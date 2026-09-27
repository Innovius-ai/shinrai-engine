"""Sentence-sized decode pieces (2026-08-22, the long-input finding).

Measured on ShinrAI v1.3-r3b: recall on 300–800-word prompts is 22% when the
whole text is one decode and 77% when each sentence is decoded on its own;
our own suite paragraphs lose recall the moment two of them are glued
together (en 95.2% → 91.2% ×2 → 80.4% ×4 → 73.8% ×8). The model was trained
on 80–160-word paragraphs; its confidence collapses with input length.

This module is the serve-side half of the fix: split a text into pieces at
sentence and paragraph boundaries, decode every piece as its own input, and
shift the spans back. Offsets are exact by construction — pieces are
``text[start:end]`` slices, never re-joined strings. Stdlib only so both
predictors (torch and ONNX/numpy) can import it.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? … (optionally followed by a closing quote or
# bracket) plus whitespace, or at a blank line. Full-width terminators cover
# ja/zh; Arabic and Hebrew use the Latin marks. Abbreviations ("Dr. Müller",
# "z. B.") are NOT special-cased: a split after "Dr." costs nothing for
# detection because the name still sits whole inside the next piece, and the
# merge step re-joins an entity only if both halves were predicted anyway.
# 2026-09-24 (D-N2 readout): a 1-2 digit ordinal before the period («15. März», «12. Juni», «3. Sept.») is not a
# sentence end - the split cut every German / Polish / Russian day-month date in two (chat DATE exact 165 vs 225
# whole-text). Four-digit years («… im Jahr 2024. Dann») still end a sentence.
_SENT_END = re.compile(r"(?<=[.!?…。！？])(?<![^\d]\d\.)(?<![^\d]\d\d\.)(?<!^\d\.)(?<!^\d\d\.)[\"'»”’)\]]*\s+|\n[ \t]*\n")

_ABBREV_RE = re.compile(r"(?<![^\W\d_])\d{1,2}\.?\s+([^\W\d_]{1,5})\.[\"'»”’)\]]*$")   # «3. Sept.», «10 окт.» - a day first


def _abbrev_before(text: str, pos: int) -> bool:
    """True when the terminator at ``pos`` closes a month abbreviation («3. Sept. 2025», «10 окт. 2023»):
    the same 15-locale abbreviation inventory the temporal-negatives training rule uses."""
    m = _ABBREV_RE.search(text[max(0, pos - 12):pos])
    if not m:
        return False
    from .temporal_negatives import _MONTH_ABBR  # local import: no training import at module load
    return re.fullmatch(_MONTH_ABBR, m.group(1).lower()) is not None


MIN_PIECE = 24  # characters; shorter tails are glued to the previous piece
MAX_PIECE = 600  # characters; a run-on without terminators is cut at whitespace


def sentence_pieces(text: str) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` covering every non-blank character of
    ``text`` in order. Pieces never overlap; ``text[s:e]`` is decoded as is."""
    pieces: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_END.finditer(text):
        if _abbrev_before(text, m.start()):
            continue
        if m.start() > pos:
            pieces.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        pieces.append((pos, len(text)))
    # trim whitespace at both ends of every piece
    trimmed = []
    for s, e in pieces:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    # glue tiny tails onto their predecessor, cut over-long runs at whitespace
    out: list[tuple[int, int]] = []
    lead: int | None = None   # a first piece shorter than MIN_PIECE («Dr.») is glued FORWARD (2026-09-24)
    for s, e in trimmed:
        if lead is not None:
            s, lead = lead, None
        if out and (e - s) < MIN_PIECE:
            ps, _ = out[-1]
            out[-1] = (ps, e)
            continue
        if not out and (e - s) < MIN_PIECE:
            lead = s
            continue
        while (e - s) > MAX_PIECE:
            cut = text.rfind(" ", s + MIN_PIECE, s + MAX_PIECE)
            if cut <= s:
                cut = s + MAX_PIECE
            out.append((s, cut))
            s = cut
            while s < e and text[s].isspace():
                s += 1
        if e > s:
            out.append((s, e))
    if lead is not None and not out:
        out.append((lead, len(text.rstrip())))
    return out


def predict_segmented(predict, texts: list[str], merge, **kwargs) -> list[list[dict]]:
    """Decode each text piece by piece. ``predict(list[str], **kwargs)`` is the
    predictor's whole-text entry point; ``merge(list[dict])`` its same-head
    overlap dedup. Spans come back in whole-text coordinates."""
    pieces_per_text = [sentence_pieces(t) for t in texts]
    flat: list[str] = []
    owner: list[tuple[int, int]] = []  # (text index, piece start)
    for ti, (text, pieces) in enumerate(zip(texts, pieces_per_text, strict=True)):
        for s, e in pieces:
            flat.append(text[s:e])
            owner.append((ti, s))
    per_text: list[list[dict]] = [[] for _ in texts]
    if flat:
        size = max(1, int(kwargs.get("batch_size", 4)))
        for offset in range(0, len(flat), size):
            predicted = predict(flat[offset:offset + size], **kwargs)
            for (ti, start), ents in zip(owner[offset:offset + size], predicted, strict=True):
                for ent in ents:
                    ent = dict(ent)
                    ent["span"] = [ent["span"][0] + start, ent["span"][1] + start]
                    per_text[ti].append(ent)
    return [merge(ents) for ents in per_text]


# ``segment="auto"`` (2026-08-22, measured on r3b): whole-text decode for
# inputs up to AUTO_CHARS, and for longer inputs the union of the whole-text
# decode with sentence-decode entities that clear AUTO_MIN_CONF and overlap
# nothing the whole pass found. Measured (overlap P / R / F1):
#   en paragraphs   whole 95.0/95.2/95.1   union@.95 91.4/95.8/93.5   sentence 84.5/90.1/87.2
#   en x4 concat    whole 98.2/80.4/88.4   union@.95 94.2/87.3/90.6   sentence 84.3/90.1/87.1
#   de paragraphs   whole 97.5/95.5/96.5   union@.95 90.1/96.5/93.2   sentence 77.8/95.5/85.7
#   de x4 concat    whole 98.2/76.5/86.0   union@.95 90.9/89.8/90.4   sentence 79.0/95.3/86.4
#   27 chat prompts whole 71.4/20.2/31.5   union@.95 80.7/50.9/62.4   sentence 63.6/71.7/67.4
# Short inputs keep the whole-text decode (best there); long inputs gain
# +2 to +4 F1 in-distribution and +31 F1 on chat prompts at precision >= 80.
# "sentence" stays the redaction-first choice (recall 72% vs 51% on prompts).
AUTO_CHARS = 1200
AUTO_MIN_CONF = 0.95


def predict_auto(predict, texts: list[str], merge, **kwargs) -> list[list[dict]]:
    whole = predict(texts, **kwargs)
    long_idx = [i for i, t in enumerate(texts) if len(t) > AUTO_CHARS]
    if not long_idx:
        return whole
    extra = predict_segmented(predict, [texts[i] for i in long_idx], merge, **kwargs)
    out = list(whole)
    for i, ents in zip(long_idx, extra, strict=True):
        base = list(whole[i])
        for e in ents:
            if float(e.get("confidence", 1.0)) < AUTO_MIN_CONF:
                continue
            # status-41 audit §6 (2026-09-06): the union must not add a sentence-decode entity
            # that overlaps a whole-text entity of ANY type — the earlier same-type test let a
            # PERSON from the sentence pass sit on top of a whole-text ORG (double coverage,
            # two spans on one surface). The whole-text decode keeps precedence.
            if any(
                not (e["span"][1] <= k["span"][0] or e["span"][0] >= k["span"][1]) for k in base
            ):
                continue
            base.append(e)
        out[i] = merge(base)
    return out
