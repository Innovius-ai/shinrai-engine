"""Constrained IOB2 decoding — the single decoder shared by training round-trip
tests, the Predictor, and the eval harness (WP-09: never fork this logic).

Importable without torch.
"""

from __future__ import annotations

import os
import re

from .labels import LabelSpace


def default_recall_floor(meta: dict | None) -> float | None:
    """The decoder threshold a predictor starts with (docs/39 D8, 2026-09-04/05).

    Precedence: ``SHINRAI_RECALL_FLOOR`` (harness-wide override) > the checkpoint's own
    decoder block (``config.json`` -> ``"shinrai"`` -> ``"decoder"`` -> ``"recall_floor"``,
    written at publish time so a release candidate ships as model + decoder) > ``None``
    (plain argmax, byte-identical to every pre-D8 decode). A per-call
    ``predict(recall_floor=...)`` still wins over all of these. Lives here (not in the torch
    predictor) because the vendored ONNX runtime imports only decode / labels / adapter."""
    env = os.environ.get("SHINRAI_RECALL_FLOOR")
    if env:
        return float(env)
    decoder = (meta or {}).get("decoder") or {}
    value = decoder.get("recall_floor")
    return float(value) if value is not None else None


def spans_from_labels(
    label_ids_by_head: dict[str, list[int]],
    offsets: list[tuple[int, int]],
    label_space: LabelSpace,
    text: str,
    confidences_by_head: dict[str, list[float]] | None = None,
    inword_gap: int = 0,
    evidence_by_head: dict[str, list[bool]] | None = None,
    proclitic_completion: str | None = None,
) -> list[dict]:
    """Decode per-token label ids into char-exact entity dicts.

    ``evidence_by_head`` (status-41 audit §6, 2026-09-06): per-token flags marking tokens the
    recall-floor decoder rescued from argmax-O. An entity with at least one rescued token
    carries ``evidence: "floor"``, every other entity ``evidence: "argmax"`` — the serving
    layer and the evaluation can tell a model decision from a floor decision. Absent = every
    entity is ``argmax`` (byte-identical spans; the key is added).

    Constrained decoding repairs invalid transitions instead of failing:
    an I-X-TIER with no open span (or after O) opens a new span (treated as B);
    an I with a tier mismatch extends the open span, keeping the opening tier.

    Tokens with start == end offsets (special tokens, padding) are skipped.
    Returns entities sorted by start: {span, text, type, tier, confidence}.

    ``inword_gap`` (docs/39 D8, 2026-09-04; default 0 = byte-identical decode):
    merge two same-type spans whose gap is at most that many characters, contains
    no whitespace and no sentence punctuation, when the joined surface is one
    word or compound. Targets the sub-word flicker measured on real text —
    «八千代出版» → «八» + «出版», «Інтерфакс-Україна» → «Інтерфакс» + «Україна»,
    «Reilly-Jenkins» → «Reilly» + «Jenkins». Opt-in; the sealed gates run with 0
    unless a candidate is evaluated WITH its decoder setting on purpose.

    ``proclitic_completion`` (DIAGNOSIS L12, 2026-09-07; default None = off,
    byte-identical decode): ``"he"`` extends a predicted span backwards over an
    attached Hebrew proclitic cluster when the span starts inside a written word
    (the whole-word convention of align.py r3). Script-gated on Hebrew letters,
    so ar — whose convention EXCLUDES the clitic — and every other track stay
    byte-identical. A decoder setting like the recall floor: off unless a
    candidate is evaluated WITH it on purpose.
    """
    entities: list[dict] = []
    for head, label_ids in label_ids_by_head.items():
        space = label_space.heads[head]
        confs = confidences_by_head.get(head) if confidences_by_head else None
        flags = evidence_by_head.get(head) if evidence_by_head else None
        open_span: dict | None = None

        def close(span: dict | None) -> None:
            if span is None:
                return
            start, end = span["start"], span["end"]
            # BPE/SP offset mappings include leading whitespace in the token;
            # entity surfaces never start or end with whitespace — trim.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start >= end:  # whitespace-only prediction — nothing to emit
                return
            entities.append(
                {
                    "span": [start, end],
                    "text": text[start:end],
                    "type": head,  # noqa: B023 — closed over per-head loop body only
                    "tier": span["tier"],
                    "confidence": (
                        round(sum(span["confs"]) / len(span["confs"]), 6) if span["confs"] else 1.0
                    ),
                    "evidence": "floor" if span["rescued"] else "argmax",
                }
            )

        for idx, (label_id, (tok_start, tok_end)) in enumerate(
            zip(label_ids, offsets, strict=False)
        ):
            if tok_start == tok_end:  # special token / padding
                continue
            label = space.labels[label_id] if 0 <= label_id < len(space.labels) else "O"
            if label == "O":
                close(open_span)
                open_span = None
                continue
            prefix, _head_name, tier = label.split("-", 2)
            tier = tier.lower()
            conf = confs[idx] if confs else None
            flag = bool(flags[idx]) if flags and idx < len(flags) else False
            if prefix == "B" or open_span is None:
                close(open_span)
                open_span = {
                    "start": tok_start,
                    "end": tok_end,
                    "tier": tier,
                    "confs": [conf] if conf is not None else [],
                    "rescued": flag,
                }
            else:  # I continuing an open span (tier of the opening token wins)
                open_span["end"] = tok_end
                if conf is not None:
                    open_span["confs"].append(conf)
                open_span["rescued"] = open_span["rescued"] or flag
        close(open_span)

    merged = _merge_geresh_splits(
        sorted(entities, key=lambda e: (e["span"][0], e["span"][1])), text
    )
    if inword_gap > 0:
        merged = merge_inword_splits(merged, text, inword_gap)
    if normalise_proclitic_completion(proclitic_completion) is not None:
        merged = complete_proclitic_spans(merged, text)
    return merged


_GAP_STOP = set('.,;:!?()[]{}"«»„“”…。、！？（）「」『』')


def merge_inword_splits(entities: list[dict], text: str, max_gap: int) -> list[dict]:
    """Merge same-type spans separated by an in-word gap of <= ``max_gap`` characters
    (no whitespace, no sentence punctuation) into one span when the joined surface
    carries no whitespace either (one word / compound). Any-script generalisation of
    the Hebrew geresh rule; confidence = mean of the parts. Pure function."""
    if not entities or max_gap <= 0:
        return entities
    out = [dict(entities[0])]
    for e in entities[1:]:
        a = out[-1]
        if e["type"] == a["type"] and e["span"][0] >= a["span"][1]:
            gap = text[a["span"][1] : e["span"][0]]
            joined = text[a["span"][0] : e["span"][1]]
            if (
                len(gap) <= max_gap
                and not any(c.isspace() for c in gap)
                and not (_GAP_STOP & set(gap))
                and not any(c.isspace() for c in joined)
            ):
                confs = [a["confidence"], e["confidence"]]
                a["span"] = [a["span"][0], e["span"][1]]
                a["text"] = joined
                a["confidence"] = round(sum(confs) / len(confs), 6)
                if e.get("evidence") == "floor":
                    a["evidence"] = "floor"
                continue
        out.append(dict(e))
    return out


_GERESH = {"'", "\u05f3", "\u05f4", "\u2019"}  # ' ׳ ״ ’
_HE_LETTER_LO, _HE_LETTER_HI = "\u05d0", "\u05ea"  # alef .. tav


def _is_hebrew(ch: str) -> bool:
    """A Hebrew LETTER, alef..tav (finals included). Combining marks are NOT letters.
    One definition, shared by the geresh merge and the L12 block below."""
    return _HE_LETTER_LO <= ch <= _HE_LETTER_HI


def _merge_geresh_splits(entities: list[dict], text: str) -> list[dict]:
    """Merge same-type spans split at an in-word geresh (he f1 stick,
    2026-08-28): the model labels the geresh token O inside names like
    גוג'ראנוואלה and the decoder emits fragments. Merge is he-scoped: the
    gap must be empty or geresh-class only, the joined surface must stay
    one word, and a flanking char must be a Hebrew letter."""
    if not entities:
        return entities
    out = [entities[0]]
    for e in entities[1:]:
        a = out[-1]
        if e["type"] == a["type"] and e["span"][0] >= a["span"][1]:
            gap = text[a["span"][1] : e["span"][0]]
            joined = text[a["span"][0] : e["span"][1]]
            flanks_hebrew = (
                (_is_hebrew(text[a["span"][1] - 1]) or _is_hebrew(text[e["span"][0]]))
                if joined
                else False
            )
            if (
                all(c in _GERESH for c in gap)
                and (_GERESH & set(joined))
                and flanks_hebrew
                and not any(c.isspace() for c in joined)
            ):
                confs = [a["confidence"], e["confidence"]]
                a["span"] = [a["span"][0], e["span"][1]]
                a["text"] = joined
                a["confidence"] = round(sum(confs) / len(confs), 6)
                if e.get("evidence") == "floor":
                    a["evidence"] = "floor"
                continue
        out.append(e)
    return out


# --- L12: proclitic span completion (he) -----------------------------------
# DIAGNOSIS-2026-09-06 L12. The he span convention since align.py r3 (2026-08-18)
# is the WHOLE WRITTEN WORD: an attached proclitic (ו ש כש ב ל מ כ ה, alone or
# stacked) is INSIDE the span. The tokenizer splits some of those words into a
# proclitic piece plus the name («▁ל|נחום»), so the model can — and often does —
# label only the name piece; the span then starts strictly INSIDE a written word
# and scores as a boundary error on both sides of the memcheck instrument
# (train fn STREET «בנחל קדרון 85», held-out fn CITY «ודאקה»).
#
# This post-pass extends such a span backwards to the start of its written word
# when the swallowed lead is a proclitic cluster. It is a pure function of the
# prediction and the text — the mirror of span_projection.permissible_lead on
# the PREDICTION side, with the same grammar as align.py ``_HE_PROCLITIC``.
#
# ar is deliberately NOT covered: the Arabic convention EXCLUDES the clitic
# (span_projection.py, align.py r3 note), so completing an Arabic span would
# move it away from its gold. The rule is script-gated on Hebrew letters, so
# any other-language text is byte-identical with the setting on.
#
# It is a DECODER SETTING, not a silent fix: off by default, stampable
# (tools/stamp_decoder --proclitic-completion he), resolved request > env >
# checkpoint stamp > off, and reported in the serve audit block.

# Hebrew combining marks, copied from generation/align.py ``_HE_MARKS``:
# cantillation U+0591-05AF, points U+05B0-05BD, 05BF-05C2, puncta 05C4-05C5,
# qamats qatan 05C7. Maqaf U+05BE is a visible hyphen and stays a separator.
_HE_MARK = re.compile("[\u0591-\u05BD\u05BF-\u05C2\u05C4\u05C5\u05C7]")
# align.py ``_HE_PROCLITIC`` verbatim: conjunction/relativizer, then a
# preposition, then optionally the article.
# grammar: (vav|shin|kaf-shin)? (bet|lamed|mem|kaf)? (he)?
_HE_PROCLITIC_RE = re.compile(
    "(?:\u05d5|\u05e9|\u05db\u05e9)?(?:\u05d1|\u05dc|\u05de|\u05db)?(?:\u05d4)?"
)

PROCLITIC_COMPLETION_LANGS = ("he",)
_PROCLITIC_OFF = ("", "off", "none", "no", "0", "false")


def normalise_proclitic_completion(value: object) -> str | None:
    """Parse a proclitic-completion setting; ``None`` = off (the default).

    Accepts a language key from ``PROCLITIC_COMPLETION_LANGS`` or an explicit
    off word. Anything else raises — a typo in an env var or a checkpoint stamp
    must fail loudly, never degrade to a silent default."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _PROCLITIC_OFF:
        return None
    if text in PROCLITIC_COMPLETION_LANGS:
        return text
    raise ValueError(
        f"proclitic_completion must be one of {PROCLITIC_COMPLETION_LANGS} "
        f"or an off word {_PROCLITIC_OFF}, got {value!r}"
    )


def default_proclitic_completion(meta: dict | None) -> str | None:
    """The proclitic-completion setting a predictor starts with (DIAGNOSIS L12).

    Precedence, mirroring ``default_recall_floor``: ``SHINRAI_PROCLITIC_COMPLETION``
    (harness-wide override) > the checkpoint's own decoder block (``config.json``
    -> ``"shinrai"`` -> ``"decoder"`` -> ``"proclitic_completion"``, written at
    publish time by tools/stamp_decoder) > ``None`` (off, byte-identical decode).
    A per-call ``predict(proclitic_completion=...)`` still wins over all of these."""
    env = os.environ.get("SHINRAI_PROCLITIC_COMPLETION")
    if env is not None and env.strip():
        return normalise_proclitic_completion(env)
    decoder = (meta or {}).get("decoder") or {}
    return normalise_proclitic_completion(decoder.get("proclitic_completion"))


def _word_start(text: str, index: int) -> int:
    """Start of the written Hebrew word containing ``index`` (which must sit on a
    Hebrew letter). Walks back over Hebrew letters and their combining marks."""
    i = index
    while i > 0 and (_is_hebrew(text[i - 1]) or _HE_MARK.match(text[i - 1])):
        i -= 1
    return i


def complete_proclitic_spans(entities: list[dict], text: str) -> list[dict]:
    """Extend every span that starts INSIDE a Hebrew written word backwards over
    its attached proclitic cluster (DIAGNOSIS L12). Pure function; ``entities``
    must be sorted by span start. Returns a new list.

    A span is extended only when ALL of these hold:

    * its first character is a Hebrew letter (script gate — ar and every other
      track is byte-identical);
    * the character before it is a Hebrew letter (the span starts inside a
      written word; a span already at a word start is never touched, so a name
      that legitimately begins with ב/ל/ה keeps its boundary);
    * the letters between the word start and the span start form a non-empty
      proclitic cluster under ``_HE_PROCLITIC_RE`` (combining marks ignored);
    * the word start is not glued to another letter (align.py's
      ``(?<![^\\W\\d_])`` word-boundary rule);
    * the extension does not run into the previous span of the same type.

    Known blind spots, pinned by tests and measured on 2026-09-07 (verify-L12):

    * the character before the span must be a LETTER, so a span whose preceding
      character is a Hebrew combining mark (pointed text) or a bidi control is
      NOT extended. Measured exposure on the ship candidate's banked he dumps:
      3 predicted spans of 104,787 on the sealed f2 suite, 6 of 111,750 on the
      legacy suite, 0 on the v1.4-he-sp1rp suite. ``l12_effect.py`` applies the
      same gate, so its convention counts share the blind spot;
    * the same-type guard is per type, so an extension may overlap a span of a
      DIFFERENT head that ends inside the same written word (multi-head decoding
      already emits overlapping spans, so this adds no new output class).
    """
    if not entities:
        return entities
    out: list[dict] = []
    prev_end: dict[str, int] = {}
    for ent in entities:
        start, end = ent["span"]
        new_start = start
        inside_word = (
            0 < start < len(text) and _is_hebrew(text[start]) and _is_hebrew(text[start - 1])
        )
        if inside_word:
            word_start = _word_start(text, start)
            lead = _HE_MARK.sub("", text[word_start:start])
            glued = word_start > 0 and text[word_start - 1].isalpha()
            if lead and not glued and _HE_PROCLITIC_RE.fullmatch(lead):
                if word_start >= prev_end.get(ent["type"], -1):
                    new_start = word_start
        if new_start != start:
            ent = {**ent, "span": [new_start, end], "text": text[new_start:end]}
        prev_end[ent["type"]] = max(prev_end.get(ent["type"], -1), ent["span"][1])
        out.append(ent)
    # an extension can move a start before an earlier entity of another head —
    # the decoder's output contract is sorted by (start, end)
    return sorted(out, key=lambda e: (e["span"][0], e["span"][1]))
