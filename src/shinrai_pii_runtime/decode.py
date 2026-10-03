"""Constrained IOB2 decoding — the single decoder shared by training round-trip
tests, the Predictor, and the eval harness (WP-09: never fork this logic).

Importable without torch.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

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
    lang: str | None = None,
    settings: DecoderSettings | None = None,
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

    ``settings`` + ``lang`` (D23 / D21, 2026-09-08): the stampable ``DecoderSettings``
    block — ko in-word gap fill, span continuity, ja particle strip — applied as
    post-passes by ``apply_decoder_settings``; the per-head recall floors of the same
    block act before this function (``head_recall_floor``). ``None`` = every setting
    off, byte-identical decode. A per-language key fires only when ``lang`` names
    that language; no ``lang`` = no per-language setting.
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
    if settings is not None and not settings.off:
        merged = apply_decoder_settings(merged, text, lang, settings)
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


# --- D23 / D21: stampable, per-language decoder settings ----------------------
# docs/38 2026-09-08: D23 GO — per-head per-language decoder floors (ja ORG 0.20,
# ko CITY 0.20) + ja particle strip + ko in-word gap fill as stamped keys; D21 — L6 span
# continuity as a stamped setting. Basis: research/competitive/eastasian/DIAGNOSIS-2026-09-07.md
# §3.6 / §5 (rules c and d measured on stored predictions; the gap fill costs ja −15, so every
# key is PER LANGUAGE) and research/competitive/medical/ERROR-ANATOMY-2026-09-07.md §5 L6
# (the ``partial_token`` boundary class: a prediction that ends or starts inside a word).
#
# The L12 principle, binding: OFF by default, stampable into ``config.json`` ->
# ``shinrai.decoder`` with a dated reason (tools/stamp_decoder), read back with the recall
# floor's precedence (request > env > checkpoint stamp > off), reported in the serve audit
# block, reached on all three decode paths, never applied silently to a candidate measured
# without it. The language of a text is the caller's ``lang`` (request ``language`` /
# record ``lang``) — no lang, no per-language setting.

DECODER_SETTINGS_ENV = "SHINRAI_DECODER_SETTINGS"
DECODER_SETTING_KEYS = (
    "recall_floor_by_head",  # {lang: {HEAD: floor}}  — layered on the global floor
    "particle_strip",        # [lang, ...]            — trailing particle trimmed (ja table)
    "inword_gap_fill",       # {lang: max_gap}        — merge_inword_splits per language
    "span_continuity",       # true                   — L6 sub-word fragment join, any language
    "numeric_span_repair",   # [HEAD, ...]            — join + extend a split numeric span over its own digits
    "word_completion",       # {lang: [HEAD, ...]}    — N1: a span cut inside a written word grows to the word
    "name_join",             # {lang: [HEAD, ...]}    — N2: same-type spans over a space / initial / hyphen join
    "name_initials",         # {lang: [HEAD, ...]}    — N2b: a span grows left over a run of initials («С. И. Гири»)
    "date_span_join",        # true                   — D-N2: DATE fragments join and grow over day / month / year tokens
    "street_number_join",    # true                   — D-N3: a STREET span takes its adjacent house number («Hauptstraße» 5, 18 «rue des Lilas»)
    "date_identifier_exclusion",  # true              — D-N4: no DATE inside an identifier (ECLI, 2004/17/EC, No 883/2004, C-83/14)
    "signoff_floor",         # {HEAD: floor}          — F001 (v1.6): a lower floor on the 1–2 lines after a closing formula
    # guard v13 (2026-10-03, research/competitive/decoder-settings/GUARD-V13-2026-10-03.md)
    "date_day_growth",       # true                   — F014: a DATE span grows left over its day number («.10.2026» -> «15.10.2026»)
    "date_year_abbrev",      # [lang, ...]            — F002: a DATE span takes the year abbreviation's period («2026 r» -> «2026 r.»)
    "name_initial_bridge",   # {lang: [PERSON]}       — F011: an initial-only PERSON span grows to its name («A.» -> «John A. Smith»)
    "name_join_cased",       # {lang: [HEAD, ...]}    — F012: name_join between two capitalised pieces only («Marana» + «Aurinete Brito»)
    "signoff_shapes",        # [shape, ...]           — F015: more sign-off regions (same_line, trailing_formula, name_before_closing)
    "person_full_span",      # true                   — F016: a multi-word PERSON span is typed as a full name by the adapter
    "age_span_repair",       # true                   — F021: AGE fragments join and grow over their number and unit («5» + «anos» -> «58 anos»)
)
# recall_floor_by_head: the language-independent default (F013, guard v13). Used when the call names no
# language, or a language the stamp has no entry for (exact key and primary subtag both missing).
DEFAULT_LANG_KEY = "*"
_SETTINGS_OFF = ("", "off", "none", "no", "0", "false", "{}")

# Rule d, taskb_model_probe.py JA_PARTICLES verbatim (results/decode-rules-ja-2026-09-07.json):
# the SentencePiece pieces fuse the following particle into the last piece of a name
# («大学の», «子が»), so the exact span is unrepresentable at token level. Longest first.
JA_PARTICLES = (
    "から", "まで", "には", "では", "とは", "らの",
    "の", "が", "は", "を", "に", "と", "も", "へ", "で", "や", "か",
)  # fmt: skip
PARTICLE_TABLES: dict[str, tuple[str, ...]] = {"ja": JA_PARTICLES}
# the analysts' rule touches PERSON / ORG / CITY predictions (gains ORG 3 / PERSON 7, 0 lost)
PARTICLE_STRIP_TYPES = ("PERSON", "ORG", "CITY")
PARTICLE_STRIP_MIN_REMAINDER = 2
# Kana-name guard (decoder-settings/S2-GUARD-2026-09-08.md, measured on the ja sealed-suite
# dump, 41,030 records): a kana given name can END in a particle character — «永海みか»,
# «かつなが», «明添 とうや» — and the unguarded rule cut 451 exact PERSON hits there. The strip
# is blocked for a PERSON span when the character before the particle is hiragana AND the
# particle is one character other than は / を: those two never end a given name, and the
# two-character particles («から», «には» …) neither — both keep stripping (gains 45 / 1 / 5).
# The guard leaves ORG / CITY alone (hiragana-preceded strips there: 12 gained, 0 lost) and a
# kanji- or katakana-final name alone (gains 257 / 2, losses 4). Dump reading: +305 net
# PERSON exact hits instead of +14; 4 lost instead of 451.
PARTICLE_STRIP_KANA_NAME_TYPES = ("PERSON",)
PARTICLE_STRIP_NOT_NAME_FINAL = ("は", "を")


def _is_hiragana(ch: str) -> bool:
    return "\u3040" <= ch <= "\u309f"


def kana_name_guard(surface: str, particle: str, ent_type: str) -> bool:
    """True when the particle strip must NOT fire: ``surface`` is a PERSON span whose character
    before the one-character particle is hiragana (a kana given name ending in か / が / と /
    な / の / も / や …). は / を and the two-character particles are never blocked."""
    if ent_type not in PARTICLE_STRIP_KANA_NAME_TYPES or len(particle) != 1:
        return False
    if particle in PARTICLE_STRIP_NOT_NAME_FINAL:
        return False
    before = len(surface) - len(particle) - 1
    return before >= 0 and _is_hiragana(surface[before])


def lang_key(lang: str | None) -> str | None:
    """Canonical language key: lower-cased, stripped; ``None`` stays ``None``."""
    if lang is None:
        return None
    key = str(lang).strip().lower().replace("_", "-")
    return key or None


def _lookup(mapping: Mapping[str, object], lang: str | None):
    """Per-language lookup: exact key first (``pt-br``), then the primary subtag (``pt``)."""
    key = lang_key(lang)
    if key is None or not mapping:
        return None
    if key in mapping:
        return mapping[key]
    primary = key.split("-", 1)[0]
    return mapping.get(primary)


# F001 (research/v16/findings/F001-signoff-names.md, 2026-10-02): a name alone on its line
# after a closing formula («Mit freundlichen Grüßen,» / «Max Munster») decodes at 0.27–0.41
# entity mass where the same name inline reads 0.93. The register has no sentence around the
# name; the floor, not the model, is the cheapest lever until the v1.6 sign-off carrier trains.
# The table covers the 15 locales of the release. A line matches when, without its trailing
# punctuation, it IS a formula (case-folded) — or, for the long formal formulas, starts with one.
CLOSING_FORMULAS: tuple[str, ...] = (
    # de
    "mit freundlichen grüßen", "mit freundlichen grüssen", "freundliche grüße", "freundliche grüsse",
    "mit besten grüßen", "beste grüße", "viele grüße", "liebe grüße", "herzliche grüße",
    "mit herzlichen grüßen", "schöne grüße", "hochachtungsvoll", "mit freundlichem gruß", "gruß", "grüße", "mfg", "lg", "vg",
    # en
    "best regards", "kind regards", "warm regards", "warmest regards", "regards", "best wishes", "best",
    "sincerely", "yours sincerely", "sincerely yours", "yours faithfully", "yours truly", "respectfully",
    "cheers", "thanks", "thank you", "many thanks", "thanks and regards", "with best regards", "with kind regards",
    # fr
    "cordialement", "bien cordialement", "très cordialement", "bien à vous", "salutations distinguées",
    "sincères salutations", "meilleures salutations", "bien amicalement", "amitiés",
    # es
    "atentamente", "saludos", "saludos cordiales", "un saludo", "un cordial saludo", "cordialmente", "muy atentamente",
    # it
    "cordiali saluti", "distinti saluti", "saluti", "un saluto", "cordialmente", "in fede", "grazie",
    # pt
    "atenciosamente", "cumprimentos", "melhores cumprimentos", "com os melhores cumprimentos", "abraços",
    "saudações", "com os meus cumprimentos", "obrigado", "obrigada",
    # pl
    "z poważaniem", "pozdrawiam", "z wyrazami szacunku", "serdecznie pozdrawiam", "pozdrawiam serdecznie", "łączę pozdrowienia",
    # ru / uk
    "с уважением", "с наилучшими пожеланиями", "всего доброго", "всего хорошего", "спасибо",
    "з повагою", "з найкращими побажаннями", "дякую", "щиро",
    # tr
    "saygılarımla", "saygılarımızla", "iyi çalışmalar", "selamlar", "teşekkürler", "sevgilerimle",
    # ar / he
    "مع خالص التحية", "مع التحية", "مع أطيب التحيات", "تحياتي", "وتفضلوا بقبول فائق الاحترام", "شكراً", "شكرا",
    "בברכה", "בכבוד רב", "תודה", "בברכה רבה",
    # ja / ko
    "よろしくお願いいたします", "よろしくお願いします", "敬具", "草々", "以上",
    "감사합니다", "고맙습니다", "안녕히 계세요",
)
# long formal formulas that run on to the end of their line («Veuillez agréer, Madame, …»)
CLOSING_PREFIXES: tuple[str, ...] = (
    "veuillez agréer", "je vous prie d'agréer", "je vous prie de croire", "recevez, madame", "recevez, monsieur",
    "le saluto cordialmente", "la saludo atentamente", "le saluda atentamente", "reciba un cordial saludo",
    "please do not hesitate", "yours sincerely,", "best regards,",
)
_CLOSING_SET = frozenset(CLOSING_FORMULAS)
_TRAILING_PUNCT = " \t,.;:!—–-…。、"
SIGNOFF_MAX_LINES = 2       # the name line and, at most, one more (title / role / company)
SIGNOFF_MAX_LINE_CHARS = 60  # a longer line is prose, not a signature
SIGNOFF_MAX_BLANK = 2        # blank lines allowed between the formula and the name


def _is_closing_formula(line: str) -> bool:
    core = line.strip().rstrip(_TRAILING_PUNCT).strip().casefold()
    if not core or len(core) > 120:
        return False
    if core in _CLOSING_SET:
        return True
    return any(core.startswith(prefix) for prefix in CLOSING_PREFIXES)


def signoff_regions(text: str, shapes: tuple[str, ...] | list[str] = ()) -> list[tuple[int, int]]:
    """Character ranges of the 1–2 short non-empty lines that follow a closing-formula line
    (up to ``SIGNOFF_MAX_BLANK`` blank lines in between). The decoder lowers the floor of the
    ``signoff_floor`` heads inside these ranges only; everything else decodes unchanged.

    ``shapes`` (guard v13, F015; empty = the guard v12 regions, byte-identical) adds:
    ``same_line`` — the rest of a line that starts with a formula («Kind regards, Jane Doe»);
    ``trailing_formula`` — a formula that ends a prose line («… 45 67. Saygılarımla,») opens the
    region on the next lines like a formula line; ``name_before_closing`` — the name before a
    trailing closing word on a short line («홍길동 드림», «김철수 올림»)."""
    if shapes:
        return _signoff_regions_shaped(text, tuple(shapes))
    if not text or "\n" not in text:
        return []
    lines: list[tuple[int, int]] = []
    pos = 0
    for raw in text.split("\n"):
        lines.append((pos, pos + len(raw)))
        pos += len(raw) + 1
    regions: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(lines):
        if not _is_closing_formula(text[start:end]):
            continue
        j, blanks, taken = i + 1, 0, 0
        while j < len(lines) and taken < SIGNOFF_MAX_LINES:
            ls, le = lines[j]
            body = text[ls:le]
            if not body.strip():
                if taken:
                    break
                blanks += 1
                if blanks > SIGNOFF_MAX_BLANK:
                    break
                j += 1
                continue
            if len(body.strip()) > SIGNOFF_MAX_LINE_CHARS or _is_closing_formula(body):
                break
            lead = len(body) - len(body.lstrip())
            regions.append((ls + lead, le - (len(body) - len(body.rstrip()))))
            taken += 1
            j += 1
    return regions


SIGNOFF_SHAPES = ("same_line", "trailing_formula", "name_before_closing")
# closing words written AFTER the name on the same line (ko: «홍길동 드림» "offered by", «올림» "presented by»,
# «배상» "respectfully"); the name is the part of the line before the word
CLOSING_WORDS_AFTER_NAME: tuple[str, ...] = ("드림", "올림", "배상")
SIGNOFF_SAME_LINE_MAX_WORDS = 5
# one-word formulas that also start ordinary sentences or names («Best Buy», «LG Electronics»): never a same-line sign-off
SAME_LINE_SKIP = frozenset({"best", "lg", "vg", "mfg"})
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_FORMULAS_LONGEST_FIRST = tuple(sorted(_CLOSING_SET, key=lambda f: len(f.casefold()), reverse=True))


def _lines(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 0
    for raw in text.split("\n"):
        out.append((pos, pos + len(raw)))
        pos += len(raw) + 1
    return out


def _trailing_formula(line: str) -> bool:
    """The last sentence of a prose line is a closing formula («Telefon: 0532 123 45 67. Saygılarımla,»)."""
    parts = _SENTENCE_SPLIT.split(line.strip())
    return len(parts) > 1 and _is_closing_formula(parts[-1])


def _same_line_rest(text: str, start: int, end: int) -> tuple[int, int] | None:
    """The name part of «<formula>[,] <name>» on one line: at most ``SIGNOFF_SAME_LINE_MAX_WORDS`` words, its first
    letter not lower-case, no sentence punctuation inside. ``None`` when the line has no such shape."""
    line = text[start:end]
    lead = len(line) - len(line.lstrip())
    folded = line[lead:].casefold()
    for formula in _FORMULAS_LONGEST_FIRST:
        key = formula.casefold()
        if not folded.startswith(key):
            continue
        if formula in SAME_LINE_SKIP:
            return None
        # casefold can change the length («ß» -> «ss», «İ» -> «i̇»): find the original prefix that folds to the formula
        size = next((k for k in range(max(1, len(key) - 3), len(key) + 3)
                     if line[lead:lead + k].casefold() == key), None)
        if size is None:
            return None
        rest_at = lead + size
        m = re.match(r"\s*[,;:\-–—]?\s+|\s*[,;:\-–—]\s*", line[rest_at:])
        if not m:
            return None                      # the formula runs into a longer word («bestellen»)
        if not re.search(r"[,;:\-–—]", m.group(0)) and " " not in formula:
            return None                      # a one-word formula needs its comma / dash («Thanks, Anna»)
        rs = rest_at + m.end()
        body = line[rs:].rstrip().rstrip(_TRAILING_PUNCT).rstrip()
        if not body or len(body) > SIGNOFF_MAX_LINE_CHARS or len(body.split()) > SIGNOFF_SAME_LINE_MAX_WORDS:
            return None
        first = next((ch for ch in body if ch.isalpha()), "")
        if not first or first.islower() or re.search(r"[.!?。！？]\s", body):
            return None
        return (start + rs, start + rs + len(body))
    return None


def _before_closing_word(text: str, start: int, end: int) -> tuple[int, int] | None:
    """«홍길동 드림» -> the range of «홍길동»: a short line that ends in a closing word after whitespace."""
    line = text[start:end].rstrip()
    for word in CLOSING_WORDS_AFTER_NAME:
        if line.endswith(word) and len(line) > len(word) and line[-len(word) - 1].isspace():
            body = line[: -len(word)].rstrip()
            lead = len(body) - len(body.lstrip())
            body = body.strip()
            if body and len(body) <= SIGNOFF_MAX_LINE_CHARS and len(body.split()) <= SIGNOFF_SAME_LINE_MAX_WORDS:
                return (start + lead, start + lead + len(body))
    return None


def _signoff_regions_shaped(text: str, shapes: tuple[str, ...]) -> list[tuple[int, int]]:
    """``signoff_regions`` with the guard v13 shapes on; the guard v12 regions are always part of the result."""
    if not text:
        return []
    lines = _lines(text)
    regions: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(lines):
        line = text[start:end]
        opens = _is_closing_formula(line) or ("trailing_formula" in shapes and _trailing_formula(line))
        if "same_line" in shapes:
            rest = _same_line_rest(text, start, end)
            if rest is not None:
                regions.append(rest)
        if "name_before_closing" in shapes:
            name = _before_closing_word(text, start, end)
            if name is not None:
                regions.append(name)
        if not opens:
            continue
        j, blanks, taken = i + 1, 0, 0
        while j < len(lines) and taken < SIGNOFF_MAX_LINES:
            ls, le = lines[j]
            body = text[ls:le]
            if not body.strip():
                if taken:
                    break
                blanks += 1
                if blanks > SIGNOFF_MAX_BLANK:
                    break
                j += 1
                continue
            if len(body.strip()) > SIGNOFF_MAX_LINE_CHARS or _is_closing_formula(body):
                break
            lead = len(body) - len(body.lstrip())
            regions.append((ls + lead, le - (len(body) - len(body.rstrip()))))
            taken += 1
            j += 1
    return sorted(set(regions))


def token_floors(
    offsets: list[tuple[int, int]],
    regions: list[tuple[int, int]],
    base: float | None,
    signoff: float,
) -> list[float] | None:
    """Per-token floors for one head: ``min(signoff, base)`` on tokens inside a sign-off region,
    ``base`` elsewhere (2.0 = never rescue, when the head decodes argmax). ``None`` when no token
    falls in a region — the caller keeps the scalar floor and the decode stays byte-identical."""
    if not regions:
        return None
    low = signoff if base is None else min(signoff, base)
    high = 2.0 if base is None else base
    floors: list[float] = []
    hit = False
    for s, e in offsets:
        inside = e > s and any(rs <= s and e <= re_ for rs, re_ in regions)
        hit = hit or inside
        floors.append(low if inside else high)
    return floors if hit else None


_INITIAL_RE = re.compile(r"^\w\.$")


def _name_like(surface: str) -> bool:
    """Every whitespace part has two or more word characters, or is an initial («J.»). Rejects the
    sub-word fragments a low floor produces («ж» from «Серж», «ל ל», a lone CJK character)."""
    parts = surface.split()
    if not parts:
        return False
    for part in parts:
        core = part.strip(".,;:()\"'«»„“”")
        if _INITIAL_RE.match(part) or (len(core) == 1 and part.endswith(".")):
            continue
        if sum(ch.isalnum() for ch in core) < 2:
            return False
    return True


def mark_signoff_spans(
    entities: list[dict],
    regions: list[tuple[int, int]],
    signoff_floor: Mapping[str, float],
    base_floor_by_head: Mapping[str, float | None],
) -> list[dict]:
    """A rescued span inside a sign-off region whose confidence sits below its head's normal floor
    exists only through the sign-off floor. Such a span is kept only when it is name-like
    (``_name_like``) and overlaps no entity of another type (a company line stays ORG); a kept
    span carries ``bar`` = that floor, the serve bar the adapter applies to it at the stamped
    operating point. ``evidence`` stays ``"floor"`` (the API contract's closed set: argmax | floor).
    Every other entity passes unchanged."""
    if not regions or not signoff_floor:
        return entities
    out: list[dict] = []
    for ent in entities:
        head = ent.get("type")
        if head in signoff_floor and ent.get("evidence") == "floor":
            s, e = ent["span"]
            base = base_floor_by_head.get(head)
            inside = any(rs <= s and e <= re_ for rs, re_ in regions)
            if inside and (base is None or float(ent.get("confidence", 1.0)) < base):
                clash = any(
                    other is not ent and other.get("type") != head
                    and other["span"][0] < e and s < other["span"][1]
                    for other in entities
                )
                if clash or not _name_like(ent.get("text") or ""):
                    continue
                ent = {**ent, "bar": float(signoff_floor[head])}
        out.append(ent)
    return out


@dataclass(frozen=True)
class DecoderSettings:
    """The four stampable settings; the default instance is OFF (byte-identical decode)."""

    recall_floor_by_head: dict[str, dict[str, float]] = field(default_factory=dict)
    particle_strip: tuple[str, ...] = ()
    inword_gap_fill: dict[str, int] = field(default_factory=dict)
    span_continuity: bool = False
    numeric_span_repair: tuple[str, ...] = ()
    word_completion: dict[str, tuple[str, ...]] = field(default_factory=dict)
    name_join: dict[str, tuple[str, ...]] = field(default_factory=dict)
    name_initials: dict[str, tuple[str, ...]] = field(default_factory=dict)
    date_span_join: bool = False
    street_number_join: bool = False
    date_identifier_exclusion: bool = False
    signoff_floor: dict[str, float] = field(default_factory=dict)
    date_day_growth: bool = False
    date_year_abbrev: tuple[str, ...] = ()
    name_initial_bridge: dict[str, tuple[str, ...]] = field(default_factory=dict)
    name_join_cased: dict[str, tuple[str, ...]] = field(default_factory=dict)
    signoff_shapes: tuple[str, ...] = ()
    person_full_span: bool = False
    age_span_repair: bool = False

    @property
    def off(self) -> bool:
        return not (
            self.recall_floor_by_head
            or self.particle_strip
            or self.inword_gap_fill
            or self.span_continuity
            or self.numeric_span_repair
            or self.word_completion
            or self.name_join
            or self.name_initials
            or self.date_span_join
            or self.street_number_join
            or self.date_identifier_exclusion
            or self.signoff_floor
            or self.date_day_growth
            or self.date_year_abbrev
            or self.name_initial_bridge
            or self.name_join_cased
            or self.signoff_shapes
            or self.person_full_span
            or self.age_span_repair
        )

    def to_json(self) -> dict:
        """Only the keys that are on — the shape stamped into the decoder block."""
        out: dict = {}
        if self.recall_floor_by_head:
            out["recall_floor_by_head"] = {
                lang: dict(heads) for lang, heads in self.recall_floor_by_head.items()
            }
        if self.particle_strip:
            out["particle_strip"] = list(self.particle_strip)
        if self.inword_gap_fill:
            out["inword_gap_fill"] = dict(self.inword_gap_fill)
        if self.span_continuity:
            out["span_continuity"] = True
        if self.numeric_span_repair:
            out["numeric_span_repair"] = list(self.numeric_span_repair)
        if self.word_completion:
            out["word_completion"] = {lang: list(heads) for lang, heads in self.word_completion.items()}
        if self.name_join:
            out["name_join"] = {lang: list(heads) for lang, heads in self.name_join.items()}
        if self.name_initials:
            out["name_initials"] = {lang: list(heads) for lang, heads in self.name_initials.items()}
        if self.date_span_join:
            out["date_span_join"] = True
        if self.street_number_join:
            out["street_number_join"] = True
        if self.date_identifier_exclusion:
            out["date_identifier_exclusion"] = True
        if self.signoff_floor:
            out["signoff_floor"] = dict(self.signoff_floor)
        if self.date_day_growth:
            out["date_day_growth"] = True
        if self.date_year_abbrev:
            out["date_year_abbrev"] = list(self.date_year_abbrev)
        if self.name_initial_bridge:
            out["name_initial_bridge"] = {lang: list(heads) for lang, heads in self.name_initial_bridge.items()}
        if self.name_join_cased:
            out["name_join_cased"] = {lang: list(heads) for lang, heads in self.name_join_cased.items()}
        if self.signoff_shapes:
            out["signoff_shapes"] = list(self.signoff_shapes)
        if self.person_full_span:
            out["person_full_span"] = True
        if self.age_span_repair:
            out["age_span_repair"] = True
        return out

    def initials_heads(self, lang: str | None) -> tuple[str, ...]:
        return tuple(_lookup(self.name_initials, lang) or ())

    def completed_heads(self, lang: str | None) -> tuple[str, ...]:
        return tuple(_lookup(self.word_completion, lang) or ())

    def joined_heads(self, lang: str | None) -> tuple[str, ...]:
        return tuple(_lookup(self.name_join, lang) or ())

    def head_floors(self, lang: str | None) -> dict[str, float]:
        """Per-head floors for ``lang``: the exact key, then the primary subtag (``pt-br`` -> ``pt``), then the
        language-independent default ``"*"`` (guard v13, F013) when the stamp has one. Without a ``"*"`` key a call
        with no language or an unknown language gets no per-head floor (byte-identical to guard v12)."""
        found = _lookup(self.recall_floor_by_head, lang)
        if found is None:
            found = self.recall_floor_by_head.get(DEFAULT_LANG_KEY)
        return dict(found or {})

    def bridged_heads(self, lang: str | None) -> tuple[str, ...]:
        return tuple(_lookup(self.name_initial_bridge, lang) or ())

    def cased_joined_heads(self, lang: str | None) -> tuple[str, ...]:
        return tuple(_lookup(self.name_join_cased, lang) or ())

    def year_abbrev(self, lang: str | None) -> str | None:
        """The year abbreviation of ``lang`` when ``date_year_abbrev`` names it (exact key or primary subtag)."""
        key = lang_key(lang)
        if key is None:
            return None
        primary = key.split("-", 1)[0]
        for k in (key, primary):
            if k in self.date_year_abbrev:
                return YEAR_ABBREVIATIONS[k]
        return None

    def strips_particles(self, lang: str | None) -> bool:
        key = lang_key(lang)
        if key is None:
            return False
        return key in self.particle_strip or key.split("-", 1)[0] in self.particle_strip

    def inword_gap(self, lang: str | None) -> int:
        return int(_lookup(self.inword_gap_fill, lang) or 0)

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> DecoderSettings:
        """Validate a stamp / env / request mapping. Loud on anything malformed: a typo in a
        checkpoint stamp must never degrade to a silent default."""
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"decoder settings must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - set(DECODER_SETTING_KEYS)
        if unknown:
            raise ValueError(
                f"unknown decoder setting(s) {sorted(unknown)}; known: {DECODER_SETTING_KEYS}"
            )
        floors: dict[str, dict[str, float]] = {}
        for lang, heads in (raw.get("recall_floor_by_head") or {}).items():
            if not isinstance(heads, Mapping):
                raise ValueError(f"recall_floor_by_head[{lang!r}] must map head -> floor")
            key = lang_key(lang)
            if key is None:
                raise ValueError("recall_floor_by_head: empty language key")
            per_head: dict[str, float] = {}
            for head, value in heads.items():
                floor = float(value)
                if not 0.0 < floor < 1.0:
                    raise ValueError(
                        f"recall_floor_by_head[{lang!r}][{head!r}] must be in (0, 1), got {value!r}"
                    )
                per_head[str(head).upper()] = floor
            if per_head:
                floors[key] = per_head
        strip_raw = raw.get("particle_strip") or ()
        if isinstance(strip_raw, str):
            strip_raw = [part for part in strip_raw.split(",") if part.strip()]
        strip: list[str] = []
        for lang in strip_raw:
            key = lang_key(lang)
            if key not in PARTICLE_TABLES:
                raise ValueError(
                    f"particle_strip: no particle table for {lang!r}; "
                    f"tables exist for {sorted(PARTICLE_TABLES)} (ko measured −6, not shipped)"
                )
            if key not in strip:
                strip.append(key)
        gaps: dict[str, int] = {}
        for lang, value in (raw.get("inword_gap_fill") or {}).items():
            key = lang_key(lang)
            gap = int(value)
            if key is None or gap < 1:
                raise ValueError(f"inword_gap_fill[{lang!r}] must be a gap >= 1, got {value!r}")
            gaps[key] = gap
        continuity = raw.get("span_continuity", False)
        if isinstance(continuity, str):
            continuity = continuity.strip().lower() not in _SETTINGS_OFF
        if not isinstance(continuity, bool | int):
            raise ValueError(f"span_continuity must be a boolean, got {continuity!r}")
        repair = raw.get("numeric_span_repair") or ()
        if isinstance(repair, str):
            repair = () if repair.strip().lower() in _SETTINGS_OFF else (repair,)
        repair = tuple(str(h).upper() for h in repair)
        for head in repair:
            if head not in _NUMERIC_REPAIR_HEADS:
                raise ValueError(
                    f"numeric_span_repair: {head!r} is not a numeric head; "
                    f"allowed: {sorted(_NUMERIC_REPAIR_HEADS)}"
                )
        completion = _heads_by_lang(raw.get("word_completion"), "word_completion", _WORD_COMPLETION_HEADS)
        joins = _heads_by_lang(raw.get("name_join"), "name_join")
        initials = _heads_by_lang(raw.get("name_initials"), "name_initials")
        date_join = raw.get("date_span_join", False)
        if isinstance(date_join, str):
            date_join = date_join.strip().lower() not in _SETTINGS_OFF
        if not isinstance(date_join, bool | int):
            raise ValueError(f"date_span_join must be a boolean, got {date_join!r}")
        street_join = raw.get("street_number_join", False)
        if isinstance(street_join, str):
            street_join = street_join.strip().lower() not in _SETTINGS_OFF
        if not isinstance(street_join, bool | int):
            raise ValueError(f"street_number_join must be a boolean, got {street_join!r}")
        id_excl = raw.get("date_identifier_exclusion", False)
        if isinstance(id_excl, str):
            id_excl = id_excl.strip().lower() not in _SETTINGS_OFF
        if not isinstance(id_excl, bool | int):
            raise ValueError(f"date_identifier_exclusion must be a boolean, got {id_excl!r}")
        signoff: dict[str, float] = {}
        raw_signoff = raw.get("signoff_floor") or {}
        if not isinstance(raw_signoff, Mapping):
            raise ValueError(f"signoff_floor must map head -> floor, got {raw_signoff!r}")
        for head, value in raw_signoff.items():
            floor = float(value)
            if not 0.0 < floor < 1.0:
                raise ValueError(f"signoff_floor[{head!r}] must be in (0, 1), got {value!r}")
            if str(head).upper() not in _NAME_RULE_HEADS:
                raise ValueError(
                    f"signoff_floor: {head!r} is not a name head; allowed: {sorted(_NAME_RULE_HEADS)}"
                )
            signoff[str(head).upper()] = floor
        day_growth = _flag(raw, "date_day_growth")
        abbrev_raw = raw.get("date_year_abbrev") or ()
        if isinstance(abbrev_raw, str):
            abbrev_raw = [part for part in abbrev_raw.split(",") if part.strip()]
        abbrev: list[str] = []
        for lang in abbrev_raw:
            key = lang_key(lang)
            if key not in YEAR_ABBREVIATIONS:
                raise ValueError(
                    f"date_year_abbrev: no year abbreviation for {lang!r}; tables exist for {sorted(YEAR_ABBREVIATIONS)}"
                )
            if key not in abbrev:
                abbrev.append(key)
        bridge = _heads_by_lang(raw.get("name_initial_bridge"), "name_initial_bridge", frozenset({"PERSON"}))
        cased_joins = _heads_by_lang(raw.get("name_join_cased"), "name_join_cased")
        shapes_raw = raw.get("signoff_shapes") or ()
        if isinstance(shapes_raw, str):
            shapes_raw = [part for part in shapes_raw.split(",") if part.strip()]
        shapes: list[str] = []
        for shape in shapes_raw:
            name = str(shape).strip().lower()
            if name not in SIGNOFF_SHAPES:
                raise ValueError(f"signoff_shapes: unknown shape {shape!r}; known: {SIGNOFF_SHAPES}")
            if name not in shapes:
                shapes.append(name)
        if shapes and not signoff:
            raise ValueError("signoff_shapes need signoff_floor: the shapes only add regions to the sign-off floor")
        full_span = _flag(raw, "person_full_span")
        age_repair = _flag(raw, "age_span_repair")
        return cls(floors, tuple(strip), gaps, bool(continuity), repair, completion, joins, initials, bool(date_join), bool(street_join),
                   bool(id_excl), signoff, day_growth, tuple(abbrev), bridge, cased_joins, tuple(shapes), full_span, age_repair)


def _flag(raw: Mapping, key: str) -> bool:
    """A boolean settings key: true / false, or an on / off word. Loud on anything else."""
    value = raw.get(key, False)
    if isinstance(value, str):
        value = value.strip().lower() not in _SETTINGS_OFF
    if not isinstance(value, bool | int):
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    return bool(value)


def _heads_by_lang(raw: object, key: str, allowed: frozenset[str] | None = None) -> dict[str, tuple[str, ...]]:
    allowed = _NAME_RULE_HEADS if allowed is None else allowed
    """``{lang: [HEAD, ...]}`` (or ``{lang: "HEAD,HEAD"}``) -> validated mapping; loud on
    anything else. Restricted to the name heads the rules were measured on."""
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must map language -> [HEAD, ...], got {type(raw).__name__}")
    out: dict[str, tuple[str, ...]] = {}
    for lang, heads in raw.items():
        lk = lang_key(lang)
        if lk is None:
            raise ValueError(f"{key}: empty language key")
        if isinstance(heads, str):
            heads = [h for h in heads.split(",") if h.strip()]
        names = tuple(dict.fromkeys(str(h).strip().upper() for h in heads))
        bad = [h for h in names if h not in allowed]
        if bad or not names:
            raise ValueError(
                f"{key}[{lang!r}]: heads must be non-empty and within {sorted(allowed)}, got {list(heads)!r}"
            )
        out[lk] = names
    return out


def normalise_decoder_settings(value: object) -> DecoderSettings:
    """Parse a settings value from any channel: ``None`` / an off word / ``DecoderSettings`` /
    a mapping / a JSON object string. Off = the default instance."""
    if value is None:
        return DecoderSettings()
    if isinstance(value, DecoderSettings):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _SETTINGS_OFF:
            return DecoderSettings()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"decoder settings must be a JSON object or an off word: {exc}"
            ) from exc
    if isinstance(value, Mapping):
        return DecoderSettings.from_mapping(value)
    raise ValueError(f"cannot read decoder settings from {type(value).__name__}")


def resolve_decoder_settings(meta: dict | None) -> tuple[DecoderSettings, dict[str, str]]:
    """The settings a predictor starts with, and where each key came from (for the audit).

    Precedence per key, mirroring ``default_recall_floor``: ``SHINRAI_DECODER_SETTINGS``
    (harness-wide override — a JSON object whose keys REPLACE the stamped keys; a key set
    to ``null`` / ``"off"`` turns that key off; the bare word ``off`` turns everything off) >
    the checkpoint's decoder block (``config.json`` -> ``shinrai`` -> ``decoder`` -> the four
    keys, written by tools/stamp_decoder) > off. A per-call ``predict(decoder_settings=...)``
    still wins over all of these."""
    decoder = (meta or {}).get("decoder") or {}
    stamped = {
        k: decoder[k]
        for k in DECODER_SETTING_KEYS
        if decoder.get(k) not in (None, {}, [], False)
    }
    sources = {
        k: ("checkpoint_decoder_stamp" if k in stamped else "off_default")
        for k in DECODER_SETTING_KEYS
    }
    merged: dict = dict(stamped)
    env = os.environ.get(DECODER_SETTINGS_ENV)
    if env is not None and env.strip():
        text = env.strip()
        if text.lower() in _SETTINGS_OFF:
            merged = {}
            sources = {k: "env" for k in DECODER_SETTING_KEYS}
        else:
            try:
                override = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{DECODER_SETTINGS_ENV} must be a JSON object or 'off': {exc}"
                ) from exc
            if not isinstance(override, Mapping):
                raise ValueError(f"{DECODER_SETTINGS_ENV} must be a JSON object, got {text!r}")
            for key, value in override.items():
                if key not in DECODER_SETTING_KEYS:
                    raise ValueError(f"{DECODER_SETTINGS_ENV}: unknown key {key!r}")
                off_word = isinstance(value, str) and value.strip().lower() in _SETTINGS_OFF
                if value is None or off_word:
                    merged.pop(key, None)
                else:
                    merged[key] = value
                sources[key] = "env"
    return DecoderSettings.from_mapping(merged), sources


def default_decoder_settings(meta: dict | None) -> DecoderSettings:
    """``resolve_decoder_settings`` without the sources — the predictor constructors' entry."""
    return resolve_decoder_settings(meta)[0]


def check_settings_heads(settings: DecoderSettings, heads: list[str] | tuple[str, ...]) -> None:
    """A per-head floor on a head the checkpoint does not have is a stamp error — loud."""
    known = set(heads)
    unknown_signoff = set(settings.signoff_floor) - known
    if unknown_signoff:
        raise ValueError(
            f"signoff_floor names head(s) {sorted(unknown_signoff)} unknown to this checkpoint ({sorted(known)})"
        )
    for lang, per_head in settings.recall_floor_by_head.items():
        unknown = set(per_head) - known
        if unknown:
            raise ValueError(
                f"recall_floor_by_head[{lang!r}] names head(s) {sorted(unknown)} "
                f"unknown to this checkpoint ({sorted(known)})"
            )


def head_recall_floor(
    settings: DecoderSettings | None, lang: str | None, head: str, recall_floor: float | None
) -> float | None:
    """The floor one head decodes with: the per-language per-head floor when ``lang`` names a
    language that stamps this head, else the global floor. Layered, not replacing: every other
    head of a ja text keeps the global 0.35."""
    if settings is None:
        return recall_floor
    per_head = settings.head_floors(lang)
    return per_head.get(head, recall_floor) if per_head else recall_floor


def is_partial_token(text: str, pos: int) -> bool:
    """True when ``pos`` cuts inside a word — both neighbours are letters, digits or ``_``.
    Verbatim the medical anatomy's class test (error_anatomy.py ``is_partial_token``)."""
    if pos <= 0 or pos >= len(text):
        return False
    a, b = text[pos - 1], text[pos]
    return (a.isalnum() or a == "_") and (b.isalnum() or b == "_")


# Scripts written WITHOUT whitespace word boundaries: kana, han (unified, ext A, compat, 々〆ー),
# hangul syllables. Under ``str.isalnum`` every character boundary in such text is a
# "sub-word cut", so the ungated join reproduced the ja in-word gap-fill loss (gold-v2 ja
# −34 hits, results/r1-ungated-s4/, 2026-09-08). The medical class was counted on Latin-script
# corpora only; a cut next to one of these characters is not the measured class.
_NO_SUBWORD_CUT = re.compile(
    "[\u3005\u3006\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)


def is_subword_cut(text: str, pos: int) -> bool:
    """A ``partial_token`` cut between two characters of a script that writes word boundaries
    (the measured class); never inside kana / han / hangul text."""
    return is_partial_token(text, pos) and not (
        _NO_SUBWORD_CUT.match(text[pos - 1]) or _NO_SUBWORD_CUT.match(text[pos])
    )


def join_split_spans(entities: list[dict], text: str) -> list[dict]:
    """L6 span continuity (ERROR-ANATOMY §5, D21): re-join two same-type predictions that are
    fragments of ONE written word. The gap between them (possibly empty) consists of word
    characters only, and both cut positions are ``partial_token`` cuts — the class the medical
    agents counted (MEDDOCAN 68 pairs, GraSCCo 19, ceiling +2.94 / +4.2 strict). Whitespace or
    punctuation in the gap is NOT a sub-word split and is never bridged (no wider rule); a cut
    inside kana / han / hangul text is not the measured class either (``is_subword_cut``).
    Pure function over a start-sorted list; confidence = mean of the parts."""
    if not entities:
        return entities
    out = [dict(entities[0])]
    for e in entities[1:]:
        a = out[-1]
        if e["type"] == a["type"] and e["span"][0] >= a["span"][1]:
            gap = text[a["span"][1] : e["span"][0]]
            if (
                all((c.isalnum() or c == "_") and not _NO_SUBWORD_CUT.match(c) for c in gap)
                and is_subword_cut(text, a["span"][1])
                and is_subword_cut(text, e["span"][0])
            ):
                confs = [a["confidence"], e["confidence"]]
                a["span"] = [a["span"][0], e["span"][1]]
                a["text"] = text[a["span"][0] : a["span"][1]]
                a["confidence"] = round(sum(confs) / len(confs), 6)
                if e.get("evidence") == "floor":
                    a["evidence"] = "floor"
                continue
        out.append(dict(e))
    return out


def strip_particles(entities: list[dict], text: str, lang: str | None) -> list[dict]:
    """Rule d (taskb_model_probe.py ``rule_strip_particle``), keyed by ``lang``: one trailing
    particle from the language's table is trimmed off a PERSON / ORG / CITY prediction when at
    least two characters remain. Longest particle first, one strip per span. No table for the
    language = byte-identical. The kana-name guard (``kana_name_guard``) keeps a PERSON span
    whose hiragana-final name merely ends in a particle character («永海みか»)."""
    table = PARTICLE_TABLES.get(lang_key(lang) or "")
    if not table:
        primary = (lang_key(lang) or "").split("-", 1)[0]
        table = PARTICLE_TABLES.get(primary)
    if not table or not entities:
        return entities
    out: list[dict] = []
    for ent in entities:
        if ent["type"] in PARTICLE_STRIP_TYPES:
            s, t = ent["span"]
            surface = text[s:t]
            for particle in table:
                remainder = len(surface) - len(particle)
                if surface.endswith(particle) and remainder >= PARTICLE_STRIP_MIN_REMAINDER:
                    if not kana_name_guard(surface, particle, ent["type"]):
                        t -= len(particle)
                        ent = {**ent, "span": [s, t], "text": text[s:t]}
                    break
        out.append(ent)
    return out


# D-N1 (2026-09-21): a 21-head arm finds an inline phone number but truncates its tail
# («+49 / 151 4568290» -> «+49 / 151», «5», «829»), so the memcheck frame criterion fails at 0.40
# while the in-sentence, label_row, native_label and bare frames are all 1.0. A partial span is a
# partial redaction, i.e. a leak. This joins same-head numeric spans separated only by number glue
# and then extends both ends over the number's own characters; it stops at a letter, at a double
# space and at anything that is not glue, so it can only ever grow a span within one number.
# only heads whose value IS a number may be repaired this way
_NUMERIC_REPAIR_HEADS = frozenset({"PHONE", "ACCOUNT", "CARD", "NATIONAL_ID", "POSTAL_CODE", "CUSTOMER_ID", "PLATE"})
_NUM_GLUE = re.compile(r"^[\s0-9()/.+-]{0,4}$")
_NUM_RUN = r"[0-9()/.+-]+(?: ?[0-9()/.+-]+)*"
_NUM_LEFT = re.compile(_NUM_RUN + r"$")
_NUM_RIGHT = re.compile(r"^" + _NUM_RUN)
_NUM_REACH = 24


def repair_numeric_spans(entities: list[dict], text: str, head: str) -> list[dict]:
    """Join and extend the spans of one numeric head over the digits they sit in."""
    mine = sorted((e for e in entities if e.get("type") == head), key=lambda e: tuple(e["span"]))
    if not mine:
        return entities
    joined: list[dict] = []
    for e in mine:
        s, t = e["span"]
        if joined and _NUM_GLUE.match(text[joined[-1]["span"][1]:s]):
            prev = joined[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], t)]
            prev["confidence"] = max(prev.get("confidence", 0.0), e.get("confidence", 0.0))
        else:
            joined.append(dict(e, span=[s, t]))
    for e in joined:
        s, t = e["span"]
        left = _NUM_LEFT.search(text[max(0, s - _NUM_REACH):s])
        if left and left.group(0).strip():
            s = max(0, s - _NUM_REACH) + left.start()
        right = _NUM_RIGHT.match(text[t:t + _NUM_REACH])
        if right and right.group(0).strip():
            t += right.end()
        e["span"] = [s, max(t, s)]
        e["text"] = text[s:e["span"][1]]
    rest = [e for e in entities if e.get("type") != head]
    return sorted(rest + joined, key=lambda e: tuple(e["span"]))


# --- N1 / N2: name span assembly (G10 readout §1.1, 2026-09-22) ------------------------
# On MultiNERD pl/ru more than half of the strict PERSON/ORG false positives are pieces of
# one name: sub-word cuts inside an inflected surname («iewskiego» ← Żółkiewskiego, «ри» ←
# Гири), a multi-token name split in two («Питер» + «Тэт»), a dropped initial («Чевакинского»
# ← С. И. Чевакинского). Each piece is one FP and one FN. Both rules are pure functions over
# the decoded list, keyed per language and per head, off unless stamped.
_NAME_RULE_HEADS = frozenset({"PERSON", "ORG", "CITY", "STREET"})
# guard v8 (2026-09-26, G12b s43 on GraSCCo): the adjectival age token («28-jährigen», «80 jährige») is a written word
# the tokenizer cuts («80 jäh», «6-jah», «5» from «57»); word completion may grow AGE spans too. name_join /
# name_initials stay name-only.
_WORD_COMPLETION_HEADS = _NAME_RULE_HEADS | {"AGE"}


def _is_word_char(ch: str) -> bool:
    return (ch.isalnum() or ch == "_") and not _NO_SUBWORD_CUT.match(ch)


def complete_word_spans(entities: list[dict], text: str, heads: tuple[str, ...]) -> list[dict]:
    """N1 word completion: a span of a listed head whose start or end is a sub-word cut
    (``is_subword_cut``) grows to the boundaries of the written word it sits in. Spans of the
    same type that overlap after growing are merged (confidence = max). Other heads pass
    through untouched; kana / han / hangul text is never touched (no sub-word cut there)."""
    if not entities or not heads:
        return entities
    grown: list[dict] = []
    for e in entities:
        e = dict(e)
        if e.get("type") in heads:
            s, t = e["span"]
            while is_subword_cut(text, s) and s > 0 and _is_word_char(text[s - 1]):
                s -= 1
            while is_subword_cut(text, t) and t < len(text) and _is_word_char(text[t]):
                t += 1
            if (s, t) != tuple(e["span"]):
                e["span"] = [s, t]
                e["text"] = text[s:t]
        grown.append(e)
    grown.sort(key=lambda e: (e["span"][0], e["span"][1]))
    out: list[dict] = []
    for e in grown:
        a = out[-1] if out else None
        if a and a["type"] == e["type"] and a["type"] in heads and e["span"][0] < a["span"][1]:
            a["span"] = [a["span"][0], max(a["span"][1], e["span"][1])]
            a["text"] = text[a["span"][0] : a["span"][1]]
            a["confidence"] = max(a.get("confidence", 0.0), e.get("confidence", 0.0))
            continue
        out.append(e)
    return out


# The gap between two pieces of one name: whitespace around at most one connector — a run of
# initials («С. И.», «A.»), a hyphen / dash, or one capitalised word (a middle name, a nobiliary
# particle written capitalised). Commas, conjunctions and lower-case words are never bridged.
_NAME_GAP = re.compile(
    r"^\s*(?:(?:[^\W\d_]\.\s*)+|[-‐‑–]|[^\W\d_]{1,20}"
    r"|[’'`]\s*[^\W\d_]{0,4})?\s*$"   # pl apostrophe genitive: «George ’ a McGoverna», «Terry ' ego»
)
_NAME_GAP_MAX = 24
# a run of initials directly before a PERSON span («С. И. Чевакинского», «A. Craig»): at most three,
# each one letter + period, the run separated from the name by whitespace only
_INITIALS_LEFT = re.compile(r"(?:(?<![^\W\d_])[^\W\d_]\.\s*){1,3}$")


def absorb_initials(entities: list[dict], text: str, heads: tuple[str, ...]) -> list[dict]:
    """N2b: a span of a listed head preceded by a run of initials grows left over them. Pure
    function; the initials must not be the tail of a longer word (``Dr.`` is not an initial)."""
    if not entities or not heads:
        return entities
    out: list[dict] = []
    for e in entities:
        e = dict(e)
        if e.get("type") in heads:
            s = e["span"][0]
            m = _INITIALS_LEFT.search(text[max(0, s - 16):s])
            if m and m.group(0).strip():
                s = max(0, s - 16) + m.start()
                e["span"] = [s, e["span"][1]]
                e["text"] = text[s:e["span"][1]]
        out.append(e)
    return out


def _starts_upper(surface: str) -> bool:
    first = next((ch for ch in surface if ch.isalpha()), "")
    return bool(first) and first.isupper()


def join_name_spans(entities: list[dict], text: str, heads: tuple[str, ...], cased: bool = False) -> list[dict]:
    """N2 name join: two consecutive spans of the same listed head whose gap is whitespace
    plus at most one connector (initials, hyphen, one capitalised token) become one span.
    The gap must contain at least one whitespace or hyphen character — an empty gap is the
    sub-word class of ``join_split_spans`` and stays with that rule.

    ``cased`` (guard v13 ``name_join_cased``, F012): both pieces must start with an upper-case letter, so a
    lower-case title the model tagged on its own («general», «rei», «imperador» before a name) is never joined."""
    if len(entities) < 2 or not heads:
        return entities
    ordered = sorted((dict(e) for e in entities), key=lambda e: (e["span"][0], e["span"][1]))
    out = [ordered[0]]
    for e in ordered[1:]:
        a = out[-1]
        if e["type"] == a["type"] and e["type"] in heads and e["span"][0] >= a["span"][1]:
            gap = text[a["span"][1] : e["span"][0]]
            connector = gap.strip()
            if (
                0 < len(gap) <= _NAME_GAP_MAX
                and (gap != connector or connector in "-‐‑–")
                and _NAME_GAP.match(gap)
                and (not connector or not connector[0].islower())
                and (not cased or (_starts_upper(a["text"]) and _starts_upper(e["text"])))
            ):
                a["span"] = [a["span"][0], e["span"][1]]
                a["text"] = text[a["span"][0] : a["span"][1]]
                a["confidence"] = round(
                    (a.get("confidence", 0.0) + e.get("confidence", 0.0)) / 2, 6
                )
                if e.get("evidence") == "floor":
                    a["evidence"] = "floor"
                continue
        out.append(e)
    return out


# --- D-N2: DATE span assembly (v1.4-vs-v1.5 comparison §3.2, 2026-09-24) ------------------
# On the 146-prompt chat test the 21-head candidate finds 123 of 301 DATE occurrences exactly and 87 as
# FRAGMENTS («January 2024» → «2024», «15. März 2024» → «März 2024»); on legacy register text the same head
# emits «2026», «03», «12» for one «2026-03-12». The pieces are one date. Rule: DATE spans separated by
# date glue (whitespace, . / - , and one lowercase connector such as «of», «de», «года») join, and a DATE
# span grows over adjacent day / month / year tokens. Month names come from the temporal-negatives
# module (the same 15-locale inventory the training rule uses). Language-agnostic like span continuity;
# never touches another head.
from . import temporal_negatives as _tn

# A month ABBREVIATION counts only with its period, as in the training rule (temporal_negatives): «sie» (pl, August)
# is the German pronoun «Sie», «mar» (es) the sea — without the period the grow step swallowed the next sentence's
# first word («12. März 2026. Sie erreichen» → DATE «12. März 2026. Sie»; v1.5.3 rollout, 2026-10-01).
_DATE_WORD = re.compile(
    r"(?i)^(?:" + _tn._MONTH_FULL[3:-1] + r"|(?:" + _tn._MONTH_ABBR[3:-1] + r")\."
    r"|\d{1,2}(?:st|nd|rd|th)?\.?|\d{4}|\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
    r"|г\.|року|года|году|r\.|roku|de|of|the|del|di|du|le|den|am|im|el|à)$"
)
_DATE_GLUE = re.compile(r"^[\s./,\-]{0,3}(?:(?:de|of|the|del|di|du|le|den|am|im|el|à|г\.|года|году|року|r\.|roku)[\s./,\-]{0,3})?$", re.I)
_DATE_TOKEN = re.compile(r"[^\s,;:()\[\]«»\"'./\-]+")
_DATE_REACH = 3   # tokens grown on each side at most


_DATE_CONNECTORS = {"de", "of", "the", "del", "di", "du", "le", "den", "am", "im", "el", "à", "г.", "года", "году", "року", "r.", "roku"}
_DATE_MONTH = re.compile(r"(?i)^(?:" + _tn._MONTH_FULL[3:-1] + r"|(?:" + _tn._MONTH_ABBR[3:-1] + r")\.)$")   # abbreviation + period only
_DATE_DAY = re.compile(r"^\d{1,2}(?:st|nd|rd|th)?\.?$")
_DATE_STRONG = re.compile(r"^(?:\d{4}|\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})$")


def _next_token(text: str, pos: int):
    """(token, end) after the glue at pos; a trailing period stays on the token («Mär.», «r.», «г.»), so dotted month
    abbreviations and connectors are recognised as such (bare abbreviations are not months: «Sie», «mar»)."""
    m = re.match(r"[\s./,\-]{0,3}", text[pos:]); gap = m.group(0)
    tok = _DATE_TOKEN.match(text[pos + len(gap):])
    if not tok:
        return None, pos
    end = pos + len(gap) + tok.end(); word = tok.group(0)
    if text[end:end + 1] == "." and re.search(r"[^\W\d_]", word):   # letters only: «2023.» keeps its sentence period outside
        word, end = word + ".", end + 1
    return word, end


def _prev_token(text: str, pos: int):
    """(token, start) before the glue at pos; a period right after the token stays on it (see _next_token)."""
    left = text[max(0, pos - 40):pos]
    m = re.search(r"([^\s,;:()\[\]«»\"'./\-]+)(\.?)([\s./,\-]{0,3})$", left)
    if not m:
        return None, pos
    word = m.group(1) + (m.group(2) if re.search(r"[^\W\d_]", m.group(1)) else "")
    return word, max(0, pos - 40) + m.start(1)


def _grow_right(text: str, t: int):
    tok, end = _next_token(text, t)
    if tok is None:
        return None
    low = tok.lower()
    if low in _DATE_CONNECTORS:
        tok2, end2 = _next_token(text, end)
        if tok2 and (_DATE_MONTH.match(tok2) or _DATE_STRONG.match(tok2)):
            return end2
        return None
    if _DATE_MONTH.match(tok) or _DATE_STRONG.match(tok):
        return end
    return None


def _grow_left(text: str, s: int):
    tok, start = _prev_token(text, s)
    if tok is None:
        return None
    low = tok.lower()
    if low in _DATE_CONNECTORS:
        tok2, start2 = _prev_token(text, start)
        if tok2 and (_DATE_MONTH.match(tok2) or _DATE_STRONG.match(tok2) or _DATE_DAY.match(tok2)):
            return start2
        return None
    if _DATE_MONTH.match(tok) or _DATE_STRONG.match(tok):
        return start
    if _DATE_DAY.match(tok):
        head, _ = _next_token(text, s)                                  # the span's first token, period kept («Mär.»)
        numeric_tail = re.match(r"\d{1,2}[./-]\d{2,4}", text[s:])   # «05.2024» ← «12.05.2024»
        return start if (head and (_DATE_MONTH.match(head) or head.lower() in _DATE_CONNECTORS)) or numeric_tail else None
    return None


def join_date_spans(entities: list[dict], text: str) -> list[dict]:
    """D-N2: join DATE fragments over date glue and grow a DATE span over adjacent date tokens."""
    mine = sorted((e for e in entities if e.get("type") == "DATE"), key=lambda e: tuple(e["span"]))
    if not mine:
        return entities
    joined: list[dict] = []
    for e in mine:
        s, t = e["span"]
        if joined and _DATE_GLUE.match(text[joined[-1]["span"][1]:s]) and s - joined[-1]["span"][1] <= 12:
            prev = joined[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], t)]
            prev["confidence"] = round((prev.get("confidence", 0.0) + e.get("confidence", 0.0)) / 2, 6)
        else:
            joined.append(dict(e, span=[s, t]))
    for e in joined:
        s, t = e["span"]
        while is_subword_cut(text, s) and s > 0 and _is_word_char(text[s - 1]):   # «5, 2023» ← «August 15, 2023»
            s -= 1
        while is_subword_cut(text, t) and t < len(text) and _is_word_char(text[t]):
            t += 1
        m = re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?", text[s:])
        if m:                                  # «30» + «.2025» ← «30.04.2025»; ISO stamps
            t = max(t, s + m.end())
        for _ in range(_DATE_REACH):          # grow right: [connector] date-word; a bare 1-2 digit token never
            nt = _grow_right(text, t)
            if nt is None:
                break
            t = nt
        for _ in range(_DATE_REACH):          # grow left: date-word [connector]; a bare day only before a month
            ns = _grow_left(text, s)
            if ns is None:
                break
            s = ns
        e["span"] = [s, t]
        e["text"] = text[s:t]
    joined.sort(key=lambda e: tuple(e["span"]))
    merged: list[dict] = []
    for e in joined:                          # grown spans may now overlap a sibling fragment: one date, one span
        if merged and e["span"][0] < merged[-1]["span"][1]:
            prev = merged[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], e["span"][1])]
            prev["text"] = text[prev["span"][0]:prev["span"][1]]
            prev["confidence"] = max(prev.get("confidence", 0.0), e.get("confidence", 0.0))
            continue
        merged.append(e)
    rest = [e for e in entities if e.get("type") != "DATE"]
    return sorted(rest + merged, key=lambda e: tuple(e["span"]))


# --- D-N3: STREET + house number (x2 chat test, 2026-09-24) ------------------------------
# 11 of 82 gold streets on the x2 test were the street name without its house number, or the number alone
# («18» + «Rua do Almada», «Hauptstraße» without «5»). A house number is 1-4 digits with an optional letter or
# «/12» suffix, glued to the street by at most one space (or «, » / « nr. »); five-digit runs are postal codes
# and never taken. Language-agnostic like continuity: the number sits right of the street (de, es, it, pl, ru)
# or left of it (en, fr, pt); both sides are tried.
_HOUSE_RIGHT = re.compile(r"^(?:[ ,]{1,2}|\s(?:nr\.?|no\.?|n°|nº)\s?)(\d{1,4}[a-zA-Z]?(?:[/-]\d{1,3})?)(?!\d|[.,:%-]\d)")
_HOUSE_LEFT = re.compile(r"(?<![\d.,:%/-])(\d{1,4}[a-zA-Z]?)[ ,]{1,2}$")


# D-N4 (2026-09-28, legal-label-policy-v1 Q1 class B): year digits that are a component of an identifier string are
# not DATE. MAPA fr / de showed the model tagging them (G14 s43: 21 of 27 fr and 23 of 44 de extra DATE spans were
# ECLI / case-number years). Spacing-tolerant: MAPA text is pre-tokenised («EU : C : 1988 : 322»). Class A years with a
# publication cue («OJ 2002 L 190», «ABl. 2010, L 338», «judgment of 17 May 1990») are not matched and stay DATE.
_SP = r"\s*"
_DASH = r"[-‑–]"
_IDENTIFIER_PATTERNS = tuple(re.compile(p) for p in (
    r"\bECLI" + _SP + ":" + _SP + r"[A-Z]{2}" + _SP + ":" + _SP + r"[A-Z0-9.]{1,20}" + _SP + ":" + _SP + r"\d{4}" + _SP + ":" + _SP + r"[\w.]+",
    r"\b(?:EU|UE)" + _SP + ":" + _SP + r"[CTF]" + _SP + ":" + _SP + r"\d{4}" + _SP + ":" + _SP + r"\d+\b",
    r"\b\d{1,4}" + _SP + "/" + _SP + r"\d{1,4}" + _SP + "/" + _SP + r"(?:EC|EU|CE|EG|UE|EWG|EEC|CEE|EGKS|Euratom|JI|GASP|PESC)\b",
    r"(?:\bNo\.?|\bNr\.?|\bn°|\bnº|\bno\.|\bn\.|\bnr)" + _SP + r"\d{1,6}" + _SP + "/" + _SP + r"\d{2,4}\b(?:" + _SP + _DASH + _SP + r"\d{1,3}\b)?",
    r"\b[CT]" + _SP + _DASH + _SP + r"\d{1,4}" + _SP + "/" + _SP + r"\d{2}\b(?:" + _SP + r"(?:P|R|RENV|DEP|SA|AJ)\b)?",
    r"\(" + _SP + r"(?:[CT]" + _SP + _DASH + _SP + r")?\d{1,4}" + _SP + "/" + _SP + r"\d{2,4}" + _SP + "," + _SP + r"(?:ECLI" + _SP + ":" + _SP + r")?(?:EU|UE)" + _SP + ":",
    r"\b[A-Z]{1,2}\s+\d{1,5}" + _SP + "/" + _SP + r"\d{4}" + _SP + _DASH + _SP + r"\d{1,3}\b",
    r"\((?:EU|EG|UE|CE|EC|EWG|EEC|CEE)\)" + _SP + r"(?:No\.?|Nr\.?|n°|nº)?" + _SP + r"\d{2,4}" + _SP + "/" + _SP + r"\d{1,4}\b",
))


def identifier_regions(text: str) -> list[tuple[int, int]]:
    """Character regions of identifier strings whose digits are never a DATE (policy class B)."""
    return sorted({(m.start(), m.end()) for p in _IDENTIFIER_PATTERNS for m in p.finditer(text)})


def drop_identifier_dates(entities: list[dict], text: str) -> list[dict]:
    """D-N4: remove every DATE span that lies inside an identifier region (ECLI, instrument or case number)."""
    if not any(e.get("type") == "DATE" for e in entities):
        return entities
    regions = identifier_regions(text)
    if not regions:
        return entities
    return [e for e in entities
            if e.get("type") != "DATE" or not any(a <= e["span"][0] and e["span"][1] <= b for a, b in regions)]


def join_street_numbers(entities: list[dict], text: str) -> list[dict]:
    """D-N3: grow every STREET span over an adjacent house number; merge STREET spans that then overlap."""
    mine = sorted((dict(e) for e in entities if e.get("type") == "STREET"), key=lambda e: tuple(e["span"]))
    if not mine:
        return entities
    for e in mine:
        s, t = e["span"]
        m = _HOUSE_RIGHT.match(text[t:t + 16])
        if m:
            t = t + m.end()
        else:
            m = _HOUSE_LEFT.search(text[max(0, s - 10):s])
            if m:
                s = max(0, s - 10) + m.start(1)
        e["span"] = [s, t]
        e["text"] = text[s:t]
    merged: list[dict] = []
    for e in mine:
        if merged and e["span"][0] <= merged[-1]["span"][1]:
            prev = merged[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], e["span"][1])]
            prev["text"] = text[prev["span"][0]:prev["span"][1]]
            prev["confidence"] = max(prev.get("confidence", 0.0), e.get("confidence", 0.0))
            continue
        merged.append(e)
    rest = [e for e in entities if e.get("type") != "STREET"]
    return sorted(rest + merged, key=lambda e: tuple(e["span"]))


# --- guard v13 (2026-10-03): F014 day growth, F002 year abbreviation, F011 initial bridge, F016 full-name type ------
# F014: with the 0.99 DATE floor (argmax only) the day token of «am 15.10.2026» can stay O: DATE «.10.2026». The span
# grows left over the 1-2 digit day when the result is a numeric date with valid day / month values.
_NUMERIC_DATE = re.compile(r"^(\d{1,2})([./-])(\d{1,2})\2(\d{4}|\d{2})$")
_DAY_BEFORE_SEP = re.compile(r"(?<![\w.,/:-])\d{1,2}$")          # «15» before «.10.2026»
_DAY_SEP_BEFORE_DIGIT = re.compile(r"(?<![\w.,/:-])\d{1,2}[./-]$")   # «15.» before «10.2026»


def _valid_numeric_date(surface: str) -> bool:
    m = _NUMERIC_DATE.match(surface)
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(3))
    return (1 <= a <= 31 and 1 <= b <= 12) or (1 <= a <= 12 and 1 <= b <= 31)


def grow_date_day_left(entities: list[dict], text: str) -> list[dict]:
    """F014: a DATE span that starts at a date separator («.10.2026») or right after one («10.2026» after «15.»)
    grows left over the day number when the grown surface is a valid numeric date. Overlapping DATE spans merge."""
    mine = sorted((dict(e) for e in entities if e.get("type") == "DATE"), key=lambda e: tuple(e["span"]))
    if not mine:
        return entities
    for e in mine:
        s, t = e["span"]
        if s <= 0:
            continue
        if text[s] in "./-":
            m = _DAY_BEFORE_SEP.search(text[max(0, s - 3):s])
        elif text[s].isdigit() and text[s - 1] in "./-":
            m = _DAY_SEP_BEFORE_DIGIT.search(text[max(0, s - 4):s])
        else:
            m = None
        if not m:
            continue
        ns = s - len(m.group(0))
        if _valid_numeric_date(text[ns:t]):
            e["span"] = [ns, t]
            e["text"] = text[ns:t]
    merged: list[dict] = []
    for e in mine:
        if merged and e["span"][0] < merged[-1]["span"][1]:
            prev = merged[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], e["span"][1])]
            prev["text"] = text[prev["span"][0]:prev["span"][1]]
            prev["confidence"] = max(prev.get("confidence", 0.0), e.get("confidence", 0.0))
            continue
        merged.append(e)
    rest = [e for e in entities if e.get("type") != "DATE"]
    return sorted(rest + merged, key=lambda e: tuple(e["span"]))


# F002: Polish writes «12 marca 2026 r.»; the gold takes the abbreviation with its period. The model ends the span at
# «r» or at the year. ru «г.» and uk «р.» have the same shape (tables kept for a later measurement).
YEAR_ABBREVIATIONS: dict[str, str] = {"pl": "r", "ru": "г", "uk": "р"}


def complete_year_abbrev(entities: list[dict], text: str, abbrev: str) -> list[dict]:
    """F002: a DATE span ending in the year abbreviation («… 2026 r») takes the period right after it; a DATE span
    ending in a 4-digit year followed by « r.» takes « r.». The period must not start a number («r.5»)."""
    after_abbrev = re.compile(r"(?<=[\s\d])" + re.escape(abbrev) + r"$")
    after_year = re.compile(r"^\s?" + re.escape(abbrev) + r"\.(?![^\W_])")
    out: list[dict] = []
    for e in entities:
        if e.get("type") == "DATE":
            s, t = e["span"]
            surface = text[s:t]
            nt = t
            if after_abbrev.search(surface) and text[t:t + 1] == "." and not text[t + 1:t + 2].isalnum():
                nt = t + 1
            elif re.search(r"(?<!\d)\d{4}$", surface):
                m = after_year.match(text[t:t + 4])
                if m:
                    nt = t + m.end()
            if nt != t:
                e = {**e, "span": [s, nt], "text": text[s:nt]}
        out.append(e)
    return out


# F011: «John A. Smith» decodes PERSON «A.» only (John 0.09, Smith 0.05 entity mass): the floor cannot reach the name,
# but a PERSON span that is nothing but initials names a person whose given name and surname are its neighbours.
_ONLY_INITIALS = re.compile(r"^(?:[^\W\d_]\.\s*){1,3}$")
_CAP_WORD_LEFT = re.compile(r"(?<![^\W\d_])([^\W\d_][^\W\d_'’-]{1,29})[ \t ]{1,2}$")
# capitalised words that sit next to a name but are not part of it: titles, greetings, sentence starters (en fr es it pt de)
BRIDGE_STOPWORDS = frozenset("""
mr mrs ms miss dr prof sir madam dear hi hello hey contact please thanks thank call email ask from to for by with
the this that these those and or but if when on in at of per via cc attn re fw fwd regards
herr frau hallo liebe lieber sehr von an mit und der die das
m mme mlle monsieur madame cher chère bonjour merci pour par avec le la les et de du des
sr sra srta don doña señor señora estimado estimada hola gracias para por con el los las y del
sig signor signora dott dottor dottoressa gentile caro cara ciao grazie il lo gli e di da
senhor senhora prezado prezada olá obrigado obrigada o os ou do da dos das em no na
""".split())
_CAP_WORD_RIGHT = re.compile(r"^[ \t ]{1,2}([^\W\d_][^\W\d_'’]{1,29}(?:-[^\W\d_][^\W\d_'’]{1,29})?)(?![^\W\d_])")


def bridge_initials(entities: list[dict], text: str, heads: tuple[str, ...]) -> list[dict]:
    """F011: a span of a listed head that consists only of initials («A.», «J. R.») grows right over one capitalised
    word (the surname) and, when present, left over one capitalised word (the given name). Both neighbours must start
    with an upper-case letter, must sit on the same line, must not be a title, greeting or sentence starter
    (``BRIDGE_STOPWORDS``) and must not lie inside any other predicted span. Without a right neighbour the span stays
    as it is."""
    if not entities or not heads:
        return entities
    out: list[dict] = []
    for e in entities:
        if e.get("type") in heads and _ONLY_INITIALS.match(e.get("text") or text[e["span"][0]:e["span"][1]]):
            s, t = e["span"]
            right = _CAP_WORD_RIGHT.match(text[t:t + 64])
            if right and right.group(1)[0].isupper() and right.group(1).casefold() not in BRIDGE_STOPWORDS:
                nt = t + right.end()
                ns = s
                left = _CAP_WORD_LEFT.search(text[max(0, s - 32):s])
                if left and left.group(1)[0].isupper() and left.group(1).casefold() not in BRIDGE_STOPWORDS:
                    ns = max(0, s - 32) + left.start(1)
                # the bridge only fills words the model left O: a neighbour inside any other span («M.» + PERSON
                # «Chauvel», a CITY) is never taken; joining two existing spans is name_join's job
                clash = any(
                    o is not e and o["span"][0] < nt and ns < o["span"][1]
                    for o in entities
                )
                if not clash:
                    e = {**e, "span": [ns, nt], "text": text[ns:nt]}
        out.append(e)
    out.sort(key=lambda x: (x["span"][0], x["span"][1]))
    merged: list[dict] = []
    for e in out:                             # a grown span may now cover a sibling piece of the same name
        a = merged[-1] if merged else None
        if a and a["type"] == e["type"] and e["type"] in heads and e["span"][0] < a["span"][1]:
            a["span"] = [a["span"][0], max(a["span"][1], e["span"][1])]
            a["text"] = text[a["span"][0]:a["span"][1]]
            a["confidence"] = max(a.get("confidence", 0.0), e.get("confidence", 0.0))
            continue
        merged.append(dict(e))
    return merged


# F016: the adapter types a PERSON span from the name_part attribute head; «Jan Kowalski» / «דוד לוי» come out SURNAME.
# A span of two or more words is a full name whatever the attribute says: the decoder marks it, the adapter maps it to
# the ``full`` API type (PERSON).
FULL_NAME_SHAPE = "full"


def mark_full_person_spans(entities: list[dict], text: str) -> list[dict]:
    """F016: every PERSON span with at least two whitespace-separated words that contain a letter gets
    ``name_shape: "full"``. No other change."""
    out: list[dict] = []
    for e in entities:
        if e.get("type") == "PERSON":
            surface = e.get("text") or text[e["span"][0]:e["span"][1]]
            if sum(1 for part in surface.split() if any(ch.isalpha() for ch in part)) >= 2:
                e = {**e, "name_shape": FULL_NAME_SHAPE}
        out.append(e)
    return out


# F021 (guard v13): under the 0.99 AGE floor (argmax only) the AGE head splits an age into number and unit pieces
# («58 anos» -> «anos» + «5», «7 meses» -> «meses», «54 anos» -> nothing); the 0.35 floor used to rescue the whole span.
# The gold convention of every AGE instrument is number + unit («59 anos», «34-year-old», «45-jähriger», «45 lat»,
# «35 years»); «old» / «alt» stay outside (the 27-prompt gold has «49 years» before «old»). Number words («trente-sept
# ans») are not covered. Units are written in the language of the text, so the table is one list for every locale.
AGE_UNITS = (
    # adjectival forms keep their inflection (word tail)
    r"years?-old", r"months?-old", r"weeks?-old", r"jährig\w*", r"monatig\w*", r"letni\w*", r"miesięczn\w*",
    r"летн\w*", r"річн\w*", r"місячн\w*",
    # pt es it fr de en pl ru uk tr ar he
    r"anos", r"ano", r"meses", r"mês", r"dias", r"dia", r"semanas", r"semana",
    r"años", r"año", r"días", r"día",
    r"anni", r"anno", r"mesi", r"mese", r"giorni", r"settimane",
    r"ans", r"mois", r"jours", r"semaines",    # not «an»: the German preposition («mit 57 an Pankreaskarzinom»)
    r"jahren", r"jahre", r"jahr", r"monaten", r"monate", r"wochen", r"tagen", r"tage",
    r"years", r"year", r"yrs", r"yr", r"y/o", r"months", r"month", r"weeks", r"days",
    r"lat", r"lata", r"miesięcy", r"miesiące", r"miesiąc", r"tygodni", r"dni",
    r"лет", r"года", r"год", r"месяцев", r"месяца", r"месяц", r"недель", r"недели",
    r"років", r"роки", r"рік", r"місяців", r"місяці", r"місяць", r"тижнів",
    r"yaşındaki", r"yaşında", r"yaşlarında", r"yaş", r"aylık", r"haftalık",
    r"عامًا", r"عاماً", r"عاما", r"عام", r"سنة", r"سنوات", r"أعوام", r"شهرا", r"أشهر",
    r"שנים", r"שנה", r"חודשים",
)
AGE_UNITS_CJK = (r"歳", r"才", r"ヶ月", r"か月", r"カ月", r"세", r"살", r"개월")
_AGE_UNIT = "(?:" + "|".join(sorted(AGE_UNITS, key=len, reverse=True)) + ")(?![^\\W\\d_])"
_AGE_UNIT_CJK = "(?:" + "|".join(AGE_UNITS_CJK) + ")"
_AGE_UNIT_AFTER_NUMBER = re.compile(r"(?i)^(?:[ \u00a0-]?" + _AGE_UNIT + "|" + _AGE_UNIT_CJK + ")")
_AGE_UNIT_START = re.compile(r"(?i)^(?:" + _AGE_UNIT + "|" + _AGE_UNIT_CJK + ")")
_AGE_NUMBER_BEFORE = re.compile(r"(?<![\d.,/:])\d{1,3}[ \u00a0-]?$")
_AGE_GAP = re.compile(r"^[ \u00a0-]?$")


def repair_age_spans(entities: list[dict], text: str) -> list[dict]:
    """F021: every AGE span grows over the digits it cuts («5» of «58»), a number span grows right over an adjacent
    unit («58» + « anos», «34» + «-year-old», «45» + «歳»), a unit span grows left over an adjacent 1-3 digit number
    («anos» -> «58 anos»), and AGE spans whose gap is empty, one space or one hyphen join («13» + «-15 lat»). Other
    heads pass through. Overlapping AGE spans merge (confidence = max)."""
    mine = sorted((dict(e) for e in entities if e.get("type") == "AGE"), key=lambda e: tuple(e["span"]))
    if not mine:
        return entities
    for e in mine:
        s, t = e["span"]
        while s > 0 and s < len(text) and text[s - 1].isdigit() and text[s].isdigit():
            s -= 1
        while 0 < t < len(text) and text[t].isdigit() and text[t - 1].isdigit():
            t += 1
        if text[t - 1:t].isdigit():
            m = _AGE_UNIT_AFTER_NUMBER.match(text[t:t + 24])
            if m:
                t += m.end()
        if _AGE_UNIT_START.match(text[s:t]):
            m = _AGE_NUMBER_BEFORE.search(text[max(0, s - 5):s])
            if m:
                s = max(0, s - 5) + m.start()
        e["span"] = [s, t]
        e["text"] = text[s:t]
    mine.sort(key=lambda e: tuple(e["span"]))
    merged: list[dict] = []
    for e in mine:
        if merged and (e["span"][0] < merged[-1]["span"][1]
                       or _AGE_GAP.match(text[merged[-1]["span"][1]:e["span"][0]])):
            prev = merged[-1]
            prev["span"] = [prev["span"][0], max(prev["span"][1], e["span"][1])]
            prev["text"] = text[prev["span"][0]:prev["span"][1]]
            prev["confidence"] = max(prev.get("confidence", 0.0), e.get("confidence", 0.0))
            if e.get("evidence") == "floor":
                prev["evidence"] = "floor"
            continue
        merged.append(e)
    rest = [e for e in entities if e.get("type") != "AGE"]
    return sorted(rest + merged, key=lambda e: tuple(e["span"]))


def apply_decoder_settings(
    entities: list[dict], text: str, lang: str | None, settings: DecoderSettings
) -> list[dict]:
    """The post-pass half of the settings block, in this order: per-language in-word gap fill
    (joins), span continuity (joins), word completion (grows), name join (joins), numeric
    span repair, particle strip (trims the final span). Off keys are skipped; a per-language
    key needs ``lang`` to name that language."""
    out = entities
    gap = settings.inword_gap(lang)
    if gap > 0:
        out = merge_inword_splits(out, text, gap)
    if settings.span_continuity:
        out = join_split_spans(out, text)
    completed = settings.completed_heads(lang)
    if completed:
        out = complete_word_spans(out, text, completed)
    joined = settings.joined_heads(lang)
    if joined:
        out = join_name_spans(out, text, joined)
    cased_joined = settings.cased_joined_heads(lang)
    if cased_joined:
        out = join_name_spans(out, text, cased_joined, cased=True)
    initials = settings.initials_heads(lang)
    if initials:
        out = absorb_initials(out, text, initials)
    bridged = settings.bridged_heads(lang)
    if bridged:
        out = bridge_initials(out, text, bridged)
    for head in settings.numeric_span_repair:
        out = repair_numeric_spans(out, text, head)
    if settings.date_span_join:
        out = join_date_spans(out, text)
    if settings.date_day_growth:
        out = grow_date_day_left(out, text)
    abbrev = settings.year_abbrev(lang)
    if abbrev:
        out = complete_year_abbrev(out, text, abbrev)
    if settings.date_identifier_exclusion:
        out = drop_identifier_dates(out, text)
    if settings.age_span_repair:
        out = repair_age_spans(out, text)
    if settings.street_number_join:
        out = join_street_numbers(out, text)
    if settings.strips_particles(lang):
        out = strip_particles(out, text, lang)
    if settings.person_full_span:
        out = mark_full_person_spans(out, text)
    return out
