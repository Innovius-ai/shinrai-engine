"""Length-preserving invisible-character scrub for serving inputs.

The v1.x tokenizers normalize nothing but " " -> "▁": zero-width, bidi,
tag and similar format characters reach the model as opaque tokens, degrade
NER quality, and sit inside entity spans — while /api/analyze character
offsets are the public contract. The scrub therefore never changes string
length: every scrubbed character becomes exactly one space, so entity offsets
computed on the scrubbed text remain valid for the caller's original text.
Entity surface strings are reported post-scrub (spaces where invisibles were).

NFC/NFKC is deliberately NOT applied — canonical (de)composition changes
string length and would corrupt the offset contract.

The character classes mirror the suite's established input sanitizers
(aiportal utils/server/llmInputSanitizer.ts + utils/app/unicodeValidator.ts,
file-handler responses.sanitize_payload_text, generation/align.py
_AR_INVISIBLES). shinrai-encryption's bert_backend.py carries the same table
for its client/in-process path; unifying every copy into one shared library
is the deferred T3 follow-up.
"""

from __future__ import annotations

import re

_SCRUBBED_RANGES: tuple[tuple[int, int], ...] = (
    (0x0000, 0x0008),    # C0 controls (TAB/LF/CR stay)
    (0x000B, 0x000C),    # VT, FF
    (0x000E, 0x001F),    # SO..US
    (0x007F, 0x009F),    # DEL + C1 controls
    (0x00AD, 0x00AD),    # soft hyphen
    (0x034F, 0x034F),    # combining grapheme joiner
    (0x061C, 0x061C),    # Arabic letter mark
    (0x115F, 0x1160),    # Hangul choseong/jungseong fillers
    (0x17B4, 0x17B5),    # Khmer inherent vowels
    (0x180E, 0x180E),    # Mongolian vowel separator
    (0x200B, 0x200F),    # zero-width space/joiners + LRM/RLM
    (0x202A, 0x202E),    # bidi embeddings/overrides
    (0x2060, 0x2064),    # word joiner + invisible operators
    (0x2066, 0x206F),    # bidi isolates + deprecated format block
    (0x2800, 0x2800),    # Braille pattern blank
    (0x3164, 0x3164),    # Hangul filler
    (0xD800, 0xDFFF),    # lone surrogates (JSON "\\ud800" escapes produce them)
    (0xFE00, 0xFE0F),    # variation selectors
    (0xFEFF, 0xFEFF),    # BOM / zero-width no-break space
    (0xFFA0, 0xFFA0),    # halfwidth Hangul filler
    (0xFFF9, 0xFFFB),    # interlinear annotation controls
    (0xE0000, 0xE007F),  # tag block
    (0xE0100, 0xE01EF),  # variation selectors supplement
)

_SCRUB_TABLE = {
    cp: 0x20 for start, end in _SCRUBBED_RANGES for cp in range(start, end + 1)
}

_SCRUB_RE = re.compile(
    "["
    + "".join(
        re.escape(chr(start)) + (("-" + re.escape(chr(end))) if end > start else "")
        for start, end in _SCRUBBED_RANGES
    )
    + "]"
)


def scrub_invisibles(text: str) -> str:
    """Replace each established invisible/format character with ONE space.

    len(result) == len(text) always. Returns the original object unchanged
    (identity, no copy) when nothing needs scrubbing — callers may use an
    ``is`` check as the fast path.
    """
    if not text or _SCRUB_RE.search(text) is None:
        return text
    return text.translate(_SCRUB_TABLE)
