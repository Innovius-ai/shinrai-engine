"""Integration adapter (WP-12, v1.0 subset): model entities -> legacy engine dicts.

Maps annotation-schema entities to the frozen shinrai-encryption entity shape:
{text, type (legacy API string via api_mapping), startIndex, endIndex (exclusive),
tier, attrs, source: "bert", confidence} — plus region/size hints the
ReplacementEngine consumes. Confidence gate defaults to 0.7.

Demo CLI:
    python -m shinrai_pii.integration.adapter --model models/parity-v1-m "Text..."
"""

from __future__ import annotations


from .labels import LabelSpace

# origin class -> static engine pool region (else GLOBAL)
ORIGIN_TO_REGION = {"DE": "DE", "US": "US", "JP": "JP", "GB": "UK", "FR": "FR", "IT": "IT"}
CITY_TIER_TO_SIZE = {"major": "large", "medium": "medium", "small": "small"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.7

_NAME_PAIR = {("given", "family"), ("family", "given")}


def merge_person_spans(entities: list[dict], text: str | None = None) -> list[dict]:
    """Merge adjacent given+family PERSON spans into one full-name span.

    Real documents carry "Mara Feldmann" as one person; the engine replaces a
    full name coherently (one surrogate identity, not an independent first and
    last name). Spans qualify when they are PERSON, exactly one character
    apart, that character is a space (assumed if ``text`` is unavailable), and
    their ``name_part`` attrs form a given/family pair.
    """
    ents = sorted(entities, key=lambda e: e["span"][0])
    merged: list[dict] = []
    i = 0
    while i < len(ents):
        cur = dict(ents[i])
        while i + 1 < len(ents):
            nxt = ents[i + 1]
            gap_ok = nxt["span"][0] - cur["span"][1] == 1 and (
                text is None or text[cur["span"][1]] == " "
            )
            parts = (
                (cur.get("attrs") or {}).get("name_part"),
                (nxt.get("attrs") or {}).get("name_part"),
            )
            if not (
                cur["type"] == "PERSON" and nxt["type"] == "PERSON"
                and gap_ok and parts in _NAME_PAIR
            ):
                break
            attrs = dict(cur.get("attrs") or {})
            given_attrs = attrs if parts[0] == "given" else dict(nxt.get("attrs") or {})
            attrs["name_part"] = "full"
            if given_attrs.get("gender_expression"):
                attrs["gender_expression"] = given_attrs["gender_expression"]
            cur["span"] = [cur["span"][0], nxt["span"][1]]
            cur["text"] = f"{cur['text']} {nxt['text']}"
            cur["attrs"] = attrs
            cur["confidence"] = min(
                float(cur.get("confidence", 1.0)), float(nxt.get("confidence", 1.0))
            )
            i += 1
        merged.append(cur)
        i += 1
    return merged


def to_legacy_entities(
    entities: list[dict],
    label_space: LabelSpace,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    text: str | None = None,
    merge_persons: bool = True,
) -> list[dict]:
    if merge_persons:
        entities = merge_person_spans(entities, text)
    legacy: list[dict] = []
    for ent in entities:
        confidence = float(ent.get("confidence", 1.0))
        if confidence < threshold:
            continue
        attrs = ent.get("attrs") or {}
        api_type = label_space.api_type(ent["type"], attrs.get("name_part"))
        item = {
            "text": ent["text"],
            "type": api_type,
            "startIndex": ent["span"][0],
            "endIndex": ent["span"][1],
            "tier": ent.get("tier"),
            "attrs": attrs,
            "source": "bert",
            "confidence": round(confidence, 4),
            "region": ORIGIN_TO_REGION.get(attrs.get("origin", ""), "GLOBAL"),
        }
        if ent["type"] == "CITY":
            item["size"] = CITY_TIER_TO_SIZE.get(ent.get("tier", ""), "medium")
        legacy.append(item)
    return sorted(legacy, key=lambda e: e["startIndex"])


