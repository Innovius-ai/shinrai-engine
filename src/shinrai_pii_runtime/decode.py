"""Constrained IOB2 decoding — the single decoder shared by training round-trip
tests, the Predictor, and the eval harness (WP-09: never fork this logic).

Importable without torch.
"""

from __future__ import annotations

from .labels import LabelSpace


def spans_from_labels(
    label_ids_by_head: dict[str, list[int]],
    offsets: list[tuple[int, int]],
    label_space: LabelSpace,
    text: str,
    confidences_by_head: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Decode per-token label ids into char-exact entity dicts.

    Constrained decoding repairs invalid transitions instead of failing:
    an I-X-TIER with no open span (or after O) opens a new span (treated as B);
    an I with a tier mismatch extends the open span, keeping the opening tier.

    Tokens with start == end offsets (special tokens, padding) are skipped.
    Returns entities sorted by start: {span, text, type, tier, confidence}.
    """
    entities: list[dict] = []
    for head, label_ids in label_ids_by_head.items():
        space = label_space.heads[head]
        confs = confidences_by_head.get(head) if confidences_by_head else None
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
                        round(sum(span["confs"]) / len(span["confs"]), 6)
                        if span["confs"]
                        else 1.0
                    ),
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
            if prefix == "B" or open_span is None:
                close(open_span)
                open_span = {
                    "start": tok_start,
                    "end": tok_end,
                    "tier": tier,
                    "confs": [conf] if conf is not None else [],
                }
            else:  # I continuing an open span (tier of the opening token wins)
                open_span["end"] = tok_end
                if conf is not None:
                    open_span["confs"].append(conf)
        close(open_span)

    return _merge_geresh_splits(
        sorted(entities, key=lambda e: (e["span"][0], e["span"][1])), text
    )


_GERESH = {"'", "\u05f3", "\u05f4", "\u2019"}  # ' ׳ ״ ’


def _is_hebrew(ch: str) -> bool:
    return "\u05d0" <= ch <= "\u05ea"


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
            flanks_hebrew = (_is_hebrew(text[a["span"][1] - 1]) or _is_hebrew(text[e["span"][0]])) if joined else False
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
                continue
        out.append(e)
    return out
