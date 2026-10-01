"""Label space: the runtime view of configs/labels/labels-v2.0.yaml.

Every label/class dimension is derived from the YAML at runtime — never hardcoded
(PLAN.md §5, WP-09). Importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class UnmappedEntityTypeError(KeyError):
    """A model head has no `api_mapping` entry in the label file.

    Subclasses KeyError so existing `except KeyError` paths keep working; the
    message names the head, the file and the mapped heads so the fix (re-vendor
    the label file, or add the mapping) is readable from the log line.
    """

    def __init__(self, entity_type: str, label_space: "LabelSpace"):
        self.entity_type = entity_type
        known = ", ".join(sorted(label_space.api_mapping))
        super().__init__(
            f"entity type {entity_type!r} has no api_mapping entry in "
            f"{label_space.source_path.name} (schema {label_space.schema_version}; "
            f"mapped: {known}) — re-vendor the label file or add the mapping"
        )

    def __str__(self) -> str:  # KeyError.__str__ would repr() the message
        return self.args[0]


@dataclass(frozen=True)
class HeadSpace:
    name: str
    tiers: tuple[str, ...]
    labels: tuple[str, ...]  # index == label id; labels[0] == "O"

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def bi_labels(self, tier: str) -> tuple[str, str]:
        upper = tier.upper()
        return f"B-{self.name}-{upper}", f"I-{self.name}-{upper}"


@dataclass(frozen=True)
class AttributeSpace:
    name: str
    applies_to: tuple[str, ...]
    classes: tuple[str, ...]

    @property
    def class_to_id(self) -> dict[str, int]:
        return {cls: i for i, cls in enumerate(self.classes)}


@dataclass(frozen=True)
class LabelSpace:
    schema_version: str
    heads: dict[str, HeadSpace]
    attributes: dict[str, AttributeSpace]
    api_mapping: dict
    source_path: Path

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(self.heads)

    def tier_of_label(self, head: str, label_id: int) -> str | None:
        """Tier encoded in a label id, or None for O."""
        label = self.heads[head].labels[label_id]
        if label == "O":
            return None
        return label.split("-", 2)[2].lower()

    def api_type(self, entity_type: str, name_part: str | None = None) -> str:
        """Map a head name (+ PERSON name_part) to the legacy shinrai-encryption
        API type string (FIRSTNAME/SURNAME/PERSON/CITY/STREET_ADDRESS/COMPANY,
        DATE/AGE since labels-v3.2).

        Raises UnmappedEntityTypeError (a KeyError) for a head the label file
        does not map. The adapter lets it propagate: the backend records
        `backend_error` and the request fails closed instead of silently
        dropping that head's spans.
        """
        mapping = self.api_mapping.get(entity_type)
        if mapping is None:
            raise UnmappedEntityTypeError(entity_type, self)
        if isinstance(mapping, str):
            return mapping
        by_part = mapping["by_name_part"]
        # name_part abstention maps to the generic PERSON type (WP-12).
        return by_part.get(name_part or "full", by_part["full"])


def load_label_space(path: str | Path) -> LabelSpace:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or raw.get("extends"):
        raise ValueError(f"{path}: expected a complete standalone label schema, not a delta")
    if not {"schema_version", "heads", "attributes", "api_mapping"} <= raw.keys():
        raise ValueError(f"{path}: incomplete label schema")

    heads = {
        name: HeadSpace(name=name, tiers=tuple(spec["tiers"]), labels=tuple(spec["labels"]))
        for name, spec in raw["heads"].items()
    }
    for head in heads.values():
        if not head.labels or len(set(head.labels)) != len(head.labels):
            raise ValueError(f"head {head.name}: labels must be nonempty and unique")
        if head.labels[0] != "O":
            raise ValueError(f"head {head.name}: labels[0] must be 'O', got {head.labels[0]}")

    attributes = {
        name: AttributeSpace(
            name=name,
            applies_to=tuple(spec["applies_to"]),
            classes=tuple(str(c) for c in spec["classes"]),
        )
        for name, spec in raw["attributes"].items()
    }

    return LabelSpace(
        schema_version=str(raw["schema_version"]),
        heads=heads,
        attributes=attributes,
        api_mapping=raw["api_mapping"],
        source_path=path.resolve(),
    )
