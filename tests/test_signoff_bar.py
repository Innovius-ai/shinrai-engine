"""0.1.7: the sign-off bar decision of /api/analyze (decoder key ``signoff_floor``)."""
from types import SimpleNamespace

from shinrai_engine.app import _signoff_bar
from shinrai_pii_runtime.adapter import to_legacy_entities
from shinrai_pii_runtime.decode import DecoderSettings, signoff_regions

STAMPED = SimpleNamespace(meta={"decoder": {"serve_threshold": 0.35}})


def test_bar_applies_at_or_below_the_stamped_threshold_only():
    assert _signoff_bar(None, STAMPED)
    assert _signoff_bar(0.35, STAMPED)       # the encryption service sends the stamp explicitly
    assert _signoff_bar(0.30, STAMPED)
    assert not _signoff_bar(0.50, STAMPED)   # stricter caller
    assert not _signoff_bar(0.35, SimpleNamespace(meta={}))


def test_vendored_runtime_carries_the_rule():
    text = "Body.\n\nMit freundlichen Grüßen,\n\nMax Munster\nEinkauf"
    assert [text[s:e] for s, e in signoff_regions(text)] == ["Max Munster", "Einkauf"]
    assert DecoderSettings.from_mapping({"signoff_floor": {"PERSON": 0.15}}).signoff_floor == {"PERSON": 0.15}
    span = {"span": [text.index("Max"), text.index("Max") + 11], "text": "Max Munster", "type": "PERSON",
            "tier": "common", "confidence": 0.27, "evidence": "floor", "bar": 0.15}
    from shinrai_pii_runtime.labels import load_label_space  # noqa: F401 - import check only
    assert to_legacy_entities.__kwdefaults__.get("signoff_bar") is False
