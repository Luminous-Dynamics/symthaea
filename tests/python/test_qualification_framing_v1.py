import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


framing = load("qualification_framing_v1")


def test_empty_record_golden_vector():
    framed = framing.frame_record("symthaea.test.empty.v1", [])
    assert framed.hex() == (
        "53594d5146524d310001001673796d74686165612e746573742e656d7074792e7631"
        "00000000"
    )
    assert framing.semantic_sha256_id("symthaea.test.empty.v1", []) == (
        "sha256:69ff1fa39afc4e827669c2e79605836ce2507f9c974a0511a5b72b0ee4170837"
    )


def test_scalar_record_golden_vector():
    fields = [
        ("name", framing.encode_text("alpha")),
        ("count", framing.encode_u64(7)),
        ("enabled", framing.encode_bool(True)),
    ]
    framed = framing.frame_record("symthaea.test.scalar.v1", fields)
    assert framed.hex() == (
        "53594d5146524d310001001773796d74686165612e746573742e7363616c61722e7631"
        "00000003"
        "00046e616d65000000000000000e540000000000000005616c706861"
        "0005636f756e740000000000000009550000000000000007"
        "0007656e61626c656400000000000000025901"
    )
    assert framing.semantic_sha256_id("symthaea.test.scalar.v1", fields) == (
        "sha256:0df9fe4f5d557dd815fbe50f7d120c866fbe7601efbb877b1eed4d30176a63f9"
    )


def test_ordered_list_and_canonical_set_golden_vector():
    fields = [
        (
            "ordered",
            framing.encode_list([framing.encode_text("b"), framing.encode_text("a")]),
        ),
        (
            "members",
            framing.encode_set([framing.encode_text("b"), framing.encode_text("a")]),
        ),
    ]
    framed = framing.frame_record("symthaea.test.collections.v1", fields)
    assert framed.hex() == (
        "53594d5146524d310001001c73796d74686165612e746573742e636f6c6c656374696f6e732e7631"
        "00000002"
        "00076f726465726564000000000000002d4c0000000000000002"
        "000000000000000a54000000000000000162"
        "000000000000000a54000000000000000161"
        "00076d656d62657273000000000000002d530000000000000002"
        "000000000000000a54000000000000000161"
        "000000000000000a54000000000000000162"
    )
    assert framing.semantic_sha256_id("symthaea.test.collections.v1", fields) == (
        "sha256:980fa1204aa741835ee0f463e6a09c3696a49af89a1021d41df5825e93b21292"
    )


def test_set_is_order_independent_but_list_is_not():
    a = framing.encode_text("a")
    b = framing.encode_text("b")
    assert framing.encode_set([a, b]) == framing.encode_set([b, a])
    assert framing.encode_list([a, b]) != framing.encode_list([b, a])


def test_duplicate_set_member_fails_closed():
    value = framing.encode_text("same")
    with pytest.raises(framing.QualificationFramingError, match="duplicate"):
        framing.encode_set([value, value])


def test_field_order_is_semantic_and_duplicate_names_fail():
    one = framing.encode_text("one")
    two = framing.encode_text("two")
    assert framing.semantic_sha256_id(
        "symthaea.test.order.v1", [("first", one), ("second", two)]
    ) != framing.semantic_sha256_id(
        "symthaea.test.order.v1", [("second", two), ("first", one)]
    )
    with pytest.raises(framing.QualificationFramingError, match="duplicate field"):
        framing.frame_record(
            "symthaea.test.order.v1", [("same", one), ("same", two)]
        )


def test_domain_version_is_identity_separation():
    fields = [("value", framing.encode_text("same"))]
    assert framing.semantic_sha256_id(
        "symthaea.test.domain.v1", fields
    ) != framing.semantic_sha256_id("symthaea.test.domain.v2", fields)


def test_optional_none_is_distinct_from_some_empty_bytes():
    assert framing.encode_optional(None) != framing.encode_optional(framing.encode_bytes(b""))


def test_non_nfc_text_and_control_text_fail_closed():
    with pytest.raises(framing.QualificationFramingError, match="NFC"):
        framing.encode_text("e\u0301")
    with pytest.raises(framing.QualificationFramingError, match="control"):
        framing.encode_text("line\nbreak")


def test_no_mapping_canonicalization_surface_exists():
    # Schema-specific field order is deliberately explicit. A dict is not accepted as a record.
    with pytest.raises(framing.QualificationFramingError, match="ordered sequence"):
        framing.frame_record("symthaea.test.map.v1", {"field": framing.encode_text("value")})
