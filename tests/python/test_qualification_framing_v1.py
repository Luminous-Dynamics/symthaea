import ast
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SOURCE = SCRIPTS / "qualification_framing_v1.py"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


framing = load("qualification_framing_v1")


def u16(value: int) -> bytes:
    return value.to_bytes(2, "big")


def u32(value: int) -> bytes:
    return value.to_bytes(4, "big")


def u64(value: int) -> bytes:
    return value.to_bytes(8, "big")


def oracle_text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return b"T" + u64(len(raw)) + raw


def oracle_u64(value: int) -> bytes:
    return b"U" + u64(value)


def oracle_bool(value: bool) -> bytes:
    return b"Y" + (b"\x01" if value else b"\x00")


def oracle_list(values: list[bytes]) -> bytes:
    return b"L" + u64(len(values)) + b"".join(u64(len(value)) + value for value in values)


def oracle_set(values: list[bytes]) -> bytes:
    ordered = sorted(values)
    return b"S" + u64(len(ordered)) + b"".join(
        u64(len(value)) + value for value in ordered
    )


def oracle_record(domain: str, fields: list[tuple[str, bytes]]) -> bytes:
    domain_raw = domain.encode("ascii")
    out = bytearray(b"SYMQFRM1")
    out.extend(u16(1))
    out.extend(u16(len(domain_raw)))
    out.extend(domain_raw)
    out.extend(u32(len(fields)))
    for name, value in fields:
        name_raw = name.encode("ascii")
        out.extend(u16(len(name_raw)))
        out.extend(name_raw)
        out.extend(u64(len(value)))
        out.extend(value)
    return bytes(out)


class QualificationFramingV1Tests(unittest.TestCase):
    def test_source_has_only_expected_stdlib_imports(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".", 1)[0])
        self.assertEqual(imports, {"__future__", "collections", "hashlib", "re", "unicodedata"})

    def test_empty_record_matches_independent_oracle_and_golden(self):
        expected = oracle_record("symthaea.test.empty.v1", [])
        self.assertEqual(framing.frame_record("symthaea.test.empty.v1", []), expected)
        self.assertEqual(
            expected.hex(),
            "53594d5146524d310001001673796d74686165612e746573742e656d7074792e7631"
            "00000000",
        )
        self.assertEqual(
            framing.semantic_sha256_id("symthaea.test.empty.v1", []),
            "sha256:69ff1fa39afc4e827669c2e79605836ce2507f9c974a0511a5b72b0ee4170837",
        )

    def test_scalar_record_matches_independent_oracle_and_golden(self):
        actual_fields = [
            ("name", framing.encode_text("alpha")),
            ("count", framing.encode_u64(7)),
            ("enabled", framing.encode_bool(True)),
        ]
        oracle_fields = [
            ("name", oracle_text("alpha")),
            ("count", oracle_u64(7)),
            ("enabled", oracle_bool(True)),
        ]
        expected = oracle_record("symthaea.test.scalar.v1", oracle_fields)
        self.assertEqual(framing.frame_record("symthaea.test.scalar.v1", actual_fields), expected)
        self.assertEqual(
            hashlib.sha256(expected).hexdigest(),
            "0df9fe4f5d557dd815fbe50f7d120c866fbe7601efbb877b1eed4d30176a63f9",
        )
        self.assertEqual(
            framing.semantic_sha256_id("symthaea.test.scalar.v1", actual_fields),
            "sha256:0df9fe4f5d557dd815fbe50f7d120c866fbe7601efbb877b1eed4d30176a63f9",
        )

    def test_collection_record_matches_independent_oracle_and_golden(self):
        a = framing.encode_text("a")
        b = framing.encode_text("b")
        actual_fields = [
            ("ordered", framing.encode_list([b, a])),
            ("members", framing.encode_set([b, a])),
        ]
        oracle_a = oracle_text("a")
        oracle_b = oracle_text("b")
        expected = oracle_record(
            "symthaea.test.collections.v1",
            [
                ("ordered", oracle_list([oracle_b, oracle_a])),
                ("members", oracle_set([oracle_b, oracle_a])),
            ],
        )
        self.assertEqual(
            framing.frame_record("symthaea.test.collections.v1", actual_fields), expected
        )
        self.assertEqual(
            hashlib.sha256(expected).hexdigest(),
            "980fa1204aa741835ee0f463e6a09c3696a49af89a1021d41df5825e93b21292",
        )
        self.assertEqual(
            framing.semantic_sha256_id("symthaea.test.collections.v1", actual_fields),
            "sha256:980fa1204aa741835ee0f463e6a09c3696a49af89a1021d41df5825e93b21292",
        )

    def test_set_is_canonical_but_list_order_is_semantic(self):
        a = framing.encode_text("a")
        b = framing.encode_text("b")
        self.assertEqual(framing.encode_set([a, b]), framing.encode_set([b, a]))
        self.assertNotEqual(framing.encode_list([a, b]), framing.encode_list([b, a]))

    def test_duplicate_set_and_duplicate_field_fail_closed(self):
        value = framing.encode_text("same")
        with self.assertRaisesRegex(framing.QualificationFramingError, "duplicate"):
            framing.encode_set([value, value])
        with self.assertRaisesRegex(framing.QualificationFramingError, "duplicate field"):
            framing.frame_record("symthaea.test.order.v1", [("same", value), ("same", value)])

    def test_field_order_and_domain_are_identity_separators(self):
        one = framing.encode_text("one")
        two = framing.encode_text("two")
        self.assertNotEqual(
            framing.semantic_sha256_id(
                "symthaea.test.order.v1", [("first", one), ("second", two)]
            ),
            framing.semantic_sha256_id(
                "symthaea.test.order.v1", [("second", two), ("first", one)]
            ),
        )
        fields = [("value", framing.encode_text("same"))]
        self.assertNotEqual(
            framing.semantic_sha256_id("symthaea.test.domain.v1", fields),
            framing.semantic_sha256_id("symthaea.test.domain.v2", fields),
        )

    def test_optional_none_is_distinct_from_some_empty_bytes(self):
        self.assertNotEqual(
            framing.encode_optional(None),
            framing.encode_optional(framing.encode_bytes(b"")),
        )

    def test_noncanonical_text_fails_closed(self):
        with self.assertRaisesRegex(framing.QualificationFramingError, "NFC"):
            framing.encode_text("e\u0301")
        with self.assertRaisesRegex(framing.QualificationFramingError, "control"):
            framing.encode_text("line\nbreak")

    def test_mapping_surface_is_rejected(self):
        with self.assertRaisesRegex(framing.QualificationFramingError, "ordered sequence"):
            framing.frame_record(
                "symthaea.test.map.v1", {"field": framing.encode_text("value")}
            )


if __name__ == "__main__":
    unittest.main()
