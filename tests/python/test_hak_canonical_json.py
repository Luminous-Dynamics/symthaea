import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("hak_canonical_json", ROOT / "scripts/hak_canonical_json.py")
assert SPEC and SPEC.loader
hak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hak)
VECTORS = json.loads((ROOT / "docs/architecture/hak/golden/hak-canonical-json-v1.vectors.json").read_text(encoding="utf-8"))
PROFILE = json.loads((ROOT / "docs/architecture/hak/canonical-json-v1.profile.json").read_text(encoding="utf-8"))


class CanonicalJsonTests(unittest.TestCase):
    def test_machine_readable_profile_matches_python_contract(self):
        expected = {
            "schema_version": "hak.canonical-json-profile.v1",
            "profile_id": hak.PROFILE_ID,
            "base_standard": {"name": "RFC 8785", "relationship": "CompatibleSubset"},
            "input_contract": {
                "encoding": "UTF-8",
                "duplicate_object_names": "Reject",
                "unicode_strings": "UnicodeScalarValuesOnly",
                "numbers": {
                    "profile": "SafeIntegerOnly",
                    "minimum": hak.MIN_SAFE_INTEGER,
                    "maximum": hak.MAX_SAFE_INTEGER,
                    "floating_point_allowed": False,
                    "negative_zero_canonicalizes_to": 0,
                },
            },
            "serialization": {
                "object_key_order": "RFC8785Utf16CodeUnits",
                "object_sorting_recursive": True,
                "array_order": "PreserveExactly",
                "whitespace": "None",
                "string_escaping": "RFC8785Compatible",
                "unicode_normalization": "None",
                "output_encoding": "UTF-8",
            },
            "digest_contract": {
                "algorithm": "SHA-256",
                "preimage": "UTF8(profile_id) || 0x00 || UTF8(domain) || 0x00 || canonical_bytes",
                "domain_must_be_nonempty": True,
                "domain_must_not_contain_nul": True,
            },
            "migration": {
                "reinterpret_existing_digests": False,
                "historical_python_sort_keys_digests_remain_historical": True,
            },
        }
        self.assertEqual(PROFILE, expected)
        self.assertEqual(VECTORS["profile_id"], hak.PROFILE_ID)

    def test_positive_golden_vectors(self):
        for vector in VECTORS["positive"]:
            with self.subTest(vector=vector["id"]):
                value = hak.parse_strict_json(vector["raw_json"])
                self.assertEqual(hak.canonical_bytes(value).decode("utf-8"), vector["canonical_utf8"])
                self.assertEqual(hak.hak_sha256(vector["digest_domain"], value), vector["hak_sha256"])

    def test_negative_golden_vectors(self):
        for vector in VECTORS["negative"]:
            with self.subTest(vector=vector["id"]):
                with self.assertRaises(hak.CanonicalJsonError):
                    hak.parse_strict_json(vector["raw_json"])

    def test_invalid_utf8_rejected(self):
        with self.assertRaises(hak.CanonicalJsonError):
            hak.parse_strict_json(b'{"x":"\xff"}')

    def test_programmatic_float_rejected(self):
        with self.assertRaises(hak.CanonicalJsonError):
            hak.canonical_bytes({"x": 1.0})

    def test_domain_nul_rejected(self):
        with self.assertRaises(hak.CanonicalJsonError):
            hak.hak_sha256("a\x00b", {"x": 1})

    def test_utf16_sort_differs_from_codepoint_order(self):
        value = {"דּ": 1, "😀": 2}
        self.assertEqual(hak.canonical_bytes(value).decode("utf-8"), '{"😀":2,"דּ":1}')

    def test_unicode_is_not_normalized(self):
        value = hak.parse_strict_json('{"é":1,"é":2}')
        self.assertEqual(hak.canonical_bytes(value).decode("utf-8"), '{"é":2,"é":1}')

    def test_profile_id_is_bound_into_digest_preimage(self):
        value = hak.parse_strict_json('{"x":1}')
        canonical = hak.canonical_bytes(value)
        import hashlib
        unprofiled = "sha256:" + hashlib.sha256(b"hak.golden.v1\x00" + canonical).hexdigest()
        self.assertNotEqual(unprofiled, hak.hak_sha256("hak.golden.v1", value))


if __name__ == "__main__":
    unittest.main()
