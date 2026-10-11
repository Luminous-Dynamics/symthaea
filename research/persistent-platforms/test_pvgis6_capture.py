"""Unit tests for strict, byte-preserving PVGIS 6 capture contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_pvgis6 import (
    API_BASE,
    OPENAPI_URL,
    CaptureError,
    build_request_url,
    capture,
    parse_json_bytes,
    sha256_bytes,
    validate_endpoint_path,
    validate_payload_size,
    validate_query_parameters,
)


class PVGISCaptureContractTests(unittest.TestCase):
    def test_accepts_documented_endpoint_families(self):
        self.assertEqual(validate_endpoint_path("performance/broadband"), "performance/broadband")
        self.assertEqual(validate_endpoint_path("solar-position/overview"), "solar-position/overview")
        self.assertEqual(validate_endpoint_path("topography/horizon-profile"), "topography/horizon-profile")

    def test_rejects_external_paths_queries_fragments_and_traversal(self):
        for value in (
            "/performance/broadband",
            "https://example.invalid/api",
            "performance/broadband?latitude=0",
            "performance/../power/broadband",
            "performance//broadband",
            "other/broadband",
            "performance/\\\\host",
        ):
            with self.subTest(value=value), self.assertRaises(CaptureError):
                validate_endpoint_path(value)

    def test_query_url_is_stable_and_booleans_have_explicit_json_style_values(self):
        first = build_request_url("performance/broadband", {"longitude": 28.0, "metadata": True, "latitude": -26.2})
        second = build_request_url("performance/broadband", {"latitude": -26.2, "metadata": True, "longitude": 28.0})
        self.assertEqual(first, second)
        self.assertEqual(first, f"{API_BASE}/performance/broadband?latitude=-26.2&longitude=28.0&metadata=true")

    def test_query_parameters_reject_empty_objects_nested_values_and_nonfinite_numbers(self):
        for value in ({}, {"latitude": None}, {"latitude": [1, 2]}, {"latitude": float("nan")}):
            with self.subTest(value=value), self.assertRaises(CaptureError):
                validate_query_parameters(value)

    def test_json_validation_rejects_duplicate_keys_nonstandard_numbers_and_invalid_utf8(self):
        with self.assertRaisesRegex(CaptureError, "duplicate JSON key"):
            parse_json_bytes(b'{"latitude":1,"latitude":2}', "fixture")
        with self.assertRaisesRegex(CaptureError, "non-standard JSON"):
            parse_json_bytes(b'{"value":NaN}', "fixture")
        with self.assertRaisesRegex(CaptureError, "UTF-8 JSON"):
            parse_json_bytes(b"\xff", "fixture")

    def test_payload_size_is_bounded(self):
        validate_payload_size(b"123", 3)
        with self.assertRaisesRegex(CaptureError, "capture limit"):
            validate_payload_size(b"1234", 3)

    @patch("capture_pvgis6._fetch_json")
    def test_capture_preserves_raw_bodies_and_binds_their_digests(self, fetch):
        openapi_bytes = b'{ "openapi": "3.1.0", "paths": {} }\n'
        source_bytes = b'{\n  "results": {"global_inclined_irradiance": [1, 2]}\n}\n'
        openapi = {
            "payload": openapi_bytes, "status": 200, "headers": [["Content-Type", "application/json"]],
            "content_type": "application/json", "final_url": OPENAPI_URL,
            "retrieved_at_utc": "2026-10-11T00:00:00Z",
        }
        source = {
            "payload": source_bytes, "status": 200, "headers": [["Content-Type", "application/json"]],
            "content_type": "application/json",
            "final_url": f"{API_BASE}/performance/broadband?latitude=-26.2&longitude=28",
            "retrieved_at_utc": "2026-10-11T00:00:01Z",
        }
        fetch.side_effect = [openapi, source]
        with tempfile.TemporaryDirectory() as temp:
            params_path = Path(temp) / "params.json"
            params_bytes = b'{ "latitude": -26.2, "longitude": 28 }\n'
            params_path.write_bytes(params_bytes)
            output = Path(temp) / "capture"
            capture("performance/broadband", params_bytes, str(output))
            self.assertEqual((output / "raw-response.json").read_bytes(), source_bytes)
            self.assertEqual((output / "openapi.json").read_bytes(), openapi_bytes)
            self.assertEqual((output / "parameters.input.json").read_bytes(), params_bytes)
            metadata = json.loads((output / "capture.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_response"]["sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(metadata["openapi"]["sha256"], hashlib.sha256(openapi_bytes).hexdigest())
            self.assertEqual(metadata["request_manifest"]["sha256"],
                             sha256_bytes((output / "request.json").read_bytes()))
            self.assertEqual(metadata["source_response"]["bytes"], len(source_bytes))

    @patch("capture_pvgis6._fetch_json")
    def test_capture_refuses_to_overwrite_existing_evidence(self, fetch):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(CaptureError, "refusing to overwrite"):
                capture("performance/broadband", b'{"latitude":1}', str(output))
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
