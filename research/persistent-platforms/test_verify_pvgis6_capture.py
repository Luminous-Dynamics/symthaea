"""Offline tamper-detection tests for PVGIS 6 capture bundles."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_pvgis6 import API_BASE, OPENAPI_URL, CaptureError, capture
from verify_pvgis6_capture import verify_bundle


class PVGISCaptureBundleVerificationTests(unittest.TestCase):
    def _capture_fixture(self, output: Path) -> None:
        openapi_bytes = b'{"openapi":"3.1.0","paths":{}}\n'
        source_bytes = b'{"results":{"opaque_fixture":[1,2]}}\n'
        openapi = {
            "payload": openapi_bytes, "status": 200,
            "headers": [["Content-Type", "application/json"]],
            "content_type": "application/json", "final_url": OPENAPI_URL,
            "retrieved_at_utc": "2026-10-11T00:00:00Z",
        }
        source = {
            "payload": source_bytes, "status": 200,
            "headers": [["Content-Type", "application/json"]],
            "content_type": "application/json",
            "final_url": f"{API_BASE}/performance/broadband?latitude=1",
            "retrieved_at_utc": "2026-10-11T00:00:01Z",
        }
        with patch("capture_pvgis6._fetch_json", side_effect=[openapi, source]):
            capture("performance/broadband", b'{"latitude":1}', str(output))

    def test_valid_bundle_verifies_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            self._capture_fixture(output)
            result = verify_bundle(str(output))
            self.assertTrue(result["valid"])
            self.assertEqual(result["file_count"], 5)
            self.assertIn("Internal consistency only", result["claim_boundary"])

    def test_modified_raw_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            self._capture_fixture(output)
            (output / "raw-response.json").write_bytes(b'{"results":{"changed":true}}')
            with self.assertRaisesRegex(CaptureError, "SHA-256 mismatch"):
                verify_bundle(str(output))

    def test_modified_request_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            self._capture_fixture(output)
            request_path = output / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["query_parameters"]["latitude"] = 2
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(CaptureError, "SHA-256 mismatch"):
                verify_bundle(str(output))

    def test_extra_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            self._capture_fixture(output)
            (output / "notes.txt").write_text("unbound note", encoding="utf-8")
            with self.assertRaisesRegex(CaptureError, "file set mismatch"):
                verify_bundle(str(output))


if __name__ == "__main__":
    unittest.main()
