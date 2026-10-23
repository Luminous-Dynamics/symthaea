"""Offline/mocked tests for integrity-checked Zenodo file acquisition."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import fetch_verified_zenodo_file as zenodo


class URLResponse(BytesIO):
    def __init__(self, payload: bytes, final_url: str):
        super().__init__(payload)
        self.final_url = final_url

    def geturl(self) -> str:
        return self.final_url


class VerifiedZenodoFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "downloaded"
        self.payload = b"synthetic zenodo archive bytes\n"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.record = {
            "id": 17306061,
            "doi": "10.5281/zenodo.17306061",
            "conceptdoi": "10.5281/zenodo.17306060",
            "metadata": {"version": "1"},
            "links": {"self_html": "https://zenodo.org/records/17306061"},
            "files": [{
                "key": "PeptoneDBs.tar.gz",
                "size": len(self.payload),
                "checksum": f"sha256:{self.digest}",
                "links": {
                    "self": "https://zenodo.org/api/records/17306061/files/PeptoneDBs.tar.gz",
                    "content": "https://zenodo.org/api/records/17306061/files/PeptoneDBs.tar.gz/content",
                },
            }],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_metadata_redirect_to_untrusted_host_is_rejected(self) -> None:
        with patch.object(zenodo, "urlopen", return_value=URLResponse(b"{}", "https://evil.invalid/api/records/1")):
            with self.assertRaisesRegex(zenodo.FetchError, "redirect to a non-Zenodo"):
                zenodo.request_bytes("https://zenodo.org/api/records/1")

    def test_file_redirect_to_untrusted_host_discards_partial(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        response = URLResponse(self.payload, "https://evil.invalid/archive.tar.gz")
        with patch.object(zenodo, "urlopen", return_value=response):
            with self.assertRaisesRegex(zenodo.FetchError, "redirect to a non-Zenodo"):
                zenodo.download_verified(item, self.output, len(self.payload))
        self.assertFalse((self.output / "PeptoneDBs.tar.gz").exists())
        self.assertEqual(list(self.output.glob("*.part")) + list(self.output.glob(".*.part")), [])

    def test_record_id_must_be_numeric(self) -> None:
        with self.assertRaises(zenodo.FetchError):
            zenodo.fetch_record_metadata("../17306061")

    def test_exact_file_metadata_is_parsed(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        self.assertEqual(item["size"], len(self.payload))
        self.assertEqual(item["checksum_algorithm"], "sha256")
        self.assertEqual(item["checksum_value"], self.digest)

    def test_missing_file_fails_closed(self) -> None:
        with self.assertRaisesRegex(zenodo.FetchError, "found 0"):
            zenodo.parse_file_metadata(self.record, "missing.tar.gz")

    def test_unsupported_checksum_fails_closed(self) -> None:
        self.record["files"][0]["checksum"] = "sha512:" + "a" * 128
        with self.assertRaisesRegex(zenodo.FetchError, "unsupported or malformed"):
            zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")

    def test_download_url_must_be_https_zenodo(self) -> None:
        self.record["files"][0]["links"]["content"] = "https://evil.invalid/data.tar.gz"
        with self.assertRaisesRegex(zenodo.FetchError, "not hosted on HTTPS Zenodo"):
            zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")

    def test_filename_must_not_be_path(self) -> None:
        with self.assertRaisesRegex(zenodo.FetchError, "exact basename"):
            zenodo.parse_file_metadata(self.record, "../PeptoneDBs.tar.gz")

    def test_embedded_credentials_in_download_url_are_rejected(self) -> None:
        self.record["files"][0]["links"]["content"] = "https://user:password@zenodo.org/data.tar.gz"
        with self.assertRaisesRegex(zenodo.FetchError, "not hosted on HTTPS Zenodo"):
            zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")

    def test_existing_sidecar_is_not_overwritten_by_default(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        self.output.mkdir()
        sidecar = self.output / "PeptoneDBs.tar.gz.zenodo.json"
        sidecar.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(zenodo.FetchError, "sidecar already exists"):
            zenodo.download_verified(item, self.output, len(self.payload))
        self.assertEqual(sidecar.read_text(encoding="utf-8"), "preserve")

    def test_size_cap_blocks_before_any_download_request(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        with patch.object(zenodo, "urlopen") as mocked:
            with self.assertRaisesRegex(zenodo.FetchError, "exceeds --max-bytes"):
                zenodo.download_verified(item, self.output, len(self.payload) - 1)
            mocked.assert_not_called()

    def test_download_verifies_checksum_and_preserves_provenance_sidecars(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        raw_metadata = json.dumps(self.record, sort_keys=True).encode()
        source_summary = {"record_id": "17306061", "doi": self.record["doi"], "version": "1"}
        with patch.object(zenodo, "urlopen", return_value=BytesIO(self.payload)):
            result = zenodo.download_verified(
                item, self.output, len(self.payload), source_metadata=source_summary, record_metadata_raw=raw_metadata
            )
        path = self.output / "PeptoneDBs.tar.gz"
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertEqual(result["local_sha256"], self.digest)
        sidecar = json.loads((self.output / "PeptoneDBs.tar.gz.zenodo.json").read_text())
        self.assertEqual(sidecar["verification_status"], "publisher_checksum_and_size_verified")
        self.assertEqual(sidecar["source_record_id"], "17306061")
        snapshot = self.output / sidecar["source_metadata_snapshot"]
        self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), sidecar["source_record_metadata_sha256"])

    def test_checksum_mismatch_discards_partial_and_final(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        item["checksum_value"] = "0" * 64
        with patch.object(zenodo, "urlopen", return_value=BytesIO(self.payload)):
            with self.assertRaisesRegex(zenodo.FetchError, "checksum mismatch"):
                zenodo.download_verified(item, self.output, len(self.payload))
        self.assertFalse((self.output / "PeptoneDBs.tar.gz").exists())
        self.assertEqual(list(self.output.glob("*.part")) + list(self.output.glob(".*.part")), [])

    def test_size_mismatch_discards_partial(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        item["size"] = len(self.payload) + 1
        with patch.object(zenodo, "urlopen", return_value=BytesIO(self.payload)):
            with self.assertRaisesRegex(zenodo.FetchError, "size mismatch"):
                zenodo.download_verified(item, self.output, len(self.payload) + 1)
        self.assertFalse((self.output / "PeptoneDBs.tar.gz").exists())

    def test_existing_destination_is_not_overwritten_by_default(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        self.output.mkdir()
        target = self.output / "PeptoneDBs.tar.gz"
        target.write_bytes(b"keep me")
        with self.assertRaisesRegex(zenodo.FetchError, "refusing to overwrite"):
            zenodo.download_verified(item, self.output, len(self.payload))
        self.assertEqual(target.read_bytes(), b"keep me")

    def test_oversized_stream_is_stopped_and_removed(self) -> None:
        item = zenodo.parse_file_metadata(self.record, "PeptoneDBs.tar.gz")
        item["size"] = len(self.payload) * 4
        item["checksum_value"] = "0" * 64
        with patch.object(zenodo, "urlopen", return_value=BytesIO(self.payload * 4)):
            with self.assertRaises(zenodo.FetchError):
                zenodo.download_verified(item, self.output, len(self.payload) * 2)
        self.assertFalse((self.output / "PeptoneDBs.tar.gz").exists())

    def test_metadata_fetch_captures_exact_response_bytes(self) -> None:
        raw = json.dumps(self.record).encode()
        with patch.object(zenodo, "request_bytes", return_value=raw):
            record, observed_raw = zenodo.fetch_record_metadata("17306061")
        self.assertEqual(record["id"], 17306061)
        self.assertEqual(hashlib.sha256(observed_raw).hexdigest(), hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
