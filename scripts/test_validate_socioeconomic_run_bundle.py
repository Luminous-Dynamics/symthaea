"""Standard-library adversarial tests for the research run-bundle verifier."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from validate_socioeconomic_run_bundle import ValidationFailure, validate_bundle

MANIFEST_SCHEMA = "civos.socioeconomic_interoperability.run_bundle_manifest.v1"
RECEIPT_SCHEMA = "civos.socioeconomic_interoperability.run_receipt.v1"
CLAIM_CEILING = "synthetic_deterministic_only"


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class RunBundleValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "payloads").mkdir()
        self.payload = b'{"seed":7}\n'
        (self.root / "payloads" / "seed.json").write_bytes(self.payload)
        self.manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "status": "research",
            "claim_ceiling": CLAIM_CEILING,
            "hash_policy": "sha256_exact_bytes_v1",
            "run_id": "run-test-1",
            "receipt_path": "receipt.json",
            "artifacts": [
                {
                    "artifact_id": "world-seed",
                    "role": "world_seed",
                    "path": "payloads/seed.json",
                    "digest": digest(self.payload),
                    "byte_length": len(self.payload),
                    "receipt_pointers": ["/run/world_seed_digest"],
                }
            ],
        }
        self.receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "record_kind": "synthetic_simulation_receipt",
            "claim_ceiling": CLAIM_CEILING,
            "run": {
                "run_id": "run-test-1",
                "artifact_manifest_digest": "",
                "world_seed_digest": digest(self.payload),
            },
            "authority_boundary": {
                "is_real_world_observation": False,
                "can_authorize_real_actions": False,
                "can_reserve_real_resources": False,
                "can_settle_real_assets": False,
                "production_ledger_import_allowed": False,
            },
        }
        self._write_pair()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_pair(self) -> None:
        manifest_bytes = json_bytes(self.manifest)
        (self.root / "bundle-manifest.json").write_bytes(manifest_bytes)
        self.receipt["run"]["artifact_manifest_digest"] = digest(manifest_bytes)
        (self.root / "receipt.json").write_bytes(json_bytes(self.receipt))

    def test_valid_bundle_verifies_payload_and_manifest_bytes(self) -> None:
        result = validate_bundle(self.root, "bundle-manifest.json")
        self.assertEqual(result["artifact_count"], 1)
        self.assertEqual(result["receipt_digest_binding_count"], 1)
        self.assertEqual(result["manifest_digest"], self.receipt["run"]["artifact_manifest_digest"])
        self.assertEqual(result["claim_ceiling"], CLAIM_CEILING)

    def test_payload_mutation_is_rejected_even_if_manifest_is_unchanged(self) -> None:
        (self.root / "payloads" / "seed.json").write_bytes(b'{"seed":8}\n')
        with self.assertRaisesRegex(ValidationFailure, "byte length mismatch|SHA-256 mismatch"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_receipt_must_bind_exact_manifest_bytes(self) -> None:
        self.receipt["run"]["artifact_manifest_digest"] = "sha256:" + "0" * 64
        (self.root / "receipt.json").write_bytes(json_bytes(self.receipt))
        with self.assertRaisesRegex(ValidationFailure, "does not match exact manifest bytes"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_every_receipt_digest_field_requires_one_payload_binding(self) -> None:
        self.receipt["run"]["environment_digest"] = digest(b"environment")
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "bindings are incomplete"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_authority_boundary_cannot_be_enabled(self) -> None:
        self.receipt["authority_boundary"]["can_settle_real_assets"] = True
        (self.root / "receipt.json").write_bytes(json_bytes(self.receipt))
        with self.assertRaisesRegex(ValidationFailure, "can_settle_real_assets must be exactly false"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_manifest_path_traversal_is_rejected(self) -> None:
        self.manifest["artifacts"][0]["path"] = "../outside.json"
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "unsafe relative path"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_receipt_digest_role_must_match_artifact_role(self) -> None:
        self.manifest["artifacts"][0]["role"] = "environment"
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "expected artifact role 'world_seed'"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_noncanonical_path_alias_is_rejected(self) -> None:
        self.manifest["artifacts"][0]["path"] = "payloads//seed.json"
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "non-canonical relative path"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_symlink_payload_is_rejected(self) -> None:
        link = self.root / "payloads" / "linked-seed.json"
        link.symlink_to(self.root / "payloads" / "seed.json")
        self.manifest["artifacts"][0]["path"] = "payloads/linked-seed.json"
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "symlink components are forbidden"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_duplicate_artifact_identity_is_rejected(self) -> None:
        duplicate = dict(self.manifest["artifacts"][0])
        duplicate["path"] = "payloads/another.json"
        self.manifest["artifacts"].append(duplicate)
        self._write_pair()
        with self.assertRaisesRegex(ValidationFailure, "duplicate artifact_id"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_nonstandard_json_constants_are_rejected(self) -> None:
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text('{"schema": "x", "bad": NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ValidationFailure, "non-standard JSON numeric constant"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_float_overflow_and_underflow_are_rejected(self) -> None:
        receipt_path = self.root / "receipt.json"
        with self.subTest(value="overflow"):
            original = receipt_path.read_text(encoding="utf-8").rstrip()
            receipt_path.write_text(original[:-1] + ',"probe":1e999}', encoding="utf-8")
            with self.assertRaisesRegex(ValidationFailure, "non-finite JSON floating-point"):
                validate_bundle(self.root, "bundle-manifest.json")
        self._write_pair()
        with self.subTest(value="underflow"):
            original = receipt_path.read_text(encoding="utf-8").rstrip()
            receipt_path.write_text(original[:-1] + ',"probe":1e-9999}', encoding="utf-8")
            with self.assertRaisesRegex(ValidationFailure, "underflow"):
                validate_bundle(self.root, "bundle-manifest.json")

    def test_duplicate_json_object_keys_are_rejected(self) -> None:
        (self.root / "receipt.json").write_text(
            '{"schema":"' + RECEIPT_SCHEMA + '","schema":"' + RECEIPT_SCHEMA + '"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationFailure, "duplicate JSON object key"):
            validate_bundle(self.root, "bundle-manifest.json")

    def test_unavailable_or_malformed_digest_cannot_be_silently_ignored(self) -> None:
        self.receipt["run"]["world_seed_digest"] = "sha256:not-a-digest"
        (self.root / "receipt.json").write_bytes(json_bytes(self.receipt))
        with self.assertRaisesRegex(ValidationFailure, "digest field is not canonical sha256"):
            validate_bundle(self.root, "bundle-manifest.json")


if __name__ == "__main__":
    unittest.main()
