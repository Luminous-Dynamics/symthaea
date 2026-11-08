# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from audit import audit, git_blob_sha1, validate_manifest


def entry(path: str, payload: bytes) -> dict:
    return {"path": path, "git_blob_sha1": git_blob_sha1(payload), "size_bytes": len(payload), "purpose": "synthetic fixture"}


def manifest_for(path: str, payload: bytes) -> dict:
    return {
        "schema": "za-lex-resource-audit/v1",
        "source": {"repository": "https://github.com/NWU-MuST/za_lex", "commit": "a" * 40},
        "status": {
            "integration": "not_integrated",
            "rights_review": "metadata_audited_per_file_not_legal_opinion",
            "model_training_rights": "not_assessed_and_not_inferred",
            "speech_synthesis_rights": "not_assessed_and_not_inferred",
            "singing_synthesis_rights": "not_assessed_and_not_inferred",
        },
        "files": [entry(path, payload)],
        "language_profiles": [{
            "id": "xh-ZA", "license": "CC-BY-3.0", "resource_files": [path],
            "rights_notes": ["synthetic rights note"],
            "capabilities": {"g2p_rules": True, "singing_synthesis": False},
            "qualification": "research_only_not_integrated_not_singing_qualified"
        }]
    }


class ZaLexAuditTests(unittest.TestCase):
    def test_verifies_pinned_file_git_blob_and_size(self) -> None:
        payload = b"phoneme inventory fixture"
        manifest = manifest_for("data/xho/phonemeset.json", payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/xho/phonemeset.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            report = audit(root, manifest, "a" * 40)
        self.assertEqual(report["verified_file_count"], 1)
        self.assertEqual(report["integration"], "not_integrated")

    def test_modified_source_bytes_fail_closed(self) -> None:
        manifest = manifest_for("data/zul/g2p.translit.txt", b"rules-v1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/zul/g2p.translit.txt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"rules-v2")
            with self.assertRaisesRegex(ValueError, "size mismatch|Git blob SHA-1 mismatch"):
                audit(root, manifest, "a" * 40)

    def test_wrong_upstream_commit_is_rejected(self) -> None:
        manifest = manifest_for("file.txt", b"x")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.txt").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "does not match pinned"):
                audit(root, manifest, "b" * 40)

    def test_unknown_license_or_untracked_profile_file_is_rejected(self) -> None:
        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["license"] = "NOASSERTION"
        with self.assertRaisesRegex(ValueError, "license"):
            validate_manifest(manifest)
        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["resource_files"] = ["untracked.dict"]
        with self.assertRaisesRegex(ValueError, "untracked"):
            validate_manifest(manifest)

    def test_synthesis_rights_cannot_be_inferred_from_dictionary_metadata(self) -> None:
        manifest = manifest_for("file.txt", b"x")
        manifest["status"]["singing_synthesis_rights"] = "permitted"
        with self.assertRaisesRegex(ValueError, "singing_synthesis_rights"):
            validate_manifest(manifest)

        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["capabilities"]["singing_synthesis"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim singing synthesis"):
            validate_manifest(manifest)

        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["qualification"] = "qualified_for_singing"
        with self.assertRaisesRegex(ValueError, "research-only"):
            validate_manifest(manifest)

    def test_manifest_requires_typed_resource_paths_and_rights_notes(self) -> None:
        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["resource_files"] = [["nested"]]
        with self.assertRaisesRegex(ValueError, "unique string list"):
            validate_manifest(manifest)

        manifest = manifest_for("file.txt", b"x")
        manifest["language_profiles"][0]["rights_notes"] = "not a list"
        with self.assertRaisesRegex(ValueError, "rights notes"):
            validate_manifest(manifest)

    def test_path_traversal_and_duplicate_entries_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_manifest(manifest_for("../secret.txt", b"x"))
        manifest = manifest_for("file.txt", b"x")
        manifest["files"].append(dict(manifest["files"][0]))
        with self.assertRaisesRegex(ValueError, "unsafe or duplicate"):
            validate_manifest(manifest)

    def test_symlink_source_is_rejected(self) -> None:
        manifest = manifest_for("data/voice.dict", b"safe")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.dict"
            outside.write_bytes(b"safe")
            (root / "data").mkdir()
            (root / "data/voice.dict").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                audit(root, manifest, "a" * 40)


if __name__ == "__main__":
    unittest.main()
