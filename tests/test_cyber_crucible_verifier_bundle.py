"""Determinism, inventory, and path-safety tests for the verifier bundle."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_cyber_crucible_verifier import (
    BUNDLE_ROOT,
    MANIFEST_NAME,
    PACKAGE_FILES,
    BundleError,
    build_bundle,
    verify_bundle,
)


TEST_COMMIT = "a" * 40


class VerifierBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cyber-bundle-")
        self.addCleanup(self.temp.cleanup)
        self.temp_root = Path(self.temp.name)

    def test_actual_source_bundle_is_deterministic(self):
        repository_root = Path(__file__).resolve().parents[1]
        first = self.temp_root / "first.tar.gz"
        second = self.temp_root / "second.tar.gz"
        result_a = build_bundle(repository_root, first, TEST_COMMIT)
        result_b = build_bundle(repository_root, second, TEST_COMMIT)
        self.assertEqual(result_a["bundle_sha256"], result_b["bundle_sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertTrue(result_a["bundle_verified"])
        self.assertEqual(result_a["file_count"], len(PACKAGE_FILES))

    def test_bundle_manifest_is_exact_and_every_file_digest_matches(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "verified.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        result = verify_bundle(bundle_path, expected_source_commit=TEST_COMMIT)
        self.assertTrue(result["bundle_verified"])
        self.assertEqual(result["source_commit"], TEST_COMMIT)

        with tarfile.open(bundle_path, "r:gz") as archive:
            manifest_member = archive.extractfile(f"{BUNDLE_ROOT}/{MANIFEST_NAME}")
            self.assertIsNotNone(manifest_member)
            manifest = json.loads(manifest_member.read().decode("utf-8"))
            self.assertEqual(
                [item["path"] for item in manifest["files"]],
                list(PACKAGE_FILES),
            )
            self.assertEqual(len(manifest["files"]), len(PACKAGE_FILES))

    def test_valid_bundle_verification_does_not_materialize_all_tar_headers(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "streamed.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        with patch("tarfile.TarFile.getmembers", side_effect=AssertionError("getmembers must not be used")):
            result = verify_bundle(bundle_path, expected_source_commit=TEST_COMMIT)
        self.assertTrue(result["bundle_verified"])
        self.assertEqual(result["file_count"], len(PACKAGE_FILES))

    def test_trusted_release_workflow_has_exact_semver_and_approval_contract(self):
        repository_root = Path(__file__).resolve().parents[1]
        trusted = (repository_root / ".github/workflows/cyber-crucible-trusted-build.yml").read_text(encoding="utf-8")
        caller = (repository_root / ".github/workflows/cyber-crucible-verifier-release.yml").read_text(encoding="utf-8")

        self.assertEqual(trusted.count("  build-and-test:"), 1)
        self.assertEqual(trusted.count("  release-approval:"), 1)
        self.assertEqual(trusted.count("  attest-exact-tested-bundle:"), 1)
        self.assertIn(r're.fullmatch(r"refs/tags/cyber-crucible-verifier-v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", source_ref)', trusted)
        self.assertIn('actual != source_commit', trusted)
        self.assertIn('needs: [build-and-test, release-approval]', trusted)
        signer = trusted[trusted.index("  attest-exact-tested-bundle:"):]
        self.assertNotIn("\n    environment:", signer)
        self.assertNotIn("actions/checkout@", signer)
        self.assertIn('refs/tags/cyber-crucible-verifier-v*', caller)
        self.assertIn('source_ref: ${{ github.ref }}', caller)
        self.assertIn('source_commit: ${{ github.sha }}', caller)
        self.assertIn("id-token: write", caller)
        self.assertIn("attestations: write", caller)
        self.assertIn("artifact-metadata: write", caller)
        self.assertIn(
            "trusted-build.yml@63e41f2ff7f1a58012a1c5fed3b28d9ecaf2f131",
            caller,
        )
        self.assertIn("retention-days: 7", trusted)
        self.assertEqual(len(re.findall(r"trusted-build\.yml@([0-9a-f]{40})", caller)), 1)
        all_workflow_text = trusted + "\n" + caller + "\n" + contract
        action_refs = re.findall(r"(?m)^\s*uses:\s*[^@\s]+@([^\s#]+)", all_workflow_text)
        self.assertGreaterEqual(len(action_refs), 8)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in action_refs))

    def test_gzip_decompression_limit_is_enforced_before_tar_iteration(self):
        bundle_path = self.temp_root / "expansion-limit.tar.gz"
        bundle_path.write_bytes(gzip.compress(b"\0" * 8192))
        with patch("scripts.package_cyber_crucible_verifier.MAX_BUNDLE_BYTES_UNCOMPRESSED", 1024):
            with self.assertRaises(BundleError):
                verify_bundle(bundle_path)

    def test_trailing_bytes_after_tar_terminator_are_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        clean = self.temp_root / "clean.tar.gz"
        tampered = self.temp_root / "trailing.tar.gz"
        build_bundle(repository_root, clean, TEST_COMMIT)
        with gzip.open(clean, "rb") as source:
            canonical_tar = source.read()
        tampered.write_bytes(gzip.compress(canonical_tar + b"untracked trailing payload"))
        with self.assertRaises(BundleError):
            verify_bundle(tampered)

    def test_release_test_dependencies_are_fully_version_pinned_and_wheel_only(self):
        repository_root = Path(__file__).resolve().parents[1]
        requirements = (repository_root / "validation/cyber_crucible_verifier_requirements_v1.txt").read_text(encoding="utf-8")
        trusted = (repository_root / ".github/workflows/cyber-crucible-trusted-build.yml").read_text(encoding="utf-8")
        contract = (repository_root / ".github/workflows/cyber-crucible-decision-receipt.yml").read_text(encoding="utf-8")

        pins = [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        self.assertGreaterEqual(len(pins), 10)
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", pin) for pin in pins))
        self.assertEqual(len({pin.split("==", 1)[0].lower() for pin in pins}), len(pins))
        for workflow in (trusted, contract):
            self.assertIn("--only-binary=:all:", workflow)
            self.assertIn("-r validation/cyber_crucible_verifier_requirements_v1.txt", workflow)
            self.assertIn("python -m pip check", workflow)

    def test_noncanonical_gzip_mtime_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        clean = self.temp_root / "gzip-clean.tar.gz"
        modified = self.temp_root / "gzip-mtime.tar.gz"
        build_bundle(repository_root, clean, TEST_COMMIT)
        raw = bytearray(clean.read_bytes())
        raw[4] = 1  # Gzip MTIME field; the DEFLATE payload is otherwise unchanged.
        modified.write_bytes(raw)
        with self.assertRaises(BundleError):
            verify_bundle(modified)

    def test_recompressed_payload_with_canonical_header_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        clean = self.temp_root / "canonical.tar.gz"
        noncanonical = self.temp_root / "recompressed.tar.gz"
        build_bundle(repository_root, clean, TEST_COMMIT)
        with gzip.open(clean, "rb") as handle:
            tar_bytes = handle.read()
        raw = bytearray(gzip.compress(tar_bytes, compresslevel=1, mtime=0))
        raw[:10] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
        noncanonical.write_bytes(raw)
        with self.assertRaises(BundleError):
            verify_bundle(noncanonical)

    def test_wrong_expected_commit_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "bundle.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        with self.assertRaises(BundleError):
            verify_bundle(bundle_path, expected_source_commit="b" * 40)

    def test_invalid_source_commit_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, self.temp_root / "bundle.tar.gz", "not-a-commit")

    def test_missing_required_source_file_is_rejected(self):
        source = self.temp_root / "source"
        source.mkdir()
        for name in PACKAGE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        missing = source / PACKAGE_FILES[0]
        missing.unlink()
        with self.assertRaises(BundleError):
            build_bundle(source, self.temp_root / "missing.tar.gz", TEST_COMMIT)

    def test_symlinked_source_file_is_rejected(self):
        source = self.temp_root / "source"
        source.mkdir()
        for name in PACKAGE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        target = self.temp_root / "outside-source"
        target.write_text("not a trusted input", encoding="utf-8")
        victim = source / "scripts/verify_cyber_crucible_verifier_provenance.py"
        victim.unlink()
        victim.symlink_to(target)
        with self.assertRaises(BundleError):
            build_bundle(source, self.temp_root / "symlink.tar.gz", TEST_COMMIT)

    def test_bundle_with_extra_member_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        clean = self.temp_root / "clean.tar.gz"
        tampered = self.temp_root / "extra.tar.gz"
        build_bundle(repository_root, clean, TEST_COMMIT)
        with tarfile.open(clean, "r:gz") as archive:
            entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers()]
        with tarfile.open(tampered, "w:gz") as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
            extra_data = b"unexpected file"
            member = tarfile.TarInfo(f"{BUNDLE_ROOT}/extra.txt")
            member.size = len(extra_data)
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(extra_data))
        with self.assertRaises(BundleError):
            verify_bundle(tampered)

    def test_bundle_path_must_not_be_a_symlink(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = self.temp_root / "target.tar.gz"
        link = self.temp_root / "link.tar.gz"
        build_bundle(repository_root, target, TEST_COMMIT)
        link.symlink_to(target)
        with self.assertRaises(BundleError):
            verify_bundle(link)

    def test_failed_self_validation_preserves_existing_output(self):
        repository_root = Path(__file__).resolve().parents[1]
        output = self.temp_root / "existing-output.tar.gz"
        output.write_bytes(b"previous trusted output")
        with patch("scripts.package_cyber_crucible_verifier.verify_bundle", side_effect=BundleError("fixture self-check failure")):
            with self.assertRaises(BundleError):
                build_bundle(repository_root, output, TEST_COMMIT)
        self.assertEqual(output.read_bytes(), b"previous trusted output")
        self.assertEqual(list(self.temp_root.glob(".existing-output.tar.gz.*.tmp")), [])

    def test_output_symlink_is_rejected_without_replacing_target(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = self.temp_root / "target.txt"
        target.write_text("protected target", encoding="utf-8")
        link = self.temp_root / "output.tar.gz"
        link.symlink_to(target)
        with self.assertRaises(BundleError):
            build_bundle(repository_root, link, TEST_COMMIT)
        self.assertEqual(target.read_text(encoding="utf-8"), "protected target")
        self.assertTrue(link.is_symlink())

    def test_output_cannot_overwrite_a_packaged_source(self):
        repository_root = Path(__file__).resolve().parents[1]
        source_file = repository_root / PACKAGE_FILES[0]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, source_file, TEST_COMMIT)

    def test_output_parent_traversal_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = repository_root / "dist" / ".." / PACKAGE_FILES[0]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, target, TEST_COMMIT)


if __name__ == "__main__":
    unittest.main()
