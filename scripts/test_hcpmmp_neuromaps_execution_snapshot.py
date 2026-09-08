#!/usr/bin/env python3
from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path

import hcpmmp_neuromaps_execution_snapshot as snapshot
from hcpmmp_neuromaps_common import REQUIRED_INPUT_ROLES, ContractError, digest_file


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ScientificInputSnapshotTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, dict[str, bytes]]:
        sources = root / "sources"
        sources.mkdir()
        payloads: dict[str, bytes] = {}
        inputs: dict[str, dict[str, str]] = {}
        for role in sorted(REQUIRED_INPUT_ROLES):
            data = f"{role}:scientific-input-v1\n".encode()
            path = sources / f"{role}.bin"
            path.write_bytes(data)
            payloads[role] = data
            inputs[role] = {"path": str(path), "sha256": sha256_bytes(data)}
        return {"inputs": inputs}, payloads

    def test_snapshot_copies_exact_closed_world_inputs_privately(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, payloads = self.fixture(td)
            result = snapshot.build_scientific_input_snapshot(run, td / "snapshot")

            self.assertEqual(result.profile, snapshot.SNAPSHOT_PROFILE)
            self.assertEqual(set(result.paths), REQUIRED_INPUT_ROLES)
            self.assertEqual(len(result.paths), 14)
            self.assertEqual(stat.S_IMODE(result.root.stat().st_mode), 0o700)
            for role, path in result.paths.items():
                self.assertNotEqual(path.resolve(), Path(run["inputs"][role]["path"]).resolve())
                self.assertEqual(path.read_bytes(), payloads[role])
                self.assertEqual(digest_file(path), run["inputs"][role]["sha256"])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)

    def test_source_mutation_after_snapshot_cannot_change_snapshot_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, payloads = self.fixture(td)
            result = snapshot.build_scientific_input_snapshot(run, td / "snapshot")
            role = sorted(REQUIRED_INPUT_ROLES)[0]
            source = Path(run["inputs"][role]["path"])
            source.write_bytes(b"mutated-after-snapshot\n")

            self.assertEqual(result.paths[role].read_bytes(), payloads[role])
            self.assertEqual(digest_file(result.paths[role]), run["inputs"][role]["sha256"])

    def test_source_replacement_after_snapshot_cannot_retarget_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, payloads = self.fixture(td)
            result = snapshot.build_scientific_input_snapshot(run, td / "snapshot")
            role = sorted(REQUIRED_INPUT_ROLES)[1]
            source = Path(run["inputs"][role]["path"])
            replacement = source.with_suffix(".replacement")
            replacement.write_bytes(b"replacement-bytes\n")
            os.replace(replacement, source)

            self.assertEqual(result.paths[role].read_bytes(), payloads[role])
            self.assertEqual(digest_file(result.paths[role]), run["inputs"][role]["sha256"])

    def test_corruption_before_snapshot_fails_and_removes_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, _ = self.fixture(td)
            role = sorted(REQUIRED_INPUT_ROLES)[5]
            Path(run["inputs"][role]["path"]).write_bytes(b"wrong-before-snapshot\n")
            root = td / "snapshot"

            with self.assertRaises(ContractError):
                snapshot.build_scientific_input_snapshot(run, root)
            self.assertFalse(root.exists())

    def test_unconfirmed_cleanup_fails_closed_without_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, _ = self.fixture(td)
            role = sorted(REQUIRED_INPUT_ROLES)[5]
            Path(run["inputs"][role]["path"]).write_bytes(b"wrong-before-snapshot\n")
            root = td / "snapshot"
            real_rmtree = snapshot.shutil.rmtree
            snapshot.shutil.rmtree = lambda *_args, **_kwargs: None
            try:
                with self.assertRaisesRegex(ContractError, "cleanup not confirmed"):
                    snapshot.build_scientific_input_snapshot(run, root)
                self.assertTrue(root.exists())
            finally:
                snapshot.shutil.rmtree = real_rmtree
                if root.exists():
                    real_rmtree(root)

    def test_opened_source_must_be_regular_file(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, _ = self.fixture(td)
            role = sorted(REQUIRED_INPUT_ROLES)[6]
            source = Path(run["inputs"][role]["path"])
            source.unlink()
            source.mkdir()
            root = td / "snapshot"

            with self.assertRaises(ContractError):
                snapshot.build_scientific_input_snapshot(run, root)
            self.assertFalse(root.exists())

    def test_destination_open_failure_closes_source_descriptor(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            source = td / "source.bin"
            payload = b"source-fd-lifetime\n"
            source.write_bytes(payload)
            destination = td / "already-exists.input"
            destination.write_bytes(b"sentinel\n")
            captured: list[int] = []
            real_open_source = snapshot._open_regular_source

            def tracking_open(path: Path) -> int:
                fd = real_open_source(path)
                captured.append(fd)
                return fd

            snapshot._open_regular_source = tracking_open
            try:
                with self.assertRaises(FileExistsError):
                    snapshot._copy_verified(source, destination, sha256_bytes(payload))
            finally:
                snapshot._open_regular_source = real_open_source

            self.assertEqual(len(captured), 1)
            with self.assertRaises(OSError) as raised:
                os.fstat(captured[0])
            self.assertEqual(raised.exception.errno, errno.EBADF)
            self.assertEqual(destination.read_bytes(), b"sentinel\n")

    def test_missing_role_fails_before_snapshot_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, _ = self.fixture(td)
            run["inputs"].pop(sorted(REQUIRED_INPUT_ROLES)[0])
            root = td / "snapshot"

            with self.assertRaises(ContractError):
                snapshot.build_scientific_input_snapshot(run, root)
            self.assertFalse(root.exists())

    def test_existing_snapshot_destination_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            run, _ = self.fixture(td)
            root = td / "snapshot"
            root.mkdir()
            sentinel = root / "sentinel"
            sentinel.write_text("retain", encoding="utf-8")

            with self.assertRaises(ContractError):
                snapshot.build_scientific_input_snapshot(run, root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "retain")

    def test_content_identity_is_path_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            first_root = td / "first"
            second_root = td / "second"
            first_root.mkdir()
            second_root.mkdir()
            run_a, _ = self.fixture(first_root)
            run_b, _ = self.fixture(second_root)

            a = snapshot.build_scientific_input_snapshot(run_a, first_root / "snapshot")
            b = snapshot.build_scientific_input_snapshot(run_b, second_root / "snapshot")
            self.assertEqual(a.input_set_digest, b.input_set_digest)
            self.assertEqual(dict(a.expected_digests), dict(b.expected_digests))


if __name__ == "__main__":
    unittest.main(verbosity=2)
