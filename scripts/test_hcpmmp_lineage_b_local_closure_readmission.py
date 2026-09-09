#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import hcpmmp_lineage_b_local_closure_readmission as m


def path_info(root: str, nar_byte: int = 0, refs: list[str] | None = None) -> bytes:
    basename = root.removeprefix("/nix/store/")
    raw = bytes([nar_byte]) * 32
    doc = {
        "version": 2,
        "storeDir": "/nix/store",
        "info": {
            basename: {
                "narHash": "sha256-" + base64.b64encode(raw).decode("ascii"),
                "references": refs or [],
            }
        },
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def completed(rc: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class ReadmissionTests(unittest.TestCase):
    def test_real_retained_trust_package_binds(self):
        verification = m.bind_qualified_trust_package()
        self.assertEqual(verification["root"], m.EXPECTED_ROOT)
        self.assertEqual(verification["closure_digest"], m.EXPECTED_CLOSURE_DIGEST)
        self.assertEqual(verification["program_content_sha256"], m.EXPECTED_PROGRAM_SHA256)

    def test_profile_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile.json"
            raw = m.qualified_run.PROFILE_PATH.read_bytes()
            p.write_bytes(raw + b" ")
            with mock.patch.object(m.qualified_run, "PROFILE_PATH", p):
                with self.assertRaisesRegex(m.ReadmissionError, "profile root mismatch"):
                    m.bind_qualified_trust_package()

    def test_verification_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "verification.json"
            raw = m.qualified_run.VERIFICATION_PATH.read_bytes()
            p.write_bytes(raw + b" ")
            with mock.patch.object(m.qualified_run, "VERIFICATION_PATH", p):
                with self.assertRaisesRegex(m.ReadmissionError, "verification root mismatch"):
                    m.bind_qualified_trust_package()

    def test_exact_nix_version_required(self):
        self.assertEqual(m.parse_nix_version(b"nix (Nix) 2.33.6\n"), "2.33.6")
        for raw in (b"nix (Nix) 2.33.5\n", b"nix (Nix) 2.34.0\n", b"2.33.6\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(m.ReadmissionError):
                    m.parse_nix_version(raw)

    def test_local_identity_compiles_only_at_expected_digest(self):
        raw = path_info(m.EXPECTED_ROOT)
        entries = m.closure_capture_verifier.parse_path_info_v2(raw)
        identity = m.closure_identity.compile_identity(m.EXPECTED_ROOT, entries)
        with mock.patch.object(m, "EXPECTED_CLOSURE_DIGEST", identity["closure_digest"]):
            self.assertEqual(m.compile_local_identity(raw)["closure_digest"], identity["closure_digest"])
        with mock.patch.object(m, "EXPECTED_CLOSURE_DIGEST", "sha256:" + "f" * 64):
            with self.assertRaisesRegex(m.ReadmissionError, "qualified closure digest mismatch"):
                m.compile_local_identity(raw)

    def test_orphan_or_reference_drift_fails_normalization(self):
        basename = m.EXPECTED_ROOT.removeprefix("/nix/store/")
        missing = "1" * 32 + "-missing"
        raw = json.dumps({
            "version": 2,
            "storeDir": "/nix/store",
            "info": {
                basename: {
                    "narHash": "sha256-" + base64.b64encode(bytes(32)).decode(),
                    "references": [missing],
                }
            },
        }, separators=(",", ":")).encode()
        with self.assertRaises(m.ReadmissionError):
            m.compile_local_identity(raw)

    def _successful_mock_context(self):
        raw = path_info(m.EXPECTED_ROOT)
        entries = m.closure_capture_verifier.parse_path_info_v2(raw)
        identity = m.closure_identity.compile_identity(m.EXPECTED_ROOT, entries)
        verification = {
            "root": m.EXPECTED_ROOT,
            "closure_digest": identity["closure_digest"],
        }
        responses = [
            completed(0, b"nix (Nix) 2.33.6\n", b""),
            completed(0, raw, b""),
            completed(0, b"", b"verified contents\n"),
        ]
        return identity, verification, responses

    def test_success_requires_exact_command_topology(self):
        identity, verification, responses = self._successful_mock_context()
        calls: list[list[str]] = []
        def run(argv):
            calls.append(list(argv))
            return responses.pop(0)
        real_resolve = Path.resolve
        real_is_file = Path.is_file
        program = Path(m.EXPECTED_ROOT) / m.PROGRAM_RELATIVE_PATH
        def resolve_side_effect(path, strict=False):
            if path == program:
                return path
            return real_resolve(path, strict=strict)
        def is_file_side_effect(path):
            if path == program:
                return True
            return real_is_file(path)
        with (
            mock.patch.object(m, "EXPECTED_CLOSURE_DIGEST", identity["closure_digest"]),
            mock.patch.object(m, "bind_qualified_trust_package", return_value=verification),
            mock.patch.object(m, "digest_regular_file", return_value=m.EXPECTED_PROGRAM_SHA256),
            mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve_side_effect),
            mock.patch.object(Path, "is_file", autospec=True, side_effect=is_file_side_effect),
            mock.patch.object(Path, "is_symlink", autospec=True, return_value=False),
        ):
            result = m.verify_local_closure(run_command=run, nix_executable=Path("/bin/true"))
        self.assertEqual(result["qualified_workbench"]["closure_digest"], identity["closure_digest"])
        self.assertTrue(result["authority"]["local_closure_readmitted"])
        self.assertIn("--offline", calls[1])
        self.assertEqual(calls[1][2:8], ["path-info", "--json", "--json-format", "2", "--recursive", m.EXPECTED_ROOT])
        self.assertEqual(calls[2][2:], ["store", "verify", "--recursive", "--no-trust", m.EXPECTED_ROOT])
        self.assertFalse(result["authority"]["workbench_execution_qualified"])
        self.assertFalse(result["authority"]["transform_executed"])

    def test_nix_version_command_failure_rejected(self):
        with (
            mock.patch.object(m, "bind_qualified_trust_package", return_value={"root": m.EXPECTED_ROOT}),
            mock.patch.object(Path, "resolve", autospec=True, return_value=Path("/bin/true")),
        ):
            with self.assertRaisesRegex(m.ReadmissionError, "nix --version failed"):
                m.verify_local_closure(run_command=lambda argv: completed(1), nix_executable=Path("/bin/true"))

    def test_path_info_failure_rejected(self):
        identity, verification, _ = self._successful_mock_context()
        responses = [completed(0, b"nix (Nix) 2.33.6\n"), completed(3, b"", b"bad")]
        program = Path(m.EXPECTED_ROOT) / m.PROGRAM_RELATIVE_PATH
        real_resolve = Path.resolve
        def resolve_side_effect(path, strict=False):
            if path == program:
                return path
            return real_resolve(path, strict=strict)
        with (
            mock.patch.object(m, "bind_qualified_trust_package", return_value=verification),
            mock.patch.object(m, "digest_regular_file", return_value=m.EXPECTED_PROGRAM_SHA256),
            mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve_side_effect),
            mock.patch.object(Path, "is_file", autospec=True, return_value=True),
            mock.patch.object(Path, "is_symlink", autospec=True, return_value=False),
        ):
            with self.assertRaisesRegex(m.ReadmissionError, "path-info failed"):
                m.verify_local_closure(run_command=lambda argv: responses.pop(0), nix_executable=Path("/bin/true"))

    def test_store_content_verification_failure_rejected(self):
        identity, verification, responses = self._successful_mock_context()
        responses[-1] = completed(1, b"", b"corrupt")
        program = Path(m.EXPECTED_ROOT) / m.PROGRAM_RELATIVE_PATH
        real_resolve = Path.resolve
        def resolve_side_effect(path, strict=False):
            if path == program:
                return path
            return real_resolve(path, strict=strict)
        with (
            mock.patch.object(m, "EXPECTED_CLOSURE_DIGEST", identity["closure_digest"]),
            mock.patch.object(m, "bind_qualified_trust_package", return_value=verification),
            mock.patch.object(m, "digest_regular_file", return_value=m.EXPECTED_PROGRAM_SHA256),
            mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve_side_effect),
            mock.patch.object(Path, "is_file", autospec=True, return_value=True),
            mock.patch.object(Path, "is_symlink", autospec=True, return_value=False),
        ):
            with self.assertRaisesRegex(m.ReadmissionError, "NAR-content verification failed"):
                m.verify_local_closure(run_command=lambda argv: responses.pop(0), nix_executable=Path("/bin/true"))

    def test_program_pre_identity_mismatch_rejected_before_closure_query(self):
        verification = {"root": m.EXPECTED_ROOT}
        calls = {"n": 0}
        def run(argv):
            calls["n"] += 1
            return completed(0, b"nix (Nix) 2.33.6\n")
        program = Path(m.EXPECTED_ROOT) / m.PROGRAM_RELATIVE_PATH
        real_resolve = Path.resolve
        def resolve_side_effect(path, strict=False):
            if path == program:
                return path
            return real_resolve(path, strict=strict)
        with (
            mock.patch.object(m, "bind_qualified_trust_package", return_value=verification),
            mock.patch.object(m, "digest_regular_file", return_value="sha256:" + "0" * 64),
            mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve_side_effect),
            mock.patch.object(Path, "is_file", autospec=True, return_value=True),
            mock.patch.object(Path, "is_symlink", autospec=True, return_value=False),
        ):
            with self.assertRaisesRegex(m.ReadmissionError, "main program bytes differ"):
                m.verify_local_closure(run_command=run, nix_executable=Path("/bin/true"))
        self.assertEqual(calls["n"], 1)

    def test_program_post_mutation_rejected(self):
        identity, verification, responses = self._successful_mock_context()
        digests = [m.EXPECTED_PROGRAM_SHA256, "sha256:" + "f" * 64]
        program = Path(m.EXPECTED_ROOT) / m.PROGRAM_RELATIVE_PATH
        real_resolve = Path.resolve
        def resolve_side_effect(path, strict=False):
            if path == program:
                return path
            return real_resolve(path, strict=strict)
        with (
            mock.patch.object(m, "EXPECTED_CLOSURE_DIGEST", identity["closure_digest"]),
            mock.patch.object(m, "bind_qualified_trust_package", return_value=verification),
            mock.patch.object(m, "digest_regular_file", side_effect=lambda *a: digests.pop(0)),
            mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve_side_effect),
            mock.patch.object(Path, "is_file", autospec=True, return_value=True),
            mock.patch.object(Path, "is_symlink", autospec=True, return_value=False),
        ):
            with self.assertRaisesRegex(m.ReadmissionError, "changed during readmission"):
                m.verify_local_closure(run_command=lambda argv: responses.pop(0), nix_executable=Path("/bin/true"))

    def test_authority_exactly_preserves_science_false(self):
        true_keys = {
            "qualified_workbench_trust_package_bound", "local_nix_version_verified",
            "local_closure_metadata_matches", "local_closure_contents_verified",
            "local_program_bytes_match", "local_closure_readmitted",
        }
        self.assertEqual({k for k, v in m.AUTHORITY.items() if v}, true_keys)
        for key in (
            "workbench_execution_qualified", "scientific_execution_qualified", "transform_executed",
            "same_host_repeatability_established", "path_equivalence_established",
            "cross_cpu_equivalence_established", "atlas_correctness_established", "fmq010_established",
            "neural_alignment_established", "consciousness_evidence",
        ):
            self.assertFalse(m.AUTHORITY[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
