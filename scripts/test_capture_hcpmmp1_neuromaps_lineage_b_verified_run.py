#!/usr/bin/env python3
from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_hcpmmp1_neuromaps_lineage_b_verified_run as m

ROOT = Path(__file__).parents[1]
METHOD_SOURCE = ROOT / "data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json"
FAKE_ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-connectome-workbench-2.1.0"


def verification_authority() -> dict[str, bool]:
    true_keys = {
        "program_membership_verified", "invocation_profile_verified",
        "invocation_executed", "version_output_bound",
    }
    return {key: key in true_keys for key in m.VERIFICATION_AUTHORITY_KEYS}


class VerifiedRunCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.td = Path(self.temp.name)
        self.method = self.td / "method.json"
        self.method.write_bytes(METHOD_SOURCE.read_bytes())

        self.program = self.td / "wb_command"
        self.program.write_bytes(b"qualified-program-bytes")
        self.program.chmod(0o700)
        self.program_sha = m.digest_regular_file(self.program, "test program")

        self.inputs: dict[str, Path] = {}
        for role in sorted(m.REQUIRED_INPUT_ROLES):
            path = self.td / f"{role}.input"
            path.write_bytes(("scientific-" + role).encode("utf-8"))
            self.inputs[role] = path

        self.verification = {
            "schema": m.VERIFICATION_SCHEMA,
            "status": m.VERIFICATION_STATUS,
            "root": FAKE_ROOT,
            "qualification_platform": "x86_64-linux",
            "closure_digest": "sha256:" + "a" * 64,
            "program_content_sha256": self.program_sha,
            "version": "2.1.0",
            "version_output_sha256": "sha256:" + "b" * 64,
            "entry_environment_sha256": "sha256:" + "c" * 64,
            "diagnostics": {
                "cpu_vendor": "TestCPU",
                "cpu_family": "1",
                "cpu_model": "2",
                "cpu_stepping": "3",
                "cpu_flags_digest": "sha256:" + "d" * 64,
                "kernel_release": "test-kernel",
            },
            "implementations": {
                "verifier_sha256": "sha256:" + "e" * 64,
                "producer_sha256": "sha256:" + "f" * 64,
                "parent_nar_verifier_sha256": "sha256:" + "1" * 64,
                "isolation_verifier_sha256": "sha256:" + "2" * 64,
            },
            "authority": verification_authority(),
        }
        self.verification_path = self.td / "verification.json"
        self.write_verification()

        self.profile = {
            "schema": m.PROFILE_SCHEMA,
            "status": "qualified-workbench-observation-bound",
            "qualification_source": {
                "pr": 976,
                "head_sha": "5" * 40,
                "workflow_run_id": 123,
                "verification_file_path": "verification.json",
                "verification_file_sha256": m.digest_file(self.verification_path),
                "independent_verification_archive_sha256": "sha256:" + "3" * 64,
                "raw_evidence_archive_sha256": "sha256:" + "4" * 64,
            },
            "qualified_workbench": {
                "root": FAKE_ROOT,
                "relative_main_program": "bin/wb_command",
                "program_content_sha256": self.program_sha,
                "version": "2.1.0",
                "version_output_sha256": "sha256:" + "b" * 64,
                "closure_digest": "sha256:" + "a" * 64,
                "qualification_platform": "x86_64-linux",
            },
            "required_verification_authority": verification_authority(),
            "authority": {
                "verified_workbench_observation_bound": True,
                "run_manifest_captured": False,
                "local_closure_reverified": False,
                "path_equivalence_established": False,
                "cross_cpu_equivalence_established": False,
                "scientific_execution_qualified": False,
                "transform_executed": False,
                "atlas_correctness_established": False,
                "fmq010_established": False,
                "neural_alignment_established": False,
                "consciousness_evidence": False,
            },
        }
        self.profile_path = self.td / "profile.json"
        self.write_profile()

    def tearDown(self):
        self.temp.cleanup()

    def write_verification(self, canonical: bool = True) -> None:
        if canonical:
            self.verification_path.write_bytes(m.canonical_json_bytes(self.verification) + b"\n")
        else:
            self.verification_path.write_text(json.dumps(self.verification, indent=2) + "\n")

    def write_profile(self, canonical: bool = True) -> None:
        if canonical:
            self.profile_path.write_bytes(m.canonical_json_bytes(self.profile) + b"\n")
        else:
            self.profile_path.write_text(json.dumps(self.profile, indent=2) + "\n")

    def input_items(self) -> list[str]:
        return [f"{role}={self.inputs[role]}" for role in sorted(self.inputs)]

    def capture(self):
        with mock.patch.object(m, "qualified_program_path", return_value=self.program):
            return m.capture_manifest(
                self.method,
                self.profile_path,
                self.verification_path,
                self.input_items(),
                "run-1",
                "synthetic-only",
            )

    def test_valid_capture_binds_verified_workbench_without_execution(self):
        doc, metadata = self.capture()
        self.assertEqual(doc["workbench"]["path"], str(self.program))
        self.assertEqual(doc["workbench"]["sha256"], self.program_sha)
        self.assertEqual(doc["workbench"]["version_output_sha256"], self.verification["version_output_sha256"])
        self.assertEqual(set(doc["inputs"]), m.REQUIRED_INPUT_ROLES)
        self.assertEqual(metadata["verified_run_profile_sha256"], m.digest_file(self.profile_path))
        self.assertEqual(metadata["workbench_verification_file_sha256"], m.digest_file(self.verification_path))
        self.assertEqual(metadata["qualified_workbench_head"], self.profile["qualification_source"]["head_sha"])
        self.assertEqual(metadata["closure_digest"], self.verification["closure_digest"])
        self.assertEqual(metadata["authority"], m.CAPTURE_AUTHORITY)

    def test_verification_is_read_exactly_once_inside_capture_window(self):
        original = Path.read_bytes
        reads = 0

        def counted(path: Path) -> bytes:
            nonlocal reads
            if path == self.verification_path:
                reads += 1
                if reads > 1:
                    raise AssertionError("verification file re-read after root binding")
            return original(path)

        with mock.patch.object(Path, "read_bytes", new=counted), \
             mock.patch.object(m, "qualified_program_path", return_value=self.program):
            m.capture_manifest(
                self.method, self.profile_path, self.verification_path,
                self.input_items(), "run-1", "synthetic-only"
            )
        self.assertEqual(reads, 1)

    def test_retained_verification_root_substitution_rejected(self):
        self.verification["version"] = "2.2.0"
        self.write_verification()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_noncanonical_retained_verification_rejected_even_with_matching_file_root(self):
        self.write_verification(canonical=False)
        self.profile["qualification_source"]["verification_file_sha256"] = m.digest_file(self.verification_path)
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_noncanonical_profile_rejected(self):
        self.write_profile(canonical=False)
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_verification_authority_escalation_rejected(self):
        self.verification["authority"]["scientific_execution_qualified"] = True
        self.write_verification()
        self.profile["qualification_source"]["verification_file_sha256"] = m.digest_file(self.verification_path)
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_profile_authority_escalation_rejected(self):
        self.profile["authority"]["transform_executed"] = True
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_bool_cannot_launder_as_profile_integer(self):
        self.profile["qualification_source"]["pr"] = True
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_profile_workbench_root_mismatch_rejected(self):
        self.profile["qualified_workbench"]["root"] = FAKE_ROOT.replace("connectome", "different")
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_local_program_byte_mismatch_rejected(self):
        self.program.write_bytes(b"mutated-program")
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_input_drift_during_capture_rejected(self):
        role = sorted(self.inputs)[0]
        victim = self.inputs[role]
        original = m.digest_regular_file
        calls = 0

        def drifting(path: Path, label: str) -> str:
            nonlocal calls
            if path == victim:
                calls += 1
                if calls == 2:
                    path.write_bytes(b"mutated-input")
            return original(path, label)

        with mock.patch.object(m, "qualified_program_path", return_value=self.program), \
             mock.patch.object(m, "digest_regular_file", side_effect=drifting):
            with self.assertRaises(m.CaptureError):
                m.capture_manifest(
                    self.method, self.profile_path, self.verification_path,
                    self.input_items(), "run-1", "synthetic-only"
                )

    def test_program_drift_during_capture_rejected(self):
        original = m.digest_regular_file
        program_calls = 0

        def drifting(path: Path, label: str) -> str:
            nonlocal program_calls
            if path == self.program:
                program_calls += 1
                if program_calls == 2:
                    path.write_bytes(b"program-drift")
            return original(path, label)

        with mock.patch.object(m, "qualified_program_path", return_value=self.program), \
             mock.patch.object(m, "digest_regular_file", side_effect=drifting):
            with self.assertRaises(m.CaptureError):
                m.capture_manifest(
                    self.method, self.profile_path, self.verification_path,
                    self.input_items(), "run-1", "synthetic-only"
                )

    def test_direct_input_symlink_rejected(self):
        role = sorted(self.inputs)[0]
        link = self.td / "input-link"
        link.symlink_to(self.inputs[role])
        items = [
            f"{candidate}={link if candidate == role else self.inputs[candidate]}"
            for candidate in sorted(self.inputs)
        ]
        with mock.patch.object(m, "qualified_program_path", return_value=self.program):
            with self.assertRaises(m.CaptureError):
                m.capture_manifest(
                    self.method, self.profile_path, self.verification_path,
                    items, "run-1", "synthetic-only"
                )

    def test_missing_role_rejected(self):
        with self.assertRaises(m.CaptureError):
            m.parse_inputs(self.input_items()[:-1])

    def test_duplicate_role_rejected(self):
        items = self.input_items()
        with self.assertRaises(m.CaptureError):
            m.parse_inputs(items + [items[0]])

    def test_unknown_verification_field_rejected(self):
        self.verification["qualified"] = True
        self.write_verification()
        self.profile["qualification_source"]["verification_file_sha256"] = m.digest_file(self.verification_path)
        self.write_profile()
        with self.assertRaises((m.CaptureError, m.ContractError)):
            self.capture()

    def test_duplicate_json_key_rejected(self):
        self.verification_path.write_text('{"schema":"x","schema":"y"}\n')
        self.profile["qualification_source"]["verification_file_sha256"] = m.digest_file(self.verification_path)
        self.write_profile()
        with self.assertRaises(m.CaptureError):
            self.capture()

    def test_output_is_private_and_never_overwritten(self):
        doc, _ = self.capture()
        target = self.td / "run.json"
        m.write_new(target, doc)
        before = target.read_bytes()
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        with self.assertRaises(m.CaptureError):
            m.write_new(target, doc)
        self.assertEqual(target.read_bytes(), before)

    def test_source_has_no_subprocess_surface(self):
        source = Path(m.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertNotIn('"-version"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
