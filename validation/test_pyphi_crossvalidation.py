# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

"""Fail-closed regression tests for validation/pyphi_crossvalidation.py.

These tests mock process and PyPhi boundaries. They test failure semantics and
model/profile binding without claiming that the reference comparison itself ran.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).with_name("pyphi_crossvalidation.py")
SPEC = importlib.util.spec_from_file_location("symthaea_phi_crossvalidation_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _validator():
    return MODULE.PyPhiValidator(symthaea_binary="cargo")


def _payload():
    return {
        "schema_version": MODULE.MODEL_SCHEMA_VERSION,
        "model_sha256": "a" * 64,
        "formalism_profile": MODULE.FORMALISM_PROFILE,
        "n_nodes": 2,
        "state": [1, 1],
        "noise_level": MODULE.NOISE_LEVEL,
        "topology_name": "fixture_pair",
        "tpm": [
            [0.02, 0.02],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.98, 0.98],
        ],
    }


def _candidate_output(payload, **overrides):
    result = {
        "schema_version": "symthaea-iit3-tpm-output-v1",
        "metric_id": MODULE.RUST_CANDIDATE_METRIC_ID,
        "formalism_profile": payload["formalism_profile"],
        "model_sha256": payload["model_sha256"],
        "request_blake3": "c" * 64,
        "n_nodes": payload["n_nodes"],
        "state": payload["state"],
        "value": 0.125,
        "partition_search": "exhaustive_bipartitions_for_full_system_mechanism_only",
        "claim_ceiling": "candidate score; formal equivalence to PyPhi is not assumed",
    }
    result.update(overrides)
    return result


class CandidateInvocationTests(unittest.TestCase):
    def test_valid_candidate_result_is_parsed(self):
        validator = _validator()
        payload = _payload()
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0,
            stdout=json.dumps(_candidate_output(payload)), stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            value, elapsed, metric_id, request_digest = validator.compute_candidate_phi(payload)
        self.assertEqual(value, 0.125)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(metric_id, MODULE.RUST_CANDIDATE_METRIC_ID)
        self.assertEqual(request_digest, "c" * 64)

    def test_nonzero_exit_fails_without_estimated_score(self):
        validator = _validator()
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=2, stdout="", stderr="example missing"
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "exited 2"):
                validator.compute_candidate_phi(_payload())
        self.assertFalse(hasattr(validator, "_estimate_hdc_phi"))

    def test_missing_executable_fails_closed(self):
        with patch.object(MODULE.subprocess, "run", side_effect=FileNotFoundError("cargo")):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                _validator().compute_candidate_phi(_payload())

    def test_timeout_fails_closed(self):
        timeout = subprocess.TimeoutExpired(cmd=["cargo"], timeout=1)
        with patch.object(MODULE.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                _validator().compute_candidate_phi(_payload())

    def test_malformed_json_fails_closed(self):
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout="this is not JSON", stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
                _validator().compute_candidate_phi(_payload())

    def test_model_profile_and_metric_identity_are_checked(self):
        validator = _validator()
        payload = _payload()
        wrong_model = _candidate_output(payload, model_sha256="b" * 64)
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout=json.dumps(wrong_model), stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "model_sha256"):
                validator.compute_candidate_phi(payload)

        wrong_metric = _candidate_output(payload, metric_id="unrelated_metric")
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout=json.dumps(wrong_metric), stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "metric identity"):
                validator.compute_candidate_phi(payload)

    def test_invalid_rust_request_digest_fails_closed(self):
        validator = _validator()
        payload = _payload()
        output = _candidate_output(payload, request_blake3="not-a-digest")
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "request_blake3"):
                validator.compute_candidate_phi(payload)

    def test_nonfinite_candidate_value_fails_closed(self):
        validator = _validator()
        payload = _payload()
        output = _candidate_output(payload, value=float("nan"))
        completed = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "invalid candidate score"):
                validator.compute_candidate_phi(payload)


class PyPhiProfileTests(unittest.TestCase):
    def test_exact_profile_and_version_are_required(self):
        validator = _validator()
        topology = MODULE.TopologySpec("pair", 2, [(0, 1)])
        tpm = MODULE.np.asarray(_payload()["tpm"], dtype=float)
        state = MODULE.np.asarray([1, 1], dtype="uint8")
        with patch.object(MODULE, "PYPHI_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                validator.compute_pyphi_phi(topology, tpm, state)

        with patch.object(MODULE, "PYPHI_AVAILABLE", True), patch.object(
            MODULE, "PYPHI_VERSION", "1.2.0"
        ):
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                validator.compute_pyphi_phi(topology, tpm, state)

    def test_pyphi_receives_pinned_profile_and_same_tpm_state(self):
        validator = _validator()
        topology = MODULE.TopologySpec("pair", 2, [(0, 1)])
        tpm = MODULE.np.asarray(_payload()["tpm"], dtype=float)
        state = MODULE.np.asarray([1, 1], dtype="uint8")
        fake_pyphi = SimpleNamespace(
            Substrate=Mock(return_value="substrate"),
            analyze=Mock(return_value=SimpleNamespace(phi=0.25)),
        )
        with patch.object(MODULE, "PYPHI_AVAILABLE", True), patch.object(
            MODULE, "PYPHI_VERSION", MODULE.EXPECTED_PYPHI_VERSION
        ), patch.object(MODULE, "pyphi", fake_pyphi):
            value, elapsed = validator.compute_pyphi_phi(topology, tpm, state)
        self.assertEqual(value, 0.25)
        self.assertGreaterEqual(elapsed, 0.0)
        fake_pyphi.Substrate.assert_called_once()
        args, kwargs = fake_pyphi.analyze.call_args
        self.assertEqual(args[0], "substrate")
        self.assertEqual(args[1], (1, 1))
        self.assertEqual(kwargs["formalism"], "IIT_3_0")


class ReportQualificationTests(unittest.TestCase):
    def test_too_few_pairs_produces_no_correlation(self):
        report = MODULE.ValidationReport(results=[
            MODULE.ValidationResult(
                topology_name="missing",
                n_nodes=2,
                model_sha256="a" * 64,
                error_message="candidate failed",
            )
        ])
        _validator()._compute_report_metrics(report)
        self.assertIsNone(report.correlation)
        self.assertIsNone(report.mean_absolute_error)
        self.assertTrue(any("at least 2" in reason for reason in report.qualification_reasons))

    def test_constant_score_does_not_produce_nan_correlation(self):
        report = MODULE.ValidationReport(results=[
            MODULE.ValidationResult(
                "a", 2, "a" * 64, rust_payload_blake3="c" * 64,
                candidate_phi=1.0, pyphi_phi_s=0.0, candidate_metric_id=MODULE.RUST_CANDIDATE_METRIC_ID,
            ),
            MODULE.ValidationResult(
                "b", 2, "b" * 64, rust_payload_blake3="d" * 64,
                candidate_phi=1.0, pyphi_phi_s=2.0, candidate_metric_id=MODULE.RUST_CANDIDATE_METRIC_ID,
            ),
        ])
        _validator()._compute_report_metrics(report)
        self.assertIsNone(report.correlation)
        self.assertTrue(any("constant" in reason for reason in report.qualification_reasons))

    def test_missing_preregistered_gates_never_qualify(self):
        validator = _validator()
        topologies = [
            MODULE.TopologySpec("a", 2, [(0, 1)]),
            MODULE.TopologySpec("b", 2, [(0, 1)]),
        ]
        results = [
            MODULE.ValidationResult(
                "a", 2, "a" * 64, rust_payload_blake3="c" * 64,
                candidate_phi=1.0, pyphi_phi_s=1.0, candidate_metric_id=MODULE.RUST_CANDIDATE_METRIC_ID,
            ),
            MODULE.ValidationResult(
                "b", 2, "b" * 64, rust_payload_blake3="d" * 64,
                candidate_phi=2.0, pyphi_phi_s=2.0, candidate_metric_id=MODULE.RUST_CANDIDATE_METRIC_ID,
            ),
        ]
        with patch.object(MODULE, "PYPHI_AVAILABLE", True), patch.object(
            MODULE, "PYPHI_VERSION", MODULE.EXPECTED_PYPHI_VERSION
        ), patch.object(validator, "create_test_topologies", return_value=topologies), patch.object(
            validator, "validate_topology", side_effect=results
        ):
            report = validator.run_validation(max_nodes=2)
        self.assertEqual(report.qualification_status, "NOT_QUALIFIED")
        self.assertIsNotNone(report.correlation)
        self.assertTrue(any("preregistered" in reason for reason in report.qualification_reasons))

    def test_failed_report_does_not_print_pass(self):
        report = MODULE.ValidationReport(
            qualification_status="NOT_QUALIFIED",
            qualification_reasons=["reference unavailable"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            _validator().generate_report(report, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("NOT_QUALIFIED", text)
        self.assertIn("reference unavailable", text)
        self.assertNotIn("VALIDATION PASSED", text)
        self.assertIn("does not establish formal equivalence", text)


if __name__ == "__main__":
    unittest.main()
