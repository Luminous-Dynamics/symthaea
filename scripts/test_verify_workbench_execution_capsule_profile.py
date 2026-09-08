#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

import verify_workbench_execution_capsule_profile as verifier

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "data/neuroscience/workbench_execution_capsule_profile_v1.json"
FLAKE_LOCK = ROOT / "flake.lock"


class CapsuleProfileTests(unittest.TestCase):
    def fixture(self, td: Path) -> tuple[Path, Path, dict, dict]:
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        lock = json.loads(FLAKE_LOCK.read_text(encoding="utf-8"))
        pp = td / "profile.json"
        lp = td / "flake.lock"
        pp.write_text(json.dumps(profile, sort_keys=True), encoding="utf-8")
        lp.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
        return pp, lp, profile, lock

    def rewrite(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def test_current_profile_binds_current_flake_lock(self):
        verifier.validate(PROFILE, FLAKE_LOCK)

    def test_nixpkgs_revision_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, profile, _ = self.fixture(Path(temp))
            profile["flake_selection"]["rev"] = "0" * 40
            self.rewrite(pp, profile)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)

    def test_root_locked_node_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, _, lock = self.fixture(Path(temp))
            root = lock["root"]
            lock["nodes"][root]["inputs"]["nixpkgs"] = "nixpkgs"
            self.rewrite(lp, lock)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)

    def test_package_metadata_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, profile, _ = self.fixture(Path(temp))
            profile["nixpkgs_package"]["version"] = "2.1.1"
            self.rewrite(pp, profile)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)

    def test_platform_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, profile, _ = self.fixture(Path(temp))
            profile["qualification_platform"] = "aarch64-linux"
            self.rewrite(pp, profile)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)

    def test_execution_environment_drift_rejected(self):
        for key, replacement in (
            ("lang", "en_US.UTF-8"),
            ("timezone", "Africa/Johannesburg"),
            ("omp_num_threads", "8"),
            ("omp_dynamic", "TRUE"),
            ("home_policy", "operator-home"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                pp, lp, profile, _ = self.fixture(Path(temp))
                profile["execution_environment"][key] = replacement
                self.rewrite(pp, profile)
                with self.assertRaises(verifier.ContractError):
                    verifier.validate(pp, lp)

    def test_closure_contract_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, profile, _ = self.fixture(Path(temp))
            profile["future_closure_receipt"]["require_per_path_nar_hash"] = False
            self.rewrite(pp, profile)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)

    def test_authority_escalation_rejected(self):
        for key in (
            "closure_realized",
            "closure_qualified",
            "transform_executed",
            "scientific_execution_qualified",
            "atlas_correctness_established",
            "fmq010_established",
            "neural_alignment_established",
            "consciousness_evidence",
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                pp, lp, profile, _ = self.fixture(Path(temp))
                profile = copy.deepcopy(profile)
                profile["authority"][key] = True
                self.rewrite(pp, profile)
                with self.assertRaises(verifier.ContractError):
                    verifier.validate(pp, lp)

    def test_unknown_profile_field_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pp, lp, profile, _ = self.fixture(Path(temp))
            profile["qualified"] = True
            self.rewrite(pp, profile)
            with self.assertRaises(verifier.ContractError):
                verifier.validate(pp, lp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
