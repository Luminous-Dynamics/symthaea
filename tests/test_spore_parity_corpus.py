#!/usr/bin/env python3
"""Negative tests for the Spore extraction parity corpus contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_spore_parity_corpus.py"
CORPUS = ROOT / "docs" / "architecture" / "spore-parity-corpus-v1.json"
MANIFEST = ROOT / "docs" / "architecture" / "spore-migration-manifest-v1.json"

spec = importlib.util.spec_from_file_location("spore_parity_checker", CHECKER)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ParityCorpusContractTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load(CORPUS)
        self.manifest = load(MANIFEST)

    def test_frozen_corpus_is_valid(self):
        self.assertEqual(mod.validate(self.corpus, self.manifest), [])

    def test_qualification_may_never_be_inherited(self):
        broken = copy.deepcopy(self.corpus)
        broken["qualification_transfer_policy"] = "inherit"
        errors = mod.validate(broken, self.manifest)
        self.assertTrue(any("never-inherit" in error for error in errors), errors)

    def test_unknown_source_artifact_is_rejected(self):
        broken = copy.deepcopy(self.corpus)
        broken["behaviors"][0]["artifact_id"] = "invented-fixture"
        errors = mod.validate(broken, self.manifest)
        self.assertTrue(any("unknown manifest artifact" in error for error in errors), errors)

    def test_duplicate_behavior_ids_are_rejected(self):
        broken = copy.deepcopy(self.corpus)
        broken["behaviors"][1]["id"] = broken["behaviors"][0]["id"]
        errors = mod.validate(broken, self.manifest)
        self.assertTrue(any("duplicate behavior id" in error for error in errors), errors)

    def test_required_domain_cannot_disappear(self):
        broken = copy.deepcopy(self.corpus)
        for behavior in broken["behaviors"]:
            behavior["domains"] = [
                domain for domain in behavior["domains"] if domain != "firmware-recovery"
            ]
        errors = mod.validate(broken, self.manifest)
        self.assertTrue(any("does not cover required domains" in error for error in errors), errors)

    def test_source_fixture_must_still_require_destination_qualification(self):
        broken_manifest = copy.deepcopy(self.manifest)
        for artifact in broken_manifest["artifacts"]:
            if artifact["id"] == "fail-open-vm":
                artifact["destination_qualification"] = "inherited"
                break
        errors = mod.validate(self.corpus, broken_manifest)
        self.assertTrue(
            any("must require destination qualification" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
