from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts" / "check_spore_migration_manifest.py"
MANIFEST_PATH = ROOT / "docs" / "architecture" / "spore-migration-manifest-v1.json"

spec = importlib.util.spec_from_file_location("check_spore_migration_manifest", CHECKER_PATH)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SporeMigrationManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_checked_in_manifest_is_valid(self) -> None:
        checker.validate_manifest(copy.deepcopy(self.manifest))

    def test_destination_qualification_cannot_be_inherited(self) -> None:
        candidate = copy.deepcopy(self.manifest)
        candidate["artifacts"][0]["destination_qualification"] = "passed"
        with self.assertRaises(checker.ManifestError):
            checker.validate_manifest(candidate)

    def test_source_mutation_requires_transformed_candidate_tier(self) -> None:
        candidate = copy.deepcopy(self.manifest)
        candidate["lineages"][0]["qualification_boundary"] = "exact-committed-source"
        with self.assertRaises(checker.ManifestError):
            checker.validate_manifest(candidate)

    def test_recovery_authority_targets_spore(self) -> None:
        candidate = copy.deepcopy(self.manifest)
        candidate["artifacts"][0]["target_owner"] = "symthaea"
        with self.assertRaises(checker.ManifestError):
            checker.validate_manifest(candidate)

    def test_destination_path_cannot_exist_before_repository(self) -> None:
        candidate = copy.deepcopy(self.manifest)
        candidate["artifacts"][0]["destination_path"] = "nix/modules/spore.nix"
        with self.assertRaises(checker.ManifestError):
            checker.validate_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
