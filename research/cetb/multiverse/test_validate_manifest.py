import copy
import json
import unittest
from pathlib import Path

from validate_manifest import ManifestError, validate_manifest


ROOT = Path(__file__).parent
SCHEMA = json.loads((ROOT / "multiverse.schema.json").read_text())
FIXTURE = json.loads((ROOT / "fixtures" / "synthetic-multiverse.json").read_text())


class MultiverseManifestTests(unittest.TestCase):
    def test_synthetic_manifest_passes(self):
        validate_manifest(FIXTURE, SCHEMA)

    def test_duplicate_option_id_fails(self):
        manifest = copy.deepcopy(FIXTURE)
        manifest["decision_nodes"][1]["options"][0]["option_id"] = "abs-wj"
        with self.assertRaisesRegex(ManifestError, "duplicate option ID"):
            validate_manifest(manifest, SCHEMA)

    def test_unknown_compatibility_reference_fails(self):
        manifest = copy.deepcopy(FIXTURE)
        manifest["compatibility_rules"][0]["option_ids"].append("not-a-real-option")
        with self.assertRaisesRegex(ManifestError, "unknown options"):
            validate_manifest(manifest, SCHEMA)

    def test_confirmatory_manifest_cannot_access_outcomes_before_freeze(self):
        manifest = copy.deepcopy(FIXTURE)
        manifest["status"] = "preregistered_confirmatory"
        manifest["freeze"]["confirmatory_outcomes_accessed"] = True
        with self.assertRaisesRegex(ManifestError, "after outcome access"):
            validate_manifest(manifest, SCHEMA)

    def test_deterministic_sampling_requires_method(self):
        manifest = copy.deepcopy(FIXTURE)
        manifest["generation"]["mode"] = "deterministic_sample"
        manifest["generation"]["sampling_method"] = " "
        with self.assertRaisesRegex(ManifestError, "sampling_method"):
            validate_manifest(manifest, SCHEMA)

    def test_primary_lane_is_required(self):
        manifest = copy.deepcopy(FIXTURE)
        for node in manifest["decision_nodes"]:
            for option in node["options"]:
                option["lane"] = "sensitivity"
        with self.assertRaisesRegex(ManifestError, "primary-lane"):
            validate_manifest(manifest, SCHEMA)

    def test_non_equivalent_estimators_are_not_implicitly_averaged(self):
        manifest = copy.deepcopy(FIXTURE)
        assessment = manifest["equivalence_assessments"][0]
        self.assertEqual(assessment["status"], "non_equivalent")
        self.assertEqual(assessment["criterion"], "construct")
        validate_manifest(manifest, SCHEMA)


if __name__ == "__main__":
    unittest.main()
