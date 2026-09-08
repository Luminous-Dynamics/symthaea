#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import verify_ci_source_tree_identity as source

PROFILE_PATH = Path("data/ci/ci_source_tree_identity_profile_v1.json")
HEAD = "1" * 40
MERGE = "2" * 40
TREE = "3" * 40
WORKFLOW_SHA = "5" * 40
BASE_REPO_ID = "1136141775"
HEAD_REPO_ID = "1136141775"
WORKFLOW_REF = "Luminous-Dynamics/symthaea/.github/workflows/ci-source-tree-identity.yml@refs/pull/1/merge"


def profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def observation(lane="exact-pr-head", observed=None):
    expected = HEAD if lane == "exact-pr-head" else MERGE
    return {
        "schema": source.OBS_SCHEMA,
        "repository": "Luminous-Dynamics/symthaea",
        "repository_id": BASE_REPO_ID,
        "head_repository": "Luminous-Dynamics/symthaea",
        "head_repository_id": HEAD_REPO_ID,
        "workflow_ref": WORKFLOW_REF,
        "workflow_sha": WORKFLOW_SHA,
        "event_name": "pull_request",
        "lane": lane,
        "event_head_sha": HEAD,
        "event_merge_sha": MERGE,
        "observed_checkout_sha": observed or expected,
        "observed_tree_sha": TREE,
    }


class SourceTreeIdentityContracts(unittest.TestCase):
    def test_profile_validates(self):
        self.assertEqual(source.verify_profile(profile())["schema"], source.PROFILE_SCHEMA)

    def test_exact_head_observation_verifies_only_head_source(self):
        result = source.compile_observation(profile(), observation())
        self.assertEqual(result["expected_checkout_sha"], HEAD)
        self.assertTrue(result["authority"]["source_tree_identity_verified"])
        self.assertTrue(result["authority"]["exact_pr_head_source_verified"])
        self.assertFalse(result["authority"]["pr_merge_source_verified"])
        self.assertFalse(result["authority"]["focused_theorem_passed"])
        self.assertEqual(result["tree_sha_identity_binding"], "diagnostic-only")
        self.assertEqual(result["workflow_definition_identity_binding"], "separate-qualification-envelope")

    def test_merge_observation_verifies_only_merge_source(self):
        result = source.compile_observation(profile(), observation("pr-merge-compatibility"))
        self.assertEqual(result["expected_checkout_sha"], MERGE)
        self.assertFalse(result["authority"]["exact_pr_head_source_verified"])
        self.assertTrue(result["authority"]["pr_merge_source_verified"])
        self.assertFalse(result["authority"]["merge_compatibility_passed"])

    def test_head_lane_rejects_merge_checkout(self):
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), observation(observed=MERGE))

    def test_merge_lane_rejects_head_checkout(self):
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), observation("pr-merge-compatibility", HEAD))

    def test_exact_head_allows_absent_merge_sha(self):
        value = observation(); value["event_merge_sha"] = None
        result = source.compile_observation(profile(), value)
        self.assertTrue(result["authority"]["exact_pr_head_source_verified"])
        self.assertIsNone(result["event_merge_sha"])

    def test_merge_lane_requires_merge_sha(self):
        value = observation("pr-merge-compatibility"); value["event_merge_sha"] = None
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_head_and_merge_need_not_be_assumed_distinct(self):
        value = observation(); value["event_merge_sha"] = HEAD
        result = source.compile_observation(profile(), value)
        self.assertTrue(result["authority"]["exact_pr_head_source_verified"])

    def test_identity_is_deterministic(self):
        a = source.compile_observation(profile(), observation())
        b = source.compile_observation(profile(), dict(reversed(list(observation().items()))))
        self.assertEqual(a, b)

    def test_tree_sha_is_retained_but_does_not_rename_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["observed_tree_sha"] = "4" * 40
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])
        self.assertNotEqual(a["observed_tree_sha"], b["observed_tree_sha"])

    def test_tree_sha_does_not_rename_merge_identity(self):
        a = source.compile_observation(profile(), observation("pr-merge-compatibility"))
        b_obs = observation("pr-merge-compatibility"); b_obs["observed_tree_sha"] = "4" * 40
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_workflow_sha_is_retained_but_does_not_rename_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["workflow_sha"] = "6" * 40
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])
        self.assertNotEqual(a["workflow_sha"], b["workflow_sha"])

    def test_workflow_ref_is_retained_but_does_not_rename_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["workflow_ref"] = "Luminous-Dynamics/symthaea/.github/workflows/ci-source-tree-identity.yml@refs/pull/2/merge"
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_workflow_definition_context_does_not_rename_merge_identity(self):
        a = source.compile_observation(profile(), observation("pr-merge-compatibility"))
        b_obs = observation("pr-merge-compatibility")
        b_obs["workflow_sha"] = "6" * 40
        b_obs["workflow_ref"] = "Luminous-Dynamics/symthaea/.github/workflows/ci-source-tree-identity.yml@refs/pull/2/merge"
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_merge_sha_does_not_rename_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["event_merge_sha"] = "4" * 40
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_absent_merge_sha_does_not_rename_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["event_merge_sha"] = None
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_head_sha_does_not_rename_same_merge_identity(self):
        a = source.compile_observation(profile(), observation("pr-merge-compatibility"))
        b_obs = observation("pr-merge-compatibility"); b_obs["event_head_sha"] = "4" * 40
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_repository_display_name_is_diagnostic_not_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["repository"] = "luminous-dynamics/SYMTHAEA"
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_head_repository_display_name_is_diagnostic_not_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["head_repository"] = "other-owner/symthaea-fork"
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_base_repository_id_changes_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["repository_id"] = "999"
        b = source.compile_observation(profile(), b_obs)
        self.assertNotEqual(a["identity_sha256"], b["identity_sha256"])

    def test_head_repository_id_changes_exact_head_identity(self):
        a = source.compile_observation(profile(), observation())
        b_obs = observation(); b_obs["head_repository_id"] = "999"
        b = source.compile_observation(profile(), b_obs)
        self.assertNotEqual(a["identity_sha256"], b["identity_sha256"])

    def test_head_repository_id_does_not_rename_merge_identity(self):
        a = source.compile_observation(profile(), observation("pr-merge-compatibility"))
        b_obs = observation("pr-merge-compatibility"); b_obs["head_repository_id"] = "999"
        b = source.compile_observation(profile(), b_obs)
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])

    def test_fork_head_identity_names_both_base_and_source_repository_ids(self):
        value = observation(); value["head_repository"] = "fork-owner/symthaea"; value["head_repository_id"] = "777"
        result = source.compile_observation(profile(), value)
        self.assertEqual(result["identity_preimage"]["base_repository_id"], BASE_REPO_ID)
        self.assertEqual(result["identity_preimage"]["source_repository_id"], "777")

    def test_merge_identity_source_repository_is_base_repository(self):
        value = observation("pr-merge-compatibility"); value["head_repository_id"] = "777"
        result = source.compile_observation(profile(), value)
        self.assertEqual(result["identity_preimage"]["source_repository_id"], BASE_REPO_ID)

    def test_workflow_dispatch_cannot_be_observation(self):
        value = observation(); value["event_name"] = "workflow_dispatch"
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_unknown_lane_rejected(self):
        value = observation(); value["lane"] = "something-else"
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_uppercase_sha_rejected(self):
        value = observation(); value["event_head_sha"] = "A" * 40
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_bad_workflow_sha_rejected(self):
        value = observation(); value["workflow_sha"] = "A" * 40
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_bad_workflow_ref_rejected(self):
        for bad in ("", "workflow.yml", "owner/repo/workflow.yml@main", "owner/repo/.github/workflows/x.yml\n@main"):
            value = observation(); value["workflow_ref"] = bad
            with self.subTest(bad=bad), self.assertRaises(source.SourceIdentityError):
                source.compile_observation(profile(), value)

    def test_short_sha_rejected(self):
        value = observation(); value["observed_checkout_sha"] = "1" * 7
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_unknown_observation_field_rejected(self):
        value = observation(); value["authority"] = True
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_bad_repository_rejected(self):
        value = observation(); value["repository"] = "symthaea"
        with self.assertRaises(source.SourceIdentityError):
            source.compile_observation(profile(), value)

    def test_path_like_repository_segment_rejected(self):
        for bad in ("../symthaea", "owner/..", "./symthaea", "owner/."):
            value = observation(); value["repository"] = bad
            with self.subTest(bad=bad), self.assertRaises(source.SourceIdentityError):
                source.compile_observation(profile(), value)

    def test_repository_id_must_be_canonical_decimal_string(self):
        for bad in (0, 1, "0", "01", "-1", "abc"):
            value = observation(); value["repository_id"] = bad
            with self.subTest(bad=bad), self.assertRaises(source.SourceIdentityError):
                source.compile_observation(profile(), value)

    def test_head_repository_id_must_be_canonical_decimal_string(self):
        for bad in (0, 1, "0", "01", "-1", "abc"):
            value = observation(); value["head_repository_id"] = bad
            with self.subTest(bad=bad), self.assertRaises(source.SourceIdentityError):
                source.compile_observation(profile(), value)

    def test_theorem_checkout_expression_drift_rejected(self):
        value = profile(); value["pull_request"]["theorem_qualification"]["checkout_ref_expression"] = "${{ github.sha }}"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_merge_checkout_expression_drift_rejected(self):
        value = profile(); value["pull_request"]["merge_compatibility"]["checkout_ref_expression"] = "${{ github.event.pull_request.head.sha }}"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_lane_meaning_drift_rejected(self):
        value = profile(); value["pull_request"]["merge_compatibility"]["meaning"] = "theorem-source-identity"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_repository_identity_expression_drift_rejected(self):
        value = profile(); value["repository_identity"]["base_repository_id_expression"] = "${{ github.repository }}"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_workflow_context_expression_drift_rejected(self):
        value = profile(); value["workflow_definition_context"]["workflow_sha_expression"] = "${{ github.sha }}"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_workflow_definition_separation_downgrade_rejected(self):
        value = profile(); value["workflow_definition_context"]["workflow_definition_identity_separate"] = False
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_dispatch_promotion_rejected(self):
        for key in ("counts_as_exact_pr_head_qualification", "counts_as_pr_merge_compatibility"):
            value = profile(); value["workflow_dispatch"][key] = True
            with self.subTest(key=key), self.assertRaises(source.SourceIdentityError):
                source.verify_profile(value)

    def test_observed_commit_command_drift_rejected(self):
        value = profile(); value["observation"]["observed_commit_command"] = "git log -1"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_observed_tree_command_drift_rejected(self):
        value = profile(); value["observation"]["observed_tree_command"] = "git rev-parse HEAD"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_observation_retention_downgrade_rejected(self):
        value = profile(); value["observation"]["retain_observed_tree_sha"] = False
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_tree_diagnostic_only_downgrade_rejected(self):
        value = profile(); value["observation"]["tree_sha_is_diagnostic_only"] = False
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_bool_int_laundering_rejected(self):
        value = profile(); value["observation"]["retain_repository_ids"] = 1
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_head_merge_conflation_rejected(self):
        for key in ("head_qualification_is_merge_compatibility", "merge_compatibility_is_head_qualification"):
            value = profile(); value["separation"][key] = True
            with self.subTest(key=key), self.assertRaises(source.SourceIdentityError):
                source.verify_profile(value)

    def test_identity_minimality_downgrade_rejected(self):
        for key in (
            "exact_head_identity_excludes_merge_sha", "exact_head_identity_excludes_tree_sha",
            "exact_head_identity_excludes_workflow_definition", "merge_identity_excludes_head_sha",
            "merge_identity_excludes_tree_sha", "merge_identity_excludes_workflow_definition",
        ):
            value = profile(); value["separation"][key] = False
            with self.subTest(key=key), self.assertRaises(source.SourceIdentityError):
                source.verify_profile(value)

    def test_premature_profile_authority_rejected(self):
        value = profile(); value["authority"]["focused_theorem_passed"] = True
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_contract_defined_must_remain_true(self):
        value = profile(); value["authority"]["source_tree_identity_contract_defined"] = False
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_unknown_profile_field_rejected(self):
        value = profile(); value["authority_root"] = "x"
        with self.assertRaises(source.SourceIdentityError):
            source.verify_profile(value)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(source.SourceIdentityError):
                source.load_json(path)

    def test_cli_profile_validation(self):
        self.assertEqual(source.main(["--profile", str(PROFILE_PATH)]), 0)

    def test_cli_observation_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observation.json"
            path.write_text(json.dumps(observation()), encoding="utf-8")
            self.assertEqual(source.main(["--profile", str(PROFILE_PATH), "--observation", str(path)]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
