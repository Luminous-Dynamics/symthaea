#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import verify_focused_ci_scope_registry as verify

REGISTRY_PATH = Path("data/ci/focused_ci_scope_registry_v1.json")
EXPECTED_PILOT_BLOBS = {
    "workbench-execution-capsule-profile-v1": "a34f09f06b1dc5e4938bc70fd8de0a2a7dc2cea1",
    "workbench-invocation-isolation-profile-v1": "6007cbc410e7e9046acf9a172ec2ca262dbd0f06",
    "workbench-nix-closure-capture-v1": "89be74793b9a133c8d460807a417910524e8309f",
    "workbench-nix-closure-capture-verifier-v1": "db11a58d4a76b6b7b24bee624cf0a07926904ba3",
    "workbench-nix-closure-identity-v1": "b69b87183abc54b6b4e124cc7c9b09b3d98b043b",
    "workbench-root-nar-membership-v1": "49d0f64d4118eba05185543d7d214ffab2003b0a",
}


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def non_workflow_files(item):
    return [path for path in item["files"] if path != item["focused_workflow"]]


def write_workflow(root: Path, item: dict, data: bytes) -> Path:
    target = root.joinpath(*item["focused_workflow"].split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


@contextmanager
def synthetic_head(registry, *, promote=True):
    candidate = copy.deepcopy(registry)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for item in candidate["scopes"]:
            if promote:
                item["qualification_source_identity_qualified"] = True
                item["executing_workflow_identity_qualified"] = True
                item["global_ci_eligible"] = True
            data = f"qualified-workflow:{item['id']}\n".encode("utf-8")
            item["focused_workflow_git_blob"] = verify.git_blob_sha(data)
            write_workflow(root, item, data)
        yield candidate, root


class FocusedCiScopeRegistryContracts(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()

    def test_registry_validates(self):
        value = verify.validate_registry(self.registry)
        self.assertEqual(value["schema"], verify.SCHEMA)
        self.assertEqual(len(value["scopes"]), 6)
        self.assertFalse(value["scientific_authority"])

    def test_current_pilot_withholds_all_admission_qualification(self):
        for item in self.registry["scopes"]:
            with self.subTest(scope=item["id"]):
                self.assertFalse(item["global_ci_eligible"])
                self.assertFalse(item["qualification_source_identity_qualified"])
                self.assertFalse(item["executing_workflow_identity_qualified"])
                self.assertEqual(item["qualification_source_mode"], "exact-pr-head")
                self.assertTrue(item["merge_compatibility_separate"])

    def test_exact_pilot_qualifier_blobs_are_bound(self):
        actual = {
            item["id"]: item["focused_workflow_git_blob"]
            for item in self.registry["scopes"]
        }
        self.assertEqual(actual, EXPECTED_PILOT_BLOBS)

    def test_git_blob_hash_matches_git_object_semantics(self):
        self.assertEqual(
            verify.git_blob_sha(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_current_owned_diff_requires_full_ci_even_with_exact_head_judge(self):
        with synthetic_head(self.registry, promote=False) as (candidate, head):
            item = candidate["scopes"][0]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("scope-not-global-ci-eligible", result["reasons"])
            self.assertIn("source-identity-not-qualified", result["reasons"])
            self.assertIn("executing-workflow-identity-not-qualified", result["reasons"])
            self.assertTrue(result["head_qualifier_blobs_verified"])

    def test_promoted_scope_can_take_focused_route_with_exact_head_judge(self):
        with synthetic_head(self.registry) as (candidate, head):
            for item in candidate["scopes"]:
                with self.subTest(scope=item["id"]):
                    result = verify.admit_changed_paths(
                        candidate,
                        non_workflow_files(item),
                        repository_root=head,
                    )
                    self.assertFalse(result["global_ci_required"])
                    self.assertTrue(result["global_ci_may_skip"])
                    self.assertTrue(result["head_qualifier_blobs_verified"])
                    self.assertIn(item["focused_workflow"], result["required_workflows"])
                    self.assertIn(
                        candidate["meta_qualifier_workflow"],
                        result["required_workflows"],
                    )
                    self.assertFalse(result["scientific_authority"])

    def test_eligible_without_source_identity_is_invalid(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["executing_workflow_identity_qualified"] = True
        candidate["scopes"][0]["global_ci_eligible"] = True
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_eligible_without_executing_workflow_identity_is_invalid(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["qualification_source_identity_qualified"] = True
        candidate["scopes"][0]["global_ci_eligible"] = True
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_promoted_metadata_requires_both_capabilities(self):
        candidate = copy.deepcopy(self.registry)
        item = candidate["scopes"][0]
        item["qualification_source_identity_qualified"] = True
        item["executing_workflow_identity_qualified"] = True
        item["global_ci_eligible"] = True
        self.assertTrue(
            verify.validate_registry(candidate)["scopes"][0]["global_ci_eligible"]
        )

    def test_source_mode_drift_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["qualification_source_mode"] = "merge-ref"
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_merge_compatibility_conflation_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["merge_compatibility_separate"] = False
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_capability_bool_int_laundering_rejected(self):
        for key in (
            "qualification_source_identity_qualified",
            "executing_workflow_identity_qualified",
        ):
            candidate = copy.deepcopy(self.registry)
            candidate["scopes"][0][key] = 0
            with self.subTest(key=key), self.assertRaises(verify.RegistryError):
                verify.validate_registry(candidate)

    def test_missing_head_observation_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, _head):
            item = candidate["scopes"][0]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("head-qualifier-blob-unverified", result["reasons"])

    def test_head_qualifier_blob_drift_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            item = candidate["scopes"][0]
            write_workflow(head, item, b"drifted\n")
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("head-qualifier-blob-drift", result["reasons"])

    def test_missing_head_qualifier_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            item = candidate["scopes"][0]
            head.joinpath(*item["focused_workflow"].split("/")).unlink()
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("head-qualifier-blob-unavailable", result["reasons"])

    def test_symlinked_head_qualifier_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            item = candidate["scopes"][0]
            target = head.joinpath(*item["focused_workflow"].split("/"))
            outside = head / "outside.yml"
            outside.write_text("qualified-workflow\n", encoding="utf-8")
            target.unlink()
            target.symlink_to(outside)
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("head-qualifier-blob-unavailable", result["reasons"])

    def test_focused_workflow_change_requires_full_ci_even_when_blob_matches(self):
        with synthetic_head(self.registry) as (candidate, head):
            item = candidate["scopes"][0]
            result = verify.admit_changed_paths(
                candidate,
                [item["focused_workflow"]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("focused-qualifier-change", result["reasons"])
            self.assertTrue(result["head_qualifier_blobs_verified"])

    def test_two_promoted_scopes_in_same_group_compose(self):
        with synthetic_head(self.registry) as (candidate, head):
            a, b = candidate["scopes"][:2]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(a)[0], non_workflow_files(b)[0]],
                repository_root=head,
            )
            self.assertFalse(result["global_ci_required"])
            self.assertEqual(result["composition_groups"], ["workbench-provenance-v1"])
            self.assertEqual(len(result["touched_scopes"]), 2)

    def test_unknown_path_requires_full_ci(self):
        result = verify.admit_changed_paths(self.registry, ["README.md"])
        self.assertTrue(result["global_ci_required"])
        self.assertIn("unowned-path", result["reasons"])

    def test_mixed_owned_and_cargo_change_requires_full_ci(self):
        item = self.registry["scopes"][0]
        result = verify.admit_changed_paths(
            self.registry,
            [non_workflow_files(item)[0], "Cargo.toml"],
        )
        self.assertTrue(result["global_ci_required"])
        self.assertIn("control-plane-change", result["reasons"])

    def test_empty_diff_requires_full_ci(self):
        result = verify.admit_changed_paths(self.registry, [])
        self.assertTrue(result["global_ci_required"])
        self.assertIn("empty-diff", result["reasons"])

    def test_control_plane_requires_full_ci(self):
        for path in (
            ".github/workflows/ci.yml",
            ".github/workflows/ci-source-tree-identity.yml",
            "data/ci/ci_source_tree_identity_profile_v1.json",
            "docs/ci/CI_SOURCE_TREE_IDENTITY_V1.md",
            "docs/ci/CI_WORKFLOW_DEFINITION_IDENTITY_V1.md",
            "scripts/verify_ci_source_tree_identity.py",
            "scripts/verify_ci_workflow_definition_identity.py",
            "scripts/test_verify_ci_source_tree_identity.py",
            "scripts/test_verify_ci_workflow_definition_identity.py",
            "Cargo.toml",
            "Cargo.lock",
            "flake.nix",
            "flake.lock",
            "rust-toolchain.toml",
            "scripts/check-class-a-changes.sh",
            "scripts/collect_ci_changed_paths.py",
            "scripts/test_collect_ci_changed_paths.py",
            "data/ci/focused_ci_scope_registry_v1.json",
            self.registry["meta_qualifier_workflow"],
            ".github/actions/example/action.yml",
        ):
            with self.subTest(path=path):
                result = verify.admit_changed_paths(self.registry, [path])
                self.assertTrue(result["global_ci_required"])
                self.assertIn("control-plane-change", result["reasons"])

    def test_changed_paths_are_sorted(self):
        item = self.registry["scopes"][0]
        paths = list(reversed(non_workflow_files(item)))
        result = verify.admit_changed_paths(self.registry, paths)
        self.assertEqual(result["changed_paths"], sorted(paths))

    def test_duplicate_changed_path_rejected(self):
        path = non_workflow_files(self.registry["scopes"][0])[0]
        with self.assertRaises(verify.RegistryError):
            verify.admit_changed_paths(self.registry, [path, path])

    def test_noncanonical_changed_path_rejected(self):
        for bad in (
            "scripts/../scripts/workbench_x.py",
            "/scripts/workbench_x.py",
            "scripts//workbench_x.py",
            "scripts\\workbench_x.py",
        ):
            with self.subTest(path=bad), self.assertRaises(verify.RegistryError):
                verify.admit_changed_paths(self.registry, [bad])

    def test_nonlist_changed_paths_rejected(self):
        with self.assertRaises(verify.RegistryError):
            verify.admit_changed_paths(self.registry, {"path": "README.md"})

    def test_unknown_top_level_field_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["authority"] = True
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_unknown_scope_field_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["authority"] = True
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_scientific_authority_escalation_rejected(self):
        for where in ("registry", "scope"):
            candidate = copy.deepcopy(self.registry)
            if where == "registry":
                candidate["scientific_authority"] = True
            else:
                candidate["scopes"][0]["scientific_authority"] = True
            with self.subTest(where=where), self.assertRaises(verify.RegistryError):
                verify.validate_registry(candidate)

    def test_composable_bool_int_laundering_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["composable"] = 1
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_bad_qualifier_blob_spelling_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["focused_workflow_git_blob"] = "A" * 40
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_duplicate_scope_id_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][1]["id"] = candidate["scopes"][0]["id"]
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_ambiguous_file_owner_rejected(self):
        candidate = copy.deepcopy(self.registry)
        path = candidate["scopes"][0]["files"][1]
        candidate["scopes"][1]["files"].append(path)
        candidate["scopes"][1]["files"].sort()
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_scope_cannot_own_control_plane(self):
        for protected in (
            ".github/workflows/ci.yml",
            "Cargo.toml",
            "flake.lock",
            "scripts/verify_ci_source_tree_identity.py",
        ):
            candidate = copy.deepcopy(self.registry)
            candidate["scopes"][0]["files"].append(protected)
            candidate["scopes"][0]["files"].sort()
            with self.subTest(path=protected), self.assertRaises(verify.RegistryError):
                verify.validate_registry(candidate)

    def test_focused_workflow_must_be_owned(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["files"].remove(
            candidate["scopes"][0]["focused_workflow"]
        )
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_focused_workflow_cannot_be_meta_qualifier(self):
        candidate = copy.deepcopy(self.registry)
        item = candidate["scopes"][0]
        item["focused_workflow"] = candidate["meta_qualifier_workflow"]
        item["files"].append(candidate["meta_qualifier_workflow"])
        item["files"].sort()
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_unsorted_scope_files_rejected(self):
        candidate = copy.deepcopy(self.registry)
        candidate["scopes"][0]["files"] = list(
            reversed(candidate["scopes"][0]["files"])
        )
        with self.assertRaises(verify.RegistryError):
            verify.validate_registry(candidate)

    def test_mixed_composition_groups_require_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            candidate["scopes"][1]["composition_group"] = "other-domain-v1"
            a, b = candidate["scopes"][:2]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(a)[0], non_workflow_files(b)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("mixed-composition-groups", result["reasons"])

    def test_noncomposable_scope_mix_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            candidate["scopes"][1]["composable"] = False
            a, b = candidate["scopes"][:2]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(a)[0], non_workflow_files(b)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("noncomposable-scope-mix", result["reasons"])

    def test_qualified_but_ineligible_scope_requires_full_ci(self):
        with synthetic_head(self.registry) as (candidate, head):
            candidate["scopes"][0]["global_ci_eligible"] = False
            item = candidate["scopes"][0]
            result = verify.admit_changed_paths(
                candidate,
                [non_workflow_files(item)[0]],
                repository_root=head,
            )
            self.assertTrue(result["global_ci_required"])
            self.assertIn("scope-not-global-ci-eligible", result["reasons"])

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(verify.RegistryError):
                verify.load_json(path)

    def test_cli_registry_validation_returns_zero(self):
        self.assertEqual(
            verify.main(["--registry", str(REGISTRY_PATH)]),
            0,
        )

    def test_cli_promoted_focused_admission_returns_zero(self):
        with synthetic_head(self.registry) as (candidate, head), tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp) / "registry.json"
            changed = Path(tmp) / "changed.json"
            reg.write_text(json.dumps(candidate), encoding="utf-8")
            item = candidate["scopes"][0]
            changed.write_text(
                json.dumps([non_workflow_files(item)[0]]),
                encoding="utf-8",
            )
            self.assertEqual(
                verify.main(
                    [
                        "--registry",
                        str(reg),
                        "--changed-paths-json",
                        str(changed),
                        "--repository-root",
                        str(head),
                    ]
                ),
                0,
            )

    def test_cli_missing_head_root_returns_three(self):
        with synthetic_head(self.registry) as (candidate, _head), tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp) / "registry.json"
            changed = Path(tmp) / "changed.json"
            reg.write_text(json.dumps(candidate), encoding="utf-8")
            item = candidate["scopes"][0]
            changed.write_text(
                json.dumps([non_workflow_files(item)[0]]),
                encoding="utf-8",
            )
            self.assertEqual(
                verify.main(
                    [
                        "--registry",
                        str(reg),
                        "--changed-paths-json",
                        str(changed),
                    ]
                ),
                3,
            )

    def test_cli_full_ci_admission_returns_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.json"
            changed.write_text(json.dumps(["Cargo.lock"]), encoding="utf-8")
            self.assertEqual(
                verify.main(
                    [
                        "--registry",
                        str(REGISTRY_PATH),
                        "--changed-paths-json",
                        str(changed),
                    ]
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
