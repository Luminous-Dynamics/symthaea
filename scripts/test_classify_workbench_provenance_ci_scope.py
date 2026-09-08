#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import classify_workbench_provenance_ci_scope as scope

REAL_STACK = {
    624: [
        ".github/workflows/workbench-execution-capsule-profile.yml",
        "data/neuroscience/workbench_execution_capsule_profile_v1.json",
        "docs/neuroscience/WORKBENCH_EXECUTION_CAPSULE_PROFILE_V1.md",
        "scripts/test_verify_workbench_execution_capsule_profile.py",
        "scripts/verify_workbench_execution_capsule_profile.py",
    ],
    629: [
        ".github/workflows/workbench-nix-closure-identity.yml",
        "docs/neuroscience/WORKBENCH_NIX_CLOSURE_IDENTITY_V1.md",
        "scripts/test_workbench_nix_closure_identity.py",
        "scripts/workbench_nix_closure_identity.py",
    ],
    638: [
        ".github/workflows/workbench-nix-closure-capture.yml",
        "docs/neuroscience/WORKBENCH_NIX_CLOSURE_CAPTURE_V1.md",
        "scripts/test_workbench_nix_closure_capture.py",
        "scripts/workbench_nix_closure_capture.py",
    ],
    667: [
        ".github/workflows/workbench-nix-closure-capture-verifier.yml",
        "docs/neuroscience/WORKBENCH_NIX_CLOSURE_CAPTURE_VERIFIER_V1.md",
        "scripts/test_verify_workbench_nix_closure_capture.py",
        "scripts/verify_workbench_nix_closure_capture.py",
    ],
    681: [
        ".github/workflows/workbench-invocation-isolation-profile.yml",
        "data/neuroscience/workbench_invocation_isolation_profile_v1.json",
        "docs/neuroscience/WORKBENCH_INVOCATION_ISOLATION_PROFILE_V1.md",
        "scripts/check_workbench_invocation_isolation_profile_static.py",
        "scripts/test_verify_workbench_invocation_isolation_profile.py",
        "scripts/verify_workbench_invocation_isolation_profile.py",
    ],
    690: [
        ".github/workflows/workbench-root-nar-membership.yml",
        "docs/neuroscience/WORKBENCH_ROOT_NAR_MEMBERSHIP_V1.md",
        "scripts/test_workbench_root_nar_membership.py",
        "scripts/test_workbench_root_nar_target_spelling.py",
        "scripts/workbench_root_nar_membership.py",
    ],
}


class WorkbenchProvenanceScopeContracts(unittest.TestCase):
    def assert_full(self, paths):
        result = scope.classify_paths(paths)
        self.assertFalse(result["global_ci_may_skip"])
        self.assertEqual(result["status"], "full-ci-required")
        self.assertTrue(result["focused_workbench_qualification_still_required"])
        return result

    def assert_focused(self, paths):
        result = scope.classify_paths(paths)
        self.assertTrue(result["global_ci_may_skip"])
        self.assertEqual(result["status"], "workbench-provenance-only")
        self.assertEqual(result["disallowed_paths"], [])
        self.assertTrue(result["focused_workbench_qualification_still_required"])
        return result

    def test_each_real_stack_pr_is_focused_only(self):
        for pr, paths in REAL_STACK.items():
            with self.subTest(pr=pr):
                self.assert_focused(paths)

    def test_combined_real_stack_surface_is_focused_only(self):
        self.assert_focused([path for paths in REAL_STACK.values() for path in paths])

    def test_empty_diff_requires_full_ci(self):
        self.assert_full([])

    def test_cargo_manifest_forces_full_ci(self):
        result = self.assert_full(REAL_STACK[690] + ["Cargo.toml"])
        self.assertEqual(result["disallowed_paths"], ["Cargo.toml"])

    def test_rust_source_forces_full_ci(self):
        self.assert_full(REAL_STACK[681] + ["crates/core/symthaea-core/src/lib.rs"])

    def test_global_ci_change_forces_full_ci(self):
        self.assert_full([".github/workflows/ci.yml"])

    def test_non_workbench_neuroscience_script_forces_full_ci(self):
        self.assert_full(["scripts/derive_hcpmmp1_neuromaps_lineage_b.py"])

    def test_broad_neuroscience_document_is_not_implicitly_exempt(self):
        self.assert_full(["docs/neuroscience/NEURAL_BENCHMARK_QUALIFICATION_V1.md"])

    def test_workbench_shell_script_is_not_exempt(self):
        self.assert_full(["scripts/workbench_capture.sh"])

    def test_workbench_workflow_yaml_extension_is_not_exempt(self):
        self.assert_full([".github/workflows/workbench-example.yaml"])

    def test_workbench_workflow_nested_path_is_not_exempt(self):
        self.assert_full([".github/workflows/archive/workbench-example.yml"])

    def test_lowercase_document_is_not_exempt(self):
        self.assert_full(["docs/neuroscience/workbench_example.md"])

    def test_uppercase_data_file_is_not_exempt(self):
        self.assert_full(["data/neuroscience/WORKBENCH_EXAMPLE.json"])

    def test_unrecognized_script_prefix_is_not_exempt(self):
        self.assert_full(["scripts/run_workbench_experiment.py"])

    def test_python_bytecode_is_not_exempt(self):
        self.assert_full(["scripts/workbench_example.pyc"])

    def test_similarly_named_top_level_directory_is_not_exempt(self):
        self.assert_full(["workbench/example.py"])

    def test_path_sorting_is_deterministic(self):
        a = scope.classify_paths(list(reversed(REAL_STACK[629])))
        b = scope.classify_paths(REAL_STACK[629])
        self.assertEqual(a, b)

    def test_duplicate_path_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths([REAL_STACK[629][0], REAL_STACK[629][0]])

    def test_absolute_path_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths(["/scripts/workbench_x.py"])

    def test_parent_traversal_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths(["scripts/../scripts/workbench_x.py"])

    def test_duplicate_separator_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths(["scripts//workbench_x.py"])

    def test_backslash_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths([r"scripts\workbench_x.py"])

    def test_control_character_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths(["scripts/workbench_x.py\nCargo.toml"])

    def test_non_string_path_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths([1])

    def test_non_list_input_rejected(self):
        with self.assertRaises(scope.ScopeError):
            scope.classify_paths({"paths": REAL_STACK[624]})

    def test_cli_focused_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paths.json"
            path.write_text(json.dumps(REAL_STACK[690]), encoding="utf-8")
            self.assertEqual(scope.main(["--paths-json", str(path)]), 0)

    def test_cli_full_ci_returns_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paths.json"
            path.write_text(json.dumps(["Cargo.lock"]), encoding="utf-8")
            self.assertEqual(scope.main(["--paths-json", str(path)]), 3)

    def test_cli_malformed_input_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paths.json"
            path.write_text("{bad", encoding="utf-8")
            self.assertEqual(scope.main(["--paths-json", str(path)]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
