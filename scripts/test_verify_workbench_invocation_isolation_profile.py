#!/usr/bin/env python3
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import verify_workbench_invocation_isolation_profile as ver

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "data/neuroscience/workbench_invocation_isolation_profile_v1.json"
PARENT = ROOT / "data/neuroscience/workbench_execution_capsule_profile_v1.json"


def current():
    return ver.load(PROFILE), ver.load(PARENT)


class InvocationIsolationProfileContracts(unittest.TestCase):
    def assert_rejects(self, mutate_profile=None, mutate_parent=None):
        profile, parent = current()
        profile = copy.deepcopy(profile)
        parent = copy.deepcopy(parent)
        if mutate_profile:
            mutate_profile(profile)
        if mutate_parent:
            mutate_parent(parent)
        with self.assertRaises(ver.ContractError):
            ver.verify_profile(profile, parent)

    def test_current_profile_validates(self):
        profile, parent = current()
        self.assertEqual(ver.verify_profile(profile, parent), profile)

    def test_unknown_top_level_field_rejected(self):
        self.assert_rejects(lambda p: p.__setitem__("qualified", True))

    def test_platform_drift_rejected(self):
        self.assert_rejects(lambda p: p.__setitem__("qualification_platform", "aarch64-linux"))

    def test_parent_schema_drift_rejected(self):
        self.assert_rejects(mutate_parent=lambda p: p.__setitem__("schema", "other"))

    def test_parent_environment_drift_rejected(self):
        self.assert_rejects(mutate_parent=lambda p: p["execution_environment"].__setitem__("omp_num_threads", "2"))

    def test_parent_main_program_drift_rejected(self):
        self.assert_rejects(mutate_parent=lambda p: p["nixpkgs_package"].__setitem__("main_program", "other"))

    def test_host_path_lookup_rejected(self):
        self.assert_rejects(lambda p: p["program_binding"].__setitem__("host_path_lookup_allowed", True))

    def test_nonabsolute_program_policy_rejected(self):
        self.assert_rejects(lambda p: p["program_binding"].__setitem__("require_absolute_exec_path", False))

    def test_program_root_substitution_rejected(self):
        self.assert_rejects(lambda p: p["program_binding"].__setitem__("root_source", "caller-path"))

    def test_program_boolean_integer_laundering_rejected(self):
        self.assert_rejects(lambda p: p["program_binding"].__setitem__("require_absolute_exec_path", 1))

    def test_environment_stage_drift_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"].__setitem__("stage", "final-workbench-environment"))

    def test_host_environment_inheritance_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"].__setitem__("inherit_host_environment", True))

    def test_fixed_environment_drift_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["fixed"].__setitem__("OMP_NUM_THREADS", "2"))

    def test_host_path_environment_reintroduced_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["fixed"].__setitem__("PATH", "/usr/bin"))

    def test_dynamic_binding_missing_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["dynamic_bindings"].pop("XDG_RUNTIME_DIR"))

    def test_dynamic_binding_retarget_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["dynamic_bindings"].__setitem__("HOME", "caller.home"))

    def test_pwd_binding_must_match_fresh_cwd(self):
        self.assert_rejects(lambda p: p["process_environment"]["dynamic_bindings"].__setitem__("PWD", "caller.cwd"))

    def test_tmp_alias_must_match_tmpdir_root(self):
        self.assert_rejects(lambda p: p["process_environment"]["dynamic_bindings"].__setitem__("TMP", "invocation.other_tmp"))

    def test_temp_alias_cannot_disappear(self):
        self.assert_rejects(lambda p: p["process_environment"]["dynamic_bindings"].pop("TEMP"))

    def test_entry_environment_cannot_be_claimed_final(self):
        self.assert_rejects(lambda p: p["process_environment"]["descendant_environment"].__setitem__("entry_environment_is_final_environment", True))

    def test_verified_program_environment_transition_cannot_be_erased(self):
        self.assert_rejects(lambda p: p["process_environment"]["descendant_environment"].__setitem__("verified_program_may_transform_environment", False))

    def test_post_execve_host_injection_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["descendant_environment"].__setitem__("host_injection_after_execve_allowed", True))

    def test_post_entry_environment_equivalence_shortcut_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["descendant_environment"].__setitem__("post_entry_environment_equivalence_assumed", True))

    def test_descendant_environment_bool_integer_laundering_rejected(self):
        self.assert_rejects(lambda p: p["process_environment"]["descendant_environment"].__setitem__("entry_environment_is_final_environment", 0))

    def test_shared_cwd_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("cwd_policy", "pipeline-shared"))

    def test_xdg_runtime_root_must_be_per_invocation(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("xdg_runtime_dir_policy", "host-default"))

    def test_dynamic_environment_must_match_created_roots(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("dynamic_environment_must_match_created_roots", False))

    def test_dynamic_environment_root_boolean_integer_laundering_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("dynamic_environment_must_match_created_roots", 1))

    def test_dynamic_environment_paths_must_be_absolute(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("dynamic_environment_paths_must_be_absolute", False))

    def test_scratch_reuse_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("reuse_allowed", True))

    def test_reuse_integer_laundering_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("reuse_allowed", 0))

    def test_inherited_file_descriptor_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("pass_fds", [3]))

    def test_uncaptured_stdout_rejected(self):
        self.assert_rejects(lambda p: p["per_invocation_isolation"].__setitem__("stdout_policy", "inherit"))

    def test_path_bytewise_equivalence_promotion_rejected(self):
        self.assert_rejects(lambda p: p["scientific_path_gate"].__setitem__("bytewise_equivalence_assumed", True))

    def test_path_semantic_equivalence_promotion_rejected(self):
        self.assert_rejects(lambda p: p["scientific_path_gate"].__setitem__("semantic_equivalence_assumed", True))

    def test_missing_path_perturbation_axis_rejected(self):
        self.assert_rejects(lambda p: p["scientific_path_gate"].__setitem__("required_perturbation_axes", ["snapshot_root"]))

    def test_x86_equivalence_shortcut_rejected(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("same_x86_64_implies_equivalence", True))

    def test_closure_cross_cpu_shortcut_rejected(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("same_closure_implies_cross_cpu_equivalence", True))

    def test_runtime_dispatch_fact_cannot_disappear(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("runtime_cpu_dispatch_known", False))

    def test_dispatch_mode_drift_rejected(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("known_dispatch_modes", ["AVX"]))

    def test_cross_cpu_gate_cannot_be_disabled(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("cross_cpu_equivalence_required_before_transfer", False))

    def test_diagnostic_context_drift_rejected(self):
        self.assert_rejects(lambda p: p["numerical_execution_gate"].__setitem__("diagnostic_context", ["cpu_model"]))

    def test_authority_escalation_rejected(self):
        self.assert_rejects(lambda p: p["authority"].__setitem__("workbench_execution_qualified", True))

    def test_authority_integer_laundering_rejected(self):
        self.assert_rejects(lambda p: p["authority"].__setitem__("invocation_executed", 0))

    def test_duplicate_json_key_rejected(self):
        raw = PROFILE.read_text(encoding="utf-8")
        raw = raw.replace('"schema":', '"schema":"forged","schema":', 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ver.ContractError):
                ver.load(path)

    def test_cli_round_trip(self):
        self.assertEqual(ver.main(["--profile", str(PROFILE), "--parent-profile", str(PARENT)]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
