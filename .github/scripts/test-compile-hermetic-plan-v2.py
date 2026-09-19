#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("compile-hermetic-plan-v2.py")
QUALIFICATION_DIR = Path(__file__).resolve().parents[1] / "qualification"
POLICY = QUALIFICATION_DIR / "hermetic-executor-v2.json"
PROFILES = QUALIFICATION_DIR / "hermetic-profiles-v2.json"

spec = importlib.util.spec_from_file_location("hermetic_plan_v2", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load hermetic plan compiler")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HermeticPlanCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths: dict[str, Path] = {}
        for name in (
            "source",
            "runtime",
            "toolchain",
            "dependencies",
            "work",
            "home",
            "tmp",
            "control-plane",
            "evidence",
        ):
            path = root / name
            path.mkdir()
            self.paths[name] = path
        (self.paths["dependencies"] / "vendor").mkdir()
        (self.paths["dependencies"] / "cargo-config.toml").write_text(
            "[source.crates-io]\nreplace-with = 'vendored-sources'\n\n"
            "[source.vendored-sources]\ndirectory = '/deps/vendor'\n",
            encoding="utf-8",
        )

        self.policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
        self.request = {
            "schema_id": module.REQUEST_SCHEMA,
            "qualification_id": "symthaea-humanoid-default-hermetic-v2",
            "source": {
                "host_path": str(self.paths["source"]),
                "subject_sha": "1" * 40,
                "tree_sha": "2" * 40,
            },
            "runtime_rootfs": {
                "host_path": str(self.paths["runtime"]),
                "sha256": "3" * 64,
            },
            "toolchain": {
                "host_path": str(self.paths["toolchain"]),
                "sha256": "4" * 64,
            },
            "dependencies": {
                "host_path": str(self.paths["dependencies"]),
                "manifest_sha256": "5" * 64,
            },
            "writable": {
                "work": str(self.paths["work"]),
                "home": str(self.paths["home"]),
                "tmp": str(self.paths["tmp"]),
            },
            "trusted": {
                "control_plane": str(self.paths["control-plane"]),
                "evidence": str(self.paths["evidence"]),
            },
            "bubblewrap": {
                "host_path": "/usr/bin/bwrap",
                "sha256": "6" * 64,
            },
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def compile(self, request=None, profiles=None, policy=None):
        return module.compile_plan(
            self.policy if policy is None else policy,
            self.profiles if profiles is None else profiles,
            self.request if request is None else request,
        )

    def test_semantic_plan_is_stable_under_json_key_reordering(self) -> None:
        first = self.compile()
        reordered = json.loads(json.dumps(self.request, sort_keys=True))
        second = self.compile(request=reordered)
        self.assertEqual(first["semantic_plan_sha256"], second["semantic_plan_sha256"])
        self.assertEqual(first["execution_plan_sha256"], second["execution_plan_sha256"])

    def test_ephemeral_host_paths_do_not_change_semantic_identity(self) -> None:
        first = self.compile()
        root = Path(self.temp.name)
        alt = {}
        for name in ("source2", "runtime2", "toolchain2", "dependencies2", "work2", "home2", "tmp2", "control2", "evidence2"):
            path = root / name
            path.mkdir()
            alt[name] = path
        (alt["dependencies2"] / "vendor").mkdir()
        (alt["dependencies2"] / "cargo-config.toml").write_text("# trusted fixture\n", encoding="utf-8")
        changed = copy.deepcopy(self.request)
        changed["source"]["host_path"] = str(alt["source2"])
        changed["runtime_rootfs"]["host_path"] = str(alt["runtime2"])
        changed["toolchain"]["host_path"] = str(alt["toolchain2"])
        changed["dependencies"]["host_path"] = str(alt["dependencies2"])
        changed["writable"] = {
            "work": str(alt["work2"]),
            "home": str(alt["home2"]),
            "tmp": str(alt["tmp2"]),
        }
        changed["trusted"] = {
            "control_plane": str(alt["control2"]),
            "evidence": str(alt["evidence2"]),
        }
        second = self.compile(request=changed)
        self.assertEqual(first["semantic_plan_sha256"], second["semantic_plan_sha256"])
        self.assertNotEqual(first["execution_plan_sha256"], second["execution_plan_sha256"])

    def test_input_identity_change_changes_semantic_digest(self) -> None:
        first = self.compile()
        changed = copy.deepcopy(self.request)
        changed["runtime_rootfs"]["sha256"] = "7" * 64
        second = self.compile(request=changed)
        self.assertNotEqual(first["semantic_plan_sha256"], second["semantic_plan_sha256"])

    def test_plan_uses_explicit_hard_namespace_flags(self) -> None:
        prefix = self.compile()["execution"]["bubblewrap_invocations"][0]
        for flag in (
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-cgroup",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ):
            self.assertIn(flag, prefix)
        self.assertNotIn("--unshare-all", prefix)
        self.assertNotIn("--share-net", prefix)

    def test_runtime_is_read_only_sandbox_root(self) -> None:
        mounts = self.compile()["execution"]["mounts"]
        self.assertEqual(mounts[0]["mode"], "ro-bind")
        self.assertEqual(mounts[0]["guest"], "/")
        self.assertEqual(mounts[0]["host"], str(self.paths["runtime"].resolve()))

    def test_cargo_home_is_writable_scratch_but_vendor_is_read_only(self) -> None:
        plan = self.compile()
        self.assertEqual(plan["semantic"]["environment"]["CARGO_HOME"], "/work/cargo-home")
        command = plan["semantic"]["commands"][0]
        self.assertEqual(command[:4], [
            "/toolchain/bin/cargo",
            "--offline",
            "--config",
            "/deps/cargo-config.toml",
        ])
        deps = next(m for m in plan["semantic"]["mounts"] if m["guest"] == "/deps")
        self.assertEqual(deps["mode"], "ro-bind")

    def test_missing_vendor_layout_is_rejected(self) -> None:
        (self.paths["dependencies"] / "cargo-config.toml").unlink()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_parent_traversal_path_is_rejected(self) -> None:
        changed = copy.deepcopy(self.request)
        changed["source"]["host_path"] = str(self.paths["source"] / ".." / "evidence")
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_overlapping_trusted_and_subject_roots_are_rejected(self) -> None:
        changed = copy.deepcopy(self.request)
        changed["trusted"]["control_plane"] = str(self.paths["source"])
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_symlink_mount_root_is_rejected(self) -> None:
        link = Path(self.temp.name) / "source-link"
        link.symlink_to(self.paths["source"], target_is_directory=True)
        changed = copy.deepcopy(self.request)
        changed["source"]["host_path"] = str(link)
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_network_enabled_profile_is_rejected(self) -> None:
        changed = copy.deepcopy(self.profiles)
        changed["profiles"][self.request["qualification_id"]]["network"] = "HOST"
        with self.assertRaises(module.PlanError):
            self.compile(profiles=changed)

    def test_environment_injection_is_rejected(self) -> None:
        changed = copy.deepcopy(self.profiles)
        changed["profiles"][self.request["qualification_id"]]["environment"]["GITHUB_TOKEN"] = "forbidden"
        with self.assertRaises(module.PlanError):
            self.compile(profiles=changed)

    def test_arbitrary_command_executable_is_rejected(self) -> None:
        changed = copy.deepcopy(self.profiles)
        changed["profiles"][self.request["qualification_id"]]["commands"][0][0] = "/bin/sh"
        with self.assertRaises(module.PlanError):
            self.compile(profiles=changed)

    def test_removing_offline_vendor_config_is_rejected(self) -> None:
        changed = copy.deepcopy(self.profiles)
        changed["profiles"][self.request["qualification_id"]]["commands"][0] = [
            "/toolchain/bin/cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"
        ]
        with self.assertRaises(module.PlanError):
            self.compile(profiles=changed)

    def test_bubblewrap_path_substitution_is_rejected_without_probing_backend(self) -> None:
        changed = copy.deepcopy(self.request)
        changed["bubblewrap"]["host_path"] = "/tmp/fake-bwrap"
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_unknown_request_policy_and_registry_fields_are_rejected(self) -> None:
        changed_request = copy.deepcopy(self.request)
        changed_request["extra"] = "not admitted"
        with self.assertRaises(module.PlanError):
            self.compile(request=changed_request)
        changed_policy = copy.deepcopy(self.policy)
        changed_policy["extra"] = "not admitted"
        with self.assertRaises(module.PlanError):
            self.compile(policy=changed_policy)
        changed_profiles = copy.deepcopy(self.profiles)
        changed_profiles["extra"] = "not admitted"
        with self.assertRaises(module.PlanError):
            self.compile(profiles=changed_profiles)

    def test_global_apparmor_relaxation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.policy)
        changed["backend"]["apparmor"]["global_unprivileged_userns_restriction_disable_allowed"] = True
        with self.assertRaises(module.PlanError):
            self.compile(policy=changed)

    def test_zero_digest_is_rejected(self) -> None:
        changed = copy.deepcopy(self.request)
        changed["bubblewrap"]["sha256"] = "0" * 64
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_production_compiler_has_no_process_execution_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("shell=True", source)

    def test_nonclaims_remain_explicit(self) -> None:
        plan = self.compile()
        self.assertEqual(plan["backend_execution_claim"], "NONE")
        self.assertEqual(plan["runtime_authority"], "NONE")
        self.assertIn("plan_compilation_is_not_namespace_enforcement", plan["nonclaims"])


if __name__ == "__main__":
    unittest.main()
