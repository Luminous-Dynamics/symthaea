#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("compile-hermetic-plan-v2b2.py")
QUALIFICATION_DIR = Path(__file__).resolve().parents[1] / "qualification"
POLICY = QUALIFICATION_DIR / "hermetic-executor-v2.json"
PROFILES = QUALIFICATION_DIR / "hermetic-profiles-v2.json"

spec = importlib.util.spec_from_file_location("hermetic_plan_v2b2", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load 003B2 plan compiler")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HermeticPlanB2Tests(unittest.TestCase):
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

        mujoco = self.paths["dependencies"] / "mujoco-3.8.0"
        (mujoco / "lib").mkdir(parents=True)
        target = mujoco / "lib" / "libmujoco.so.3.8.0"
        target.write_bytes(b"synthetic-mujoco-3.8.0-fixture\n")
        (mujoco / "lib" / "libmujoco.so").symlink_to("libmujoco.so.3.8.0")
        lib_sha = hashlib.sha256(target.read_bytes()).hexdigest()

        self.manifest_path = mujoco / "runtime-manifest.json"
        self.manifest = {
            "schema_id": module.MUJOCO_RUNTIME_SCHEMA,
            "platform": "x86_64-linux",
            "version": module.MUJOCO_VERSION,
            "mujoco_rs_compatibility": module.MUJOCO_RS_COMPATIBILITY,
            "upstream_release_commit": module.MUJOCO_RELEASE_COMMIT,
            "upstream_asset_id": 404891937,
            "upstream_asset_name": "mujoco-3.8.0-linux-x86_64.tar.gz",
            "upstream_asset_size": 20812715,
            "upstream_asset_sha256": "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab",
            "extracted_tree_digest_algorithm": module.EXTRACTED_TREE_DIGEST_ALGORITHM,
            "extracted_tree_sha256": "7" * 64,
            "libmujoco_sha256": lib_sha,
        }
        self._write_manifest()

        self.policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
        self.request = {
            "schema_id": module.REQUEST_SCHEMA,
            "qualification_id": module.MUJOCO_PROFILE_ID,
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

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def compile(self, request=None):
        return module.compile_plan(
            self.policy,
            self.profiles,
            self.request if request is None else request,
        )

    def test_exact_runtime_is_bound_and_explicit_link_dir_is_injected(self) -> None:
        plan = self.compile()
        runtime = plan["semantic"]["mujoco_runtime"]
        self.assertEqual(runtime["version"], "3.8.0")
        self.assertEqual(runtime["mujoco_rs_compatibility"], "4.0.1+mj-3.8.0")
        self.assertEqual(
            runtime["extracted_tree_digest_algorithm"],
            module.EXTRACTED_TREE_DIGEST_ALGORITHM,
        )
        self.assertEqual(runtime["upstream_asset_id"], 404891937)
        self.assertEqual(
            runtime["guest_dynamic_link_dir"], "/deps/mujoco-3.8.0/lib"
        )
        self.assertEqual(runtime["download_dir_policy"], "ABSENT")
        self.assertEqual(
            plan["semantic"]["environment"]["MUJOCO_DYNAMIC_LINK_DIR"],
            "/deps/mujoco-3.8.0/lib",
        )
        self.assertNotIn("MUJOCO_DOWNLOAD_DIR", plan["semantic"]["environment"])

    def test_bubblewrap_argv_sets_only_explicit_dynamic_link_dir(self) -> None:
        argv = self.compile()["execution"]["bubblewrap_invocations"][0]
        flattened = "\0".join(argv)
        self.assertIn("MUJOCO_DYNAMIC_LINK_DIR", flattened)
        self.assertIn("/deps/mujoco-3.8.0/lib", flattened)
        self.assertNotIn("MUJOCO_DOWNLOAD_DIR", flattened)

    def test_tampered_library_bytes_are_rejected(self) -> None:
        target = self.paths["dependencies"] / "mujoco-3.8.0" / "lib" / "libmujoco.so.3.8.0"
        target.write_bytes(b"tampered\n")
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_wrong_release_asset_identity_is_rejected(self) -> None:
        self.manifest["upstream_asset_id"] = 1
        self._write_manifest()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_wrong_wrapper_compatibility_is_rejected(self) -> None:
        self.manifest["mujoco_rs_compatibility"] = "4.0.1+mj-3.3.7"
        self._write_manifest()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_wrong_tree_digest_algorithm_is_rejected(self) -> None:
        self.manifest["extracted_tree_digest_algorithm"] = "other-tree-hash-v1"
        self._write_manifest()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_missing_manifest_is_rejected(self) -> None:
        self.manifest_path.unlink()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_missing_library_is_rejected(self) -> None:
        link = self.paths["dependencies"] / "mujoco-3.8.0" / "lib" / "libmujoco.so"
        link.unlink()
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_default_profile_is_not_admitted_by_b2(self) -> None:
        changed = copy.deepcopy(self.request)
        changed["qualification_id"] = "symthaea-humanoid-default-hermetic-v2"
        with self.assertRaises(module.PlanError):
            self.compile(request=changed)

    def test_manifest_identity_changes_semantic_plan_identity(self) -> None:
        first = self.compile()
        self.manifest["extracted_tree_sha256"] = "8" * 64
        self._write_manifest()
        second = self.compile()
        self.assertNotEqual(
            first["semantic_plan_sha256"], second["semantic_plan_sha256"]
        )

    def test_relative_confined_lib_symlink_is_admitted(self) -> None:
        plan = self.compile()
        self.assertEqual(
            plan["execution"]["mujoco_runtime"]["libmujoco_sha256"],
            self.manifest["libmujoco_sha256"],
        )

    def test_external_runtime_symlink_is_rejected(self) -> None:
        link = self.paths["dependencies"] / "mujoco-3.8.0" / "lib" / "libmujoco.so"
        link.unlink()
        outside = self.paths["evidence"] / "libmujoco.so"
        outside.write_bytes(b"outside\n")
        link.symlink_to(outside)
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_binding_revision_is_part_of_both_plan_identities(self) -> None:
        plan = self.compile()
        self.assertEqual(plan["mujoco_binding_revision"], module.BINDING_REVISION)
        self.assertEqual(
            plan["semantic"]["mujoco_binding_revision"], module.BINDING_REVISION
        )
        self.assertEqual(
            plan["execution"]["mujoco_binding_revision"], module.BINDING_REVISION
        )

    def test_b2_compiler_has_no_process_execution_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
