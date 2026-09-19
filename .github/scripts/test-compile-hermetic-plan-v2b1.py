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

SCRIPT = Path(__file__).with_name("compile-hermetic-plan-v2b1.py")
QUALIFICATION_DIR = Path(__file__).resolve().parents[1] / "qualification"
POLICY = QUALIFICATION_DIR / "hermetic-executor-v2.json"
PROFILES = QUALIFICATION_DIR / "hermetic-profiles-v2.json"

spec = importlib.util.spec_from_file_location("hermetic_plan_v2b1", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load 003B1 plan compiler")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HermeticPlanB1Tests(unittest.TestCase):
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

    def compile(self, request=None):
        return module.compile_plan(
            self.policy,
            self.profiles,
            self.request if request is None else request,
        )

    def test_nested_user_namespaces_are_hard_disabled(self) -> None:
        plan = self.compile()
        flags = plan["semantic"]["namespace_flags"]
        self.assertIn("--unshare-user", flags)
        self.assertIn("--disable-userns", flags)
        self.assertEqual(flags.count("--disable-userns"), 1)
        self.assertLess(flags.index("--unshare-user"), flags.index("--disable-userns"))
        self.assertNotIn("--unshare-all", flags)
        self.assertNotIn("--share-net", flags)

    def test_dev_is_tmpfs_plus_exact_closed_device_allowlist(self) -> None:
        plan = self.compile()
        mounts = plan["semantic"]["mounts"]
        self.assertNotIn({"mode": "dev", "guest": "/dev"}, mounts)
        self.assertIn({"mode": "tmpfs", "guest": "/dev"}, mounts)
        device_mounts = [m for m in mounts if m["mode"] == "dev-bind"]
        self.assertEqual(
            device_mounts,
            [
                {"mode": "dev-bind", "device": d, "guest": d}
                for d in module.DEVICE_ALLOWLIST
            ],
        )
        self.assertEqual(
            plan["semantic"]["device_allowlist"],
            list(module.DEVICE_ALLOWLIST),
        )

    def test_stdio_links_are_private_proc_links(self) -> None:
        mounts = self.compile()["semantic"]["mounts"]
        symlinks = [m for m in mounts if m["mode"] == "symlink"]
        self.assertEqual(
            symlinks,
            [
                {"mode": "symlink", "target": target, "guest": guest}
                for target, guest in module.STDIO_SYMLINKS
            ],
        )

    def test_invocation_contains_no_generic_dev_mount(self) -> None:
        argv = self.compile()["execution"]["bubblewrap_invocations"][0]
        self.assertNotIn("--dev", argv)
        self.assertIn("--tmpfs", argv)
        self.assertIn("--disable-userns", argv)
        for device in module.DEVICE_ALLOWLIST:
            self.assertIn(device, argv)

    def test_external_source_symlink_is_rejected(self) -> None:
        outside = self.paths["evidence"] / "secret"
        outside.write_text("secret\n", encoding="utf-8")
        (self.paths["source"] / "escape").symlink_to(outside)
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_internal_source_symlink_is_admitted(self) -> None:
        target = self.paths["source"] / "target"
        target.write_text("safe\n", encoding="utf-8")
        (self.paths["source"] / "inside").symlink_to("target")
        self.compile()

    def test_absolute_internal_source_symlink_is_rejected(self) -> None:
        target = self.paths["source"] / "target-absolute"
        target.write_text("safe bytes, unsafe link form\n", encoding="utf-8")
        (self.paths["source"] / "absolute").symlink_to(target)
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_broken_source_symlink_is_rejected(self) -> None:
        (self.paths["source"] / "broken").symlink_to(
            self.paths["source"] / "missing"
        )
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_external_dependency_symlink_is_rejected(self) -> None:
        outside = self.paths["evidence"] / "secret-dep"
        outside.write_text("secret\n", encoding="utf-8")
        (self.paths["dependencies"] / "vendor" / "escape").symlink_to(outside)
        with self.assertRaises(module.PlanError):
            self.compile()

    def test_ephemeral_host_paths_still_do_not_change_semantic_identity(self) -> None:
        first = self.compile()
        root = Path(self.temp.name)
        alt: dict[str, Path] = {}
        for name in (
            "source2",
            "runtime2",
            "toolchain2",
            "dependencies2",
            "work2",
            "home2",
            "tmp2",
            "control2",
            "evidence2",
        ):
            path = root / name
            path.mkdir()
            alt[name] = path
        (alt["dependencies2"] / "vendor").mkdir()
        (alt["dependencies2"] / "cargo-config.toml").write_text(
            "# trusted fixture\n", encoding="utf-8"
        )
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
        self.assertEqual(
            first["semantic_plan_sha256"], second["semantic_plan_sha256"]
        )
        self.assertNotEqual(
            first["execution_plan_sha256"], second["execution_plan_sha256"]
        )

    def test_hardening_revision_is_bound_into_both_identities(self) -> None:
        plan = self.compile()
        self.assertEqual(
            plan["compiler_hardening_revision"], module.HARDENING_REVISION
        )
        self.assertEqual(
            plan["semantic"]["hardening_revision"], module.HARDENING_REVISION
        )
        self.assertEqual(
            plan["execution"]["hardening_revision"], module.HARDENING_REVISION
        )

    def test_hardening_compiler_has_no_execution_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
