#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("generate-mujoco-runtime-manifest.py")
spec = importlib.util.spec_from_file_location("mujoco_runtime_manifest", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load MuJoCo runtime manifest generator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MujocoRuntimeManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "mujoco"
        (self.root / "lib").mkdir(parents=True)
        (self.root / "include").mkdir()
        self.lib = self.root / "lib" / "libmujoco.so.3.8.0"
        self.lib.write_bytes(b"synthetic-libmujoco\n")
        (self.root / "lib" / "libmujoco.so").symlink_to("libmujoco.so.3.8.0")
        (self.root / "include" / "mujoco.h").write_text(
            "/* synthetic header */\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_manifest_matches_b2_schema_and_x86_asset_identity(self) -> None:
        manifest = module.build_manifest(self.root, "x86_64-linux")
        self.assertEqual(manifest["schema_id"], module.SCHEMA_ID)
        self.assertEqual(
            manifest["extracted_tree_digest_algorithm"],
            module.EXTRACTED_TREE_DIGEST_ALGORITHM,
        )
        self.assertEqual(manifest["version"], "3.8.0")
        self.assertEqual(manifest["mujoco_rs_compatibility"], "4.0.1+mj-3.8.0")
        self.assertEqual(manifest["upstream_asset_id"], 404891937)
        self.assertEqual(
            manifest["upstream_asset_sha256"],
            "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab",
        )

    def test_manifest_write_is_idempotent_and_excluded_from_tree_digest(self) -> None:
        before = module.extracted_tree_sha256(self.root)
        output = module.write_manifest(self.root, "x86_64-linux")
        first_bytes = output.read_bytes()
        after_first = module.extracted_tree_sha256(self.root)
        module.write_manifest(self.root, "x86_64-linux")
        second_bytes = output.read_bytes()
        after_second = module.extracted_tree_sha256(self.root)
        self.assertEqual(before, after_first)
        self.assertEqual(after_first, after_second)
        self.assertEqual(first_bytes, second_bytes)

    def test_tree_digest_is_root_path_independent(self) -> None:
        other = Path(self.temp.name) / "other"
        (other / "lib").mkdir(parents=True)
        (other / "include").mkdir()
        (other / "lib" / "libmujoco.so.3.8.0").write_bytes(self.lib.read_bytes())
        (other / "lib" / "libmujoco.so").symlink_to("libmujoco.so.3.8.0")
        (other / "include" / "mujoco.h").write_text(
            "/* synthetic header */\n", encoding="utf-8"
        )
        self.assertEqual(
            module.extracted_tree_sha256(self.root),
            module.extracted_tree_sha256(other),
        )

    def test_root_mode_change_changes_tree_digest(self) -> None:
        self.root.chmod(0o755)
        first = module.extracted_tree_sha256(self.root)
        self.root.chmod(0o750)
        second = module.extracted_tree_sha256(self.root)
        self.assertNotEqual(first, second)

    def test_file_content_change_changes_tree_digest(self) -> None:
        first = module.extracted_tree_sha256(self.root)
        (self.root / "include" / "mujoco.h").write_text(
            "/* changed header */\n", encoding="utf-8"
        )
        second = module.extracted_tree_sha256(self.root)
        self.assertNotEqual(first, second)

    def test_lib_change_changes_lib_and_tree_digests(self) -> None:
        first = module.build_manifest(self.root, "x86_64-linux")
        self.lib.write_bytes(b"changed-libmujoco\n")
        second = module.build_manifest(self.root, "x86_64-linux")
        self.assertNotEqual(first["libmujoco_sha256"], second["libmujoco_sha256"])
        self.assertNotEqual(first["extracted_tree_sha256"], second["extracted_tree_sha256"])

    def test_relative_confined_symlink_is_admitted(self) -> None:
        manifest = module.build_manifest(self.root, "x86_64-linux")
        self.assertRegex(manifest["libmujoco_sha256"], r"^[0-9a-f]{64}$")

    def test_absolute_symlink_is_rejected(self) -> None:
        target = self.root / "include" / "mujoco.h"
        (self.root / "absolute-link").symlink_to(target)
        with self.assertRaises(module.ManifestError):
            module.extracted_tree_sha256(self.root)

    def test_escaping_symlink_is_rejected(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.write_text("outside\n", encoding="utf-8")
        (self.root / "escape").symlink_to("../outside")
        with self.assertRaises(module.ManifestError):
            module.extracted_tree_sha256(self.root)

    def test_broken_symlink_is_rejected(self) -> None:
        (self.root / "broken").symlink_to("missing")
        with self.assertRaises(module.ManifestError):
            module.extracted_tree_sha256(self.root)

    def test_special_files_are_rejected(self) -> None:
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(module.ManifestError):
            module.extracted_tree_sha256(self.root)

    def test_unsupported_platform_is_rejected(self) -> None:
        with self.assertRaises(module.ManifestError):
            module.build_manifest(self.root, "riscv64-linux")

    def test_manifest_is_canonical_json_with_exact_field_set(self) -> None:
        output = module.write_manifest(self.root, "x86_64-linux")
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            set(parsed),
            {
                "schema_id",
                "platform",
                "version",
                "mujoco_rs_compatibility",
                "upstream_release_commit",
                "upstream_asset_id",
                "upstream_asset_name",
                "upstream_asset_size",
                "upstream_asset_sha256",
                "extracted_tree_digest_algorithm",
                "extracted_tree_sha256",
                "libmujoco_sha256",
            },
        )
        self.assertEqual(output.read_bytes().count(b"\n"), 1)


if __name__ == "__main__":
    unittest.main()
