#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import render_workbench_provenance_ci_paths_ignore as render

SYNTHETIC = b"""name: CI\n\non:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:\n\njobs:\n  test:\n    runs-on: ubuntu-latest\n"""


class PathsIgnoreRendererContracts(unittest.TestCase):
    def test_exempt_path_set_is_exact_current_stack_union(self):
        self.assertEqual(len(render.EXEMPT_PATHS), 28)
        self.assertEqual(len(set(render.EXEMPT_PATHS)), 28)
        self.assertEqual(tuple(sorted(render.EXEMPT_PATHS)), render.EXEMPT_PATHS)

    def test_expected_input_blob_is_current_reviewed_ci(self):
        self.assertEqual(render.EXPECTED_INPUT_GIT_BLOB, "a48366076b30eb8e12d22c927a3b8bf333181409")

    def test_synthetic_render_changes_only_pull_request_stanza(self):
        out = render.render_bytes(SYNTHETIC, require_reviewed_input=False)
        expected = SYNTHETIC.replace(render.PULL_REQUEST_STANZA.encode(), render.rendered_stanza().encode(), 1)
        self.assertEqual(out, expected)

    def test_rendered_stanza_has_exact_paths(self):
        text = render.rendered_stanza()
        self.assertTrue(text.startswith("  pull_request:\n    paths-ignore:\n"))
        for path in render.EXEMPT_PATHS:
            self.assertEqual(text.count(f"      - '{path}'\n"), 1)

    def test_workbench_stack_workflows_are_exempt(self):
        for name in (
            "workbench-execution-capsule-profile.yml",
            "workbench-nix-closure-identity.yml",
            "workbench-nix-closure-capture.yml",
            "workbench-nix-closure-capture-verifier.yml",
            "workbench-invocation-isolation-profile.yml",
            "workbench-root-nar-membership.yml",
        ):
            self.assertIn(f".github/workflows/{name}", render.EXEMPT_PATHS)

    def test_global_ci_is_not_exempt(self):
        self.assertNotIn(".github/workflows/ci.yml", render.EXEMPT_PATHS)

    def test_cargo_is_not_exempt(self):
        self.assertNotIn("Cargo.toml", render.EXEMPT_PATHS)
        self.assertNotIn("Cargo.lock", render.EXEMPT_PATHS)

    def test_non_workbench_neuro_is_not_exempt(self):
        self.assertNotIn("scripts/derive_hcpmmp1_neuromaps_lineage_b.py", render.EXEMPT_PATHS)

    def test_unknown_input_blob_rejected(self):
        with self.assertRaises(render.RenderError):
            render.render_bytes(SYNTHETIC)

    def test_missing_pull_request_stanza_rejected(self):
        with self.assertRaises(render.RenderError):
            render.render_bytes(b"name: CI\n", require_reviewed_input=False)

    def test_duplicate_pull_request_stanza_rejected(self):
        with self.assertRaises(render.RenderError):
            render.render_bytes(SYNTHETIC + b"  pull_request:\n", require_reviewed_input=False)

    def test_already_patched_source_rejected(self):
        first = render.render_bytes(SYNTHETIC, require_reviewed_input=False)
        with self.assertRaises(render.RenderError):
            render.render_bytes(first, require_reviewed_input=False)

    def test_candidate_tamper_rejected(self):
        candidate = render.render_bytes(SYNTHETIC, require_reviewed_input=False) + b"# forged\n"
        with self.assertRaises(render.RenderError):
            render.verify_render(SYNTHETIC, candidate, require_reviewed_input=False)

    def test_candidate_non_workbench_ignore_injection_rejected(self):
        candidate = render.render_bytes(SYNTHETIC, require_reviewed_input=False)
        candidate = candidate.replace(b"    paths-ignore:\n", b"    paths-ignore:\n      - 'Cargo.toml'\n", 1)
        with self.assertRaises(render.RenderError):
            render.verify_render(SYNTHETIC, candidate, require_reviewed_input=False)

    def test_rest_of_workflow_is_byte_preserved(self):
        out = render.render_bytes(SYNTHETIC, require_reviewed_input=False)
        before_suffix = SYNTHETIC.split(render.PULL_REQUEST_STANZA.encode(), 1)[1]
        after_suffix = out.split(render.rendered_stanza().encode(), 1)[1]
        self.assertEqual(after_suffix, before_suffix)

    def test_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ci.yml"
            path.write_bytes(b"existing")
            with self.assertRaises(render.RenderError):
                render.write_exclusive(path, b"new")

    def test_exclusive_output_writes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ci.yml"
            data = b"candidate"
            render.write_exclusive(path, data)
            self.assertEqual(path.read_bytes(), data)

    def test_git_blob_identity_changes_when_bytes_change(self):
        self.assertNotEqual(render.git_blob_sha(SYNTHETIC), render.git_blob_sha(SYNTHETIC + b"x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
