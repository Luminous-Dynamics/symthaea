import unittest

from scripts.ci.apply_ci_lifecycle_tiering import (
    FULL_GUARD,
    MARKER,
    PR_TRIGGER_NEW,
    git_blob_sha,
    transform,
)
from scripts.ci.validate_ci_lifecycle_registry import extract_job_blocks


class LifecycleTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = {
            "schema": "symthaea.ci-lifecycle-jobs.v1",
            "policy": {"unknown_job_default": "full"},
            "iteration_jobs": ["fmt", "safety"],
            "full_only_examples": ["test"],
        }
        self.workflow = """name: CI
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
jobs:
  fmt:
    name: Format
    runs-on: ubuntu-latest
    steps: []
  safety:
    name: Safety
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps: []
  test:
    name: Test
    runs-on: ubuntu-latest
    steps: []
  child:
    name: Child
    runs-on: ubuntu-latest
    needs: test
    steps: []
  scheduled:
    name: Scheduled
    runs-on: ubuntu-latest
    if: |
      github.event_name == 'schedule' ||
      github.event_name == 'workflow_dispatch'
    steps: []
"""

    def test_git_blob_sha_uses_git_blob_framing(self) -> None:
        # Canonical Git test vector: SHA-1("blob 0\\0")
        self.assertEqual(
            git_blob_sha(b""),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        )

    def test_expands_pull_request_lifecycle_types(self) -> None:
        result = transform(self.workflow, self.registry)
        self.assertIn(PR_TRIGGER_NEW, result.text)
        self.assertNotIn("  pull_request:\njobs:", result.text)

    def test_tier1_jobs_remain_unguarded(self) -> None:
        result = transform(self.workflow, self.registry)
        blocks = extract_job_blocks(result.text)
        for job_id in ("fmt", "safety"):
            block = "\n".join(blocks[job_id].lines)
            self.assertNotIn(MARKER, block)
            self.assertNotIn(FULL_GUARD, block)

    def test_unknown_job_defaults_to_full_guard(self) -> None:
        result = transform(self.workflow, self.registry)
        blocks = extract_job_blocks(result.text)
        block = "\n".join(blocks["child"].lines)
        self.assertIn(MARKER, block)
        self.assertIn(FULL_GUARD, block)

    def test_full_job_without_if_gets_one_guard(self) -> None:
        result = transform(self.workflow, self.registry)
        block = "\n".join(extract_job_blocks(result.text)["test"].lines)
        self.assertEqual(block.count("    if:"), 1)
        self.assertIn(FULL_GUARD, block)

    def test_existing_single_line_if_is_conjoined_for_full_job(self) -> None:
        registry = dict(self.registry)
        registry["iteration_jobs"] = ["fmt"]
        result = transform(self.workflow, registry)
        block = "\n".join(extract_job_blocks(result.text)["safety"].lines)
        self.assertEqual(block.count("    if:"), 1)
        self.assertIn(FULL_GUARD, block)
        self.assertIn("github.event_name == 'pull_request'", block)

    def test_existing_multiline_if_is_conjoined(self) -> None:
        result = transform(self.workflow, self.registry)
        block = "\n".join(extract_job_blocks(result.text)["scheduled"].lines)
        self.assertEqual(block.count("    if:"), 1)
        self.assertIn(FULL_GUARD, block)
        self.assertIn("github.event_name == 'schedule'", block)
        self.assertIn("github.event_name == 'workflow_dispatch'", block)

    def test_needs_dependency_is_preserved(self) -> None:
        result = transform(self.workflow, self.registry)
        block = "\n".join(extract_job_blocks(result.text)["child"].lines)
        self.assertIn("    needs: test", block)

    def test_every_full_job_is_marked_exactly_once(self) -> None:
        result = transform(self.workflow, self.registry)
        blocks = extract_job_blocks(result.text)
        for job_id in ("test", "child", "scheduled"):
            block = "\n".join(blocks[job_id].lines)
            self.assertEqual(block.count(MARKER), 1)
            self.assertEqual(block.count(FULL_GUARD), 1)

    def test_transform_refuses_already_transformed_input(self) -> None:
        first = transform(self.workflow, self.registry)
        with self.assertRaisesRegex(ValueError, "broad pull_request trigger"):
            transform(first.text, self.registry)


if __name__ == "__main__":
    unittest.main()
