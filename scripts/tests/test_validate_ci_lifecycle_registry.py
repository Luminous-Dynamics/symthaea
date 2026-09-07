import unittest

from scripts.ci.validate_ci_lifecycle_registry import extract_job_blocks, validate


class LifecycleRegistryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = """name: CI
jobs:
  fmt:
    runs-on: ubuntu-latest
    steps: []
  safety:
    runs-on: ubuntu-latest
    steps: []
  test:
    runs-on: ubuntu-latest
    needs: fmt
    steps: []
"""
        self.registry = {
            "schema": "symthaea.ci-lifecycle-jobs.v1",
            "policy": {"unknown_job_default": "full"},
            "iteration_jobs": ["fmt", "safety"],
            "full_only_examples": ["test"],
        }

    def test_extracts_only_top_level_jobs(self) -> None:
        blocks = extract_job_blocks(self.workflow)
        self.assertEqual(set(blocks), {"fmt", "safety", "test"})

    def test_unlisted_job_defaults_to_full(self) -> None:
        result = validate(self.workflow, self.registry)
        self.assertEqual(result["iteration_job_count"], 2)
        self.assertEqual(result["full_only_default_count"], 1)
        self.assertEqual(result["full_only_jobs"], ["test"])
        self.assertEqual(result["unknown_job_default"], "full")
        self.assertFalse(result["tier1_green_is_merge_qualification"])

    def test_missing_iteration_job_fails(self) -> None:
        registry = dict(self.registry)
        registry["iteration_jobs"] = ["fmt", "does-not-exist"]
        with self.assertRaisesRegex(ValueError, "absent from ci.yml"):
            validate(self.workflow, registry)

    def test_iteration_job_may_not_depend_on_full_job(self) -> None:
        registry = dict(self.registry)
        registry["iteration_jobs"] = ["fmt", "test"]
        with self.assertRaisesRegex(ValueError, "may not have job-level needs"):
            validate(self.workflow, registry)

    def test_documented_full_only_example_must_exist(self) -> None:
        registry = dict(self.registry)
        registry["full_only_examples"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "full-only example jobs absent"):
            validate(self.workflow, registry)

    def test_duplicate_job_id_fails_closed(self) -> None:
        malformed = """jobs:
  fmt:
    runs-on: ubuntu-latest
  fmt:
    runs-on: ubuntu-latest
"""
        with self.assertRaisesRegex(ValueError, "duplicate top-level job id"):
            extract_job_blocks(malformed)

    def test_missing_jobs_root_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no exact top-level 'jobs:'"):
            extract_job_blocks("name: CI\n")


if __name__ == "__main__":
    unittest.main()
