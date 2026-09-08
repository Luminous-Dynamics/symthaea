#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import verify_ci_workflow_definition_identity as verify

WORKFLOW = ".github/workflows/example-focused.yml"
DATA = b"name: Example\non: pull_request\n"


def write(root: Path, data: bytes = DATA) -> Path:
    target = root / WORKFLOW
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


class WorkflowDefinitionIdentityContracts(unittest.TestCase):
    def setUp(self):
        self.expected = verify.git_blob_sha(DATA)

    def roots(self):
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        head = base / "head"
        executing = base / "executing"
        head.mkdir()
        executing.mkdir()
        write(head)
        write(executing)
        return tmp, head, executing

    def test_exact_registered_judge_verifies(self):
        tmp, head, executing = self.roots()
        try:
            result = verify.verify_workflow_definition(
                workflow=WORKFLOW,
                expected_git_blob=self.expected,
                head_root=head,
                executing_root=executing,
            )
            self.assertTrue(result["authority"]["workflow_definition_identity_verified"])
            self.assertFalse(result["authority"]["focused_theorem_passed"])
            self.assertFalse(result["authority"]["scientific_execution_qualified"])
        finally:
            tmp.cleanup()

    def test_git_blob_semantics(self):
        self.assertEqual(
            verify.git_blob_sha(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_head_drift_rejected(self):
        tmp, head, executing = self.roots()
        try:
            write(head, b"drift\n")
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_executing_drift_rejected(self):
        tmp, head, executing = self.roots()
        try:
            write(executing, b"drift\n")
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_both_same_but_unregistered_rejected(self):
        tmp, head, executing = self.roots()
        try:
            write(head, b"other\n")
            write(executing, b"other\n")
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_missing_head_rejected(self):
        tmp, head, executing = self.roots()
        try:
            (head / WORKFLOW).unlink()
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_missing_executing_rejected(self):
        tmp, head, executing = self.roots()
        try:
            (executing / WORKFLOW).unlink()
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_symlinked_head_rejected(self):
        tmp, head, executing = self.roots()
        try:
            target = head / WORKFLOW
            outside = head / "outside.yml"
            outside.write_bytes(DATA)
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_symlinked_executing_rejected(self):
        tmp, head, executing = self.roots()
        try:
            target = executing / WORKFLOW
            outside = executing / "outside.yml"
            outside.write_bytes(DATA)
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=self.expected,
                    head_root=head,
                    executing_root=executing,
                )
        finally:
            tmp.cleanup()

    def test_bad_workflow_path_rejected(self):
        for bad in (
            ".github/workflows/../x.yml",
            ".github/workflows/x.yaml",
            "/.github/workflows/x.yml",
            "scripts/x.yml",
            ".github/workflows/X.yml",
        ):
            with self.subTest(path=bad), self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=bad,
                    expected_git_blob=self.expected,
                    head_root=Path("."),
                    executing_root=Path("."),
                )

    def test_bad_blob_spelling_rejected(self):
        for bad in ("", "A" * 40, "a" * 39, 1):
            with self.subTest(blob=bad), self.assertRaises(verify.WorkflowIdentityError):
                verify.verify_workflow_definition(
                    workflow=WORKFLOW,
                    expected_git_blob=bad,
                    head_root=Path("."),
                    executing_root=Path("."),
                )

    def test_cli_success(self):
        tmp, head, executing = self.roots()
        try:
            self.assertEqual(
                verify.main([
                    "--workflow", WORKFLOW,
                    "--expected-git-blob", self.expected,
                    "--head-root", str(head),
                    "--executing-root", str(executing),
                ]),
                0,
            )
        finally:
            tmp.cleanup()

    def test_cli_drift_fails(self):
        tmp, head, executing = self.roots()
        try:
            write(executing, b"drift\n")
            self.assertEqual(
                verify.main([
                    "--workflow", WORKFLOW,
                    "--expected-git-blob", self.expected,
                    "--head-root", str(head),
                    "--executing-root", str(executing),
                ]),
                2,
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
