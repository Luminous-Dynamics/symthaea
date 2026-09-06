#!/usr/bin/env python3
"""Adversarial tests for scripts/check-repo-integrity.py."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("check-repo-integrity.py")
SPEC = importlib.util.spec_from_file_location("repo_integrity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repo_integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repo_integrity)


class RepoIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo, self.claims = repo_integrity.validate()
        self.original_load_toml = repo_integrity.load_toml

    def _validate_with(self, repo: dict, claims: dict) -> None:
        def fake_load(path: Path) -> dict:
            if path.name == "repo.toml":
                return repo
            if path.name == "claims.toml":
                return claims
            return self.original_load_toml(path)

        with mock.patch.object(repo_integrity, "load_toml", side_effect=fake_load):
            repo_integrity.validate()

    def test_current_contract_validates(self) -> None:
        self.assertEqual(self.repo["repository"], "Luminous-Dynamics/symthaea")
        self.assertEqual(self.repo["conformance_level"], "RC-1")
        self.assertGreater(len(self.claims["claim"]), 0)

    def test_parent_escape_is_rejected(self) -> None:
        with self.assertRaises(repo_integrity.ContractError):
            repo_integrity.safe_repo_path("../outside", "test")

    def test_absolute_path_is_rejected(self) -> None:
        with self.assertRaises(repo_integrity.ContractError):
            repo_integrity.safe_repo_path("/tmp/outside", "test")

    def test_symlink_escape_is_rejected(self) -> None:
        link = repo_integrity.ROOT / ".repo-integrity-escape-test"
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / "probe.txt").write_text("outside", encoding="utf-8")
            try:
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaises(repo_integrity.ContractError):
                    repo_integrity.safe_repo_path(
                        ".repo-integrity-escape-test/probe.txt",
                        "test",
                    )
            finally:
                link.unlink(missing_ok=True)

    def test_malformed_repo_schema_is_rejected(self) -> None:
        repo = copy.deepcopy(self.repo)
        repo["schema_version"] = 99
        with self.assertRaises(repo_integrity.ContractError):
            self._validate_with(repo, self.claims)

    def test_duplicate_claim_id_is_rejected(self) -> None:
        claims = copy.deepcopy(self.claims)
        claims["claim"].append(copy.deepcopy(claims["claim"][0]))
        with self.assertRaises(repo_integrity.ContractError):
            self._validate_with(self.repo, claims)

    def test_missing_claim_scope_is_rejected(self) -> None:
        claims = copy.deepcopy(self.claims)
        claims["claim"][0].pop("scope")
        with self.assertRaises(repo_integrity.ContractError):
            self._validate_with(self.repo, claims)

    def test_invalid_claim_scope_is_rejected(self) -> None:
        claims = copy.deepcopy(self.claims)
        claims["claim"][0]["scope"] = "Repository Scope"
        with self.assertRaises(repo_integrity.ContractError):
            self._validate_with(self.repo, claims)

    def test_rc3_without_protected_promotion_is_rejected(self) -> None:
        repo = copy.deepcopy(self.repo)
        repo["conformance_level"] = "RC-3"
        repo["policy"]["require_protected_promotion"] = True
        repo["promotion"]["protected"] = False
        with self.assertRaises(repo_integrity.ContractError):
            self._validate_with(repo, self.claims)

    def test_renderer_is_deterministic(self) -> None:
        first = repo_integrity.render_claims(self.claims)
        second = repo_integrity.render_claims(copy.deepcopy(self.claims))
        self.assertEqual(first, second)

    def test_renderer_collapses_real_newlines_inside_cells(self) -> None:
        rendered = repo_integrity.render_claims(
            {
                "coverage": "partial",
                "claim": [
                    {
                        "id": "TEST-001",
                        "scope": "test-scope",
                        "kind": "non_claim",
                        "status": "declared_boundary",
                        "statement": "line one\nline two",
                        "evidence": ["README.md"],
                        "limitations": ["first\nsecond"],
                    }
                ],
            }
        )
        self.assertIn("line one line two", rendered)
        self.assertIn("first second", rendered)
        self.assertNotIn("line one\nline two", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
