#!/usr/bin/env python3
"""Unit tests for the mutation generators and their preregistered catalog mapping."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = Path(__file__).with_name("run_mutation_campaign.py")
SOURCE_PATH = ROOT / "tools/vleb/ledger-reconcile/src/lib.rs"
CATALOG_PATH = ROOT / "tools/vleb/mutation-catalog.json"

SPEC = importlib.util.spec_from_file_location("vleb_run_mutation_campaign", CAMPAIGN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load mutation campaign module")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class MutationCampaignGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.mutants = {mutation_id: (kind, transform, killer) for mutation_id, kind, transform, killer in campaign.MUTANTS}

    def test_active_campaign_exactly_matches_registered_catalog(self) -> None:
        active = self.catalog["active_campaign"]["required_mutation_ids"]
        self.assertEqual(set(active), set(self.mutants))
        self.assertEqual(len(active), len(set(active)))
        registered = {row["mutation_id"]: row for row in self.catalog["mutations"]}
        self.assertTrue(set(active).issubset(registered))
        for mutation_id, (_, _, expected_killer) in self.mutants.items():
            self.assertEqual(registered[mutation_id]["expected_killer_test"], expected_killer)
            self.assertEqual(registered[mutation_id]["expected_equivalence"], "non_equivalent")

    def test_each_generator_changes_source_and_retains_target_test(self) -> None:
        for mutation_id, (_, transform, expected_killer) in self.mutants.items():
            with self.subTest(mutation_id=mutation_id):
                mutated = transform(self.source)
                self.assertNotEqual(mutated, self.source)
                self.assertIn(expected_killer, self.source)
                self.assertIn("fn " + expected_killer + "(", self.source)

    def test_overflow_mutant_is_a_compilable_wrapping_operation(self) -> None:
        _, transform, _ = self.mutants["VLEB-LDG-004"]
        mutated = transform(self.source)
        self.assertEqual(mutated.count(".wrapping_add(amount_minor)"), 4)
        self.assertEqual(mutated.count(".checked_add(amount_minor)"), 0)
        self.assertNotIn("ok_or_else(|| LedgerError::new(\"account credit total overflow\"))?;", mutated)
        self.assertIn("fn arithmetic_overflow_is_rejected(", self.source)

    def test_credit_debit_mutant_swaps_branch_meaning_not_random_text(self) -> None:
        _, transform, _ = self.mutants["VLEB-LDG-007"]
        mutated = transform(self.source)
        match_start = mutated.index("match side {")
        match_end = mutated.index("\n        }\n\n        account.unique_transactions", match_start)
        block = mutated[match_start:match_end]
        debit_start = block.index("Side::Debit =>")
        credit_start = block.index("Side::Credit =>")
        self.assertLess(debit_start, credit_start, "mutant must swap branch discriminants")
        debit_labeled_branch = block[debit_start:credit_start]
        credit_labeled_branch = block[credit_start:]
        self.assertIn("account.credits_minor", debit_labeled_branch)
        self.assertIn("totals.credits_minor", debit_labeled_branch)
        self.assertIn("account.debits_minor", credit_labeled_branch)
        self.assertIn("totals.debits_minor", credit_labeled_branch)

    def test_duplicate_mutant_only_removes_exact_duplicate_short_circuit(self) -> None:
        _, transform, _ = self.mutants["VLEB-LDG-001"]
        mutated = transform(self.source)
        self.assertIn("MUTANT: exact duplicates fall through", mutated)
        self.assertIn("conflicting duplicate transaction_id", mutated)
        self.assertIn("if previous == &transaction", mutated)

    def test_conflict_mutant_only_accepts_conflicting_duplicate(self) -> None:
        _, transform, _ = self.mutants["VLEB-LDG-002"]
        mutated = transform(self.source)
        self.assertIn("MUTANT: conflicting transaction-ID reuse is silently ignored.", mutated)
        self.assertIn("if previous == &transaction", mutated)

    def test_catalog_is_not_mislabeled_as_fully_executed(self) -> None:
        self.assertEqual(len(self.catalog["mutations"]), 17)
        self.assertEqual(len(self.mutants), 4)
        self.assertNotEqual(set(self.catalog["active_campaign"]["required_mutation_ids"]), {
            row["mutation_id"] for row in self.catalog["mutations"]
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
