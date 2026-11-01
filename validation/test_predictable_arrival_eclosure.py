import copy
import unittest
from fractions import Fraction

from predictable_arrival_eclosure import (
    activation_state_digest,
    audit_predictable_arrival_ledger,
    select_predictable_arrival_capacity_actions,
    verify_predictable_arrival_capacity_result,
)


def make_ledger(hypotheses, rounds):
    """Build canonical event batches; factors default to one for active ids."""
    active = set()
    activation_history = []
    evidence_history = []
    events = []
    for t, (new_ids, factors) in enumerate(rounds):
        state = activation_state_digest(
            hypotheses, t, sorted(active), activation_history, evidence_history
        )
        for hypothesis_id in sorted(new_ids):
            events.append({
                "seq": len(events),
                "round": t,
                "kind": "activation_commit",
                "hypothesis_id": hypothesis_id,
                "pre_state_sha256": state,
            })
            active.add(hypothesis_id)
            activation_history.append([t, hypothesis_id])
        if not active:
            raise ValueError("test fixture must activate at least one hypothesis")
        for hypothesis_id in sorted(active):
            factor = factors.get(hypothesis_id, 1)
            events.append({
                "seq": len(events),
                "round": t,
                "kind": "evidence_increment",
                "hypothesis_id": hypothesis_id,
                "factor": factor,
            })
            evidence_history.append([t, hypothesis_id, Fraction(factor)])
    return events


class PredictableArrivalEclosureTests(unittest.TestCase):
    def setUp(self):
        self.hypotheses = [
            {"id": "h1", "priority_score": 1, "value": 10, "cost": 1},
            {"id": "h2", "priority_score": 1, "value": 8, "cost": 1},
            {"id": "h3", "priority_score": 1, "value": 1, "cost": 1},
        ]
        # h2 arrives only after round-1 evidence for h1 has been observed.
        # Its activation receipt commits to that exact pre-round history.
        self.rounds = [
            (["h1"], {"h1": 1}),
            ([], {"h1": 8}),
            (["h2"], {"h1": 1, "h2": 16}),
        ]
        self.events = make_ledger(self.hypotheses, self.rounds)
        self.alpha = Fraction(1, 2)
        self.capacity = [0, 1, 2]
        self.thresholds = [2, 4, 8, 16, 32, 64]
        self.adjuster = [1, 1, 1, 1, 1, 1]

    def test_predictable_random_arrival_ledger_and_capacity_selection(self):
        audit = audit_predictable_arrival_ledger(
            self.hypotheses, self.events, len(self.capacity)
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["catalogue_ids"], ["h1", "h2", "h3"])
        self.assertEqual(audit["active_masks"], [0b001, 0b001, 0b011])
        self.assertEqual(audit["activation_rounds"], {"h1": 0, "h2": 2})
        self.assertEqual(audit["activation_commitments_checked"], 2)
        self.assertEqual(audit["evidence_increments_checked"], 4)

        result = select_predictable_arrival_capacity_actions(
            self.hypotheses, self.events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        self.assertEqual(
            [row["rejections"] for row in result["selection"]["decisions"]],
            [[], [1], [1, 2]],
        )
        check = verify_predictable_arrival_capacity_result(
            result, self.hypotheses, self.events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        self.assertTrue(check["valid"], check)
        self.assertTrue(check["optimality_verified"])

    def test_activation_commit_is_bound_to_prior_evidence(self):
        # The round-2 digest includes the h1=8 evidence observed in round 1.
        expected = activation_state_digest(
            self.hypotheses,
            2,
            ["h1"],
            [[0, "h1"]],
            [[0, "h1", 1], [1, "h1", 8]],
        )
        event = next(
            e for e in self.events
            if e["kind"] == "activation_commit" and e["round"] == 2
        )
        self.assertEqual(event["pre_state_sha256"], expected)

        altered_history_digest = activation_state_digest(
            self.hypotheses,
            2,
            ["h1"],
            [[0, "h1"]],
            [[0, "h1", 1], [1, "h1", 1]],
        )
        self.assertNotEqual(expected, altered_history_digest)

    def test_activation_after_current_round_evidence_is_rejected(self):
        bad = copy.deepcopy(self.events)
        activation_index = next(
            i for i, event in enumerate(bad)
            if event["kind"] == "activation_commit" and event["round"] == 2
        )
        evidence_index = next(
            i for i, event in enumerate(bad)
            if event["kind"] == "evidence_increment" and event["round"] == 2
        )
        activation = bad.pop(activation_index)
        evidence_index = next(
            i for i, event in enumerate(bad)
            if event["kind"] == "evidence_increment" and event["round"] == 2
        )
        bad.insert(evidence_index + 1, activation)
        for seq, event in enumerate(bad):
            event["seq"] = seq
        with self.assertRaisesRegex(ValueError, "activation_after_evidence"):
            audit_predictable_arrival_ledger(
                self.hypotheses, bad, len(self.capacity)
            )

    def test_tampered_pre_evidence_digest_is_rejected(self):
        bad = copy.deepcopy(self.events)
        activation = next(
            e for e in bad
            if e["kind"] == "activation_commit" and e["round"] == 2
        )
        activation["pre_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pre-evidence filtration state"):
            audit_predictable_arrival_ledger(
                self.hypotheses, bad, len(self.capacity)
            )

    def test_activation_outside_finite_envelope_is_rejected(self):
        bad = copy.deepcopy(self.events)
        activation = next(
            e for e in bad if e["kind"] == "activation_commit" and e["round"] == 2
        )
        activation["hypothesis_id"] = "not-in-catalogue"
        with self.assertRaisesRegex(ValueError, "outside the deterministic finite envelope"):
            audit_predictable_arrival_ledger(
                self.hypotheses, bad, len(self.capacity)
            )

    def test_missing_or_duplicate_active_coordinate_increment_is_rejected(self):
        missing = [
            copy.deepcopy(e) for e in self.events
            if not (
                e["round"] == 2
                and e["kind"] == "evidence_increment"
                and e["hypothesis_id"] == "h2"
            )
        ]
        for seq, event in enumerate(missing):
            event["seq"] = seq
        with self.assertRaisesRegex(ValueError, "missing=.*h2"):
            audit_predictable_arrival_ledger(
                self.hypotheses, missing, len(self.capacity)
            )

        duplicate = copy.deepcopy(self.events)
        source = next(
            e for e in duplicate
            if e["round"] == 2
            and e["kind"] == "evidence_increment"
            and e["hypothesis_id"] == "h2"
        )
        extra = copy.deepcopy(source)
        duplicate.append(extra)
        for seq, event in enumerate(duplicate):
            event["seq"] = seq
        with self.assertRaisesRegex(ValueError, "duplicate coordinate increment"):
            audit_predictable_arrival_ledger(
                self.hypotheses, duplicate, len(self.capacity)
            )

    def test_inactive_coordinate_cannot_emit_evidence(self):
        bad = copy.deepcopy(self.events)
        event = next(
            e for e in bad
            if e["round"] == 1 and e["kind"] == "evidence_increment"
        )
        event["hypothesis_id"] = "h3"
        with self.assertRaisesRegex(ValueError, "not-yet-active"):
            audit_predictable_arrival_ledger(
                self.hypotheses, bad, len(self.capacity)
            )

    def test_contiguous_integer_sequence_is_enforced(self):
        bad = copy.deepcopy(self.events)
        bad[0]["seq"] = 7
        with self.assertRaisesRegex(ValueError, "contiguous and ordered"):
            audit_predictable_arrival_ledger(
                self.hypotheses, bad, len(self.capacity)
            )
        non_integer = copy.deepcopy(self.events)
        non_integer[0]["seq"] = 0.0
        with self.assertRaisesRegex(ValueError, "contiguous and ordered"):
            audit_predictable_arrival_ledger(
                self.hypotheses, non_integer, len(self.capacity)
            )

    def test_same_round_activation_order_is_canonical(self):
        rounds = [(["h1", "h2"], {"h1": 1, "h2": 1})]
        events = make_ledger(self.hypotheses, rounds)
        activations = [
            i for i, event in enumerate(events)
            if event["kind"] == "activation_commit"
        ]
        events[activations[0]], events[activations[1]] = (
            events[activations[1]], events[activations[0]]
        )
        for seq, event in enumerate(events):
            event["seq"] = seq
        with self.assertRaisesRegex(ValueError, "canonical catalogue order"):
            audit_predictable_arrival_ledger(self.hypotheses, events, 1)

    def test_result_verifier_rejects_receipt_tampering(self):
        result = select_predictable_arrival_capacity_actions(
            self.hypotheses, self.events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        tampered = copy.deepcopy(result)
        tampered["arrival_audit"]["ledger_sha256"] = "f" * 64
        check = verify_predictable_arrival_capacity_result(
            tampered, self.hypotheses, self.events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        self.assertFalse(check["valid"])
        self.assertIn(
            ("arrival_ledger_digest_mismatch",), check["failures"]
        )

    def test_future_factor_change_does_not_rewrite_earlier_decisions(self):
        first = select_predictable_arrival_capacity_actions(
            self.hypotheses, self.events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        changed_events = copy.deepcopy(self.events)
        changed_events[-1]["factor"] = 32
        second = select_predictable_arrival_capacity_actions(
            self.hypotheses, changed_events, self.alpha, self.capacity,
            self.thresholds, self.adjuster, share_policy="equal_active",
        )
        first_decisions = first["selection"]["decisions"]
        second_decisions = second["selection"]["decisions"]
        self.assertEqual(
            [row["rejections"] for row in first_decisions[:2]],
            [row["rejections"] for row in second_decisions[:2]],
        )
        self.assertEqual(
            [row["prefix_sha256"] for row in first_decisions[:2]],
            [row["prefix_sha256"] for row in second_decisions[:2]],
        )

    def test_wrong_horizon_and_non_nested_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "outside the declared horizon"):
            audit_predictable_arrival_ledger(
                self.hypotheses, self.events, 2
            )
        bad_catalogue = copy.deepcopy(self.hypotheses)
        bad_catalogue.append(copy.deepcopy(bad_catalogue[0]))
        with self.assertRaisesRegex(ValueError, "duplicate hypothesis id"):
            audit_predictable_arrival_ledger(
                bad_catalogue, self.events, len(self.capacity)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
