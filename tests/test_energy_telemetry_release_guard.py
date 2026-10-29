"""Offline tests for local energy telemetry release policy."""
from __future__ import annotations

from datetime import datetime
import unittest

from tools.energy_telemetry.release_guard import ReleasePolicyError, ReleasePolicyGuard


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def batch(ch: str) -> str:
    return ch * 32


def digest(ch: str) -> str:
    return ch * 64


def people(start: int = 0, count: int = 20):
    return [f"local-token-{i:04d}" for i in range(start, start + count)]


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.guard = ReleasePolicyGuard()
        self.a = dt("2026-10-12T10:00:00Z")
        self.b = dt("2026-10-12T10:15:00Z")
        self.c = dt("2026-10-12T10:30:00Z")

    def submit(self, *, batch_id="a", payload="b", participants=None, start=None, end=None,
               recipient="grid_operations", epsilon=None, delta=None):
        return self.guard.submit(
            batch_id=batch(batch_id),
            payload_sha256=digest(payload),
            participant_tokens=people() if participants is None else participants,
            start_utc=self.a if start is None else start,
            end_utc=self.b if end is None else end,
            recipient_class=recipient,
            epsilon=epsilon,
            delta=delta,
        )

    def test_grid_operations_aggregate_is_accepted(self):
        self.assertEqual(self.submit(), "accepted")

    def test_minimum_cohort_is_enforced_against_local_participant_set(self):
        with self.assertRaisesRegex(ReleasePolicyError, "below the required minimum"):
            self.submit(participants=people(count=19))

    def test_participant_tokens_must_be_non_empty(self):
        with self.assertRaisesRegex(ReleasePolicyError, "non-empty opaque"):
            self.submit(participants=[""])

    def test_research_requires_dp(self):
        with self.assertRaisesRegex(ReleasePolicyError, "requires differential privacy"):
            self.submit(recipient="approved_research")

    def test_public_requires_dp(self):
        with self.assertRaisesRegex(ReleasePolicyError, "requires differential privacy"):
            self.submit(recipient="public_reporting")

    def test_dp_release_is_accepted_within_budget(self):
        self.assertEqual(self.submit(recipient="approved_research", epsilon=0.5, delta=0.0), "accepted")

    def test_overlapping_window_for_shared_participant_is_rejected(self):
        self.submit()
        with self.assertRaisesRegex(ReleasePolicyError, "overlapping release windows"):
            self.submit(batch_id="c", payload="d", start=dt("2026-10-12T10:10:00Z"),
                        end=dt("2026-10-12T10:20:00Z"))

    def test_adjacent_windows_are_not_overlapping(self):
        self.submit()
        self.assertEqual(self.submit(batch_id="c", payload="d", start=self.b, end=self.c), "accepted")

    def test_disjoint_participant_cohorts_can_have_overlapping_windows(self):
        self.submit()
        self.assertEqual(self.submit(batch_id="c", payload="d", participants=people(20, 20)), "accepted")

    def test_exact_retransmission_is_idempotent_and_not_charged_twice(self):
        self.submit(recipient="approved_research", epsilon=1.0, delta=0.0)
        self.assertEqual(self.submit(recipient="approved_research", epsilon=1.0, delta=0.0), "idempotent_retransmission")
        # A second disjoint time window consumes only the second unit of epsilon.
        self.assertEqual(
            self.submit(batch_id="c", payload="d", start=self.b, end=self.c,
                        recipient="approved_research", epsilon=1.0, delta=0.0),
            "accepted",
        )

    def test_reused_batch_id_with_different_payload_is_rejected(self):
        self.submit()
        with self.assertRaisesRegex(ReleasePolicyError, "reused with different"):
            self.submit(payload="d")

    def test_cumulative_epsilon_budget_is_enforced(self):
        self.submit(recipient="approved_research", epsilon=1.0, delta=0.0)
        self.submit(batch_id="c", payload="d", start=self.b, end=self.c,
                    recipient="approved_research", epsilon=1.0, delta=0.0)
        with self.assertRaisesRegex(ReleasePolicyError, "cumulative epsilon budget exceeded"):
            self.submit(batch_id="e", payload="f", start=self.c,
                        end=dt("2026-10-12T10:45:00Z"), recipient="approved_research",
                        epsilon=0.01, delta=0.0)

    def test_cumulative_delta_budget_is_enforced(self):
        self.guard = ReleasePolicyGuard(max_epsilon_per_participant=3.0, max_delta_per_participant=1e-6)
        self.submit(recipient="approved_research", epsilon=0.5, delta=6e-7)
        with self.assertRaisesRegex(ReleasePolicyError, "cumulative delta budget exceeded"):
            self.submit(batch_id="c", payload="d", start=self.b, end=self.c,
                        recipient="approved_research", epsilon=0.5, delta=6e-7)

    def test_mixed_epsilon_delta_arguments_are_rejected(self):
        with self.assertRaisesRegex(ReleasePolicyError, "both be supplied"):
            self.submit(recipient="approved_research", epsilon=0.5, delta=None)

    def test_non_utc_window_is_rejected(self):
        with self.assertRaisesRegex(ReleasePolicyError, "must be UTC"):
            self.submit(start=dt("2026-10-12T12:00:00+02:00"))

    def test_reversed_window_is_rejected(self):
        with self.assertRaisesRegex(ReleasePolicyError, "end_utc must be later"):
            self.submit(start=self.b, end=self.a)

    def test_extremely_large_epsilon_fails_closed_without_overflow(self):
        with self.assertRaisesRegex(ReleasePolicyError, "epsilon must be finite"):
            self.submit(recipient="approved_research", epsilon=10 ** 10000, delta=0.0)

    def test_extremely_large_budget_configuration_fails_closed_without_overflow(self):
        with self.assertRaisesRegex(ValueError, "max_epsilon_per_participant"):
            ReleasePolicyGuard(max_epsilon_per_participant=10 ** 10000)


if __name__ == "__main__":
    unittest.main()
