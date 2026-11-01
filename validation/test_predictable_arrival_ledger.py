import copy
import hashlib
import random
import unittest

from predictable_arrival_ledger import (
    build_predictable_arrival_ledger,
    build_predictable_arrival_ledger_from_source,
    build_same_round_lookahead_negative_control,
    verify_predictable_arrival_ledger,
)
from countable_global_projective_eclosure import (
    projective_certificate,
    verify_countable_global_projective_path,
)


def _scores():
    # h1 has positive pre-round mean after rounds 0 and 1. h2 is activated
    # at the beginning of round 2, before round-2 observations are exposed.
    return [
        [1.0, 100.0, -100.0],
        [1.0, 100.0, -100.0],
        [0.5, 0.0, 0.0],
        [0.2, 0.3, 0.0],
        [-0.1, 0.4, 0.0],
    ]


class PredictableArrivalLedgerTests(unittest.TestCase):
    def test_valid_policy_replays_with_pre_observation_activation(self):
        ledger = build_predictable_arrival_ledger(
            _scores(),
            ["h1", "h2", "h3"],
            initial_active_count=1,
            min_observations=2,
            mean_threshold=0.0,
        )
        verified = verify_predictable_arrival_ledger(ledger)
        self.assertTrue(verified["valid"])
        self.assertEqual(
            verified["active_by_round"],
            [
                ("h1",),
                ("h1",),
                ("h1", "h2"),
                ("h1", "h2"),
                ("h1", "h2", "h3"),
            ],
        )

    def test_streaming_source_is_queried_only_after_active_set_freezes(self):
        calls = []
        score_rows = {
            0: {"h1": 1.0},
            1: {"h1": 1.0},
            2: {"h1": 0.5, "h2": 4.0},
            3: {"h1": 0.2, "h2": 0.3},
        }

        def source(round_index, active_ids):
            # The callback sees only the active IDs, never dormant coordinates.
            calls.append((round_index, active_ids))
            self.assertEqual(set(score_rows[round_index]), set(active_ids))
            return score_rows[round_index]

        ledger = build_predictable_arrival_ledger_from_source(
            source,
            4,
            ["h1", "h2", "h3"],
            initial_active_count=1,
            min_observations=2,
            mean_threshold=0.0,
        )
        self.assertEqual(
            calls,
            [
                (0, ("h1",)),
                (1, ("h1",)),
                (2, ("h1", "h2")),
                (3, ("h1", "h2")),
            ],
        )
        verified = verify_predictable_arrival_ledger(ledger)
        self.assertEqual(verified["active_by_round"][-1], ("h1", "h2"))

    def test_streaming_source_must_return_exact_active_roster(self):
        def missing(round_index, active_ids):
            return {}

        with self.assertRaisesRegex(ValueError, "exactly the active"):
            build_predictable_arrival_ledger_from_source(
                missing,
                2,
                ["h1", "h2"],
                initial_active_count=1,
                min_observations=2,
            )

    def test_same_round_lookahead_negative_control_is_hash_consistent_but_rejected(self):
        scores = [
            [0.0, 10.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [0.0, 0.0],
        ]
        invalid = build_same_round_lookahead_negative_control(
            scores,
            ["h1", "h2"],
            initial_active_count=1,
            min_observations=2,
            mean_threshold=0.0,
        )
        # The builder has chained every hash consistently; rejection is based
        # on the policy seeing evidence from the current round, not hash damage.
        with self.assertRaisesRegex(ValueError, "arrival policy replay mismatch"):
            verify_predictable_arrival_ledger(invalid)

    def test_history_digest_tamper_fails(self):
        ledger = build_predictable_arrival_ledger(
            _scores(), ["h1", "h2", "h3"], min_observations=2
        )
        changed = copy.deepcopy(ledger)
        changed["events"][2]["pre_history_digest"] = hashlib.sha256(b"wrong").hexdigest()
        with self.assertRaisesRegex(ValueError, "pre-round evidence digest"):
            verify_predictable_arrival_ledger(changed)

    def test_observation_omission_and_reordering_fail(self):
        ledger = build_predictable_arrival_ledger(
            _scores(), ["h1", "h2", "h3"], min_observations=2
        )
        omitted = copy.deepcopy(ledger)
        omitted["events"][2]["observed_scores"].pop()
        with self.assertRaises(ValueError):
            verify_predictable_arrival_ledger(omitted)

        reordered = copy.deepcopy(ledger)
        reordered["events"][2]["observed_scores"].reverse()
        with self.assertRaises(ValueError):
            verify_predictable_arrival_ledger(reordered)

    def test_policy_spec_and_event_schema_tampering_fail(self):
        ledger = build_predictable_arrival_ledger(
            _scores(), ["h1", "h2", "h3"], min_observations=2
        )
        changed_header = copy.deepcopy(ledger)
        changed_header["header"]["policy_spec_digest"] = "0" * 64
        # Recompute only the header hash: the verifier must still reject an
        # unknown policy specification, not merely detect an old digest.
        changed_header["header_hash"] = hashlib.sha256(
            __import__("json").dumps(
                changed_header["header"], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "policy specification"):
            verify_predictable_arrival_ledger(changed_header)

        changed_event = copy.deepcopy(ledger)
        changed_event["events"][0]["unrecognized_field"] = "injected"
        with self.assertRaisesRegex(ValueError, "event fields"):
            verify_predictable_arrival_ledger(changed_event)

    def test_event_chain_tamper_fails(self):
        ledger = build_predictable_arrival_ledger(
            _scores(), ["h1", "h2", "h3"], min_observations=2
        )
        changed = copy.deepcopy(ledger)
        changed["events"][1]["previous_event_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "predecessor"):
            verify_predictable_arrival_ledger(changed)

    def test_out_of_universe_or_nonfinite_scores_fail_at_construction(self):
        with self.assertRaises(ValueError):
            build_predictable_arrival_ledger(
                [[1.0, 2.0]], ["h1", "h2", "h3"], min_observations=2
            )
        with self.assertRaises(ValueError):
            build_predictable_arrival_ledger(
                [[float("nan")]], ["h1"], min_observations=2
            )

    def test_nested_projective_certificate_matches_full_true_set(self):
        ledger = build_predictable_arrival_ledger(
            _scores(),
            ["h1", "h2", "h3"],
            initial_active_count=1,
            min_observations=2,
            mean_threshold=0.0,
        )
        replay = verify_predictable_arrival_ledger(ledger)
        histories = {hypothesis_id: [] for hypothesis_id in ledger["header"]["hypothesis_ids"]}
        coordinate_paths = {i: [] for i in range(1, 4)}
        active_by_time = []
        for event, active_ids in zip(ledger["events"], replay["active_by_round"]):
            for hypothesis_id, score_hex in event["observed_scores"]:
                histories[hypothesis_id].append(float.fromhex(score_hex))
            # Construct a nonnegative finite algebraic evidence snapshot. A
            # dormant coordinate is exactly one; this checks padding only,
            # not the stochastic e-process assumption for these scores.
            values = {
                hypothesis_id: (
                    1.0 + sum(histories[hypothesis_id]) ** 2 / 10.0
                    if hypothesis_id in active_ids
                    else 1.0
                )
                for hypothesis_id in ledger["header"]["hypothesis_ids"]
            }
            value_map = {i + 1: values[f"h{i+1}"] for i in range(3)}
            full = projective_certificate((1, 2, 3), value_map)
            active_nulls = tuple(int(h[1:]) for h in active_ids)
            active = projective_certificate(active_nulls, value_map)
            self.assertEqual(full, active)
            active_by_time.append(tuple(int(h[1:]) for h in active_ids))
            for i in range(1, 4):
                coordinate_paths[i].append(value_map[i])

        # Bridge the verified event order into the finite global-projective
        # closure oracle. This validates the algebraic all-subsets contract
        # and neutral-one padding along the exact same arrival path.
        closure = verify_countable_global_projective_path(
            active_by_time,
            [[] for _ in active_by_time],
            (1, 2, 3),
            coordinate_paths,
            0.05,
            scheme="geometric",
        )
        self.assertTrue(closure["valid"], closure["failures"])
        self.assertTrue(all(row["projective_identity"] for row in closure["rows"]))

    def test_randomized_replay_is_deterministic_and_nested(self):
        rng = random.Random(980601)
        for case in range(100):
            d = rng.randint(1, 8)
            rounds = rng.randint(3, 20)
            rows = [
                [rng.uniform(-2.0, 2.0) for _ in range(d)]
                for _ in range(rounds)
            ]
            ledger = build_predictable_arrival_ledger(
                rows,
                [f"hypothesis-{i}" for i in range(d)],
                initial_active_count=1,
                min_observations=rng.randint(1, 4),
                mean_threshold=rng.uniform(-0.5, 0.5),
            )
            first = verify_predictable_arrival_ledger(ledger)
            second = verify_predictable_arrival_ledger(ledger)
            self.assertEqual(first, second)
            for before, after in zip(
                first["active_by_round"], first["active_by_round"][1:]
            ):
                self.assertTrue(set(before).issubset(set(after)))


if __name__ == "__main__":
    unittest.main()
