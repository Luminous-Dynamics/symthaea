"""Tests for dependency cascade, absence, and revision currentness semantics."""
from __future__ import annotations

import unittest
from pathlib import Path

from dependency_cascade import evaluate_dataset, evaluate_snapshot
from eligibility_filter import read_jsonl_bytes

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mutations" / "dependency_snapshots.jsonl"


def claim(
    claim_id: str,
    revision: int,
    value: str,
    dependencies: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "revision": revision,
        "value": value,
        "dependencies": dependencies or [],
    }


def dep(claim_id: str, revision: int) -> dict[str, object]:
    return {"claim_id": claim_id, "revision": revision}


def by_key(report: dict, claim_id: str, revision: int | None = None) -> dict:
    matches = [
        row for row in report["evaluations"]
        if row["claim_id"] == claim_id and (revision is None or row["revision"] == revision)
    ]
    if revision is None:
        matches = [row for row in matches if row["is_latest_revision"]]
    if len(matches) != 1:
        raise AssertionError(f"expected one {claim_id}@{revision}, got {matches}")
    return matches[0]


class DependencyCascadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshots, _ = read_jsonl_bytes(FIXTURE)
        cls.reports = {r["snapshot_id"]: r for r in evaluate_dataset(cls.snapshots)["snapshots"]}

    def test_fixture_has_seven_distinct_lifecycle_snapshots(self) -> None:
        self.assertEqual(
            list(self.reports),
            [
                "S0_initial",
                "S1_upstream_updated",
                "S2_service_refreshed_worker_stale",
                "S3_dependency_chain_refreshed",
                "S4_dependency_value_reverts_without_refresh",
                "S5_service_refreshed_after_reversion_worker_stale",
                "S6_dependency_chain_refreshed_after_reversion",
            ],
        )

    def test_initial_dependency_chain_is_current(self) -> None:
        report = self.reports["S0_initial"]
        for claim_id in ("db_endpoint", "service_dsn", "worker_dsn", "alert_threshold"):
            with self.subTest(claim_id=claim_id):
                self.assertEqual(by_key(report, claim_id)["status"], "current")

    def test_upstream_change_invalidates_direct_and_transitive_dependents(self) -> None:
        report = self.reports["S1_upstream_updated"]
        self.assertEqual(by_key(report, "db_endpoint")["value"], "db-primary-b.internal")
        self.assertEqual(by_key(report, "service_dsn")["status"], "stale_dependency")
        self.assertEqual(by_key(report, "worker_dsn")["status"], "stale_dependency")
        self.assertEqual(by_key(report, "alert_threshold")["status"], "current")

    def test_stale_derived_values_are_not_returned_as_current(self) -> None:
        report = self.reports["S1_upstream_updated"]
        current_ids = {row["claim_id"] for row in report["current_claims"]}
        self.assertEqual(current_ids, {"db_endpoint", "alert_threshold"})
        self.assertNotIn("service_dsn", current_ids)
        self.assertNotIn("worker_dsn", current_ids)
        absence = {row["claim_id"]: row["status"] for row in report["absence_claims"]}
        self.assertEqual(absence["service_dsn"], "stale_dependency")
        self.assertEqual(absence["worker_dsn"], "stale_dependency")

    def test_refreshing_one_dependency_does_not_resurrect_downstream_stale_value(self) -> None:
        report = self.reports["S2_service_refreshed_worker_stale"]
        self.assertEqual(by_key(report, "service_dsn")["status"], "current")
        self.assertEqual(by_key(report, "worker_dsn")["status"], "stale_dependency")
        self.assertNotIn(
            "worker_dsn",
            {row["claim_id"] for row in report["current_claims"]},
        )

    def test_refreshing_entire_dependency_chain_restores_currentness(self) -> None:
        report = self.reports["S3_dependency_chain_refreshed"]
        for claim_id in ("db_endpoint", "service_dsn", "worker_dsn", "alert_threshold"):
            with self.subTest(claim_id=claim_id):
                self.assertEqual(by_key(report, claim_id)["status"], "current")
        self.assertEqual(report["absence_claims"], [])

    def test_old_revision_remains_auditable_but_is_not_latest(self) -> None:
        report = self.reports["S1_upstream_updated"]
        old = by_key(report, "db_endpoint", 1)
        self.assertEqual(old["status"], "historical")
        self.assertFalse(old["is_latest_revision"])
        self.assertEqual(by_key(report, "db_endpoint", 2)["status"], "current")

    def test_unknown_dependency_fails_closed_and_is_explicitly_absent(self) -> None:
        report = evaluate_snapshot({
            "snapshot_id": "unknown-dep",
            "claims": [claim("service_dsn", 1, "dsn://unknown", [dep("db_endpoint", 7)])],
        })
        self.assertEqual(by_key(report, "service_dsn")["status"], "unknown_dependency")
        self.assertEqual(
            report["current_claims"],
            [],
        )
        self.assertEqual(report["absence_claims"][0]["claim_id"], "service_dsn")

    def test_aba_value_reversion_does_not_resurrect_old_dependencies(self) -> None:
        report = self.reports["S4_dependency_value_reverts_without_refresh"]
        endpoint = by_key(report, "db_endpoint")
        service = by_key(report, "service_dsn")
        worker = by_key(report, "worker_dsn")
        self.assertEqual(endpoint["value"], "db-primary-a.internal")
        self.assertEqual(endpoint["status"], "current")
        self.assertEqual(service["value"], "postgres://db-primary-b.internal/service")
        self.assertEqual(service["status"], "stale_dependency")
        self.assertEqual(worker["status"], "stale_dependency")
        self.assertNotIn(
            "service_dsn",
            {row["claim_id"] for row in report["current_claims"]},
        )
        self.assertNotIn(
            "worker_dsn",
            {row["claim_id"] for row in report["current_claims"]},
        )

    def test_aba_partial_refresh_still_leaves_transitive_worker_stale(self) -> None:
        report = self.reports["S5_service_refreshed_after_reversion_worker_stale"]
        self.assertEqual(by_key(report, "db_endpoint")["value"], "db-primary-a.internal")
        self.assertEqual(by_key(report, "service_dsn")["value"], "postgres://db-primary-a.internal/service")
        self.assertEqual(by_key(report, "service_dsn")["status"], "current")
        self.assertEqual(by_key(report, "worker_dsn")["status"], "stale_dependency")
        self.assertNotIn(
            "worker_dsn",
            {row["claim_id"] for row in report["current_claims"]},
        )

    def test_same_value_is_not_enough_without_current_dependency_revision(self) -> None:
        report = self.reports["S6_dependency_chain_refreshed_after_reversion"]
        old_worker = by_key(report, "worker_dsn", 1)
        current_worker = by_key(report, "worker_dsn", 3)
        self.assertEqual(old_worker["value"], current_worker["value"])
        self.assertEqual(old_worker["status"], "historical")
        self.assertFalse(old_worker["is_latest_revision"])
        self.assertEqual(current_worker["status"], "current")
        self.assertEqual(
            {row["claim_id"] for row in report["current_claims"]},
            {"db_endpoint", "service_dsn", "worker_dsn", "alert_threshold"},
        )

    def test_deep_dependency_chain_does_not_depend_on_python_recursion_limit(self) -> None:
        depth = 1500
        claims = []
        for index in range(depth):
            dependencies = [] if index == 0 else [dep(f"node-{index - 1:04d}", 1)]
            claims.append(claim(f"node-{index:04d}", 1, f"value-{index}", dependencies))
        report = evaluate_snapshot({"snapshot_id": "deep-chain", "claims": claims})
        self.assertEqual(report["latest_claim_count"], depth)
        self.assertEqual(report["state_counts"]["current"], depth)
        self.assertEqual(len(report["current_claims"]), depth)
        self.assertEqual(report["absence_claims"], [])

    def test_claim_input_order_does_not_change_revision_statuses(self) -> None:
        original = {
            "snapshot_id": "order-invariance",
            "claims": [
                claim("root", 1, "R"),
                claim("left", 1, "L", [dep("root", 1)]),
                claim("right", 1, "X", [dep("root", 1)]),
                claim("leaf", 1, "Y", [dep("left", 1), dep("right", 1)]),
            ],
        }
        reversed_snapshot = {
            "snapshot_id": original["snapshot_id"],
            "claims": list(reversed(original["claims"])),
        }
        a = evaluate_snapshot(original)
        b = evaluate_snapshot(reversed_snapshot)
        key = lambda report: {
            (row["claim_id"], row["revision"]): row["status"]
            for row in report["evaluations"]
        }
        self.assertEqual(key(a), key(b))
        self.assertEqual(
            {row["claim_id"] for row in a["current_claims"]},
            {row["claim_id"] for row in b["current_claims"]},
        )

    def test_duplicate_claim_revision_fails_closed(self) -> None:
        snapshot = {
            "snapshot_id": "duplicate",
            "claims": [claim("db_endpoint", 1, "db-a"), claim("db_endpoint", 1, "db-b")],
        }
        with self.assertRaisesRegex(ValueError, "duplicate claim revision"):
            evaluate_snapshot(snapshot)

    def test_dependency_cycle_fails_closed(self) -> None:
        snapshot = {
            "snapshot_id": "cycle",
            "claims": [
                claim("a", 1, "A", [dep("b", 1)]),
                claim("b", 1, "B", [dep("a", 1)]),
            ],
        }
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            evaluate_snapshot(snapshot)

    def test_boolean_revision_is_not_accepted_as_integer_revision(self) -> None:
        snapshot = {
            "snapshot_id": "bool-revision",
            "claims": [claim("db_endpoint", True, "db-a")],
        }
        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_snapshot(snapshot)

    def test_duplicate_snapshot_ids_fail_closed(self) -> None:
        snapshot = {"snapshot_id": "S0", "claims": [claim("a", 1, "a")]}
        with self.assertRaisesRegex(ValueError, "snapshot IDs must be unique"):
            evaluate_dataset([snapshot, snapshot])


if __name__ == "__main__":
    unittest.main()
