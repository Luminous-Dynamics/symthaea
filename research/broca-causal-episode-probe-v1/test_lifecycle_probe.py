"""Tests for the explicit supersession worker-probe assertions."""
from __future__ import annotations

import hashlib
import unittest

from run_lifecycle_probe import assert_transition_result


def qresult(ids: list[str]) -> dict[str, object]:
    return {
        "audit": [
            {"evidence_id": item, "content_sha256": hashlib.sha256(item.encode("utf-8")).hexdigest()}
            for item in ids
        ],
        "context": [{"type": "text", "value": item} for item in ids],
    }


class LifecycleProbeTests(unittest.TestCase):
    def test_supersession_succeeds_when_old_is_gone_and_new_remains(self) -> None:
        assert_transition_result(
            "old-id",
            "new-id",
            {
                "lifecycle": "superseded",
                "old_evidence_id": "old-id",
                "new_evidence_id": "new-id",
            },
            qresult(["new-id", "other-id"]),
            qresult(["new-id"]),
        )

    def test_retired_evidence_still_returned_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "retired evidence remained"):
            assert_transition_result(
                "old-id",
                "new-id",
                {"lifecycle": "superseded", "old_evidence_id": "old-id", "new_evidence_id": "new-id"},
                qresult(["old-id"]),
                qresult(["new-id"]),
            )

    def test_missing_replacement_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "replacement evidence is not retrievable"):
            assert_transition_result(
                "old-id",
                "new-id",
                {"lifecycle": "superseded", "old_evidence_id": "old-id", "new_evidence_id": "new-id"},
                qresult(["other-id"]),
                qresult(["other-id"]),
            )

    def test_wrong_transition_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong old evidence ID"):
            assert_transition_result(
                "old-id",
                "new-id",
                {"lifecycle": "superseded", "old_evidence_id": "some-other-id", "new_evidence_id": "new-id"},
                qresult([]),
                qresult(["new-id"]),
            )

    def test_context_audit_cardinality_mismatch_fails_closed(self) -> None:
        bad_context = {"audit": [{"evidence_id": "new-id"}], "context": []}
        with self.assertRaisesRegex(ValueError, "cardinalities disagree"):
            assert_transition_result(
                "old-id",
                "new-id",
                {"lifecycle": "superseded", "old_evidence_id": "old-id", "new_evidence_id": "new-id"},
                bad_context,
                qresult(["new-id"]),
            )

    def test_context_digest_mismatch_fails_closed(self) -> None:
        bad = {
            "audit": [{"evidence_id": "new-id", "content_sha256": "0" * 64}],
            "context": [{"type": "text", "value": "altered"}],
        }
        with self.assertRaisesRegex(ValueError, "does not match its audit digest"):
            assert_transition_result(
                "old-id",
                "new-id",
                {"lifecycle": "superseded", "old_evidence_id": "old-id", "new_evidence_id": "new-id"},
                qresult([]),
                bad,
            )


if __name__ == "__main__":
    unittest.main()
