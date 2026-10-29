#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from forecast_event_journal import (
    GENESIS_SHA256, audit_event_journal, main, seal_event,
)
from forecast_validation import make_outcome_record, forecast_sha256

MODEL = "a" * 64
STATE = "b" * 64


def forecast(forecast_id, probabilities=(0.7, 0.3), regime="nominal"):
    return {
        "forecast_id": forecast_id, "task_id": forecast_id, "regime": regime,
        "model_sha256": MODEL, "state_sha256": STATE,
        "probabilities": list(probabilities),
    }


def journal_for(forecasts, outcomes, publication_times=None, observation_times=None):
    publication_times = publication_times or [f"2026-10-29T08:00:{i:02d}Z" for i in range(len(forecasts))]
    observation_times = observation_times or [f"2026-10-29T09:00:{i:02d}Z" for i in range(len(outcomes))]
    events = []
    publications = {}
    prior = GENESIS_SHA256
    for index, item in enumerate(forecasts):
        event = seal_event({
            "sequence": index, "event_id": f"publish:{item['forecast_id']}",
            "kind": "forecast_published", "forecast_id": item["forecast_id"],
            "occurred_at": publication_times[index],
            "artifact_sha256": forecast_sha256(item),
            "previous_event_sha256": prior,
        })
        events.append(event)
        publications[item["forecast_id"]] = event
        prior = event["event_sha256"]
    for index, outcome in enumerate(outcomes):
        forecast_id = outcome["forecast_id"]
        event = seal_event({
            "sequence": len(events), "event_id": f"observe:{forecast_id}",
            "kind": "outcome_observed", "forecast_id": forecast_id,
            "occurred_at": observation_times[index],
            "artifact_sha256": outcome["outcome_sha256"],
            "forecast_event_sha256": publications[forecast_id]["event_sha256"],
            "previous_event_sha256": prior,
        })
        events.append(event)
        prior = event["event_sha256"]
    return events


def records(ids=("f1",)):
    forecasts = [forecast(item) for item in ids]
    outcomes = [make_outcome_record(item, 0) for item in forecasts]
    return forecasts, outcomes


class ForecastEventJournalTests(unittest.TestCase):
    def test_valid_journal_binds_artifacts_and_temporal_order(self):
        forecasts, outcomes = records(("f1", "f2"))
        events = journal_for(forecasts, outcomes)
        result = audit_event_journal(forecasts, outcomes, events)
        self.assertEqual(result["forecast_count"], 2)
        self.assertEqual(result["outcome_count"], 2)
        self.assertEqual(result["event_count"], 4)
        self.assertEqual(result["journal_head_sha256"], events[-1]["event_sha256"])
        self.assertIn("does not establish truthful event times", result["claim_ceiling"])

    def test_rejects_event_byte_tampering(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        events[0]["occurred_at"] = "2026-10-29T08:10:00Z"
        with self.assertRaisesRegex(ValueError, "event digest mismatch"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_reordering_and_sequence_gaps(self):
        forecasts, outcomes = records(("f1", "f2"))
        events = journal_for(forecasts, outcomes)
        with self.assertRaisesRegex(ValueError, "sequence must be contiguous"):
            audit_event_journal(forecasts, outcomes, [events[1], events[0], *events[2:]])
        events = journal_for(forecasts, outcomes)
        events[1]["sequence"] = 7
        events[1] = seal_event(events[1])
        with self.assertRaisesRegex(ValueError, "sequence must be contiguous"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_outcome_that_predates_publication_even_when_resealed(self):
        forecasts, outcomes = records()
        events = journal_for(
            forecasts, outcomes,
            publication_times=["2026-10-29T08:00:10Z"],
            observation_times=["2026-10-29T08:00:09Z"],
        )
        # The event records have valid self-hashes; chronology must still fail.
        with self.assertRaisesRegex(ValueError, "event timestamps decrease"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_equal_forecast_and_outcome_times(self):
        forecasts, outcomes = records()
        events = journal_for(
            forecasts, outcomes,
            publication_times=["2026-10-29T08:00:00Z"],
            observation_times=["2026-10-29T08:00:00Z"],
        )
        with self.assertRaisesRegex(ValueError, "strictly later"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_wrong_forecast_artifact_digest(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        events[0]["artifact_sha256"] = "c" * 64
        events[0] = seal_event(events[0])
        events[1]["previous_event_sha256"] = events[0]["event_sha256"]
        events[1] = seal_event(events[1])
        with self.assertRaisesRegex(ValueError, "publication artifact digest mismatch"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_wrong_outcome_artifact_digest(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        events[-1]["artifact_sha256"] = "c" * 64
        events[-1] = seal_event(events[-1])
        with self.assertRaisesRegex(ValueError, "observation artifact digest mismatch"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_bad_forecast_publication_link(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        events[-1]["forecast_event_sha256"] = "c" * 64
        events[-1] = seal_event(events[-1])
        with self.assertRaisesRegex(ValueError, "exact publication event"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_missing_events_and_duplicate_event_ids(self):
        forecasts, outcomes = records(("f1", "f2"))
        incomplete = journal_for(forecasts[1:], outcomes[1:])
        with self.assertRaisesRegex(ValueError, "publication event coverage mismatch"):
            audit_event_journal(forecasts, outcomes, incomplete)
        events = journal_for(forecasts, outcomes)
        events[1]["event_id"] = events[0]["event_id"]
        events[1] = seal_event(events[1])
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            audit_event_journal(forecasts, outcomes, events)

    def test_rejects_timezone_ambiguity(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        events[0]["occurred_at"] = "2026-10-29T08:00:00+02:00"
        events[0] = seal_event(events[0])
        events[1]["previous_event_sha256"] = events[0]["event_sha256"]
        events[1] = seal_event(events[1])
        with self.assertRaisesRegex(ValueError, "UTC timestamp ending in Z"):
            audit_event_journal(forecasts, outcomes, events)

    def test_cli_binds_raw_manifests_and_evaluator_digest(self):
        forecasts, outcomes = records()
        events = journal_for(forecasts, outcomes)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("forecasts.json", "outcomes.json", "events.json")]
            for path, data in zip(paths, (forecasts, outcomes, events), strict=True):
                path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            raw_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            output = root / "receipt.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([*(str(path) for path in paths), "--output", str(output)])
            self.assertEqual(code, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["receipt_label"], "FORECAST_EVENT_JOURNAL_VALIDATION_NOT_TRUSTED_TIMESTAMPING")
            source = receipt["source_artifacts"]
            self.assertEqual(source["forecast_manifest_sha256"], raw_hashes[0])
            self.assertEqual(source["outcome_manifest_sha256"], raw_hashes[1])
            self.assertEqual(source["event_journal_sha256"], raw_hashes[2])
            self.assertEqual(len(receipt["evaluator_sha256"]), 64)

    def test_cli_rejects_duplicate_json_keys_and_does_not_emit_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast_path, outcome_path, event_path, output = [
                root / name for name in ("f.json", "o.json", "e.json", "r.json")
            ]
            forecast_path.write_text('[{"forecast_id":"a","forecast_id":"b"}]', encoding="utf-8")
            outcome_path.write_text("[]", encoding="utf-8")
            event_path.write_text("[]", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(forecast_path), str(outcome_path), str(event_path), "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("duplicate JSON key", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_refuses_to_overwrite_input_or_alias_input_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecasts, outcomes = records()
            events = journal_for(forecasts, outcomes)
            paths = [root / name for name in ("f.json", "o.json", "e.json")]
            for path, data in zip(paths, (forecasts, outcomes, events), strict=True):
                path.write_text(json.dumps(data), encoding="utf-8")
            original = paths[0].read_bytes()
            self.assertEqual(main([*(str(path) for path in paths), "--output", str(paths[0])]), 2)
            self.assertEqual(paths[0].read_bytes(), original)
            self.assertEqual(main([str(paths[0]), str(paths[0]), str(paths[2]), "--output", str(root / "r.json")]), 2)


if __name__ == "__main__":
    unittest.main()
