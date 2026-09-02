#!/usr/bin/env python3
from __future__ import annotations

import verify_vart_world_creative_001_qualified as q

rows_only = {
    "trials": [
        {"trial_id": "T1"},
        {"trial_id": "T2"},
    ]
}
assert q.expected_trial_ids(rows_only) == ["T1", "T2"]

both = {
    "trial_ids": ["T2", "T1"],
    "trials": [
        {"trial_id": "T1"},
        {"trial_id": "T2"},
    ],
}
assert q.expected_trial_ids(both) == ["T2", "T1"]

bad = {
    "trial_ids": ["T1", "T3"],
    "trials": [
        {"trial_id": "T1"},
        {"trial_id": "T2"},
    ],
}
try:
    q.expected_trial_ids(bad)
except q.Reject as exc:
    assert exc.code == "PREREGISTRATION_INVALID"
else:
    raise AssertionError("mismatched trial_ids/trials representations unexpectedly passed")

print("PASS: v3 trial-row membership + legacy trial_ids compatibility + mismatch rejection")
