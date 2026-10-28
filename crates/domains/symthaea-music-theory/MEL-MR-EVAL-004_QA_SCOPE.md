{
  "status": "PASS",
  "dataset_version": "MEL-EDIT-LOCALITY-SMOKE-1.0",
  "case_count": 132,
  "expected_verdicts": {
    "accept": 60,
    "reject": 72
  },
  "requested_success": 120,
  "protected_pass": 72,
  "families": 12,
  "per_family_cases": 11,
  "split_cases": {
    "development": 66,
    "calibration": 22,
    "test": 44
  },
  "split_families": {
    "development": 6,
    "calibration": 2,
    "test": 4
  },
  "family_isolation": true,
  "fixture_scope": "structural and oracle consistency only; no Rust execution or human evaluation"
}
