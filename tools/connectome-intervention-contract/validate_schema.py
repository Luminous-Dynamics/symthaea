#!/usr/bin/env python3
"""CI-only Draft 2020-12 schema conformance checks for the intervention manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_MANIFEST.schema.json"
EXAMPLE_PATH = ROOT / "docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_EXAMPLE_V1.json"


def rejected(validator: Draft202012Validator, instance: dict, label: str) -> None:
    errors = list(validator.iter_errors(instance))
    if not errors:
        raise AssertionError(f"negative schema control unexpectedly passed: {label}")


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    # The public example is intentionally incomplete and should remain valid
    # as a planning manifest, without being promoted to a completed result.
    validator.validate(example)
    assert example["run_state"] == "incomplete"
    assert example["results"]["status"] == "not_run"

    wrong_claim = copy.deepcopy(example)
    wrong_claim["claim_ceiling"] = "observational_prediction_only"
    rejected(validator, wrong_claim, "evidence tier / claim ceiling mismatch")

    unknown_property = copy.deepcopy(example)
    unknown_property["unreviewed_assertion"] = "pass"
    rejected(validator, unknown_property, "unknown top-level property")

    forged_complete = copy.deepcopy(example)
    forged_complete["run_state"] = "complete"
    forged_complete["results"]["status"] = "complete"
    rejected(validator, forged_complete, "complete run with null source/split/result/receipt evidence")

    measured_without_permission = copy.deepcopy(example)
    measured_without_permission["evidence_tier"] = "measured_intervention"
    measured_without_permission["claim_ceiling"] = "measured_intervention_prediction"
    measured_without_permission["source"]["source_kind"] = "measured_intervention"
    measured_without_permission["source"]["rights_state"] = "unknown"
    rejected(validator, measured_without_permission, "measured intervention without permission receipt")

    print(
        "OK: Draft 2020-12 schema is valid; the incomplete planning example passes; "
        "four adversarial mutations are rejected."
    )


if __name__ == "__main__":
    main()
