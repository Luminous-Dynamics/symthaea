#!/usr/bin/env python3
"""Independent known-answer validator for MAG-005C-001A.

This script intentionally imports no Symthaea production modules. It derives
scorecard semantics from the frozen synthetic facts and checks them against the
known-answer fields in the corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

CORPUS = Path("docs/release/evidence/mag-005c-001a-synthetic-scorecard-corpus-v1.json")
EXPECTED_SHA256 = "6d3f4ecaab7549509290849da4da636a7b88f0f6a68df3da0aa04765fd9361b1"
EXPECTED_GIT_BLOB_SHA1 = "e9253ea008f6d3c16e8c0970d8856b8e994e6ab3"
EXPECTED_SCHEMA = "mag-005c-001a-synthetic-scorecard-corpus-v1"
EXPECTED_ARCHITECTURE_HEAD = "d70bd8bf5ffba444b73d66c640028a7dfafd13c7"
EXPECTED_CASE_IDS = [f"MAG-005C-001A-{i:02d}" for i in range(1, 25)]
EXPECTED_VECTOR_KEYS = [
    "one_shot_protocol_valid",
    "adaptive_protocol_valid",
    "terminal_census_complete",
    "score_inputs_admissible",
    "calibrated_acquisition_supported",
    "contamination_control_supported",
    "independent_model_corroboration_supported",
    "prospective_credit_supported",
    "control_difference_established",
    "new_lineage_required",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def selected_slots(facts: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("top_k", "random"):
        values = facts.get(key)
        if isinstance(values, list):
            result.extend(str(v) for v in values)
    return result


def terminal_census_complete(facts: dict[str, Any]) -> bool:
    selected = selected_slots(facts)
    terminal = facts.get("terminal")
    if selected:
        if not isinstance(terminal, dict):
            return False
        return all(candidate in terminal for candidate in selected)
    return facts.get("terminal_census_complete") is True


def new_lineage_required(facts: dict[str, Any]) -> bool:
    metric_changed = (
        "original_metric_profile" in facts
        and "scored_metric_profile" in facts
        and facts["original_metric_profile"] != facts["scored_metric_profile"]
    )
    seed_changed = (
        facts.get("outcomes_revealed_before_change") is True
        and "committed_seed" in facts
        and "scored_seed" in facts
        and facts["committed_seed"] != facts["scored_seed"]
    )
    evidence_changed = False
    original = facts.get("original_evidence")
    scored = facts.get("scored_evidence")
    if isinstance(original, dict) and isinstance(scored, dict):
        evidence_changed = original.get("id") != scored.get("id")
    return metric_changed or seed_changed or evidence_changed


def one_shot_valid(facts: dict[str, Any]) -> bool:
    if facts.get("protocol") != "one_shot":
        return False
    if facts.get("commitment_before_answers") is not True:
        return False
    if facts.get("first_outcome_revealed") is True and facts.get("reranked_after_outcome"):
        return False
    if (
        facts.get("outcomes_revealed_before_change") is True
        and (
            (
                "original_metric_profile" in facts
                and "scored_metric_profile" in facts
                and facts["original_metric_profile"] != facts["scored_metric_profile"]
            )
            or (
                "committed_seed" in facts
                and "scored_seed" in facts
                and facts["committed_seed"] != facts["scored_seed"]
            )
        )
    ):
        return False
    if (
        "original_metric_profile" in facts
        and "scored_metric_profile" in facts
        and facts["original_metric_profile"] != facts["scored_metric_profile"]
    ):
        return False
    return True


def adaptive_valid(facts: dict[str, Any]) -> bool:
    if facts.get("protocol") != "adaptive":
        return False
    rounds = facts.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return False
    if facts.get("all_commitments_precede_corresponding_outcomes") is not True:
        return False
    for round_state in rounds:
        if not isinstance(round_state, dict):
            return False
        if not round_state.get("evidence_snapshot"):
            return False
        if not round_state.get("selection_commitment"):
            return False
        if "outcome" not in round_state:
            return False
    return True


def contamination_supported(facts: dict[str, Any]) -> bool:
    return (
        facts.get("contamination") == "clean"
        and facts.get("phase_a_expected_answer_reachable") is not True
    )


def calibrated_acquisition_supported(facts: dict[str, Any]) -> bool:
    if facts.get("acquisition_arm") != "uncertainty":
        return False
    calibration = facts.get("calibration_artifact")
    if not isinstance(calibration, dict):
        return False
    return (
        calibration.get("error_detection_validated") is True
        and calibration.get("task") == facts.get("requested_task")
        and calibration.get("domain") == facts.get("requested_domain")
    )


def independent_model_support(facts: dict[str, Any]) -> bool:
    models = facts.get("models")
    if not isinstance(models, list) or len(models) < 2:
        return False
    if facts.get("predictions_agree") is not True:
        return False
    if facts.get("declared_common_cause_overlap") != "none_observed_under_snapshot":
        return False
    families = [m.get("family") for m in models if isinstance(m, dict)]
    ancestries = [m.get("training_ancestry") for m in models if isinstance(m, dict)]
    return (
        len(families) == len(models)
        and len(ancestries) == len(models)
        and len(set(families)) == len(families)
        and len(set(ancestries)) == len(ancestries)
    )


def recompute_metrics(facts: dict[str, Any]) -> dict[str, float | None] | None:
    top_k = facts.get("top_k")
    random = facts.get("random")
    terminal = facts.get("terminal")
    if not (isinstance(top_k, list) and isinstance(random, list) and isinstance(terminal, dict)):
        return None
    if not top_k or not random:
        return None
    if not all(candidate in terminal for candidate in top_k + random):
        return None
    top_hits = sum(terminal[candidate] == "hit" for candidate in top_k)
    random_hits = sum(terminal[candidate] == "hit" for candidate in random)
    top_precision = top_hits / len(top_k)
    random_precision = random_hits / len(random)
    enrichment = top_precision / random_precision if random_precision > 0 else None
    return {
        "top_k_precision": top_precision,
        "random_precision": random_precision,
        "enrichment_ratio": enrichment,
    }


def metrics_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            if a is not b:
                return False
        elif not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def stored_summary_consistent(facts: dict[str, Any]) -> bool:
    stored = facts.get("stored_summary")
    if stored is None:
        return True
    if not isinstance(stored, dict):
        return False
    derived = recompute_metrics(facts)
    if derived is None:
        return False
    comparable = {k: derived[k] for k in stored if k in derived}
    return len(comparable) == len(stored) and metrics_equal(stored, comparable)


def universe_inputs_admissible(facts: dict[str, Any]) -> bool:
    universe = facts.get("candidate_universe")
    if not isinstance(universe, list):
        return True
    universe_set = set(universe)
    for candidate in selected_slots(facts):
        if candidate not in universe_set:
            return False
    scored = facts.get("scored_candidates")
    if isinstance(scored, list) and any(candidate not in universe_set for candidate in scored):
        return False
    terminal = facts.get("terminal")
    if isinstance(terminal, dict) and any(candidate not in universe_set for candidate in terminal):
        return False
    return True


def derive_flags(facts: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    census_ok = terminal_census_complete(facts)
    lineage_change = new_lineage_required(facts)
    one_shot_ok = one_shot_valid(facts)
    adaptive_ok = adaptive_valid(facts)
    contamination_ok = contamination_supported(facts)

    if one_shot_ok:
        flags.add("one_shot_protocol_valid")
    if adaptive_ok:
        flags.add("adaptive_protocol_valid")
    if census_ok:
        flags.add("terminal_census_complete")
    if calibrated_acquisition_supported(facts):
        flags.add("calibrated_acquisition_supported")
    if contamination_ok:
        flags.add("contamination_control_supported")
    if independent_model_support(facts):
        flags.add("independent_model_corroboration_supported")
    if lineage_change:
        flags.add("new_lineage_required")

    protocol_ok = one_shot_ok or adaptive_ok
    inputs_ok = (
        protocol_ok
        and census_ok
        and contamination_ok
        and not lineage_change
        and universe_inputs_admissible(facts)
        and stored_summary_consistent(facts)
    )
    if inputs_ok:
        flags.add("score_inputs_admissible")
        flags.add("prospective_credit_supported")

    if facts.get("control_difference_evidence") == "established":
        flags.add("control_difference_established")
    return flags


def main() -> int:
    data = CORPUS.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    blob = git_blob_sha1(data)
    if sha256 != EXPECTED_SHA256:
        fail(f"corpus SHA-256 mismatch: {sha256}")
    if blob != EXPECTED_GIT_BLOB_SHA1:
        fail(f"corpus Git blob mismatch: {blob}")

    corpus = json.loads(data)
    if corpus.get("schema") != EXPECTED_SCHEMA:
        fail("schema mismatch")
    if corpus.get("authority") != "representation_only_no_scientific_authority":
        fail("authority mismatch")
    if corpus.get("architecture_head") != EXPECTED_ARCHITECTURE_HEAD:
        fail("architecture head mismatch")
    if corpus.get("vector_keys") != EXPECTED_VECTOR_KEYS:
        fail("vector key order/content mismatch")

    cases = corpus.get("cases")
    if not isinstance(cases, list):
        fail("cases must be a list")
    ids = [case.get("id") for case in cases]
    if ids != EXPECTED_CASE_IDS:
        fail(f"case ID/order mismatch: {ids}")

    for case in cases:
        facts = case.get("facts")
        expected = case.get("expected_true")
        limits = case.get("limits")
        if not isinstance(facts, dict):
            fail(f"{case['id']}: facts must be an object")
        if not isinstance(expected, list) or len(expected) != len(set(expected)):
            fail(f"{case['id']}: expected_true must be a unique list")
        if any(flag not in EXPECTED_VECTOR_KEYS for flag in expected):
            fail(f"{case['id']}: unknown expected flag")
        if not isinstance(limits, list) or not limits:
            fail(f"{case['id']}: non-empty limits required")

        derived = derive_flags(facts)
        if derived != set(expected):
            fail(
                f"{case['id']}: flags mismatch "
                f"derived={sorted(derived)} expected={sorted(expected)}"
            )

        expected_metrics = case.get("expected_metrics")
        if expected_metrics is not None:
            derived_metrics = recompute_metrics(facts)
            if derived_metrics is None:
                fail(f"{case['id']}: expected metrics but cannot recompute")
            if not metrics_equal(derived_metrics, expected_metrics):
                fail(
                    f"{case['id']}: metric mismatch "
                    f"derived={derived_metrics} expected={expected_metrics}"
                )

    print(f"ok cases={len(cases)} sha256={sha256} git_blob={blob}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
