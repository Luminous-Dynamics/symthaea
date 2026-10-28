#!/usr/bin/env python3
"""Paired causal-replay effect benchmark for PE interventions.

Reference research harness only: consumes recorded run summaries and does not execute
production Symthaea Rust. Statistical efficacy on generated fixtures is not a product claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean, stdev

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def digest(value: object) -> str:
    b = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(b"symthaea.pev.causal-replay.v1\x00" + b).hexdigest()


def valid_digest(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def make_pair(index: int, arm: str, effect: float, rng: random.Random) -> dict:
    pair_id = f"{arm}-{index:05d}"
    decision_id = f"decision-{arm}-{index:05d}"
    scenario = {"arm": arm, "index": index, "observation": rng.randrange(0, 1_000_000),
                "action": rng.randrange(0, 16), "seed": rng.randrange(0, 2**32)}
    scenario_digest = digest(scenario)
    initial_digest = digest({"initial_state": scenario_digest, "latent": rng.randrange(0, 2**32)})
    prefix_digest = digest({"decision": decision_id, "prefix": scenario_digest})
    tape_digest = digest({"exogenous_randomness_tape": scenario_digest, "key": rng.randrange(0, 2**32)})
    calibration_digest = digest({"external_outcome_calibration": rng.randrange(0, 2**32)})
    y0 = round(0.30 + 0.25 * rng.random(), 8)
    delta = effect + rng.gauss(0.0, 0.04)
    y1 = round(y0 + delta, 8)
    if not 0.0 <= y1 <= 1.0:
        raise AssertionError("synthetic outcome outside [0,1]; adjust DGP rather than clip")
    intervention = "valid_pe" if arm == "valid_pe" else "invalid_pe_payload"
    invalid_payloads = [None, 0.0, 1.0, -1.0, "NaN", "+Infinity", "-Infinity"]
    invalid_payload = invalid_payloads[index % len(invalid_payloads)] if intervention == "invalid_pe_payload" else None
    common = {"decision_id": decision_id, "scenario_digest": scenario_digest,
              "initial_state_digest": initial_digest, "pre_intervention_state_digest": prefix_digest,
              "randomness_tape_digest": tape_digest, "outcome_calibration_digest": calibration_digest,
              "prefix_replay_verified": True}
    control = {**common, "run_id": f"run-{pair_id}-control", "branch": "no_pe",
               "intervention": "no_pe", "outcome_score": y0, "execution_path_digest": digest({"path": "control", "pair": pair_id}),
               "evidence_id": None, "pe_transition_count": 0, "compatibility_payload": None}
    treatment = {**common, "run_id": f"run-{pair_id}-treatment", "branch": "treatment",
                 "intervention": intervention, "outcome_score": y1,
                 "execution_path_digest": digest({"path": intervention, "pair": pair_id}),
                 "evidence_id": f"evidence-{pair_id}" if intervention == "valid_pe" else None,
                 "pe_transition_count": 1 if intervention == "valid_pe" else 0,
                 "compatibility_payload": round(effect, 8) if intervention == "valid_pe" else invalid_payload,
                 "pe_value": round(effect, 8) if intervention == "valid_pe" else None}
    return {"pair_id": pair_id, "expected_intervention": intervention,
            "control": control, "treatment": treatment}


def audit_pair(pair: dict) -> dict:
    errors = []
    if not isinstance(pair, dict):
        return {"accepted": False, "errors": ["PAIR_NOT_OBJECT"]}
    pid = pair.get("pair_id")
    if not isinstance(pid, str) or not pid:
        errors.append("PAIR_ID_MISSING")
    expected = pair.get("expected_intervention")
    if not isinstance(expected, str) or expected not in {"valid_pe", "invalid_pe_payload"}:
        errors.append("EXPECTED_INTERVENTION_INVALID")
        expected = None
    control, treatment = pair.get("control"), pair.get("treatment")
    if not isinstance(control, dict) or not isinstance(treatment, dict):
        return {"accepted": False, "pair_id": pid, "errors": errors + ["PAIR_BRANCH_MISSING"]}
    shared = ("decision_id", "scenario_digest", "initial_state_digest", "pre_intervention_state_digest",
              "randomness_tape_digest", "outcome_calibration_digest")
    for field in shared:
        a, b = control.get(field), treatment.get(field)
        if field == "decision_id":
            if not isinstance(a, str) or not a or a != b: errors.append("DECISION_ID_MISMATCH")
        else:
            if not valid_digest(a) or not valid_digest(b): errors.append(f"{field.upper()}_INVALID")
            if a != b: errors.append(f"PAIRED_{field.upper()}_MISMATCH")
    if control.get("branch") != "no_pe" or control.get("intervention") != "no_pe":
        errors.append("CONTROL_ROLE_MISMATCH")
    if treatment.get("branch") != "treatment": errors.append("TREATMENT_BRANCH_LABEL_MISMATCH")
    if treatment.get("intervention") != expected: errors.append("TREATMENT_INTERVENTION_MISMATCH")
    for label, run in (("control", control), ("treatment", treatment)):
        if not isinstance(run.get("run_id"), str) or not run["run_id"]:
            errors.append(f"{label.upper()}_RUN_ID_MISSING")
        score = run.get("outcome_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            errors.append(f"{label.upper()}_OUTCOME_INVALID")
        if run.get("prefix_replay_verified") is not True:
            errors.append(f"{label.upper()}_PREFIX_REPLAY_NOT_VERIFIED")
    control_count = control.get("pe_transition_count")
    if (control.get("evidence_id") is not None or not isinstance(control_count, int)
            or isinstance(control_count, bool) or control_count != 0):
        errors.append("CONTROL_CONTAINS_PE_INFLUENCE")
    if expected == "valid_pe":
        if not isinstance(treatment.get("evidence_id"), str) or not treatment.get("evidence_id"):
            errors.append("VALID_PE_EVIDENCE_ID_MISSING")
        treatment_count = treatment.get("pe_transition_count")
        if not isinstance(treatment_count, int) or isinstance(treatment_count, bool) or treatment_count != 1:
            errors.append("VALID_PE_ADMISSION_NOT_EXACTLY_ONCE")
        pv, cv = treatment.get("pe_value"), treatment.get("compatibility_payload")
        if not isinstance(pv, (int, float)) or isinstance(pv, bool) or not math.isfinite(float(pv)):
            errors.append("VALID_PE_VALUE_INVALID")
        if not isinstance(cv, (int, float)) or isinstance(cv, bool) or not math.isfinite(float(cv)) or cv != pv:
            errors.append("VALID_PE_COMPATIBILITY_MISMATCH")
    elif expected == "invalid_pe_payload":
        treatment_count = treatment.get("pe_transition_count")
        if (treatment.get("evidence_id") is not None or not isinstance(treatment_count, int)
                or isinstance(treatment_count, bool) or treatment_count != 0):
            errors.append("INVALID_PE_ADMISSION")
        if treatment.get("pe_value") is not None:
            errors.append("INVALID_PE_TYPED_VALUE_PRESENT")
        if payload_key(treatment.get("compatibility_payload")) not in {payload_key(None), payload_key(0.0), payload_key(1.0), payload_key(-1.0), payload_key("NaN"), payload_key("+Infinity"), payload_key("-Infinity")}:
            errors.append("INVALID_PE_PAYLOAD_UNEXPECTED")
    return {"accepted": not errors, "pair_id": pid, "errors": errors,
            "effect": float(treatment.get("outcome_score", 0.0)) - float(control.get("outcome_score", 0.0))
            if isinstance(treatment.get("outcome_score"), (int, float)) and isinstance(control.get("outcome_score"), (int, float)) else None}


def payload_key(v: object) -> tuple:
    if v is None: return ("null", None)
    if isinstance(v, bool): return ("bool", v)
    if isinstance(v, float): return ("float", v.hex())
    if isinstance(v, int): return ("int", v)
    if isinstance(v, str): return ("str", v)
    return (type(v).__name__, repr(v))


def audit_panel(pairs: list) -> dict:
    if not isinstance(pairs, list) or not pairs:
        return {"accepted": False, "status": "FAIL", "errors": ["PAIR_PANEL_EMPTY_OR_INVALID"], "pairs": 0}
    errors, pair_ids, run_ids = [], set(), set()
    arm_effects = {"valid_pe": [], "invalid_pe_payload": []}
    for i, pair in enumerate(pairs):
        result = audit_pair(pair)
        if not result["accepted"]: errors.extend(f"pair[{i}] {x}" for x in result["errors"])
        if not isinstance(pair, dict): continue
        pid = pair.get("pair_id")
        if isinstance(pid, str):
            if pid in pair_ids: errors.append(f"PAIR_ID_DUPLICATE:{pid}")
            pair_ids.add(pid)
        for key in ("control", "treatment"):
            run = pair.get(key)
            if not isinstance(run, dict): continue
            rid = run.get("run_id")
            if isinstance(rid, str):
                if rid in run_ids: errors.append(f"RUN_ID_DUPLICATE:{rid}")
                run_ids.add(rid)
        arm = pair.get("expected_intervention")
        if isinstance(arm, str) and arm in arm_effects and result.get("effect") is not None:
            arm_effects[arm].append(result["effect"])
    return {"accepted": not errors, "status": "PASS" if not errors else "FAIL", "pairs": len(pairs),
            "errors": errors, "effects": arm_effects}


def bootstrap_ci(values: list[float], *, seed: int, replicates: int = 3000, alpha: float = 0.05) -> dict:
    if not values: raise ValueError("values must not be empty")
    if replicates < 100: raise ValueError("replicates must be >= 100")
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(replicates):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    lo = estimates[max(0, int((alpha / 2) * replicates))]
    hi = estimates[min(replicates - 1, int((1 - alpha / 2) * replicates))]
    sd = stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    return {"n_pairs": n, "mean_effect": mean(values), "paired_sd": sd, "paired_se": se,
            "bootstrap_ci_level": 1 - alpha, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "bootstrap_replicates": replicates, "ci_excludes_zero": lo > 0 or hi < 0}


def run_experiment(n_per_arm: int = 512, seed: int = 20261028, replicates: int = 3000) -> dict:
    if n_per_arm < 100: raise ValueError("n_per_arm must be >= 100 for this reference fixture")
    rng = random.Random(seed)
    pairs = [make_pair(i, "valid_pe", 0.08, rng) for i in range(n_per_arm)]
    pairs += [make_pair(i, "invalid_pe_payload", 0.0, rng) for i in range(n_per_arm)]
    panel = audit_panel(pairs)
    if not panel["accepted"]:
        raise AssertionError(panel["errors"][:10])
    valid = panel["effects"]["valid_pe"]
    placebo = panel["effects"]["invalid_pe_payload"]
    valid_stats = bootstrap_ci(valid, seed=seed + 1, replicates=replicates)
    placebo_stats = bootstrap_ci(placebo, seed=seed + 2, replicates=replicates)
    equivalence_margin = 0.02
    result = {
        "status": "PASS_CAUSAL_REPLAY_REFERENCE_ONLY",
        "seed": seed, "pairs_per_arm": n_per_arm, "total_paired_runs": len(pairs) * 2,
        "bootstrap_replicates_per_arm": replicates, "audit_panel_accepted": panel["accepted"],
        "valid_pe_effect": valid_stats,
        "invalid_pe_placebo_effect": placebo_stats,
        "placebo_equivalence_margin": equivalence_margin,
        "placebo_ci_inside_equivalence_margin": placebo_stats["bootstrap_ci_low"] > -equivalence_margin and placebo_stats["bootstrap_ci_high"] < equivalence_margin,
        "positive_control_ci_excludes_zero": valid_stats["ci_excludes_zero"],
        "positive_control_lower_bound_exceeds_0_05": valid_stats["bootstrap_ci_low"] > 0.05,
        "synthetic_ground_truth_valid_pe_effect": 0.08,
        "synthetic_ground_truth_invalid_pe_effect": 0.0,
        "production_symthaea_imported": False, "rust_compiled_or_executed": False,
        "scope_note": "Synthetic stochastic outcomes with planted effects; estimates validate this reference estimator only, not Symthaea causal influence or task utility."
    }
    mutations = run_mutation_tests(pairs[0])
    result["mutation_results"] = mutations
    result["mutations_injected"] = len(mutations)
    result["mutations_detected"] = sum(v["detected"] for v in mutations.values())
    result["self_test_passed"] = (result["positive_control_ci_excludes_zero"] and result["positive_control_lower_bound_exceeds_0_05"] and result["placebo_ci_inside_equivalence_margin"] and result["mutations_detected"] == result["mutations_injected"])
    if not result["self_test_passed"]: result["status"] = "FAIL_CAUSAL_REPLAY_REFERENCE_ONLY"
    return result


def run_mutation_tests(pair: dict) -> dict:
    mutations = {}
    def check(name: str, mutated_pairs: list, expected: str):
        result = audit_panel(mutated_pairs)
        mutations[name] = {"detected": (not result["accepted"] and any(expected in e for e in result["errors"])),
                           "expected_error": expected}
        if not mutations[name]["detected"]: raise AssertionError(f"mutation {name} not detected: {result['errors'][:5]}")
    base = json.loads(json.dumps(pair))
    m = json.loads(json.dumps(base)); m["treatment"]["scenario_digest"] = digest({"wrong": "input"})
    check("paired_input_mismatch", [m], "PAIRED_SCENARIO_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["treatment"]["pre_intervention_state_digest"] = digest({"wrong": "prefix"})
    check("pre_intervention_state_mismatch", [m], "PAIRED_PRE_INTERVENTION_STATE_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["treatment"]["randomness_tape_digest"] = digest({"different": "randomness"})
    check("randomness_tape_mismatch", [m], "PAIRED_RANDOMNESS_TAPE_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["treatment"]["outcome_calibration_digest"] = digest({"different": "outcome"})
    check("outcome_calibration_mismatch", [m], "PAIRED_OUTCOME_CALIBRATION_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["treatment"]["outcome_score"] = 1.5
    check("outcome_out_of_range", [m], "TREATMENT_OUTCOME_INVALID")
    m = json.loads(json.dumps(base)); m["treatment"]["prefix_replay_verified"] = False
    check("unverified_prefix_replay", [m], "TREATMENT_PREFIX_REPLAY_NOT_VERIFIED")
    valid_pair = make_pair(999999, "valid_pe", 0.08, random.Random(0))
    valid_pair["treatment"]["evidence_id"] = None
    check("valid_pe_missing_evidence", [valid_pair], "VALID_PE_EVIDENCE_ID_MISSING")
    placebo_pair = make_pair(999998, "invalid_pe_payload", 0.0, random.Random(1))
    placebo_pair["treatment"]["evidence_id"] = "illicit-evidence"
    check("invalid_pe_admitted", [placebo_pair], "INVALID_PE_ADMISSION")
    m = json.loads(json.dumps(base)); m["treatment"]["intervention"] = "no_pe"
    check("treatment_role_swapped", [m], "TREATMENT_INTERVENTION_MISMATCH")
    m = json.loads(json.dumps(base)); m["treatment"]["pe_transition_count"] = True
    check("boolean_transition_count", [m], "VALID_PE_ADMISSION_NOT_EXACTLY_ONCE")
    m = json.loads(json.dumps(base)); m["expected_intervention"] = ["invalid", "structure"]
    check("malformed_intervention_type", [m], "EXPECTED_INTERVENTION_INVALID")
    pair2 = json.loads(json.dumps(base)); pair2["pair_id"] = "different-pair"
    pair2["control"]["run_id"] = base["control"]["run_id"]
    panel = audit_panel([base, pair2])
    if panel["accepted"] or not any("RUN_ID_DUPLICATE:" in e for e in panel["errors"]): raise AssertionError("duplicate run ID mutation not detected")
    mutations["duplicate_run_id_across_pairs"] = {"detected": True, "expected_error": "RUN_ID_DUPLICATE"}
    pair2 = json.loads(json.dumps(base)); pair2["pair_id"] = base["pair_id"]
    panel = audit_panel([base, pair2])
    if panel["accepted"] or not any("PAIR_ID_DUPLICATE:" in e for e in panel["errors"]): raise AssertionError("duplicate pair ID mutation not detected")
    mutations["duplicate_pair_id"] = {"detected": True, "expected_error": "PAIR_ID_DUPLICATE"}
    return mutations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-per-arm", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--bootstrap-replicates", type=int, default=3000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run_experiment(args.pairs_per_arm, args.seed, args.bootstrap_replicates)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["self_test_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
