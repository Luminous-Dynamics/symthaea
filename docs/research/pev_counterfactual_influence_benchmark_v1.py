#!/usr/bin/env python3
"""Paired counterfactual influence audit for prediction-error (PE) consumers.

Dependency-free reference benchmark. It consumes explicit state-transition summaries;
it does not execute or import production Symthaea Rust. A passing synthetic test is
protocol evidence only, not evidence of production influence or task-utility gains.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_INVALID_PAYLOADS = [None, 0.0, 1.0, -1.0, "NaN", "+Infinity", "-Infinity"]


def is_digest(value: object) -> bool:
    return isinstance(value, str) and DIGEST_PATTERN.fullmatch(value) is not None


def payload_key(value: object) -> tuple:
    """Type-sensitive key so bools/ints cannot masquerade as float variants."""
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        return ("str", value)
    return (type(value).__name__, repr(value))


def canonical_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"symthaea.pev.counterfactual.v1\x00" + raw).hexdigest()


def run_summary(trial_id: str, branch: str, input_digest: str, initial_digest: str,
                calibration_digest: str, consumers: dict, *, pe_status: str,
                compatibility_payload=None, evidence_id=None, pe_value=None) -> dict:
    return {
        "trial_id": trial_id, "branch": branch, "input_digest": input_digest,
        "initial_state_digest": initial_digest, "gate_and_outcome_digest": calibration_digest,
        "pe_status": pe_status, "compatibility_payload": compatibility_payload,
        "evidence_id": evidence_id, "pe_value": pe_value, "consumers": consumers,
    }


def consumer_state(trial_id: str, consumer_id: str, state_label: str,
                   evidence_ids=(), pe_transition_count: int = 0) -> dict:
    return {
        "consumer_id": consumer_id,
        "before_state_digest": canonical_digest({"trial": trial_id, "consumer": consumer_id, "state": "before"}),
        "after_state_digest": canonical_digest({"trial": trial_id, "consumer": consumer_id, "state": state_label}),
        "accepted_pe_evidence_ids": list(evidence_ids),
        "pe_transition_count": pe_transition_count,
    }


def generated_trial(index: int, rng: random.Random) -> dict:
    trial_id = f"cf-{index:07d}"
    input_digest = canonical_digest({"seed_case": index, "observation": rng.randrange(1, 1_000_000),
                                     "action": rng.randrange(0, 17)})
    initial_digest = canonical_digest({"agent_state": index, "seed": rng.randrange(0, 2**32)})
    calibration_digest = canonical_digest({"gate": bool(rng.getrandbits(1)),
                                           "external_outcome": bool(rng.getrandbits(1)),
                                           "uncertainty": round(rng.random(), 8)})
    no_pe_consumers = {
        "learning_manager": consumer_state(trial_id, "learning_manager", "same"),
        "coherence_tracker": consumer_state(trial_id, "coherence_tracker", "same"),
        "uncertainty_monitor": consumer_state(trial_id, "uncertainty_monitor", "same"),
    }
    base = run_summary(trial_id, "no_pe", input_digest, initial_digest, calibration_digest,
                       no_pe_consumers, pe_status="missing")
    invalid = []
    for j, payload in enumerate(EXPECTED_INVALID_PAYLOADS):
        consumers = {k: consumer_state(trial_id, k, "same") for k in no_pe_consumers}
        invalid.append(run_summary(trial_id, f"invalid_{j}", input_digest, initial_digest,
                                   calibration_digest, consumers, pe_status="invalid",
                                   compatibility_payload=payload, evidence_id=None))
    evidence_id = f"e-{trial_id}"
    valid_consumers = {
        "learning_manager": consumer_state(trial_id, "learning_manager", f"updated-by-{evidence_id}", [evidence_id], 1),
        "coherence_tracker": consumer_state(trial_id, "coherence_tracker", "same"),
        "uncertainty_monitor": consumer_state(trial_id, "uncertainty_monitor", "same"),
    }
    valid_value = round(rng.random(), 8)
    valid = run_summary(trial_id, "valid_pe", input_digest, initial_digest, calibration_digest,
                        valid_consumers, pe_status="valid", compatibility_payload=valid_value,
                        evidence_id=evidence_id, pe_value=valid_value)
    return {"trial_id": trial_id, "target_consumers": ["learning_manager"],
            "protected_consumers": ["coherence_tracker", "uncertainty_monitor"],
            "no_pe": base, "invalid_variants": invalid, "valid_pe": valid}


def _consumer_map(run: dict, errors: list[str], label: str) -> dict:
    src = run.get("consumers")
    if not isinstance(src, dict):
        errors.append(f"{label}: CONSUMERS_NOT_OBJECT")
        return {}
    out = {}
    for key, entry in src.items():
        if not isinstance(entry, dict):
            errors.append(f"{label}: CONSUMER_NOT_OBJECT:{key}")
            continue
        cid = entry.get("consumer_id")
        if not isinstance(cid, str) or not cid or cid != key:
            errors.append(f"{label}: CONSUMER_ID_MISMATCH:{key}")
            continue
        if cid in out:
            errors.append(f"{label}: CONSUMER_DUPLICATE:{cid}")
        out[cid] = entry
        for digest_field in ("before_state_digest", "after_state_digest"):
            if not is_digest(entry.get(digest_field)):
                errors.append(f"{label}: CONSUMER_STATE_DIGEST_INVALID:{cid}:{digest_field}")
        ids, count = entry.get("accepted_pe_evidence_ids"), entry.get("pe_transition_count")
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            errors.append(f"{label}: PE_EVIDENCE_IDS_INVALID:{cid}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append(f"{label}: PE_TRANSITION_COUNT_INVALID:{cid}")
    return out


def audit_trial(trial: dict) -> dict:
    errors: list[str] = []
    if not isinstance(trial, dict):
        return {"accepted": False, "errors": ["TRIAL_NOT_OBJECT"]}
    tid = trial.get("trial_id")
    if not isinstance(tid, str) or not tid:
        errors.append("TRIAL_ID_MISSING")
    target, protected = trial.get("target_consumers"), trial.get("protected_consumers")
    if not isinstance(target, list) or not target or any(not isinstance(x, str) or not x for x in target):
        errors.append("TARGET_CONSUMERS_INVALID"); target = []
    if not isinstance(protected, list) or any(not isinstance(x, str) or not x for x in protected):
        errors.append("PROTECTED_CONSUMERS_INVALID"); protected = []
    if len(set(target)) != len(target) or len(set(protected)) != len(protected) or set(target) & set(protected):
        errors.append("CONSUMER_ROLES_OVERLAP_OR_DUPLICATE")
    no_pe, valid, variants = trial.get("no_pe"), trial.get("valid_pe"), trial.get("invalid_variants")
    if not isinstance(no_pe, dict) or not isinstance(valid, dict) or not isinstance(variants, list) or not variants:
        return {"accepted": False, "errors": errors + ["COUNTERFACTUAL_BRANCHES_MISSING"]}
    base_label, valid_label = f"{tid}/no_pe", f"{tid}/valid_pe"
    base_consumers = _consumer_map(no_pe, errors, base_label)
    valid_consumers = _consumer_map(valid, errors, valid_label)
    if len(variants) != len(EXPECTED_INVALID_PAYLOADS):
        errors.append("INVALID_VARIANT_COUNT")
    if no_pe.get("branch") != "no_pe": errors.append("NO_PE_BRANCH_LABEL_MISMATCH")
    if valid.get("branch") != "valid_pe": errors.append("VALID_BRANCH_LABEL_MISMATCH")
    for label, run in [(base_label, no_pe), (valid_label, valid)]:
        if run.get("trial_id") != tid: errors.append(f"{label}: TRIAL_ID_MISMATCH")
        for field in ("input_digest", "initial_state_digest", "gate_and_outcome_digest"):
            if not is_digest(run.get(field)):
                errors.append(f"{label}: {field.upper()}_INVALID")
    if no_pe.get("pe_status") != "missing" or no_pe.get("evidence_id") is not None:
        errors.append("NO_PE_BRANCH_NOT_MISSING")
    if no_pe.get("pe_value") is not None or no_pe.get("compatibility_payload") is not None:
        errors.append("NO_PE_BRANCH_HAS_VALUE")
    if valid.get("pe_status") != "valid" or not isinstance(valid.get("evidence_id"), str) or not valid.get("evidence_id"):
        errors.append("VALID_BRANCH_EVIDENCE_MISSING")
    if not isinstance(valid.get("pe_value"), (int, float)) or isinstance(valid.get("pe_value"), bool):
        errors.append("VALID_BRANCH_VALUE_MISSING")
    elif not math.isfinite(float(valid["pe_value"])):
        errors.append("VALID_BRANCH_VALUE_NONFINITE")
    if not isinstance(valid.get("compatibility_payload"), (int, float)) or isinstance(valid.get("compatibility_payload"), bool) or not math.isfinite(float(valid.get("compatibility_payload", float("nan")))):
        errors.append("VALID_COMPATIBILITY_PAYLOAD_INVALID")
    elif valid.get("compatibility_payload") != valid.get("pe_value"):
        errors.append("VALID_COMPATIBILITY_PAYLOAD_MISMATCH")

    observed_payloads = [r.get("compatibility_payload") if isinstance(r, dict) else object() for r in variants]
    if len({payload_key(x) for x in observed_payloads}) != len(observed_payloads) or Counter(payload_key(x) for x in observed_payloads) != Counter(payload_key(x) for x in EXPECTED_INVALID_PAYLOADS):
        errors.append("INVALID_PAYLOAD_COVERAGE")
    all_runs = [(f"{tid}/invalid[{i}]", r) for i, r in enumerate(variants)] + [(valid_label, valid)]
    base_names = set(base_consumers)
    if not set(target).issubset(base_names): errors.append("TARGET_CONSUMER_MISSING")
    if not set(protected).issubset(base_names): errors.append("PROTECTED_CONSUMER_MISSING")
    if (set(target) | set(protected)) != base_names: errors.append("UNCLASSIFIED_CONSUMER")
    if base_names != set(valid_consumers): errors.append("CONSUMER_SET_CHANGED_VALID_COUNTERFACTUAL")
    for label, run in all_runs:
        if not isinstance(run, dict):
            errors.append(f"{label}: BRANCH_NOT_OBJECT")
            continue
        if run.get("trial_id") != tid: errors.append(f"{label}: TRIAL_ID_MISMATCH")
        if label.startswith(f"{tid}/invalid["):
            index = int(label.rsplit("[", 1)[1].rstrip("]"))
            if run.get("branch") != f"invalid_{index}": errors.append(f"{label}: INVALID_BRANCH_LABEL_MISMATCH")
        for field in ("input_digest", "initial_state_digest", "gate_and_outcome_digest"):
            if run.get(field) != no_pe.get(field):
                errors.append(f"{label}: COUNTERFACTUAL_{field.upper()}_MISMATCH")
        cm = _consumer_map(run, errors, label)
        if set(cm) != base_names: errors.append(f"{label}: CONSUMER_SET_CHANGED")
        if run.get("gate_and_outcome_digest") != no_pe.get("gate_and_outcome_digest"):
            errors.append(f"{label}: INDEPENDENT_OUTCOME_CALIBRATION_CHANGED")
        is_valid = run.get("pe_status") == "valid"
        for cid in base_names & set(cm):
            state, base = cm[cid], base_consumers[cid]
            if state.get("before_state_digest") != base.get("before_state_digest"):
                errors.append(f"{label}: INITIAL_CONSUMER_STATE_CHANGED:{cid}")
            if cid in protected:
                if state.get("after_state_digest") != base.get("after_state_digest"):
                    errors.append(f"{label}: NON_TARGET_CHANNEL_CHANGED:{cid}")
                if state.get("accepted_pe_evidence_ids") != [] or state.get("pe_transition_count") != 0:
                    errors.append(f"{label}: PROTECTED_CHANNEL_PE_INFLUENCE:{cid}")
            if not is_valid:
                if state.get("after_state_digest") != base.get("after_state_digest"):
                    errors.append(f"{label}: INVALID_PE_COUNTERFACTUAL_DIVERGENCE:{cid}")
                if state.get("accepted_pe_evidence_ids") != [] or state.get("pe_transition_count") != 0:
                    errors.append(f"{label}: INVALID_PE_ADMISSION:{cid}")
            if is_valid and cid in target:
                if state.get("accepted_pe_evidence_ids") != [valid.get("evidence_id")]:
                    errors.append(f"{label}: TARGET_EVIDENCE_NOT_ADMITTED_EXACTLY_ONCE:{cid}")
                if state.get("pe_transition_count") != 1:
                    errors.append(f"{label}: TARGET_TRANSITION_COUNT_NOT_ONE:{cid}")
                if state.get("after_state_digest") == base.get("after_state_digest"):
                    errors.append(f"{label}: POSITIVE_CONTROL_NO_EFFECT:{cid}")
        if not is_valid:
            if run.get("pe_status") != "invalid": errors.append(f"{label}: INVALID_BRANCH_STATUS_MISMATCH")
            if run.get("evidence_id") is not None: errors.append(f"{label}: INVALID_BRANCH_HAS_ADMITTED_EVIDENCE")
            if run.get("pe_value") is not None: errors.append(f"{label}: INVALID_BRANCH_VALUE_NOT_NULL")
    for cid, state in base_consumers.items():
        if state.get("accepted_pe_evidence_ids") != [] or state.get("pe_transition_count") != 0:
            errors.append(f"{base_label}: BASELINE_HAS_PE_INFLUENCE:{cid}")
    return {"accepted": not errors, "errors": errors, "trial_id": tid,
            "invalid_variants_checked": len(variants), "target_consumers_checked": len(target),
            "protected_consumers_checked": len(protected)}


def audit_trials(trials: list) -> dict:
    """Audit a batch and enforce unique trial/evidence identity across trials."""
    errors = []
    if not isinstance(trials, list) or not trials:
        return {"accepted": False, "status": "FAIL", "errors": ["TRIAL_BATCH_EMPTY_OR_INVALID"], "trials": 0}
    seen_trials, seen_evidence = set(), set()
    evaluated = []
    for index, trial in enumerate(trials):
        result = audit_trial(trial)
        evaluated.append(result)
        if not isinstance(trial, dict):
            continue
        tid = trial.get("trial_id")
        if isinstance(tid, str):
            if tid in seen_trials:
                errors.append(f"TRIAL_ID_DUPLICATE:{tid}")
            seen_trials.add(tid)
        valid = trial.get("valid_pe")
        eid = valid.get("evidence_id") if isinstance(valid, dict) else None
        if isinstance(eid, str) and eid:
            if eid in seen_evidence:
                errors.append(f"EVIDENCE_ID_DUPLICATE:{eid}")
            seen_evidence.add(eid)
        if not result.get("accepted"):
            errors.extend(f"trial[{index}] {x}" for x in result.get("errors", []))
    return {"accepted": not errors, "status": "PASS" if not errors else "FAIL",
            "trials": len(trials), "accepted_trials": sum(bool(x.get("accepted")) for x in evaluated),
            "errors": errors}


def self_test(cases: int = 100_000, seed: int = 20261028) -> dict:
    if cases <= 0: raise ValueError("cases must be positive")
    rng = random.Random(seed)
    for i in range(cases):
        result = audit_trial(generated_trial(i, rng))
        if not result["accepted"]:
            raise AssertionError(f"clean counterfactual {i} rejected: {result['errors'][:5]}")
    base = generated_trial(cases + 1, rng)
    mutations = {}
    m = json.loads(json.dumps(base)); m["invalid_variants"][1]["consumers"]["coherence_tracker"]["after_state_digest"] = "tampered-state"
    mutations["invalid_payload_changes_protected_state"] = (m, "INVALID_PE_COUNTERFACTUAL_DIVERGENCE")
    m = json.loads(json.dumps(base)); m["valid_pe"]["consumers"]["learning_manager"]["after_state_digest"] = m["no_pe"]["consumers"]["learning_manager"]["after_state_digest"]
    mutations["valid_pe_noop_learner"] = (m, "POSITIVE_CONTROL_NO_EFFECT")
    m = json.loads(json.dumps(base)); m["valid_pe"]["consumers"]["learning_manager"]["accepted_pe_evidence_ids"] = []
    mutations["positive_control_missing_admission"] = (m, "TARGET_EVIDENCE_NOT_ADMITTED_EXACTLY_ONCE")
    m = json.loads(json.dumps(base)); m["valid_pe"]["consumers"]["learning_manager"]["accepted_pe_evidence_ids"] *= 2
    mutations["duplicate_positive_admission"] = (m, "TARGET_EVIDENCE_NOT_ADMITTED_EXACTLY_ONCE")
    m = json.loads(json.dumps(base)); m["valid_pe"]["consumers"]["uncertainty_monitor"]["after_state_digest"] = "perturbed-nontarget"
    mutations["valid_pe_collateral_state_change"] = (m, "NON_TARGET_CHANNEL_CHANGED")
    m = json.loads(json.dumps(base)); m["invalid_variants"][0]["input_digest"] = "different-input"
    mutations["counterfactual_input_mismatch"] = (m, "COUNTERFACTUAL_INPUT_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["invalid_variants"][0]["gate_and_outcome_digest"] = "outcome-was-suppressed"
    mutations["independent_outcome_suppression"] = (m, "COUNTERFACTUAL_GATE_AND_OUTCOME_DIGEST_MISMATCH")
    m = json.loads(json.dumps(base)); m["target_consumers"] = ["consumer-that-does-not-exist"]
    mutations["missing_target_consumer"] = (m, "TARGET_CONSUMER_MISSING")
    m = json.loads(json.dumps(base)); m["invalid_variants"] = m["invalid_variants"][:-1]
    mutations["missing_invalid_variant"] = (m, "INVALID_VARIANT_COUNT")
    m = json.loads(json.dumps(base)); m["invalid_variants"][6]["compatibility_payload"] = "NaN"
    mutations["duplicate_invalid_payload_coverage"] = (m, "INVALID_PAYLOAD_COVERAGE")
    m = json.loads(json.dumps(base)); m["valid_pe"]["compatibility_payload"] = m["valid_pe"]["pe_value"] + 0.1
    mutations["valid_scalar_evidence_mismatch"] = (m, "VALID_COMPATIBILITY_PAYLOAD_MISMATCH")
    m = json.loads(json.dumps(base)); m["no_pe"]["consumers"]["learning_manager"]["after_state_digest"] = "not-a-digest"
    mutations["malformed_consumer_state_digest"] = (m, "CONSUMER_STATE_DIGEST_INVALID")
    m = json.loads(json.dumps(base)); m["no_pe"]["consumers"]["unclassified_cache"] = consumer_state(m["trial_id"], "unclassified_cache", "same")
    for br in m["invalid_variants"]: br["consumers"]["unclassified_cache"] = consumer_state(m["trial_id"], "unclassified_cache", "same")
    m["valid_pe"]["consumers"]["unclassified_cache"] = consumer_state(m["trial_id"], "unclassified_cache", "same")
    mutations["unclassified_consumer"] = (m, "UNCLASSIFIED_CONSUMER")
    m = json.loads(json.dumps(base)); m["invalid_variants"][0]["branch"] = "invalid_wrong"
    mutations["invalid_branch_label_mismatch"] = (m, "INVALID_BRANCH_LABEL_MISMATCH")
    checked = {}
    for name, (mutated, expected) in mutations.items():
        result = audit_trial(mutated)
        if result["accepted"] or not any(expected in e for e in result["errors"]):
            raise AssertionError(f"mutation {name} not detected: {result['errors'][:8]}")
        checked[name] = {"rejected": True, "expected_error_detected": True}
    duplicate_batch = audit_trials([base, json.loads(json.dumps(base))])
    if duplicate_batch["accepted"] or not any(x.startswith("TRIAL_ID_DUPLICATE:") for x in duplicate_batch["errors"]):
        raise AssertionError(f"duplicate trial ID not detected: {duplicate_batch}")
    checked["duplicate_trial_id_across_batch"] = {"rejected": True, "expected_error_detected": True}
    other = generated_trial(cases + 2, rng)
    other["valid_pe"]["evidence_id"] = base["valid_pe"]["evidence_id"]
    other["valid_pe"]["consumers"]["learning_manager"]["accepted_pe_evidence_ids"] = [base["valid_pe"]["evidence_id"]]
    duplicate_evidence_batch = audit_trials([base, other])
    if duplicate_evidence_batch["accepted"] or not any(x.startswith("EVIDENCE_ID_DUPLICATE:") for x in duplicate_evidence_batch["errors"]):
        raise AssertionError(f"duplicate evidence ID not detected: {duplicate_evidence_batch}")
    checked["duplicate_evidence_id_across_batch"] = {"rejected": True, "expected_error_detected": True}
    return {"status": "PASS_COUNTERFACTUAL_REFERENCE_AUDITOR_ONLY", "seed": seed,
            "generated_paired_trials": cases, "generated_trials_accepted": cases,
            "invalid_payload_variants_per_trial": 7, "branches_per_trial": 9,
            "counterfactual_runs_audited": cases * 9, "mutations_injected": len(mutations) + 2,
            "mutations_detected": len(checked), "mutation_results": checked,
            "production_symthaea_imported": False, "rust_compiled_or_executed": False,
            "scope_note": "Synthetic paired state-digest fixtures only; not production causal verification or task-utility/SOTA evidence."}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cases", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.self_test:
        report = self_test(a.cases, a.seed)
    elif a.input:
        loaded = json.loads(a.input.read_text(encoding="utf-8"))
        trials = loaded if isinstance(loaded, list) else loaded.get("trials", [])
        batch = audit_trials(trials)
        report = {"status": batch["status"], "trials": batch["trials"],
                  "accepted": batch["accepted_trials"], "errors": batch["errors"]}
    else:
        p.error("choose --self-test or --input")
    out = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(out, encoding="utf-8")
    print(out, end="")
    return 0 if str(report.get("status", "")).startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
