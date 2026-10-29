#!/usr/bin/env python3
"""Calibrate and stress-test the reliability of external assumption-bound issuers.

Research prototype only. The confidence bound is justified only when audited
violation indicators are independent/exchangeable and representative of the
future claims in the same issuer/scope/epoch. It cannot protect against unmodeled
drift or an invalid audit process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

SCHEMA = "symthaea-assumption-source-reliability-budget-v1"


def _prob(name: str, x: float, *, allow_one: bool = True) -> float:
    if not math.isfinite(x) or x < 0.0 or x > 1.0 or (not allow_one and x == 1.0):
        raise ValueError(f"{name} must be a finite probability in the allowed range")
    return x


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P[X <= k] for X~Binomial(n,p), using log-sum-exp for stability."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("require n >= 0 and 0 <= k <= n")
    p = _prob("p", p)
    if k == n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * lp + (n - i) * lq
        for i in range(k + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


def clopper_pearson_upper(failures: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided exact (1-alpha) upper confidence bound for a Bernoulli rate."""
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("require trials > 0 and 0 <= failures <= trials")
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if failures == trials:
        return 1.0
    lo, hi = 0.0, 1.0
    # The binomial CDF P_p(X <= failures) is monotone decreasing in p.
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if binomial_cdf(failures, trials, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compose_familywise_budget(*, alpha_stat: float, alpha_cal: float,
                              issuer_failure_upper: float,
                              decisions: int = 1) -> float:
    """Unconditional union-bound budget over a fixed number of future decisions.

    The calibration event is shared: with probability at least 1-alpha_cal,
    future issuer failures have rate <= issuer_failure_upper. Conditional on
    that event, each decision has failure <= issuer_failure_upper+alpha_stat.
    Thus a conservative familywise bound is alpha_cal + M*(delta_issuer+alpha_stat).
    This requires all component assumptions and the fixed decision horizon to hold.
    """
    if decisions < 1:
        raise ValueError("decisions must be at least 1")
    for name, value in (("alpha_stat", alpha_stat), ("alpha_cal", alpha_cal),
                         ("issuer_failure_upper", issuer_failure_upper)):
        _prob(name, value)
    return min(1.0, alpha_cal + decisions * (issuer_failure_upper + alpha_stat))


def assess_issuer(*, failures: int, audits: int, alpha_cal: float,
                  alpha_stat: float, decisions: int, target_budget: float,
                  evidence_scope: str, requested_scope: str, expired: bool,
                  audit_process_qualified: bool) -> dict:
    """Return a fail-closed scope check and reliability budget for one issuer."""
    _prob("target_budget", target_budget, allow_one=False)
    if not evidence_scope or not requested_scope:
        scope_ok = False
    else:
        scope_ok = evidence_scope == requested_scope
    if expired or not scope_ok or not audit_process_qualified or audits <= 0:
        if expired:
            reason = "evidence_expired"
        elif not scope_ok:
            reason = "scope_mismatch_or_missing"
        elif not audit_process_qualified:
            reason = "audit_process_unqualified"
        else:
            reason = "no_independent_audits"
        return {
            "status": "refuse_unresolved", "reason": reason,
            "observations_used_for_certificate": 0,
            "issuer_failure_probability_upper": None,
            "familywise_failure_budget_upper": None,
            "target_budget": target_budget,
        }
    upper = clopper_pearson_upper(failures, audits, alpha_cal)
    familywise = compose_familywise_budget(
        alpha_stat=alpha_stat, alpha_cal=alpha_cal,
        issuer_failure_upper=upper, decisions=decisions,
    )
    return {
        "status": "assumption_budget_accepted" if familywise <= target_budget else "refuse_unresolved",
        "reason": "calibrated_reliability_budget" if familywise <= target_budget else "budget_exceeds_target",
        "observations_used_for_certificate": 0,
        "audited_bound_violations": failures,
        "independent_audits": audits,
        "calibration_alpha": alpha_cal,
        "statistical_alpha_per_decision": alpha_stat,
        "planned_decisions": decisions,
        "issuer_failure_probability_upper": upper,
        "familywise_failure_budget_upper": familywise,
        "target_budget": target_budget,
        "evidence_scope": evidence_scope,
        "audit_process_qualified": True,
        "assumption": "audited violation indicators are independent/exchangeable and representative for future claims in this exact scope and epoch",
    }


def run_drift_stress(*, repeats: int = 1000, audits: int = 1000,
                     historical_failure_rate: float = 0.01,
                     future_rates: tuple[float, ...] = (0.01, 0.10),
                     alpha_cal: float = 0.005, alpha_stat: float = 0.005,
                     target_budget: float = 0.05, seed: int = 20261029) -> dict:
    """Show how a historically calibrated issuer can fail after a scope/epoch shift."""
    if repeats < 1 or audits < 1:
        raise ValueError("repeats and audits must be positive")
    _prob("historical_failure_rate", historical_failure_rate)
    for p in future_rates:
        _prob("future_rate", p)
    rng = np.random.default_rng(seed)
    outputs = {f"future_rate_{p:.3f}": {"rate": p, "accepted": 0,
                 "issuer_ucb_below_future_rate": 0, "runs": []} for p in future_rates}
    for _ in range(repeats):
        observed = int(rng.binomial(audits, historical_failure_rate))
        ucb = clopper_pearson_upper(observed, audits, alpha_cal)
        familywise = compose_familywise_budget(
            alpha_stat=alpha_stat, alpha_cal=alpha_cal,
            issuer_failure_upper=ucb, decisions=1,
        )
        accepted = familywise <= target_budget
        for p in future_rates:
            row = outputs[f"future_rate_{p:.3f}"]
            row["accepted"] += int(accepted)
            row["issuer_ucb_below_future_rate"] += int(ucb < p)
            row["runs"].append({"audit_failures": observed, "issuer_ucb": ucb,
                                "familywise_budget": familywise, "accepted": accepted,
                                "future_rate": p, "ucb_below_future_rate": ucb < p})
    for row in outputs.values():
        row["acceptance_rate"] = row["accepted"] / repeats
        row["undercoverage_rate"] = row["issuer_ucb_below_future_rate"] / repeats
        row.pop("runs")
    source_text = Path(__file__).read_bytes()
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_ISSUER_DRIFT_STRESS_NOT_PRODUCTION_VALIDATION",
        "protocol": {"seed": seed, "repeats": repeats, "independent_audits_per_issuer": audits,
                     "historical_failure_rate": historical_failure_rate,
                     "future_failure_rates": list(future_rates), "alpha_cal": alpha_cal,
                     "alpha_stat": alpha_stat, "decisions_per_calibration": 1,
                     "target_familywise_budget": target_budget},
        "results": outputs,
        "interpretation": "Historical iid calibration does not cover an unannounced change in issuer reliability. Scope/epoch qualification and drift response are separate gates.",
        "limitations": [
            "simulation uses Bernoulli independent audit violations within the historical epoch",
            "future failure rates are simulator truth and not observable at certification time",
            "Clopper-Pearson coverage applies only to its sampling assumptions",
            "a shift detector can miss undetectable shifts; this artifact does not prove arbitrary shift detection",
            "the result does not authenticate audit labels or prove the audit process is independent",
        ],
        "source_sha256": hashlib.sha256(source_text).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def build_witness() -> dict:
    # Worked gate examples: zero observed misses are not a zero failure rate.
    cp0_200 = clopper_pearson_upper(0, 200, 0.05)
    cp0_1000 = clopper_pearson_upper(0, 1000, 0.005)
    missing = assess_issuer(failures=0, audits=1000, alpha_cal=0.005,
        alpha_stat=0.005, decisions=1, target_budget=0.05,
        evidence_scope="issuer-A/risk-bounds/epoch-4", requested_scope="issuer-A/risk-bounds/epoch-5",
        expired=False, audit_process_qualified=True)
    return {
        "schema": SCHEMA,
        "status": "RELIABILITY_BUDGET_EXAMPLE_NOT_EXTERNAL_ASSURANCE",
        "zero_failures_200_audits_at_95pct_confidence": cp0_200,
        "zero_failures_1000_audits_at_99_5pct_confidence": cp0_1000,
        "wrong_epoch_negative_control": missing,
        "notes": [
            "zero observed audit failures does not mean the issuer failure probability is zero",
            "scope mismatch refuses before decision inference, even if historical audit count is large",
            "a per-decision allowance does not imply familywise safety over repeated decisions",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-output", type=Path)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--audits", type=int, default=1000)
    args = parser.parse_args()
    witness = build_witness()
    raw = (json.dumps(witness, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    summary = {"output": str(args.output), "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(),
               "zero_failure_upper_200": witness["zero_failures_200_audits_at_95pct_confidence"],
               "wrong_epoch_status": witness["wrong_epoch_negative_control"]["status"]}
    if args.stress_output:
        stress = run_drift_stress(repeats=args.repeats, audits=args.audits)
        stress_raw = (json.dumps(stress, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        args.stress_output.parent.mkdir(parents=True, exist_ok=True)
        args.stress_output.write_bytes(stress_raw)
        summary["stress_output"] = str(args.stress_output)
        summary["stress_sha256"] = hashlib.sha256(stress_raw).hexdigest()
        summary["stress_results"] = {k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in stress["results"].items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
