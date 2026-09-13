#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import pathlib
import tempfile
from typing import Any, Callable

POLICY_SCHEMA = "ll009ae.rms-stress-sensitivity-policy.v1"
OUT_SCHEMA = "ll009ae.rms-stress-sensitivity-frontier-receipt.v1"
Y_K_SCHEMA = "ll009k.horizon-pack.v1"
Y_PRODUCER = "ll009y.memberwise-hybrid-horizon-receipt.v1"
S_SCHEMA = "ll009s.semantic-visibility-receipt.v1"

class AEError(RuntimeError):
    pass

def canonical_bytes(v: Any) -> bytes:
    return (json.dumps(v, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def read_obj(p: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        v = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise AEError(f"cannot read {label}: {e}") from e
    if not isinstance(v, dict):
        raise AEError(f"{label} must contain object")
    return v

def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def verify_self_hash(v: dict[str, Any], label: str) -> None:
    d = v.get("receipt_sha256")
    if not isinstance(d, str) or len(d) != 64:
        raise AEError(f"{label} receipt self-hash missing")
    b = json.loads(json.dumps(v))
    b.pop("receipt_sha256", None)
    if sha256_bytes(canonical_bytes(b)) != d:
        raise AEError(f"{label} receipt self-hash mismatch")

def validate_policy(p: dict[str, Any]) -> dict[str, Any]:
    if p.get("schema_version") != POLICY_SCHEMA:
        raise AEError(f"policy schema must be {POLICY_SCHEMA}")
    if p.get("semantics_class") != "model_conditional_rms_stress_sensitivity_frontier":
        raise AEError("unexpected AE semantics")
    ls = p.get("stress_factors")
    if not isinstance(ls, list) or len(ls) < 2 or not all(finite(x) and float(x) > 0 for x in ls):
        raise AEError("stress_factors must contain >=2 positive finite values")
    vals = [float(x) for x in ls]
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise AEError("stress_factors must be strictly increasing")
    if 1.0 not in vals:
        raise AEError("stress_factors must contain exact lambda=1.0")
    if p.get("intervention") != "multiply_far_product90_rms_before_geometry_aware_markov_solver":
        raise AEError("AE intervention drift")
    if p.get("baseline_reproduction") != "byte_exact_y_and_s_at_lambda_1":
        raise AEError("AE must require byte-exact lambda=1 Y/S reproduction")
    if p.get("probability_calibration_claim") != "prohibited":
        raise AEError("AE may not claim probability calibration")
    if p.get("lambda_below_one_semantics") != "diagnostic_non_conservative":
        raise AEError("lambda<1 semantics must remain diagnostic_non_conservative")
    tol = p.get("horizon_monotonicity_tolerance_deg")
    if not finite(tol) or float(tol) < 0:
        raise AEError("invalid monotonicity tolerance")
    return p

def validate_baselines(y: dict[str, Any], s: dict[str, Any]) -> None:
    if y.get("schema_version") != Y_K_SCHEMA or y.get("producer_schema_version") != Y_PRODUCER or y.get("status") != "pass":
        raise AEError("baseline Y must be passing LL-009Y K pack")
    verify_self_hash(y, "baseline Y")
    if s.get("schema_version") != S_SCHEMA or s.get("status") not in {"pass", "claim_blocked_geometry_available"}:
        raise AEError("baseline S receipt invalid")
    verify_self_hash(s, "baseline S")
    if s.get("k_horizon_pack_sha256") is None:
        raise AEError("baseline S lacks K binding")

def _scaled_solver(original: Callable, lam: float) -> Callable:
    def wrapped(records, observer_radii, alpha_bin, uniform_k, iterations,
                risk_tolerance, member_batch_size, pixel_chunk_size, np):
        scaled = np.array(records, copy=True)
        scaled["rms_m"] *= lam
        if np.any(~np.isfinite(scaled["rms_m"])) or np.any(scaled["rms_m"] < 0):
            raise AEError("stress scaling produced invalid RMS")
        return original(
            scaled, observer_radii, alpha_bin, uniform_k, iterations,
            risk_tolerance, member_batch_size, pixel_chunk_size, np
        )
    return wrapped

@contextlib.contextmanager
def stress_y_solver(Y: Any, lam: float):
    original = Y.solve_members_for_bin
    Y.solve_members_for_bin = _scaled_solver(original, lam)
    try:
        yield
    finally:
        Y.solve_members_for_bin = original

def y_horizon_vector(y: dict[str, Any]) -> list[float]:
    bins = y.get("bins")
    if not isinstance(bins, list) or not bins:
        raise AEError("Y K pack bins missing")
    out = []
    for i, b in enumerate(bins):
        if not isinstance(b, dict) or b.get("bin_index") != i or not finite(b.get("conservative_elevation_deg")):
            raise AEError("Y K bins invalid/noncanonical")
        out.append(float(b["conservative_elevation_deg"]))
    return out

def extract_metric_central(s: dict[str, Any]) -> dict[str, float]:
    m = s.get("metrics")
    if not isinstance(m, dict) or not m:
        raise AEError("S metrics missing")
    out: dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, dict) and finite(v.get("central")):
            out[k] = float(v["central"])
    if not out:
        raise AEError("S has no central metrics")
    return out

def visibility_direction(metric: str) -> str | None:
    if metric.endswith("_visibility_fraction") or metric.endswith("_availability_fraction"):
        return "nonincreasing"
    if "occlusion" in metric or "outage" in metric:
        return "nondecreasing"
    return None

def check_frontier_monotonicity(rows: list[dict[str, Any]], tol_deg: float) -> None:
    for a, b in zip(rows, rows[1:]):
        ha, hb = a["horizon_deg"], b["horizon_deg"]
        if len(ha) != len(hb) or any(y + tol_deg < x for x, y in zip(ha, hb)):
            raise AEError(f"horizon monotonicity violated between lambda={a['lambda']} and {b['lambda']}")
        ma, mb = a["metrics"], b["metrics"]
        for k in set(ma) & set(mb):
            direction = visibility_direction(k)
            if direction == "nonincreasing" and mb[k] > ma[k] + 1e-15:
                raise AEError(f"visibility falsely improves with larger RMS stress: {k}")
            if direction == "nondecreasing" and mb[k] + 1e-15 < ma[k]:
                raise AEError(f"blocked duration falsely improves with larger RMS stress: {k}")

def first_threshold_brackets(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric, spec in thresholds.items():
        if not isinstance(spec, dict) or spec.get("relation") not in {"at_most", "at_least"} or not finite(spec.get("value")):
            raise AEError(f"invalid threshold spec for {metric}")
        relation, value = spec["relation"], float(spec["value"])
        previous = None
        for row in rows:
            if metric not in row["metrics"]:
                continue
            x = row["metrics"][metric]
            crossed = x <= value if relation == "at_most" else x >= value
            if crossed:
                out[metric] = {
                    "relation": relation, "value": value,
                    "first_crossing_lambda": row["lambda"],
                    "bracket_low_lambda": None if previous is None else previous,
                    "bracket_high_lambda": row["lambda"],
                    "interpretation": "declared_stress_ladder_bracket_not_calibration_estimate",
                }
                break
            previous = row["lambda"]
        if metric not in out:
            out[metric] = {
                "relation": relation, "value": value, "first_crossing_lambda": None,
                "bracket_low_lambda": previous, "bracket_high_lambda": None,
                "interpretation": "no_crossing_on_declared_stress_ladder",
            }
    return out

def materialize(
    policy_path: pathlib.Path,
    y_policy_path: pathlib.Path,
    q_path: pathlib.Path,
    v_path: pathlib.Path,
    w_path: pathlib.Path,
    x_policy_path: pathlib.Path,
    x_path: pathlib.Path,
    r_path: pathlib.Path,
    l_path: pathlib.Path,
    baseline_y_path: pathlib.Path,
    s_config_path: pathlib.Path,
    o_path: pathlib.Path,
    baseline_s_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    policy = validate_policy(read_obj(policy_path, "AE policy"))
    baseline_y = read_obj(baseline_y_path, "baseline Y")
    baseline_s = read_obj(baseline_s_path, "baseline S")
    validate_baselines(baseline_y, baseline_s)
    if baseline_s.get("k_horizon_pack_sha256") != sha256_file(baseline_y_path):
        raise AEError("baseline S does not bind exact baseline Y K pack")

    Y = importlib.import_module("materialize_ll009y_memberwise_hybrid_horizon")
    S = importlib.import_module("generate_ll009s_semantic_visibility")
    rows = []
    baseline_index = policy["stress_factors"].index(1.0)

    with tempfile.TemporaryDirectory(prefix="ll009ae-") as td:
        root = pathlib.Path(td)
        for lam_raw in policy["stress_factors"]:
            lam = float(lam_raw)
            with stress_y_solver(Y, lam):
                y = Y.materialize(
                    y_policy_path, q_path, v_path, w_path, x_policy_path,
                    x_path, r_path, l_path, artifact_root
                )
            y_bytes = canonical_bytes(y)
            if lam == 1.0 and y_bytes != baseline_y_path.read_bytes():
                raise AEError("lambda=1 failed byte-exact Y baseline reproduction")
            y_path = root / f"y-lambda-{lam:.17g}.json"
            y_path.write_bytes(y_bytes)

            s = S.generate(s_config_path, y_path, o_path, r_path, artifact_root)
            s_bytes = canonical_bytes(s)
            if lam == 1.0 and s_bytes != baseline_s_path.read_bytes():
                raise AEError("lambda=1 failed byte-exact S baseline reproduction")
            rows.append({
                "lambda": lam,
                "conservative": lam >= 1.0,
                "lambda_semantics": "stress_or_baseline" if lam >= 1.0 else "diagnostic_non_conservative",
                "internal_counterfactual_y_sha256": sha256_bytes(y_bytes),
                "internal_counterfactual_s_sha256": sha256_bytes(s_bytes),
                "horizon_deg": y_horizon_vector(y),
                "metrics": extract_metric_central(s),
            })

    check_frontier_monotonicity(rows, float(policy["horizon_monotonicity_tolerance_deg"]))
    base = rows[baseline_index]
    for row in rows:
        row["max_horizon_increase_vs_lambda1_deg"] = max(
            (b - a for a, b in zip(base["horizon_deg"], row["horizon_deg"])), default=0.0
        )
        row["metric_delta_vs_lambda1"] = {
            k: row["metrics"][k] - base["metrics"][k]
            for k in set(base["metrics"]) & set(row["metrics"])
        }

    out = {
        "schema_version": OUT_SCHEMA,
        "status": "pass",
        "semantics_class": policy["semantics_class"],
        "study_id": baseline_y.get("study_id"),
        "lineage": {
            "policy_sha256": sha256_file(policy_path),
            "y_policy_sha256": sha256_file(y_policy_path),
            "q_receipt_sha256": sha256_file(q_path),
            "v_receipt_sha256": sha256_file(v_path),
            "w_receipt_sha256": sha256_file(w_path),
            "x_policy_sha256": sha256_file(x_policy_path),
            "x_receipt_sha256": sha256_file(x_path),
            "r_receipt_sha256": sha256_file(r_path),
            "l_config_sha256": sha256_file(l_path),
            "baseline_y_k_pack_sha256": sha256_file(baseline_y_path),
            "s_config_sha256": sha256_file(s_config_path),
            "o_receipt_sha256": sha256_file(o_path),
            "baseline_s_receipt_sha256": sha256_file(baseline_s_path),
        },
        "intervention": policy["intervention"],
        "baseline_reproduction": {"lambda": 1.0, "y_byte_exact": True, "s_byte_exact": True},
        "frontier": rows,
        "engineering_threshold_brackets": first_threshold_brackets(rows, policy.get("engineering_thresholds", {})),
        "probability_calibration_claim": False,
        "joint_q_far_probability_claim": False,
        "real_world_calibration_multiplier_selected": False,
        "claim_rule": (
            "Each lambda is a declared counterfactual scale applied only to the exact Product90 far RMS "
            "input before the existing LL-009Y geometry-aware Markov solver. Visibility is recomputed by "
            "the exact LL-009S engine. Lambda is a sensitivity coordinate, not an inferred calibration."
        ),
        "non_claims": [
            "No lambda is identified as the true Product90 calibration multiplier.",
            "Lambda below one is diagnostic and is not conservative.",
            "No Gaussian, independence, covariance, or Q-by-far joint probability assumption is introduced.",
            "LL-009R spatial-support limitations remain unchanged.",
            "Counterfactual Y/S objects are internal numeric intermediates and are not promoted as new Y/S evidence receipts.",
        ],
    }
    out["receipt_sha256"] = sha256_bytes(canonical_bytes(out))
    return out

def self_test() -> None:
    validate_policy({
        "schema_version": POLICY_SCHEMA,
        "semantics_class": "model_conditional_rms_stress_sensitivity_frontier",
        "stress_factors": [0.75, 1.0, 1.25, 2.0],
        "intervention": "multiply_far_product90_rms_before_geometry_aware_markov_solver",
        "baseline_reproduction": "byte_exact_y_and_s_at_lambda_1",
        "probability_calibration_claim": "prohibited",
        "lambda_below_one_semantics": "diagnostic_non_conservative",
        "horizon_monotonicity_tolerance_deg": 1e-12,
    })
    rows = [
        {"lambda": 0.75, "horizon_deg": [1.0, 2.0], "metrics": {"solar_full_disc_visibility_fraction": .8, "max_contiguous_dte_outage_h": 2.0}},
        {"lambda": 1.0, "horizon_deg": [1.1, 2.2], "metrics": {"solar_full_disc_visibility_fraction": .7, "max_contiguous_dte_outage_h": 3.0}},
        {"lambda": 2.0, "horizon_deg": [1.5, 3.0], "metrics": {"solar_full_disc_visibility_fraction": .5, "max_contiguous_dte_outage_h": 5.0}},
    ]
    check_frontier_monotonicity(rows, 1e-12)
    b = first_threshold_brackets(rows, {"solar_full_disc_visibility_fraction": {"relation": "at_most", "value": .6}})
    assert b["solar_full_disc_visibility_fraction"]["first_crossing_lambda"] == 2.0
    bad = json.loads(json.dumps(rows))
    bad[2]["horizon_deg"][0] = 0.9
    try:
        check_frontier_monotonicity(bad, 1e-12)
        raise AssertionError("nonmonotone horizon accepted")
    except AEError:
        pass

    import numpy as np
    rec = np.zeros(3, dtype=[("rms_m", "<f8"), ("c", "<f8")])
    rec["rms_m"] = [1.0, 2.0, 4.0]
    observed = {}
    def fake_solver(records, observer_radii, alpha_bin, uniform_k, iterations,
                    risk_tolerance, member_batch_size, pixel_chunk_size, np_mod):
        observed["rms"] = records["rms_m"].tolist()
        return np_mod.array([0.0]), np_mod.array([0.0]), np_mod.array([0.0])
    wrapped = _scaled_solver(fake_solver, 1.5)
    wrapped(rec, np.array([1.0]), .01, 10.0, 32, 1e-12, 1, 1024, np)
    assert observed["rms"] == [1.5, 3.0, 6.0]
    assert rec["rms_m"].tolist() == [1.0, 2.0, 4.0], "baseline records mutated"

    class FakeY:
        pass
    fy = FakeY()
    fy.solve_members_for_bin = fake_solver
    original = fy.solve_members_for_bin
    with stress_y_solver(fy, 2.0):
        assert fy.solve_members_for_bin is not original
    assert fy.solve_members_for_bin is original, "Y solver patch not restored"
    print("LL-009AE RMS stress sensitivity self-test: PASS")

def write_once(path: pathlib.Path, value: dict[str, Any]) -> None:
    b = canonical_bytes(value)
    if path.exists() and path.read_bytes() != b:
        raise AEError(f"refusing to overwrite differing output {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b)

def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--self-test", action="store_true")
    for x in ("policy","y-policy","q-receipt","v-receipt","w-receipt","x-policy","x-receipt",
              "r-receipt","l-config","baseline-y","s-config","o-receipt","baseline-s","artifact-root","out"):
        a.add_argument("--" + x)
    ns = a.parse_args()
    if ns.self_test:
        self_test()
        return 0
    required = ["policy","y_policy","q_receipt","v_receipt","w_receipt","x_policy","x_receipt",
                "r_receipt","l_config","baseline_y","s_config","o_receipt","baseline_s","artifact_root","out"]
    if any(getattr(ns, x) is None for x in required):
        a.error("all materialization arguments are required unless --self-test")
    out = materialize(
        pathlib.Path(ns.policy), pathlib.Path(ns.y_policy), pathlib.Path(ns.q_receipt),
        pathlib.Path(ns.v_receipt), pathlib.Path(ns.w_receipt), pathlib.Path(ns.x_policy),
        pathlib.Path(ns.x_receipt), pathlib.Path(ns.r_receipt), pathlib.Path(ns.l_config),
        pathlib.Path(ns.baseline_y), pathlib.Path(ns.s_config), pathlib.Path(ns.o_receipt),
        pathlib.Path(ns.baseline_s), pathlib.Path(ns.artifact_root),
    )
    write_once(pathlib.Path(ns.out), out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
