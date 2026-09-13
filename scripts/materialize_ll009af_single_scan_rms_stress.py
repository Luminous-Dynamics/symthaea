#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import tempfile

import ll009af_engine as E
import ll009af_support as A

def self_test():
    import numpy as np
    AE = importlib.import_module("materialize_ll009ae_rms_stress_sensitivity")
    Y = importlib.import_module("materialize_ll009y_memberwise_hybrid_horizon")
    p = A.validate_policy({
        "schema_version": A.POLICY_SCHEMA,
        "semantics_class": A.SEMANTICS,
        "stress_source": "inherit_exact_ll009ae_policy",
        "raster_source_passes_required": 1,
        "intervention": "multiply_only_rms_m_in_ephemeral_float64_bin_copy",
        "persist_stressed_rasters": False,
        "baseline_reproduction": "byte_exact_y_and_s_at_lambda_1",
        "production_ae_reference_requirement": "not_required_after_synthetic_equivalence_gate",
        "l2_identity_relative_tolerance": 1e-12,
        "l2_identity_absolute_tolerance": 1e-12,
    })
    dt = np.dtype(Y.RECORD_DTYPE_FIELDS)
    r = np.zeros(4, dtype=dt)
    r["c"] = [.9995, .999, .9985, .998]
    r["radius_m"] = [1001, 1002, 1004, 1008]
    r["rms_m"] = [1, 2, 4, 8]
    r["row"] = [10, 11, 12, 13]
    r["col"] = [20, 21, 22, 23]
    r["nominal_elevation_deg"] = [-1, -.5, 0, .5]
    orig = r.copy()
    obs = np.asarray([1000., 1003.])
    lambdas = [.75, 1., 1.5, 3.]
    af_far, ae_far = [], []
    for lam in lambdas:
        s, w = A.scale_with_witness(
            r, lam, np, p["l2_identity_relative_tolerance"], p["l2_identity_absolute_tolerance"]
        )
        af = Y.solve_members_for_bin(s, obs, .25, 10., 64, 1e-12, 2, 4, np)
        ae = AE._scaled_solver(Y.solve_members_for_bin, lam)(r, obs, .25, 10., 64, 1e-12, 2, 4, np)
        if any(not np.array_equal(x, y) for x, y in zip(af, ae)):
            raise AssertionError("AF != AE solver")
        assert w["max_abs_error_vs_float64_lambda_times_baseline_rms"] == 0.
        af_far.append(af[0]); ae_far.append(ae[0])
    assert np.array_equal(r, orig)

    full_af, full_ae = [], []
    for af, ae in zip(af_far, ae_far):
        fa = [float(np.max(np.maximum(af, -100.))), 100.]
        fe = [float(np.max(np.maximum(ae, -100.))), 100.]
        assert fa == fe
        full_af.append(fa); full_ae.append(fe)
    assert all(x[1] == 100. for x in full_af)
    low, high = full_af[1][0], full_af[-1][0]
    if high <= low:
        raise AssertionError("synthetic far-dominant stress did not move horizon")
    target = (low + high) / 2

    S = importlib.import_module("generate_ll009s_semantic_visibility")
    with tempfile.TemporaryDirectory(prefix="ll009af-selftest-") as td:
        root = pathlib.Path(td)
        src = root / "source.txt"
        src.write_text("synthetic ephemeris witness\n")
        cfg = {
            "schema_version": "ll009s.semantic-visibility-config.v1",
            "requested_claim_class": "descriptive_geometry_only",
            "policies": {"max_time_gap_s": 10.0, "los_margin_deg": 0.0},
            "sources": [{"source_id": "syn", "path": "source.txt", "sha256": A.hf(src)}],
            "targets": [{
                "target_id": "earth", "kind": "earth", "frame": "SYN", "source_refs": ["syn"],
                "samples": [
                    {"t_s": t, "direction": [math.cos(math.radians(target)), 0.0, math.sin(math.radians(target))]}
                    for t in (0.0, 1.0, 2.0)
                ],
            }],
        }
        cp = root / "cfg.json"; cp.write_bytes(A.cb(cfg))
        op = root / "o.json"; rp = root / "r.json"
        op.write_bytes(A.cb({
            "schema_version": "ll009o.uncertainty-semantics-receipt.v1", "study_id": "syn",
            "deterministic_upper_bound_eligible": False, "risk_qualified_horizon_eligible": False,
        }))
        rp.write_bytes(A.cb({
            "schema_version": "ll009r.spatial-support-classification-receipt.v1", "study_id": "syn",
            "strongest_common_spatial_support": "sample_points_only",
        }))
        s_af, s_ae = [], []
        for idx, (ha, he) in enumerate(zip(full_af, full_ae)):
            def pack(h):
                vals = [h[0], h[1], -90.0, -90.0]
                return {
                    "schema_version": "ll009k.horizon-pack.v1", "status": "pass", "study_id": "syn",
                    "frame_contract_id": "F", "epoch_contract_id": "E", "site_ref": "site", "frame": "SYN",
                    "azimuth_bin_width_deg": 90.0, "bin_count": 4,
                    "basis": {"north": [1.,0.,0.], "east": [0.,1.,0.], "up": [0.,0.,1.], "fallback_used": False},
                    "bins": [
                        {"bin_index": bi, "azimuth_start_deg": bi*90.0,
                         "azimuth_end_deg": (bi+1)*90.0, "conservative_elevation_deg": float(v)}
                        for bi, v in enumerate(vals)
                    ],
                }
            kap, kbp = root / f"ka-{idx}.json", root / f"kb-{idx}.json"
            kap.write_bytes(A.cb(pack(ha))); kbp.write_bytes(A.cb(pack(he)))
            sa, sb = S.generate(cp, kap, op, rp, root), S.generate(cp, kbp, op, rp, root)
            if sa.get("metrics") != sb.get("metrics"):
                raise AssertionError("AF/AE S metric payload mismatch")
            s_af.append(sa["metrics"]["dte_los_availability_fraction"]["central"])
            s_ae.append(sb["metrics"]["dte_los_availability_fraction"]["central"])
        assert s_af == s_ae and s_af[1] != s_af[-1]
    print("LL-009AF single-scan intervention/equivalence self-test: PASS")

def main():
    p = argparse.ArgumentParser(description="LL-009AF single-scan RMS stress campaign")
    p.add_argument("--self-test", action="store_true")
    for n in (
        "af-policy","ae-policy","y-policy","q-receipt","v-receipt","w-receipt","x-policy",
        "x-receipt","r-receipt","l-config","baseline-y","s-config","o-receipt","baseline-s",
        "artifact-root","output","witness-output","performance-output"
    ):
        p.add_argument("--" + n)
    z = p.parse_args()
    if z.self_test:
        self_test()
        return 0
    names = (
        "af_policy","ae_policy","y_policy","q_receipt","v_receipt","w_receipt","x_policy",
        "x_receipt","r_receipt","l_config","baseline_y","s_config","o_receipt","baseline_s",
        "artifact_root","output","witness_output","performance_output"
    )
    if any(getattr(z, n) is None for n in names):
        p.error("all materialization arguments required unless --self-test")
    out, wit, perf = E.materialize(
        pathlib.Path(z.af_policy), pathlib.Path(z.ae_policy), pathlib.Path(z.y_policy),
        pathlib.Path(z.q_receipt), pathlib.Path(z.v_receipt), pathlib.Path(z.w_receipt),
        pathlib.Path(z.x_policy), pathlib.Path(z.x_receipt), pathlib.Path(z.r_receipt),
        pathlib.Path(z.l_config), pathlib.Path(z.baseline_y), pathlib.Path(z.s_config),
        pathlib.Path(z.o_receipt), pathlib.Path(z.baseline_s), pathlib.Path(z.artifact_root)
    )
    A.wr(pathlib.Path(z.output), out)
    A.wr(pathlib.Path(z.witness_output), wit)
    A.wr_diagnostic(pathlib.Path(z.performance_output), perf)
    print(json.dumps(out, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
