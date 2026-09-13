#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib
import math
import pathlib
import tempfile
import time

import ll009af_support as A

def rebuild_y(Y, baseline_y, y_policy, q_info, r_layers, far_info, observer_radii,
              far_h, far_risk, far_nom, np):
    mc, bc = q_info["member_count"], q_info["bin_count"]
    near_margin = r_layers[y_policy["near_layer_id"]]["margin_deg"]
    far_margin = r_layers[y_policy["far_layer_id"]]["margin_deg"]
    near = np.asarray([m["near_horizon_deg"] for m in q_info["members"]], dtype="float64")
    full = np.maximum(near + near_margin, far_h + far_margin)
    nw = (near + near_margin) >= (far_h + far_margin)
    q = float(y_policy["quantile"])
    summaries, selected = [], []
    for b in range(bc):
        vals = full[:, b]
        ordered = np.sort(vals, kind="mergesort")
        rank = max(1, int(math.ceil(q * mc)))
        sel = float(ordered[rank - 1])
        nwin = int(np.count_nonzero(nw[:, b]))
        selected.append(sel)
        summaries.append({
            "bin_index": b,
            "member_count": mc,
            "min_deg": float(ordered[0]),
            "median_deg": Y.nearest_rank(vals.tolist(), 0.5),
            "selected_quantile": q,
            "selected_quantile_deg": sel,
            "finite_observed_max_deg": float(ordered[-1]),
            "max_member_ordinal": int(np.argmax(vals)),
            "near_winner_member_count": nwin,
            "far_winner_member_count": mc - nwin,
            "maximum_far_markov_bound": float(np.max(far_risk[:, b])),
            "alpha_bin": float(far_info["x_bins"][b]["alpha_bin"]),
        })
    y = copy.deepcopy(baseline_y)
    for b, sel in enumerate(selected):
        y["bins"][b]["conservative_elevation_deg"] = sel
    ev = y["memberwise_hybrid_evidence"]
    ev["per_bin_summary"] = summaries
    dig = hashlib.sha256()
    for i, qm in enumerate(q_info["members"]):
        m = copy.deepcopy(ev["members"][i])
        if m.get("ordinal") != qm["ordinal"] or m.get("source_id") != qm["source_id"]:
            raise A.AFError("Q/Y member identity drift")
        m["observer_radius_m"] = float(observer_radii[i])
        m["far_scenario_horizon_deg"] = far_h[i].tolist()
        m["far_nominal_horizon_deg"] = far_nom[i].tolist()
        m["far_markov_bound"] = far_risk[i].tolist()
        m["full_horizon_deg"] = full[i].tolist()
        m["winner_by_bin"] = ["near" if bool(v) else "far" for v in nw[i]]
        ev["members"][i] = m
        dig.update(Y.canonical_bytes(m))
    y["statistical_horizon_binding"]["memberwise_solver_digest_sha256"] = dig.hexdigest()
    y.pop("receipt_sha256", None)
    y["receipt_sha256"] = Y.sha256_bytes(Y.canonical_bytes(y))
    return y

def materialize(afp, aep, yp, qp, vp, wp, xpp, xp, rp, lp, byp, scp, op, bsp, root):
    AF = A.validate_policy(A.rd(afp, "AF policy"))
    AE = importlib.import_module("materialize_ll009ae_rms_stress_sensitivity")
    Y = importlib.import_module("materialize_ll009y_memberwise_hybrid_horizon")
    S = importlib.import_module("generate_ll009s_semantic_visibility")
    ae = AE.validate_policy(AE.read_obj(aep, "AE policy"))
    lambdas = [float(x) for x in ae["stress_factors"]]
    by = AE.read_obj(byp, "baseline Y")
    bs = AE.read_obj(bsp, "baseline S")
    AE.validate_baselines(by, bs)
    if bs.get("k_horizon_pack_sha256") != A.hf(byp):
        raise A.AFError("baseline S/Y binding mismatch")

    ypol = Y.validate_policy(Y.read_obj(yp, "Y policy"))
    q, v = Y.read_obj(qp, "Q"), Y.read_obj(vp, "V")
    w, x = Y.read_obj(wp, "W"), Y.read_obj(xp, "X")
    r, l = Y.read_obj(rp, "R"), Y.read_obj(lp, "L")
    qi = Y.validate_q(q, ypol, lp)
    fi = Y.validate_v_w_x(v, w, x, vp, wp, xpp, lp, ypol)
    rl = Y.validate_r(r, ypol)
    Y.validate_l(l, ypol, qi, fi)
    if by.get("study_id") != ypol["study_id"]:
        raise A.AFError("baseline Y study mismatch")

    solver_calls, capture, rows, wm = {}, [], [], {}
    scan_t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ll009af-") as td:
        scratch = pathlib.Path(td)
        with A.capture_y_bins(Y, scratch, capture, solver_calls):
            baseline_generated = Y.materialize(yp, qp, vp, wp, xpp, xp, rp, lp, root)
        scan_s = time.perf_counter() - scan_t0
        if Y.canonical_bytes(baseline_generated) != byp.read_bytes():
            raise A.AFError("lambda=1 failed byte-exact Y baseline reproduction")
        if len(capture) != qi["bin_count"]:
            raise A.AFError("capture bin count mismatch")
        np = Y.W.require_rasterio()[0]
        for c in capture:
            b = c["bin_index"]
            xb = fi["x_bins"][b]
            if c["count"] != xb["admitted_pixel_count"] or A.hf(c["path"]) != xb["input_population_digest_sha256"]:
                raise A.AFError("captured Y population does not match exact X bin population")

        bc, mc = qi["bin_count"], qi["member_count"]
        states = [{"h": np.empty((mc, bc)), "r": np.empty((mc, bc)), "n": np.empty((mc, bc))} for _ in lambdas]
        solve_t0 = time.perf_counter()
        for c in capture:
            b = c["bin_index"]
            dt = np.dtype(c["dtype_descr"])
            rec = np.memmap(c["path"], dtype=dt, mode="r", shape=(c["count"],))
            for li, lam in enumerate(lambdas):
                scaled, wit = A.scale_with_witness(
                    rec, lam, np, AF["l2_identity_relative_tolerance"],
                    AF["l2_identity_absolute_tolerance"]
                )
                if lam == 1.0:
                    m = by["memberwise_hybrid_evidence"]["members"]
                    states[li]["h"][:, b] = [z["far_scenario_horizon_deg"][b] for z in m]
                    states[li]["r"][:, b] = [z["far_markov_bound"][b] for z in m]
                    states[li]["n"][:, b] = [z["far_nominal_horizon_deg"][b] for z in m]
                else:
                    sh, sr, sn = Y.solve_members_for_bin(
                        scaled, c["observer_radii"], c["alpha_bin"], c["uniform_k"],
                        c["iterations"], c["risk_tolerance"], c["member_batch_size"],
                        c["pixel_chunk_size"], np
                    )
                    solver_calls[(lam, b)] = solver_calls.get((lam, b), 0) + 1
                    states[li]["h"][:, b] = sh
                    states[li]["r"][:, b] = sr
                    states[li]["n"][:, b] = sn
                wm[(li, b)] = wit
                del scaled
            del rec
        solve_s = time.perf_counter() - solve_t0

        for lam in lambdas:
            for b in range(bc):
                if solver_calls.get((lam, b), 0) != 1:
                    raise A.AFError(f"missing/duplicated solver call lambda={lam} bin={b}")

        for li, lam in enumerate(lambdas):
            y = by if lam == 1.0 else rebuild_y(
                Y, by, ypol, qi, rl, fi, capture[0]["observer_radii"],
                states[li]["h"], states[li]["r"], states[li]["n"], np
            )
            ybytes = Y.canonical_bytes(y)
            if lam == 1.0 and ybytes != byp.read_bytes():
                raise A.AFError("lambda=1 Y identity drift")
            ytmp = scratch / f"y-{li:02d}.json"
            ytmp.write_bytes(ybytes)
            s = S.generate(scp, ytmp, op, rp, root)
            sbytes = AE.canonical_bytes(s)
            if lam == 1.0 and sbytes != bsp.read_bytes():
                raise A.AFError("lambda=1 failed byte-exact S baseline reproduction")
            rows.append({
                "lambda": lam,
                "conservative": lam >= 1.0,
                "lambda_semantics": "stress_or_baseline" if lam >= 1.0 else "diagnostic_non_conservative",
                "internal_counterfactual_y_sha256": A.hb(ybytes),
                "internal_counterfactual_s_sha256": A.hb(sbytes),
                "horizon_deg": AE.y_horizon_vector(y),
                "metrics": AE.extract_metric_central(s),
            })
        popdig = [A.hf(c["path"]) for c in capture]
        maxbytes = max((c["count"] * np.dtype(c["dtype_descr"]).itemsize for c in capture), default=0)

    AE.check_frontier_monotonicity(rows, float(ae["horizon_monotonicity_tolerance_deg"]))
    base = rows[lambdas.index(1.0)]
    for row in rows:
        row["max_horizon_increase_vs_lambda1_deg"] = max(
            (b - a for a, b in zip(base["horizon_deg"], row["horizon_deg"])), default=0.0
        )
        row["metric_delta_vs_lambda1"] = {
            k: row["metrics"][k] - base["metrics"][k]
            for k in set(base["metrics"]) & set(row["metrics"])
        }

    manifest = A.build_manifest(by["study_id"], lambdas, len(capture), wm, popdig)
    for li, lam in enumerate(lambdas):
        b2 = math.fsum(wm[(li, b)]["baseline_rms_l2_sq"] for b in range(len(capture)))
        s2 = math.fsum(wm[(li, b)]["stressed_rms_l2_sq"] for b in range(len(capture)))
        e2 = lam * lam * b2
        err = abs(s2 - e2)
        lim = AF["l2_identity_absolute_tolerance"] + AF["l2_identity_relative_tolerance"] * max(abs(e2), 1.0)
        if err > lim:
            raise A.AFError(f"global L2 identity failed at lambda={lam}")

    manifest_file_sha256 = A.hb(A.cb(manifest))
    out = {
        "schema_version": A.OUT_SCHEMA,
        "producer_schema_version": A.OUT_SCHEMA,
        "status": "pass",
        "semantics_class": A.SEMANTICS,
        "study_id": by["study_id"],
        "lineage": {
            "af_policy_sha256": A.hf(afp), "ae_policy_sha256": A.hf(aep),
            "y_policy_sha256": A.hf(yp), "q_receipt_sha256": A.hf(qp),
            "v_receipt_sha256": A.hf(vp), "w_receipt_sha256": A.hf(wp),
            "x_policy_sha256": A.hf(xpp), "x_receipt_sha256": A.hf(xp),
            "r_receipt_sha256": A.hf(rp), "l_config_sha256": A.hf(lp),
            "baseline_y_k_pack_sha256": A.hf(byp), "s_config_sha256": A.hf(scp),
            "o_receipt_sha256": A.hf(op), "baseline_s_receipt_sha256": A.hf(bsp),
        },
        "execution_theorem": {
            "source_raster_passes": 1,
            "admitted_population_captured_once_from_exact_y": True,
            "stressed_source_rasters_persisted": False,
            "solver_calls_per_lambda_bin": 1,
            "lambda_count": len(lambdas),
            "bin_count": len(capture),
            "intervention_witness_manifest_body_sha256": manifest["manifest_sha256"],
            "intervention_witness_manifest_file_sha256": manifest_file_sha256,
            "intervention_witness_entry_count": manifest["entry_count"],
        },
        "baseline_reproduction": {"lambda": 1.0, "y_byte_exact": True, "s_byte_exact": True},
        "frontier": rows,
        "engineering_threshold_brackets": AE.first_threshold_brackets(rows, ae.get("engineering_thresholds", {})),
        "probability_calibration_claim": False,
        "joint_q_far_probability_claim": False,
        "real_world_calibration_multiplier_selected": False,
        "claim_rule": "AF is numerically the same sensitivity experiment as LL-009AE, but the exact admitted Product90 population is captured from one exact Y scan and every ephemeral lambda/bin RMS intervention is cryptographically witnessed.",
        "non_claims": [
            "No lambda is identified as the true Product90 calibration multiplier.",
            "Lambda below one is diagnostic and is not conservative.",
            "No Gaussian, independence, covariance, or Q-by-far joint probability assumption is introduced.",
            "LL-009R spatial-support limitations remain unchanged.",
            "Intervention witnesses prove the declared admitted-record transform; they do not calibrate Product90 RMS.",
            "Performance diagnostics have no scientific authority.",
        ],
    }
    out["receipt_sha256"] = A.hb(A.cb(out))
    perf = {
        "schema_version": A.PERF_SCHEMA,
        "scientific_authority": False,
        "study_id": by["study_id"],
        "source_raster_passes": 1,
        "scan_and_baseline_y_wall_clock_s": scan_s,
        "stress_solver_wall_clock_s": solve_s,
        "max_captured_baseline_bin_bytes": maxbytes,
        "max_ephemeral_stressed_bin_copy_bytes": maxbytes,
        "member_batch_size": int(ypol["member_batch_size"]),
        "pixel_chunk_size": int(ypol["pixel_chunk_size"]),
        "note": "Wall-clock values are descriptive runtime observations and are not part of scientific authority.",
    }
    return out, manifest, perf
