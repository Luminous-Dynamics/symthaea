#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import pathlib
from typing import Any

POLICY_SCHEMA = "ll009af.single-scan-rms-stress-policy.v1"
OUT_SCHEMA = "ll009af.single-scan-rms-stress-frontier-receipt.v1"
WITNESS_SCHEMA = "ll009af.intervention-witness-manifest.v1"
PERF_SCHEMA = "ll009af.performance-diagnostic.v1"
SEMANTICS = "model_conditional_rms_stress_sensitivity_frontier"
NON_RMS_FIELDS = ("c", "radius_m", "row", "col", "nominal_elevation_deg")

class AFError(RuntimeError):
    pass

def cb(v: Any) -> bytes:
    return (json.dumps(v, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()

def hb(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hf(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def rd(p: pathlib.Path, n: str) -> dict[str, Any]:
    try:
        v = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise AFError(f"cannot read {n}: {e}") from e
    if not isinstance(v, dict):
        raise AFError(f"{n} must contain object")
    return v

def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def validate_policy(p):
    if p.get("schema_version") != POLICY_SCHEMA:
        raise AFError("invalid AF schema")
    if p.get("semantics_class") != SEMANTICS:
        raise AFError("AF must preserve AE semantics")
    if p.get("stress_source") != "inherit_exact_ll009ae_policy":
        raise AFError("AF must inherit exact AE stress ladder")
    if p.get("raster_source_passes_required") != 1:
        raise AFError("AF requires one source-raster pass")
    if p.get("intervention") != "multiply_only_rms_m_in_ephemeral_float64_bin_copy":
        raise AFError("AF intervention drift")
    if p.get("persist_stressed_rasters") is not False:
        raise AFError("stressed rasters may not be persisted")
    if p.get("baseline_reproduction") != "byte_exact_y_and_s_at_lambda_1":
        raise AFError("lambda=1 identity required")
    if p.get("production_ae_reference_requirement") != "not_required_after_synthetic_equivalence_gate":
        raise AFError("AE reference policy drift")
    for k in ("l2_identity_relative_tolerance", "l2_identity_absolute_tolerance"):
        if not finite(p.get(k)) or float(p[k]) < 0:
            raise AFError(f"invalid {k}")
    return p

def non_rms_bytes(records, np):
    dt = np.dtype([
        ("c", "<f8"),
        ("radius_m", "<f8"),
        ("row", "<i8"),
        ("col", "<i8"),
        ("nominal_elevation_deg", "<f8"),
    ])
    out = np.empty(len(records), dtype=dt)
    for k in NON_RMS_FIELDS:
        out[k] = records[k]
    return out.tobytes(order="C")

def scale_with_witness(records, lam, np, rel_tol, abs_tol):
    lam = float(lam)
    base_geom = hb(non_rms_bytes(records, np))
    base_rms = np.asarray(records["rms_m"], dtype="<f8")
    if np.any(~np.isfinite(base_rms)) or np.any(base_rms < 0):
        raise AFError("invalid baseline RMS")
    scaled = np.array(records, copy=True)
    expected = np.asarray(base_rms * lam, dtype="<f8")
    scaled["rms_m"] = expected
    stressed = np.asarray(scaled["rms_m"], dtype="<f8")
    if hb(non_rms_bytes(scaled, np)) != base_geom:
        raise AFError("non-RMS mutation detected")
    if not np.array_equal(stressed.view("<u8"), expected.view("<u8")):
        raise AFError("float64 scaling mismatch")
    maxerr = float(np.max(np.abs(stressed - expected))) if len(stressed) else 0.0
    if maxerr != 0.0:
        raise AFError("per-element RMS scaling not exact")
    b2 = float(np.sum(base_rms * base_rms, dtype="float64"))
    s2 = float(np.sum(stressed * stressed, dtype="float64"))
    e2 = float(lam * lam * b2)
    err = abs(s2 - e2)
    lim = float(abs_tol) + float(rel_tol) * max(abs(e2), 1.0)
    if err > lim:
        raise AFError(f"L2 scaling identity failed: {err} > {lim}")
    return scaled, {
        "record_count": int(len(records)),
        "non_rms_geometry_identity_sha256": base_geom,
        "baseline_rms_sha256": hb(base_rms.tobytes(order="C")),
        "stressed_rms_sha256": hb(stressed.tobytes(order="C")),
        "max_abs_error_vs_float64_lambda_times_baseline_rms": maxerr,
        "baseline_rms_l2_sq": b2,
        "stressed_rms_l2_sq": s2,
        "expected_stressed_rms_l2_sq": e2,
        "l2_identity_abs_error": err,
        "float_representation": "ieee754_binary64_little_endian",
    }

@contextlib.contextmanager
def capture_y_bins(Y, scratch: pathlib.Path, capture: list[dict[str, Any]], solver_calls: dict):
    original = Y.solve_members_for_bin
    def wrapped(records, observer_radii, alpha_bin, uniform_k, iterations, risk_tolerance,
                member_batch_size, pixel_chunk_size, np):
        b = len(capture)
        path = scratch / f"captured-bin-{b:04d}.bin"
        path.write_bytes(records.tobytes(order="C"))
        capture.append({
            "bin_index": b,
            "path": path,
            "count": int(len(records)),
            "dtype_descr": records.dtype.descr,
            "observer_radii": np.asarray(observer_radii, dtype="float64").copy(),
            "alpha_bin": float(alpha_bin),
            "uniform_k": float(uniform_k),
            "iterations": int(iterations),
            "risk_tolerance": float(risk_tolerance),
            "member_batch_size": int(member_batch_size),
            "pixel_chunk_size": int(pixel_chunk_size),
        })
        solver_calls[(1.0, b)] = solver_calls.get((1.0, b), 0) + 1
        return original(
            records, observer_radii, alpha_bin, uniform_k, iterations,
            risk_tolerance, member_batch_size, pixel_chunk_size, np
        )
    Y.solve_members_for_bin = wrapped
    try:
        yield
    finally:
        Y.solve_members_for_bin = original

def build_manifest(study, lambdas, bin_count, wm, popdig):
    entries = []
    for li, lam in enumerate(lambdas):
        for b in range(bin_count):
            e = copy.deepcopy(wm[(li, b)])
            e["lambda"] = lam
            e["bin_index"] = b
            e["baseline_full_record_population_sha256"] = popdig[b]
            entries.append(e)
    m = {
        "schema_version": WITNESS_SCHEMA,
        "study_id": study,
        "ordered_by": "lambda_ascending_then_bin_index_ascending",
        "entry_count": len(entries),
        "entries": entries,
    }
    m["manifest_sha256"] = hb(cb(m))
    return m

def wr(p, v):
    b = cb(v)
    if p.exists() and p.read_bytes() != b:
        raise AFError(f"refusing to overwrite differing output {p}")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)

def wr_diagnostic(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(cb(v))
