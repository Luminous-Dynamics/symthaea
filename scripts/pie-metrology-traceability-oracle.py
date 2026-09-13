#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib, json, math

class ModelError(ValueError):
    pass

LOCAL_PRIMARY = "LocalPrimary"
IMPORTED_PRIMARY = "ImportedPrimary"
DERIVED = "Derived"
QUAL_LOCAL = "QualifiedLocalTraceability"
QUAL_IMPORT = "QualifiedImportTraceability"
INSUFFICIENT = "InsufficientPrecision"
EXPIRED = "Expired"
UNREACHABLE = "Unreachable"
EPS = 1e-12

@dataclass(frozen=True)
class Reference:
    id: str
    parent: Optional[str]
    kind: str
    bound_at_calibration: float
    drift_bound_per_step: float
    calibrated_at: int
    valid_through: int

@dataclass(frozen=True)
class Decision:
    id: str
    leaf_reference: str
    allowed_total_bound: float

@dataclass(frozen=True)
class Evaluation:
    decision_id: str
    status: str
    total_bound: Optional[float]
    anchor_id: Optional[str]
    path: tuple[str, ...]
    digest: str

def _finite_nonneg(x: float, name: str) -> None:
    if not math.isfinite(x) or x < 0:
        raise ModelError(f"{name} must be finite and non-negative")

def validate(refs: list[Reference], decisions: list[Decision]) -> dict[str, Reference]:
    if not refs:
        raise ModelError("empty reference registry")
    by_id: dict[str, Reference] = {}
    for r in refs:
        if not r.id or r.id in by_id:
            raise ModelError("duplicate/empty reference id")
        if r.kind not in (LOCAL_PRIMARY, IMPORTED_PRIMARY, DERIVED):
            raise ModelError("unknown reference kind")
        _finite_nonneg(r.bound_at_calibration, "bound_at_calibration")
        _finite_nonneg(r.drift_bound_per_step, "drift_bound_per_step")
        if r.calibrated_at < 0 or r.valid_through < r.calibrated_at:
            raise ModelError("invalid reference timing")
        if r.kind in (LOCAL_PRIMARY, IMPORTED_PRIMARY) and r.parent is not None:
            raise ModelError("primary anchor cannot have parent")
        if r.kind == DERIVED and not r.parent:
            raise ModelError("derived reference requires parent")
        if r.parent == r.id:
            raise ModelError("direct self traceability")
        by_id[r.id] = r
    for r in refs:
        if r.parent is not None and r.parent not in by_id:
            raise ModelError("unknown parent reference")
    seen_decisions = set()
    for d in decisions:
        if not d.id or d.id in seen_decisions:
            raise ModelError("duplicate/empty decision id")
        seen_decisions.add(d.id)
        if d.leaf_reference not in by_id:
            raise ModelError("unknown decision leaf")
        _finite_nonneg(d.allowed_total_bound, "allowed_total_bound")
    for start in by_id:
        visiting = set()
        cur = start
        while True:
            if cur in visiting:
                raise ModelError("traceability cycle")
            visiting.add(cur)
            parent = by_id[cur].parent
            if parent is None:
                break
            cur = parent
    return by_id

def _component_bound(r: Reference, step: int) -> float:
    if step < r.calibrated_at:
        raise ModelError("evaluation before calibration")
    age = step - r.calibrated_at
    return r.bound_at_calibration + r.drift_bound_per_step * age

def trace_path(by_id: dict[str, Reference], leaf: str) -> tuple[list[Reference], Optional[str]]:
    path = []
    cur = by_id.get(leaf)
    if cur is None:
        return [], None
    visited = set()
    while cur is not None:
        if cur.id in visited:
            raise ModelError("traceability cycle")
        visited.add(cur.id)
        path.append(cur)
        if cur.parent is None:
            if cur.kind not in (LOCAL_PRIMARY, IMPORTED_PRIMARY):
                return path, None
            return path, cur.kind
        cur = by_id.get(cur.parent)
        if cur is None:
            return path, None
    return path, None

def evaluate(refs: list[Reference], decisions: list[Decision], decision_id: str, step: int) -> Evaluation:
    by_id = validate(refs, decisions)
    dmap = {d.id: d for d in decisions}
    if decision_id not in dmap:
        raise ModelError("unknown decision")
    if step < 0:
        raise ModelError("negative step")
    d = dmap[decision_id]
    path, anchor_kind = trace_path(by_id, d.leaf_reference)
    ids = tuple(r.id for r in path)
    payload = {
        "decision": asdict(d),
        "step": step,
        "path": [asdict(r) for r in path],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not path or anchor_kind is None:
        return Evaluation(d.id, UNREACHABLE, None, None, ids, digest)
    if any(step > r.valid_through for r in path):
        return Evaluation(d.id, EXPIRED, None, path[-1].id, ids, digest)
    total = sum(_component_bound(r, step) for r in path)
    if total > d.allowed_total_bound + EPS:
        return Evaluation(d.id, INSUFFICIENT, total, path[-1].id, ids, digest)
    status = QUAL_LOCAL if anchor_kind == LOCAL_PRIMARY else QUAL_IMPORT
    return Evaluation(d.id, status, total, path[-1].id, ids, digest)

def self_test() -> None:
    refs = [
        Reference("earth_master", None, IMPORTED_PRIMARY, 0.10, 0.01, 0, 10),
        Reference("transfer", "earth_master", DERIVED, 0.05, 0.005, 0, 10),
        Reference("shop", "transfer", DERIVED, 0.05, 0.005, 0, 8),
        Reference("local_primary", None, LOCAL_PRIMARY, 0.15, 0.005, 0, 20),
        Reference("local_shop", "local_primary", DERIVED, 0.05, 0.005, 0, 20),
        Reference("coarse_local_primary", None, LOCAL_PRIMARY, 0.40, 0.01, 0, 20),
        Reference("coarse_shop", "coarse_local_primary", DERIVED, 0.20, 0.01, 0, 20),
    ]
    decisions = [
        Decision("import_decision", "shop", 0.40),
        Decision("local_decision", "local_shop", 0.30),
        Decision("coarse_decision", "coarse_shop", 0.50),
    ]
    e0 = evaluate(refs, decisions, "import_decision", 0)
    assert e0.status == QUAL_IMPORT and abs(e0.total_bound - 0.20) < EPS
    l0 = evaluate(refs, decisions, "local_decision", 0)
    assert l0.status == QUAL_LOCAL and abs(l0.total_bound - 0.20) < EPS
    c0 = evaluate(refs, decisions, "coarse_decision", 0)
    assert c0.status == INSUFFICIENT and c0.total_bound > 0.50
    e5 = evaluate(refs, decisions, "import_decision", 5)
    assert e5.total_bound >= e0.total_bound and e5.status in (QUAL_IMPORT, INSUFFICIENT)

    eq_refs = [
        Reference("p", None, LOCAL_PRIMARY, 0.20, 0.0, 0, 10),
        Reference("d", "p", DERIVED, 0.10, 0.0, 0, 10),
    ]
    eq = evaluate(eq_refs, [Decision("eq", "d", 0.30)], "eq", 0)
    assert eq.status == QUAL_LOCAL and abs(eq.total_bound - 0.30) < EPS

    ex = evaluate(refs, decisions, "import_decision", 11)
    assert ex.status == EXPIRED

    broken = [r for r in refs if r.id != "earth_master"]
    try:
        evaluate(broken, decisions, "import_decision", 0)
        raise AssertionError("unknown parent should fail")
    except ModelError:
        pass

    cyc = [
        Reference("a", "b", DERIVED, 0.1, 0.0, 0, 10),
        Reference("b", "a", DERIVED, 0.1, 0.0, 0, 10),
    ]
    try:
        evaluate(cyc, [Decision("x", "a", 1.0)], "x", 0)
        raise AssertionError("cycle should fail")
    except ModelError:
        pass

    d1 = evaluate(refs, decisions, "local_decision", 3).digest
    d2 = evaluate(refs, decisions, "local_decision", 3).digest
    assert d1 == d2
    changed = refs.copy()
    changed[3] = Reference("local_primary", None, LOCAL_PRIMARY, 0.16, 0.005, 0, 20)
    d3 = evaluate(changed, decisions, "local_decision", 3).digest
    assert d3 != d1

    bad = [Reference("bad", None, LOCAL_PRIMARY, float("nan"), 0.0, 0, 1)]
    try:
        evaluate(bad, [Decision("bad_dec", "bad", 1.0)], "bad_dec", 0)
        raise AssertionError("NaN should fail")
    except ModelError:
        pass
    print("ok")

if __name__ == "__main__":
    self_test()
