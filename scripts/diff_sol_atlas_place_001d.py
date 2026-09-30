#!/usr/bin/env python3
"""Differential gate for the Sol Atlas -> CIV-PLACE adapter.

The Python side intentionally re-derives the projection/evaluation contract
from the fixture rather than importing Rust semantics. Rust remains the
qualified implementation under test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs" / "engineering" / "fixtures" / "sol-atlas-place-001d.json"
BIN = ROOT / "target" / "debug" / "sol-atlas-place-001d-oracle"


def canonical_plan(plan: dict) -> bytes:
    p = copy.deepcopy(plan)
    p["site"]["external_refs"].sort(key=lambda x: (x["namespace"], x["external_id"]))
    p["source_snapshots"].sort(key=lambda x: (x["id"], x["provider"], x["profile"], x["release"]))
    p["intent"]["non_goals"].sort()
    p["elements"].sort(key=lambda x: (
        x["id"], x["kind"], x.get("parent_id"), x.get("geometry_ref")
    ))
    for e in p["elements"]:
        e["external_refs"].sort(key=lambda x: (x["namespace"], x["external_id"]))
    p["dependencies"].sort(key=lambda x: (
        x["id"], x["from_id"], x["to_id"], x["service"], x["required"], x.get("common_mode_group")
    ))
    p["assumptions"].sort(key=lambda x: (
        x["id"], x["statement"], x["status"], x.get("source_ref")
    ))
    return json.dumps(p, sort_keys=True, separators=(",", ":")).encode()


def project_py(inp: dict) -> dict:
    plan = inp["plan"]
    nodes = {
        b["element_id"]: {
            "id": b["element_id"],
            "currentness": b["currentness"],
            "projection": b["projection"],
            "state": b["state"],
        }
        for b in inp["node_bindings"]
    }
    deps = sorted(
        [{
            "id": d["id"],
            "source": d["to_id"],
            "target": d["from_id"],
            "class": d["service"],
            "required": d["required"],
            "common_mode": d.get("common_mode_group"),
        } for d in plan["dependencies"]],
        key=lambda d: d["id"],
    )
    projections = sorted(inp["projections"], key=lambda p: p["id"])
    services = []
    for s in inp["services"]:
        services.append({
            "id": s["id"],
            "currentness_required": s["currentness_required"],
            "dependency_discovery": s["dependency_discovery"],
            "included_dependency_classes": sorted(set(s["included_dependency_classes"])),
            "required_dependency_classes": sorted(set(s["required_dependency_classes"])),
            "root": s["root_element_id"],
        })
    services.sort(key=lambda s: s["id"])

    receipt = {
        "profile": "SOL-PLACE-001D",
        "plan_id": plan["plan_id"],
        "plan_version": plan["version"],
        "plan_sha256": hashlib.sha256(canonical_plan(plan)).hexdigest(),
        "projected_dependency_ids": [d["id"] for d in deps],
        "service_ids": [s["id"] for s in services],
        "kernel_schema_version": "civ-place-001b-v1",
        "claim_ceiling": plan["claim_ceiling"],
    }
    return {
        "nodes": [nodes[k] for k in sorted(nodes)],
        "dependencies": deps,
        "projections": projections,
        "services": services,
        "receipt": receipt,
    }


def evaluate_py(inp: dict, projection: dict) -> dict:
    nodes = {n["id"]: n for n in projection["nodes"]}
    projections = {p["id"]: p for p in projection["projections"]}
    deps = projection["dependencies"]
    by_target = {}
    for d in deps:
        by_target.setdefault(d["target"], []).append(d)
    for values in by_target.values():
        values.sort(key=lambda d: d["id"])

    out = {}
    for service in projection["services"]:
        seen = set()
        edges = set()
        queue = [service["root"]]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            if len(seen) >= 128:
                break
            seen.add(node)
            for d in by_target.get(node, []):
                if d["class"] not in service["included_dependency_classes"]:
                    continue
                edges.add(d["id"])
                if d["source"] not in seen:
                    queue.append(d["source"])

        edge_by_id = {d["id"]: d for d in deps}
        required = {
            eid for eid in edges
            if edge_by_id[eid]["required"]
            or edge_by_id[eid]["class"] in service["required_dependency_classes"]
        }
        optional = edges - required
        required_sources = {edge_by_id[e]["source"] for e in required}
        optional_sources = {edge_by_id[e]["source"] for e in optional}

        reasons = set()
        currentness = set()
        blocked = False
        conflicted = False
        unavailable = False
        degraded = False
        unresolved = service["dependency_discovery"] != "Complete"
        if unresolved:
            reasons.add(f"DependencyDiscovery:{service['dependency_discovery']}")

        for node_id in sorted(seen):
            node = nodes[node_id]
            p = projections[node["projection"]]
            if node["state"] != "Available":
                if node_id in required_sources or node_id == service["root"]:
                    unavailable = True
                    reasons.add(f"UnavailableNode:{node_id}")
                elif node_id in optional_sources:
                    degraded = True
                    reasons.add(f"OptionalUnavailable:{node_id}")

            currentness.add(node["currentness"])
            if p["contradiction"] != "None":
                conflicted = True
                reasons.add(f"ProjectionConflict:{node['projection']}")
            elif (
                p["completeness"] != "Complete"
                or p["convergence"] != "Converged"
            ) and service["currentness_required"]:
                blocked = True
                reasons.add(f"ProjectionIncomplete:{node['projection']}")

            if service["currentness_required"] and node["currentness"] != "Current":
                if node["currentness"] == "Conflicted":
                    conflicted = True
                else:
                    blocked = True
                reasons.add(f"NodeCurrentness:{node_id}:{node['currentness']}")

        status = (
            "Conflicted" if conflicted else
            "Blocked" if blocked else
            "Unavailable" if unavailable else
            "UnresolvedDependencies" if unresolved else
            "DegradedService" if degraded else
            "FullService"
        )
        out[service["id"]] = {
            "service": service["id"],
            "status": status,
            "closure": {
                "root": service["root"],
                "nodes": sorted(seen),
                "edges": sorted(edges),
                "bounded": len(seen) >= 128,
                "common_mode_groups": sorted({
                    d["common_mode"] for d in deps
                    if d["id"] in edges and d["common_mode"] is not None
                }),
            },
            "currentness": sorted(currentness),
            "reasons": sorted(reasons),
        }
    return out


def run_rust(inp: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
        json.dump(inp, f, separators=(",", ":"))
        path = f.name
    try:
        proc = subprocess.run([str(BIN), path], cwd=ROOT, text=True, capture_output=True, check=False)
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit("Sol Atlas Rust oracle failed")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("Sol Atlas Rust oracle emitted no JSON")
    return json.loads(lines[-1])


def expected_semantics(inp: dict) -> dict:
    projection = project_py(inp)
    return {
        "profile": inp["profile"],
        "schema_version": inp["plan"]["schema_version"],
        "projection": projection,
        "evaluations": evaluate_py(inp, projection),
    }


def actual_semantics(rust: dict) -> dict:
    return {
        "profile": rust["profile"],
        "schema_version": rust["schema_version"],
        "projection": rust["projection"],
        "evaluations": rust["evaluations"],
    }


def main() -> int:
    subprocess.run(
        ["cargo", "build", "-p", "symthaea-civ-place", "--bin", "sol-atlas-place-001d-oracle", "--quiet"],
        cwd=ROOT, check=True,
    )

    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = []

    cases.append(("nominal", base))

    permutation = copy.deepcopy(base)
    permutation["plan"]["elements"].reverse()
    permutation["plan"]["dependencies"].reverse()
    permutation["node_bindings"].reverse()
    permutation["services"].reverse()
    cases.append(("permutation", permutation))

    stale = copy.deepcopy(base)
    stale["node_bindings"][3]["currentness"] = "Stale"
    cases.append(("stale_home", stale))

    partial = copy.deepcopy(base)
    partial["projections"][0]["completeness"] = "Partial"
    cases.append(("partial_projection", partial))

    incomplete = copy.deepcopy(base)
    incomplete["services"][0]["dependency_discovery"] = "Partial"
    cases.append(("incomplete_dependency_discovery", incomplete))

    unrelated = copy.deepcopy(base)
    unrelated["plan"]["geometry_revision"] = "geom:place-r4-unrelated"
    cases.append(("unrelated_geometry_revision", unrelated))

    results = []
    for name, case in cases:
        expected = expected_semantics(case)
        rust = run_rust(case)
        actual = actual_semantics(rust)
        if actual != expected:
            print(f"DIFFERENTIAL FAIL: {name}", file=sys.stderr)
            print(json.dumps({"expected": expected, "actual": actual}, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        results.append({
            "case": name,
            "plan_sha256": rust["plan_sha256"],
            "statuses": {k: v["status"] for k, v in rust["evaluations"].items()},
        })

    if results[0]["statuses"]["svc:home-01:electric"] != "FullService":
        raise SystemExit("nominal case did not remain FullService")
    if results[2]["statuses"]["svc:home-01:electric"] != "Blocked":
        raise SystemExit("stale home did not reopen the service")
    if results[3]["statuses"]["svc:home-01:electric"] != "Blocked":
        raise SystemExit("partial projection did not block currentness-required service")
    if results[4]["statuses"]["svc:home-01:electric"] != "UnresolvedDependencies":
        raise SystemExit("incomplete dependency discovery was promoted")

    print(json.dumps({
        "result": "PASS",
        "profile": "SOL-PLACE-001D",
        "cases": results,
        "python_reference": "PASS",
        "rust_adapter": "PASS",
        "differential": "PASS",
        "authority_claim": "none",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
