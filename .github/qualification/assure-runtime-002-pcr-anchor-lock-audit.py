#!/usr/bin/env python3
"""Audit a focused multi-package capsule lock against the frozen external lock graph."""
import json, pathlib, sys, tomllib

source_path, capsule_path, audit_path, closure_path = map(pathlib.Path, sys.argv[1:5])
source = tomllib.loads(source_path.read_text())
capsule = tomllib.loads(capsule_path.read_text())

def ident(pkg):
    return (pkg["name"], pkg["version"], pkg.get("source", ""))

def idx(doc, label):
    packages = doc.get("package", [])
    out = {}
    for pkg in packages:
        key = ident(pkg)
        if key in out:
            raise SystemExit(f"{label}: duplicate package identity: {key!r}")
        out[key] = pkg
    return packages, out

def resolve(ref, packages, label):
    text = ref.strip()
    source_ref = None
    if text.endswith(")"):
        marker = text.rfind(" (")
        if marker < 0:
            raise SystemExit(f"{label}: malformed dependency reference: {ref!r}")
        source_ref = text[marker + 2:-1]
        text = text[:marker]
    parts = text.split()
    if not 1 <= len(parts) <= 2:
        raise SystemExit(f"{label}: malformed dependency reference: {ref!r}")
    name = parts[0]
    version = parts[1] if len(parts) == 2 else None
    matches = [
        ident(pkg) for pkg in packages
        if pkg["name"] == name
        and (version is None or pkg["version"] == version)
        and (source_ref is None or pkg.get("source", "") == source_ref)
    ]
    if len(matches) != 1:
        raise SystemExit(f"{label}: dependency must resolve uniquely: {ref!r} -> {matches!r}")
    return matches[0]

closure_rows = [line.split("\t") for line in closure_path.read_text().splitlines()[1:] if line]
closure_names = {row[0] for row in closure_rows}
if len(closure_names) != len(closure_rows):
    raise SystemExit("closure contains duplicate package names")

source_packages, source_index = idx(source, "source lock")
capsule_packages, capsule_index = idx(capsule, "capsule lock")
capsule_locals = [pkg for pkg in capsule_packages if not pkg.get("source")]
capsule_local_names = {pkg["name"] for pkg in capsule_locals}
if capsule_local_names != closure_names:
    raise SystemExit(
        f"capsule local closure mismatch: expected={sorted(closure_names)!r} "
        f"actual={sorted(capsule_local_names)!r}"
    )

external = []
for pkg in capsule_packages:
    key = ident(pkg)
    if not key[2]:
        continue
    frozen = source_index.get(key)
    if frozen is None:
        raise SystemExit(f"capsule external package absent frozen source lock: {key!r}")
    if pkg.get("checksum") != frozen.get("checksum"):
        raise SystemExit(f"checksum drift for {key!r}")
    external.append(key)

source_edges = {}
for key, pkg in source_index.items():
    if not key[2]:
        continue
    source_edges[key] = {
        resolve(ref, source_packages, f"source lock {key!r}")
        for ref in pkg.get("dependencies", [])
    }

local_local_edges = 0
local_external_edges = 0
external_edges = 0
for pkg in capsule_packages:
    src = ident(pkg)
    for ref in pkg.get("dependencies", []):
        dst = resolve(ref, capsule_packages, f"capsule {src!r}")
        if not src[2]:
            if dst[2]:
                local_external_edges += 1
                if dst not in source_index:
                    raise SystemExit(f"local external root absent frozen source lock: {src!r} -> {dst!r}")
            else:
                local_local_edges += 1
                if dst[0] not in closure_names:
                    raise SystemExit(f"local edge escapes focused closure: {src!r} -> {dst!r}")
            continue
        external_edges += 1
        if not dst[2]:
            raise SystemExit(f"forbidden external-to-local edge: {src!r} -> {dst!r}")
        if dst not in source_edges.get(src, set()):
            raise SystemExit(f"external edge absent frozen source graph: {src!r} -> {dst!r}")

audit = {
    "schema": "symthaea.assurance.focused-closure-lock-graph-audit.v1",
    "result": "PASS",
    "local_package_provenance": "exact-git-object-source-census",
    "source_lock_local_nodes_required": False,
    "focused_local_packages": sorted(closure_names),
    "focused_local_package_count": len(closure_names),
    "capsule_external_packages": len(external),
    "source_package_count": len(source_packages),
    "capsule_package_count": len(capsule_packages),
    "local_to_local_edges_verified": local_local_edges,
    "local_to_external_edges_verified": local_external_edges,
    "external_to_external_edges_verified": external_edges,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(
    "focused_lock_graph_audit=PASS "
    f"local={len(closure_names)} external={len(external)} "
    f"local_local={local_local_edges} local_external={local_external_edges} "
    f"external_external={external_edges}"
)
