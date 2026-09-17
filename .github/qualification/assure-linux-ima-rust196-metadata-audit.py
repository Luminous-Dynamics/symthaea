#!/usr/bin/env python3
"""Audit offline Cargo metadata against audited local roots and frozen external graph."""
import json
import pathlib
import sys
import tomllib

source = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
capsule = tomllib.loads(pathlib.Path(sys.argv[2]).read_text())
metadata = json.loads(pathlib.Path(sys.argv[3]).read_text())
audit_path = pathlib.Path(sys.argv[4])
expected = (sys.argv[5], sys.argv[6], "")

def ident(pkg):
    return (pkg["name"], pkg["version"], pkg.get("source") or "")

def index(doc, label):
    out = {}
    for pkg in doc.get("package", []):
        key = ident(pkg)
        if key in out:
            raise SystemExit(f"{label}: duplicate package identity: {key!r}")
        out[key] = pkg
    return out

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
        and (source_ref is None or (pkg.get("source") or "") == source_ref)
    ]
    if len(matches) != 1:
        raise SystemExit(f"{label}: dependency must resolve uniquely: {ref!r} -> {matches!r}")
    return matches[0]

source_packages = source.get("package", [])
capsule_packages = capsule.get("package", [])
source_index = index(source, "source lock")
capsule_index = index(capsule, "capsule lock")
local = capsule_index.get(expected)
if local is None:
    raise SystemExit(f"capsule lock missing expected local package: {expected!r}")
if [ident(p) for p in capsule_packages if not (p.get("source") or "")] != [expected]:
    raise SystemExit("capsule lock contains unexpected local packages")

local_roots = {
    resolve(ref, capsule_packages, f"capsule local {expected!r}")
    for ref in local.get("dependencies", [])
}
if any(not key[2] or key not in source_index for key in local_roots):
    raise SystemExit(f"capsule local root is not a frozen external identity: {sorted(local_roots)!r}")

source_edges = {}
for key, pkg in source_index.items():
    if key[2]:
        source_edges[key] = {
            resolve(ref, source_packages, f"source lock {key!r}")
            for ref in pkg.get("dependencies", [])
        }

by_id = {}
identity_by_id = {}
local_ids = []
for pkg in metadata.get("packages", []):
    pkg_id = pkg["id"]
    if pkg_id in by_id:
        raise SystemExit(f"duplicate metadata package id: {pkg_id}")
    key = ident(pkg)
    if key not in capsule_index:
        raise SystemExit(f"metadata package absent audited capsule lock: {key!r}")
    if not key[2]:
        if key != expected:
            raise SystemExit(f"unexpected local metadata package: {key!r}")
        manifest = pkg.get("manifest_path", "").replace("\\", "/")
        if not manifest.endswith("/crates/domains/symthaea-linux-ima-replay/Cargo.toml"):
            raise SystemExit(f"unexpected local manifest path: {manifest!r}")
        local_ids.append(pkg_id)
    by_id[pkg_id] = pkg
    identity_by_id[pkg_id] = key

if len(local_ids) != 1:
    raise SystemExit(f"expected exactly one local metadata package: {local_ids!r}")
if metadata.get("workspace_members", []) != local_ids:
    raise SystemExit("offline workspace_members does not equal the sole audited local package")
default_members = metadata.get("workspace_default_members")
if default_members is not None and default_members != local_ids:
    raise SystemExit("offline workspace_default_members does not equal the sole audited local package")

resolve_graph = metadata.get("resolve")
if not resolve_graph:
    raise SystemExit("cargo metadata missing resolve graph")
nodes = resolve_graph.get("nodes", [])
if {n["id"] for n in nodes} != set(by_id):
    raise SystemExit("cargo metadata resolve-node/package identity mismatch")

local_edges = 0
external_edges = 0
for node in nodes:
    src = identity_by_id[node["id"]]
    for dst_id in node.get("dependencies", []):
        if dst_id not in identity_by_id:
            raise SystemExit(f"resolved dependency id absent package set: {dst_id}")
        dst = identity_by_id[dst_id]
        if not src[2]:
            local_edges += 1
            if not dst[2] or dst not in local_roots:
                raise SystemExit(f"unaudited local root edge: {src!r} -> {dst!r}")
        else:
            external_edges += 1
            if not dst[2]:
                raise SystemExit(f"forbidden external-to-local edge: {src!r} -> {dst!r}")
            if dst not in source_edges.get(src, set()):
                raise SystemExit(f"external edge absent frozen source graph: {src!r} -> {dst!r}")

audit = {
    "schema": "symthaea.assurance.offline-resolved-graph-audit.v2",
    "result": "PASS",
    "cargo_net_offline": True,
    "local_package_provenance": "exact-git-object-source-census",
    "resolved_packages": len(by_id),
    "resolved_local_root_edges": local_edges,
    "resolved_external_edges": external_edges,
    "local_package": {"name": expected[0], "version": expected[1], "source": expected[2]},
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(
    "ima_offline_resolved_graph_audit=PASS "
    f"packages={len(by_id)} local_edges={local_edges} external_edges={external_edges}"
)
