#!/usr/bin/env python3
"""Audit offline Cargo metadata for a focused multi-package closure."""
import json, pathlib, sys, tomllib

source_path = pathlib.Path(sys.argv[1])
capsule_path = pathlib.Path(sys.argv[2])
metadata_path = pathlib.Path(sys.argv[3])
audit_path = pathlib.Path(sys.argv[4])
closure_path = pathlib.Path(sys.argv[5])
subject_name = sys.argv[6]

source = tomllib.loads(source_path.read_text())
capsule = tomllib.loads(capsule_path.read_text())
metadata = json.loads(metadata_path.read_text())

def ident(pkg):
    return (pkg["name"], pkg["version"], pkg.get("source") or "")

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
        and (source_ref is None or (pkg.get("source") or "") == source_ref)
    ]
    if len(matches) != 1:
        raise SystemExit(f"{label}: dependency must resolve uniquely: {ref!r} -> {matches!r}")
    return matches[0]

closure_rows = [line.split("\t") for line in closure_path.read_text().splitlines()[1:] if line]
closure_paths = {row[0]: row[1] for row in closure_rows}
closure_names = set(closure_paths)
if len(closure_paths) != len(closure_rows):
    raise SystemExit("closure contains duplicate package names")
if subject_name not in closure_names:
    raise SystemExit(f"subject package absent focused closure: {subject_name}")

source_packages, source_index = idx(source, "source lock")
capsule_packages, capsule_index = idx(capsule, "capsule lock")
capsule_locals = {ident(p): p for p in capsule_packages if not (p.get("source") or "")}
if {k[0] for k in capsule_locals} != closure_names:
    raise SystemExit("capsule local packages do not equal focused closure")

allowed_local_edges = {}
for src, pkg in capsule_locals.items():
    allowed_local_edges[src] = {
        resolve(ref, capsule_packages, f"capsule local {src!r}")
        for ref in pkg.get("dependencies", [])
    }

source_edges = {}
for src, pkg in source_index.items():
    if src[2]:
        source_edges[src] = {
            resolve(ref, source_packages, f"source lock {src!r}")
            for ref in pkg.get("dependencies", [])
        }

by_id = {}
identity_by_id = {}
local_ids = set()
subject_ids = set()
for pkg in metadata.get("packages", []):
    pkg_id = pkg["id"]
    if pkg_id in by_id:
        raise SystemExit(f"duplicate metadata package id: {pkg_id}")
    key = ident(pkg)
    if key not in capsule_index:
        raise SystemExit(f"metadata package absent audited capsule lock: {key!r}")
    if not key[2]:
        if key[0] not in closure_names:
            raise SystemExit(f"unexpected local metadata package: {key!r}")
        manifest = pkg.get("manifest_path", "").replace("\\", "/")
        suffix = "/" + closure_paths[key[0]].strip("/") + "/Cargo.toml"
        if not manifest.endswith(suffix):
            raise SystemExit(f"local manifest path mismatch for {key[0]}: {manifest!r}")
        local_ids.add(pkg_id)
        if key[0] == subject_name:
            subject_ids.add(pkg_id)
    by_id[pkg_id] = pkg
    identity_by_id[pkg_id] = key

if {identity_by_id[x][0] for x in local_ids} != closure_names:
    raise SystemExit("offline metadata local package set does not equal focused closure")
if set(metadata.get("workspace_members", [])) != local_ids:
    raise SystemExit("offline workspace_members does not equal focused closure")
defaults = set(metadata.get("workspace_default_members") or [])
if defaults and defaults != subject_ids:
    raise SystemExit("offline default member is not exactly the subject package")

resolve_graph = metadata.get("resolve")
if not resolve_graph:
    raise SystemExit("cargo metadata missing resolve graph")
nodes = resolve_graph.get("nodes", [])
if {n["id"] for n in nodes} != set(by_id):
    raise SystemExit("metadata resolve-node/package identity mismatch")

ll = le = ee = 0
for node in nodes:
    src = identity_by_id[node["id"]]
    for dst_id in node.get("dependencies", []):
        if dst_id not in identity_by_id:
            raise SystemExit(f"resolved dependency id absent package set: {dst_id}")
        dst = identity_by_id[dst_id]
        if not src[2]:
            if dst not in allowed_local_edges.get(src, set()):
                raise SystemExit(f"offline local edge absent audited capsule graph: {src!r} -> {dst!r}")
            if dst[2]:
                le += 1
            else:
                ll += 1
        else:
            ee += 1
            if not dst[2]:
                raise SystemExit(f"forbidden offline external-to-local edge: {src!r} -> {dst!r}")
            if dst not in source_edges.get(src, set()):
                raise SystemExit(f"offline external edge absent frozen source graph: {src!r} -> {dst!r}")

audit = {
    "schema": "symthaea.assurance.focused-closure-offline-metadata-audit.v1",
    "result": "PASS",
    "cargo_net_offline": True,
    "focused_local_packages": sorted(closure_names),
    "focused_local_package_count": len(closure_names),
    "resolved_packages": len(by_id),
    "local_to_local_edges_verified": ll,
    "local_to_external_edges_verified": le,
    "external_to_external_edges_verified": ee,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(
    "focused_offline_metadata_audit=PASS "
    f"local={len(closure_names)} packages={len(by_id)} "
    f"local_local={ll} local_external={le} external_external={ee}"
)
