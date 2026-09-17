#!/usr/bin/env python3
"""Audit minimal capsule lock against frozen external source-lock graph."""
import json
import pathlib
import sys
import tomllib

source_path = pathlib.Path(sys.argv[1])
capsule_path = pathlib.Path(sys.argv[2])
audit_path = pathlib.Path(sys.argv[3])
expected_local_name = sys.argv[4]
expected_local_version = sys.argv[5]

source_doc = tomllib.loads(source_path.read_text())
capsule_doc = tomllib.loads(capsule_path.read_text())

def identity(pkg):
    return (pkg["name"], pkg["version"], pkg.get("source", ""))

def identity_json(key):
    return {"name": key[0], "version": key[1], "source": key[2]}

def build_index(doc, label):
    packages = doc.get("package", [])
    index = {}
    for pkg in packages:
        key = identity(pkg)
        if key in index:
            raise SystemExit(f"{label}: duplicate package identity: {key!r}")
        index[key] = pkg
    return packages, index

def resolve_dep(ref, packages, label):
    text = ref.strip()
    source = None
    if text.endswith(")"):
        marker = text.rfind(" (")
        if marker == -1:
            raise SystemExit(f"{label}: malformed dependency reference: {ref!r}")
        source = text[marker + 2:-1]
        text = text[:marker]
    parts = text.split()
    if not 1 <= len(parts) <= 2:
        raise SystemExit(f"{label}: malformed dependency reference: {ref!r}")
    name = parts[0]
    version = parts[1] if len(parts) == 2 else None
    candidates = []
    for pkg in packages:
        if pkg["name"] != name:
            continue
        if version is not None and pkg["version"] != version:
            continue
        if source is not None and pkg.get("source", "") != source:
            continue
        candidates.append(identity(pkg))
    if len(candidates) != 1:
        raise SystemExit(
            f"{label}: dependency reference must resolve uniquely: "
            f"{ref!r} -> {candidates!r}"
        )
    return candidates[0]

source_packages, source_index = build_index(source_doc, "source lock")
capsule_packages, capsule_index = build_index(capsule_doc, "capsule lock")

capsule_locals = [pkg for pkg in capsule_packages if "source" not in pkg]
local_identities = [identity(pkg) for pkg in capsule_locals]
expected_local = (expected_local_name, expected_local_version, "")
if local_identities != [expected_local]:
    raise SystemExit(f"unexpected capsule local packages: {local_identities!r}")

# Local package provenance is the exact Git-object source census established
# before lock reconciliation. It is deliberately NOT inferred from a workspace
# Cargo.lock node, because Cargo may omit a local workspace member from that lock.
external_identities = []
for pkg in capsule_packages:
    key = identity(pkg)
    if not key[2]:
        continue
    frozen = source_index.get(key)
    if frozen is None:
        raise SystemExit(f"capsule external package absent from frozen source lock: {key!r}")
    if pkg.get("checksum") != frozen.get("checksum"):
        raise SystemExit(
            f"capsule checksum drift for {key!r}: "
            f"{pkg.get('checksum')!r} != {frozen.get('checksum')!r}"
        )
    external_identities.append(key)

source_external_edges = {}
for src_key, pkg in source_index.items():
    if not src_key[2]:
        continue
    edges = set()
    for dep_ref in pkg.get("dependencies", []):
        edges.add(resolve_dep(dep_ref, source_packages, f"source lock {src_key!r}"))
    source_external_edges[src_key] = edges

local_roots = []
external_edge_count = 0
for pkg in capsule_packages:
    src_key = identity(pkg)
    if not src_key[2]:
        for dep_ref in pkg.get("dependencies", []):
            dst_key = resolve_dep(dep_ref, capsule_packages, f"capsule local {src_key!r}")
            if not dst_key[2]:
                raise SystemExit(
                    f"local package depends on unexpected local package: "
                    f"{src_key!r} -> {dst_key!r}"
                )
            if dst_key not in source_index:
                raise SystemExit(
                    f"local dependency root absent from frozen source lock: "
                    f"{src_key!r} -> {dst_key!r}"
                )
            local_roots.append(dst_key)
        continue

    frozen_edges = source_external_edges[src_key]
    for dep_ref in pkg.get("dependencies", []):
        dst_key = resolve_dep(dep_ref, capsule_packages, f"capsule external {src_key!r}")
        external_edge_count += 1
        if not dst_key[2]:
            raise SystemExit(
                f"external package resolved an unexpected local dependency: "
                f"{src_key!r} -> {dst_key!r}"
            )
        if dst_key not in frozen_edges:
            raise SystemExit(
                f"capsule external dependency edge absent from frozen source graph: "
                f"{src_key!r} -> {dst_key!r}"
            )

audit = {
    "schema": "symthaea.assurance.semantic-lock-graph-audit.v2",
    "result": "PASS",
    "local_package_provenance": "exact-git-object-source-census",
    "source_lock_local_node_required": False,
    "capsule_local_package": identity_json(expected_local),
    "capsule_local_external_roots": [identity_json(k) for k in local_roots],
    "capsule_external_packages": len(external_identities),
    "capsule_package_count": len(capsule_packages),
    "source_package_count": len(source_packages),
    "capsule_external_dependency_edges_verified": external_edge_count,
    "rule": (
        "the sole local package is bound by the exact Git-object source census; "
        "every capsule external package must match frozen name/version/source/checksum; "
        "every local root must resolve to a frozen external identity; every "
        "external-to-external capsule edge must be contained in the corresponding "
        "frozen source-lock edge set; external-to-local edges are forbidden"
    ),
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(
    "ima_lock_graph_audit=PASS "
    f"packages={len(capsule_packages)} external={len(external_identities)} "
    f"local_roots={len(local_roots)} external_edges={external_edge_count}"
)
