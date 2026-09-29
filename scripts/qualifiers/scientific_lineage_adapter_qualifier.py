#!/usr/bin/env python3
"""Independent cryptographic and structural verifier for the Rust-to-CP-04 vector.

The digest routines intentionally mirror the Rust wire contracts, not a generic
JSON canonicalization standard. This makes a contract change explicit and testable.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json"
IDENTITY_SCHEMA = "symthaea.engineering-object-id.v1"
RELATION_SCHEMA = "symthaea.engineering-relation.v1"
PROJECTION_SCHEMA = "symthaea.qualification-projection.v1"
LINEAGE_SCHEMA = "symthaea.scientific-lineage.v2"
ADAPTER_SCHEMA = "symthaea.cp-04-qualification-adapter.v1"
CP04_SCHEMA = "cp-04-compute-evidence-dag-v1-1"
POLICY = "symthaea.qualification-policy.v1"
AUTHORITY = "SyntheticQualification"
CLAIM_CEILING = ("typed dependency and invalidation semantics over synthetic/reference workflows only; "
                 "no physical execution, performance, compiler, accelerator, model, deployment, safety, "
                 "or operational-authority claim")
NODE_TYPES = (
    "requirement", "representation", "model", "model_parameters", "runtime",
    "toolchain", "accelerator", "deployment_artifact", "execution_context",
    "observation", "uncertainty", "statistics", "provenance_reference",
    "currentness", "applicability", "disposition",
)
ENDPOINTS = {
    "requires": ("requirement", "representation"),
    "implements": ("representation", "model"),
    "parameterizes": ("model", "model_parameters"),
    "executes_with": ("model", "runtime"),
    "compiled_by": ("runtime", "toolchain"),
    "runs_on": ("runtime", "accelerator"),
    "deploys": ("accelerator", "deployment_artifact"),
    "executes": ("deployment_artifact", "execution_context"),
    "observes": ("execution_context", "observation"),
    "quantifies": ("observation", "uncertainty"),
    "summarizes": ("observation", "statistics"),
    "traces_to": ("observation", "provenance_reference"),
    "currentness_for": ("currentness", "runtime"),
    "applicable_to": ("applicability", "execution_context"),
    "derives": ("statistics", "disposition"),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_FIELDS = ("namespace", "object_kind", "canonical_identifier", "version", "content_digest")


def require(condition, message):
    if not condition:
        raise AssertionError("[SCI-LINEAGE-ADAPTER] " + message)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def push_field(out, value):
    """Rust Vec<u8>: decimal UTF-8 byte length, colon, then raw UTF-8."""
    encoded = value.encode("utf-8")
    out.extend(str(len(encoded)).encode("ascii"))
    out.extend(b":")
    out.extend(encoded)


def identity_digest(identity):
    out = bytearray()
    for value in (IDENTITY_SCHEMA, *(identity[k] for k in IDENTITY_FIELDS)):
        push_field(out, value)
    return sha256_hex(out)


def relation_digest(source_digest, edge_type, target_digest):
    out = bytearray()
    for value in (RELATION_SCHEMA, source_digest, target_digest, edge_type):
        push_field(out, value)
    return sha256_hex(out)


def projection_digest(relation_digests):
    out = bytearray()
    for value in (PROJECTION_SCHEMA, POLICY, "synthetic-qualification", *sorted(relation_digests)):
        out.extend(value.encode("utf-8"))
        out.append(0)
    return sha256_hex(out)


def graph_digest(node_digests, relation_digests):
    out = bytearray(LINEAGE_SCHEMA.encode("utf-8"))
    out.extend(b"nodes\x00")
    for digest in sorted(node_digests):
        out.extend(digest.encode("ascii"))
        out.append(0)
    out.extend(b"relations\x00")
    for digest in sorted(relation_digests):
        out.extend(digest.encode("ascii"))
        out.append(0)
    return sha256_hex(out)


def artifact_digest(artifact):
    # serde_json serializes Rust struct fields in declaration order. Rebuild that
    # order explicitly; compact UTF-8 JSON matches serde_json::to_vec for this
    # string-only schema (no floats, maps, or implementation-dependent numbers).
    value = {
        "schema": artifact["schema"],
        "cp04_schema": artifact["cp04_schema"],
        "source_graph_digest": artifact["source_graph_digest"],
        "projection_digest": artifact["projection_digest"],
        "qualification_policy": artifact["qualification_policy"],
        "authority_ceiling": artifact["authority_ceiling"],
        "claim_ceiling": artifact["claim_ceiling"],
        "nodes": [
            {
                "identity": {k: n["identity"][k] for k in IDENTITY_FIELDS},
                "identity_digest": n["identity_digest"],
                "object_kind": n["object_kind"],
            } for n in artifact["nodes"]
        ],
        "edges": [
            {
                "source_identity_digest": e["source_identity_digest"],
                "edge_type": e["edge_type"],
                "target_identity_digest": e["target_identity_digest"],
                "relation_digest": e["relation_digest"],
            } for e in artifact["edges"]
        ],
        "artifact_digest": "",
    }
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_hex(encoded)


def main():
    artifact = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require(set(artifact) == {
        "schema", "cp04_schema", "source_graph_digest", "projection_digest",
        "qualification_policy", "authority_ceiling", "claim_ceiling",
        "nodes", "edges", "artifact_digest",
    }, "unexpected adapter envelope fields")
    require(artifact["schema"] == ADAPTER_SCHEMA, "adapter schema mismatch")
    require(artifact["cp04_schema"] == CP04_SCHEMA, "CP-04 schema mismatch")
    require(artifact["qualification_policy"] == POLICY, "policy mismatch")
    require(artifact["authority_ceiling"] == AUTHORITY, "authority ceiling changed")
    require(artifact["claim_ceiling"] == CLAIM_CEILING, "claim ceiling changed")
    for field in ("source_graph_digest", "projection_digest", "artifact_digest"):
        require(isinstance(artifact[field], str) and HEX64.fullmatch(artifact[field]),
                field + " must be lowercase SHA-256 hex")

    nodes, edges = artifact["nodes"], artifact["edges"]
    require(len(nodes) == 16 and len(edges) == 15, "fixture cardinality mismatch")
    require({n["object_kind"] for n in nodes} == set(NODE_TYPES),
            "node set differs from canonical vocabulary")
    by_digest = {}
    for node in nodes:
        require(set(node) == {"identity", "identity_digest", "object_kind"}, "unexpected node fields")
        ident = node["identity"]
        require(set(ident) == set(IDENTITY_FIELDS), "unexpected identity fields")
        require(all(isinstance(ident[k], str) for k in IDENTITY_FIELDS), "identity values must be strings")
        require(all(ident[k] and ident[k].strip() == ident[k] for k in IDENTITY_FIELDS[:4]),
                "empty or noncanonical identity field")
        require(ident["object_kind"] == node["object_kind"], "identity/object kind mismatch")
        require(node["object_kind"] in NODE_TYPES, "unsupported CP-04 node kind")
        require(bool(HEX64.fullmatch(ident["content_digest"])), "content digest malformed")
        expected = identity_digest(ident)
        require(node["identity_digest"] == expected,
                "identity digest does not match Rust canonical identity bytes")
        require(expected not in by_digest, "duplicate node identity digest")
        by_digest[expected] = node

    seen_edges, relation_digests = set(), []
    adjacency = {digest: [] for digest in by_digest}
    for edge in edges:
        require(set(edge) == {"source_identity_digest", "edge_type", "target_identity_digest", "relation_digest"},
                "unexpected edge fields")
        src, kind, dst = edge["source_identity_digest"], edge["edge_type"], edge["target_identity_digest"]
        require(src in by_digest and dst in by_digest, "dangling endpoint")
        require(kind in ENDPOINTS, "unsupported edge type (epistemic/scientific leakage)")
        require((by_digest[src]["object_kind"], by_digest[dst]["object_kind"]) == ENDPOINTS[kind],
                "edge endpoint ontology mismatch")
        expected = relation_digest(src, kind, dst)
        require(edge["relation_digest"] == expected,
                "relation digest does not match Rust canonical relation bytes")
        key = (src, kind, dst, expected)
        require(key not in seen_edges, "duplicate edge")
        seen_edges.add(key)
        relation_digests.append(expected)
        adjacency[src].append(dst)

    require(len(set(relation_digests)) == len(relation_digests), "duplicate relation digest")

    visiting, visited = set(), set()
    def visit(node):
        require(node not in visiting, "qualification projection contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)
    for node in adjacency:
        visit(node)

    expected_projection = projection_digest(relation_digests)
    require(artifact["projection_digest"] == expected_projection,
            "projection digest does not match Rust qualification projection bytes")
    expected_graph = graph_digest(by_digest.keys(), relation_digests)
    require(artifact["source_graph_digest"] == expected_graph,
            "source graph digest does not match reconstructed fixture graph")
    expected_artifact = artifact_digest(artifact)
    require(artifact["artifact_digest"] == expected_artifact,
            "artifact digest does not match Rust serde_json compact struct serialization")

    print("SCI-LINEAGE-ADAPTER: PASS (identity, relation, projection, graph, and artifact digests)")
    print("SCI-LINEAGE-ADAPTER: PASS (16 typed nodes, 15 typed edges, endpoint ontology, acyclicity, authority ceiling)")
    print("NOTE: fixture graph is reconstructed from adapter nodes/edges; this does not prove physical execution.")


if __name__ == "__main__":
    main()
