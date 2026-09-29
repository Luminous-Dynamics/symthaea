#!/usr/bin/env python3
"""Independent structural verifier for the Rust-to-CP-04 adapter vector (stdlib only)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json"
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
CLAIM_CEILING = ("typed dependency and invalidation semantics over synthetic/reference workflows only; "
                 "no physical execution, performance, compiler, accelerator, model, deployment, safety, "
                 "or operational-authority claim")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition, message):
    if not condition:
        raise AssertionError("[SCI-LINEAGE-ADAPTER] " + message)


def main():
    artifact = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require(set(artifact) == {
        "schema", "cp04_schema", "source_graph_digest", "projection_digest",
        "qualification_policy", "authority_ceiling", "claim_ceiling",
        "nodes", "edges", "artifact_digest",
    }, "unexpected adapter envelope fields")
    require(artifact["schema"] == "symthaea.cp-04-qualification-adapter.v1", "adapter schema mismatch")
    require(artifact["cp04_schema"] == "cp-04-compute-evidence-dag-v1-1", "CP-04 schema mismatch")
    require(artifact["qualification_policy"] == "symthaea.qualification-policy.v1", "policy mismatch")
    require(artifact["authority_ceiling"] == "SyntheticQualification", "authority ceiling changed")
    require(artifact["claim_ceiling"] == CLAIM_CEILING, "claim ceiling changed")
    for field in ("source_graph_digest", "projection_digest", "artifact_digest"):
        require(bool(HEX64.fullmatch(artifact[field])), field + " must be lowercase SHA-256 hex")

    nodes = artifact["nodes"]
    edges = artifact["edges"]
    require(len(nodes) == 16 and len(edges) == 15, "fixture cardinality mismatch")
    require(len({n["object_kind"] for n in nodes}) == len(NODE_TYPES)
            and {n["object_kind"] for n in nodes} == set(NODE_TYPES),
            "node set differs from canonical vocabulary")
    by_digest = {}
    for node in nodes:
        require(set(node) == {"identity", "identity_digest", "object_kind"}, "unexpected node fields")
        ident = node["identity"]
        require(set(ident) == {"namespace", "object_kind", "canonical_identifier", "version", "content_digest"},
                "unexpected identity fields")
        require(all(isinstance(ident[k], str) and ident[k].strip() == ident[k] and ident[k]
                    for k in ("namespace", "object_kind", "canonical_identifier", "version")),
                "empty or noncanonical identity field")
        require(ident["object_kind"] == node["object_kind"], "identity/object kind mismatch")
        require(node["object_kind"] in NODE_TYPES, "unsupported CP-04 node kind")
        require(bool(HEX64.fullmatch(ident["content_digest"])), "content digest malformed")
        require(bool(HEX64.fullmatch(node["identity_digest"])), "identity digest malformed")
        require(node["identity_digest"] not in by_digest, "duplicate node identity digest")
        by_digest[node["identity_digest"]] = node

    seen_edges = set()
    adjacency = {d: [] for d in by_digest}
    for edge in edges:
        require(set(edge) == {"source_identity_digest", "edge_type", "target_identity_digest", "relation_digest"},
                "unexpected edge fields")
        src, kind, dst = edge["source_identity_digest"], edge["edge_type"], edge["target_identity_digest"]
        require(src in by_digest and dst in by_digest, "dangling endpoint")
        require(kind in ENDPOINTS, "unsupported edge type (epistemic/scientific leakage)")
        require((by_digest[src]["object_kind"], by_digest[dst]["object_kind"]) == ENDPOINTS[kind],
                "edge endpoint ontology mismatch")
        require(bool(HEX64.fullmatch(edge["relation_digest"])), "relation digest malformed")
        key = (src, kind, dst, edge["relation_digest"])
        require(key not in seen_edges, "duplicate edge")
        seen_edges.add(key)
        adjacency[src].append(dst)

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

    print("SCI-LINEAGE-ADAPTER: PASS (16 typed nodes, 15 typed edges, endpoint ontology, acyclicity, authority ceiling)")
    print("NOTE: structural verification only; does not recompute Rust canonical digests or prove physical execution.")


if __name__ == "__main__":
    main()
