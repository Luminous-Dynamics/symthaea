#!/usr/bin/env python3
"""Independent stdlib-only qualifier for CP-04 typed evidence DAG v1.1."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "docs/engineering/data/cp-04-compute-evidence-dag-v1-1.json"
MANIFEST = ROOT / "docs/engineering/data/cp-04-compute-evidence-dag-mutation-manifest-v1-1.json"
SCHEMA = "cp-04-compute-evidence-dag-v1-1"
CLAIM_CEILING = "typed dependency and invalidation semantics over synthetic/reference workflows only; no physical execution, performance, compiler, accelerator, model, deployment, safety, or operational-authority claim"

GUARDS = {
    "CP-COMP-DAG-SCHEMA",
    "CP-COMP-DAG-TYPING",
    "CP-COMP-DAG-ACYCLIC",
    "CP-COMP-DAG-CLOSURE",
    "CP-COMP-DAG-IDENTITY",
    "CP-COMP-DAG-AUTHORITY",
    "CP-COMP-DAG-MUTATION",
    "CP-COMP-DAG-INVARIANT",
}

EXPECTED_NODES = (
    "requirement", "representation", "model", "model_parameters", "runtime",
    "toolchain", "accelerator", "deployment_artifact", "execution_context",
    "observation", "uncertainty", "statistics", "provenance_reference",
    "currentness", "applicability", "disposition",
)
EXPECTED_EDGES = (
    ("requirement", "requires", "representation"),
    ("representation", "implements", "model"),
    ("model", "parameterizes", "model_parameters"),
    ("model", "executes_with", "runtime"),
    ("runtime", "compiled_by", "toolchain"),
    ("runtime", "runs_on", "accelerator"),
    ("accelerator", "deploys", "deployment_artifact"),
    ("deployment_artifact", "executes", "execution_context"),
    ("execution_context", "observes", "observation"),
    ("observation", "quantifies", "uncertainty"),
    ("observation", "summarizes", "statistics"),
    ("observation", "traces_to", "provenance_reference"),
    ("currentness", "currentness_for", "runtime"),
    ("applicability", "applicable_to", "execution_context"),
    ("statistics", "derives", "disposition"),
)
EXPECTED_TYPES = (
    "requires", "implements", "parameterizes", "executes_with", "compiled_by",
    "runs_on", "deploys", "executes", "observes", "quantifies", "summarizes",
    "traces_to", "currentness_for", "applicable_to", "derives",
)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def fail(message, guard_id):
    if guard_id not in GUARDS:
        raise AssertionError(f"unregistered guard: {guard_id}")
    raise AssertionError(f"[{guard_id}] {message}")

def source_audit():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"symthaea", "torch", "numpy", "pandas", "tensorflow", "onnx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    fail("forbidden production/runtime import", "CP-COMP-DAG-INVARIANT")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in forbidden:
                fail("forbidden production/runtime import", "CP-COMP-DAG-INVARIANT")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fail":
            if len(node.args) != 2:
                fail("fail() must carry a stable guard id", "CP-COMP-DAG-INVARIANT")
            guard = node.args[1]
            if isinstance(guard, ast.Constant):
                if guard.value not in GUARDS:
                    fail("fail() references unregistered guard", "CP-COMP-DAG-INVARIANT")
            elif not (isinstance(guard, ast.Name) and guard.id == "guard_id"):
                fail("fail() guard must be stable literal or guard_id", "CP-COMP-DAG-INVARIANT")

def load():
    try:
        return json.loads(GRAPH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot load graph: {exc}", "CP-COMP-DAG-SCHEMA")

def validate_shape(g):
    if set(g) != {"schema", "node_types", "edge_types", "edges", "claim_ceiling"}:
        fail("unexpected top-level graph fields", "CP-COMP-DAG-SCHEMA")
    if g["schema"] != SCHEMA or g["claim_ceiling"] != CLAIM_CEILING:
        fail("schema or claim ceiling mismatch", "CP-COMP-DAG-SCHEMA")
    if tuple(g["node_types"]) != EXPECTED_NODES:
        fail("node vocabulary/order mismatch", "CP-COMP-DAG-TYPING")
    if tuple(g["edge_types"]) != EXPECTED_TYPES:
        fail("edge-type vocabulary/order mismatch", "CP-COMP-DAG-TYPING")
    edges = tuple(tuple(e) for e in g["edges"])
    if edges != EXPECTED_EDGES:
        fail("typed edge oracle mismatch", "CP-COMP-DAG-TYPING")
    if len(g["node_types"]) != len(set(g["node_types"])):
        fail("duplicate node type", "CP-COMP-DAG-SCHEMA")
    if len(g["edge_types"]) != len(set(g["edge_types"])):
        fail("duplicate edge type", "CP-COMP-DAG-SCHEMA")
    if len(edges) != len(set(edges)):
        fail("duplicate typed edge", "CP-COMP-DAG-SCHEMA")
    nodes = set(g["node_types"])
    types = set(g["edge_types"])
    for edge in edges:
        if len(edge) != 3 or edge[0] not in nodes or edge[2] not in nodes or edge[1] not in types:
            fail("malformed or dangling typed edge", "CP-COMP-DAG-TYPING")

def validate_acyclic(g):
    adj = {n: [] for n in g["node_types"]}
    for src, _, dst in g["edges"]:
        adj[src].append(dst)
    visiting, visited = set(), set()
    def walk(node):
        if node in visiting:
            fail("cycle detected in evidence DAG", "CP-COMP-DAG-ACYCLIC")
        if node in visited:
            return
        visiting.add(node)
        for child in adj[node]:
            walk(child)
        visiting.remove(node)
        visited.add(node)
    for node in g["node_types"]:
        walk(node)

def closure(node, g):
    adj = {n: [] for n in g["node_types"]}
    for src, _, dst in g["edges"]:
        adj[src].append(dst)
    found = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        stack.extend(adj[current])
    return tuple(n for n in g["node_types"] if n in found)

def expected_closures(g):
    return {node: closure(node, g) for node in g["node_types"]}

def graph_identity(g):
    return digest({
        "schema": g["schema"],
        "node_types": g["node_types"],
        "edge_types": g["edge_types"],
        "edges": g["edges"],
        "claim_ceiling": g["claim_ceiling"],
    })

MUTATIONS = (
    ("remove-model-runtime-edge", "CP-COMP-DAG-TYPING", lambda g: g["edges"].remove(["model", "executes_with", "runtime"])),
    ("reverse-model-runtime-edge", "CP-COMP-DAG-TYPING", lambda g: g["edges"].__setitem__(3, ["runtime", "executes_with", "model"])),
    ("duplicate-model-runtime-edge", "CP-COMP-DAG-TYPING", lambda g: g["edges"].append(["model", "executes_with", "runtime"])),
    ("change-edge-type", "CP-COMP-DAG-TYPING", lambda g: g["edges"].__setitem__(3, ["model", "requires", "runtime"])),
    ("rename-node", "CP-COMP-DAG-TYPING", lambda g: g["node_types"].__setitem__(2, "model_v2")),
    ("add-graph-field", "CP-COMP-DAG-SCHEMA", lambda g: g.__setitem__("unexpected", True)),
    ("malformed-edge", "CP-COMP-DAG-TYPING", lambda g: g["edges"].__setitem__(3, ["model", "executes_with"])),
    ("dangling-edge", "CP-COMP-DAG-TYPING", lambda g: g["edges"].__setitem__(3, ["model", "executes_with", "missing"])),
    ("duplicate-node", "CP-COMP-DAG-SCHEMA", lambda g: g["node_types"].append("model")),
    ("add-cycle", "CP-COMP-DAG-ACYCLIC", lambda g: g["edges"].append(["disposition", "derives", "model"])),
)

def validate_mutation_manifest():
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot load mutation manifest: {exc}", "CP-COMP-DAG-MUTATION")
    if set(manifest) != {"schema", "mutations"} or manifest["schema"] != "cp-04-compute-evidence-dag-mutation-manifest-v1-1":
        fail("mutation manifest schema mismatch", "CP-COMP-DAG-MUTATION")
    expected = [{"mutation_id": label, "expected_guard_id": guard} for label, guard, _ in MUTATIONS]
    if manifest["mutations"] != expected:
        fail("mutation manifest does not exactly bind executable mutation suite", "CP-COMP-DAG-MUTATION")
    return digest(manifest)

def validate(g):
    validate_shape(g)
    validate_acyclic(g)
    closures = expected_closures(g)
    if "deployment_artifact" not in closures["accelerator"] or "model" in closures["deployment_artifact"]:
        fail("deployment closure escaped its dependency boundary", "CP-COMP-DAG-CLOSURE")
    if closures["runtime"] != (
        "runtime", "toolchain", "accelerator", "deployment_artifact",
        "execution_context", "observation", "uncertainty", "statistics",
        "provenance_reference", "disposition",
    ):
        fail("runtime closure does not match typed dependency semantics", "CP-COMP-DAG-CLOSURE")
    if "provenance_reference" not in closures["observation"]:
        fail("observation lost provenance dependency", "CP-COMP-DAG-CLOSURE")
    if "disposition" not in closures["statistics"]:
        fail("statistics lost disposition dependency", "CP-COMP-DAG-CLOSURE")
    if "execution_context" not in closures["applicability"]:
        fail("applicability boundary missing", "CP-COMP-DAG-CLOSURE")

def replay_identity(g):
    return digest(g)

def run_mutations(base):
    results = []
    for label, guard, mutate in MUTATIONS:
        candidate = copy.deepcopy(base)
        mutate(candidate)
        rejected = False
        try:
            validate(candidate)
        except AssertionError as exc:
            rejected = guard in str(exc)
        if not rejected:
            fail(f"mutation {label} escaped expected guard {guard}", "CP-COMP-DAG-MUTATION")
        results.append({"mutation_id": label, "guard_id": guard, "rejected": True})
    return results

def main():
    source_audit()
    manifest_identity = validate_mutation_manifest()
    graph = load()
    validate(graph)

    identity = graph_identity(graph)
    equivalent = copy.deepcopy(graph)
    if graph_identity(equivalent) != identity:
        fail("equivalent graph changed identity", "CP-COMP-DAG-IDENTITY")

    changed_type = copy.deepcopy(graph)
    changed_type["edges"][3][1] = "requires"
    if graph_identity(changed_type) == identity:
        fail("edge-type mutation did not change graph identity", "CP-COMP-DAG-IDENTITY")

    disposition_only = copy.deepcopy(graph)
    # Derived disposition is not an input node value in this fixture; an external
    # disposition annotation therefore cannot alter the dependency graph identity.
    disposition_only["derived_disposition"] = "CurrentAndApplicable"
    if graph_identity(disposition_only) == identity:
        # This check documents the intentional boundary: graph identity ignores
        # derived output annotations.
        pass
    else:
        fail("derived disposition altered immutable graph identity", "CP-COMP-DAG-INVARIANT")

    results = run_mutations(graph)
    receipt = {
        "schema": "cp-04-compute-evidence-dag-qualification-receipt-v1-1",
        "graph_identity": identity,
        "mutation_manifest_identity": manifest_identity,
        "mutation_results": results,
        "claim_ceiling": CLAIM_CEILING,
        "physical_execution_authority": False,
    }
    receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    print("PASS"
          f" schema={SCHEMA}"
          f" graph_identity={identity}"
          f" mutations={len(results)}"
          f" receipt_sha256={receipt['receipt_sha256']}")

if __name__ == "__main__":
    main()
