#!/usr/bin/env python3
"""End-to-end MATH-RET-RUNTIME-001B contract-adapter qualification.

Builds a real content-addressed experiment/retrieval graph from the normative
validator fixtures, validates it, invokes the typed Rust runtime seam for arm S,
then content-addresses the emitted trace/materialized payloads and validates the
frozen MATH-RET-TRACE-001A + MATH-RET-PAYLOAD-001A documents.

Stdlib only. The Rust seam remains dependency-free.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
HERE = Path(__file__).resolve().parent
WORK = HERE / "target" / "contract-adapter-fixture"
GRAPH = WORK / "graph"
RUNTIME = WORK / "runtime"
AUTH = "MeasurementOnly"


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_label(label: str) -> str:
    return digest_bytes(label.encode("utf-8"))


def canonical_bytes(doc: object) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, doc: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(doc)
    path.write_bytes(data)
    return digest_bytes(data)


def run(*args: object) -> None:
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_graph_fixture():
    exp_v21 = load_script("validate-math-search-experiment-v2.1.py", "adapter_exp_v21")
    index_v1 = load_script("validate-math-retrieval-index.py", "adapter_index_v1")
    fusion_v1 = load_script("validate-math-retrieval-fusion.py", "adapter_fusion_v1")
    packer_v1 = load_script("validate-math-retrieval-context-packer.py", "adapter_packer_v1")
    graph_v1 = load_script("validate-math-retrieval-graph.py", "adapter_graph_v1")

    exp = exp_v21.fixture()
    exp["experiment_id"] = "math-ret-runtime-001b-fixture"
    shared = exp["shared_contract"]
    budget = exp["budget"]

    packer = packer_v1.fixture()
    packer["packer_id"] = "math-ret-runtime-001b-packer"
    packer["source"]["source_object_contract_sha256"] = shared["source_object_contract_sha256"]
    packer["source"]["source_fetch_policy_sha256"] = digest_label("runtime-001b-source-fetch")
    packer["source"]["payload_serialization_sha256"] = digest_label("runtime-001b-payload-serialization")
    packer["budget"] = {
        "max_output_items": budget["retrieved_items_max"],
        "max_output_bytes": budget["retrieval_context_bytes_max"],
        "max_output_item_bytes": budget["retrieved_item_bytes_max"],
    }
    packer_path = GRAPH / "context-packer.json"
    packer_digest = write_json(packer_path, packer)

    candidate_universe = copy.deepcopy(index_v1.fixture()["candidate_universe"])
    candidate_universe.update(
        corpus_snapshot_sha256=shared["corpus_snapshot_sha256"],
        knowledge_boundary_sha256=shared["knowledge_boundary_sha256"],
        candidate_eligibility_policy_sha256=digest_label("runtime-001b-candidate-eligibility"),
        candidate_set_sha256=digest_label("runtime-001b-candidate-set"),
        candidate_count=32,
    )

    arm_specs = {
        "L": ("Syntax", "Lexical", "None"),
        "R": ("None", "None", "RandomRetrieval"),
        "S": ("Syntax", "CanonicalSparse", "None"),
        "H": ("Syntax", "HDC", "None"),
        "N": ("ExactNormalForm", "CanonicalSparse", "None"),
        "SHUF": ("Syntax", "HDC", "ShuffledHdcVectors"),
        "PERM": ("Syntax", "CanonicalSparse", "PermutedChallengeAssociations"),
    }

    indices: dict[str, dict] = {}
    index_paths: dict[str, Path] = {}
    index_digests: dict[str, str] = {}

    for arm_id, (channel, family, control) in arm_specs.items():
        doc = index_v1.fixture()
        doc["index_id"] = f"math-ret-runtime-001b-{arm_id.lower()}"
        doc["candidate_universe"] = copy.deepcopy(candidate_universe)
        rep = doc["representation"]
        rep["channel"] = channel
        rep["representation_family"] = family
        rep["representation_sha256"] = digest_label(f"runtime-001b-representation-{arm_id}")
        rep["item_serialization_sha256"] = digest_label(f"runtime-001b-item-serialization-{arm_id}")
        rep["max_serialized_item_bytes"] = budget["retrieved_item_bytes_max"]
        rep["control_transform"] = control
        rep.pop("normalization_contract_sha256", None)
        rep.pop("normalization_implementation_sha256", None)
        rep.pop("control_seed", None)
        rep.pop("control_artifact_sha256", None)
        if channel == "ExactNormalForm":
            rep["normalization_contract_sha256"] = shared["normalization_contract_sha256"]
            rep["normalization_implementation_sha256"] = shared["normalization_implementation_sha256"]
        if control != "None":
            rep["control_seed"] = exp["seeds"][0]
            rep["control_artifact_sha256"] = digest_label(f"runtime-001b-control-artifact-{arm_id}")

        ix = doc["index"]
        ix["index_build_policy_sha256"] = digest_label(f"runtime-001b-build-{arm_id}")
        ix["index_artifact_sha256"] = digest_label(f"runtime-001b-index-artifact-{arm_id}")
        ix["index_seed"] = exp["seeds"][0]
        ix["top_k_supported"] = budget["retrieved_items_max"]
        ix["scoring_metric"] = (
            "RandomDeterministic"
            if control == "RandomRetrieval"
            else "BM25"
            if family == "Lexical"
            else "HammingSimilarity"
            if family == "HDC"
            else "Cosine"
        )
        ix["scoring_policy_sha256"] = digest_label(f"runtime-001b-score-{arm_id}")
        ix["score_precision_policy_sha256"] = digest_label(f"runtime-001b-precision-{arm_id}")
        ix["query_normalization_policy_sha256"] = digest_label(f"runtime-001b-query-{arm_id}")

        path = GRAPH / "indices" / f"{arm_id}.json"
        indices[arm_id] = doc
        index_paths[arm_id] = path
        index_digests[arm_id] = write_json(path, doc)

    fusion = fusion_v1.fixture()
    fusion["version"] = "math-retrieval-fusion-v1.1"
    fusion["fusion_id"] = "math-ret-runtime-001b-fusion"
    fusion["inputs"] = [
        {
            "channel": "Syntax",
            "index_manifest_sha256": index_digests["S"],
            "max_input_items": budget["retrieved_items_max"] // 2,
            "max_input_bytes": budget["retrieval_context_bytes_max"] // 2,
            "rank_weight": 1,
        },
        {
            "channel": "ExactNormalForm",
            "index_manifest_sha256": index_digests["N"],
            "max_input_items": budget["retrieved_items_max"] // 2,
            "max_input_bytes": budget["retrieval_context_bytes_max"] // 2,
            "rank_weight": 1,
        },
    ]
    fusion["fusion"].update(
        input_rank_interpretation="OneBasedAscendingIndexRank",
        rrf_arithmetic="ExactRational",
        rrf_formula="SumOneOverKPlusRank",
    )
    fusion["global_budget"] = {
        "max_output_items": budget["retrieved_items_max"],
        "max_output_bytes": budget["retrieval_context_bytes_max"],
        "max_output_item_bytes": budget["retrieved_item_bytes_max"],
        "byte_accounting": "CanonicalUtf8Bytes",
        "partial_item_policy": "RejectWholeItem",
        "packing_policy": "FusedRankOrderWholeItems",
    }
    fusion["output"]["payload_serialization_sha256"] = packer["source"]["payload_serialization_sha256"]
    fusion_path = GRAPH / "fusion.json"
    fusion_digest = write_json(fusion_path, fusion)

    bindings: dict[str, dict] = {}
    binding_paths: dict[str, Path] = {}
    binding_digests: dict[str, str] = {}
    for arm_id in ("L", "R", "S", "H", "N", "SHUF", "PERM"):
        doc = {
            "version": "math-retrieval-binding-v1",
            "binding_id": f"math-ret-runtime-001b-binding-{arm_id}",
            "authority": AUTH,
            "arm_id": arm_id,
            "mode": "SingleIndex",
            "context_packer_sha256": packer_digest,
            "index_manifest_sha256": index_digests[arm_id],
        }
        path = GRAPH / "bindings" / f"{arm_id}.json"
        bindings[arm_id] = doc
        binding_paths[arm_id] = path
        binding_digests[arm_id] = write_json(path, doc)

    f_binding = {
        "version": "math-retrieval-binding-v1",
        "binding_id": "math-ret-runtime-001b-binding-F",
        "authority": AUTH,
        "arm_id": "F",
        "mode": "Fusion",
        "context_packer_sha256": packer_digest,
        "syntax_index_manifest_sha256": index_digests["S"],
        "normal_form_index_manifest_sha256": index_digests["N"],
        "fusion_policy_sha256": fusion_digest,
    }
    f_binding_path = GRAPH / "bindings" / "F.json"
    bindings["F"] = f_binding
    binding_paths["F"] = f_binding_path
    binding_digests["F"] = write_json(f_binding_path, f_binding)

    for arm in exp["arms"]:
        arm_id = arm["arm_id"]
        if arm["retriever_family"] == "None":
            arm.pop("retrieval_binding_sha256", None)
            continue
        arm["retrieval_binding_sha256"] = binding_digests[arm_id]
        if arm_id == "F":
            arm["fusion_policy_sha256"] = fusion_digest
            arm["representation_sha256"] = graph_v1.fusion_rep_identity(
                indices["S"]["representation"]["representation_sha256"],
                indices["N"]["representation"]["representation_sha256"],
                fusion_digest,
            )
        else:
            arm["representation_sha256"] = indices[arm_id]["representation"]["representation_sha256"]

    experiment_path = GRAPH / "experiment.json"
    experiment_digest = write_json(experiment_path, exp)

    artifact_rows = [
        ("ExperimentManifest", experiment_path, experiment_digest),
        ("ContextPacker", packer_path, packer_digest),
        ("FusionPolicy", fusion_path, fusion_digest),
    ]
    artifact_rows.extend(("RetrievalIndex", index_paths[a], index_digests[a]) for a in sorted(index_paths))
    artifact_rows.extend(("RetrievalBinding", binding_paths[a], binding_digests[a]) for a in sorted(binding_paths))
    bundle = {
        "version": "math-retrieval-graph-bundle-v1",
        "bundle_id": "math-ret-runtime-001b-fixture-bundle",
        "authority": AUTH,
        "artifacts": [
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha,
            }
            for kind, path, sha in artifact_rows
        ],
    }
    bundle_path = GRAPH / "bundle.json"
    write_json(bundle_path, bundle)

    return {
        "experiment": exp,
        "experiment_path": experiment_path,
        "experiment_digest": experiment_digest,
        "packer": packer,
        "packer_digest": packer_digest,
        "bundle_path": bundle_path,
        "candidate_universe": candidate_universe,
        "s_binding_digest": binding_digests["S"],
        "s_index_digest": index_digests["S"],
        "s_index_artifact_digest": indices["S"]["index"]["index_artifact_sha256"],
    }


def write_runtime_config(graph: dict, graph_report: dict, graph_report_path: Path) -> Path:
    exp = graph["experiment"]
    packer = graph["packer"]
    budget = exp["budget"]
    normalized = int(round(float(budget["normalized_compute_units_max"]) * 1_000_000))
    values = {
        "experiment_id": exp["experiment_id"],
        "experiment_seed": exp["seeds"][0],
        "bundle_sha256": digest_bytes(graph["bundle_path"].read_bytes()),
        "graph_report_sha256": digest_bytes(graph_report_path.read_bytes()),
        "experiment_sha256": graph["experiment_digest"],
        "retrieval_binding_sha256": graph["s_binding_digest"],
        "candidate_set_sha256": graph_report["shared_candidate_set_sha256"],
        "candidate_count": graph_report["shared_candidate_count"],
        "context_packer_sha256": graph["packer_digest"],
        "source_object_contract_sha256": packer["source"]["source_object_contract_sha256"],
        "source_fetch_policy_sha256": packer["source"]["source_fetch_policy_sha256"],
        "payload_serialization_sha256": packer["source"]["payload_serialization_sha256"],
        "index_manifest_sha256": graph["s_index_digest"],
        "index_artifact_sha256": graph["s_index_artifact_digest"],
        "max_output_items": budget["retrieved_items_max"],
        "max_output_bytes": budget["retrieval_context_bytes_max"],
        "max_output_item_bytes": budget["retrieved_item_bytes_max"],
        "max_retrieval_queries": budget["retrieval_queries_max"],
        "max_normalized_compute_microunits": normalized,
        "max_wall_time_ms": budget["wall_time_ms_max"],
    }
    path = WORK / "runtime-config.txt"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


def build_payload_audit(trace_path: Path, graph: dict, graph_report_path: Path) -> Path:
    plan_path = RUNTIME / "payload-plan.tsv"
    rows = plan_path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0] != "role\trank\tsource_object_sha256\tpayload_path\tcanonical_payload_bytes":
        raise RuntimeError("unexpected payload-plan header")
    entries = []
    for row in rows[1:]:
        role, rank, source, rel, claimed = row.split("\t")
        p = RUNTIME / rel
        raw = p.read_bytes()
        if len(raw) != int(claimed):
            raise RuntimeError(f"Rust payload-plan byte mismatch for {rel}")
        entries.append(
            {
                "role": role,
                "rank": int(rank),
                "source_object_sha256": source,
                "payload_path": rel,
                "payload_sha256": digest_bytes(raw),
                "canonical_payload_bytes": len(raw),
            }
        )
    audit = {
        "version": "math-retrieval-payload-audit-v1",
        "audit_id": "math-ret-runtime-001b-fixture-audit",
        "authority": AUTH,
        "trace_sha256": digest_bytes(trace_path.read_bytes()),
        "graph_report_sha256": digest_bytes(graph_report_path.read_bytes()),
        "context_packer_sha256": graph["packer_digest"],
        "payload_serialization_sha256": graph["packer"]["source"]["payload_serialization_sha256"],
        "entries": entries,
    }
    path = RUNTIME / "payload-audit.json"
    write_json(path, audit)
    return path


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    GRAPH.mkdir(parents=True)
    RUNTIME.mkdir(parents=True)

    graph = build_graph_fixture()
    graph_report_path = WORK / "graph-report.json"
    run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-graph-v1.1.py",
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--report",
        graph_report_path,
    )
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if not graph_report.get("all_checks_passed"):
        raise RuntimeError("graph qualification did not pass")

    config_path = write_runtime_config(graph, graph_report, graph_report_path)
    run(
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        HERE / "Cargo.toml",
        "--locked",
        "--bin",
        "emit_fixture",
        "--",
        config_path,
        RUNTIME,
    )

    trace_path = RUNTIME / "trace.json"
    trace_report_path = WORK / "trace-report.json"
    run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-trace.py",
        trace_path,
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--report",
        trace_report_path,
    )
    trace_report = json.loads(trace_report_path.read_text(encoding="utf-8"))
    if not trace_report.get("all_checks_passed"):
        raise RuntimeError("trace qualification did not pass")

    audit_path = build_payload_audit(trace_path, graph, graph_report_path)
    payload_report_path = WORK / "payload-report.json"
    run(
        sys.executable,
        SCRIPTS / "validate-math-retrieval-payload-audit.py",
        audit_path,
        trace_path,
        graph["bundle_path"],
        "--repo-root",
        ROOT,
        "--payload-root",
        RUNTIME,
        "--report",
        payload_report_path,
    )
    payload_report = json.loads(payload_report_path.read_text(encoding="utf-8"))
    if not payload_report.get("all_checks_passed"):
        raise RuntimeError("payload qualification did not pass")

    summary = {
        "authority": AUTH,
        "experiment_sha256": graph["experiment_digest"],
        "bundle_sha256": digest_bytes(graph["bundle_path"].read_bytes()),
        "graph_report_sha256": digest_bytes(graph_report_path.read_bytes()),
        "trace_sha256": digest_bytes(trace_path.read_bytes()),
        "payload_audit_sha256": digest_bytes(audit_path.read_bytes()),
        "graph_all_checks_passed": True,
        "trace_all_checks_passed": True,
        "payload_all_checks_passed": True,
    }
    (WORK / "qualification-summary.json").write_bytes(canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
