#!/usr/bin/env python3
"""Generate all 4-node loop-free directed graphs for query-oracle differential CI.

The generated universe has 2^(4*3) = 4096 directed simple graph topologies.
Each possible directed edge has a fixed, distinct positive weight so edge,
weighted in/out-degree, and reachability queries are covered in the same run.
This is a synthetic software-semantics fixture, not a biological data set.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path


def graph_digest(edges: list[dict[str, int | float]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"connectome-query-scenario-graph-v1")
    for edge in sorted(edges, key=lambda item: (item["source"], item["target"])):
        digest.update(struct.pack(">Q", edge["source"]))
        digest.update(struct.pack(">Q", edge["target"]))
        digest.update(struct.pack(">d", float(edge["weight"])))
    return digest.hexdigest()


def build_input() -> dict[str, object]:
    nodes = range(4)
    possible_edges = [
        (source, target)
        for source in nodes
        for target in nodes
        if source != target
    ]
    weights = [0.5 + (index + 1) / 8.0 for index in range(len(possible_edges))]

    scenarios: list[dict[str, object]] = []
    for mask in range(1 << len(possible_edges)):
        edges = [
            {"source": source, "target": target, "weight": weights[index]}
            for index, (source, target) in enumerate(possible_edges)
            if mask & (1 << index)
        ]
        scenario_id = f"graph-{mask:04d}"
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "graph_sha256": graph_digest(edges),
                "lineage_groups": ["synthetic-exhaustive-four-node-v1"],
                "edges": edges,
            }
        )

    queries: list[dict[str, object]] = []
    for source, target in possible_edges:
        queries.append(
            {
                "query_id": f"edge-weight-{source}-{target}",
                "metric": {"kind": "edge_weight", "source": source, "target": target},
                "decision": {"kind": "greater_or_equal", "threshold": 0.5},
            }
        )
    for node in nodes:
        queries.append(
            {
                "query_id": f"in-degree-{node}",
                "metric": {"kind": "in_degree", "node": node},
                "decision": {"kind": "greater_than", "threshold": 0.0},
            }
        )
        queries.append(
            {
                "query_id": f"out-degree-{node}",
                "metric": {"kind": "out_degree", "node": node},
                "decision": {"kind": "greater_than", "threshold": 0.0},
            }
        )
    for source in nodes:
        for target in nodes:
            queries.append(
                {
                    "query_id": f"path-exists-{source}-{target}",
                    "metric": {"kind": "path_exists", "source": source, "target": target},
                    "decision": {"kind": "boolean_is", "value": True},
                }
            )

    return {
        "schema_version": 1,
        "analysis_id": "synthetic-exhaustive-four-node-directed-graph-v1",
        "dataset_revision_commit_sha": "a" * 40,
        "scope_coverage": "exhaustive_declared_scenario_space",
        "scenarios": scenarios,
        "queries": queries,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: generate_exhaustive_query_ensemble_input.py OUTPUT.json",
            file=sys.stderr,
        )
        return 2
    destination = Path(sys.argv[1])
    document = build_input()
    rendered = json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n"
    destination.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "EXHAUSTIVE_SYNTHETIC_INPUT_GENERATED",
                "scenario_count": len(document["scenarios"]),
                "query_count": len(document["queries"]),
                "directed_graph_topologies": len(document["scenarios"]),
                "query_scenario_evaluations": len(document["scenarios"]) * len(document["queries"]),
                "output_bytes": len(rendered.encode("utf-8")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
