#!/usr/bin/env python3
"""Hydrate and audit only public task-network assets from Zenodo v2.0.

No participant/session/move-history records are selected or exported. Downloaded
LFS objects exist only in the runner's temporary directory. The output contains
aggregate network/solution diagnostics and per-object integrity receipts.
"""

from __future__ import annotations

import argparse
import base64
import os
import hashlib
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

ZENODO_MD5 = "55fbf9a0ffbc2f605da1792226471711"
ZENODO_SIZE = 28_279_082
DEFAULT_LFS_BATCH = (
    "https://github.com/center-for-humans-and-machines/"
    "reward-network-iii.git/info/lfs/objects/batch"
)

TASK_ASSETS = {
    "data/networks_solutions_models/networks_train.json": "networks_train",
    "data/networks_solutions_models/networks_test.json": "networks_test",
    "data/networks_solutions_models/networks_exp.json": "networks_experiment",
    "data/networks_solutions_models/solution_train__myopic.json": "train_myopic",
    "data/networks_solutions_models/solution_train__take_loss.json": "train_take_loss",
    "data/networks_solutions_models/solution_train__random.json": "train_random",
    "data/networks_solutions_models/solution_test__myopic.json": "test_myopic",
    "data/networks_solutions_models/solution_test__take_loss.json": "test_take_loss",
    "data/networks_solutions_models/solution_test__random.json": "test_random",
    "data/networks_solutions_models/solution_exp__myopic.json": "experiment_myopic",
    "data/networks_solutions_models/solution_exp__take_loss.json": "experiment_take_loss",
    "data/networks_solutions_models/solution_exp__random.json": "experiment_random",
}

POINTER_RE = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\s+"
    rb"oid sha256:([a-f0-9]{64})\s+size ([0-9]+)"
)


def digest_bytes(data: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_pointer(archive: zipfile.ZipFile, member_name: str) -> dict[str, Any]:
    matches = [
        name for name in archive.namelist()
        if name.endswith("/" + member_name) or name == member_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one archive member for {member_name}, found {len(matches)}")
    content = archive.read(matches[0])
    match = POINTER_RE.fullmatch(content.strip())
    if not match:
        raise RuntimeError(f"member is not a valid Git LFS pointer: {matches[0]}")
    return {
        "archive_path": matches[0],
        "relative_path": member_name,
        "oid": match.group(1).decode("ascii"),
        "size_bytes": int(match.group(2)),
    }


def lfs_batch_url(archive: zipfile.ZipFile, cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    for name in archive.namelist():
        if name.endswith("/.lfsconfig") or name == ".lfsconfig":
            text = archive.read(name).decode("utf-8", errors="replace")
            match = re.search(r"(?im)^\s*url\s*=\s*(\S+)\s*$", text)
            if match:
                url = match.group(1).rstrip("/")
                return url if url.endswith("/objects/batch") else url + "/objects/batch"
    return DEFAULT_LFS_BATCH


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "CivBench-TaskNetwork-Audit/0.1",
    }

    # The workflow's GITHUB_TOKEN is scoped to this repository and must not
    # be reused as an identity for the external research repository. Anonymous
    # access is attempted by default; an explicitly configured upstream
    # read-only credential may be supplied through the dedicated environment
    # variable. Never log or persist credentials or signed download URLs.
    token = os.environ.get("CIVBENCH_REWARD_NETWORK_LFS_TOKEN")
    if token:
        credential = base64.b64encode(
            f"x-access-token:{token}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {credential}"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code == 401:
            guidance = (
                "Configure CIVBENCH_REWARD_NETWORK_LFS_TOKEN with a read-only "
                "credential authorized for the upstream research repository."
            )
        elif exc.code == 404:
            guidance = (
                "The upstream LFS service reports that the repository is not "
                "available to the supplied identity. Verify the upstream URL "
                "and access rights; the current repository's GITHUB_TOKEN is "
                "intentionally not used for this external repository."
            )
        else:
            guidance = "Check the upstream LFS service response and retry policy."
        raise RuntimeError(
            f"Git LFS batch endpoint returned HTTP {exc.code}: {details}. {guidance}"
        ) from exc


def download_lfs_object(action: dict[str, Any], target: str) -> bytes:
    request = urllib.request.Request(
        action["href"],
        headers=action.get("header", {}),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    with open(target, "wb") as output:
        output.write(data)
    return data


def node_index(network: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(node["node_num"]): node for node in network.get("nodes", [])}


def summarize_networks(networks: list[dict[str, Any]]) -> dict[str, Any]:
    node_counts = Counter()
    edge_counts = Counter()
    levels = Counter()
    rewards = Counter()
    transition_counts = Counter()
    start_nodes_valid = 0
    invalid_endpoints = 0
    duplicate_directed_edges = 0
    reverse_pair_count = 0
    unique_ids: set[str] = set()
    duplicate_ids = 0

    for network in networks:
        network_id = str(network.get("network_id", ""))
        if network_id in unique_ids:
            duplicate_ids += 1
        unique_ids.add(network_id)

        nodes = network.get("nodes", [])
        edges = network.get("edges", [])
        node_counts[len(nodes)] += 1
        edge_counts[len(edges)] += 1

        index = node_index(network)
        levels.update(str(node.get("level")) for node in nodes)
        try:
            start_nodes_valid += int(int(network["starting_node"]) in index)
        except (KeyError, TypeError, ValueError):
            pass

        directed: set[tuple[int, int]] = set()
        for edge in edges:
            source = int(edge["source_num"])
            target = int(edge["target_num"])
            rewards[str(edge["reward"])] += 1
            if source not in index or target not in index:
                invalid_endpoints += 1
            if (source, target) in directed:
                duplicate_directed_edges += 1
            directed.add((source, target))
        reverse_pair_count += sum(
            1 for source, target in directed
            if source < target and (target, source) in directed
        )

    return {
        "network_count": len(networks),
        "unique_network_id_count": len(unique_ids),
        "duplicate_network_id_count": duplicate_ids,
        "node_count_distribution": dict(sorted(node_counts.items())),
        "edge_count_distribution": dict(sorted(edge_counts.items())),
        "node_level_distribution": dict(sorted(levels.items())),
        "edge_reward_distribution": dict(sorted(rewards.items())),
        "starting_node_valid_count": start_nodes_valid,
        "invalid_edge_endpoint_count": invalid_endpoints,
        "duplicate_directed_edge_count": duplicate_directed_edges,
        "reverse_directed_pair_count": reverse_pair_count,
        "network_schema_keys": sorted(networks[0].keys()) if networks else [],
    }


def summarize_solutions(
    solutions: list[dict[str, Any]],
    networks: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(net.get("network_id")): net for net in networks}
    solution_ids: set[str] = set()
    duplicate_solution_ids = 0
    unknown_network_ids = 0
    invalid_paths = 0
    transition_length = Counter()
    score_distribution = Counter()
    total_score = 0
    scored_count = 0

    for solution in solutions:
        network_id = str(solution.get("network_id", ""))
        if network_id in solution_ids:
            duplicate_solution_ids += 1
        solution_ids.add(network_id)
        network = by_id.get(network_id)
        if network is None:
            unknown_network_ids += 1
            continue
        moves = [int(move) for move in solution.get("moves", [])]
        transition_length[max(0, len(moves) - 1)] += 1
        edge_lookup = {
            (int(edge["source_num"]), int(edge["target_num"])): int(edge["reward"])
            for edge in network.get("edges", [])
        }
        score = 0
        valid = True
        for source, target in zip(moves[:-1], moves[1:]):
            reward = edge_lookup.get((source, target))
            if reward is None:
                valid = False
                break
            score += reward
        if not valid:
            invalid_paths += 1
            continue
        score_distribution[str(score)] += 1
        total_score += score
        scored_count += 1

    return {
        "solution_count": len(solutions),
        "unique_solution_network_id_count": len(solution_ids),
        "duplicate_solution_network_id_count": duplicate_solution_ids,
        "unknown_network_id_count": unknown_network_ids,
        "invalid_directed_path_count": invalid_paths,
        "transition_count_distribution": dict(sorted(transition_length.items())),
        "directed_edge_reward_sum_distribution": dict(sorted(score_distribution.items())),
        "mean_directed_edge_reward_sum": total_score / scored_count if scored_count else None,
        "scored_solution_count": scored_count,
        "solution_schema_keys": sorted(solutions[0].keys()) if solutions else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--lfs-batch-url")
    args = parser.parse_args()

    with open(args.archive, "rb") as source:
        archive_bytes = source.read()
    if len(archive_bytes) != ZENODO_SIZE:
        raise SystemExit(f"Zenodo archive size mismatch: expected {ZENODO_SIZE}, got {len(archive_bytes)}")
    archive_md5 = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()
    if archive_md5 != ZENODO_MD5:
        raise SystemExit(f"Zenodo MD5 mismatch: expected {ZENODO_MD5}, got {archive_md5}")
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    with zipfile.ZipFile(args.archive) as archive:
        pointers = {path: read_pointer(archive, path) for path in TASK_ASSETS}
        batch_url = lfs_batch_url(archive, args.lfs_batch_url)
    # Safe diagnostic: endpoint only, never credentials or signed object URLs.
    print(f"Git LFS batch endpoint selected: {batch_url}", file=sys.stderr)

    batch = post_json(
        batch_url,
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [
                {"oid": item["oid"], "size": item["size_bytes"]}
                for item in pointers.values()
            ],
        },
    )

    returned = {item["oid"]: item for item in batch.get("objects", [])}
    object_receipts: dict[str, Any] = {}
    hydrated: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="civbench-rn3-") as temporary:
        for relative_path, pointer in pointers.items():
            remote = returned.get(pointer["oid"])
            if remote is None:
                raise RuntimeError(f"LFS server omitted object {relative_path} ({pointer['oid']})")
            if remote.get("error"):
                raise RuntimeError(f"LFS error for {relative_path}: {remote['error']}")
            action = remote.get("actions", {}).get("download")
            if not action:
                raise RuntimeError(f"LFS server returned no download action for {relative_path}")
            local_path = str(PurePosixPath(temporary) / PurePosixPath(relative_path).name)
            payload = download_lfs_object(action, local_path)
            actual = digest_bytes(payload)
            if actual["size_bytes"] != pointer["size_bytes"]:
                raise RuntimeError(f"size mismatch for {relative_path}")
            if actual["sha256"] != pointer["oid"]:
                raise RuntimeError(f"SHA-256 mismatch for {relative_path}")
            object_receipts[relative_path] = {
                "oid_sha256": pointer["oid"],
                "size_bytes": pointer["size_bytes"],
                "verified": True,
            }
            hydrated[relative_path] = json.loads(payload.decode("utf-8"))

        network_paths = {
            "data/networks_solutions_models/networks_train.json": "train",
            "data/networks_solutions_models/networks_test.json": "test",
            "data/networks_solutions_models/networks_exp.json": "experiment",
        }
        network_objects = {
            label: hydrated[path] for path, label in network_paths.items()
        }
        all_ids: dict[str, str] = {}
        cross_split_duplicates: list[str] = []
        summaries = {}
        for label, networks in network_objects.items():
            ids = [str(network.get("network_id", "")) for network in networks]
            for network_id in ids:
                previous = all_ids.get(network_id)
                if previous is not None and previous != label:
                    cross_split_duplicates.append(network_id)
                all_ids[network_id] = label
            summaries[label] = summarize_networks(networks)

        solution_summaries = {}
        solution_mapping = {
            "train_myopic": ("data/networks_solutions_models/solution_train__myopic.json", "train"),
            "train_take_loss": ("data/networks_solutions_models/solution_train__take_loss.json", "train"),
            "train_random": ("data/networks_solutions_models/solution_train__random.json", "train"),
            "test_myopic": ("data/networks_solutions_models/solution_test__myopic.json", "test"),
            "test_take_loss": ("data/networks_solutions_models/solution_test__take_loss.json", "test"),
            "test_random": ("data/networks_solutions_models/solution_test__random.json", "test"),
            "experiment_myopic": ("data/networks_solutions_models/solution_exp__myopic.json", "experiment"),
            "experiment_take_loss": ("data/networks_solutions_models/solution_exp__take_loss.json", "experiment"),
            "experiment_random": ("data/networks_solutions_models/solution_exp__random.json", "experiment"),
        }
        for label, (path, split) in solution_mapping.items():
            solution_summaries[label] = summarize_solutions(
                hydrated[path], network_objects[split]
            )

    report = {
        "schema_version": "0.1.0",
        "source": {
            "zenodo_record": "https://zenodo.org/records/18547122",
            "doi": "10.5281/zenodo.18547122",
            "version": "v2.0",
            "archive_size_bytes": len(archive_bytes),
            "archive_md5": archive_md5,
            "archive_sha256": archive_sha256,
            "lfs_batch_endpoint": batch_url,
        },
        "assets": object_receipts,
        "network_splits": summaries,
        "split_audit": {
            "unique_network_ids_across_splits": len(all_ids),
            "cross_split_duplicate_id_count": len(set(cross_split_duplicates)),
            "cross_split_duplicate_ids_exported": False,
        },
        "solution_summaries": solution_summaries,
        "data_boundary": {
            "participant_data_hydrated": False,
            "raw_session_data_hydrated": False,
            "move_history_data_hydrated": False,
            "network_task_assets_hydrated_in_memory_only": True,
            "raw_network_or_solution_records_exported": False,
        },
        "limitations": [
            "This report contains aggregate generated-task network and solution diagnostics only.",
            "It does not reproduce the authors' participant-level analysis or DQN training.",
            "Solution sums use the source network's directed source_num -> target_num edges; missing-step penalties and interface timing are not modeled here.",
            "The exact game-action interpretation and ten-move horizon must be checked against the archived experiment code and article protocol.",
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({
        "network_splits": {key: value["network_count"] for key, value in summaries.items()},
        "cross_split_duplicate_count": report["split_audit"]["cross_split_duplicate_id_count"],
        "hydrated_asset_count": len(object_receipts),
        "participant_data_hydrated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
