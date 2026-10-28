"""Fixed-budget episode-neighborhood expansion ablation for Broca.

Uses only indexed fields from the trajectory fixture and BM25 audit candidates.
It never reads question support labels, expected answers, or evaluator metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"
MAX_LINE_BYTES = 32 * 1024 * 1024
POLICY_VERSION = "broca-episode-neighborhood-expand-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(item)
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_candidate_manifest(predictions_path: Path, trajectory_path: Path, evaluation_k: int) -> tuple[dict[str, Any], Path]:
    """Require exact source-run provenance before deriving another retrieval arm."""
    manifest_path = predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"candidate-run manifest is required: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"candidate-run manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("candidate-run manifest must be an object")
    if manifest.get("schema") != "broca-causal-episode-probe-v1":
        raise ValueError("candidate-run manifest schema mismatch")
    if manifest.get("prediction_sha256") != sha256_bytes(predictions_path.read_bytes()):
        raise ValueError("candidate prediction digest does not match its manifest")
    if manifest.get("trajectory_fixture_sha256") != sha256_bytes(trajectory_path.read_bytes()):
        raise ValueError("candidate manifest trajectory fixture digest mismatch")
    if manifest.get("scope_id") != "web":
        raise ValueError("candidate manifest scope must be exactly 'web' for this fixture")
    worker_sha256 = manifest.get("worker_sha256")
    if not isinstance(worker_sha256, str) or len(worker_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in worker_sha256
    ):
        raise ValueError("candidate manifest must bind a lowercase worker SHA-256")
    bm25 = manifest.get("bm25")
    if not isinstance(bm25, dict):
        raise ValueError("candidate manifest requires a BM25 configuration")
    candidate_k = bm25.get("max_items")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or not evaluation_k <= candidate_k <= 100:
        raise ValueError("candidate pool must be an integer in [evaluation_k, 100]")
    question_count = manifest.get("question_count")
    if isinstance(question_count, bool) or not isinstance(question_count, int) or question_count < 1:
        raise ValueError("candidate manifest question_count must be a positive integer")
    return manifest, manifest_path


def extract_indexed_refs(trajectories: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Mirror the worker's public-state/content text allowlist for this probe."""
    result: dict[tuple[str, str], dict[str, Any]] = {}

    def add(trajectory_id: str, pointer: str, text: Any, step_index: int | None, role: str) -> None:
        if isinstance(text, str) and text.strip():
            key = (trajectory_id, pointer)
            if key in result:
                raise ValueError(f"duplicate indexed source pointer: {key}")
            result[key] = {
                "trajectory_id": trajectory_id,
                "source_pointer": pointer,
                "step_index": step_index,
                "role": role,
            }

    for trajectory in trajectories:
        trajectory_id = trajectory.get("id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("every trajectory requires a non-empty id")

        states = trajectory.get("states")
        public_states = isinstance(states, list) and bool(states)
        raw_goal = trajectory.get("goal")
        if isinstance(raw_goal, str) and raw_goal.strip():
            add(trajectory_id, "/goal", raw_goal, None, "goal")
        elif isinstance(trajectory.get("metadata"), dict):
            original_goal = trajectory["metadata"].get("original_goal")
            if isinstance(original_goal, str):
                add(trajectory_id, "/metadata/original_goal", original_goal, None, "goal")
            elif isinstance(original_goal, list):
                for i, value in enumerate(original_goal):
                    add(trajectory_id, f"/metadata/original_goal/{i}", value, None, "goal")

        start_url = trajectory.get("start_url")
        if public_states and (not isinstance(trajectory.get("goal"), str)
                              or not isinstance(start_url, str) or not start_url.strip()):
            raise ValueError(f"{trajectory_id}: public-state trajectories require goal and start_url")
        add(trajectory_id, "/start_url", start_url, None, "url")

        if public_states:
            for i, state in enumerate(states):
                if not isinstance(state, dict):
                    raise ValueError(f"{trajectory_id}: states[{i}] must be an object")
                prefix = f"/states/{i}"
                url = state.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(f"{trajectory_id}: states[{i}].url must be non-empty")
                screenshot = state.get("screenshot")
                if not isinstance(screenshot, str) or not screenshot.strip():
                    raise ValueError(f"{trajectory_id}: states[{i}].screenshot must be non-empty")
                add(trajectory_id, f"{prefix}/url", url, i, "url")
                action = state.get("action")
                if action is not None and not isinstance(action, str):
                    raise ValueError(f"{trajectory_id}: states[{i}].action must be string/null")
                add(trajectory_id, f"{prefix}/action", action, i, "action")
                thought_key = "thought" if "thought" in state else "thoughts"
                thought = state.get(thought_key)
                if thought is not None and not isinstance(thought, str):
                    raise ValueError(f"{trajectory_id}: states[{i}].{thought_key} must be string/null")
                add(trajectory_id, f"{prefix}/{thought_key}", thought, i, "thought")
                text_key = "accessibility_tree" if "accessibility_tree" in state else "text"
                text = state.get(text_key)
                if not isinstance(text, str):
                    raise ValueError(f"{trajectory_id}: states[{i}].{text_key} must be a string")
                add(trajectory_id, f"{prefix}/{text_key}", text, i, "observation")
        else:
            content = trajectory.get("content")
            if not isinstance(content, list) or not content:
                raise ValueError(f"{trajectory_id}: require non-empty states or content")
            if not isinstance(trajectory.get("metadata"), dict):
                raise ValueError(f"{trajectory_id}: content-schema trajectories require metadata")
            for i, state in enumerate(content):
                if not isinstance(state, dict):
                    raise ValueError(f"{trajectory_id}: content[{i}] must be an object")
                prefix = f"/content/{i}"
                url = state.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(f"{trajectory_id}: content[{i}].url must be non-empty")
                add(trajectory_id, f"{prefix}/url", url, i, "url")
                for key, role in (("action", "action"), ("thoughts", "thought")):
                    value = state.get(key)
                    if value is not None and not isinstance(value, str):
                        raise ValueError(f"{trajectory_id}: content[{i}].{key} must be string/null")
                    add(trajectory_id, f"{prefix}/{key}", value, i, role)
                observation = state.get("observation")
                if not isinstance(observation, dict) or not isinstance(observation.get("text"), str):
                    raise ValueError(f"{trajectory_id}: content[{i}].observation.text must be a string")
                screenshot = observation.get("screenshot")
                if not isinstance(screenshot, str) or not screenshot.strip():
                    raise ValueError(f"{trajectory_id}: content[{i}].observation.screenshot must be non-empty")
                add(trajectory_id, f"{prefix}/observation/text", observation["text"], i, "observation")
    return result


ROLE_PRIORITY = {"action": 0, "observation": 1, "thought": 2, "url": 3, "goal": 4}


def expand_question(
    audit: list[dict[str, Any]],
    indexed_refs: dict[tuple[str, str], dict[str, Any]],
    k: int,
    window: int = 1,
) -> list[dict[str, Any]]:
    """Expand ranked seeds through same/adjacent steps; strict k-item final budget."""
    if not isinstance(audit, list):
        raise ValueError("audit must be an array")
    if not 1 <= k <= 100:
        raise ValueError("k must be in [1, 100]")
    if not 0 <= window <= 10:
        raise ValueError("window must be in [0, 10]")

    trajectories: dict[str, list[tuple[str, str]]] = {}
    for key, meta in indexed_refs.items():
        if meta["step_index"] is not None:
            trajectories.setdefault(meta["trajectory_id"], []).append(key)
    for keys in trajectories.values():
        keys.sort(key=lambda key: (
            indexed_refs[key]["step_index"],
            ROLE_PRIORITY[indexed_refs[key]["role"]],
            key[1],
        ))

    seed_keys: list[tuple[str, str]] = []
    seen_seed: set[tuple[str, str]] = set()
    for rank, item in enumerate(audit, 1):
        if not isinstance(item, dict):
            raise ValueError(f"audit item {rank} must be an object")
        trajectory_id, pointer = item.get("trajectory_id"), item.get("source_pointer")
        key = (trajectory_id, pointer)
        meta = indexed_refs.get(key)
        if meta is None:
            raise ValueError(f"BM25 audit points outside indexed source allowlist: {key}")
        if key in seen_seed:
            raise ValueError(f"duplicate BM25 evidence pointer: {key}")
        seen_seed.add(key)
        observed_step = item.get("step_index")
        if observed_step != meta["step_index"]:
            raise ValueError(f"BM25 step_index disagrees with source pointer: {key}")
        seed_keys.append(key)

    output: list[dict[str, Any]] = []
    selected: set[tuple[str, str]] = set()

    def add(key: tuple[str, str], role: str, seed: tuple[str, str], distance: int) -> None:
        if key in selected or len(output) >= k:
            return
        meta = indexed_refs.get(key)
        if meta is None:
            raise ValueError(f"episode expansion generated unindexed pointer: {key}")
        selected.add(key)
        output.append({
            "trajectory_id": key[0],
            "source_pointer": key[1],
            "step_index": meta["step_index"],
            "retrieval_role": role,
            "seed_source_pointer": seed[1],
            "episode_distance": distance,
        })

    for seed in seed_keys:
        if len(output) >= k:
            break
        seed_meta = indexed_refs[seed]
        add(seed, "bm25_seed", seed, 0)
        if len(output) >= k:
            break
        seed_step = seed_meta["step_index"]
        if seed_step is None:
            continue
        neighbors = []
        for candidate in trajectories.get(seed[0], []):
            if candidate == seed:
                continue
            candidate_meta = indexed_refs[candidate]
            distance = abs(candidate_meta["step_index"] - seed_step)
            if distance <= window:
                # Same step first; then newer step before older step at equal
                # distance. Role priority is fixed before looking at gold labels.
                direction = 0 if distance == 0 else (0 if candidate_meta["step_index"] > seed_step else 1)
                neighbors.append((
                    distance,
                    direction,
                    ROLE_PRIORITY[candidate_meta["role"]],
                    candidate[1],
                    candidate,
                ))
        neighbors.sort()
        for distance, _direction, _role_priority, _pointer, candidate in neighbors:
            add(candidate, "episode_neighbor", seed, distance)
            if len(output) >= k:
                break

    return output


def expand_predictions(
    trajectories: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    k: int,
    window: int = 1,
) -> list[dict[str, Any]]:
    indexed_refs = extract_indexed_refs(trajectories)
    question_ids: set[str] = set()
    expanded: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        question_id = prediction.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"prediction {index}: invalid question_id")
        if question_id in question_ids:
            raise ValueError(f"duplicate prediction question_id: {question_id}")
        question_ids.add(question_id)
        expanded.append({
            "question_id": question_id,
            "audit": expand_question(prediction.get("audit"), indexed_refs, k, window),
        })
    if not expanded:
        raise ValueError("predictions must not be empty")
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="BM25 candidate-pool JSONL from run_probe.py")
    parser.add_argument("--k", type=int, default=5, help="fixed final evidence budget (1..100)")
    parser.add_argument("--window", type=int, default=1, help="same-step / adjacent-step expansion window (0..10)")
    parser.add_argument("--output", required=True, type=Path, help="episode-expanded prediction JSONL")
    args = parser.parse_args()
    if not 1 <= args.k <= 100:
        raise SystemExit("--k must be in [1, 100]")
    if not 0 <= args.window <= 10:
        raise SystemExit("--window must be in [0, 10]")

    trajectory_path = FIXTURE_DIR / "trajectories.jsonl"
    candidate_manifest, candidate_manifest_path = load_candidate_manifest(
        args.predictions, trajectory_path, args.k
    )
    trajectories = read_jsonl(trajectory_path)
    predictions = read_jsonl(args.predictions)
    if len(predictions) != candidate_manifest["question_count"]:
        raise ValueError("candidate prediction count does not match its manifest")
    candidate_k = candidate_manifest["bm25"]["max_items"]
    for index, prediction in enumerate(predictions):
        audit = prediction.get("audit")
        if not isinstance(audit, list) or len(audit) > candidate_k:
            raise ValueError(f"prediction {index}: audit exceeds the declared candidate pool")
    expanded = expand_predictions(trajectories, predictions, args.k, args.window)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in expanded
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    manifest = {
        "schema": POLICY_VERSION,
        "trajectory_fixture_sha256": sha256_bytes(trajectory_path.read_bytes()),
        "input_candidate_predictions_sha256": sha256_bytes(args.predictions.read_bytes()),
        "input_candidate_manifest_sha256": sha256_bytes(candidate_manifest_path.read_bytes()),
        "input_worker_sha256": candidate_manifest["worker_sha256"],
        "input_bm25_config": candidate_manifest["bm25"],
        "input_scope_id": candidate_manifest["scope_id"],
        "input_candidate_k": candidate_manifest["bm25"]["max_items"],
        "output_predictions_sha256": sha256_bytes(output.read_bytes()),
        "output_path": str(output),
        "evaluation_k": args.k,
        "candidate_window": args.window,
        "policy": f"BM25 rank-order seeds; expand within step distance <= {args.window} by fixed role order; de-duplicate; stop at exactly k",
        "reads_gold_support_labels": False,
        "claim_ceiling": "retrieval policy ablation only; no causal truth or benchmark result",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(output), "manifest": str(manifest_path), "question_count": len(expanded)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
