"""Run the Broca causal-episode retrieval probe against a built Rust worker.

This is a research fixture runner, not an official LongMemEval evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"
MAX_LINE_BYTES = 32 * 1024 * 1024
RUNNER_VERSION = "broca-causal-episode-probe-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each record must be an object")
            rows.append(value)
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def worker_request(request_id: str, op: str, **fields: Any) -> dict[str, Any]:
    return {"request_id": request_id, "op": op, **fields}


def resolve_pointer(root: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid absolute JSON Pointer: {pointer!r}")
    current = root
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current) or str(index) != token:
                raise ValueError(f"invalid array index in JSON Pointer: {pointer}")
            current = current[index]
        else:
            raise ValueError(f"JSON Pointer does not resolve: {pointer}")
    return current


def validate_fixtures(trajectories: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    trajectory_by_id: dict[str, dict[str, Any]] = {}
    for index, trajectory in enumerate(trajectories):
        trajectory_id = trajectory.get("id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"trajectory {index}: id must be a non-empty string")
        if trajectory_id in trajectory_by_id:
            raise ValueError(f"duplicate trajectory id: {trajectory_id}")
        trajectory_by_id[trajectory_id] = trajectory

    question_ids: set[str] = set()
    for index, question in enumerate(questions):
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"question {index}: question_id must be a non-empty string")
        if question_id in question_ids:
            raise ValueError(f"duplicate question id: {question_id}")
        question_ids.add(question_id)
        if not isinstance(question.get("query"), str) or not question["query"].strip():
            raise ValueError(f"{question_id}: query must be a non-empty string")
        for field in ("required_support", "forbidden_support", "distractor_support"):
            refs = question.get(field, [])
            if not isinstance(refs, list):
                raise ValueError(f"{question_id}: {field} must be an array")
            seen_refs: set[tuple[str, str]] = set()
            for index_ref, ref in enumerate(refs, 1):
                if not isinstance(ref, dict):
                    raise ValueError(f"{question_id}: {field}[{index_ref}] must be an object")
                target_id = ref.get("trajectory_id")
                pointer = ref.get("source_pointer")
                if not isinstance(target_id, str) or target_id not in trajectory_by_id:
                    raise ValueError(f"{question_id}: {field}[{index_ref}] references unknown trajectory")
                key = (target_id, pointer) if isinstance(pointer, str) else (target_id, "")
                if key in seen_refs:
                    raise ValueError(f"{question_id}: duplicate pointer in {field}")
                seen_refs.add(key)
                if not isinstance(resolve_pointer(trajectory_by_id[target_id], pointer), str):
                    raise ValueError(f"{question_id}: {field}[{index_ref}] must resolve to a string")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path, help="built broca-lme-worker executable")
    parser.add_argument("--k", type=int, default=5, help="final comparison budget/top-k (1..100)")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="BM25 candidate pool to retain for fixed-budget retrieval ablations (must be >= k)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", required=True, type=Path, help="output prediction JSONL path")
    args = parser.parse_args()

    worker = args.worker.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not worker.is_file() or not worker.stat().st_mode & 0o111:
        raise SystemExit(f"worker must exist and be executable: {worker}")
    if not 1 <= args.k <= 100:
        raise SystemExit("--k must be in [1, 100]")
    if isinstance(args.candidate_k, bool) or not args.k <= args.candidate_k <= 100:
        raise SystemExit("--candidate-k must be in [k, 100]")
    if not 0.1 <= args.timeout_seconds <= 3600:
        raise SystemExit("--timeout-seconds must be in [0.1, 3600]")

    trajectory_path = FIXTURE_DIR / "trajectories.jsonl"
    question_path = FIXTURE_DIR / "questions.jsonl"
    trajectories = read_jsonl(trajectory_path)
    questions = read_jsonl(question_path)
    if len({row.get("id") for row in trajectories}) != len(trajectories):
        raise SystemExit("trajectory IDs must be unique")
    if len({row.get("question_id") for row in questions}) != len(questions):
        raise SystemExit("question IDs must be unique")
    if not trajectories or not questions:
        raise SystemExit("fixtures must contain trajectories and questions")
    validate_fixtures(trajectories, questions)

    requests: list[dict[str, Any]] = [
        worker_request(
            "probe:init",
            "init",
            scope_id="web",
            now_unix_ms=0,
            bm25={"k1": 1.2, "b": 0.75, "max_items": args.candidate_k},
        )
    ]
    expected_ids: list[str] = ["probe:init"]
    for trajectory in trajectories:
        request_id = "probe:insert:" + str(trajectory["id"])
        requests.append(worker_request(request_id, "insert", trajectory=trajectory))
        expected_ids.append(request_id)
    for question in questions:
        request_id = "probe:query:" + str(question["question_id"])
        requests.append(worker_request(request_id, "query", query=question["query"]))
        expected_ids.append(request_id)

    payload = "".join(
        json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for request in requests
    )
    if any(len(line.encode("utf-8")) > MAX_LINE_BYTES for line in payload.splitlines()):
        raise SystemExit("a generated request exceeds the worker line bound")

    completed = subprocess.run(
        [str(worker)],
        input=payload,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"worker exited {completed.returncode}; stderr follows:\n{completed.stderr[-8000:]}"
        )
    if completed.stderr.strip():
        raise SystemExit(f"worker wrote unexpected stderr: {completed.stderr[-8000:]}")

    response_lines = completed.stdout.splitlines()
    if len(response_lines) != len(expected_ids):
        raise SystemExit(
            f"expected {len(expected_ids)} response lines, received {len(response_lines)}"
        )
    responses: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(response_lines, 1):
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise SystemExit(f"response line {line_number} exceeds byte bound")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise SystemExit(f"response line {line_number} is not an object")
        request_id = response.get("request_id")
        if request_id not in expected_ids or request_id in responses:
            raise SystemExit(f"unexpected or duplicate response request_id: {request_id!r}")
        if response.get("ok") is not True:
            raise SystemExit(
                f"worker rejected {request_id}: {response.get('error', 'unknown error')}"
            )
        if not isinstance(response.get("result"), dict):
            raise SystemExit(f"worker response {request_id} has no result object")
        responses[request_id] = response["result"]
    if set(responses) != set(expected_ids):
        raise SystemExit("worker response ID set does not match request ID set")

    init_result = responses["probe:init"]
    capabilities = init_result.get("capabilities", {})
    if capabilities.get("retrieval_type") != "lexical_bm25":
        raise SystemExit("worker did not identify itself as lexical_bm25")
    if capabilities.get("semantic_encoder") is not False:
        raise SystemExit("worker capability declaration is missing or misleading")

    predictions: list[dict[str, Any]] = []
    for question in questions:
        request_id = "probe:query:" + str(question["question_id"])
        result = responses[request_id]
        audit = result.get("audit")
        if not isinstance(audit, list):
            raise SystemExit(f"{request_id}: audit must be an array")
        predictions.append({"question_id": question["question_id"], "audit": audit})

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in predictions
    )
    output.write_text(serialized, encoding="utf-8")
    manifest = {
        "schema": RUNNER_VERSION,
        "worker_path": str(worker),
        "worker_sha256": sha256_bytes(worker.read_bytes()),
        "scope_id": "web",
        "bm25": {"k1": 1.2, "b": 0.75, "max_items": args.candidate_k},
        "evaluation_k": args.k,
        "protocol": "bounded JSONL; worker request_id and ok checked",
        "trajectory_fixture_sha256": sha256_bytes(trajectory_path.read_bytes()),
        "question_fixture_sha256": sha256_bytes(question_path.read_bytes()),
        "trajectory_count": len(trajectories),
        "question_count": len(questions),
        "prediction_path": str(output),
        "prediction_sha256": sha256_bytes(output.read_bytes()),
        "claim_ceiling": "synthetic retrieval probe only; not answer accuracy or causal proof",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(output), "manifest": str(manifest_path), "question_count": len(questions)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
