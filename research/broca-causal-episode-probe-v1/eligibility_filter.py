"""Eligibility-first retrieval ablation with oracle and keyword-intent modes.

This is a research evaluator, not a production intent classifier or worker patch.
It filters BM25 candidates using a versioned source eligibility ledger before the
final top-k budget. Gold support labels are never read by the selection policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "mutations"
MAX_LINE_BYTES = 32 * 1024 * 1024
RUNNER_SCHEMA = "broca-causal-episode-probe-v1"
FILTER_SCHEMA = "broca-eligibility-first-ablation-v2"
INTENT_CLASSIFIER_SCHEMA = "broca-keyword-intent-v1"
LEDGER_SCHEMA = "broca-eligibility-ledger-v1"

ALLOWED_STATUSES = {
    "contextual", "current", "historical", "revoked", "tentative",
    "not_applied", "causal",
}
ELIGIBLE_BY_INTENT: dict[str, frozenset[str]] = {
    "current_state": frozenset({"current", "contextual"}),
    "historical_state": frozenset({"historical", "contextual"}),
    "revocation_status": frozenset({"revoked", "contextual"}),
    "approved_state": frozenset({"current", "contextual"}),
    "proposal_history": frozenset({"tentative", "not_applied", "contextual"}),
    # A causal-chain query is allowed to inspect old and failed events; otherwise
    # filtering would erase the negative intervention needed to interpret outcome.
    "causal_chain": frozenset(ALLOWED_STATUSES),
}


def infer_retrieval_intent(query: str) -> tuple[str | None, str]:
    """Transparent keyword baseline; unknown queries return None for fail-closed abstention."""
    if not isinstance(query, str) or not query.strip():
        return None, "empty_query"
    text = " ".join(query.casefold().split())

    # Order matters: proposal history is more specific than the word "approved";
    # revocation is more specific than generic state/currentness language.
    if any(term in text for term in ("proposed", "proposal", "never approved", "not approved")):
        return "proposal_history", "proposal_lexeme"
    if "credential" in text and any(term in text for term in ("revok", "still be used", "active")):
        return "revocation_status", "credential_status"
    if "approved" in text and "deployed" in text:
        return "approved_state", "approved_and_deployed"
    if (
        "preceded" in text and any(term in text for term in ("recovery", "spike", "outcome"))
    ) or ("failed" in text and "which action" in text):
        return "causal_chain", "event_sequence"
    if (
        any(term in text for term in ("before ", "prior to ", "during migration testing", "previously"))
    ):
        return "historical_state", "historical_temporal_cue"
    if any(term in text for term in ("current", "currently", "right now", "at present", "today", "now")):
        return "current_state", "current_temporal_cue"
    return None, "no_confident_rule"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl_bytes(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"{path}: JSONL must not be empty")
    return records, payload


def evidence_key(item: dict[str, Any], context: str) -> tuple[str, str]:
    trajectory_id = item.get("trajectory_id")
    source_pointer = item.get("source_pointer")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"{context}: trajectory_id must be a non-empty string")
    if not isinstance(source_pointer, str) or not source_pointer.startswith("/"):
        raise ValueError(f"{context}: source_pointer must be an absolute JSON Pointer")
    if "~" in source_pointer.replace("~0", "").replace("~1", ""):
        raise ValueError(f"{context}: malformed JSON Pointer escape")
    return trajectory_id, source_pointer


def load_ledger(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    ledger: dict[tuple[str, str], dict[str, str]] = {}
    for index, row in enumerate(records):
        if row.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"ledger row {index}: wrong or missing schema")
        key = evidence_key(row, f"ledger row {index}")
        status = row.get("status")
        reason = row.get("reason")
        if not isinstance(status, str) or status not in ALLOWED_STATUSES:
            raise ValueError(f"ledger row {index}: unsupported status {status!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"ledger row {index}: reason must be a non-empty string")
        if key in ledger:
            raise ValueError(f"duplicate ledger key: {key}")
        ledger[key] = {"status": status, "reason": reason}
    return ledger


def resolve_json_pointer(root: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer with canonical array indexes."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid absolute source pointer: {pointer!r}")
    current = root
    for raw in pointer.split("/")[1:]:
        if "~" in raw.replace("~0", "").replace("~1", ""):
            raise ValueError(f"malformed JSON Pointer escape: {pointer}")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and str(int(token)) == token:
            index = int(token)
            if index >= len(current):
                raise ValueError(f"JSON Pointer array index out of range: {pointer}")
            current = current[index]
        else:
            raise ValueError(f"JSON Pointer does not resolve: {pointer}")
    return current


def iter_string_pointers(value: Any, prefix: str = "") -> list[str]:
    """Return all JSON Pointers to string-valued leaves in deterministic traversal order."""
    if isinstance(value, str):
        return [prefix]
    if isinstance(value, list):
        return [
            pointer
            for index, child in enumerate(value)
            for pointer in iter_string_pointers(child, f"{prefix}/{index}")
        ]
    if isinstance(value, dict):
        pointers: list[str] = []
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            pointers.extend(iter_string_pointers(child, f"{prefix}/{escaped}"))
        return pointers
    return []


def validate_ledger_coverage(
    trajectories: list[dict[str, Any]],
    ledger_records: list[dict[str, Any]],
) -> None:
    """Require exact coverage of every source string leaf and reject dangling pointers."""
    indexed: dict[str, dict[str, Any]] = {}
    for index, trajectory in enumerate(trajectories):
        trajectory_id = trajectory.get("id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"trajectory {index}: id must be a non-empty string")
        if trajectory_id in indexed:
            raise ValueError(f"duplicate trajectory id: {trajectory_id}")
        indexed[trajectory_id] = trajectory

    ledger_keys = set(load_ledger(ledger_records))
    expected = {
        (trajectory["id"], pointer)
        for trajectory in trajectories
        for pointer in iter_string_pointers(trajectory)
    }
    missing = expected - ledger_keys
    extra = ledger_keys - expected
    if missing:
        raise ValueError(f"eligibility ledger omits source string pointers: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"eligibility ledger has dangling source pointers: {sorted(extra)[:5]}")
    for trajectory_id, pointer in ledger_keys:
        try:
            value = resolve_json_pointer(indexed[trajectory_id], pointer)
        except ValueError as error:
            raise ValueError(f"eligibility ledger pointer does not resolve: {trajectory_id}{pointer}") from error
        if not isinstance(value, str):
            raise ValueError(f"eligibility ledger target is not a string: {trajectory_id}{pointer}")


def _index_questions(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(questions):
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"question {index}: question_id must be a non-empty string")
        if question_id in indexed:
            raise ValueError(f"duplicate question_id: {question_id}")
        intent = question.get("retrieval_intent")
        if not isinstance(intent, str) or intent not in ELIGIBLE_BY_INTENT:
            raise ValueError(f"{question_id}: unknown or missing retrieval_intent {intent!r}")
        indexed[question_id] = question
    if not indexed:
        raise ValueError("questions must not be empty")
    return indexed


def filter_dataset(
    questions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    ledger_records: list[dict[str, Any]],
    k: int,
    intent_source: str = "annotated",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter candidate evidence in rank order, then enforce an exact top-k ceiling.

    Unknown ledger entries fail closed (they are discarded, never passed through).
    This function intentionally does not read required_support, forbidden_support,
    distractor_support, answers, or any evaluator outcome.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k must be an integer in [1, 100]")
    if not isinstance(intent_source, str) or intent_source not in {"annotated", "heuristic"}:
        raise ValueError("intent_source must be annotated or heuristic")
    question_by_id = _index_questions(questions)
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        question_id = prediction.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"prediction {index}: question_id must be a non-empty string")
        if question_id in prediction_by_id:
            raise ValueError(f"duplicate prediction question_id: {question_id}")
        if question_id not in question_by_id:
            raise ValueError(f"unexpected prediction question_id: {question_id}")
        if not isinstance(prediction.get("audit"), list):
            raise ValueError(f"{question_id}: audit must be an array")
        prediction_by_id[question_id] = prediction
    missing = sorted(set(question_by_id) - set(prediction_by_id))
    if missing:
        raise ValueError(f"missing predictions for question IDs: {missing}")

    ledger = load_ledger(ledger_records)
    output: list[dict[str, Any]] = []
    aggregate = {
        "question_count": len(question_by_id),
        "input_candidate_count": 0,
        "eligible_before_budget_count": 0,
        "output_count": 0,
        "removed_unannotated_count": 0,
        "removed_by_status": {status: 0 for status in sorted(ALLOWED_STATUSES)},
        "questions_below_k": 0,
        "intent_source": intent_source,
        "intent_evaluation": {"correct": 0, "unknown": 0, "rows": []},
        "per_intent": {},
    }

    for question_id in sorted(question_by_id):
        question = question_by_id[question_id]
        prediction = prediction_by_id[question_id]
        expected_intent = question["retrieval_intent"]
        if intent_source == "annotated":
            intent: str | None = expected_intent
            rule = "oracle_annotation"
        else:
            intent, rule = infer_retrieval_intent(question.get("query", ""))
        policy_intent = intent if intent is not None else "abstain"
        allowed = ELIGIBLE_BY_INTENT.get(intent, frozenset())
        intent_correct = intent == expected_intent
        aggregate["intent_evaluation"]["correct"] += int(intent_correct)
        aggregate["intent_evaluation"]["unknown"] += int(intent is None)
        aggregate["intent_evaluation"]["rows"].append({
            "question_id": question_id,
            "expected_intent": expected_intent,
            "predicted_intent": intent,
            "rule": rule,
            "correct": intent_correct,
        })
        seen: set[tuple[str, str]] = set()
        eligible: list[dict[str, Any]] = []
        candidates = prediction["audit"]
        if len(candidates) > 100:
            raise ValueError(f"{question_id}: candidate pool exceeds the 100-item contract")
        for candidate_index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError(f"{question_id}: audit item {candidate_index} must be an object")
            key = evidence_key(item, f"{question_id} candidate {candidate_index}")
            if key in seen:
                raise ValueError(f"{question_id}: duplicate candidate evidence key {key}")
            seen.add(key)
            record = ledger.get(key)
            if record is None:
                aggregate["removed_unannotated_count"] += 1
                continue
            status = record["status"]
            if status not in allowed:
                aggregate["removed_by_status"][status] += 1
                continue
            eligible.append(item)

        final = eligible[:k]
        aggregate["input_candidate_count"] += len(candidates)
        aggregate["eligible_before_budget_count"] += len(eligible)
        aggregate["output_count"] += len(final)
        if len(final) < k:
            aggregate["questions_below_k"] += 1
        intent_stats = aggregate["per_intent"].setdefault(
            policy_intent, {"questions": 0, "input": 0, "eligible": 0, "output": 0}
        )
        intent_stats["questions"] += 1
        intent_stats["input"] += len(candidates)
        intent_stats["eligible"] += len(eligible)
        intent_stats["output"] += len(final)
        # Preserve the original audit item fields and order among eligible records.
        output.append({"question_id": question_id, "audit": final})

    aggregate["removed_total"] = (
        aggregate["input_candidate_count"] - aggregate["output_count"]
    )
    aggregate["intent_evaluation"]["accuracy"] = round(
        aggregate["intent_evaluation"]["correct"] / len(question_by_id), 6
    )
    aggregate["interpretation"] = (
        "oracle-intent eligibility upper bound"
        if intent_source == "annotated"
        else "transparent keyword-intent baseline; fixture scores are not held-out generalization evidence"
    )
    aggregate["interpretation"] += (
        "; unknown intent fails closed to an empty evidence context; no gold support arrays consumed"
    )
    return output, aggregate


def verify_input_manifest(
    predictions_path: Path,
    questions_path: Path,
    trajectories_path: Path,
    k: int,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    prediction_rows, prediction_bytes = read_jsonl_bytes(predictions_path)
    questions, question_bytes = read_jsonl_bytes(questions_path)
    trajectories, trajectory_bytes = read_jsonl_bytes(trajectories_path)
    manifest_path = predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != RUNNER_SCHEMA:
        raise ValueError("candidate receipt has wrong or missing schema")
    if manifest.get("prediction_sha256") != sha256_bytes(prediction_bytes):
        raise ValueError("candidate prediction digest does not match its receipt")
    if manifest.get("question_fixture_sha256") != sha256_bytes(question_bytes):
        raise ValueError("question fixture digest does not match candidate receipt")
    if manifest.get("trajectory_fixture_sha256") != sha256_bytes(trajectory_bytes):
        raise ValueError("trajectory fixture digest does not match candidate receipt")
    worker_sha = manifest.get("worker_sha256")
    if not isinstance(worker_sha, str) or len(worker_sha) != 64 or any(
        character not in "0123456789abcdef" for character in worker_sha
    ):
        raise ValueError("candidate receipt has invalid worker SHA-256")
    bm25 = manifest.get("bm25")
    if not isinstance(bm25, dict):
        raise ValueError("candidate receipt has no BM25 configuration")
    candidate_k = bm25.get("max_items")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < k:
        raise ValueError("candidate pool must be at least the final top-k budget")
    if manifest.get("question_count") != len(questions):
        raise ValueError("candidate receipt question_count mismatch")
    if manifest.get("trajectory_count") != len(trajectories):
        raise ValueError("candidate receipt trajectory_count mismatch")
    trajectory_ids = [row.get("id") for row in trajectories]
    question_ids = [row.get("question_id") for row in questions]
    if any(not isinstance(value, str) or not value for value in trajectory_ids):
        raise ValueError("trajectory IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in question_ids):
        raise ValueError("question IDs must be non-empty strings")
    if len(set(trajectory_ids)) != len(trajectories):
        raise ValueError("trajectory IDs must be unique")
    if len(set(question_ids)) != len(questions):
        raise ValueError("question IDs must be unique")
    prediction_ids = [row.get("question_id") for row in prediction_rows]
    if any(not isinstance(value, str) or not value for value in prediction_ids):
        raise ValueError("candidate prediction IDs must be non-empty strings")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("candidate prediction IDs must be unique")
    if set(prediction_ids) != set(question_ids):
        raise ValueError("candidate prediction question IDs do not match the fixture")
    return manifest, prediction_bytes, question_bytes, trajectory_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=FIXTURE_DIR / "questions.jsonl")
    parser.add_argument("--trajectories", type=Path, default=FIXTURE_DIR / "trajectories.jsonl")
    parser.add_argument("--ledger", type=Path, default=FIXTURE_DIR / "eligibility-ledger.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--intent-source",
        choices=("annotated", "heuristic"),
        default="annotated",
        help="use annotated oracle intent or the transparent keyword baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.predictions.expanduser().resolve()
    questions_path = args.questions.expanduser().resolve()
    trajectories_path = args.trajectories.expanduser().resolve()
    ledger_path = args.ledger.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_input_path = predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")
    output_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    input_paths = {predictions_path, questions_path, trajectories_path, ledger_path, manifest_input_path}
    if output_path in input_paths or output_manifest_path in input_paths:
        raise SystemExit("output paths must not overwrite predictions, manifests, fixtures, or the policy ledger")
    input_manifest, prediction_bytes, question_bytes, trajectory_bytes = verify_input_manifest(
        predictions_path, questions_path, trajectories_path, args.k
    )
    predictions, _ = read_jsonl_bytes(predictions_path)
    questions, _ = read_jsonl_bytes(questions_path)
    ledger_records, ledger_bytes = read_jsonl_bytes(ledger_path)
    trajectories, _ = read_jsonl_bytes(trajectories_path)
    validate_ledger_coverage(trajectories, ledger_records)
    filtered, metrics = filter_dataset(
        questions, predictions, ledger_records, args.k, intent_source=args.intent_source
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in filtered
    ).encode("utf-8")
    output_path.write_bytes(output_bytes)
    ledger = {
        "schema": FILTER_SCHEMA,
        "ledger_schema": LEDGER_SCHEMA,
        "candidate_manifest_sha256": sha256_bytes(
            predictions_path.with_suffix(predictions_path.suffix + ".manifest.json").read_bytes()
        ),
        "filter_code_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "candidate_prediction_sha256": sha256_bytes(prediction_bytes),
        "worker_sha256": input_manifest["worker_sha256"],
        "question_fixture_sha256": sha256_bytes(question_bytes),
        "trajectory_fixture_sha256": sha256_bytes(trajectory_bytes),
        "eligibility_ledger_sha256": sha256_bytes(ledger_bytes),
        "scope_id": input_manifest.get("scope_id"),
        "bm25": input_manifest["bm25"],
        "candidate_pool_size": input_manifest["bm25"]["max_items"],
        "final_k": args.k,
        "intent_source": args.intent_source,
        "intent_classifier_schema": (
            INTENT_CLASSIFIER_SCHEMA if args.intent_source == "heuristic" else None
        ),
        "intent_policy": {
            key: sorted(value) for key, value in sorted(ELIGIBLE_BY_INTENT.items())
        },
        "question_count": len(filtered),
        "prediction_sha256": sha256_bytes(output_bytes),
        "metrics": metrics,
        "claim_ceiling": (
            "synthetic currentness ablation; not official answer accuracy; intent_source=" + args.intent_source
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({
        "predictions": str(output_path),
        "manifest": str(manifest_path),
        "question_count": len(filtered),
        "safe_eligibility_policy": True,
        "intent_source": args.intent_source,
        "intent_accuracy": metrics["intent_evaluation"]["accuracy"],
        "intent_unknown_count": metrics["intent_evaluation"]["unknown"],
        "output_sha256": sha256_bytes(output_bytes),
        "removed_total": metrics["removed_total"],
        "questions_below_k": metrics["questions_below_k"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
