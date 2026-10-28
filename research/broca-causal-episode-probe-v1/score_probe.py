"""Strict scorer for Broca causal-episode probe retrieval audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"
MAX_LINE_BYTES = 32 * 1024 * 1024


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each record must be an object")
            records.append(value)
    return records


def evidence_key(item: dict[str, Any], context: str) -> tuple[str, str]:
    trajectory_id = item.get("trajectory_id")
    source_pointer = item.get("source_pointer")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"{context}: trajectory_id must be a non-empty string")
    if not isinstance(source_pointer, str) or not source_pointer.startswith("/"):
        raise ValueError(f"{context}: source_pointer must be an RFC 6901-style absolute pointer")
    return trajectory_id, source_pointer


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _metrics(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    required_total = required_hits = complete = reciprocal_sum = 0
    forbidden_total = forbidden_hits = forbidden_questions = 0
    distractor_total = distractor_hits = 0

    for row in rows:
        rank = row["ranked_keys"][:k]
        rank_set = set(rank)
        required = set(row["required"])
        forbidden = set(row["forbidden"])
        distractors = set(row["distractors"])

        hit_count = len(required & rank_set)
        required_total += len(required)
        required_hits += hit_count
        if required and required <= rank_set:
            complete += 1
        first_rank = next((index + 1 for index, key in enumerate(rank) if key in required), None)
        reciprocal_sum += 0.0 if first_rank is None else 1.0 / first_rank

        row_forbidden_hits = len(forbidden & rank_set)
        forbidden_total += len(forbidden)
        forbidden_hits += row_forbidden_hits
        forbidden_questions += int(row_forbidden_hits > 0)

        distractor_total += len(distractors)
        distractor_hits += len(distractors & rank_set)

    question_count = len(rows)
    return {
        "question_count": question_count,
        "required_support_items": required_total,
        "required_support_recall_at_k": _rate(required_hits, required_total),
        "complete_chain_rate_at_k": _rate(complete, question_count),
        "first_required_support_mrr_at_k": _rate(round(reciprocal_sum, 9), question_count),
        "forbidden_support_items": forbidden_total,
        "forbidden_support_hit_rate_at_k": _rate(forbidden_hits, forbidden_total),
        "questions_with_any_forbidden_hit_rate_at_k": _rate(forbidden_questions, question_count),
        "distractor_items": distractor_total,
        "distractor_retrieval_rate_at_k": _rate(distractor_hits, distractor_total),
    }


def score_dataset(
    questions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k must be an integer in [1, 100]")
    question_by_id: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(questions):
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"question {index}: question_id must be a non-empty string")
        if question_id in question_by_id:
            raise ValueError(f"duplicate question_id in fixtures: {question_id}")
        question_by_id[question_id] = question

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        question_id = prediction.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"prediction {index}: question_id must be a non-empty string")
        if question_id in prediction_by_id:
            raise ValueError(f"duplicate prediction question_id: {question_id}")
        if question_id not in question_by_id:
            raise ValueError(f"unexpected prediction question_id: {question_id}")
        audit = prediction.get("audit")
        if not isinstance(audit, list):
            raise ValueError(f"{question_id}: audit must be an array")
        ranked_keys = []
        seen = set()
        for rank_index, item in enumerate(audit, 1):
            if not isinstance(item, dict):
                raise ValueError(f"{question_id}: audit item {rank_index} must be an object")
            key = evidence_key(item, f"{question_id} audit item {rank_index}")
            if key in seen:
                raise ValueError(f"{question_id}: duplicate ranked evidence pointer {key}")
            seen.add(key)
            ranked_keys.append(key)
        prediction_by_id[question_id] = {"ranked_keys": ranked_keys}

    missing = sorted(set(question_by_id) - set(prediction_by_id))
    if missing:
        raise ValueError(f"missing predictions for question IDs: {missing}")

    prepared: list[dict[str, Any]] = []
    for question_id, question in question_by_id.items():
        prediction = prediction_by_id[question_id]
        def refs(field: str) -> list[tuple[str, str]]:
            raw = question.get(field, [])
            if not isinstance(raw, list):
                raise ValueError(f"{question_id}: {field} must be an array")
            parsed = []
            for position, item in enumerate(raw, 1):
                if not isinstance(item, dict):
                    raise ValueError(f"{question_id}: {field}[{position}] must be an object")
                parsed.append(evidence_key(item, f"{question_id} {field}[{position}]"))
            if len(parsed) != len(set(parsed)):
                raise ValueError(f"{question_id}: duplicate pointer in {field}")
            return parsed

        required = refs("required_support")
        if not required:
            raise ValueError(f"{question_id}: required_support must not be empty")
        prepared.append({
            "question_id": question_id,
            "category": str(question.get("category", "uncategorized")),
            "ranked_keys": prediction["ranked_keys"],
            "required": required,
            "forbidden": refs("forbidden_support"),
            "distractors": refs("distractor_support"),
        })

    overall = _metrics(prepared, k)
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in prepared:
        categories.setdefault(row["category"], []).append(row)
    by_category = {
        category: _metrics(sorted(rows, key=lambda row: row["question_id"]), k)
        for category, rows in sorted(categories.items())
    }

    per_question = []
    for row in prepared:
        top = set(row["ranked_keys"][:k])
        required = set(row["required"])
        forbidden = set(row["forbidden"])
        distractors = set(row["distractors"])
        per_question.append({
            "question_id": row["question_id"],
            "category": row["category"],
            "required_support_count": len(required),
            "required_support_hits_at_k": len(required & top),
            "complete_chain_at_k": required <= top,
            "forbidden_hits_at_k": len(forbidden & top),
            "distractor_hits_at_k": len(distractors & top),
        })

    return {
        "schema": "broca-causal-episode-probe-score-v1",
        "k": k,
        "overall": overall,
        "by_category": by_category,
        "per_question": per_question,
        "interpretation": {
            "primary_metric": "complete_chain_rate_at_k",
            "claim_ceiling": "synthetic evidence retrieval diagnostic only; not answer accuracy, causal truth, task success, or SOTA",
            "compare_only_when": "identical fixtures, questions, context budget, and disclosed method configuration",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=FIXTURE_DIR / "questions.jsonl")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    predictions = load_jsonl(args.predictions)
    report = score_dataset(questions, predictions, args.k)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
