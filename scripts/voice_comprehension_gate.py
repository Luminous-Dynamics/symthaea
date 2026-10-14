#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare and score a blinded spoken-passage comprehension pack.

This is a scoring/data-integrity tool, not a speech recognizer and not a
validated universal release threshold. Audio must be rendered separately with
the exact case text and named by clip_id before a listening session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CASES = Path(__file__).with_name("voice_comprehension_cases.json")
REQUIRED_CATEGORIES = {
    "negation", "quantities", "amounts", "dates", "time", "names",
    "units", "sequence", "conditions", "safety",
}
OPTION_IDS = ("A", "B", "C", "D")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _write_exclusive(path: Path, value: Any, mode: int = 0o600) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def load_case_bank(path: Path = DEFAULT_CASES) -> dict[str, Any]:
    bank = _read_json(path)
    if not isinstance(bank, dict) or bank.get("schema_version") != 1:
        raise ValueError("case bank must be an object with schema_version=1")
    if bank.get("language") != "en" or not isinstance(bank.get("cases"), list):
        raise ValueError("case bank requires language='en' and a cases array")
    if not bank["cases"]:
        raise ValueError("case bank must not be empty")

    seen_cases: set[str] = set()
    seen_categories: set[str] = set()
    total_questions = 0
    for index, case in enumerate(bank["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = case.get("case_id")
        category = case.get("category")
        text = case.get("text")
        questions = case.get("questions")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_cases:
            raise ValueError(f"case {index} has an empty or duplicate case_id")
        seen_cases.add(case_id)
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"case {case_id} requires a category")
        seen_categories.add(category)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"case {case_id} requires non-empty source text")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"case {case_id} requires at least one question")
        seen_questions: set[str] = set()
        for qi, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(f"question {qi} in {case_id} must be an object")
            qid = question.get("question_id")
            prompt = question.get("question")
            options = question.get("options")
            answer = question.get("answer")
            if not isinstance(qid, str) or not qid.strip() or qid in seen_questions:
                raise ValueError(f"case {case_id} has an empty or duplicate question_id")
            seen_questions.add(qid)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{case_id}/{qid} requires a question prompt")
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(f"{case_id}/{qid} must have exactly four options")
            option_ids: list[str] = []
            for option in options:
                if not isinstance(option, dict):
                    raise ValueError(f"{case_id}/{qid} options must be objects")
                oid, label = option.get("id"), option.get("text")
                if not isinstance(oid, str) or not isinstance(label, str) or not label.strip():
                    raise ValueError(f"{case_id}/{qid} contains a malformed option")
                option_ids.append(oid)
            if tuple(option_ids) != OPTION_IDS:
                raise ValueError(f"{case_id}/{qid} options must appear exactly as A, B, C, D")
            if answer not in OPTION_IDS:
                raise ValueError(f"{case_id}/{qid} answer must be A, B, C, or D")
            total_questions += 1
    missing = REQUIRED_CATEGORIES - seen_categories
    if missing:
        raise ValueError(f"case bank missing required categories: {sorted(missing)}")
    if len(seen_cases) != len(bank["cases"]):
        raise ValueError("case IDs must be unique")
    if total_questions < 24:
        raise ValueError("case bank must contain at least 24 questions")
    return bank


def build_documents(bank: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (public questions, private answer key, private render manifest)."""
    public_cases = []
    answer_rows = []
    render_cases = []
    for case in bank["cases"]:
        clip_id = case["case_id"]
        public_questions = []
        question_ids = []
        for question in case["questions"]:
            question_ids.append(question["question_id"])
            public_questions.append({
                "question_id": question["question_id"],
                "question": question["question"],
                "options": [{"id": opt["id"], "text": opt["text"]} for opt in question["options"]],
            })
            answer_rows.append({
                "clip_id": clip_id,
                "question_id": question["question_id"],
                "category": case["category"],
                "answer": question["answer"],
            })
        public_cases.append({"clip_id": clip_id, "questions": public_questions})
        source_text = case["text"]
        render_cases.append({
            "clip_id": clip_id,
            "language": bank["language"],
            "category": case["category"],
            "text": source_text,
            "text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "question_ids": question_ids,
        })
    public_doc = {"format_version": 1, "cases": public_cases}
    answer_key = {"format_version": 1, "answers": answer_rows}
    render_manifest = {"format_version": 1, "cases": render_cases}
    return public_doc, answer_key, render_manifest


def score_responses(
    public_doc: dict[str, Any],
    answer_key: dict[str, Any],
    responses: Any,
) -> dict[str, Any]:
    """Strictly score one response for every published question."""
    if not isinstance(public_doc, dict) or public_doc.get("format_version") != 1:
        raise ValueError("public question pack must be an object with format_version=1")
    if not isinstance(answer_key, dict) or answer_key.get("format_version") != 1:
        raise ValueError("private answer key must be an object with format_version=1")
    if not isinstance(responses, list):
        raise ValueError("responses must be a JSON array")
    public_cases = public_doc.get("cases")
    if not isinstance(public_cases, list) or not public_cases:
        raise ValueError("public question pack requires a non-empty cases array")
    public_questions: dict[tuple[str, str], set[str]] = {}
    seen_clip_ids: set[str] = set()
    for case in public_cases:
        if not isinstance(case, dict):
            raise ValueError("public question pack has malformed case identity")
        clip_id = case.get("clip_id")
        questions = case.get("questions")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("public question pack has malformed case identity")
        if clip_id in seen_clip_ids:
            raise ValueError(f"duplicate public clip ID {clip_id}")
        seen_clip_ids.add(clip_id)
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"public case {clip_id} requires a non-empty question list")
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError(f"public questions for {clip_id} are malformed")
            qid, prompt, options = question.get("question_id"), question.get("question"), question.get("options")
            if not isinstance(qid, str) or not qid.strip() or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"public question for {clip_id} lacks a valid ID/prompt")
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(f"{clip_id}/{qid} must expose exactly four options A-D")
            option_ids = []
            for option in options:
                if not isinstance(option, dict):
                    raise ValueError(f"{clip_id}/{qid} contains a malformed option")
                option_id, label = option.get("id"), option.get("text")
                if not isinstance(option_id, str) or not isinstance(label, str) or not label.strip():
                    raise ValueError(f"{clip_id}/{qid} contains a malformed option")
                option_ids.append(option_id)
            if tuple(option_ids) != OPTION_IDS:
                raise ValueError(f"{clip_id}/{qid} must expose options A-D exactly")
            key = (clip_id, qid)
            if key in public_questions:
                raise ValueError(f"duplicate public question {clip_id}/{qid}")
            public_questions[key] = set(option_ids)

    key_rows = answer_key.get("answers")
    if not isinstance(key_rows, list) or not key_rows:
        raise ValueError("private answer key requires a non-empty answers array")
    expected: dict[tuple[str, str], tuple[str, str]] = {}
    for row in key_rows:
        if not isinstance(row, dict):
            raise ValueError("private answer-key row must be an object")
        clip_id, question_id = row.get("clip_id"), row.get("question_id")
        answer, category = row.get("answer"), row.get("category")
        if not isinstance(clip_id, str) or not clip_id.strip() or not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("answer-key rows require non-empty string clip/question IDs")
        key = (clip_id, question_id)
        if key not in public_questions:
            raise ValueError(f"answer key references unpublished question {key}")
        if not isinstance(answer, str) or answer not in public_questions[key] or not isinstance(category, str) or not category:
            raise ValueError(f"answer key has invalid answer/category for {key}")
        if key in expected:
            raise ValueError(f"duplicate answer-key question {key}")
        expected[key] = (answer, category)
    if set(expected) != set(public_questions):
        raise ValueError("public question coverage and private answer key do not match")

    received: dict[tuple[str, str], str] = {}
    for index, row in enumerate(responses):
        if not isinstance(row, dict) or set(row) != {"clip_id", "question_id", "answer"}:
            raise ValueError(f"response row {index} must contain only clip_id, question_id, answer")
        clip_id, question_id, answer = row["clip_id"], row["question_id"], row["answer"]
        if (
            not isinstance(clip_id, str) or not clip_id.strip()
            or not isinstance(question_id, str) or not question_id.strip()
        ):
            raise ValueError(f"response row {index} requires non-empty string clip/question IDs")
        key = (clip_id, question_id)
        if key not in expected:
            raise ValueError(f"response row {index} references unknown question {key}")
        if key in received:
            raise ValueError(f"duplicate response for {key}")
        if not isinstance(answer, str) or answer not in public_questions[key]:
            raise ValueError(f"response for {key} must select A, B, C, or D")
        received[key] = answer
    missing = set(expected) - set(received)
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"missing {len(missing)} response(s), including {sample}")

    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    case_stats: dict[str, dict[str, Any]] = {}
    for (clip_id, qid), (answer, category) in expected.items():
        correct = received[(clip_id, qid)] == answer
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(correct)
        case = case_stats.setdefault(clip_id, {"correct": 0, "total": 0})
        case["total"] += 1
        case["correct"] += int(correct)

    total = len(expected)
    correct_total = sum(stats["correct"] for stats in category_stats.values())
    categories = {
        category: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"],
        }
        for category, stats in sorted(category_stats.items())
    }
    cases = {
        clip_id: {
            "correct": stats["correct"],
            "total": stats["total"],
            "all_correct": stats["correct"] == stats["total"],
        }
        for clip_id, stats in sorted(case_stats.items())
    }
    all_correct_cases = sum(int(row["all_correct"]) for row in cases.values())
    return {
        "format_version": 1,
        "measure": "spoken_passage_question_accuracy",
        "responses_complete": True,
        "question_accuracy": correct_total / total if total else 0.0,
        "correct_questions": correct_total,
        "total_questions": total,
        "case_all_questions_correct_rate": all_correct_cases / len(cases) if cases else 0.0,
        "all_correct_cases": all_correct_cases,
        "total_cases": len(cases),
        "category_scores": categories,
        "case_scores": cases,
        "release_threshold": None,
        "release_decision": "not_set_requires_audio-linked_baseline_and_review",
        "limitations": [
            "The scorer only checks answers after listening; it does not render or inspect audio.",
            "Synthetic fixture accuracy is not a population estimate or validated release threshold.",
            "A complete response pack is required; missing answers are not counted as correct or dropped.",
        ],
    }


def prepare(output: Path, cases_path: Path) -> None:
    bank = load_case_bank(cases_path)
    public_doc, answer_key, render_manifest = build_documents(bank)
    output = output.expanduser()
    private = Path(str(output) + ".private")
    if output.is_symlink() or private.is_symlink():
        raise ValueError("output and private output must not be symlinks")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"public output must be a new or empty directory: {output}")
    if private.exists() and (not private.is_dir() or any(private.iterdir())):
        raise ValueError(f"private output must be a new or empty directory: {private}")
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(private, 0o700)
    _write_exclusive(output / "public_questions.json", public_doc, 0o644)
    _write_exclusive(private / "answer_key.json", answer_key, 0o600)
    _write_exclusive(private / "render_manifest.json", render_manifest, 0o600)
    print(json.dumps({
        "public_questions": str(output / "public_questions.json"),
        "private_answer_key": str(private / "answer_key.json"),
        "private_render_manifest": str(private / "render_manifest.json"),
        "cases": len(bank["cases"]),
        "questions": len(answer_key["answers"]),
        "note": "No audio was rendered. Render each private manifest text to a WAV named by clip_id before collecting listening responses.",
    }, indent=2))


def score(output: Path, responses_path: Path) -> None:
    output = output.expanduser()
    private = Path(str(output) + ".private")
    if output.is_symlink() or private.is_symlink():
        raise ValueError("public and private output paths must not be symlinks")
    if not output.is_dir() or not private.is_dir():
        raise ValueError("prepared public and sibling private directories are required")
    public_path = output / "public_questions.json"
    key_path = private / "answer_key.json"
    manifest_path = private / "render_manifest.json"
    if responses_path.is_symlink():
        raise ValueError("response file must not be a symlink")
    response_abs = responses_path.expanduser().resolve()
    if response_abs.is_relative_to(output.resolve()) or response_abs.is_relative_to(private.resolve()):
        raise ValueError("responses must be stored outside the public and private artifact directories")
    public_doc = _read_json(public_path)
    answer_key = _read_json(key_path)
    render_manifest = _read_json(manifest_path)
    responses = _read_json(responses_path)
    # Bind the render manifest to the answer key's clip inventory and text digests.
    render_cases = render_manifest.get("cases")
    if not isinstance(render_cases, list):
        raise ValueError("private render manifest is malformed")
    render_ids = [case.get("clip_id") for case in render_cases if isinstance(case, dict)]
    if len(render_ids) != len(render_cases):
        raise ValueError("private render manifest has malformed case records")
    if not all(isinstance(clip_id, str) and clip_id.strip() for clip_id in render_ids):
        raise ValueError("private render manifest contains invalid clip IDs")
    if len(set(render_ids)) != len(render_ids):
        raise ValueError("private render manifest has duplicate clip IDs")
    key_document = _read_json(key_path)
    key_rows = key_document.get("answers") if isinstance(key_document, dict) else None
    if not isinstance(key_rows, list) or not key_rows:
        raise ValueError("private answer key requires a non-empty answers array")
    answer_ids = set()
    for row in key_rows:
        if not isinstance(row, dict) or not isinstance(row.get("clip_id"), str):
            raise ValueError("private answer key contains malformed clip identity")
        answer_ids.add(row["clip_id"])
    if set(render_ids) != answer_ids:
        raise ValueError("render manifest and answer-key clip inventory differ")
    public_doc = _read_json(public_path)
    public_case_map = {}
    for case in public_doc.get("cases", []) if isinstance(public_doc, dict) else []:
        if not isinstance(case, dict) or not isinstance(case.get("clip_id"), str):
            raise ValueError("public question pack has malformed case identity")
        questions = case.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"public questions for {case['clip_id']} are malformed")
        qids = [question.get("question_id") for question in questions if isinstance(question, dict)]
        if len(qids) != len(questions) or not all(isinstance(qid, str) for qid in qids):
            raise ValueError(f"public question IDs are malformed for {case['clip_id']}")
        if case["clip_id"] in public_case_map:
            raise ValueError(f"duplicate public clip ID {case['clip_id']}")
        public_case_map[case["clip_id"]] = set(qids)
    for case in render_cases:
        if not isinstance(case, dict):
            raise ValueError("private render manifest contains a malformed case")
        text = case.get("text")
        if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != case.get("text_sha256"):
            raise ValueError(f"source text digest mismatch for {case.get('clip_id')}")
        expected_question_ids = case.get("question_ids")
        if not isinstance(expected_question_ids, list) or not all(isinstance(qid, str) for qid in expected_question_ids):
            raise ValueError(f"private question IDs are malformed for {case.get('clip_id')}")
        if public_case_map.get(case.get("clip_id")) != set(expected_question_ids):
            raise ValueError(f"render manifest/public question IDs differ for {case.get('clip_id')}")
    report = score_responses(public_doc, answer_key, responses)
    report["render_text_digests_verified"] = True
    report_path = private / "comprehension_report.json"
    _write_exclusive(report_path, report, 0o600)
    print(json.dumps({"private_report": str(report_path), **{
        key: report[key] for key in (
            "question_accuracy", "correct_questions", "total_questions",
            "case_all_questions_correct_rate", "all_correct_cases", "total_cases",
            "release_decision",
        )
    }}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="create a blinded question pack")
    prepare_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    prepare_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score", help="score complete listener responses")
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.output, args.cases)
        else:
            score(args.output, args.responses)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
