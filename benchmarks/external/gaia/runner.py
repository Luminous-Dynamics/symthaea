#!/usr/bin/env python3
"""GAIA dev runner with optional agent command."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_items(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"GAIA data not found: {path}")
    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "tasks", "examples", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unrecognized GAIA data format")


def normalize_files(files: Any, base_dir: Path) -> List[Any]:
    if not isinstance(files, list):
        return []
    normalized: List[Any] = []
    for entry in files:
        if isinstance(entry, str):
            candidate = (base_dir / entry).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                    normalized.append({"path": entry, "content": content})
                except OSError:
                    normalized.append(entry)
            else:
                normalized.append(entry)
        elif isinstance(entry, dict):
            path_val = entry.get("path")
            if path_val and "content" not in entry:
                candidate = (base_dir / str(path_val)).resolve()
                if candidate.exists() and candidate.is_file():
                    try:
                        content = candidate.read_text(encoding="utf-8", errors="replace")
                        entry = dict(entry)
                        entry["content"] = content
                    except OSError:
                        pass
            normalized.append(entry)
        else:
            normalized.append(entry)
    return normalized


def first_text_field(item: dict, fields: List[str]) -> Optional[str]:
    for key in fields:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_task(item: dict) -> str:
    fields = ["question", "instruction", "input", "task", "problem", "query", "prompt"]
    task = first_text_field(item, fields)
    if task:
        return task
    # Fallback: stringify
    return json.dumps(item, ensure_ascii=True)


def extract_answer(item: dict) -> Optional[str]:
    fields = ["answer", "final_answer", "output", "target", "gold"]
    ans = first_text_field(item, fields)
    if ans:
        return ans
    return None


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r\f\v\"'`.,;:!?")
    return text


def try_parse_number(text: str) -> Optional[float]:
    cleaned = text.strip().replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def compare_answers(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if pred_n == gold_n:
        return True
    pred_num = try_parse_number(pred)
    gold_num = try_parse_number(gold)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return False


def call_agent(agent_cmd: Optional[str], payload: dict) -> Tuple[str, dict]:
    if not agent_cmd:
        return "I don't know", {"status": "null_agent"}
    cmd = shlex.split(agent_cmd)
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if not stdout:
        return "", {"status": "agent_no_output", "stderr": stderr}
    try:
        data = json.loads(stdout)
        answer = data.get("answer")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=True)
        meta = {"status": data.get("status", "ok")}
        if "actions" in data:
            meta["actions"] = data["actions"]
        if stderr:
            meta["stderr"] = stderr
        return answer, meta
    except json.JSONDecodeError:
        meta = {"status": "agent_plaintext"}
        if stderr:
            meta["stderr"] = stderr
        return stdout, meta


def build_result_schema(benchmark_id: str, benchmark_name: str, metrics: dict, notes: str) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": {
            "backend": "ollama",
            "model_id": os.environ.get("SYMTHAEA_LLM_MODEL", "unknown"),
            "translation_only": True,
            "quantized": True,
        },
        "policy": None,
        "metrics": metrics,
        "artifacts": {
            "raw_logs": None,
            "run_notes": notes,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GAIA dev benchmark")
    parser.add_argument("--input", default="data/dev.json", help="Path to GAIA dev JSON")
    parser.add_argument("--output", default="../../results/gaia-dev.json", help="Output JSON")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    items = load_items(input_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    agent_cmd = os.environ.get("SYMTHAEA_AGENT_CMD")
    scored = 0
    correct = 0
    results = []

    for idx, item in enumerate(items):
        task_id = item.get("id") or item.get("task_id") or f"gaia-dev-{idx:04d}"
        task_text = extract_task(item)
        gold = extract_answer(item)

        files = normalize_files(item.get("files", []), input_path.parent)
        payload = {
            "task_id": task_id,
            "task": task_text,
            "files": files,
            "metadata": {"source": "gaia", "index": idx},
        }

        answer, meta = call_agent(agent_cmd, payload)
        is_correct = None
        if gold is not None:
            scored += 1
            is_correct = compare_answers(answer, gold)
            if is_correct:
                correct += 1

        results.append(
            {
                "task_id": task_id,
                "answer": answer,
                "expected": gold,
                "correct": is_correct,
                "meta": meta,
            }
        )

    accuracy = None
    if scored > 0:
        accuracy = correct / scored

    metrics = {
        "primary": accuracy,
        "secondary": {
            "total": len(items),
            "scored": scored,
            "correct": correct,
            "agent": agent_cmd or "null_agent",
        },
    }

    notes = "GAIA dev run"
    if not agent_cmd:
        notes = "null_agent baseline (no SYMTHAEA_AGENT_CMD)"

    output = build_result_schema("gaia-dev", "GAIA (Public Dev)", metrics, notes)
    output["artifacts"]["task_results"] = results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
