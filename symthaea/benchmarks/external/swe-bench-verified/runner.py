#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""SWE-bench Verified (Mini) runner wrapper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_tasks(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench data not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "tasks", "instances", "examples"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unrecognized SWE-bench data format")


def build_result(benchmark_id: str, benchmark_name: str, metrics: dict, notes: str) -> dict:
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
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified (Mini)")
    parser.add_argument("--input", default="data/mini.json", help="Path to SWE-bench mini JSON")
    parser.add_argument("--output", default="../../results/swe-bench-verified-mini.json", help="Output JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    tasks = load_tasks(input_path)

    runner_cmd = os.environ.get("SYMTHAEA_SWEBENCH_RUNNER")
    harness_cmd = os.environ.get("SYMTHAEA_SWEBENCH_HARNESS_CMD")
    harness_args = os.environ.get("SYMTHAEA_SWEBENCH_HARNESS_ARGS", "")
    external_results = os.environ.get("SYMTHAEA_SWEBENCH_RESULT_JSON")
    harness_results = os.environ.get("SYMTHAEA_SWEBENCH_HARNESS_RESULT_JSON")

    resolved = 0
    attempted = 0
    notes = "SWE-bench mini baseline"

    if runner_cmd:
        # Invoke external harness and expect it to write a result JSON to stdout.
        proc = subprocess.run([runner_cmd, str(input_path)], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                payload = json.loads(proc.stdout)
                resolved = int(payload.get("resolved", 0))
                attempted = int(payload.get("attempted", 0))
            except json.JSONDecodeError:
                notes = "runner output was not JSON; no score computed"
        else:
            notes = f"runner failed (code {proc.returncode})"
    elif harness_cmd:
        cmd = [harness_cmd] + shlex.split(harness_args) + [str(input_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                payload = json.loads(proc.stdout)
                resolved = int(payload.get("resolved", 0))
                attempted = int(payload.get("attempted", 0))
            except json.JSONDecodeError:
                notes = "harness output was not JSON; no score computed"
        elif harness_results and Path(harness_results).exists():
            try:
                payload = json.loads(Path(harness_results).read_text(encoding="utf-8"))
                resolved = int(payload.get("resolved", 0))
                attempted = int(payload.get("attempted", 0))
                notes = f"wrapped harness results: {harness_results}"
            except Exception:
                notes = f"failed to parse harness results: {harness_results}"
        else:
            notes = f"harness failed (code {proc.returncode})"
    elif external_results and Path(external_results).exists():
        # Wrap externally generated results if available.
        notes = f"wrapped external results: {external_results}"
        try:
            payload = json.loads(Path(external_results).read_text(encoding="utf-8"))
            resolved = int(payload.get("resolved", 0))
            attempted = int(payload.get("attempted", 0))
        except Exception:
            notes = f"failed to parse external results: {external_results}"
    else:
        notes = "no runner configured (set SYMTHAEA_SWEBENCH_RUNNER)"

    primary = None
    if attempted > 0:
        primary = resolved / attempted

    metrics = {
        "primary": primary,
        "secondary": {
            "total": len(tasks),
            "attempted": attempted,
            "resolved": resolved,
            "runner": runner_cmd or "none",
            "external_results": external_results or "none",
        },
    }

    output = build_result("swe-bench-verified-mini", "SWE-bench Verified (Mini)", metrics, notes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
