#!/usr/bin/env python3
"""OSWorld-Verified runner wrapper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Run OSWorld-Verified (NixOS)")
    parser.add_argument("--output", default="../../results/osworld-verified.json", help="Output JSON")
    args = parser.parse_args()

    output_path = Path(args.output)

    runner_cmd = os.environ.get("SYMTHAEA_OSWORLD_RUNNER")
    external_results = os.environ.get("SYMTHAEA_OSWORLD_RESULT_JSON")

    success = None
    notes = "OSWorld baseline"

    if runner_cmd:
        proc = subprocess.run([runner_cmd], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                payload = json.loads(proc.stdout)
                success = payload.get("success_rate")
            except json.JSONDecodeError:
                notes = "runner output was not JSON; no score computed"
        else:
            notes = f"runner failed (code {proc.returncode})"
    elif external_results and Path(external_results).exists():
        notes = f"wrapped external results: {external_results}"
        try:
            payload = json.loads(Path(external_results).read_text(encoding="utf-8"))
            success = payload.get("success_rate")
        except Exception:
            notes = f"failed to parse external results: {external_results}"
    else:
        notes = "no runner configured (set SYMTHAEA_OSWORLD_RUNNER)"

    env_hash = os.environ.get("SYMTHAEA_OSWORLD_ENV_HASH")
    metrics = {
        "primary": success,
        "secondary": {
            "runner": runner_cmd or "none",
            "external_results": external_results or "none",
            "env_hash": env_hash or "unknown",
        },
    }

    output = build_result("osworld-verified", "OSWorld-Verified (NixOS)", metrics, notes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
