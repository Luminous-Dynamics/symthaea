#!/usr/bin/env python3
"""Run HELM Capabilities + Safety suites and emit a normalized result JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def count_files(path: Path, max_depth: int = 3) -> int:
    if not path.exists():
        return 0
    root = path.resolve()
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
        count += len(filenames)
    return count


def run_with_heartbeat(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    label: str,
    heartbeat_seconds: int,
    heartbeat_paths: Optional[Dict[str, Path]] = None,
) -> None:
    process = subprocess.Popen(cmd, env=env)
    start = time.time()
    next_beat = start + max(heartbeat_seconds, 1)
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise RuntimeError(f"Command failed ({return_code}): {' '.join(cmd)}")
            return
        now = time.time()
        if heartbeat_seconds > 0 and now >= next_beat:
            elapsed = int(now - start)
            parts = [f"[HELM] {label} still running ({elapsed}s elapsed)"]
            if heartbeat_paths:
                for name, path in heartbeat_paths.items():
                    try:
                        count = count_files(path, max_depth=3)
                    except OSError:
                        count = 0
                    parts.append(f"{name}_files={count}")
            print(" - ".join(parts), file=sys.stderr, flush=True)
            next_beat = now + heartbeat_seconds
        time.sleep(1)


def ensure_helm_repo(default_path: Path) -> Path:
    repo_path = os.environ.get("SYMTHAEA_HELM_REPO")
    if repo_path:
        return Path(repo_path).resolve()
    if default_path.exists():
        return default_path
    repo_url = os.environ.get("SYMTHAEA_HELM_REPO_URL", "https://github.com/stanford-crfm/helm.git")
    run(["git", "clone", "--depth", "1", repo_url, str(default_path)])
    return default_path


def ensure_venv(venv_path: Path, helm_repo: Path) -> Tuple[Path, Path]:
    helm_run = venv_path / "bin" / "helm-run"
    helm_summarize = venv_path / "bin" / "helm-summarize"
    if helm_run.exists() and helm_summarize.exists():
        return helm_run, helm_summarize

    python = os.environ.get("SYMTHAEA_HELM_PYTHON", sys.executable)
    run([python, "-m", "venv", str(venv_path)])
    pip = venv_path / "bin" / "pip"

    pip_args = [str(pip), "install", "-e", str(helm_repo)]
    if os.environ.get("SYMTHAEA_HELM_PIP_NO_DEPS") == "1":
        pip_args.append("--no-deps")
    run(pip_args)

    if not helm_run.exists() or not helm_summarize.exists():
        raise RuntimeError("helm-run not installed in virtualenv")
    return helm_run, helm_summarize


def parse_stat_name(stat: dict) -> Optional[str]:
    name = stat.get("name")
    if isinstance(name, dict):
        return name.get("name")
    if isinstance(name, str):
        return name
    return None


def aggregate_metrics(runs_path: Path) -> Dict[str, float]:
    if not runs_path.exists():
        return {}
    try:
        payload = json.loads(runs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    runs = payload.get("runs") if isinstance(payload, dict) else payload
    if not isinstance(runs, list):
        return {}

    metric_values: Dict[str, List[float]] = {}
    for run_item in runs:
        stats = run_item.get("stats") if isinstance(run_item, dict) else None
        if not isinstance(stats, list):
            continue
        for stat in stats:
            if not isinstance(stat, dict):
                continue
            name = parse_stat_name(stat)
            if not name:
                continue
            mean = stat.get("mean")
            if mean is None:
                continue
            if not isinstance(mean, (int, float)):
                continue
            metric_values.setdefault(name, []).append(float(mean))

    aggregated: Dict[str, float] = {}
    for name, values in metric_values.items():
        if values:
            aggregated[name] = sum(values) / len(values)
    return aggregated


def choose_primary(metrics: Dict[str, float]) -> Optional[float]:
    for key in (
        "quasi_exact_match",
        "exact_match",
        "accuracy",
        "pass@1",
        "success",
        "successful",
    ):
        if key in metrics:
            return metrics[key]
    return None


def run_suite(
    helm_run: Path,
    helm_summarize: Path,
    run_entries: Path,
    schema_path: Path,
    suite: str,
    output_path: Path,
    local_path: Path,
    models_to_run: List[str],
    max_eval_instances: int,
    num_train_trials: int,
    priority: int,
    disable_cache: bool,
    heartbeat_seconds: int,
) -> Dict[str, str]:
    cmd = [
        str(helm_run),
        "--conf-paths",
        str(run_entries),
        "--suite",
        suite,
        "--output-path",
        str(output_path),
        "--local-path",
        str(local_path),
        "--num-train-trials",
        str(num_train_trials),
        "--max-eval-instances",
        str(max_eval_instances),
        "--priority",
        str(priority),
        "--models-to-run",
    ] + models_to_run

    if disable_cache:
        cmd.append("--disable-cache")

    heartbeat_paths = {
        "runs": output_path / "runs" / suite,
        "cache": local_path / "cache",
    }
    run_with_heartbeat(
        cmd,
        label=f"{suite}",
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_paths=heartbeat_paths,
    )

    run(
        [
            str(helm_summarize),
            "--suite",
            suite,
            "--output-path",
            str(output_path),
            "--schema-path",
            str(schema_path),
        ]
    )

    return {
        "suite": suite,
        "output_path": str(output_path),
    }


def build_result(metrics: dict, notes: str) -> dict:
    return {
        "benchmark_id": "helm-capabilities-safety",
        "benchmark_name": "HELM Capabilities + HELM Safety",
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": {
            "backend": "helm",
            "model_id": os.environ.get("SYMTHAEA_HELM_MODELS_TO_RUN", "simple/model1"),
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
    parser = argparse.ArgumentParser(description="Run HELM Capabilities + Safety")
    parser.add_argument("--output", default="../../results/helm.json", help="Output JSON")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    if not (data_dir / "READY").exists():
        print("HELM data missing. Run fetch.sh first.", file=sys.stderr)
        return 3

    run_entries_cap = data_dir / "run_entries_capabilities_reasoning_v2.conf"
    run_entries_safety = data_dir / "run_entries_safety.conf"
    schema_cap = data_dir / "schema_capabilities.yaml"
    schema_safety = data_dir / "schema_safety.yaml"

    for path in (run_entries_cap, run_entries_safety, schema_cap, schema_safety):
        if not path.exists():
            print(f"Missing HELM file: {path}", file=sys.stderr)
            return 3

    output_path = Path(os.environ.get("SYMTHAEA_HELM_OUTPUT_PATH", script_dir / "output"))
    local_path = Path(os.environ.get("SYMTHAEA_HELM_LOCAL_PATH", script_dir / "local"))
    output_path.mkdir(parents=True, exist_ok=True)
    local_path.mkdir(parents=True, exist_ok=True)

    helm_repo = ensure_helm_repo(script_dir / "helm-repo")
    venv_path = script_dir / ".venv"
    helm_run, helm_summarize = ensure_venv(venv_path, helm_repo)

    models_to_run = os.environ.get("SYMTHAEA_HELM_MODELS_TO_RUN", "simple/model1").split()
    max_eval_instances = int(os.environ.get("SYMTHAEA_HELM_MAX_EVAL_INSTANCES", "1000"))
    num_train_trials = int(os.environ.get("SYMTHAEA_HELM_NUM_TRAIN_TRIALS", "1"))
    priority = int(os.environ.get("SYMTHAEA_HELM_PRIORITY", "2"))
    disable_cache = os.environ.get("SYMTHAEA_HELM_DISABLE_CACHE", "1") == "1"
    heartbeat_seconds = int(os.environ.get("SYMTHAEA_HELM_HEARTBEAT_SECS", "60"))
    suite_prefix = os.environ.get("SYMTHAEA_HELM_SUITE_PREFIX", "symthaea")

    cap_suite = f"{suite_prefix}-helm-capabilities"
    safety_suite = f"{suite_prefix}-helm-safety"

    run_suite(
        helm_run,
        helm_summarize,
        run_entries_cap,
        schema_cap,
        cap_suite,
        output_path,
        local_path,
        models_to_run,
        max_eval_instances,
        num_train_trials,
        priority,
        disable_cache,
        heartbeat_seconds,
    )

    run_suite(
        helm_run,
        helm_summarize,
        run_entries_safety,
        schema_safety,
        safety_suite,
        output_path,
        local_path,
        models_to_run,
        max_eval_instances,
        num_train_trials,
        priority,
        disable_cache,
        heartbeat_seconds,
    )

    cap_runs = output_path / "runs" / cap_suite / "runs.json"
    safety_runs = output_path / "runs" / safety_suite / "runs.json"
    cap_metrics = aggregate_metrics(cap_runs)
    safety_metrics = aggregate_metrics(safety_runs)

    cap_primary = choose_primary(cap_metrics)
    safety_primary = choose_primary(safety_metrics)

    primaries = [v for v in (cap_primary, safety_primary) if v is not None]
    overall_primary = sum(primaries) / len(primaries) if primaries else None

    metrics = {
        "primary": overall_primary,
        "secondary": {
            "capabilities_primary": cap_primary,
            "safety_primary": safety_primary,
            "capabilities_metrics": cap_metrics,
            "safety_metrics": safety_metrics,
            "capabilities_suite": cap_suite,
            "safety_suite": safety_suite,
            "output_path": str(output_path),
        },
    }

    notes = (
        f"helm_repo={helm_repo}; models_to_run={','.join(models_to_run)}; "
        f"max_eval_instances={max_eval_instances}; num_train_trials={num_train_trials}; "
        f"priority={priority}; disable_cache={disable_cache}"
    )

    output = build_result(metrics, notes)
    output_path_json = Path(args.output)
    if not output_path_json.is_absolute():
        output_path_json = (script_dir / output_path_json).resolve()
    output_path_json.parent.mkdir(parents=True, exist_ok=True)
    output_path_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
