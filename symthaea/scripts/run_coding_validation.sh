#!/usr/bin/env bash
# Focused validation lane for Symthaea's code-generation honesty and benchmark path.

set -euo pipefail

cd "$(dirname "$0")/.."

export RUSTC_WRAPPER="${RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

report_path="${SYMTHAEA_CODING_VALIDATION_REPORT:-target/coding-validation/humaneval-smoke.json}"
mkdir -p "$(dirname "$report_path")"

echo "== coding honesty =="
cargo test --test coding_honesty --features school_learning -- --nocapture

echo "== verified generation honesty =="
cargo test --test verified_generation_honesty --features code_generation -- --nocapture

echo "== polyglot executor honesty =="
cargo test --test polyglot_executor_honesty --features code_generation -- --nocapture

echo "== code executor targeted tests =="
cargo test --lib --features code_generation code_executor::tests:: -- --nocapture

echo "== verified HumanEval-style smoke benchmark =="
cargo run --quiet --example benchmark_humaneval_verified --features code_generation -- \
  --input data/benchmarks/humaneval/rust_smoke.jsonl \
  --json > "$report_path"

python3 - "$report_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())

task_count = report.get("task_count", 0)
pass_at_1 = report.get("pass_at_1", 0.0)
compile_rate = report.get("compile_rate", 0.0)

if task_count < 2:
    raise SystemExit(f"expected at least 2 smoke tasks, got {task_count}")
if pass_at_1 < 1.0:
    raise SystemExit(f"expected smoke pass@1 >= 1.0, got {pass_at_1}")
if compile_rate < 1.0:
    raise SystemExit(f"expected smoke compile_rate >= 1.0, got {compile_rate}")

for task in report.get("tasks", []):
    if not task.get("compiled"):
        raise SystemExit(f"{task.get('id')} did not compile: {task.get('first_error')}")
    if not task.get("guaranteed"):
        raise SystemExit(f"{task.get('id')} was not guaranteed: {task.get('first_error')}")

print(
    "coding validation passed: "
    f"tasks={task_count} pass_at_1={pass_at_1:.3f} compile_rate={compile_rate:.3f}"
)
PY

echo "report: $report_path"
