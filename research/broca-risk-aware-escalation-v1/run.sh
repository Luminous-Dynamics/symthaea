#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
out=${1:-"$root/run-$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $out" >&2; exit 2
fi
python3 "$root/benchmark.py" --out "$out"
python3 "$root/verify_run.py" "$out"
(cd "$root" && python3 -m unittest -v test_benchmark test_verify_run)
echo "PASS: risk-aware escalation run validated at $out"
