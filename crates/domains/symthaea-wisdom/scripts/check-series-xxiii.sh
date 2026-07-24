#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$root/scripts/validate-rust-lexical.py"
python3 "$root/scripts/validate-durable-uncertainty.py"
python3 "$root/scripts/validate-transition-journals.py"
python3 "$root/scripts/validate-authority-encapsulation.py"
python3 "$root/scripts/validate-state-encapsulation.py"
