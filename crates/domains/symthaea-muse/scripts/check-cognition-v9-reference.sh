#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m py_compile "$ROOT/scripts/verify_cognition_study_v9.py"
python3 "$ROOT/scripts/verify_cognition_study_v9.py" self-test
