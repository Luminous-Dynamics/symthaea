#!/usr/bin/env bash
set -euo pipefail
MUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m py_compile "$MUSE_DIR/scripts/verify_cognition_study_v10.py"
python3 "$MUSE_DIR/scripts/verify_cognition_study_v10.py" self-test >/dev/null
git -C "$MUSE_DIR" diff --check
