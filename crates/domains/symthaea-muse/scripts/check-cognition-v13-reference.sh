#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PYCOMPILE' "$ROOT/scripts/verify_cognition_study_v13.py"
from pathlib import Path
import sys
path = Path(sys.argv[1])
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PYCOMPILE
python3 "$ROOT/scripts/verify_cognition_study_v13.py" --self-test >/dev/null
for file in \
  src/replication_protocol.rs \
  src/replication_site_registry.rs \
  src/replication_package.rs \
  src/replication_execution.rs \
  src/replication_synthesis.rs \
  src/replication_orchestration.rs \
  src/research_revision_governance.rs \
  src/research_archive.rs \
  src/stewardship_governance.rs \
  src/research_release_promotion.rs \
  src/stewardship_release.rs; do
  test -s "$ROOT/$file"
done
python3 - <<'PY' "$ROOT/data/cognition-study/templates/v13"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in root.glob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
PY
git -C "$ROOT" diff --check
