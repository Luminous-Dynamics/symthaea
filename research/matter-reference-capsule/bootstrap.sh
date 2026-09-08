#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-symthaea-matter-si-001}"
email="${2:-matter-capsule@localhost.invalid}"
state_root="${SYMTHAEA_MATTER_CAPSULE_STATE:-$here/state/$profile}"
work_root="$state_root/work"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ "${SYMTHAEA_MATTER_CAPSULE_SHELL:-}" == 1 ]] || fail "enter the pinned shell first: nix-shell $here/shell.nix"
[[ -f "$here/uv.lock" ]] || fail "uv.lock is not committed; this environment is scaffold-only and may not execute a reference capsule"
[[ -n "${SYMTHAEA_PYTHON:-}" && -x "$SYMTHAEA_PYTHON" ]] || fail "pinned Python executable is unavailable"
[[ -n "${SYMTHAEA_QE_STORE:-}" && -d "$SYMTHAEA_QE_STORE" ]] || fail "pinned Quantum ESPRESSO store path is unavailable"
command -v uv >/dev/null || fail "uv is unavailable"

for executable in pw.x ph.x q2r.x matdyn.x; do
  path="$(command -v "$executable" || true)"
  [[ -n "$path" ]] || fail "$executable is unavailable in the pinned Nix shell"
  resolved="$(readlink -f "$path")"
  case "$resolved" in
    "$SYMTHAEA_QE_STORE"/*) ;;
    *) fail "$executable does not resolve inside the pinned Quantum ESPRESSO store path" ;;
  esac
done

export UV_PROJECT_ENVIRONMENT="$here/.venv"
uv sync --project "$here" --frozen --no-dev --python "$SYMTHAEA_PYTHON"
# shellcheck disable=SC1091
source "$here/.venv/bin/activate"

python - <<'PY'
import importlib.metadata
expected = {
    "aiida-core": "2.9.2",
    "aiida-quantumespresso": "5.0.0",
}
for distribution, version in expected.items():
    actual = importlib.metadata.version(distribution)
    if actual != version:
        raise SystemExit(f"{distribution} version mismatch: expected {version}, got {actual}")
PY

export AIIDA_PATH="$state_root/aiida"
mkdir -p "$AIIDA_PATH" "$work_root"

# Fail closed if the profile name is already present. Reference capsule v1 uses
# a fresh provenance store rather than mutating/reusing an existing run state.
if verdi profile list 2>/dev/null | grep -Fq "$profile"; then
  fail "AiiDA profile already exists under this isolated AIIDA_PATH: $profile"
fi

verdi presto --profile-name "$profile" --email "$email" --no-broker --non-interactive
verdi -p "$profile" config set caching.default_enabled False

computer="matter-local"
verdi -p "$profile" computer setup \
  --label "$computer" \
  --hostname localhost \
  --transport core.local \
  --scheduler core.direct \
  --work-dir "$work_root" \
  --non-interactive
verdi -p "$profile" computer configure core.local "$computer" --safe-interval 0 --non-interactive
verdi -p "$profile" computer test "$computer"

create_code() {
  local label="$1"
  local plugin="$2"
  local executable="$3"
  local path
  path="$(readlink -f "$(command -v "$executable")")"
  verdi -p "$profile" code create core.code.installed \
    --label "$label" \
    --computer "$computer" \
    --default-calc-job-plugin "$plugin" \
    --filepath-executable "$path" \
    --non-interactive
}

create_code qe-pw quantumespresso.pw pw.x
create_code qe-ph quantumespresso.ph ph.x
create_code qe-q2r quantumespresso.q2r q2r.x
create_code qe-matdyn quantumespresso.matdyn matdyn.x

python - "$here" "$profile" "$state_root" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

here = Path(sys.argv[1])
profile = sys.argv[2]
state_root = Path(sys.argv[3])

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

executables = {}
for name in ("pw.x", "ph.x", "q2r.x", "matdyn.x"):
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"missing executable after bootstrap: {name}")
    resolved = Path(found).resolve()
    executables[name] = {"path": str(resolved), "sha256": digest(resolved)}

manifest = {
    "schema_version": "symthaea.matter.reference-capsule-environment/v1",
    "profile": profile,
    "aiida_path": os.environ["AIIDA_PATH"],
    "state_root": str(state_root),
    "python": {
        "executable": sys.executable,
        "version": sys.version.split()[0],
    },
    "python_distributions": {
        "aiida-core": importlib.metadata.version("aiida-core"),
        "aiida-quantumespresso": importlib.metadata.version("aiida-quantumespresso"),
    },
    "quantum_espresso_store": os.environ["SYMTHAEA_QE_STORE"],
    "executables": executables,
    "caching": "disabled",
    "broker": "none",
    "storage": "SQLite via verdi presto",
    "authority": "ExecutionEnvironmentPreparedOnly",
    "limitations": [
        "environment preparation does not establish that any scientific calculation was executed",
        "AiiDA SQLite/no-broker mode is intended for the local reference capsule, not high-throughput production",
        "pseudopotential identity is deliberately not selected by bootstrap and must be supplied as frozen benchmark evidence",
    ],
}
output = state_root / "environment-manifest.json"
output.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
print(output)
PY

printf '\nPrepared isolated Matter reference-capsule environment.\n'
printf '  profile:    %s\n' "$profile"
printf '  AIIDA_PATH: %s\n' "$AIIDA_PATH"
printf '  state:      %s\n' "$state_root"
printf '\nNo scientific calculation has been executed by this bootstrap.\n'
