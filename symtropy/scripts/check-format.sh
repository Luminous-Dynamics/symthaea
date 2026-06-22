#!/usr/bin/env bash
# Format/check only the packages that are members of the Symtropy workspace.

set -euo pipefail

mode="${1:-check}"
if [[ "$mode" != "check" && "$mode" != "fix" ]]; then
  echo "usage: $0 [check|fix]" >&2
  exit 2
fi

mapfile -t packages < <(
  python3 - <<'PY'
import json
import subprocess

metadata = json.loads(
    subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        text=True,
    )
)
workspace_ids = set(metadata["workspace_members"])
for package in metadata["packages"]:
    if package["id"] in workspace_ids:
        print(package["name"])
PY
)

for package in "${packages[@]}"; do
  if [[ "$mode" == "check" ]]; then
    cargo fmt -p "$package" -- --check
  else
    cargo fmt -p "$package"
  fi
done
