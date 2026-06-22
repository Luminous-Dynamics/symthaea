#!/usr/bin/env bash
# Check that workspace Rust files have SPDX headers matching their package license.

set -euo pipefail

python3 - <<'PY'
import json
import pathlib
import subprocess
import sys
import tomllib

metadata = json.loads(
    subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        text=True,
    )
)

failed = False
root = pathlib.Path(metadata["workspace_root"])

for package in metadata["packages"]:
    manifest_path = pathlib.Path(package["manifest_path"])
    package_root = manifest_path.parent
    manifest = tomllib.loads(manifest_path.read_text())
    license_id = manifest.get("package", {}).get("license")

    if not license_id:
        print(f"No license found in {manifest_path.relative_to(root)}")
        failed = True
        continue

    for source_root in ("src", "tests", "examples", "benches"):
        tree = package_root / source_root
        if not tree.exists():
            continue
        for rust_file in sorted(tree.rglob("*.rs")):
            text = rust_file.read_text(errors="replace")
            spdx = None
            for line in text.splitlines()[:10]:
                marker = "SPDX-License-Identifier:"
                if marker in line:
                    spdx = line.split(marker, 1)[1].strip()
                    break

            rel = rust_file.relative_to(root)
            if spdx is None:
                print(f"Missing SPDX header: {rel}")
                failed = True
            elif spdx != license_id:
                print(f"License mismatch: {rel} (Cargo: {license_id}, Header: {spdx})")
                failed = True

if failed:
    print("License check failed.")
    sys.exit(1)

print("License check passed.")
PY
