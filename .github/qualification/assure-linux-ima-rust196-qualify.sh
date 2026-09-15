#!/usr/bin/env bash
set -euo pipefail

package="symthaea-linux-ima-replay"
crate="crates/domains/symthaea-linux-ima-replay"
work="${RUNNER_TEMP:-/tmp}/assure-linux-ima-rust196"
capsule="$work/capsule"
source_census="$work/source-census.tsv"
source_lock="$work/Cargo.lock.source"
capsule_lock="$work/Cargo.lock.capsule"
lock_audit="$work/lock-audit.json"

rm -rf "$work"
mkdir -p "$capsule/$crate"
cp Cargo.lock "$source_lock"

# Copy only exact Git-tracked bytes for the one repaired crate and bind every byte
# to the frozen product commit. The source manifest is kept byte-exact.
printf 'path\tgit_blob_sha1\tsha256\tbytes\n' > "$source_census"
while IFS= read -r -d '' path; do
  mkdir -p "$capsule/$(dirname "$path")"
  cp "$path" "$capsule/$path"
  printf '%s\t%s\t%s\t%s\n' \
    "$path" \
    "$(git rev-parse "$PRODUCT_HEAD:$path")" \
    "$(sha256sum "$path" | awk '{print $1}')" \
    "$(wc -c < "$path" | tr -d ' ')" \
    >> "$source_census"
done < <(git ls-files -z -- "$crate/")

python3 - "$source_census" "$capsule" <<'PY'
import hashlib
import os
import pathlib
import subprocess
import sys

census = pathlib.Path(sys.argv[1])
capsule = pathlib.Path(sys.argv[2])
product = os.environ["PRODUCT_HEAD"]
rows = census.read_text().splitlines()[1:]
if not rows:
    raise SystemExit("empty IMA source census")
for row in rows:
    path, blob, digest, size = row.split("\t")
    src = pathlib.Path(path).read_bytes()
    copied = (capsule / path).read_bytes()
    if src != copied:
        raise SystemExit(f"capsule copy mismatch: {path}")
    if hashlib.sha256(src).hexdigest() != digest or len(src) != int(size):
        raise SystemExit(f"source census mismatch: {path}")
    actual_blob = subprocess.check_output(
        ["git", "rev-parse", f"{product}:{path}"], text=True
    ).strip()
    if actual_blob != blob:
        raise SystemExit(f"product blob mismatch: {path}")
print(f"ima_source_census=PASS files={len(rows)}")
PY

# Recreate only the workspace inheritance actually used by the exact crate manifest.
cat > "$capsule/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["crates/domains/symthaea-linux-ima-replay"]
default-members = ["crates/domains/symthaea-linux-ima-replay"]

[workspace.dependencies]
blake3 = "1.5"
serde = { version = "1.0", features = ["derive"] }
TOML

cmp "$crate/Cargo.toml" "$capsule/$crate/Cargo.toml"
cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package" -- --check

# Reconcile only inside the temporary capsule, then prove every external package entry
# is byte-semantically present in the frozen source lock. The repository lock is immutable.
cp "$source_lock" "$capsule/Cargo.lock"
cargo check --manifest-path "$capsule/Cargo.toml" -p "$package"
cp "$capsule/Cargo.lock" "$capsule_lock"

python3 - "$source_lock" "$capsule_lock" "$lock_audit" <<'PY'
import json
import pathlib
import sys
import tomllib

source_path, capsule_path, audit_path = map(pathlib.Path, sys.argv[1:])
source = tomllib.loads(source_path.read_text())
capsule = tomllib.loads(capsule_path.read_text())

def key(pkg):
    return (pkg["name"], pkg["version"], pkg.get("source", ""))

source_packages = {key(pkg): pkg for pkg in source.get("package", [])}
local = [pkg for pkg in capsule.get("package", []) if "source" not in pkg]
if [(pkg["name"], pkg["version"]) for pkg in local] != [("symthaea-linux-ima-replay", "0.1.0")]:
    raise SystemExit(f"unexpected capsule local packages: {local!r}")

external = [pkg for pkg in capsule.get("package", []) if "source" in pkg]
missing = []
changed = []
for pkg in external:
    frozen = source_packages.get(key(pkg))
    if frozen is None:
        missing.append(key(pkg))
    elif frozen != pkg:
        changed.append(key(pkg))
if missing:
    raise SystemExit(f"capsule introduced external packages absent from source lock: {missing!r}")
if changed:
    raise SystemExit(f"capsule external package entries drifted from source lock: {changed!r}")

audit = {
    "capsule_local_packages": [pkg["name"] for pkg in local],
    "external_packages_bound_to_source_lock": len(external),
    "source_package_count": len(source_packages),
    "capsule_package_count": len(capsule.get("package", [])),
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print("ima_lock_audit=PASS")
PY

cargo check --locked --manifest-path "$capsule/Cargo.toml" -p "$package"
cargo test --locked --manifest-path "$capsule/Cargo.toml" -p "$package"
cargo clippy --locked --manifest-path "$capsule/Cargo.toml" -p "$package" -- -D warnings

printf 'qualification_contract_revision=%s\n' 'assure-linux-ima-rust196-v1'
printf 'product_head=%s\n' "$PRODUCT_HEAD"
printf 'source_census_sha256=%s\n' "$(sha256sum "$source_census" | awk '{print $1}')"
printf 'source_lock_sha256=%s\n' "$(sha256sum "$source_lock" | awk '{print $1}')"
printf 'capsule_lock_sha256=%s\n' "$(sha256sum "$capsule_lock" | awk '{print $1}')"
printf 'lock_audit_sha256=%s\n' "$(sha256sum "$lock_audit" | awk '{print $1}')"
printf 'qualification_result=PASS\n'
