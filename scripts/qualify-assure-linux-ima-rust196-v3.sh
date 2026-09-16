#!/usr/bin/env bash
set -euo pipefail

: "${PRODUCT_HEAD:?PRODUCT_HEAD required}"
: "${PRODUCT_PARENT:?PRODUCT_PARENT required}"
: "${EXPECTED_PRODUCT_BLOB:?EXPECTED_PRODUCT_BLOB required}"
: "${EXPECTED_PRODUCT_SHA256:?EXPECTED_PRODUCT_SHA256 required}"
: "${DERIVATION_ID:?DERIVATION_ID required}"
: "${ARCHIVE_ID:?ARCHIVE_ID required}"

package="symthaea-linux-ima-replay"
crate="crates/domains/symthaea-linux-ima-replay"
source_path="$crate/src/lib.rs"
work="${RUNNER_TEMP:-/tmp}/assure-linux-ima-rust196-v3"
capsule="$work/capsule"
source_census="$work/source-census.tsv"
source_lock="$work/Cargo.lock.source"
capsule_lock="$work/Cargo.lock.capsule"
lock_audit="$work/lock-audit.json"
metadata_json="$work/cargo-metadata.offline.json"
test_list="$work/test-list.txt"
golden_log="$work/golden-replay-test.log"
full_test_log="$work/full-test.log"
clippy_log="$work/clippy-all-targets.log"
receipt="$work/qualification-receipt.txt"

rm -rf "$work"
mkdir -p "$capsule/$crate"

stage() {
  printf 'qualification_gate=%s\n' "$1"
}

stage product_identity

git cat-file -e "$PRODUCT_HEAD^{commit}"
test "$(git rev-parse "$PRODUCT_HEAD^")" = "$PRODUCT_PARENT"
test "$(git rev-list --count "$PRODUCT_PARENT".."$PRODUCT_HEAD")" = "1"
test "$(git rev-list --parents -n 1 "$PRODUCT_HEAD" | awk '{print NF}')" = "2"
test "$(git diff --name-status "$PRODUCT_PARENT" "$PRODUCT_HEAD")" = $'M\tcrates/domains/symthaea-linux-ima-replay/src/lib.rs'
test "$(git rev-parse "$PRODUCT_HEAD:$source_path")" = "$EXPECTED_PRODUCT_BLOB"
test "$(git show "$PRODUCT_HEAD:$source_path" | sha256sum | awk '{print $1}')" = "$EXPECTED_PRODUCT_SHA256"

stage git_object_capsule

printf 'path\tgit_blob_sha1\tsha256\tbytes\n' > "$source_census"
while IFS= read -r -d '' path; do
  mkdir -p "$capsule/$(dirname "$path")"
  git show "$PRODUCT_HEAD:$path" > "$capsule/$path"
  cmp "$path" "$capsule/$path"
  printf '%s\t%s\t%s\t%s\n' \
    "$path" \
    "$(git rev-parse "$PRODUCT_HEAD:$path")" \
    "$(sha256sum "$capsule/$path" | awk '{print $1}')" \
    "$(wc -c < "$capsule/$path" | tr -d ' ')" \
    >> "$source_census"
done < <(git ls-tree -r -z --name-only "$PRODUCT_HEAD" -- "$crate/")

python3 - "$source_census" "$capsule" "$PRODUCT_HEAD" <<'PY'
import hashlib
import pathlib
import subprocess
import sys

census = pathlib.Path(sys.argv[1])
capsule = pathlib.Path(sys.argv[2])
product = sys.argv[3]
rows = census.read_text().splitlines()[1:]
if not rows:
    raise SystemExit("empty IMA source census")
paths = []
for row in rows:
    path, blob, digest, size = row.split("\t")
    if path in paths:
        raise SystemExit(f"duplicate source census path: {path}")
    paths.append(path)
    data = (capsule / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest or len(data) != int(size):
        raise SystemExit(f"capsule census mismatch: {path}")
    actual_blob = subprocess.check_output(
        ["git", "rev-parse", f"{product}:{path}"], text=True
    ).strip()
    if actual_blob != blob:
        raise SystemExit(f"product blob mismatch: {path}")
print(f"ima_source_census=PASS files={len(rows)} unique_paths={len(paths)}")
PY

git show "$PRODUCT_HEAD:Cargo.lock" > "$source_lock"

cat > "$capsule/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["crates/domains/symthaea-linux-ima-replay"]
default-members = ["crates/domains/symthaea-linux-ima-replay"]

[workspace.dependencies]
blake3 = "1.5"
serde = { version = "1.0", features = ["derive"] }
TOML

stage rustfmt
cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package" -- --check

stage capsule_lock_reconciliation_non_authoritative
cp "$source_lock" "$capsule/Cargo.lock"
# This first command may use the network. It exists only to derive the minimal capsule lock
# and populate the Cargo cache. No qualification PASS can be emitted from this phase.
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

# Explicitly finish all network acquisition before authoritative Cargo execution.
cargo fetch --locked --manifest-path "$capsule/Cargo.toml"
export CARGO_NET_OFFLINE=true

stage offline_metadata
cargo metadata --locked --offline --format-version 1 \
  --manifest-path "$capsule/Cargo.toml" > "$metadata_json"

stage offline_check
cargo check --locked --offline --manifest-path "$capsule/Cargo.toml" -p "$package"

stage offline_test_census
cargo test --locked --offline --manifest-path "$capsule/Cargo.toml" -p "$package" -- --list \
  > "$test_list"
grep -F 'checked_in_golden_vector_replays_to_frozen_sha256_pcr' "$test_list" >/dev/null

stage offline_golden_replay
set -o pipefail
cargo test --locked --offline --manifest-path "$capsule/Cargo.toml" -p "$package" \
  checked_in_golden_vector_replays_to_frozen_sha256_pcr -- --nocapture --test-threads=1 \
  2>&1 | tee "$golden_log"
grep -E 'test .*checked_in_golden_vector_replays_to_frozen_sha256_pcr .* ok' "$golden_log" >/dev/null

stage offline_full_tests
cargo test --locked --offline --manifest-path "$capsule/Cargo.toml" -p "$package" \
  2>&1 | tee "$full_test_log"

stage offline_strict_clippy_all_targets
cargo clippy --locked --offline --manifest-path "$capsule/Cargo.toml" -p "$package" \
  --all-targets -- -D warnings \
  2>&1 | tee "$clippy_log"

stage qualification_receipt
cat > "$receipt" <<EOF_RECEIPT
qualification_contract_revision=assure-linux-ima-rust196-v3-artifact-derived
qualification_result=PASS
product_head=$PRODUCT_HEAD
product_parent=$PRODUCT_PARENT
expected_product_blob=$EXPECTED_PRODUCT_BLOB
expected_product_sha256=$EXPECTED_PRODUCT_SHA256
derivation_id=$DERIVATION_ID
archive_id=$ARCHIVE_ID
authoritative_cargo_network=OFFLINE
source_census_sha256=$(sha256sum "$source_census" | awk '{print $1}')
source_lock_sha256=$(sha256sum "$source_lock" | awk '{print $1}')
capsule_lock_sha256=$(sha256sum "$capsule_lock" | awk '{print $1}')
lock_audit_sha256=$(sha256sum "$lock_audit" | awk '{print $1}')
metadata_sha256=$(sha256sum "$metadata_json" | awk '{print $1}')
test_list_sha256=$(sha256sum "$test_list" | awk '{print $1}')
golden_log_sha256=$(sha256sum "$golden_log" | awk '{print $1}')
full_test_log_sha256=$(sha256sum "$full_test_log" | awk '{print $1}')
clippy_log_sha256=$(sha256sum "$clippy_log" | awk '{print $1}')
EOF_RECEIPT
cat "$receipt"
