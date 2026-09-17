#!/usr/bin/env bash
set -euo pipefail

: "${PRODUCT_HEAD:?PRODUCT_HEAD required}"
: "${PRODUCT_PARENT:?PRODUCT_PARENT required}"
: "${EXPECTED_PRODUCT_BLOB:?EXPECTED_PRODUCT_BLOB required}"
: "${EXPECTED_PRODUCT_SHA256:?EXPECTED_PRODUCT_SHA256 required}"
: "${EXPECTED_PRODUCT_BYTES:?EXPECTED_PRODUCT_BYTES required}"
: "${DERIVATION_ID:?DERIVATION_ID required}"
: "${ARCHIVE_ID:?ARCHIVE_ID required}"

package="symthaea-linux-ima-replay"
package_version="0.1.0"
crate="crates/domains/symthaea-linux-ima-replay"
source_path="$crate/src/lib.rs"
work="${RUNNER_TEMP:-/tmp}/assure-linux-ima-rust196-v5"
capsule="$work/capsule"
source_census="$work/source-census.tsv"
source_lock="$work/Cargo.lock.source"
capsule_lock="$work/Cargo.lock.capsule"
lock_audit="$work/lock-audit.json"
metadata_json="$work/cargo-metadata.offline.json"
metadata_audit="$work/metadata-audit.json"
test_list="$work/test-list.txt"
golden_log="$work/golden-replay-test.log"
full_test_log="$work/full-test.log"
clippy_log="$work/clippy-all-targets.log"
receipt="$work/qualification-receipt.txt"
outcome="$work/contract-outcome.txt"
current_stage="$work/current-stage.txt"

rm -rf "$work"
mkdir -p "$capsule/$crate"

stage() {
  printf '%s\n' "$1" > "$current_stage"
  printf 'qualification_gate=%s\n' "$1"
}

on_exit() {
  rc=$?
  last_gate="not_started"
  if [[ -f "$current_stage" ]]; then
    last_gate="$(cat "$current_stage")"
  fi
  if [[ "$rc" -eq 0 ]]; then
    cat > "$outcome" <<EOF_OUTCOME
contract_result=PASS
qualification_result=PENDING_POSTFLIGHT
authority=ContractPassedPendingPostflight
last_gate=$last_gate
exit_code=0
EOF_OUTCOME
  else
    cat > "$outcome" <<EOF_OUTCOME
contract_result=FAIL
qualification_result=NOT_ESTABLISHED
authority=None
failed_gate=$last_gate
exit_code=$rc
EOF_OUTCOME
  fi
}
trap on_exit EXIT

stage product_identity

git cat-file -e "$PRODUCT_HEAD^{commit}"
test "$(git rev-parse "$PRODUCT_HEAD^")" = "$PRODUCT_PARENT"
test "$(git rev-list --count "$PRODUCT_PARENT".."$PRODUCT_HEAD")" = "1"
test "$(git rev-list --parents -n 1 "$PRODUCT_HEAD" | awk '{print NF}')" = "2"
test "$(git diff --name-status "$PRODUCT_PARENT" "$PRODUCT_HEAD")" = $'M\tcrates/domains/symthaea-linux-ima-replay/src/lib.rs'
test "$(git rev-parse "$PRODUCT_HEAD:$source_path")" = "$EXPECTED_PRODUCT_BLOB"
test "$(git cat-file -s "$PRODUCT_HEAD:$source_path")" = "$EXPECTED_PRODUCT_BYTES"
test "$(git show "$PRODUCT_HEAD:$source_path" | sha256sum | awk '{print $1}')" = "$EXPECTED_PRODUCT_SHA256"

stage git_object_capsule

printf 'path\tgit_blob_sha1\tsha256\tbytes\n' > "$source_census"
while IFS= read -r -d '' path; do
  mkdir -p "$capsule/$(dirname "$path")"
  git show "$PRODUCT_HEAD:$path" > "$capsule/$path"
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
# This command may use the network. It only derives the minimal capsule lock and
# populates the Cargo cache. Nothing in this phase has qualification authority.
cargo check --manifest-path "$capsule/Cargo.toml" -p "$package"
cp "$capsule/Cargo.lock" "$capsule_lock"

stage semantic_lock_graph_audit
python3 .github/qualification/assure-linux-ima-rust196-lock-audit.py \
  "$source_lock" "$capsule_lock" "$lock_audit" "$package" "$package_version"

# Finish all network acquisition before authoritative Cargo execution.
cargo fetch --locked --manifest-path "$capsule/Cargo.toml"
export CARGO_NET_OFFLINE=true

stage offline_metadata
cargo metadata --locked --offline --format-version 1 \
  --manifest-path "$capsule/Cargo.toml" > "$metadata_json"

stage offline_resolved_graph_audit
python3 .github/qualification/assure-linux-ima-rust196-metadata-audit.py \
  "$source_lock" "$capsule_lock" "$metadata_json" "$metadata_audit" "$package" "$package_version"

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
qualification_contract_revision=assure-linux-ima-rust196-v5-local-git-external-lock-graph
contract_result=PASS
qualification_result=PENDING_POSTFLIGHT
authority=ContractPassedPendingPostflight
product_head=$PRODUCT_HEAD
product_parent=$PRODUCT_PARENT
expected_product_blob=$EXPECTED_PRODUCT_BLOB
expected_product_sha256=$EXPECTED_PRODUCT_SHA256
expected_product_bytes=$EXPECTED_PRODUCT_BYTES
derivation_id=$DERIVATION_ID
archive_id=$ARCHIVE_ID
authoritative_cargo_network=OFFLINE
source_census_sha256=$(sha256sum "$source_census" | awk '{print $1}')
source_lock_sha256=$(sha256sum "$source_lock" | awk '{print $1}')
capsule_lock_sha256=$(sha256sum "$capsule_lock" | awk '{print $1}')
lock_audit_sha256=$(sha256sum "$lock_audit" | awk '{print $1}')
metadata_sha256=$(sha256sum "$metadata_json" | awk '{print $1}')
metadata_audit_sha256=$(sha256sum "$metadata_audit" | awk '{print $1}')
test_list_sha256=$(sha256sum "$test_list" | awk '{print $1}')
golden_log_sha256=$(sha256sum "$golden_log" | awk '{print $1}')
full_test_log_sha256=$(sha256sum "$full_test_log" | awk '{print $1}')
clippy_log_sha256=$(sha256sum "$clippy_log" | awk '{print $1}')
EOF_RECEIPT
cat "$receipt"
