#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
VERIFIER="$ROOT/scripts/cogsec-verify-qualification-receipt.sh"
[[ -f "$VERIFIER" ]] || {
  printf 'ERROR: verifier missing: %s\n' "$VERIFIER" >&2
  exit 1
}

TMP="$(mktemp -d "${TMPDIR:-/tmp}/cogsec-receipt-verifier.XXXXXX")"
cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

REPO="$TMP/repo"
mkdir -p "$REPO/scripts"
cd "$REPO"
git init -q
git config user.name 'CogSec Receipt Test'
git config user.email 'cogsec-receipt-test@example.invalid'

cat > Cargo.lock <<'LOCK'
# synthetic lock fixture
version = 4
LOCK
cat > Cargo.toml <<'TOML'
[workspace]
members = []
TOML
cat > scripts/cogsec-focused-qualification.sh <<'QUAL'
#!/usr/bin/env bash
# synthetic qualification fixture
exit 0
QUAL

git add Cargo.lock Cargo.toml scripts/cogsec-focused-qualification.sh
git commit -q -m 'synthetic qualified state'

HEAD_SHA="$(git rev-parse HEAD)"
TREE_SHA="$(git rev-parse 'HEAD^{tree}')"
LOCK_SHA="$(sha256sum Cargo.lock | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum Cargo.toml | awk '{print $1}')"
SCRIPT_SHA="$(sha256sum scripts/cogsec-focused-qualification.sh | awk '{print $1}')"
PACKAGE_SET='symthaea-cogsec,symthaea-cogsec-evidence,symthaea-cogsec-qualification,symthaea-cogsec-shadow-runtime'

write_valid_receipt() {
  local path="$1"
  cat > "$path" <<EOF_RECEIPT
schema_version	2
status	PASS
exit_code	0
head	$HEAD_SHA
tree	$TREE_SHA
cargo_lock_sha256	$LOCK_SHA
workspace_manifest_sha256	$MANIFEST_SHA
qualification_script_sha256	$SCRIPT_SHA
package_set	$PACKAGE_SET
rustc	rustc 1.96.0 (synthetic)
cargo	cargo 1.96.0 (synthetic)
current_gate	complete
last_completed_gate	tracked-state-postcondition
EOF_RECEIPT
}

expect_pass() {
  local name="$1"
  local receipt="$2"
  if ! bash "$VERIFIER" "$receipt" >"$TMP/$name.out" 2>"$TMP/$name.err"; then
    printf 'FAIL: expected verifier success for %s\n' "$name" >&2
    cat "$TMP/$name.err" >&2 || true
    exit 1
  fi
  grep -Fq 'authenticity:        NOT ESTABLISHED' "$TMP/$name.out" || {
    printf 'FAIL: %s did not preserve authenticity non-claim\n' "$name" >&2
    exit 1
  }
}

expect_fail() {
  local name="$1"
  local receipt="$2"
  if bash "$VERIFIER" "$receipt" >"$TMP/$name.out" 2>"$TMP/$name.err"; then
    printf 'FAIL: expected verifier rejection for %s\n' "$name" >&2
    cat "$TMP/$name.out" >&2 || true
    exit 1
  fi
}

VALID="$TMP/valid.tsv"
write_valid_receipt "$VALID"
expect_pass valid "$VALID"

DUP="$TMP/duplicate.tsv"
cp "$VALID" "$DUP"
printf 'status\tPASS\n' >> "$DUP"
expect_fail duplicate-field "$DUP"

UNKNOWN="$TMP/unknown.tsv"
cp "$VALID" "$UNKNOWN"
sed -i '1s/schema_version/unknown_field/' "$UNKNOWN"
expect_fail unknown-field "$UNKNOWN"

TREE_BAD="$TMP/tree-bad.tsv"
cp "$VALID" "$TREE_BAD"
sed -i "s/^tree\t.*/tree\t0000000000000000000000000000000000000000/" "$TREE_BAD"
expect_fail tree-mismatch "$TREE_BAD"

LOCK_BAD="$TMP/lock-bad.tsv"
cp "$VALID" "$LOCK_BAD"
sed -i "s/^cargo_lock_sha256\t.*/cargo_lock_sha256\t$(printf bad | sha256sum | awk '{print $1}')/" "$LOCK_BAD"
expect_fail lock-mismatch "$LOCK_BAD"

PACKAGE_BAD="$TMP/package-bad.tsv"
cp "$VALID" "$PACKAGE_BAD"
sed -i 's/^package_set\t.*/package_set\tsymthaea-cogsec/' "$PACKAGE_BAD"
expect_fail package-scope "$PACKAGE_BAD"

TOOLCHAIN_BAD="$TMP/toolchain-bad.tsv"
cp "$VALID" "$TOOLCHAIN_BAD"
sed -i 's/^rustc\t.*/rustc\trustc 1.95.0 (synthetic)/' "$TOOLCHAIN_BAD"
expect_fail toolchain "$TOOLCHAIN_BAD"

GATE_BAD="$TMP/gate-bad.tsv"
cp "$VALID" "$GATE_BAD"
sed -i 's/^last_completed_gate\t.*/last_completed_gate\tclippy/' "$GATE_BAD"
expect_fail incomplete-postcondition "$GATE_BAD"

STATUS_BAD="$TMP/status-bad.tsv"
cp "$VALID" "$STATUS_BAD"
sed -i 's/^status\tPASS/status\tFAIL/' "$STATUS_BAD"
expect_fail failed-receipt "$STATUS_BAD"

printf 'PASS: CogSec qualification receipt verifier protocol self-test\n'
