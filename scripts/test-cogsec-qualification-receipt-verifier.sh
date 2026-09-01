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
TAB=$'\t'

write_valid_receipt() {
  local path="$1"
  {
    printf 'schema_version\t2\n'
    printf 'status\tPASS\n'
    printf 'exit_code\t0\n'
    printf 'head\t%s\n' "$HEAD_SHA"
    printf 'tree\t%s\n' "$TREE_SHA"
    printf 'cargo_lock_sha256\t%s\n' "$LOCK_SHA"
    printf 'workspace_manifest_sha256\t%s\n' "$MANIFEST_SHA"
    printf 'qualification_script_sha256\t%s\n' "$SCRIPT_SHA"
    printf 'package_set\t%s\n' "$PACKAGE_SET"
    printf 'rustc\trustc 1.96.0 (synthetic)\n'
    printf 'cargo\tcargo 1.96.0 (synthetic)\n'
    printf 'current_gate\tcomplete\n'
    printf 'last_completed_gate\ttracked-state-postcondition\n'
  } > "$path"
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
sed -i "s/^tree${TAB}.*/tree${TAB}0000000000000000000000000000000000000000/" "$TREE_BAD"
expect_fail tree-mismatch "$TREE_BAD"

LOCK_BAD="$TMP/lock-bad.tsv"
cp "$VALID" "$LOCK_BAD"
sed -i "s/^cargo_lock_sha256${TAB}.*/cargo_lock_sha256${TAB}$(printf bad | sha256sum | awk '{print $1}')/" "$LOCK_BAD"
expect_fail lock-mismatch "$LOCK_BAD"

PACKAGE_BAD="$TMP/package-bad.tsv"
cp "$VALID" "$PACKAGE_BAD"
sed -i "s/^package_set${TAB}.*/package_set${TAB}symthaea-cogsec/" "$PACKAGE_BAD"
expect_fail package-scope "$PACKAGE_BAD"

TOOLCHAIN_BAD="$TMP/toolchain-bad.tsv"
cp "$VALID" "$TOOLCHAIN_BAD"
sed -i "s/^rustc${TAB}.*/rustc${TAB}rustc 1.95.0 (synthetic)/" "$TOOLCHAIN_BAD"
expect_fail toolchain "$TOOLCHAIN_BAD"

GATE_BAD="$TMP/gate-bad.tsv"
cp "$VALID" "$GATE_BAD"
sed -i "s/^last_completed_gate${TAB}.*/last_completed_gate${TAB}clippy/" "$GATE_BAD"
expect_fail incomplete-postcondition "$GATE_BAD"

STATUS_BAD="$TMP/status-bad.tsv"
cp "$VALID" "$STATUS_BAD"
sed -i "s/^status${TAB}PASS/status${TAB}FAIL/" "$STATUS_BAD"
expect_fail failed-receipt "$STATUS_BAD"

printf 'PASS: CogSec qualification receipt verifier protocol self-test\n'
