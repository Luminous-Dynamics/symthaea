#!/usr/bin/env bash
set -euo pipefail

: "${FROZEN_BASE:?}"
: "${FORMATTER_PRODUCT_HEAD:?}"
: "${EXPECTED_FORMATTER_BLOB:?}"
: "${PRODUCT_HEAD:?}"
: "${PRODUCT_PARENT:?}"
: "${EXPECTED_PRODUCT_BLOB:?}"
: "${EXPECTED_PRODUCT_SHA256:?}"
: "${EXPECTED_PRODUCT_BYTES:?}"
: "${SUBJECT_SHA:?}"
: "${BASE_SHA:?}"
: "${RECIPE_ID:?}"
: "${RECIPE_SCRIPT_BLOB:?}"
: "${RECIPE_SCRIPT_SHA256:?}"
: "${LOCK_AUDIT_BLOB:?}"
: "${LOCK_AUDIT_SHA256:?}"
: "${METADATA_AUDIT_BLOB:?}"
: "${METADATA_AUDIT_SHA256:?}"
: "${RUNNER_SCRIPT_BLOB:?}"
: "${RUNNER_SCRIPT_SHA256:?}"

root="${RUNNER_TEMP:?}/assure-linux-ima-rust196-v5"
outer="${RUNNER_TEMP}/assure-linux-ima-rust196-v6-outer-receipt.txt"
delta="${RUNNER_TEMP}/qualification-delta.txt"
contract='.github/qualification/assure-linux-ima-rust196-qualify.sh'
lock_audit='.github/qualification/assure-linux-ima-rust196-lock-audit.py'
metadata_audit='.github/qualification/assure-linux-ima-rust196-metadata-audit.py'
runner='.github/qualification/assure-linux-ima-rust196-runner.sh'
workflow='.github/workflows/assure-linux-ima-rust196.yml'
target='crates/domains/symthaea-linux-ima-replay/src/lib.rs'

preflight() {
  actual="$(git rev-parse HEAD)"
  test "$actual" = "$SUBJECT_SHA"
  test "$BASE_SHA" = "$FROZEN_BASE"

  test "$PRODUCT_PARENT" = "$FORMATTER_PRODUCT_HEAD"
  test "$(git rev-parse "$FORMATTER_PRODUCT_HEAD^")" = "$FROZEN_BASE"
  test "$(git rev-parse "$PRODUCT_HEAD^")" = "$PRODUCT_PARENT"
  test "$(git rev-parse HEAD^)" = "$PRODUCT_HEAD"

  test "$(git rev-list --count "$FROZEN_BASE".."$FORMATTER_PRODUCT_HEAD")" = 1
  test "$(git rev-list --count "$FROZEN_BASE".."$PRODUCT_HEAD")" = 2
  test "$(git rev-list --count "$PRODUCT_PARENT".."$PRODUCT_HEAD")" = 1
  test "$(git rev-list --count "$PRODUCT_HEAD"..HEAD)" = 1
  test "$(git rev-list --parents -n1 "$FORMATTER_PRODUCT_HEAD" | awk '{print NF}')" = 2
  test "$(git rev-list --parents -n1 "$PRODUCT_HEAD" | awk '{print NF}')" = 2
  test "$(git rev-list --parents -n1 HEAD | awk '{print NF}')" = 2

  test "$(git diff --name-status "$FROZEN_BASE" "$FORMATTER_PRODUCT_HEAD")" = $'M\tcrates/domains/symthaea-linux-ima-replay/src/lib.rs'
  test "$(git rev-parse "$FORMATTER_PRODUCT_HEAD:$target")" = "$EXPECTED_FORMATTER_BLOB"
  test "$(git diff --name-status "$PRODUCT_PARENT" "$PRODUCT_HEAD")" = $'M\tcrates/domains/symthaea-linux-ima-replay/src/lib.rs'
  test "$(git diff --name-status "$FROZEN_BASE" "$PRODUCT_HEAD")" = $'M\tcrates/domains/symthaea-linux-ima-replay/src/lib.rs'
  test "$(git rev-parse "$PRODUCT_HEAD:$target")" = "$EXPECTED_PRODUCT_BLOB"
  test "$(git cat-file -s "$PRODUCT_HEAD:$target")" = "$EXPECTED_PRODUCT_BYTES"
  test "$(git show "$PRODUCT_HEAD:$target" | sha256sum | awk '{print $1}')" = "$EXPECTED_PRODUCT_SHA256"

  git diff --name-status "$PRODUCT_HEAD" HEAD > "$delta"
  python3 - "$delta" <<'PY'
import pathlib, sys
expected = {
 ('A','.github/qualification/assure-linux-ima-rust196-lock-audit.py'),
 ('A','.github/qualification/assure-linux-ima-rust196-metadata-audit.py'),
 ('A','.github/qualification/assure-linux-ima-rust196-qualify.sh'),
 ('A','.github/qualification/assure-linux-ima-rust196-runner.sh'),
 ('A','.github/workflows/assure-linux-ima-rust196.yml'),
}
actual={tuple(x.split('\t',1)) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x}
if actual != expected: raise SystemExit(f'qualification delta mismatch: {actual!r}')
print('qualification_delta=PASS files=5')
PY

  test "$(git rev-parse "HEAD:$contract")" = "$RECIPE_SCRIPT_BLOB"
  test "$(sha256sum "$contract" | awk '{print $1}')" = "$RECIPE_SCRIPT_SHA256"
  test "$(git rev-parse "HEAD:$lock_audit")" = "$LOCK_AUDIT_BLOB"
  test "$(sha256sum "$lock_audit" | awk '{print $1}')" = "$LOCK_AUDIT_SHA256"
  test "$(git rev-parse "HEAD:$metadata_audit")" = "$METADATA_AUDIT_BLOB"
  test "$(sha256sum "$metadata_audit" | awk '{print $1}')" = "$METADATA_AUDIT_SHA256"
  test "$(git rev-parse "HEAD:$runner")" = "$RUNNER_SCRIPT_BLOB"
  test "$(sha256sum "$runner" | awk '{print $1}')" = "$RUNNER_SCRIPT_SHA256"

  payload="${RUNNER_TEMP}/assure-linux-ima-rust196-v5-recipe-payload.txt"
  printf '%s\n' \
    'schema=symthaea.assurance.linux-ima-rust196-recipe.v5' \
    "contract_sha256=$RECIPE_SCRIPT_SHA256" \
    "lock_audit_sha256=$LOCK_AUDIT_SHA256" \
    "metadata_audit_sha256=$METADATA_AUDIT_SHA256" \
    > "$payload"
  test "$RECIPE_ID" = "sha256:$(sha256sum "$payload" | awk '{print $1}')"

  cat > "$outer" <<EOF_OUTER
schema=symthaea.assurance.linux-ima-rust196-exact-head-qualification.v6
authority=QualificationPending
qualification_state_before_execution=NOT_ESTABLISHED
subject_sha=$actual
subject_tree=$(git rev-parse 'HEAD^{tree}')
frozen_base=$FROZEN_BASE
formatter_product_head=$FORMATTER_PRODUCT_HEAD
formatter_product_blob=$EXPECTED_FORMATTER_BLOB
product_head=$PRODUCT_HEAD
product_parent=$PRODUCT_PARENT
product_blob=$EXPECTED_PRODUCT_BLOB
product_sha256=$EXPECTED_PRODUCT_SHA256
product_bytes=$EXPECTED_PRODUCT_BYTES
recipe_id=$RECIPE_ID
recipe_script_blob=$RECIPE_SCRIPT_BLOB
lock_audit_blob=$LOCK_AUDIT_BLOB
metadata_audit_blob=$METADATA_AUDIT_BLOB
runner_script_blob=$RUNNER_SCRIPT_BLOB
authoritative_cargo_network=OFFLINE
workflow_sha256=$(sha256sum "$workflow" | awk '{print $1}')
source_lock_sha256=$(sha256sum Cargo.lock | awk '{print $1}')
EOF_OUTER
  git rev-parse HEAD > "${RUNNER_TEMP}/pre-head"
  git rev-parse 'HEAD^{tree}' > "${RUNNER_TEMP}/pre-tree"
  sha256sum Cargo.lock | awk '{print $1}' > "${RUNNER_TEMP}/pre-lock"
}

execute_contract() {
  preflight
  bash "$contract" 2>&1 | tee "${RUNNER_TEMP}/assure-linux-ima-rust196-v6-contract.log"
}

postflight() {
  test -f "${RUNNER_TEMP}/pre-head"
  test "$(git rev-parse HEAD)" = "$(cat "${RUNNER_TEMP}/pre-head")"
  test "$(git rev-parse 'HEAD^{tree}')" = "$(cat "${RUNNER_TEMP}/pre-tree")"
  test "$(sha256sum Cargo.lock | awk '{print $1}')" = "$(cat "${RUNNER_TEMP}/pre-lock")"
  test -z "$(git status --porcelain=v1 --untracked-files=all)"
  echo 'postflight_immutable=PASS' >> "$outer"
}

finalize() {
  test -f "$root/contract-outcome.txt"
  test -f "$root/qualification-receipt.txt"
  grep -Fx 'contract_result=PASS' "$root/contract-outcome.txt"
  grep -Fx 'qualification_result=PENDING_POSTFLIGHT' "$root/contract-outcome.txt"
  grep -Fx 'contract_result=PASS' "$root/qualification-receipt.txt"
  grep -Fx 'qualification_result=PENDING_POSTFLIGHT' "$root/qualification-receipt.txt"
  grep -Fx 'authoritative_cargo_network=OFFLINE' "$root/qualification-receipt.txt"
  grep -Fx 'postflight_immutable=PASS' "$outer"
  echo 'qualification_result=PASS' >> "$outer"
  echo 'authority=QualifiedExactHead' >> "$outer"
}

case "${1:-}" in
  execute) execute_contract ;;
  postflight) postflight ;;
  finalize) finalize ;;
  *) echo 'usage: runner.sh execute|postflight|finalize' >&2; exit 64 ;;
esac
