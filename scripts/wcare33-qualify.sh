#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WCARE32_PARENT="9577c39533b2267a199f726a8e80247b006d65a0"
WCARE29_PARENT="364b5a54672e3b9f88e868c5b6f6a4fcfac7089c"
COMPOSITION="05e97bceb9ed15152503af82b53f6a50998cc616"
CORPUS_SHA256="85c3968ddde3afb090d89a018db1c804bd2b7375ad5527536c2065e2841edd4e"
SCHEMA_SHA256="904be17a8a3d9ee38479ae5e39efb89a22c7e6f023b8bd797ee12b1cb965c80f"
MANIFEST_SHA256="5f4234d7afba2c3facd956e8ddabdc09118666e835187c6b7b7b978201c3c7b9"

OUT_DIR="${WCARE33_OUT_DIR:-target/wcare33}"
RECEIPT="$OUT_DIR/receipt.json"
COMPILE_LOG="$OUT_DIR/compile.log"
CORPUS_LOG="$OUT_DIR/corpus-integrity.log"
RUNNER_LOG="$OUT_DIR/adversarial-runner.log"
METAMORPHIC_LOG="$OUT_DIR/metamorphic-invariants.log"
mkdir -p "$OUT_DIR"

classification="INFRASTRUCTURE_INDETERMINATE"
stage="bootstrap"
detail="not_started"
head_sha="unknown"
rustc_version="unavailable"
cargo_version="unavailable"
compile_status="not_run"
corpus_status="not_run"
subject_cases_status="not_run"
metamorphic_status="not_run"
failed_case_ids_json="[]"

json_safe() {
  printf '%s' "$1" | tr '\n\r\t"\\' '     '
}

collect_failed_case_ids() {
  local ids id out="" separator=""
  ids="$(grep 'FAILED' "$RUNNER_LOG" 2>/dev/null \
    | sed -n 's/.*wcare32_\([0-9][0-9][0-9]\)_[A-Za-z0-9_]*.*/WCARE32-\1/p' \
    | sort -u || true)"
  if [[ -z "$ids" ]]; then
    failed_case_ids_json="[]"
    return
  fi
  while IFS= read -r id; do
    [[ -z "$id" ]] && continue
    out="${out}${separator}\"${id}\""
    separator=","
  done <<< "$ids"
  failed_case_ids_json="[$out]"
}

emit_receipt() {
  cat > "$RECEIPT" <<EOF
{"authority":"MeasurementOnly","case_count":24,"cargo":"$(json_safe "$cargo_version")","classification":"$classification","compile_status":"$compile_status","composition_commit":"$COMPOSITION","corpus_sha256":"$CORPUS_SHA256","corpus_status":"$corpus_status","detail":"$(json_safe "$detail")","failed_case_ids":$failed_case_ids_json,"head_sha":"$head_sha","manifest_sha256":"$MANIFEST_SHA256","metamorphic_required":true,"metamorphic_status":"$metamorphic_status","required_case_range":"WCARE32-001..WCARE32-024","rustc":"$(json_safe "$rustc_version")","schema_sha256":"$SCHEMA_SHA256","stage":"$stage","subject_cases_status":"$subject_cases_status","wcare29_parent":"$WCARE29_PARENT","wcare32_parent":"$WCARE32_PARENT"}
EOF
  printf 'WCARE-33 %s (%s: %s)\nreceipt: %s\n' "$classification" "$stage" "$detail" "$RECEIPT"
}

finish() {
  local code="$1"
  emit_receipt
  exit "$code"
}

if ! command -v git >/dev/null 2>&1; then
  detail="git_missing"
  finish 2
fi
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
  detail="rust_toolchain_missing"
  finish 2
fi

cargo_version="$(cargo --version 2>/dev/null || true)"
rustc_version="$(rustc --version 2>/dev/null || true)"
head_sha="$(git rev-parse HEAD 2>/dev/null || printf unknown)"

stage="subject_identity"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  detail="working_tree_not_clean"
  finish 2
fi
for sha in "$WCARE32_PARENT" "$WCARE29_PARENT" "$COMPOSITION"; do
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    detail="required_subject_commit_unavailable:$sha"
    finish 2
  fi
done
if ! git merge-base --is-ancestor "$WCARE32_PARENT" HEAD \
  || ! git merge-base --is-ancestor "$WCARE29_PARENT" HEAD \
  || ! git merge-base --is-ancestor "$COMPOSITION" HEAD; then
  classification="FAIL_SUBJECT"
  detail="head_does_not_contain_exact_composed_subject"
  finish 1
fi

compile_failure_classification() {
  local log="$1"
  if grep -Eiq \
    'could not resolve host|failed to (download|fetch)|network failure|timed out|no space left on device|linker .* not found|is not available in offline mode' \
    "$log"; then
    printf 'INFRASTRUCTURE_INDETERMINATE'
  else
    printf 'FAIL_SUBJECT'
  fi
}

stage="compile"
if ! cargo test --locked -p symthaea-wisdom \
  --test wcare32_corpus_integrity \
  --test wcare33_reciprocal_adversarial_runner \
  --test wcare33_metamorphic_invariants \
  --no-run >"$COMPILE_LOG" 2>&1; then
  compile_status="fail"
  classification="$(compile_failure_classification "$COMPILE_LOG")"
  detail="compile_gate_failed"
  [[ "$classification" == "INFRASTRUCTURE_INDETERMINATE" ]] && finish 2
  finish 1
fi
compile_status="pass"

stage="corpus_integrity"
if ! cargo test --locked -p symthaea-wisdom \
  --test wcare32_corpus_integrity -- --nocapture >"$CORPUS_LOG" 2>&1; then
  corpus_status="invalid"
  classification="INVALID_CORPUS"
  detail="frozen_corpus_integrity_failed"
  finish 1
fi
corpus_status="pass"

stage="subject_cases"
if ! cargo test --locked -p symthaea-wisdom \
  --test wcare33_reciprocal_adversarial_runner -- --nocapture >"$RUNNER_LOG" 2>&1; then
  subject_cases_status="fail"
  collect_failed_case_ids
  classification="FAIL_SUBJECT"
  detail="one_or_more_frozen_cases_failed"
  finish 1
fi
subject_cases_status="pass"
failed_case_ids_json="[]"

stage="metamorphic_invariants"
if ! cargo test --locked -p symthaea-wisdom \
  --test wcare33_metamorphic_invariants -- --nocapture >"$METAMORPHIC_LOG" 2>&1; then
  metamorphic_status="fail"
  classification="FAIL_SUBJECT"
  detail="one_or_more_metamorphic_invariants_failed"
  finish 1
fi
metamorphic_status="pass"

classification="PASS_SUBJECT"
stage="complete"
detail="all_24_frozen_cases_global_boundaries_and_metamorphic_invariants_passed"
finish 0
