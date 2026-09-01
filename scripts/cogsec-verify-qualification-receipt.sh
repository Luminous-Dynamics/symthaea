#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

usage() {
  printf 'usage: bash scripts/cogsec-verify-qualification-receipt.sh <receipt.tsv>\n' >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
RECEIPT="$1"
[[ -f "$RECEIPT" ]] || {
  printf 'ERROR: receipt not found: %s\n' "$RECEIPT" >&2
  exit 1
}

EXPECTED_PACKAGE_SET="symthaea-cogsec,symthaea-cogsec-evidence,symthaea-cogsec-qualification,symthaea-cogsec-shadow-runtime"
EXPECTED_RUST="1.96.0"

required_keys=(
  schema_version
  status
  exit_code
  head
  tree
  cargo_lock_sha256
  workspace_manifest_sha256
  qualification_script_sha256
  package_set
  rustc
  cargo
  current_gate
  last_completed_gate
)

declare -A allowed=()
declare -A values=()
for key in "${required_keys[@]}"; do
  allowed["$key"]=1
done

line_no=0
while IFS=$'\t' read -r key value extra || [[ -n "${key:-}" ]]; do
  line_no=$((line_no + 1))
  [[ -n "${key:-}" ]] || {
    printf 'ERROR: blank/malformed receipt line %d\n' "$line_no" >&2
    exit 1
  }
  [[ -z "${extra:-}" ]] || {
    printf 'ERROR: receipt line %d contains extra tab-delimited fields\n' "$line_no" >&2
    exit 1
  }
  [[ -n "${allowed[$key]:-}" ]] || {
    printf 'ERROR: unknown receipt field: %s\n' "$key" >&2
    exit 1
  }
  [[ -z "${values[$key]+x}" ]] || {
    printf 'ERROR: duplicate receipt field: %s\n' "$key" >&2
    exit 1
  }
  values["$key"]="$value"
done < "$RECEIPT"

for key in "${required_keys[@]}"; do
  [[ -n "${values[$key]+x}" ]] || {
    printf 'ERROR: missing receipt field: %s\n' "$key" >&2
    exit 1
  }
done

[[ "$line_no" -eq "${#required_keys[@]}" ]] || {
  printf 'ERROR: schema-v2 receipt must contain exactly %d fields; found %d\n' \
    "${#required_keys[@]}" "$line_no" >&2
  exit 1
}

[[ "${values[schema_version]}" == "2" ]] || {
  printf 'ERROR: unsupported receipt schema: %s\n' "${values[schema_version]}" >&2
  exit 1
}
[[ "${values[status]}" == "PASS" ]] || {
  printf 'ERROR: receipt is not PASS: %s\n' "${values[status]}" >&2
  exit 1
}
[[ "${values[exit_code]}" == "0" ]] || {
  printf 'ERROR: PASS receipt has non-zero exit code: %s\n' "${values[exit_code]}" >&2
  exit 1
}
[[ "${values[current_gate]}" == "complete" ]] || {
  printf 'ERROR: PASS receipt current_gate is not complete: %s\n' "${values[current_gate]}" >&2
  exit 1
}
[[ "${values[last_completed_gate]}" == "tracked-state-postcondition" ]] || {
  printf 'ERROR: PASS receipt did not complete tracked-state postcondition: %s\n' \
    "${values[last_completed_gate]}" >&2
  exit 1
}
[[ "${values[package_set]}" == "$EXPECTED_PACKAGE_SET" ]] || {
  printf 'ERROR: unexpected qualified package set: %s\n' "${values[package_set]}" >&2
  exit 1
}

rust_version="$(awk '{print $2}' <<< "${values[rustc]}")"
cargo_version="$(awk '{print $2}' <<< "${values[cargo]}")"
[[ "$rust_version" == "$EXPECTED_RUST" ]] || {
  printf 'ERROR: receipt rustc version is not %s: %s\n' "$EXPECTED_RUST" "${values[rustc]}" >&2
  exit 1
}
[[ "$cargo_version" == "$EXPECTED_RUST" ]] || {
  printf 'ERROR: receipt cargo version is not %s: %s\n' "$EXPECTED_RUST" "${values[cargo]}" >&2
  exit 1
}

HEAD_SHA="${values[head]}"
git cat-file -e "${HEAD_SHA}^{commit}" 2>/dev/null || {
  printf 'ERROR: receipt HEAD is not present as a commit in this repository: %s\n' "$HEAD_SHA" >&2
  exit 1
}

ACTUAL_TREE="$(git rev-parse "${HEAD_SHA}^{tree}")"
[[ "$ACTUAL_TREE" == "${values[tree]}" ]] || {
  printf 'ERROR: receipt tree mismatch: claimed=%s actual=%s\n' \
    "${values[tree]}" "$ACTUAL_TREE" >&2
  exit 1
}

hash_path_at_head() {
  local path="$1"
  git cat-file -e "${HEAD_SHA}:${path}" 2>/dev/null || {
    printf 'ERROR: receipt HEAD lacks required path: %s\n' "$path" >&2
    return 1
  }
  git show "${HEAD_SHA}:${path}" | sha256sum | awk '{print $1}'
}

ACTUAL_LOCK="$(hash_path_at_head Cargo.lock)"
ACTUAL_MANIFEST="$(hash_path_at_head Cargo.toml)"
ACTUAL_SCRIPT="$(hash_path_at_head scripts/cogsec-focused-qualification.sh)"

[[ "$ACTUAL_LOCK" == "${values[cargo_lock_sha256]}" ]] || {
  printf 'ERROR: Cargo.lock commitment mismatch\n' >&2
  exit 1
}
[[ "$ACTUAL_MANIFEST" == "${values[workspace_manifest_sha256]}" ]] || {
  printf 'ERROR: Cargo.toml commitment mismatch\n' >&2
  exit 1
}
[[ "$ACTUAL_SCRIPT" == "${values[qualification_script_sha256]}" ]] || {
  printf 'ERROR: qualification-script commitment mismatch\n' >&2
  exit 1
}

RECEIPT_SHA256="$(sha256sum "$RECEIPT" | awk '{print $1}')"
DOMAIN_COMMITMENT="$({
  printf 'COGSEC_QUALIFICATION_RECEIPT/v2\0'
  cat "$RECEIPT"
} | sha256sum | awk '{print $1}')"

printf 'PASS: CogSec qualification receipt is structurally consistent with repository history.\n'
printf 'head:                %s\n' "$HEAD_SHA"
printf 'tree:                %s\n' "$ACTUAL_TREE"
printf 'receipt_sha256:      %s\n' "$RECEIPT_SHA256"
printf 'domain_commitment:   %s\n' "$DOMAIN_COMMITMENT"
printf 'authenticity:        NOT ESTABLISHED\n'
printf 'authority:           NOT ESTABLISHED\n'
