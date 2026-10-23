#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portable exact-head bootstrap qualification harness for research crates.
#
# Qualification theorem v2. CI is only an execution provider. This script may
# run under GitHub Actions, Nix, a local checkout, or another CI provider.

set -euo pipefail

readonly RECEIPT_SCHEMA="symthaea.research-bootstrap-receipt.v2"
readonly PROGRAM_RE='^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$'
readonly SHA_RE='^[0-9a-f]{40}$'
readonly RUST_RE='^[0-9]+\.[0-9]+\.[0-9]+$'
readonly PACKAGE_RE='^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$'
readonly PATH_RE='^[A-Za-z0-9._/-]+$'
readonly MAX_LOCK_BYTES=$((16 * 1024 * 1024))
readonly MAX_PATCH_BYTES=$((16 * 1024 * 1024))
readonly MAX_RECEIPT_BYTES=$((256 * 1024))

usage() {
  cat <<'USAGE'
Usage:
  scripts/qualify-research-crate.sh \
    --program SCI-001A \
    --source-parent <40-hex-sha> \
    --expected-head <40-hex-sha> \
    --expected-rust 1.96.0 \
    --package symthaea-science-research \
    --source-path crates/core/symthaea-science-research \
    --qualifier-path .github/research-qualifiers/SCI-001AQ.json \
    --output-dir /tmp/sci-001a-evidence

Repeat --package, --source-path, and --qualifier-path as needed.

The theorem requires:
  * HEAD is exactly --expected-head and has exactly one parent;
  * that sole parent is exactly --source-parent;
  * the parent->HEAD diff is exactly the canonical qualifier-path set with
    rename/external-diff interpretation disabled;
  * frozen source paths have identical regular Git objects parent->HEAD and
    contain no symlink/gitlink descendants;
  * direct CLI identities/paths are canonical, bounded, and duplicate-free;
  * rustc and cargo version tokens equal --expected-rust exactly;
  * Cargo.lock is a tracked regular non-symlink file;
  * evidence output is a fresh external directory;
  * Cargo materializes a non-empty additive-only bootstrap lock candidate;
  * package check, rustfmt, tests, and strict all-target Clippy pass;
  * no working-tree mutation exists except temporary Cargo.lock drift;
  * Cargo.lock is restored and the repository is clean before success.

This is bootstrap source qualification only. It grants no scientific claim,
replication, independence, or final locked-build authority.
USAGE
}

fail() {
  echo "qualification error: $*" >&2
  exit 1
}

is_program() { [[ "$1" =~ $PROGRAM_RE ]]; }
is_sha() { [[ "$1" =~ $SHA_RE ]]; }
is_rust_version() { [[ "$1" =~ $RUST_RE ]]; }
is_package() { [[ "$1" =~ $PACKAGE_RE ]]; }

is_repo_path() {
  local value="$1"
  [[ -n "$value" ]] || return 1
  ((${#value} <= 512)) || return 1
  [[ "$value" =~ $PATH_RE ]] || return 1
  [[ "$value" != /* ]] || return 1
  [[ "$value" != ./* ]] || return 1
  [[ "$value" != */ ]] || return 1
  [[ "$value" != *//* ]] || return 1
  [[ "$value" != *:* ]] || return 1
  local component
  IFS='/' read -r -a _components <<< "$value"
  ((${#_components[@]} > 0)) || return 1
  for component in "${_components[@]}"; do
    [[ -n "$component" && "$component" != "." && "$component" != ".." ]] || return 1
  done
}

require_unique() {
  local label="$1"
  shift
  local -A seen=()
  local value
  for value in "$@"; do
    [[ -z "${seen[$value]+x}" ]] || fail "duplicate $label entry: $value"
    seen["$value"]=1
  done
}

require_git_source_object() {
  local commit="$1"
  local path="$2"
  local line metadata observed_path mode type object
  line="$(git ls-tree "$commit" -- "$path")"
  [[ -n "$line" && "$line" == *$'\t'* ]] || fail "missing or ambiguous Git source object at $commit:$path"
  metadata="${line%%$'\t'*}"
  observed_path="${line#*$'\t'}"
  [[ "$observed_path" == "$path" ]] || fail "unexpected Git path while resolving source object: $observed_path"
  read -r mode type object <<< "$metadata"
  case "$mode:$type" in
    100644:blob|100755:blob|040000:tree) ;;
    *) fail "forbidden source Git object mode/type at $commit:$path: $mode/$type" ;;
  esac
  is_sha "$object" || fail "invalid source Git object identity at $commit:$path"

  while IFS=$'\t' read -r metadata observed_path; do
    [[ -n "$metadata" ]] || continue
    read -r mode type object <<< "$metadata"
    case "$mode:$type" in
      100644:blob|100755:blob) ;;
      *) fail "source tree contains forbidden Git object $mode/$type: $observed_path" ;;
    esac
    is_sha "$object" || fail "invalid descendant Git object identity under $path"
  done < <(git ls-tree -r "$commit" -- "$path")
}

require_git_qualifier_object() {
  local commit="$1"
  local path="$2"
  local line metadata observed_path mode type object
  line="$(git ls-tree "$commit" -- "$path")"
  [[ -n "$line" && "$line" == *$'\t'* ]] || fail "missing or ambiguous qualifier Git object at $commit:$path"
  metadata="${line%%$'\t'*}"
  observed_path="${line#*$'\t'}"
  [[ "$observed_path" == "$path" ]] || fail "unexpected Git path while resolving qualifier object: $observed_path"
  read -r mode type object <<< "$metadata"
  case "$mode:$type" in
    100644:blob|100755:blob) ;;
    *) fail "qualifier path must be a regular Git blob, observed $mode/$type: $path" ;;
  esac
  is_sha "$object" || fail "invalid qualifier Git object identity: $path"
}

publish_regular_file() {
  local source="$1"
  local destination="$2"
  local temporary="${destination}.tmp.$$"
  [[ -f "$source" && ! -L "$source" ]] || fail "evidence source is not a regular file: $source"
  rm -f -- "$temporary"
  install -m 600 -- "$source" "$temporary"
  mv -fT -- "$temporary" "$destination"
  [[ -f "$destination" && ! -L "$destination" ]] || fail "failed to publish regular evidence file: $destination"
}

self_test() {
  is_program "SCI-001A"
  ! is_program $'SCI-001A\nforged'
  is_sha "1111111111111111111111111111111111111111"
  ! is_sha "1111111"
  is_rust_version "1.96.0"
  ! is_rust_version "1.96"
  is_package "symthaea-science-research"
  ! is_package "bad package"
  is_repo_path "crates/core/symthaea-science-research"
  is_repo_path ".github/research-qualifiers/SCI-001AQ.json"
  ! is_repo_path "/absolute/path"
  ! is_repo_path "./relative/path"
  ! is_repo_path "crates//core"
  ! is_repo_path "crates/../core"
  ! is_repo_path "crates/./core"
  ! is_repo_path "path:ambiguous"
  require_unique "self-test" alpha beta gamma
  echo "research_qualification_harness_self_test=PASS"
}

program=""
source_parent=""
expected_head=""
expected_rust="1.96.0"
output_dir=""
packages=()
source_paths=()
qualifier_paths=()
self_test_only=false

while (($#)); do
  case "$1" in
    --program)
      program="${2:?missing value for --program}"
      shift 2
      ;;
    --source-parent)
      source_parent="${2:?missing value for --source-parent}"
      shift 2
      ;;
    --expected-head)
      expected_head="${2:?missing value for --expected-head}"
      shift 2
      ;;
    --expected-rust)
      expected_rust="${2:?missing value for --expected-rust}"
      shift 2
      ;;
    --package)
      packages+=("${2:?missing value for --package}")
      shift 2
      ;;
    --source-path)
      source_paths+=("${2:?missing value for --source-path}")
      shift 2
      ;;
    --qualifier-path)
      qualifier_paths+=("${2:?missing value for --qualifier-path}")
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --self-test)
      self_test_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$self_test_only" == true ]]; then
  self_test
  exit 0
fi

[[ -n "$program" && -n "$source_parent" && -n "$expected_head" && -n "$output_dir" ]] || {
  echo "--program, --source-parent, --expected-head, and --output-dir are required" >&2
  exit 2
}
((${#packages[@]} > 0 && ${#source_paths[@]} > 0 && ${#qualifier_paths[@]} > 0)) || {
  echo "at least one --package, --source-path, and --qualifier-path is required" >&2
  exit 2
}

is_program "$program" || fail "non-canonical --program: $program"
is_sha "$source_parent" || fail "--source-parent must be exactly 40 lowercase hex characters"
is_sha "$expected_head" || fail "--expected-head must be exactly 40 lowercase hex characters"
is_rust_version "$expected_rust" || fail "--expected-rust must have x.y.z form"

for package in "${packages[@]}"; do
  is_package "$package" || fail "non-canonical package identity: $package"
done
for path in "${source_paths[@]}"; do
  is_repo_path "$path" || fail "non-canonical source path: $path"
done
for path in "${qualifier_paths[@]}"; do
  is_repo_path "$path" || fail "non-canonical qualifier path: $path"
done
require_unique "package" "${packages[@]}"
require_unique "source path" "${source_paths[@]}"
require_unique "qualifier path" "${qualifier_paths[@]}"

repo_root="$(git rev-parse --show-toplevel)"
repo_root="$(cd "$repo_root" && pwd -P)"
cd "$repo_root"

actual_head="$(git rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || fail "head mismatch: expected $expected_head, observed $actual_head"

read -r -a commit_and_parents <<< "$(git rev-list --parents -n 1 HEAD)"
((${#commit_and_parents[@]} == 2)) || fail "qualifier HEAD must have exactly one parent"
actual_parent="${commit_and_parents[1]}"
[[ "$actual_parent" == "$source_parent" ]] || fail "parent mismatch: expected $source_parent, observed $actual_parent"

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "qualification must start from a clean working tree" >&2
  git status --short >&2
  exit 1
fi

mapfile -t actual_scope < <(git diff --no-ext-diff --no-renames --name-only "$source_parent" HEAD -- | LC_ALL=C sort -u)
mapfile -t expected_scope < <(printf '%s\n' "${qualifier_paths[@]}" | LC_ALL=C sort -u)
if ! diff -u \
  <(printf '%s\n' "${expected_scope[@]}") \
  <(printf '%s\n' "${actual_scope[@]}"); then
  fail "qualification diff is not exactly the declared qualifier path set"
fi

for path in "${qualifier_paths[@]}"; do
  require_git_qualifier_object HEAD "$path"
  [[ -f "$path" && ! -L "$path" ]] || fail "qualifier path must check out as a regular non-symlink file: $path"
done

for path in "${source_paths[@]}"; do
  require_git_source_object "$source_parent" "$path"
  require_git_source_object HEAD "$path"
  [[ -e "$path" && ! -L "$path" ]] || fail "source path must check out as a regular non-symlink file/directory: $path"
  resolved_path="$(realpath -e -- "$path")"
  [[ "$resolved_path" == "$repo_root" || "$resolved_path" == "$repo_root/"* ]] || fail "source path resolves outside repository: $path"
  source_object="$(git rev-parse "$source_parent:$path")"
  head_object="$(git rev-parse "HEAD:$path")"
  [[ "$source_object" == "$head_object" ]] || fail "source path changed in qualification commit: $path"
done

command -v rustc >/dev/null || fail "rustc not found"
command -v cargo >/dev/null || fail "cargo not found"
command -v rustfmt >/dev/null || fail "rustfmt not found"
command -v sha256sum >/dev/null || fail "sha256sum not found"
command -v realpath >/dev/null || fail "realpath not found"
command -v install >/dev/null || fail "install not found"
cargo clippy --version >/dev/null 2>&1 || fail "cargo clippy not available"

rustc_full="$(rustc --version)"
cargo_full="$(cargo --version)"
rustfmt_full="$(rustfmt --version)"
clippy_full="$(cargo clippy --version)"
rustc_version="$(awk '{print $2}' <<< "$rustc_full")"
cargo_version="$(awk '{print $2}' <<< "$cargo_full")"
[[ "$rustc_version" == "$expected_rust" ]] || fail "rustc version mismatch: expected $expected_rust, observed $rustc_version"
[[ "$cargo_version" == "$expected_rust" ]] || fail "cargo version mismatch: expected $expected_rust, observed $cargo_version"

[[ -f Cargo.lock && ! -L Cargo.lock ]] || fail "Cargo.lock must be a regular non-symlink file"
git ls-files --error-unmatch -- Cargo.lock >/dev/null 2>&1 || fail "Cargo.lock must be tracked"
lock_mode_type="$(git ls-tree HEAD -- Cargo.lock | awk '{print $1 ":" $2}')"
[[ "$lock_mode_type" == "100644:blob" ]] || fail "Cargo.lock must be a non-executable regular Git blob"

output_parent="$(dirname -- "$output_dir")"
output_name="$(basename -- "$output_dir")"
[[ "$output_name" != "." && "$output_name" != ".." && -n "$output_name" ]] || fail "invalid --output-dir basename"
[[ -d "$output_parent" && ! -L "$output_parent" ]] || fail "--output-dir parent must already exist as a non-symlink directory"
output_parent="$(cd "$output_parent" && pwd -P)"
output_dir="$output_parent/$output_name"
if [[ "$output_dir" == "$repo_root" || "$output_dir" == "$repo_root/"* ]]; then
  fail "--output-dir must resolve outside the subject repository"
fi
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || fail "--output-dir must not already exist"
mkdir -m 700 -- "$output_dir"

# Scratch evidence is built privately and published into the external evidence
# directory only through regular-file replacement. The theorem still does not
# claim protection against hostile same-UID background processes.
tmp_dir="$(mktemp -d)"
chmod 700 "$tmp_dir"
lock_before="$tmp_dir/Cargo.lock.before"
generated_tmp="$tmp_dir/Cargo.lock.generated"
patch_tmp="$tmp_dir/Cargo.lock.patch"
receipt_tmp="$tmp_dir/receipt.txt"
cp -- Cargo.lock "$lock_before"

restore_lock() {
  cp -- "$lock_before" Cargo.lock
  rm -rf -- "$tmp_dir"
}
trap restore_lock EXIT

for package in "${packages[@]}"; do
  cargo check -p "$package"
done

unexpected_after_check="$({ git status --porcelain=v1 || true; } | grep -v '^ M Cargo.lock$' || true)"
if [[ -n "$unexpected_after_check" ]]; then
  echo "unexpected working-tree mutation during bootstrap cargo check:" >&2
  printf '%s\n' "$unexpected_after_check" >&2
  exit 1
fi

cp -- Cargo.lock "$generated_tmp"
git diff --no-ext-diff --no-renames -- Cargo.lock > "$patch_tmp"
[[ -s "$patch_tmp" ]] || fail "expected a non-empty bootstrap lock candidate"
read -r lock_additions lock_deletions lock_path < <(git diff --no-ext-diff --numstat -- Cargo.lock)
[[ "$lock_path" == "Cargo.lock" ]] || fail "unexpected Cargo.lock numstat path: ${lock_path:-<empty>}"
[[ "$lock_additions" =~ ^[0-9]+$ && "$lock_deletions" =~ ^[0-9]+$ ]] || fail "Cargo.lock numstat was not numeric"
(( lock_additions > 0 )) || fail "bootstrap lock candidate must add at least one line"
(( lock_deletions == 0 )) || fail "bootstrap lock candidate must not delete or replace existing lock lines"
(( $(stat -c '%s' "$generated_tmp") <= MAX_LOCK_BYTES )) || fail "generated Cargo.lock exceeds evidence size bound"
(( $(stat -c '%s' "$patch_tmp") <= MAX_PATCH_BYTES )) || fail "Cargo.lock patch exceeds evidence size bound"
for package in "${packages[@]}"; do
  grep -F "name = \"$package\"" Cargo.lock >/dev/null || fail "generated lock does not contain package: $package"
done

fmt_args=()
for package in "${packages[@]}"; do
  fmt_args+=("-p" "$package")
done
cargo fmt --check "${fmt_args[@]}"

for package in "${packages[@]}"; do
  cargo test -p "$package"
  cargo clippy -p "$package" --all-targets -- -D warnings
done

for path in "${source_paths[@]}"; do
  git diff --no-ext-diff --quiet HEAD -- "$path" || fail "working-tree source path changed during qualification: $path"
  git diff --no-ext-diff --cached --quiet HEAD -- "$path" || fail "index source path changed during qualification: $path"
done

unexpected="$({ git status --porcelain=v1 || true; } | grep -v '^ M Cargo.lock$' || true)"
if [[ -n "$unexpected" ]]; then
  echo "unexpected working-tree mutation during qualification:" >&2
  printf '%s\n' "$unexpected" >&2
  exit 1
fi

lock_sha256="$(sha256sum "$generated_tmp" | cut -d' ' -f1)"
patch_sha256="$(sha256sum "$patch_tmp" | cut -d' ' -f1)"
harness_sha256="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
host_triple="$(rustc -vV | awk -F': ' '$1 == "host" {print $2}')"

{
  echo "schema=$RECEIPT_SCHEMA"
  echo "program=$program"
  echo "authority=bootstrap-source-qualification-only"
  echo "scientific_claim=NONE"
  echo "source_subject_sha=$source_parent"
  echo "qualifier_sha=$actual_head"
  echo "qualifier_parent_count=1"
  echo "harness_sha256=$harness_sha256"
  echo "rustc_version=$rustc_full"
  echo "cargo_version=$cargo_full"
  echo "rustfmt_version=$rustfmt_full"
  echo "clippy_version=$clippy_full"
  echo "rust_host_triple=$host_triple"
  echo "generated_lock_sha256=$lock_sha256"
  echo "lock_patch_sha256=$patch_sha256"
  for package in "${packages[@]}"; do
    echo "package=$package"
  done
  for path in "${source_paths[@]}"; do
    echo "source_path=$path"
    echo "source_object=$path:$(git rev-parse "$source_parent:$path")"
  done
  for path in "${qualifier_paths[@]}"; do
    echo "qualifier_path=$path"
  done
  echo "scope_gate=PASS"
  echo "source_immutable_gate=PASS"
  echo "lock_additive_gate=PASS"
  echo "cargo_check_gate=PASS"
  echo "rustfmt_gate=PASS"
  echo "cargo_test_gate=PASS"
  echo "clippy_gate=PASS"
  echo "postflight_worktree_immutable_gate=PASS"
} > "$receipt_tmp"
(( $(stat -c '%s' "$receipt_tmp") <= MAX_RECEIPT_BYTES )) || fail "receipt exceeds evidence size bound"

publish_regular_file "$generated_tmp" "$output_dir/Cargo.lock.generated"
publish_regular_file "$patch_tmp" "$output_dir/Cargo.lock.patch"
publish_regular_file "$receipt_tmp" "$output_dir/receipt.txt"

mapfile -t evidence_entries < <(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
expected_evidence=("Cargo.lock.generated" "Cargo.lock.patch" "receipt.txt")
if ! diff -u \
  <(printf '%s\n' "${expected_evidence[@]}" | LC_ALL=C sort) \
  <(printf '%s\n' "${evidence_entries[@]}"); then
  fail "external evidence directory contains unexpected entries"
fi

restore_lock
trap - EXIT

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "working tree is not clean after Cargo.lock restoration" >&2
  git status --short >&2
  exit 1
fi

printf 'bootstrap qualification PASS for %s at %s\n' "$program" "$actual_head"
printf 'evidence: %s\n' "$output_dir"
