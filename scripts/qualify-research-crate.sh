#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portable exact-head bootstrap qualification harness for research crates.
#
# This script intentionally owns the qualification theorem logic; CI is only a
# scheduler/environment provider. The same command can run under GitHub Actions,
# Nix, or a local checkout with the pinned toolchain available.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/qualify-research-crate.sh \
    --program SCI-001A \
    --source-parent <sha> \
    --expected-head <sha> \
    --expected-rust 1.96.0 \
    --package symthaea-science-research \
    --source-path crates/core/symthaea-science-research \
    --qualifier-path .github/workflows/sci-001a-science-research-kernel.yml \
    --output-dir /tmp/sci-001a-evidence

Repeat --package, --source-path, and --qualifier-path as needed.

The harness:
  * binds current HEAD to --expected-head and HEAD^ to --source-parent;
  * requires the qualifier diff to contain exactly --qualifier-path entries;
  * proves every --source-path is byte-identical between source parent and HEAD;
  * verifies the requested Rust/Cargo toolchain version;
  * checks every requested package and materializes Cargo's lock candidate when needed;
  * accepts an already-resolved lock or an additive-only Cargo.lock materialization;
  * records the compiler/runner environment and locked dependency graph;
  * reruns locked all-feature/all-target check/test/Clippy plus doctest/rustdoc;
  * rechecks source and lock immutability and restores the original Cargo.lock;
  * writes retained evidence and receipt.txt outside the source tree.

This is bootstrap source qualification only. It never grants scientific claim
or qualification authority to the code being checked.
EOF
}

program=""
source_parent=""
expected_head=""
expected_rust="1.96.0"
output_dir=""
packages=()
source_paths=()
qualifier_paths=()

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

if [[ -z "$program" || -z "$source_parent" || -z "$expected_head" || -z "$output_dir" ]]; then
  echo "--program, --source-parent, --expected-head, and --output-dir are required" >&2
  exit 2
fi
if ((${#packages[@]} == 0 || ${#source_paths[@]} == 0 || ${#qualifier_paths[@]} == 0)); then
  echo "at least one --package, --source-path, and --qualifier-path is required" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_head="$(git rev-parse HEAD)"
actual_parent="$(git rev-parse HEAD^)"
if [[ "$actual_head" != "$expected_head" ]]; then
  echo "head mismatch: expected $expected_head, observed $actual_head" >&2
  exit 1
fi
if [[ "$actual_parent" != "$source_parent" ]]; then
  echo "parent mismatch: expected $source_parent, observed $actual_parent" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "qualification must start from a clean working tree" >&2
  git status --short >&2
  exit 1
fi

mapfile -t actual_scope < <(git diff --name-only "$source_parent" HEAD | LC_ALL=C sort -u)
mapfile -t expected_scope < <(printf '%s\n' "${qualifier_paths[@]}" | LC_ALL=C sort -u)
if ! diff -u \
  <(printf '%s\n' "${expected_scope[@]}") \
  <(printf '%s\n' "${actual_scope[@]}"); then
  echo "qualification diff is not qualifier-only" >&2
  exit 1
fi

for path in "${source_paths[@]}"; do
  git cat-file -e "$source_parent:$path"
  git cat-file -e "HEAD:$path"
  if [[ "$(git rev-parse "$source_parent:$path")" != "$(git rev-parse "HEAD:$path")" ]]; then
    echo "source path changed in qualification commit: $path" >&2
    exit 1
  fi
done

command -v rustc >/dev/null
command -v cargo >/dev/null
command -v rustfmt >/dev/null
command -v sha256sum >/dev/null
rustc --version | grep -F "rustc $expected_rust"
cargo --version | grep -F "cargo $expected_rust"
rustfmt --version
cargo clippy --version

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
tmp_dir="$(mktemp -d)"
lock_before="$tmp_dir/Cargo.lock.before"
cp Cargo.lock "$lock_before"

restore_lock() {
  cp "$lock_before" Cargo.lock
  rm -rf "$tmp_dir"
}
trap restore_lock EXIT

{
  echo "uname=$(uname -srm)"
  echo "runner_os=${RUNNER_OS:-unknown}"
  echo "runner_arch=${RUNNER_ARCH:-unknown}"
  echo "image_os=${ImageOS:-unknown}"
  echo "image_version=${ImageVersion:-unknown}"
  echo "rustc_verbose_begin"
  rustc -Vv
  echo "rustc_verbose_end"
  echo "cargo=$(cargo --version)"
  echo "rustfmt=$(rustfmt --version)"
  echo "clippy=$(cargo clippy --version)"
} > "$output_dir/environment.txt"

# First pass may materialize missing lock entries for the complete feature/target surface.
for package in "${packages[@]}"; do
  cargo check -p "$package" --all-targets --all-features
done

cp Cargo.lock "$output_dir/Cargo.lock.generated"
git diff -- Cargo.lock > "$output_dir/Cargo.lock.patch"
lock_state="already-resolved"
if [[ -s "$output_dir/Cargo.lock.patch" ]]; then
  lock_state="additive-materialized"
  if grep -E '^-[^-]' "$output_dir/Cargo.lock.patch"; then
    echo "bootstrap lock patch contains deletions/replacements; only additive materialization is allowed" >&2
    exit 1
  fi
fi
for package in "${packages[@]}"; do
  grep -F "name = \"$package\"" Cargo.lock >/dev/null
done

# Bind the exact resolved graph used by the locked verification phase.
cargo metadata --locked --format-version 1 > "$output_dir/Cargo.metadata.json"
: > "$output_dir/dependency-tree.txt"
for package in "${packages[@]}"; do
  printf 'package=%s\n' "$package" >> "$output_dir/dependency-tree.txt"
  cargo tree --locked -p "$package" --edges normal,build,dev --prefix none --charset ascii \
    >> "$output_dir/dependency-tree.txt"
done

# From this point onward the candidate lock must be sufficient and immutable.
for package in "${packages[@]}"; do
  cargo check --locked -p "$package" --all-targets --all-features
done

fmt_args=()
for package in "${packages[@]}"; do
  fmt_args+=("-p" "$package")
done
cargo fmt --check "${fmt_args[@]}"

for package in "${packages[@]}"; do
  cargo test --locked -p "$package" --all-targets --all-features
  cargo test --locked -p "$package" --doc --all-features
  cargo clippy --locked -p "$package" --all-targets --all-features -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc --locked -p "$package" --all-features --no-deps
done

# Prove the locked verification phase did not mutate the candidate lock.
if ! cmp -s Cargo.lock "$output_dir/Cargo.lock.generated"; then
  echo "Cargo.lock changed after locked verification began" >&2
  exit 1
fi

for path in "${source_paths[@]}"; do
  if [[ "$(git rev-parse "$source_parent:$path")" != "$(git rev-parse "HEAD:$path")" ]]; then
    echo "source path changed during qualification: $path" >&2
    exit 1
  fi
done

unexpected="$({ git status --porcelain=v1 || true; } | grep -v '^ M Cargo.lock$' || true)"
if [[ -n "$unexpected" ]]; then
  echo "unexpected working-tree mutation during qualification:" >&2
  printf '%s\n' "$unexpected" >&2
  exit 1
fi

lock_sha256="$(sha256sum "$output_dir/Cargo.lock.generated" | cut -d' ' -f1)"
patch_sha256="$(sha256sum "$output_dir/Cargo.lock.patch" | cut -d' ' -f1)"
metadata_sha256="$(sha256sum "$output_dir/Cargo.metadata.json" | cut -d' ' -f1)"
dependency_tree_sha256="$(sha256sum "$output_dir/dependency-tree.txt" | cut -d' ' -f1)"
environment_sha256="$(sha256sum "$output_dir/environment.txt" | cut -d' ' -f1)"

{
  echo "program=$program"
  echo "authority=bootstrap-source-qualification-only"
  echo "scientific_claim=NONE"
  echo "source_subject_sha=$source_parent"
  echo "qualifier_sha=$actual_head"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "lock_state=$lock_state"
  echo "generated_lock_sha256=$lock_sha256"
  echo "lock_patch_sha256=$patch_sha256"
  echo "cargo_metadata_sha256=$metadata_sha256"
  echo "dependency_tree_sha256=$dependency_tree_sha256"
  echo "environment_sha256=$environment_sha256"
  for package in "${packages[@]}"; do
    echo "package=$package"
  done
  for path in "${source_paths[@]}"; do
    echo "source_path=$path"
    echo "source_object=$path:$(git rev-parse "$source_parent:$path")"
  done
  echo "scope=PASS"
  echo "source_immutable=PASS"
  echo "lock_resolved_or_additive=PASS"
  echo "resolved_graph_bound=PASS"
  echo "locked_all_features_all_targets_check=PASS"
  echo "format=PASS"
  echo "locked_all_features_all_targets_tests=PASS"
  echo "locked_doctests=PASS"
  echo "locked_all_features_all_targets_clippy=PASS"
  echo "locked_rustdoc=PASS"
  echo "postflight_lock_immutable=PASS"
  echo "postflight_source_immutable=PASS"
} > "$output_dir/receipt.txt"

restore_lock
trap - EXIT

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "working tree is not clean after lock restoration" >&2
  git status --short >&2
  exit 1
fi

printf 'bootstrap qualification PASS for %s at %s\n' "$program" "$actual_head"
printf 'evidence: %s\n' "$output_dir"
