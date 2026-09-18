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
  * materializes Cargo's lock candidate by checking every requested package;
  * requires a non-empty additive-only Cargo.lock patch;
  * runs package rustfmt, tests, and strict all-target Clippy;
  * rechecks source immutability and restores the original Cargo.lock;
  * writes Cargo.lock.generated, Cargo.lock.patch, and receipt.txt.

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

for package in "${packages[@]}"; do
  cargo check -p "$package"
done

cp Cargo.lock "$output_dir/Cargo.lock.generated"
git diff -- Cargo.lock > "$output_dir/Cargo.lock.patch"
if [[ ! -s "$output_dir/Cargo.lock.patch" ]]; then
  echo "expected Cargo to materialize a non-empty bootstrap lock candidate" >&2
  exit 1
fi
if grep -E '^-[^-]' "$output_dir/Cargo.lock.patch"; then
  echo "bootstrap lock patch is not additive-only" >&2
  exit 1
fi
for package in "${packages[@]}"; do
  grep -F "name = \"$package\"" Cargo.lock >/dev/null
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

{
  echo "program=$program"
  echo "authority=bootstrap-source-qualification-only"
  echo "scientific_claim=NONE"
  echo "source_subject_sha=$source_parent"
  echo "qualifier_sha=$actual_head"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "generated_lock_sha256=$lock_sha256"
  echo "lock_patch_sha256=$patch_sha256"
  for package in "${packages[@]}"; do
    echo "package=$package"
  done
  for path in "${source_paths[@]}"; do
    echo "source_path=$path"
    echo "source_object=$path:$(git rev-parse "$source_parent:$path")"
  done
  echo "scope=PASS"
  echo "source_immutable=PASS"
  echo "lock_additive=PASS"
  echo "check=PASS"
  echo "format=PASS"
  echo "tests=PASS"
  echo "clippy=PASS"
  echo "postflight_source_immutable=PASS"
} > "$output_dir/receipt.txt"

# Restore Cargo.lock before returning so local/Nix use does not leave a mutated
# research checkout. Artifacts remain outside the repository when callers use
# the recommended temporary output directory.
restore_lock
trap - EXIT

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "working tree is not clean after lock restoration" >&2
  git status --short >&2
  exit 1
fi

printf 'bootstrap qualification PASS for %s at %s\n' "$program" "$actual_head"
printf 'evidence: %s\n' "$output_dir"
