#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

command -v cargo >/dev/null || {
  echo "cargo is required" >&2
  exit 1
}
command -v rustc >/dev/null || {
  echo "rustc is required" >&2
  exit 1
}

expected_msrv="$(sed -n 's/^rust-version = "\([^"]*\)"/\1/p' Cargo.toml)"
actual_release="$(rustc --version | awk '{print $2}')"
printf 'MSRV declared: %s; rustc selected: %s\n' "$expected_msrv" "$actual_release"

cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo metadata --no-deps --format-version 1 >/dev/null

if cargo tree --depth 1 | tail -n +2 | grep -q .; then
  echo "runtime or development dependencies detected; review zero-dependency policy" >&2
  cargo tree --depth 1 >&2
  exit 1
fi

if grep -RIn --include='*.rs' -E 'unsafe[[:space:]]*\{|#!\[allow\(unsafe_code\)\]' src tests; then
  echo "unsafe Rust marker detected" >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
fi

echo "symthaea-legal-reasoning release verification passed"
