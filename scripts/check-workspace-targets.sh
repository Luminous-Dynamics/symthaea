#!/usr/bin/env bash
# check-workspace-targets.sh — fail if any workspace member lacks a build target.
#
# WHY: a member Cargo.toml with no `src/lib.rs`, `src/main.rs`, `src/bin/*.rs`,
# `[lib]`, or `[[bin]]` makes `cargo` refuse to load the ENTIRE workspace
# ("error: no targets specified in the manifest ..."), which breaks every cargo
# command for every concurrent session — not just the offending crate. This
# happened twice on 2026-07-06 (`symthaea-organic-chemistry`, and separately a
# half-scaffolded crate) and cost real time to diagnose from cargo's opaque error.
#
# USE:
#   scripts/check-workspace-targets.sh        # CI gate + pre-commit check
#   scripts/check-workspace-targets.sh        # ALSO the fast diagnostic: when
#     `cargo` suddenly fails workspace-wide, run this to name the broken crate.
#
# No build required — pure filesystem/Cargo.toml scan, so it works even when the
# workspace is already broken (cargo metadata can't, because it fails on the break).
set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."  # -> symthaea/

# Member globs — must mirror [workspace].members in Cargo.toml (immediate children).
MEMBER_GLOBS=(crates/core/* crates/domains/* crates/bridges/*)

# Excluded members — must mirror [workspace].exclude in Cargo.toml.
EXCLUDES=(
  "crates/core/symthaea-zkproof/methods"
  "crates/core/symthaea-zkproof/host"
  "crates/domains/symthaea-spore/fuzz"
  "crates/domains/spark-engine"
  "crates/domains/symthaea-lab"
)

is_excluded() {
  local d="$1" e
  for e in "${EXCLUDES[@]}"; do [[ "$d" == "$e" ]] && return 0; done
  return 1
}

# A crate has a build target if any of these hold.
has_target() {
  local d="$1"
  [[ -f "$d/src/lib.rs" || -f "$d/src/main.rs" ]] && return 0
  local b; for b in "$d"/src/bin/*.rs; do [[ -e "$b" ]] && return 0; done
  grep -qE '^\[\[bin\]\]|^\[lib\]' "$d/Cargo.toml" && return 0
  return 1
}

fail=0
for glob in "${MEMBER_GLOBS[@]}"; do
  for dir in $glob; do
    [[ -f "$dir/Cargo.toml" ]] || continue
    is_excluded "$dir" && continue
    if ! has_target "$dir"; then
      echo "MISSING TARGET: $dir  (Cargo.toml but no src/lib.rs, src/main.rs, src/bin/*.rs, [lib], or [[bin]])"
      fail=1
    fi
  done
done

if [[ $fail -ne 0 ]]; then
  echo ""
  echo "The member(s) above break 'cargo' for the ENTIRE workspace. Fix by adding a"
  echo "src/lib.rs (a stub 'pub fn placeholder() {}' is enough) or excluding the crate"
  echo "in Cargo.toml's [workspace].exclude before committing."
  exit 1
fi
echo "OK: all workspace members have a build target."
