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

# ---------------------------------------------------------------------------
# Check 2 (added 2026-07-30): every DECLARED target resolves to a TRACKED file.
#
# Check 1 above catches "member crate with no build target" -- the break that took
# the workspace down twice in 2026-07 (organic-chemistry). It does NOT catch the
# inverse: a [[example]]/[[test]]/[[bin]]/[[bench]] declared in a manifest whose
# source file does not exist. That break is worse, because cargo then fails to
# PARSE the manifest, so every cargo command in the workspace dies -- not just the
# one target.
#
# It happened twice on 2026-07-30 alone:
#   df4d6c7f56  committed one half of a cross-crate addition
#   3c2353af54  committed `[[example]] name = "recurrent_mask_characterization"`
#               while examples/recurrent_mask_characterization.rs stayed UNTRACKED
#
# WHY THIS TESTS TRACKED-NESS AND NOT MERE EXISTENCE
#
# In the tree where the author works, the untracked file is sitting right there, so
# an -f existence test PASSES. The break only appears in a fresh clone, a worktree,
# or CI. Testing `git ls-files` reproduces what everyone else will get, which is the
# whole point of the check. An existence check would have been green on the machine
# that introduced the defect.
declared_fail=0
check_declared_targets() {
  local manifest="$1" crate_dir; crate_dir="$(dirname "$manifest")"
  # Emit "kind<TAB>name<TAB>path" for each declared target ("-" when path is absent).
  awk '
    /^\[\[(example|test|bin|bench)\]\]/ {
      if (kind != "") { print kind "\t" name "\t" (path == "" ? "-" : path) }
      kind = $0; gsub(/[^a-z]/, "", kind); name = ""; path = ""; next
    }
    /^\[/ && !/^\[\[(example|test|bin|bench)\]\]/ {
      if (kind != "") { print kind "\t" name "\t" (path == "" ? "-" : path) }
      kind = ""; name = ""; path = ""; next
    }
    kind != "" && /^[[:space:]]*name[[:space:]]*=/ { v=$0; sub(/^[^=]*=[[:space:]]*/, "", v); gsub(/["'"'"']/, "", v); sub(/[[:space:]]*(#.*)?$/, "", v); name = v }
    kind != "" && /^[[:space:]]*path[[:space:]]*=/ { v=$0; sub(/^[^=]*=[[:space:]]*/, "", v); gsub(/["'"'"']/, "", v); sub(/[[:space:]]*(#.*)?$/, "", v); path = v }
    END { if (kind != "") print kind "\t" name "\t" (path == "" ? "-" : path) }
  ' "$manifest" | while IFS=$'\t' read -r kind name path; do
    [[ -n "$name" ]] || continue
    local candidates=()
    if [[ "$path" != "-" ]]; then
      candidates=("$crate_dir/$path")
    else
      local subdir
      case "$kind" in
        example) subdir="examples" ;;
        test)    subdir="tests" ;;
        bench)   subdir="benches" ;;
        bin)     subdir="src/bin" ;;
      esac
      candidates=("$crate_dir/$subdir/$name.rs" "$crate_dir/$subdir/$name/main.rs")
    fi
    local found=""
    for c in "${candidates[@]}"; do
      # `git ls-files --error-unmatch` succeeds only for a TRACKED path.
      if git ls-files --error-unmatch "$c" >/dev/null 2>&1; then found="$c"; break; fi
    done
    if [[ -z "$found" ]]; then
      local hint="not tracked by git"
      for c in "${candidates[@]}"; do
        [[ -f "$c" ]] && hint="EXISTS ON DISK BUT IS UNTRACKED -- run: git add $c"
      done
      echo "ORPHANED TARGET: [[$kind]] name = \"$name\" in $manifest"
      echo "                 expected: ${candidates[*]}"
      echo "                 $hint"
      echo "orphan" >> "$ORPHAN_FLAG"
    fi
  done
}

ORPHAN_FLAG="$(mktemp)"
trap 'rm -f "$ORPHAN_FLAG"' EXIT
check_declared_targets "Cargo.toml"
for glob in "${MEMBER_GLOBS[@]}"; do
  for dir in $glob; do
    [[ -f "$dir/Cargo.toml" ]] || continue
    is_excluded "$dir" && continue
    check_declared_targets "$dir/Cargo.toml"
  done
done
[[ -s "$ORPHAN_FLAG" ]] && declared_fail=1

if [[ $fail -ne 0 ]]; then
  echo ""
  echo "The member(s) above break 'cargo' for the ENTIRE workspace. Fix by adding a"
  echo "src/lib.rs (a stub 'pub fn placeholder() {}' is enough) or excluding the crate"
  echo "in Cargo.toml's [workspace].exclude before committing."
fi
if [[ $declared_fail -ne 0 ]]; then
  echo ""
  echo "The declared target(s) above make cargo fail to PARSE the manifest, which"
  echo "breaks EVERY cargo command in the workspace -- for everyone except the author,"
  echo "whose working tree still has the untracked file. Stage the file, or remove the"
  echo "declaration, before committing."
fi
(( fail != 0 || declared_fail != 0 )) && exit 1
echo "OK: all workspace members have a build target, and all declared targets are tracked."
