#!/usr/bin/env bash
set -euo pipefail

SOURCE_HEAD="27dd014a191a6055f2d60b0641a5b7954daab615"
CHECKED_BLOB="aadeac20400fcc789d23fbcc91f012549cb0a886"
LIB_BLOB="85e960207b3c4268c596df1f38dd403c87688f54"
CHECKED_PATH="crates/domains/symthaea-structural/src/checked.rs"
LIB_PATH="crates/domains/symthaea-structural/src/lib.rs"

actual_parent="$(git rev-parse HEAD^)"
if [[ "$actual_parent" != "$SOURCE_HEAD" ]]; then
  echo "qualifier parent mismatch: $actual_parent" >&2
  exit 1
fi

if [[ "$(git rev-list --count "$SOURCE_HEAD"..HEAD)" != "1" ]]; then
  echo "qualifier must be exactly one commit" >&2
  exit 1
fi

actual_changed="$(git diff --name-only "$SOURCE_HEAD"..HEAD | LC_ALL=C sort)"
expected_changed="$(printf '%s\n' '.github/workflows/struct-ref-001a-checked.yml' 'scripts/qualify_struct_ref_001a_checked.sh' | LC_ALL=C sort)"
if [[ "$actual_changed" != "$expected_changed" ]]; then
  echo "unexpected qualifier delta" >&2
  printf '%s\n' "$actual_changed" >&2
  exit 1
fi

[[ "$(git rev-parse "$SOURCE_HEAD:$CHECKED_PATH")" == "$CHECKED_BLOB" ]]
[[ "$(git rev-parse "$SOURCE_HEAD:$LIB_PATH")" == "$LIB_BLOB" ]]

echo "== cargo check =="
cargo check -p symthaea-structural

echo "== checked facade tests =="
cargo test -p symthaea-structural --lib checked::tests

echo "== full structural crate regression =="
cargo test -p symthaea-structural

echo "== clean postflight =="
git diff --exit-code
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  exit 1
fi

echo "PASS_STRUCT_REF_001A_CHECKED source=$SOURCE_HEAD"
