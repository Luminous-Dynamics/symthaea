#!/bin/sh
# Ensure the known-answer suite kills representative unsafe normalizer mutations.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v sed >/dev/null 2>&1 || fail "sed required"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

run_mutant(){
  label=$1
  mutator=$2
  dir="$TMP/$label"
  mkdir -p "$dir"
  cp "$ROOT/cases.json" "$ROOT/oracle.json" "$ROOT/verify.sh" "$ROOT/normalize.jq" "$dir/"
  case "$mutator" in
    drop-hash)
      sed -i 's/^    | normalize_paths(\$root)$/    | (if .__typename__=="HashReport" then del(.sha256) else . end)\
    | normalize_paths($root)/' "$dir/normalize.jq" ;;
    retain-provider-ids)
      sed -i 's/^      then del(\.id) else \. end)$/      then . else . end)/' "$dir/normalize.jq" ;;
    retain-root-paths)
      sed -i 's/^    | normalize_paths(\$root)$/    | ./' "$dir/normalize.jq" ;;
    drop-handler)
      sed -i 's/^    | normalize_paths(\$root)$/    | del(.handler_name)\
    | normalize_paths($root)/' "$dir/normalize.jq" ;;
    *) fail "unknown mutant: $mutator" ;;
  esac
  if sh "$dir/verify.sh" > "$dir/output.log" 2>&1; then
    cat "$dir/output.log" >&2
    fail "mutant survived: $label"
  fi
  grep -q '^FAIL:' "$dir/output.log" || { cat "$dir/output.log" >&2; fail "$label failed for an unexpected reason"; }
  printf 'PASS mutation killed: %s\n' "$label"
}

run_mutant drop-sha256 drop-hash
run_mutant keep-provider-generated-IDs retain-provider-ids
run_mutant keep-absolute-extraction-roots retain-root-paths
run_mutant erase-handler-semantics drop-handler
printf '\nPASS ENG-REA-005 mutation controls: 4/4 weakened normalizers were detected.\n'
