#!/bin/sh
# Mutation checks for ENG-REA-003: each deliberate contract regression must fail.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
command -v jq >/dev/null 2>&1 || { echo 'jq required' >&2; exit 1; }
command -v sed >/dev/null 2>&1 || { echo 'sed required' >&2; exit 1; }

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

work=$(mktemp -d "${TMPDIR:-/tmp}/rea-coverage-mutations.XXXXXX")
trap 'rm -rf "$work"' 0
trap 'exit 1' HUP INT TERM

prepare() {
    dir="$work/$1"
    mkdir -p "$dir"
    cp "$ROOT/cases.json" "$ROOT/oracle.json" "$ROOT/verify.sh" "$dir/"
    printf '%s\n' "$dir"
}

expect_rejection() {
    label=$1
    dir=$2
    if (cd "$dir" && sh ./verify.sh) >"$work/$label.stdout" 2>"$work/$label.stderr"; then
        fail "$label mutation unexpectedly passed"
    fi
    printf 'PASS mutation rejected: %s\n' "$label"
}

# Mutation 1: remove the required gap from a report that claims coverage
# is partial. The fixed oracle still requires this report to be consistent.
dir=$(prepare missing-gap)
jq '(.cases[] | select(.id == "RCG-005").extraction_gaps) = []' \
    "$dir/cases.json" >"$dir/cases.new.json" || fail "cannot mutate missing-gap case"
mv "$dir/cases.new.json" "$dir/cases.json"
expect_rejection missing-gap "$dir"

# Mutation 2: duplicate an ID, which must break exact manifest/oracle identity.
dir=$(prepare duplicate-id)
jq '(.cases[1].id) = .cases[0].id' \
    "$dir/cases.json" >"$dir/cases.new.json" || fail "cannot mutate duplicate ID"
mv "$dir/cases.new.json" "$dir/cases.json"
expect_rejection duplicate-id "$dir"

# Mutation 3: deliberately remove both integer-range guards. The fractional
# RCG-016 fixture must then be misclassified, causing the oracle gate to fail.
dir=$(prepare fractional-range)
sed -e '/(.range.offset | floor) != .range.offset or/d' \
    -e '/(.range.length | floor) != .range.length or/d' \
    "$dir/verify.sh" >"$dir/verify.new.sh" || fail "cannot mutate range validator"
mv "$dir/verify.new.sh" "$dir/verify.sh"
expect_rejection fractional-range "$dir"

printf 'PASS: all three expected-failure mutation controls\n'
printf 'NOTE: mutation testing of the synthetic contract only; no REA/provider or hardware executed.\n'
