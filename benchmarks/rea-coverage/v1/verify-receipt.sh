#!/bin/sh
# Verify the exact tracked bytes named by the ENG-REA-003 receipt.
# The receipt does not hash itself; it hashes every listed corpus input and verifier.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RECEIPT="$ROOT/receipt.json"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

for command_name in jq sha256sum git; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command unavailable: $command_name"
done
[ -f "$RECEIPT" ] || fail "missing receipt.json"
jq -e '.schema_version == "symthaea.rea-coverage-receipt.v1" and
       .benchmark_id == "ENG-REA-003" and
       .run_status == "pass" and (.files | type == "object")' "$RECEIPT" >/dev/null ||
    fail "invalid receipt identity or status"

while IFS="$(printf '\t')" read -r name expected_git expected_sha; do
    [ -n "$name" ] || fail "receipt contains an empty file name"
    [ -n "$expected_git" ] || fail "$name has no expected Git blob SHA-1"
    file="$ROOT/$name"
    [ -f "$file" ] || fail "receipt file missing: $name"

    actual_git=$(git hash-object "$file") || fail "cannot hash Git blob: $name"
    [ "$actual_git" = "$expected_git" ] ||
        fail "$name Git blob mismatch: expected $expected_git, got $actual_git"

    if [ -n "$expected_sha" ]; then
        actual_sha=$(sha256sum "$file" | awk '{print $1}')
        [ "$actual_sha" = "$expected_sha" ] ||
            fail "$name SHA-256 mismatch: expected $expected_sha, got $actual_sha"
    fi
    printf 'PASS receipt identity: %s\n' "$name"
done <<EOF_FILES
$(jq -er '.files | to_entries[] | [.key, .value.git_blob_sha1, (.value.sha256 // "")] | @tsv' "$RECEIPT")
EOF_FILES

printf 'PASS: all %s receipt file identities match\n' "$(jq '.files | length' "$RECEIPT")"
printf 'NOTE: receipt is not self-hashed; real REA/Unblob provider and hardware remain untested.\n'
