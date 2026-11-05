#!/usr/bin/env bash
set -euo pipefail

script="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)/check_workspace_members.sh"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

mkdir -p \
    "$fixture/xtask" \
    "$fixture/crates/core/symthaea/examples" \
    "$fixture/crates/core/valid-core/src" \
    "$fixture/crates/domains/valid-domain/src" \
    "$fixture/crates/bridges/valid-bridge/src"

cat > "$fixture/Cargo.toml" <<'TOML'
[workspace]
members = [
    "xtask",
    "crates/core/*",
    "crates/domains/*",
    "crates/bridges/*",
]
exclude = [
    # examples-only directory must be explicitly excluded
    "crates/core/symthaea",
]
TOML

for package in \
    "$fixture/xtask" \
    "$fixture/crates/core/valid-core" \
    "$fixture/crates/domains/valid-domain" \
    "$fixture/crates/bridges/valid-bridge"; do
    printf '[package]\nname = "fixture-package"\nversion = "0.1.0"\n' > "$package/Cargo.toml"
done

output="$("$script" "$fixture")"
grep -q '^EXCLUDED crates/core/symthaea$' <<< "$output"
grep -q '^MEMBER crates/core/valid-core$' <<< "$output"
grep -q '^MEMBER crates/domains/valid-domain$' <<< "$output"
grep -q '^MEMBER crates/bridges/valid-bridge$' <<< "$output"

mkdir -p "$fixture/crates/domains/missing-manifest"
if "$script" "$fixture" >"$fixture/stdout" 2>"$fixture/stderr"; then
    printf 'workspace preflight test failed: missing manifest was accepted\n' >&2
    exit 1
fi
grep -q 'MISSING_MANIFEST crates/domains/missing-manifest' "$fixture/stderr"

printf 'workspace preflight tests passed\n'
