#!/usr/bin/env bash
# Fail-closed preflight for every directory selected by root Cargo workspace
# member globs. Exclusions are read from Cargo.toml, not duplicated here.
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
root="$(cd "$root" && pwd)"
manifest="$root/Cargo.toml"

if [[ ! -f "$manifest" ]]; then
    printf 'workspace preflight: missing root manifest: %s\n' "$manifest" >&2
    exit 2
fi

array_values() {
    local key="$1"
    awk -v key="$key" '
        $0 ~ ("^" key " = \\[") { in_array = 1 }
        in_array {
            print
            line = $0
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line == "]") exit
        }
    ' "$manifest" |
        sed 's/#.*$//' |
        grep -oE '"[^"]+"' |
        sed 's/^"//; s/"$//'
}

mapfile -t members < <(array_values members)
mapfile -t exclusions < <(array_values exclude)

if [[ ${#members[@]} -eq 0 ]]; then
    printf 'workspace preflight: no root workspace members found\n' >&2
    exit 2
fi

matched=0
eligible=0
excluded_count=0
failed=0

for pattern in "${members[@]}"; do
    shopt -s nullglob
    candidates=( "$root"/$pattern )
    shopt -u nullglob

    if [[ "$pattern" != *'*'* && "$pattern" != *'?'* && "$pattern" != *'['* && ${#candidates[@]} -eq 0 ]]; then
        printf 'workspace preflight: literal member path is missing: %s\n' "$pattern" >&2
        failed=1
        continue
    fi

    for candidate in "${candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        relative="${candidate#"$root"/}"
        matched=$((matched + 1))
        is_excluded=0

        for exclusion in "${exclusions[@]}"; do
            if [[ "$relative" == $exclusion ]]; then
                is_excluded=1
                break
            fi
        done

        if [[ "$is_excluded" -eq 1 ]]; then
            excluded_count=$((excluded_count + 1))
            printf 'EXCLUDED %s\n' "$relative"
            continue
        fi

        if [[ ! -f "$candidate/Cargo.toml" ]]; then
            printf 'MISSING_MANIFEST %s (matched by %s)\n' "$relative" "$pattern" >&2
            failed=1
            continue
        fi

        eligible=$((eligible + 1))
        printf 'MEMBER %s\n' "$relative"
    done
done

if [[ "$matched" -eq 0 ]]; then
    printf 'workspace preflight: no directories matched root workspace members\n' >&2
    exit 2
fi

printf 'workspace preflight: matched=%d members=%d excluded=%d\n' \
    "$matched" "$eligible" "$excluded_count"

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
