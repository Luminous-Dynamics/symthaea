#!/usr/bin/env bash
# CLS Field-Count Ratchet
#
# CognitiveLoopService ("CLS") regressed from ~59 fields (post-refactor) back up
# to 131+ once already because nothing enforced a ceiling. This script counts
# the struct's current top-level field declarations and fails if that count
# exceeds the checked-in maximum below — the same mechanism a snapshot test
# uses, just without needing a compile. Bump MAX_CLS_FIELDS in this file (in
# the same PR that adds the field) if a new field is genuinely justified; the
# rule per SYMTHAEA_IMPROVEMENT_PLAN_2026-07.md Phase 3 is "no new field on the
# service struct, ever — new state goes in a manager," so treat a bump as the
# exception, not the default response to a failure here.
#
# Usage: scripts/check-cls-field-count.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCT_FILE="${SCRIPT_DIR}/../src/cognitive_loop/mod.rs"

# Verified 2026-07-05 baseline. See header comment before bumping.
MAX_CLS_FIELDS=89

if [[ ! -f "${STRUCT_FILE}" ]]; then
    echo "check-cls-field-count: struct file not found: ${STRUCT_FILE}" >&2
    exit 1
fi

# Extract the CognitiveLoopService struct body via brace depth tracking (Rust
# field declarations never introduce their own braces, so a simple depth
# counter keyed on '{'/'}' correctly isolates the struct body from everything
# after it).
struct_body="$(awk '
    /^pub struct CognitiveLoopService/ { in_struct = 1 }
    in_struct {
        print
        n = length($0)
        for (i = 1; i <= n; i++) {
            c = substr($0, i, 1)
            if (c == "{") depth++
            else if (c == "}") {
                depth--
                if (depth == 0) { exit }
            }
        }
    }
' "${STRUCT_FILE}")"

if [[ -z "${struct_body}" ]]; then
    echo "check-cls-field-count: could not locate 'pub struct CognitiveLoopService' in ${STRUCT_FILE}" >&2
    exit 1
fi

field_count="$(grep -cE '^\s+(pub\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*' <<< "${struct_body}")"

echo "CognitiveLoopService field count: ${field_count} (max: ${MAX_CLS_FIELDS})"

if (( field_count > MAX_CLS_FIELDS )); then
    echo "" >&2
    echo "FAIL: CognitiveLoopService grew from ${MAX_CLS_FIELDS} to ${field_count} fields." >&2
    echo "New state belongs in a manager (cognitive_loop/managers/), not a new field" >&2
    echo "on the service struct itself. If this field is genuinely justified, bump" >&2
    echo "MAX_CLS_FIELDS in scripts/check-cls-field-count.sh in this same PR, with a" >&2
    echo "justification in the commit message." >&2
    exit 1
fi

echo "OK: within the ${MAX_CLS_FIELDS}-field ceiling."
