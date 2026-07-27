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

# Bumped 2026-07-26 from the 2026-07-05 baseline of 89, after auditing all 3
# new fields individually against the "no scattered state" rule above (not a
# rubber-stamp bump):
#   - train_history_snapshots: VecDeque<TemporalStateBackup> -- the 2026-07-17
#     sequence-prediction fix (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md
#     Phase 5) that made the loop's temporal learning actually work (all 3
#     pre-registered acceptance gates passed for the first time). Read/written
#     every cycle from cycle_phase_dynamics/{training,planning}.rs -- core
#     hot-path state, not a bolt-on.
#   - training_frozen: bool -- a single kill-switch bool gating one existing
#     condition in cycle_phase_dynamics/training.rs, set only by the
#     Predictive Compression C1 experiment harness (examples/compression_bits.rs).
#     Minimal footprint; not scattered scalars.
#   - memetic_immune: symthaea_memetics::MemeticImmuneSystem -- a whole
#     encapsulated subsystem (screens incoming memes, tracks contagion,
#     psi-gated guardian posture; MEMETICS_ANTIMEMETICS_PLAN.md) with its own
#     accessor module (accessors/memetics.rs), structurally identical to how
#     other subsystem-holding fields already live on this struct -- not raw
#     loose state that belongs in a new manager.
# None of the three are unprincipled sprawl; each traces to a specific,
# already-landed, previously-verified initiative. Re-derive this list, don't
# just bump the number, next time this fails.
MAX_CLS_FIELDS=92

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
