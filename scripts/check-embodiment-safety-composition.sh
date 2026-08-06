#!/usr/bin/env bash
# check-embodiment-safety-composition.sh — fail if an EmbodimentBridge platform
# gates motor authority on Phi WITHOUT also admitting a safety override and a
# moral verdict.
#
# WHY: `EmbodimentBridge::step(..., phi)` hands every platform a consciousness
# scalar, and `MotorSafetyLevel::from_phi` turns it into Green/Yellow/Orange/Red,
# which physically caps torque, grip force, and flight envelopes. But Phi is only
# ONE of the three inputs a platform is supposed to honour. The other two are:
#
#   * `safety_override`  — how an external SafetyAgent forces a lower tier
#   * `moral_safety`     — how an ahimsa/consent verdict forces a lower tier
#
# The fleet convention is `max(phi_level, safety_override, moral_safety)` on the
# derived `Ord`, so any one of the three can only ever make the tier MORE
# restrictive. A platform that reads Phi but never composes the other two has no
# route from a SafetyAgent override or an ethics verdict to its actuators — the
# gate exists and is simply unreachable.
#
# A 2026-07-29 audit (SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md,
# "all-platform tier-semantics uniformity") found exactly one such platform:
# `symthaea-gravcraft`, whose module doc additionally claimed to be an
# "EmbodimentBridge implementation" while implementing no such trait. It was
# harmless only because nothing depends on it yet. This check exists so the NEXT
# one is caught mechanically rather than by someone happening to run an audit.
#
# ── WHAT THIS IS NOT ────────────────────────────────────────────────────────
# This is a SOURCE LINT, not a proof. It greps for the composition idiom; it does
# not execute anything, so it verifies that the code *mentions* the right inputs,
# not that it *behaves* correctly with them. A platform could satisfy this script
# and still compose the values wrongly.
#
# It strips comments before grepping, and that is not a detail. The FIRST version of
# this script did not, and `symthaea-gravcraft` -- the very platform that motivated
# it -- passed, because the doc comment documenting the absence of a moral gate
# mentions the words "safety_override" and "moral_safety". A text lint is defeated by
# prose ABOUT the thing it looks for, including prose saying the thing is missing.
# That is a fair summary of how much a grep is worth as evidence.
#
# It is a lint rather than a Rust test for a concrete reason: each platform sits
# behind its own Cargo feature (`humanoid`, `vehicle`, `surgical`, ...) and there
# is no feature configuration in which all of them compile together, so no single
# `#[test]` can construct the fleet and assert the property behaviourally. If that
# ever changes, replace this script with a real test — a test would be strictly
# better evidence.
#
# USE:
#   scripts/check-embodiment-safety-composition.sh     # CI gate + local check
set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."  # -> symthaea/

fail=0
checked=0

# Documented exceptions. An entry here must say WHY, and is reviewed whenever this
# check changes. Silent exclusion would defeat the purpose.
is_exempt() {
    case "$1" in
        # Not a robot: no actuators, so "a verdict cannot reach the actuators" does
        # not apply. Gates browser ACTIONS via a separate per-action `required_phi`
        # ladder rather than a motor tier; its MotorSafetyLevel use is telemetry.
        # Also not an EmbodimentBridge impl (duck-typed lookalike).
        symthaea-browser) return 0 ;;
        *) return 1 ;;
    esac
}

for f in crates/domains/*/src/embodiment.rs; do
    crate="$(basename "$(dirname "$(dirname "$f")")")"
    is_exempt "$crate" && continue

    # Strip whole-line comments before matching -- see the "WHAT THIS IS NOT" note.
    # Without this, a doc comment merely NAMING these fields makes a platform pass.
    code="$(sed -E 's://!.*::; s:^[[:space:]]*//.*::' "$f")"

    # Only platforms that actually derive a tier from Phi are in scope. A file
    # that never calls from_phi() isn't gating motor authority on consciousness
    # and has nothing to compose.
    grep -q "from_phi" <<<"$code" || continue
    checked=$((checked + 1))

    missing=()
    grep -q "safety_override" <<<"$code" || missing+=("safety_override")
    # Accept either the composed field or a call to the trait's default helper.
    grep -qE "moral_safety|apply_moral_gate" <<<"$code" || missing+=("moral_safety/apply_moral_gate")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "FAIL: $crate ($f)"
        echo "      derives a safety tier from Phi but never references: ${missing[*]}"
        echo "      => a SafetyAgent override or an ethics verdict has no route to this"
        echo "         platform's actuators. Compose it:"
        echo "         max(phi_level, safety_override, moral_safety)"
        fail=1
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "FAIL: no embodiment.rs files matched — the glob or layout changed."
    echo "      This check silently passing on zero files would be worse than failing."
    exit 1
fi

if [ "$fail" -eq 0 ]; then
    echo "OK: all $checked Phi-gating embodiment platforms compose safety_override + moral gate."
else
    echo
    echo "See SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md,"
    echo "'all-platform tier-semantics uniformity', for the audit this encodes."
fi

exit "$fail"
