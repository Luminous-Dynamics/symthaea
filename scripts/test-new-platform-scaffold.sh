#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Textual contract test for scripts/new-platform.sh.
#
# This deliberately runs in a temporary empty directory: it verifies that the
# generator's own outputs/instructions stay conservative and current without
# mutating the repository. Full generated-crate compilation remains a separate
# integration gate because a new first-party platform also requires an explicit
# EmbodimentPlatform variant and root feature/dependency registration.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/new-platform.sh"

bash -n "$GENERATOR"

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$tmp_dir/scripts"
cp "$GENERATOR" "$tmp_dir/scripts/new-platform.sh"
chmod +x "$tmp_dir/scripts/new-platform.sh"

pushd "$tmp_dir" >/dev/null
output="$(./scripts/new-platform.sh test-rover 3 7)"
crate="crates/domains/symthaea-test-rover"

# Current workspace/domain layout and lean module set.
test -f "$crate/Cargo.toml"
test -f "$crate/src/types.rs"
test -f "$crate/src/encoder.rs"
test -f "$crate/src/controller.rs"
test -f "$crate/src/simulator.rs"
test -f "$crate/src/embodiment.rs"
test -f "$crate/src/plugin.rs"
test -f "$crate/src/lib.rs"
test ! -e "crates/symthaea-test-rover"

grep -Fq 'symthaea-core = { path = "../../core/symthaea-core" }' "$crate/Cargo.toml"
grep -Fq 'edition.workspace = true' "$crate/Cargo.toml"
grep -Fq 'pub mod plugin;' "$crate/src/lib.rs"
grep -Fq 'impl PlatformPlugin for TestRoverPlugin' "$crate/src/plugin.rs"
grep -Fq 'EmbodimentPlatform::TestRover' "$crate/src/plugin.rs"

# Generated control starts non-actuating rather than random/unqualified.
grep -Fq 'TestRoverCommand::zero()' "$crate/src/controller.rs"
! grep -Fq 'HdcLtcUnifiedNetwork' "$crate/src/controller.rs"

# #1281: temporal novelty is not emitted as prediction error/confidence.
grep -Fq 'last_temporal_novelty' "$crate/src/embodiment.rs"
grep -Fq 'prediction_error: LEGACY_NO_PREDICTION_SENTINEL' "$crate/src/embodiment.rs"
grep -Fq 'observation_confidence: LEGACY_NO_OBSERVATION_CONFIDENCE' "$crate/src/embodiment.rs"
! grep -Fq 'grounding_from_prediction_error' "$crate/src/embodiment.rs"

# Typed safety telemetry and restrictive pre-step/reset state.
grep -Fq 'safety_level: self.current_safety' "$crate/src/embodiment.rs"
grep -Fq 'current_safety: MotorSafetyLevel::Red' "$crate/src/embodiment.rs"

# The current plugin registry workflow replaces the pre-2026 constructor match arm.
grep -Fq 'src/cognitive_loop/platform_registry.rs' <<<"$output"
! grep -Fq 'constructor.rs' <<<"$output"
grep -Fq 'No workspace-members edit is needed' <<<"$output"

# Invalid inputs fail without leaving a generated body behind.
if ./scripts/new-platform.sh Bad_Name 3 7 >/dev/null 2>&1; then
    echo 'invalid platform name unexpectedly succeeded' >&2
    exit 1
fi
if ./scripts/new-platform.sh invalid-count 0 7 >/dev/null 2>&1; then
    echo 'zero actuator count unexpectedly succeeded' >&2
    exit 1
fi
test ! -e 'crates/domains/symthaea-Bad_Name'
test ! -e 'crates/domains/symthaea-invalid-count'

popd >/dev/null

echo 'new-platform scaffold contract: PASS'