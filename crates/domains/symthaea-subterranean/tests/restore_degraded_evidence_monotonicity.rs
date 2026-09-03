// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-surface red gate for degraded-operation evidence across restore.
//!
//! A stale checkpoint must not move watchdog/link-loss evidence backward and
//! thereby delay a safety transition, even when the visible degraded mode is
//! unchanged at the instant of restore.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_subterranean::degraded_operations::DegradedMode;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;

fn thought(seed: u64) -> ContinuousHV {
    ContinuousHV::random(HDC_DIMENSION, seed)
}

#[test]
fn stale_checkpoint_must_not_roll_back_watchdog_failure_evidence() {
    let genesis = GenesisSeed::from_phrase("restore degraded evidence monotonicity");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    // Default policy requires three consecutive unhealthy control-loop
    // observations before latching RecoveryRequired. After one failure the
    // visible mode is still Normal.
    embodiment.set_runtime_health(true, false, true, 0);
    embodiment.step(&thought(92_001), 0.005, 0.9);
    assert_eq!(embodiment.degraded_mode(), DegradedMode::Normal);
    let older = embodiment.operational_checkpoint();

    // Consume a second distinct unhealthy observation. The visible mode remains
    // Normal, but the live evidence now contains two consecutive failures.
    embodiment.set_runtime_health(true, false, true, 0);
    embodiment.step(&thought(92_002), 0.005, 0.9);
    assert_eq!(embodiment.degraded_mode(), DegradedMode::Normal);

    let restore = embodiment.load_operational_checkpoint(&older);
    if restore.is_err() {
        return;
    }
    assert_eq!(
        embodiment.degraded_mode(),
        DegradedMode::Normal,
        "the test isolates hidden degraded evidence rather than a mode change"
    );

    // The third unhealthy observation must still cross the watchdog threshold.
    // If restore replaced the live failure streak with the older checkpoint's
    // count, this incorrectly remains Normal for one extra control interval.
    embodiment.set_runtime_health(true, false, true, 0);
    embodiment.step(&thought(92_003), 0.005, 0.9);
    assert_eq!(
        embodiment.degraded_mode(),
        DegradedMode::RecoveryRequired,
        "stale restore rolled watchdog evidence backward and delayed RecoveryRequired"
    );
}
