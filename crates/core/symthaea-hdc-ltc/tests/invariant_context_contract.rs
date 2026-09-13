// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{ContinuousHV, InvariantContextMixer, UnitaryRole};

#[test]
fn context_invariance_holds_across_roles_and_seeds() {
    for seed in 0_u64..16 {
        let mixer = InvariantContextMixer::try_new(256, vec![1, 7, 31, 127], seed).unwrap();
        let state = ContinuousHV::new_random(256, seed + 100);
        let input = ContinuousHV::new_random(256, seed + 200);
        let role = UnitaryRole::new(256, seed + 300);
        let error = mixer
            .role_invariance_error(&role, &state, &input)
            .unwrap();
        assert_eq!(error, 0.0, "context invariance failed for seed={seed}");
    }
}

#[test]
fn mixer_has_genuine_cross_coordinate_receptive_field() {
    let mixer = InvariantContextMixer::try_new(64, vec![5], 42).unwrap();
    let baseline = ContinuousHV::new(64);
    let input = ContinuousHV::new(64);
    let mut changed = baseline.clone();

    // Target coordinate 0 reads source coordinate 5 through the configured offset.
    changed.values[5] = 2.0;
    let before = mixer.mix(&baseline, &input).unwrap();
    let after = mixer.mix(&changed, &input).unwrap();
    assert_ne!(before.values[0], after.values[0]);

    // The local signed state coordinate itself did not change; influence arrived
    // through cross-dimensional invariant context rather than direct recurrence.
    assert_eq!(baseline.values[0], changed.values[0]);
}
