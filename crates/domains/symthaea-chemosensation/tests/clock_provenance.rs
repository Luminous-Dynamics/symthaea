// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-module regressions for acquisition-clock provenance.
//!
//! The deterministic sensor simulators accept caller-supplied numeric timestamps,
//! but they do not own an acquisition clock and must not silently label those
//! values as Unix time or as a shared device timebase. A later acquisition or
//! replay layer may attach an explicit `ChemicalClockDomainId` when justified.

use symthaea_chemosensation::{
    ElectronicTongueSimulator, GustatoryStimulus, MoxArraySimulator, MoxChannelModel,
    OlfactoryStimulus, PotentiometricChannelModel,
};

#[test]
fn olfaction_fixture_does_not_invent_clock_provenance() {
    let mut nose = MoxArraySimulator::new(vec![MoxChannelModel::new("mox-a", 100_000.0, 1.0)]);
    let observation = nose
        .step(
            &OlfactoryStimulus {
                concentration_ppm: 10.0,
                affinities: vec![1.0],
                temperature_c: 25.0,
                humidity_rh: 50.0,
            },
            0.1,
            123,
        )
        .unwrap();

    assert_eq!(observation.timestamp_us, 123);
    assert!(observation.clock_domain.is_none());
}

#[test]
fn gustation_fixture_does_not_invent_clock_provenance() {
    let tongue = ElectronicTongueSimulator::new(vec![PotentiometricChannelModel::new(
        "ion-a",
        1,
        vec![1.0],
    )]);
    let observation = tongue
        .sample(
            &GustatoryStimulus {
                ph: 7.0,
                conductivity_s_m: 1.0,
                ion_activities: vec![0.1],
                temperature_c: 25.0,
            },
            456,
        )
        .unwrap();

    assert_eq!(observation.timestamp_us, 456);
    assert!(observation.clock_domain.is_none());
}
