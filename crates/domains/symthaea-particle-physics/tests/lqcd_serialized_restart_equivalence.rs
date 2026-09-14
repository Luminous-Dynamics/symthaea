// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LQCD-021D/021E serialized split-run restart qualification.
//!
//! Strengthens the in-memory theorem in #2791 by crossing the canonical
//! `wilson_gauge_field_be_f64_v1` byte boundary between N and M transitions.
//! It still does not claim filesystem durability or cryptographic checkpoint
//! authorization; those are separate higher-level gates.

use symthaea_particle_physics::lattice_rng::{
    LatticeCampaignPhase, LatticeChaCha8Stream, LatticeStreamCoordinatesV2,
    LatticeStreamPurpose,
};
use symthaea_particle_physics::{
    HeatbathOverrelaxationSchedule, SubgroupForceBackend, WilsonGaugeField,
    decode_wilson_gauge_field, encode_wilson_gauge_field, heatbath_overrelaxation_cycle,
};

const DIMS: [usize; 4] = [2, 2, 2, 2];
const BETA: f64 = 5.7;
const N: usize = 2;
const M: usize = 3;

fn schedule() -> HeatbathOverrelaxationSchedule {
    HeatbathOverrelaxationSchedule {
        force_backend: SubgroupForceBackend::Staple,
        overrelaxation_sweeps: 3,
        max_heatbath_attempts: 256,
    }
}

fn stream() -> LatticeChaCha8Stream {
    LatticeChaCha8Stream::new_v2(
        [0xe1; 32],
        LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Pilot,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 0x021e,
            replica: 13,
            rank: 0,
        },
    )
    .unwrap()
}

fn run_cycles(field: &mut WilsonGaugeField, rng: &mut LatticeChaCha8Stream, cycles: usize) {
    for _ in 0..cycles {
        heatbath_overrelaxation_cycle(field, BETA, schedule(), rng).unwrap();
    }
}

fn assert_field_bitwise_equal(left: &WilsonGaugeField, right: &WilsonGaugeField) {
    assert_eq!(left.dims(), right.dims());
    let dims = left.dims();
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        let a = left.link(site, mu).unwrap();
                        let b = right.link(site, mu).unwrap();
                        for row in 0..3 {
                            for col in 0..3 {
                                assert_eq!(a[row][col].re.to_bits(), b[row][col].re.to_bits());
                                assert_eq!(a[row][col].im.to_bits(), b[row][col].im.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn serialized_checkpoint_split_run_matches_uninterrupted_bit_for_bit() {
    let mut uninterrupted_field = WilsonGaugeField::identity(DIMS).unwrap();
    let mut uninterrupted_rng = stream();
    run_cycles(&mut uninterrupted_field, &mut uninterrupted_rng, N + M);

    let mut prefix_field = WilsonGaugeField::identity(DIMS).unwrap();
    let mut prefix_rng = stream();
    run_cycles(&mut prefix_field, &mut prefix_rng, N);

    let checkpoint_bytes = encode_wilson_gauge_field(&prefix_field).unwrap();
    let checkpoint_rng = prefix_rng.replay_state();
    drop(prefix_field);
    drop(prefix_rng);

    let mut resumed_field = decode_wilson_gauge_field(&checkpoint_bytes).unwrap();
    assert_eq!(
        encode_wilson_gauge_field(&resumed_field).unwrap(),
        checkpoint_bytes,
        "persistent field roundtrip changed checkpoint bytes"
    );
    let mut resumed_rng = LatticeChaCha8Stream::from_replay_state(checkpoint_rng);
    run_cycles(&mut resumed_field, &mut resumed_rng, M);

    assert_field_bitwise_equal(&uninterrupted_field, &resumed_field);
    assert_eq!(uninterrupted_rng.stream_id(), resumed_rng.stream_id());
    assert_eq!(uninterrupted_rng.word_pos(), resumed_rng.word_pos());
    assert_eq!(
        uninterrupted_field.average_plaquette().unwrap().to_bits(),
        resumed_field.average_plaquette().unwrap().to_bits()
    );
    assert_eq!(
        uninterrupted_field.wilson_action(BETA).unwrap().to_bits(),
        resumed_field.wilson_action(BETA).unwrap().to_bits()
    );

    // Continuation remains synchronized after the final compared field state.
    for _ in 0..64 {
        assert_eq!(uninterrupted_rng.next_u64(), resumed_rng.next_u64());
    }
}
