// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LQCD-021D exact split-run transition qualification.
//!
//! This test proves the in-memory transition theorem only: given an exact
//! `WilsonGaugeField` clone plus the exact ChaCha8 replay state, running N cycles,
//! restoring, then running M cycles must reproduce an uninterrupted N+M run bit
//! for bit. Canonical field serialization and cryptographic checkpoint authority
//! remain separate later gates.

use symthaea_particle_physics::lattice_rng::{
    LatticeCampaignPhase, LatticeChaCha8Stream, LatticeStreamCoordinatesV2,
    LatticeStreamPurpose,
};
use symthaea_particle_physics::{
    HeatbathOverrelaxationSchedule, SubgroupForceBackend, WilsonGaugeField,
    heatbath_overrelaxation_cycle,
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
        [0xd1; 32],
        LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Pilot,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 0x021d,
            replica: 11,
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
                                assert_eq!(
                                    a[row][col].re.to_bits(),
                                    b[row][col].re.to_bits(),
                                    "real mismatch at site={site:?} mu={mu} row={row} col={col}",
                                );
                                assert_eq!(
                                    a[row][col].im.to_bits(),
                                    b[row][col].im.to_bits(),
                                    "imag mismatch at site={site:?} mu={mu} row={row} col={col}",
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn split_run_matches_uninterrupted_transition_bit_for_bit() {
    let mut uninterrupted_field = WilsonGaugeField::identity(DIMS).unwrap();
    let mut uninterrupted_rng = stream();
    run_cycles(&mut uninterrupted_field, &mut uninterrupted_rng, N + M);

    let mut prefix_field = WilsonGaugeField::identity(DIMS).unwrap();
    let mut prefix_rng = stream();
    run_cycles(&mut prefix_field, &mut prefix_rng, N);

    // The field clone stands in for exact checkpoint bytes in this tranche.
    // Later checkpoint work must prove serialization roundtrip before replacing
    // this in-memory identity theorem with persistent restart authority.
    let checkpoint_field = prefix_field.clone();
    let checkpoint_rng = prefix_rng.replay_state();
    drop(prefix_field);
    drop(prefix_rng);

    let mut resumed_field = checkpoint_field;
    let mut resumed_rng = LatticeChaCha8Stream::from_replay_state(checkpoint_rng);
    run_cycles(&mut resumed_field, &mut resumed_rng, M);

    assert_field_bitwise_equal(&uninterrupted_field, &resumed_field);
    assert_eq!(uninterrupted_rng.stream_id(), resumed_rng.stream_id());
    assert_eq!(uninterrupted_rng.word_pos(), resumed_rng.word_pos());

    assert_eq!(
        uninterrupted_field.average_plaquette().unwrap().to_bits(),
        resumed_field.average_plaquette().unwrap().to_bits(),
    );
    assert_eq!(
        uninterrupted_field.wilson_action(BETA).unwrap().to_bits(),
        resumed_field.wilson_action(BETA).unwrap().to_bits(),
    );

    // The continuation theorem extends beyond the compared field state: the
    // next random words must also be identical after the split run.
    for _ in 0..64 {
        assert_eq!(uninterrupted_rng.next_u64(), resumed_rng.next_u64());
    }
}
