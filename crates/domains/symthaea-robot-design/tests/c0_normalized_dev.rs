// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development-only integration harness for C0 normalized mechanics.

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;

use c0_normalized::{
    C0MatchedInvariantProfileId, C0RectangularSectionV1, PositiveRationalV1,
    evaluate_c0_normalized,
};
use design_parameters::{DesignLengthUm, DesignParameterSetId};

fn set_id(byte: u8) -> DesignParameterSetId {
    DesignParameterSetId::from_bytes([byte; 32])
}

fn section(id_byte: u8, width_um: u64, height_um: u64) -> C0RectangularSectionV1 {
    C0RectangularSectionV1::new(
        set_id(id_byte),
        DesignLengthUm::from_micrometres(width_um),
        DesignLengthUm::from_micrometres(height_um),
    )
    .unwrap()
}

#[test]
fn independent_known_answer_matches_frozen_c0_oracle_values() {
    let result = evaluate_c0_normalized(
        section(1, 20_000, 6_000),
        section(2, 16_000, 7_200),
        C0MatchedInvariantProfileId::from_bytes([9; 32]),
    )
    .unwrap();

    assert_eq!(result.mass_ratio, PositiveRationalV1::new(24, 25).unwrap());
    assert_eq!(
        result.deflection_ratio,
        PositiveRationalV1::new(625, 864).unwrap()
    );
    assert_eq!(
        result.stress_ratio,
        PositiveRationalV1::new(125, 144).unwrap()
    );
}

#[test]
fn same_ratios_do_not_collapse_distinct_candidate_subjects() {
    let profile = C0MatchedInvariantProfileId::from_bytes([7; 32]);
    let first = evaluate_c0_normalized(
        section(1, 20_000, 6_000),
        section(2, 16_000, 7_200),
        profile,
    )
    .unwrap();
    let second = evaluate_c0_normalized(
        section(1, 20_000, 6_000),
        section(3, 16_000, 7_200),
        profile,
    )
    .unwrap();

    assert_eq!(first.mass_ratio, second.mass_ratio);
    assert_eq!(first.deflection_ratio, second.deflection_ratio);
    assert_eq!(first.stress_ratio, second.stress_ratio);
    assert_ne!(first.id(), second.id());
}
