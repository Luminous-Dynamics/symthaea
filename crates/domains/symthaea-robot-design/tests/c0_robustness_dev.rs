// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development harness for exact C0 tolerance-box robustness.

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;
#[path = "../src/c0_robustness.rs"]
mod c0_robustness;

use c0_normalized::PositiveRationalV1;
use c0_robustness::{
    C0DimensionBoxV1, C0RobustnessDispositionV1, C0RobustnessPolicyV1,
    assess_c0_tolerance_robustness,
};
use design_parameters::DesignLengthUm;

fn um(value: u64) -> DesignLengthUm {
    DesignLengthUm::from_micrometres(value)
}

fn policy() -> C0RobustnessPolicyV1 {
    C0RobustnessPolicyV1 {
        max_mass_ratio: PositiveRationalV1::new(97, 100).unwrap(),
        max_deflection_ratio: PositiveRationalV1::new(19, 20).unwrap(),
        max_stress_ratio: PositiveRationalV1::new(1, 1).unwrap(),
    }
}

#[test]
fn synthetic_narrow_and_wide_boxes_produce_distinct_expected_dispositions() {
    let narrow = assess_c0_tolerance_robustness(
        um(20_000),
        um(6_000),
        C0DimensionBoxV1 {
            width_lower: um(15_950),
            width_upper: um(16_050),
            height_lower: um(7_150),
            height_upper: um(7_250),
        },
        policy(),
    )
    .unwrap();
    assert_eq!(
        narrow.disposition,
        C0RobustnessDispositionV1::RobustUnderDeclaredBox
    );

    let wide = assess_c0_tolerance_robustness(
        um(20_000),
        um(6_000),
        C0DimensionBoxV1 {
            width_lower: um(15_500),
            width_upper: um(16_750),
            height_lower: um(7_000),
            height_upper: um(7_500),
        },
        policy(),
    )
    .unwrap();
    assert_eq!(wide.disposition, C0RobustnessDispositionV1::MassTargetNotRobust);
    assert_eq!(wide.worst_mass_ratio, PositiveRationalV1::new(67, 64).unwrap());
}
