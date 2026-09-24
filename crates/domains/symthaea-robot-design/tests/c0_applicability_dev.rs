// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development integration harness for C0E applicability screening.

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;
#[path = "../src/c0_applicability.rs"]
mod c0_applicability;

use c0_applicability::{
    C0ApplicabilityDispositionV1, C0ApplicabilityProfileV1,
    assess_c0_geometry_applicability,
};
use c0_normalized::PositiveRationalV1;
use design_parameters::DesignLengthUm;

fn um(value: u64) -> DesignLengthUm {
    DesignLengthUm::from_micrometres(value)
}

fn profile() -> C0ApplicabilityProfileV1 {
    C0ApplicabilityProfileV1 {
        max_conservative_shear_to_bending_ratio: PositiveRationalV1::new(1, 100).unwrap(),
        max_height_to_width_ratio: PositiveRationalV1::new(2, 1).unwrap(),
    }
}

#[test]
fn reference_baseline_and_candidate_pass_geometry_only_screen() {
    for (width, height) in [(20_000, 6_000), (16_000, 7_200)] {
        let assessment = assess_c0_geometry_applicability(
            um(300_000),
            um(width),
            um(height),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
    }
}

#[test]
fn known_hostile_geometry_is_rejected_without_becoming_a_safety_claim() {
    let assessment = assess_c0_geometry_applicability(
        um(300_000),
        um(40_000),
        um(30_000),
        profile(),
    )
    .unwrap();
    assert_eq!(
        assessment.disposition,
        C0ApplicabilityDispositionV1::RejectedConservativeShearBound
    );
}
