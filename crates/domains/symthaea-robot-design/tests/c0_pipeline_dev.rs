// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development-only end-to-end synthetic C0 pipeline harness.

pub use symthaea_robot_design::{
    ContentDigest, DesignComponentId, DesignComponentV1, GeometryDesignId, GeometryDesignRefV1,
    GeometryRepresentationV1, MaterialAssignmentV1, RobotDesignError, RobotDesignId,
    RobotDesignSubjectV1, RobotMorphologyDesignV1,
};

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_joint_link_coupon.rs"]
mod c0_joint_link_coupon;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;
#[path = "../src/c0_enumeration.rs"]
mod c0_enumeration;
#[path = "../src/c0_applicability.rs"]
mod c0_applicability;

use c0_applicability::{
    C0ApplicabilityDispositionV1, C0ApplicabilityProfileV1,
    assess_c0_geometry_applicability,
};
use c0_enumeration::{
    C0_HEIGHT_PARAMETER_ID as ENUM_HEIGHT_ID, C0_WIDTH_PARAMETER_ID as ENUM_WIDTH_ID,
    C0SelectionPolicyV1, enumerate_c0_domain, select_c0_candidate,
};
use c0_joint_link_coupon::{C0JointLinkCouponTemplateV1, compile_c0_coupon};
use c0_normalized::{C0MatchedInvariantProfileId, PositiveRationalV1};
use design_parameters::{
    DesignLengthDomainV1, DesignLengthParameterV1, DesignLengthUm, DesignParameterId,
    DesignParameterSetV1, DesignSearchDomainV1,
};

fn digest(byte: u8) -> ContentDigest {
    ContentDigest::from_bytes([byte; 32])
}

fn id(value: &str) -> DesignParameterId {
    DesignParameterId::new(value).unwrap()
}

fn reference_domain() -> DesignSearchDomainV1 {
    DesignSearchDomainV1::new(vec![
        DesignLengthDomainV1::Explicit {
            id: id(ENUM_WIDTH_ID),
            values: [16_000, 18_000, 20_000, 22_000, 24_000]
                .into_iter()
                .map(DesignLengthUm::from_micrometres)
                .collect(),
        },
        DesignLengthDomainV1::Explicit {
            id: id(ENUM_HEIGHT_ID),
            values: [4_800, 5_400, 6_000, 6_600, 7_200]
                .into_iter()
                .map(DesignLengthUm::from_micrometres)
                .collect(),
        },
    ])
    .unwrap()
}

fn baseline_parameters() -> DesignParameterSetV1 {
    DesignParameterSetV1::new(vec![
        DesignLengthParameterV1 {
            id: id(ENUM_WIDTH_ID),
            value: DesignLengthUm::from_micrometres(20_000),
        },
        DesignLengthParameterV1 {
            id: id(ENUM_HEIGHT_ID),
            value: DesignLengthUm::from_micrometres(6_000),
        },
    ])
    .unwrap()
}

fn template() -> C0JointLinkCouponTemplateV1 {
    C0JointLinkCouponTemplateV1 {
        fixed_length: DesignLengthUm::from_micrometres(300_000),
        requirement_snapshot: digest(1),
        material_digest: digest(2),
        fabrication_constraints: digest(3),
    }
}

#[test]
fn synthetic_c0_pipeline_selects_compiles_and_admits_one_consistent_candidate() {
    let baseline_params = baseline_parameters();
    let baseline_design = compile_c0_coupon(template(), &baseline_params).unwrap();
    let profile_id = C0MatchedInvariantProfileId::from_bytes([0xC0; 32]);

    let candidates = enumerate_c0_domain(&reference_domain(), &baseline_params, profile_id).unwrap();
    assert_eq!(candidates.len(), 25);

    let selected = select_c0_candidate(
        &candidates,
        C0SelectionPolicyV1 {
            max_deflection_ratio: PositiveRationalV1::new(19, 20).unwrap(),
            max_stress_ratio: PositiveRationalV1::new(1, 1).unwrap(),
        },
    )
    .unwrap();
    assert_eq!(selected.width.micrometres(), 16_000);
    assert_eq!(selected.height.micrometres(), 7_200);

    let candidate_design = compile_c0_coupon(template(), &selected.parameter_set).unwrap();
    assert_eq!(
        selected.normalized.candidate_parameter_set_id,
        candidate_design.parameter_set_id
    );
    assert_eq!(
        selected.normalized.baseline_parameter_set_id,
        baseline_design.parameter_set_id
    );
    assert_ne!(baseline_design.design_id, candidate_design.design_id);

    let applicability = assess_c0_geometry_applicability(
        template().fixed_length,
        selected.width,
        selected.height,
        C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: PositiveRationalV1::new(1, 100).unwrap(),
            max_height_to_width_ratio: PositiveRationalV1::new(2, 1).unwrap(),
        },
    )
    .unwrap();
    assert_eq!(
        applicability.disposition,
        C0ApplicabilityDispositionV1::AdmittedGeometryScreen
    );

    assert_eq!(selected.normalized.mass_ratio, PositiveRationalV1::new(24, 25).unwrap());
    assert_eq!(
        selected.normalized.deflection_ratio,
        PositiveRationalV1::new(625, 864).unwrap()
    );
    assert_eq!(
        selected.normalized.stress_ratio,
        PositiveRationalV1::new(125, 144).unwrap()
    );
}
