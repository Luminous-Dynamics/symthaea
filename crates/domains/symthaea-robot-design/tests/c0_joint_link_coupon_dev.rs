// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development integration harness for C0A coupon compilation.

pub use symthaea_robot_design::{
    ContentDigest, DesignComponentId, DesignComponentV1, GeometryDesignId, GeometryDesignRefV1,
    GeometryRepresentationV1, MaterialAssignmentV1, RobotDesignError, RobotDesignId,
    RobotDesignSubjectV1, RobotMorphologyDesignV1,
};

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_joint_link_coupon.rs"]
mod c0_joint_link_coupon;

use c0_joint_link_coupon::{
    C0_HEIGHT_PARAMETER_ID, C0_WIDTH_PARAMETER_ID, C0JointLinkCouponTemplateV1,
    compile_c0_coupon,
};
use design_parameters::{
    DesignLengthParameterV1, DesignLengthUm, DesignParameterId, DesignParameterSetV1,
};

fn digest(byte: u8) -> ContentDigest {
    ContentDigest::from_bytes([byte; 32])
}

fn id(value: &str) -> DesignParameterId {
    DesignParameterId::new(value).unwrap()
}

fn params(width: u64, height: u64) -> DesignParameterSetV1 {
    DesignParameterSetV1::new(vec![
        DesignLengthParameterV1 {
            id: id(C0_WIDTH_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(width),
        },
        DesignLengthParameterV1 {
            id: id(C0_HEIGHT_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(height),
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
fn baseline_and_candidate_compile_to_distinct_exact_robot_design_subjects() {
    let baseline = compile_c0_coupon(template(), &params(20_000, 6_000)).unwrap();
    let candidate = compile_c0_coupon(template(), &params(16_000, 7_200)).unwrap();

    assert_ne!(baseline.parameter_set_id, candidate.parameter_set_id);
    assert_ne!(baseline.geometry_intent_digest, candidate.geometry_intent_digest);
    assert_ne!(baseline.design_id, candidate.design_id);
    assert_eq!(baseline.subject.morphology.components.len(), 1);
    assert!(baseline.subject.morphology.joints.is_empty());
}

#[test]
fn exact_geometry_intent_preserves_length_width_height_axes() {
    let compiled = compile_c0_coupon(template(), &params(16_000, 7_200)).unwrap();
    assert_eq!(compiled.geometry_intent.length.micrometres(), 300_000);
    assert_eq!(compiled.geometry_intent.width.micrometres(), 16_000);
    assert_eq!(compiled.geometry_intent.height.micrometres(), 7_200);
}
