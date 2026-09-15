// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_core::two_stage_robust_reachability::{
    AffineStage1Policy1D, ConstantStage1Policy1D, OpenLoopTwoStageStrategy1D,
    Stage1FeedbackStrategy1D, Stage1Policy1D, TwoStageChronologyEvent,
    TwoStageInformationPattern1D, TwoStageInformationPatternKind,
    MANIFOLD_006D_PREREGISTRATION,
};

#[test]
fn audited_preregistration_binding_is_carried_without_identity_domain_conflation() {
    let binding = MANIFOLD_006D_PREREGISTRATION;
    assert_eq!(binding.hosted_run_id(), 34_919_856_614);
    assert_eq!(binding.artifact_id(), 10_377_995_319);
    assert_eq!(
        binding.preregistration_json_sha256().as_bytes(),
        &[
            0x40, 0x92, 0xb9, 0x71, 0xb2, 0x9a, 0x64, 0xa7, 0xeb, 0x62, 0x2e, 0x5f, 0xde,
            0xe1, 0xf8, 0x0d, 0x0b, 0xaf, 0xbe, 0x7b, 0x9e, 0x6d, 0xc6, 0x51, 0x39, 0x95,
            0xf6, 0x92, 0xac, 0x74, 0x37, 0x1b,
        ]
    );
    assert_eq!(
        binding.preregistration_id().as_bytes(),
        &[
            0xfa, 0x62, 0x08, 0x4a, 0x56, 0x2a, 0x06, 0x14, 0x45, 0xc7, 0x40, 0x88, 0x91,
            0x46, 0x16, 0x48, 0x0e, 0xfc, 0x60, 0x08, 0xd2, 0xa1, 0x6d, 0xb5, 0x6b, 0x23,
            0xe9, 0x34, 0x06, 0x12, 0x33, 0x7e,
        ]
    );

    let open_loop = TwoStageInformationPattern1D::open_loop();
    let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
    assert_ne!(open_loop.runtime_identity(), feedback.runtime_identity());
}

#[test]
fn information_pattern_chronology_keeps_decision_order_explicit() {
    let open_loop = TwoStageInformationPattern1D::open_loop();
    assert_eq!(open_loop.kind(), TwoStageInformationPatternKind::OpenLoopTwoStage);
    assert_eq!(
        open_loop.chronology(),
        &[
            TwoStageChronologyEvent::SelectOpenLoopControls,
            TwoStageChronologyEvent::RealizeFirstDisturbance,
            TwoStageChronologyEvent::EstablishStage1State,
            TwoStageChronologyEvent::ApplySecondOpenLoopControl,
            TwoStageChronologyEvent::RealizeSecondDisturbance,
            TwoStageChronologyEvent::EstablishTerminalState,
        ]
    );

    let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
    assert_eq!(
        feedback.kind(),
        TwoStageInformationPatternKind::Stage1ExactObservation1D
    );
    assert_eq!(
        feedback.chronology(),
        &[
            TwoStageChronologyEvent::SelectFeedbackStrategy,
            TwoStageChronologyEvent::RealizeFirstDisturbance,
            TwoStageChronologyEvent::EstablishStage1State,
            TwoStageChronologyEvent::PublishStage1Observation,
            TwoStageChronologyEvent::EvaluateFixedStage1Policy,
            TwoStageChronologyEvent::RealizeSecondDisturbance,
            TwoStageChronologyEvent::EstablishTerminalState,
        ]
    );
}

#[test]
fn strategies_are_fixed_pure_data_objects_without_theorem_verdicts() {
    let open_loop = OpenLoopTwoStageStrategy1D::new(0.0, 0.25).expect("finite open-loop strategy");
    assert_eq!(open_loop.first_control(), 0.0);
    assert_eq!(open_loop.second_control(), 0.25);
    assert_eq!(
        open_loop.information_pattern_identity(),
        TwoStageInformationPattern1D::open_loop().runtime_identity()
    );

    let constant = ConstantStage1Policy1D::new(0.25).expect("finite constant policy");
    let constant_strategy = Stage1FeedbackStrategy1D::new(0.0, constant.into())
        .expect("constant feedback strategy");
    assert_eq!(
        constant_strategy.information_pattern_identity(),
        TwoStageInformationPattern1D::stage1_exact_observation().runtime_identity()
    );

    let affine = AffineStage1Policy1D::new(-1.0, 0.0, -1.0, 1.0)
        .expect("finite preregistered affine syntax");
    let affine_strategy = Stage1FeedbackStrategy1D::new(0.0, Stage1Policy1D::Affine(affine))
        .expect("affine feedback strategy");
    assert_ne!(constant_strategy.runtime_identity(), affine_strategy.runtime_identity());
}
