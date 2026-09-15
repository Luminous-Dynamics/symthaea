// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_core::two_stage_robust_reachability::{
    AffineStage1Policy1D, ConstantStage1Policy1D, MANIFOLD_006D_PREREGISTRATION,
    OpenLoopTwoStageStrategy1D, Stage1FeedbackStrategy1D, TwoStageChronologyEvent,
    TwoStageInformationPattern1D, TwoStageInformationPatternKind,
};

#[test]
fn audited_preregistration_binding_is_exact_and_identity_domains_remain_typed() {
    let binding = MANIFOLD_006D_PREREGISTRATION;
    assert_eq!(
        binding.control_commit().as_hex(),
        "c0f5776d28d321d46801ca0f9620a622d35ca28c"
    );
    assert_eq!(binding.hosted_run_id(), 34_919_856_614);
    assert_eq!(binding.artifact_id(), 10_377_995_319);
    assert_eq!(
        binding.artifact_zip_sha256().as_hex(),
        "c98edc977228a0946d3a1a961ef45ff02f5fad2a87beed093726639a4924b214"
    );
    assert_eq!(
        binding.preregistration_id().as_hex(),
        "fa62084a562a061445c74088914616480efc6008d2a16db56b23e9340612337e"
    );
    assert_eq!(binding.chronology_version().as_str(), "m006d.chron.v3");
    assert_eq!(
        binding.information_pattern_version().as_str(),
        "m006d.ip.v3"
    );
    assert_eq!(binding.observation_version().as_str(), "m006d.obs.v3");
    assert_eq!(
        binding.affine_policy_version().as_str(),
        "m006d.affine_pi.v3"
    );
    assert_eq!(binding.strategy_version().as_str(), "m006d.strategy.v2");
    assert_eq!(binding.policy_classes_version().as_str(), "m006d.classes.v2");
}

#[test]
fn information_patterns_freeze_distinct_chronology_and_external_contract_ids() {
    let open_loop = TwoStageInformationPattern1D::open_loop();
    let feedback = TwoStageInformationPattern1D::stage1_exact_observation();

    assert_eq!(
        open_loop.kind(),
        TwoStageInformationPatternKind::OpenLoopTwoStage
    );
    assert_eq!(
        feedback.kind(),
        TwoStageInformationPatternKind::Stage1ExactObservation1D
    );
    assert_ne!(open_loop.runtime_identity(), feedback.runtime_identity());
    assert_eq!(
        open_loop.preregistered_chronology_id(),
        MANIFOLD_006D_PREREGISTRATION.open_loop_chronology_id()
    );
    assert_eq!(
        feedback.preregistered_chronology_id(),
        MANIFOLD_006D_PREREGISTRATION.feedback_chronology_id()
    );
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
fn strategies_are_fixed_pure_data_and_bind_the_preregistered_contracts() {
    let open_loop = OpenLoopTwoStageStrategy1D::new(-0.0, 0.25).expect("open-loop strategy");
    assert_eq!(open_loop.first_control().to_bits(), 0.0f64.to_bits());
    assert_eq!(
        open_loop.strategy_contract_id(),
        MANIFOLD_006D_PREREGISTRATION.open_loop_strategy_contract_id()
    );

    let constant = ConstantStage1Policy1D::new(0.25).expect("constant policy");
    assert_eq!(
        constant.policy_classes_id(),
        MANIFOLD_006D_PREREGISTRATION.policy_classes_id()
    );
    let constant_strategy =
        Stage1FeedbackStrategy1D::new(0.0, constant.into()).expect("constant strategy");
    assert_eq!(
        constant_strategy.strategy_contract_id(),
        MANIFOLD_006D_PREREGISTRATION.feedback_strategy_contract_id()
    );

    let affine = AffineStage1Policy1D::new(-1.0, -0.0, -1.0, 1.0).expect("affine policy");
    assert_eq!(affine.bias().to_bits(), 0.0f64.to_bits());
    assert_eq!(affine.semantics_version().as_str(), "m006d.affine_pi.v3");
    let affine_strategy =
        Stage1FeedbackStrategy1D::new(0.0, affine.into()).expect("affine strategy");

    assert_ne!(
        constant_strategy.runtime_identity(),
        affine_strategy.runtime_identity()
    );
    assert_eq!(
        MANIFOLD_006D_PREREGISTRATION.reference_affine_policy_id().as_hex(),
        "f84ef73cdebf2ed9ef887b3e7be05bcc3200d36dd43dc9027221522283e84fb3"
    );
}
