// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::existential_reachability::{
    ExistentialReachabilityError, certify_existential_terminal_witness,
    verify_existential_terminal_witness,
};
use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, PlantTimeProfile,
};
use symthaea_core::robust_reachability::{
    BoundedDisturbance1D, RobustReachabilityPolicy1D, RobustReachabilityResult1D,
    RobustSingleIntegratorQuery1D, TerminalInterval1D, existential_terminal_interval,
    solve_robust_single_integrator_analytic,
};

fn model() -> BoundedSingleIntegrator1D {
    BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap())
}

fn query(
    target_lower: f64,
    target_upper: f64,
    disturbance_lower: f64,
    disturbance_upper: f64,
    horizon: f64,
) -> RobustSingleIntegratorQuery1D {
    let model = model();
    RobustSingleIntegratorQuery1D::new(
        &model,
        0.0,
        TerminalInterval1D::new(target_lower, target_upper).unwrap(),
        BoundedDisturbance1D::new(disturbance_lower, disturbance_upper).unwrap(),
        PlantTimeProfile::new(horizon).unwrap(),
        RobustReachabilityPolicy1D::reference_v1(),
    )
    .unwrap()
}

#[test]
fn existential_can_be_constructively_true_while_robust_is_certified_false() {
    let model = model();
    let query = query(-0.1, 0.1, -0.25, 0.25, 2.0);

    assert!(matches!(
        solve_robust_single_integrator_analytic(&model, &query).unwrap(),
        RobustReachabilityResult1D::CertifiedNotRobust { .. }
    ));

    let certified = certify_existential_terminal_witness(&model, &query, 0.0, 0.0).unwrap();
    assert_eq!(certified.witness().control(), 0.0);
    assert_eq!(certified.witness().disturbance(), 0.0);
    assert_eq!(certified.witness().terminal_enclosure().lower(), 0.0);
    assert_eq!(certified.witness().terminal_enclosure().upper(), 0.0);
    assert_eq!(
        certified.verification().witness_identity(),
        certified.witness().identity()
    );
}

#[test]
fn outer_possibility_overlap_is_not_used_as_existence_authority() {
    let model = model();
    let query = query(-0.1, 0.1, -0.25, 0.25, 2.0);
    let outer = existential_terminal_interval(&model, &query).unwrap();
    assert!(outer.intersection(query.target()).is_some());

    assert!(matches!(
        certify_existential_terminal_witness(&model, &query, 1.0, 0.25),
        Err(ExistentialReachabilityError::CandidateNotEstablished)
    ));
}

#[test]
fn rounding_trap_candidate_is_rejected_by_outward_realization_replay() {
    let model = model();
    let query = query(0.0, 0.03, 0.0, 0.1, 0.3);

    assert_eq!(0.1_f64.mul_add(0.3, 0.0), 0.03);
    assert!(matches!(
        certify_existential_terminal_witness(&model, &query, 0.0, 0.1),
        Err(ExistentialReachabilityError::CandidateNotEstablished)
    ));
}

#[test]
fn candidate_control_and_disturbance_membership_are_replayed_exactly() {
    let model = model();
    let query = query(-10.0, 10.0, -0.25, 0.25, 1.0);

    assert!(matches!(
        certify_existential_terminal_witness(&model, &query, 1.01, 0.0),
        Err(ExistentialReachabilityError::ControlOutsideBounds { .. })
    ));
    assert!(matches!(
        certify_existential_terminal_witness(&model, &query, 0.0, 0.251),
        Err(ExistentialReachabilityError::DisturbanceOutsideBounds { .. })
    ));
}

#[test]
fn witness_is_bound_to_original_query_not_only_point_disturbance_replay() {
    let model = model();
    let original = query(-0.1, 0.1, -0.25, 0.25, 2.0);
    let different = query(-0.2, 0.2, -0.25, 0.25, 2.0);
    let certified = certify_existential_terminal_witness(&model, &original, 0.0, 0.0).unwrap();

    assert!(matches!(
        verify_existential_terminal_witness(&model, &different, certified.witness()),
        Err(ExistentialReachabilityError::WitnessSubjectMismatch)
    ));
}

#[test]
fn independent_verifier_replays_fixed_realization_and_subject_identity() {
    let model = model();
    let query = query(-0.5, 0.5, -0.25, 0.25, 2.0);
    let certified = certify_existential_terminal_witness(&model, &query, 0.0, 0.0).unwrap();
    let replay = verify_existential_terminal_witness(&model, &query, certified.witness()).unwrap();

    assert_eq!(replay.model_identity(), model.identity());
    assert_eq!(replay.query_identity(), query.identity());
    assert_eq!(replay.witness_identity(), certified.witness().identity());
    assert_ne!(replay.verifier_identity(), [0_u8; 32]);
}

#[test]
fn existential_and_robust_verifier_authority_domains_are_distinct() {
    let model = model();
    let query = query(-0.5, 0.5, -0.25, 0.25, 2.0);
    let existential = certify_existential_terminal_witness(&model, &query, 0.0, 0.0).unwrap();
    let robust_verifier = match solve_robust_single_integrator_analytic(&model, &query).unwrap() {
        RobustReachabilityResult1D::RobustFeasible { verification, .. } => {
            verification.verifier_identity()
        }
        other => panic!("expected robust feasible result, got {other:?}"),
    };

    assert_ne!(existential.verification().verifier_identity(), robust_verifier);
}

#[test]
fn signed_zero_does_not_split_constructive_witness_identity() {
    let model = model();
    let query = query(-0.1, 0.1, -0.25, 0.25, 2.0);
    let positive = certify_existential_terminal_witness(&model, &query, 0.0, 0.0).unwrap();
    let negative = certify_existential_terminal_witness(&model, &query, -0.0, -0.0).unwrap();

    assert_eq!(positive.witness().identity(), negative.witness().identity());
    assert_eq!(positive.witness().control().to_bits(), 0.0_f64.to_bits());
    assert_eq!(positive.witness().disturbance().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn exact_power_of_two_point_realization_remains_constructively_provable() {
    let model = model();
    let query = query(1.0, 1.0, 0.0, 0.0, 2.0);
    let certified = certify_existential_terminal_witness(&model, &query, 0.5, 0.0).unwrap();

    assert_eq!(certified.witness().terminal_enclosure().lower(), 1.0);
    assert_eq!(certified.witness().terminal_enclosure().upper(), 1.0);
}
