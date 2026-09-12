use symthaea_maritime_core::{
    calibrate_regenerative_viability_role, evaluate_regenerative_policy_sensitivity_surface,
    RegenerativeGenerationCountV1, RegenerativeHorizon, RegenerativeLineageRoleReportV1,
    RegenerativeLineageRoleV1, RegenerativeLineageViabilityReportV1,
    RegenerativeReproductionPolicySpecV1, RegenerativeViabilityCalibrationClassV1,
    RegenerativeViabilityDynamicObservationV1,
};

fn role(
    role: RegenerativeLineageRoleV1,
    horizon: RegenerativeHorizon,
) -> RegenerativeLineageRoleReportV1 {
    RegenerativeLineageRoleReportV1 {
        role,
        conservative_horizon: horizon,
        limiting_requirement_ids: vec![format!("requirement:{role:?}")],
        root_limiting_dependency_ids: vec![format!("dependency:{role:?}")],
        fully_modeled_support: true,
        externally_conditioned_requirement_ids: Vec::new(),
    }
}

fn report(successor_horizon: u64) -> RegenerativeLineageViabilityReportV1 {
    RegenerativeLineageViabilityReportV1 {
        profile_id: format!("profile:manta-v3-h{successor_horizon}"),
        genome_id: "genome:manta-v3".into(),
        closure_model_id: format!("closure:manta-v3-h{successor_horizon}"),
        flow_support_id: format!("support:manta-v3-h{successor_horizon}"),
        operation: role(
            RegenerativeLineageRoleV1::Operation,
            RegenerativeHorizon::FinitePeriods(8),
        ),
        successor_construction: role(
            RegenerativeLineageRoleV1::SuccessorConstruction,
            RegenerativeHorizon::FinitePeriods(successor_horizon),
        ),
        successor_qualification: role(
            RegenerativeLineageRoleV1::SuccessorQualification,
            RegenerativeHorizon::FinitePeriods(successor_horizon + 1),
        ),
        successor_reproduction_horizon: RegenerativeHorizon::FinitePeriods(successor_horizon),
        regenerative_viability_horizon: RegenerativeHorizon::FinitePeriods(successor_horizon),
        limiting_roles: vec![RegenerativeLineageRoleV1::SuccessorConstruction],
        fully_modeled_regenerative_viability: true,
    }
}

fn policies() -> Vec<RegenerativeReproductionPolicySpecV1> {
    (1..=4)
        .map(|maturity_periods| RegenerativeReproductionPolicySpecV1 {
            reproduction_policy_id: format!("manta-v3-maturity-{maturity_periods}"),
            reproduction_policy_evidence_binding: format!(
                "policy:manta-v3:maturity:{maturity_periods}:v1"
            ),
            maturity_periods,
        })
        .collect()
}

fn finite(value: RegenerativeGenerationCountV1) -> u64 {
    match value {
        RegenerativeGenerationCountV1::Finite(value) => value,
        RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
            panic!("expected finite control surface")
        }
    }
}

#[test]
fn post_handoff_reserve_loss_invalidates_direct_h4_calibration_comparison() {
    // Dynamic #814 removes one tooling and one safeguarded-service reserve unit
    // after the H=4 successor has already been established. The resulting dynamic
    // system first becomes unavailable on tick 4, equivalent to three complete
    // periods under the changed state.
    let observation = RegenerativeViabilityDynamicObservationV1 {
        role_id: "successor_reproduction".into(),
        static_horizon: RegenerativeHorizon::FinitePeriods(4),
        first_unavailable_tick: Some(4),
        observed_through_tick: 4,
        static_assumptions_held: false,
        observation_binding:
            "git:Luminous-Dynamics/symtropy:ed96021386a6130119420e1db987d7b7f90a3b33"
                .into(),
    };
    let calibrated = calibrate_regenerative_viability_role(&observation).unwrap();

    assert_eq!(
        calibrated.classification,
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    );
    assert_eq!(calibrated.expected_first_unavailable_tick, Some(5));
    assert_eq!(calibrated.observed_first_unavailable_tick, Some(4));
    assert!(!calibrated.static_model_conflict);
    assert!(!calibrated.model_refinement_opportunity);
}

#[test]
fn rebaselined_h3_model_corroborates_the_changed_dynamic_boundary() {
    let observation = RegenerativeViabilityDynamicObservationV1 {
        role_id: "successor_reproduction".into(),
        static_horizon: RegenerativeHorizon::FinitePeriods(3),
        first_unavailable_tick: Some(4),
        observed_through_tick: 4,
        static_assumptions_held: true,
        observation_binding:
            "git:Luminous-Dynamics/symtropy:ed96021386a6130119420e1db987d7b7f90a3b33"
                .into(),
    };
    let calibrated = calibrate_regenerative_viability_role(&observation).unwrap();

    assert_eq!(
        calibrated.classification,
        RegenerativeViabilityCalibrationClassV1::CorroboratedFiniteBoundary
    );
    assert_eq!(calibrated.expected_first_unavailable_tick, Some(4));
    assert!(!calibrated.static_model_conflict);
}

#[test]
fn assumption_change_requires_a_new_policy_surface_not_mutation_of_the_old_one() {
    let h4 = evaluate_regenerative_policy_sensitivity_surface(&report(4), &policies()).unwrap();
    let h3 = evaluate_regenerative_policy_sensitivity_surface(&report(3), &policies()).unwrap();

    assert_eq!(
        h4.physical_successor_reproduction_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert_eq!(
        h3.physical_successor_reproduction_horizon,
        RegenerativeHorizon::FinitePeriods(3)
    );
    assert_ne!(h4, h3);

    let h4_counts: Vec<_> = h4
        .policy_points
        .iter()
        .map(|point| {
            let projection = &point.generation_policy_projection;
            (
                finite(projection.founded_descendant_generations),
                finite(projection.maturity_completed_descendant_generations),
                finite(projection.descendant_reproduction_transitions),
            )
        })
        .collect();
    let h3_counts: Vec<_> = h3
        .policy_points
        .iter()
        .map(|point| {
            let projection = &point.generation_policy_projection;
            (
                finite(projection.founded_descendant_generations),
                finite(projection.maturity_completed_descendant_generations),
                finite(projection.descendant_reproduction_transitions),
            )
        })
        .collect();

    assert_eq!(h4_counts, vec![(4, 4, 3), (2, 2, 1), (2, 1, 1), (1, 1, 0)]);
    assert_eq!(h3_counts, vec![(3, 3, 2), (2, 1, 1), (1, 1, 0), (1, 0, 0)]);
}
