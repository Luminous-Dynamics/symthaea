include!("manta_forge_lineage_success_control_v1.rs");

use symthaea_maritime_core::{
    evaluate_policy_normalized_lineage_profile, RegenerativeGenerationCountV1,
};

fn finite_generation_count(value: RegenerativeGenerationCountV1) -> u64 {
    match value {
        RegenerativeGenerationCountV1::Finite(value) => value,
        RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
            panic!("success-control policy view must be finite")
        }
    }
}

#[test]
fn one_physical_v3_state_has_multiple_policy_views_without_physical_drift() {
    let v3_model = model("v3", 4, 0, 4, 18);
    let v3_report = evaluate_regenerative_lineage_viability(
        &profile("v3"),
        &genome("v3", Some("genome:manta-v2-success-control")),
        &v3_model,
        &support("v3"),
    )
    .unwrap();

    assert_eq!(
        v3_report.successor_reproduction_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );
    assert_eq!(
        v3_report.regenerative_viability_horizon,
        RegenerativeHorizon::FinitePeriods(4)
    );

    let profiles = (1u64..=4)
        .map(|maturity| {
            evaluate_policy_normalized_lineage_profile(
                &v3_report,
                format!("manta-reproduction-maturity-{maturity}"),
                format!("policy:manta-reproduction:maturity-{maturity}:v1"),
                maturity,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    for normalized in &profiles {
        assert_eq!(normalized.source_profile_id, v3_report.profile_id);
        assert_eq!(normalized.genome_id, v3_report.genome_id);
        assert_eq!(normalized.closure_model_id, v3_report.closure_model_id);
        assert_eq!(normalized.flow_support_id, v3_report.flow_support_id);
        assert_eq!(
            normalized.physical_successor_reproduction_horizon,
            RegenerativeHorizon::FinitePeriods(4)
        );
        assert_eq!(
            normalized.physical_regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(4)
        );
        assert!(normalized.fully_modeled_successor_reproduction);
        assert!(normalized.fully_modeled_regenerative_viability);
    }

    let founded = profiles
        .iter()
        .map(|profile| {
            finite_generation_count(
                profile
                    .generation_policy_projection
                    .founded_descendant_generations,
            )
        })
        .collect::<Vec<_>>();
    let matured = profiles
        .iter()
        .map(|profile| {
            finite_generation_count(
                profile
                    .generation_policy_projection
                    .maturity_completed_descendant_generations,
            )
        })
        .collect::<Vec<_>>();
    let transitions = profiles
        .iter()
        .map(|profile| {
            finite_generation_count(
                profile
                    .generation_policy_projection
                    .descendant_reproduction_transitions,
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(founded, vec![4, 2, 2, 1]);
    assert_eq!(matured, vec![4, 2, 1, 1]);
    assert_eq!(transitions, vec![3, 1, 1, 0]);
}
