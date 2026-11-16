use fusion_decision_compiler_contract::decision_value_of_information::{
    AlternativeEligibility, DecisionAlternative, DecisionEvidenceClass, DecisionScenarioWeight,
    DecisionValuePlan, DecisionVoIDisposition, ExperimentSignal, InformationExperiment,
    ScenarioUtility, evaluate_decision_value_of_information,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn scenario(id: &str, weight: f64) -> DecisionScenarioWeight {
    DecisionScenarioWeight {
        scenario_id: id.to_string(),
        probability_weight: weight,
        weight_evidence_ref: format!("synthetic-weight-{id}"),
    }
}

fn utility(scenario_id: &str, alternative_id: &str, value: f64) -> ScenarioUtility {
    ScenarioUtility {
        scenario_id: scenario_id.to_string(),
        alternative_id: alternative_id.to_string(),
        utility_value: value,
        evidence_ref: format!("synthetic-utility-{scenario_id}-{alternative_id}"),
    }
}

fn experiment(
    id: &str,
    direct_cost: f64,
    opportunity_cost: f64,
    predictive_variance_score: f64,
    signals: &[(&str, &str)],
) -> InformationExperiment {
    InformationExperiment {
        experiment_id: id.to_string(),
        experiment_design_sha256: sha('d'),
        direct_cost_value: direct_cost,
        opportunity_cost_value: opportunity_cost,
        cost_unit_ref: "USD-2026-real-million".to_string(),
        cost_basis_ref: "USD-2026-real-discount-6pct-v1".to_string(),
        cost_evidence_ref: format!("synthetic-cost-{id}"),
        evidence_ref: format!("synthetic-experiment-{id}"),
        predictive_variance_score: Some(predictive_variance_score),
        variance_score_evidence_ref: Some(format!("synthetic-variance-{id}")),
        signal_map: signals
            .iter()
            .map(|(scenario_id, signal_id)| ExperimentSignal {
                scenario_id: (*scenario_id).to_string(),
                signal_id: (*signal_id).to_string(),
                evidence_ref: format!("synthetic-signal-{id}-{scenario_id}-{signal_id}"),
            })
            .collect(),
    }
}

fn main() {
    let plan = DecisionValuePlan {
        decision_case_id: "synthetic-fusion-service-decision-v1".to_string(),
        decision_case_sha256: sha('a'),
        scenario_weight_manifest_ref: "synthetic-scenario-weight-manifest-v1".to_string(),
        scenario_weight_manifest_sha256: sha('b'),
        utility_manifest_ref: "synthetic-action-utility-manifest-v1".to_string(),
        utility_manifest_sha256: sha('c'),
        cost_basis_ref: "USD-2026-real-discount-6pct-v1".to_string(),
        decision_value_unit_ref: "USD-2026-real-million".to_string(),
        lifetime_horizon_ref: "plant-life-30-years-v1".to_string(),
        validity_domain_ref: "synthetic-demonstrator-service-domain-v1".to_string(),
        joint_scenario_model_ref: "synthetic-joint-scenario-model-v1".to_string(),
        evidence_class: DecisionEvidenceClass::SyntheticFixture,
        weight_sum_tolerance: 1e-12,
        numerical_tolerance: 1e-10,
        scenarios: vec![
            scenario("s1", 0.25),
            scenario("s2", 0.25),
            scenario("s3", 0.25),
            scenario("s4", 0.25),
        ],
        alternatives: vec![
            DecisionAlternative {
                alternative_id: "A".to_string(),
                eligibility: AlternativeEligibility::Eligible,
                eligibility_evidence_ref: "synthetic-eligibility-A".to_string(),
            },
            DecisionAlternative {
                alternative_id: "B".to_string(),
                eligibility: AlternativeEligibility::Eligible,
                eligibility_evidence_ref: "synthetic-eligibility-B".to_string(),
            },
        ],
        utilities: vec![
            utility("s1", "A", 100.0),
            utility("s1", "B", 60.0),
            utility("s2", "A", 90.0),
            utility("s2", "B", 75.0),
            utility("s3", "A", 40.0),
            utility("s3", "B", 95.0),
            utility("s4", "A", 20.0),
            utility("s4", "B", 110.0),
        ],
        experiments: vec![
            experiment(
                "decision-discriminating",
                3.0,
                2.0,
                1.0,
                &[("s1", "left"), ("s2", "left"), ("s3", "right"), ("s4", "right")],
            ),
            experiment(
                "high-variance-uninformative",
                0.5,
                0.5,
                100.0,
                &[("s1", "same"), ("s2", "same"), ("s3", "same"), ("s4", "same")],
            ),
            experiment(
                "perfect-but-overpriced",
                20.0,
                0.0,
                25.0,
                &[("s1", "s1"), ("s2", "s2"), ("s3", "s3"), ("s4", "s4")],
            ),
        ],
    };

    let report = evaluate_decision_value_of_information(&plan);
    println!("contract_version={}", report.contract_version);
    println!("decision_case={}", report.decision_case_id);
    println!("disposition={:?}", report.disposition);
    println!(
        "current_best_alternative={}",
        report.current_best_alternative_id.as_deref().unwrap_or("NONE")
    );
    println!(
        "current_best_expected_value={:.6}",
        report.current_best_expected_value.unwrap_or(f64::NAN)
    );
    println!(
        "expected_value_with_perfect_information={:.6}",
        report
            .expected_value_with_perfect_information
            .unwrap_or(f64::NAN)
    );
    println!("evpi={:.6}", report.evpi.unwrap_or(f64::NAN));
    for experiment in &report.experiments {
        println!(
            "experiment={},variance_score={},EVSI={:.6},direct_cost={:.6},opportunity_cost={:.6},net_VoI={:.6},remaining_regret={:.6}",
            experiment.experiment_id,
            experiment.predictive_variance_score.unwrap_or(f64::NAN),
            experiment.evsi,
            experiment.direct_cost_value,
            experiment.opportunity_cost_value,
            experiment.net_voi,
            experiment.expected_regret_after_information
        );
        for signal in &experiment.signals {
            println!(
                "  signal={},p={:.3},action={},conditional_value={:.6}",
                signal.signal_id,
                signal.signal_probability,
                signal.selected_alternative_id,
                signal.conditional_expected_utility
            );
        }
    }
    println!(
        "selected_experiment={}",
        report.selected_experiment_id.as_deref().unwrap_or("NONE")
    );
    println!(
        "selected_net_voi={:.6}",
        report.selected_net_voi.unwrap_or(f64::NAN)
    );
    println!(
        "SCOPE: synthetic finite-ensemble decision-value example only; no calibrated fusion probabilities, utility model or physical experiment recommendation"
    );

    if report.disposition != DecisionVoIDisposition::SyntheticOnly
        || report.current_best_alternative_id.as_deref() != Some("B")
        || (report.evpi.unwrap_or(f64::NAN) - 13.75).abs() > 1e-10
        || report.selected_experiment_id.as_deref() != Some("decision-discriminating")
        || (report.selected_net_voi.unwrap_or(f64::NAN) - 8.75).abs() > 1e-10
    {
        eprintln!("finite-ensemble decision VoI example failed its known-answer checks");
        std::process::exit(1);
    }
}
