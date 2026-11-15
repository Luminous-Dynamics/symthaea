use fusion_reject_audit::{
    AcceptedCandidate, AcceptedCandidateManifest, AuditOutcome, FrozenCandidateUniverse,
    RejectAuditPlan, RejectStratum, deterministic_audit_sample_ids,
    scenario_opportunity::{
        CandidateScenarioBundle, ScenarioDefinition, ScenarioEvaluation,
        ScenarioOpportunityPlan, StratumScenarioBundles, audit_scenario_opportunities,
    },
    decision_opportunity::{
        IncumbentFrontierPoint, ObjectiveDefinition, ObjectiveDirection, ObjectiveValue,
    },
};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn objective_values(cost: f64, availability: f64) -> Vec<ObjectiveValue> {
    vec![
        ObjectiveValue {
            objective_id: "lifetime-cost".to_string(),
            unit: "MUSD_2026".to_string(),
            value: cost,
        },
        ObjectiveValue {
            objective_id: "availability".to_string(),
            unit: "fraction".to_string(),
            value: availability,
        },
    ]
}

fn evaluation(
    scenario_id: &str,
    candidate_id: &str,
    outcome: AuditOutcome,
    within_declared_domain: bool,
    objectives: Vec<ObjectiveValue>,
    evidence_ref: String,
) -> ScenarioEvaluation {
    ScenarioEvaluation {
        scenario_id: scenario_id.to_string(),
        run_id: format!("synthetic-{scenario_id}-run-{candidate_id}"),
        reference_outcome: outcome,
        within_declared_domain,
        objectives,
        evidence_ref,
    }
}

fn main() {
    let seed = 0x1234_5678_9abc_def0_u64;
    let accepted = vec![
        AcceptedCandidate {
            candidate_id: "accepted-0".to_string(),
            outcome: AuditOutcome::ConfirmedFeasible,
            evidence_ref: "synthetic-primary-run-accepted-0".to_string(),
        },
        AcceptedCandidate {
            candidate_id: "accepted-1".to_string(),
            outcome: AuditOutcome::Failed,
            evidence_ref: "synthetic-primary-run-accepted-1".to_string(),
        },
    ];
    let rejected_ids = vec![
        "rejected-0".to_string(),
        "rejected-1".to_string(),
        "rejected-2".to_string(),
    ];
    let primary_outcomes = [
        ("rejected-0", AuditOutcome::ConfirmedFeasible),
        ("rejected-1", AuditOutcome::ConfirmedInfeasible),
        ("rejected-2", AuditOutcome::NonConverged),
    ];
    let selected = deterministic_audit_sample_ids("rejected", &rejected_ids, 3, seed)
        .expect("frozen sample must be reproducible");
    let samples = selected
        .iter()
        .map(|candidate_id| {
            let outcome = primary_outcomes
                .iter()
                .find(|(id, _)| *id == candidate_id)
                .expect("each sample has a frozen primary disposition")
                .1;
            fusion_reject_audit::AuditSample {
                candidate_id: candidate_id.clone(),
                outcome,
                evidence_ref: format!("synthetic-primary-run-{candidate_id}"),
            }
        })
        .collect();

    let mut frozen_ids = accepted
        .iter()
        .map(|candidate| candidate.candidate_id.clone())
        .collect::<Vec<_>>();
    frozen_ids.extend(rejected_ids.clone());
    frozen_ids.sort();

    let base_plan = RejectAuditPlan {
        frozen_universe: FrozenCandidateUniverse {
            candidate_ids: frozen_ids,
            manifest_sha256: digest('a'),
        },
        sampling_seed: seed,
        seed_commitment_ref: "synthetic-precommitted-seed-record".to_string(),
        seed_committed_before_outcomes: true,
        familywise_confidence_level: 0.95,
        target_feasible_recall: 0.99,
        accepted: AcceptedCandidateManifest {
            candidates: accepted,
            manifest_sha256: digest('b'),
            evidence_ref: "synthetic-accepted-population-manifest".to_string(),
        },
        strata: vec![RejectStratum {
            stratum_id: "rejected".to_string(),
            stratum_definition_sha256: digest('c'),
            population_manifest_sha256: digest('d'),
            selection_receipt_ref: "synthetic-selection-receipt".to_string(),
            rejected_candidate_ids: rejected_ids,
            sample_size: 3,
            samples,
        }],
    };

    let scenarios = vec![
        ScenarioDefinition {
            scenario_id: "nominal".to_string(),
            scenario_sha256: digest('e'),
            incumbent_frontier_sha256: digest('f'),
            incumbent_frontier: vec![
                IncumbentFrontierPoint {
                    candidate_id: "nominal-incumbent-a".to_string(),
                    objectives: objective_values(100.0, 0.90),
                },
                IncumbentFrontierPoint {
                    candidate_id: "nominal-incumbent-b".to_string(),
                    objectives: objective_values(120.0, 0.95),
                },
            ],
        },
        ScenarioDefinition {
            scenario_id: "stress".to_string(),
            scenario_sha256: digest('1'),
            incumbent_frontier_sha256: digest('2'),
            incumbent_frontier: vec![
                IncumbentFrontierPoint {
                    candidate_id: "stress-incumbent-a".to_string(),
                    objectives: objective_values(110.0, 0.88),
                },
                IncumbentFrontierPoint {
                    candidate_id: "stress-incumbent-b".to_string(),
                    objectives: objective_values(125.0, 0.93),
                },
            ],
        },
    ];

    let accepted_bundles = vec![
        CandidateScenarioBundle {
            candidate_id: "accepted-0".to_string(),
            evaluations: vec![
                evaluation(
                    "nominal",
                    "accepted-0",
                    AuditOutcome::ConfirmedFeasible,
                    true,
                    objective_values(95.0, 0.94),
                    "synthetic-primary-run-accepted-0".to_string(),
                ),
                evaluation(
                    "stress",
                    "accepted-0",
                    AuditOutcome::ConfirmedFeasible,
                    true,
                    objective_values(150.0, 0.70),
                    "synthetic-stress-run-accepted-0".to_string(),
                ),
            ],
        },
        CandidateScenarioBundle {
            candidate_id: "accepted-1".to_string(),
            evaluations: vec![
                evaluation(
                    "nominal",
                    "accepted-1",
                    AuditOutcome::Failed,
                    true,
                    Vec::new(),
                    "synthetic-primary-run-accepted-1".to_string(),
                ),
                evaluation(
                    "stress",
                    "accepted-1",
                    AuditOutcome::Unsupported,
                    true,
                    Vec::new(),
                    "synthetic-stress-run-accepted-1".to_string(),
                ),
            ],
        },
    ];

    let rejected_bundles = selected
        .iter()
        .map(|candidate_id| {
            let (nominal_outcome, nominal_values, stress_outcome, stress_in_domain, stress_values) =
                match candidate_id.as_str() {
                    "rejected-0" => (
                        AuditOutcome::ConfirmedFeasible,
                        objective_values(95.0, 0.94),
                        AuditOutcome::ConfirmedFeasible,
                        true,
                        objective_values(100.0, 0.96),
                    ),
                    "rejected-1" => (
                        AuditOutcome::ConfirmedInfeasible,
                        Vec::new(),
                        AuditOutcome::ConfirmedFeasible,
                        true,
                        objective_values(105.0, 0.97),
                    ),
                    _ => (
                        AuditOutcome::NonConverged,
                        Vec::new(),
                        AuditOutcome::Inconclusive,
                        false,
                        Vec::new(),
                    ),
                };
            CandidateScenarioBundle {
                candidate_id: candidate_id.clone(),
                evaluations: vec![
                    evaluation(
                        "nominal",
                        candidate_id,
                        nominal_outcome,
                        true,
                        nominal_values,
                        format!("synthetic-primary-run-{candidate_id}"),
                    ),
                    evaluation(
                        "stress",
                        candidate_id,
                        stress_outcome,
                        stress_in_domain,
                        stress_values,
                        format!("synthetic-stress-run-{candidate_id}"),
                    ),
                ],
            }
        })
        .collect();

    let plan = ScenarioOpportunityPlan {
        campaign_id: "synthetic-two-scenario-opportunity-v1".to_string(),
        scenario_manifest_sha256: digest('3'),
        objective_manifest_sha256: digest('4'),
        opportunity_rule_sha256: digest('5'),
        primary_scenario_id: "nominal".to_string(),
        target_opportunity_recall: 0.99,
        base_plan,
        objective_definitions: vec![
            ObjectiveDefinition {
                objective_id: "lifetime-cost".to_string(),
                unit: "MUSD_2026".to_string(),
                direction: ObjectiveDirection::Minimize,
                epsilon: 1.0,
            },
            ObjectiveDefinition {
                objective_id: "availability".to_string(),
                unit: "fraction".to_string(),
                direction: ObjectiveDirection::Maximize,
                epsilon: 0.01,
            },
        ],
        scenarios,
        accepted_bundles,
        rejected_strata: vec![StratumScenarioBundles {
            stratum_id: "rejected".to_string(),
            bundles: rejected_bundles,
        }],
    };

    match audit_scenario_opportunities(&plan) {
        Ok(report) => {
            println!("contract_version={}", report.contract_version);
            println!("campaign={}", report.campaign_id);
            println!("primary_scenario={}", report.primary_scenario_id);
            println!("scenario_count={}", report.scenarios.len());
            println!(
                "simultaneous_bounds={},familywise_confidence={:.3}",
                report.simultaneous_bound_count,
                report.familywise_confidence_level
            );
            println!(
                "any_scenario_opportunity_recall_lower_bound={:.6}",
                report.union
                    .conditional_any_scenario_recall_lower_bound
                    .unwrap_or(0.0)
            );
            for scenario in &report.scenarios {
                println!(
                    "scenario={},opportunities={},unresolved={},missed_upper={},recall_lower={:.6}",
                    scenario.scenario_id,
                    scenario.rejected_opportunities_observed,
                    scenario.rejected_unresolved_observed,
                    scenario.missed_opportunities_upper_bound,
                    scenario.conditional_opportunity_recall_lower_bound.unwrap_or(0.0)
                );
            }
            println!(
                "SCOPE: synthetic scenario-family audit only; not robust physical validation or global Pareto recall"
            );
        }
        Err(errors) => {
            eprintln!("scenario opportunity audit refused: {errors:?}");
            std::process::exit(1);
        }
    }
}
