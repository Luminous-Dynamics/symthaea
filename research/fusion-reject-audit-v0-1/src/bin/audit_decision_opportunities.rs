use std::collections::BTreeMap;

use fusion_reject_audit::{
    AcceptedCandidate, AcceptedCandidateManifest, AuditOutcome, FrozenCandidateUniverse,
    RejectAuditPlan, RejectStratum,
    decision_opportunity::{
        audit_decision_opportunities, DecisionEvaluation, DecisionOpportunityPlan, IncumbentFrontierPoint, ObjectiveDefinition,
        ObjectiveDirection, ObjectiveValue, StratumDecisionEvaluations,
    },
    deterministic_audit_sample_ids,
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
    candidate_id: &str,
    outcome: AuditOutcome,
    within_declared_domain: bool,
    objectives: Vec<ObjectiveValue>,
) -> DecisionEvaluation {
    DecisionEvaluation {
        candidate_id: candidate_id.to_string(),
        reference_outcome: outcome,
        within_declared_domain,
        objectives,
        evidence_ref: format!("synthetic-reference-run-{candidate_id}"),
    }
}

fn main() {
    let seed = 0xdec1_5100_0f01_2026;
    let rejected_ids = vec![
        "rejected-0".to_string(),
        "rejected-1".to_string(),
        "rejected-2".to_string(),
        "rejected-3".to_string(),
    ];
    let selected = deterministic_audit_sample_ids("reject-all", &rejected_ids, 4, seed)
        .expect("frozen audit sample must be valid");

    let rejected_outcomes = BTreeMap::from([
        (
            "rejected-0",
            (
                AuditOutcome::ConfirmedFeasible,
                true,
                objective_values(95.0, 0.94),
            ),
        ),
        ("rejected-1", (AuditOutcome::NonConverged, true, Vec::new())),
        (
            "rejected-2",
            (AuditOutcome::Inconclusive, false, Vec::new()),
        ),
        (
            "rejected-3",
            (
                AuditOutcome::ConfirmedFeasible,
                true,
                objective_values(140.0, 0.98),
            ),
        ),
    ]);

    let rejected_samples = selected
        .iter()
        .map(|candidate_id| {
            let (outcome, _, _) = rejected_outcomes
                .get(candidate_id.as_str())
                .expect("each frozen sample has an explicit synthetic outcome");
            fusion_reject_audit::AuditSample {
                candidate_id: candidate_id.clone(),
                outcome: *outcome,
                evidence_ref: format!("synthetic-reference-run-{candidate_id}"),
            }
        })
        .collect();

    let accepted_rows = vec![
        (
            "accepted-0",
            AuditOutcome::ConfirmedFeasible,
            true,
            objective_values(90.0, 0.93),
        ),
        ("accepted-1", AuditOutcome::Failed, true, Vec::new()),
        (
            "accepted-2",
            AuditOutcome::ConfirmedInfeasible,
            true,
            Vec::new(),
        ),
        (
            "accepted-3",
            AuditOutcome::ConfirmedFeasible,
            true,
            objective_values(150.0, 0.70),
        ),
    ];

    let accepted_manifest = accepted_rows
        .iter()
        .map(|(candidate_id, outcome, _, _)| AcceptedCandidate {
            candidate_id: (*candidate_id).to_string(),
            outcome: *outcome,
            evidence_ref: format!("synthetic-reference-run-{candidate_id}"),
        })
        .collect::<Vec<_>>();

    let mut frozen_ids = accepted_rows
        .iter()
        .map(|(candidate_id, _, _, _)| (*candidate_id).to_string())
        .collect::<Vec<_>>();
    frozen_ids.extend(rejected_ids.iter().cloned());
    frozen_ids.sort();

    let base_plan = RejectAuditPlan {
        frozen_universe: FrozenCandidateUniverse {
            candidate_ids: frozen_ids,
            manifest_sha256: digest('a'),
        },
        sampling_seed: seed,
        seed_commitment_ref: "synthetic-frozen-seed-record".to_string(),
        seed_committed_before_outcomes: true,
        familywise_confidence_level: 0.95,
        target_feasible_recall: 0.99,
        accepted: AcceptedCandidateManifest {
            candidates: accepted_manifest,
            manifest_sha256: digest('b'),
            evidence_ref: "synthetic-accepted-population-manifest".to_string(),
        },
        strata: vec![RejectStratum {
            stratum_id: "reject-all".to_string(),
            stratum_definition_sha256: digest('c'),
            population_manifest_sha256: digest('d'),
            selection_receipt_ref: "synthetic-selection-receipt".to_string(),
            rejected_candidate_ids: rejected_ids,
            sample_size: 4,
            samples: rejected_samples,
        }],
    };

    let accepted_evaluations = accepted_rows
        .iter()
        .map(|(candidate_id, outcome, in_domain, objectives)| {
            evaluation(candidate_id, *outcome, *in_domain, objectives.clone())
        })
        .collect();

    let rejected_evaluations = selected
        .iter()
        .map(|candidate_id| {
            let (outcome, in_domain, objectives) = rejected_outcomes
                .get(candidate_id.as_str())
                .expect("each frozen sample has an explicit synthetic outcome");
            evaluation(candidate_id, *outcome, *in_domain, objectives.clone())
        })
        .collect();

    let plan = DecisionOpportunityPlan {
        campaign_id: "synthetic-decision-opportunity-demo-v1".to_string(),
        objective_manifest_sha256: digest('e'),
        opportunity_rule_sha256: digest('f'),
        incumbent_frontier_sha256: digest('1'),
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
        incumbent_frontier: vec![
            IncumbentFrontierPoint {
                candidate_id: "incumbent-a".to_string(),
                objectives: objective_values(100.0, 0.90),
            },
            IncumbentFrontierPoint {
                candidate_id: "incumbent-b".to_string(),
                objectives: objective_values(120.0, 0.95),
            },
        ],
        accepted_evaluations,
        rejected_evaluations: vec![StratumDecisionEvaluations {
            stratum_id: "reject-all".to_string(),
            evaluations: rejected_evaluations,
        }],
    };

    match audit_decision_opportunities(&plan) {
        Ok(report) => {
            println!("decision_rule={}", report.decision_rule);
            println!("campaign={}", report.campaign_id);
            println!(
                "familywise_confidence={:.3}",
                report.familywise_confidence_level
            );
            println!(
                "accepted_opportunities={},accepted_no_opportunity={},accepted_unresolved={}",
                report.accepted_confirmed_opportunities,
                report.accepted_confirmed_no_opportunity,
                report.accepted_unresolved
            );
            println!(
                "rejected_population={},audited_rejected={},observed_opportunities={},unresolved={}",
                report.rejected_population,
                report.audited_rejected_candidates,
                report.rejected_opportunities_observed,
                report.rejected_unresolved_observed
            );
            println!(
                "missed_decision_opportunities_upper_bound={}",
                report.missed_decision_opportunities_upper_bound
            );
            match report.conditional_opportunity_recall_lower_bound {
                Some(value) => println!("conditional_opportunity_recall_lower_bound={value:.6}"),
                None => println!("conditional_opportunity_recall_lower_bound=unavailable"),
            }
            match report.feasible_recall_lower_bound {
                Some(value) => println!("feasible_recall_lower_bound={value:.6}"),
                None => println!("feasible_recall_lower_bound=unavailable"),
            }
            println!(
                "target_met={},feasibility_target_met={}",
                report.opportunity_target_met, report.feasibility_target_met
            );
            println!(
                "SCOPE: synthetic conditional reference-front audit only; not global Pareto recall, solver or physical evidence"
            );
        }
        Err(errors) => {
            eprintln!("decision-opportunity audit refused: {errors:?}");
            std::process::exit(1);
        }
    }
}
