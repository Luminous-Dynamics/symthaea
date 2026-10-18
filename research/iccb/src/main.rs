use iccb_evaluator::{
    CandidateScenario, DecisionOption, EvidenceClaim, EvidenceState, Mandate,
    cash_maximizing_feasible_baseline, pareto_frontier,
    policy::{CapitalPolicyScenario, DemandCase},
    robust_decision::{RobustDecisionScenario, RobustPayoff, StateProbabilityBound},
    staging::TrancheRequest,
    uncertainty_contract::{
        UncertaintyAssessmentRef, UncertaintyEvidenceKind,
    },
    value_of_information::{
        DecisionPayoff, InformationState, SignalLikelihood, SignalOutcome,
        ValueOfInformationScenario,
    },
};

fn main() {
    let claim = EvidenceClaim {
        id: "capex-source-001".into(),
        state: EvidenceState::Qualified,
        source_id: Some("synthetic-fixture-v1".into()),
        observed_day: Some(1),
        valid_through_day: Some(30),
    };
    let scenario = CandidateScenario {
        id: "demo-maintenance-gate".into(),
        mandate: Mandate {
            budget_limit_minor: 100_000,
            minimum_liquidity_minor: 10_000,
            maximum_maintenance_backlog_delta_minor: 5_000,
            evaluation_day: 12,
            required_evidence: true,
        },
        options: vec![
            DecisionOption {
                id: "high-return-defers-maintenance".into(),
                expected_net_cash_minor: 50_000,
                reported_valuation_change_minor: 500_000,
                required_capital_minor: 60_000,
                available_liquidity_after_minor: 20_000,
                productive_capacity_delta: 2,
                maintenance_backlog_delta_minor: 8_000,
                dependency_ids: vec!["grid-a".into()],
                evidence: vec![claim.clone()],
                authority_granted: true,
            },
            DecisionOption {
                id: "balanced-cash-option".into(),
                expected_net_cash_minor: 42_000,
                reported_valuation_change_minor: 150_000,
                required_capital_minor: 55_000,
                available_liquidity_after_minor: 21_000,
                productive_capacity_delta: 5,
                maintenance_backlog_delta_minor: 2_000,
                dependency_ids: vec!["grid-a".into()],
                evidence: vec![claim.clone()],
                authority_granted: true,
            },
            DecisionOption {
                id: "lower-return-maintained-expansion".into(),
                expected_net_cash_minor: 35_000,
                reported_valuation_change_minor: 20_000,
                required_capital_minor: 50_000,
                available_liquidity_after_minor: 22_000,
                productive_capacity_delta: 12,
                maintenance_backlog_delta_minor: 1_000,
                dependency_ids: vec!["grid-a".into()],
                evidence: vec![claim],
                authority_granted: true,
            },
        ],
    };

    let receipt = scenario.evaluate();
    let baseline = cash_maximizing_feasible_baseline(&receipt.evaluated_options);
    let frontier = pareto_frontier(&receipt.evaluated_options);

    println!("ICCB synthetic evaluation: {}", receipt.scenario_id);
    println!("No composite score is calculated; hard-gate failures cannot be offset by return.");
    println!("Cash-maximizing feasible baseline: {:?}", baseline);
    println!("Constraint-gated Pareto frontier: {:?}", frontier);
    for evaluated in receipt.evaluated_options {
        println!(
            "{}: accepted={}, rejection_reasons={:?}, net_cash_minor={}, valuation_change_minor={}, capacity_delta={}, maintenance_delta_minor={}, declared_dependencies={}",
            evaluated.option_id,
            evaluated.accepted,
            evaluated.rejection_reasons,
            evaluated.outcomes.expected_net_cash_minor,
            evaluated.outcomes.reported_valuation_change_minor,
            evaluated.outcomes.productive_capacity_delta,
            evaluated.outcomes.maintenance_backlog_delta_minor,
            evaluated.outcomes.declared_dependency_count,
        );
    }

    let tranche = TrancheRequest {
        tranche_id: "stage-2-after-inspection".into(),
        amount_minor: 25_000,
        previously_committed_minor: 30_000,
        total_commitment_cap_minor: 80_000,
        evaluation_day: 12,
        milestone_satisfied: false,
        authority_granted: true,
        milestone_evidence: vec![EvidenceClaim {
            id: "inspection-001".into(),
            state: EvidenceState::Stale,
            source_id: Some("synthetic-inspection-record".into()),
            observed_day: Some(1),
            valid_through_day: Some(7),
        }],
    }
    .evaluate_release();
    println!(
        "Next tranche approved: {}, reasons: {:?}",
        tranche.approved, tranche.rejection_reasons
    );

    // These probabilities and cashflows are synthetic and exist only to
    // demonstrate that the best policy changes with delay costs and demand.
    let policy_scenario = CapitalPolicyScenario {
        id: "synthetic-stage-vs-wait-policy".into(),
        capital_budget_minor: 120,
        full_commitment_minor: 100,
        staged_commitment_minor: 15,
        expansion_commitment_minor: 90,
        wait_cost_minor: 70,
        discount_factor_bps: 9_000,
        demand_cases: vec![
            DemandCase {
                id: "high-demand".into(),
                probability_bps: 5_000,
                full_capacity_cashflow_period1_minor: 40,
                full_capacity_cashflow_period2_minor: 200,
                staged_cashflow_period1_minor: 30,
                expanded_cashflow_period2_minor: 180,
            },
            DemandCase {
                id: "low-demand".into(),
                probability_bps: 5_000,
                full_capacity_cashflow_period1_minor: 10,
                full_capacity_cashflow_period2_minor: 20,
                staged_cashflow_period1_minor: 10,
                expanded_cashflow_period2_minor: 20,
            },
        ],
    };
    match policy_scenario.evaluate() {
        Ok(comparison) => {
            println!(
                "Synthetic policy comparison: best={:?}, improvement_vs_one_shot_minor={:?}",
                comparison.best_policy_within_catalog,
                comparison.improvement_vs_one_shot_minor,
            );
            for assessment in comparison.assessments {
                println!(
                    "  policy={:?}, feasible={}, expected_npv_minor={:?}, max_commitment_minor={}",
                    assessment.policy,
                    assessment.feasible,
                    assessment.expected_npv_minor,
                    assessment.maximum_authorized_commitment_minor,
                );
            }
        }
        Err(error) => eprintln!("Synthetic policy comparison rejected invalid input: {error:?}"),
    }

    let voi_scenario = ValueOfInformationScenario {
        id: "synthetic-diligence-survey".into(),
        information_cost_minor: 20,
        delay_cost_minor: 0,
        financing_cost_minor: 0,
        states: vec![
            InformationState {
                id: "high-demand".into(),
                prior_probability_bps: 5_000,
            },
            InformationState {
                id: "low-demand".into(),
                prior_probability_bps: 5_000,
            },
        ],
        signal_outcomes: vec![
            SignalOutcome {
                id: "survey-says-high".into(),
                likelihoods: vec![
                    SignalLikelihood {
                        state_id: "high-demand".into(),
                        likelihood_bps: 10_000,
                    },
                    SignalLikelihood {
                        state_id: "low-demand".into(),
                        likelihood_bps: 0,
                    },
                ],
            },
            SignalOutcome {
                id: "survey-says-low".into(),
                likelihoods: vec![
                    SignalLikelihood {
                        state_id: "high-demand".into(),
                        likelihood_bps: 0,
                    },
                    SignalLikelihood {
                        state_id: "low-demand".into(),
                        likelihood_bps: 10_000,
                    },
                ],
            },
        ],
        action_ids: vec!["build".into(), "avoid".into()],
        payoffs: vec![
            DecisionPayoff {
                action_id: "build".into(),
                state_id: "high-demand".into(),
                net_value_minor: 100,
            },
            DecisionPayoff {
                action_id: "build".into(),
                state_id: "low-demand".into(),
                net_value_minor: -100,
            },
            DecisionPayoff {
                action_id: "avoid".into(),
                state_id: "high-demand".into(),
                net_value_minor: 0,
            },
            DecisionPayoff {
                action_id: "avoid".into(),
                state_id: "low-demand".into(),
                net_value_minor: 0,
            },
        ],
    };
    match voi_scenario.evaluate() {
        Ok(report) => println!(
            "Synthetic diligence value: baseline={:?}, gross_evsi_minor={}, survey_cost_minor={}, delay_cost_minor={}, financing_cost_minor={}, total_cost_minor={}, net_evsi_minor={}, evpi_minor={}, worthwhile={}",
            report.baseline_action_id,
            report.gross_evsi_minor,
            report.information_cost_minor,
            report.delay_cost_minor,
            report.financing_cost_minor,
            report.total_cost_minor,
            report.net_evsi_minor,
            report.evpi_minor,
            report.worthwhile_at_declared_cost,
        ),
        Err(error) => eprintln!("Synthetic value-of-information scenario rejected: {error:?}"),
    }

    // Probability bounds are deliberately broad enough to make the action
    // ranking ambiguous; minimax regret is printed separately, not executed.
    let robust_scenario = RobustDecisionScenario {
        id: "synthetic-ambiguous-capital-choice".into(),
        state_probability_bounds: vec![
            StateProbabilityBound {
                state_id: "high-demand".into(),
                lower_bps: 3_000,
                upper_bps: 7_000,
            },
            StateProbabilityBound {
                state_id: "low-demand".into(),
                lower_bps: 3_000,
                upper_bps: 7_000,
            },
        ],
        action_ids: vec!["build".into(), "avoid".into()],
        payoffs: vec![
            RobustPayoff {
                action_id: "build".into(),
                state_id: "high-demand".into(),
                payoff_minor: 10,
            },
            RobustPayoff {
                action_id: "build".into(),
                state_id: "low-demand".into(),
                payoff_minor: -5,
            },
            RobustPayoff {
                action_id: "avoid".into(),
                state_id: "high-demand".into(),
                payoff_minor: 0,
            },
            RobustPayoff {
                action_id: "avoid".into(),
                state_id: "low-demand".into(),
                payoff_minor: 0,
            },
        ],
    };
    match robust_scenario.evaluate() {
        Ok(report) => {
            println!(
                "Synthetic robust choice: status={:?}, minimax_regret_actions={:?}",
                report.status, report.minimax_regret_action_ids
            );
            for action in report.action_envelopes {
                println!(
                    "  action={}, expected_value_bounds=[{}, {}], worst_case_regret_upper={}",
                    action.action_id,
                    action.lower_expected_value_minor,
                    action.upper_expected_value_minor,
                    action.worst_case_expected_regret_upper_minor,
                );
            }
        }
        Err(error) => eprintln!("Synthetic robust decision scenario rejected: {error:?}"),
    }

    // The source semantics choose the analysis route. Incompatible uncertainty
    // types are never silently converted to point probabilities.
    for evidence_kind in [
        UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "synthetic-bounds-receipt".into(),
        },
        UncertaintyEvidenceKind::OutcomeCompatibilitySet {
            compatibility_receipt_id: "synthetic-compatibility-receipt".into(),
        },
        UncertaintyEvidenceKind::ModelDiscrepancyEnvelope {
            envelope_receipt_id: "synthetic-discrepancy-receipt".into(),
        },
        UncertaintyEvidenceKind::OutOfDomain {
            reason: "synthetic-domain-gap".into(),
        },
    ] {
        let reference = UncertaintyAssessmentRef {
            scenario_id: "synthetic-uncertainty-router".into(),
            producer_version: "synthetic-producer-v1".into(),
            source_manifest_digest: "sha256:synthetic-manifest".into(),
            evidence_kind,
        };
        match reference.route() {
            Ok(receipt) => println!(
                "Uncertainty route: scenario={}, route={:?}, evidence_ref={}, limit={}",
                receipt.scenario_id,
                receipt.route,
                receipt.evidence_reference,
                receipt.limitation,
            ),
            Err(error) => eprintln!("Uncertainty routing rejected invalid packet: {error:?}"),
        }
    }
}
