use iccb_evaluator::{
    CandidateScenario, DecisionOption, EvidenceClaim, EvidenceState, Mandate,
};
use std::collections::BTreeSet;

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
        known_critical_dependency_ids: ["grid-a".to_owned()].into_iter().collect::<BTreeSet<_>>(),
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
    println!("ICCB synthetic evaluation: {}", receipt.scenario_id);
    println!("No composite score is calculated; hard-gate failures cannot be offset by return.");
    for evaluated in receipt.evaluated_options {
        println!(
            "{}: accepted={}, rejection_reasons={:?}, net_cash_minor={}, valuation_change_minor={}, capacity_delta={}, maintenance_delta_minor={}, critical_dependencies={}",
            evaluated.option_id,
            evaluated.accepted,
            evaluated.rejection_reasons,
            evaluated.outcomes.expected_net_cash_minor,
            evaluated.outcomes.reported_valuation_change_minor,
            evaluated.outcomes.productive_capacity_delta,
            evaluated.outcomes.maintenance_backlog_delta_minor,
            evaluated.outcomes.critical_dependency_count,
        );
    }
}
