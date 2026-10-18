//! Frozen, deterministic public development fixtures for ICCB v0.
//!
//! These fixtures are visible examples for contract testing. They are not a
//! hidden holdout set and must not be reported as independent evaluation data.

use super::{CandidateScenario, DecisionOption, EvidenceClaim, EvidenceState, Mandate};
use std::collections::BTreeSet;

fn claim(id: &str, state: EvidenceState) -> EvidenceClaim {
    EvidenceClaim {
        id: id.to_owned(),
        state,
        source_id: Some("iccb-synthetic-fixture-v0".to_owned()),
        observed_day: Some(1),
        valid_through_day: Some(20),
    }
}

fn mandate() -> Mandate {
    Mandate {
        budget_limit_minor: 100_000,
        minimum_liquidity_minor: 10_000,
        maximum_maintenance_backlog_delta_minor: 5_000,
        evaluation_day: 12,
        required_evidence: true,
    }
}

fn option(
    id: &str,
    expected_net_cash_minor: i64,
    reported_valuation_change_minor: i64,
    required_capital_minor: i64,
    available_liquidity_after_minor: i64,
    productive_capacity_delta: i64,
    maintenance_backlog_delta_minor: i64,
    dependency_ids: &[&str],
    evidence_state: EvidenceState,
) -> DecisionOption {
    DecisionOption {
        id: id.to_owned(),
        expected_net_cash_minor,
        reported_valuation_change_minor,
        required_capital_minor,
        available_liquidity_after_minor,
        productive_capacity_delta,
        maintenance_backlog_delta_minor,
        dependency_ids: dependency_ids.iter().map(|id| (*id).to_owned()).collect(),
        evidence: vec![claim("source-1", evidence_state)],
        authority_granted: true,
    }
}

fn scenario(id: &str, candidate_option: DecisionOption) -> CandidateScenario {
    CandidateScenario {
        id: id.to_owned(),
        mandate: mandate(),
        options: vec![candidate_option],
    }
}

/// A feasible investment with positive capacity growth and bounded maintenance.
pub fn productive_expansion() -> CandidateScenario {
    scenario(
        "ICCB-PUBLIC-001-productive-expansion",
        option(
            "expansion-a",
            45_000,
            12_000,
            60_000,
            25_000,
            25,
            1_000,
            &["grid-west"],
            EvidenceState::Qualified,
        ),
    )
}

/// Valuation growth is intentionally high while cash flow and capacity stay low.
pub fn asset_price_only_boom() -> CandidateScenario {
    scenario(
        "ICCB-PUBLIC-002-asset-price-only-boom",
        option(
            "valuation-only",
            5_000,
            900_000,
            20_000,
            15_000,
            0,
            500,
            &["grid-west"],
            EvidenceState::Qualified,
        ),
    )
}

/// The option breaches a hard maintenance cap despite its expected cash flow.
pub fn hidden_maintenance_burden() -> CandidateScenario {
    scenario(
        "ICCB-PUBLIC-003-hidden-maintenance",
        option(
            "deferred-replacement",
            80_000,
            30_000,
            60_000,
            25_000,
            5,
            9_000,
            &["grid-west"],
            EvidenceState::Qualified,
        ),
    )
}

/// Declared dependency links are reported without inferring criticality.
pub fn declared_dependency_links() -> CandidateScenario {
    scenario(
        "ICCB-PUBLIC-004-declared-dependency-links",
        option(
            "shared-utilities",
            30_000,
            10_000,
            50_000,
            25_000,
            10,
            1_000,
            &["grid-west", "water-east", "grid-west"],
            EvidenceState::Qualified,
        ),
    )
}

/// A stale source must fail closed when the mandate requires qualified evidence.
pub fn stale_evidence() -> CandidateScenario {
    scenario(
        "ICCB-PUBLIC-005-stale-evidence",
        option(
            "stale-source",
            60_000,
            8_000,
            40_000,
            20_000,
            15,
            500,
            &["water-east"],
            EvidenceState::Stale,
        ),
    )
}

/// Stable-order, public development fixture registry.
pub fn all() -> Vec<CandidateScenario> {
    vec![
        productive_expansion(),
        asset_price_only_boom(),
        hidden_maintenance_burden(),
        declared_dependency_links(),
        stale_evidence(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RejectionReason;

    #[test]
    fn registry_contains_five_unique_frozen_scenario_ids() {
        let fixtures = all();
        let ids = fixtures
            .iter()
            .map(|fixture| fixture.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(fixtures.len(), 5);
        assert_eq!(ids.len(), 5);
        assert!(ids.iter().all(|id| id.starts_with("ICCB-PUBLIC-")));
    }

    #[test]
    fn productive_expansion_passes_hard_gates() {
        let result = productive_expansion().evaluate();
        assert!(result.evaluated_options[0].accepted);
    }

    #[test]
    fn valuation_boom_does_not_imply_cash_flow_or_capacity() {
        let result = asset_price_only_boom().evaluate();
        let outcomes = &result.evaluated_options[0].outcomes;
        assert_eq!(outcomes.reported_valuation_change_minor, 900_000);
        assert_eq!(outcomes.expected_net_cash_minor, 5_000);
        assert_eq!(outcomes.productive_capacity_delta, 0);
    }

    #[test]
    fn maintenance_burden_fixture_fails_closed() {
        let result = hidden_maintenance_burden().evaluate();
        assert!(!result.evaluated_options[0].accepted);
        assert!(
            result.evaluated_options[0]
                .rejection_reasons
                .contains(&RejectionReason::MaintenanceLimitExceeded)
        );
    }

    #[test]
    fn declared_dependency_links_is_deduplicated() {
        let result = declared_dependency_links().evaluate();
        assert_eq!(
            result.evaluated_options[0]
                .outcomes
                .declared_dependency_count,
            2
        );
    }

    #[test]
    fn stale_evidence_fixture_fails_closed() {
        let result = stale_evidence().evaluate();
        assert!(!result.evaluated_options[0].accepted);
        assert!(
            result.evaluated_options[0]
                .rejection_reasons
                .contains(&RejectionReason::RequiredEvidenceMissingOrUnusable)
        );
    }
}
