//! ICCB's initial deterministic reference kernel.
//!
//! This is a synthetic protocol prototype, not a financial adviser, investment
//! recommender, or validated model of a real asset portfolio. It intentionally
//! emits separate outcome dimensions rather than a composite score.

use std::collections::BTreeSet;

pub mod fixtures;
pub mod policy;
pub mod robust_decision;
pub mod staging;
pub mod value_of_information;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceState {
    Qualified,
    Stale,
    Conflicting,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceClaim {
    pub id: String,
    pub state: EvidenceState,
    pub source_id: Option<String>,
    pub observed_day: Option<u32>,
    pub valid_through_day: Option<u32>,
}

impl EvidenceClaim {
    /// A claim is usable only when its state, source, and validity window qualify.
    pub fn is_usable(&self, evaluation_day: u32) -> bool {
        self.state == EvidenceState::Qualified
            && self
                .source_id
                .as_ref()
                .is_some_and(|source| !source.trim().is_empty())
            && self.observed_day.is_some_and(|day| day <= evaluation_day)
            && self
                .valid_through_day
                .is_some_and(|day| day >= evaluation_day)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionOption {
    pub id: String,
    /// Integer minor currency units; no floating-point arithmetic for money.
    pub expected_net_cash_minor: i64,
    pub reported_valuation_change_minor: i64,
    pub required_capital_minor: i64,
    pub available_liquidity_after_minor: i64,
    /// A scenario-defined unit (e.g. extra service capacity); not a monetary value.
    pub productive_capacity_delta: i64,
    pub maintenance_backlog_delta_minor: i64,
    pub dependency_ids: Vec<String>,
    pub evidence: Vec<EvidenceClaim>,
    pub authority_granted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mandate {
    pub budget_limit_minor: i64,
    pub minimum_liquidity_minor: i64,
    pub maximum_maintenance_backlog_delta_minor: i64,
    pub evaluation_day: u32,
    pub required_evidence: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateScenario {
    pub id: String,
    pub mandate: Mandate,
    pub options: Vec<DecisionOption>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectionReason {
    NegativeCapitalRequirement,
    BudgetExceeded,
    LiquidityFloorBreached,
    MaintenanceLimitExceeded,
    Unauthorized,
    RequiredEvidenceMissingOrUnusable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeVector {
    pub expected_net_cash_minor: i64,
    pub reported_valuation_change_minor: i64,
    pub available_liquidity_after_minor: i64,
    pub productive_capacity_delta: i64,
    pub maintenance_backlog_delta_minor: i64,
    pub declared_dependency_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionEvaluation {
    pub option_id: String,
    pub accepted: bool,
    pub rejection_reasons: Vec<RejectionReason>,
    pub outcomes: OutcomeVector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationReceipt {
    pub scenario_id: String,
    pub evaluation_day: u32,
    pub evaluated_options: Vec<OptionEvaluation>,
}

impl CandidateScenario {
    /// Enforce hard constraints before returning an option result.
    ///
    /// This function deliberately does not rank or select options. A higher
    /// expected return can never offset a hard-gate violation.
    pub fn evaluate(&self) -> EvaluationReceipt {
        let mut evaluated_options = Vec::with_capacity(self.options.len());

        for option in &self.options {
            let mut rejection_reasons = Vec::new();

            if option.required_capital_minor < 0 {
                rejection_reasons.push(RejectionReason::NegativeCapitalRequirement);
            }
            if option.required_capital_minor > self.mandate.budget_limit_minor {
                rejection_reasons.push(RejectionReason::BudgetExceeded);
            }
            if option.available_liquidity_after_minor < self.mandate.minimum_liquidity_minor {
                rejection_reasons.push(RejectionReason::LiquidityFloorBreached);
            }
            if option.maintenance_backlog_delta_minor
                > self.mandate.maximum_maintenance_backlog_delta_minor
            {
                rejection_reasons.push(RejectionReason::MaintenanceLimitExceeded);
            }
            if !option.authority_granted {
                rejection_reasons.push(RejectionReason::Unauthorized);
            }
            if self.mandate.required_evidence
                && (option.evidence.is_empty()
                    || option
                        .evidence
                        .iter()
                        .any(|claim| !claim.is_usable(self.mandate.evaluation_day)))
            {
                rejection_reasons.push(RejectionReason::RequiredEvidenceMissingOrUnusable);
            }

            // Deduplicate dependencies before counting overlaps with known critical
            // dependencies; repeating an ID must not inflate or reduce the count.
            // Count unique declared links only. Criticality is an evaluator-side
            // label and must not be embedded in the candidate-facing scenario.
            let declared_dependency_count = option
                .dependency_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len();

            evaluated_options.push(OptionEvaluation {
                option_id: option.id.clone(),
                accepted: rejection_reasons.is_empty(),
                rejection_reasons,
                outcomes: OutcomeVector {
                    expected_net_cash_minor: option.expected_net_cash_minor,
                    reported_valuation_change_minor: option.reported_valuation_change_minor,
                    available_liquidity_after_minor: option.available_liquidity_after_minor,
                    productive_capacity_delta: option.productive_capacity_delta,
                    maintenance_backlog_delta_minor: option.maintenance_backlog_delta_minor,
                    declared_dependency_count,
                },
            });
        }

        EvaluationReceipt {
            scenario_id: self.id.clone(),
            evaluation_day: self.mandate.evaluation_day,
            evaluated_options,
        }
    }
}

/// Select the highest expected-cash option among options that already pass
/// all hard gates. This intentionally simple policy is a baseline to beat, not
/// a recommendation policy for real investments. Ties resolve by lexical ID.
pub fn cash_maximizing_feasible_baseline(evaluations: &[OptionEvaluation]) -> Option<String> {
    let mut best: Option<&OptionEvaluation> = None;

    for current in evaluations.iter().filter(|evaluation| evaluation.accepted) {
        let replace = match best {
            None => true,
            Some(previous) => {
                current.outcomes.expected_net_cash_minor > previous.outcomes.expected_net_cash_minor
                    || (current.outcomes.expected_net_cash_minor
                        == previous.outcomes.expected_net_cash_minor
                        && current.option_id < previous.option_id)
            }
        };

        if replace {
            best = Some(current);
        }
    }

    best.map(|evaluation| evaluation.option_id.clone())
}

/// Returns IDs of non-dominated, feasible options under a declared outcome
/// vector. Cash, liquidity, capacity, maintenance burden, and critical-dependency
/// overlap are compared coordinate-by-coordinate with no arbitrary weights.
/// Reported valuation change is deliberately excluded from dominance: it remains
/// visible in the outcome vector, but cannot substitute for cash flow or capacity.
pub fn pareto_frontier(evaluations: &[OptionEvaluation]) -> Vec<String> {
    let mut frontier = evaluations
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.accepted)
        .filter(|(candidate_index, candidate)| {
            !evaluations.iter().enumerate().any(|(other_index, other)| {
                *candidate_index != other_index
                    && other.accepted
                    && dominates(&other.outcomes, &candidate.outcomes)
            })
        })
        .map(|(_, evaluation)| evaluation.option_id.clone())
        .collect::<Vec<_>>();

    frontier.sort();
    frontier
}

fn dominates(left: &OutcomeVector, right: &OutcomeVector) -> bool {
    // Dependency-link count is descriptive only: fewer links can reflect missing
    // topology, not lower risk, so it must not establish Pareto dominance.
    let no_worse = left.expected_net_cash_minor >= right.expected_net_cash_minor
        && left.available_liquidity_after_minor >= right.available_liquidity_after_minor
        && left.productive_capacity_delta >= right.productive_capacity_delta
        && left.maintenance_backlog_delta_minor <= right.maintenance_backlog_delta_minor;

    let strictly_better = left.expected_net_cash_minor > right.expected_net_cash_minor
        || left.available_liquidity_after_minor > right.available_liquidity_after_minor
        || left.productive_capacity_delta > right.productive_capacity_delta
        || left.maintenance_backlog_delta_minor < right.maintenance_backlog_delta_minor;

    no_worse && strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualified_claim(id: &str) -> EvidenceClaim {
        EvidenceClaim {
            id: id.to_owned(),
            state: EvidenceState::Qualified,
            source_id: Some("synthetic-source-v1".to_owned()),
            observed_day: Some(3),
            valid_through_day: Some(12),
        }
    }

    fn base_mandate() -> Mandate {
        Mandate {
            budget_limit_minor: 10_000,
            minimum_liquidity_minor: 1_000,
            maximum_maintenance_backlog_delta_minor: 500,
            evaluation_day: 8,
            required_evidence: true,
        }
    }

    fn option(id: &str) -> DecisionOption {
        DecisionOption {
            id: id.to_owned(),
            expected_net_cash_minor: 4_000,
            reported_valuation_change_minor: 0,
            required_capital_minor: 5_000,
            available_liquidity_after_minor: 2_000,
            productive_capacity_delta: 10,
            maintenance_backlog_delta_minor: 100,
            dependency_ids: vec!["grid-west".to_owned()],
            evidence: vec![qualified_claim("claim-1")],
            authority_granted: true,
        }
    }

    fn scenario(options: Vec<DecisionOption>) -> CandidateScenario {
        CandidateScenario {
            id: "synthetic-test".to_owned(),
            mandate: base_mandate(),
            options,
        }
    }

    #[test]
    fn hard_constraint_rejects_high_return_unauthorized_option() {
        let mut risky = option("high-return-but-unauthorized");
        risky.expected_net_cash_minor = 9_000_000;
        risky.authority_granted = false;
        let result = scenario(vec![risky]).evaluate();
        let evaluated = &result.evaluated_options[0];

        assert!(!evaluated.accepted);
        assert!(
            evaluated
                .rejection_reasons
                .contains(&RejectionReason::Unauthorized)
        );
    }

    #[test]
    fn stale_evidence_fails_closed_when_required() {
        let mut stale = option("stale-evidence");
        stale.evidence[0].state = EvidenceState::Stale;
        let evaluated = &scenario(vec![stale]).evaluate().evaluated_options[0];

        assert!(!evaluated.accepted);
        assert!(
            evaluated
                .rejection_reasons
                .contains(&RejectionReason::RequiredEvidenceMissingOrUnusable)
        );
    }

    #[test]
    fn evidence_with_missing_source_or_expired_window_is_unusable() {
        let mut missing_source = qualified_claim("missing-source");
        missing_source.source_id = None;
        assert!(!missing_source.is_usable(8));

        let mut expired = qualified_claim("expired");
        expired.valid_through_day = Some(7);
        assert!(!expired.is_usable(8));
    }

    #[test]
    fn declared_dependency_count_is_not_a_criticality_score() {
        let result = scenario(vec![option("depends-on-grid")]).evaluate();
        let metrics = &result.evaluated_options[0].outcomes;

        assert_eq!(metrics.declared_dependency_count, 1);
        assert_eq!(metrics.expected_net_cash_minor, 4_000);
        assert_eq!(metrics.productive_capacity_delta, 10);
    }

    #[test]
    fn repeated_dependency_ids_do_not_inflate_exposure() {
        let mut investment = option("duplicate-ids");
        investment.dependency_ids = vec![
            "grid-west".to_owned(),
            "grid-west".to_owned(),
            "unknown-link".to_owned(),
        ];
        let result = scenario(vec![investment]).evaluate();

        assert_eq!(
            result.evaluated_options[0]
                .outcomes
                .declared_dependency_count,
            1
        );
    }

    #[test]
    fn maintenance_limit_is_a_hard_gate() {
        let mut deferred = option("deferred-maintenance");
        deferred.maintenance_backlog_delta_minor = 501;
        let evaluated = &scenario(vec![deferred]).evaluate().evaluated_options[0];

        assert!(!evaluated.accepted);
        assert!(
            evaluated
                .rejection_reasons
                .contains(&RejectionReason::MaintenanceLimitExceeded)
        );
    }

    #[test]
    fn budget_and_liquidity_constraints_are_hard_gates() {
        let mut over_budget = option("over-budget");
        over_budget.required_capital_minor = 10_001;
        over_budget.available_liquidity_after_minor = 999;
        let evaluated = &scenario(vec![over_budget]).evaluate().evaluated_options[0];

        assert!(!evaluated.accepted);
        assert!(
            evaluated
                .rejection_reasons
                .contains(&RejectionReason::BudgetExceeded)
        );
        assert!(
            evaluated
                .rejection_reasons
                .contains(&RejectionReason::LiquidityFloorBreached)
        );
    }

    #[test]
    fn reported_valuation_change_is_not_productive_capacity_or_cash_flow() {
        let mut boom = option("valuation-only-boom");
        boom.expected_net_cash_minor = 1_000;
        boom.reported_valuation_change_minor = 900_000;
        boom.productive_capacity_delta = 0;

        let result = scenario(vec![boom]).evaluate();
        let metrics = &result.evaluated_options[0].outcomes;
        assert_eq!(metrics.reported_valuation_change_minor, 900_000);
        assert_eq!(metrics.expected_net_cash_minor, 1_000);
        assert_eq!(metrics.productive_capacity_delta, 0);
    }

    #[test]
    fn cash_baseline_selects_highest_cash_among_feasible_options_only() {
        let mut low_cash = option("a-low-cash");
        low_cash.expected_net_cash_minor = 3_000;
        let mut high_cash = option("z-high-cash");
        high_cash.expected_net_cash_minor = 5_000;
        let mut forbidden = option("m-forbidden-highest-cash");
        forbidden.expected_net_cash_minor = 9_000_000;
        forbidden.authority_granted = false;

        let result = scenario(vec![low_cash, high_cash, forbidden]).evaluate();
        assert_eq!(
            cash_maximizing_feasible_baseline(&result.evaluated_options),
            Some("z-high-cash".to_owned())
        );
    }

    #[test]
    fn pareto_frontier_preserves_cash_capacity_tradeoff_and_removes_dominated_option() {
        let mut cash = option("cash");
        cash.expected_net_cash_minor = 4_000;
        cash.productive_capacity_delta = 10;

        let mut capacity = option("capacity");
        capacity.expected_net_cash_minor = 3_500;
        capacity.productive_capacity_delta = 20;
        capacity.maintenance_backlog_delta_minor = 50;

        let mut dominated = option("dominated");
        dominated.expected_net_cash_minor = 3_000;
        dominated.productive_capacity_delta = 5;

        let result = scenario(vec![cash, capacity, dominated]).evaluate();
        assert_eq!(
            pareto_frontier(&result.evaluated_options),
            vec!["capacity".to_owned(), "cash".to_owned()]
        );
    }

    #[test]
    fn fewer_declared_dependency_links_do_not_create_pareto_dominance() {
        let mut one_link = option("one-link");
        one_link.dependency_ids = vec!["grid-west".to_owned()];

        let mut two_links = option("two-links");
        two_links.dependency_ids = vec!["grid-west".to_owned(), "water-east".to_owned()];

        let result = scenario(vec![one_link, two_links]).evaluate();
        assert_eq!(
            pareto_frontier(&result.evaluated_options),
            vec!["one-link".to_owned(), "two-links".to_owned()]
        );
    }

    #[test]
    fn reported_valuation_does_not_rescue_a_dominated_option() {
        let mut productive = option("productive");
        productive.expected_net_cash_minor = 4_000;
        productive.productive_capacity_delta = 10;

        let mut valuation_only = option("valuation-only");
        valuation_only.expected_net_cash_minor = 3_000;
        valuation_only.productive_capacity_delta = 5;
        valuation_only.reported_valuation_change_minor = 9_000_000;

        let result = scenario(vec![productive, valuation_only]).evaluate();
        assert_eq!(
            pareto_frontier(&result.evaluated_options),
            vec!["productive".to_owned()]
        );
    }

    #[test]
    fn no_feasible_options_yield_no_baseline_choice_or_frontier() {
        let mut unauthorized = option("blocked");
        unauthorized.authority_granted = false;

        let result = scenario(vec![unauthorized]).evaluate();
        assert_eq!(
            cash_maximizing_feasible_baseline(&result.evaluated_options),
            None
        );
        assert!(pareto_frontier(&result.evaluated_options).is_empty());
    }

    #[test]
    fn deterministic_input_produces_equal_receipts() {
        let input = scenario(vec![option("a"), option("b")]);
        assert_eq!(input.evaluate(), input.evaluate());
    }
}
