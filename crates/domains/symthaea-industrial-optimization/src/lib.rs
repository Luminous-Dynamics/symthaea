// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research-stage portfolio selection for methane-abatement interventions.
//!
//! This crate is deliberately a small, auditable baseline rather than an AI
//! controller or an emissions estimator. Inputs are supplied by the caller;
//! the built-in example data are synthetic and must never be interpreted as
//! typical site performance, measured emissions, or an operating recommendation.
//!
//! The exhaustive optimizer maximizes a caller-supplied conservative scenario
//! floor under a capital budget. It prevents two proposed interventions from
//! claiming the same source group, and excludes options not marked eligible for
//! planning. Scenario floors are not statistical confidence bounds unless the
//! caller has independently established that interpretation.

use std::collections::BTreeSet;
use std::fmt;

pub const MAX_EXACT_CANDIDATES: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbatementEstimate {
    /// Conservative scenario floor, in tonnes of CH4 per reporting year.
    pub floor_tonnes_ch4_per_year: u64,
    /// Central scenario estimate, in tonnes of CH4 per reporting year.
    pub expected_tonnes_ch4_per_year: u64,
    /// High scenario, in tonnes of CH4 per reporting year.
    pub ceiling_tonnes_ch4_per_year: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intervention {
    /// Stable, unique identifier for this candidate.
    pub id: String,
    /// Identifier for the same underlying emissions source/inventory line.
    /// At most one intervention per source group is selected, avoiding simple
    /// double counting when options are alternatives for the same source.
    pub source_group: String,
    /// Up-front capital cost in whole USD, on a common declared price basis.
    pub capex_usd: u64,
    /// Scenario estimates; these values must share a common reporting period.
    pub abatement: AbatementEstimate,
    /// Caller-provided screening flag. This is not safety approval or permission
    /// to procure, install, or operate equipment.
    pub eligible_for_planning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Portfolio {
    /// Sorted IDs make reports deterministic regardless of input order.
    pub selected_ids: Vec<String>,
    pub capex_usd: u64,
    pub floor_tonnes_ch4_per_year: u64,
    pub expected_tonnes_ch4_per_year: u64,
    pub ceiling_tonnes_ch4_per_year: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkError {
    TooManyCandidates { actual: usize, maximum: usize },
    EmptyId { index: usize },
    EmptySourceGroup { index: usize },
    ZeroCapex { id: String },
    InvalidAbatementRange { id: String },
    DuplicateId { id: String },
    ArithmeticOverflow,
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyCandidates { actual, maximum } => {
                write!(f, "exact search supports at most {maximum} candidates, got {actual}")
            }
            Self::EmptyId { index } => write!(f, "candidate at index {index} has an empty ID"),
            Self::EmptySourceGroup { index } => {
                write!(f, "candidate at index {index} has an empty source group")
            }
            Self::ZeroCapex { id } => write!(f, "candidate {id} has zero capex; ratio baseline requires positive capex"),
            Self::InvalidAbatementRange { id } => {
                write!(f, "candidate {id} must satisfy floor <= expected <= ceiling")
            }
            Self::DuplicateId { id } => write!(f, "candidate ID {id} is duplicated"),
            Self::ArithmeticOverflow => write!(f, "portfolio totals overflow u64"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

fn validate(candidates: &[Intervention]) -> Result<(), BenchmarkError> {
    if candidates.len() > MAX_EXACT_CANDIDATES {
        return Err(BenchmarkError::TooManyCandidates {
            actual: candidates.len(),
            maximum: MAX_EXACT_CANDIDATES,
        });
    }

    let mut ids = BTreeSet::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.id.trim().is_empty() {
            return Err(BenchmarkError::EmptyId { index });
        }
        if candidate.source_group.trim().is_empty() {
            return Err(BenchmarkError::EmptySourceGroup { index });
        }
        if candidate.capex_usd == 0 {
            return Err(BenchmarkError::ZeroCapex {
                id: candidate.id.clone(),
            });
        }
        if candidate.abatement.floor_tonnes_ch4_per_year
            > candidate.abatement.expected_tonnes_ch4_per_year
            || candidate.abatement.expected_tonnes_ch4_per_year
                > candidate.abatement.ceiling_tonnes_ch4_per_year
        {
            return Err(BenchmarkError::InvalidAbatementRange {
                id: candidate.id.clone(),
            });
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(BenchmarkError::DuplicateId {
                id: candidate.id.clone(),
            });
        }
    }
    Ok(())
}

fn portfolio_from_indices(
    candidates: &[Intervention],
    indexes: &[usize],
) -> Result<Portfolio, BenchmarkError> {
    let mut portfolio = Portfolio::default();
    for &index in indexes {
        let candidate = &candidates[index];
        portfolio.capex_usd = portfolio
            .capex_usd
            .checked_add(candidate.capex_usd)
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        portfolio.floor_tonnes_ch4_per_year = portfolio
            .floor_tonnes_ch4_per_year
            .checked_add(candidate.abatement.floor_tonnes_ch4_per_year)
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        portfolio.expected_tonnes_ch4_per_year = portfolio
            .expected_tonnes_ch4_per_year
            .checked_add(candidate.abatement.expected_tonnes_ch4_per_year)
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        portfolio.ceiling_tonnes_ch4_per_year = portfolio
            .ceiling_tonnes_ch4_per_year
            .checked_add(candidate.abatement.ceiling_tonnes_ch4_per_year)
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        portfolio.selected_ids.push(candidate.id.clone());
    }
    portfolio.selected_ids.sort();
    Ok(portfolio)
}

fn better(candidate: &Portfolio, incumbent: &Portfolio) -> bool {
    candidate.floor_tonnes_ch4_per_year > incumbent.floor_tonnes_ch4_per_year
        || (candidate.floor_tonnes_ch4_per_year == incumbent.floor_tonnes_ch4_per_year
            && candidate.expected_tonnes_ch4_per_year > incumbent.expected_tonnes_ch4_per_year)
        || (candidate.floor_tonnes_ch4_per_year == incumbent.floor_tonnes_ch4_per_year
            && candidate.expected_tonnes_ch4_per_year == incumbent.expected_tonnes_ch4_per_year
            && candidate.capex_usd < incumbent.capex_usd)
        || (candidate.floor_tonnes_ch4_per_year == incumbent.floor_tonnes_ch4_per_year
            && candidate.expected_tonnes_ch4_per_year == incumbent.expected_tonnes_ch4_per_year
            && candidate.capex_usd == incumbent.capex_usd
            && candidate.selected_ids < incumbent.selected_ids)
}

/// Maximize summed scenario-floor methane abatement under a capex budget.
///
/// Constraints: candidates must pass basic schema validation; only candidates
/// marked eligible_for_planning are considered; at most one item from each
/// source group is selected; total capex must not exceed budget_usd. Exhaustive
/// search is only suitable for small frozen benchmark cases. It does not model
/// interactions between distinct sources, annual OPEX, gas-sale revenue, NPV,
/// lifecycle emissions, process physics, or safety cases.
pub fn optimize_robust_portfolio(
    candidates: &[Intervention],
    budget_usd: u64,
) -> Result<Portfolio, BenchmarkError> {
    validate(candidates)?;

    let subset_count = 1_u64 << candidates.len();
    let mut best = Portfolio::default();

    for mask in 1..subset_count {
        let mut indexes = Vec::new();
        let mut source_groups = BTreeSet::new();
        let mut capex = 0_u64;
        let mut feasible = true;

        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1_u64 << index) == 0 {
                continue;
            }
            if !candidate.eligible_for_planning
                || !source_groups.insert(candidate.source_group.as_str())
            {
                feasible = false;
                break;
            }
            let Some(next_capex) = capex.checked_add(candidate.capex_usd) else {
                feasible = false;
                break;
            };
            capex = next_capex;
            if capex > budget_usd {
                feasible = false;
                break;
            }
            indexes.push(index);
        }

        if !feasible {
            continue;
        }
        let portfolio = portfolio_from_indices(candidates, &indexes)?;
        if better(&portfolio, &best) {
            best = portfolio;
        }
    }

    Ok(best)
}

/// Simple benchmark baseline: greedily rank eligible interventions by expected
/// methane abatement per dollar, respecting the same capex and source-group
/// constraints. This is a transparent comparison rule, not a claim about every
/// operator's current workflow or the industry state of the art.
pub fn greedy_expected_per_dollar(
    candidates: &[Intervention],
    budget_usd: u64,
) -> Result<Portfolio, BenchmarkError> {
    validate(candidates)?;

    let mut order: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| candidate.eligible_for_planning.then_some(index))
        .collect();

    order.sort_by(|&a, &b| {
        let left = u128::from(candidates[a].abatement.expected_tonnes_ch4_per_year)
            * u128::from(candidates[b].capex_usd);
        let right = u128::from(candidates[b].abatement.expected_tonnes_ch4_per_year)
            * u128::from(candidates[a].capex_usd);
        right
            .cmp(&left)
            .then_with(|| candidates[a].capex_usd.cmp(&candidates[b].capex_usd))
            .then_with(|| candidates[a].id.cmp(&candidates[b].id))
    });

    let mut selected = Vec::new();
    let mut source_groups = BTreeSet::new();
    let mut capex = 0_u64;

    for index in order {
        let candidate = &candidates[index];
        if source_groups.contains(candidate.source_group.as_str()) {
            continue;
        }
        let Some(next_capex) = capex.checked_add(candidate.capex_usd) else {
            continue;
        };
        if next_capex > budget_usd {
            continue;
        }
        capex = next_capex;
        source_groups.insert(candidate.source_group.as_str());
        selected.push(index);
    }

    portfolio_from_indices(candidates, &selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(
        id: &str,
        source_group: &str,
        capex_usd: u64,
        floor: u64,
        expected: u64,
        ceiling: u64,
        eligible_for_planning: bool,
    ) -> Intervention {
        Intervention {
            id: id.to_owned(),
            source_group: source_group.to_owned(),
            capex_usd,
            abatement: AbatementEstimate {
                floor_tonnes_ch4_per_year: floor,
                expected_tonnes_ch4_per_year: expected,
                ceiling_tonnes_ch4_per_year: ceiling,
            },
            eligible_for_planning,
        }
    }

    #[test]
    fn exact_optimizer_can_beat_expected_value_greedy_on_floor() {
        let candidates = vec![
            item("a-high-variance", "source-a", 6, 4, 20, 30, true),
            item("b-stable", "source-b", 5, 3, 6, 8, true),
            item("c-stable", "source-c", 5, 3, 6, 8, true),
        ];

        let robust = optimize_robust_portfolio(&candidates, 10).unwrap();
        let greedy = greedy_expected_per_dollar(&candidates, 10).unwrap();

        assert_eq!(robust.selected_ids, vec!["b-stable", "c-stable"]);
        assert_eq!(robust.floor_tonnes_ch4_per_year, 6);
        assert_eq!(robust.capex_usd, 10);
        assert_eq!(greedy.selected_ids, vec!["a-high-variance"]);
        assert_eq!(greedy.floor_tonnes_ch4_per_year, 4);
    }

    #[test]
    fn neither_strategy_exceeds_capital_budget() {
        let candidates = vec![
            item("a", "source-a", 10, 2, 3, 4, true),
            item("b", "source-b", 11, 5, 6, 7, true),
        ];
        assert!(optimize_robust_portfolio(&candidates, 10).unwrap().capex_usd <= 10);
        assert!(greedy_expected_per_dollar(&candidates, 10).unwrap().capex_usd <= 10);
    }

    #[test]
    fn same_source_group_cannot_be_double_counted() {
        let candidates = vec![
            item("repair", "same-metered-source", 10, 2, 4, 6, true),
            item("replace", "same-metered-source", 12, 3, 5, 7, true),
        ];
        let result = optimize_robust_portfolio(&candidates, 30).unwrap();
        assert_eq!(result.selected_ids.len(), 1);
        assert_eq!(result.floor_tonnes_ch4_per_year, 3);
    }

    #[test]
    fn planning_ineligible_interventions_are_never_selected() {
        let candidates = vec![
            item("eligible", "source-a", 10, 1, 2, 3, true),
            item("unreviewed", "source-b", 1, 100, 100, 100, false),
        ];
        let result = optimize_robust_portfolio(&candidates, 100).unwrap();
        assert_eq!(result.selected_ids, vec!["eligible"]);
    }

    #[test]
    fn malformed_scenario_order_fails_closed() {
        let candidates = vec![item("bad", "source-a", 10, 8, 4, 12, true)];
        assert_eq!(
            optimize_robust_portfolio(&candidates, 100),
            Err(BenchmarkError::InvalidAbatementRange {
                id: "bad".to_owned()
            })
        );
    }

    #[test]
    fn duplicate_ids_fail_closed() {
        let candidates = vec![
            item("same", "source-a", 10, 1, 2, 3, true),
            item("same", "source-b", 11, 1, 2, 3, true),
        ];
        assert_eq!(
            optimize_robust_portfolio(&candidates, 100),
            Err(BenchmarkError::DuplicateId {
                id: "same".to_owned()
            })
        );
    }

    #[test]
    fn empty_case_returns_empty_portfolio() {
        assert_eq!(
            optimize_robust_portfolio(&[], 100).unwrap(),
            Portfolio::default()
        );
    }
}
