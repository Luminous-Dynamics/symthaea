// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-009: exact post-commit oracle with safety and tail-risk constraints.
//!
//! Evaluator-only reference. It must never be linked into an online policy that can
//! see paired counterfactual outcomes. Exhaustive enumeration is intentionally
//! bounded; exceeding the cap returns an error, never an approximate "exact" result.
//!
//! Run tests with:
//!   rustc --edition=2024 --test tools/vision-dvpc-safety-tail-oracle.rs -o /tmp/vis-dvpc-009-tests
//!   /tmp/vis-dvpc-009-tests

#![deny(unsafe_code)]

use std::collections::BTreeMap;

pub const MAX_EXACT_TRACES: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct PairedTraceOutcomeV1 {
    pub trace_id: String,
    pub group_id: String,
    pub cheap_utility: f64,
    pub full_utility: f64,
    pub cheap_safety_violations: u64,
    pub full_safety_violations: u64,
    pub extra_latency_us: u64,
    pub extra_energy_uj: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafetyTailConstraintsV1 {
    pub latency_budget_us: u64,
    pub energy_budget_uj: u64,
    /// Cap on the sum of safety-violation counts under the chosen actions.
    pub max_total_safety_violations: u64,
    /// Maximum allowed CVaR95 of per-trace incremental decision loss.
    pub max_cvar95_incremental_loss: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleErrorV1 {
    EmptyTraceSet,
    InstanceTooLarge { traces: usize, maximum: usize },
    InvalidTraceId(String),
    InvalidGroupId(String),
    DuplicateTraceId(String),
    InvalidUtility(String),
    InvalidConstraints,
    AggregateCostOverflow,
    AggregateSafetyCountOverflow,
    NoFeasiblePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafetyTailOracleResultV1 {
    pub selected_full_trace_ids: Vec<String>,
    pub group_balanced_utility: f64,
    pub record_mean_utility: f64,
    pub chosen_safety_violations: u64,
    pub cvar95_incremental_loss: f64,
    pub incremental_latency_us: u64,
    pub incremental_energy_uj: u64,
    pub subsets_examined: u64,
}

impl SafetyTailOracleResultV1 {
    /// Deterministic tab-separated evaluator receipt. Trace IDs cannot contain
    /// separators, so the selected set is unambiguous when joined with commas.
    pub fn to_tsv(&self, constraints: SafetyTailConstraintsV1) -> String {
        format!(
            "ORACLE\tEXACT_SAFETY_TAIL_RESOURCE\t{}\t{:.6}\t{:.6}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.selected_full_trace_ids.join(","),
            self.group_balanced_utility,
            self.record_mean_utility,
            self.chosen_safety_violations,
            self.cvar95_incremental_loss,
            self.incremental_latency_us,
            self.incremental_energy_uj,
            constraints.latency_budget_us,
            constraints.energy_budget_uj,
            constraints.max_total_safety_violations,
            constraints.max_cvar95_incremental_loss,
            self.subsets_examined,
        )
    }
}

#[derive(Debug, Clone)]
struct CandidateMetrics {
    selected: Vec<String>,
    group_balanced_utility: f64,
    record_mean_utility: f64,
    safety_violations: u64,
    cvar95: f64,
    latency_us: u64,
    energy_uj: u64,
    escalations: usize,
}

/// Exact hindsight reference that maximizes equal-group-weighted utility subject
/// to all four declared constraints: total latency, total energy, total chosen
/// safety violations, and CVaR95 incremental decision loss. Outcomes are evaluator
/// truth, so this result is a diagnostic upper reference, never a deployable policy.
pub fn select_exact_safety_tail_oracle_v1(
    input: &[PairedTraceOutcomeV1],
    constraints: SafetyTailConstraintsV1,
) -> Result<SafetyTailOracleResultV1, OracleErrorV1> {
    validate_input(input, constraints)?;
    let mut traces = input.to_vec();
    traces.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    let group_sizes = group_sizes(&traces);
    let subset_count =
        1_u64
            .checked_shl(traces.len() as u32)
            .ok_or(OracleErrorV1::InstanceTooLarge {
                traces: traces.len(),
                maximum: MAX_EXACT_TRACES,
            })?;

    let mut best: Option<CandidateMetrics> = None;
    for mask in 0..subset_count {
        let candidate = score_mask(mask, &traces, &group_sizes)?;
        if candidate.latency_us > constraints.latency_budget_us
            || candidate.energy_uj > constraints.energy_budget_uj
            || candidate.safety_violations > constraints.max_total_safety_violations
            || candidate.cvar95 > constraints.max_cvar95_incremental_loss
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|current| candidate_is_better(&candidate, current))
        {
            best = Some(candidate);
        }
    }

    let best = best.ok_or(OracleErrorV1::NoFeasiblePolicy)?;
    Ok(SafetyTailOracleResultV1 {
        selected_full_trace_ids: best.selected,
        group_balanced_utility: best.group_balanced_utility,
        record_mean_utility: best.record_mean_utility,
        chosen_safety_violations: best.safety_violations,
        cvar95_incremental_loss: best.cvar95,
        incremental_latency_us: best.latency_us,
        incremental_energy_uj: best.energy_uj,
        subsets_examined: subset_count,
    })
}

fn validate_input(
    input: &[PairedTraceOutcomeV1],
    constraints: SafetyTailConstraintsV1,
) -> Result<(), OracleErrorV1> {
    if input.is_empty() {
        return Err(OracleErrorV1::EmptyTraceSet);
    }
    if input.len() > MAX_EXACT_TRACES {
        return Err(OracleErrorV1::InstanceTooLarge {
            traces: input.len(),
            maximum: MAX_EXACT_TRACES,
        });
    }
    if !constraints.max_cvar95_incremental_loss.is_finite()
        || !(0.0..=1.0).contains(&constraints.max_cvar95_incremental_loss)
    {
        return Err(OracleErrorV1::InvalidConstraints);
    }

    let mut trace_ids = std::collections::BTreeSet::new();
    let mut max_latency = 0_u64;
    let mut max_energy = 0_u64;
    let mut max_safety = 0_u64;
    for trace in input {
        if !valid_identifier(&trace.trace_id) {
            return Err(OracleErrorV1::InvalidTraceId(trace.trace_id.clone()));
        }
        if !valid_identifier(&trace.group_id) {
            return Err(OracleErrorV1::InvalidGroupId(trace.group_id.clone()));
        }
        if !trace_ids.insert(trace.trace_id.as_str()) {
            return Err(OracleErrorV1::DuplicateTraceId(trace.trace_id.clone()));
        }
        if !trace.cheap_utility.is_finite()
            || !trace.full_utility.is_finite()
            || !(0.0..=1.0).contains(&trace.cheap_utility)
            || !(0.0..=1.0).contains(&trace.full_utility)
        {
            return Err(OracleErrorV1::InvalidUtility(trace.trace_id.clone()));
        }
        max_latency = max_latency
            .checked_add(trace.extra_latency_us)
            .ok_or(OracleErrorV1::AggregateCostOverflow)?;
        max_energy = max_energy
            .checked_add(trace.extra_energy_uj)
            .ok_or(OracleErrorV1::AggregateCostOverflow)?;
        max_safety = max_safety
            .checked_add(
                trace
                    .cheap_safety_violations
                    .max(trace.full_safety_violations),
            )
            .ok_or(OracleErrorV1::AggregateSafetyCountOverflow)?;
    }

    // Keep these explicitly evaluated so impossible large totals cannot be hidden
    // by a particular subset's action choices.
    let _maximum_possible_latency = max_latency;
    let _maximum_possible_energy = max_energy;
    let _maximum_possible_safety = max_safety;
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty()
        && !value
            .chars()
            .any(|ch| matches!(ch, '\t' | '\r' | '\n' | ','))
}

fn group_sizes(traces: &[PairedTraceOutcomeV1]) -> BTreeMap<&str, usize> {
    let mut sizes = BTreeMap::new();
    for trace in traces {
        *sizes.entry(trace.group_id.as_str()).or_insert(0) += 1;
    }
    sizes
}

fn score_mask(
    mask: u64,
    traces: &[PairedTraceOutcomeV1],
    group_sizes: &BTreeMap<&str, usize>,
) -> Result<CandidateMetrics, OracleErrorV1> {
    let mut selected = Vec::new();
    let mut group_utility_sums = BTreeMap::<&str, f64>::new();
    let mut record_utility_sum = 0.0;
    let mut safety_violations = 0_u64;
    let mut latency_us = 0_u64;
    let mut energy_uj = 0_u64;
    let mut incremental_losses = Vec::with_capacity(traces.len());
    let mut escalations = 0;

    for (index, trace) in traces.iter().enumerate() {
        let full = (mask & (1_u64 << index)) != 0;
        let utility = if full {
            selected.push(trace.trace_id.clone());
            escalations += 1;
            trace.full_utility
        } else {
            trace.cheap_utility
        };
        record_utility_sum += utility;
        *group_utility_sums
            .entry(trace.group_id.as_str())
            .or_insert(0.0) += utility;
        safety_violations = safety_violations
            .checked_add(if full {
                trace.full_safety_violations
            } else {
                trace.cheap_safety_violations
            })
            .ok_or(OracleErrorV1::AggregateSafetyCountOverflow)?;

        if full {
            latency_us = latency_us
                .checked_add(trace.extra_latency_us)
                .ok_or(OracleErrorV1::AggregateCostOverflow)?;
            energy_uj = energy_uj
                .checked_add(trace.extra_energy_uj)
                .ok_or(OracleErrorV1::AggregateCostOverflow)?;
            incremental_losses.push(trace.cheap_utility - trace.full_utility);
        } else {
            incremental_losses.push(0.0);
        }
    }

    let group_count = group_utility_sums.len();
    let group_balanced_utility = group_utility_sums
        .iter()
        .map(|(group, sum)| sum / *group_sizes.get(group).expect("group size exists") as f64)
        .sum::<f64>()
        / group_count as f64;
    let record_mean_utility = record_utility_sum / traces.len() as f64;
    let cvar95 = cvar95(&mut incremental_losses);

    Ok(CandidateMetrics {
        selected,
        group_balanced_utility,
        record_mean_utility,
        safety_violations,
        cvar95,
        latency_us,
        energy_uj,
        escalations,
    })
}

fn cvar95(losses: &mut [f64]) -> f64 {
    losses.sort_by(|left, right| right.total_cmp(left));
    let tail_count = (losses.len() + 19) / 20;
    losses.iter().take(tail_count).sum::<f64>() / tail_count as f64
}

fn candidate_is_better(candidate: &CandidateMetrics, current: &CandidateMetrics) -> bool {
    match candidate
        .group_balanced_utility
        .total_cmp(&current.group_balanced_utility)
    {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }

    for ordering in [
        current.cvar95.total_cmp(&candidate.cvar95),
        current.safety_violations.cmp(&candidate.safety_violations),
        current.latency_us.cmp(&candidate.latency_us),
        current.energy_uj.cmp(&candidate.energy_uj),
        current.escalations.cmp(&candidate.escalations),
    ] {
        match ordering {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    candidate.selected < current.selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(
        id: &str,
        group: &str,
        cheap: f64,
        full: f64,
        cheap_safety: u64,
        full_safety: u64,
        latency: u64,
        energy: u64,
    ) -> PairedTraceOutcomeV1 {
        PairedTraceOutcomeV1 {
            trace_id: id.to_owned(),
            group_id: group.to_owned(),
            cheap_utility: cheap,
            full_utility: full,
            cheap_safety_violations: cheap_safety,
            full_safety_violations: full_safety,
            extra_latency_us: latency,
            extra_energy_uj: energy,
        }
    }

    fn constraints(latency: u64, energy: u64, safety: u64, cvar: f64) -> SafetyTailConstraintsV1 {
        SafetyTailConstraintsV1 {
            latency_budget_us: latency,
            energy_budget_uj: energy,
            max_total_safety_violations: safety,
            max_cvar95_incremental_loss: cvar,
        }
    }

    #[test]
    fn safety_cap_excludes_high_utility_but_unsafe_escalation() {
        let traces = vec![
            trace("unsafe-high-gain", "g1", 0.40, 0.90, 0, 1, 10, 10),
            trace("safe-gain", "g2", 0.50, 0.60, 0, 0, 10, 10),
        ];
        let utility_only =
            select_exact_safety_tail_oracle_v1(&traces, constraints(20, 20, u64::MAX, 1.0))
                .unwrap();
        assert_eq!(
            utility_only.selected_full_trace_ids,
            vec!["safe-gain".to_owned(), "unsafe-high-gain".to_owned()]
        );

        let constrained =
            select_exact_safety_tail_oracle_v1(&traces, constraints(20, 20, 0, 0.0)).unwrap();
        assert_eq!(
            constrained.selected_full_trace_ids,
            vec!["safe-gain".to_owned()]
        );
        assert_eq!(constrained.chosen_safety_violations, 0);
        assert!((constrained.group_balanced_utility - 0.50).abs() < 1e-12);
    }

    #[test]
    fn latency_and_energy_limits_both_constrain_the_optimum() {
        let traces = vec![
            trace("a", "ga", 0.10, 0.50, 0, 0, 8, 2),
            trace("b", "gb", 0.10, 0.70, 0, 0, 2, 8),
            trace("c", "gc", 0.10, 0.55, 0, 0, 5, 5),
        ];
        let result =
            select_exact_safety_tail_oracle_v1(&traces, constraints(10, 10, 0, 0.0)).unwrap();
        assert_eq!(
            result.selected_full_trace_ids,
            vec!["a".to_owned(), "b".to_owned()]
        );
        assert_eq!(result.incremental_latency_us, 10);
        assert_eq!(result.incremental_energy_uj, 10);
    }

    #[test]
    fn cvar_bound_excludes_harmful_tail_action() {
        let traces = vec![
            trace("beneficial", "g1", 0.40, 0.60, 0, 0, 1, 1),
            trace("harmful-tail", "g2", 0.50, 0.30, 0, 0, 1, 1),
        ];
        let result =
            select_exact_safety_tail_oracle_v1(&traces, constraints(2, 2, 0, 0.0)).unwrap();
        assert_eq!(
            result.selected_full_trace_ids,
            vec!["beneficial".to_owned()]
        );
        assert!((result.cvar95_incremental_loss - 0.0).abs() < 1e-12);
    }

    #[test]
    fn group_balanced_objective_can_prefer_the_small_group() {
        let traces = vec![
            trace("common-a", "common", 0.0, 0.50, 0, 0, 1, 1),
            trace("common-b", "common", 0.0, 0.50, 0, 0, 1, 1),
            trace("rare", "rare", 0.0, 0.30, 0, 0, 1, 1),
        ];
        let result =
            select_exact_safety_tail_oracle_v1(&traces, constraints(1, 1, 0, 0.0)).unwrap();
        assert_eq!(result.selected_full_trace_ids, vec!["rare".to_owned()]);
        assert!((result.group_balanced_utility - 0.15).abs() < 1e-12);
        assert!((result.record_mean_utility - 0.10).abs() < 1e-12);
    }

    #[test]
    fn no_feasible_policy_returns_explicit_infeasibility() {
        let traces = vec![trace("already-unsafe", "g", 0.10, 0.20, 1, 2, 1, 1)];
        let error =
            select_exact_safety_tail_oracle_v1(&traces, constraints(10, 10, 0, 0.0)).unwrap_err();
        assert_eq!(error, OracleErrorV1::NoFeasiblePolicy);
    }

    #[test]
    fn result_is_deterministic_and_input_order_invariant() {
        let traces = vec![
            trace("z", "g2", 0.2, 0.4, 0, 0, 1, 1),
            trace("a", "g1", 0.2, 0.3, 0, 0, 1, 1),
            trace("m", "g3", 0.2, 0.5, 0, 0, 1, 1),
        ];
        let limits = constraints(2, 2, 0, 0.0);
        let forward = select_exact_safety_tail_oracle_v1(&traces, limits).unwrap();
        let mut reversed = traces.clone();
        reversed.reverse();
        let backward = select_exact_safety_tail_oracle_v1(&reversed, limits).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(forward.to_tsv(limits), backward.to_tsv(limits));
    }

    #[test]
    fn oversized_instance_is_rejected_without_approximation() {
        let traces = (0..=MAX_EXACT_TRACES)
            .map(|index| trace(&format!("trace-{index:02}"), "g", 0.1, 0.2, 0, 0, 1, 1))
            .collect::<Vec<_>>();
        assert_eq!(
            select_exact_safety_tail_oracle_v1(
                &traces,
                constraints(u64::MAX, u64::MAX, u64::MAX, 1.0)
            ),
            Err(OracleErrorV1::InstanceTooLarge {
                traces: MAX_EXACT_TRACES + 1,
                maximum: MAX_EXACT_TRACES,
            })
        );
    }

    #[test]
    fn duplicate_ids_and_nonfinite_utilities_are_rejected() {
        let duplicate = vec![
            trace("dup", "g1", 0.1, 0.2, 0, 0, 1, 1),
            trace("dup", "g2", 0.1, 0.3, 0, 0, 1, 1),
        ];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(&duplicate, constraints(10, 10, 0, 1.0)),
            Err(OracleErrorV1::DuplicateTraceId("dup".to_owned()))
        );

        let invalid = vec![trace("bad", "g", f64::NAN, 0.5, 0, 0, 1, 1)];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(&invalid, constraints(10, 10, 0, 1.0)),
            Err(OracleErrorV1::InvalidUtility("bad".to_owned()))
        );
    }

    #[test]
    fn aggregate_resource_cost_overflow_is_rejected() {
        let traces = vec![
            trace("large", "g1", 0.1, 0.2, 0, 0, u64::MAX, 1),
            trace("overflow", "g2", 0.1, 0.2, 0, 0, 1, 1),
        ];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(
                &traces,
                constraints(u64::MAX, u64::MAX, u64::MAX, 1.0),
            ),
            Err(OracleErrorV1::AggregateCostOverflow)
        );
    }

    #[test]
    fn aggregate_safety_count_overflow_is_rejected() {
        let traces = vec![
            trace("large", "g1", 0.1, 0.2, u64::MAX, 0, 1, 1),
            trace("overflow", "g2", 0.1, 0.2, 1, 0, 1, 1),
        ];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(
                &traces,
                constraints(u64::MAX, u64::MAX, u64::MAX, 1.0),
            ),
            Err(OracleErrorV1::AggregateSafetyCountOverflow)
        );
    }

    #[test]
    fn invalid_identifier_and_constraint_are_rejected() {
        let invalid_id = vec![trace("bad,id", "g", 0.1, 0.2, 0, 0, 1, 1)];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(&invalid_id, constraints(10, 10, 0, 1.0)),
            Err(OracleErrorV1::InvalidTraceId("bad,id".to_owned()))
        );
        let valid = vec![trace("ok", "g", 0.1, 0.2, 0, 0, 1, 1)];
        assert_eq!(
            select_exact_safety_tail_oracle_v1(&valid, constraints(10, 10, 0, f64::NAN)),
            Err(OracleErrorV1::InvalidConstraints)
        );
    }

    #[test]
    fn receipt_contains_constraints_and_reports_every_examined_subset() {
        let traces = vec![trace("x", "g", 0.1, 0.2, 0, 0, 1, 1)];
        let limits = constraints(4, 5, 0, 0.0);
        let result = select_exact_safety_tail_oracle_v1(&traces, limits).unwrap();
        let receipt = result.to_tsv(limits);
        assert!(receipt.starts_with("ORACLE\tEXACT_SAFETY_TAIL_RESOURCE\tx\t0.200000\t0.200000"));
        assert!(receipt.ends_with("\t2\n"));
    }
}
