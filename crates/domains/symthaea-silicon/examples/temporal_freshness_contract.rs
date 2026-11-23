// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference contract for decision freshness at the time information is consumed.
//!
//! This example adapts established AoI / Query AoI concepts to independently
//! verified temporal decisions. It is not a new definition of AoI or AoII.
//!
//! It keeps these questions separate:
//! - How old was the evidence when the output was emitted?
//! - How old was that evidence when the consumer actually used the output?
//! - Was the output correct at use, fresh at use, and before its deadline?
//! - What fraction of all evaluator-defined opportunities received such an output?
//!
//! Run from the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_freshness_contract
//!   cargo run --locked -p symthaea-silicon --example temporal_freshness_contract
//!
//! Logical/event timestamps are independent of wall-clock execution time.
//! The work-ledger receipt digest binds this result to the same benchmark run.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const SCHEMA: &str = "symthaea.temporal-decision-freshness";
const SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Answered,
    Abstained,
    Unserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Correctness {
    Correct,
    Incorrect,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionOpportunity {
    opportunity_id: u64,
    /// Time the evaluator says the decision-relevant event occurred.
    source_event_time_ns: u64,
    /// Time this opportunity became available to the candidate.
    available_at_ns: u64,
    /// Absolute task deadline at the point the consumer uses the output.
    deadline_at_ns: u64,
    /// Maximum acceptable evidence age at the point of use.
    freshness_budget_ns: u64,
    disposition: Disposition,
    /// Candidate output time; absent for abstention/unserved.
    emitted_at_ns: Option<u64>,
    /// Time the downstream consumer/controller actually used the output.
    consumed_at_ns: Option<u64>,
    /// Source/event time of the newest evidence watermark used by this output.
    evidence_source_time_ns: Option<u64>,
    /// Availability time of that evidence watermark.
    evidence_available_at_ns: Option<u64>,
    /// Evaluator-only label, never exposed to the candidate.
    correctness: Option<Correctness>,
}

impl DecisionOpportunity {
    fn validate(&self) -> Result<(), &'static str> {
        if self.source_event_time_ns > self.available_at_ns {
            return Err("source event time cannot follow opportunity availability");
        }

        match self.disposition {
            Disposition::Answered => {
                let emitted = self.emitted_at_ns.ok_or("answered output requires emitted time")?;
                let consumed = self.consumed_at_ns.ok_or("answered output requires consumed time")?;
                let evidence_source = self
                    .evidence_source_time_ns
                    .ok_or("answered output requires evidence source time")?;
                let evidence_available = self
                    .evidence_available_at_ns
                    .ok_or("answered output requires evidence availability time")?;
                self.correctness.ok_or("answered output requires evaluator correctness")?;

                if emitted < self.available_at_ns {
                    return Err("output cannot be emitted before opportunity availability");
                }
                if consumed < emitted {
                    return Err("output cannot be consumed before it is emitted");
                }
                if evidence_source > evidence_available {
                    return Err("evidence source time cannot follow evidence availability");
                }
                if evidence_available > emitted {
                    return Err("evidence unavailable at emission is future-information leakage");
                }
                if evidence_source > emitted {
                    return Err("evidence source time cannot follow output emission");
                }
                if consumed < evidence_source {
                    return Err("evidence source time cannot follow output consumption");
                }
            }
            Disposition::Abstained | Disposition::Unserved => {
                if self.emitted_at_ns.is_some()
                    || self.consumed_at_ns.is_some()
                    || self.evidence_source_time_ns.is_some()
                    || self.evidence_available_at_ns.is_some()
                    || self.correctness.is_some()
                {
                    return Err("abstained/unserved opportunity cannot carry output evidence");
                }
            }
        }
        // A deadline earlier than availability is valid input: the opportunity
        // arrived already expired and must be counted as late/unserved, not erased.
        Ok(())
    }

    fn availability_to_use_latency_ns(&self) -> Option<u64> {
        self.consumed_at_ns?.checked_sub(self.available_at_ns)
    }

    fn source_to_use_latency_ns(&self) -> Option<u64> {
        self.consumed_at_ns?.checked_sub(self.source_event_time_ns)
    }

    fn source_to_availability_delay_ns(&self) -> Option<u64> {
        self.available_at_ns.checked_sub(self.source_event_time_ns)
    }

    fn evidence_age_at_emission_ns(&self) -> Option<u64> {
        self.emitted_at_ns?
            .checked_sub(self.evidence_source_time_ns?)
    }

    fn evidence_age_at_use_ns(&self) -> Option<u64> {
        self.consumed_at_ns?
            .checked_sub(self.evidence_source_time_ns?)
    }

    fn is_late(&self) -> bool {
        self.consumed_at_ns
            .is_some_and(|consumed| consumed > self.deadline_at_ns)
    }

    fn is_fresh_at_use(&self) -> Option<bool> {
        Some(self.evidence_age_at_use_ns()? <= self.freshness_budget_ns)
    }

    fn is_fresh_correct_on_time(&self) -> bool {
        self.disposition == Disposition::Answered
            && !self.is_late()
            && self.is_fresh_at_use() == Some(true)
            && self.correctness == Some(Correctness::Correct)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FreshnessRun {
    schema: String,
    schema_version: String,
    workload_id: String,
    profile_id: String,
    source_revision: String,
    configuration_id: String,
    decision_contract_id: String,
    independent_verifier_id: String,
    /// SHA-256 of the #13061 temporal-work receipt from this exact run.
    temporal_work_receipt_sha256: String,
    /// End of the evaluator's logical-time observation horizon.
    logical_horizon_end_ns: u64,
    opportunities: Vec<DecisionOpportunity>,
}

impl FreshnessRun {
    fn validate(&self) -> Result<(), &'static str> {
        if self.schema != SCHEMA || self.schema_version != SCHEMA_VERSION {
            return Err("unsupported freshness schema identity or version");
        }
        for value in [
            self.workload_id.as_str(),
            self.profile_id.as_str(),
            self.source_revision.as_str(),
            self.configuration_id.as_str(),
            self.decision_contract_id.as_str(),
            self.independent_verifier_id.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err("run identity fields cannot be empty");
            }
        }
        if self.temporal_work_receipt_sha256.len() != 64
            || !self
                .temporal_work_receipt_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("work receipt identity must be a 64-character SHA-256 hex digest");
        }
        if self.opportunities.is_empty() {
            return Err("freshness run requires at least one decision opportunity");
        }

        let mut ids = BTreeSet::new();
        for opportunity in &self.opportunities {
            opportunity.validate()?;
            if !ids.insert(opportunity.opportunity_id) {
                return Err("decision opportunity IDs must be unique");
            }
            if opportunity.source_event_time_ns > self.logical_horizon_end_ns
                || opportunity.available_at_ns > self.logical_horizon_end_ns
                || opportunity.emitted_at_ns.is_some_and(|t| t > self.logical_horizon_end_ns)
                || opportunity.consumed_at_ns.is_some_and(|t| t > self.logical_horizon_end_ns)
                || opportunity
                    .evidence_source_time_ns
                    .is_some_and(|t| t > self.logical_horizon_end_ns)
                || opportunity
                    .evidence_available_at_ns
                    .is_some_and(|t| t > self.logical_horizon_end_ns)
            {
                return Err("opportunity timestamp exceeds the logical evaluation horizon");
            }
        }
        Ok(())
    }

    fn summary(&self) -> Result<FreshnessSummary, &'static str> {
        self.validate()?;

        let opportunities = self.opportunities.len() as u64;
        let answered: Vec<&DecisionOpportunity> = self
            .opportunities
            .iter()
            .filter(|item| item.disposition == Disposition::Answered)
            .collect();
        let on_time: Vec<&DecisionOpportunity> =
            answered.iter().copied().filter(|item| !item.is_late()).collect();
        let late = answered.len() as u64 - on_time.len() as u64;
        let abstained = self
            .opportunities
            .iter()
            .filter(|item| item.disposition == Disposition::Abstained)
            .count() as u64;
        let unserved = self
            .opportunities
            .iter()
            .filter(|item| item.disposition == Disposition::Unserved)
            .count() as u64;
        let fresh_at_use = answered
            .iter()
            .filter(|item| item.is_fresh_at_use() == Some(true))
            .count() as u64;
        let expired_unserved = self
            .opportunities
            .iter()
            .filter(|item| {
                item.disposition == Disposition::Unserved
                    && item.deadline_at_ns <= self.logical_horizon_end_ns
            })
            .count() as u64;
        let pending_unserved = unserved - expired_unserved;

        // Deadline failures are late outputs or unserved opportunities whose
        // deadlines have elapsed by the evaluator horizon. Abstentions remain
        // separate coverage failures and do not masquerade as missed responses.
        let mut ordered: Vec<&DecisionOpportunity> = self.opportunities.iter().collect();
        ordered.sort_by_key(|item| (item.available_at_ns, item.opportunity_id));
        let mut max_consecutive_misses = 0_u64;
        let mut current_miss_run = 0_u64;
        let mut previous_was_miss: Option<bool> = None;
        let mut previous_miss_pairs = 0_u64;
        let mut following_miss_pairs = 0_u64;
        for item in ordered {
            let miss = match item.disposition {
                Disposition::Answered => Some(item.is_late()),
                Disposition::Abstained => Some(false),
                Disposition::Unserved if item.deadline_at_ns <= self.logical_horizon_end_ns => {
                    Some(true)
                }
                Disposition::Unserved => None,
            };
            match miss {
                Some(true) => {
                    current_miss_run += 1;
                    max_consecutive_misses = max_consecutive_misses.max(current_miss_run);
                    if previous_was_miss == Some(true) {
                        previous_miss_pairs += 1;
                        following_miss_pairs += 1;
                    }
                    previous_was_miss = Some(true);
                }
                Some(false) => {
                    if previous_was_miss == Some(true) {
                        previous_miss_pairs += 1;
                    }
                    current_miss_run = 0;
                    previous_was_miss = Some(false);
                }
                None => {
                    current_miss_run = 0;
                    previous_was_miss = None;
                }
            }
        }
        let deadline_misses = late + expired_unserved;
        let fresh_correct_on_time = self
            .opportunities
            .iter()
            .filter(|item| item.is_fresh_correct_on_time())
            .count() as u64;
        let correct_on_time = on_time
            .iter()
            .filter(|item| item.correctness == Some(Correctness::Correct))
            .count() as u64;
        let fresh_but_incorrect = answered
            .iter()
            .filter(|item| {
                item.is_fresh_at_use() == Some(true)
                    && item.correctness == Some(Correctness::Incorrect)
            })
            .count() as u64;
        let correct_but_stale = answered
            .iter()
            .filter(|item| {
                item.is_fresh_at_use() == Some(false)
                    && item.correctness == Some(Correctness::Correct)
            })
            .count() as u64;

        let mut availability_to_use: Vec<u64> = answered
            .iter()
            .filter_map(|item| item.availability_to_use_latency_ns())
            .collect();
        let mut source_to_use: Vec<u64> = answered
            .iter()
            .filter_map(|item| item.source_to_use_latency_ns())
            .collect();
        let mut source_to_availability: Vec<u64> = answered
            .iter()
            .filter_map(|item| item.source_to_availability_delay_ns())
            .collect();
        let mut evidence_age_at_use: Vec<u64> = answered
            .iter()
            .filter_map(|item| item.evidence_age_at_use_ns())
            .collect();
        let mut time_to_fresh_correct: Vec<u64> = self
            .opportunities
            .iter()
            .filter(|item| item.is_fresh_correct_on_time())
            .filter_map(|item| item.availability_to_use_latency_ns())
            .collect();

        Ok(FreshnessSummary {
            opportunities,
            answered: answered.len() as u64,
            on_time_answered: on_time.len() as u64,
            late_answered: late,
            abstained,
            unserved,
            expired_unserved,
            pending_unserved,
            deadline_misses,
            max_consecutive_deadline_misses: max_consecutive_misses,
            miss_to_miss_transition_rate: ratio(following_miss_pairs, previous_miss_pairs),
            fresh_at_use,
            fresh_correct_on_time,
            correct_on_time,
            fresh_but_incorrect,
            correct_but_stale,
            on_time_decision_coverage: ratio(on_time.len() as u64, opportunities),
            fresh_correct_on_time_coverage: ratio(fresh_correct_on_time, opportunities),
            verified_accuracy_on_time: ratio(correct_on_time, on_time.len() as u64),
            availability_to_use_latency_p50_ns: percentile(&mut availability_to_use, 50),
            availability_to_use_latency_p95_ns: percentile(&mut availability_to_use, 95),
            availability_to_use_latency_p99_ns: percentile(&mut availability_to_use, 99),
            availability_to_use_latency_p99_9_ns: percentile_per_mille(&mut availability_to_use, 999),
            availability_to_use_latency_max_ns: percentile(&mut availability_to_use, 100),
            source_to_use_latency_p50_ns: percentile(&mut source_to_use, 50),
            source_to_use_latency_p95_ns: percentile(&mut source_to_use, 95),
            source_to_use_latency_p99_ns: percentile(&mut source_to_use, 99),
            source_to_use_latency_p99_9_ns: percentile_per_mille(&mut source_to_use, 999),
            source_to_use_latency_max_ns: percentile(&mut source_to_use, 100),
            source_to_availability_delay_p95_ns: percentile(&mut source_to_availability, 95),
            source_to_availability_delay_p99_ns: percentile(&mut source_to_availability, 99),
            evidence_age_at_use_p50_ns: percentile(&mut evidence_age_at_use, 50),
            evidence_age_at_use_p95_ns: percentile(&mut evidence_age_at_use, 95),
            evidence_age_at_use_p99_ns: percentile(&mut evidence_age_at_use, 99),
            evidence_age_at_use_p99_9_ns: percentile_per_mille(&mut evidence_age_at_use, 999),
            evidence_age_at_use_max_ns: percentile(&mut evidence_age_at_use, 100),
            time_to_fresh_correct_p95_ns: percentile(&mut time_to_fresh_correct, 95),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FreshnessSummary {
    opportunities: u64,
    answered: u64,
    on_time_answered: u64,
    late_answered: u64,
    abstained: u64,
    unserved: u64,
    expired_unserved: u64,
    pending_unserved: u64,
    deadline_misses: u64,
    max_consecutive_deadline_misses: u64,
    /// P(next resolved opportunity misses | previous resolved opportunity missed).
    miss_to_miss_transition_rate: Option<f64>,
    fresh_at_use: u64,
    fresh_correct_on_time: u64,
    correct_on_time: u64,
    fresh_but_incorrect: u64,
    correct_but_stale: u64,
    /// Timely outputs / all evaluator-defined opportunities.
    on_time_decision_coverage: Option<f64>,
    /// Fresh + correct + timely outputs / all evaluator-defined opportunities.
    fresh_correct_on_time_coverage: Option<f64>,
    /// Correct timely outputs / all timely outputs.
    verified_accuracy_on_time: Option<f64>,
    availability_to_use_latency_p50_ns: Option<u64>,
    availability_to_use_latency_p95_ns: Option<u64>,
    availability_to_use_latency_p99_ns: Option<u64>,
    availability_to_use_latency_p99_9_ns: Option<u64>,
    availability_to_use_latency_max_ns: Option<u64>,
    source_to_use_latency_p50_ns: Option<u64>,
    source_to_use_latency_p95_ns: Option<u64>,
    source_to_use_latency_p99_ns: Option<u64>,
    source_to_use_latency_p99_9_ns: Option<u64>,
    source_to_use_latency_max_ns: Option<u64>,
    source_to_availability_delay_p95_ns: Option<u64>,
    source_to_availability_delay_p99_ns: Option<u64>,
    evidence_age_at_use_p50_ns: Option<u64>,
    evidence_age_at_use_p95_ns: Option<u64>,
    evidence_age_at_use_p99_ns: Option<u64>,
    evidence_age_at_use_p99_9_ns: Option<u64>,
    evidence_age_at_use_max_ns: Option<u64>,
    time_to_fresh_correct_p95_ns: Option<u64>,
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

/// Nearest-rank percentile. Empty samples produce None, never a fabricated zero.
fn percentile(values: &mut [u64], percentile: usize) -> Option<u64> {
    if percentile == 0 || percentile > 100 {
        return None;
    }
    percentile_per_mille(values, percentile * 10)
}

/// Quantile expressed in per-mille units, so 999 means p99.9 and 1000 means max.
fn percentile_per_mille(values: &mut [u64], quantile_per_mille: usize) -> Option<u64> {
    if values.is_empty() || quantile_per_mille == 0 || quantile_per_mille > 1000 {
        return None;
    }
    // Do not present p99.9 as a meaningful tail statistic with fewer than
    // 1,000 observations. This is only a minimum resolvability gate; rigorous
    // tail claims still require confidence intervals and much larger samples.
    if quantile_per_mille >= 999 && values.len() < 1_000 {
        return None;
    }
    values.sort_unstable();
    let rank = (quantile_per_mille * values.len()).div_ceil(1000);
    values.get(rank.saturating_sub(1)).copied()
}

fn fixture_run() -> FreshnessRun {
    FreshnessRun {
        schema: SCHEMA.to_owned(),
        schema_version: SCHEMA_VERSION.to_owned(),
        workload_id: "synthetic-freshness-fixture-v0.1".to_owned(),
        profile_id: "four-outcome-control".to_owned(),
        source_revision: "synthetic-source-revision".to_owned(),
        configuration_id: "frozen-config-v1".to_owned(),
        decision_contract_id: "decision-contract-v1".to_owned(),
        independent_verifier_id: "synthetic-oracle-v1".to_owned(),
        temporal_work_receipt_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        logical_horizon_end_ns: 1_000,
        opportunities: vec![
            DecisionOpportunity {
                opportunity_id: 1,
                source_event_time_ns: 100,
                available_at_ns: 110,
                deadline_at_ns: 200,
                freshness_budget_ns: 50,
                disposition: Disposition::Answered,
                emitted_at_ns: Some(120),
                consumed_at_ns: Some(130),
                evidence_source_time_ns: Some(105),
                evidence_available_at_ns: Some(115),
                correctness: Some(Correctness::Correct),
            },
            DecisionOpportunity {
                opportunity_id: 2,
                source_event_time_ns: 200,
                available_at_ns: 210,
                deadline_at_ns: 230,
                freshness_budget_ns: 10,
                disposition: Disposition::Answered,
                emitted_at_ns: Some(220),
                consumed_at_ns: Some(240),
                evidence_source_time_ns: Some(215),
                evidence_available_at_ns: Some(218),
                correctness: Some(Correctness::Incorrect),
            },
            DecisionOpportunity {
                opportunity_id: 3,
                source_event_time_ns: 300,
                available_at_ns: 310,
                deadline_at_ns: 400,
                freshness_budget_ns: 50,
                disposition: Disposition::Abstained,
                emitted_at_ns: None,
                consumed_at_ns: None,
                evidence_source_time_ns: None,
                evidence_available_at_ns: None,
                correctness: None,
            },
            DecisionOpportunity {
                opportunity_id: 4,
                source_event_time_ns: 400,
                available_at_ns: 410,
                deadline_at_ns: 500,
                freshness_budget_ns: 50,
                disposition: Disposition::Unserved,
                emitted_at_ns: None,
                consumed_at_ns: None,
                evidence_source_time_ns: None,
                evidence_available_at_ns: None,
                correctness: None,
            },
        ],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let run = fixture_run();
    let summary = run
        .summary()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    println!("{}", serde_json::to_string_pretty(&run)?);
    println!("{}", serde_json::to_string_pretty(&summary)?);
    println!("scope: synthetic freshness contract fixture; no model or hardware claim");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn age_at_use_is_distinct_from_age_at_emission() {
        let run = fixture_run();
        let item = &run.opportunities[0];
        assert_eq!(item.evidence_age_at_emission_ns(), Some(15));
        assert_eq!(item.evidence_age_at_use_ns(), Some(25));
        assert_eq!(item.availability_to_use_latency_ns(), Some(20));
        assert_eq!(item.source_to_use_latency_ns(), Some(30));
        assert_eq!(item.source_to_availability_delay_ns(), Some(10));
    }

    #[test]
    fn summary_keeps_freshness_correctness_and_deadlines_separate() {
        let summary = fixture_run().summary().unwrap();
        assert_eq!(summary.opportunities, 4);
        assert_eq!(summary.answered, 2);
        assert_eq!(summary.on_time_answered, 1);
        assert_eq!(summary.late_answered, 1);
        assert_eq!(summary.abstained, 1);
        assert_eq!(summary.unserved, 1);
        assert_eq!(summary.expired_unserved, 1);
        assert_eq!(summary.pending_unserved, 0);
        assert_eq!(summary.deadline_misses, 2);
        assert_eq!(summary.max_consecutive_deadline_misses, 1);
        assert_eq!(summary.miss_to_miss_transition_rate, Some(0.0));
        assert_eq!(summary.fresh_correct_on_time, 1);
        assert_eq!(summary.fresh_correct_on_time_coverage, Some(0.25));
        assert_eq!(summary.on_time_decision_coverage, Some(0.25));
        assert_eq!(summary.verified_accuracy_on_time, Some(1.0));
        assert_eq!(summary.correct_but_stale, 0);
        assert_eq!(summary.fresh_but_incorrect, 0);
        assert_eq!(summary.availability_to_use_latency_p99_9_ns, None);
        assert_eq!(summary.availability_to_use_latency_max_ns, Some(30));
        assert_eq!(summary.source_to_use_latency_p50_ns, Some(30));
        assert_eq!(summary.source_to_use_latency_p95_ns, Some(40));
        assert_eq!(summary.source_to_availability_delay_p95_ns, Some(10));
        assert_eq!(summary.evidence_age_at_use_p99_9_ns, None);
        assert_eq!(summary.evidence_age_at_use_max_ns, Some(25));
    }

    #[test]
    fn correct_but_stale_is_not_fresh_correct_coverage() {
        let mut run = fixture_run();
        run.opportunities[0].evidence_source_time_ns = Some(60);
        run.opportunities[0].evidence_available_at_ns = Some(90);
        let summary = run.summary().unwrap();
        assert_eq!(summary.correct_but_stale, 1);
        assert_eq!(summary.fresh_correct_on_time, 0);
    }

    #[test]
    fn fresh_but_incorrect_is_not_fresh_correct_coverage() {
        let mut run = fixture_run();
        run.opportunities[0].correctness = Some(Correctness::Incorrect);
        let summary = run.summary().unwrap();
        assert_eq!(summary.fresh_but_incorrect, 1);
        assert_eq!(summary.fresh_correct_on_time, 0);
    }

    #[test]
    fn output_consumed_after_deadline_is_late_even_if_emitted_before_it() {
        let mut run = fixture_run();
        run.opportunities[0].deadline_at_ns = 125;
        let summary = run.summary().unwrap();
        assert_eq!(summary.late_answered, 2);
        assert_eq!(summary.fresh_correct_on_time, 0);
    }

    #[test]
    fn an_expired_opportunity_at_arrival_is_counted_not_rejected() {
        let mut run = fixture_run();
        run.opportunities[0].deadline_at_ns = 100;
        let summary = run.summary().unwrap();
        assert_eq!(summary.late_answered, 2);
        assert_eq!(summary.opportunities, 4);
    }

    #[test]
    fn deadline_tail_reports_bursts_and_conditional_miss_probability() {
        let mut run = fixture_run();
        let third = &mut run.opportunities[2];
        third.disposition = Disposition::Answered;
        third.emitted_at_ns = Some(320);
        third.consumed_at_ns = Some(350);
        third.evidence_source_time_ns = Some(315);
        third.evidence_available_at_ns = Some(318);
        third.correctness = Some(Correctness::Incorrect);
        third.deadline_at_ns = 340;
        run.opportunities[3].deadline_at_ns = 400;

        let summary = run.summary().unwrap();
        assert_eq!(summary.deadline_misses, 3);
        assert_eq!(summary.max_consecutive_deadline_misses, 3);
        assert_eq!(summary.miss_to_miss_transition_rate, Some(1.0));
    }

    #[test]
    fn unserved_opportunities_with_future_deadlines_are_censored_not_counted_as_misses() {
        let mut run = fixture_run();
        run.opportunities[3].deadline_at_ns = 1_500;
        let summary = run.summary().unwrap();
        assert_eq!(summary.unserved, 1);
        assert_eq!(summary.expired_unserved, 0);
        assert_eq!(summary.pending_unserved, 1);
        assert_eq!(summary.deadline_misses, 1);
    }

    #[test]
    fn rejects_evidence_that_was_unavailable_when_output_was_emitted() {
        let mut run = fixture_run();
        run.opportunities[0].evidence_available_at_ns = Some(121);
        assert_eq!(
            run.validate(),
            Err("evidence unavailable at emission is future-information leakage")
        );
    }

    #[test]
    fn rejects_inconsistent_source_and_availability_time() {
        let mut run = fixture_run();
        run.opportunities[0].evidence_source_time_ns = Some(116);
        run.opportunities[0].evidence_available_at_ns = Some(115);
        assert_eq!(
            run.validate(),
            Err("evidence source time cannot follow evidence availability")
        );
    }

    #[test]
    fn rejects_opportunity_available_before_its_source_event() {
        let mut run = fixture_run();
        run.opportunities[0].available_at_ns = 99;
        assert_eq!(
            run.validate(),
            Err("source event time cannot follow opportunity availability")
        );
    }

    #[test]
    fn rejects_duplicate_opportunity_ids() {
        let mut run = fixture_run();
        run.opportunities[1].opportunity_id = run.opportunities[0].opportunity_id;
        assert_eq!(
            run.validate(),
            Err("decision opportunity IDs must be unique")
        );
    }

    #[test]
    fn abstention_and_unserved_cannot_hide_output_fields() {
        let mut run = fixture_run();
        run.opportunities[2].emitted_at_ns = Some(320);
        assert_eq!(
            run.validate(),
            Err("abstained/unserved opportunity cannot carry output evidence")
        );
    }

    #[test]
    fn unknown_percentiles_and_empty_samples_remain_unknown() {
        let mut values = Vec::new();
        assert_eq!(percentile(&mut values, 99), None);
        let mut values = vec![2, 9, 4, 1, 7];
        assert_eq!(percentile(&mut values, 50), Some(4));
        assert_eq!(percentile(&mut values, 95), Some(9));
        assert_eq!(percentile(&mut values, 101), None);

        let mut insufficient_tail = (1_u64..=999).collect::<Vec<_>>();
        assert_eq!(percentile_per_mille(&mut insufficient_tail, 999), None);
        insufficient_tail.push(1_000);
        assert_eq!(percentile_per_mille(&mut insufficient_tail, 999), Some(999));
    }

    #[test]
    fn rejects_invalid_work_receipt_digest() {
        let mut run = fixture_run();
        run.temporal_work_receipt_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            run.validate(),
            Err("work receipt identity must be a 64-character SHA-256 hex digest")
        );
    }

    #[test]
    fn deterministic_serialization_of_frozen_run() {
        let run = fixture_run();
        let first = serde_json::to_string(&run).unwrap();
        let second = serde_json::to_string(&run).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("\"temporal_work_receipt_sha256\""));
    }
}
