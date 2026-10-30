//! Run-wide confidence budgeting for adaptive, selection-conditioned pruning audits.
//!
//! This module composes the finite-population audit primitive across a bounded sequence of
//! adaptive pruning events. Each event receives a predictable, predeclared share of the
//! run-wide confidence-failure budget. The per-event audit is recomputed here from the
//! committed sample receipt and raw evaluator outcomes; callers cannot submit a hand-built
//! aggregate report as evidence.
//!
//! The ledger certifies only the per-event finite-population false-prune-rate statements under
//! the assumptions of pruning_audit::audit_pruning. It does not certify global-optimum or
//! top-K retention, and it does not protect an untrusted process boundary. The seed binding
//! digest records identity/provenance; it is not a cryptographic RNG or proof of entropy.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::pruning_audit::{
    audit_pruning, AuditScore, PruningAuditConfig, PruningAuditError, PruningAuditSample,
    PruningDisposition,
};

const LEDGER_DOMAIN: &[u8] = b"symthaea.nas.run-wide-pruning-risk-ledger.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.run-wide-pruning-risk-event.v0.1";
const SEED_BINDING_DOMAIN: &[u8] = b"symthaea.nas.pruning-audit-seed-binding.v0.1";

/// Immutable risk and confidence limits for one search run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunWidePruningRiskConfig {
    pub run_id: String,
    /// Maximum probability budget for one or more event-level audit bounds to fail.
    /// This is not the probability that the global optimum or final top-K is retained.
    pub run_confidence_failure_budget: f64,
    /// Hard upper bound on the number of adaptive pruning audits in this run.
    pub max_audit_events: usize,
    /// Largest acceptable event-level upper bound on the false-pruned population fraction.
    pub max_false_prune_rate_per_event: f64,
    /// Minimum number of uniformly sampled discarded candidates at every event.
    pub min_audit_samples: usize,
}

/// A chained, aggregate-only receipt for one pruning decision.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningRiskEventReceipt {
    pub run_id: String,
    pub event_id: String,
    pub event_index: usize,
    pub fidelity_id: String,
    pub fidelity_index: usize,
    pub selected_frontier_sha256: String,
    pub pruned_population_sha256: String,
    pub audit_sample_sha256: String,
    /// Binds run/event identity to the seed and sample. It does not prove cryptographic entropy.
    pub audit_seed_binding_sha256: String,
    pub pruned_population_size: usize,
    pub audit_sample_size: usize,
    pub false_pruned_count: usize,
    pub empirical_false_prune_rate: f64,
    pub upper_bound_false_pruned_count: usize,
    pub upper_bound_false_prune_rate: f64,
    pub event_confidence_failure_budget: f64,
    pub event_confidence_level: f64,
    pub cumulative_confidence_failure_budget_spent: f64,
    pub max_false_prune_rate: f64,
    pub disposition: PruningDisposition,
    pub pruning_remains_enabled: bool,
    pub previous_receipt_sha256: String,
    pub receipt_sha256: String,
}

impl PruningRiskEventReceipt {
    /// Recompute the receipt digest to detect edits after the event was committed.
    pub fn verify_digest(&self) -> bool {
        digest_event_receipt(self) == self.receipt_sha256
    }
}

/// Aggregate run summary. No sentinel identities or raw outcomes are exposed.
#[derive(Debug, Clone, PartialEq)]
pub struct RunWidePruningRiskSummary {
    pub run_id: String,
    pub events_recorded: usize,
    pub max_audit_events: usize,
    pub confidence_failure_budget: f64,
    pub confidence_failure_budget_spent: f64,
    pub pruning_enabled: bool,
    pub stop_reason: Option<String>,
    pub genesis_sha256: String,
    pub last_receipt_sha256: String,
}

/// A fail-closed validation or audit error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunWidePruningRiskError {
    pub field: String,
    pub message: String,
}

impl RunWidePruningRiskError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for RunWidePruningRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for RunWidePruningRiskError {}

impl From<PruningAuditError> for RunWidePruningRiskError {
    fn from(error: PruningAuditError) -> Self {
        Self::new(error.field, error.message)
    }
}

/// Stateful run-level audit ledger. Its configuration and event history are private so callers
/// can only append events through the verified audit path.
pub struct RunWidePruningRiskLedger {
    config: RunWidePruningRiskConfig,
    genesis_sha256: String,
    events: Vec<PruningRiskEventReceipt>,
    pruning_enabled: bool,
    stop_reason: Option<String>,
}

impl RunWidePruningRiskLedger {
    pub fn new(
        config: RunWidePruningRiskConfig,
    ) -> Result<Self, RunWidePruningRiskError> {
        validate_config(&config)?;
        let genesis_sha256 = digest_ledger_genesis(&config);
        Ok(Self {
            config,
            genesis_sha256,
            events: Vec::new(),
            pruning_enabled: true,
            stop_reason: None,
        })
    }

    /// The predictable uniform alpha-spending schedule.
    ///
    /// Every event receives delta_run / max_audit_events, irrespective of prior observations.
    /// Therefore the sum of all event allocations cannot exceed delta_run, without assuming
    /// independent audit outcomes. Unused event allocations are simply not spent.
    pub fn event_confidence_failure_budget(
        &self,
        event_index: usize,
    ) -> Result<f64, RunWidePruningRiskError> {
        if event_index >= self.config.max_audit_events {
            return Err(RunWidePruningRiskError::new(
                "event_index",
                "event index exceeds the configured maximum audit count",
            ));
        }
        Ok(self.config.run_confidence_failure_budget
            / self.config.max_audit_events as f64)
    }

    /// Verify a committed sentinel sample, recompute its exact finite-population audit, and
    /// append a chained event receipt. The raw sentinel outcomes are consumed here; only the
    /// aggregate receipt is returned.
    ///
    /// Any invalid event disables further pruning for this run. An audit whose sample is too
    /// small or whose upper false-prune bound exceeds the configured limit is recorded but
    /// also permanently disables further pruning.
    pub fn record_event(
        &mut self,
        event_id: &str,
        fidelity_id: &str,
        selected_scores: &[AuditScore],
        audit_sample: &PruningAuditSample,
        audited_pruned_scores: &[AuditScore],
    ) -> Result<PruningRiskEventReceipt, RunWidePruningRiskError> {
        if !self.pruning_enabled {
            return Err(RunWidePruningRiskError::new(
                "pruning_enabled",
                self.stop_reason
                    .clone()
                    .unwrap_or_else(|| "pruning was disabled by an earlier audit".to_owned()),
            ));
        }

        if event_id.trim().is_empty() {
            return self.fail_closed("event_id", "must be a non-empty string");
        }
        if fidelity_id.trim().is_empty() {
            return self.fail_closed("fidelity_id", "must be a non-empty string");
        }
        if self.events.len() >= self.config.max_audit_events {
            return self.fail_closed(
                "max_audit_events",
                "the run has exhausted its predeclared audit-event budget",
            );
        }
        if self.events.iter().any(|event| event.event_id == event_id) {
            return self.fail_closed("event_id", "duplicate event ID in this run");
        }
        if selected_scores.is_empty() {
            return self.fail_closed("selected_scores", "promoted frontier must not be empty");
        }

        let event_index = self.events.len();
        let delta_event = self.event_confidence_failure_budget(event_index)?;
        let confidence_level = 1.0 - delta_event;
        let audit_config = PruningAuditConfig {
            min_audit_samples: self.config.min_audit_samples,
            max_false_prune_rate: self.config.max_false_prune_rate_per_event,
            confidence_level,
        };

        let report = match audit_pruning(
            selected_scores,
            audit_sample,
            audited_pruned_scores,
            &audit_config,
        ) {
            Ok(report) => report,
            Err(error) => {
                let field = error.field.clone();
                let message = error.message.clone();
                return self.fail_closed(&field, &message);
            }
        };

        let selected_frontier_sha256 = match digest_selected_frontier(selected_scores) {
            Ok(digest) => digest,
            Err(error) => return self.fail_closed(&error.field, &error.message),
        };

        let previous_receipt_sha256 = self.events.last()
            .map(|event| event.receipt_sha256.clone())
            .unwrap_or_else(|| self.genesis_sha256.clone());
        let cumulative_spent = delta_event * (event_index + 1) as f64;
        if cumulative_spent > self.config.run_confidence_failure_budget + 1e-12 {
            return self.fail_closed(
                "confidence_budget",
                "cumulative event allocation exceeds the run-wide budget",
            );
        }

        let audit_seed_binding_sha256 = digest_seed_binding(
            &self.config.run_id,
            event_id,
            event_index,
            report.sample_seed,
            &report.population_sha256,
            &report.sample_sha256,
        );
        let pruning_remains_enabled = report.disposition == PruningDisposition::ContinuePruning;
        let mut receipt = PruningRiskEventReceipt {
            run_id: self.config.run_id.clone(),
            event_id: event_id.to_owned(),
            event_index,
            fidelity_id: fidelity_id.to_owned(),
            fidelity_index: report.target_fidelity_index,
            selected_frontier_sha256,
            pruned_population_sha256: report.population_sha256,
            audit_sample_sha256: report.sample_sha256,
            audit_seed_binding_sha256,
            pruned_population_size: report.pruned_population_size,
            audit_sample_size: report.audit_sample_size,
            false_pruned_count: report.false_pruned_count,
            empirical_false_prune_rate: report.empirical_false_prune_rate,
            upper_bound_false_pruned_count: report.upper_bound_false_pruned_count,
            upper_bound_false_prune_rate: report.upper_bound_false_prune_rate,
            event_confidence_failure_budget: delta_event,
            event_confidence_level: report.confidence_level,
            cumulative_confidence_failure_budget_spent: cumulative_spent,
            max_false_prune_rate: self.config.max_false_prune_rate_per_event,
            disposition: report.disposition,
            pruning_remains_enabled,
            previous_receipt_sha256,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = digest_event_receipt(&receipt);
        self.events.push(receipt.clone());

        if !pruning_remains_enabled {
            self.pruning_enabled = false;
            self.stop_reason = Some(match report.disposition {
                PruningDisposition::DisablePruningInsufficientAudit =>
                    "audit sample did not meet the minimum sample-size requirement".to_owned(),
                PruningDisposition::DisablePruningObservedRisk =>
                    "upper confidence bound exceeded the pre-registered false-prune risk limit".to_owned(),
                PruningDisposition::ContinuePruning => unreachable!(),
            });
        }

        Ok(receipt)
    }

    /// Verify event ordering, alpha spending, chained digests and fail-closed state.
    pub fn verify_integrity(&self) -> Result<(), RunWidePruningRiskError> {
        validate_config(&self.config)?;
        if digest_ledger_genesis(&self.config) != self.genesis_sha256 {
            return Err(RunWidePruningRiskError::new(
                "genesis_sha256",
                "ledger configuration does not match its genesis digest",
            ));
        }
        if self.events.len() > self.config.max_audit_events {
            return Err(RunWidePruningRiskError::new(
                "events",
                "recorded event count exceeds the configured maximum",
            ));
        }

        let delta_event = self.config.run_confidence_failure_budget
            / self.config.max_audit_events as f64;
        let mut seen_event_ids = HashSet::with_capacity(self.events.len());
        let mut previous = self.genesis_sha256.clone();
        let mut expected_enabled = true;

        for (index, event) in self.events.iter().enumerate() {
            if !expected_enabled {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}]"),
                    "an event was recorded after a prior audit disabled pruning",
                ));
            }
            if event.run_id != self.config.run_id
                || event.event_index != index
                || event.event_id.trim().is_empty()
                || !seen_event_ids.insert(event.event_id.as_str())
                || event.fidelity_id.trim().is_empty()
            {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}]"),
                    "run identity, event ordering, unique ID or fidelity identity is invalid",
                ));
            }
            let expected_cumulative = delta_event * (index + 1) as f64;
            if !approximately_equal(event.event_confidence_failure_budget, delta_event)
                || !approximately_equal(event.event_confidence_level, 1.0 - delta_event)
                || !approximately_equal(
                    event.cumulative_confidence_failure_budget_spent,
                    expected_cumulative,
                )
                || expected_cumulative > self.config.run_confidence_failure_budget + 1e-12
                || !approximately_equal(
                    event.max_false_prune_rate,
                    self.config.max_false_prune_rate_per_event,
                )
            {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}].confidence_budget"),
                    "event allocation, confidence level, cumulative spend or risk threshold is invalid",
                ));
            }
            if event.audit_sample_size > event.pruned_population_size
                || event.pruned_population_size == 0
                || event.false_pruned_count > event.audit_sample_size
                || !approximately_equal(
                    event.empirical_false_prune_rate,
                    event.false_pruned_count as f64 / event.audit_sample_size as f64,
                )
                || event.upper_bound_false_pruned_count > event.pruned_population_size
                || !approximately_equal(
                    event.upper_bound_false_prune_rate,
                    event.upper_bound_false_pruned_count as f64
                        / event.pruned_population_size as f64,
                )
            {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}].audit_statistics"),
                    "audit population counts or rates are inconsistent",
                ));
            }
            if event.previous_receipt_sha256 != previous || !event.verify_digest() {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}].receipt_sha256"),
                    "receipt chain or event digest does not verify",
                ));
            }
            if event.pruning_remains_enabled
                != (event.disposition == PruningDisposition::ContinuePruning)
            {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}].disposition"),
                    "pruning state does not match the audit disposition",
                ));
            }
            if event.disposition == PruningDisposition::ContinuePruning
                && (event.audit_sample_size < self.config.min_audit_samples
                    || event.upper_bound_false_prune_rate
                        > self.config.max_false_prune_rate_per_event)
            {
                return Err(RunWidePruningRiskError::new(
                    format!("events[{index}].disposition"),
                    "continue-pruning decision violates the sample or risk threshold",
                ));
            }

            expected_enabled = event.pruning_remains_enabled;
            previous = event.receipt_sha256.clone();
        }

        if self.pruning_enabled != expected_enabled {
            return Err(RunWidePruningRiskError::new(
                "pruning_enabled",
                "ledger state does not match the final event disposition",
            ));
        }
        if !self.pruning_enabled && self.stop_reason.as_deref().unwrap_or("").is_empty() {
            return Err(RunWidePruningRiskError::new(
                "stop_reason",
                "a disabled ledger must retain its stop reason",
            ));
        }
        Ok(())
    }

    pub fn events(&self) -> &[PruningRiskEventReceipt] {
        &self.events
    }

    pub fn pruning_enabled(&self) -> bool {
        self.pruning_enabled
    }

    pub fn summary(&self) -> RunWidePruningRiskSummary {
        let spent = self.events.last()
            .map(|event| event.cumulative_confidence_failure_budget_spent)
            .unwrap_or(0.0);
        let last_receipt_sha256 = self.events.last()
            .map(|event| event.receipt_sha256.clone())
            .unwrap_or_else(|| self.genesis_sha256.clone());
        RunWidePruningRiskSummary {
            run_id: self.config.run_id.clone(),
            events_recorded: self.events.len(),
            max_audit_events: self.config.max_audit_events,
            confidence_failure_budget: self.config.run_confidence_failure_budget,
            confidence_failure_budget_spent: spent,
            pruning_enabled: self.pruning_enabled,
            stop_reason: self.stop_reason.clone(),
            genesis_sha256: self.genesis_sha256.clone(),
            last_receipt_sha256,
        }
    }

    fn fail_closed<T>(
        &mut self,
        field: &str,
        message: &str,
    ) -> Result<T, RunWidePruningRiskError> {
        self.pruning_enabled = false;
        self.stop_reason = Some(format!("{field}: {message}"));
        Err(RunWidePruningRiskError::new(field, message))
    }
}

fn validate_config(
    config: &RunWidePruningRiskConfig,
) -> Result<(), RunWidePruningRiskError> {
    if config.run_id.trim().is_empty() {
        return Err(RunWidePruningRiskError::new("run_id", "must be non-empty"));
    }
    if !config.run_confidence_failure_budget.is_finite()
        || config.run_confidence_failure_budget <= 0.0
        || config.run_confidence_failure_budget >= 1.0
    {
        return Err(RunWidePruningRiskError::new(
            "run_confidence_failure_budget",
            "must be finite and strictly between zero and one",
        ));
    }
    if config.max_audit_events == 0 {
        return Err(RunWidePruningRiskError::new(
            "max_audit_events",
            "must be positive",
        ));
    }
    if !config.max_false_prune_rate_per_event.is_finite()
        || !(0.0..=1.0).contains(&config.max_false_prune_rate_per_event)
    {
        return Err(RunWidePruningRiskError::new(
            "max_false_prune_rate_per_event",
            "must be finite and in [0, 1]",
        ));
    }
    if config.min_audit_samples == 0 {
        return Err(RunWidePruningRiskError::new(
            "min_audit_samples",
            "must be positive",
        ));
    }
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

fn digest_selected_frontier(
    selected_scores: &[AuditScore],
) -> Result<String, RunWidePruningRiskError> {
    let mut rows: Vec<&AuditScore> = selected_scores.iter().collect();
    rows.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nas.pruning-selected-frontier.v0.1");
    let mut previous_id: Option<&str> = None;
    for (index, row) in rows.iter().enumerate() {
        if row.candidate_id.trim().is_empty() || !row.score.is_finite() {
            return Err(RunWidePruningRiskError::new(
                format!("selected_scores[{index}]"),
                "candidate ID must be non-empty and score finite",
            ));
        }
        if previous_id == Some(row.candidate_id.as_str()) {
            return Err(RunWidePruningRiskError::new(
                format!("selected_scores[{index}].candidate_id"),
                "duplicate candidate ID in promoted frontier",
            ));
        }
        previous_id = Some(row.candidate_id.as_str());
        update_string(&mut hasher, &row.candidate_id);
        hasher.update((row.fidelity_index as u64).to_be_bytes());
        hasher.update(row.score.to_bits().to_be_bytes());
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

fn digest_ledger_genesis(config: &RunWidePruningRiskConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(LEDGER_DOMAIN);
    update_string(&mut hasher, &config.run_id);
    hasher.update(config.run_confidence_failure_budget.to_bits().to_be_bytes());
    hasher.update((config.max_audit_events as u64).to_be_bytes());
    hasher.update(config.max_false_prune_rate_per_event.to_bits().to_be_bytes());
    hasher.update((config.min_audit_samples as u64).to_be_bytes());
    hex_digest(hasher.finalize().as_slice())
}

fn digest_seed_binding(
    run_id: &str,
    event_id: &str,
    event_index: usize,
    seed: u64,
    population_sha256: &str,
    sample_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SEED_BINDING_DOMAIN);
    update_string(&mut hasher, run_id);
    update_string(&mut hasher, event_id);
    hasher.update((event_index as u64).to_be_bytes());
    hasher.update(seed.to_be_bytes());
    update_string(&mut hasher, population_sha256);
    update_string(&mut hasher, sample_sha256);
    hex_digest(hasher.finalize().as_slice())
}

fn digest_event_receipt(receipt: &PruningRiskEventReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DOMAIN);
    update_string(&mut hasher, &receipt.run_id);
    update_string(&mut hasher, &receipt.event_id);
    hasher.update((receipt.event_index as u64).to_be_bytes());
    update_string(&mut hasher, &receipt.fidelity_id);
    hasher.update((receipt.fidelity_index as u64).to_be_bytes());
    update_string(&mut hasher, &receipt.selected_frontier_sha256);
    update_string(&mut hasher, &receipt.pruned_population_sha256);
    update_string(&mut hasher, &receipt.audit_sample_sha256);
    update_string(&mut hasher, &receipt.audit_seed_binding_sha256);
    hasher.update((receipt.pruned_population_size as u64).to_be_bytes());
    hasher.update((receipt.audit_sample_size as u64).to_be_bytes());
    hasher.update((receipt.false_pruned_count as u64).to_be_bytes());
    hasher.update(receipt.empirical_false_prune_rate.to_bits().to_be_bytes());
    hasher.update((receipt.upper_bound_false_pruned_count as u64).to_be_bytes());
    hasher.update(receipt.upper_bound_false_prune_rate.to_bits().to_be_bytes());
    hasher.update(receipt.event_confidence_failure_budget.to_bits().to_be_bytes());
    hasher.update(receipt.event_confidence_level.to_bits().to_be_bytes());
    hasher.update(receipt.cumulative_confidence_failure_budget_spent.to_bits().to_be_bytes());
    hasher.update(receipt.max_false_prune_rate.to_bits().to_be_bytes());
    hasher.update([match receipt.disposition {
        PruningDisposition::ContinuePruning => 0,
        PruningDisposition::DisablePruningInsufficientAudit => 1,
        PruningDisposition::DisablePruningObservedRisk => 2,
    }]);
    hasher.update([u8::from(receipt.pruning_remains_enabled)]);
    update_string(&mut hasher, &receipt.previous_receipt_sha256);
    hex_digest(hasher.finalize().as_slice())
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    let alphabet = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        encoded.push(alphabet[(byte >> 4) as usize] as char);
        encoded.push(alphabet[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pruning_audit::prepare_pruning_audit_sample;

    fn config(max_audit_events: usize, max_false_prune_rate_per_event: f64) -> RunWidePruningRiskConfig {
        RunWidePruningRiskConfig {
            run_id: "run-001".to_owned(),
            run_confidence_failure_budget: 0.05,
            max_audit_events,
            max_false_prune_rate_per_event,
            min_audit_samples: 2,
        }
    }

    fn make_sample(ids: &[&str], seed: u64) -> PruningAuditSample {
        let population: Vec<String> = ids.iter().map(|id| (*id).to_owned()).collect();
        prepare_pruning_audit_sample(&population, population.len(), seed)
            .expect("sample preparation should succeed")
    }

    fn selected(score: f64) -> Vec<AuditScore> {
        vec![AuditScore {
            candidate_id: "promoted-1".to_owned(),
            fidelity_index: 2,
            score,
        }]
    }

    fn audited(sample: &PruningAuditSample, scores: &[f64]) -> Vec<AuditScore> {
        assert_eq!(sample.sampled_candidate_ids().len(), scores.len());
        sample.sampled_candidate_ids().iter().zip(scores).map(|(id, score)| AuditScore {
            candidate_id: id.clone(),
            fidelity_index: 2,
            score: *score,
        }).collect()
    }

    #[test]
    fn spends_a_predictable_uniform_fraction_of_the_run_budget() {
        let ledger = RunWidePruningRiskLedger::new(config(4, 0.25)).unwrap();
        let delta = ledger.event_confidence_failure_budget(0).unwrap();
        assert!((delta - 0.0125).abs() < 1e-12);
        assert!((ledger.event_confidence_failure_budget(3).unwrap() - delta).abs() < 1e-12);
        assert!(ledger.event_confidence_failure_budget(4).is_err());
    }

    #[test]
    fn records_a_valid_census_audit_and_verifies_the_chain() {
        let mut ledger = RunWidePruningRiskLedger::new(config(4, 0.25)).unwrap();
        let sample = make_sample(&["a", "b", "c", "d"], 11);
        let report = ledger.record_event(
            "rung-0",
            "epoch-15",
            &selected(0.8),
            &sample,
            &audited(&sample, &[0.1, 0.2, 0.3, 0.4]),
        ).unwrap();

        assert_eq!(report.event_index, 0);
        assert_eq!(report.disposition, PruningDisposition::ContinuePruning);
        assert!(report.pruning_remains_enabled);
        assert!((report.event_confidence_failure_budget - 0.0125).abs() < 1e-12);
        assert!((report.event_confidence_level - 0.9875).abs() < 1e-12);
        assert!(report.verify_digest());
        ledger.verify_integrity().unwrap();
        assert!(ledger.pruning_enabled());
    }

    #[test]
    fn cumulative_confidence_spend_never_exceeds_run_budget() {
        let mut ledger = RunWidePruningRiskLedger::new(config(2, 0.25)).unwrap();
        for index in 0..2 {
            let ids = if index == 0 { vec!["a", "b"] } else { vec!["c", "d"] };
            let sample = make_sample(&ids, index as u64 + 1);
            let event_id = format!("rung-{index}");
            ledger.record_event(
                &event_id,
                "epoch-15",
                &selected(0.8),
                &sample,
                &audited(&sample, &[0.1, 0.2]),
            ).unwrap();
        }
        let summary = ledger.summary();
        assert_eq!(summary.events_recorded, 2);
        assert!((summary.confidence_failure_budget_spent - 0.05).abs() < 1e-12);
        assert!(summary.confidence_failure_budget_spent <= summary.confidence_failure_budget + 1e-12);
        ledger.verify_integrity().unwrap();
    }

    #[test]
    fn observed_false_prune_risk_disables_future_pruning() {
        let mut ledger = RunWidePruningRiskLedger::new(config(3, 0.0)).unwrap();
        let sample = make_sample(&["a", "b"], 2);
        let result = ledger.record_event(
            "rung-0",
            "epoch-15",
            &selected(0.8),
            &sample,
            &audited(&sample, &[0.9, 0.1]),
        ).unwrap();

        assert_eq!(result.disposition, PruningDisposition::DisablePruningObservedRisk);
        assert!(!ledger.pruning_enabled());
        assert!(ledger.record_event(
            "rung-1",
            "epoch-30",
            &selected(0.8),
            &sample,
            &audited(&sample, &[0.1, 0.2]),
        ).is_err());
        ledger.verify_integrity().unwrap();
    }

    #[test]
    fn insufficient_audit_disables_future_pruning() {
        let mut cfg = config(2, 0.5);
        cfg.min_audit_samples = 3;
        let mut ledger = RunWidePruningRiskLedger::new(cfg).unwrap();
        let sample = make_sample(&["a", "b"], 3);
        let result = ledger.record_event(
            "rung-0",
            "epoch-15",
            &selected(0.8),
            &sample,
            &audited(&sample, &[0.1, 0.2]),
        ).unwrap();
        assert_eq!(result.disposition, PruningDisposition::DisablePruningInsufficientAudit);
        assert!(!ledger.pruning_enabled());
        ledger.verify_integrity().unwrap();
    }

    #[test]
    fn malformed_audit_evidence_disables_pruning() {
        let mut ledger = RunWidePruningRiskLedger::new(config(2, 0.5)).unwrap();
        let sample = make_sample(&["a", "b"], 4);
        let wrong_fidelity = vec![AuditScore {
            candidate_id: sample.sampled_candidate_ids()[0].clone(),
            fidelity_index: 1,
            score: 0.1,
        }, AuditScore {
            candidate_id: sample.sampled_candidate_ids()[1].clone(),
            fidelity_index: 2,
            score: 0.2,
        }];
        assert!(ledger.record_event(
            "rung-0",
            "epoch-15",
            &selected(0.8),
            &sample,
            &wrong_fidelity,
        ).is_err());
        assert!(!ledger.pruning_enabled());
    }

    #[test]
    fn duplicate_event_ids_fail_closed() {
        let mut ledger = RunWidePruningRiskLedger::new(config(3, 0.5)).unwrap();
        let sample = make_sample(&["a", "b"], 5);
        for _ in 0..1 {
            ledger.record_event(
                "same-event",
                "epoch-15",
                &selected(0.8),
                &sample,
                &audited(&sample, &[0.1, 0.2]),
            ).unwrap();
        }
        let next = make_sample(&["c", "d"], 6);
        assert!(ledger.record_event(
            "same-event",
            "epoch-30",
            &selected(0.8),
            &next,
            &audited(&next, &[0.1, 0.2]),
        ).is_err());
        assert!(!ledger.pruning_enabled());
    }

    #[test]
    fn receipt_tampering_is_detected() {
        let mut ledger = RunWidePruningRiskLedger::new(config(2, 0.5)).unwrap();
        let sample = make_sample(&["a", "b"], 7);
        ledger.record_event(
            "rung-0",
            "epoch-15",
            &selected(0.8),
            &sample,
            &audited(&sample, &[0.1, 0.2]),
        ).unwrap();
        ledger.events[0].upper_bound_false_prune_rate = 0.99;
        assert!(ledger.verify_integrity().is_err());
    }

    #[test]
    fn frontier_digest_is_independent_of_selected_input_order() {
        let left = vec![
            AuditScore { candidate_id: "a".into(), fidelity_index: 2, score: 0.4 },
            AuditScore { candidate_id: "b".into(), fidelity_index: 2, score: 0.7 },
        ];
        let right = vec![left[1].clone(), left[0].clone()];
        assert_eq!(digest_selected_frontier(&left).unwrap(), digest_selected_frontier(&right).unwrap());
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert!(RunWidePruningRiskLedger::new(config(0, 0.5)).is_err());
        assert!(RunWidePruningRiskLedger::new(config(2, f64::NAN)).is_err());
        let mut invalid = config(2, 0.5);
        invalid.run_confidence_failure_budget = 1.0;
        assert!(RunWidePruningRiskLedger::new(invalid).is_err());
    }
}
