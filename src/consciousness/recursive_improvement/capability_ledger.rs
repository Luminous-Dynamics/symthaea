// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound operational self-knowledge for capability and skill claims.
//!
//! A capability claim is separate from evidence that qualifies it. Predictions must be recorded
//! before outcomes, outcomes must bind to the exact subject and forecast, and only pre-approved
//! evaluator identities/revisions on held-out or transfer cases can qualify a claim. Training
//! success remains useful telemetry but cannot grant qualification.
//!
//! This module validates provenance metadata and policy binding; it does not verify evaluator
//! signatures, fetch runner artifacts, or prove that a supplied receipt is authentic. Those checks
//! belong to the independent evidence/verifier layer. A constructed receipt or this in-memory
//! ledger, by itself, is not proof that a real run occurred.
//!
//! This first version measures prospective success-probability calibration. Expected compute cost
//! and realized compute are retained for analysis, but do not yet participate in qualification.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Exact software/evaluation subject to which all forecasts and outcomes are bound.
///
/// Digest fields are supplied by the build/evaluation authority, not derived by this type.
/// SHA-1- or SHA-256-sized Git object IDs are accepted for source identities to support both
/// repository object formats; configuration, evaluation-profile, claim, and receipt digests use
/// SHA-256.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectIdentity {
    pub source_commit: String,
    pub source_tree: String,
    pub configuration_sha256: String,
    pub evaluation_profile_sha256: String,
    /// Optional only when the subject genuinely has no separate model/data artifact bundle.
    pub model_data_sha256: Option<String>,
}

impl SubjectIdentity {
    pub fn validate(&self) -> Result<(), CapabilityLedgerError> {
        if !is_hex_digest(&self.source_commit, &[40, 64])
            || !is_hex_digest(&self.source_tree, &[40, 64])
            || !is_hex_digest(&self.configuration_sha256, &[64])
            || !is_hex_digest(&self.evaluation_profile_sha256, &[64])
            || self
                .model_data_sha256
                .as_ref()
                .is_some_and(|value| !is_hex_digest(value, &[64]))
        {
            return Err(CapabilityLedgerError::InvalidSubjectIdentity);
        }
        Ok(())
    }
}

/// Where a forecast is evaluated. Only held-out and transfer cases can qualify a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilitySplit {
    Training,
    HeldOut,
    Transfer,
    Diagnostic,
}

impl CapabilitySplit {
    fn may_qualify(self) -> bool {
        matches!(self, Self::HeldOut | Self::Transfer)
    }
}

/// Lifecycle of an operational capability claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityLifecycle {
    Proposed,
    Discovered,
    Candidate,
    Qualified,
    Restricted,
    Suspended,
    Retired,
}

/// Versioned, scoped statement about what the subject can do.
///
/// claim_sha256 must be the digest of the canonical claim payload maintained by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityClaim {
    pub claim_id: String,
    pub claim_sha256: String,
    pub description: String,
    pub scope: String,
    pub lifecycle: CapabilityLifecycle,
}

impl CapabilityClaim {
    pub fn new(
        claim_id: impl Into<String>,
        claim_sha256: impl Into<String>,
        description: impl Into<String>,
        scope: impl Into<String>,
    ) -> Result<Self, CapabilityLedgerError> {
        let claim = Self {
            claim_id: claim_id.into(),
            claim_sha256: claim_sha256.into(),
            description: description.into(),
            scope: scope.into(),
            lifecycle: CapabilityLifecycle::Proposed,
        };
        if !non_empty(&claim.claim_id)
            || !is_hex_digest(&claim.claim_sha256, &[64])
            || !non_empty(&claim.description)
            || !non_empty(&claim.scope)
        {
            return Err(CapabilityLedgerError::InvalidClaim);
        }
        Ok(claim)
    }
}

/// Prospective forecast written before the outcome is known.
///
/// sequence is a monotonically increasing logical sequence supplied by the evaluation transcript;
/// wall-clock timestamps are deliberately not used to establish ordering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityForecast {
    pub forecast_id: String,
    pub claim_id: String,
    pub claim_sha256: String,
    pub subject: SubjectIdentity,
    pub sequence: u64,
    pub context_id: String,
    pub split: CapabilitySplit,
    pub predicted_success_probability: f64,
    pub expected_compute_units: f64,
    pub predicted_failure_mode: Option<String>,
}

/// Outcome evidence linked to one previously recorded forecast.
///
/// The receipt's evaluator identity/revision must match an externally configured
/// QualificationPolicy. receipt_sha256 is an evidence root/reference, not a signature; the
/// evidence pipeline must independently authenticate and verify it before admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityOutcomeReceipt {
    pub forecast_id: String,
    pub subject: SubjectIdentity,
    pub sequence: u64,
    pub observed_success: bool,
    pub split: CapabilitySplit,
    pub evaluator_identity: String,
    pub evaluator_revision: String,
    pub receipt_sha256: String,
    pub actual_compute_units: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityMetrics {
    /// Total forecasts associated with the claim, including unresolved forecasts.
    pub forecast_count: usize,
    pub resolved_count: usize,
    /// Resolved held-out and transfer cases (not training or diagnostics).
    pub qualification_samples: usize,
    pub held_out_samples: usize,
    pub transfer_samples: usize,
    /// None means no resolved observations are available; it is not a zero error score.
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub mean_predicted_success_probability: Option<f64>,
    pub empirical_success_rate: Option<f64>,
    pub mean_absolute_compute_error: Option<f64>,
}

/// Externally supplied qualification policy. Evaluator identity and accepted revisions must be
/// pinned by the trusted evaluation/verification layer, not selected by the capability claimant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationPolicy {
    pub minimum_resolved: usize,
    pub minimum_held_out: usize,
    pub maximum_brier_score: f64,
    pub maximum_expected_calibration_error: f64,
    pub trusted_evaluator_identity: String,
    pub trusted_evaluator_revisions: Vec<String>,
}

impl QualificationPolicy {
    pub fn validate(&self) -> Result<(), CapabilityLedgerError> {
        if self.minimum_resolved == 0
            || self.minimum_held_out == 0
            || !self.maximum_brier_score.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_brier_score)
            || !self.maximum_expected_calibration_error.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_expected_calibration_error)
            || !non_empty(&self.trusted_evaluator_identity)
            || self.trusted_evaluator_revisions.is_empty()
            || self.trusted_evaluator_revisions.iter().any(|r| !non_empty(r))
        {
            return Err(CapabilityLedgerError::InvalidQualificationPolicy);
        }
        Ok(())
    }

    fn accepts(&self, receipt: &CapabilityOutcomeReceipt) -> bool {
        receipt.evaluator_identity == self.trusted_evaluator_identity
            && self
                .trusted_evaluator_revisions
                .iter()
                .any(|revision| revision == &receipt.evaluator_revision)
            && receipt.split.may_qualify()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityLedgerError {
    InvalidSubjectIdentity,
    InvalidClaim,
    DuplicateClaim,
    UnknownClaim,
    InvalidLifecycleTransition,
    InvalidForecast,
    DuplicateForecast,
    UnknownForecast,
    AlreadyResolved,
    SubjectMismatch,
    ClaimVersionMismatch,
    SequenceNotMonotonic,
    SplitMismatch,
    InvalidOutcomeReceipt,
    ReceiptReplay,
    InvalidQualificationPolicy,
    InsufficientEvidence,
    QualificationRejected,
}

impl fmt::Display for CapabilityLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CapabilityLedgerError {}

#[derive(Debug, Clone)]
struct ForecastRecord {
    forecast: CapabilityForecast,
    receipt: Option<CapabilityOutcomeReceipt>,
}

/// In-memory ledger pinned to one exact subject identity.
#[derive(Debug, Clone)]
pub struct CapabilityLedger {
    subject: SubjectIdentity,
    claims: BTreeMap<String, CapabilityClaim>,
    forecasts: BTreeMap<String, ForecastRecord>,
    /// Most recent accepted forecast/outcome event sequence for the whole ledger.
    last_sequence: u64,
}

impl CapabilityLedger {
    pub fn new(subject: SubjectIdentity) -> Result<Self, CapabilityLedgerError> {
        subject.validate()?;
        Ok(Self {
            subject,
            claims: BTreeMap::new(),
            forecasts: BTreeMap::new(),
            last_sequence: 0,
        })
    }

    pub fn subject(&self) -> &SubjectIdentity {
        &self.subject
    }

    pub fn add_claim(&mut self, claim: CapabilityClaim) -> Result<(), CapabilityLedgerError> {
        if self.claims.contains_key(&claim.claim_id) {
            return Err(CapabilityLedgerError::DuplicateClaim);
        }
        // Re-validate even if the claim was deserialized rather than built with new().
        let validated = CapabilityClaim::new(
            claim.claim_id.clone(),
            claim.claim_sha256.clone(),
            claim.description.clone(),
            claim.scope.clone(),
        )?;
        self.claims.insert(
            validated.claim_id.clone(),
            CapabilityClaim {
                lifecycle: CapabilityLifecycle::Proposed,
                ..validated
            },
        );
        Ok(())
    }

    pub fn claim(&self, claim_id: &str) -> Option<&CapabilityClaim> {
        self.claims.get(claim_id)
    }

    /// Advance a non-qualification lifecycle transition. A caller cannot set Qualified directly.
    pub fn transition_claim(
        &mut self,
        claim_id: &str,
        next: CapabilityLifecycle,
    ) -> Result<(), CapabilityLedgerError> {
        if next == CapabilityLifecycle::Qualified {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        let claim = self
            .claims
            .get_mut(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        let allowed = match claim.lifecycle {
            CapabilityLifecycle::Proposed => {
                matches!(next, CapabilityLifecycle::Discovered | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Discovered => {
                matches!(next, CapabilityLifecycle::Candidate | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Candidate => matches!(
                next,
                CapabilityLifecycle::Restricted
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Qualified => matches!(
                next,
                CapabilityLifecycle::Restricted
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Restricted => matches!(
                next,
                CapabilityLifecycle::Candidate
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Suspended => {
                matches!(next, CapabilityLifecycle::Candidate | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Retired => false,
        };
        if !allowed {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        claim.lifecycle = next;
        Ok(())
    }

    /// Admit a forecast bound to the exact subject and immutable claim version.
    pub fn record_forecast(
        &mut self,
        forecast: CapabilityForecast,
    ) -> Result<(), CapabilityLedgerError> {
        if !non_empty(&forecast.forecast_id)
            || !non_empty(&forecast.context_id)
            || !is_valid_probability(forecast.predicted_success_probability)
            || !forecast.expected_compute_units.is_finite()
            || forecast.expected_compute_units < 0.0
            || forecast
                .predicted_failure_mode
                .as_ref()
                .is_some_and(|reason| !non_empty(reason))
        {
            return Err(CapabilityLedgerError::InvalidForecast);
        }
        forecast.subject.validate()?;
        if forecast.subject != self.subject {
            return Err(CapabilityLedgerError::SubjectMismatch);
        }
        if forecast.sequence <= self.last_sequence {
            return Err(CapabilityLedgerError::SequenceNotMonotonic);
        }
        let claim = self
            .claims
            .get(&forecast.claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        if claim.claim_sha256 != forecast.claim_sha256 {
            return Err(CapabilityLedgerError::ClaimVersionMismatch);
        }
        if !matches!(
            claim.lifecycle,
            CapabilityLifecycle::Candidate
                | CapabilityLifecycle::Qualified
                | CapabilityLifecycle::Restricted
        ) {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        if self.forecasts.contains_key(&forecast.forecast_id) {
            return Err(CapabilityLedgerError::DuplicateForecast);
        }
        let sequence = forecast.sequence;
        self.forecasts.insert(
            forecast.forecast_id.clone(),
            ForecastRecord {
                forecast,
                receipt: None,
            },
        );
        self.last_sequence = sequence;
        Ok(())
    }

    /// Resolve a forecast only with later, subject-matched, split-matched evidence.
    ///
    /// This checks chronology and bindings, not receipt authenticity. Only admit receipts after
    /// independent verification by the external evidence pipeline.
    pub fn resolve_forecast(
        &mut self,
        receipt: CapabilityOutcomeReceipt,
    ) -> Result<(), CapabilityLedgerError> {
        receipt.subject.validate()?;
        if receipt.subject != self.subject {
            return Err(CapabilityLedgerError::SubjectMismatch);
        }
        if !non_empty(&receipt.evaluator_identity)
            || !non_empty(&receipt.evaluator_revision)
            || !is_hex_digest(&receipt.receipt_sha256, &[64])
            || receipt
                .actual_compute_units
                .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
        {
            return Err(CapabilityLedgerError::InvalidOutcomeReceipt);
        }
        if receipt.sequence <= self.last_sequence {
            return Err(CapabilityLedgerError::SequenceNotMonotonic);
        }
        {
            let record = self
                .forecasts
                .get(&receipt.forecast_id)
                .ok_or(CapabilityLedgerError::UnknownForecast)?;
            if record.receipt.is_some() {
                return Err(CapabilityLedgerError::AlreadyResolved);
            }
            if receipt.sequence <= record.forecast.sequence {
                return Err(CapabilityLedgerError::SequenceNotMonotonic);
            }
            if receipt.split != record.forecast.split {
                return Err(CapabilityLedgerError::SplitMismatch);
            }
            if receipt.subject != record.forecast.subject {
                return Err(CapabilityLedgerError::SubjectMismatch);
            }
        }
        if self.forecasts.values().any(|existing| {
            existing.receipt.as_ref().is_some_and(|prior| {
                prior.receipt_sha256 == receipt.receipt_sha256
            })
        }) {
            return Err(CapabilityLedgerError::ReceiptReplay);
        }
        let sequence = receipt.sequence;
        self.forecasts
            .get_mut(&receipt.forecast_id)
            .ok_or(CapabilityLedgerError::UnknownForecast)?
            .receipt = Some(receipt);
        self.last_sequence = sequence;
        Ok(())
    }

    /// Return descriptive metrics for all resolved evidence attached to a claim.
    ///
    /// Qualification decisions use a stricter, policy-filtered subset via qualification_metrics.
    pub fn metrics(&self, claim_id: &str) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        if !self.claims.contains_key(claim_id) {
            return Err(CapabilityLedgerError::UnknownClaim);
        }
        let records: Vec<&ForecastRecord> = self
            .forecasts
            .values()
            .filter(|record| record.forecast.claim_id == claim_id)
            .collect();
        Ok(calculate_metrics(&records))
    }

    /// Metrics computed only from resolved held-out/transfer receipts issued by the evaluator
    /// revision explicitly accepted by the supplied policy.
    pub fn qualification_metrics(
        &self,
        claim_id: &str,
        policy: &QualificationPolicy,
    ) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        policy.validate()?;
        if !self.claims.contains_key(claim_id) {
            return Err(CapabilityLedgerError::UnknownClaim);
        }
        let records: Vec<&ForecastRecord> = self
            .forecasts
            .values()
            .filter(|record| {
                record.forecast.claim_id == claim_id
                    && record
                        .receipt
                        .as_ref()
                        .is_some_and(|receipt| policy.accepts(receipt))
            })
            .collect();
        Ok(calculate_metrics(&records))
    }

    /// Promote a candidate only when a pinned evaluator policy and adequate out-of-training
    /// evidence satisfy both outcome-score and calibration-error gates.
    pub fn qualify_claim(
        &mut self,
        claim_id: &str,
        policy: &QualificationPolicy,
    ) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        policy.validate()?;
        let claim = self
            .claims
            .get(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        if claim.lifecycle != CapabilityLifecycle::Candidate {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        let metrics = self.qualification_metrics(claim_id, policy)?;
        if metrics.resolved_count < policy.minimum_resolved
            || metrics.held_out_samples < policy.minimum_held_out
            || metrics.qualification_samples < policy.minimum_resolved
        {
            return Err(CapabilityLedgerError::InsufficientEvidence);
        }
        if metrics
            .brier_score
            .map_or(true, |score| score > policy.maximum_brier_score)
            || metrics
                .expected_calibration_error
                .map_or(true, |ece| ece > policy.maximum_expected_calibration_error)
        {
            return Err(CapabilityLedgerError::QualificationRejected);
        }
        self.claims
            .get_mut(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?
            .lifecycle = CapabilityLifecycle::Qualified;
        Ok(metrics)
    }
}

fn calculate_metrics(records: &[&ForecastRecord]) -> CapabilityMetrics {
    let resolved: Vec<(&CapabilityForecast, &CapabilityOutcomeReceipt)> = records
        .iter()
        .filter_map(|record| {
            record
                .receipt
                .as_ref()
                .map(|receipt| (&record.forecast, receipt))
        })
        .collect();

    let resolved_count = resolved.len();
    let held_out_samples = resolved
        .iter()
        .filter(|(_, receipt)| receipt.split == CapabilitySplit::HeldOut)
        .count();
    let transfer_samples = resolved
        .iter()
        .filter(|(_, receipt)| receipt.split == CapabilitySplit::Transfer)
        .count();
    let qualification_samples = held_out_samples + transfer_samples;

    if resolved.is_empty() {
        return CapabilityMetrics {
            forecast_count: records.len(),
            resolved_count,
            qualification_samples,
            held_out_samples,
            transfer_samples,
            brier_score: None,
            expected_calibration_error: None,
            mean_predicted_success_probability: None,
            empirical_success_rate: None,
            mean_absolute_compute_error: None,
        };
    }

    let n = resolved_count as f64;
    let brier_score = resolved
        .iter()
        .map(|(forecast, receipt)| {
            let observed = if receipt.observed_success { 1.0 } else { 0.0 };
            (forecast.predicted_success_probability - observed).powi(2)
        })
        .sum::<f64>()
        / n;

    // Ten fixed probability bins. Empty bins contribute zero weight; ECE is unavailable rather
    // than zero when there are no resolved outcomes.
    let mut bins = [(0usize, 0.0f64, 0.0f64); 10];
    for (forecast, receipt) in &resolved {
        let p = forecast.predicted_success_probability;
        let idx = ((p * 10.0).floor() as usize).min(9);
        bins[idx].0 += 1;
        bins[idx].1 += p;
        bins[idx].2 += if receipt.observed_success { 1.0 } else { 0.0 };
    }
    let ece = bins
        .iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, p_sum, observed_sum)| {
            let count_f = *count as f64;
            (count_f / n) * ((p_sum / count_f) - (observed_sum / count_f)).abs()
        })
        .sum::<f64>();

    let compute_errors: Vec<f64> = resolved
        .iter()
        .filter_map(|(forecast, receipt)| {
            receipt
                .actual_compute_units
                .map(|actual| (forecast.expected_compute_units - actual).abs())
        })
        .collect();

    CapabilityMetrics {
        forecast_count: records.len(),
        resolved_count,
        qualification_samples,
        held_out_samples,
        transfer_samples,
        brier_score: Some(brier_score),
        expected_calibration_error: Some(ece),
        mean_predicted_success_probability: Some(
            resolved
                .iter()
                .map(|(forecast, _)| forecast.predicted_success_probability)
                .sum::<f64>()
                / n,
        ),
        empirical_success_rate: Some(
            resolved
                .iter()
                .filter(|(_, receipt)| receipt.observed_success)
                .count() as f64
                / n,
        ),
        mean_absolute_compute_error: if compute_errors.is_empty() {
            None
        } else {
            Some(compute_errors.iter().sum::<f64>() / compute_errors.len() as f64)
        },
    }
}

fn is_valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_hex_digest(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> SubjectIdentity {
        SubjectIdentity {
            source_commit: "a".repeat(40),
            source_tree: "b".repeat(40),
            configuration_sha256: "c".repeat(64),
            evaluation_profile_sha256: "d".repeat(64),
            model_data_sha256: Some("e".repeat(64)),
        }
    }

    fn setup_claim() -> CapabilityLedger {
        let mut ledger = CapabilityLedger::new(subject()).unwrap();
        ledger
            .add_claim(
                CapabilityClaim::new(
                    "rust-debugging",
                    "f".repeat(64),
                    "Diagnose bounded Rust compiler failures",
                    "Rust 1.96 workspace, declared dependencies and frozen test profile",
                )
                .unwrap(),
            )
            .unwrap();
        ledger
            .transition_claim("rust-debugging", CapabilityLifecycle::Discovered)
            .unwrap();
        ledger
            .transition_claim("rust-debugging", CapabilityLifecycle::Candidate)
            .unwrap();
        ledger
    }

    fn policy() -> QualificationPolicy {
        QualificationPolicy {
            minimum_resolved: 3,
            minimum_held_out: 3,
            maximum_brier_score: 0.05,
            maximum_expected_calibration_error: 0.15,
            trusted_evaluator_identity: "independent-qualification-runner".to_string(),
            trusted_evaluator_revisions: vec!["v1".to_string()],
        }
    }

    fn add_case(
        ledger: &mut CapabilityLedger,
        id: &str,
        forecast_seq: u64,
        split: CapabilitySplit,
        probability: f64,
        actual: bool,
    ) {
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: id.to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: forecast_seq,
                context_id: format!("frozen-context-{id}"),
                split,
                predicted_success_probability: probability,
                expected_compute_units: 10.0,
                predicted_failure_mode: Some("Missing compiler feature context".to_string()),
            })
            .unwrap();
        ledger
            .resolve_forecast(CapabilityOutcomeReceipt {
                forecast_id: id.to_string(),
                subject: subject(),
                sequence: forecast_seq + 1,
                observed_success: actual,
                split,
                evaluator_identity: "independent-qualification-runner".to_string(),
                evaluator_revision: "v1".to_string(),
                receipt_sha256: format!("{:064x}", forecast_seq + 1),
                actual_compute_units: Some(12.0),
            })
            .unwrap();
    }

    #[test]
    fn training_success_cannot_qualify_a_skill() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("train-{i}"),
                2 * i + 1,
                CapabilitySplit::Training,
                0.9,
                true,
            );
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging", &policy()),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
        assert_eq!(
            ledger.claim("rust-debugging").unwrap().lifecycle,
            CapabilityLifecycle::Candidate
        );
    }

    #[test]
    fn held_out_evidence_can_qualify_a_calibrated_candidate() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("heldout-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.9,
                true,
            );
        }
        let metrics = ledger.qualify_claim("rust-debugging", &policy()).unwrap();
        assert_eq!(metrics.resolved_count, 3);
        assert_eq!(metrics.held_out_samples, 3);
        assert!((metrics.brier_score.unwrap() - 0.01).abs() < 1e-9);
        assert_eq!(
            ledger.claim("rust-debugging").unwrap().lifecycle,
            CapabilityLifecycle::Qualified
        );
    }

    #[test]
    fn miscalibrated_held_out_evidence_is_rejected() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("wrong-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.99,
                false,
            );
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging", &policy()),
            Err(CapabilityLedgerError::QualificationRejected)
        );
    }

    #[test]
    fn outcome_must_follow_forecast_and_match_split() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "chronology".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 10,
                context_id: "context".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();

        let mut receipt = CapabilityOutcomeReceipt {
            forecast_id: "chronology".to_string(),
            subject: subject(),
            sequence: 10,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "2".repeat(64),
            actual_compute_units: None,
        };
        assert_eq!(
            ledger.resolve_forecast(receipt.clone()),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );
        receipt.sequence = 11;
        receipt.split = CapabilitySplit::Training;
        assert_eq!(
            ledger.resolve_forecast(receipt),
            Err(CapabilityLedgerError::SplitMismatch)
        );
    }

    #[test]
    fn subject_mismatch_is_rejected() {
        let mut ledger = setup_claim();
        let mut forecast = CapabilityForecast {
            forecast_id: "wrong-subject".to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "context".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.7,
            expected_compute_units: 1.0,
            predicted_failure_mode: None,
        };
        forecast.subject.source_commit = "9".repeat(40);
        assert_eq!(
            ledger.record_forecast(forecast),
            Err(CapabilityLedgerError::SubjectMismatch)
        );
    }

    #[test]
    fn adding_a_claim_cannot_import_qualified_status() {
        let mut ledger = CapabilityLedger::new(subject()).unwrap();
        ledger
            .add_claim(CapabilityClaim {
                claim_id: "forged".to_string(),
                claim_sha256: "f".repeat(64),
                description: "Claim with caller-supplied status".to_string(),
                scope: "Test only".to_string(),
                lifecycle: CapabilityLifecycle::Qualified,
            })
            .unwrap();
        assert_eq!(
            ledger.claim("forged").unwrap().lifecycle,
            CapabilityLifecycle::Proposed
        );
    }

    #[test]
    fn evaluator_must_match_pinned_qualification_policy() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("heldout-untrusted-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.9,
                true,
            );
        }
        let mut untrusted_policy = policy();
        untrusted_policy.trusted_evaluator_identity = "different-evaluator".to_string();
        assert_eq!(
            ledger.qualify_claim("rust-debugging", &untrusted_policy),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
    }

    #[test]
    fn trusted_evaluator_revision_is_pinned() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            let id = format!("wrong-revision-{i}");
            let seq = 2 * i + 1;
            ledger
                .record_forecast(CapabilityForecast {
                    forecast_id: id.clone(),
                    claim_id: "rust-debugging".to_string(),
                    claim_sha256: "f".repeat(64),
                    subject: subject(),
                    sequence: seq,
                    context_id: format!("held-out-{i}"),
                    split: CapabilitySplit::HeldOut,
                    predicted_success_probability: 0.9,
                    expected_compute_units: 10.0,
                    predicted_failure_mode: None,
                })
                .unwrap();
            ledger
                .resolve_forecast(CapabilityOutcomeReceipt {
                    forecast_id: id,
                    subject: subject(),
                    sequence: seq + 1,
                    observed_success: true,
                    split: CapabilitySplit::HeldOut,
                    evaluator_identity: "independent-qualification-runner".to_string(),
                    evaluator_revision: "unreviewed-revision".to_string(),
                    receipt_sha256: format!("{:064x}", seq + 1),
                    actual_compute_units: Some(10.0),
                })
                .unwrap();
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging", &policy()),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
    }

    #[test]
    fn invalid_probabilities_and_costs_fail_closed() {
        let mut ledger = setup_claim();
        for (id, probability, cost) in [
            ("nan-probability", f64::NAN, 1.0),
            ("negative-probability", -0.01, 1.0),
            ("over-one-probability", 1.01, 1.0),
            ("negative-cost", 0.5, -1.0),
            ("infinite-cost", 0.5, f64::INFINITY),
        ] {
            let forecast = CapabilityForecast {
                forecast_id: id.to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: format!("context-{id}"),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: probability,
                expected_compute_units: cost,
                predicted_failure_mode: None,
            };
            assert_eq!(
                ledger.record_forecast(forecast),
                Err(CapabilityLedgerError::InvalidForecast),
                "invalid input should be rejected: {id}"
            );
        }
    }

    #[test]
    fn unqualified_claim_cannot_record_forecasts() {
        let mut ledger = CapabilityLedger::new(subject()).unwrap();
        ledger
            .add_claim(
                CapabilityClaim::new(
                    "not-ready",
                    "f".repeat(64),
                    "Not ready for evaluation",
                    "Test scope",
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            ledger.record_forecast(CapabilityForecast {
                forecast_id: "premature".to_string(),
                claim_id: "not-ready".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: "context".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            }),
            Err(CapabilityLedgerError::InvalidLifecycleTransition)
        );
    }

    #[test]
    fn duplicate_claim_ids_are_rejected() {
        let mut ledger = setup_claim();
        let duplicate = CapabilityClaim::new(
            "rust-debugging",
            "1".repeat(64),
            "A conflicting claim version",
            "A different scope",
        )
        .unwrap();
        assert_eq!(
            ledger.add_claim(duplicate),
            Err(CapabilityLedgerError::DuplicateClaim)
        );
    }

    #[test]
    fn forecast_and_outcome_events_share_one_monotonic_sequence() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "first".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 5,
                context_id: "context-first".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.8,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();

        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "first".to_string(),
            subject: subject(),
            sequence: 5,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "3".repeat(64),
            actual_compute_units: None,
        };
        assert_eq!(
            ledger.resolve_forecast(receipt.clone()),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );

        let mut later_receipt = receipt;
        later_receipt.sequence = 6;
        ledger.resolve_forecast(later_receipt).unwrap();

        assert_eq!(
            ledger.record_forecast(CapabilityForecast {
                forecast_id: "out-of-order".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 5,
                context_id: "context-out-of-order".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.8,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            }),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );
    }

    #[test]
    fn receipt_root_cannot_be_replayed_for_a_second_forecast() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "first".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: "context-first".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();
        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "first".to_string(),
            subject: subject(),
            sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "2".repeat(64),
            actual_compute_units: None,
        };
        ledger.resolve_forecast(receipt.clone()).unwrap();

        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "second".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 3,
                context_id: "context-second".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();
        let replay = CapabilityOutcomeReceipt {
            forecast_id: "second".to_string(),
            sequence: 4,
            ..receipt
        };
        assert_eq!(
            ledger.resolve_forecast(replay),
            Err(CapabilityLedgerError::ReceiptReplay)
        );
    }

    #[test]
    fn unavailable_metrics_are_not_reported_as_perfect_calibration() {
        let ledger = setup_claim();
        let metrics = ledger.metrics("rust-debugging").unwrap();
        assert_eq!(metrics.resolved_count, 0);
        assert_eq!(metrics.brier_score, None);
        assert_eq!(metrics.expected_calibration_error, None);
    }
}
