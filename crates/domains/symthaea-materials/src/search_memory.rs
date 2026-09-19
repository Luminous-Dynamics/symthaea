// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only search memory for materials discovery.
//!
//! Search memory records what Symthaea tried, why it tried it, what it cost,
//! and how the attempt ended. These records can change future acquisition
//! priorities, but they are not scientific evidence and cannot advance the
//! MAT-001 evidence ladder.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const SHA256_HEX_LEN: usize = 64;

/// Broad outcome class for one attempted material evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchAttemptOutcome {
    /// Exact attempt had already been evaluated.
    Duplicate,
    /// Proposed material subject or architecture was invalid.
    InvalidSubject,
    /// A MAT-009 hard constraint was violated before expensive evaluation.
    HardConstraintViolation,
    /// Required external data/provider result was unavailable.
    ProviderUnavailable,
    /// Surrogate/model declared the subject outside its applicability domain.
    ModelOutOfDomain,
    /// Numerical calculation failed to converge.
    CalculationNonConvergence,
    /// Candidate was evaluated successfully but performed poorly for the campaign.
    BelowTarget,
    /// Thermodynamic evidence was unfavorable.
    ThermodynamicallyUnfavorable,
    /// Dynamic/phonon evidence was unfavorable.
    DynamicallyUnstable,
    /// Physical synthesis was attempted and failed.
    SynthesisFailed,
    /// Material was produced but the intended phase was not established.
    TargetPhaseNotEstablished,
    /// Characterization completed with a scientifically meaningful null result.
    CharacterizationNull,
    /// Evidence sources contradicted one another beyond declared uncertainty.
    ContradictoryEvidence,
    /// Attempt could not finish within campaign resource ceilings.
    BudgetExhausted,
    /// Strategy was explicitly abandoned for a documented reason.
    Abandoned,
    /// Evaluation completed and produced usable downstream evidence.
    Completed,
}

/// Resources consumed by one attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchAttemptCost {
    /// Compute core-hours consumed.
    pub compute_core_hours: f64,
    /// Wall-clock execution/experiment time in hours.
    pub wall_time_hours: f64,
    /// Physical material/feedstock consumed in kg.
    pub material_mass_kg: f64,
    /// Direct monetary cost, when known.
    pub direct_cost: Option<f64>,
    /// Currency for direct cost.
    pub currency: Option<String>,
}

impl SearchAttemptCost {
    fn validate(&self) -> Result<(), SearchMemoryError> {
        nonnegative("compute_core_hours", self.compute_core_hours)?;
        nonnegative("wall_time_hours", self.wall_time_hours)?;
        nonnegative("material_mass_kg", self.material_mass_kg)?;
        match (self.direct_cost, self.currency.as_deref()) {
            (Some(cost), Some(currency)) => {
                nonnegative("direct_cost", cost)?;
                nonempty("currency", currency)
            }
            (None, None) => Ok(()),
            _ => Err(SearchMemoryError::IncompleteMonetaryCost),
        }
    }
}

/// Exact evaluator/action identity used for an attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchEvaluatorRef {
    /// Stable evaluator/action identifier.
    pub evaluator_id: String,
    /// SHA-256 of exact model/code/protocol artifact.
    pub artifact_sha256: String,
    /// Named fidelity/method class, for example `ml-surrogate`, `dft-pbe`, or `xrd`.
    pub fidelity_id: String,
}

/// Optional neighborhood fingerprint for near-duplicate avoidance.
///
/// A neighborhood match is search telemetry only. It never transfers evidence
/// between distinct MAT-007 material subjects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchNeighborhoodFingerprint {
    /// Fingerprint method/version.
    pub method_id: String,
    /// Deterministic digest or canonical neighborhood key.
    pub fingerprint: String,
}

/// One append-only record of a material search/evaluation attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialSearchAttempt {
    /// Deterministic ID for this exact attempt definition.
    pub attempt_id: String,
    /// Exact MAT-009 campaign identity.
    pub campaign_identity: String,
    /// Exact MAT-007 subject identity, or canonical proposed-subject identity.
    pub subject_identity: String,
    /// Optional search-neighborhood fingerprint.
    pub neighborhood: Option<SearchNeighborhoodFingerprint>,
    /// Evaluator/action that was attempted.
    pub evaluator: SearchEvaluatorRef,
    /// Stable proposal/acquisition strategy identifier.
    pub proposal_strategy_id: String,
    /// Non-authoritative acquisition priority at proposal time, when available.
    pub acquisition_priority: Option<f64>,
    /// Human-readable rationale. Excluded from exact attempt identity.
    pub rationale: String,
    /// Outcome class.
    pub outcome: SearchAttemptOutcome,
    /// Stable reason code suitable for aggregation.
    pub reason_code: String,
    /// Human-readable outcome detail.
    pub detail: String,
    /// Resource cost actually consumed.
    pub cost: SearchAttemptCost,
    /// Evidence/artifact record IDs created by the attempt, if any.
    pub produced_evidence_ids: Vec<String>,
    /// Exact result/log artifact bindings.
    pub result_artifact_sha256: Vec<String>,
}

impl MaterialSearchAttempt {
    /// Construct and validate an attempt, deriving its exact deterministic ID.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        campaign_identity: String,
        subject_identity: String,
        neighborhood: Option<SearchNeighborhoodFingerprint>,
        evaluator: SearchEvaluatorRef,
        proposal_strategy_id: String,
        acquisition_priority: Option<f64>,
        rationale: String,
        outcome: SearchAttemptOutcome,
        reason_code: String,
        detail: String,
        cost: SearchAttemptCost,
        produced_evidence_ids: Vec<String>,
        result_artifact_sha256: Vec<String>,
    ) -> Result<Self, SearchMemoryError> {
        let mut attempt = Self {
            attempt_id: String::new(),
            campaign_identity,
            subject_identity,
            neighborhood,
            evaluator,
            proposal_strategy_id,
            acquisition_priority,
            rationale,
            outcome,
            reason_code,
            detail,
            cost,
            produced_evidence_ids,
            result_artifact_sha256,
        };
        attempt.validate_without_id()?;
        attempt.attempt_id = attempt.exact_attempt_identity();
        Ok(attempt)
    }

    /// Validate this stored record including deterministic identity consistency.
    pub fn validate(&self) -> Result<(), SearchMemoryError> {
        self.validate_without_id()?;
        if self.attempt_id != self.exact_attempt_identity() {
            return Err(SearchMemoryError::AttemptIdentityMismatch);
        }
        Ok(())
    }

    /// Canonical exact-attempt identity.
    ///
    /// Outcome, rationale, costs, and produced evidence are deliberately excluded:
    /// they are consequences of executing the attempt, not part of the action that
    /// was proposed. Re-running the same exact action therefore derives the same ID.
    pub fn exact_attempt_identity(&self) -> String {
        format!(
            "materials-attempt:v1|campaign={}|subject={}|evaluator={}|artifact={}|fidelity={}|strategy={}",
            token(&self.campaign_identity),
            token(&self.subject_identity),
            token(&self.evaluator.evaluator_id),
            self.evaluator.artifact_sha256.to_ascii_lowercase(),
            token(&self.evaluator.fidelity_id),
            token(&self.proposal_strategy_id)
        )
    }

    fn validate_without_id(&self) -> Result<(), SearchMemoryError> {
        nonempty("campaign_identity", &self.campaign_identity)?;
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("evaluator_id", &self.evaluator.evaluator_id)?;
        sha256(&self.evaluator.artifact_sha256)?;
        nonempty("fidelity_id", &self.evaluator.fidelity_id)?;
        nonempty("proposal_strategy_id", &self.proposal_strategy_id)?;
        nonempty("rationale", &self.rationale)?;
        nonempty("reason_code", &self.reason_code)?;
        nonempty("detail", &self.detail)?;
        if let Some(priority) = self.acquisition_priority {
            finite("acquisition_priority", priority)?;
        }
        if let Some(neighborhood) = &self.neighborhood {
            nonempty("neighborhood method_id", &neighborhood.method_id)?;
            nonempty("neighborhood fingerprint", &neighborhood.fingerprint)?;
        }
        self.cost.validate()?;
        unique_nonempty(&self.produced_evidence_ids, "produced_evidence_id")?;
        for digest in &self.result_artifact_sha256 {
            sha256(digest)?;
        }
        let unique_digests: HashSet<_> = self.result_artifact_sha256.iter().collect();
        if unique_digests.len() != self.result_artifact_sha256.len() {
            return Err(SearchMemoryError::DuplicateResultArtifact);
        }
        Ok(())
    }
}

/// Append-only materials search ledger.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterialsSearchMemory {
    attempts: Vec<MaterialSearchAttempt>,
}

impl MaterialsSearchMemory {
    /// Create empty search memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Immutable view of all recorded attempts in append order.
    pub fn attempts(&self) -> &[MaterialSearchAttempt] {
        &self.attempts
    }

    /// Append an attempt. Exact repeated actions are rejected so callers can avoid
    /// expensive duplicate work before execution.
    pub fn append(&mut self, attempt: MaterialSearchAttempt) -> Result<(), SearchMemoryError> {
        attempt.validate()?;
        if self
            .attempts
            .iter()
            .any(|existing| existing.attempt_id == attempt.attempt_id)
        {
            return Err(SearchMemoryError::DuplicateExactAttempt(attempt.attempt_id));
        }
        self.attempts.push(attempt);
        Ok(())
    }

    /// Whether this exact campaign/subject/evaluator/strategy action is already recorded.
    pub fn contains_exact_attempt(&self, attempt_id: &str) -> bool {
        self.attempts
            .iter()
            .any(|attempt| attempt.attempt_id == attempt_id)
    }

    /// Find all attempts sharing a neighborhood fingerprint under the same method.
    /// This is advisory search memory and transfers no scientific evidence.
    pub fn neighborhood_attempts(
        &self,
        neighborhood: &SearchNeighborhoodFingerprint,
    ) -> Vec<&MaterialSearchAttempt> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.neighborhood.as_ref() == Some(neighborhood))
            .collect()
    }

    /// Aggregate consumed resources for one exact campaign identity.
    pub fn campaign_cost(&self, campaign_identity: &str) -> SearchAttemptCost {
        let mut total = SearchAttemptCost::default();
        let mut currency: Option<String> = None;
        let mut money_known = true;
        let mut monetary_total = 0.0;
        for attempt in self
            .attempts
            .iter()
            .filter(|attempt| attempt.campaign_identity == campaign_identity)
        {
            total.compute_core_hours += attempt.cost.compute_core_hours;
            total.wall_time_hours += attempt.cost.wall_time_hours;
            total.material_mass_kg += attempt.cost.material_mass_kg;
            match (attempt.cost.direct_cost, attempt.cost.currency.as_ref()) {
                (Some(value), Some(code)) => {
                    match &currency {
                        None => currency = Some(code.clone()),
                        Some(existing) if existing == code => {}
                        Some(_) => money_known = false,
                    }
                    monetary_total += value;
                }
                (None, None) => money_known = false,
                _ => money_known = false,
            }
        }
        if money_known && currency.is_some() {
            total.direct_cost = Some(monetary_total);
            total.currency = currency;
        }
        total
    }
}

fn unique_nonempty(values: &[String], field: &'static str) -> Result<(), SearchMemoryError> {
    let mut seen = HashSet::new();
    for value in values {
        nonempty(field, value)?;
        if !seen.insert(value.as_str()) {
            return Err(SearchMemoryError::DuplicateEvidenceId(value.clone()));
        }
    }
    Ok(())
}

fn nonempty(field: &'static str, value: &str) -> Result<(), SearchMemoryError> {
    if value.trim().is_empty() {
        Err(SearchMemoryError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), SearchMemoryError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SearchMemoryError::NonFiniteValue { field, value })
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), SearchMemoryError> {
    finite(field, value)?;
    if value < 0.0 {
        Err(SearchMemoryError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn sha256(value: &str) -> Result<(), SearchMemoryError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(SearchMemoryError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Search-memory validation or append failure.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchMemoryError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN or infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// A resource value was negative.
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Direct cost and currency were not supplied together.
    IncompleteMonetaryCost,
    /// SHA-256 binding was malformed.
    InvalidSha256,
    /// Stored attempt ID does not match the derived exact action identity.
    AttemptIdentityMismatch,
    /// Exact action has already been recorded.
    DuplicateExactAttempt(String),
    /// Evidence ID was repeated within one attempt.
    DuplicateEvidenceId(String),
    /// Result artifact digest was repeated within one attempt.
    DuplicateResultArtifact,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn attempt(subject: &str, outcome: SearchAttemptOutcome) -> MaterialSearchAttempt {
        MaterialSearchAttempt::new(
            "campaign-v1".to_string(),
            subject.to_string(),
            Some(SearchNeighborhoodFingerprint {
                method_id: "composition-l1-v1".to_string(),
                fingerprint: "ti-zr-nb-ta-er-cell-17".to_string(),
            }),
            SearchEvaluatorRef {
                evaluator_id: "dft-relax-v1".to_string(),
                artifact_sha256: A64.to_string(),
                fidelity_id: "dft-pbe".to_string(),
            },
            "pareto-information-gain-v1".to_string(),
            Some(0.73),
            "resolve phase uncertainty".to_string(),
            outcome,
            "fixture".to_string(),
            "fixture outcome".to_string(),
            SearchAttemptCost {
                compute_core_hours: 12.0,
                wall_time_hours: 2.0,
                material_mass_kg: 0.0,
                direct_cost: Some(3.0),
                currency: Some("USD".to_string()),
            },
            vec!["evidence-1".to_string()],
            vec![B64.to_string()],
        )
        .unwrap()
    }

    #[test]
    fn outcome_does_not_change_exact_action_identity() {
        let success = attempt("subject-a", SearchAttemptOutcome::Completed);
        let failure = attempt("subject-a", SearchAttemptOutcome::CalculationNonConvergence);
        assert_eq!(success.attempt_id, failure.attempt_id);
    }

    #[test]
    fn subject_change_changes_attempt_identity() {
        assert_ne!(
            attempt("subject-a", SearchAttemptOutcome::Completed).attempt_id,
            attempt("subject-b", SearchAttemptOutcome::Completed).attempt_id
        );
    }

    #[test]
    fn exact_duplicate_is_rejected_before_repeat_work() {
        let mut memory = MaterialsSearchMemory::new();
        let first = attempt("subject-a", SearchAttemptOutcome::Completed);
        let duplicate = attempt("subject-a", SearchAttemptOutcome::CalculationNonConvergence);
        memory.append(first).unwrap();
        assert!(matches!(
            memory.append(duplicate),
            Err(SearchMemoryError::DuplicateExactAttempt(_))
        ));
    }

    #[test]
    fn neighborhood_match_does_not_require_same_subject() {
        let mut memory = MaterialsSearchMemory::new();
        memory
            .append(attempt("subject-a", SearchAttemptOutcome::BelowTarget))
            .unwrap();
        let probe = attempt("subject-b", SearchAttemptOutcome::BelowTarget);
        let matches = memory.neighborhood_attempts(probe.neighborhood.as_ref().unwrap());
        assert_eq!(matches.len(), 1);
        assert_ne!(matches[0].subject_identity, probe.subject_identity);
    }

    #[test]
    fn failed_synthesis_is_preserved_as_failure_not_authority() {
        let mut memory = MaterialsSearchMemory::new();
        memory
            .append(attempt("subject-a", SearchAttemptOutcome::SynthesisFailed))
            .unwrap();
        assert_eq!(memory.attempts()[0].outcome, SearchAttemptOutcome::SynthesisFailed);
        // Search-memory types intentionally contain no MAT-001 evidence-stage field.
    }

    #[test]
    fn campaign_cost_accumulates_same_currency() {
        let mut memory = MaterialsSearchMemory::new();
        memory
            .append(attempt("subject-a", SearchAttemptOutcome::BelowTarget))
            .unwrap();
        memory
            .append(attempt("subject-b", SearchAttemptOutcome::Completed))
            .unwrap();
        let total = memory.campaign_cost("campaign-v1");
        assert_eq!(total.compute_core_hours, 24.0);
        assert_eq!(total.wall_time_hours, 4.0);
        assert_eq!(total.direct_cost, Some(6.0));
        assert_eq!(total.currency.as_deref(), Some("USD"));
    }

    #[test]
    fn tampered_attempt_identity_is_rejected() {
        let mut invalid = attempt("subject-a", SearchAttemptOutcome::Completed);
        invalid.attempt_id = "wrong".to_string();
        assert_eq!(invalid.validate(), Err(SearchMemoryError::AttemptIdentityMismatch));
    }

    #[test]
    fn acquisition_priority_can_be_negative_without_becoming_resource_cost() {
        let mut value = attempt("subject-a", SearchAttemptOutcome::Completed);
        value.acquisition_priority = Some(-1.0);
        value.validate().unwrap();
    }
}
