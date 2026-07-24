// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Uncertainty envelopes and mandatory abstention.
//!
//! Inferred and simulated values must carry provenance, bounded uncertainty,
//! freshness, and a decision policy. Missing or weak uncertainty information is
//! a reason to abstain, not a reason to present a precise-looking value.

use crate::model_registry::ModelExecutionReceipt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EstimateSource {
    DirectObservation,
    UserReport,
    ClinicianReport,
    RuleBasedInference,
    ModelInference,
    Simulation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EstimateUse {
    SupportiveWording,
    StrategyRanking,
    InterventionAuthorization,
    CrisisDecision,
    ClinicalDecision,
    ResearchAnalysis,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EstimateEnvelope {
    pub estimate_id: String,
    /// Fixed-point value in domain-defined milli-units.
    pub value_milliunits: i64,
    pub lower_milliunits: i64,
    pub upper_milliunits: i64,
    /// Confidence in basis points. 10_000 = 100%.
    pub confidence_basis_points: u16,
    pub source: EstimateSource,
    pub generated_at_unix: u64,
    pub expires_at_unix: u64,
    pub model_receipt: Option<ModelExecutionReceipt>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbstentionReason {
    MissingIdentifier,
    InvalidInterval,
    InvalidConfidence,
    Expired,
    ClockBeforeGeneration,
    ProvenanceRequired,
    LimitationsMissing,
    ConfidenceTooLow,
    IntervalTooWide,
    UseProhibited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EstimateDecision {
    Authorized,
    Abstain(AbstentionReason),
}

impl EstimateDecision {
    pub const fn is_authorized(self) -> bool {
        matches!(self, Self::Authorized)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstentionPolicy {
    pub policy_version: String,
    pub minimum_confidence_basis_points: u16,
    pub maximum_interval_width_milliunits: u64,
    pub prohibit_inferred_crisis_decisions: bool,
    pub prohibit_simulated_clinical_decisions: bool,
}

impl EstimateEnvelope {
    pub fn interval_width(&self) -> Option<u64> {
        self.upper_milliunits
            .checked_sub(self.lower_milliunits)
            .and_then(|width| u64::try_from(width).ok())
    }

    pub fn validate(&self, now_unix: u64) -> Result<(), AbstentionReason> {
        if self.estimate_id.trim().is_empty() {
            return Err(AbstentionReason::MissingIdentifier);
        }
        if self.lower_milliunits > self.value_milliunits
            || self.value_milliunits > self.upper_milliunits
        {
            return Err(AbstentionReason::InvalidInterval);
        }
        if self.confidence_basis_points > 10_000 {
            return Err(AbstentionReason::InvalidConfidence);
        }
        if now_unix < self.generated_at_unix {
            return Err(AbstentionReason::ClockBeforeGeneration);
        }
        if now_unix >= self.expires_at_unix {
            return Err(AbstentionReason::Expired);
        }
        if matches!(
            self.source,
            EstimateSource::ModelInference | EstimateSource::Simulation
        ) && self.model_receipt.is_none()
        {
            return Err(AbstentionReason::ProvenanceRequired);
        }
        if matches!(
            self.source,
            EstimateSource::RuleBasedInference
                | EstimateSource::ModelInference
                | EstimateSource::Simulation
        ) && self.limitations.is_empty()
        {
            return Err(AbstentionReason::LimitationsMissing);
        }
        Ok(())
    }
}

impl AbstentionPolicy {
    pub fn authorize(
        &self,
        estimate: &EstimateEnvelope,
        intended_use: EstimateUse,
        now_unix: u64,
    ) -> EstimateDecision {
        if let Err(reason) = estimate.validate(now_unix) {
            return EstimateDecision::Abstain(reason);
        }
        if estimate.confidence_basis_points < self.minimum_confidence_basis_points {
            return EstimateDecision::Abstain(AbstentionReason::ConfidenceTooLow);
        }
        let Some(width) = estimate.interval_width() else {
            return EstimateDecision::Abstain(AbstentionReason::InvalidInterval);
        };
        if width > self.maximum_interval_width_milliunits {
            return EstimateDecision::Abstain(AbstentionReason::IntervalTooWide);
        }
        if self.prohibit_inferred_crisis_decisions
            && intended_use == EstimateUse::CrisisDecision
            && !matches!(
                estimate.source,
                EstimateSource::DirectObservation
                    | EstimateSource::UserReport
                    | EstimateSource::ClinicianReport
            )
        {
            return EstimateDecision::Abstain(AbstentionReason::UseProhibited);
        }
        if self.prohibit_simulated_clinical_decisions
            && intended_use == EstimateUse::ClinicalDecision
            && estimate.source == EstimateSource::Simulation
        {
            return EstimateDecision::Abstain(AbstentionReason::UseProhibited);
        }
        EstimateDecision::Authorized
    }

    pub fn fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-therapeutic-abstention-policy-v1\0");
        hasher.update(self.policy_version.as_bytes());
        hasher.update(&self.minimum_confidence_basis_points.to_le_bytes());
        hasher.update(&self.maximum_interval_width_milliunits.to_le_bytes());
        hasher.update(&[
            self.prohibit_inferred_crisis_decisions as u8,
            self.prohibit_simulated_clinical_decisions as u8,
        ]);
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AbstentionPolicy {
        AbstentionPolicy {
            policy_version: "uncertainty-1".to_string(),
            minimum_confidence_basis_points: 7_500,
            maximum_interval_width_milliunits: 400,
            prohibit_inferred_crisis_decisions: true,
            prohibit_simulated_clinical_decisions: true,
        }
    }

    fn reported_estimate() -> EstimateEnvelope {
        EstimateEnvelope {
            estimate_id: "reported-distress".to_string(),
            value_milliunits: 500,
            lower_milliunits: 400,
            upper_milliunits: 600,
            confidence_basis_points: 9_000,
            source: EstimateSource::UserReport,
            generated_at_unix: 100,
            expires_at_unix: 200,
            model_receipt: None,
            limitations: Vec::new(),
        }
    }

    #[test]
    fn sufficiently_bounded_user_report_is_authorized() {
        assert!(
            policy()
                .authorize(&reported_estimate(), EstimateUse::SupportiveWording, 120)
                .is_authorized()
        );
    }

    #[test]
    fn low_confidence_abstains() {
        let mut estimate = reported_estimate();
        estimate.confidence_basis_points = 7_499;
        assert_eq!(
            policy().authorize(&estimate, EstimateUse::StrategyRanking, 120),
            EstimateDecision::Abstain(AbstentionReason::ConfidenceTooLow)
        );
    }

    #[test]
    fn inferred_crisis_decision_is_prohibited() {
        let mut estimate = reported_estimate();
        estimate.source = EstimateSource::RuleBasedInference;
        estimate.limitations = vec!["Heuristic inference".to_string()];
        assert_eq!(
            policy().authorize(&estimate, EstimateUse::CrisisDecision, 120),
            EstimateDecision::Abstain(AbstentionReason::UseProhibited)
        );
    }

    #[test]
    fn model_inference_requires_execution_receipt() {
        let mut estimate = reported_estimate();
        estimate.source = EstimateSource::ModelInference;
        estimate.limitations = vec!["Model estimate".to_string()];
        assert_eq!(
            policy().authorize(&estimate, EstimateUse::SupportiveWording, 120),
            EstimateDecision::Abstain(AbstentionReason::ProvenanceRequired)
        );
    }
}
