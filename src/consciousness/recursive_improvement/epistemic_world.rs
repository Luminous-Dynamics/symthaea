// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic typing for replay, prediction, and imagination.
//!
//! These types make the distinction between observation, replay, prediction, and
//! counterfactual generation explicit. The serialized `EpistemicWorldRecord` shape
//! remains compatible with the frozen SYM-RSI grounded-dream receipts, but its
//! historical `empirically_validated` boolean is explicitly non-authorizing.
//!
//! A generated record can gain confidence-promotion authority only through the
//! non-deserializable `ValidatedEpistemicWorldRecord` capability, constructed from
//! a separate Recorded or ReplayDerived evidence record.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldEvidenceKind {
    /// Directly observed in an executed environment.
    Recorded,
    /// Derived only by selecting/traversing recorded transitions.
    ReplayDerived,
    /// Interpolation between supported observations.
    Interpolated,
    /// Output of a learned predictive model.
    ModelPredicted,
    /// Explicit alternative to what actually occurred.
    Counterfactual,
    /// Prediction outside demonstrated support.
    Extrapolated,
    /// Deliberately generated stress/adversarial world.
    AdversarialGenerated,
}

impl WorldEvidenceKind {
    /// Whether this class is already empirical outcome evidence without any
    /// separate validation capability.
    pub fn is_empirical(self) -> bool {
        matches!(self, Self::Recorded | Self::ReplayDerived)
    }
}

/// Provenance of the separate empirical record that validated a generated record.
///
/// Intentionally not serializable/deserializable. This is runtime authority, not a
/// boolean that a mutable receipt can mint for itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpiricalValidationEvidence {
    kind: WorldEvidenceKind,
    provenance_digest: String,
}

impl EmpiricalValidationEvidence {
    fn try_from_record(record: &EpistemicWorldRecord) -> Result<Self, EpistemicWorldError> {
        record.validate()?;
        if !record.kind.is_empirical() {
            return Err(EpistemicWorldError::ValidationSourceNotEmpirical);
        }
        Ok(Self {
            kind: record.kind,
            provenance_digest: record.provenance_digest.clone(),
        })
    }

    pub fn kind(&self) -> WorldEvidenceKind {
        self.kind
    }

    pub fn provenance_digest(&self) -> &str {
        &self.provenance_digest
    }

    fn validate(&self) -> Result<(), EpistemicWorldError> {
        if !self.kind.is_empirical() {
            return Err(EpistemicWorldError::ValidationSourceNotEmpirical);
        }
        if self.provenance_digest.trim().is_empty() {
            return Err(EpistemicWorldError::EmptyProvenanceDigest);
        }
        Ok(())
    }
}

/// Serialized epistemic metadata carried by predictions and replay products.
///
/// The public fields are retained for compatibility with the preregistered v7
/// grounded-dream receipt construction. `empirically_validated` is legacy metadata
/// only: `may_promote_confidence()` deliberately ignores it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicWorldRecord {
    pub kind: WorldEvidenceKind,
    pub provenance_digest: String,
    pub model_version: Option<String>,
    pub confidence: Option<f64>,
    /// Distance from demonstrated support, if defined by the producer.
    pub support_distance: Option<f64>,
    pub causal_assumptions: Vec<String>,
    /// Legacy serialized compatibility field. NON-AUTHORIZING.
    ///
    /// Setting this value to true does not permit confidence promotion. Generated
    /// evidence requires `validated_by()` and the resulting runtime capability.
    pub empirically_validated: bool,
}

impl EpistemicWorldRecord {
    pub fn new(
        kind: WorldEvidenceKind,
        provenance_digest: impl Into<String>,
    ) -> Result<Self, EpistemicWorldError> {
        let record = Self {
            kind,
            provenance_digest: provenance_digest.into(),
            model_version: None,
            confidence: None,
            support_distance: None,
            causal_assumptions: Vec::new(),
            empirically_validated: false,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn kind(&self) -> WorldEvidenceKind {
        self.kind
    }

    pub fn provenance_digest(&self) -> &str {
        &self.provenance_digest
    }

    pub fn model_version(&self) -> Option<&str> {
        self.model_version.as_deref()
    }

    pub fn confidence(&self) -> Option<f64> {
        self.confidence
    }

    pub fn support_distance(&self) -> Option<f64> {
        self.support_distance
    }

    pub fn causal_assumptions(&self) -> &[String] {
        &self.causal_assumptions
    }

    pub fn with_model_version(mut self, model_version: impl Into<String>) -> Self {
        let model_version = model_version.into();
        self.model_version = (!model_version.trim().is_empty()).then_some(model_version);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Result<Self, EpistemicWorldError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(EpistemicWorldError::InvalidConfidence);
        }
        self.confidence = Some(confidence);
        Ok(self)
    }

    pub fn with_support_distance(
        mut self,
        support_distance: f64,
    ) -> Result<Self, EpistemicWorldError> {
        if !support_distance.is_finite() || support_distance < 0.0 {
            return Err(EpistemicWorldError::InvalidSupportDistance);
        }
        self.support_distance = Some(support_distance);
        Ok(self)
    }

    pub fn with_causal_assumption(mut self, assumption: impl Into<String>) -> Self {
        let assumption = assumption.into();
        if !assumption.trim().is_empty() {
            self.causal_assumptions.push(assumption);
        }
        self
    }

    pub fn validate(&self) -> Result<(), EpistemicWorldError> {
        if self.provenance_digest.trim().is_empty() {
            return Err(EpistemicWorldError::EmptyProvenanceDigest);
        }
        if self
            .confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(EpistemicWorldError::InvalidConfidence);
        }
        if self
            .support_distance
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(EpistemicWorldError::InvalidSupportDistance);
        }
        Ok(())
    }

    /// Raw serialized records may promote confidence only when already empirical.
    /// The legacy boolean is intentionally ignored.
    pub fn may_promote_confidence(&self) -> bool {
        self.validate().is_ok() && self.kind.is_empirical()
    }

    /// Bind this record to a separate empirical validation record.
    ///
    /// The returned capability is deliberately not deserializable. A mutable dream
    /// receipt cannot authorize itself by flipping `empirically_validated`.
    pub fn validated_by(
        &self,
        empirical_record: &EpistemicWorldRecord,
    ) -> Result<ValidatedEpistemicWorldRecord, EpistemicWorldError> {
        self.validate()?;
        let validation = EmpiricalValidationEvidence::try_from_record(empirical_record)?;
        Ok(ValidatedEpistemicWorldRecord {
            record: self.clone(),
            validation,
        })
    }
}

/// Runtime capability proving that a source record has separate empirical support.
///
/// This type does not derive Serialize/Deserialize. Recovery after restart must
/// revalidate the source record against independently retained empirical evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedEpistemicWorldRecord {
    record: EpistemicWorldRecord,
    validation: EmpiricalValidationEvidence,
}

impl ValidatedEpistemicWorldRecord {
    pub fn record(&self) -> &EpistemicWorldRecord {
        &self.record
    }

    pub fn validation(&self) -> &EmpiricalValidationEvidence {
        &self.validation
    }

    pub fn may_promote_confidence(&self) -> bool {
        self.record.validate().is_ok() && self.validation.validate().is_ok()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicWorldError {
    EmptyProvenanceDigest,
    InvalidConfidence,
    InvalidSupportDistance,
    ValidationSourceNotEmpirical,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded(digest: &str) -> EpistemicWorldRecord {
        EpistemicWorldRecord::new(WorldEvidenceKind::Recorded, digest).unwrap()
    }

    #[test]
    fn counterfactual_is_not_empirical_by_default() {
        let record = EpistemicWorldRecord::new(
            WorldEvidenceKind::Counterfactual,
            "blake3:dream-source",
        )
        .unwrap()
        .with_model_version("wm-v1")
        .with_confidence(0.9)
        .unwrap()
        .with_support_distance(0.1)
        .unwrap()
        .with_causal_assumption("do(action=x)");
        assert!(!record.may_promote_confidence());
    }

    #[test]
    fn legacy_boolean_cannot_self_authorize_generated_evidence() {
        let mut record = EpistemicWorldRecord::new(
            WorldEvidenceKind::Counterfactual,
            "blake3:dream-source",
        )
        .unwrap();
        record.empirically_validated = true;
        assert!(!record.may_promote_confidence());
    }

    #[test]
    fn recorded_and_replay_derived_are_empirical() {
        let recorded = recorded("blake3:observation");
        let replay = EpistemicWorldRecord::new(
            WorldEvidenceKind::ReplayDerived,
            "blake3:exact-replay",
        )
        .unwrap();
        assert!(recorded.may_promote_confidence());
        assert!(replay.may_promote_confidence());
    }

    #[test]
    fn generated_record_requires_separate_empirical_validation_capability() {
        let empirical = recorded("blake3:independent-observation");
        let generated = EpistemicWorldRecord::new(
            WorldEvidenceKind::Counterfactual,
            "blake3:dream-source",
        )
        .unwrap();
        let validated = generated.validated_by(&empirical).unwrap();

        assert!(!generated.may_promote_confidence());
        assert!(validated.may_promote_confidence());
        assert_eq!(validated.record().kind(), WorldEvidenceKind::Counterfactual);
        assert_eq!(validated.validation().kind(), WorldEvidenceKind::Recorded);
        assert_eq!(
            validated.validation().provenance_digest(),
            "blake3:independent-observation"
        );
    }

    #[test]
    fn counterfactual_cannot_validate_counterfactual() {
        let source = EpistemicWorldRecord::new(
            WorldEvidenceKind::Counterfactual,
            "blake3:other-dream",
        )
        .unwrap();
        let generated = EpistemicWorldRecord::new(
            WorldEvidenceKind::ModelPredicted,
            "blake3:model-output",
        )
        .unwrap();
        assert_eq!(
            generated.validated_by(&source),
            Err(EpistemicWorldError::ValidationSourceNotEmpirical)
        );
    }

    #[test]
    fn invalid_numeric_metadata_is_rejected() {
        let record = recorded("blake3:observation");
        assert_eq!(
            record.clone().with_confidence(f64::NAN),
            Err(EpistemicWorldError::InvalidConfidence)
        );
        assert_eq!(
            record.with_support_distance(-0.1),
            Err(EpistemicWorldError::InvalidSupportDistance)
        );
    }
}
