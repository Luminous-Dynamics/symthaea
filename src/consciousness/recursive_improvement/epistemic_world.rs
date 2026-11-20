// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic typing for observed, replayed, predicted and counterfactual worlds.
//!
//! Serialized records are metadata, not authority. In particular,
//! `empirically_validated` is retained for compatibility and is deliberately
//! ignored by every promotion check.
//!
//! A generated record can gain a runtime-only promotion capability only when a
//! separate empirical record binds the same claim and scope from a different
//! declared source family and source event. Canonical claim, scope and source
//! identities should be supplied by the owners in #8265, #9016 and #9075.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldEvidenceKind {
    Recorded,
    ReplayDerived,
    Interpolated,
    ModelPredicted,
    Counterfactual,
    Extrapolated,
    AdversarialGenerated,
}

impl WorldEvidenceKind {
    pub fn is_empirical(self) -> bool {
        matches!(self, Self::Recorded | Self::ReplayDerived)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicWorldRecord {
    pub kind: WorldEvidenceKind,
    pub provenance_digest: String,
    pub model_version: Option<String>,
    pub confidence: Option<f64>,
    pub support_distance: Option<f64>,
    pub causal_assumptions: Vec<String>,
    /// Legacy serialized compatibility field. It never grants authority.
    pub empirically_validated: bool,
    /// Canonical claim identity from the claim-identity owner (#8265).
    #[serde(default)]
    pub claim_identity: String,
    /// Canonical use-scope identity from the authority/scope owner (#9016).
    #[serde(default)]
    pub scope_identity: String,
    /// Declared independent source-family identity from evidence provenance (#9075).
    #[serde(default)]
    pub source_family_identity: String,
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
            claim_identity: String::new(),
            scope_identity: String::new(),
            source_family_identity: String::new(),
        };
        record.validate()?;
        Ok(record)
    }

    pub fn with_binding(
        mut self,
        claim_identity: impl Into<String>,
        scope_identity: impl Into<String>,
        source_family_identity: impl Into<String>,
    ) -> Result<Self, EpistemicWorldError> {
        self.claim_identity = claim_identity.into();
        self.scope_identity = scope_identity.into();
        self.source_family_identity = source_family_identity.into();
        self.validate_binding()?;
        Ok(self)
    }

    pub fn with_model_version(mut self, model_version: impl Into<String>) -> Self {
        let value = model_version.into();
        self.model_version = (!value.trim().is_empty()).then_some(value);
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
        let value = assumption.into();
        if !value.trim().is_empty() {
            self.causal_assumptions.push(value);
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

    fn validate_binding(&self) -> Result<(), EpistemicWorldError> {
        if self.claim_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingClaimIdentity);
        }
        if self.scope_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingScopeIdentity);
        }
        if self.source_family_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingSourceFamilyIdentity);
        }
        Ok(())
    }

    /// A raw record is authoritative only when its own class is empirical and
    /// its canonical claim/scope/source bindings are present and valid.
    pub fn may_promote_confidence(&self) -> bool {
        self.validate().is_ok()
            && self.validate_binding().is_ok()
            && self.kind.is_empirical()
    }

    /// Bind a generated record to a distinct empirical event about the same claim
    /// and scope, from a different declared source family.
    pub fn validated_by(
        &self,
        empirical_record: &EpistemicWorldRecord,
    ) -> Result<ValidatedEpistemicWorldRecord, EpistemicWorldError> {
        self.validate()?;
        self.validate_binding()?;
        if self.kind.is_empirical() {
            return Err(EpistemicWorldError::TargetAlreadyEmpirical);
        }
        if !empirical_record.may_promote_confidence() {
            return Err(EpistemicWorldError::ValidationSourceNotEmpirical);
        }
        if self.claim_identity != empirical_record.claim_identity {
            return Err(EpistemicWorldError::ClaimIdentityMismatch);
        }
        if self.scope_identity != empirical_record.scope_identity {
            return Err(EpistemicWorldError::ScopeIdentityMismatch);
        }
        if self.source_family_identity == empirical_record.source_family_identity {
            return Err(EpistemicWorldError::SourceFamilyNotIndependent);
        }
        if self.provenance_digest == empirical_record.provenance_digest {
            return Err(EpistemicWorldError::SameProvenanceDigest);
        }

        let validation = EmpiricalValidationEvidence {
            kind: empirical_record.kind,
            provenance_digest: empirical_record.provenance_digest.clone(),
            claim_identity: empirical_record.claim_identity.clone(),
            scope_identity: empirical_record.scope_identity.clone(),
            source_family_identity: empirical_record.source_family_identity.clone(),
        };
        validation.validate()?;

        Ok(ValidatedEpistemicWorldRecord {
            record: self.clone(),
            validation,
        })
    }
}

/// Runtime-only proof that a generated record was checked against separate
/// empirical evidence with matching claim and scope identity.
/// This type intentionally does not implement Serialize or Deserialize.
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
        self.record.validate().is_ok()
            && self.record.validate_binding().is_ok()
            && self.validation.validate().is_ok()
            && self.record.claim_identity == self.validation.claim_identity
            && self.record.scope_identity == self.validation.scope_identity
            && self.record.source_family_identity != self.validation.source_family_identity
            && self.record.provenance_digest != self.validation.provenance_digest
    }
}

/// Evidence that carries authority; it is retained inside a non-serializable
/// capability rather than accepted from a caller-controlled boolean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpiricalValidationEvidence {
    kind: WorldEvidenceKind,
    provenance_digest: String,
    claim_identity: String,
    scope_identity: String,
    source_family_identity: String,
}

impl EmpiricalValidationEvidence {
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
        if self.claim_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingClaimIdentity);
        }
        if self.scope_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingScopeIdentity);
        }
        if self.source_family_identity.trim().is_empty() {
            return Err(EpistemicWorldError::MissingSourceFamilyIdentity);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicWorldError {
    EmptyProvenanceDigest,
    InvalidConfidence,
    InvalidSupportDistance,
    MissingClaimIdentity,
    MissingScopeIdentity,
    MissingSourceFamilyIdentity,
    ValidationSourceNotEmpirical,
    ClaimIdentityMismatch,
    ScopeIdentityMismatch,
    SourceFamilyNotIndependent,
    SameProvenanceDigest,
    TargetAlreadyEmpirical,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound_record(
        kind: WorldEvidenceKind,
        provenance: &str,
        claim: &str,
        scope: &str,
        family: &str,
    ) -> EpistemicWorldRecord {
        EpistemicWorldRecord::new(kind, provenance)
            .unwrap()
            .with_binding(claim, scope, family)
            .unwrap()
    }

    #[test]
    fn recorded_and_replay_records_are_directly_empirical_when_bound() {
        let recorded = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:alpha",
            "scope:site-a",
            "family:sensor-a",
        );
        let replay = bound_record(
            WorldEvidenceKind::ReplayDerived,
            "event:replay",
            "claim:alpha",
            "scope:site-a",
            "family:replay-a",
        );
        assert!(recorded.may_promote_confidence());
        assert!(replay.may_promote_confidence());
    }

    #[test]
    fn legacy_boolean_cannot_authorize_generated_evidence() {
        let mut generated = bound_record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:dream-model",
        );
        generated.empirically_validated = true;
        assert!(!generated.may_promote_confidence());
    }

    #[test]
    fn validated_generated_record_requires_same_claim_and_scope() {
        let generated = bound_record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:dream-model",
        );
        let recorded = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:alpha",
            "scope:site-a",
            "family:external-sensor",
        );
        let validated = generated.validated_by(&recorded).unwrap();
        assert!(validated.may_promote_confidence());
        assert_eq!(validated.record().kind, WorldEvidenceKind::Counterfactual);
        assert_eq!(validated.validation().kind(), WorldEvidenceKind::Recorded);
    }

    #[test]
    fn different_claim_cannot_validate_generated_record() {
        let generated = bound_record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:dream-model",
        );
        let recorded = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:beta",
            "scope:site-a",
            "family:external-sensor",
        );
        assert_eq!(
            generated.validated_by(&recorded),
            Err(EpistemicWorldError::ClaimIdentityMismatch)
        );
    }

    #[test]
    fn different_scope_cannot_validate_generated_record() {
        let generated = bound_record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:dream-model",
        );
        let recorded = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:alpha",
            "scope:site-b",
            "family:external-sensor",
        );
        assert_eq!(
            generated.validated_by(&recorded),
            Err(EpistemicWorldError::ScopeIdentityMismatch)
        );
    }

    #[test]
    fn same_source_family_or_same_event_cannot_validate() {
        let generated = bound_record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:same",
        );
        let same_family = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:alpha",
            "scope:site-a",
            "family:same",
        );
        let same_event = bound_record(
            WorldEvidenceKind::Recorded,
            "event:dream",
            "claim:alpha",
            "scope:site-a",
            "family:external",
        );
        assert_eq!(
            generated.validated_by(&same_family),
            Err(EpistemicWorldError::SourceFamilyNotIndependent)
        );
        assert_eq!(
            generated.validated_by(&same_event),
            Err(EpistemicWorldError::SameProvenanceDigest)
        );
    }

    #[test]
    fn unbound_records_cannot_authorize() {
        let record = EpistemicWorldRecord::new(
            WorldEvidenceKind::Recorded,
            "event:recorded",
        )
        .unwrap();
        assert!(!record.may_promote_confidence());
    }

    #[test]
    fn invalid_numeric_metadata_is_rejected() {
        let record = bound_record(
            WorldEvidenceKind::Recorded,
            "event:recorded",
            "claim:alpha",
            "scope:site-a",
            "family:sensor-a",
        );
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
