//! Conflict-preserving evidence analysis for Planetary Perception.
//!
//! Conflicting evidence is not averaged away and a plausible explanation is
//! not the same thing as a resolution. This crate retains the original
//! `EvidenceConflict`, records candidate explanations and discriminating
//! evidence needs, and requires explicit verification evidence before a conflict
//! may be marked resolved.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_earth_observation::{EvidenceConflict, EvidenceRef, EvidenceStage};

pub type Result<T> = std::result::Result<T, ConflictError>;

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictError {
    EmptyField(&'static str),
    InvalidSupport(f64),
    MissingExplanation,
    MissingDiscriminatingNeed,
    ResolutionRequiresVerification,
}

impl Display for ConflictError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::InvalidSupport(value) => {
                write!(f, "explanation support must be finite and in [0, 1], got {value}")
            }
            Self::MissingExplanation => write!(f, "conflict assessment requires at least one candidate explanation"),
            Self::MissingDiscriminatingNeed => write!(f, "open conflict assessment requires at least one discriminating evidence need"),
            Self::ResolutionRequiresVerification => write!(
                f,
                "a conflict may be marked resolved only with explicit verification-stage evidence"
            ),
        }
    }
}

impl Error for ConflictError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ConflictError::EmptyField(field));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictCauseClass {
    SpatialMismatch,
    TemporalMismatch,
    CalibrationMismatch,
    SensorFault,
    ProcessingArtifact,
    ModelFailure,
    SamplingBias,
    RealWorldHeterogeneity,
    DefinitionMismatch,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConflictExplanation {
    pub class: ConflictCauseClass,
    pub statement: String,
    /// Relative support for this explanation inside this assessment. This is
    /// not necessarily a calibrated probability.
    pub support: f64,
    pub supporting_evidence: Vec<EvidenceRef>,
    pub contradicting_evidence: Vec<EvidenceRef>,
}

impl ConflictExplanation {
    pub fn new(
        class: ConflictCauseClass,
        statement: impl Into<String>,
        support: f64,
        supporting_evidence: Vec<EvidenceRef>,
        contradicting_evidence: Vec<EvidenceRef>,
    ) -> Result<Self> {
        let statement = statement.into();
        non_empty(&statement, "conflict explanation statement")?;
        if !support.is_finite() || !(0.0..=1.0).contains(&support) {
            return Err(ConflictError::InvalidSupport(support));
        }
        Ok(Self {
            class,
            statement,
            support,
            supporting_evidence,
            contradicting_evidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscriminatingEvidenceNeed {
    pub question: String,
    pub rationale: String,
    pub preferred_evidence_stage: EvidenceStage,
}

impl DiscriminatingEvidenceNeed {
    pub fn new(
        question: impl Into<String>,
        rationale: impl Into<String>,
        preferred_evidence_stage: EvidenceStage,
    ) -> Result<Self> {
        let question = question.into();
        let rationale = rationale.into();
        non_empty(&question, "discriminating question")?;
        non_empty(&rationale, "discriminating rationale")?;
        Ok(Self {
            question,
            rationale,
            preferred_evidence_stage,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictDisposition {
    Open,
    ExplainedButUnresolved,
    Resolved,
}

/// An assessment never replaces or mutates the original competing evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceConflictAssessment {
    pub conflict_id: String,
    pub original_competing_evidence: Vec<EvidenceRef>,
    pub original_note: String,
    pub explanations: Vec<ConflictExplanation>,
    pub discriminating_needs: Vec<DiscriminatingEvidenceNeed>,
    pub disposition: ConflictDisposition,
    pub resolution_evidence: Vec<EvidenceRef>,
    pub resolution_note: Option<String>,
}

impl EvidenceConflictAssessment {
    pub fn open(
        conflict: &EvidenceConflict,
        explanations: Vec<ConflictExplanation>,
        discriminating_needs: Vec<DiscriminatingEvidenceNeed>,
    ) -> Result<Self> {
        if explanations.is_empty() {
            return Err(ConflictError::MissingExplanation);
        }
        if discriminating_needs.is_empty() {
            return Err(ConflictError::MissingDiscriminatingNeed);
        }
        Ok(Self {
            conflict_id: conflict.id.clone(),
            original_competing_evidence: conflict.competing_evidence.clone(),
            original_note: conflict.note.clone(),
            explanations,
            discriminating_needs,
            disposition: ConflictDisposition::Open,
            resolution_evidence: Vec::new(),
            resolution_note: None,
        })
    }

    pub fn explain_without_resolution(mut self) -> Self {
        self.disposition = ConflictDisposition::ExplainedButUnresolved;
        self
    }

    pub fn resolve(
        mut self,
        verification_evidence: Vec<EvidenceRef>,
        note: impl Into<String>,
    ) -> Result<Self> {
        let note = note.into();
        non_empty(&note, "conflict resolution note")?;
        if verification_evidence.is_empty()
            || !verification_evidence
                .iter()
                .any(|reference| reference.stage == EvidenceStage::Verification)
        {
            return Err(ConflictError::ResolutionRequiresVerification);
        }
        self.disposition = ConflictDisposition::Resolved;
        self.resolution_evidence = verification_evidence;
        self.resolution_note = Some(note);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str, stage: EvidenceStage) -> EvidenceRef {
        EvidenceRef::new(id, stage).unwrap()
    }

    fn conflict() -> EvidenceConflict {
        EvidenceConflict::new(
            "flood-conflict",
            vec![
                evidence("sar-flood", EvidenceStage::DerivedFeature),
                evidence("gauge-normal", EvidenceStage::Measurement),
            ],
            "SAR indicates water while the gauge remains normal",
        )
        .unwrap()
    }

    fn explanation() -> ConflictExplanation {
        ConflictExplanation::new(
            ConflictCauseClass::SpatialMismatch,
            "the gauge may not sample the inundated floodplain",
            0.6,
            vec![evidence("map", EvidenceStage::Observation)],
            vec![],
        )
        .unwrap()
    }

    fn need() -> DiscriminatingEvidenceNeed {
        DiscriminatingEvidenceNeed::new(
            "is standing water present at the SAR-positive footprint?",
            "a local observation distinguishes gauge mismatch from SAR artefact",
            EvidenceStage::Verification,
        )
        .unwrap()
    }

    #[test]
    fn original_competing_evidence_is_preserved() {
        let source = conflict();
        let assessment = EvidenceConflictAssessment::open(
            &source,
            vec![explanation()],
            vec![need()],
        )
        .unwrap();
        assert_eq!(assessment.original_competing_evidence, source.competing_evidence);
        assert_eq!(assessment.disposition, ConflictDisposition::Open);
    }

    #[test]
    fn plausible_explanation_does_not_resolve_conflict() {
        let assessment = EvidenceConflictAssessment::open(
            &conflict(),
            vec![explanation()],
            vec![need()],
        )
        .unwrap()
        .explain_without_resolution();
        assert_eq!(
            assessment.disposition,
            ConflictDisposition::ExplainedButUnresolved
        );
        assert!(assessment.resolution_evidence.is_empty());
    }

    #[test]
    fn ordinary_observation_cannot_mark_conflict_resolved() {
        let assessment = EvidenceConflictAssessment::open(
            &conflict(),
            vec![explanation()],
            vec![need()],
        )
        .unwrap();
        let result = assessment.resolve(
            vec![evidence("another-image", EvidenceStage::Observation)],
            "looks consistent",
        );
        assert_eq!(
            result.unwrap_err(),
            ConflictError::ResolutionRequiresVerification
        );
    }

    #[test]
    fn explicit_verification_can_resolve_while_retaining_history() {
        let source = conflict();
        let assessment = EvidenceConflictAssessment::open(
            &source,
            vec![explanation()],
            vec![need()],
        )
        .unwrap()
        .resolve(
            vec![evidence("field-check", EvidenceStage::Verification)],
            "field verification confirmed floodplain water outside the gauge location",
        )
        .unwrap();
        assert_eq!(assessment.disposition, ConflictDisposition::Resolved);
        assert_eq!(assessment.original_competing_evidence, source.competing_evidence);
        assert_eq!(assessment.resolution_evidence.len(), 1);
    }

    #[test]
    fn support_is_not_claimed_to_be_calibrated_probability() {
        let explanation = explanation();
        assert_eq!(explanation.support, 0.6);
        // Semantics are deliberately documented as relative assessment support.
    }
}
