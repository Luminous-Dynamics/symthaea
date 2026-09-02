use serde::{Deserialize, Serialize};
use std::fmt;

/// Reality-domain provenance carried independently of confidence or authority.
///
/// Grounded domains cannot be constructed implicitly through [`ProvenanceEnvelope::new`]
/// or [`ProvenanceEnvelope::derive`]. They require explicit [`GroundingEvidence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RealityDomain {
    PhysicalGrounded,
    DigitalCommitted,
    Counterfactual,
    Replay,
    Dream,
    Imported,
    #[default]
    Unknown,
}

impl RealityDomain {
    pub const fn is_grounded(self) -> bool {
        matches!(self, Self::PhysicalGrounded | Self::DigitalCommitted)
    }

    pub const fn is_intrinsically_counterfactual(self) -> bool {
        matches!(self, Self::Counterfactual | Self::Dream)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GroundingEvidenceKind {
    DirectObservation,
    CommitReceipt,
}

impl GroundingEvidenceKind {
    pub const fn target_domain(self) -> RealityDomain {
        match self {
            Self::DirectObservation => RealityDomain::PhysicalGrounded,
            Self::CommitReceipt => RealityDomain::DigitalCommitted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundingEvidence {
    pub kind: GroundingEvidenceKind,
    pub evidence_id: String,
    pub source_id: String,
    pub event_time_ns: Option<u64>,
    pub confidence: f32,
}

impl GroundingEvidence {
    pub fn direct_observation(
        evidence_id: impl Into<String>,
        source_id: impl Into<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
    ) -> Result<Self, ProvenanceError> {
        Self::new(
            GroundingEvidenceKind::DirectObservation,
            evidence_id,
            source_id,
            event_time_ns,
            confidence,
        )
    }

    pub fn commit_receipt(
        evidence_id: impl Into<String>,
        source_id: impl Into<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
    ) -> Result<Self, ProvenanceError> {
        Self::new(
            GroundingEvidenceKind::CommitReceipt,
            evidence_id,
            source_id,
            event_time_ns,
            confidence,
        )
    }

    pub fn new(
        kind: GroundingEvidenceKind,
        evidence_id: impl Into<String>,
        source_id: impl Into<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
    ) -> Result<Self, ProvenanceError> {
        validate_confidence(confidence)?;
        let evidence_id = evidence_id.into();
        let source_id = source_id.into();
        if evidence_id.trim().is_empty() {
            return Err(ProvenanceError::EmptyEvidenceId);
        }
        if source_id.trim().is_empty() {
            return Err(ProvenanceError::EmptySourceId);
        }
        Ok(Self {
            kind,
            evidence_id,
            source_id,
            event_time_ns,
            confidence,
        })
    }
}

/// Provenance metadata that prevents counterfactual information from silently
/// becoming grounded history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceEnvelope {
    pub domain: RealityDomain,
    pub source_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub event_time_ns: Option<u64>,
    pub confidence: f32,
    /// True while the value depends on counterfactual/dream content that has not
    /// undergone an explicit grounding transition.
    pub counterfactual_taint: bool,
    /// Historical fact retained even after explicit grounding clears active taint.
    pub counterfactual_ancestry: bool,
    pub derivation_depth: u32,
}

impl ProvenanceEnvelope {
    /// Construct an ungrounded provenance envelope.
    ///
    /// PhysicalGrounded and DigitalCommitted require explicit evidence and are
    /// therefore rejected here.
    pub fn new(
        domain: RealityDomain,
        source_ids: Vec<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
    ) -> Result<Self, ProvenanceError> {
        if domain.is_grounded() {
            return Err(ProvenanceError::GroundedDomainRequiresEvidence(domain));
        }
        validate_confidence(confidence)?;
        let source_ids = normalize_ids(source_ids)?;
        let taint = domain.is_intrinsically_counterfactual();
        Ok(Self {
            domain,
            source_ids,
            evidence_ids: Vec::new(),
            event_time_ns,
            confidence,
            counterfactual_taint: taint,
            counterfactual_ancestry: taint,
            derivation_depth: 0,
        })
    }

    /// Construct a grounded envelope directly from grounding evidence.
    pub fn from_grounding(evidence: GroundingEvidence) -> Self {
        Self {
            domain: evidence.kind.target_domain(),
            source_ids: vec![evidence.source_id],
            evidence_ids: vec![evidence.evidence_id],
            event_time_ns: evidence.event_time_ns,
            confidence: evidence.confidence,
            counterfactual_taint: false,
            counterfactual_ancestry: false,
            derivation_depth: 0,
        }
    }

    /// Derive an ungrounded value from parent envelopes.
    ///
    /// Counterfactual taint propagates transitively. A derived value cannot name
    /// a grounded reality domain; grounding requires an explicit evidence object.
    pub fn derive<'a, I>(
        domain: RealityDomain,
        source_ids: Vec<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
        parents: I,
    ) -> Result<Self, ProvenanceError>
    where
        I: IntoIterator<Item = &'a ProvenanceEnvelope>,
    {
        if domain.is_grounded() {
            return Err(ProvenanceError::GroundedDomainRequiresEvidence(domain));
        }
        validate_confidence(confidence)?;

        let mut parent_count = 0usize;
        let mut inherited_taint = false;
        let mut inherited_ancestry = false;
        let mut max_depth = 0u32;
        let mut inherited_sources = Vec::new();
        let mut inherited_evidence = Vec::new();

        for parent in parents {
            parent_count += 1;
            inherited_taint |= parent.counterfactual_taint;
            inherited_ancestry |= parent.counterfactual_ancestry;
            max_depth = max_depth.max(parent.derivation_depth);
            inherited_sources.extend(parent.source_ids.iter().cloned());
            inherited_evidence.extend(parent.evidence_ids.iter().cloned());
        }

        if parent_count == 0 {
            return Err(ProvenanceError::DerivationRequiresParent);
        }

        inherited_sources.extend(source_ids);
        let source_ids = normalize_ids(inherited_sources)?;
        let evidence_ids = normalize_optional_ids(inherited_evidence);
        let intrinsic = domain.is_intrinsically_counterfactual();

        Ok(Self {
            domain,
            source_ids,
            evidence_ids,
            event_time_ns,
            confidence,
            counterfactual_taint: inherited_taint || intrinsic,
            counterfactual_ancestry: inherited_ancestry || intrinsic,
            derivation_depth: max_depth.saturating_add(1),
        })
    }

    /// Explicitly ground this value with a direct observation or digital commit receipt.
    ///
    /// Grounding clears active counterfactual taint while preserving the fact that
    /// the value has counterfactual ancestry. This makes "was once imagined" and
    /// "is currently justified as grounded" independently inspectable.
    pub fn ground(&self, evidence: GroundingEvidence) -> Result<Self, ProvenanceError> {
        validate_confidence(evidence.confidence)?;
        let mut source_ids = self.source_ids.clone();
        source_ids.push(evidence.source_id);
        let source_ids = normalize_ids(source_ids)?;
        let mut evidence_ids = self.evidence_ids.clone();
        evidence_ids.push(evidence.evidence_id);
        let evidence_ids = normalize_optional_ids(evidence_ids);

        Ok(Self {
            domain: evidence.kind.target_domain(),
            source_ids,
            evidence_ids,
            event_time_ns: evidence.event_time_ns.or(self.event_time_ns),
            confidence: evidence.confidence,
            counterfactual_taint: false,
            counterfactual_ancestry: self.counterfactual_ancestry || self.counterfactual_taint,
            derivation_depth: self.derivation_depth.saturating_add(1),
        })
    }

    pub const fn may_enter_grounded_history(&self) -> bool {
        self.domain.is_grounded() && !self.counterfactual_taint
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceError {
    ConfidenceOutOfRange(f32),
    EmptySourceId,
    EmptyEvidenceId,
    GroundedDomainRequiresEvidence(RealityDomain),
    DerivationRequiresParent,
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfidenceOutOfRange(value) => {
                write!(f, "confidence must be finite and within [0,1], got {value}")
            }
            Self::EmptySourceId => write!(f, "source identifiers must be non-empty"),
            Self::EmptyEvidenceId => write!(f, "grounding evidence identifier must be non-empty"),
            Self::GroundedDomainRequiresEvidence(domain) => {
                write!(f, "grounded domain {domain:?} requires explicit grounding evidence")
            }
            Self::DerivationRequiresParent => write!(f, "derived provenance requires at least one parent"),
        }
    }
}

impl std::error::Error for ProvenanceError {}

fn validate_confidence(confidence: f32) -> Result<(), ProvenanceError> {
    if confidence.is_finite() && (0.0..=1.0).contains(&confidence) {
        Ok(())
    } else {
        Err(ProvenanceError::ConfidenceOutOfRange(confidence))
    }
}

fn normalize_ids(ids: Vec<String>) -> Result<Vec<String>, ProvenanceError> {
    let mut out = Vec::new();
    for id in ids {
        if id.trim().is_empty() {
            return Err(ProvenanceError::EmptySourceId);
        }
        if !out.contains(&id) {
            out.push(id);
        }
    }
    Ok(out)
}

fn normalize_optional_ids(ids: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for id in ids {
        if !id.trim().is_empty() && !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counterfactual_derivation_propagates_taint() {
        let imagined = ProvenanceEnvelope::new(
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            Some(10),
            0.7,
        )
        .unwrap();
        let derived = ProvenanceEnvelope::derive(
            RealityDomain::Imported,
            vec!["reasoner".into()],
            Some(11),
            0.6,
            [&imagined],
        )
        .unwrap();
        assert!(derived.counterfactual_taint);
        assert!(derived.counterfactual_ancestry);
        assert!(!derived.may_enter_grounded_history());
    }

    #[test]
    fn grounded_domains_cannot_be_declared_without_evidence() {
        let err = ProvenanceEnvelope::new(
            RealityDomain::PhysicalGrounded,
            vec!["sensor".into()],
            Some(1),
            0.9,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ProvenanceError::GroundedDomainRequiresEvidence(RealityDomain::PhysicalGrounded)
        ));
    }

    #[test]
    fn observation_grounding_clears_active_taint_but_preserves_ancestry() {
        let imagined = ProvenanceEnvelope::new(
            RealityDomain::Dream,
            vec!["dream-engine".into()],
            None,
            0.4,
        )
        .unwrap();
        let evidence = GroundingEvidence::direct_observation(
            "obs-123",
            "camera-7",
            Some(100),
            0.95,
        )
        .unwrap();
        let grounded = imagined.ground(evidence).unwrap();
        assert_eq!(grounded.domain, RealityDomain::PhysicalGrounded);
        assert!(!grounded.counterfactual_taint);
        assert!(grounded.counterfactual_ancestry);
        assert!(grounded.may_enter_grounded_history());
        assert_eq!(grounded.evidence_ids, vec!["obs-123"]);
    }

    #[test]
    fn commit_receipt_creates_digital_grounding() {
        let evidence = GroundingEvidence::commit_receipt(
            "receipt-abc",
            "world-ledger",
            Some(42),
            1.0,
        )
        .unwrap();
        let grounded = ProvenanceEnvelope::from_grounding(evidence);
        assert_eq!(grounded.domain, RealityDomain::DigitalCommitted);
        assert!(grounded.may_enter_grounded_history());
        assert!(!grounded.counterfactual_ancestry);
    }

    #[test]
    fn mixed_parent_derivation_is_tainted_if_any_parent_is_tainted() {
        let observed = ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation("obs", "sensor", Some(1), 0.9).unwrap(),
        );
        let imagined = ProvenanceEnvelope::new(
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            Some(1),
            0.8,
        )
        .unwrap();
        let mixed = ProvenanceEnvelope::derive(
            RealityDomain::Replay,
            vec![],
            Some(2),
            0.75,
            [&observed, &imagined],
        )
        .unwrap();
        assert!(mixed.counterfactual_taint);
        assert_eq!(mixed.derivation_depth, 1);
        assert!(mixed.source_ids.contains(&"sensor".to_string()));
        assert!(mixed.source_ids.contains(&"planner".to_string()));
    }

    #[test]
    fn invalid_confidence_is_rejected() {
        assert!(matches!(
            ProvenanceEnvelope::new(RealityDomain::Unknown, vec![], None, f32::NAN),
            Err(ProvenanceError::ConfidenceOutOfRange(_))
        ));
    }

    #[test]
    fn serde_round_trip_preserves_taint_and_ancestry() {
        let value = ProvenanceEnvelope::new(
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            Some(9),
            0.5,
        )
        .unwrap();
        let encoded = serde_json::to_string(&value).unwrap();
        let decoded: ProvenanceEnvelope = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, value);
    }
}
