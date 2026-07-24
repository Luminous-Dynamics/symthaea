// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Modality-neutral communication analysis.
//!
//! The types in this crate deliberately separate detecting a signal, discovering
//! recurring units and structure, and making grounded claims about reference or
//! intent. A model must not report a capability above the evidence it supplies.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod adapters;
pub mod animal;
pub mod artifact;
pub mod benchmark;
pub mod human;
pub mod metrics;
pub mod pilot;
pub mod pipeline;
pub mod provider;
pub mod run;
pub mod unknown;

/// The strongest claim supported by a result, ordered from weakest to strongest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityLevel {
    #[default]
    Signal,
    Unit,
    Structure,
    Reference,
    Intent,
    Dialogue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    Audio {
        sample_rate_hz: u32,
        channels: u16,
    },
    Text {
        language: Option<String>,
        script: Option<String>,
    },
    RadioSpectrum,
    Optical,
    Image,
    TemporalEvents,
    NumericStream,
    Other(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub start_s: f64,
    pub end_s: f64,
}

impl TimeSpan {
    pub fn is_valid(self) -> bool {
        self.start_s.is_finite() && self.end_s.is_finite() && self.start_s <= self.end_s
    }
}

/// Calibration and transformation details supplied by the instrument owner.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensorCalibration {
    pub instrument_id: Option<String>,
    pub calibrated_at: Option<String>,
    pub units: Option<String>,
    pub parameters: BTreeMap<String, f64>,
}

/// An observation preserves raw material and derived features side by side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalObservation {
    pub id: String,
    pub modality: Modality,
    pub samples: Vec<f32>,
    pub features: BTreeMap<String, Vec<f32>>,
    pub original_text: Option<String>,
    pub normalized_text: Option<String>,
    pub uncertain_spans: Vec<TimeSpan>,
    pub timing: TimeSpan,
    pub location: Option<[f64; 3]>,
    pub calibration: SensorCalibration,
    pub source_identity: Option<String>,
    pub environment: BTreeMap<String, String>,
}

impl SignalObservation {
    pub fn computed_id(&self) -> Result<String, serde_json::Error> {
        let mut canonical = self.clone();
        canonical.id.clear();
        serde_json::to_vec(&canonical).map(|bytes| content_hash(&bytes))
    }

    pub fn validate_identity(&self) -> Result<(), CommunicationError> {
        if !self.timing.is_valid() {
            return Err(CommunicationError::InvalidObservation(
                "invalid timing".into(),
            ));
        }
        let expected = self
            .computed_id()
            .map_err(|error| CommunicationError::InvalidObservation(error.to_string()))?;
        if self.id != expected {
            return Err(CommunicationError::InvalidObservation(
                "observation id is not content-addressed".into(),
            ));
        }
        Ok(())
    }

    pub fn refresh_id(&mut self) -> Result<(), serde_json::Error> {
        self.id = self.computed_id()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnitRepresentation {
    Features(Vec<f32>),
    Symbol(String),
    ExternalEmbedding { provider: String, values: Vec<f32> },
    Opaque(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommunicationUnit {
    pub id: String,
    pub observation_id: String,
    pub boundary: TimeSpan,
    pub boundary_confidence: f32,
    pub representation: UnitRepresentation,
    pub recurrence_count: u64,
    pub parent_unit: Option<String>,
    pub alternatives: Vec<AlternativeBoundary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlternativeBoundary {
    pub boundary: TimeSpan,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionContext {
    pub agents: Vec<String>,
    pub behaviors: Vec<String>,
    pub objects: Vec<String>,
    pub physiological_state: BTreeMap<String, f32>,
    pub preceding_events: Vec<EventFrame>,
    pub observable_outcomes: Vec<EventFrame>,
}

/// Grounded interlingua node. It is not restricted to words or NSM primes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventFrame {
    pub predicate: String,
    pub roles: BTreeMap<String, String>,
    pub attributes: BTreeMap<String, String>,
}

/// A modality-neutral grounded concept/event graph used as an interlingua.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundedConceptGraph {
    pub nodes: Vec<ConceptNode>,
    pub edges: Vec<ConceptEdge>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub kind: ConceptKind,
    pub label: Option<String>,
    /// Observation, unit, or context identifiers grounding this node.
    pub grounded_by: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptKind {
    Agent,
    Object,
    Event,
    Action,
    State,
    Property,
    Relation,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConceptEdge {
    pub source: String,
    pub relation: String,
    pub target: String,
    pub evidence_ids: Vec<String>,
    pub confidence: f32,
}

/// Named representation families may map to/from the graph without becoming
/// assumptions of the core. For example, NSM is a human-language adapter only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepresentationFamily {
    HumanText,
    Nsm,
    Hdc,
    MultilingualEmbedding,
    BrocaThought,
    AudioUnit,
    Vision,
    Sensorimotor,
    Custom(String),
}

pub trait ConceptGraphAdapter {
    type Input;
    type Output;

    fn family(&self) -> RepresentationFamily;
    fn to_graph(
        &self,
        input: &Self::Input,
        context: &InteractionContext,
    ) -> Result<CommunicationResult<GroundedConceptGraph>, CommunicationError>;
    fn from_graph(
        &self,
        graph: &GroundedConceptGraph,
    ) -> Result<CommunicationResult<Self::Output>, CommunicationError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeaningHypothesis {
    pub id: String,
    pub unit_ids: Vec<String>,
    pub referent: Option<String>,
    pub intent: Option<String>,
    pub event_frame: Option<EventFrame>,
    pub composition: Vec<String>,
    pub counter_hypotheses: Vec<String>,
    pub supporting_evidence: Vec<String>,
    pub contradicting_evidence: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStatus {
    Unreplicated,
    InternallyReplicated,
    IndependentlyReplicated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub predicted_confidence: f32,
    pub observed_frequency: f32,
    pub sample_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub training_ids_hash: Option<String>,
    pub validation_ids_hash: Option<String>,
    pub test_ids_hash: Option<String>,
    pub held_out_identities: bool,
    pub held_out_sites: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommunicationEvidence {
    pub id: String,
    pub dataset_uri: String,
    pub dataset_hash: String,
    pub model_hash: String,
    pub lineage: Vec<String>,
    pub split: DatasetSplit,
    pub evidence_records: Vec<EvidenceRecord>,
    pub preregistration_uri: Option<String>,
    pub replication: ReplicationStatus,
    pub calibration: Vec<CalibrationPoint>,
    pub experimental: bool,
    pub domain: EvidenceDomain,
}

impl CommunicationEvidence {
    pub fn computed_id(&self) -> Result<String, serde_json::Error> {
        let mut canonical = self.clone();
        canonical.id.clear();
        serde_json::to_vec(&canonical).map(|bytes| content_hash(&bytes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDomain {
    HumanLanguage,
    AnimalCommunication,
    UnknownSignal,
    SyntheticProtocol,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub kind: EvidenceKind,
    pub description: String,
    pub result_hash: String,
    pub preregistered: bool,
    pub independently_replicated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    ObservationalCorrelation,
    HeldOutPrediction,
    PermutationControl,
    Intervention,
    IndependentReplication,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub provider: String,
    pub provider_version: String,
    pub model_hash: String,
    pub feature_flags: Vec<String>,
    pub transformations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alternative<T> {
    pub value: T,
    pub confidence: f32,
}

/// Common envelope used at every pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommunicationResult<T> {
    pub value: T,
    pub capability: CapabilityLevel,
    pub confidence: f32,
    pub alternatives: Vec<Alternative<T>>,
    pub provenance: Provenance,
    pub evidence: Vec<CommunicationEvidence>,
}

impl<T> CommunicationResult<T> {
    pub fn validate(&self, stage_ceiling: CapabilityLevel) -> Result<(), CommunicationError> {
        if !valid_confidence(self.confidence)
            || self
                .alternatives
                .iter()
                .any(|alternative| !valid_confidence(alternative.confidence))
        {
            return Err(CommunicationError::InvalidResult(
                "confidence must be finite and in [0, 1]".into(),
            ));
        }
        if self.capability > stage_ceiling {
            return Err(CommunicationError::InvalidResult(
                "capability exceeds the pipeline stage ceiling".into(),
            ));
        }
        if self.provenance.provider.trim().is_empty()
            || self.provenance.model_hash.trim().is_empty()
        {
            return Err(CommunicationError::InvalidResult(
                "provider and model hash are required".into(),
            ));
        }
        if self.evidence.is_empty() {
            return Err(CommunicationError::InsufficientEvidence(
                "every communication result requires explicit evidence, including experimental evidence".into(),
            ));
        }
        if self.capability >= CapabilityLevel::Reference
            && requires_grounded_intervention(&self.evidence)
            && !has_grounded_intervention(&self.evidence)
        {
            return Err(CommunicationError::InsufficientEvidence(
                "reference and intent require preregistered intervention evidence".into(),
            ));
        }
        Ok(())
    }
}

pub fn valid_confidence(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

pub fn has_grounded_intervention(evidence: &[CommunicationEvidence]) -> bool {
    evidence
        .iter()
        .flat_map(|item| &item.evidence_records)
        .any(|record| record.kind == EvidenceKind::Intervention && record.preregistered)
}

pub fn requires_grounded_intervention(evidence: &[CommunicationEvidence]) -> bool {
    evidence.iter().any(|item| {
        matches!(
            item.domain,
            EvidenceDomain::AnimalCommunication | EvidenceDomain::UnknownSignal
        )
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommunicationError {
    InvalidObservation(String),
    UnsupportedModality,
    InsufficientEvidence(String),
    Provider(String),
    InvalidResult(String),
}

pub trait CommunicationModel {
    type Encoding;
    type Structure;

    fn observe(
        &mut self,
        observation: SignalObservation,
    ) -> Result<CommunicationResult<SignalObservation>, CommunicationError>;
    fn segment(
        &mut self,
        observation: &SignalObservation,
    ) -> Result<CommunicationResult<Vec<CommunicationUnit>>, CommunicationError>;
    fn encode(
        &self,
        units: &[CommunicationUnit],
    ) -> Result<CommunicationResult<Self::Encoding>, CommunicationError>;
    fn infer_structure(
        &self,
        encoding: &Self::Encoding,
    ) -> Result<CommunicationResult<Self::Structure>, CommunicationError>;
    fn propose_meanings(
        &self,
        structure: &Self::Structure,
        context: &InteractionContext,
    ) -> Result<CommunicationResult<Vec<MeaningHypothesis>>, CommunicationError>;
    fn update_from_outcome(
        &mut self,
        hypotheses: &[MeaningHypothesis],
        outcome: &EventFrame,
    ) -> Result<(), CommunicationError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpressionTarget {
    Human,
    AnimalPlayback,
    UnknownOrExtraterrestrial,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpressionAuthorization {
    pub protocol_id: String,
    pub reviewer: String,
    pub expires_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpressionDecision {
    Allow,
    Block(&'static str),
}

/// Conservative default: animal playback needs review; unknown targets are never transmitted to.
#[derive(Clone, Debug, Default)]
pub struct ExpressionPolicy;

impl ExpressionPolicy {
    pub fn evaluate(
        &self,
        target: &ExpressionTarget,
        capability: CapabilityLevel,
        evidence: &[CommunicationEvidence],
        authorization: Option<&ExpressionAuthorization>,
    ) -> ExpressionDecision {
        if evidence.is_empty() || evidence.iter().any(|item| item.experimental) {
            return ExpressionDecision::Block("missing release-grade evidence");
        }
        match target {
            ExpressionTarget::UnknownOrExtraterrestrial => {
                ExpressionDecision::Block("autonomous transmission is prohibited")
            }
            ExpressionTarget::AnimalPlayback if authorization.is_none() => {
                ExpressionDecision::Block("animal playback requires a reviewed protocol")
            }
            ExpressionTarget::AnimalPlayback if capability < CapabilityLevel::Intent => {
                ExpressionDecision::Block("evidence does not establish communicative intent")
            }
            ExpressionTarget::Human if capability < CapabilityLevel::Intent => {
                ExpressionDecision::Block("insufficient evidence for an intentional reply")
            }
            _ => ExpressionDecision::Allow,
        }
    }
}

pub fn content_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(experimental: bool) -> CommunicationEvidence {
        let mut e = CommunicationEvidence {
            id: String::new(),
            dataset_uri: "dataset:test".into(),
            dataset_hash: "d".into(),
            model_hash: "m".into(),
            lineage: vec![],
            split: DatasetSplit::default(),
            evidence_records: vec![],
            preregistration_uri: None,
            replication: ReplicationStatus::Unreplicated,
            calibration: vec![],
            experimental,
            domain: EvidenceDomain::HumanLanguage,
        };
        e.id = e.computed_id().unwrap();
        e
    }

    #[test]
    fn unknown_transmission_is_always_blocked() {
        assert!(matches!(
            ExpressionPolicy.evaluate(
                &ExpressionTarget::UnknownOrExtraterrestrial,
                CapabilityLevel::Dialogue,
                &[evidence(false)],
                None
            ),
            ExpressionDecision::Block(_)
        ));
    }

    #[test]
    fn experimental_results_cannot_authorize_expression() {
        assert!(matches!(
            ExpressionPolicy.evaluate(
                &ExpressionTarget::Human,
                CapabilityLevel::Dialogue,
                &[evidence(true)],
                None
            ),
            ExpressionDecision::Block(_)
        ));
    }

    #[test]
    fn capability_order_is_monotonic() {
        assert!(CapabilityLevel::Structure < CapabilityLevel::Reference);
    }

    // --- Schema round-trip tests ---
    // These fail if field names, enum variants, or serialization tags change
    // without a corresponding schema-version bump. Never delete these tests;
    // append new ones when the schema evolves.

    #[test]
    fn capability_level_schema_is_stable() {
        let cases = [
            (CapabilityLevel::Signal, "\"Signal\""),
            (CapabilityLevel::Unit, "\"Unit\""),
            (CapabilityLevel::Structure, "\"Structure\""),
            (CapabilityLevel::Reference, "\"Reference\""),
            (CapabilityLevel::Intent, "\"Intent\""),
            (CapabilityLevel::Dialogue, "\"Dialogue\""),
        ];
        for (level, expected) in cases {
            let serialized = serde_json::to_string(&level).unwrap();
            assert_eq!(
                serialized, expected,
                "CapabilityLevel::{level:?} serialization changed"
            );
            let round_tripped: CapabilityLevel = serde_json::from_str(&serialized).unwrap();
            assert_eq!(round_tripped, level);
        }
    }

    #[test]
    fn evidence_domain_schema_is_stable() {
        for (domain, expected) in [
            (EvidenceDomain::HumanLanguage, "\"HumanLanguage\""),
            (
                EvidenceDomain::AnimalCommunication,
                "\"AnimalCommunication\"",
            ),
            (EvidenceDomain::UnknownSignal, "\"UnknownSignal\""),
            (EvidenceDomain::SyntheticProtocol, "\"SyntheticProtocol\""),
        ] {
            assert_eq!(serde_json::to_string(&domain).unwrap(), expected);
        }
    }

    #[test]
    fn evidence_kind_schema_is_stable() {
        for (kind, expected) in [
            (
                EvidenceKind::ObservationalCorrelation,
                "\"ObservationalCorrelation\"",
            ),
            (EvidenceKind::HeldOutPrediction, "\"HeldOutPrediction\""),
            (EvidenceKind::PermutationControl, "\"PermutationControl\""),
            (EvidenceKind::Intervention, "\"Intervention\""),
            (
                EvidenceKind::IndependentReplication,
                "\"IndependentReplication\"",
            ),
        ] {
            assert_eq!(serde_json::to_string(&kind).unwrap(), expected);
        }
    }

    #[test]
    fn observation_id_is_stable_across_serialization() {
        let mut obs = SignalObservation {
            id: String::new(),
            modality: Modality::Audio {
                sample_rate_hz: 16000,
                channels: 1,
            },
            samples: vec![0.1, 0.2],
            features: BTreeMap::new(),
            original_text: None,
            normalized_text: None,
            uncertain_spans: vec![],
            timing: TimeSpan {
                start_s: 0.0,
                end_s: 0.125,
            },
            location: None,
            calibration: SensorCalibration::default(),
            source_identity: None,
            environment: BTreeMap::new(),
        };
        obs.refresh_id().unwrap();
        let id_before = obs.id.clone();
        // Serialise and deserialise; the ID must be identical.
        let json = serde_json::to_string(&obs).unwrap();
        let restored: SignalObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, id_before);
        // Validate that the restored observation passes the identity check.
        assert!(restored.validate_identity().is_ok());
    }

    #[test]
    fn evidence_id_is_stable_across_serialization() {
        let e = evidence(false);
        let json = serde_json::to_string(&e).unwrap();
        let restored: CommunicationEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, e.id);
        assert_eq!(restored.computed_id().unwrap(), e.id);
    }

    #[test]
    fn valid_confidence_accepts_boundary_values() {
        assert!(valid_confidence(0.0));
        assert!(valid_confidence(1.0));
        assert!(valid_confidence(0.5));
        assert!(!valid_confidence(-0.001));
        assert!(!valid_confidence(1.001));
        assert!(!valid_confidence(f32::NAN));
        assert!(!valid_confidence(f32::INFINITY));
    }

    #[test]
    fn modality_audio_roundtrips_cleanly() {
        let m = Modality::Audio {
            sample_rate_hz: 22050,
            channels: 2,
        };
        let json = serde_json::to_string(&m).unwrap();
        let restored: Modality = serde_json::from_str(&json).unwrap();
        assert_eq!(m, restored);
    }

    #[test]
    fn timespan_validity_edge_cases() {
        assert!(
            TimeSpan {
                start_s: 0.0,
                end_s: 0.0
            }
            .is_valid()
        ); // zero-length is valid
        assert!(
            !TimeSpan {
                start_s: 1.0,
                end_s: 0.0
            }
            .is_valid()
        ); // reversed
        assert!(
            !TimeSpan {
                start_s: f64::NAN,
                end_s: 0.0
            }
            .is_valid()
        );
        assert!(
            !TimeSpan {
                start_s: 0.0,
                end_s: f64::INFINITY
            }
            .is_valid()
        );
    }
}
