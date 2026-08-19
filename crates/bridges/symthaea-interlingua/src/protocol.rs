// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, modality-neutral contracts for the Symthaea Cognitive Interchange Protocol (SCIP).
//!
//! `GroundedConceptGraph` remains the canonical semantic object. HDC, structured
//! data, references, and text are negotiated representations of that meaning.
//! An HDC vector alone is never evidence that a particular meaning is grounded.

use serde::{Deserialize, Serialize};
use symthaea_communication::{
    GroundedConceptGraph, Provenance, content_hash, valid_confidence,
};

pub const SCIP_PROTOCOL_ID: &str = "symthaea-cognitive-interchange";
pub const SCIP_V1: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };
pub const GROUNDED_GRAPH_SCHEMA_V1: &str = "symthaea.grounded-concept-graph/v1";

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterchangeRepresentation {
    GroundedGraph,
    Hdc,
    StructuredJson,
    HumanText,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcProfile {
    pub dimension: usize,
    pub algebra: String,
    pub atom_derivation: String,
    pub namespace: String,
    pub codebook_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticProfile {
    pub representation: InterchangeRepresentation,
    pub schema_id: String,
    pub hdc: Option<HdcProfile>,
}

impl SemanticProfile {
    pub fn grounded_graph() -> Self {
        Self {
            representation: InterchangeRepresentation::GroundedGraph,
            schema_id: GROUNDED_GRAPH_SCHEMA_V1.into(),
            hdc: None,
        }
    }

    pub fn hdc(profile: HdcProfile) -> Self {
        Self {
            representation: InterchangeRepresentation::Hdc,
            schema_id: GROUNDED_GRAPH_SCHEMA_V1.into(),
            hdc: Some(profile),
        }
    }

    pub fn structured_json() -> Self {
        Self {
            representation: InterchangeRepresentation::StructuredJson,
            schema_id: GROUNDED_GRAPH_SCHEMA_V1.into(),
            hdc: None,
        }
    }

    pub fn human_text() -> Self {
        Self {
            representation: InterchangeRepresentation::HumanText,
            schema_id: GROUNDED_GRAPH_SCHEMA_V1.into(),
            hdc: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcPayload {
    pub values: Vec<f32>,
    pub semantic_hash: String,
    pub profile_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticReference {
    pub semantic_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InterchangePayload {
    GroundedGraph(GroundedConceptGraph),
    Hdc(HdcPayload),
    StructuredJson(Vec<u8>),
    HumanText(String),
    Reference(SemanticReference),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEnvelope {
    pub protocol: String,
    pub version: ProtocolVersion,
    pub message_id: String,
    pub parent_id: Option<String>,
    pub profile: SemanticProfile,
    pub payload: InterchangePayload,
    pub confidence: f32,
    pub provenance: Provenance,
    pub evidence_ids: Vec<String>,
}

impl CognitiveEnvelope {
    pub fn new(
        profile: SemanticProfile,
        payload: InterchangePayload,
        confidence: f32,
        provenance: Provenance,
    ) -> Result<Self, InterchangeError> {
        let mut value = Self {
            protocol: SCIP_PROTOCOL_ID.into(),
            version: SCIP_V1,
            message_id: String::new(),
            parent_id: None,
            profile,
            payload,
            confidence,
            provenance,
            evidence_ids: vec![],
        };
        value.validate_content()?;
        value.refresh_id()?;
        Ok(value)
    }

    pub fn from_graph(
        graph: GroundedConceptGraph,
        confidence: f32,
        provenance: Provenance,
    ) -> Result<Self, InterchangeError> {
        Self::new(
            SemanticProfile::grounded_graph(),
            InterchangePayload::GroundedGraph(graph),
            confidence,
            provenance,
        )
    }

    pub fn computed_id(&self) -> Result<String, InterchangeError> {
        let mut canonical = self.clone();
        canonical.message_id.clear();
        Ok(content_hash(&serde_json::to_vec(&canonical)?))
    }

    pub fn refresh_id(&mut self) -> Result<(), InterchangeError> {
        self.message_id = self.computed_id()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        self.validate_content()?;
        if self.computed_id()? != self.message_id {
            return Err(InterchangeError::InvalidIdentity);
        }
        Ok(())
    }

    fn validate_content(&self) -> Result<(), InterchangeError> {
        if self.protocol != SCIP_PROTOCOL_ID {
            return Err(InterchangeError::UnsupportedProtocol(self.protocol.clone()));
        }
        if self.version != SCIP_V1 {
            return Err(InterchangeError::UnsupportedVersion(self.version));
        }
        if self.profile.schema_id != GROUNDED_GRAPH_SCHEMA_V1 {
            return Err(InterchangeError::UnsupportedSchema(self.profile.schema_id.clone()));
        }
        if !valid_confidence(self.confidence) {
            return Err(InterchangeError::InvalidConfidence);
        }
        if self.provenance.provider.trim().is_empty()
            || self.provenance.model_hash.trim().is_empty()
        {
            return Err(InterchangeError::InvalidProvenance);
        }

        match (&self.profile.representation, &self.profile.hdc, &self.payload) {
            (
                InterchangeRepresentation::GroundedGraph,
                None,
                InterchangePayload::GroundedGraph(graph),
            ) => validate_graph(graph)?,
            (
                InterchangeRepresentation::Hdc,
                Some(profile),
                InterchangePayload::Hdc(payload),
            ) => {
                if profile.dimension == 0
                    || payload.values.len() != profile.dimension
                    || payload.profile_fingerprint != profile.codebook_fingerprint
                    || payload.values.iter().any(|value| !value.is_finite())
                    || payload.semantic_hash.is_empty()
                {
                    return Err(InterchangeError::InvalidHdcPayload);
                }
            }
            (
                InterchangeRepresentation::StructuredJson,
                None,
                InterchangePayload::StructuredJson(_),
            )
            | (
                InterchangeRepresentation::HumanText,
                None,
                InterchangePayload::HumanText(_),
            ) => {}
            (_, _, InterchangePayload::Reference(reference))
                if !reference.semantic_hash.trim().is_empty() => {}
            _ => return Err(InterchangeError::ProfilePayloadMismatch),
        }

        Ok(())
    }

    pub fn semantic_hash(&self) -> Result<Option<String>, InterchangeError> {
        match &self.payload {
            InterchangePayload::GroundedGraph(graph) => Ok(Some(graph_semantic_hash(graph)?)),
            InterchangePayload::Hdc(payload) => Ok(Some(payload.semantic_hash.clone())),
            InterchangePayload::Reference(reference) => Ok(Some(reference.semantic_hash.clone())),
            InterchangePayload::StructuredJson(_) | InterchangePayload::HumanText(_) => Ok(None),
        }
    }
}

pub fn validate_graph(graph: &GroundedConceptGraph) -> Result<(), InterchangeError> {
    let mut ids = std::collections::BTreeSet::new();
    for node in &graph.nodes {
        if node.id.trim().is_empty() || !ids.insert(node.id.as_str()) {
            return Err(InterchangeError::InvalidGraph("empty or duplicate node id".into()));
        }
        if !valid_confidence(node.confidence) {
            return Err(InterchangeError::InvalidGraph(format!(
                "invalid confidence for node {}",
                node.id
            )));
        }
    }
    for edge in &graph.edges {
        if edge.source.trim().is_empty()
            || edge.relation.trim().is_empty()
            || edge.target.trim().is_empty()
            || !valid_confidence(edge.confidence)
        {
            return Err(InterchangeError::InvalidGraph("invalid edge".into()));
        }
    }
    Ok(())
}

pub fn canonical_graph_bytes(graph: &GroundedConceptGraph) -> Result<Vec<u8>, InterchangeError> {
    validate_graph(graph)?;
    let mut canonical = graph.clone();
    for node in &mut canonical.nodes {
        node.grounded_by.sort();
    }
    canonical.nodes.sort_by(|a, b| a.id.cmp(&b.id));
    for edge in &mut canonical.edges {
        edge.evidence_ids.sort();
    }
    canonical.edges.sort_by(|a, b| {
        (&a.source, &a.relation, &a.target).cmp(&(&b.source, &b.relation, &b.target))
    });
    Ok(serde_json::to_vec(&canonical)?)
}

pub fn graph_semantic_hash(graph: &GroundedConceptGraph) -> Result<String, InterchangeError> {
    Ok(content_hash(&canonical_graph_bytes(graph)?))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterchangeError {
    Serialization(String),
    UnsupportedProtocol(String),
    UnsupportedVersion(ProtocolVersion),
    UnsupportedSchema(String),
    InvalidGraph(String),
    InvalidConfidence,
    InvalidProvenance,
    InvalidIdentity,
    ProfilePayloadMismatch,
    InvalidHdcPayload,
    NegotiationFailed,
    MissingSemanticReference(String),
    InvalidDelta(String),
}

impl std::fmt::Display for InterchangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(message) => write!(f, "serialization error: {message}"),
            Self::UnsupportedProtocol(value) => write!(f, "unsupported protocol: {value}"),
            Self::UnsupportedVersion(value) => {
                write!(f, "unsupported SCIP version: {}.{}", value.major, value.minor)
            }
            Self::UnsupportedSchema(value) => write!(f, "unsupported schema: {value}"),
            Self::InvalidGraph(value) => write!(f, "invalid grounded graph: {value}"),
            Self::InvalidConfidence => write!(f, "invalid confidence"),
            Self::InvalidProvenance => write!(f, "invalid provenance"),
            Self::InvalidIdentity => write!(f, "invalid content-addressed message id"),
            Self::ProfilePayloadMismatch => write!(f, "profile does not match payload"),
            Self::InvalidHdcPayload => write!(f, "invalid HDC payload"),
            Self::NegotiationFailed => write!(f, "no mutually supported SCIP representation"),
            Self::MissingSemanticReference(value) => {
                write!(f, "missing semantic reference: {value}")
            }
            Self::InvalidDelta(value) => write!(f, "invalid HDC delta: {value}"),
        }
    }
}

impl std::error::Error for InterchangeError {}

impl From<serde_json::Error> for InterchangeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_communication::{ConceptKind, ConceptNode};

    fn provenance() -> Provenance {
        Provenance {
            provider: "scip-test".into(),
            provider_version: "1".into(),
            model_hash: "model".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn graph() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "b".into(),
                    kind: ConceptKind::Object,
                    label: Some("reactor".into()),
                    grounded_by: vec!["obs-2".into(), "obs-1".into()],
                    confidence: 0.8,
                },
                ConceptNode {
                    id: "a".into(),
                    kind: ConceptKind::Agent,
                    label: Some("alice".into()),
                    grounded_by: vec!["obs-0".into()],
                    confidence: 0.9,
                },
            ],
            edges: vec![],
        }
    }

    #[test]
    fn graph_hash_ignores_graph_order() {
        let first = graph();
        let mut second = first.clone();
        second.nodes.reverse();
        second.nodes[1].grounded_by.reverse();
        assert_eq!(
            graph_semantic_hash(&first).unwrap(),
            graph_semantic_hash(&second).unwrap()
        );
    }

    #[test]
    fn envelope_identity_detects_mutation() {
        let mut envelope = CognitiveEnvelope::from_graph(graph(), 0.8, provenance()).unwrap();
        assert!(envelope.validate().is_ok());
        envelope.confidence = 0.7;
        assert_eq!(envelope.validate(), Err(InterchangeError::InvalidIdentity));
    }
}
