// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, modality-neutral contracts for the Symthaea Cognitive Interchange Protocol (SCIP).
//!
//! `GroundedConceptGraph` remains the canonical semantic object. HDC, structured
//! data, references, and text are negotiated representations of that meaning.
//! An HDC vector alone is never evidence that a particular meaning is grounded.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_communication::{GroundedConceptGraph, Provenance, content_hash, valid_confidence};

pub const SCIP_PROTOCOL_ID: &str = "symthaea-cognitive-interchange";
pub const SCIP_V1: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };
pub const GROUNDED_GRAPH_SCHEMA_V1: &str = "symthaea.grounded-concept-graph/v1";
pub const SCIP_CONTENT_HASH_HEX_LEN: usize = 64;

/// Defensive ceilings for peer-controlled SCIP data.
///
/// These are validation limits, not recommendations for normal messages. The
/// default HDC profile remains 16,384D; the larger ceiling permits controlled
/// experimentation without treating arbitrary peer sizes as trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScipLimits {
    pub max_hdc_dimension: usize,
    pub max_graph_nodes: usize,
    pub max_graph_edges: usize,
    pub max_identifier_bytes: usize,
    pub max_label_bytes: usize,
    pub max_grounding_refs_per_node: usize,
    pub max_evidence_refs_per_edge: usize,
    pub max_envelope_evidence_ids: usize,
    pub max_structured_json_bytes: usize,
    pub max_human_text_bytes: usize,
    pub max_provenance_entries: usize,
}

impl Default for ScipLimits {
    fn default() -> Self {
        Self {
            max_hdc_dimension: 65_536,
            max_graph_nodes: 16_384,
            max_graph_edges: 65_536,
            max_identifier_bytes: 4 * 1024,
            max_label_bytes: 64 * 1024,
            max_grounding_refs_per_node: 4_096,
            max_evidence_refs_per_edge: 4_096,
            max_envelope_evidence_ids: 16_384,
            max_structured_json_bytes: 8 * 1024 * 1024,
            max_human_text_bytes: 2 * 1024 * 1024,
            max_provenance_entries: 4_096,
        }
    }
}

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

/// Canonical JSON encoding of the grounded graph, bound to its semantic hash.
///
/// `bytes` are not arbitrary JSON. They must be byte-for-byte equal to
/// `canonical_graph_bytes(decoded_graph)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredJsonPayload {
    pub bytes: Vec<u8>,
    pub semantic_hash: String,
}

impl StructuredJsonPayload {
    pub fn from_graph(graph: &GroundedConceptGraph) -> Result<Self, InterchangeError> {
        Self::from_graph_with_limits(graph, &ScipLimits::default())
    }

    pub fn from_graph_with_limits(
        graph: &GroundedConceptGraph,
        limits: &ScipLimits,
    ) -> Result<Self, InterchangeError> {
        let bytes = canonical_graph_bytes_with_limits(graph, limits)?;
        if bytes.len() > limits.max_structured_json_bytes {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "structured JSON is {} bytes; maximum is {}",
                bytes.len(),
                limits.max_structured_json_bytes
            )));
        }
        Ok(Self {
            semantic_hash: content_hash(&bytes),
            bytes,
        })
    }

    pub fn decode_graph(&self) -> Result<GroundedConceptGraph, InterchangeError> {
        self.decode_graph_with_limits(&ScipLimits::default())
    }

    pub fn decode_graph_with_limits(
        &self,
        limits: &ScipLimits,
    ) -> Result<GroundedConceptGraph, InterchangeError> {
        if self.bytes.len() > limits.max_structured_json_bytes {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "structured JSON is {} bytes; maximum is {}",
                self.bytes.len(),
                limits.max_structured_json_bytes
            )));
        }
        require_content_hash(&self.semantic_hash, "structured JSON semantic hash")?;
        let graph: GroundedConceptGraph = serde_json::from_slice(&self.bytes)?;
        let canonical = canonical_graph_bytes_with_limits(&graph, limits)?;
        if canonical != self.bytes {
            return Err(InterchangeError::InvalidStructuredJson(
                "payload is not canonical SCIP graph JSON".into(),
            ));
        }
        if content_hash(&canonical) != self.semantic_hash {
            return Err(InterchangeError::InvalidStructuredJson(
                "semantic hash does not match canonical graph bytes".into(),
            ));
        }
        Ok(graph)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticReference {
    pub semantic_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InterchangePayload {
    GroundedGraph(GroundedConceptGraph),
    Hdc(HdcPayload),
    StructuredJson(StructuredJsonPayload),
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

    pub fn from_structured_graph(
        graph: &GroundedConceptGraph,
        confidence: f32,
        provenance: Provenance,
    ) -> Result<Self, InterchangeError> {
        Self::new(
            SemanticProfile::structured_json(),
            InterchangePayload::StructuredJson(StructuredJsonPayload::from_graph(graph)?),
            confidence,
            provenance,
        )
    }

    pub fn computed_id(&self) -> Result<String, InterchangeError> {
        let mut canonical = self.clone();
        canonical.message_id.clear();
        canonical.evidence_ids.sort();
        canonical.provenance.feature_flags.sort();
        canonical.provenance.transformations.sort();
        if let InterchangePayload::GroundedGraph(graph) = &mut canonical.payload {
            *graph = canonicalize_graph(graph)?;
        }
        Ok(content_hash(&serde_json::to_vec(&canonical)?))
    }

    pub fn refresh_id(&mut self) -> Result<(), InterchangeError> {
        self.message_id = self.computed_id()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        self.validate_with_limits(&ScipLimits::default())
    }

    pub fn validate_with_limits(&self, limits: &ScipLimits) -> Result<(), InterchangeError> {
        self.validate_content_with_limits(limits)?;
        if self.computed_id()? != self.message_id {
            return Err(InterchangeError::InvalidIdentity);
        }
        Ok(())
    }

    fn validate_content(&self) -> Result<(), InterchangeError> {
        self.validate_content_with_limits(&ScipLimits::default())
    }

    fn validate_content_with_limits(&self, limits: &ScipLimits) -> Result<(), InterchangeError> {
        if self.protocol != SCIP_PROTOCOL_ID {
            return Err(InterchangeError::UnsupportedProtocol(self.protocol.clone()));
        }
        if self.version != SCIP_V1 {
            return Err(InterchangeError::UnsupportedVersion(self.version));
        }
        if self.profile.schema_id != GROUNDED_GRAPH_SCHEMA_V1 {
            return Err(InterchangeError::UnsupportedSchema(
                self.profile.schema_id.clone(),
            ));
        }
        if !valid_confidence(self.confidence) {
            return Err(InterchangeError::InvalidConfidence);
        }
        validate_profile(&self.profile, limits)?;
        validate_provenance(&self.provenance, limits)?;
        if let Some(parent_id) = &self.parent_id {
            require_content_hash(parent_id, "parent message id")?;
        }
        validate_unique_strings(
            &self.evidence_ids,
            limits.max_envelope_evidence_ids,
            limits.max_identifier_bytes,
            "envelope evidence ids",
        )?;

        match (
            &self.profile.representation,
            &self.profile.hdc,
            &self.payload,
        ) {
            (
                InterchangeRepresentation::GroundedGraph,
                None,
                InterchangePayload::GroundedGraph(graph),
            ) => validate_graph_with_limits(graph, limits)?,
            (InterchangeRepresentation::Hdc, Some(profile), InterchangePayload::Hdc(payload)) => {
                if payload.values.len() != profile.dimension
                    || payload.profile_fingerprint != profile.codebook_fingerprint
                    || payload.values.iter().any(|value| !value.is_finite())
                {
                    return Err(InterchangeError::InvalidHdcPayload);
                }
                require_content_hash(&payload.semantic_hash, "HDC semantic hash")?;
            }
            (
                InterchangeRepresentation::StructuredJson,
                None,
                InterchangePayload::StructuredJson(payload),
            ) => {
                payload.decode_graph_with_limits(limits)?;
            }
            (
                InterchangeRepresentation::HumanText,
                None,
                InterchangePayload::HumanText(text),
            ) => {
                if text.len() > limits.max_human_text_bytes {
                    return Err(InterchangeError::ResourceLimitExceeded(format!(
                        "human text is {} bytes; maximum is {}",
                        text.len(),
                        limits.max_human_text_bytes
                    )));
                }
            }
            (_, _, InterchangePayload::Reference(reference)) => {
                require_content_hash(&reference.semantic_hash, "semantic reference")?;
            }
            _ => return Err(InterchangeError::ProfilePayloadMismatch),
        }

        Ok(())
    }

    pub fn semantic_hash(&self) -> Result<Option<String>, InterchangeError> {
        match &self.payload {
            InterchangePayload::GroundedGraph(graph) => Ok(Some(graph_semantic_hash(graph)?)),
            InterchangePayload::Hdc(payload) => Ok(Some(payload.semantic_hash.clone())),
            InterchangePayload::StructuredJson(payload) => Ok(Some(payload.semantic_hash.clone())),
            InterchangePayload::Reference(reference) => Ok(Some(reference.semantic_hash.clone())),
            InterchangePayload::HumanText(_) => Ok(None),
        }
    }
}

pub fn validate_graph(graph: &GroundedConceptGraph) -> Result<(), InterchangeError> {
    validate_graph_with_limits(graph, &ScipLimits::default())
}

pub fn validate_graph_with_limits(
    graph: &GroundedConceptGraph,
    limits: &ScipLimits,
) -> Result<(), InterchangeError> {
    if graph.nodes.len() > limits.max_graph_nodes {
        return Err(InterchangeError::ResourceLimitExceeded(format!(
            "graph has {} nodes; maximum is {}",
            graph.nodes.len(),
            limits.max_graph_nodes
        )));
    }
    if graph.edges.len() > limits.max_graph_edges {
        return Err(InterchangeError::ResourceLimitExceeded(format!(
            "graph has {} edges; maximum is {}",
            graph.edges.len(),
            limits.max_graph_edges
        )));
    }

    let mut ids = BTreeSet::new();
    for node in &graph.nodes {
        validate_identifier(&node.id, limits, "node id")?;
        if !ids.insert(node.id.as_str()) {
            return Err(InterchangeError::InvalidGraph(format!(
                "duplicate node id {}",
                node.id
            )));
        }
        if let Some(label) = &node.label {
            if label.len() > limits.max_label_bytes {
                return Err(InterchangeError::ResourceLimitExceeded(format!(
                    "node label is {} bytes; maximum is {}",
                    label.len(),
                    limits.max_label_bytes
                )));
            }
        }
        if !valid_confidence(node.confidence) {
            return Err(InterchangeError::InvalidGraph(format!(
                "invalid confidence for node {}",
                node.id
            )));
        }
        validate_unique_strings(
            &node.grounded_by,
            limits.max_grounding_refs_per_node,
            limits.max_identifier_bytes,
            "node grounding references",
        )?;
    }

    for edge in &graph.edges {
        validate_identifier(&edge.source, limits, "edge source")?;
        validate_identifier(&edge.relation, limits, "edge relation")?;
        validate_identifier(&edge.target, limits, "edge target")?;
        if !ids.contains(edge.source.as_str()) || !ids.contains(edge.target.as_str()) {
            return Err(InterchangeError::InvalidGraph(format!(
                "edge {} -> {} has a dangling endpoint",
                edge.source, edge.target
            )));
        }
        if !valid_confidence(edge.confidence) {
            return Err(InterchangeError::InvalidGraph("invalid edge confidence".into()));
        }
        validate_unique_strings(
            &edge.evidence_ids,
            limits.max_evidence_refs_per_edge,
            limits.max_identifier_bytes,
            "edge evidence references",
        )?;
    }
    Ok(())
}

pub fn canonicalize_graph(
    graph: &GroundedConceptGraph,
) -> Result<GroundedConceptGraph, InterchangeError> {
    canonicalize_graph_with_limits(graph, &ScipLimits::default())
}

pub fn canonicalize_graph_with_limits(
    graph: &GroundedConceptGraph,
    limits: &ScipLimits,
) -> Result<GroundedConceptGraph, InterchangeError> {
    validate_graph_with_limits(graph, limits)?;
    let mut canonical = graph.clone();
    for node in &mut canonical.nodes {
        node.grounded_by.sort();
    }
    canonical.nodes.sort_by(|left, right| left.id.cmp(&right.id));
    for edge in &mut canonical.edges {
        edge.evidence_ids.sort();
    }
    canonical.edges.sort_by(|left, right| {
        (
            &left.source,
            &left.relation,
            &left.target,
            &left.evidence_ids,
            left.confidence.to_bits(),
        )
            .cmp(&(
                &right.source,
                &right.relation,
                &right.target,
                &right.evidence_ids,
                right.confidence.to_bits(),
            ))
    });
    Ok(canonical)
}

pub fn canonical_graph_bytes(graph: &GroundedConceptGraph) -> Result<Vec<u8>, InterchangeError> {
    canonical_graph_bytes_with_limits(graph, &ScipLimits::default())
}

pub fn canonical_graph_bytes_with_limits(
    graph: &GroundedConceptGraph,
    limits: &ScipLimits,
) -> Result<Vec<u8>, InterchangeError> {
    Ok(serde_json::to_vec(&canonicalize_graph_with_limits(
        graph, limits,
    )?)?)
}

pub fn graph_semantic_hash(graph: &GroundedConceptGraph) -> Result<String, InterchangeError> {
    Ok(content_hash(&canonical_graph_bytes(graph)?))
}

fn validate_profile(profile: &SemanticProfile, limits: &ScipLimits) -> Result<(), InterchangeError> {
    if profile.schema_id.len() > limits.max_identifier_bytes {
        return Err(InterchangeError::ResourceLimitExceeded(
            "schema id exceeds maximum identifier size".into(),
        ));
    }
    if let InterchangeRepresentation::Custom(value) = &profile.representation {
        validate_identifier(value, limits, "custom representation")?;
    }
    if let Some(hdc) = &profile.hdc {
        if hdc.dimension == 0 || hdc.dimension > limits.max_hdc_dimension {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "HDC dimension {} is outside 1..={} ",
                hdc.dimension, limits.max_hdc_dimension
            )));
        }
        validate_identifier(&hdc.algebra, limits, "HDC algebra")?;
        validate_identifier(&hdc.atom_derivation, limits, "HDC atom derivation")?;
        validate_identifier(&hdc.namespace, limits, "HDC namespace")?;
        require_content_hash(&hdc.codebook_fingerprint, "HDC codebook fingerprint")?;
    }
    Ok(())
}

fn validate_provenance(
    provenance: &Provenance,
    limits: &ScipLimits,
) -> Result<(), InterchangeError> {
    validate_identifier(&provenance.provider, limits, "provenance provider")?;
    if provenance.model_hash.trim().is_empty() {
        return Err(InterchangeError::InvalidProvenance);
    }
    if provenance.model_hash.len() > limits.max_identifier_bytes
        || provenance.provider_version.len() > limits.max_identifier_bytes
    {
        return Err(InterchangeError::ResourceLimitExceeded(
            "provenance field exceeds maximum identifier size".into(),
        ));
    }
    validate_unique_strings(
        &provenance.feature_flags,
        limits.max_provenance_entries,
        limits.max_identifier_bytes,
        "provenance feature flags",
    )?;
    validate_unique_strings(
        &provenance.transformations,
        limits.max_provenance_entries,
        limits.max_identifier_bytes,
        "provenance transformations",
    )?;
    Ok(())
}

fn validate_identifier(
    value: &str,
    limits: &ScipLimits,
    field: &str,
) -> Result<(), InterchangeError> {
    if value.trim().is_empty() {
        return Err(InterchangeError::InvalidGraph(format!("empty {field}")));
    }
    if value.len() > limits.max_identifier_bytes {
        return Err(InterchangeError::ResourceLimitExceeded(format!(
            "{field} is {} bytes; maximum is {}",
            value.len(),
            limits.max_identifier_bytes
        )));
    }
    Ok(())
}

fn validate_unique_strings(
    values: &[String],
    maximum_entries: usize,
    maximum_bytes: usize,
    field: &str,
) -> Result<(), InterchangeError> {
    if values.len() > maximum_entries {
        return Err(InterchangeError::ResourceLimitExceeded(format!(
            "{field} has {} entries; maximum is {maximum_entries}",
            values.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(InterchangeError::InvalidGraph(format!(
                "{field} contains an empty value"
            )));
        }
        if value.len() > maximum_bytes {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "{field} contains a value of {} bytes; maximum is {maximum_bytes}",
                value.len()
            )));
        }
        if !seen.insert(value.as_str()) {
            return Err(InterchangeError::InvalidGraph(format!(
                "{field} contains duplicate value {value}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn require_content_hash(value: &str, field: &str) -> Result<(), InterchangeError> {
    if value.len() != SCIP_CONTENT_HASH_HEX_LEN
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(InterchangeError::InvalidContentHash(field.into()));
    }
    Ok(())
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
    InvalidContentHash(String),
    InvalidStructuredJson(String),
    ResourceLimitExceeded(String),
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
            Self::InvalidContentHash(value) => write!(f, "invalid content hash: {value}"),
            Self::InvalidStructuredJson(value) => write!(f, "invalid structured JSON: {value}"),
            Self::ResourceLimitExceeded(value) => write!(f, "SCIP resource limit exceeded: {value}"),
            Self::ProfilePayloadMismatch => write!(f, "profile does not match payload"),
            Self::InvalidHdcPayload => write!(f, "invalid HDC payload"),
            Self::NegotiationFailed => write!(f, "no mutually supported SCIP representation"),
            Self::MissingSemanticReference(value) => {
                write!(f, "missing semantic reference: {value}")
            }
            Self::InvalidDelta(value) => write!(f, "invalid SCIP delta: {value}"),
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
    use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode};

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
            edges: vec![ConceptEdge {
                source: "a".into(),
                relation: "observes".into(),
                target: "b".into(),
                evidence_ids: vec!["ev-2".into(), "ev-1".into()],
                confidence: 0.7,
            }],
        }
    }

    #[test]
    fn graph_hash_ignores_graph_order() {
        let first = graph();
        let mut second = first.clone();
        second.nodes.reverse();
        second.nodes[0].grounded_by.reverse();
        second.edges[0].evidence_ids.reverse();
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

    #[test]
    fn dangling_edge_is_rejected() {
        let mut value = graph();
        value.edges[0].target = "missing".into();
        assert!(matches!(
            validate_graph(&value),
            Err(InterchangeError::InvalidGraph(_))
        ));
    }

    #[test]
    fn structured_json_is_canonical_and_hash_bound() {
        let graph = graph();
        let payload = StructuredJsonPayload::from_graph(&graph).unwrap();
        assert_eq!(payload.decode_graph().unwrap(), canonicalize_graph(&graph).unwrap());

        let mut tampered = payload;
        tampered.bytes.push(b' ');
        assert!(tampered.decode_graph().is_err());
    }

    #[test]
    fn envelope_id_ignores_nonsemantic_graph_order() {
        let first = CognitiveEnvelope::from_graph(graph(), 0.8, provenance()).unwrap();
        let mut reordered_graph = graph();
        reordered_graph.nodes.reverse();
        reordered_graph.edges[0].evidence_ids.reverse();
        let second = CognitiveEnvelope::from_graph(reordered_graph, 0.8, provenance()).unwrap();
        assert_eq!(first.message_id, second.message_id);
    }

    #[test]
    fn hdc_dimension_limit_is_enforced_before_acceptance() {
        let limits = ScipLimits {
            max_hdc_dimension: 8,
            ..Default::default()
        };
        let profile = HdcProfile {
            dimension: 16,
            algebra: "a".into(),
            atom_derivation: "d".into(),
            namespace: "n".into(),
            codebook_fingerprint: "0".repeat(64),
        };
        let envelope = CognitiveEnvelope {
            protocol: SCIP_PROTOCOL_ID.into(),
            version: SCIP_V1,
            message_id: String::new(),
            parent_id: None,
            profile: SemanticProfile::hdc(profile),
            payload: InterchangePayload::Hdc(HdcPayload {
                values: vec![0.0; 16],
                semantic_hash: "1".repeat(64),
                profile_fingerprint: "0".repeat(64),
            }),
            confidence: 0.8,
            provenance: provenance(),
            evidence_ids: vec![],
        };
        assert!(matches!(
            envelope.validate_with_limits(&limits),
            Err(InterchangeError::ResourceLimitExceeded(_))
        ));
    }
}
