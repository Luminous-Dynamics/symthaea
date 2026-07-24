//! Common ingress and provider adapter boundaries.

use crate::human::PreservedText;
use crate::{
    CapabilityLevel, CommunicationError, CommunicationResult, ConceptKind, ConceptNode,
    GroundedConceptGraph, InteractionContext, Provenance, RepresentationFamily, SignalObservation,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CommunicationIngress {
    Observation(SignalObservation),
    HumanText(PreservedText),
    HdcVector(Vec<f32>),
    MultilingualEmbedding {
        provider: String,
        values: Vec<f32>,
    },
    BrocaThought {
        channels: Vec<f32>,
        labels: Vec<String>,
    },
    SensorimotorState(Vec<f32>),
}

pub trait IngressAdapter {
    fn family(&self) -> RepresentationFamily;
    fn ground(
        &self,
        input: &CommunicationIngress,
        context: &InteractionContext,
    ) -> Result<CommunicationResult<GroundedConceptGraph>, CommunicationError>;
}

pub trait GenerationProvider {
    fn provider_id(&self) -> &str;
    fn generate(
        &mut self,
        graph: &GroundedConceptGraph,
        target_language: &str,
    ) -> Result<CommunicationResult<PreservedText>, CommunicationError>;
}

/// Lossless text ingress. Labels aid human rendering but grounding identifiers
/// come from source spans, never from an assumed universal vocabulary.
pub struct HumanTextGraphAdapter {
    pub provenance: Provenance,
}

impl IngressAdapter for HumanTextGraphAdapter {
    fn family(&self) -> RepresentationFamily {
        RepresentationFamily::HumanText
    }

    fn ground(
        &self,
        input: &CommunicationIngress,
        _context: &InteractionContext,
    ) -> Result<CommunicationResult<GroundedConceptGraph>, CommunicationError> {
        let CommunicationIngress::HumanText(text) = input else {
            return Err(CommunicationError::UnsupportedModality);
        };
        text.validate()
            .map_err(CommunicationError::InvalidObservation)?;
        let nodes = text
            .spans
            .iter()
            .enumerate()
            .map(|(index, span)| ConceptNode {
                id: format!("span-{index}"),
                kind: ConceptKind::Unknown,
                label: Some(span.original.clone()),
                grounded_by: vec![format!("text:{index}")],
                confidence: span.confidence,
            })
            .collect();
        Ok(CommunicationResult {
            value: GroundedConceptGraph {
                nodes,
                edges: vec![],
            },
            capability: CapabilityLevel::Unit,
            confidence: 1.0,
            alternatives: vec![],
            provenance: self.provenance.clone(),
            evidence: vec![],
        })
    }
}

/// HDC and embedding vectors remain opaque evidence until context grounds them.
pub struct OpaqueVectorAdapter {
    pub family: RepresentationFamily,
    pub provenance: Provenance,
}

impl IngressAdapter for OpaqueVectorAdapter {
    fn family(&self) -> RepresentationFamily {
        self.family.clone()
    }

    fn ground(
        &self,
        input: &CommunicationIngress,
        _context: &InteractionContext,
    ) -> Result<CommunicationResult<GroundedConceptGraph>, CommunicationError> {
        let values = match input {
            CommunicationIngress::HdcVector(values)
            | CommunicationIngress::SensorimotorState(values)
            | CommunicationIngress::MultilingualEmbedding { values, .. } => values,
            CommunicationIngress::BrocaThought { channels, .. } => channels,
            _ => return Err(CommunicationError::UnsupportedModality),
        };
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(CommunicationError::InvalidObservation(
                "vector is empty or non-finite".into(),
            ));
        }
        Ok(CommunicationResult {
            value: GroundedConceptGraph {
                nodes: vec![ConceptNode {
                    id: crate::content_hash(bytemuck_free_f32_bytes(values).as_slice()),
                    kind: ConceptKind::Unknown,
                    label: None,
                    grounded_by: vec![],
                    confidence: 0.0,
                }],
                edges: vec![],
            },
            capability: CapabilityLevel::Unit,
            confidence: 0.0,
            alternatives: vec![],
            provenance: self.provenance.clone(),
            evidence: vec![],
        })
    }
}

fn bytemuck_free_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human::{PreservedSpan, PreservedText};

    fn stub_provenance() -> Provenance {
        Provenance {
            provider: "test".into(),
            provider_version: "0".into(),
            model_hash: "mhash".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn make_span(original: &str, confidence: f32) -> PreservedSpan {
        PreservedSpan {
            original: original.into(),
            normalized: None,
            timing: None,
            language: Some("en".into()),
            script: None,
            confidence,
            alternatives: vec![],
            named_entity: None,
            prosody: None,
        }
    }

    #[test]
    fn human_text_adapter_rejects_observation_ingress() {
        let mut obs = crate::SignalObservation {
            id: String::new(),
            modality: crate::Modality::Text {
                language: None,
                script: None,
            },
            samples: vec![],
            features: Default::default(),
            original_text: Some("hello".into()),
            normalized_text: None,
            uncertain_spans: vec![],
            timing: crate::TimeSpan {
                start_s: 0.0,
                end_s: 0.0,
            },
            location: None,
            calibration: crate::SensorCalibration::default(),
            source_identity: None,
            environment: Default::default(),
        };
        obs.refresh_id().unwrap();
        let adapter = HumanTextGraphAdapter {
            provenance: stub_provenance(),
        };
        assert!(matches!(
            adapter.ground(
                &CommunicationIngress::Observation(obs),
                &InteractionContext::default()
            ),
            Err(CommunicationError::UnsupportedModality)
        ));
    }

    #[test]
    fn human_text_adapter_grounds_spans() {
        let text = PreservedText {
            original: "Hello world".into(),
            normalized: None,
            primary_language: Some("en".into()),
            script: None,
            spans: vec![make_span("Hello", 0.9), make_span("world", 0.8)],
        };
        let adapter = HumanTextGraphAdapter {
            provenance: stub_provenance(),
        };
        let result = adapter
            .ground(
                &CommunicationIngress::HumanText(text),
                &InteractionContext::default(),
            )
            .unwrap();
        assert_eq!(result.value.nodes.len(), 2);
        assert_eq!(result.value.nodes[0].label.as_deref(), Some("Hello"));
        assert_eq!(result.capability, CapabilityLevel::Unit);
    }

    #[test]
    fn opaque_vector_adapter_rejects_nan() {
        let adapter = OpaqueVectorAdapter {
            family: RepresentationFamily::Hdc,
            provenance: stub_provenance(),
        };
        assert!(matches!(
            adapter.ground(
                &CommunicationIngress::HdcVector(vec![1.0, f32::NAN, 0.5]),
                &InteractionContext::default()
            ),
            Err(CommunicationError::InvalidObservation(_))
        ));
    }

    #[test]
    fn opaque_vector_adapter_rejects_empty() {
        let adapter = OpaqueVectorAdapter {
            family: RepresentationFamily::MultilingualEmbedding,
            provenance: stub_provenance(),
        };
        assert!(matches!(
            adapter.ground(
                &CommunicationIngress::HdcVector(vec![]),
                &InteractionContext::default()
            ),
            Err(CommunicationError::InvalidObservation(_))
        ));
    }

    #[test]
    fn opaque_vector_adapter_produces_single_node_with_hash_id() {
        let adapter = OpaqueVectorAdapter {
            family: RepresentationFamily::Hdc,
            provenance: stub_provenance(),
        };
        let result = adapter
            .ground(
                &CommunicationIngress::HdcVector(vec![0.1, 0.2, 0.3]),
                &InteractionContext::default(),
            )
            .unwrap();
        assert_eq!(result.value.nodes.len(), 1);
        // ID must be a non-empty hex string derived from the vector content.
        assert!(!result.value.nodes[0].id.is_empty());
    }
}
