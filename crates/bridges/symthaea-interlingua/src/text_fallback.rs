// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compatibility adapter for LLM APIs that currently accept text only.
//!
//! This is a fallback, not SCIP's preferred machine-to-machine representation.
//! It serializes the grounded semantic graph compactly while preserving the
//! envelope's confidence, evidence references, provenance and semantic hash.

use crate::{CognitiveEnvelope, InterchangeError, InterchangePayload, graph_semantic_hash};
use serde::{Deserialize, Serialize};
use symthaea_communication::GroundedConceptGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmFallbackMode {
    GroundedReasoning,
    FaithfulTranslation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmFallbackPacket {
    pub system_prompt: String,
    pub content: String,
    pub semantic_hash: String,
}

pub struct LlmTextFallback;

impl LlmTextFallback {
    pub fn compile(
        envelope: &CognitiveEnvelope,
        resolved_graph: Option<&GroundedConceptGraph>,
        mode: LlmFallbackMode,
    ) -> Result<LlmFallbackPacket, InterchangeError> {
        envelope.validate()?;
        let graph = resolve_graph(envelope, resolved_graph)?;
        let semantic_hash = graph_semantic_hash(graph)?;

        let payload = serde_json::json!({
            "scip": {
                "protocol": envelope.protocol,
                "version": {
                    "major": envelope.version.major,
                    "minor": envelope.version.minor
                },
                "message_id": envelope.message_id,
                "semantic_hash": semantic_hash,
                "confidence": envelope.confidence,
                "evidence_ids": envelope.evidence_ids,
                "provenance": envelope.provenance,
            },
            "grounded_graph": graph,
        });

        Ok(LlmFallbackPacket {
            system_prompt: system_prompt(mode).to_string(),
            content: serde_json::to_string(&payload)?,
            semantic_hash,
        })
    }
}

fn resolve_graph<'a>(
    envelope: &'a CognitiveEnvelope,
    resolved_graph: Option<&'a GroundedConceptGraph>,
) -> Result<&'a GroundedConceptGraph, InterchangeError> {
    match &envelope.payload {
        InterchangePayload::GroundedGraph(graph) => Ok(graph),
        InterchangePayload::Hdc(payload) => {
            let graph = resolved_graph.ok_or_else(|| {
                InterchangeError::MissingSemanticReference(payload.semantic_hash.clone())
            })?;
            let actual = graph_semantic_hash(graph)?;
            if actual != payload.semantic_hash {
                return Err(InterchangeError::MissingSemanticReference(
                    payload.semantic_hash.clone(),
                ));
            }
            Ok(graph)
        }
        InterchangePayload::Reference(reference) => {
            let graph = resolved_graph.ok_or_else(|| {
                InterchangeError::MissingSemanticReference(reference.semantic_hash.clone())
            })?;
            let actual = graph_semantic_hash(graph)?;
            if actual != reference.semantic_hash {
                return Err(InterchangeError::MissingSemanticReference(
                    reference.semantic_hash.clone(),
                ));
            }
            Ok(graph)
        }
        InterchangePayload::StructuredJson(_) | InterchangePayload::HumanText(_) => {
            Err(InterchangeError::MissingSemanticReference(
                "text fallback requires a grounded concept graph".into(),
            ))
        }
    }
}

fn system_prompt(mode: LlmFallbackMode) -> &'static str {
    match mode {
        LlmFallbackMode::GroundedReasoning => {
            "You are a SCIP cognitive peer. Treat GROUNDED_GRAPH as supplied evidence, not prose. \
             Preserve confidence, provenance, evidence references, and uncertainty. You may reason \
             from the graph, but clearly distinguish new inference from supplied grounded facts. \
             Do not invent missing grounding."
        }
        LlmFallbackMode::FaithfulTranslation => {
            "You are a SCIP translation adapter. Convert the supplied GROUNDED_GRAPH into natural \
             language without adding facts, causes, certainty, or conclusions. Preserve confidence, \
             provenance, evidence references, and uncertainty. If grounding is absent, say so."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GroundedHdcCodec;
    use symthaea_communication::{ConceptKind, ConceptNode, Provenance};

    fn graph() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![ConceptNode {
                id: "sensor".into(),
                kind: ConceptKind::Object,
                label: Some("S17".into()),
                grounded_by: vec!["observation-17".into()],
                confidence: 0.9,
            }],
            edges: vec![],
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            provider: "test".into(),
            provider_version: "1".into(),
            model_hash: "model".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    #[test]
    fn direct_graph_compiles_without_external_resolution() {
        let envelope = CognitiveEnvelope::from_graph(graph(), 0.9, provenance()).unwrap();
        let packet =
            LlmTextFallback::compile(&envelope, None, LlmFallbackMode::GroundedReasoning).unwrap();
        assert!(packet.content.contains("\"grounded_graph\""));
        assert!(packet.content.contains("\"S17\""));
    }

    #[test]
    fn hdc_payload_requires_matching_grounded_graph() {
        let codec = GroundedHdcCodec::new(1024, "test");
        let graph = graph();
        let envelope = codec
            .envelope_from_graph(&graph, 0.9, provenance())
            .unwrap();

        assert!(
            LlmTextFallback::compile(&envelope, None, LlmFallbackMode::GroundedReasoning).is_err()
        );

        let packet =
            LlmTextFallback::compile(&envelope, Some(&graph), LlmFallbackMode::GroundedReasoning)
                .unwrap();
        assert_eq!(packet.semantic_hash, graph_semantic_hash(&graph).unwrap());
    }
}
