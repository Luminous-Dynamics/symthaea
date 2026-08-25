// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use symthaea::language::{
    llm_backend::{GenerationParams, LLMBackend},
    llm_organ::{LLMOrgan, LLMOrganConfig, MessageRole},
};
use symthaea_communication::{ConceptKind, ConceptNode, GroundedConceptGraph, Provenance};
use symthaea_interlingua::{CognitiveEnvelope, LlmFallbackMode};
use symthaea_scip_llm_adapter::{MAX_SCIP_LLM_OUTPUT_BYTES, ScipLlmError, ScipLlmRequest};
use symthaea_scip_llm_transactional::execute_accounted_transactional;

enum Step {
    Output(String),
    Failure,
}

struct SequenceBackend {
    steps: Mutex<VecDeque<Step>>,
}

impl SequenceBackend {
    fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            steps: Mutex::new(steps.into_iter().collect()),
        }
    }
}

#[async_trait::async_trait]
impl LLMBackend for SequenceBackend {
    async fn generate(&self, _prompt: &str, _params: &GenerationParams) -> Result<String> {
        match self.steps.lock().unwrap().pop_front() {
            Some(Step::Output(text)) => Ok(text),
            Some(Step::Failure) => anyhow::bail!("operator-only sequence backend detail"),
            None => anyhow::bail!("test backend exhausted"),
        }
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "transaction-sequence-backend"
    }
}

fn request() -> ScipLlmRequest {
    let graph = GroundedConceptGraph {
        nodes: vec![ConceptNode {
            id: "sensor".into(),
            kind: ConceptKind::Object,
            label: Some("S17".into()),
            grounded_by: vec!["observation-17".into()],
            confidence: 0.9,
        }],
        edges: vec![],
    };
    let provenance = Provenance {
        provider: "nonzero-state-ratchet".into(),
        provider_version: "1".into(),
        model_hash: "test-model".into(),
        feature_flags: vec![],
        transformations: vec![],
    };
    let envelope = CognitiveEnvelope::from_graph(graph, 0.91, provenance).unwrap();
    ScipLlmRequest::compile(&envelope, None, LlmFallbackMode::FaithfulTranslation).unwrap()
}

#[derive(Debug, PartialEq)]
struct OrganSnapshot {
    queries_processed: u64,
    tokens_generated: u64,
    avg_generation_time_bits: u64,
    cache_hits: u64,
    errors: u64,
    history: Vec<(MessageRole, String, u64)>,
}

fn snapshot(organ: &LLMOrgan) -> OrganSnapshot {
    let stats = organ.stats();
    OrganSnapshot {
        queries_processed: stats.queries_processed,
        tokens_generated: stats.tokens_generated,
        avg_generation_time_bits: stats.avg_generation_time_ms.to_bits(),
        cache_hits: stats.cache_hits,
        errors: stats.errors,
        history: organ
            .conversation_history()
            .iter()
            .map(|message| (message.role, message.content.clone(), message.timestamp))
            .collect(),
    }
}

async fn seed_nonzero_state(organ: &mut LLMOrgan, request: &ScipLlmRequest) {
    let output = execute_accounted_transactional(request, organ)
        .await
        .expect("seed realization must be accepted");
    assert_eq!(output.text, "seed surface");
    assert_eq!(organ.stats().queries_processed, 1);
    assert!(organ.stats().tokens_generated > 0);
    assert_eq!(organ.conversation_history().len(), 2);
}

#[tokio::test]
async fn blank_rejection_preserves_preexisting_state_exactly() {
    let backend = Arc::new(SequenceBackend::new([
        Step::Output("seed surface".into()),
        Step::Output("   ".into()),
    ]));
    let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
    let request = request();

    seed_nonzero_state(&mut organ, &request).await;
    let before = snapshot(&organ);

    assert!(matches!(
        execute_accounted_transactional(&request, &mut organ).await,
        Err(ScipLlmError::EmptyOutput { .. })
    ));
    assert_eq!(snapshot(&organ), before);
}

#[tokio::test]
async fn oversized_rejection_preserves_preexisting_state_exactly() {
    let backend = Arc::new(SequenceBackend::new([
        Step::Output("seed surface".into()),
        Step::Output("x".repeat(MAX_SCIP_LLM_OUTPUT_BYTES + 1)),
    ]));
    let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
    let request = request();

    seed_nonzero_state(&mut organ, &request).await;
    let before = snapshot(&organ);

    assert!(matches!(
        execute_accounted_transactional(&request, &mut organ).await,
        Err(ScipLlmError::OutputTooLarge { .. })
    ));
    assert_eq!(snapshot(&organ), before);
}

#[tokio::test]
async fn backend_failure_from_nonzero_state_changes_only_error_counter() {
    let backend = Arc::new(SequenceBackend::new([
        Step::Output("seed surface".into()),
        Step::Failure,
    ]));
    let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
    let request = request();

    seed_nonzero_state(&mut organ, &request).await;
    let before = snapshot(&organ);

    let error = execute_accounted_transactional(&request, &mut organ)
        .await
        .expect_err("backend failure must be surfaced");
    assert!(matches!(error, ScipLlmError::BackendFailure { .. }));
    assert!(!error.to_string().contains("operator-only sequence"));

    let after = snapshot(&organ);
    assert_eq!(after.queries_processed, before.queries_processed);
    assert_eq!(after.tokens_generated, before.tokens_generated);
    assert_eq!(
        after.avg_generation_time_bits,
        before.avg_generation_time_bits
    );
    assert_eq!(after.cache_hits, before.cache_hits);
    assert_eq!(after.history, before.history);
    assert_eq!(after.errors, before.errors + 1);
}
