// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional accounting bridge for strict SCIP LLM realization.
//!
//! Phase B1 proves that SCIP realization fails closed, but deliberately bypasses
//! `LLMOrgan` accounting. Phase B2 adds protocol-agnostic strict backend execution
//! with normal accounting. This bridge composes those two contracts without
//! allowing a surface realization that SCIP rejects to remain in organ history,
//! statistics, or the embedding cache.
//!
//! The current implementation uses clone-and-commit staging. That is deliberately
//! correctness-first and is not claimed to be the final zero-copy runtime path.

#![forbid(unsafe_code)]

use symthaea::language::llm_organ::{
    LLMBackendExecutionError, LLMOrgan, LLMQuery, LLMQueryParams, QueryType,
};
use symthaea_interlingua::LlmFallbackMode;
use symthaea_scip_llm_adapter::{
    MAX_SCIP_LLM_OUTPUT_BYTES, SCIP_LLM_ADAPTER_PROFILE_V1, ScipLlmError, ScipLlmOutput,
    ScipLlmRequest, digest_surface_text,
};

const FAITHFUL_TRANSLATION_TEMPERATURE: f32 = 0.2;
const FAITHFUL_TRANSLATION_MAX_TOKENS: usize = 512;
const GROUNDED_REASONING_TEMPERATURE: f32 = 0.3;
const GROUNDED_REASONING_MAX_TOKENS: usize = 768;

/// Execute one already-compiled SCIP request with transactional organ accounting.
///
/// The backend call and ordinary `LLMOrgan` accounting occur on a cloned staging
/// organ. The staged state is committed only after the returned surface satisfies
/// the SCIP adapter's acceptance policy (nonblank and at most 1 MiB). Therefore a
/// model output rejected by SCIP cannot leak into conversation history, token
/// counters, latency averages, or the embedding cache.
///
/// Backend generation failures are different: Phase B2 deliberately counts them
/// as organ errors. On that failure path the staged organ is committed because its
/// only organ-level mutation is the error counter; no generated surface or history
/// entry was accepted.
///
/// This function never invokes `LLMOrgan::query_async`, so there is no simulation
/// fallback.
pub async fn execute_accounted_transactional(
    request: &ScipLlmRequest,
    organ: &mut LLMOrgan,
) -> Result<ScipLlmOutput, ScipLlmError> {
    let backend_name = organ
        .get_backend()
        .map(|backend| backend.name().to_owned())
        .ok_or(ScipLlmError::MissingBackend)?;

    let query = locked_query(request);
    let mut staged = organ.clone();

    let result = match staged.execute_backend_strict(&query).await {
        Ok(result) => result,
        Err(LLMBackendExecutionError::MissingBackend) => {
            return Err(ScipLlmError::MissingBackend);
        }
        Err(LLMBackendExecutionError::Generation { backend, .. }) => {
            // Strict root execution increments only the error counter on this
            // path. Commit that accounting while still returning a redacted SCIP
            // failure and never entering simulation.
            *organ = staged;
            return Err(ScipLlmError::BackendFailure { backend });
        }
    };

    let generation_time_ms = result.generation_time_ms;
    let text = result.text;

    if text.trim().is_empty() {
        return Err(ScipLlmError::EmptyOutput {
            backend: backend_name,
        });
    }
    if text.len() > MAX_SCIP_LLM_OUTPUT_BYTES {
        return Err(ScipLlmError::OutputTooLarge {
            backend: backend_name,
            bytes: text.len(),
            maximum: MAX_SCIP_LLM_OUTPUT_BYTES,
        });
    }

    let output = ScipLlmOutput {
        surface_digest: digest_surface_text(&text),
        text,
        adapter_profile: SCIP_LLM_ADAPTER_PROFILE_V1,
        request_digest: request.request_digest().to_owned(),
        source_message_id: request.source_message_id().to_owned(),
        source_semantic_hash: request.source_semantic_hash().to_owned(),
        source_confidence: request.source_confidence(),
        source_evidence_ids: request.source_evidence_ids().to_vec(),
        source_provenance: request.source_provenance().clone(),
        backend_name,
        mode: request.mode(),
        generation_time_ms,
    };

    *organ = staged;
    Ok(output)
}

/// Reconstruct the locked Phase B1 generation policy as an `LLMQuery` for the
/// protocol-agnostic Phase B2 strict executor.
///
/// These values intentionally mirror `symthaea-scip-llm-adapter` v1. The test
/// suite executes both paths against recording backends and requires their actual
/// `GenerationParams` to match, turning any future policy drift into a test
/// failure rather than a silent change in audited request semantics.
fn locked_query(request: &ScipLlmRequest) -> LLMQuery {
    let (query_type, temperature, max_length) = match request.mode() {
        LlmFallbackMode::FaithfulTranslation => (
            QueryType::Translation,
            FAITHFUL_TRANSLATION_TEMPERATURE,
            FAITHFUL_TRANSLATION_MAX_TOKENS,
        ),
        LlmFallbackMode::GroundedReasoning => (
            QueryType::Analysis,
            GROUNDED_REASONING_TEMPERATURE,
            GROUNDED_REASONING_MAX_TOKENS,
        ),
    };

    LLMQuery {
        query_type,
        content: request.content().to_owned(),
        context: Vec::new(),
        system_prompt: Some(request.system_prompt().to_owned()),
        params: Some(LLMQueryParams {
            temperature: Some(temperature),
            max_length: Some(max_length),
            stop_sequences: Vec::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::sync::{Arc, Mutex};
    use symthaea::language::{
        llm_backend::{GenerationParams, LLMBackend},
        llm_organ::LLMOrganConfig,
    };
    use symthaea_communication::{ConceptKind, ConceptNode, GroundedConceptGraph, Provenance};
    use symthaea_interlingua::CognitiveEnvelope;

    fn graph(label: &str) -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![ConceptNode {
                id: "sensor".into(),
                kind: ConceptKind::Object,
                label: Some(label.into()),
                grounded_by: vec!["observation-17".into()],
                confidence: 0.9,
            }],
            edges: vec![],
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            provider: "scip-transaction-test".into(),
            provider_version: "1".into(),
            model_hash: "test-model".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn request(mode: LlmFallbackMode) -> ScipLlmRequest {
        let envelope = CognitiveEnvelope::from_graph(graph("S17"), 0.91, provenance()).unwrap();
        ScipLlmRequest::compile(&envelope, None, mode).unwrap()
    }

    struct RecordingBackend {
        response: String,
        fail: bool,
        calls: Arc<Mutex<Vec<(String, GenerationParams)>>>,
    }

    impl RecordingBackend {
        fn success(
            response: impl Into<String>,
            calls: Arc<Mutex<Vec<(String, GenerationParams)>>>,
        ) -> Self {
            Self {
                response: response.into(),
                fail: false,
                calls,
            }
        }

        fn failure(calls: Arc<Mutex<Vec<(String, GenerationParams)>>>) -> Self {
            Self {
                response: String::new(),
                fail: true,
                calls,
            }
        }
    }

    #[async_trait::async_trait]
    impl LLMBackend for RecordingBackend {
        async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
            self.calls
                .lock()
                .unwrap()
                .push((prompt.to_owned(), params.clone()));
            if self.fail {
                anyhow::bail!("operator-only transactional backend detail");
            }
            Ok(self.response.clone())
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "transaction-test-backend"
        }
    }

    fn organ_with_backend(backend: Arc<dyn LLMBackend>) -> LLMOrgan {
        LLMOrgan::with_backend(LLMOrganConfig::default(), backend)
    }

    fn assert_generation_params_equal(left: &GenerationParams, right: &GenerationParams) {
        assert_eq!(left.temperature.to_bits(), right.temperature.to_bits());
        assert_eq!(left.max_tokens, right.max_tokens);
        assert_eq!(left.system_prompt, right.system_prompt);
        assert_eq!(
            left.consciousness_context.is_none(),
            right.consciousness_context.is_none()
        );
    }

    async fn assert_policy_matches_direct_adapter(mode: LlmFallbackMode) {
        let request = request(mode);

        let direct_calls = Arc::new(Mutex::new(Vec::new()));
        let direct_organ = organ_with_backend(Arc::new(RecordingBackend::success(
            "accepted surface",
            direct_calls.clone(),
        )));
        request.execute(&direct_organ).await.unwrap();

        let transactional_calls = Arc::new(Mutex::new(Vec::new()));
        let mut transactional_organ = organ_with_backend(Arc::new(RecordingBackend::success(
            "accepted surface",
            transactional_calls.clone(),
        )));
        execute_accounted_transactional(&request, &mut transactional_organ)
            .await
            .unwrap();

        let direct_calls = direct_calls.lock().unwrap();
        let transactional_calls = transactional_calls.lock().unwrap();
        assert_eq!(direct_calls.len(), 1);
        assert_eq!(transactional_calls.len(), 1);
        assert_eq!(direct_calls[0].0, transactional_calls[0].0);
        assert_generation_params_equal(&direct_calls[0].1, &transactional_calls[0].1);
    }

    #[tokio::test]
    async fn transactional_policy_matches_direct_adapter_for_both_modes() {
        assert_policy_matches_direct_adapter(LlmFallbackMode::FaithfulTranslation).await;
        assert_policy_matches_direct_adapter(LlmFallbackMode::GroundedReasoning).await;
    }

    #[tokio::test]
    async fn accepted_surface_commits_accounting_and_source_identity() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut organ = organ_with_backend(Arc::new(RecordingBackend::success(
            "Sensor S17.",
            calls,
        )));

        let output = execute_accounted_transactional(&request, &mut organ)
            .await
            .unwrap();

        assert_eq!(output.text, "Sensor S17.");
        assert_eq!(output.request_digest, request.request_digest());
        assert_eq!(output.source_message_id, request.source_message_id());
        assert_eq!(output.source_semantic_hash, request.source_semantic_hash());
        assert_eq!(output.source_confidence, request.source_confidence());
        assert_eq!(output.backend_name, "transaction-test-backend");
        assert_eq!(organ.stats().queries_processed, 1);
        assert_eq!(organ.stats().errors, 0);
        assert_eq!(organ.conversation_history().len(), 2);
        assert_eq!(organ.conversation_history()[1].content, "Sensor S17.");
    }

    #[tokio::test]
    async fn blank_surface_is_rejected_without_persistent_accounting() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut organ = organ_with_backend(Arc::new(RecordingBackend::success("   ", calls)));

        assert!(matches!(
            execute_accounted_transactional(&request, &mut organ).await,
            Err(ScipLlmError::EmptyOutput { .. })
        ));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().tokens_generated, 0);
        assert_eq!(organ.stats().cache_hits, 0);
        assert_eq!(organ.stats().errors, 0);
        assert!(organ.conversation_history().is_empty());
    }

    #[tokio::test]
    async fn oversized_surface_is_rejected_without_persistent_accounting() {
        let request = request(LlmFallbackMode::GroundedReasoning);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut organ = organ_with_backend(Arc::new(RecordingBackend::success(
            "x".repeat(MAX_SCIP_LLM_OUTPUT_BYTES + 1),
            calls,
        )));

        assert!(matches!(
            execute_accounted_transactional(&request, &mut organ).await,
            Err(ScipLlmError::OutputTooLarge { .. })
        ));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().tokens_generated, 0);
        assert_eq!(organ.stats().cache_hits, 0);
        assert_eq!(organ.stats().errors, 0);
        assert!(organ.conversation_history().is_empty());
    }

    #[tokio::test]
    async fn backend_failure_commits_only_root_error_accounting() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut organ = organ_with_backend(Arc::new(RecordingBackend::failure(calls.clone())));

        let error = execute_accounted_transactional(&request, &mut organ)
            .await
            .unwrap_err();
        assert!(matches!(error, ScipLlmError::BackendFailure { .. }));
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().tokens_generated, 0);
        assert_eq!(organ.stats().errors, 1);
        assert!(organ.conversation_history().is_empty());
        assert!(!error.to_string().contains("operator-only transactional"));
    }

    #[tokio::test]
    async fn missing_backend_is_state_preserving() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let mut organ = LLMOrgan::new(LLMOrganConfig::default());

        assert!(matches!(
            execute_accounted_transactional(&request, &mut organ).await,
            Err(ScipLlmError::MissingBackend)
        ));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().errors, 0);
        assert!(organ.conversation_history().is_empty());
    }
}
