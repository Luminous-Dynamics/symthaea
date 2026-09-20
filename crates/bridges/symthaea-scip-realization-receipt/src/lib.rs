// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Opaque evidence receipts for accepted SCIP language surfaces.
//!
//! This crate deliberately makes a narrow claim:
//!
//! ```text
//! backend returned bytes
//!     != surface accepted
//!     != semantic fidelity established
//!     != grounded truth
//! ```
//!
//! The upstream transactional bridge establishes the middle boundary: a backend
//! result is nonblank, within the configured size ceiling, and committed to
//! `LLMOrgan` accounting only after surface acceptance. This crate binds that
//! accepted result into a deterministic content receipt without upgrading it to
//! a semantic, epistemic, authentication, or execution-authority claim.
//!
//! A later semantic verifier must consume this receipt and produce a *different*
//! typed capability. It must not turn this receipt into "faithful" by mutating a
//! boolean or widening this type's claim scope.

#![forbid(unsafe_code)]

use std::fmt;

use blake3::Hasher;
use symthaea::language::llm_organ::LLMOrgan;
use symthaea_communication::Provenance;
use symthaea_interlingua::LlmFallbackMode;
use symthaea_scip_llm_adapter::{
    ScipLlmError, ScipLlmOutput, ScipLlmRequest, digest_surface_text,
};
use symthaea_scip_llm_transactional::execute_accounted_transactional;

/// Versioned profile for the v1 accepted-surface receipt.
pub const SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1: &str =
    "symthaea.scip-surface-realization-receipt/v1";

const RECEIPT_DOMAIN_V1: &[u8] = b"symthaea-scip-surface-realization-receipt-v1\0";
const EVIDENCE_SET_DOMAIN_V1: &[u8] = b"symthaea-scip-surface-evidence-set-v1\0";
const PROVENANCE_DOMAIN_V1: &[u8] = b"symthaea-scip-surface-provenance-v1\0";

/// The complete positive claim made by a v1 receipt.
///
/// There is intentionally no `SemanticFaithful`, `Grounded`, `Verified`, or
/// `Authorized` variant. Those claims require independent evidence and should
/// be represented by later types rather than by widening this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealizationClaimScope {
    SurfaceAcceptedOnly,
}

/// Content-addressed evidence that one SCIP surface crossed the transactional
/// acceptance boundary.
///
/// Fields are private and the type is not deserializable. A receipt can be
/// minted only by [`execute_with_surface_receipt`], which first executes the
/// upstream transactional boundary. The digest is a deterministic content
/// commitment, not a signature, timestamp, nonce, anti-replay token, or proof
/// that a named backend actually produced the bytes.
#[derive(Clone)]
pub struct ScipSurfaceRealizationReceiptV1 {
    receipt_digest: String,
    adapter_profile: String,
    request_digest: String,
    surface_digest: String,
    source_message_id: String,
    source_semantic_hash: String,
    source_confidence_bits: u32,
    source_evidence_digest: String,
    source_evidence_count: u64,
    source_provenance_digest: String,
    backend_name: String,
    mode: LlmFallbackMode,
    surface_bytes: u64,
}

impl fmt::Debug for ScipSurfaceRealizationReceiptV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScipSurfaceRealizationReceiptV1")
            .field("profile", &SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1)
            .field("claim_scope", &RealizationClaimScope::SurfaceAcceptedOnly)
            .field("mode", &self.mode)
            .field("surface_bytes", &self.surface_bytes)
            .field("source_evidence_count", &self.source_evidence_count)
            .finish()
    }
}

impl ScipSurfaceRealizationReceiptV1 {
    fn from_accepted_output(output: &ScipLlmOutput) -> Self {
        let surface_digest = digest_surface_text(&output.text);
        debug_assert_eq!(surface_digest, output.surface_digest);

        let source_evidence_digest = evidence_set_digest(&output.source_evidence_ids);
        let source_provenance_digest = provenance_digest(&output.source_provenance);
        let source_confidence_bits = output.source_confidence.to_bits();
        let source_evidence_count = output.source_evidence_ids.len() as u64;
        let surface_bytes = output.text.len() as u64;

        let receipt_digest = receipt_digest(
            output.adapter_profile,
            &output.request_digest,
            &surface_digest,
            &output.source_message_id,
            &output.source_semantic_hash,
            source_confidence_bits,
            &source_evidence_digest,
            source_evidence_count,
            &source_provenance_digest,
            &output.backend_name,
            output.mode,
            surface_bytes,
        );

        Self {
            receipt_digest,
            adapter_profile: output.adapter_profile.to_owned(),
            request_digest: output.request_digest.clone(),
            surface_digest,
            source_message_id: output.source_message_id.clone(),
            source_semantic_hash: output.source_semantic_hash.clone(),
            source_confidence_bits,
            source_evidence_digest,
            source_evidence_count,
            source_provenance_digest,
            backend_name: output.backend_name.clone(),
            mode: output.mode,
            surface_bytes,
        }
    }

    /// Deterministic BLAKE3 content identity for this accepted-surface record.
    pub fn receipt_digest(&self) -> &str {
        &self.receipt_digest
    }

    pub fn profile(&self) -> &'static str {
        SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1
    }

    pub fn claim_scope(&self) -> RealizationClaimScope {
        RealizationClaimScope::SurfaceAcceptedOnly
    }

    pub fn adapter_profile(&self) -> &str {
        &self.adapter_profile
    }

    pub fn request_digest(&self) -> &str {
        &self.request_digest
    }

    pub fn surface_digest(&self) -> &str {
        &self.surface_digest
    }

    pub fn source_message_id(&self) -> &str {
        &self.source_message_id
    }

    pub fn source_semantic_hash(&self) -> &str {
        &self.source_semantic_hash
    }

    pub fn source_confidence_bits(&self) -> u32 {
        self.source_confidence_bits
    }

    pub fn source_evidence_digest(&self) -> &str {
        &self.source_evidence_digest
    }

    pub fn source_evidence_count(&self) -> u64 {
        self.source_evidence_count
    }

    pub fn source_provenance_digest(&self) -> &str {
        &self.source_provenance_digest
    }

    /// Human-readable backend label retained from the upstream adapter.
    ///
    /// This value is bound into the receipt but is not cryptographic backend or
    /// model attestation.
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn mode(&self) -> LlmFallbackMode {
        self.mode
    }

    pub fn surface_bytes(&self) -> u64 {
        self.surface_bytes
    }

    /// V1 deliberately cannot establish semantic fidelity.
    pub fn semantic_fidelity_established(&self) -> bool {
        false
    }

    /// V1 binds a backend label but does not authenticate a provider, runtime,
    /// model deployment, or model weights.
    pub fn backend_authenticated(&self) -> bool {
        false
    }

    /// The v1 identity intentionally excludes generation latency and contains no
    /// nonce or trusted timestamp, so identical executions can share an ID.
    pub fn unique_execution_established(&self) -> bool {
        false
    }

    /// Recompute all receipt-bearing content from an output and compare it with
    /// this receipt.
    ///
    /// This is an internal-consistency check only. It does not authenticate an
    /// independently supplied `ScipLlmOutput`.
    pub fn matches_output(&self, output: &ScipLlmOutput) -> bool {
        let surface_digest = digest_surface_text(&output.text);
        let evidence_digest = evidence_set_digest(&output.source_evidence_ids);
        let provenance_digest = provenance_digest(&output.source_provenance);
        let evidence_count = output.source_evidence_ids.len() as u64;
        let surface_bytes = output.text.len() as u64;

        self.adapter_profile == output.adapter_profile
            && self.request_digest == output.request_digest
            && self.surface_digest == surface_digest
            && output.surface_digest == surface_digest
            && self.source_message_id == output.source_message_id
            && self.source_semantic_hash == output.source_semantic_hash
            && self.source_confidence_bits == output.source_confidence.to_bits()
            && self.source_evidence_digest == evidence_digest
            && self.source_evidence_count == evidence_count
            && self.source_provenance_digest == provenance_digest
            && self.backend_name == output.backend_name
            && self.mode == output.mode
            && self.surface_bytes == surface_bytes
            && self.receipt_digest
                == receipt_digest(
                    output.adapter_profile,
                    &output.request_digest,
                    &surface_digest,
                    &output.source_message_id,
                    &output.source_semantic_hash,
                    output.source_confidence.to_bits(),
                    &evidence_digest,
                    evidence_count,
                    &provenance_digest,
                    &output.backend_name,
                    output.mode,
                    surface_bytes,
                )
    }
}

/// Execute one compiled SCIP request transactionally and mint a receipt only
/// after the surface is accepted.
///
/// Backend absence/failure, blank output, and oversized output return the
/// upstream error and produce no receipt. The transactional bridge retains its
/// existing accounting semantics.
pub async fn execute_with_surface_receipt(
    request: &ScipLlmRequest,
    organ: &mut LLMOrgan,
) -> Result<(ScipLlmOutput, ScipSurfaceRealizationReceiptV1), ScipLlmError> {
    let output = execute_accounted_transactional(request, organ).await?;
    let receipt = ScipSurfaceRealizationReceiptV1::from_accepted_output(&output);
    debug_assert!(receipt.matches_output(&output));
    Ok((output, receipt))
}

fn evidence_set_digest(evidence_ids: &[String]) -> String {
    let mut canonical = evidence_ids.to_vec();
    canonical.sort();

    let mut hasher = Hasher::new();
    hasher.update(EVIDENCE_SET_DOMAIN_V1);
    update_u64(&mut hasher, canonical.len() as u64);
    for evidence_id in canonical {
        update_str(&mut hasher, &evidence_id);
    }
    hasher.finalize().to_hex().to_string()
}

fn provenance_digest(provenance: &Provenance) -> String {
    let mut feature_flags = provenance.feature_flags.clone();
    let mut transformations = provenance.transformations.clone();
    feature_flags.sort();
    transformations.sort();

    let mut hasher = Hasher::new();
    hasher.update(PROVENANCE_DOMAIN_V1);
    update_str(&mut hasher, &provenance.provider);
    update_str(&mut hasher, &provenance.provider_version);
    update_str(&mut hasher, &provenance.model_hash);
    update_u64(&mut hasher, feature_flags.len() as u64);
    for feature in feature_flags {
        update_str(&mut hasher, &feature);
    }
    update_u64(&mut hasher, transformations.len() as u64);
    for transformation in transformations {
        update_str(&mut hasher, &transformation);
    }
    hasher.finalize().to_hex().to_string()
}

#[allow(clippy::too_many_arguments)]
fn receipt_digest(
    adapter_profile: &str,
    request_digest: &str,
    surface_digest: &str,
    source_message_id: &str,
    source_semantic_hash: &str,
    source_confidence_bits: u32,
    source_evidence_digest: &str,
    source_evidence_count: u64,
    source_provenance_digest: &str,
    backend_name: &str,
    mode: LlmFallbackMode,
    surface_bytes: u64,
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(RECEIPT_DOMAIN_V1);
    update_str(&mut hasher, SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1);
    hasher.update(&[0]); // SurfaceAcceptedOnly
    update_str(&mut hasher, adapter_profile);
    update_str(&mut hasher, request_digest);
    update_str(&mut hasher, surface_digest);
    update_str(&mut hasher, source_message_id);
    update_str(&mut hasher, source_semantic_hash);
    hasher.update(&source_confidence_bits.to_be_bytes());
    update_str(&mut hasher, source_evidence_digest);
    update_u64(&mut hasher, source_evidence_count);
    update_str(&mut hasher, source_provenance_digest);
    update_str(&mut hasher, backend_name);
    hasher.update(&[mode_code(mode)]);
    update_u64(&mut hasher, surface_bytes);
    hasher.finalize().to_hex().to_string()
}

fn mode_code(mode: LlmFallbackMode) -> u8 {
    match mode {
        LlmFallbackMode::FaithfulTranslation => 0,
        LlmFallbackMode::GroundedReasoning => 1,
    }
}

fn update_str(hasher: &mut Hasher, value: &str) {
    update_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn update_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use std::sync::Arc;
    use symthaea::language::{
        llm_backend::{GenerationParams, LLMBackend},
        llm_organ::LLMOrganConfig,
    };
    use symthaea_communication::{ConceptKind, ConceptNode, GroundedConceptGraph};
    use symthaea_interlingua::CognitiveEnvelope;

    struct FixedBackend {
        name: String,
        response: String,
    }

    #[async_trait::async_trait]
    impl LLMBackend for FixedBackend {
        async fn generate(&self, _prompt: &str, _params: &GenerationParams) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            provider: "receipt-test-provider".into(),
            provider_version: "1".into(),
            model_hash: "test-model".into(),
            feature_flags: vec!["b".into(), "a".into()],
            transformations: vec!["z".into(), "x".into()],
        }
    }

    fn request(mode: LlmFallbackMode) -> ScipLlmRequest {
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
        let mut envelope = CognitiveEnvelope::from_graph(graph, 0.91, provenance()).unwrap();
        envelope.evidence_ids = vec!["evidence-b".into(), "evidence-a".into()];
        ScipLlmRequest::compile(&envelope, None, mode).unwrap()
    }

    fn organ(name: &str, response: &str) -> LLMOrgan {
        LLMOrgan::with_backend(
            LLMOrganConfig::default(),
            Arc::new(FixedBackend {
                name: name.into(),
                response: response.into(),
            }),
        )
    }

    #[tokio::test]
    async fn accepted_surface_mints_narrow_receipt_after_accounting_commit() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let mut organ = organ("receipt-test-backend", "Sensor S17 is observed.");

        let (output, receipt) = execute_with_surface_receipt(&request, &mut organ)
            .await
            .unwrap();

        assert_eq!(receipt.claim_scope(), RealizationClaimScope::SurfaceAcceptedOnly);
        assert!(!receipt.semantic_fidelity_established());
        assert!(!receipt.backend_authenticated());
        assert!(!receipt.unique_execution_established());
        assert!(receipt.matches_output(&output));
        assert_eq!(receipt.surface_bytes(), output.text.len() as u64);
        assert_eq!(organ.stats().queries_processed, 1);
        assert_eq!(organ.stats().errors, 0);
        assert_eq!(organ.conversation_history().len(), 2);
    }

    #[tokio::test]
    async fn blank_surface_mints_no_receipt_and_preserves_success_accounting_state() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let mut organ = organ("receipt-test-backend", "   ");

        assert!(matches!(
            execute_with_surface_receipt(&request, &mut organ).await,
            Err(ScipLlmError::EmptyOutput { .. })
        ));
        assert_eq!(organ.stats().queries_processed, 0);
        assert_eq!(organ.stats().tokens_generated, 0);
        assert_eq!(organ.stats().errors, 0);
        assert!(organ.conversation_history().is_empty());
    }

    #[tokio::test]
    async fn receipt_identity_is_stable_for_same_content_not_runtime_latency() {
        let request = request(LlmFallbackMode::GroundedReasoning);
        let mut first = organ("receipt-test-backend", "Grounded answer.");
        let mut second = organ("receipt-test-backend", "Grounded answer.");

        let (first_output, first_receipt) =
            execute_with_surface_receipt(&request, &mut first).await.unwrap();
        let (second_output, second_receipt) =
            execute_with_surface_receipt(&request, &mut second).await.unwrap();

        assert_eq!(first_receipt.receipt_digest(), second_receipt.receipt_digest());
        assert!(first_receipt.matches_output(&first_output));
        assert!(second_receipt.matches_output(&second_output));
    }

    #[tokio::test]
    async fn receipt_identity_changes_with_surface_or_backend_label() {
        let request = request(LlmFallbackMode::FaithfulTranslation);

        let mut surface_a = organ("backend-a", "Alpha");
        let mut surface_b = organ("backend-a", "Beta");
        let mut backend_b = organ("backend-b", "Alpha");

        let (_, receipt_a) = execute_with_surface_receipt(&request, &mut surface_a)
            .await
            .unwrap();
        let (_, receipt_b) = execute_with_surface_receipt(&request, &mut surface_b)
            .await
            .unwrap();
        let (_, receipt_backend_b) = execute_with_surface_receipt(&request, &mut backend_b)
            .await
            .unwrap();

        assert_ne!(receipt_a.receipt_digest(), receipt_b.receipt_digest());
        assert_ne!(
            receipt_a.receipt_digest(),
            receipt_backend_b.receipt_digest()
        );
    }

    #[tokio::test]
    async fn receipt_detects_post_acceptance_output_mutation() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let mut organ = organ("receipt-test-backend", "Original surface.");

        let (mut output, receipt) = execute_with_surface_receipt(&request, &mut organ)
            .await
            .unwrap();
        assert!(receipt.matches_output(&output));

        output.text = "Mutated surface.".into();
        assert!(!receipt.matches_output(&output));
    }

    #[tokio::test]
    async fn debug_view_redacts_bound_identifiers_and_backend_label() {
        let request = request(LlmFallbackMode::FaithfulTranslation);
        let mut organ = organ("sensitive-backend-label", "Sensitive surface.");

        let (output, receipt) = execute_with_surface_receipt(&request, &mut organ)
            .await
            .unwrap();
        let debug = format!("{receipt:?}");

        assert!(!debug.contains(receipt.receipt_digest()));
        assert!(!debug.contains(receipt.request_digest()));
        assert!(!debug.contains(receipt.surface_digest()));
        assert!(!debug.contains(receipt.source_semantic_hash()));
        assert!(!debug.contains("sensitive-backend-label"));
        assert!(!debug.contains(output.text.as_str()));
        assert!(debug.contains("SurfaceAcceptedOnly"));
    }
}
