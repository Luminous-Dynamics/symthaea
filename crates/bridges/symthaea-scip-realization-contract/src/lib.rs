// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grounded semantic-verification obligations for accepted SCIP language surfaces.
//!
//! This crate deliberately stops before semantic judgment:
//!
//! ```text
//! accepted surface receipt
//!     + exact grounded SCIP source
//!         -> semantic verification contract
//!         != semantic fidelity established
//!         != grounded truth
//! ```
//!
//! The contract makes the verification surface explicit and content-addressed.
//! A later verifier must resolve every required semantic dimension using
//! independently qualified evidence. No scalar similarity score, LLM-as-judge
//! response, embedding cosine, or caller-provided boolean can turn this contract
//! into a positive semantic capability.

#![forbid(unsafe_code)]

use std::fmt;

use blake3::Hasher;
use symthaea_communication::GroundedConceptGraph;
use symthaea_interlingua::{InterchangeError, LlmFallbackMode, graph_semantic_hash};
use symthaea_scip_realization_receipt::{
    RealizationClaimScope, SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1,
    ScipSurfaceRealizationReceiptV1,
};

/// Versioned semantic-verification contract profile.
pub const SCIP_SEMANTIC_REALIZATION_CONTRACT_PROFILE_V1: &str =
    "symthaea.scip-semantic-realization-contract/v1";

const CONTRACT_DOMAIN_V1: &[u8] = b"symthaea-scip-semantic-realization-contract-v1\0";
const DIMENSIONS_DOMAIN_V1: &[u8] = b"symthaea-scip-semantic-realization-dimensions-v1\0";

/// The complete positive claim made by this type.
///
/// It says only that the accepted surface has been bound to the exact grounded
/// source and to a fixed semantic verification checklist. It does not say that
/// any checklist item has passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticContractClaimScope {
    VerificationRequired,
}

/// Independent dimensions that a later semantic verifier must resolve.
///
/// The ordering and numeric codes are protocol-stable in v1 and are committed
/// into every contract identity. A later profile may add or refine dimensions,
/// but must not silently reinterpret these codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SemanticVerificationDimension {
    EntityReference = 0,
    RelationDirection = 1,
    NumericValueAndUnit = 2,
    PolarityAndNegation = 3,
    QuantifierAndCardinality = 4,
    TemporalScope = 5,
    EpistemicModality = 6,
    AttributionAndSource = 7,
    CausalStrength = 8,
    UnsupportedAdditions = 9,
    RequiredDetailCoverage = 10,
}

impl SemanticVerificationDimension {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::EntityReference => "entity-reference",
            Self::RelationDirection => "relation-direction",
            Self::NumericValueAndUnit => "numeric-value-and-unit",
            Self::PolarityAndNegation => "polarity-and-negation",
            Self::QuantifierAndCardinality => "quantifier-and-cardinality",
            Self::TemporalScope => "temporal-scope",
            Self::EpistemicModality => "epistemic-modality",
            Self::AttributionAndSource => "attribution-and-source",
            Self::CausalStrength => "causal-strength",
            Self::UnsupportedAdditions => "unsupported-additions",
            Self::RequiredDetailCoverage => "required-detail-coverage",
        }
    }
}

pub const REQUIRED_SEMANTIC_DIMENSIONS_V1: [SemanticVerificationDimension; 11] = [
    SemanticVerificationDimension::EntityReference,
    SemanticVerificationDimension::RelationDirection,
    SemanticVerificationDimension::NumericValueAndUnit,
    SemanticVerificationDimension::PolarityAndNegation,
    SemanticVerificationDimension::QuantifierAndCardinality,
    SemanticVerificationDimension::TemporalScope,
    SemanticVerificationDimension::EpistemicModality,
    SemanticVerificationDimension::AttributionAndSource,
    SemanticVerificationDimension::CausalStrength,
    SemanticVerificationDimension::UnsupportedAdditions,
    SemanticVerificationDimension::RequiredDetailCoverage,
];

/// Content-addressed obligation contract for one accepted surface and one exact
/// grounded source graph.
///
/// Fields are private. Construction validates that the supplied grounded graph
/// hashes to the exact `source_semantic_hash` already bound by the surface
/// receipt. The contract cannot be deserialized into existence and exposes no
/// method that can mark semantic fidelity as established.
#[derive(Clone)]
pub struct SemanticRealizationContractV1 {
    contract_digest: String,
    surface_receipt_digest: String,
    source_semantic_hash: String,
    request_digest: String,
    surface_digest: String,
    mode: LlmFallbackMode,
    source_graph_nodes: u64,
    source_graph_edges: u64,
    dimensions_digest: String,
}

impl fmt::Debug for SemanticRealizationContractV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticRealizationContractV1")
            .field("profile", &SCIP_SEMANTIC_REALIZATION_CONTRACT_PROFILE_V1)
            .field("claim_scope", &SemanticContractClaimScope::VerificationRequired)
            .field("mode", &self.mode)
            .field("source_graph_nodes", &self.source_graph_nodes)
            .field("source_graph_edges", &self.source_graph_edges)
            .field("required_dimensions", &REQUIRED_SEMANTIC_DIMENSIONS_V1.len())
            .finish()
    }
}

impl SemanticRealizationContractV1 {
    /// Bind one accepted surface receipt to the exact grounded source graph.
    pub fn bind_grounded_source(
        receipt: &ScipSurfaceRealizationReceiptV1,
        source_graph: &GroundedConceptGraph,
    ) -> Result<Self, SemanticContractError> {
        if receipt.claim_scope() != RealizationClaimScope::SurfaceAcceptedOnly {
            return Err(SemanticContractError::UnexpectedReceiptScope);
        }

        let source_semantic_hash = graph_semantic_hash(source_graph)?;
        if source_semantic_hash != receipt.source_semantic_hash() {
            return Err(SemanticContractError::SourceSemanticHashMismatch);
        }

        let source_graph_nodes = u64::try_from(source_graph.nodes.len())
            .map_err(|_| SemanticContractError::GraphSizeOverflow)?;
        let source_graph_edges = u64::try_from(source_graph.edges.len())
            .map_err(|_| SemanticContractError::GraphSizeOverflow)?;
        let dimensions_digest = dimensions_digest_v1();
        let contract_digest = contract_digest_v1(
            receipt.receipt_digest(),
            &source_semantic_hash,
            receipt.request_digest(),
            receipt.surface_digest(),
            receipt.mode(),
            source_graph_nodes,
            source_graph_edges,
            &dimensions_digest,
        );

        Ok(Self {
            contract_digest,
            surface_receipt_digest: receipt.receipt_digest().to_owned(),
            source_semantic_hash,
            request_digest: receipt.request_digest().to_owned(),
            surface_digest: receipt.surface_digest().to_owned(),
            mode: receipt.mode(),
            source_graph_nodes,
            source_graph_edges,
            dimensions_digest,
        })
    }

    pub fn profile(&self) -> &'static str {
        SCIP_SEMANTIC_REALIZATION_CONTRACT_PROFILE_V1
    }

    pub fn claim_scope(&self) -> SemanticContractClaimScope {
        SemanticContractClaimScope::VerificationRequired
    }

    pub fn contract_digest(&self) -> &str {
        &self.contract_digest
    }

    pub fn surface_receipt_digest(&self) -> &str {
        &self.surface_receipt_digest
    }

    pub fn source_semantic_hash(&self) -> &str {
        &self.source_semantic_hash
    }

    pub fn request_digest(&self) -> &str {
        &self.request_digest
    }

    pub fn surface_digest(&self) -> &str {
        &self.surface_digest
    }

    pub fn mode(&self) -> LlmFallbackMode {
        self.mode
    }

    pub fn source_graph_nodes(&self) -> u64 {
        self.source_graph_nodes
    }

    pub fn source_graph_edges(&self) -> u64 {
        self.source_graph_edges
    }

    pub fn dimensions_digest(&self) -> &str {
        &self.dimensions_digest
    }

    pub fn required_dimensions(&self) -> &'static [SemanticVerificationDimension] {
        &REQUIRED_SEMANTIC_DIMENSIONS_V1
    }

    /// This contract cannot establish semantic fidelity by construction.
    pub fn semantic_fidelity_established(&self) -> bool {
        false
    }

    /// V1 deliberately defines no universal scalar semantic-quality score.
    pub fn universal_scalar_score_defined(&self) -> bool {
        false
    }

    /// Validate that this contract still corresponds to the supplied receipt
    /// and exact grounded source.
    pub fn matches(
        &self,
        receipt: &ScipSurfaceRealizationReceiptV1,
        source_graph: &GroundedConceptGraph,
    ) -> Result<bool, SemanticContractError> {
        let candidate = Self::bind_grounded_source(receipt, source_graph)?;
        Ok(self.contract_digest == candidate.contract_digest
            && self.surface_receipt_digest == candidate.surface_receipt_digest
            && self.source_semantic_hash == candidate.source_semantic_hash
            && self.request_digest == candidate.request_digest
            && self.surface_digest == candidate.surface_digest
            && self.mode == candidate.mode
            && self.source_graph_nodes == candidate.source_graph_nodes
            && self.source_graph_edges == candidate.source_graph_edges
            && self.dimensions_digest == candidate.dimensions_digest)
    }
}

#[derive(Debug)]
pub enum SemanticContractError {
    Interchange(String),
    UnexpectedReceiptScope,
    SourceSemanticHashMismatch,
    GraphSizeOverflow,
}

impl fmt::Display for SemanticContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interchange(_) => write!(f, "grounded SCIP source validation failed"),
            Self::UnexpectedReceiptScope => write!(f, "unexpected surface receipt claim scope"),
            Self::SourceSemanticHashMismatch => {
                write!(f, "grounded source does not match the accepted surface receipt")
            }
            Self::GraphSizeOverflow => write!(f, "grounded graph size cannot be represented in v1"),
        }
    }
}

impl std::error::Error for SemanticContractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<InterchangeError> for SemanticContractError {
    fn from(value: InterchangeError) -> Self {
        Self::Interchange(value.to_string())
    }
}

fn dimensions_digest_v1() -> String {
    let mut hasher = Hasher::new();
    hasher.update(DIMENSIONS_DOMAIN_V1);
    update_u64(&mut hasher, REQUIRED_SEMANTIC_DIMENSIONS_V1.len() as u64);
    for dimension in REQUIRED_SEMANTIC_DIMENSIONS_V1 {
        hasher.update(&[dimension.code()]);
        update_str(&mut hasher, dimension.name());
    }
    hasher.finalize().to_hex().to_string()
}

#[allow(clippy::too_many_arguments)]
fn contract_digest_v1(
    surface_receipt_digest: &str,
    source_semantic_hash: &str,
    request_digest: &str,
    surface_digest: &str,
    mode: LlmFallbackMode,
    source_graph_nodes: u64,
    source_graph_edges: u64,
    dimensions_digest: &str,
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(CONTRACT_DOMAIN_V1);
    update_str(&mut hasher, SCIP_SEMANTIC_REALIZATION_CONTRACT_PROFILE_V1);
    hasher.update(&[0]); // VerificationRequired
    update_str(&mut hasher, SCIP_SURFACE_REALIZATION_RECEIPT_PROFILE_V1);
    update_str(&mut hasher, surface_receipt_digest);
    update_str(&mut hasher, source_semantic_hash);
    update_str(&mut hasher, request_digest);
    update_str(&mut hasher, surface_digest);
    hasher.update(&[mode_code(mode)]);
    update_u64(&mut hasher, source_graph_nodes);
    update_u64(&mut hasher, source_graph_edges);
    update_str(&mut hasher, dimensions_digest);
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
        llm_organ::{LLMOrgan, LLMOrganConfig},
    };
    use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, Provenance};
    use symthaea_interlingua::CognitiveEnvelope;
    use symthaea_scip_llm_adapter::ScipLlmRequest;
    use symthaea_scip_realization_receipt::execute_with_surface_receipt;

    struct FixedBackend {
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
            "semantic-contract-test-backend"
        }
    }

    fn graph(sensor_label: &str) -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "sensor".into(),
                    kind: ConceptKind::Object,
                    label: Some(sensor_label.into()),
                    grounded_by: vec!["observation-17".into()],
                    confidence: 0.9,
                },
                ConceptNode {
                    id: "state".into(),
                    kind: ConceptKind::Property,
                    label: Some("nominal".into()),
                    grounded_by: vec!["observation-17".into()],
                    confidence: 0.8,
                },
            ],
            edges: vec![ConceptEdge {
                source: "sensor".into(),
                relation: "has-state".into(),
                target: "state".into(),
                evidence_ids: vec!["evidence-17".into()],
                confidence: 0.8,
            }],
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            provider: "semantic-contract-test".into(),
            provider_version: "1".into(),
            model_hash: "fixture".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    async fn accepted_receipt(
        source_graph: &GroundedConceptGraph,
        response: &str,
    ) -> ScipSurfaceRealizationReceiptV1 {
        let envelope =
            CognitiveEnvelope::from_graph(source_graph.clone(), 0.91, provenance()).unwrap();
        let request = ScipLlmRequest::compile(
            &envelope,
            None,
            LlmFallbackMode::FaithfulTranslation,
        )
        .unwrap();
        let mut organ = LLMOrgan::with_backend(
            LLMOrganConfig::default(),
            Arc::new(FixedBackend {
                response: response.into(),
            }),
        );
        let (_, receipt) = execute_with_surface_receipt(&request, &mut organ)
            .await
            .unwrap();
        receipt
    }

    #[tokio::test]
    async fn exact_grounded_source_binds_verification_contract() {
        let source = graph("S17");
        let receipt = accepted_receipt(&source, "Sensor S17 is nominal.").await;
        let contract = SemanticRealizationContractV1::bind_grounded_source(&receipt, &source)
            .unwrap();

        assert_eq!(
            contract.claim_scope(),
            SemanticContractClaimScope::VerificationRequired
        );
        assert_eq!(contract.source_semantic_hash(), receipt.source_semantic_hash());
        assert_eq!(contract.surface_receipt_digest(), receipt.receipt_digest());
        assert_eq!(contract.source_graph_nodes(), 2);
        assert_eq!(contract.source_graph_edges(), 1);
        assert!(!contract.semantic_fidelity_established());
        assert!(!contract.universal_scalar_score_defined());
        assert!(contract.matches(&receipt, &source).unwrap());
    }

    #[tokio::test]
    async fn different_grounded_source_is_rejected() {
        let source = graph("S17");
        let receipt = accepted_receipt(&source, "Sensor S17 is nominal.").await;
        let substituted = graph("S18");

        assert!(matches!(
            SemanticRealizationContractV1::bind_grounded_source(&receipt, &substituted),
            Err(SemanticContractError::SourceSemanticHashMismatch)
        ));
    }

    #[tokio::test]
    async fn changed_surface_changes_contract_identity() {
        let source = graph("S17");
        let receipt_a = accepted_receipt(&source, "Sensor S17 is nominal.").await;
        let receipt_b = accepted_receipt(&source, "Sensor S17 may be nominal.").await;
        let contract_a = SemanticRealizationContractV1::bind_grounded_source(&receipt_a, &source)
            .unwrap();
        let contract_b = SemanticRealizationContractV1::bind_grounded_source(&receipt_b, &source)
            .unwrap();

        assert_ne!(receipt_a.surface_digest(), receipt_b.surface_digest());
        assert_ne!(contract_a.contract_digest(), contract_b.contract_digest());
    }

    #[test]
    fn required_dimensions_are_complete_stable_and_unique() {
        assert_eq!(REQUIRED_SEMANTIC_DIMENSIONS_V1.len(), 11);
        for (expected_code, dimension) in REQUIRED_SEMANTIC_DIMENSIONS_V1.iter().enumerate() {
            assert_eq!(usize::from(dimension.code()), expected_code);
            assert!(!dimension.name().is_empty());
        }
        let digest = dimensions_digest_v1();
        assert_eq!(digest.len(), 64);
    }

    #[tokio::test]
    async fn debug_view_redacts_content_identities() {
        let source = graph("S17");
        let receipt = accepted_receipt(&source, "Sensor S17 is nominal.").await;
        let contract = SemanticRealizationContractV1::bind_grounded_source(&receipt, &source)
            .unwrap();
        let debug = format!("{contract:?}");

        assert!(!debug.contains(contract.contract_digest()));
        assert!(!debug.contains(contract.surface_receipt_digest()));
        assert!(!debug.contains(contract.source_semantic_hash()));
        assert!(!debug.contains(contract.request_digest()));
        assert!(!debug.contains(contract.surface_digest()));
    }
}
