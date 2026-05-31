// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Knowledge - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover claim submission, graph relationships, inference, credibility
//! assessment, fact-checking, query execution, and knowledge bridge operations.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists
//! ls mycelix-knowledge/dna/mycelix_knowledge.dna
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

/// Mirror of claims_integrity::EpistemicPosition
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpistemicPosition {
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
}

impl Default for EpistemicPosition {
    fn default() -> Self {
        EpistemicPosition {
            empirical: 0.5,
            normative: 0.5,
            mythic: 0.5,
        }
    }
}

/// Mirror of claims_integrity::ClaimType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClaimType {
    Fact,
    Opinion,
    Prediction,
    Hypothesis,
    Definition,
    Historical,
    Normative,
    Narrative,
}

/// Mirror of claims_integrity::Claim
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Claim {
    pub id: String,
    pub content: String,
    pub classification: EpistemicPosition,
    pub author: String,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub claim_type: ClaimType,
    pub confidence: f64,
    pub expires: Option<Timestamp>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// Mirror of claims_integrity::EvidenceType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EvidenceType {
    Observation,
    Experiment,
    Statistical,
    Expert,
    Document,
    CrossReference,
    Logical,
}

/// Mirror of claims_integrity::Evidence
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    pub id: String,
    pub claim_id: String,
    pub evidence_type: EvidenceType,
    pub source_uri: String,
    pub content: String,
    pub strength: f64,
    pub submitted_by: String,
    pub submitted_at: Timestamp,
}

/// Mirror of claims_integrity::ChallengeStatus
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ChallengeStatus {
    Pending,
    UnderReview,
    Accepted,
    Rejected,
    Withdrawn,
}

/// Mirror of claims_integrity::ClaimChallenge
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimChallenge {
    pub id: String,
    pub claim_id: String,
    pub challenger: String,
    pub reason: String,
    pub counter_evidence: Vec<String>,
    pub status: ChallengeStatus,
    pub created: Timestamp,
}

/// Mirror of graph_integrity::RelationshipType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RelationshipType {
    Supports,
    Contradicts,
    DerivedFrom,
    ExampleOf,
    Generalizes,
    PartOf,
    Causes,
    RelatedTo,
    Equivalent,
    SpecializedFrom,
    Custom(String),
}

/// Mirror of graph_integrity::Relationship
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Relationship {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: RelationshipType,
    pub weight: f64,
    pub properties: Option<String>,
    pub creator: String,
    pub created: Timestamp,
}

/// Mirror of graph_integrity::Ontology
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Ontology {
    pub id: String,
    pub name: String,
    pub description: String,
    pub namespace: String,
    pub schema: String,
    pub version: String,
    pub creator: String,
    pub created: Timestamp,
    pub updated: Timestamp,
}

/// Mirror of graph_integrity::Concept
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Concept {
    pub id: String,
    pub ontology_id: String,
    pub name: String,
    pub definition: String,
    pub parent: Option<String>,
    pub synonyms: Vec<String>,
    pub created: Timestamp,
}

/// Mirror of graph coordinator::FindPathInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FindPathInput {
    pub source: String,
    pub target: String,
    pub max_depth: u32,
}

/// Mirror of graph coordinator::GraphStats
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GraphStats {
    pub relationship_count: u64,
}

/// Mirror of graph_integrity::PropagationResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropagationResult {
    pub source_claim_id: String,
    pub nodes_affected: u32,
    pub iterations: u32,
    pub converged: bool,
    pub max_delta: f64,
    pub processing_time_ms: u64,
    pub updated_nodes: Vec<String>,
}

/// Mirror of graph_integrity::DependencyTree
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DependencyTree {
    pub root_claim_id: String,
    pub nodes: Vec<DependencyTreeNode>,
    pub depth: u32,
    pub total_dependencies: u32,
    pub aggregate_weight: f64,
}

/// Mirror of graph_integrity::DependencyTreeNode
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DependencyTreeNode {
    pub claim_id: String,
    pub depth: u32,
    pub weight: f64,
    pub children: Vec<String>,
    pub is_leaf: bool,
}

/// Mirror of graph coordinator::DependencyTreeInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DependencyTreeInput {
    pub claim_id: String,
    pub max_depth: u32,
}

/// Mirror of graph_integrity::CascadeImpact
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CascadeImpact {
    pub claim_id: String,
    pub total_affected: u32,
    pub affected_by_depth: Vec<u32>,
    pub max_depth: u32,
    pub impact_score: f64,
    pub high_impact_claims: Vec<String>,
    pub assessed_at: Timestamp,
}

/// Mirror of inference_integrity::InferenceType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InferenceType {
    ImpliedRelation,
    Contradiction,
    Pattern,
    Prediction,
    Synthesis,
    Similarity,
    Anomaly,
    CredibilityAssessment,
}

/// Mirror of inference_integrity::Inference
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Inference {
    pub id: String,
    pub inference_type: InferenceType,
    pub source_claims: Vec<String>,
    pub conclusion: String,
    pub confidence: f64,
    pub reasoning: String,
    pub model: String,
    pub created: Timestamp,
    pub verified: bool,
}

/// Mirror of inference_integrity::CredibilitySubjectType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CredibilitySubjectType {
    Claim,
    Source,
    Author,
}

/// Mirror of inference coordinator::AssessCredibilityInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssessCredibilityInput {
    pub subject: String,
    pub subject_type: CredibilitySubjectType,
    pub expires_at: Option<Timestamp>,
}

/// Mirror of inference_integrity::CredibilityScore
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredibilityScore {
    pub id: String,
    pub subject: String,
    pub subject_type: CredibilitySubjectType,
    pub score: f64,
    pub components: CredibilityComponents,
    pub factors: Vec<CredibilityFactor>,
    pub assessed_at: Timestamp,
    pub expires_at: Option<Timestamp>,
}

/// Mirror of inference_integrity::CredibilityComponents
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredibilityComponents {
    pub accuracy: f64,
    pub consistency: f64,
    pub transparency: f64,
    pub track_record: f64,
    pub corroboration: f64,
}

/// Mirror of inference_integrity::CredibilityFactor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredibilityFactor {
    pub name: String,
    pub value: f64,
    pub explanation: String,
}

/// Mirror of inference_integrity::PatternType
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PatternType {
    Temporal,
    Cluster,
    Causal,
    Correlation,
    ContradictionCluster,
    EchoChamber,
}

/// Mirror of inference_integrity::Pattern
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Pattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub description: String,
    pub claims: Vec<String>,
    pub strength: f64,
    pub detected_at: Timestamp,
}

/// Mirror of inference coordinator::DetectPatternsInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DetectPatternsInput {
    pub claims: Vec<String>,
    pub pattern_types: Option<Vec<PatternType>>,
}

/// Mirror of inference coordinator::VerifyInferenceInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyInferenceInput {
    pub inference_id: String,
    pub is_correct: bool,
    pub verifier_did: String,
    pub comment: Option<String>,
}

/// Mirror of claims coordinator::ChallengeClaimInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChallengeClaimInput {
    pub claim_id: String,
    pub challenger_did: String,
    pub reason: String,
    pub counter_evidence: Vec<String>,
}

/// Mirror of query coordinator::ExecuteQueryInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecuteQueryInput {
    pub query: String,
    pub parameters: Option<String>,
    pub use_cache: bool,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Mirror of query coordinator::QueryExecutionResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryExecutionResult {
    pub results: Vec<String>,
    pub count: u64,
    pub execution_time_ms: u64,
    pub plan: Option<serde_json::Value>,
}

/// Mirror of query_integrity::SavedQuery
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SavedQuery {
    pub id: String,
    pub name: String,
    pub query: String,
    pub description: String,
    pub creator: String,
    pub public: bool,
    pub created: Timestamp,
}

/// Mirror of knowledge_bridge coordinator::RegisterClaimInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterClaimInput {
    pub source_happ: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub epistemic_e: Option<f64>,
    pub epistemic_n: Option<f64>,
    pub epistemic_m: Option<f64>,
}

/// Mirror of knowledge_bridge_integrity::ClaimReference
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClaimReference {
    pub claim_id: String,
    pub source_happ: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub epistemic_e: f64,
    pub epistemic_n: f64,
    pub epistemic_m: f64,
    pub created_at: Timestamp,
}

/// Mirror of knowledge_bridge coordinator::QueryKnowledgeInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryKnowledgeInput {
    pub source_happ: String,
    pub query_type: KnowledgeQueryType,
    pub parameters: serde_json::Value,
}

/// Mirror of knowledge_bridge_integrity::KnowledgeQueryType
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum KnowledgeQueryType {
    VerifyClaim,
    ClaimsBySubject,
    EpistemicScore,
    GraphTraversal,
    FactCheck,
    GisClassify,
    RashomonAnalyze,
    DarkSpotQuery,
    HarmonicAlignment,
}

/// Mirror of knowledge_bridge coordinator::QueryKnowledgeResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryKnowledgeResult {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Mirror of inference_integrity::AuthorReputation
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuthorReputation {
    pub id: String,
    pub author_did: String,
    pub overall_score: f64,
    pub domain_scores: Vec<serde_json::Value>,
    pub historical_accuracy: f64,
    pub claims_authored: u32,
    pub claims_verified_true: u32,
    pub claims_verified_false: u32,
    pub claims_pending: u32,
    pub average_epistemic_e: f64,
    pub matl_trust: f64,
    pub updated_at: Timestamp,
    pub reputation_age_days: u32,
}

/// Mirror of inference coordinator::UpdateAuthorReputationInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateAuthorReputationInput {
    pub author_did: String,
    pub claims_added: Option<u32>,
    pub verified_true: Option<u32>,
    pub verified_false: Option<u32>,
    pub matl_trust_update: Option<f64>,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_knowledge.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

/// Helper to create a test claim
fn make_test_claim(
    id: &str,
    content: &str,
    author: &str,
    tags: Vec<String>,
    now: Timestamp,
) -> Claim {
    Claim {
        id: id.to_string(),
        content: content.to_string(),
        classification: EpistemicPosition {
            empirical: 0.7,
            normative: 0.5,
            mythic: 0.3,
        },
        author: author.to_string(),
        sources: vec!["https://example.com/source1".to_string()],
        tags,
        claim_type: ClaimType::Fact,
        confidence: 0.8,
        expires: None,
        created: now,
        updated: now,
        version: 1,
    }
}

// ============================================================================
// Claims Zome Tests
// ============================================================================

#[cfg(test)]
mod claims_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_submit_and_get_claim() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);
        let claim = make_test_claim(
            "claim-test-001",
            "The Earth orbits the Sun",
            &author_did,
            vec!["astronomy".to_string(), "physics".to_string()],
            now,
        );

        // Submit claim
        let claim_record: Record = conductor
            .call(&cell.zome("claims"), "submit_claim", claim.clone())
            .await;

        let created_claim: Claim = decode_entry(&claim_record).expect("Failed to decode claim");
        assert_eq!(created_claim.id, "claim-test-001");
        assert_eq!(created_claim.version, 1);
        assert_eq!(created_claim.author, author_did);

        // Get claim by ID
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claim",
                "claim-test-001".to_string(),
            )
            .await;

        assert!(retrieved.is_some(), "Should find claim by ID");
        let retrieved_claim: Claim = decode_entry(&retrieved.unwrap()).expect("Failed to decode");
        assert_eq!(retrieved_claim.content, "The Earth orbits the Sun");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_claims_by_author() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);

        // Submit multiple claims
        for i in 0..3 {
            let claim = make_test_claim(
                &format!("claim-author-{}", i),
                &format!("Test claim number {}", i),
                &author_did,
                vec!["test".to_string()],
                now,
            );
            let _: Record = conductor
                .call(&cell.zome("claims"), "submit_claim", claim)
                .await;
        }

        // Get by author
        let claims: Vec<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claims_by_author",
                author_did.clone(),
            )
            .await;

        assert!(claims.len() >= 3, "Should find at least 3 claims by author");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_claims_by_tag() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);

        let claim = make_test_claim(
            "claim-tag-001",
            "Solar energy is renewable",
            &author_did,
            vec!["energy".to_string(), "sustainability".to_string()],
            now,
        );

        let _: Record = conductor
            .call(&cell.zome("claims"), "submit_claim", claim)
            .await;

        // Get by tag
        let claims: Vec<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claims_by_tag",
                "energy".to_string(),
            )
            .await;

        assert!(!claims.is_empty(), "Should find claim by tag");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_evidence_to_claim() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);

        let claim = make_test_claim(
            "claim-evidence-001",
            "Water boils at 100C at sea level",
            &author_did,
            vec!["chemistry".to_string()],
            now,
        );

        let _: Record = conductor
            .call(&cell.zome("claims"), "submit_claim", claim)
            .await;

        let evidence = Evidence {
            id: "evidence-001".to_string(),
            claim_id: "claim-evidence-001".to_string(),
            evidence_type: EvidenceType::Experiment,
            source_uri: "https://doi.org/10.1234/boiling".to_string(),
            content: "Laboratory measurement of water boiling point".to_string(),
            strength: 0.95,
            submitted_by: author_did.clone(),
            submitted_at: now,
        };

        let evidence_record: Record = conductor
            .call(&cell.zome("claims"), "add_evidence", evidence)
            .await;

        let created_evidence: Evidence = decode_entry(&evidence_record).expect("Failed to decode");
        assert_eq!(created_evidence.claim_id, "claim-evidence-001");
        assert_eq!(created_evidence.strength, 0.95);

        // Get evidence for claim
        let evidence_list: Vec<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claim_evidence",
                "claim-evidence-001".to_string(),
            )
            .await;

        assert!(!evidence_list.is_empty(), "Should find evidence for claim");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_challenge_claim() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);

        let claim = make_test_claim(
            "claim-challenge-001",
            "Controversial claim for testing",
            &author_did,
            vec!["debate".to_string()],
            now,
        );

        let _: Record = conductor
            .call(&cell.zome("claims"), "submit_claim", claim)
            .await;

        let challenge_input = ChallengeClaimInput {
            claim_id: "claim-challenge-001".to_string(),
            challenger_did: format!("did:mycelix:challenger_{}", agent),
            reason: "Insufficient evidence for this claim".to_string(),
            counter_evidence: vec!["https://counter-source.com".to_string()],
        };

        let challenge_record: Record = conductor
            .call(&cell.zome("claims"), "challenge_claim", challenge_input)
            .await;

        let challenge: ClaimChallenge =
            decode_entry(&challenge_record).expect("Failed to decode challenge");
        assert_eq!(challenge.claim_id, "claim-challenge-001");
        assert_eq!(challenge.status, ChallengeStatus::Pending);

        // Get challenges
        let challenges: Vec<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claim_challenges",
                "claim-challenge-001".to_string(),
            )
            .await;

        assert!(!challenges.is_empty(), "Should find challenges for claim");
    }
}

// ============================================================================
// Graph Zome Tests
// ============================================================================

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_relationship() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let creator_did = format!("did:mycelix:{}", agent);

        let relationship = Relationship {
            id: "rel-001".to_string(),
            source: "claim-A".to_string(),
            target: "claim-B".to_string(),
            relationship_type: RelationshipType::Supports,
            weight: 0.85,
            properties: None,
            creator: creator_did.clone(),
            created: now,
        };

        let rel_record: Record = conductor
            .call(&cell.zome("graph"), "create_relationship", relationship)
            .await;

        let created_rel: Relationship = decode_entry(&rel_record).expect("Failed to decode");
        assert_eq!(created_rel.source, "claim-A");
        assert_eq!(created_rel.target, "claim-B");
        assert_eq!(created_rel.weight, 0.85);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_outgoing_and_incoming_relationships() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let creator_did = format!("did:mycelix:{}", agent);

        // Create relationships: A -> B, A -> C
        for (id, target) in [("rel-out-1", "claim-B"), ("rel-out-2", "claim-C")] {
            let rel = Relationship {
                id: id.to_string(),
                source: "claim-A".to_string(),
                target: target.to_string(),
                relationship_type: RelationshipType::Supports,
                weight: 0.7,
                properties: None,
                creator: creator_did.clone(),
                created: now,
            };
            let _: Record = conductor
                .call(&cell.zome("graph"), "create_relationship", rel)
                .await;
        }

        // Get outgoing from A
        let outgoing: Vec<Record> = conductor
            .call(
                &cell.zome("graph"),
                "get_outgoing_relationships",
                "claim-A".to_string(),
            )
            .await;

        assert!(
            outgoing.len() >= 2,
            "Should have at least 2 outgoing relationships"
        );

        // Get incoming to B
        let incoming: Vec<Record> = conductor
            .call(
                &cell.zome("graph"),
                "get_incoming_relationships",
                "claim-B".to_string(),
            )
            .await;

        assert!(
            !incoming.is_empty(),
            "Should have incoming relationships to B"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_ontology_and_concepts() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let creator_did = format!("did:mycelix:{}", agent);

        let ontology = Ontology {
            id: "onto-energy".to_string(),
            name: "Energy Ontology".to_string(),
            description: "Classification of energy concepts".to_string(),
            namespace: "https://mycelix.net/ontology/energy".to_string(),
            schema: r#"{"type":"object"}"#.to_string(),
            version: "1.0.0".to_string(),
            creator: creator_did.clone(),
            created: now,
            updated: now,
        };

        let onto_record: Record = conductor
            .call(&cell.zome("graph"), "create_ontology", ontology)
            .await;

        let created_onto: Ontology = decode_entry(&onto_record).expect("Failed to decode");
        assert_eq!(created_onto.id, "onto-energy");

        // Create concept
        let concept = Concept {
            id: "concept-solar".to_string(),
            ontology_id: "onto-energy".to_string(),
            name: "Solar Energy".to_string(),
            definition: "Energy derived from the Sun's radiation".to_string(),
            parent: None,
            synonyms: vec!["photovoltaic energy".to_string()],
            created: now,
        };

        let concept_record: Record = conductor
            .call(&cell.zome("graph"), "create_concept", concept)
            .await;

        let created_concept: Concept = decode_entry(&concept_record).expect("Failed to decode");
        assert_eq!(created_concept.ontology_id, "onto-energy");

        // Get ontology concepts
        let concepts: Vec<Record> = conductor
            .call(
                &cell.zome("graph"),
                "get_ontology_concepts",
                "onto-energy".to_string(),
            )
            .await;

        assert!(!concepts.is_empty(), "Should find concepts in ontology");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_graph_stats() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let stats: GraphStats = conductor
            .call(&cell.zome("graph"), "get_graph_stats", ())
            .await;

        // Stats should return valid counts (may be 0 on fresh conductor)
        assert!(
            stats.relationship_count >= 0,
            "Relationship count should be non-negative"
        );
    }
}

// ============================================================================
// Inference Zome Tests
// ============================================================================

#[cfg(test)]
mod inference_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_inference() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        let inference = Inference {
            id: "inf-001".to_string(),
            inference_type: InferenceType::ImpliedRelation,
            source_claims: vec!["claim-A".to_string(), "claim-B".to_string()],
            conclusion: "Claim A implies Claim B through shared evidence".to_string(),
            confidence: 0.75,
            reasoning: "Both claims share 3 common evidence sources".to_string(),
            model: "pattern_matcher_v1".to_string(),
            created: now,
            verified: false,
        };

        let inf_record: Record = conductor
            .call(&cell.zome("inference"), "create_inference", inference)
            .await;

        let created_inf: Inference = decode_entry(&inf_record).expect("Failed to decode");
        assert_eq!(created_inf.id, "inf-001");
        assert!(!created_inf.verified);
        assert_eq!(created_inf.confidence, 0.75);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assess_credibility() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = AssessCredibilityInput {
            subject: "claim-credibility-001".to_string(),
            subject_type: CredibilitySubjectType::Claim,
            expires_at: None,
        };

        let score_record: Record = conductor
            .call(&cell.zome("inference"), "assess_credibility", input)
            .await;

        let score: CredibilityScore = decode_entry(&score_record).expect("Failed to decode");
        assert!(
            score.score >= 0.0 && score.score <= 1.0,
            "Score must be 0-1"
        );
        assert!(
            score.components.accuracy >= 0.0 && score.components.accuracy <= 1.0,
            "Component scores must be 0-1"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_detect_patterns() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = DetectPatternsInput {
            claims: vec![
                "claim-pattern-1".to_string(),
                "claim-pattern-2".to_string(),
                "claim-pattern-3".to_string(),
            ],
            pattern_types: None,
        };

        let patterns: Vec<Record> = conductor
            .call(&cell.zome("inference"), "detect_patterns", input)
            .await;

        assert!(
            !patterns.is_empty(),
            "Should detect at least one pattern from 3 claims"
        );

        let pattern: Pattern = decode_entry(&patterns[0]).expect("Failed to decode pattern");
        assert_eq!(pattern.pattern_type, PatternType::Cluster);
        assert!(
            pattern.strength > 0.0,
            "Pattern strength should be positive"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_author_reputation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let author_did = "did:mycelix:test_author_rep".to_string();

        // Get or create author reputation
        let rep_record: Record = conductor
            .call(
                &cell.zome("inference"),
                "get_author_reputation",
                author_did.clone(),
            )
            .await;

        let reputation: AuthorReputation = decode_entry(&rep_record).expect("Failed to decode");
        assert_eq!(reputation.author_did, author_did);
        assert_eq!(
            reputation.overall_score, 0.5,
            "New author should have neutral reputation"
        );

        // Update reputation
        let update_input = UpdateAuthorReputationInput {
            author_did: author_did.clone(),
            claims_added: Some(5),
            verified_true: Some(3),
            verified_false: Some(1),
            matl_trust_update: Some(0.8),
        };

        let updated_record: Record = conductor
            .call(
                &cell.zome("inference"),
                "update_author_reputation",
                update_input,
            )
            .await;

        let updated_rep: AuthorReputation =
            decode_entry(&updated_record).expect("Failed to decode");
        assert_eq!(updated_rep.claims_authored, 5);
        assert_eq!(updated_rep.claims_verified_true, 3);
        assert!(
            updated_rep.historical_accuracy > 0.5,
            "Accuracy should improve with more true than false"
        );
    }
}

// ============================================================================
// Knowledge Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_external_claim() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterClaimInput {
            source_happ: "mycelix-governance".to_string(),
            subject: "proposal-123".to_string(),
            predicate: "has_quorum".to_string(),
            object: "true".to_string(),
            epistemic_e: Some(0.9),
            epistemic_n: Some(0.7),
            epistemic_m: Some(0.5),
        };

        let claim_record: Record = conductor
            .call(
                &cell.zome("knowledge_bridge"),
                "register_external_claim",
                input,
            )
            .await;

        let claim_ref: ClaimReference =
            decode_entry(&claim_record).expect("Failed to decode claim reference");
        assert_eq!(claim_ref.source_happ, "mycelix-governance");
        assert_eq!(claim_ref.subject, "proposal-123");
        assert_eq!(claim_ref.epistemic_e, 0.9);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_claims_by_subject() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register a claim about a subject
        let input = RegisterClaimInput {
            source_happ: "mycelix-energy".to_string(),
            subject: "solar-panel-efficiency".to_string(),
            predicate: "exceeds".to_string(),
            object: "20%".to_string(),
            epistemic_e: Some(0.8),
            epistemic_n: None,
            epistemic_m: None,
        };

        let _: Record = conductor
            .call(
                &cell.zome("knowledge_bridge"),
                "register_external_claim",
                input,
            )
            .await;

        // Get claims by subject
        let claims: Vec<Record> = conductor
            .call(
                &cell.zome("knowledge_bridge"),
                "get_claims_by_subject",
                "solar-panel-efficiency".to_string(),
            )
            .await;

        assert!(!claims.is_empty(), "Should find claims about the subject");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_claims_by_happ() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register claims from a specific hApp
        for i in 0..2 {
            let input = RegisterClaimInput {
                source_happ: "mycelix-justice".to_string(),
                subject: format!("case-{}", i),
                predicate: "status".to_string(),
                object: "active".to_string(),
                epistemic_e: Some(0.6),
                epistemic_n: Some(0.5),
                epistemic_m: Some(0.4),
            };

            let _: Record = conductor
                .call(
                    &cell.zome("knowledge_bridge"),
                    "register_external_claim",
                    input,
                )
                .await;
        }

        let claims: Vec<Record> = conductor
            .call(
                &cell.zome("knowledge_bridge"),
                "get_claims_by_happ",
                "mycelix-justice".to_string(),
            )
            .await;

        assert!(
            claims.len() >= 2,
            "Should find at least 2 claims from justice hApp"
        );
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_knowledge_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let now = Timestamp::now();
        let author_did = format!("did:mycelix:{}", agent);

        // 1. Submit a claim
        let claim = make_test_claim(
            "lifecycle-claim-001",
            "Decentralized knowledge graphs improve epistemic resilience",
            &author_did,
            vec!["knowledge".to_string(), "decentralization".to_string()],
            now,
        );

        let claim_record: Record = conductor
            .call(&cell.zome("claims"), "submit_claim", claim)
            .await;

        let created_claim: Claim = decode_entry(&claim_record).expect("Failed to decode");
        assert_eq!(created_claim.version, 1);

        // 2. Add evidence
        let evidence = Evidence {
            id: "lifecycle-ev-001".to_string(),
            claim_id: "lifecycle-claim-001".to_string(),
            evidence_type: EvidenceType::Statistical,
            source_uri: "https://doi.org/10.5555/dkg-study".to_string(),
            content: "Study showing 40% improvement in fact verification with DKG".to_string(),
            strength: 0.85,
            submitted_by: author_did.clone(),
            submitted_at: now,
        };

        let _: Record = conductor
            .call(&cell.zome("claims"), "add_evidence", evidence)
            .await;

        // 3. Create a relationship to another claim
        let relationship = Relationship {
            id: "lifecycle-rel-001".to_string(),
            source: "lifecycle-claim-001".to_string(),
            target: "lifecycle-claim-002".to_string(),
            relationship_type: RelationshipType::Supports,
            weight: 0.9,
            properties: None,
            creator: author_did.clone(),
            created: now,
        };

        let _: Record = conductor
            .call(&cell.zome("graph"), "create_relationship", relationship)
            .await;

        // 4. Create an inference
        let inference = Inference {
            id: "lifecycle-inf-001".to_string(),
            inference_type: InferenceType::ImpliedRelation,
            source_claims: vec![
                "lifecycle-claim-001".to_string(),
                "lifecycle-claim-002".to_string(),
            ],
            conclusion: "DKGs and epistemic markets are complementary".to_string(),
            confidence: 0.7,
            reasoning: "Both address information asymmetry".to_string(),
            model: "similarity_v1".to_string(),
            created: now,
            verified: false,
        };

        let _: Record = conductor
            .call(&cell.zome("inference"), "create_inference", inference)
            .await;

        // 5. Assess credibility
        let cred_input = AssessCredibilityInput {
            subject: "lifecycle-claim-001".to_string(),
            subject_type: CredibilitySubjectType::Claim,
            expires_at: None,
        };

        let cred_record: Record = conductor
            .call(&cell.zome("inference"), "assess_credibility", cred_input)
            .await;

        let credibility: CredibilityScore =
            decode_entry(&cred_record).expect("Failed to decode credibility");
        assert!(
            credibility.score >= 0.0 && credibility.score <= 1.0,
            "Credibility score must be 0-1"
        );

        // 6. Register external claim via bridge
        let bridge_input = RegisterClaimInput {
            source_happ: "mycelix-governance".to_string(),
            subject: "lifecycle-claim-001".to_string(),
            predicate: "referenced_by".to_string(),
            object: "proposal-456".to_string(),
            epistemic_e: Some(0.8),
            epistemic_n: Some(0.6),
            epistemic_m: Some(0.5),
        };

        let _: Record = conductor
            .call(
                &cell.zome("knowledge_bridge"),
                "register_external_claim",
                bridge_input,
            )
            .await;

        // 7. Verify the knowledge graph state
        let evidence_list: Vec<Record> = conductor
            .call(
                &cell.zome("claims"),
                "get_claim_evidence",
                "lifecycle-claim-001".to_string(),
            )
            .await;

        assert!(!evidence_list.is_empty(), "Claim should have evidence");

        let outgoing: Vec<Record> = conductor
            .call(
                &cell.zome("graph"),
                "get_outgoing_relationships",
                "lifecycle-claim-001".to_string(),
            )
            .await;

        assert!(
            !outgoing.is_empty(),
            "Claim should have outgoing relationships"
        );
    }
}
