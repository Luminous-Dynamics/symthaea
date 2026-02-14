//! LUCID Sweettest Integration Tests
//!
//! Rust-based integration tests using Holochain's sweettest framework.

use holochain::sweettest::*;
use holochain_types::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Path to the LUCID hApp bundle
pub fn lucid_happ_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("lucid.happ")
}

// ============================================================================
// TYPES (matching zome types)
// ============================================================================

/// Thought type enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum ThoughtType {
    #[default]
    Note,
    Belief,
    Question,
    Evidence,
    Hypothesis,
    Idea,
    Claim,
    Argument,
    Memory,
    Insight,
}

/// Empirical level for epistemic classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, Default)]
pub enum EmpiricalLevel {
    #[default]
    E0,  // Pure speculation
    E1,  // Personal anecdote
    E2,  // Multiple anecdotes
    E3,  // Correlational evidence
    E4,  // Single study
    E5,  // Multiple studies
    E6,  // Meta-analysis
    E7,  // Scientific consensus
}

/// Normative level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum NormativeLevel {
    #[default]
    N0,  // Purely descriptive
    N1,  // Mild preference
    N2,  // Strong preference
    N3,  // Moral claim
}

/// Materiality level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum MaterialityLevel {
    #[default]
    M0,  // Pure abstraction
    M1,  // Conceptual
    M2,  // Social/institutional
    M3,  // Physical
}

/// Harmonic level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum HarmonicLevel {
    #[default]
    H0,  // Isolated fact
    H1,  // Connected to few concepts
    H2,  // Connected to many concepts
    H3,  // Highly integrated worldview
}

/// Epistemic classification
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EpistemicClassification {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
    pub harmonic: HarmonicLevel,
}

/// Input for creating a thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateThoughtInput {
    pub content: String,
    pub thought_type: Option<ThoughtType>,
    pub epistemic: Option<EpistemicClassification>,
    pub confidence: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub domain: Option<String>,
    pub related_thoughts: Option<Vec<String>>,
    pub parent_thought: Option<String>,
    pub embedding: Option<Vec<f32>>,
}

impl CreateThoughtInput {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            thought_type: None,
            epistemic: None,
            confidence: None,
            tags: None,
            domain: None,
            related_thoughts: None,
            parent_thought: None,
            embedding: None,
        }
    }

    pub fn with_type(mut self, thought_type: ThoughtType) -> Self {
        self.thought_type = Some(thought_type);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }
}

/// Input for updating a thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateThoughtInput {
    pub thought_id: String,
    pub content: Option<String>,
    pub thought_type: Option<ThoughtType>,
    pub epistemic: Option<EpistemicClassification>,
    pub confidence: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub domain: Option<String>,
    pub related_thoughts: Option<Vec<String>>,
}

/// Thought entry (for deserializing responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub id: String,
    pub content: String,
    pub thought_type: ThoughtType,
    pub epistemic: EpistemicClassification,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub domain: Option<String>,
    pub related_thoughts: Vec<String>,
    pub source_hashes: Vec<String>,
    pub parent_thought: Option<String>,
    pub created_at: holochain_types::prelude::Timestamp,
    pub updated_at: holochain_types::prelude::Timestamp,
    pub version: u32,
    pub embedding: Option<Vec<f32>>,
    pub embedding_version: Option<u32>,
    pub coherence_score: Option<f64>,
    pub phi_score: Option<f64>,
}

/// Search input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchThoughtsInput {
    pub tags: Option<Vec<String>>,
    pub domain: Option<String>,
    pub thought_type: Option<ThoughtType>,
    pub min_empirical: Option<EmpiricalLevel>,
    pub min_confidence: Option<f64>,
    pub content_contains: Option<String>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

// ============================================================================
// SYMTHAEA INTEGRATION TYPES
// ============================================================================

/// Input for updating a thought's embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateEmbeddingInput {
    pub thought_id: String,
    pub embedding: Vec<f32>,
}

/// Input for updating coherence/phi scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCoherenceInput {
    pub thought_id: String,
    pub coherence_score: f64,
    pub phi_score: Option<f64>,
}

/// Input for semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchInput {
    pub query_embedding: Vec<f32>,
    pub threshold: f64,
    pub limit: Option<u32>,
}

/// Result of semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchResult {
    pub thought_id: String,
    pub action_hash: ActionHash,
    pub similarity: f64,
    pub content_preview: String,
}

/// Input for coherence range query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceRangeInput {
    pub min_coherence: f64,
    pub max_coherence: f64,
}

// ============================================================================
// EPISTEMIC GARDEN TYPES
// ============================================================================

/// Input for exploring the epistemic garden
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreGardenInput {
    pub max_clusters: u32,
    pub min_cluster_size: u32,
    pub domain_filter: Option<String>,
}

/// A cluster of semantically related thoughts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtCluster {
    pub cluster_id: u32,
    pub centroid_thought_id: String,
    pub thought_ids: Vec<String>,
    pub avg_confidence: f64,
    pub avg_coherence: Option<f64>,
    pub dominant_domain: Option<String>,
    pub dominant_tags: Vec<String>,
}

/// Input for suggesting connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestConnectionsInput {
    pub thought_id: String,
    pub max_suggestions: u32,
    pub min_similarity: f64,
}

/// A suggestion for connecting two unlinked thoughts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSuggestion {
    pub thought_a_id: String,
    pub thought_b_id: String,
    pub similarity: f64,
    pub shared_tags: Vec<String>,
}

/// A knowledge gap in a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub domain: String,
    pub thought_count: u32,
    pub avg_confidence: f64,
    pub low_confidence_count: u32,
    pub missing_embeddings: u32,
    pub sparse: bool,
}

/// A discovered pattern in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPattern {
    pub pattern_type: String,
    pub description: String,
    pub involved_thought_ids: Vec<String>,
    pub confidence: f64,
}

// ============================================================================
// TEMPORAL CONSCIOUSNESS TYPES
// ============================================================================

/// What triggered a belief snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotTrigger {
    Created,
    Edited,
    Scheduled,
    ContradictionDetected,
    ReviewedConfirmed,
    ReviewedModified,
    EvidenceUpdated,
    CoherenceChanged,
}

/// Input for recording a belief snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordSnapshotInput {
    pub thought_id: String,
    pub epistemic_code: String,
    pub confidence: f64,
    pub phi: f64,
    pub coherence: f64,
    pub trigger: SnapshotTrigger,
}

/// Input for recording consciousness evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordEvolutionInput {
    pub period_start: holochain_types::prelude::Timestamp,
    pub period_end: holochain_types::prelude::Timestamp,
    pub avg_phi: f64,
    pub phi_trend: f64,
    pub avg_coherence: f64,
    pub coherence_trend: f64,
    pub stable_belief_count: u32,
    pub growing_belief_count: u32,
    pub weakening_belief_count: u32,
    pub entrenched_belief_count: u32,
    pub insights: Vec<String>,
}

// ============================================================================
// HELPERS
// ============================================================================

/// Generate a random 16,384D embedding (normalized)
pub fn generate_random_embedding(seed: u64) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(16384);
    let mut state = seed;
    for _ in 0..16384 {
        // Simple LCG for reproducibility
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        embedding.push(val);
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }
    embedding
}

/// Generate a similar embedding (with some noise)
pub fn generate_similar_embedding(base: &[f32], noise_factor: f32, seed: u64) -> Vec<f32> {
    let mut result = base.to_vec();
    let mut state = seed;
    for v in &mut result {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        *v += noise * noise_factor;
    }
    // Normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut result {
            *v /= norm;
        }
    }
    result
}

/// Extract a Thought from a Record
pub fn thought_from_record(record: &Record) -> Option<Thought> {
    record
        .entry()
        .to_app_option::<Thought>()
        .ok()
        .flatten()
}

/// Create a sweetest conductor with LUCID installed
pub async fn setup_lucid_conductor() -> (SweetConductor, SweetZome) {
    let conductor = SweetConductor::from_standard_config().await;

    let happ_path = lucid_happ_path();
    assert!(happ_path.exists(), "hApp not found at {:?}", happ_path);

    let app = conductor.setup_app("lucid", &[&happ_path]).await.unwrap();
    let cells = app.into_cells();
    let lucid_cell = cells.into_iter().next().expect("No cells in app");

    let zome = SweetZome::new(lucid_cell.cell_id().clone(), "lucid".into());

    (conductor, zome)
}
