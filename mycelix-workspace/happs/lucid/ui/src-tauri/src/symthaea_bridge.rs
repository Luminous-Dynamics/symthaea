//! Symthaea Bridge for LUCID Desktop
//!
//! This module provides Tauri commands that integrate Symthaea's consciousness
//! engine with LUCID's personal knowledge graph. It replaces the transformers.js
//! based semantic search with 16,384-dimensional HDC embeddings.
//!
//! ## Commands
//!
//! - `analyze_thought`: Process text through Symthaea, returning epistemic classification
//! - `semantic_search`: Generate HDC embedding for similarity search
//! - `check_coherence`: Analyze coherence across multiple thoughts
//!
//! ## Architecture
//!
//! Symthaea runs as a managed state in the Tauri application, initialized once
//! at startup and shared across all commands via `tauri::State`.

use anyhow::{Context, Result};
use lucid_symthaea::{
    convert_to_lucid, from_raw_scores,
    cosine_similarity, embedding_to_bytes, validate_dimension,
    CoherenceResult, Contradiction, EpistemicClassification, LucidThought,
    symthaea_cube_to_lucid,
    HDC_DIMENSION,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use symthaea::Symthaea;
use tauri::{Emitter, State};
use tokio::sync::Mutex;

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/// Symthaea state wrapper for Tauri managed state
pub struct SymthaeaState {
    /// The Symthaea instance (wrapped in Arc<Mutex> for async access)
    /// Public for access from mind_bridge module
    pub symthaea: Arc<Mutex<Option<Symthaea>>>,
    /// Whether initialization is in progress
    pub initializing: Arc<Mutex<bool>>,
    /// HDC dimension being used
    pub dimension: usize,
}

impl SymthaeaState {
    /// Create a new uninitialized state
    pub fn new() -> Self {
        Self {
            symthaea: Arc::new(Mutex::new(None)),
            initializing: Arc::new(Mutex::new(false)),
            dimension: HDC_DIMENSION,
        }
    }

    /// Check if Symthaea is ready
    pub async fn is_ready(&self) -> bool {
        self.symthaea.lock().await.is_some()
    }
}

impl Default for SymthaeaState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Bridge-specific errors
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Symthaea not initialized")]
    NotInitialized,

    #[error("Symthaea initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl serde::Serialize for BridgeError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/// Initialize Symthaea (call once at app startup)
///
/// This command initializes the Symthaea consciousness engine. It should be
/// called early in the application lifecycle to ensure embeddings are ready
/// when needed.
#[tauri::command]
pub async fn initialize_symthaea(
    state: State<'_, SymthaeaState>,
) -> Result<InitializeResponse, BridgeError> {
    // Check if already initializing
    let mut initializing = state.initializing.lock().await;
    if *initializing {
        return Ok(InitializeResponse {
            success: false,
            message: "Initialization already in progress".to_string(),
            dimension: state.dimension,
        });
    }

    // Check if already initialized
    if state.symthaea.lock().await.is_some() {
        return Ok(InitializeResponse {
            success: true,
            message: "Symthaea already initialized".to_string(),
            dimension: state.dimension,
        });
    }

    // Mark as initializing
    *initializing = true;
    drop(initializing);

    // Initialize Symthaea
    let result = Symthaea::new(state.dimension, 128).await;

    // Update state
    let mut initializing = state.initializing.lock().await;
    *initializing = false;

    match result {
        Ok(symthaea) => {
            let mut locked = state.symthaea.lock().await;
            *locked = Some(symthaea);

            tracing::info!(
                target: "lucid::symthaea",
                dimension = state.dimension,
                "Symthaea initialized successfully"
            );

            Ok(InitializeResponse {
                success: true,
                message: "Symthaea initialized".to_string(),
                dimension: state.dimension,
            })
        }
        Err(e) => {
            tracing::error!(
                target: "lucid::symthaea",
                error = %e,
                "Symthaea initialization failed"
            );

            Err(BridgeError::InitializationFailed(e.to_string()))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResponse {
    pub success: bool,
    pub message: String,
    pub dimension: usize,
}

/// Check if Symthaea is ready
#[tauri::command]
pub async fn symthaea_ready(state: State<'_, SymthaeaState>) -> Result<bool, BridgeError> {
    Ok(state.is_ready().await)
}

// ============================================================================
// THOUGHT ANALYSIS
// ============================================================================

/// Analyze thought content through Symthaea
///
/// This is the primary command for processing user input. It returns:
/// - Epistemic classification (E/N/M/H)
/// - HDC embedding (16,384 dimensions)
/// - Confidence and coherence scores
/// - Detected thought type
#[tauri::command]
pub async fn analyze_thought(
    content: String,
    state: State<'_, SymthaeaState>,
) -> Result<LucidThought, BridgeError> {
    if content.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Content cannot be empty".to_string()));
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Process through Symthaea
    let response = symthaea
        .process(&content)
        .await
        .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

    // Get embedding
    let embedding = get_embedding_internal(symthaea, &content)?;

    // Convert to LucidThought
    let thought = convert_to_lucid(&response, embedding);

    tracing::debug!(
        target: "lucid::symthaea",
        content_len = content.len(),
        confidence = thought.confidence,
        epistemic = ?thought.epistemic.code(),
        "Thought analyzed"
    );

    Ok(thought)
}

// ============================================================================
// SEMANTIC SEARCH (EMBEDDING)
// ============================================================================

/// Generate HDC embedding for text
///
/// Returns a 16,384-dimensional vector suitable for semantic similarity search.
/// The embedding is returned as raw bytes (f32 little-endian) for efficiency.
#[tauri::command]
pub async fn semantic_search(
    query: String,
    state: State<'_, SymthaeaState>,
) -> Result<Vec<u8>, BridgeError> {
    if query.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Query cannot be empty".to_string()));
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let embedding = get_embedding_internal(symthaea, &query)?;
    Ok(embedding_to_bytes(&embedding))
}

/// Generate HDC embedding as Float32Array (for direct JS use)
#[tauri::command]
pub async fn embed_text(
    text: String,
    state: State<'_, SymthaeaState>,
) -> Result<Vec<f32>, BridgeError> {
    if text.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Text cannot be empty".to_string()));
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    get_embedding_internal(symthaea, &text)
}

/// Batch embed multiple texts
#[tauri::command]
pub async fn batch_embed(
    texts: Vec<String>,
    state: State<'_, SymthaeaState>,
) -> Result<Vec<Vec<f32>>, BridgeError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut embeddings = Vec::with_capacity(texts.len());
    for text in texts {
        if text.trim().is_empty() {
            embeddings.push(vec![0.0f32; HDC_DIMENSION]);
        } else {
            let embedding = get_embedding_internal(symthaea, &text)?;
            embeddings.push(embedding);
        }
    }

    Ok(embeddings)
}

/// Internal function to get embedding from Symthaea
///
/// Uses Symthaea's native HDC encoding which provides:
/// - Neural Bridge v2 (BGE-M3) when available for high-quality semantic encoding
/// - Hash-based fallback when Neural Bridge is not compiled in
///
/// This replaces the previous hash_based_embedding() fallback with real semantic vectors.
fn get_embedding_internal(symthaea: &mut Symthaea, text: &str) -> Result<Vec<f32>, BridgeError> {
    // Use Symthaea's public embed API for real HDC embeddings
    let embedding = symthaea.embed_vec(text);

    // Validate dimension
    if let Err(e) = validate_dimension(&embedding) {
        return Err(BridgeError::ProcessingFailed(format!(
            "Invalid embedding dimension: {} (expected {}, got {})",
            e, HDC_DIMENSION, embedding.len()
        )));
    }

    Ok(embedding)
}

/// Hash-based embedding fallback (DEPRECATED)
///
/// This creates a deterministic 16,384-dim embedding from text using a hash function.
/// It is retained for testing and offline scenarios where Symthaea is not available.
///
/// **Note**: This function is no longer used in production. Real embeddings now
/// come from `Symthaea::embed_vec()` which uses either Neural Bridge v2 (BGE-M3)
/// or Symthaea's internal hash-based encoding.
#[allow(dead_code)]
fn hash_based_embedding(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; HDC_DIMENSION];

    // Hash each token and spread across dimensions
    for (word_idx, word) in text.split_whitespace().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let hash = hasher.finish();

        // Use hash to set values in embedding
        for i in 0..64 {
            let bit = (hash >> i) & 1;
            let dim = (word_idx * 64 + i) % HDC_DIMENSION;
            embedding[dim] += if bit == 1 { 1.0 } else { -1.0 };
        }
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

// ============================================================================
// SIMILARITY COMPUTATION
// ============================================================================

/// Compute similarity between two embeddings
#[tauri::command]
pub async fn compute_similarity(
    embedding_a: Vec<f32>,
    embedding_b: Vec<f32>,
) -> Result<f64, BridgeError> {
    if embedding_a.len() != embedding_b.len() {
        return Err(BridgeError::InvalidInput(format!(
            "Embedding dimension mismatch: {} vs {}",
            embedding_a.len(),
            embedding_b.len()
        )));
    }

    cosine_similarity(&embedding_a, &embedding_b)
        .map_err(|e: lucid_symthaea::EmbeddingError| BridgeError::ProcessingFailed(e.to_string()))
}

/// Find most similar embeddings from a set
#[tauri::command]
pub async fn find_similar(
    query_embedding: Vec<f32>,
    candidate_embeddings: Vec<(String, Vec<f32>)>,
    limit: usize,
    threshold: f64,
) -> Result<Vec<SimilarityResult>, BridgeError> {
    let mut results: Vec<SimilarityResult> = Vec::new();

    for (id, embedding) in candidate_embeddings {
        if let Ok(similarity) = cosine_similarity(&query_embedding, &embedding) {
            if similarity >= threshold {
                results.push(SimilarityResult { id, similarity });
            }
        }
    }

    // Sort by similarity descending
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(limit);

    Ok(results)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub id: String,
    pub similarity: f64,
}

// ============================================================================
// COHERENCE ANALYSIS
// ============================================================================

/// Check coherence across multiple thoughts using Symthaea's consciousness engine
///
/// This provides real coherence analysis by:
/// 1. Processing each thought through Symthaea to extract phi and coherence scores
/// 2. Computing cross-thought semantic similarity via HDC embeddings
/// 3. Detecting logical contradictions based on opposing epistemic stances
/// 4. Calculating aggregate coherence metrics
#[tauri::command]
pub async fn check_coherence(
    thought_contents: Vec<String>,
    state: State<'_, SymthaeaState>,
) -> Result<CoherenceResult, BridgeError> {
    if thought_contents.is_empty() {
        return Ok(CoherenceResult {
            overall: 1.0,
            logical: 1.0,
            temporal: 1.0,
            epistemic: 1.0,
            harmonic: 1.0,
            contradictions: Vec::new(),
        });
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Phase 1: Process each thought through Symthaea to get phi and coherence
    let mut thought_analyses: Vec<ThoughtAnalysis> = Vec::new();
    for (idx, content) in thought_contents.iter().enumerate() {
        let response = symthaea
            .process(content)
            .await
            .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

        let embedding = get_embedding_internal(symthaea, content)?;

        let (phi, coherence, epistemic_status) = if let Some(ref st) = response.structured_thought {
            (st.phi, st.coherence, format!("{:?}", st.epistemic_status))
        } else {
            (response.confidence as f64, response.confidence as f64, "Unknown".to_string())
        };

        thought_analyses.push(ThoughtAnalysis {
            index: idx,
            content: content.clone(),
            embedding,
            phi,
            coherence,
            epistemic_status,
        });
    }

    // Phase 2: Compute cross-thought coherence
    let mut total_similarity = 0.0;
    let mut phi_weighted_similarity = 0.0;
    let mut total_phi_weight = 0.0;
    let mut count = 0;
    let mut contradictions = Vec::new();

    for i in 0..thought_analyses.len() {
        for j in (i + 1)..thought_analyses.len() {
            let a = &thought_analyses[i];
            let b = &thought_analyses[j];

            if let Ok(sim) = cosine_similarity(&a.embedding, &b.embedding) {
                total_similarity += sim;
                count += 1;

                // Weight by combined phi (more integrated thoughts are more important)
                let phi_weight = (a.phi + b.phi) / 2.0;
                phi_weighted_similarity += sim * phi_weight;
                total_phi_weight += phi_weight;

                // Detect contradictions using multiple signals
                let is_contradiction = detect_contradiction(a, b, sim);
                if let Some(contradiction) = is_contradiction {
                    contradictions.push(contradiction);
                }
            }
        }
    }

    // Phase 3: Aggregate coherence metrics
    let avg_similarity = if count > 0 {
        total_similarity / count as f64
    } else {
        1.0
    };

    let weighted_similarity = if total_phi_weight > 0.0 {
        phi_weighted_similarity / total_phi_weight
    } else {
        avg_similarity
    };

    // Average phi across all thoughts (integrated information coherence)
    let avg_phi = thought_analyses.iter().map(|t| t.phi).sum::<f64>()
        / thought_analyses.len().max(1) as f64;

    // Average internal coherence scores
    let avg_internal_coherence = thought_analyses.iter().map(|t| t.coherence).sum::<f64>()
        / thought_analyses.len().max(1) as f64;

    // Penalize for contradictions
    let contradiction_penalty = (contradictions.len() as f64 * 0.1).min(0.5);

    let overall = (weighted_similarity * 0.4 + avg_phi * 0.3 + avg_internal_coherence * 0.3)
        - contradiction_penalty;

    // Compute temporal coherence: are semantically-related thoughts temporally clustered?
    // Uses index ordering as a temporal proxy (lower index = earlier thought).
    let temporal = if thought_analyses.len() >= 2 {
        // For each pair with high similarity, check if they are temporally close
        let max_gap = thought_analyses.len() as f64;
        let mut temporal_sum = 0.0;
        let mut temporal_count = 0usize;
        for i in 0..thought_analyses.len() {
            for j in (i + 1)..thought_analyses.len() {
                let emb_a = &thought_analyses[i].embedding;
                let emb_b = &thought_analyses[j].embedding;
                if let Ok(sim) = cosine_similarity(emb_a, emb_b) {
                    if sim > 0.5 {
                        // High similarity: temporal proximity should be high
                        let index_gap = (thought_analyses[j].index as f64
                            - thought_analyses[i].index as f64)
                            .abs();
                        let proximity = 1.0 - (index_gap / max_gap);
                        temporal_sum += proximity;
                        temporal_count += 1;
                    }
                }
            }
        }
        if temporal_count > 0 {
            temporal_sum / temporal_count as f64
        } else {
            1.0 // No highly similar pairs: no temporal penalty
        }
    } else {
        1.0 // Single thought: perfect temporal coherence
    };

    Ok(CoherenceResult {
        overall: overall.clamp(0.0, 1.0),
        logical: (avg_similarity - contradiction_penalty * 2.0).clamp(0.0, 1.0),
        temporal,
        epistemic: avg_internal_coherence,
        harmonic: avg_phi,
        contradictions,
    })
}

/// Internal analysis result for a single thought
struct ThoughtAnalysis {
    index: usize,
    content: String,
    embedding: Vec<f32>,
    phi: f64,
    coherence: f64,
    epistemic_status: String,
}

/// Detect if two thoughts contradict each other
fn detect_contradiction(a: &ThoughtAnalysis, b: &ThoughtAnalysis, similarity: f64) -> Option<Contradiction> {
    // Signal 1: Very low semantic similarity with high phi suggests meaningful contradiction
    if similarity < 0.2 && (a.phi > 0.5 || b.phi > 0.5) {
        return Some(Contradiction {
            thought_a: a.index.to_string(),
            thought_b: b.index.to_string(),
            description: format!(
                "Semantic divergence detected (similarity: {:.2}). Consider reconciling these perspectives.",
                similarity
            ),
            severity: (1.0 - similarity) * ((a.phi + b.phi) / 2.0),
        });
    }

    // Signal 2: Negation patterns in content
    let a_lower = a.content.to_lowercase();
    let b_lower = b.content.to_lowercase();

    let negation_patterns = [
        ("i believe ", "i don't believe "),
        ("i think ", "i don't think "),
        ("is true", "is false"),
        ("is correct", "is incorrect"),
        ("is right", "is wrong"),
        ("i love ", "i hate "),
        ("i support ", "i oppose "),
        ("agree", "disagree"),
    ];

    for (positive, negative) in negation_patterns {
        // Check if one contains positive and other contains negative about similar topic
        if (a_lower.contains(positive) && b_lower.contains(negative))
            || (a_lower.contains(negative) && b_lower.contains(positive))
        {
            return Some(Contradiction {
                thought_a: a.index.to_string(),
                thought_b: b.index.to_string(),
                description: format!(
                    "Opposing stance detected: '{}' vs '{}'. This may indicate a belief evolution or unresolved conflict.",
                    positive.trim(), negative.trim()
                ),
                severity: 0.7,
            });
        }
    }

    // Signal 3: High similarity but opposite epistemic status (confident vs uncertain)
    if similarity > 0.7 {
        let certain_uncertain = (a.epistemic_status.contains("Certain") && b.epistemic_status.contains("Uncertain"))
            || (a.epistemic_status.contains("Uncertain") && b.epistemic_status.contains("Certain"));

        if certain_uncertain {
            return Some(Contradiction {
                thought_a: a.index.to_string(),
                thought_b: b.index.to_string(),
                description: "Similar content with conflicting confidence levels. Review and consolidate your epistemic stance.".to_string(),
                severity: 0.4,
            });
        }
    }

    None
}

// ============================================================================
// EPISTEMIC CLASSIFICATION (Standalone)
// ============================================================================

/// Classify text epistemically without full Symthaea processing
///
/// This is a lighter-weight alternative to analyze_thought when you only
/// need the epistemic classification.
#[tauri::command]
pub async fn classify_epistemic(
    content: String,
    confidence: f64,
    state: State<'_, SymthaeaState>,
) -> Result<EpistemicClassification, BridgeError> {
    if content.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Content cannot be empty".to_string()));
    }

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Process to get epistemic data
    let response = symthaea
        .process(&content)
        .await
        .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

    // Extract epistemic classification
    if let Some(ref st) = response.structured_thought {
        if let Some(ref ctx) = st.domain_context {
            if let Some(ref cube) = ctx.cube {
                return Ok(symthaea_cube_to_lucid(
                    cube, st.phi, st.coherence,
                ));
            }
        }

        // Derive from confidence metrics
        return Ok(from_raw_scores(
            st.phi,
            0.0, // Default normative
            0.0, // Default materiality
            st.phi,
            st.coherence,
        ));
    }

    // Fallback: derive from confidence only
    Ok(from_raw_scores(confidence, 0.0, 0.0, 0.5, 0.5))
}

// ============================================================================
// UTILITY COMMANDS
// ============================================================================

/// Get the HDC dimension being used
#[tauri::command]
pub async fn get_hdc_dimension(state: State<'_, SymthaeaState>) -> Result<usize, BridgeError> {
    Ok(state.dimension)
}

/// Get Symthaea status information
#[tauri::command]
pub async fn get_symthaea_status(
    state: State<'_, SymthaeaState>,
) -> Result<SymthaeaStatus, BridgeError> {
    let is_ready = state.is_ready().await;
    let is_initializing = *state.initializing.lock().await;

    Ok(SymthaeaStatus {
        ready: is_ready,
        initializing: is_initializing,
        dimension: state.dimension,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SymthaeaStatus {
    pub ready: bool,
    pub initializing: bool,
    pub dimension: usize,
}

// ============================================================================
// COHERENCE FEEDBACK LOOP
// ============================================================================

/// Input for automatic coherence checking after thought creation
#[derive(Debug, Serialize, Deserialize)]
pub struct AutoCoherenceInput {
    /// The newly created thought's content
    pub new_thought_content: String,
    /// The newly created thought's ID
    pub new_thought_id: String,
    /// Contents of existing thoughts to check coherence against
    pub existing_thoughts: Vec<ExistingThought>,
    /// Minimum similarity threshold to consider for coherence check
    pub similarity_threshold: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExistingThought {
    pub id: String,
    pub content: String,
    /// Pre-computed embedding (if available)
    pub embedding: Option<Vec<f32>>,
}

/// Result of automatic coherence check for a new thought
#[derive(Debug, Serialize, Deserialize)]
pub struct AutoCoherenceResult {
    /// The new thought's ID
    pub thought_id: String,
    /// Overall coherence score with knowledge graph
    pub coherence_score: f64,
    /// Phi (integrated information) score
    pub phi_score: f64,
    /// Whether the thought is coherent with existing knowledge
    pub is_coherent: bool,
    /// The new thought's embedding (to store)
    pub embedding: Vec<f32>,
    /// Detected contradictions with existing thoughts
    pub contradictions: Vec<AutoContradiction>,
    /// Suggestions for the user
    pub suggestions: Vec<String>,
    /// IDs of thoughts that were analyzed (for storing coherence analysis)
    pub analyzed_thought_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AutoContradiction {
    /// ID of the conflicting thought
    pub conflicting_thought_id: String,
    /// Preview of the conflicting thought
    pub conflicting_content_preview: String,
    /// Severity of the contradiction (0.0-1.0)
    pub severity: f64,
    /// Type of contradiction detected
    pub contradiction_type: String,
    /// Human-readable description
    pub description: String,
}

/// Automatically check coherence when a new thought is created
///
/// This is the main entry point for the coherence feedback loop:
/// 1. Compute embedding for the new thought
/// 2. Find semantically similar existing thoughts
/// 3. Run coherence analysis
/// 4. Return results for UI display and Holochain storage
#[tauri::command]
pub async fn auto_check_coherence(
    input: AutoCoherenceInput,
    state: State<'_, SymthaeaState>,
) -> Result<AutoCoherenceResult, BridgeError> {
    let threshold = input.similarity_threshold.unwrap_or(0.3);

    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Step 1: Process the new thought through Symthaea
    let new_thought_response = symthaea
        .process(&input.new_thought_content)
        .await
        .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

    let new_embedding = get_embedding_internal(symthaea, &input.new_thought_content)?;

    let (new_phi, new_coherence) = if let Some(ref st) = new_thought_response.structured_thought {
        (st.phi, st.coherence)
    } else {
        (new_thought_response.confidence as f64, new_thought_response.confidence as f64)
    };

    // Step 2: Find similar thoughts and compute/use their embeddings
    let mut similar_thoughts: Vec<(ExistingThought, Vec<f32>, f64)> = Vec::new();

    for existing in &input.existing_thoughts {
        let existing_embedding = if let Some(ref emb) = existing.embedding {
            emb.clone()
        } else {
            // Compute embedding for existing thought
            get_embedding_internal(symthaea, &existing.content)?
        };

        if let Ok(similarity) = cosine_similarity(&new_embedding, &existing_embedding) {
            if similarity >= threshold {
                similar_thoughts.push((existing.clone(), existing_embedding, similarity));
            }
        }
    }

    // Sort by similarity descending and limit
    similar_thoughts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    similar_thoughts.truncate(10);

    // Step 3: Run coherence analysis on similar thoughts
    let mut contradictions: Vec<AutoContradiction> = Vec::new();
    let mut total_coherence = 0.0;
    let mut phi_sum = new_phi;
    let mut count = 1usize;
    let mut analyzed_ids = vec![input.new_thought_id.clone()];

    for (existing, _existing_embedding, similarity) in &similar_thoughts {
        analyzed_ids.push(existing.id.clone());

        // Process existing thought for deeper analysis
        let existing_response = symthaea
            .process(&existing.content)
            .await
            .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

        let (existing_phi, existing_coherence, _existing_status) = if let Some(ref st) = existing_response.structured_thought {
            (st.phi, st.coherence, format!("{:?}", st.epistemic_status))
        } else {
            (existing_response.confidence as f64, existing_response.confidence as f64, "Unknown".to_string())
        };

        phi_sum += existing_phi;
        total_coherence += existing_coherence;
        count += 1;

        // Check for contradiction
        if let Some(contradiction) = detect_auto_contradiction(
            &input.new_thought_content,
            &existing.content,
            &existing.id,
            *similarity,
            new_phi,
            existing_phi,
        ) {
            contradictions.push(contradiction);
        }
    }

    // Step 4: Calculate aggregate scores
    let avg_phi = phi_sum / count as f64;
    let avg_coherence = if count > 1 {
        total_coherence / (count - 1) as f64 // Exclude new thought from average
    } else {
        1.0
    };

    let avg_similarity = if similar_thoughts.is_empty() {
        1.0
    } else {
        similar_thoughts.iter().map(|(_, _, s)| s).sum::<f64>() / similar_thoughts.len() as f64
    };

    let contradiction_penalty = (contradictions.len() as f64 * 0.15).min(0.5);

    let overall_coherence = (
        avg_similarity * 0.3 +
        avg_phi * 0.3 +
        new_coherence * 0.2 +
        avg_coherence * 0.2
    ) - contradiction_penalty;

    // Step 5: Generate suggestions
    let mut suggestions: Vec<String> = Vec::new();

    if overall_coherence < 0.5 {
        suggestions.push("This thought has low coherence with your existing knowledge. Consider reviewing related thoughts.".to_string());
    }

    if !contradictions.is_empty() {
        suggestions.push(format!(
            "Found {} potential contradiction(s). Review and resolve for better cognitive clarity.",
            contradictions.len()
        ));
    }

    if similar_thoughts.is_empty() {
        suggestions.push("This thought introduces a new topic area. Consider linking it to related concepts.".to_string());
    }

    let is_coherent = overall_coherence >= 0.5 && contradictions.is_empty();

    Ok(AutoCoherenceResult {
        thought_id: input.new_thought_id,
        coherence_score: overall_coherence.clamp(0.0, 1.0),
        phi_score: new_phi,
        is_coherent,
        embedding: new_embedding,
        contradictions,
        suggestions,
        analyzed_thought_ids: analyzed_ids,
    })
}

/// Detect contradiction between new thought and an existing thought
fn detect_auto_contradiction(
    new_content: &str,
    existing_content: &str,
    existing_id: &str,
    similarity: f64,
    new_phi: f64,
    existing_phi: f64,
) -> Option<AutoContradiction> {
    let new_lower = new_content.to_lowercase();
    let existing_lower = existing_content.to_lowercase();

    // Signal 1: Negation patterns
    let negation_patterns = [
        ("i believe ", "i don't believe "),
        ("i think ", "i don't think "),
        ("is true", "is false"),
        ("is correct", "is incorrect"),
        ("is right", "is wrong"),
        ("i love ", "i hate "),
        ("i support ", "i oppose "),
        ("agree", "disagree"),
        ("always", "never"),
        ("everyone", "no one"),
        ("must", "must not"),
    ];

    for (positive, negative) in negation_patterns {
        if (new_lower.contains(positive) && existing_lower.contains(negative))
            || (new_lower.contains(negative) && existing_lower.contains(positive))
        {
            return Some(AutoContradiction {
                conflicting_thought_id: existing_id.to_string(),
                conflicting_content_preview: existing_content.chars().take(100).collect(),
                severity: 0.7,
                contradiction_type: "negation".to_string(),
                description: format!(
                    "Opposing stance detected ('{}'). This may represent a belief evolution or unresolved conflict.",
                    if new_lower.contains(positive) { positive.trim() } else { negative.trim() }
                ),
            });
        }
    }

    // Signal 2: High similarity with very different phi (cognitive dissonance)
    if similarity > 0.7 && (new_phi - existing_phi).abs() > 0.4 {
        return Some(AutoContradiction {
            conflicting_thought_id: existing_id.to_string(),
            conflicting_content_preview: existing_content.chars().take(100).collect(),
            severity: 0.4,
            contradiction_type: "cognitive_dissonance".to_string(),
            description: "Similar content with conflicting levels of integrated understanding. Consider consolidating.".to_string(),
        });
    }

    // Signal 3: Very low similarity but same topic markers
    if similarity < 0.2 {
        // Check for topic overlap via shared important words
        let new_words: std::collections::HashSet<&str> = new_lower.split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();
        let existing_words: std::collections::HashSet<&str> = existing_lower.split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();

        let overlap: Vec<_> = new_words.intersection(&existing_words).collect();
        if overlap.len() >= 2 {
            let avg_phi = (new_phi + existing_phi) / 2.0;
            return Some(AutoContradiction {
                conflicting_thought_id: existing_id.to_string(),
                conflicting_content_preview: existing_content.chars().take(100).collect(),
                severity: (1.0 - similarity) * avg_phi * 0.5,
                contradiction_type: "semantic_divergence".to_string(),
                description: format!(
                    "Different perspectives on shared topic ({}). May indicate conflicting views.",
                    overlap.into_iter().take(2).copied().collect::<Vec<&str>>().join(", ")
                ),
            });
        }
    }

    None
}

/// Batch compute embeddings for thoughts that don't have them
/// Called to backfill embeddings for existing thoughts
#[tauri::command]
pub async fn backfill_embeddings(
    thoughts: Vec<ExistingThought>,
    state: State<'_, SymthaeaState>,
) -> Result<Vec<(String, Vec<f32>)>, BridgeError> {
    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut results: Vec<(String, Vec<f32>)> = Vec::new();

    for thought in thoughts {
        if thought.embedding.is_none() {
            let embedding = get_embedding_internal(symthaea, &thought.content)?;
            results.push((thought.id, embedding));
        }
    }

    Ok(results)
}

/// Backfill embeddings and emit events for frontend DHT persistence.
///
/// Wraps `backfill_embeddings` and emits an `embeddings_computed` event
/// so the frontend can persist them to Holochain DHT.
#[tauri::command]
pub async fn backfill_and_persist_embeddings(
    thoughts: Vec<ExistingThought>,
    app_handle: tauri::AppHandle,
    state: State<'_, SymthaeaState>,
) -> Result<Vec<(String, Vec<f32>)>, BridgeError> {
    let mut locked = state.symthaea.lock().await;
    let symthaea = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut results: Vec<(String, Vec<f32>)> = Vec::new();

    for thought in thoughts {
        if thought.embedding.is_none() {
            let embedding = get_embedding_internal(symthaea, &thought.content)?;
            results.push((thought.id.clone(), embedding));
        }
    }

    // Emit event so the frontend can persist embeddings to DHT
    if !results.is_empty() {
        let _ = app_handle.emit("embeddings_computed", &results);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // hash_based_embedding tests
    // ========================================================================

    #[test]
    fn hash_embedding_deterministic() {
        let a = hash_based_embedding("consciousness is integrated information");
        let b = hash_based_embedding("consciousness is integrated information");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_embedding_correct_dimension() {
        let embedding = hash_based_embedding("hello world");
        assert_eq!(embedding.len(), HDC_DIMENSION);
    }

    #[test]
    fn hash_embedding_normalized() {
        let embedding = hash_based_embedding("some meaningful text with several words");
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}, expected ~1.0", norm);
    }

    #[test]
    fn hash_embedding_different_texts_differ() {
        let a = hash_based_embedding("the sun is bright");
        let b = hash_based_embedding("the moon is dark");
        assert_ne!(a, b);
    }

    #[test]
    fn hash_embedding_empty_text() {
        let embedding = hash_based_embedding("");
        assert_eq!(embedding.len(), HDC_DIMENSION);
        assert!(embedding.iter().all(|x| *x == 0.0));
    }

    // ========================================================================
    // detect_contradiction tests
    // ========================================================================

    fn make_thought(index: usize, content: &str, phi: f64, coherence: f64, status: &str) -> ThoughtAnalysis {
        ThoughtAnalysis {
            index,
            content: content.to_string(),
            embedding: vec![0.0; 10],
            phi,
            coherence,
            epistemic_status: status.to_string(),
        }
    }

    #[test]
    fn contradiction_negation_pattern() {
        let a = make_thought(0, "I believe in free will", 0.5, 0.7, "Certain");
        let b = make_thought(1, "I don't believe in determinism", 0.5, 0.6, "Certain");
        let result = detect_contradiction(&a, &b, 0.5);
        assert!(result.is_some(), "Should detect negation pattern");
        let c = result.unwrap();
        assert_eq!(c.severity, 0.7);
    }

    #[test]
    fn contradiction_low_similarity_high_phi() {
        let a = make_thought(0, "quantum mechanics", 0.8, 0.9, "Certain");
        let b = make_thought(1, "classical physics", 0.7, 0.8, "Certain");
        let result = detect_contradiction(&a, &b, 0.1);
        assert!(result.is_some(), "Should detect semantic divergence with high phi");
        let c = result.unwrap();
        assert!(c.severity > 0.5);
    }

    #[test]
    fn contradiction_epistemic_conflict() {
        let a = make_thought(0, "climate change is real", 0.5, 0.8, "Certain-E3");
        let b = make_thought(1, "climate change is happening", 0.5, 0.6, "Uncertain-E1");
        let result = detect_contradiction(&a, &b, 0.8);
        assert!(result.is_some(), "Should detect epistemic conflict");
        let c = result.unwrap();
        assert_eq!(c.severity, 0.4);
    }

    #[test]
    fn no_contradiction_similar_aligned_thoughts() {
        let a = make_thought(0, "I enjoy learning new things", 0.5, 0.7, "Certain");
        let b = make_thought(1, "education is valuable for growth", 0.5, 0.7, "Certain");
        let result = detect_contradiction(&a, &b, 0.6);
        assert!(result.is_none(), "Should not detect contradiction for aligned thoughts");
    }

    // ========================================================================
    // detect_auto_contradiction tests
    // ========================================================================

    #[test]
    fn auto_contradiction_negation() {
        let result = detect_auto_contradiction(
            "I think this approach is correct",
            "I don't think the method works",
            "existing-1",
            0.5,
            0.6,
            0.5,
        );
        assert!(result.is_some());
        let c = result.unwrap();
        assert_eq!(c.contradiction_type, "negation");
        assert_eq!(c.severity, 0.7);
    }

    #[test]
    fn auto_contradiction_cognitive_dissonance() {
        let result = detect_auto_contradiction(
            "Neural networks are powerful tools",
            "Neural networks are transformative tools",
            "existing-2",
            0.85,
            0.9,
            0.3,
        );
        assert!(result.is_some());
        let c = result.unwrap();
        assert_eq!(c.contradiction_type, "cognitive_dissonance");
        assert_eq!(c.severity, 0.4);
    }

    #[test]
    fn auto_contradiction_semantic_divergence() {
        let result = detect_auto_contradiction(
            "consciousness emerges from neural integration patterns",
            "consciousness cannot arise from simple integration alone",
            "existing-3",
            0.1,
            0.7,
            0.6,
        );
        assert!(result.is_some());
        let c = result.unwrap();
        assert_eq!(c.contradiction_type, "semantic_divergence");
    }

    #[test]
    fn auto_contradiction_none_for_unrelated() {
        let result = detect_auto_contradiction(
            "the weather is nice today",
            "rust is a great programming language",
            "existing-4",
            0.4,
            0.5,
            0.5,
        );
        assert!(result.is_none());
    }

    // ========================================================================
    // SymthaeaState tests
    // ========================================================================

    #[tokio::test]
    async fn state_new_not_ready() {
        let state = SymthaeaState::new();
        assert!(!state.is_ready().await);
        assert_eq!(state.dimension, HDC_DIMENSION);
    }

    #[tokio::test]
    async fn state_default_matches_new() {
        let a = SymthaeaState::new();
        let b = SymthaeaState::default();
        assert_eq!(a.dimension, b.dimension);
        assert!(!a.is_ready().await);
        assert!(!b.is_ready().await);
    }

    // ========================================================================
    // BridgeError serialization test
    // ========================================================================

    #[test]
    fn bridge_error_serializes_to_string() {
        let err = BridgeError::NotInitialized;
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("Symthaea not initialized"));

        let err2 = BridgeError::InvalidInput("empty".to_string());
        let json2 = serde_json::to_string(&err2).unwrap();
        assert!(json2.contains("empty"));
    }
}
