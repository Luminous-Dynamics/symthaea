//! LucidMind Bridge - ContinuousMind wrapper for LUCID Desktop
//!
//! This module provides a persistent working memory context that wraps
//! Symthaea's ContinuousMind. It enables:
//!
//! - **Session Memory**: Thoughts persist across multiple analyses
//! - **Context-Aware Analysis**: References like "that" resolve correctly
//! - **Working Memory Seeding**: Pre-load context for better understanding
//! - **Consciousness Profile**: Full access to phi, meta-awareness, cognitive load
//!
//! ## Architecture
//!
//! LucidMind maintains a separate ContinuousMind instance from the main Symthaea
//! facade, allowing persistent state between analyses. The main Symthaea instance
//! is still used for embeddings and one-shot processing.

use anyhow::Result;
use lucid_symthaea::{ConsciousnessProfile, EpistemicClassification, LucidThought, convert_to_lucid, from_raw_scores};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use symthaea::mind::{ContinuousMind, MindConfig, MindState};
use symthaea::hdc::real_hv::RealHV;
use tauri::State;
use tokio::sync::Mutex;

use crate::symthaea_bridge::{BridgeError, SymthaeaState};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/// LucidMind state wrapper for Tauri managed state
pub struct LucidMindState {
    /// The ContinuousMind instance for persistent context
    mind: Arc<Mutex<Option<ContinuousMindWrapper>>>,
    /// Whether initialization is in progress
    initializing: Arc<Mutex<bool>>,
}

impl LucidMindState {
    /// Create a new uninitialized state
    pub fn new() -> Self {
        Self {
            mind: Arc::new(Mutex::new(None)),
            initializing: Arc::new(Mutex::new(false)),
        }
    }

    /// Check if LucidMind is ready
    pub async fn is_ready(&self) -> bool {
        self.mind.lock().await.is_some()
    }
}

impl Default for LucidMindState {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper around ContinuousMind with session tracking
struct ContinuousMindWrapper {
    /// The underlying ContinuousMind
    mind: ContinuousMind,
    /// Session memory: recent thought contents for context
    session_memory: Vec<SessionThought>,
    /// HDC dimension
    dimension: usize,
    /// Total thoughts analyzed in this session
    thoughts_analyzed: u64,
}

/// A thought stored in session memory
#[derive(Debug, Clone)]
struct SessionThought {
    content: String,
    embedding: RealHV,
    timestamp: std::time::Instant,
}

impl ContinuousMindWrapper {
    fn new(dimension: usize) -> Self {
        let config = MindConfig {
            dimension,
            ..MindConfig::default()
        };
        let mut mind = ContinuousMind::new(config);
        mind.awaken();
        mind.seed_memory();

        Self {
            mind,
            session_memory: Vec::new(),
            dimension,
            thoughts_analyzed: 0,
        }
    }

    /// Add a thought to session memory
    fn add_to_session(&mut self, content: String, embedding: RealHV) {
        self.session_memory.push(SessionThought {
            content,
            embedding,
            timestamp: std::time::Instant::now(),
        });

        // Keep session memory bounded (last 50 thoughts)
        if self.session_memory.len() > 50 {
            self.session_memory.remove(0);
        }
    }

    /// Get recent context as text
    fn recent_context(&self, n: usize) -> Vec<&str> {
        self.session_memory
            .iter()
            .rev()
            .take(n)
            .map(|t| t.content.as_str())
            .collect()
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/// Initialize the LucidMind working memory system
#[tauri::command]
pub async fn initialize_lucid_mind(
    state: State<'_, LucidMindState>,
    symthaea_state: State<'_, SymthaeaState>,
) -> Result<MindInitResponse, BridgeError> {
    // Check if already initializing
    let mut initializing = state.initializing.lock().await;
    if *initializing {
        return Ok(MindInitResponse {
            success: false,
            message: "Initialization already in progress".to_string(),
        });
    }

    // Check if already initialized
    if state.mind.lock().await.is_some() {
        return Ok(MindInitResponse {
            success: true,
            message: "LucidMind already initialized".to_string(),
        });
    }

    // Mark as initializing
    *initializing = true;
    drop(initializing);

    // Get dimension from Symthaea state
    let dimension = {
        let locked = symthaea_state.symthaea.lock().await;
        if let Some(ref symthaea) = *locked {
            symthaea.dimension()
        } else {
            16384 // Default HDC dimension
        }
    };

    // Create the wrapper
    let wrapper = ContinuousMindWrapper::new(dimension);

    // Update state
    let mut initializing = state.initializing.lock().await;
    *initializing = false;

    let mut locked = state.mind.lock().await;
    *locked = Some(wrapper);

    tracing::info!(
        target: "lucid::mind",
        dimension = dimension,
        "LucidMind initialized with persistent working memory"
    );

    Ok(MindInitResponse {
        success: true,
        message: "LucidMind initialized".to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MindInitResponse {
    pub success: bool,
    pub message: String,
}

/// Check if LucidMind is ready
#[tauri::command]
pub async fn lucid_mind_ready(state: State<'_, LucidMindState>) -> Result<bool, BridgeError> {
    Ok(state.is_ready().await)
}

// ============================================================================
// WORKING MEMORY OPERATIONS
// ============================================================================

/// Seed working memory with context thoughts
///
/// Call this to pre-load context before analyzing thoughts that contain
/// references (like "that", "it", "the above", etc.)
#[tauri::command]
pub async fn seed_working_memory(
    thoughts: Vec<String>,
    state: State<'_, LucidMindState>,
    symthaea_state: State<'_, SymthaeaState>,
) -> Result<SeedResponse, BridgeError> {
    let mut mind_locked = state.mind.lock().await;
    let wrapper = mind_locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut symthaea_locked = symthaea_state.symthaea.lock().await;
    let symthaea = symthaea_locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut seeded_count = 0;

    for thought in thoughts {
        if thought.trim().is_empty() {
            continue;
        }

        // Get embedding from Symthaea
        let hv = symthaea.embed(&thought);

        // Perceive in the mind
        wrapper.mind.perceive_text(&thought, hv.clone());
        wrapper.add_to_session(thought.clone(), hv);

        seeded_count += 1;
    }

    // Run a tick to integrate
    wrapper.mind.tick();

    tracing::debug!(
        target: "lucid::mind",
        seeded = seeded_count,
        session_size = wrapper.session_memory.len(),
        "Working memory seeded"
    );

    Ok(SeedResponse {
        seeded_count,
        session_memory_size: wrapper.session_memory.len(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SeedResponse {
    pub seeded_count: usize,
    pub session_memory_size: usize,
}

/// Clear working memory (start fresh session)
#[tauri::command]
pub async fn clear_working_memory(
    state: State<'_, LucidMindState>,
) -> Result<(), BridgeError> {
    let mut locked = state.mind.lock().await;
    let wrapper = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Clear session memory
    wrapper.session_memory.clear();

    // Re-initialize the mind
    let config = MindConfig {
        dimension: wrapper.dimension,
        ..MindConfig::default()
    };
    wrapper.mind = ContinuousMind::new(config);
    wrapper.mind.awaken();
    wrapper.mind.seed_memory();

    tracing::info!(
        target: "lucid::mind",
        "Working memory cleared"
    );

    Ok(())
}

// ============================================================================
// CONTEXT-AWARE ANALYSIS
// ============================================================================

/// Analyze a thought with full session context
///
/// Unlike `analyze_thought` which is one-shot, this maintains context between
/// calls and can resolve references to previous thoughts.
#[tauri::command]
pub async fn analyze_with_context(
    content: String,
    state: State<'_, LucidMindState>,
    symthaea_state: State<'_, SymthaeaState>,
) -> Result<ContextualAnalysis, BridgeError> {
    if content.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Content cannot be empty".to_string()));
    }

    let mut mind_locked = state.mind.lock().await;
    let wrapper = mind_locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    let mut symthaea_locked = symthaea_state.symthaea.lock().await;
    let symthaea = symthaea_locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Get embedding
    let hv = symthaea.embed(&content);
    let embedding = hv.values.clone();

    // Perceive in the mind
    wrapper.mind.perceive_text(&content, hv.clone());
    wrapper.mind.tick();

    // Extract structured thought (with context awareness)
    let structured_thought = wrapper.mind.extract_structured_thought();

    // Track in session
    wrapper.add_to_session(content.clone(), hv);
    wrapper.thoughts_analyzed += 1;

    // Build epistemic classification
    let epistemic = from_raw_scores(
        structured_thought.phi,
        0.0,
        0.0,
        structured_thought.phi,
        structured_thought.coherence,
    );

    // Get recent context for response
    let recent_context: Vec<String> = wrapper
        .recent_context(3)
        .into_iter()
        .map(String::from)
        .collect();

    Ok(ContextualAnalysis {
        content,
        epistemic,
        embedding,
        phi: structured_thought.phi,
        coherence: structured_thought.coherence,
        meta_awareness: structured_thought.meta_awareness,
        semantic_intent: format!("{:?}", structured_thought.semantic_intent),
        response_type: format!("{:?}", structured_thought.response_type),
        epistemic_status: format!("{:?}", structured_thought.epistemic_status),
        activated_concepts: structured_thought.activated_concepts
            .iter()
            .map(|c| c.name.clone())
            .collect(),
        recent_context,
        session_thought_count: wrapper.thoughts_analyzed,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContextualAnalysis {
    pub content: String,
    pub epistemic: EpistemicClassification,
    pub embedding: Vec<f32>,
    pub phi: f64,
    pub coherence: f64,
    pub meta_awareness: f64,
    pub semantic_intent: String,
    pub response_type: String,
    pub epistemic_status: String,
    pub activated_concepts: Vec<String>,
    pub recent_context: Vec<String>,
    pub session_thought_count: u64,
}

// ============================================================================
// CONSCIOUSNESS PROFILE
// ============================================================================

/// Get the full consciousness profile from the working memory mind
#[tauri::command]
pub async fn get_mind_consciousness_profile(
    state: State<'_, LucidMindState>,
) -> Result<ConsciousnessProfile, BridgeError> {
    let locked = state.mind.lock().await;
    let wrapper = locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    let mind_state = wrapper.mind.snapshot();

    Ok(ConsciousnessProfile {
        phi: mind_state.consciousness_level,
        meta_awareness: mind_state.meta_awareness,
        cognitive_load: mind_state.cognitive_load,
        emotional_valence: mind_state.emotional_valence as f64,
        arousal: mind_state.arousal as f64,
    })
}

/// Get detailed mind state for debugging/introspection
#[tauri::command]
pub async fn get_mind_state(
    state: State<'_, LucidMindState>,
) -> Result<MindStateInfo, BridgeError> {
    let locked = state.mind.lock().await;
    let wrapper = locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    let mind_state = wrapper.mind.snapshot();

    Ok(MindStateInfo {
        tick: mind_state.tick,
        is_active: mind_state.is_active,
        is_conscious: mind_state.is_conscious,
        consciousness_level: mind_state.consciousness_level,
        meta_awareness: mind_state.meta_awareness,
        cognitive_load: mind_state.cognitive_load,
        emotional_valence: mind_state.emotional_valence,
        arousal: mind_state.arousal,
        memory_utilization: mind_state.memory_utilization,
        time_awake_ms: mind_state.time_awake_ms,
        working_memory_size: wrapper.mind.working_memory().len(),
        session_memory_size: wrapper.session_memory.len(),
        thoughts_analyzed: wrapper.thoughts_analyzed,
        is_seeded: wrapper.mind.is_seeded(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MindStateInfo {
    pub tick: u64,
    pub is_active: bool,
    pub is_conscious: bool,
    pub consciousness_level: f64,
    pub meta_awareness: f64,
    pub cognitive_load: f64,
    pub emotional_valence: f32,
    pub arousal: f32,
    pub memory_utilization: f32,
    pub time_awake_ms: u64,
    pub working_memory_size: usize,
    pub session_memory_size: usize,
    pub thoughts_analyzed: u64,
    pub is_seeded: bool,
}

// ============================================================================
// SESSION MEMORY OPERATIONS
// ============================================================================

/// Get the current session memory contents (for debugging)
#[tauri::command]
pub async fn get_session_memory(
    state: State<'_, LucidMindState>,
) -> Result<Vec<SessionMemoryItem>, BridgeError> {
    let locked = state.mind.lock().await;
    let wrapper = locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    Ok(wrapper.session_memory.iter().enumerate().map(|(i, t)| {
        SessionMemoryItem {
            index: i,
            content: t.content.clone(),
            age_secs: t.timestamp.elapsed().as_secs(),
        }
    }).collect())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionMemoryItem {
    pub index: usize,
    pub content: String,
    pub age_secs: u64,
}

/// Run a cognitive tick on the mind (for explicit integration)
#[tauri::command]
pub async fn mind_tick(
    state: State<'_, LucidMindState>,
) -> Result<u64, BridgeError> {
    let mut locked = state.mind.lock().await;
    let wrapper = locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    wrapper.mind.tick();

    Ok(wrapper.mind.state().tick)
}

// ============================================================================
// REFERENCE RESOLUTION
// ============================================================================

/// Reference patterns to look for
const REFERENCE_PATTERNS: &[&str] = &[
    "that", "this", "it", "the above", "previous", "earlier", "mentioned",
    "the idea", "the thought", "the concept", "what i said",
];

/// Check if content contains a reference that needs resolution
fn contains_reference(content: &str) -> bool {
    let lower = content.to_lowercase();
    REFERENCE_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Resolve references in content to actual thought content
///
/// When content contains references like "that idea" or "the above",
/// this finds the most relevant recent thought and returns context.
#[tauri::command]
pub async fn resolve_reference(
    content: String,
    state: State<'_, LucidMindState>,
    symthaea_state: State<'_, SymthaeaState>,
) -> Result<ReferenceResolution, BridgeError> {
    if content.trim().is_empty() {
        return Err(BridgeError::InvalidInput("Content cannot be empty".to_string()));
    }

    let mind_locked = state.mind.lock().await;
    let wrapper = mind_locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    // Check if content contains references
    if !contains_reference(&content) {
        return Ok(ReferenceResolution {
            contains_reference: false,
            resolved_content: None,
            referenced_thought: None,
            confidence: 1.0,
        });
    }

    let mut symthaea_locked = symthaea_state.symthaea.lock().await;
    let symthaea = symthaea_locked
        .as_mut()
        .ok_or(BridgeError::NotInitialized)?;

    // Get embedding of current content
    let content_hv = symthaea.embed(&content);
    let content_embedding = &content_hv.values;

    // Find most similar thought in session memory
    let mut best_match: Option<(usize, f64, &SessionThought)> = None;

    for (idx, session_thought) in wrapper.session_memory.iter().enumerate().rev() {
        // Compute similarity
        let similarity = cosine_similarity_vec(content_embedding, &session_thought.embedding.values);

        if let Some((_, best_sim, _)) = best_match {
            if similarity > best_sim {
                best_match = Some((idx, similarity, session_thought));
            }
        } else if similarity > 0.3 {
            best_match = Some((idx, similarity, session_thought));
        }
    }

    if let Some((idx, similarity, thought)) = best_match {
        // Build resolved content by providing context
        let resolved = format!(
            "{} [Reference: \"{}\"]",
            content,
            thought.content.chars().take(100).collect::<String>()
        );

        Ok(ReferenceResolution {
            contains_reference: true,
            resolved_content: Some(resolved),
            referenced_thought: Some(ReferencedThought {
                index: idx,
                content: thought.content.clone(),
                similarity,
            }),
            confidence: similarity,
        })
    } else {
        Ok(ReferenceResolution {
            contains_reference: true,
            resolved_content: None,
            referenced_thought: None,
            confidence: 0.0,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReferenceResolution {
    pub contains_reference: bool,
    pub resolved_content: Option<String>,
    pub referenced_thought: Option<ReferencedThought>,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReferencedThought {
    pub index: usize,
    pub content: String,
    pub similarity: f64,
}

/// Helper function to compute cosine similarity between two vectors
fn cosine_similarity_vec(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        dot += (a[i] as f64) * (b[i] as f64);
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

// ============================================================================
// CONSCIOUSNESS EVOLUTION TRACKING
// ============================================================================

/// Snapshot of consciousness state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    /// When this snapshot was taken (Unix timestamp ms)
    pub timestamp: u64,
    /// Tick number when snapshot was taken
    pub tick: u64,
    /// Phi (integrated information)
    pub phi: f64,
    /// Meta-awareness level
    pub meta_awareness: f64,
    /// Cognitive load
    pub cognitive_load: f64,
    /// Emotional valence
    pub emotional_valence: f64,
    /// Arousal level
    pub arousal: f64,
    /// Number of thoughts analyzed at this point
    pub thoughts_analyzed: u64,
    /// Working memory size
    pub working_memory_size: usize,
}

/// Take a consciousness snapshot for evolution tracking
#[tauri::command]
pub async fn take_consciousness_snapshot(
    state: State<'_, LucidMindState>,
) -> Result<ConsciousnessSnapshot, BridgeError> {
    let locked = state.mind.lock().await;
    let wrapper = locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    let mind_state = wrapper.mind.snapshot();

    Ok(ConsciousnessSnapshot {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        tick: mind_state.tick,
        phi: mind_state.consciousness_level,
        meta_awareness: mind_state.meta_awareness,
        cognitive_load: mind_state.cognitive_load,
        emotional_valence: mind_state.emotional_valence as f64,
        arousal: mind_state.arousal as f64,
        thoughts_analyzed: wrapper.thoughts_analyzed,
        working_memory_size: wrapper.mind.working_memory().len(),
    })
}

/// Get a summary of consciousness evolution over the session
#[tauri::command]
pub async fn get_consciousness_evolution_summary(
    state: State<'_, LucidMindState>,
) -> Result<ConsciousnessEvolutionSummary, BridgeError> {
    let locked = state.mind.lock().await;
    let wrapper = locked
        .as_ref()
        .ok_or(BridgeError::NotInitialized)?;

    let mind_state = wrapper.mind.snapshot();

    // Calculate trends based on current state
    // In a full implementation, we'd track historical snapshots
    let phi_trend = if mind_state.consciousness_level > 0.5 {
        "increasing"
    } else if mind_state.consciousness_level > 0.3 {
        "stable"
    } else {
        "decreasing"
    };

    let cognitive_health = if mind_state.cognitive_load < 0.6 && mind_state.meta_awareness > 0.4 {
        "good"
    } else if mind_state.cognitive_load < 0.8 {
        "moderate"
    } else {
        "strained"
    };

    Ok(ConsciousnessEvolutionSummary {
        session_duration_secs: mind_state.time_awake_ms / 1000,
        total_thoughts_analyzed: wrapper.thoughts_analyzed,
        current_phi: mind_state.consciousness_level,
        current_meta_awareness: mind_state.meta_awareness,
        current_cognitive_load: mind_state.cognitive_load,
        phi_trend: phi_trend.to_string(),
        cognitive_health: cognitive_health.to_string(),
        memory_utilization: mind_state.memory_utilization as f64,
        recommendations: generate_recommendations(&mind_state, wrapper.thoughts_analyzed),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConsciousnessEvolutionSummary {
    pub session_duration_secs: u64,
    pub total_thoughts_analyzed: u64,
    pub current_phi: f64,
    pub current_meta_awareness: f64,
    pub current_cognitive_load: f64,
    pub phi_trend: String,
    pub cognitive_health: String,
    pub memory_utilization: f64,
    pub recommendations: Vec<String>,
}

/// Generate recommendations based on consciousness state
fn generate_recommendations(state: &MindState, thoughts_analyzed: u64) -> Vec<String> {
    let mut recommendations = Vec::new();

    if state.cognitive_load > 0.8 {
        recommendations.push("Consider taking a break - cognitive load is high.".to_string());
    }

    if state.meta_awareness < 0.3 && thoughts_analyzed > 10 {
        recommendations.push("Try reviewing your thoughts to increase meta-awareness.".to_string());
    }

    if state.consciousness_level < 0.4 && thoughts_analyzed > 5 {
        recommendations.push("Consider connecting related thoughts to increase integration.".to_string());
    }

    if state.memory_utilization > 0.9 {
        recommendations.push("Working memory is near capacity. Consider consolidating thoughts.".to_string());
    }

    if thoughts_analyzed > 50 && state.consciousness_level > 0.6 {
        recommendations.push("Excellent session! Your thoughts are well-integrated.".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Session is progressing well. Keep exploring!".to_string());
    }

    recommendations
}
