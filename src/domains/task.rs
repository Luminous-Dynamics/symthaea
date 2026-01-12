//! Task Domain - Semantic Concept Manipulation for MMLU Benchmarking
//!
//! This module implements Phase 2a of the Generalization Refactoring Plan:
//! A domain where states are HV16 hypervectors representing semantic concepts,
//! and actions manipulate these concepts through binding and querying primitives.
//!
//! ## Purpose
//!
//! GridWorld validated that the generalization infrastructure WORKS.
//! TaskState validates that it works for REASONING PROBLEMS (like MMLU).
//!
//! ## Key Innovation: States as Semantic Working Memory
//!
//! Unlike GridWorld where state = (x,y) position:
//! - TaskState = HV16 representing the current "thought" or working memory
//! - Actions = Primitive applications that transform the thought
//! - Goal = Semantic similarity to target answer
//!
//! This enables measuring whether Φ correlates with reasoning accuracy.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      TaskState                              │
//! │  ┌──────────────────────────────────────────────────────┐  │
//! │  │ working_memory: HV16 (current semantic state)        │  │
//! │  │ attention_focus: Vec<String> (active concepts)       │  │
//! │  │ reasoning_depth: usize (steps taken)                 │  │
//! │  │ confidence: f64 (certainty in current state)         │  │
//! │  └──────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼ Actions
//! ┌─────────────────────────────────────────────────────────────┐
//! │ ApplyPrimitive(name)  - Bind primitive to working memory    │
//! │ QueryKnowledge(query) - Search for relevant concepts        │
//! │ Compose(a, b)         - Bind two concepts together          │
//! │ Attend(concept)       - Focus attention on concept          │
//! │ Recall(pattern)       - Retrieve from episodic memory       │
//! │ Evaluate              - Check confidence in current state   │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼ Goal
//! ┌─────────────────────────────────────────────────────────────┐
//! │ SemanticGoal(target: HV16, threshold: f64)                 │
//! │   is_satisfied = working_memory.similarity(target) > thresh │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // Create initial state from question encoding
//! let question = HV16::from_text("What is the capital of France?");
//! let state = TaskState::from_question(question);
//!
//! // Apply reasoning steps
//! let action = TaskAction::ApplyPrimitive("LOCATION".into());
//! let new_state = dynamics.predict(&state, &action);
//!
//! // Check if we've reached the answer
//! let answer = HV16::from_text("Paris");
//! let goal = SemanticGoal::new(answer, 0.7);
//! assert!(goal.is_satisfied(&new_state));
//! ```

use crate::core::domain_traits::{
    Action, DomainAdapter, Goal, HdcEncodable, QualitySignal, State, WorldModel,
};
use crate::consciousness::recursive_improvement::world_model::ActionProvider;
use crate::hdc::binary_hv::HV16;
use crate::hdc::primitive_system::PrimitiveSystem;
use std::fmt;
use std::hash::{Hash, Hasher};

// ═══════════════════════════════════════════════════════════════════════════════
// TASK STATE - Semantic Working Memory
// ═══════════════════════════════════════════════════════════════════════════════

/// Task state represents the agent's current "thought" as a hypervector.
///
/// This is the semantic working memory - a single HV16 vector that encodes
/// the current understanding, modified by reasoning actions.
#[derive(Clone)]
pub struct TaskState {
    /// The current semantic representation in HV16 space
    working_memory: HV16,

    /// Concepts currently in the attention buffer
    attention_focus: Vec<String>,

    /// Number of reasoning steps taken to reach this state
    reasoning_depth: usize,

    /// Confidence in the current state (0.0 to 1.0)
    confidence: f64,

    /// History of applied primitives (for debugging/analysis)
    applied_primitives: Vec<String>,
}

impl TaskState {
    /// Create a new task state from an initial question/concept encoding
    pub fn from_question(encoding: HV16) -> Self {
        Self {
            working_memory: encoding,
            attention_focus: Vec::new(),
            reasoning_depth: 0,
            confidence: 0.5, // Start uncertain
            applied_primitives: Vec::new(),
        }
    }

    /// Create an empty task state (tabula rasa)
    pub fn empty() -> Self {
        Self {
            working_memory: HV16::zero(),
            attention_focus: Vec::new(),
            reasoning_depth: 0,
            confidence: 0.0,
            applied_primitives: Vec::new(),
        }
    }

    /// Get the current working memory hypervector
    pub fn working_memory(&self) -> &HV16 {
        &self.working_memory
    }

    /// Get the current attention focus
    pub fn attention(&self) -> &[String] {
        &self.attention_focus
    }

    /// Get reasoning depth
    pub fn depth(&self) -> usize {
        self.reasoning_depth
    }

    /// Get confidence
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Get applied primitives
    pub fn history(&self) -> &[String] {
        &self.applied_primitives
    }

    /// Create a new state by applying a primitive to working memory
    pub fn with_primitive_applied(&self, primitive_name: &str, primitive_hv: &HV16) -> Self {
        // Bind the primitive to current working memory
        let new_memory = self.working_memory.bind(primitive_hv);

        let mut new_focus = self.attention_focus.clone();
        new_focus.push(primitive_name.to_string());
        if new_focus.len() > 5 {
            new_focus.remove(0); // Keep attention buffer bounded
        }

        let mut new_history = self.applied_primitives.clone();
        new_history.push(primitive_name.to_string());

        Self {
            working_memory: new_memory,
            attention_focus: new_focus,
            reasoning_depth: self.reasoning_depth + 1,
            confidence: (self.confidence * 0.9 + 0.1).min(0.95), // Slight confidence boost
            applied_primitives: new_history,
        }
    }

    /// Create a new state by bundling (averaging) with another concept
    pub fn with_concept_bundled(&self, concept_hv: &HV16) -> Self {
        let new_memory = HV16::bundle(&[self.working_memory.clone(), concept_hv.clone()]);

        Self {
            working_memory: new_memory,
            attention_focus: self.attention_focus.clone(),
            reasoning_depth: self.reasoning_depth + 1,
            confidence: self.confidence * 0.95, // Slight confidence loss from mixing
            applied_primitives: self.applied_primitives.clone(),
        }
    }

    /// Update confidence based on similarity to a target
    pub fn with_confidence_update(&self, target: &HV16) -> Self {
        let similarity = self.working_memory.similarity(target);
        let new_confidence = (self.confidence + similarity as f64) / 2.0;

        Self {
            working_memory: self.working_memory.clone(),
            attention_focus: self.attention_focus.clone(),
            reasoning_depth: self.reasoning_depth,
            confidence: new_confidence,
            applied_primitives: self.applied_primitives.clone(),
        }
    }
}

impl fmt::Debug for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskState")
            .field("reasoning_depth", &self.reasoning_depth)
            .field("confidence", &self.confidence)
            .field("attention", &self.attention_focus)
            .field("primitives_applied", &self.applied_primitives.len())
            .finish()
    }
}

impl State for TaskState {
    fn to_features(&self) -> Vec<f64> {
        // Convert HV16 to feature vector
        // HV16 is binary, so we convert bits to f64
        let mut features = Vec::with_capacity(HV16::DIM + 2);

        // Add metadata features
        features.push(self.reasoning_depth as f64 / 100.0); // Normalized
        features.push(self.confidence);

        // Add HV16 features (first 1024 bits for efficiency)
        // Access internal bytes through the pub field .0
        for byte in self.working_memory.0.iter().take(128) {
            for bit in 0..8 {
                features.push(if (byte >> bit) & 1 == 1 { 1.0 } else { 0.0 });
            }
        }

        features
    }

    fn distance(&self, other: &Self) -> f64 {
        // Hamming distance normalized to [0, 1]
        let hamming = self.working_memory.hamming_distance(&other.working_memory);
        hamming as f64 / HV16::DIM as f64
    }

    fn is_equivalent(&self, other: &Self, tolerance: f64) -> bool {
        // Use semantic similarity for equivalence
        let similarity = self.working_memory.similarity(&other.working_memory);
        (1.0 - similarity as f64) < tolerance
    }
}

impl HdcEncodable for TaskState {
    type HyperVector = HV16;

    fn to_hv(&self) -> Self::HyperVector {
        self.working_memory.clone()
    }

    fn from_hv(hv: &Self::HyperVector) -> Option<Self> {
        Some(Self {
            working_memory: hv.clone(),
            attention_focus: Vec::new(),
            reasoning_depth: 0,
            confidence: 0.5,
            applied_primitives: Vec::new(),
        })
    }

    fn semantic_similarity(&self, other: &Self) -> f64 {
        self.working_memory.similarity(&other.working_memory) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TASK ACTION - Semantic Manipulation Operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Actions available in the task domain
#[derive(Clone, Debug)]
pub enum TaskAction {
    /// Apply a primitive by name (binds to working memory)
    ApplyPrimitive(String),

    /// Query knowledge base for relevant concept (returns bundled result)
    QueryKnowledge(String),

    /// Compose two primitives together (a ⊗ b)
    Compose(String, String),

    /// Focus attention on a specific concept
    Attend(String),

    /// Recall from episodic memory (pattern match)
    Recall(String),

    /// Evaluate current state (updates confidence)
    Evaluate,

    /// Reset working memory to baseline
    Reset,

    /// No operation (for passing)
    Noop,
}

impl PartialEq for TaskAction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ApplyPrimitive(a), Self::ApplyPrimitive(b)) => a == b,
            (Self::QueryKnowledge(a), Self::QueryKnowledge(b)) => a == b,
            (Self::Compose(a1, a2), Self::Compose(b1, b2)) => a1 == b1 && a2 == b2,
            (Self::Attend(a), Self::Attend(b)) => a == b,
            (Self::Recall(a), Self::Recall(b)) => a == b,
            (Self::Evaluate, Self::Evaluate) => true,
            (Self::Reset, Self::Reset) => true,
            (Self::Noop, Self::Noop) => true,
            _ => false,
        }
    }
}

impl Eq for TaskAction {}

impl Hash for TaskAction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::ApplyPrimitive(name) => name.hash(state),
            Self::QueryKnowledge(query) => query.hash(state),
            Self::Compose(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            Self::Attend(concept) => concept.hash(state),
            Self::Recall(pattern) => pattern.hash(state),
            _ => {}
        }
    }
}

impl Action for TaskAction {
    fn action_id(&self) -> u64 {
        // Generate deterministic ID based on action type and content
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        std::hash::Hasher::finish(&hasher)
    }

    fn describe(&self) -> String {
        match self {
            Self::ApplyPrimitive(name) => format!("Apply primitive '{}'", name),
            Self::QueryKnowledge(query) => format!("Query knowledge: '{}'", query),
            Self::Compose(a, b) => format!("Compose {} ⊗ {}", a, b),
            Self::Attend(concept) => format!("Attend to '{}'", concept),
            Self::Recall(pattern) => format!("Recall pattern '{}'", pattern),
            Self::Evaluate => "Evaluate current state".to_string(),
            Self::Reset => "Reset working memory".to_string(),
            Self::Noop => "No operation".to_string(),
        }
    }

    fn cost(&self) -> f64 {
        match self {
            Self::ApplyPrimitive(_) => 1.0,
            Self::QueryKnowledge(_) => 2.0, // More expensive
            Self::Compose(_, _) => 1.5,
            Self::Attend(_) => 0.5,
            Self::Recall(_) => 2.0,
            Self::Evaluate => 0.1,
            Self::Reset => 0.0,
            Self::Noop => 0.0,
        }
    }

    fn is_reversible(&self) -> bool {
        match self {
            Self::Reset => false, // Can't undo reset
            _ => true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION PROVIDER - Enumerate available actions (static)
// ═══════════════════════════════════════════════════════════════════════════════

impl ActionProvider<TaskAction> for TaskAction {
    fn all_actions() -> Vec<TaskAction> {
        let mut actions = Vec::new();

        // Always available
        actions.push(TaskAction::Evaluate);
        actions.push(TaskAction::Noop);
        actions.push(TaskAction::Reset);

        // Get core primitives for MMLU-style reasoning
        let core_primitives = vec![
            // Logical
            "IMPLIES",
            "AND",
            "OR",
            "NOT",
            "IF_THEN",
            "EQUALS",
            // Causal
            "CAUSES",
            "ENABLES",
            "PREVENTS",
            // Relational
            "IS_A",
            "PART_OF",
            "SIMILAR_TO",
            // Quantifiers
            "ALL",
            "SOME",
            "NONE",
            // Temporal
            "BEFORE",
            "AFTER",
            "DURING",
            // Knowledge
            "KNOWS",
            "BELIEVES",
            "WANTS",
        ];

        for prim in core_primitives {
            actions.push(TaskAction::ApplyPrimitive(prim.to_string()));
        }

        actions
    }
}

impl TaskState {
    /// Get available actions from the current state context
    pub fn get_available_actions(&self) -> Vec<TaskAction> {
        let mut actions = TaskAction::all_actions();

        // Add composition actions if we have enough attention focus
        if self.attention_focus.len() >= 2 {
            let last = self.attention_focus.last().unwrap();
            let second_last = &self.attention_focus[self.attention_focus.len() - 2];
            actions.push(TaskAction::Compose(
                second_last.clone(),
                last.clone(),
            ));
        }

        actions
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC GOAL - Target answer similarity
// ═══════════════════════════════════════════════════════════════════════════════

/// Goal based on semantic similarity to target answer
#[derive(Clone, Debug)]
pub struct SemanticGoal {
    /// Target answer encoding
    target: HV16,

    /// Similarity threshold for satisfaction
    threshold: f64,

    /// Human-readable description of the goal
    description: String,

    /// Maximum reasoning depth allowed
    max_depth: usize,
}

impl SemanticGoal {
    /// Create a new semantic goal
    pub fn new(target: HV16, threshold: f64) -> Self {
        Self {
            target,
            threshold,
            description: "Reach semantic similarity threshold".to_string(),
            max_depth: 50,
        }
    }

    /// Create goal with description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set maximum reasoning depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Get the target hypervector
    pub fn target(&self) -> &HV16 {
        &self.target
    }

    /// Get the similarity threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Goal<TaskState> for SemanticGoal {
    fn is_satisfied(&self, state: &TaskState) -> bool {
        let similarity = state.working_memory.similarity(&self.target) as f64;
        similarity >= self.threshold
    }

    fn distance_to_goal(&self, state: &TaskState) -> f64 {
        let similarity = state.working_memory.similarity(&self.target) as f64;
        // Distance is inverse of similarity, bounded to [0, 1]
        (self.threshold - similarity).max(0.0).min(1.0)
    }

    fn reward(&self, state: &TaskState) -> f64 {
        if self.is_satisfied(state) {
            1.0
        } else if state.reasoning_depth > self.max_depth {
            -1.0 // Penalty for taking too long
        } else {
            // Reward progress toward goal
            let similarity = state.working_memory.similarity(&self.target) as f64;
            similarity - self.threshold // Can be negative
        }
    }

    fn priority(&self) -> f64 {
        1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TASK DYNAMICS - How actions transform states
// ═══════════════════════════════════════════════════════════════════════════════

/// Dynamics predictor for the task domain
pub struct TaskDynamics {
    /// Reference to the primitive system
    primitive_cache: std::collections::HashMap<String, HV16>,
}

impl TaskDynamics {
    /// Create new task dynamics (caches primitive encodings)
    pub fn new() -> Self {
        let system = PrimitiveSystem::global();
        let mut cache = std::collections::HashMap::new();

        // Cache commonly used primitives
        for name in [
            "IMPLIES", "AND", "OR", "NOT", "IF_THEN", "EQUALS", "CAUSES", "ENABLES",
            "PREVENTS", "IS_A", "PART_OF", "SIMILAR_TO", "ALL", "SOME", "NONE",
            "BEFORE", "AFTER", "DURING", "KNOWS", "BELIEVES", "WANTS",
        ] {
            if let Some(prim) = system.get(name) {
                cache.insert(name.to_string(), prim.encoding.clone());
            }
        }

        Self {
            primitive_cache: cache,
        }
    }

    /// Get or create a hypervector for a concept name
    fn get_concept_hv(&self, name: &str) -> HV16 {
        if let Some(hv) = self.primitive_cache.get(name) {
            hv.clone()
        } else {
            // Generate deterministic HV from name
            use std::hash::Hash;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut hasher);
            HV16::random(std::hash::Hasher::finish(&hasher))
        }
    }
}

impl Default for TaskDynamics {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldModel<TaskState, TaskAction> for TaskDynamics {
    fn predict(&self, state: &TaskState, action: &TaskAction) -> TaskState {
        match action {
            TaskAction::ApplyPrimitive(name) => {
                let prim_hv = self.get_concept_hv(name);
                state.with_primitive_applied(name, &prim_hv)
            }

            TaskAction::QueryKnowledge(query) => {
                // Generate query-based concept and bundle with working memory
                let query_hv = self.get_concept_hv(query);
                state.with_concept_bundled(&query_hv)
            }

            TaskAction::Compose(a, b) => {
                // Bind two concepts together, then apply to state
                let hv_a = self.get_concept_hv(a);
                let hv_b = self.get_concept_hv(b);
                let composed = hv_a.bind(&hv_b);
                state.with_primitive_applied(&format!("{}⊗{}", a, b), &composed)
            }

            TaskAction::Attend(concept) => {
                // Update attention buffer without changing working memory
                let mut new_state = state.clone();
                new_state.attention_focus.push(concept.clone());
                if new_state.attention_focus.len() > 5 {
                    new_state.attention_focus.remove(0);
                }
                new_state
            }

            TaskAction::Recall(pattern) => {
                // Pattern match against working memory, bundle with result
                let pattern_hv = self.get_concept_hv(pattern);
                state.with_concept_bundled(&pattern_hv)
            }

            TaskAction::Evaluate => {
                // Just update confidence based on current state
                let mut new_state = state.clone();
                // Use working memory density as confidence proxy
                let density = state.working_memory.popcount() as f64 / HV16::DIM as f64;
                new_state.confidence = (density * 2.0 - 1.0).abs(); // Near 0.5 = high confidence
                new_state
            }

            TaskAction::Reset => TaskState::empty(),

            TaskAction::Noop => state.clone(),
        }
    }

    fn confidence(&self, _state: &TaskState, action: &TaskAction) -> f64 {
        // Higher confidence for well-known actions
        match action {
            TaskAction::ApplyPrimitive(name) => {
                if self.primitive_cache.contains_key(name) {
                    0.9
                } else {
                    0.6
                }
            }
            TaskAction::Evaluate | TaskAction::Noop | TaskAction::Reset => 1.0,
            _ => 0.7,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN ADAPTER - Pluggable domain configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Task domain adapter for the planning infrastructure
pub struct TaskDomainAdapter {
    /// Dynamics model
    dynamics: TaskDynamics,
}

impl TaskDomainAdapter {
    pub fn new() -> Self {
        Self {
            dynamics: TaskDynamics::new(),
        }
    }

    /// Get reference to dynamics
    pub fn dynamics(&self) -> &TaskDynamics {
        &self.dynamics
    }
}

impl Default for TaskDomainAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainAdapter<TaskState, TaskAction> for TaskDomainAdapter {
    fn domain_name(&self) -> &'static str {
        "task"
    }

    fn available_actions(&self, state: &TaskState) -> Vec<TaskAction> {
        state.get_available_actions()
    }

    fn initial_state(&self) -> TaskState {
        TaskState::empty()
    }

    fn quality_signals(&self) -> Vec<Box<dyn QualitySignal<TaskState>>> {
        vec![
            Box::new(ConfidenceSignal),
            Box::new(CoherenceSignal),
            Box::new(DepthSignal),
        ]
    }

    fn is_valid_state(&self, state: &TaskState) -> bool {
        // States are always valid (HV16 is always well-formed)
        state.reasoning_depth < 1000 // Sanity limit
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUALITY SIGNALS - Metrics for task reasoning
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence quality signal
struct ConfidenceSignal;

impl QualitySignal<TaskState> for ConfidenceSignal {
    fn measure(&self, state: &TaskState) -> f64 {
        state.confidence
    }

    fn name(&self) -> &'static str {
        "confidence"
    }

    fn weight(&self) -> f64 {
        1.0
    }
}

/// Coherence quality signal (based on HV16 structure)
struct CoherenceSignal;

impl QualitySignal<TaskState> for CoherenceSignal {
    fn measure(&self, state: &TaskState) -> f64 {
        // Measure coherence as deviation from random (50% density)
        let density = state.working_memory.popcount() as f64 / HV16::DIM as f64;
        // Coherent states have biased density
        (density - 0.5).abs() * 2.0
    }

    fn name(&self) -> &'static str {
        "coherence"
    }

    fn weight(&self) -> f64 {
        0.5
    }
}

/// Reasoning depth signal (penalizes very long chains)
struct DepthSignal;

impl QualitySignal<TaskState> for DepthSignal {
    fn measure(&self, state: &TaskState) -> f64 {
        // Quality decreases with excessive depth
        let optimal_depth = 5.0;
        let depth = state.reasoning_depth as f64;
        (1.0 - (depth - optimal_depth).abs() / optimal_depth).max(0.0)
    }

    fn name(&self) -> &'static str {
        "depth_efficiency"
    }

    fn weight(&self) -> f64 {
        0.3
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state_creation() {
        let state = TaskState::empty();
        assert_eq!(state.depth(), 0);
        assert!(state.attention().is_empty());
        assert_eq!(state.confidence(), 0.0);
    }

    #[test]
    fn test_task_state_from_question() {
        let hv = HV16::random(42);
        let state = TaskState::from_question(hv.clone());
        assert_eq!(state.depth(), 0);
        assert_eq!(state.confidence(), 0.5);
        assert_eq!(state.working_memory().hamming_distance(&hv), 0);
    }

    #[test]
    fn test_primitive_application() {
        let state = TaskState::from_question(HV16::random(42));
        let prim_hv = HV16::random(123);
        let new_state = state.with_primitive_applied("TEST", &prim_hv);

        assert_eq!(new_state.depth(), 1);
        assert_eq!(new_state.history().len(), 1);
        assert!(new_state.attention().contains(&"TEST".to_string()));
    }

    #[test]
    fn test_state_distance() {
        let state1 = TaskState::from_question(HV16::random(1));
        let state2 = TaskState::from_question(HV16::random(2));
        let state_same = TaskState::from_question(HV16::random(1));

        // Same seed = same HV = zero distance
        assert_eq!(state1.distance(&state_same), 0.0);

        // Different seeds = different HVs = some distance
        let dist = state1.distance(&state2);
        assert!(dist > 0.3 && dist < 0.7); // Random HVs are ~0.5 distance
    }

    #[test]
    fn test_semantic_similarity() {
        let state1 = TaskState::from_question(HV16::random(42));
        let state2 = TaskState::from_question(HV16::random(42));
        let state3 = TaskState::from_question(HV16::random(99));

        assert!((state1.semantic_similarity(&state2) - 1.0).abs() < 0.01);
        assert!(state1.semantic_similarity(&state3) < 0.6);
    }

    #[test]
    fn test_task_action_properties() {
        let action = TaskAction::ApplyPrimitive("IMPLIES".into());
        assert!(action.action_id() != 0);
        assert!(action.describe().contains("IMPLIES"));
        assert_eq!(action.cost(), 1.0);
        assert!(action.is_reversible());

        let reset = TaskAction::Reset;
        assert!(!reset.is_reversible());
    }

    #[test]
    fn test_action_provider() {
        // Test static all_actions()
        let actions = TaskAction::all_actions();
        assert!(actions.contains(&TaskAction::Evaluate));
        assert!(actions.contains(&TaskAction::Noop));
        assert!(actions.iter().any(|a| matches!(a, TaskAction::ApplyPrimitive(_))));

        // Test state-aware get_available_actions()
        let state = TaskState::empty();
        let state_actions = state.get_available_actions();
        assert!(state_actions.contains(&TaskAction::Evaluate));
    }

    #[test]
    fn test_semantic_goal() {
        let target = HV16::random(42);
        let goal = SemanticGoal::new(target.clone(), 0.9);

        let matching_state = TaskState::from_question(target);
        assert!(goal.is_satisfied(&matching_state));
        assert!(goal.distance_to_goal(&matching_state) < 0.1);

        let different_state = TaskState::from_question(HV16::random(99));
        assert!(!goal.is_satisfied(&different_state));
        assert!(goal.distance_to_goal(&different_state) > 0.3);
    }

    #[test]
    fn test_task_dynamics() {
        let dynamics = TaskDynamics::new();
        let state = TaskState::from_question(HV16::random(42));

        // Test ApplyPrimitive
        let action = TaskAction::ApplyPrimitive("AND".into());
        let new_state = dynamics.predict(&state, &action);
        assert_eq!(new_state.depth(), 1);
        assert!(new_state.history().contains(&"AND".to_string()));

        // Test Noop
        let noop_state = dynamics.predict(&state, &TaskAction::Noop);
        assert_eq!(noop_state.depth(), state.depth());

        // Test Reset
        let reset_state = dynamics.predict(&state, &TaskAction::Reset);
        assert_eq!(reset_state.depth(), 0);
        assert_eq!(reset_state.confidence(), 0.0);
    }

    #[test]
    fn test_domain_adapter() {
        let adapter = TaskDomainAdapter::new();

        assert_eq!(adapter.domain_name(), "task");

        let initial = adapter.initial_state();
        assert_eq!(initial.depth(), 0);

        let signals = adapter.quality_signals();
        assert!(signals.len() >= 2);
    }

    #[test]
    fn test_goal_reward_signal() {
        let target = HV16::random(42);
        let goal = SemanticGoal::new(target.clone(), 0.7);

        // Matching state should get positive reward
        let good_state = TaskState::from_question(target);
        assert!(goal.reward(&good_state) > 0.0);

        // State close to target should get progress reward
        let mut partial_state = TaskState::from_question(HV16::random(1));
        // Can't easily make a "close" state without more machinery
        // But we can verify the reward logic exists
        let reward = goal.reward(&partial_state);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_hdc_encodable_trait() {
        let state = TaskState::from_question(HV16::random(42));

        let hv = state.to_hv();
        assert_eq!(hv.hamming_distance(state.working_memory()), 0);

        let recovered = TaskState::from_hv(&hv).unwrap();
        assert_eq!(recovered.semantic_similarity(&state), 1.0);
    }

    #[test]
    fn test_reasoning_chain() {
        let dynamics = TaskDynamics::new();
        let mut state = TaskState::from_question(HV16::random(42));

        // Apply a sequence of reasoning steps
        let actions = [
            TaskAction::ApplyPrimitive("IS_A".into()),
            TaskAction::ApplyPrimitive("CAUSES".into()),
            TaskAction::ApplyPrimitive("IMPLIES".into()),
            TaskAction::Evaluate,
        ];

        for action in actions {
            state = dynamics.predict(&state, &action);
        }

        assert_eq!(state.depth(), 3); // Only ApplyPrimitive increments depth (Evaluate is assessment)
        assert_eq!(state.history().len(), 3); // Evaluate doesn't add to history
    }
}
