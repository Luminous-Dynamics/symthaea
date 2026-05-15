// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Theory Consciousness: Unified Framework Beyond IIT
//!
//! This module integrates multiple consciousness theories into a unified architecture:
//!
//! 1. **Global Workspace Theory (GWT)** - Baars: Conscious broadcast to all subsystems
//! 2. **Attention Schema Theory (AST)** - Graziano: Self-model of attention
//! 3. **Predictive Processing (PP)** - Clark/Friston: Prediction-error minimization
//! 4. **Recurrent Processing Theory (RPT)** - Lamme: Feedback loops for awareness
//! 5. **Embodied Cognition (4E)** - Varela: Grounded in system state
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    MULTI-THEORY CONSCIOUSNESS                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
//! │  │   GLOBAL    │    │  ATTENTION  │    │ PREDICTIVE  │                │
//! │  │  WORKSPACE  │◄──►│   SCHEMA    │◄──►│ PROCESSING  │                │
//! │  │   (GWT)     │    │   (AST)     │    │    (PP)     │                │
//! │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │
//! │         │                  │                  │                        │
//! │         └──────────────────┼──────────────────┘                        │
//! │                            │                                           │
//! │                    ┌───────▼───────┐                                   │
//! │                    │   RECURRENT   │                                   │
//! │                    │   PROCESSING  │                                   │
//! │                    │     (RPT)     │                                   │
//! │                    └───────┬───────┘                                   │
//! │                            │                                           │
//! │                    ┌───────▼───────┐                                   │
//! │                    │   EMBODIED    │                                   │
//! │                    │   COGNITION   │                                   │
//! │                    │  (NixOS Body) │                                   │
//! │                    └───────────────┘                                   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Theory Integration
//!
//! Each theory contributes a unique perspective:
//! - **IIT (Φ)**: How integrated is the processing? (already in codebase)
//! - **GWT**: What enters the global workspace for broadcast?
//! - **AST**: What does the system think it's attending to?
//! - **PP**: How well do predictions match reality?
//! - **RPT**: How many recurrent passes refine understanding?
//! - **4E**: How grounded is understanding in actual system state?

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime};
use symthaea_core::hdc::universal_semantics::SemanticPrime;

// ============================================================================
// GLOBAL WORKSPACE THEORY (GWT) - Baars
// ============================================================================
// "Consciousness is a global broadcast of information to all brain regions"

/// A message that can be broadcast to the global workspace
#[derive(Debug, Clone)]
pub struct WorkspaceMessage {
    /// Unique identifier
    pub id: u64,
    /// Source subsystem
    pub source: SubsystemId,
    /// Content summary
    pub content: String,
    /// Semantic primes involved
    pub primes: Vec<SemanticPrime>,
    /// Salience (importance/urgency)
    pub salience: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Identifier for cognitive subsystems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubsystemId {
    /// Language understanding
    Language,
    /// Intent recognition
    Intent,
    /// Semantic processing
    Semantics,
    /// Memory systems
    Memory,
    /// Action planning
    Action,
    /// Error detection
    Error,
    /// Prediction engine
    Prediction,
    /// Embodied state
    Embodiment,
}

impl SubsystemId {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Language => "Language",
            Self::Intent => "Intent",
            Self::Semantics => "Semantics",
            Self::Memory => "Memory",
            Self::Action => "Action",
            Self::Error => "Error",
            Self::Prediction => "Prediction",
            Self::Embodiment => "Embodiment",
        }
    }

    /// All subsystems
    pub fn all() -> Vec<SubsystemId> {
        vec![
            Self::Language,
            Self::Intent,
            Self::Semantics,
            Self::Memory,
            Self::Action,
            Self::Error,
            Self::Prediction,
            Self::Embodiment,
        ]
    }
}

/// Response from a subsystem to a broadcast
#[derive(Debug, Clone)]
pub struct SubsystemResponse {
    /// Responding subsystem
    pub subsystem: SubsystemId,
    /// Relevance score (0-1)
    pub relevance: f64,
    /// Response content
    pub response: String,
    /// Vote: should this stay in workspace?
    pub vote_keep: bool,
}

/// The Global Workspace - central conscious broadcast mechanism
#[derive(Debug, Clone)]
pub struct GlobalWorkspace {
    /// Current content in the workspace (limited capacity)
    contents: VecDeque<WorkspaceMessage>,
    /// Maximum workspace capacity
    capacity: usize,
    /// Broadcast history
    broadcast_history: VecDeque<BroadcastEvent>,
    /// Subsystem response accumulator
    pending_responses: HashMap<u64, Vec<SubsystemResponse>>,
    /// Message ID counter
    next_id: u64,
    /// Competition threshold (salience needed to enter)
    competition_threshold: f64,
}

/// Record of a broadcast event
#[derive(Debug, Clone)]
pub struct BroadcastEvent {
    /// Message that was broadcast
    pub message: WorkspaceMessage,
    /// Responses from subsystems
    pub responses: Vec<SubsystemResponse>,
    /// Total relevance (sum of subsystem relevances)
    pub total_relevance: f64,
    /// Whether message was retained
    pub retained: bool,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self {
            contents: VecDeque::with_capacity(7), // Miller's 7±2
            capacity: 7,
            broadcast_history: VecDeque::with_capacity(100),
            pending_responses: HashMap::new(),
            next_id: 0,
            competition_threshold: 0.3,
        }
    }
}

impl GlobalWorkspace {
    /// Create a new global workspace
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a message for potential broadcast (competition)
    pub fn submit(
        &mut self,
        source: SubsystemId,
        content: String,
        primes: Vec<SemanticPrime>,
        salience: f64,
    ) -> Option<u64> {
        // Competition: only high-salience messages enter
        if salience < self.competition_threshold {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        let message = WorkspaceMessage {
            id,
            source,
            content,
            primes,
            salience,
            timestamp: SystemTime::now(),
        };

        // If at capacity, evict lowest salience
        if self.contents.len() >= self.capacity {
            // Find minimum salience
            let min_idx = self
                .contents
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.salience
                        .partial_cmp(&b.salience)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(idx) = min_idx {
                if self.contents[idx].salience < salience {
                    self.contents.remove(idx);
                } else {
                    return None; // New message not salient enough
                }
            }
        }

        self.contents.push_back(message);
        self.pending_responses.insert(id, Vec::new());
        Some(id)
    }

    /// Broadcast current workspace to all subsystems (returns messages to broadcast)
    pub fn get_broadcast(&self) -> Vec<&WorkspaceMessage> {
        self.contents.iter().collect()
    }

    /// Receive response from a subsystem
    pub fn receive_response(&mut self, message_id: u64, response: SubsystemResponse) {
        if let Some(responses) = self.pending_responses.get_mut(&message_id) {
            responses.push(response);
        }
    }

    /// Process responses and update workspace
    pub fn process_responses(&mut self) {
        let mut to_remove = Vec::new();

        for (id, responses) in self.pending_responses.drain() {
            if let Some(msg_idx) = self.contents.iter().position(|m| m.id == id) {
                let message = self.contents[msg_idx].clone();
                let total_relevance: f64 = responses.iter().map(|r| r.relevance).sum();
                let votes_keep: usize = responses.iter().filter(|r| r.vote_keep).count();
                let retained = votes_keep > responses.len() / 2;

                // Record broadcast event
                let event = BroadcastEvent {
                    message: message.clone(),
                    responses: responses.clone(),
                    total_relevance,
                    retained,
                    timestamp: SystemTime::now(),
                };

                if self.broadcast_history.len() >= 100 {
                    self.broadcast_history.pop_front();
                }
                self.broadcast_history.push_back(event);

                if !retained {
                    to_remove.push(msg_idx);
                }
            }
        }

        // Remove non-retained messages (in reverse order to preserve indices)
        to_remove.sort();
        to_remove.reverse();
        for idx in to_remove {
            self.contents.remove(idx);
        }
    }

    /// Get current workspace contents
    pub fn contents(&self) -> &VecDeque<WorkspaceMessage> {
        &self.contents
    }

    /// Get broadcast history
    pub fn history(&self) -> &VecDeque<BroadcastEvent> {
        &self.broadcast_history
    }

    /// Calculate GWT consciousness metric (workspace utilization × average salience)
    pub fn gwt_metric(&self) -> f64 {
        if self.contents.is_empty() {
            return 0.0;
        }
        let utilization = self.contents.len() as f64 / self.capacity as f64;
        let avg_salience =
            self.contents.iter().map(|m| m.salience).sum::<f64>() / self.contents.len() as f64;
        utilization * avg_salience
    }
}

// ============================================================================
// ATTENTION SCHEMA THEORY (AST) - Graziano
// ============================================================================
// "Consciousness is a model of attention - we're aware of what we think we're attending to"

/// What the system believes it's currently attending to
#[derive(Debug, Clone)]
pub struct AttentionFocus {
    /// Target of attention
    pub target: String,
    /// Why this target (reason)
    pub reason: String,
    /// Confidence in this being the focus
    pub confidence: f64,
    /// Semantic category
    pub category: AttentionCategory,
    /// Duration of attention
    pub duration: Duration,
    /// Start time
    pub started: SystemTime,
}

/// Categories of attention targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionCategory {
    /// Attending to user intent
    Intent,
    /// Attending to a specific word/phrase
    Word,
    /// Attending to a package/service name
    Entity,
    /// Attending to an error/problem
    Error,
    /// Attending to a prediction mismatch
    Surprise,
    /// Attending to memory/context
    Memory,
    /// Attending to system state
    SystemState,
}

impl AttentionCategory {
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Intent => "understanding what you want to do",
            Self::Word => "processing a specific word or phrase",
            Self::Entity => "identifying a package, service, or resource",
            Self::Error => "noticing something that might be wrong",
            Self::Surprise => "something unexpected that needs attention",
            Self::Memory => "recalling relevant context",
            Self::SystemState => "checking the actual system state",
        }
    }
}

/// The Attention Schema - self-model of attention
#[derive(Debug, Clone)]
pub struct AttentionSchema {
    /// Current primary focus
    pub primary_focus: Option<AttentionFocus>,
    /// Secondary foci (peripheral attention)
    pub secondary_foci: Vec<AttentionFocus>,
    /// Attention history
    history: VecDeque<AttentionFocus>,
    /// Maximum history size
    max_history: usize,
    /// Attention shift count (higher = more scattered)
    shift_count: usize,
}

impl Default for AttentionSchema {
    fn default() -> Self {
        Self {
            primary_focus: None,
            secondary_foci: Vec::new(),
            history: VecDeque::with_capacity(50),
            max_history: 50,
            shift_count: 0,
        }
    }
}

impl AttentionSchema {
    /// Create a new attention schema
    pub fn new() -> Self {
        Self::default()
    }

    /// Set primary attention focus
    pub fn focus_on(
        &mut self,
        target: &str,
        reason: &str,
        confidence: f64,
        category: AttentionCategory,
    ) {
        // Move current focus to history
        if let Some(current) = self.primary_focus.take() {
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back(current);
            self.shift_count += 1;
        }

        self.primary_focus = Some(AttentionFocus {
            target: target.to_string(),
            reason: reason.to_string(),
            confidence,
            category,
            duration: Duration::from_secs(0),
            started: SystemTime::now(),
        });
    }

    /// Add secondary (peripheral) focus
    pub fn add_secondary(
        &mut self,
        target: &str,
        reason: &str,
        confidence: f64,
        category: AttentionCategory,
    ) {
        // Limit secondary foci
        if self.secondary_foci.len() >= 3 {
            self.secondary_foci.remove(0);
        }

        self.secondary_foci.push(AttentionFocus {
            target: target.to_string(),
            reason: reason.to_string(),
            confidence,
            category,
            duration: Duration::from_secs(0),
            started: SystemTime::now(),
        });
    }

    /// Generate natural language description of attention state
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();

        if let Some(focus) = &self.primary_focus {
            parts.push(format!(
                "I'm {} - specifically looking at \"{}\"",
                focus.category.description(),
                focus.target
            ));
        }

        for secondary in &self.secondary_foci {
            parts.push(format!(
                "I also noticed \"{}\" ({})",
                secondary.target, secondary.reason
            ));
        }

        if parts.is_empty() {
            "I'm not focused on anything specific right now.".to_string()
        } else {
            parts.join(". ")
        }
    }

    /// Get attention stability (inverse of shift rate)
    pub fn stability(&self) -> f64 {
        if self.shift_count == 0 {
            return 1.0;
        }
        let history_time = self
            .history
            .iter()
            .filter_map(|f| f.started.elapsed().ok())
            .map(|d| d.as_secs_f64())
            .sum::<f64>();

        if history_time == 0.0 {
            return 1.0;
        }

        // Shifts per second, inverted and clamped
        let shift_rate = self.shift_count as f64 / history_time.max(1.0);
        (1.0 / (1.0 + shift_rate)).clamp(0.0, 1.0)
    }

    /// Calculate AST metric (attention coherence)
    pub fn ast_metric(&self) -> f64 {
        let focus_confidence = self
            .primary_focus
            .as_ref()
            .map(|f| f.confidence)
            .unwrap_or(0.0);
        let stability = self.stability();
        let has_focus = if self.primary_focus.is_some() {
            0.5
        } else {
            0.0
        };

        (focus_confidence * 0.4 + stability * 0.3 + has_focus * 0.3).clamp(0.0, 1.0)
    }

    /// Reset shift counter
    pub fn reset_shifts(&mut self) {
        self.shift_count = 0;
    }
}

// ============================================================================
// PREDICTIVE PROCESSING (PP) - Clark/Friston
// ============================================================================
// "The brain is a prediction machine - consciousness is prediction error"

/// A prediction about what to expect
#[derive(Debug, Clone)]
pub struct Prediction {
    /// What we predict
    pub expected: String,
    /// Confidence in prediction
    pub confidence: f64,
    /// Source of prediction
    pub source: PredictionSource,
    /// Semantic primes expected
    pub expected_primes: Vec<SemanticPrime>,
    /// When made
    pub timestamp: SystemTime,
}

/// Sources of predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionSource {
    /// Based on past experience
    Experience,
    /// Based on current context
    Context,
    /// Based on user patterns
    UserModel,
    /// Based on language patterns
    Language,
    /// Based on NixOS knowledge
    Domain,
}

/// Result of comparing prediction to reality
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// The original prediction
    pub prediction: Prediction,
    /// What actually occurred
    pub actual: String,
    /// Actual primes found
    pub actual_primes: Vec<SemanticPrime>,
    /// Error magnitude (0 = perfect prediction, 1 = completely wrong)
    pub error: f64,
    /// Surprise (log of inverse probability)
    pub surprise: f64,
}

/// The Predictive Processing engine
#[derive(Debug, Clone)]
pub struct PredictiveProcessor {
    /// Active predictions
    active_predictions: Vec<Prediction>,
    /// Prediction error history
    error_history: VecDeque<PredictionError>,
    /// Running prediction accuracy
    accuracy_window: VecDeque<f64>,
    /// Maximum history sizes
    max_history: usize,
    /// Prior beliefs (learned patterns)
    priors: HashMap<String, Vec<(String, f64)>>, // context -> [(prediction, confidence)]
}

impl Default for PredictiveProcessor {
    fn default() -> Self {
        Self {
            active_predictions: Vec::new(),
            error_history: VecDeque::with_capacity(100),
            accuracy_window: VecDeque::with_capacity(20),
            max_history: 100,
            priors: HashMap::new(),
        }
    }
}

impl PredictiveProcessor {
    /// Create a new predictive processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a prediction
    pub fn predict(
        &mut self,
        expected: &str,
        confidence: f64,
        source: PredictionSource,
        expected_primes: Vec<SemanticPrime>,
    ) {
        self.active_predictions.push(Prediction {
            expected: expected.to_string(),
            confidence,
            source,
            expected_primes,
            timestamp: SystemTime::now(),
        });
    }

    /// Check predictions against actual outcome
    pub fn check(&mut self, actual: &str, actual_primes: &[SemanticPrime]) -> Vec<PredictionError> {
        let mut errors = Vec::new();

        for prediction in self.active_predictions.drain(..) {
            // Calculate semantic overlap
            let expected_set: HashSet<_> = prediction.expected_primes.iter().cloned().collect();
            let actual_set: HashSet<_> = actual_primes.iter().cloned().collect();

            let intersection = expected_set.intersection(&actual_set).count();
            let union = expected_set.union(&actual_set).count();

            let semantic_match = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.5 // Neutral if no primes
            };

            // String similarity (simple)
            let string_match = if prediction.expected.to_lowercase() == actual.to_lowercase() {
                1.0
            } else if actual
                .to_lowercase()
                .contains(&prediction.expected.to_lowercase())
            {
                0.7
            } else {
                0.0
            };

            let match_score = semantic_match * 0.6 + string_match * 0.4;
            let error = 1.0 - match_score;
            let surprise = -match_score.max(0.01).ln(); // Avoid ln(0)

            let pred_error = PredictionError {
                prediction,
                actual: actual.to_string(),
                actual_primes: actual_primes.to_vec(),
                error,
                surprise,
            };

            // Update accuracy window
            if self.accuracy_window.len() >= 20 {
                self.accuracy_window.pop_front();
            }
            self.accuracy_window.push_back(match_score);

            // Store in history
            if self.error_history.len() >= self.max_history {
                self.error_history.pop_front();
            }
            self.error_history.push_back(pred_error.clone());

            errors.push(pred_error);
        }

        errors
    }

    /// Learn a prior (association)
    pub fn learn_prior(&mut self, context: &str, prediction: &str, strength: f64) {
        let priors = self.priors.entry(context.to_string()).or_default();

        // Update or add
        if let Some(existing) = priors.iter_mut().find(|(p, _)| p == prediction) {
            existing.1 = (existing.1 + strength) / 2.0; // Running average
        } else {
            priors.push((prediction.to_string(), strength));
            // Limit priors per context
            if priors.len() > 10 {
                priors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                priors.truncate(10);
            }
        }
    }

    /// Get predictions based on learned priors
    pub fn get_priors(&self, context: &str) -> Vec<(String, f64)> {
        self.priors.get(context).cloned().unwrap_or_default()
    }

    /// Get recent prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.accuracy_window.is_empty() {
            return 0.5; // Neutral
        }
        let result = self.accuracy_window.iter().sum::<f64>() / self.accuracy_window.len() as f64;
        if result.is_finite() { result } else { 0.5 }
    }

    /// Get average surprise (higher = more unpredictable)
    pub fn average_surprise(&self) -> f64 {
        if self.error_history.is_empty() {
            return 0.0;
        }
        let result = self.error_history.iter().map(|e| e.surprise).sum::<f64>()
            / self.error_history.len() as f64;
        if result.is_finite() { result } else { 0.0 }
    }

    /// Calculate PP metric (prediction success)
    pub fn pp_metric(&self) -> f64 {
        self.accuracy()
    }
}

// ============================================================================
// RECURRENT PROCESSING THEORY (RPT) - Lamme
// ============================================================================
// "Consciousness requires recurrent (feedback) processing, not just feedforward"

/// A processing pass through the understanding pipeline
#[derive(Debug, Clone)]
pub struct ProcessingPass {
    /// Pass number (1 = initial feedforward)
    pub pass: usize,
    /// Input at this pass
    pub input: String,
    /// Understanding at this pass
    pub understanding: String,
    /// Confidence at this pass
    pub confidence: f64,
    /// What changed from previous pass
    pub changes: Vec<String>,
    /// Whether understanding improved
    pub improved: bool,
}

/// Recurrent processing manager
#[derive(Debug, Clone)]
pub struct RecurrentProcessor {
    /// Processing passes for current input
    passes: Vec<ProcessingPass>,
    /// Maximum passes before stopping
    max_passes: usize,
    /// Improvement threshold (stop if improvement < this)
    improvement_threshold: f64,
    /// History of pass counts
    pass_history: VecDeque<usize>,
}

impl Default for RecurrentProcessor {
    fn default() -> Self {
        Self {
            passes: Vec::new(),
            max_passes: 5,
            improvement_threshold: 0.05,
            pass_history: VecDeque::with_capacity(50),
        }
    }
}

impl RecurrentProcessor {
    /// Create a new recurrent processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Start processing a new input
    pub fn start(&mut self, input: &str) {
        self.passes.clear();
        self.passes.push(ProcessingPass {
            pass: 1,
            input: input.to_string(),
            understanding: String::new(),
            confidence: 0.0,
            changes: vec!["Initial feedforward pass".to_string()],
            improved: true,
        });
    }

    /// Record a processing pass
    pub fn record_pass(
        &mut self,
        understanding: &str,
        confidence: f64,
        changes: Vec<String>,
    ) -> bool {
        let pass_num = self.passes.len() + 1;

        let improved = if let Some(prev) = self.passes.last() {
            confidence > prev.confidence + self.improvement_threshold
        } else {
            true
        };

        self.passes.push(ProcessingPass {
            pass: pass_num,
            input: self
                .passes
                .first()
                .map(|p| p.input.clone())
                .unwrap_or_default(),
            understanding: understanding.to_string(),
            confidence,
            changes,
            improved,
        });

        // Should continue? (improved and not at max)
        improved && pass_num < self.max_passes
    }

    /// Finish processing
    pub fn finish(&mut self) {
        if self.pass_history.len() >= 50 {
            self.pass_history.pop_front();
        }
        self.pass_history.push_back(self.passes.len());
    }

    /// Get current pass count
    pub fn current_depth(&self) -> usize {
        self.passes.len()
    }

    /// Get all passes
    pub fn passes(&self) -> &[ProcessingPass] {
        &self.passes
    }

    /// Calculate recurrence depth score (more passes = deeper processing)
    pub fn depth_score(&self) -> f64 {
        (self.passes.len() as f64 / self.max_passes as f64).clamp(0.0, 1.0)
    }

    /// Average recurrence depth over history
    pub fn average_depth(&self) -> f64 {
        if self.pass_history.is_empty() {
            return 1.0;
        }
        self.pass_history.iter().map(|&d| d as f64).sum::<f64>() / self.pass_history.len() as f64
    }

    /// Calculate RPT metric (recurrent depth)
    pub fn rpt_metric(&self) -> f64 {
        // Combine current depth with improvement trajectory
        let depth = self.depth_score();
        let improvement = self.passes.iter().filter(|p| p.improved).count() as f64
            / self.passes.len().max(1) as f64;

        (depth * 0.5 + improvement * 0.5).clamp(0.0, 1.0)
    }
}

// ============================================================================
// EMBODIED COGNITION (4E) - Varela/Thompson/Rosch
// ============================================================================
// "Cognition is embodied, embedded, enacted, and extended"

/// The system's "body" - its connection to NixOS
#[derive(Debug, Clone)]
pub struct EmbodiedState {
    /// Known packages (grounded symbols)
    known_packages: HashMap<String, PackageState>,
    /// Known services
    known_services: HashMap<String, ServiceState>,
    /// System observations
    observations: VecDeque<SystemObservation>,
    /// Proprioceptive state (self-awareness of capabilities)
    proprioception: Proprioception,
}

/// State of a known package
#[derive(Debug, Clone)]
pub struct PackageState {
    /// Package name
    pub name: String,
    /// Whether installed
    pub installed: bool,
    /// Last observed
    pub last_observed: SystemTime,
    /// Confidence in state
    pub confidence: f64,
}

/// State of a known service
#[derive(Debug, Clone)]
pub struct ServiceState {
    /// Service name
    pub name: String,
    /// Whether enabled
    pub enabled: bool,
    /// Whether running
    pub running: bool,
    /// Last observed
    pub last_observed: SystemTime,
    /// Confidence in state
    pub confidence: f64,
}

/// An observation of the system
#[derive(Debug, Clone)]
pub struct SystemObservation {
    /// What was observed
    pub observation: String,
    /// Category
    pub category: ObservationCategory,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Success/failure
    pub success: bool,
}

/// Categories of system observations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationCategory {
    /// Package query
    Package,
    /// Service query
    Service,
    /// File system
    FileSystem,
    /// Command execution
    Command,
    /// Configuration
    Config,
}

/// Proprioceptive awareness of capabilities
#[derive(Debug, Clone)]
pub struct Proprioception {
    /// Can we query packages?
    pub can_query_packages: bool,
    /// Can we install packages?
    pub can_install: bool,
    /// Can we manage services?
    pub can_manage_services: bool,
    /// Can we read files?
    pub can_read_files: bool,
    /// Can we write files?
    pub can_write_files: bool,
    /// Last capability check
    pub last_check: Option<SystemTime>,
}

impl Default for Proprioception {
    fn default() -> Self {
        Self {
            can_query_packages: true, // Assume capable until proven otherwise
            can_install: false,       // Conservative default
            can_manage_services: false,
            can_read_files: true,
            can_write_files: false,
            last_check: None,
        }
    }
}

impl Default for EmbodiedState {
    fn default() -> Self {
        Self {
            known_packages: HashMap::new(),
            known_services: HashMap::new(),
            observations: VecDeque::with_capacity(100),
            proprioception: Proprioception::default(),
        }
    }
}

impl EmbodiedState {
    /// Create new embodied state
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a package observation
    pub fn observe_package(&mut self, name: &str, installed: bool, confidence: f64) {
        self.known_packages.insert(
            name.to_string(),
            PackageState {
                name: name.to_string(),
                installed,
                last_observed: SystemTime::now(),
                confidence,
            },
        );

        self.add_observation(
            &format!(
                "Package '{}' is {}",
                name,
                if installed {
                    "installed"
                } else {
                    "not installed"
                }
            ),
            ObservationCategory::Package,
            true,
        );
    }

    /// Record a service observation
    pub fn observe_service(&mut self, name: &str, enabled: bool, running: bool, confidence: f64) {
        self.known_services.insert(
            name.to_string(),
            ServiceState {
                name: name.to_string(),
                enabled,
                running,
                last_observed: SystemTime::now(),
                confidence,
            },
        );

        self.add_observation(
            &format!(
                "Service '{}' is {} and {}",
                name,
                if enabled { "enabled" } else { "disabled" },
                if running { "running" } else { "stopped" }
            ),
            ObservationCategory::Service,
            true,
        );
    }

    /// Add a general observation
    pub fn add_observation(
        &mut self,
        observation: &str,
        category: ObservationCategory,
        success: bool,
    ) {
        if self.observations.len() >= 100 {
            self.observations.pop_front();
        }
        self.observations.push_back(SystemObservation {
            observation: observation.to_string(),
            category,
            timestamp: SystemTime::now(),
            success,
        });
    }

    /// Check if a symbol is grounded (has real-world correspondence)
    pub fn is_grounded(&self, symbol: &str) -> bool {
        let symbol_lower = symbol.to_lowercase();
        self.known_packages.contains_key(&symbol_lower)
            || self.known_services.contains_key(&symbol_lower)
    }

    /// Get grounding confidence for a symbol
    pub fn grounding_confidence(&self, symbol: &str) -> f64 {
        let symbol_lower = symbol.to_lowercase();

        if let Some(pkg) = self.known_packages.get(&symbol_lower) {
            // Decay confidence over time
            let age = pkg
                .last_observed
                .elapsed()
                .unwrap_or(Duration::from_secs(3600));
            let decay = (-age.as_secs_f64() / 3600.0).exp(); // 1-hour half-life
            return pkg.confidence * decay;
        }

        if let Some(svc) = self.known_services.get(&symbol_lower) {
            let age = svc
                .last_observed
                .elapsed()
                .unwrap_or(Duration::from_secs(3600));
            let decay = (-age.as_secs_f64() / 3600.0).exp();
            return svc.confidence * decay;
        }

        0.0
    }

    /// Update proprioception based on observation
    pub fn update_proprioception(&mut self, capability: &str, available: bool) {
        match capability {
            "query_packages" => self.proprioception.can_query_packages = available,
            "install" => self.proprioception.can_install = available,
            "manage_services" => self.proprioception.can_manage_services = available,
            "read_files" => self.proprioception.can_read_files = available,
            "write_files" => self.proprioception.can_write_files = available,
            _ => {}
        }
        self.proprioception.last_check = Some(SystemTime::now());
    }

    /// Calculate embodiment metric (groundedness)
    pub fn embodiment_metric(&self) -> f64 {
        // Combine observation recency, grounding coverage, and proprioceptive confidence
        let observation_score = if self.observations.is_empty() {
            0.0
        } else {
            let recent = self
                .observations
                .iter()
                .filter(|o| {
                    o.timestamp
                        .elapsed()
                        .map(|d| d.as_secs() < 300)
                        .unwrap_or(false)
                })
                .count();
            (recent as f64 / self.observations.len() as f64).min(1.0)
        };

        let grounding_score = (self.known_packages.len() + self.known_services.len()) as f64 / 20.0;
        let grounding_score = grounding_score.min(1.0);

        let proprio_score = [
            self.proprioception.can_query_packages,
            self.proprioception.can_read_files,
        ]
        .iter()
        .filter(|&&b| b)
        .count() as f64
            / 2.0;

        (observation_score * 0.3 + grounding_score * 0.4 + proprio_score * 0.3).clamp(0.0, 1.0)
    }
}

// ============================================================================
// UNIFIED MULTI-THEORY CONSCIOUSNESS
// ============================================================================

/// Combined consciousness metrics from all theories
#[derive(Debug, Clone)]
pub struct MultiTheoryMetrics {
    /// IIT: Integration (Φ) - from existing system
    pub phi: f64,
    /// GWT: Global workspace utilization
    pub gwt: f64,
    /// AST: Attention coherence
    pub ast: f64,
    /// PP: Prediction accuracy
    pub pp: f64,
    /// RPT: Recurrent depth
    pub rpt: f64,
    /// 4E: Embodiment/grounding
    pub embodiment: f64,
    /// Overall consciousness score (weighted combination)
    pub unified: f64,
}

impl MultiTheoryMetrics {
    /// Calculate unified consciousness score
    pub fn calculate_unified(&mut self) {
        // Each theory contributes equally by default
        // Could be weighted based on task type
        self.unified = (
            self.phi * 0.20 +         // IIT: Integration
            self.gwt * 0.15 +         // GWT: Broadcast
            self.ast * 0.15 +         // AST: Attention model
            self.pp * 0.20 +          // PP: Prediction
            self.rpt * 0.15 +         // RPT: Recurrence
            self.embodiment * 0.15
            // 4E: Grounding
        )
        .clamp(0.0, 1.0);
    }

    /// Get dominant theory (highest contribution)
    pub fn dominant_theory(&self) -> &'static str {
        let theories = [
            (self.phi, "IIT (Integration)"),
            (self.gwt, "GWT (Broadcast)"),
            (self.ast, "AST (Attention)"),
            (self.pp, "PP (Prediction)"),
            (self.rpt, "RPT (Recurrence)"),
            (self.embodiment, "4E (Embodiment)"),
        ];

        theories
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("Unknown")
    }
}

/// The unified multi-theory consciousness engine
#[derive(Debug)]
pub struct MultiTheoryConsciousness {
    /// Global Workspace (GWT)
    pub workspace: GlobalWorkspace,
    /// Attention Schema (AST)
    pub attention: AttentionSchema,
    /// Predictive Processor (PP)
    pub predictor: PredictiveProcessor,
    /// Recurrent Processor (RPT)
    pub recurrence: RecurrentProcessor,
    /// Embodied State (4E)
    pub embodiment: EmbodiedState,
    /// Current metrics
    pub metrics: MultiTheoryMetrics,
}

impl Default for MultiTheoryConsciousness {
    fn default() -> Self {
        Self {
            workspace: GlobalWorkspace::new(),
            attention: AttentionSchema::new(),
            predictor: PredictiveProcessor::new(),
            recurrence: RecurrentProcessor::new(),
            embodiment: EmbodiedState::new(),
            metrics: MultiTheoryMetrics {
                phi: 0.0,
                gwt: 0.0,
                ast: 0.0,
                pp: 0.0,
                rpt: 0.0,
                embodiment: 0.0,
                unified: 0.0,
            },
        }
    }
}

impl MultiTheoryConsciousness {
    /// Create a new multi-theory consciousness engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Process input through all consciousness theories
    pub fn process(
        &mut self,
        input: &str,
        primes: &[SemanticPrime],
        phi: f64,
    ) -> MultiTheoryResult {
        // 1. Start recurrent processing
        self.recurrence.start(input);

        // 2. Set attention focus
        self.attention.focus_on(
            input,
            "Processing new input",
            0.8,
            AttentionCategory::Intent,
        );

        // 3. Generate predictions based on priors
        let priors = self.predictor.get_priors("input_start");
        for (expected, confidence) in priors.iter().take(3) {
            self.predictor
                .predict(expected, *confidence, PredictionSource::Experience, vec![]);
        }

        // 4. Submit to global workspace
        let _workspace_id = self.workspace.submit(
            SubsystemId::Language,
            input.to_string(),
            primes.to_vec(),
            0.7, // Initial salience
        );

        // 5. Check predictions
        let prediction_errors = self.predictor.check(input, primes);
        let avg_error = if prediction_errors.is_empty() {
            0.5
        } else {
            prediction_errors.iter().map(|e| e.error).sum::<f64>() / prediction_errors.len() as f64
        };

        // 6. Record recurrent pass
        self.recurrence.record_pass(
            input,
            1.0 - avg_error,
            vec!["Initial understanding".to_string()],
        );

        // 7. Check embodiment
        let words: Vec<&str> = input.split_whitespace().collect();
        let grounded_count = words
            .iter()
            .filter(|w| self.embodiment.is_grounded(w))
            .count();
        let grounding_ratio = grounded_count as f64 / words.len().max(1) as f64;

        // 8. Update attention based on results
        if avg_error > 0.5 {
            self.attention.add_secondary(
                "high prediction error",
                "Unexpected input pattern",
                avg_error,
                AttentionCategory::Surprise,
            );
        }

        // 9. Calculate all metrics
        self.metrics.phi = phi;
        self.metrics.gwt = self.workspace.gwt_metric();
        self.metrics.ast = self.attention.ast_metric();
        self.metrics.pp = self.predictor.pp_metric();
        self.metrics.rpt = self.recurrence.rpt_metric();
        self.metrics.embodiment = self.embodiment.embodiment_metric();
        self.metrics.calculate_unified();

        // 10. Learn from this interaction
        self.predictor.learn_prior(
            "input_start",
            &words.first().unwrap_or(&"").to_string(),
            0.5,
        );

        MultiTheoryResult {
            input: input.to_string(),
            metrics: self.metrics.clone(),
            attention_description: self.attention.describe(),
            prediction_errors,
            recurrence_depth: self.recurrence.current_depth(),
            grounding_ratio,
            workspace_contents: self.workspace.contents().len(),
            dominant_theory: self.metrics.dominant_theory().to_string(),
        }
    }

    /// Record feedback for learning
    pub fn record_feedback(&mut self, success: bool, context: &str) {
        if success {
            self.predictor.learn_prior(context, "success", 0.8);
        } else {
            self.predictor.learn_prior(context, "failure", 0.2);
        }

        // Update embodiment based on interaction
        self.embodiment.add_observation(
            &format!(
                "Interaction {} - {}",
                context,
                if success { "succeeded" } else { "failed" }
            ),
            ObservationCategory::Command,
            success,
        );
    }

    /// Get comprehensive consciousness state description
    pub fn describe_state(&self) -> ConsciousnessStateDescription {
        ConsciousnessStateDescription {
            // Integration (IIT)
            integration_level: match self.metrics.phi {
                x if x > 0.7 => "Highly integrated processing",
                x if x > 0.4 => "Moderate integration",
                _ => "Fragmented processing",
            }
            .to_string(),

            // Broadcast (GWT)
            workspace_state: format!(
                "{} items in global workspace ({}% capacity)",
                self.workspace.contents().len(),
                (self.workspace.contents().len() as f64 / 7.0 * 100.0) as u32
            ),

            // Attention (AST)
            attention_state: self.attention.describe(),

            // Prediction (PP)
            prediction_state: format!(
                "Prediction accuracy: {:.0}%, Average surprise: {:.2}",
                self.predictor.accuracy() * 100.0,
                self.predictor.average_surprise()
            ),

            // Recurrence (RPT)
            recurrence_state: format!(
                "Processing depth: {} passes, Average: {:.1}",
                self.recurrence.current_depth(),
                self.recurrence.average_depth()
            ),

            // Embodiment (4E)
            embodiment_state: format!(
                "Grounded symbols: {}, Recent observations: {}",
                self.embodiment.known_packages.len() + self.embodiment.known_services.len(),
                self.embodiment.observations.len()
            ),

            // Overall
            unified_score: self.metrics.unified,
            dominant_theory: self.metrics.dominant_theory().to_string(),
        }
    }
}

/// Result of multi-theory processing
#[derive(Debug, Clone)]
pub struct MultiTheoryResult {
    /// Original input
    pub input: String,
    /// All consciousness metrics
    pub metrics: MultiTheoryMetrics,
    /// Natural language description of attention
    pub attention_description: String,
    /// Prediction errors encountered
    pub prediction_errors: Vec<PredictionError>,
    /// Recurrent processing depth
    pub recurrence_depth: usize,
    /// Ratio of grounded symbols
    pub grounding_ratio: f64,
    /// Items in global workspace
    pub workspace_contents: usize,
    /// Which theory is dominant
    pub dominant_theory: String,
}

/// Description of consciousness state
#[derive(Debug, Clone)]
pub struct ConsciousnessStateDescription {
    /// IIT integration level
    pub integration_level: String,
    /// GWT workspace state
    pub workspace_state: String,
    /// AST attention state
    pub attention_state: String,
    /// PP prediction state
    pub prediction_state: String,
    /// RPT recurrence state
    pub recurrence_state: String,
    /// 4E embodiment state
    pub embodiment_state: String,
    /// Unified score
    pub unified_score: f64,
    /// Dominant theory
    pub dominant_theory: String,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_workspace() {
        let mut workspace = GlobalWorkspace::new();

        // Submit messages
        let id1 = workspace.submit(SubsystemId::Language, "hello".into(), vec![], 0.5);
        assert!(id1.is_some());

        let id2 = workspace.submit(SubsystemId::Intent, "greeting".into(), vec![], 0.8);
        assert!(id2.is_some());

        // Low salience should be rejected
        let id3 = workspace.submit(SubsystemId::Memory, "old".into(), vec![], 0.1);
        assert!(id3.is_none());

        assert_eq!(workspace.contents().len(), 2);
    }

    #[test]
    fn test_attention_schema() {
        let mut attention = AttentionSchema::new();

        attention.focus_on(
            "install command",
            "Detected action verb",
            0.9,
            AttentionCategory::Intent,
        );
        assert!(attention.primary_focus.is_some());

        attention.add_secondary(
            "firefox",
            "Package name detected",
            0.7,
            AttentionCategory::Entity,
        );
        assert_eq!(attention.secondary_foci.len(), 1);

        let description = attention.describe();
        assert!(description.contains("install"));
    }

    #[test]
    fn test_predictive_processor() {
        let mut predictor = PredictiveProcessor::new();

        predictor.predict(
            "install",
            0.8,
            PredictionSource::Context,
            vec![SemanticPrime::Do],
        );

        let errors = predictor.check("install firefox", &[SemanticPrime::Do, SemanticPrime::Move]);
        assert!(!errors.is_empty());
        assert!(errors[0].error < 1.0); // Partial match
    }

    #[test]
    fn test_recurrent_processor() {
        let mut recurrent = RecurrentProcessor::new();

        recurrent.start("test input");
        assert_eq!(recurrent.current_depth(), 1);

        let should_continue = recurrent.record_pass("understanding v1", 0.5, vec![]);
        assert!(should_continue);

        let should_continue = recurrent.record_pass("understanding v2", 0.8, vec![]);
        assert!(should_continue);

        assert_eq!(recurrent.current_depth(), 3);
    }

    #[test]
    fn test_embodied_state() {
        let mut embodiment = EmbodiedState::new();

        embodiment.observe_package("firefox", true, 0.9);
        assert!(embodiment.is_grounded("firefox"));
        assert!(embodiment.grounding_confidence("firefox") > 0.8);

        assert!(!embodiment.is_grounded("unknownpackage"));
    }

    #[test]
    fn test_multi_theory_consciousness() {
        let mut consciousness = MultiTheoryConsciousness::new();

        let result = consciousness.process(
            "install firefox",
            &[SemanticPrime::Do, SemanticPrime::Move],
            0.5,
        );

        assert!(result.metrics.unified > 0.0);
        assert!(!result.attention_description.is_empty());
        assert!(result.recurrence_depth >= 1);
    }
}
