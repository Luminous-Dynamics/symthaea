// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Memory Module
//!
//! Implements memory structures for experience-based cognition:
//!
//! - **ExperienceRecord**: Individual experiences with context and outcomes
//! - **ThoughtTrace**: Internal reasoning traces for learning
//! - **UserEpistemicMirror**: Theory of Mind for users (from GIS v3.0)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::kosmic_state::KosmicSnapshot;
use super::signals::PrincipledSignals;

/// A recorded experience (governance outcomes, promoted knowledge-graph facts) held in
/// [`ExperienceBus`]'s in-process cache.
///
/// **Not the canonical episodic memory store.** That role belongs to
/// `symthaea_memory::episodic_replay::EpisodicMemory`, which is Phi/Psi-weighted, retrains the
/// live CfC network, and is partially persisted to SQLite — see its doc comment. This type was
/// previously also named `EpisodicMemory`, creating a name collision with no reconciling owner
/// (Symthaea Cognitive Core Reconciliation Plan, Phase 3 / Audit Overlap #1); it was renamed
/// here to end that collision. As of this rename, nothing constructs one in production (its two
/// former call sites were removed — see `ExperienceBus::generate_with_experience`'s doc comment
/// for why the read side was already dead code before this change).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRecord {
    /// Unique identifier
    pub id: String,

    /// When this experience occurred
    pub timestamp: u64,

    /// HDV embedding of the input (16384 f32)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub hdv_embedding: Vec<f32>,

    /// Primitives that were selected
    pub thought_primitives: Vec<String>,

    /// Hash of broader context
    pub context_hash: String,

    /// User identifier (if available)
    pub user_id: Option<String>,

    // ===== Principled Signals at this moment =====
    pub prediction_error: f32,
    pub uncertainty: f32,
    pub coherence: f32,
    pub confidence: f32,
    pub salience: f32,

    // ===== GIS Integration =====
    pub kosmic_snapshot: Option<KosmicSnapshot>,

    // ===== Outcome for learning =====
    pub outcome: Option<ExperienceOutcome>,

    // ===== Summaries for display/debugging =====
    pub input_summary: String,
    pub output_summary: String,
}

impl ExperienceRecord {
    /// Create a new episodic memory
    pub fn new(id: &str, input_summary: &str) -> Self {
        Self {
            id: id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            hdv_embedding: Vec::new(),
            thought_primitives: Vec::new(),
            context_hash: String::new(),
            user_id: None,
            prediction_error: 0.5,
            uncertainty: 0.5,
            coherence: 0.5,
            confidence: 0.5,
            salience: 0.5,
            kosmic_snapshot: None,
            outcome: None,
            input_summary: input_summary.to_string(),
            output_summary: String::new(),
        }
    }

    /// Create from signals
    pub fn from_signals(id: &str, input: &str, signals: &PrincipledSignals) -> Self {
        Self {
            id: id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            hdv_embedding: Vec::new(),
            thought_primitives: Vec::new(),
            context_hash: String::new(),
            user_id: None,
            prediction_error: signals.prediction_error,
            uncertainty: signals.uncertainty,
            coherence: signals.coherence,
            confidence: signals.confidence,
            salience: signals.salience,
            kosmic_snapshot: None,
            outcome: None,
            input_summary: input.chars().take(100).collect(),
            output_summary: String::new(),
        }
    }

    /// Set HDV embedding
    pub fn with_hdv(mut self, hdv: Vec<f32>) -> Self {
        self.hdv_embedding = hdv;
        self
    }

    /// Set primitives
    pub fn with_primitives(mut self, primitives: Vec<String>) -> Self {
        self.thought_primitives = primitives;
        self
    }

    /// Set outcome
    pub fn with_outcome(mut self, outcome: ExperienceOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Set kosmic snapshot
    pub fn with_kosmic(mut self, snapshot: KosmicSnapshot) -> Self {
        self.kosmic_snapshot = Some(snapshot);
        self
    }

    /// Set output summary
    pub fn with_output(mut self, output: &str) -> Self {
        self.output_summary = output.chars().take(200).collect();
        self
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.timestamp)
    }

    /// Check if outcome was successful
    pub fn was_successful(&self) -> bool {
        self.outcome
            .as_ref()
            .map(|o| o.task_completion)
            .unwrap_or(false)
    }

    /// Compute learning value (how valuable for learning)
    pub fn learning_value(&self) -> f32 {
        // High prediction error + has outcome = valuable for learning
        let surprise_value = self.prediction_error * 0.3;
        let outcome_value = if self.outcome.is_some() { 0.4 } else { 0.0 };
        let recency_value = 1.0 / (1.0 + (self.age_seconds() as f32 / 86400.0)); // Decay over days

        surprise_value + outcome_value + recency_value * 0.3
    }
}

/// Outcome of an experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceOutcome {
    /// User satisfaction (0-1)
    pub user_satisfaction: f32,

    /// Was the task completed successfully?
    pub task_completion: bool,

    /// Was follow-up needed?
    pub follow_up_needed: bool,

    /// Any explicit feedback
    pub feedback: Option<String>,

    /// Time to completion (ms)
    pub completion_time_ms: Option<u64>,
}

impl ExperienceOutcome {
    pub fn success() -> Self {
        Self {
            user_satisfaction: 0.8,
            task_completion: true,
            follow_up_needed: false,
            feedback: None,
            completion_time_ms: None,
        }
    }

    pub fn failure(reason: &str) -> Self {
        Self {
            user_satisfaction: 0.3,
            task_completion: false,
            follow_up_needed: true,
            feedback: Some(reason.to_string()),
            completion_time_ms: None,
        }
    }

    pub fn partial(satisfaction: f32) -> Self {
        Self {
            user_satisfaction: satisfaction,
            task_completion: satisfaction > 0.5,
            follow_up_needed: satisfaction < 0.7,
            feedback: None,
            completion_time_ms: None,
        }
    }
}

/// Internal reasoning trace for debugging and learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtTrace {
    /// Unique identifier
    pub id: String,

    /// Session this trace belongs to
    pub session_id: String,

    /// Sequence number within session
    pub sequence_num: u32,

    /// Input HDV (optional, for space)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_hdv_summary: Option<HdvSummary>,

    /// Codec output HDV summary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_output_summary: Option<HdvSummary>,

    /// LTC state snapshots
    pub ltc_activations: Vec<f32>,

    /// Primitives selected and their scores
    pub primitive_selections: Vec<PrimitiveSelection>,

    /// Signals at this point
    pub signals: PrincipledSignals,

    /// Any rules that fired
    pub rules_fired: Vec<String>,

    /// Processing time (ms)
    pub processing_time_ms: u64,
}

impl ThoughtTrace {
    pub fn new(session_id: &str, sequence: u32) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            sequence_num: sequence,
            input_hdv_summary: None,
            codec_output_summary: None,
            ltc_activations: Vec::new(),
            primitive_selections: Vec::new(),
            signals: PrincipledSignals::default(),
            rules_fired: Vec::new(),
            processing_time_ms: 0,
        }
    }
}

/// Summary of an HDV for storage efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdvSummary {
    /// Norm of the vector
    pub norm: f32,

    /// Mean value
    pub mean: f32,

    /// Standard deviation
    pub std: f32,

    /// Top 10 indices by absolute value
    pub top_indices: Vec<usize>,

    /// Corresponding values
    pub top_values: Vec<f32>,
}

impl HdvSummary {
    pub fn from_hdv(hdv: &[f32]) -> Self {
        let norm: f32 = hdv.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n = hdv.len().max(1) as f32;
        let mean: f32 = hdv.iter().sum::<f32>() / n;
        let variance: f32 = hdv.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        // Find top 10 by absolute value
        let mut indexed: Vec<(usize, f32)> =
            hdv.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: Vec<usize> = indexed.iter().take(10).map(|(i, _)| *i).collect();
        let top_values: Vec<f32> = top_indices.iter().map(|&i| hdv[i]).collect();

        Self {
            norm,
            mean,
            std,
            top_indices,
            top_values,
        }
    }
}

/// A primitive selection with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveSelection {
    pub primitive: String,
    pub score: f32,
    pub reason: String,
}

/// User Epistemic Mirror - Theory of Mind for a user
///
/// From GIS v3.0: Models what we believe about a user's knowledge state,
/// preferences, and communication needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEpistemicMirror {
    /// User identifier (hashed for privacy)
    pub user_hash: String,

    /// HDV profile - semantic signature of this user's interests
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub hdv_profile: Vec<f32>,

    /// Knowledge state per domain
    pub knowledge_state: HashMap<String, f32>,

    /// Learning velocity - how fast they pick things up
    pub learning_velocity: f32,

    /// Preferred explanation depth
    pub preferred_depth: ExplanationDepth,

    /// Harmonic resonances - which harmonies resonate with this user
    pub harmonic_resonance: HarmonicResonance,

    /// Predicted needs for this user
    pub predicted_needs: Vec<String>,

    /// Communication style preference
    pub communication_style: CommunicationStyle,

    /// Interaction history summary
    pub interaction_count: u64,

    /// Trust level (built over time)
    pub trust_level: f32,

    /// Last interaction timestamp
    pub last_interaction: u64,
}

impl UserEpistemicMirror {
    pub fn new(user_hash: &str) -> Self {
        Self {
            user_hash: user_hash.to_string(),
            hdv_profile: Vec::new(),
            knowledge_state: HashMap::new(),
            learning_velocity: 0.5,
            preferred_depth: ExplanationDepth::Moderate,
            harmonic_resonance: HarmonicResonance::default(),
            predicted_needs: vec!["clarity".to_string()],
            communication_style: CommunicationStyle::Balanced,
            interaction_count: 0,
            trust_level: 0.5,
            last_interaction: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Update from interaction
    pub fn update_from_interaction(
        &mut self,
        domain: Option<&str>,
        success: bool,
        _satisfaction: f32,
    ) {
        self.interaction_count += 1;
        self.last_interaction = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Update trust based on outcomes
        let trust_delta = if success { 0.02 } else { -0.01 };
        self.trust_level = (self.trust_level + trust_delta).clamp(0.0, 1.0);

        // Update domain knowledge if specified
        if let Some(d) = domain {
            let current = self.knowledge_state.get(d).unwrap_or(&0.5);
            let delta = if success { 0.05 } else { -0.02 };
            self.knowledge_state
                .insert(d.to_string(), (current + delta).clamp(0.0, 1.0));
        }

        // Adjust learning velocity based on success rate
        if success && self.interaction_count > 5 {
            self.learning_velocity = (self.learning_velocity + 0.01).min(1.0);
        }
    }

    /// Check if user is a beginner in domain
    pub fn is_beginner(&self, domain: &str) -> bool {
        self.knowledge_state.get(domain).unwrap_or(&0.3) < &0.3
    }

    /// Check if user is advanced in domain
    pub fn is_advanced(&self, domain: &str) -> bool {
        self.knowledge_state.get(domain).unwrap_or(&0.3) > &0.7
    }

    /// Get recommended explanation style
    pub fn recommended_style(&self, domain: &str) -> ExplanationDepth {
        let knowledge = self.knowledge_state.get(domain).unwrap_or(&0.5);
        if *knowledge < 0.3 {
            ExplanationDepth::Detailed
        } else if *knowledge > 0.7 {
            ExplanationDepth::Brief
        } else {
            self.preferred_depth.clone()
        }
    }
}

/// Explanation depth preference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExplanationDepth {
    Brief,
    #[default]
    Moderate,
    Detailed,
    Expert,
}

/// Communication style preference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CommunicationStyle {
    Technical,
    #[default]
    Balanced,
    Casual,
    Formal,
}

/// Harmonic resonance - which harmonies resonate with a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicResonance {
    pub coherence: f32,
    pub flourishing: f32,
    pub wisdom: f32,
    pub play: f32,
    pub interconnect: f32,
    pub reciprocity: f32,
    pub evolution: f32,
}

impl Default for HarmonicResonance {
    fn default() -> Self {
        Self {
            coherence: 0.5,
            flourishing: 0.5,
            wisdom: 0.5,
            play: 0.5,
            interconnect: 0.5,
            reciprocity: 0.5,
            evolution: 0.5,
        }
    }
}

impl HarmonicResonance {
    /// Get dominant harmony for this user
    pub fn dominant(&self) -> &'static str {
        let harmonies = [
            (self.coherence, "COHERENCE"),
            (self.flourishing, "FLOURISHING"),
            (self.wisdom, "WISDOM"),
            (self.play, "PLAY"),
            (self.interconnect, "INTERCONNECT"),
            (self.reciprocity, "RECIPROCITY"),
            (self.evolution, "EVOLUTION"),
        ];

        harmonies
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("WISDOM")
    }

    /// Update resonance based on feedback
    pub fn update(&mut self, harmony: &str, delta: f32) {
        let learning_rate = 0.1;
        let d = delta * learning_rate;
        match harmony {
            "COHERENCE" => self.coherence = (self.coherence + d).clamp(0.0, 1.0),
            "FLOURISHING" => self.flourishing = (self.flourishing + d).clamp(0.0, 1.0),
            "WISDOM" => self.wisdom = (self.wisdom + d).clamp(0.0, 1.0),
            "PLAY" => self.play = (self.play + d).clamp(0.0, 1.0),
            "INTERCONNECT" => self.interconnect = (self.interconnect + d).clamp(0.0, 1.0),
            "RECIPROCITY" => self.reciprocity = (self.reciprocity + d).clamp(0.0, 1.0),
            "EVOLUTION" => self.evolution = (self.evolution + d).clamp(0.0, 1.0),
            _ => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_episodic_memory_creation() {
        let mem = ExperienceRecord::new("test-001", "test input");
        assert_eq!(mem.id, "test-001");
        assert!(!mem.was_successful());
    }

    #[test]
    fn test_episodic_memory_with_outcome() {
        let mem =
            ExperienceRecord::new("test-002", "test").with_outcome(ExperienceOutcome::success());
        assert!(mem.was_successful());
    }

    #[test]
    fn test_user_mirror_update() {
        let mut mirror = UserEpistemicMirror::new("user123");
        assert_eq!(mirror.interaction_count, 0);

        mirror.update_from_interaction(Some("nix"), true, 0.9);
        assert_eq!(mirror.interaction_count, 1);
        assert!(mirror.trust_level > 0.5);
    }

    #[test]
    fn test_user_beginner_detection() {
        let mut mirror = UserEpistemicMirror::new("user123");
        mirror.knowledge_state.insert("nix".to_string(), 0.2);

        assert!(mirror.is_beginner("nix"));
        assert!(!mirror.is_advanced("nix"));
    }

    #[test]
    fn test_hdv_summary() {
        let hdv: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let summary = HdvSummary::from_hdv(&hdv);

        assert!(summary.norm > 0.0);
        assert_eq!(summary.top_indices.len(), 10);
    }

    #[test]
    fn test_harmonic_resonance_dominant() {
        let mut res = HarmonicResonance::default();
        res.wisdom = 0.9;
        assert_eq!(res.dominant(), "WISDOM");
    }
}
