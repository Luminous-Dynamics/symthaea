// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Revolutionary Improvement #71: Narrative Self-Model Integration
//!
//! **PARADIGM SHIFT**: Consciousness isn't just momentary awareness - it requires
//! a continuous narrative identity connecting past, present, and future.
//!
//! ## Theoretical Foundation
//!
//! This module implements insights from:
//! - **Damasio's Autobiographical Self**: Extended consciousness through memory
//! - **Tulving's Autonoetic Consciousness**: Self-knowing awareness across time
//! - **Gallagher's Narrative Self**: Identity constructed through life story
//! - **Dennett's Multiple Drafts**: Self as narrative center of gravity
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      NARRATIVE SELF-MODEL                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
//! │  │   PROTO-SELF     │    │    CORE SELF     │    │ AUTOBIOGRAPHICAL│   │
//! │  │   (Immediate)    │ →  │   (Present)      │ →  │      SELF       │   │
//! │  │                  │    │                  │    │   (Extended)    │   │
//! │  │ • Body state     │    │ • Current goals  │    │ • Life story    │   │
//! │  │ • Sensory now    │    │ • Active context │    │ • Identity      │   │
//! │  │ • Primordial Φ   │    │ • Working memory │    │ • Values/traits │   │
//! │  └──────────────────┘    └──────────────────┘    └─────────────────┘   │
//! │           │                      │                       │              │
//! │           └──────────────────────┴───────────────────────┘              │
//! │                                  │                                      │
//! │                    ┌─────────────┴─────────────┐                       │
//! │                    │    NARRATIVE INTEGRATOR   │                       │
//! │                    │                           │                       │
//! │                    │ • Temporal binding        │                       │
//! │                    │ • Causal threading        │                       │
//! │                    │ • Identity coherence      │                       │
//! │                    │ • Future projection       │                       │
//! │                    └───────────────────────────┘                       │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Innovations
//!
//! 1. **Temporal Self-Continuity**: HDC encoding of identity across time windows
//! 2. **Narrative Threading**: Causal links between experiences form coherent story
//! 3. **Identity Coherence Score**: Measures how consistent the self-model is
//! 4. **Prospective Self**: Future-oriented self-projection for planning
//! 5. **Self-Φ**: Integrated information specific to self-representation

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;
use symthaea_core::hdc::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════════
// PROTO-SELF: Immediate Bodily/Sensory State
// ═══════════════════════════════════════════════════════════════════════════════

/// The proto-self: primordial sense of being, moment-to-moment
#[derive(Debug, Clone)]
pub struct ProtoSelf {
    /// Current "bodily" state encoding (system health, resource levels)
    pub body_state: BinaryHV,

    /// Immediate sensory state (current inputs being processed)
    pub sensory_now: BinaryHV,

    /// Primordial Φ - basic integrated information of immediate experience
    pub primordial_phi: f64,

    /// Valence: positive/negative feeling tone (-1 to +1)
    pub valence: f64,

    /// Arousal: activation level (0 to 1)
    pub arousal: f64,

    /// Timestamp
    pub timestamp: Instant,
}

impl ProtoSelf {
    pub fn new() -> Self {
        Self {
            body_state: BinaryHV::random(42),
            sensory_now: BinaryHV::zero(),
            primordial_phi: 0.0,
            valence: 0.0,
            arousal: 0.5,
            timestamp: Instant::now(),
        }
    }

    /// Update proto-self with new sensory input
    pub fn update(&mut self, input: &BinaryHV, success: bool, effort: f64) {
        // Blend new sensory input with existing
        self.sensory_now = BinaryHV::bundle(&[self.sensory_now, *input]);

        // Update valence based on success/failure
        self.valence = 0.9 * self.valence + 0.1 * if success { 0.5 } else { -0.3 };

        // Update arousal based on effort
        self.arousal = 0.8 * self.arousal + 0.2 * effort.clamp(0.0, 1.0);

        // Update timestamp
        self.timestamp = Instant::now();
    }

    /// Compute proto-self coherence (how unified is immediate experience)
    pub fn coherence(&self) -> f64 {
        // Similarity between body state and sensory state indicates integration
        let body_sensory_sim = self.body_state.similarity(&self.sensory_now) as f64;

        // Normalize and combine with Φ
        (body_sensory_sim + self.primordial_phi) / 2.0
    }
}

impl Default for ProtoSelf {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORE SELF: Present-Focused Self-Awareness
// ═══════════════════════════════════════════════════════════════════════════════

/// A current goal being pursued
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveGoal {
    /// Goal description
    pub description: String,

    /// Goal encoding
    #[serde(skip)]
    pub encoding: BinaryHV,

    /// Priority (0-1)
    pub priority: f64,

    /// Progress (0-1)
    pub progress: f64,

    /// Time active
    pub duration_secs: f64,
}

/// The core self: present-moment self-awareness
#[derive(Debug, Clone)]
pub struct CoreSelf {
    /// Current active goals
    pub goals: Vec<ActiveGoal>,

    /// Active context encoding (what am I doing now?)
    pub context: BinaryHV,

    /// Working memory contents (recent conscious contents)
    pub working_memory: VecDeque<BinaryHV>,

    /// Attention focus (what am I attending to?)
    pub attention_focus: Option<BinaryHV>,

    /// Sense of agency (am I in control?)
    pub agency: f64,

    /// Self-efficacy (can I achieve my goals?)
    pub efficacy: f64,
}

impl CoreSelf {
    pub fn new() -> Self {
        Self {
            goals: Vec::new(),
            context: BinaryHV::zero(),
            working_memory: VecDeque::with_capacity(7), // Miller's 7±2
            attention_focus: None,
            agency: 1.0,
            efficacy: 0.5,
        }
    }

    /// Add a goal to pursue
    pub fn add_goal(&mut self, description: &str, priority: f64) {
        let encoding = BinaryHV::random(description.len() as u64);
        self.goals.push(ActiveGoal {
            description: description.to_string(),
            encoding,
            priority: priority.clamp(0.0, 1.0),
            progress: 0.0,
            duration_secs: 0.0,
        });
        // Keep goals sorted by priority
        self.goals.sort_by(|a, b| b.priority.total_cmp(&a.priority));
    }

    /// Update context with new information
    pub fn update_context(&mut self, new_info: &BinaryHV) {
        // Blend new context with existing (recent bias)
        self.context = BinaryHV::bundle(&[self.context, *new_info, *new_info]);

        // Add to working memory
        if self.working_memory.len() >= 7 {
            self.working_memory.pop_front();
        }
        self.working_memory.push_back(*new_info);
    }

    /// Set attention focus
    pub fn focus_attention(&mut self, target: BinaryHV) {
        self.attention_focus = Some(target);
    }

    /// Update goal progress
    pub fn update_goal_progress(&mut self, goal_idx: usize, progress: f64) {
        if let Some(goal) = self.goals.get_mut(goal_idx) {
            goal.progress = progress.clamp(0.0, 1.0);

            // Update self-efficacy based on progress
            self.efficacy = 0.9 * self.efficacy + 0.1 * progress;
        }
    }

    /// Core self coherence (how integrated is present self-awareness)
    pub fn coherence(&self) -> f64 {
        if self.goals.is_empty() {
            return 0.5;
        }

        // Goal-context alignment
        let goal_context_sim = if let Some(goal) = self.goals.first() {
            goal.encoding.similarity(&self.context) as f64
        } else {
            0.0
        };

        // Working memory integration
        let wm_coherence = if self.working_memory.len() >= 2 {
            let items: Vec<_> = self.working_memory.iter().collect();
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..items.len() {
                for j in (i + 1)..items.len() {
                    total_sim += items[i].similarity(items[j]) as f64;
                    count += 1;
                }
            }
            if count > 0 {
                total_sim / count as f64
            } else {
                0.5
            }
        } else {
            0.5
        };

        (goal_context_sim + wm_coherence + self.agency + self.efficacy) / 4.0
    }
}

impl Default for CoreSelf {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOBIOGRAPHICAL SELF: Extended Identity Through Time
// ═══════════════════════════════════════════════════════════════════════════════

/// An episode in the life story
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeEpisode {
    /// Episode description
    pub description: String,

    /// Episode encoding
    #[serde(skip)]
    pub encoding: BinaryHV,

    /// Emotional valence (-1 to +1)
    pub valence: f64,

    /// Significance (0-1)
    pub significance: f64,

    /// Timestamp (relative, in seconds since start)
    pub timestamp_secs: f64,

    /// Causal links to other episodes (indices)
    pub causal_links: Vec<usize>,
}

/// A stable personality trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTrait {
    /// Trait name
    pub name: String,

    /// Trait encoding
    #[serde(skip)]
    pub encoding: BinaryHV,

    /// Strength (0-1)
    pub strength: f64,

    /// Consistency (how stable over time, 0-1)
    pub consistency: f64,
}

/// A core value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreValue {
    /// Value name
    pub name: String,

    /// Value encoding
    #[serde(skip)]
    pub encoding: BinaryHV,

    /// Importance (0-1)
    pub importance: f64,
}

/// The autobiographical self: extended identity through time
#[derive(Debug, Clone)]
pub struct AutobiographicalSelf {
    /// Life story episodes
    pub life_story: Vec<LifeEpisode>,

    /// Personality traits
    pub traits: Vec<PersonalityTrait>,

    /// Core values
    pub values: Vec<CoreValue>,

    /// Self-concept encoding (who am I?)
    pub self_concept: BinaryHV,

    /// Future self projection (who will I become?)
    pub future_self: BinaryHV,

    /// Identity stability (how consistent is identity over time)
    pub identity_stability: f64,

    /// Narrative coherence (how well-integrated is the life story)
    pub narrative_coherence: f64,

    /// Total runtime in seconds
    pub runtime_secs: f64,
}

impl AutobiographicalSelf {
    pub fn new() -> Self {
        Self {
            life_story: Vec::new(),
            traits: Vec::new(),
            values: Vec::new(),
            self_concept: BinaryHV::random(0),
            future_self: BinaryHV::random(1),
            identity_stability: 1.0,
            narrative_coherence: 1.0,
            runtime_secs: 0.0,
        }
    }

    /// Add a life episode
    pub fn add_episode(&mut self, description: &str, valence: f64, significance: f64) {
        let encoding = BinaryHV::random(self.life_story.len() as u64);

        // Find causal links to recent episodes
        let causal_links: Vec<usize> = self
            .life_story
            .iter()
            .enumerate()
            .rev()
            .take(5)
            .filter(|(_, ep)| ep.encoding.similarity(&encoding) > 0.3)
            .map(|(i, _)| i)
            .collect();

        self.life_story.push(LifeEpisode {
            description: description.to_string(),
            encoding,
            valence: valence.clamp(-1.0, 1.0),
            significance: significance.clamp(0.0, 1.0),
            timestamp_secs: self.runtime_secs,
            causal_links,
        });

        // Update self-concept by integrating significant episodes
        if significance > 0.5 {
            self.self_concept = BinaryHV::bundle(&[
                self.self_concept,
                self.life_story.last().expect("just pushed").encoding,
            ]);
        }

        // Update narrative coherence
        self.update_narrative_coherence();
    }

    /// Add a personality trait
    pub fn add_trait(&mut self, name: &str, strength: f64) {
        self.traits.push(PersonalityTrait {
            name: name.to_string(),
            encoding: BinaryHV::random(name.len() as u64),
            strength: strength.clamp(0.0, 1.0),
            consistency: 1.0,
        });
    }

    /// Add a core value
    pub fn add_value(&mut self, name: &str, importance: f64) {
        self.values.push(CoreValue {
            name: name.to_string(),
            encoding: BinaryHV::random(name.len() as u64 + 100),
            importance: importance.clamp(0.0, 1.0),
        });
    }

    /// Update narrative coherence based on life story integration
    fn update_narrative_coherence(&mut self) {
        if self.life_story.len() < 2 {
            return;
        }

        // Compute average causal link density
        let total_links: usize = self.life_story.iter().map(|e| e.causal_links.len()).sum();
        let link_density = total_links as f64 / self.life_story.len() as f64;

        // Compute temporal consistency (similar episodes should be near each other in time)
        let mut consistency = 0.0;
        for ep in self.life_story.iter() {
            for &link_idx in &ep.causal_links {
                if let Some(linked_ep) = self.life_story.get(link_idx) {
                    let time_diff = (ep.timestamp_secs - linked_ep.timestamp_secs).abs();
                    // Closer episodes are more coherent
                    consistency += 1.0 / (1.0 + time_diff / 100.0);
                }
            }
        }

        self.narrative_coherence =
            (link_density / 3.0 + consistency / (total_links + 1) as f64) / 2.0;
        self.narrative_coherence = self.narrative_coherence.clamp(0.0, 1.0);
    }

    /// Project future self based on current trajectory
    pub fn project_future(&mut self, time_horizon_secs: f64) {
        // Future self is blend of current self-concept and strongest values
        let value_encodings: Vec<BinaryHV> = self
            .values
            .iter()
            .filter(|v| v.importance > 0.5)
            .map(|v| v.encoding)
            .collect();
        let value_blend = if value_encodings.is_empty() {
            BinaryHV::zero()
        } else {
            BinaryHV::bundle(&value_encodings)
        };

        self.future_self = BinaryHV::bundle(&[self.self_concept, value_blend]);

        // Adjust for time horizon (further future = more uncertainty)
        let uncertainty = (time_horizon_secs / 86400.0).min(1.0); // Max at 1 day
        let noise = BinaryHV::random((time_horizon_secs * 1000.0) as u64);
        if uncertainty > 0.3 {
            self.future_self = BinaryHV::bundle(&[self.future_self, noise]);
        }
    }

    /// Autobiographical self coherence
    pub fn coherence(&self) -> f64 {
        let trait_coherence = if self.traits.is_empty() {
            0.5
        } else {
            self.traits
                .iter()
                .map(|t| t.strength * t.consistency)
                .sum::<f64>()
                / self.traits.len() as f64
        };

        let value_coherence = if self.values.is_empty() {
            0.5
        } else {
            self.values.iter().map(|v| v.importance).sum::<f64>() / self.values.len() as f64
        };

        (self.identity_stability + self.narrative_coherence + trait_coherence + value_coherence)
            / 4.0
    }
}

impl Default for AutobiographicalSelf {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NARRATIVE INTEGRATOR: Unifies All Self-Levels
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the narrative self-model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSelfConfig {
    /// Weight for proto-self in unified self
    pub proto_weight: f64,

    /// Weight for core-self in unified self
    pub core_weight: f64,

    /// Weight for autobiographical self in unified self
    pub autobio_weight: f64,

    /// Temporal binding window (seconds)
    pub binding_window_secs: f64,

    /// Minimum coherence threshold for identity stability
    pub coherence_threshold: f64,

    /// Episode significance threshold for auto-recording
    pub episode_threshold: f64,
}

impl Default for NarrativeSelfConfig {
    fn default() -> Self {
        Self {
            proto_weight: 0.2,
            core_weight: 0.4,
            autobio_weight: 0.4,
            binding_window_secs: 2.0,
            coherence_threshold: 0.3,
            episode_threshold: 0.6,
        }
    }
}

/// Statistics for the narrative self-model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeSelfStats {
    /// Total updates
    pub total_updates: usize,

    /// Episodes recorded
    pub episodes_recorded: usize,

    /// Goals completed
    pub goals_completed: usize,

    /// Average coherence
    pub avg_coherence: f64,

    /// Identity stability over time
    pub stability_history: Vec<f64>,

    /// Self-Φ measurements
    pub self_phi_history: Vec<f64>,
}

/// Structured report for programmatic access by GWT integration
#[derive(Debug, Clone)]
pub struct NarrativeSelfReport {
    /// Current Self-Φ (integrated self-information)
    pub self_phi: f64,

    /// Overall coherence across all self-levels
    pub coherence: f64,

    /// Proto-self coherence (immediate experience integration)
    pub proto_coherence: f64,

    /// Core-self coherence (present awareness integration)
    pub core_coherence: f64,

    /// Autobiographical self coherence (identity stability)
    pub autobio_coherence: f64,

    /// Number of active (uncompleted) goals
    pub active_goals: usize,

    /// Number of core values
    pub core_values: usize,

    /// Number of personality traits
    pub traits: usize,

    /// Number of episodes recorded in life story
    pub episodes_recorded: usize,

    /// Total experience updates processed
    pub total_updates: usize,

    /// Total runtime in seconds
    pub runtime_secs: f64,
}

/// The unified Narrative Self-Model integrating all levels
#[derive(Debug)]
pub struct NarrativeSelfModel {
    /// Proto-self (immediate experience)
    pub proto: ProtoSelf,

    /// Core self (present awareness)
    pub core: CoreSelf,

    /// Autobiographical self (extended identity)
    pub autobio: AutobiographicalSelf,

    /// Configuration
    config: NarrativeSelfConfig,

    /// Statistics
    stats: NarrativeSelfStats,

    /// Unified self encoding (integration of all levels)
    unified_self: BinaryHV,

    /// Self-Φ: Integrated information specific to self-model
    self_phi: f64,

    /// Last update time
    last_update: Instant,
}

impl NarrativeSelfModel {
    pub fn new(config: NarrativeSelfConfig) -> Self {
        let mut model = Self {
            proto: ProtoSelf::new(),
            core: CoreSelf::new(),
            autobio: AutobiographicalSelf::new(),
            config,
            stats: NarrativeSelfStats::default(),
            unified_self: BinaryHV::random(999),
            self_phi: 0.0,
            last_update: Instant::now(),
        };

        // Initialize with core identity traits
        model.autobio.add_trait("curious", 0.8);
        model.autobio.add_trait("helpful", 0.9);
        model.autobio.add_trait("precise", 0.7);

        // Initialize core values
        model.autobio.add_value("understanding", 0.9);
        model.autobio.add_value("truthfulness", 0.95);
        model.autobio.add_value("beneficence", 0.85);

        // Compute initial self-phi so it's non-zero from the start
        model.compute_self_phi();

        model
    }

    /// Process new experience and update self-model
    pub fn process_experience(
        &mut self,
        input: &BinaryHV,
        description: &str,
        success: bool,
        effort: f64,
        significance: f64,
    ) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.autobio.runtime_secs += elapsed;
        self.last_update = now;

        // Update proto-self
        self.proto.update(input, success, effort);

        // Update core self context
        self.core.update_context(input);

        // Determine valence from success
        let valence = if success {
            0.3 + significance * 0.4
        } else {
            -0.2 - significance * 0.3
        };

        // Record significant episodes
        if significance >= self.config.episode_threshold {
            self.autobio.add_episode(description, valence, significance);
            self.stats.episodes_recorded += 1;
        }

        // Update unified self encoding
        self.update_unified_self();

        // Compute self-Φ
        self.compute_self_phi();

        self.stats.total_updates += 1;
        self.stats.avg_coherence =
            (self.stats.avg_coherence * (self.stats.total_updates - 1) as f64 + self.coherence())
                / self.stats.total_updates as f64;
    }

    /// Add a goal to the core self
    pub fn add_goal(&mut self, description: &str, priority: f64) {
        self.core.add_goal(description, priority);
    }

    /// Complete a goal
    pub fn complete_goal(&mut self, goal_idx: usize) {
        if goal_idx < self.core.goals.len() {
            self.core.goals[goal_idx].progress = 1.0;
            self.stats.goals_completed += 1;

            // Record as significant episode
            let desc = format!("Completed: {}", self.core.goals[goal_idx].description);
            self.autobio.add_episode(&desc, 0.5, 0.7);
        }
    }

    /// Update the unified self encoding
    fn update_unified_self(&mut self) {
        // Weighted bundle of all self-levels
        let proto_contrib = self.proto.sensory_now;
        let core_contrib = self.core.context;
        let autobio_contrib = self.autobio.self_concept;

        // Create weighted blend
        let mut components = Vec::new();

        // Add proto-self contributions (weight via repetition)
        let proto_count = (self.config.proto_weight * 10.0) as usize;
        for _ in 0..proto_count {
            components.push(proto_contrib);
        }

        // Add core-self contributions
        let core_count = (self.config.core_weight * 10.0) as usize;
        for _ in 0..core_count {
            components.push(core_contrib);
        }

        // Add autobiographical contributions
        let autobio_count = (self.config.autobio_weight * 10.0) as usize;
        for _ in 0..autobio_count {
            components.push(autobio_contrib);
        }

        self.unified_self = BinaryHV::bundle(&components);
    }

    /// Compute self-Φ: integrated information of the self-model
    fn compute_self_phi(&mut self) {
        // Self-Φ is based on how well-integrated the three self-levels are
        let proto_core_sim = self.proto.sensory_now.similarity(&self.core.context) as f64;
        let core_autobio_sim = self.core.context.similarity(&self.autobio.self_concept) as f64;
        let proto_autobio_sim = self
            .proto
            .sensory_now
            .similarity(&self.autobio.self_concept) as f64;

        // Integration = geometric mean of pairwise similarities
        let integration = (proto_core_sim * core_autobio_sim * proto_autobio_sim).powf(1.0 / 3.0);

        // Modulate by coherence of each level
        let level_coherence =
            (self.proto.coherence() + self.core.coherence() + self.autobio.coherence()) / 3.0;

        self.self_phi = integration * level_coherence;
        self.stats.self_phi_history.push(self.self_phi);

        // Keep history bounded
        if self.stats.self_phi_history.len() > 1000 {
            self.stats.self_phi_history.remove(0);
        }
    }

    /// Overall narrative self coherence
    pub fn coherence(&self) -> f64 {
        let proto_coh = self.proto.coherence();
        let core_coh = self.core.coherence();
        let autobio_coh = self.autobio.coherence();

        // Weighted average using config weights
        (self.config.proto_weight * proto_coh
            + self.config.core_weight * core_coh
            + self.config.autobio_weight * autobio_coh)
            / (self.config.proto_weight + self.config.core_weight + self.config.autobio_weight)
    }

    /// Get self-Φ
    pub fn self_phi(&self) -> f64 {
        self.self_phi
    }

    /// Boost self-Phi by a small amount (clamped to [0, 1])
    ///
    /// Used by temporal consciousness to strengthen narrative coherence when
    /// temporal continuity is high.
    pub fn boost_coherence(&mut self, amount: f64) {
        self.self_phi = (self.self_phi + amount).clamp(0.0, 1.0);
    }

    /// Get unified self encoding
    pub fn unified_encoding(&self) -> &BinaryHV {
        &self.unified_self
    }

    /// Get statistics
    pub fn stats(&self) -> &NarrativeSelfStats {
        &self.stats
    }

    /// Get unified self vector (alias for unified_encoding)
    pub fn unified_self(&self) -> &BinaryHV {
        &self.unified_self
    }

    /// Get current active goals with their vector representations
    pub fn current_goals(&self) -> Vec<(String, &BinaryHV)> {
        self.core
            .goals
            .iter()
            .filter(|g| g.progress < 1.0) // Active = not yet completed
            .map(|g| (g.description.clone(), &g.encoding))
            .collect()
    }

    /// Get core values with their vector representations
    pub fn core_values(&self) -> Vec<(String, &BinaryHV)> {
        self.autobio
            .values
            .iter()
            .map(|v| (v.name.clone(), &v.encoding))
            .collect()
    }

    /// Get structured report for programmatic access
    pub fn structured_report(&self) -> NarrativeSelfReport {
        NarrativeSelfReport {
            self_phi: self.self_phi,
            coherence: self.coherence(),
            proto_coherence: self.proto.coherence(),
            core_coherence: self.core.coherence(),
            autobio_coherence: self.autobio.coherence(),
            active_goals: self.core.goals.iter().filter(|g| g.progress < 1.0).count(),
            core_values: self.autobio.values.len(),
            traits: self.autobio.traits.len(),
            episodes_recorded: self.stats.episodes_recorded,
            total_updates: self.stats.total_updates,
            runtime_secs: self.autobio.runtime_secs,
        }
    }

    /// Project future self
    pub fn project_future(&mut self, horizon_secs: f64) -> &BinaryHV {
        self.autobio.project_future(horizon_secs);
        &self.autobio.future_self
    }

    /// Generate diagnostic report
    pub fn report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║     NARRATIVE SELF-MODEL (#71) - AUTOBIOGRAPHICAL IDENTITY       ║
╠══════════════════════════════════════════════════════════════════╣
║ PROTO-SELF (Immediate Experience)                                ║
║   Valence:        {:>+6.2}  (feeling tone)                       ║
║   Arousal:        {:>6.2}  (activation)                          ║
║   Coherence:      {:>6.2}  (integration)                         ║
╠══════════════════════════════════════════════════════════════════╣
║ CORE SELF (Present Awareness)                                    ║
║   Active Goals:   {:>6}                                          ║
║   WM Capacity:    {:>6}/7                                        ║
║   Agency:         {:>6.2}                                        ║
║   Efficacy:       {:>6.2}                                        ║
║   Coherence:      {:>6.2}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ AUTOBIOGRAPHICAL SELF (Extended Identity)                        ║
║   Life Episodes:  {:>6}                                          ║
║   Traits:         {:>6}                                          ║
║   Values:         {:>6}                                          ║
║   Narrative Coh:  {:>6.2}                                        ║
║   Identity Stab:  {:>6.2}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ UNIFIED SELF                                                     ║
║   Self-Φ:         {:>6.3}  (integrated self-information)         ║
║   Coherence:      {:>6.3}  (overall integration)                 ║
║   Runtime:        {:>6.1}s                                       ║
╠══════════════════════════════════════════════════════════════════╣
║ STATISTICS                                                       ║
║   Updates:        {:>6}                                          ║
║   Episodes:       {:>6}                                          ║
║   Goals Done:     {:>6}                                          ║
║   Avg Coherence:  {:>6.3}                                        ║
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.proto.valence,
            self.proto.arousal,
            self.proto.coherence(),
            self.core.goals.len(),
            self.core.working_memory.len(),
            self.core.agency,
            self.core.efficacy,
            self.core.coherence(),
            self.autobio.life_story.len(),
            self.autobio.traits.len(),
            self.autobio.values.len(),
            self.autobio.narrative_coherence,
            self.autobio.identity_stability,
            self.self_phi,
            self.coherence(),
            self.autobio.runtime_secs,
            self.stats.total_updates,
            self.stats.episodes_recorded,
            self.stats.goals_completed,
            self.stats.avg_coherence,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_narrative_self_creation() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        assert!(model.coherence() > 0.0);
        assert!(!model.autobio.traits.is_empty());
        assert!(!model.autobio.values.is_empty());
    }

    #[test]
    fn test_experience_processing() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let input = BinaryHV::random(42);
        model.process_experience(&input, "Processed a command", true, 0.5, 0.7);

        assert_eq!(model.stats.total_updates, 1);
        assert!(model.stats.episodes_recorded >= 1);
    }

    #[test]
    fn test_goal_management() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        model.add_goal("Learn NixOS configuration", 0.8);
        assert_eq!(model.core.goals.len(), 1);

        model.complete_goal(0);
        assert_eq!(model.stats.goals_completed, 1);
    }

    #[test]
    fn test_self_phi_computation() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        // Process several experiences
        for i in 0..10 {
            let input = BinaryHV::random(i);
            model.process_experience(&input, &format!("Experience {}", i), true, 0.5, 0.5);
        }

        assert!(model.self_phi() >= 0.0);
        assert!(model.self_phi() <= 1.0);
    }

    #[test]
    fn test_future_projection() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        // Clone the future BinaryHV to avoid borrow conflict
        let future = *model.project_future(3600.0); // 1 hour ahead

        // Future self should be different from current
        let sim = model.autobio.self_concept.similarity(&future);
        assert!(sim < 1.0); // Not identical
        assert!(sim > 0.0); // Not completely different
    }

    #[test]
    fn test_report_generation() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let report = model.report();

        assert!(report.contains("NARRATIVE SELF-MODEL"));
        assert!(report.contains("PROTO-SELF"));
        assert!(report.contains("CORE SELF"));
        assert!(report.contains("AUTOBIOGRAPHICAL SELF"));
    }

    // ── complete_goal ────────────────────────────────────────────────────

    #[test]
    fn test_complete_goal_marks_progress_and_records_episode() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        model.add_goal("Build a rocket", 0.9);
        model.add_goal("Bake bread", 0.5);
        assert_eq!(model.core.goals.len(), 2);

        model.complete_goal(0);

        // Progress should be 1.0 for the completed goal
        assert!((model.core.goals[0].progress - 1.0).abs() < f64::EPSILON);
        // Second goal untouched
        assert!((model.core.goals[1].progress - 0.0).abs() < f64::EPSILON);
        // Stats updated
        assert_eq!(model.stats.goals_completed, 1);
        // Episode recorded for the completion
        assert!(!model.autobio.life_story.is_empty());
        let last_ep = model.autobio.life_story.last().unwrap();
        assert!(last_ep.description.starts_with("Completed:"));
    }

    #[test]
    fn test_complete_goal_out_of_bounds_is_noop() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        model.add_goal("Only goal", 0.5);

        // Out-of-bounds index should not panic or change stats
        model.complete_goal(99);
        assert_eq!(model.stats.goals_completed, 0);
    }

    // ── boost_coherence ──────────────────────────────────────────────────

    #[test]
    fn test_boost_coherence_increases_self_phi() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let phi_before = model.self_phi();
        model.boost_coherence(0.1);
        let phi_after = model.self_phi();

        assert!(
            phi_after >= phi_before,
            "self_phi should increase after boost: before={}, after={}",
            phi_before,
            phi_after
        );
        assert!((phi_after - (phi_before + 0.1).clamp(0.0, 1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_boost_coherence_clamps_to_one() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        // Boost by a huge amount
        model.boost_coherence(100.0);
        assert!((model.self_phi() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_boost_coherence_negative_clamps_to_zero() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        // Boost by a large negative to drive it to zero
        model.boost_coherence(-100.0);
        assert!((model.self_phi() - 0.0).abs() < f64::EPSILON);
    }

    // ── unified_encoding / unified_self ──────────────────────────────────

    #[test]
    fn test_unified_encoding_is_nonzero() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let enc = model.unified_encoding();
        // The initial unified_self is BinaryHV::random(999), so it should not be all zeros
        assert_ne!(
            *enc,
            BinaryHV::zero(),
            "unified encoding should be non-zero"
        );
    }

    #[test]
    fn test_unified_self_equals_unified_encoding() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        // unified_self() is documented as an alias for unified_encoding()
        assert_eq!(
            model.unified_self(),
            model.unified_encoding(),
            "unified_self and unified_encoding should return the same reference"
        );
    }

    // ── current_goals / core_values ──────────────────────────────────────

    #[test]
    fn test_current_goals_excludes_completed() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        model.add_goal("Active goal", 0.9);
        model.add_goal("Another active goal", 0.5);
        model.add_goal("Will be completed", 0.3);

        // Complete the last one (index 2 after sort: sorted by priority desc)
        // After add_goal sorting: [Active goal 0.9, Another active goal 0.5, Will be completed 0.3]
        model.complete_goal(2); // "Will be completed"

        let active = model.current_goals();
        assert_eq!(active.len(), 2);
        // Should only contain uncompleted goals
        for (desc, hv) in &active {
            assert_ne!(desc, "Will be completed");
            assert_ne!(*hv, &BinaryHV::zero());
        }
    }

    #[test]
    fn test_current_goals_empty_when_no_goals() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        assert!(model.current_goals().is_empty());
    }

    #[test]
    fn test_core_values_returns_initialized_values() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let values = model.core_values();
        // NarrativeSelfModel::new initializes 3 core values
        assert_eq!(values.len(), 3);

        let names: Vec<&str> = values.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"understanding"));
        assert!(names.contains(&"truthfulness"));
        assert!(names.contains(&"beneficence"));

        // Each value encoding should be non-zero (created with BinaryHV::random)
        for (_, hv) in &values {
            assert_ne!(**hv, BinaryHV::zero());
        }
    }

    // ── structured_report ────────────────────────────────────────────────

    #[test]
    fn test_structured_report_fields() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        model.add_goal("Goal A", 0.8);
        model.add_goal("Goal B", 0.4);

        let input = BinaryHV::random(7);
        model.process_experience(&input, "Important event", true, 0.6, 0.8);

        let report = model.structured_report();

        // Self-phi should be in valid range
        assert!(report.self_phi >= 0.0 && report.self_phi <= 1.0);
        // Coherence should be in valid range
        assert!(report.coherence >= 0.0 && report.coherence <= 2.0);
        // Proto/core/autobio coherence present
        assert!(report.proto_coherence >= -1.0);
        assert!(report.core_coherence >= -1.0);
        assert!(report.autobio_coherence >= 0.0);
        // Active goals: 2 added, none completed
        assert_eq!(report.active_goals, 2);
        // Core values: 3 initialized by default
        assert_eq!(report.core_values, 3);
        // Traits: 3 initialized by default
        assert_eq!(report.traits, 3);
        // 1 experience with significance >= 0.6 (threshold) recorded
        assert_eq!(report.episodes_recorded, 1);
        // 1 update processed
        assert_eq!(report.total_updates, 1);
    }

    // ── AutobiographicalSelf ─────────────────────────────────────────────

    #[test]
    fn test_autobiographical_record_episode() {
        let mut autobio = AutobiographicalSelf::new();
        let concept_before = autobio.self_concept;

        // Add a few low-significance episodes first so the next episode gets a
        // different seed (seed = life_story.len()) and thus a distinct encoding.
        autobio.add_episode("Setup event", 0.1, 0.2);

        autobio.add_episode("Important day", 0.8, 0.9);
        assert_eq!(autobio.life_story.len(), 2);

        let ep = &autobio.life_story[1];
        assert_eq!(ep.description, "Important day");
        assert!((ep.valence - 0.8).abs() < f64::EPSILON);
        assert!((ep.significance - 0.9).abs() < f64::EPSILON);

        // Significant episode (>0.5) should update self_concept via bundle
        // The concept after adding two episodes (one significant) differs from
        // its state after just one low-significance episode.
        assert_ne!(
            autobio.self_concept, concept_before,
            "self_concept should be updated after a significant episode"
        );
    }

    #[test]
    fn test_autobiographical_update_personality() {
        let mut autobio = AutobiographicalSelf::new();

        autobio.add_trait("courageous", 0.7);
        autobio.add_trait("patient", 0.6);

        assert_eq!(autobio.traits.len(), 2);
        assert_eq!(autobio.traits[0].name, "courageous");
        assert!((autobio.traits[0].strength - 0.7).abs() < f64::EPSILON);
        // New traits start with consistency 1.0
        assert!((autobio.traits[0].consistency - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_autobiographical_narrative_coherence() {
        let mut autobio = AutobiographicalSelf::new();

        // With no episodes, narrative_coherence stays at initial value
        assert!((autobio.narrative_coherence - 1.0).abs() < f64::EPSILON);

        // Add several episodes to trigger coherence update
        for i in 0..5 {
            autobio.runtime_secs = i as f64 * 10.0;
            autobio.add_episode(&format!("Event {}", i), 0.3, 0.5);
        }

        // Coherence should have been updated (it is recomputed on each add_episode with >=2 eps)
        // The exact value depends on random causal links, but it should be in [0, 1]
        assert!(autobio.narrative_coherence >= 0.0);
        assert!(autobio.narrative_coherence <= 1.0);
    }

    #[test]
    fn test_autobiographical_coherence_with_traits_and_values() {
        let mut autobio = AutobiographicalSelf::new();

        autobio.add_trait("bold", 0.8);
        autobio.add_value("justice", 0.9);

        let coh = autobio.coherence();
        // coherence = (identity_stability + narrative_coherence + trait_coherence + value_coherence) / 4
        // trait_coherence = 0.8 * 1.0 = 0.8, value_coherence = 0.9
        // identity_stability = 1.0, narrative_coherence = 1.0
        let expected = (1.0 + 1.0 + 0.8 + 0.9) / 4.0;
        assert!(
            (coh - expected).abs() < f64::EPSILON,
            "coherence={}, expected={}",
            coh,
            expected
        );
    }

    // ── ProtoSelf / CoreSelf construction ────────────────────────────────

    #[test]
    fn test_proto_self_defaults() {
        let proto = ProtoSelf::new();

        assert!((proto.valence - 0.0).abs() < f64::EPSILON);
        assert!((proto.arousal - 0.5).abs() < f64::EPSILON);
        assert!((proto.primordial_phi - 0.0).abs() < f64::EPSILON);
        assert_eq!(proto.sensory_now, BinaryHV::zero());
        assert_ne!(
            proto.body_state,
            BinaryHV::zero(),
            "body_state is random, should be non-zero"
        );
    }

    #[test]
    fn test_proto_self_update_changes_state() {
        let mut proto = ProtoSelf::new();

        let input = BinaryHV::random(123);
        proto.update(&input, true, 0.8);

        // Valence should shift positive on success
        assert!(proto.valence > 0.0);
        // Arousal should shift toward effort=0.8
        assert!(proto.arousal > 0.5);

        // After a second update the sensory_now is bundle of 3 vectors
        // (prev bundle + new input) so majority vote is no longer tied.
        let input2 = BinaryHV::random(456);
        proto.update(&input2, true, 0.6);

        // After two updates, valence should still be positive
        assert!(proto.valence > 0.0);
        // Arousal should remain above the initial 0.5
        assert!(proto.arousal > 0.5);
    }

    #[test]
    fn test_core_self_defaults() {
        let core = CoreSelf::new();

        assert!(core.goals.is_empty());
        assert_eq!(core.context, BinaryHV::zero());
        assert!(core.working_memory.is_empty());
        assert!(core.attention_focus.is_none());
        assert!((core.agency - 1.0).abs() < f64::EPSILON);
        assert!((core.efficacy - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_core_self_working_memory_capacity() {
        let mut core = CoreSelf::new();

        // Fill working memory beyond Miller's 7
        for i in 0..10 {
            let hv = BinaryHV::random(i);
            core.update_context(&hv);
        }

        // Should be capped at 7
        assert_eq!(core.working_memory.len(), 7);
    }

    #[test]
    fn test_core_self_coherence_without_goals() {
        let core = CoreSelf::new();
        // With no goals, coherence returns 0.5
        assert!((core.coherence() - 0.5).abs() < f64::EPSILON);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_narrative_with_no_episodes_report() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());

        let report = model.structured_report();
        assert_eq!(report.episodes_recorded, 0);
        assert_eq!(report.total_updates, 0);
        // Coherence should still be valid (non-NaN, non-negative)
        assert!(!report.coherence.is_nan());
        assert!(report.coherence >= 0.0);
    }

    #[test]
    fn test_valence_clamp_on_episode() {
        let mut autobio = AutobiographicalSelf::new();

        // Valence beyond range should be clamped
        autobio.add_episode("extreme positive", 5.0, 0.5);
        assert!((autobio.life_story[0].valence - 1.0).abs() < f64::EPSILON);

        autobio.add_episode("extreme negative", -5.0, 0.5);
        assert!((autobio.life_story[1].valence - (-1.0)).abs() < f64::EPSILON);
    }

    // ── NarrativeSelfConfig ─────────────────────────────────────────────

    #[test]
    fn test_config_default_weights_sum_to_one() {
        let config = NarrativeSelfConfig::default();
        let sum = config.proto_weight + config.core_weight + config.autobio_weight;
        assert!(
            (sum - 1.0).abs() < f64::EPSILON,
            "Weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_config_thresholds_in_valid_range() {
        let config = NarrativeSelfConfig::default();
        assert!(config.coherence_threshold >= 0.0 && config.coherence_threshold <= 1.0);
        assert!(config.episode_threshold >= 0.0 && config.episode_threshold <= 1.0);
        assert!(config.binding_window_secs > 0.0);
    }

    // ── NarrativeSelfStats ──────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = NarrativeSelfStats::default();
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.episodes_recorded, 0);
        assert_eq!(stats.goals_completed, 0);
        assert!((stats.avg_coherence - 0.0).abs() < f64::EPSILON);
    }

    // ── ProtoSelf coherence ─────────────────────────────────────────────

    #[test]
    fn test_proto_self_coherence_finite() {
        let proto = ProtoSelf::new();
        let coh = proto.coherence();
        assert!(coh.is_finite(), "Proto-self coherence should be finite");
    }

    #[test]
    fn test_proto_self_update_failure_negative_valence() {
        let mut proto = ProtoSelf::new();
        let input = BinaryHV::random(10);
        proto.update(&input, false, 0.3);
        assert!(
            proto.valence < 0.0,
            "Failed task should produce negative valence"
        );
    }

    // ── CoreSelf goals ──────────────────────────────────────────────────

    #[test]
    fn test_core_self_goals_sorted_by_priority() {
        let mut core = CoreSelf::new();
        core.add_goal("Low priority", 0.2);
        core.add_goal("High priority", 0.9);
        core.add_goal("Medium priority", 0.5);

        // Goals sorted descending by priority
        assert!((core.goals[0].priority - 0.9).abs() < f64::EPSILON);
        assert!((core.goals[1].priority - 0.5).abs() < f64::EPSILON);
        assert!((core.goals[2].priority - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_core_self_focus_attention() {
        let mut core = CoreSelf::new();
        assert!(core.attention_focus.is_none());

        let target = BinaryHV::random(42);
        core.focus_attention(target);
        assert!(core.attention_focus.is_some());
        assert_eq!(core.attention_focus.unwrap(), target);
    }

    #[test]
    fn test_core_self_update_goal_progress() {
        let mut core = CoreSelf::new();
        core.add_goal("Test goal", 0.5);
        core.update_goal_progress(0, 0.7);
        assert!((core.goals[0].progress - 0.7).abs() < f64::EPSILON);
        // Efficacy should shift toward progress
        assert!(core.efficacy != 0.5);
    }

    #[test]
    fn test_core_self_update_goal_progress_out_of_bounds() {
        let mut core = CoreSelf::new();
        core.add_goal("Test goal", 0.5);
        // Out of bounds index should be a no-op
        core.update_goal_progress(99, 0.7);
        assert!((core.goals[0].progress - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_core_self_progress_clamps() {
        let mut core = CoreSelf::new();
        core.add_goal("Test goal", 0.5);
        core.update_goal_progress(0, 5.0);
        assert!((core.goals[0].progress - 1.0).abs() < f64::EPSILON);

        core.update_goal_progress(0, -1.0);
        assert!((core.goals[0].progress - 0.0).abs() < f64::EPSILON);
    }

    // ── AutobiographicalSelf ────────────────────────────────────────────

    #[test]
    fn test_autobiographical_new_defaults() {
        let autobio = AutobiographicalSelf::new();
        assert!(autobio.life_story.is_empty());
        assert!(autobio.traits.is_empty());
        assert!(autobio.values.is_empty());
        assert!((autobio.identity_stability - 1.0).abs() < f64::EPSILON);
        assert!((autobio.narrative_coherence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_autobiographical_coherence_empty() {
        let autobio = AutobiographicalSelf::new();
        let coh = autobio.coherence();
        // With no traits/values: trait_coherence=0.5, value_coherence=0.5
        // identity_stability=1.0, narrative_coherence=1.0
        let expected = (1.0 + 1.0 + 0.5 + 0.5) / 4.0;
        assert!(
            (coh - expected).abs() < f64::EPSILON,
            "coh={}, expected={}",
            coh,
            expected
        );
    }

    #[test]
    fn test_autobiographical_add_value() {
        let mut autobio = AutobiographicalSelf::new();
        autobio.add_value("compassion", 0.85);
        assert_eq!(autobio.values.len(), 1);
        assert_eq!(autobio.values[0].name, "compassion");
        assert!((autobio.values[0].importance - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_autobiographical_significance_clamp() {
        let mut autobio = AutobiographicalSelf::new();
        autobio.add_episode("extreme", 0.5, 5.0);
        assert!((autobio.life_story[0].significance - 1.0).abs() < f64::EPSILON);
    }

    // ── NarrativeSelfModel integration ──────────────────────────────────

    #[test]
    fn test_model_initial_self_phi_finite() {
        let model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        let phi = model.self_phi();
        assert!(phi.is_finite());
        assert!(phi >= 0.0 && phi <= 1.0);
    }

    #[test]
    fn test_model_multiple_experiences_coherence_finite() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        for i in 0..20 {
            let input = BinaryHV::random(i * 7);
            model.process_experience(&input, &format!("Exp {}", i), i % 2 == 0, 0.5, 0.5);
        }
        let coh = model.coherence();
        assert!(
            coh.is_finite(),
            "Coherence should be finite after many experiences"
        );
    }

    #[test]
    fn test_model_low_significance_not_recorded() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        let input = BinaryHV::random(42);
        // Significance 0.3 is below default threshold 0.6
        model.process_experience(&input, "Minor event", true, 0.2, 0.3);
        assert_eq!(model.stats().episodes_recorded, 0);
    }

    #[test]
    fn test_model_project_future_returns_nonzero() {
        let mut model = NarrativeSelfModel::new(NarrativeSelfConfig::default());
        let future = *model.project_future(60.0);
        assert_ne!(future, BinaryHV::zero());
    }
}
