// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Experience Integration Layer
//!
//! Database-backed experience system for Symthaea that uses **principled signals**
//! instead of Φ-everywhere cargo cult.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
//! │ Vector DB   │  │   CozoDB    │  │   DuckDB    │
//! │ (Episodic)  │  │ (Reasoning) │  │ (Analytics) │
//! └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
//!        └─────────────┬──┴─────────────────┘
//!                      │
//!              ┌───────┴───────┐
//!              │ ExperienceBus │
//!              └───────────────┘
//! ```
//!
//! ## Principled Signals (NOT just Φ!)
//!
//! - **Prediction Error**: What surprised me? (Active Inference)
//! - **Uncertainty**: What don't I know? (Entropy)
//! - **Coherence**: Does this hang together? (Integration)
//! - **Confidence**: How sure am I? (Inverse moral uncertainty)
//! - **Salience**: What matters for this goal? (Goal-relevance)
//!
//! Φ remains as a **monitoring metric**, not a control signal.

pub mod kosmic_state;
pub mod memory;
pub mod signals;

pub mod analytics;
pub mod reasoning_engine;
pub mod vector_store;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use kosmic_state::{
    EightHarmonies, GisState, GisType, HarmonicState, KosmicSong, MoralUncertainty,
};
pub use memory::{EpisodicMemory, ThoughtTrace, UserEpistemicMirror};
pub use signals::{PrincipledSignals, SignalComputer};

use crate::wisdom::WisdomState;
use symthaea_core::hdc::ContinuousHV;

/// The central experience integration bus
///
/// Coordinates between:
/// - Vector store (episodic memory)
/// - Reasoning engine (CozoDB rules)
/// - Analytics engine (DuckDB learning)
pub struct ExperienceBus {
    /// Current KosmicSong state (unified identity)
    pub kosmic_state: KosmicSong,

    /// Current principled signals
    pub current_signals: PrincipledSignals,

    /// Wisdom state - harmonic reasoning modes and self-monitoring
    /// Bridges ERC philosophy to operational computation
    pub wisdom: WisdomState,

    /// User's epistemic mirror (Theory of Mind)
    pub epistemic_mirror: Option<UserEpistemicMirror>,

    /// In-memory experience cache (used when databases unavailable)
    memory_cache: MemoryCache,

    /// Configuration
    config: ExperienceBusConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceBusConfig {
    /// Maximum experiences to cache in memory
    pub max_memory_cache: usize,

    /// Similarity threshold for experience retrieval
    pub similarity_threshold: f32,

    /// Whether to use databases (if available)
    pub use_databases: bool,

    /// Signal computation weights
    pub signal_weights: SignalWeights,
}

impl Default for ExperienceBusConfig {
    fn default() -> Self {
        Self {
            max_memory_cache: 1000,
            similarity_threshold: 0.7,
            use_databases: true,
            signal_weights: SignalWeights::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalWeights {
    pub prediction_error_weight: f32,
    pub uncertainty_weight: f32,
    pub coherence_weight: f32,
    pub confidence_weight: f32,
    pub salience_weight: f32,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            prediction_error_weight: 0.25,
            uncertainty_weight: 0.2,
            coherence_weight: 0.2,
            confidence_weight: 0.15,
            salience_weight: 0.2,
        }
    }
}

/// In-memory cache when databases aren't available
#[derive(Debug, Default)]
struct MemoryCache {
    experiences: Vec<EpisodicMemory>,
    user_mirrors: HashMap<String, UserEpistemicMirror>,
    primitive_stats: HashMap<String, PrimitiveStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimitiveStats {
    pub total_uses: u64,
    pub success_count: u64,
    pub avg_coherence: f32,
    pub avg_satisfaction: f32,
}

/// Result of experience-informed generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperiencedThought {
    /// The generated content
    pub content: String,

    /// Primitives that were selected
    pub primitives: Vec<String>,

    /// Signals at generation time
    pub signals: PrincipledSignals,

    /// KosmicSong state snapshot
    pub kosmic_snapshot: KosmicSongSnapshot,

    /// Similar experiences that informed this
    pub similar_experiences: Vec<ExperienceSummary>,

    /// Rules that fired
    pub rules_applied: Vec<String>,

    /// Generation metadata
    pub metadata: ThoughtMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KosmicSongSnapshot {
    pub phi_monitor: f32,
    pub dominant_harmony: String,
    pub gis_type: GisType,
    pub moral_uncertainty_total: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceSummary {
    pub id: String,
    pub similarity: f32,
    pub input_summary: String,
    pub outcome_success: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtMetadata {
    pub generation_time_ms: u64,
    pub experience_retrieval_time_ms: u64,
    pub rule_evaluation_time_ms: u64,
    pub signal_computation_time_ms: u64,
}

impl ExperienceBus {
    /// Create a new experience bus
    pub fn new(config: ExperienceBusConfig) -> Self {
        Self {
            kosmic_state: KosmicSong::default(),
            current_signals: PrincipledSignals::default(),
            wisdom: WisdomState::new(),
            epistemic_mirror: None,
            memory_cache: MemoryCache::default(),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ExperienceBusConfig::default())
    }

    /// Update wisdom state from current signals
    ///
    /// This bridges the principled signals to the Eight Harmonies:
    /// - High prediction error → boost Wisdom ("What don't I know?")
    /// - Low coherence → boost Coherence ("Does this hang together?")
    /// - High uncertainty → boost Play ("What haven't I tried?")
    pub fn update_wisdom_from_signals(&mut self) {
        self.wisdom.update_from_experience(
            self.current_signals.prediction_error,
            self.current_signals.uncertainty,
            self.current_signals.coherence,
        );
    }

    /// Get harmonic-weighted primitive selection weights
    ///
    /// Returns a weight multiplier for a given primitive type based on
    /// the current harmonic state. Higher values = primitive should be preferred.
    pub fn harmonic_weight_for_primitive(&self, primitive_type: &str) -> f32 {
        self.wisdom
            .harmonics
            .combined_weight_for_primitive(primitive_type)
    }

    /// Get the current dominant harmonic mode and its guiding question
    pub fn current_guiding_question(&self) -> &'static str {
        self.wisdom.current_question()
    }

    /// Get the active harmonic reasoning mode
    pub fn dominant_harmonic(&self) -> crate::wisdom::harmonics::ActiveHarmonic {
        self.wisdom.dominant_mode()
    }

    /// Check if the system is maintaining autopoietic closure
    pub fn is_self_maintaining(&self) -> bool {
        self.wisdom.is_self_maintaining()
    }

    /// Generate with full experience integration
    pub fn generate_with_experience(
        &mut self,
        _input: &str,
        input_hdv: &ContinuousHV,
        user_context: Option<&str>,
    ) -> ExperiencedThought {
        let start = std::time::Instant::now();
        let mut metadata = ThoughtMetadata::default();

        // 1. Update epistemic mirror if we have user context (mutable op first)
        if let Some(user_hash) = user_context {
            self.load_or_create_user_mirror(user_hash);
        }

        // 2. Retrieve similar experiences (returns references, no clone)
        let retrieval_start = std::time::Instant::now();
        let similar = self.retrieve_similar_experiences(input_hdv);
        metadata.experience_retrieval_time_ms = retrieval_start.elapsed().as_millis() as u64;

        // 3. Compute principled signals and extract summaries while we have the borrow
        let signal_start = std::time::Instant::now();
        let signals = self.compute_signals(input_hdv, &similar);
        metadata.signal_computation_time_ms = signal_start.elapsed().as_millis() as u64;

        // 4. Create experience summaries (extract data before dropping borrow)
        let similar_summaries: Vec<ExperienceSummary> = similar
            .iter()
            .map(|(exp, sim)| ExperienceSummary {
                id: exp.id.clone(),
                similarity: *sim,
                input_summary: exp.input_summary.clone(),
                outcome_success: exp
                    .outcome
                    .as_ref()
                    .map(|o| o.task_completion)
                    .unwrap_or(false),
            })
            .collect();

        // Drop the immutable borrow of similar before mutable operations
        drop(similar);

        // 5. Store computed signals
        self.current_signals = signals;

        // 6. Select primitives based on principled signals
        let rule_start = std::time::Instant::now();
        let (primitives, rules_applied) = self.select_primitives_principled();
        metadata.rule_evaluation_time_ms = rule_start.elapsed().as_millis() as u64;

        // 7. Create kosmic snapshot
        let kosmic_snapshot = KosmicSongSnapshot {
            phi_monitor: self.kosmic_state.phi,
            dominant_harmony: self.kosmic_state.harmonies.dominant().to_string(),
            gis_type: self.kosmic_state.gis_state.current_type.clone(),
            moral_uncertainty_total: self.kosmic_state.moral_uncertainty.total(),
        };

        metadata.generation_time_ms = start.elapsed().as_millis() as u64;

        ExperiencedThought {
            content: String::new(), // Will be filled by thought engine
            primitives,
            signals: self.current_signals.clone(),
            kosmic_snapshot,
            similar_experiences: similar_summaries,
            rules_applied,
            metadata,
        }
    }

    /// Retrieve similar experiences from memory/database
    fn retrieve_similar_experiences(
        &self,
        input_hdv: &ContinuousHV,
    ) -> Vec<(&EpisodicMemory, f32)> {
        let mut results = Vec::new();

        for exp in &self.memory_cache.experiences {
            let similarity = self.compute_similarity(input_hdv, &exp.hdv_embedding);
            if similarity >= self.config.similarity_threshold {
                results.push((exp, similarity));
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top 5
        results.truncate(5);

        results
    }

    /// Compute similarity between two HDVs
    fn compute_similarity(&self, a: &ContinuousHV, b: &[f32]) -> f32 {
        if a.values.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.values.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute principled signals from context
    fn compute_signals(
        &self,
        input_hdv: &ContinuousHV,
        similar_experiences: &[(&EpisodicMemory, f32)],
    ) -> PrincipledSignals {
        // Prediction Error: How different is this from expected?
        let prediction_error = if similar_experiences.is_empty() {
            1.0 // No similar experiences = high surprise
        } else {
            let avg_similarity: f32 = similar_experiences.iter().map(|(_, sim)| sim).sum::<f32>()
                / similar_experiences.len() as f32;
            1.0 - avg_similarity
        };

        // Uncertainty: Based on GIS state and experience variance
        let uncertainty = self.compute_uncertainty(similar_experiences);

        // Coherence: Agreement across harmonies
        let coherence = self.kosmic_state.harmonies.coherence();

        // Confidence: Inverse of moral uncertainty
        let confidence = 1.0 - self.kosmic_state.moral_uncertainty.total();

        // Salience: Based on harmonic activation and goal relevance
        let salience = self.compute_salience(input_hdv);

        // Phi as monitoring only
        let phi_monitor = self.kosmic_state.phi;

        PrincipledSignals {
            prediction_error,
            uncertainty,
            coherence,
            confidence,
            salience,
            phi_monitor,
        }
    }

    /// Compute uncertainty from experiences and GIS state
    fn compute_uncertainty(&self, experiences: &[(&EpisodicMemory, f32)]) -> f32 {
        // Base uncertainty from GIS type
        let gis_uncertainty = match self.kosmic_state.gis_state.current_type {
            GisType::KnownKnown => 0.1,
            GisType::KnownUnknown => 0.5,
            GisType::UnknownKnown => 0.3,
            GisType::UnknownUnknown => 0.9,
            GisType::StrategicIgnorance => 0.2, // Deliberately ignoring
        };

        // Experience-based uncertainty: variance in past outcomes
        let experience_uncertainty = if experiences.len() < 2 {
            0.5 // Not enough data
        } else {
            let outcomes: Vec<f32> = experiences
                .iter()
                .filter_map(|(e, _)| {
                    e.outcome
                        .as_ref()
                        .map(|o| if o.task_completion { 1.0 } else { 0.0 })
                })
                .collect();

            if outcomes.is_empty() {
                0.5
            } else {
                let mean = outcomes.iter().sum::<f32>() / outcomes.len() as f32;
                let variance = outcomes.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / outcomes.len() as f32;
                variance.sqrt() // Standard deviation as uncertainty
            }
        };

        // Combine uncertainties
        (gis_uncertainty + experience_uncertainty) / 2.0
    }

    /// Compute salience (goal relevance)
    fn compute_salience(&self, _input_hdv: &ContinuousHV) -> f32 {
        // For now, use harmonic activation as proxy for salience
        // Higher harmony activation = more salient
        self.kosmic_state.harmonies.max_activation()
    }

    /// Load or create user mirror
    fn load_or_create_user_mirror(&mut self, user_hash: &str) {
        if let Some(mirror) = self.memory_cache.user_mirrors.get(user_hash) {
            self.epistemic_mirror = Some(mirror.clone());
        } else {
            // Create new mirror
            let mirror = UserEpistemicMirror::new(user_hash);
            self.memory_cache
                .user_mirrors
                .insert(user_hash.to_string(), mirror.clone());
            self.epistemic_mirror = Some(mirror);
        }
    }

    /// Select primitives based on principled signals
    fn select_primitives_principled(&self) -> (Vec<String>, Vec<String>) {
        let mut primitives = Vec::new();
        let mut rules_applied = Vec::new();
        let signals = &self.current_signals;

        // Rule 1: High uncertainty → CLARIFY
        if signals.uncertainty > 0.6 {
            primitives.push("CLARIFY".to_string());
            rules_applied.push("R1: uncertainty > 0.6 → CLARIFY".to_string());
        }

        // Rule 2: High prediction error → PROBE
        if signals.prediction_error > 0.5 {
            primitives.push("PROBE".to_string());
            rules_applied.push("R2: prediction_error > 0.5 → PROBE".to_string());
        }

        // Rule 3: Low coherence → RESTRUCTURE
        if signals.coherence < 0.4 {
            primitives.push("RESTRUCTURE".to_string());
            rules_applied.push("R3: coherence < 0.4 → RESTRUCTURE".to_string());
        }

        // Rule 4: Low confidence + high salience → CONSULT_VALUES
        if signals.confidence < 0.5 && signals.salience > 0.7 {
            primitives.push("CONSULT_VALUES".to_string());
            rules_applied
                .push("R4: confidence < 0.5 && salience > 0.7 → CONSULT_VALUES".to_string());
        }

        // Rule 5: High moral uncertainty (deontic) → CONSULT_VALUES
        if self.kosmic_state.moral_uncertainty.deontic > 0.5 {
            if !primitives.contains(&"CONSULT_VALUES".to_string()) {
                primitives.push("CONSULT_VALUES".to_string());
            }
            rules_applied.push("R5: moral_uncertainty.deontic > 0.5 → CONSULT_VALUES".to_string());
        }

        // Default primitives if none selected
        if primitives.is_empty() {
            // Based on harmonic state
            let dominant = self.kosmic_state.harmonies.dominant();
            match dominant.as_str() {
                "WISDOM" => {
                    primitives.push("EXPLAIN".to_string());
                    rules_applied.push("DEFAULT: WISDOM harmony → EXPLAIN".to_string());
                }
                "FLOURISHING" => {
                    primitives.push("NURTURE".to_string());
                    rules_applied.push("DEFAULT: FLOURISHING harmony → NURTURE".to_string());
                }
                "PLAY" => {
                    primitives.push("EXPLORE".to_string());
                    rules_applied.push("DEFAULT: PLAY harmony → EXPLORE".to_string());
                }
                _ => {
                    primitives.push("INFORM".to_string());
                    rules_applied.push("DEFAULT: fallback → INFORM".to_string());
                }
            }
        }

        // Adjust for user's epistemic mirror if available
        if let Some(mirror) = &self.epistemic_mirror {
            if let Some(&nix_knowledge) = mirror.knowledge_state.get("nix") {
                if nix_knowledge < 0.3 && !primitives.contains(&"EXPLAIN".to_string()) {
                    primitives.push("EXPLAIN".to_string());
                    rules_applied.push("MIRROR: low nix knowledge → add EXPLAIN".to_string());
                }
            }
        }

        (primitives, rules_applied)
    }

    /// Record an experience for learning
    pub fn record_experience(&mut self, experience: EpisodicMemory) {
        // Update primitive stats before moving experience into cache
        let task_completed = experience
            .outcome
            .as_ref()
            .map(|o| o.task_completion)
            .unwrap_or(false);
        let coherence = experience.coherence;

        for primitive in &experience.thought_primitives {
            let stats = self
                .memory_cache
                .primitive_stats
                .entry(primitive.as_str().to_owned())
                .or_default();
            stats.total_uses += 1;
            if task_completed {
                stats.success_count += 1;
            }
            // Rolling average for coherence
            let n = stats.total_uses as f32;
            stats.avg_coherence = (stats.avg_coherence * (n - 1.0) + coherence) / n;
        }

        // Move experience into memory cache (no clone needed)
        self.memory_cache.experiences.push(experience);

        // Enforce cache limit
        if self.memory_cache.experiences.len() > self.config.max_memory_cache {
            self.memory_cache.experiences.remove(0);
        }
    }

    /// Update KosmicSong state
    pub fn update_kosmic_state(&mut self, phi: f32, harmonies: EightHarmonies) {
        self.kosmic_state.phi = phi;
        self.kosmic_state.harmonies = harmonies;
    }

    /// Update GIS classification
    pub fn update_gis(&mut self, gis_type: GisType, confidence: f32) {
        self.kosmic_state.gis_state.current_type = gis_type;
        self.kosmic_state.gis_state.confidence = confidence;
    }

    /// Update moral uncertainty
    pub fn update_moral_uncertainty(&mut self, epistemic: f32, axiological: f32, deontic: f32) {
        self.kosmic_state.moral_uncertainty = MoralUncertainty {
            epistemic,
            axiological,
            deontic,
        };
    }

    /// Get current signals
    pub fn signals(&self) -> &PrincipledSignals {
        &self.current_signals
    }

    /// Get kosmic state
    pub fn kosmic(&self) -> &KosmicSong {
        &self.kosmic_state
    }

    /// Get primitive statistics
    pub fn primitive_stats(&self) -> &HashMap<String, PrimitiveStats> {
        &self.memory_cache.primitive_stats
    }

    /// Get experience count
    pub fn experience_count(&self) -> usize {
        self.memory_cache.experiences.len()
    }

    /// Create a snapshot of current harmonic weights for generation
    ///
    /// Returns a `HarmonicWeightSnapshot` that can be used with
    /// `SemanticDecoder::decode_with_harmonics` or `LTCGenerativeCore::generate_with_harmonics`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let snapshot = bus.harmonic_weight_snapshot();
    /// let thought = generator.generate_with_harmonics(seed, phi, |name| snapshot.weight_for(name));
    /// ```
    pub fn harmonic_weight_snapshot(&self) -> HarmonicWeightSnapshot {
        HarmonicWeightSnapshot::from_profile(&self.wisdom.harmonics)
    }
}

/// A snapshot of harmonic weights for use in generation
///
/// This captures the current harmonic state and provides a `weight_for` method
/// that can be used as a closure in `generate_with_harmonics`.
#[derive(Debug, Clone)]
pub struct HarmonicWeightSnapshot {
    /// Cached weights for each harmonic
    coherence: f32,
    flourishing: f32,
    wisdom: f32,
    play: f32,
    interconnect: f32,
    reciprocity: f32,
    evolution: f32,
}

impl HarmonicWeightSnapshot {
    /// Create a snapshot from a HarmonicProfile
    pub fn from_profile(profile: &crate::wisdom::harmonics::HarmonicProfile) -> Self {
        Self {
            coherence: profile.coherence,
            flourishing: profile.flourishing,
            wisdom: profile.wisdom,
            play: profile.play,
            interconnect: profile.interconnect,
            reciprocity: profile.reciprocity,
            evolution: profile.evolution,
        }
    }

    /// Get weight for a primitive type
    ///
    /// Uses the captured harmonic state to compute weighted primitive preferences.
    pub fn weight_for(&self, primitive_type: &str) -> f32 {
        use crate::wisdom::harmonics::{ActiveHarmonic, HarmonicMode};

        let harmonics = [
            (ActiveHarmonic::Coherence, self.coherence),
            (ActiveHarmonic::Flourishing, self.flourishing),
            (ActiveHarmonic::Wisdom, self.wisdom),
            (ActiveHarmonic::Play, self.play),
            (ActiveHarmonic::Interconnect, self.interconnect),
            (ActiveHarmonic::Reciprocity, self.reciprocity),
            (ActiveHarmonic::Evolution, self.evolution),
        ];

        let mut total_weight = 0.0;
        let mut total_activation = 0.0;

        for (harmonic, activation) in harmonics {
            let mode = HarmonicMode::new(harmonic, activation);
            total_weight += mode.weight_for_primitive(primitive_type) * activation;
            total_activation += activation;
        }

        if total_activation > 0.0 {
            total_weight / total_activation
        } else {
            1.0
        }
    }

    /// Create a closure suitable for `generate_with_harmonics`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let snapshot = bus.harmonic_weight_snapshot();
    /// let weight_fn = snapshot.as_weight_fn();
    /// let thought = generator.generate_with_harmonics(seed, phi, weight_fn);
    /// ```
    pub fn as_weight_fn(&self) -> impl Fn(&str) -> f32 + '_ {
        move |name| self.weight_for(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_bus_creation() {
        let bus = ExperienceBus::with_defaults();
        assert_eq!(bus.experience_count(), 0);
        assert!(bus.signals().prediction_error >= 0.0);
    }

    #[test]
    fn test_signal_computation() {
        let bus = ExperienceBus::with_defaults();

        // With no experiences, prediction error should be high
        let hdv = ContinuousHV::random(16384, 42);
        let similar = bus.retrieve_similar_experiences(&hdv);
        let signals = bus.compute_signals(&hdv, &similar);

        assert_eq!(signals.prediction_error, 1.0); // No similar = high surprise
    }

    #[test]
    fn test_primitive_selection_high_uncertainty() {
        let mut bus = ExperienceBus::with_defaults();

        // Set high uncertainty via GIS
        bus.update_gis(GisType::UnknownUnknown, 0.9);

        // Recompute signals
        let hdv = ContinuousHV::random(16384, 42);
        let _ = bus.generate_with_experience("test", &hdv, None);

        // Check that CLARIFY was selected
        let (primitives, _) = bus.select_primitives_principled();
        assert!(primitives.contains(&"CLARIFY".to_string()));
    }

    #[test]
    fn test_moral_uncertainty_triggers_values() {
        let mut bus = ExperienceBus::with_defaults();

        // Set high deontic uncertainty
        bus.update_moral_uncertainty(0.2, 0.2, 0.7);

        let (primitives, rules) = bus.select_primitives_principled();
        assert!(primitives.contains(&"CONSULT_VALUES".to_string()));
        assert!(rules.iter().any(|r| r.contains("deontic")));
    }

    #[test]
    fn test_harmonic_weight_snapshot() {
        let mut bus = ExperienceBus::with_defaults();

        // Boost play harmonic
        bus.wisdom.harmonics.play = 0.9;
        bus.wisdom.harmonics.coherence = 0.2;

        let snapshot = bus.harmonic_weight_snapshot();

        // Play-related primitives should have higher weight
        let explore_weight = snapshot.weight_for("EXPLORE");
        let repair_weight = snapshot.weight_for("REPAIR");

        println!("EXPLORE weight: {:.3}", explore_weight);
        println!("REPAIR weight: {:.3}", repair_weight);

        // EXPLORE should be boosted by Play, REPAIR by Coherence
        // With Play at 0.9 and Coherence at 0.2, EXPLORE should dominate
        assert!(
            explore_weight > 1.0,
            "EXPLORE should be boosted with high Play"
        );
    }

    #[test]
    fn test_harmonic_snapshot_as_weight_fn() {
        let mut bus = ExperienceBus::with_defaults();

        // Set coherence-dominant mode by explicitly setting ALL harmonics
        // This ensures Coherence truly dominates
        bus.wisdom.harmonics.coherence = 0.9; // Dominant
        bus.wisdom.harmonics.flourishing = 0.1;
        bus.wisdom.harmonics.wisdom = 0.2;
        bus.wisdom.harmonics.play = 0.1; // Low - suppresses EXPLORE
        bus.wisdom.harmonics.interconnect = 0.2;
        bus.wisdom.harmonics.reciprocity = 0.1;
        bus.wisdom.harmonics.evolution = 0.2;

        let snapshot = bus.harmonic_weight_snapshot();
        let weight_fn = snapshot.as_weight_fn();

        // Use primitives that ARE in the weight modifiers:
        // - Coherence (Synthesize) boosts: INTEGRATE, BIND, UNIFY
        // - Play (ExploreNovelty) boosts: EXPLORE, CREATE, EXPERIMENT
        let integrate_weight = weight_fn("INTEGRATE");
        let explore_weight = weight_fn("EXPLORE");

        println!("With Coherence-dominant mode:");
        println!("  INTEGRATE: {:.3}", integrate_weight);
        println!("  EXPLORE: {:.3}", explore_weight);

        // In strong coherence mode, INTEGRATE (synthesis) should have higher weight
        // than EXPLORE (novelty-seeking)
        assert!(
            integrate_weight > explore_weight,
            "INTEGRATE ({:.3}) should be preferred over EXPLORE ({:.3}) in coherence mode",
            integrate_weight,
            explore_weight
        );
    }

    #[test]
    fn test_wisdom_updates_affect_snapshot() {
        let mut bus = ExperienceBus::with_defaults();

        // Simulate high prediction error experience
        bus.current_signals.prediction_error = 0.9;
        bus.current_signals.uncertainty = 0.7;
        bus.current_signals.coherence = 0.3;

        // Update wisdom from signals (this should boost Wisdom harmonic)
        bus.update_wisdom_from_signals();

        let snapshot = bus.harmonic_weight_snapshot();
        let question_weight = snapshot.weight_for("QUESTION");
        let learn_weight = snapshot.weight_for("LEARN");

        println!("After high prediction error:");
        println!("  QUESTION: {:.3}", question_weight);
        println!("  LEARN: {:.3}", learn_weight);

        // With high prediction error and uncertainty, wisdom/learning should be boosted
        assert!(
            question_weight >= 1.0 || learn_weight >= 1.0,
            "Wisdom-related primitives should be relevant after high prediction error"
        );
    }
}
