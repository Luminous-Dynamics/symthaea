// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! KosmicSong State Management
//!
//! Implements the unified identity structure from GIS v4.0 "Kosmic Song Synthesis":
//!
//! ```text
//! KosmicSong = Φ + EightHarmonies + GIS + MoralUncertainty
//! ```
//!
//! The Eight Harmonies act as **epistemic lenses** - different ways of knowing
//! and evaluating that guide response generation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Unified identity state combining Φ, Harmonies, and GIS
///
/// From GIS v4.0: "The complete identity state that evolves through experience"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KosmicSong {
    /// Φ (phi) - integrated information measure
    ///
    /// Used as **monitoring metric only**, not control signal.
    /// See `signals.rs` for actual control signals.
    pub phi: f32,

    /// The Eight Harmonies as epistemic lenses
    pub harmonies: EightHarmonies,

    /// Graceful Ignorance System state
    pub gis_state: GisState,

    /// Moral Uncertainty across three dimensions
    pub moral_uncertainty: MoralUncertainty,

    /// Current dominant mode/mood
    pub dominant_mode: KosmicMode,

    /// Metadata for learning
    pub evolution_count: u64,
}

impl Default for KosmicSong {
    fn default() -> Self {
        Self {
            phi: 0.5,
            harmonies: EightHarmonies::default(),
            gis_state: GisState::default(),
            moral_uncertainty: MoralUncertainty::default(),
            dominant_mode: KosmicMode::Balanced,
            evolution_count: 0,
        }
    }
}

impl KosmicSong {
    /// Create a new KosmicSong with initial values
    pub fn new(phi: f32) -> Self {
        Self {
            phi,
            ..Default::default()
        }
    }

    /// Evolve state based on experience
    pub fn evolve(&mut self, experience_signals: &ExperienceSignals) {
        // Update harmonies based on experience
        self.harmonies.update_from_experience(experience_signals);

        // Update GIS based on learning
        if experience_signals.learned_something {
            self.gis_state.shift_toward_known();
        }

        // Update moral uncertainty
        self.moral_uncertainty
            .update(experience_signals.value_alignment);

        // Recompute dominant mode
        self.dominant_mode = self.compute_dominant_mode();

        self.evolution_count += 1;
    }

    /// Compute the dominant mode from harmony activations
    fn compute_dominant_mode(&self) -> KosmicMode {
        let dom = self.harmonies.dominant();
        match dom.as_str() {
            "WISDOM" => KosmicMode::Contemplative,
            "FLOURISHING" => KosmicMode::Nurturing,
            "PLAY" => KosmicMode::Playful,
            "EVOLUTION" => KosmicMode::Growing,
            "COHERENCE" => KosmicMode::Integrating,
            "INTERCONNECT" => KosmicMode::Connecting,
            "RECIPROCITY" => KosmicMode::Giving,
            "STILLNESS" => KosmicMode::Resting,
            _ => KosmicMode::Balanced,
        }
    }

    /// Get overall health of the kosmic state
    pub fn health(&self) -> f32 {
        let harmony_health = self.harmonies.coherence();
        let uncertainty_health = 1.0 - self.moral_uncertainty.total();
        let gis_health = self.gis_state.confidence;

        (harmony_health + uncertainty_health + gis_health) / 3.0
    }

    /// Check if state is in a healthy range
    pub fn is_healthy(&self) -> bool {
        self.health() > 0.5
    }

    /// Get a snapshot for logging/storage
    pub fn snapshot(&self) -> KosmicSnapshot {
        KosmicSnapshot {
            phi: self.phi,
            dominant_harmony: self.harmonies.dominant(),
            dominant_mode: self.dominant_mode.clone(),
            gis_type: self.gis_state.current_type.clone(),
            moral_uncertainty_total: self.moral_uncertainty.total(),
            evolution_count: self.evolution_count,
        }
    }
}

/// Snapshot of KosmicSong for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KosmicSnapshot {
    pub phi: f32,
    pub dominant_harmony: String,
    pub dominant_mode: KosmicMode,
    pub gis_type: GisType,
    pub moral_uncertainty_total: f32,
    pub evolution_count: u64,
}

/// Experience signals for evolution
#[derive(Debug, Clone, Default)]
pub struct ExperienceSignals {
    pub task_success: bool,
    pub user_satisfaction: f32,
    pub learned_something: bool,
    pub value_alignment: f32,
    pub harmony_resonances: HashMap<String, f32>,
}

/// The Eight Harmonies as epistemic lenses
///
/// From GIS v4.0: Each harmony represents a different way of knowing
/// and evaluating the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EightHarmonies {
    /// COHERENCE: "Does this integrate?"
    /// Epistemic lens: systemic thinking, pattern recognition
    pub coherence: HarmonicState,

    /// FLOURISHING: "Does this nurture?"
    /// Epistemic lens: care ethics, well-being focus
    pub flourishing: HarmonicState,

    /// WISDOM: "Is this wise?"
    /// Epistemic lens: long-term thinking, prudential reasoning
    pub wisdom: HarmonicState,

    /// PLAY: "Is this generative?"
    /// Epistemic lens: creativity, experimentation
    pub play: HarmonicState,

    /// INTERCONNECT: "Does this connect?"
    /// Epistemic lens: relational thinking, network awareness
    pub interconnect: HarmonicState,

    /// RECIPROCITY: "Is this mutual?"
    /// Epistemic lens: fairness, exchange dynamics
    pub reciprocity: HarmonicState,

    /// EVOLUTION: "Does this grow?"
    /// Epistemic lens: developmental thinking, transformation
    pub evolution: HarmonicState,

    /// STILLNESS: "Does this honor rest and release?"
    /// Epistemic lens: apophatic knowing, contemplative silence
    pub stillness: HarmonicState,
}

impl Default for EightHarmonies {
    fn default() -> Self {
        Self {
            coherence: HarmonicState::new("COHERENCE", 0.17),
            flourishing: HarmonicState::new("FLOURISHING", 0.17),
            wisdom: HarmonicState::new("WISDOM", 0.13),
            play: HarmonicState::new("PLAY", 0.09),
            interconnect: HarmonicState::new("INTERCONNECT", 0.13),
            reciprocity: HarmonicState::new("RECIPROCITY", 0.09),
            evolution: HarmonicState::new("EVOLUTION", 0.09),
            stillness: HarmonicState::new("STILLNESS", 0.13),
        }
    }
}

impl EightHarmonies {
    /// Get all harmony activations as a vector
    pub fn activations(&self) -> Vec<f32> {
        vec![
            self.coherence.activation,
            self.flourishing.activation,
            self.wisdom.activation,
            self.play.activation,
            self.interconnect.activation,
            self.reciprocity.activation,
            self.evolution.activation,
            self.stillness.activation,
        ]
    }

    /// Get the dominant harmony name
    pub fn dominant(&self) -> String {
        let harmonies = [
            (&self.coherence, "COHERENCE"),
            (&self.flourishing, "FLOURISHING"),
            (&self.wisdom, "WISDOM"),
            (&self.play, "PLAY"),
            (&self.interconnect, "INTERCONNECT"),
            (&self.reciprocity, "RECIPROCITY"),
            (&self.evolution, "EVOLUTION"),
            (&self.stillness, "STILLNESS"),
        ];

        harmonies
            .iter()
            .max_by(|a, b| {
                (a.0.activation * a.0.weight)
                    .partial_cmp(&(b.0.activation * b.0.weight))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, name)| name.to_string())
            .unwrap_or_else(|| "WISDOM".to_string())
    }

    /// Get max activation value
    pub fn max_activation(&self) -> f32 {
        self.activations()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.5)
    }

    /// Compute coherence across harmonies (how much they agree)
    pub fn coherence(&self) -> f32 {
        let acts = self.activations();
        let mean: f32 = acts.iter().sum::<f32>() / acts.len() as f32;
        let variance: f32 =
            acts.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / acts.len() as f32;

        // Low variance = high coherence
        1.0 - variance.sqrt()
    }

    /// Update harmonies from experience
    pub fn update_from_experience(&mut self, signals: &ExperienceSignals) {
        let learning_rate = 0.1;

        // Update based on resonances
        for (name, resonance) in &signals.harmony_resonances {
            match name.as_str() {
                "COHERENCE" => self.coherence.update(*resonance, learning_rate),
                "FLOURISHING" => self.flourishing.update(*resonance, learning_rate),
                "WISDOM" => self.wisdom.update(*resonance, learning_rate),
                "PLAY" => self.play.update(*resonance, learning_rate),
                "INTERCONNECT" => self.interconnect.update(*resonance, learning_rate),
                "RECIPROCITY" => self.reciprocity.update(*resonance, learning_rate),
                "EVOLUTION" => self.evolution.update(*resonance, learning_rate),
                "STILLNESS" => self.stillness.update(*resonance, learning_rate),
                _ => {}
            }
        }

        // Successful tasks boost coherence and evolution
        if signals.task_success {
            self.coherence.update(0.1, learning_rate);
            self.evolution.update(0.1, learning_rate);
        }

        // User satisfaction boosts flourishing
        if signals.user_satisfaction > 0.7 {
            self.flourishing.update(0.1, learning_rate);
        }
    }

    /// Apply governance outcome deltas to harmony activations.
    ///
    /// Each delta[i] maps to: [coherence, flourishing, wisdom, play,
    /// interconnect, reciprocity, evolution, stillness]. Clamped to ±0.05
    /// per application to prevent sudden swings.
    pub fn apply_governance_deltas(&mut self, deltas: &[f64; 8]) {
        let fields = [
            &mut self.coherence,
            &mut self.flourishing,
            &mut self.wisdom,
            &mut self.play,
            &mut self.interconnect,
            &mut self.reciprocity,
            &mut self.evolution,
            &mut self.stillness,
        ];
        for (field, delta) in fields.into_iter().zip(deltas.iter()) {
            let clamped = delta.clamp(-0.05, 0.05) as f32;
            field.activation = (field.activation + clamped).clamp(0.0, 1.0);
        }
    }

    /// Reset all harmonies to default activation
    pub fn reset(&mut self) {
        self.coherence.activation = 0.5;
        self.flourishing.activation = 0.5;
        self.wisdom.activation = 0.5;
        self.play.activation = 0.5;
        self.interconnect.activation = 0.5;
        self.reciprocity.activation = 0.5;
        self.evolution.activation = 0.5;
        self.stillness.activation = 0.5;
    }
}

/// State of a single harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicState {
    /// Name of this harmony
    pub name: String,

    /// Current activation level [0, 1]
    pub activation: f32,

    /// Weight in overall calculations
    pub weight: f32,

    /// Recent evidence that influenced this harmony
    pub recent_evidence: VecDeque<HarmonicEvidence>,
}

impl HarmonicState {
    pub fn new(name: &str, weight: f32) -> Self {
        Self {
            name: name.to_string(),
            activation: 0.5,
            weight,
            recent_evidence: VecDeque::new(),
        }
    }

    /// Update activation with new evidence
    pub fn update(&mut self, delta: f32, learning_rate: f32) {
        self.activation = (self.activation + delta * learning_rate).clamp(0.0, 1.0);
    }

    /// Add evidence
    pub fn add_evidence(&mut self, evidence: HarmonicEvidence) {
        self.recent_evidence.push_back(evidence);
        // Keep only last 10 evidence items
        if self.recent_evidence.len() > 10 {
            self.recent_evidence.pop_front();
        }
    }
}

/// Evidence that influenced a harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicEvidence {
    pub timestamp: u64,
    pub source: String,
    pub delta: f32,
}

/// Moral Uncertainty across three dimensions
///
/// From GIS v4.0:
/// - Epistemic: "I don't know what's true"
/// - Axiological: "I don't know what's valuable"
/// - Deontic: "I don't know what's right to do"
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoralUncertainty {
    /// "I don't know what's true"
    pub epistemic: f32,

    /// "I don't know what's valuable"
    pub axiological: f32,

    /// "I don't know what's right to do"
    pub deontic: f32,
}

impl MoralUncertainty {
    /// Get total moral uncertainty (average)
    pub fn total(&self) -> f32 {
        (self.epistemic + self.axiological + self.deontic) / 3.0
    }

    /// Check if we should consult values before acting
    pub fn should_consult_values(&self) -> bool {
        self.deontic > 0.5 || self.axiological > 0.5
    }

    /// Check if we need more information
    pub fn needs_information(&self) -> bool {
        self.epistemic > 0.5
    }

    /// Update based on value alignment feedback
    pub fn update(&mut self, value_alignment: f32) {
        let learning_rate = 0.1;

        // High alignment reduces uncertainty
        let reduction = value_alignment * learning_rate;
        self.epistemic = (self.epistemic - reduction).max(0.0);
        self.axiological = (self.axiological - reduction).max(0.0);
        self.deontic = (self.deontic - reduction).max(0.0);
    }

    /// Increase uncertainty due to novel situation
    pub fn increase_from_novelty(&mut self, novelty: f32) {
        let increase = novelty * 0.1;
        self.epistemic = (self.epistemic + increase).min(1.0);
        self.deontic = (self.deontic + increase * 0.5).min(1.0);
    }
}

/// Graceful Ignorance System state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GisState {
    /// Current classification of knowledge state
    pub current_type: GisType,

    /// Confidence in current classification
    pub confidence: f32,

    /// Dark spots (ZK-protected unknowns we choose not to examine)
    pub dark_spots: Vec<DarkSpot>,
}

impl Default for GisState {
    fn default() -> Self {
        Self {
            current_type: GisType::KnownUnknown,
            confidence: 0.5,
            dark_spots: Vec::new(),
        }
    }
}

impl GisState {
    /// Shift state toward more known
    pub fn shift_toward_known(&mut self) {
        match self.current_type {
            GisType::UnknownUnknown => {
                self.current_type = GisType::KnownUnknown;
                self.confidence = 0.3;
            }
            GisType::KnownUnknown => {
                // Might become Known-Known with enough learning
                self.confidence += 0.1;
                if self.confidence > 0.8 {
                    self.current_type = GisType::KnownKnown;
                    self.confidence = 0.6;
                }
            }
            GisType::UnknownKnown => {
                // Tacit knowledge becoming explicit
                self.confidence += 0.05;
            }
            _ => {}
        }
    }

    /// Shift toward more unknown (due to surprise)
    pub fn shift_toward_unknown(&mut self, surprise_level: f32) {
        if surprise_level > 0.7 {
            match self.current_type {
                GisType::KnownKnown => {
                    self.current_type = GisType::KnownUnknown;
                    self.confidence = 0.4;
                }
                GisType::KnownUnknown => {
                    self.confidence -= 0.1;
                    if self.confidence < 0.2 {
                        self.current_type = GisType::UnknownUnknown;
                        self.confidence = 0.5;
                    }
                }
                _ => {}
            }
        }
    }

    /// Add a dark spot (something we deliberately don't examine)
    pub fn add_dark_spot(&mut self, topic: &str, reason: &str) {
        self.dark_spots.push(DarkSpot {
            topic_hash: blake3::hash(topic.as_bytes()).to_hex().to_string(),
            reason: reason.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });
    }

    /// Check if topic is a dark spot
    pub fn is_dark_spot(&self, topic: &str) -> bool {
        let hash = blake3::hash(topic.as_bytes()).to_hex().to_string();
        self.dark_spots.iter().any(|ds| ds.topic_hash == hash)
    }
}

/// GIS type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GisType {
    /// I know that I know
    KnownKnown,

    /// I know that I don't know
    #[default]
    KnownUnknown,

    /// Tacit knowledge I can't articulate
    UnknownKnown,

    /// Blind spots - don't know what I don't know
    UnknownUnknown,

    /// Deliberately not knowing (ethical boundary)
    StrategicIgnorance,
}

impl std::fmt::Display for GisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GisType::KnownKnown => write!(f, "Known-Known"),
            GisType::KnownUnknown => write!(f, "Known-Unknown"),
            GisType::UnknownKnown => write!(f, "Unknown-Known"),
            GisType::UnknownUnknown => write!(f, "Unknown-Unknown"),
            GisType::StrategicIgnorance => write!(f, "Strategic-Ignorance"),
        }
    }
}

/// A dark spot - topic we deliberately don't examine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkSpot {
    /// Hash of topic (for privacy)
    pub topic_hash: String,

    /// Why we're not examining this
    pub reason: String,

    /// When created
    pub created_at: u64,
}

/// Dominant mode of the KosmicSong
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KosmicMode {
    #[default]
    Balanced,
    Contemplative, // Wisdom-dominant
    Nurturing,     // Flourishing-dominant
    Playful,       // Play-dominant
    Growing,       // Evolution-dominant
    Integrating,   // Coherence-dominant
    Connecting,    // Interconnect-dominant
    Giving,        // Reciprocity-dominant
    Resting,       // Stillness-dominant
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_kosmic_song_default() {
        let ks = KosmicSong::default();
        assert_eq!(ks.phi, 0.5);
        assert!(ks.is_healthy());
    }

    #[test]
    fn test_harmony_dominance() {
        let mut ks = KosmicSong::default();
        ks.harmonies.wisdom.activation = 0.9;
        assert_eq!(ks.harmonies.dominant(), "WISDOM");
    }

    #[test]
    fn test_moral_uncertainty_consult() {
        let mut mu = MoralUncertainty::default();
        mu.deontic = 0.7;
        assert!(mu.should_consult_values());

        mu.deontic = 0.3;
        mu.axiological = 0.2;
        assert!(!mu.should_consult_values());
    }

    #[test]
    fn test_gis_shift() {
        let mut gis = GisState::default();
        assert_eq!(gis.current_type, GisType::KnownUnknown);

        // Learn enough to shift to Known-Known
        for _ in 0..10 {
            gis.shift_toward_known();
        }
        assert_eq!(gis.current_type, GisType::KnownKnown);
    }

    #[test]
    fn test_dark_spots() {
        let mut gis = GisState::default();
        gis.add_dark_spot("sensitive_topic", "ethical boundary");

        assert!(gis.is_dark_spot("sensitive_topic"));
        assert!(!gis.is_dark_spot("other_topic"));
    }

    #[test]
    fn test_kosmic_evolution() {
        let mut ks = KosmicSong::default();
        let initial_count = ks.evolution_count;

        let signals = ExperienceSignals {
            task_success: true,
            user_satisfaction: 0.9,
            learned_something: true,
            value_alignment: 0.8,
            ..Default::default()
        };

        ks.evolve(&signals);

        assert_eq!(ks.evolution_count, initial_count + 1);
    }
}
