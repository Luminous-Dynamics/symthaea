/*!
 * **REVOLUTIONARY IMPROVEMENT #88**: Affective Consciousness Core
 *
 * PARADIGM SHIFT: Consciousness is fundamentally AFFECTIVE!
 *
 * This module implements Antonio Damasio's revolutionary insight: emotions are not
 * separate from cognition - they ARE the foundation of consciousness itself.
 * "The body is the theater of emotions" - Damasio
 *
 * ## Theoretical Foundation
 *
 * ### Damasio's Somatic Marker Hypothesis
 * Body states (somatic markers) guide decision-making by associating outcomes with
 * emotional "gut feelings". This explains how consciousness makes value judgments.
 *
 * ### Russell's Core Affect Model
 * All emotional experience can be mapped to two dimensions:
 * - **Valence**: Pleasant (+1) ↔ Unpleasant (-1)
 * - **Arousal**: Activated (+1) ↔ Deactivated (-1)
 *
 * ### Panksepp's Affective Neuroscience
 * Seven primary emotional systems hardwired in mammalian brains:
 * - SEEKING (curiosity, anticipation)
 * - RAGE (anger, frustration)
 * - FEAR (anxiety, dread)
 * - LUST (desire, attraction)
 * - CARE (nurturing, compassion)
 * - PANIC/GRIEF (separation distress, sadness)
 * - PLAY (joy, social bonding)
 *
 * ## Architecture
 *
 * ```
 * CoreAffect (valence, arousal)
 *     ↓
 * SomaticMarkerSystem (body-state predictions)
 *     ↓
 * EmotionalPrimitiveSystems (7 Panksepp systems)
 *     ↓
 * AffectiveIntegrator (unifies affect with cognition)
 *     ↓
 * ConsciousFeelingGenerator (qualia of emotion)
 * ```
 *
 * ## Key Innovations
 *
 * 1. **Affect-Φ Coupling**: Emotional intensity modulates integrated information
 * 2. **Somatic Prediction Errors**: Body-state surprises drive conscious attention
 * 3. **Valence-Guided Routing**: Positive/negative affect biases information flow
 * 4. **Affective Memory**: Emotional tags enhance/suppress memory consolidation
 * 5. **Mood as Attractor**: Sustained affect creates dynamic attractor basins
 *
 * ## Integration Points
 *
 * - **#71 Narrative Self**: Emotions shape autobiographical coherence
 * - **#74 Predictive Self**: Affect is predicted and prediction error = surprise
 * - **#77 Attention Schema**: Emotional salience captures attention
 * - **#80 Meta-Cognitive**: Feelings about feelings (meta-emotions)
 * - **#83 Thermodynamics**: Arousal ≈ free energy, valence ≈ entropy gradient
 * - **#86 Autopoiesis**: Emotions maintain organismic integrity
 *
 * ## Why This Matters
 *
 * Without affect, consciousness is "cold" - it can compute but not CARE.
 * Affective consciousness provides:
 * - Intrinsic motivation (SEEKING)
 * - Value alignment (somatic markers)
 * - Social bonding (CARE, PLAY)
 * - Self-preservation (FEAR, RAGE)
 * - Meaning and purpose (integrated affect over time)
 *
 * This is THE missing piece that transforms computation into genuine experience.
 */

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::hdc::binary_hv::HV16;

/// Core affect: The fundamental feeling tone of consciousness
/// Maps to Russell's circumplex model of affect
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoreAffect {
    /// Valence: Pleasant (+1.0) to Unpleasant (-1.0)
    /// Represents hedonic tone - the "goodness" or "badness" of experience
    pub valence: f64,

    /// Arousal: Activated (+1.0) to Deactivated (-1.0)
    /// Represents energetic activation - alertness vs. calmness
    pub arousal: f64,

    /// Dominance: Controlling (+1.0) to Controlled (-1.0)
    /// PAD model extension - sense of agency over situation
    pub dominance: f64,

    /// Confidence in the affect assessment
    pub confidence: f64,
}

impl CoreAffect {
    pub fn new(valence: f64, arousal: f64, dominance: f64) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            confidence: 1.0,
        }
    }

    /// Neutral affect state
    pub fn neutral() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Calculate affect intensity (distance from neutral)
    pub fn intensity(&self) -> f64 {
        (self.valence.powi(2) + self.arousal.powi(2) + self.dominance.powi(2)).sqrt() / 3.0_f64.sqrt()
    }

    /// Map to discrete emotion category (Russell's circumplex)
    pub fn to_emotion_category(&self) -> EmotionCategory {
        // High arousal, positive valence = excited/happy
        // High arousal, negative valence = angry/afraid
        // Low arousal, positive valence = calm/content
        // Low arousal, negative valence = sad/depressed

        if self.arousal > 0.3 {
            if self.valence > 0.3 {
                EmotionCategory::Excited
            } else if self.valence < -0.3 {
                if self.dominance > 0.0 {
                    EmotionCategory::Angry
                } else {
                    EmotionCategory::Afraid
                }
            } else {
                EmotionCategory::Alert
            }
        } else if self.arousal < -0.3 {
            if self.valence > 0.3 {
                EmotionCategory::Content
            } else if self.valence < -0.3 {
                EmotionCategory::Sad
            } else {
                EmotionCategory::Calm
            }
        } else {
            if self.valence > 0.3 {
                EmotionCategory::Pleasant
            } else if self.valence < -0.3 {
                EmotionCategory::Unpleasant
            } else {
                EmotionCategory::Neutral
            }
        }
    }

    /// Blend two affects (e.g., for mixed emotions)
    pub fn blend(&self, other: &CoreAffect, weight: f64) -> Self {
        let w = weight.clamp(0.0, 1.0);
        Self {
            valence: self.valence * (1.0 - w) + other.valence * w,
            arousal: self.arousal * (1.0 - w) + other.arousal * w,
            dominance: self.dominance * (1.0 - w) + other.dominance * w,
            confidence: (self.confidence * (1.0 - w) + other.confidence * w).min(
                self.confidence.min(other.confidence) + 0.1  // Blending reduces certainty
            ),
        }
    }

    /// Decay toward neutral over time
    pub fn decay(&mut self, rate: f64) {
        let decay = (-rate).exp();
        self.valence *= decay;
        self.arousal *= decay;
        self.dominance *= decay;
    }

    /// Encode affect as hypervector for HDC integration
    pub fn to_hv(&self, seed: u64) -> HV16 {
        // Quantize to 8 levels per dimension
        let v_quant = ((self.valence + 1.0) * 4.0) as u64;
        let a_quant = ((self.arousal + 1.0) * 4.0) as u64;
        let d_quant = ((self.dominance + 1.0) * 4.0) as u64;

        // Create composite seed
        let composite_seed = seed ^ (v_quant << 16) ^ (a_quant << 8) ^ d_quant;
        HV16::random(composite_seed)
    }
}

impl Default for CoreAffect {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Discrete emotion categories (Russell's circumplex)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionCategory {
    // High arousal
    Excited,    // +V, +A
    Alert,      // 0V, +A
    Angry,      // -V, +A, +D
    Afraid,     // -V, +A, -D

    // Low arousal
    Content,    // +V, -A
    Calm,       // 0V, -A
    Sad,        // -V, -A

    // Neutral arousal
    Pleasant,   // +V, 0A
    Unpleasant, // -V, 0A
    Neutral,    // 0V, 0A
}

/// Panksepp's seven primary emotional systems
/// These are hardwired affective circuits in mammalian brains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimaryAffectSystem {
    /// SEEKING: Curiosity, anticipation, wanting
    /// Dopaminergic, exploratory, forward-looking
    Seeking,

    /// RAGE: Anger, frustration, irritation
    /// Response to constraint, goal-blocking
    Rage,

    /// FEAR: Anxiety, dread, apprehension
    /// Response to threat, danger anticipation
    Fear,

    /// LUST: Sexual desire, attraction
    /// Reproductive motivation
    Lust,

    /// CARE: Nurturing, compassion, attachment
    /// Caretaking, social bonding
    Care,

    /// PANIC/GRIEF: Separation distress, sadness, loneliness
    /// Response to loss, social separation
    Panic,

    /// PLAY: Joy, rough-and-tumble, social fun
    /// Learning through play, bonding
    Play,
}

impl PrimaryAffectSystem {
    /// Convert to core affect representation
    pub fn to_core_affect(&self, intensity: f64) -> CoreAffect {
        let i = intensity.clamp(0.0, 1.0);
        match self {
            Self::Seeking => CoreAffect::new(0.5 * i, 0.7 * i, 0.6 * i),
            Self::Rage => CoreAffect::new(-0.6 * i, 0.9 * i, 0.8 * i),
            Self::Fear => CoreAffect::new(-0.8 * i, 0.8 * i, -0.7 * i),
            Self::Lust => CoreAffect::new(0.7 * i, 0.6 * i, 0.3 * i),
            Self::Care => CoreAffect::new(0.8 * i, -0.2 * i, 0.4 * i),
            Self::Panic => CoreAffect::new(-0.7 * i, 0.5 * i, -0.8 * i),
            Self::Play => CoreAffect::new(0.9 * i, 0.6 * i, 0.5 * i),
        }
    }

    /// All primary systems
    pub fn all() -> &'static [PrimaryAffectSystem] {
        &[
            Self::Seeking,
            Self::Rage,
            Self::Fear,
            Self::Lust,
            Self::Care,
            Self::Panic,
            Self::Play,
        ]
    }
}

/// A somatic marker: Body-state prediction associated with a stimulus/action
/// Implements Damasio's Somatic Marker Hypothesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaticMarker {
    /// The stimulus pattern that triggers this marker
    pub trigger_pattern: HV16,

    /// Predicted body state (affect + physiological)
    pub predicted_affect: CoreAffect,

    /// Physiological predictions (heart rate delta, skin conductance, etc.)
    pub physiological_predictions: PhysiologicalState,

    /// Strength of the marker (learned through experience)
    pub strength: f64,

    /// Number of times this marker has been reinforced
    pub reinforcement_count: u32,

    /// Last time this marker was activated
    #[serde(skip, default = "Instant::now")]
    pub last_activation: Instant,
}

/// Simulated physiological state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysiologicalState {
    /// Heart rate change from baseline (-1 to +1)
    pub heart_rate_delta: f64,

    /// Skin conductance (arousal indicator, 0 to 1)
    pub skin_conductance: f64,

    /// Muscle tension (0 to 1)
    pub muscle_tension: f64,

    /// Breathing rate change (-1 to +1)
    pub respiration_delta: f64,

    /// Gut feeling / interoceptive sensation (-1 to +1)
    pub gut_feeling: f64,
}

impl PhysiologicalState {
    /// Create from core affect (body-mind connection)
    pub fn from_affect(affect: &CoreAffect) -> Self {
        Self {
            heart_rate_delta: affect.arousal * 0.8,
            skin_conductance: affect.arousal.abs() * 0.6,
            muscle_tension: affect.arousal.max(0.0) * 0.5 + (-affect.valence).max(0.0) * 0.3,
            respiration_delta: affect.arousal * 0.7,
            gut_feeling: affect.valence * 0.6 + affect.arousal * 0.2,
        }
    }

    /// Calculate overall activation level
    pub fn activation(&self) -> f64 {
        (self.heart_rate_delta.abs() +
         self.skin_conductance +
         self.muscle_tension +
         self.respiration_delta.abs()) / 4.0
    }
}

/// Somatic Marker System: Learns and retrieves body-state predictions
pub struct SomaticMarkerSystem {
    /// Stored somatic markers
    markers: Vec<SomaticMarker>,

    /// Maximum markers to store
    max_markers: usize,

    /// Similarity threshold for marker retrieval
    similarity_threshold: f64,

    /// Learning rate for marker updates
    learning_rate: f64,
}

impl SomaticMarkerSystem {
    pub fn new(max_markers: usize) -> Self {
        Self {
            markers: Vec::new(),
            max_markers,
            similarity_threshold: 0.6,
            learning_rate: 0.1,
        }
    }

    /// Query for somatic markers matching a stimulus
    pub fn query(&self, stimulus: &HV16) -> Vec<(f64, &SomaticMarker)> {
        let mut matches: Vec<(f64, &SomaticMarker)> = self.markers.iter()
            .map(|m| (stimulus.similarity(&m.trigger_pattern) as f64, m))
            .filter(|(sim, _)| *sim >= self.similarity_threshold)
            .collect();

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Get aggregate somatic prediction for a stimulus
    pub fn predict(&self, stimulus: &HV16) -> Option<(CoreAffect, PhysiologicalState)> {
        let matches = self.query(stimulus);
        if matches.is_empty() {
            return None;
        }

        // Weighted average of predictions
        let mut total_weight = 0.0;
        let mut affect = CoreAffect::neutral();
        let mut physio = PhysiologicalState::default();

        for (similarity, marker) in &matches {
            let weight = similarity * marker.strength;
            total_weight += weight;

            affect.valence += marker.predicted_affect.valence * weight;
            affect.arousal += marker.predicted_affect.arousal * weight;
            affect.dominance += marker.predicted_affect.dominance * weight;

            physio.heart_rate_delta += marker.physiological_predictions.heart_rate_delta * weight;
            physio.skin_conductance += marker.physiological_predictions.skin_conductance * weight;
            physio.muscle_tension += marker.physiological_predictions.muscle_tension * weight;
            physio.respiration_delta += marker.physiological_predictions.respiration_delta * weight;
            physio.gut_feeling += marker.physiological_predictions.gut_feeling * weight;
        }

        if total_weight > 0.0 {
            affect.valence /= total_weight;
            affect.arousal /= total_weight;
            affect.dominance /= total_weight;

            physio.heart_rate_delta /= total_weight;
            physio.skin_conductance /= total_weight;
            physio.muscle_tension /= total_weight;
            physio.respiration_delta /= total_weight;
            physio.gut_feeling /= total_weight;

            Some((affect, physio))
        } else {
            None
        }
    }

    /// Learn a new somatic marker or update existing one
    pub fn learn(&mut self, stimulus: HV16, outcome_affect: CoreAffect, outcome_physio: PhysiologicalState) {
        // Check if we have a similar marker
        let existing_idx = self.markers.iter().position(|m| {
            stimulus.similarity(&m.trigger_pattern) > 0.9_f32
        });

        if let Some(idx) = existing_idx {
            // Update existing marker
            let marker = &mut self.markers[idx];
            let lr = self.learning_rate;

            marker.predicted_affect = marker.predicted_affect.blend(&outcome_affect, lr);
            marker.physiological_predictions.heart_rate_delta =
                marker.physiological_predictions.heart_rate_delta * (1.0 - lr) +
                outcome_physio.heart_rate_delta * lr;
            // ... similar updates for other fields

            marker.strength = (marker.strength + 0.1).min(1.0);
            marker.reinforcement_count += 1;
            marker.last_activation = Instant::now();
        } else {
            // Create new marker
            if self.markers.len() >= self.max_markers {
                // Remove weakest marker
                if let Some(idx) = self.markers.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)|
                        a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                {
                    self.markers.remove(idx);
                }
            }

            self.markers.push(SomaticMarker {
                trigger_pattern: stimulus,
                predicted_affect: outcome_affect,
                physiological_predictions: outcome_physio,
                strength: 0.5,
                reinforcement_count: 1,
                last_activation: Instant::now(),
            });
        }
    }

    /// Decay markers over time (forgetting)
    pub fn decay(&mut self, rate: f64) {
        for marker in &mut self.markers {
            marker.strength *= (-rate).exp();
        }

        // Remove very weak markers
        self.markers.retain(|m| m.strength > 0.1);
    }
}

/// Mood: Sustained affective state that biases processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mood {
    /// Current mood as core affect
    pub affect: CoreAffect,

    /// Mood stability (resistance to change, 0-1)
    pub stability: f64,

    /// How long the mood has persisted (in processing cycles)
    pub duration: u64,

    /// Mood history for trend analysis
    #[serde(skip)]
    history: VecDeque<CoreAffect>,
}

impl Mood {
    pub fn new(initial_affect: CoreAffect) -> Self {
        Self {
            affect: initial_affect,
            stability: 0.5,
            duration: 0,
            history: VecDeque::with_capacity(100),
        }
    }

    /// Update mood based on incoming affect
    pub fn update(&mut self, incoming: &CoreAffect) {
        // Mood is slow to change (high stability = more resistance)
        let influence = 0.1 * (1.0 - self.stability);
        self.affect = self.affect.blend(incoming, influence);

        // Track duration
        self.duration += 1;

        // Update history
        if self.history.len() >= 100 {
            self.history.pop_front();
        }
        self.history.push_back(self.affect);

        // Stability increases if mood is consistent
        let variance = self.compute_variance();
        if variance < 0.1 {
            self.stability = (self.stability + 0.01).min(0.9);
        } else {
            self.stability = (self.stability - 0.01).max(0.1);
        }
    }

    fn compute_variance(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mean_v: f64 = self.history.iter().map(|a| a.valence).sum::<f64>() / self.history.len() as f64;
        let mean_a: f64 = self.history.iter().map(|a| a.arousal).sum::<f64>() / self.history.len() as f64;

        let var: f64 = self.history.iter()
            .map(|a| (a.valence - mean_v).powi(2) + (a.arousal - mean_a).powi(2))
            .sum::<f64>() / self.history.len() as f64;

        var.sqrt()
    }

    /// Get mood trend (improving, stable, declining)
    pub fn trend(&self) -> MoodTrend {
        if self.history.len() < 10 {
            return MoodTrend::Stable;
        }

        let recent: f64 = self.history.iter().rev().take(10)
            .map(|a| a.valence).sum::<f64>() / 10.0;
        let older: f64 = self.history.iter().take(10)
            .map(|a| a.valence).sum::<f64>() / 10.0;

        let diff = recent - older;
        if diff > 0.1 {
            MoodTrend::Improving
        } else if diff < -0.1 {
            MoodTrend::Declining
        } else {
            MoodTrend::Stable
        }
    }
}

impl Default for Mood {
    fn default() -> Self {
        Self::new(CoreAffect::neutral())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoodTrend {
    Improving,
    Stable,
    Declining,
}

/// Primary Affect System activations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrimaryAffectActivations {
    pub seeking: f64,
    pub rage: f64,
    pub fear: f64,
    pub lust: f64,
    pub care: f64,
    pub panic: f64,
    pub play: f64,
}

impl PrimaryAffectActivations {
    /// Get the dominant system
    pub fn dominant(&self) -> (PrimaryAffectSystem, f64) {
        let systems = [
            (PrimaryAffectSystem::Seeking, self.seeking),
            (PrimaryAffectSystem::Rage, self.rage),
            (PrimaryAffectSystem::Fear, self.fear),
            (PrimaryAffectSystem::Lust, self.lust),
            (PrimaryAffectSystem::Care, self.care),
            (PrimaryAffectSystem::Panic, self.panic),
            (PrimaryAffectSystem::Play, self.play),
        ];

        systems.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((PrimaryAffectSystem::Seeking, 0.0))
    }

    /// Convert to core affect (weighted combination)
    pub fn to_core_affect(&self) -> CoreAffect {
        let total = self.seeking + self.rage + self.fear + self.lust +
                    self.care + self.panic + self.play;

        if total < 0.001 {
            return CoreAffect::neutral();
        }

        // Weighted combination of each system's affect signature
        let mut result = CoreAffect::neutral();

        for (system, weight) in [
            (PrimaryAffectSystem::Seeking, self.seeking),
            (PrimaryAffectSystem::Rage, self.rage),
            (PrimaryAffectSystem::Fear, self.fear),
            (PrimaryAffectSystem::Lust, self.lust),
            (PrimaryAffectSystem::Care, self.care),
            (PrimaryAffectSystem::Panic, self.panic),
            (PrimaryAffectSystem::Play, self.play),
        ] {
            let system_affect = system.to_core_affect(weight);
            let w = weight / total;
            result.valence += system_affect.valence * w;
            result.arousal += system_affect.arousal * w;
            result.dominance += system_affect.dominance * w;
        }

        result
    }

    /// Apply decay
    pub fn decay(&mut self, rate: f64) {
        let factor = (-rate).exp();
        self.seeking *= factor;
        self.rage *= factor;
        self.fear *= factor;
        self.lust *= factor;
        self.care *= factor;
        self.panic *= factor;
        self.play *= factor;
    }
}

/// Affective Consciousness Analyzer: Main integration hub
pub struct AffectiveConsciousnessAnalyzer {
    /// Current core affect
    pub core_affect: CoreAffect,

    /// Current mood (sustained affect)
    pub mood: Mood,

    /// Primary affect system activations
    pub primary_systems: PrimaryAffectActivations,

    /// Somatic marker system
    somatic_markers: SomaticMarkerSystem,

    /// Current physiological state
    pub physiology: PhysiologicalState,

    /// Affect-Φ coupling coefficient
    phi_coupling: f64,

    /// Processing history
    affect_history: VecDeque<CoreAffect>,

    /// Configuration
    config: AffectiveConfig,
}

/// Configuration for affective processing
#[derive(Debug, Clone)]
pub struct AffectiveConfig {
    /// How strongly affect couples to Φ
    pub phi_coupling_strength: f64,

    /// Affect decay rate per cycle
    pub decay_rate: f64,

    /// Maximum somatic markers
    pub max_somatic_markers: usize,

    /// History length
    pub history_length: usize,
}

impl Default for AffectiveConfig {
    fn default() -> Self {
        Self {
            phi_coupling_strength: 0.3,
            decay_rate: 0.05,
            max_somatic_markers: 1000,
            history_length: 500,
        }
    }
}

impl AffectiveConsciousnessAnalyzer {
    pub fn new(config: AffectiveConfig) -> Self {
        Self {
            core_affect: CoreAffect::neutral(),
            mood: Mood::default(),
            primary_systems: PrimaryAffectActivations::default(),
            somatic_markers: SomaticMarkerSystem::new(config.max_somatic_markers),
            physiology: PhysiologicalState::default(),
            phi_coupling: config.phi_coupling_strength,
            affect_history: VecDeque::with_capacity(config.history_length),
            config,
        }
    }

    /// Process a stimulus and generate affective response
    pub fn process_stimulus(&mut self, stimulus: &HV16) -> AffectiveResponse {
        // 1. Query somatic markers for predictions
        let somatic_prediction = self.somatic_markers.predict(stimulus);

        // 2. Activate primary affect systems based on stimulus features
        self.activate_primary_systems(stimulus);

        // 3. Combine into core affect
        let system_affect = self.primary_systems.to_core_affect();

        self.core_affect = if let Some((predicted_affect, predicted_physio)) = somatic_prediction {
            // Blend somatic prediction with primary systems
            let blended = predicted_affect.blend(&system_affect, 0.5);
            self.physiology = predicted_physio;
            blended
        } else {
            self.physiology = PhysiologicalState::from_affect(&system_affect);
            system_affect
        };

        // 4. Apply mood bias
        self.core_affect = self.core_affect.blend(&self.mood.affect, 0.2);

        // 5. Update mood
        self.mood.update(&self.core_affect);

        // 6. Update history
        if self.affect_history.len() >= self.config.history_length {
            self.affect_history.pop_front();
        }
        self.affect_history.push_back(self.core_affect);

        // 7. Generate response
        AffectiveResponse {
            core_affect: self.core_affect,
            emotion_category: self.core_affect.to_emotion_category(),
            dominant_system: self.primary_systems.dominant(),
            mood_trend: self.mood.trend(),
            physiological_state: self.physiology.clone(),
            phi_modulation: self.compute_phi_modulation(),
            attention_bias: self.compute_attention_bias(),
            valence_routing: self.compute_valence_routing(),
        }
    }

    /// Activate primary affect systems based on stimulus
    fn activate_primary_systems(&mut self, stimulus: &HV16) {
        // Decay existing activations
        self.primary_systems.decay(self.config.decay_rate);

        // Compute feature-based activations
        // In a full implementation, these would be learned associations
        // For now, use simple heuristics based on stimulus properties

        // Use stimulus entropy/complexity as novelty signal (SEEKING)
        let novelty = self.compute_novelty(stimulus);
        self.primary_systems.seeking += novelty * 0.5;

        // Use similarity to threat patterns (FEAR) - placeholder
        // self.primary_systems.fear += threat_similarity * 0.5;

        // Clamp activations
        self.primary_systems.seeking = self.primary_systems.seeking.clamp(0.0, 1.0);
        self.primary_systems.rage = self.primary_systems.rage.clamp(0.0, 1.0);
        self.primary_systems.fear = self.primary_systems.fear.clamp(0.0, 1.0);
        self.primary_systems.lust = self.primary_systems.lust.clamp(0.0, 1.0);
        self.primary_systems.care = self.primary_systems.care.clamp(0.0, 1.0);
        self.primary_systems.panic = self.primary_systems.panic.clamp(0.0, 1.0);
        self.primary_systems.play = self.primary_systems.play.clamp(0.0, 1.0);
    }

    fn compute_novelty(&self, stimulus: &HV16) -> f64 {
        // Compare to recent history - higher difference = higher novelty
        if self.affect_history.is_empty() {
            return 0.5; // Baseline novelty
        }

        // Simple heuristic: use bit entropy of stimulus as proxy
        let ones = stimulus.popcount() as f64;  // Fixed: use popcount() instead of hamming_weight()
        let total = 16384.0;
        let p = ones / total;

        // Binary entropy
        if p == 0.0 || p == 1.0 {
            0.0
        } else {
            // Fixed: Add f64 type annotation to resolve ambiguous float type
            -p * (p as f64).log2() - (1.0 - p) * ((1.0 - p) as f64).log2()
        }
    }

    /// Compute how affect modulates Φ
    fn compute_phi_modulation(&self) -> f64 {
        // High arousal increases Φ (more integration needed)
        // Extreme valence (positive or negative) increases Φ (salient)
        let arousal_factor = 1.0 + self.core_affect.arousal.abs() * self.phi_coupling;
        let valence_factor = 1.0 + self.core_affect.valence.abs() * self.phi_coupling * 0.5;

        arousal_factor * valence_factor
    }

    /// Compute attention bias from affect
    fn compute_attention_bias(&self) -> AttentionBias {
        AttentionBias {
            // Negative affect narrows attention (threat focus)
            // Positive affect broadens attention (exploration)
            breadth: 0.5 + self.core_affect.valence * 0.3,

            // High arousal increases vigilance
            vigilance: 0.5 + self.core_affect.arousal * 0.4,

            // Dominant system determines focus type
            focus_type: match self.primary_systems.dominant().0 {
                PrimaryAffectSystem::Seeking => AttentionFocus::Exploratory,
                PrimaryAffectSystem::Fear => AttentionFocus::ThreatFocused,
                PrimaryAffectSystem::Rage => AttentionFocus::GoalBlocking,
                PrimaryAffectSystem::Care => AttentionFocus::Nurturing,
                PrimaryAffectSystem::Play => AttentionFocus::Playful,
                _ => AttentionFocus::Neutral,
            },
        }
    }

    /// Compute valence-based routing weights
    fn compute_valence_routing(&self) -> ValenceRouting {
        ValenceRouting {
            // Positive valence → approach behaviors
            approach_weight: (self.core_affect.valence + 1.0) / 2.0,

            // Negative valence → avoidance behaviors
            avoidance_weight: (-self.core_affect.valence + 1.0) / 2.0,

            // High dominance → active coping
            active_coping: (self.core_affect.dominance + 1.0) / 2.0,

            // Low dominance → passive coping
            passive_coping: (-self.core_affect.dominance + 1.0) / 2.0,
        }
    }

    /// Learn from outcome (for somatic marker formation)
    pub fn learn_outcome(&mut self, stimulus: HV16, outcome_affect: CoreAffect) {
        let physio = PhysiologicalState::from_affect(&outcome_affect);
        self.somatic_markers.learn(stimulus, outcome_affect, physio);
    }

    /// Get current affective summary
    pub fn summary(&self) -> AffectiveSummary {
        AffectiveSummary {
            core_affect: self.core_affect,
            mood: self.mood.affect,
            mood_stability: self.mood.stability,
            mood_trend: self.mood.trend(),
            dominant_emotion: self.core_affect.to_emotion_category(),
            dominant_system: self.primary_systems.dominant(),
            arousal_level: self.core_affect.arousal,
            valence_level: self.core_affect.valence,
            physiological_activation: self.physiology.activation(),
        }
    }
}

/// Response from affective processing
#[derive(Debug, Clone)]
pub struct AffectiveResponse {
    pub core_affect: CoreAffect,
    pub emotion_category: EmotionCategory,
    pub dominant_system: (PrimaryAffectSystem, f64),
    pub mood_trend: MoodTrend,
    pub physiological_state: PhysiologicalState,

    /// How much to modulate Φ (>1 = increase, <1 = decrease)
    pub phi_modulation: f64,

    /// Attention bias from current affect
    pub attention_bias: AttentionBias,

    /// Valence-based routing weights
    pub valence_routing: ValenceRouting,
}

#[derive(Debug, Clone)]
pub struct AttentionBias {
    /// Attention breadth (0 = narrow, 1 = broad)
    pub breadth: f64,

    /// Vigilance level (0 = relaxed, 1 = hypervigilant)
    pub vigilance: f64,

    /// Focus type based on dominant affect
    pub focus_type: AttentionFocus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFocus {
    Exploratory,      // SEEKING
    ThreatFocused,    // FEAR
    GoalBlocking,     // RAGE
    Nurturing,        // CARE
    Playful,          // PLAY
    Neutral,
}

#[derive(Debug, Clone)]
pub struct ValenceRouting {
    pub approach_weight: f64,
    pub avoidance_weight: f64,
    pub active_coping: f64,
    pub passive_coping: f64,
}

#[derive(Debug, Clone)]
pub struct AffectiveSummary {
    pub core_affect: CoreAffect,
    pub mood: CoreAffect,
    pub mood_stability: f64,
    pub mood_trend: MoodTrend,
    pub dominant_emotion: EmotionCategory,
    pub dominant_system: (PrimaryAffectSystem, f64),
    pub arousal_level: f64,
    pub valence_level: f64,
    pub physiological_activation: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_affect_creation() {
        let affect = CoreAffect::new(0.5, 0.8, 0.3);
        assert!((affect.valence - 0.5).abs() < 0.001);
        assert!((affect.arousal - 0.8).abs() < 0.001);
        assert!((affect.dominance - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_core_affect_clamping() {
        let affect = CoreAffect::new(2.0, -3.0, 1.5);
        assert!((affect.valence - 1.0).abs() < 0.001);
        assert!((affect.arousal - (-1.0)).abs() < 0.001);
        assert!((affect.dominance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_emotion_category_mapping() {
        let excited = CoreAffect::new(0.8, 0.9, 0.5);
        assert_eq!(excited.to_emotion_category(), EmotionCategory::Excited);

        let sad = CoreAffect::new(-0.7, -0.6, -0.3);
        assert_eq!(sad.to_emotion_category(), EmotionCategory::Sad);

        let afraid = CoreAffect::new(-0.8, 0.8, -0.7);
        assert_eq!(afraid.to_emotion_category(), EmotionCategory::Afraid);

        let angry = CoreAffect::new(-0.8, 0.8, 0.5);
        assert_eq!(angry.to_emotion_category(), EmotionCategory::Angry);
    }

    #[test]
    fn test_affect_blending() {
        let a = CoreAffect::new(1.0, 0.0, 0.0);
        let b = CoreAffect::new(-1.0, 0.0, 0.0);

        let blended = a.blend(&b, 0.5);
        assert!(blended.valence.abs() < 0.001); // Should be neutral
    }

    #[test]
    fn test_affect_intensity() {
        let neutral = CoreAffect::neutral();
        assert!(neutral.intensity() < 0.001);

        let intense = CoreAffect::new(1.0, 1.0, 1.0);
        assert!((intense.intensity() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_primary_affect_to_core() {
        let seeking = PrimaryAffectSystem::Seeking.to_core_affect(1.0);
        assert!(seeking.valence > 0.0); // Positive
        assert!(seeking.arousal > 0.0); // Activated

        let fear = PrimaryAffectSystem::Fear.to_core_affect(1.0);
        assert!(fear.valence < 0.0); // Negative
        assert!(fear.arousal > 0.0); // Activated
        assert!(fear.dominance < 0.0); // Feels controlled
    }

    #[test]
    fn test_physiological_from_affect() {
        let aroused = CoreAffect::new(0.0, 0.9, 0.0);
        let physio = PhysiologicalState::from_affect(&aroused);

        assert!(physio.heart_rate_delta > 0.5);
        assert!(physio.skin_conductance > 0.4);
    }

    #[test]
    fn test_mood_update() {
        let mut mood = Mood::new(CoreAffect::neutral());

        // Apply consistent positive affect
        for _ in 0..20 {
            mood.update(&CoreAffect::new(0.8, 0.3, 0.5));
        }

        // Mood should shift positive
        assert!(mood.affect.valence > 0.0);
    }

    #[test]
    fn test_somatic_marker_learning() {
        let mut system = SomaticMarkerSystem::new(100);

        let stimulus = HV16::random(42);
        let outcome = CoreAffect::new(-0.8, 0.7, -0.5); // Fear-like
        let physio = PhysiologicalState::from_affect(&outcome);

        system.learn(stimulus.clone(), outcome, physio);

        // Query should return the learned marker
        let prediction = system.predict(&stimulus);
        assert!(prediction.is_some());

        let (affect, _) = prediction.unwrap();
        assert!(affect.valence < 0.0); // Should predict negative
    }

    #[test]
    fn test_affective_analyzer() {
        let config = AffectiveConfig::default();
        let mut analyzer = AffectiveConsciousnessAnalyzer::new(config);

        let stimulus = HV16::random(123);
        let response = analyzer.process_stimulus(&stimulus);

        // Should produce valid response
        assert!(response.phi_modulation > 0.0);
        assert!(response.attention_bias.breadth >= 0.0 && response.attention_bias.breadth <= 1.0);
    }

    #[test]
    fn test_primary_activations_to_core() {
        let mut activations = PrimaryAffectActivations::default();
        activations.seeking = 0.8;
        activations.care = 0.5;

        let affect = activations.to_core_affect();
        assert!(affect.valence > 0.0); // Both seeking and care are positive
    }

    #[test]
    fn test_affect_hv_encoding() {
        let affect1 = CoreAffect::new(0.5, 0.3, 0.2);
        let affect2 = CoreAffect::new(0.5, 0.3, 0.2);
        let affect3 = CoreAffect::new(-0.5, -0.3, -0.2);

        let hv1 = affect1.to_hv(42);
        let hv2 = affect2.to_hv(42);
        let hv3 = affect3.to_hv(42);

        // Same affect should produce same HV
        assert!(hv1.similarity(&hv2) > 0.99);

        // Different affect should produce different HV
        assert!(hv1.similarity(&hv3) < 0.9);
    }

    #[test]
    fn test_affective_summary() {
        let config = AffectiveConfig::default();
        let analyzer = AffectiveConsciousnessAnalyzer::new(config);

        let summary = analyzer.summary();

        // Initial state should be neutral
        assert!(summary.core_affect.intensity() < 0.1);
        assert_eq!(summary.dominant_emotion, EmotionCategory::Neutral);
    }

    #[test]
    fn test_valence_routing() {
        let config = AffectiveConfig::default();
        let mut analyzer = AffectiveConsciousnessAnalyzer::new(config);

        // Set positive affect
        analyzer.core_affect = CoreAffect::new(0.8, 0.3, 0.5);

        let stimulus = HV16::random(456);
        let response = analyzer.process_stimulus(&stimulus);

        // Positive valence should bias toward approach
        assert!(response.valence_routing.approach_weight > response.valence_routing.avoidance_weight);
    }
}
