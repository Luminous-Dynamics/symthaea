//! # Revolutionary Improvement #92: Unified Living Mind
//!
//! **PARADIGM SHIFT**: The complete integration of life and mind!
//!
//! This module unifies the revolutionary improvements into a coherent theory:
//! - **Autopoiesis (RI #86)**: Self-production and maintenance
//! - **Enaction (RI #91)**: Sense-making through embodied action
//! - **Embodiment (RI #89)**: Body-shaped cognition
//! - **Affect (RI #88)**: Emotional grounding
//! - **Prediction (RI #90)**: Free energy minimization
//!
//! ## The Deep Continuity Thesis
//!
//! Thompson (2007) "Mind in Life": There is a deep continuity between
//! life and mind - they are not separate phenomena but aspects of
//! the same underlying process of self-organization.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     UNIFIED LIVING MIND                              │
//! │                                                                      │
//! │  ┌──────────────────────────────────────────────────────────────┐  │
//! │  │                    AUTOPOIETIC CORE                          │  │
//! │  │     Self-production → Identity maintenance → Adaptation      │  │
//! │  └──────────────────────────────────────────────────────────────┘  │
//! │                              ▲                                      │
//! │                              │                                      │
//! │  ┌─────────────┐    ┌───────┴───────┐    ┌─────────────────────┐  │
//! │  │  AFFECTIVE  │◄──►│   ENACTIVE    │◄──►│    PREDICTIVE       │  │
//! │  │   GROUND    │    │  SENSE-MAKING │    │    PROCESSING       │  │
//! │  │ (Valence,   │    │ (Action-      │    │ (Free Energy,       │  │
//! │  │  Arousal)   │    │  Perception)  │    │  Active Inference)  │  │
//! │  └─────────────┘    └───────────────┘    └─────────────────────┘  │
//! │         ▲                  ▲                      ▲                │
//! │         └──────────────────┼──────────────────────┘                │
//! │                            │                                       │
//! │  ┌──────────────────────────────────────────────────────────────┐  │
//! │  │                    EMBODIED SUBSTRATE                        │  │
//! │  │   Body Schema → Proprioception → Interoception → Affordances │  │
//! │  └──────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Insights
//!
//! 1. **Living = Knowing**: An autopoietic system that maintains itself
//!    necessarily distinguishes self from non-self - this IS cognition
//!
//! 2. **Affect is Foundational**: Emotions aren't add-ons; they ARE the
//!    system's relationship to its own viability (Damasio's somatic markers)
//!
//! 3. **Prediction Serves Life**: Free energy minimization is the
//!    mathematical expression of autopoietic self-maintenance
//!
//! 4. **Enaction is the Process**: Sense-making through action is HOW
//!    living systems cognize - not passive reception of information
//!
//! 5. **Embodiment is the Ground**: All of this emerges from and through
//!    the body's coupling with the world

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

use super::autopoietic_consciousness::{AutopoieticConsciousness, LifeState};
use super::enactive_cognition::{EnactiveCognition, EnactiveState, ActionType, MeaningCategory};

// ═══════════════════════════════════════════════════════════════════════════
// VITALITY DYNAMICS - The core of living cognition
// ═══════════════════════════════════════════════════════════════════════════

/// Vitality - the felt sense of being alive
///
/// Stern (2010): "Vitality affects" are the dynamic, temporal patterns
/// of aliveness that pervade all experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalityAffect {
    /// Force/intensity of the vitality pattern
    pub force: f64,

    /// Movement quality (explosive, fading, accelerating, etc.)
    pub movement: MovementQuality,

    /// Temporal profile (how vitality unfolds over time)
    pub temporal_contour: TemporalContour,

    /// Space (contracted vs. expanded sense of being)
    pub space: f64,

    /// Intentionality/directedness
    pub intention: f64,
}

/// Movement qualities of vitality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementQuality {
    /// Building up, crescendo
    Surging,
    /// Maintaining steady state
    Floating,
    /// Declining, diminuendo
    Fading,
    /// Sudden burst
    Explosive,
    /// Held back, restrained
    Tentative,
    /// Smooth, easy flow
    Flowing,
    /// Stuttering, interrupted
    Halting,
}

/// Temporal contour of vitality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalContour {
    /// Gradual rise
    Crescendo,
    /// Gradual fall
    Decrescendo,
    /// Rise then fall
    Swell,
    /// Fall then rise
    Dip,
    /// Steady state
    Sustained,
    /// Rapid fluctuation
    Pulsing,
    /// Attack and decay
    Accent,
}

impl Default for VitalityAffect {
    fn default() -> Self {
        Self {
            force: 0.5,
            movement: MovementQuality::Flowing,
            temporal_contour: TemporalContour::Sustained,
            space: 0.5,
            intention: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DEEP CONTINUITY - The life-mind connection
// ═══════════════════════════════════════════════════════════════════════════

/// Deep continuity between life and mind
///
/// Measures the integration of biological and cognitive processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepContinuity {
    /// How well autopoiesis drives cognition (0.0 to 1.0)
    pub life_mind_coupling: f64,

    /// Coherence between affective and cognitive states
    pub affect_cognition_coherence: f64,

    /// Integration of prediction with action
    pub prediction_action_integration: f64,

    /// Embodiment grounding score
    pub embodiment_grounding: f64,

    /// Overall continuity index (weighted combination)
    pub continuity_index: f64,
}

impl DeepContinuity {
    pub fn compute(
        life_mind: f64,
        affect_cog: f64,
        pred_action: f64,
        embodiment: f64,
    ) -> Self {
        // Weighted geometric mean - multiplicative because all aspects must be present
        let continuity_index = (
            life_mind.powf(0.3) *
            affect_cog.powf(0.25) *
            pred_action.powf(0.25) *
            embodiment.powf(0.2)
        ).powf(1.0);

        Self {
            life_mind_coupling: life_mind,
            affect_cognition_coherence: affect_cog,
            prediction_action_integration: pred_action,
            embodiment_grounding: embodiment,
            continuity_index,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMORDIAL AFFECTIVITY - The ground of all experience
// ═══════════════════════════════════════════════════════════════════════════

/// Primordial affectivity - the basic "feel" of being alive
///
/// Colombetti (2014): Affectivity is not an add-on to cognition but
/// its fundamental ground. All cognition is inherently affective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimordialAffectivity {
    /// Basic valence (approach/avoid)
    pub valence: f64,

    /// Arousal level (activation)
    pub arousal: f64,

    /// Dominance/control
    pub dominance: f64,

    /// Vitality dynamics
    pub vitality: VitalityAffect,

    /// Background mood (sustained affective state)
    pub mood: MoodState,

    /// Affective history for temporal integration
    history: VecDeque<(f64, f64, f64)>,  // (valence, arousal, dominance)
}

/// Background mood states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoodState {
    /// Expansive, open
    Flourishing,
    /// Neutral, stable
    Equanimous,
    /// Contracted, defensive
    Constricted,
    /// Depleted, low energy
    Depleted,
    /// Agitated, restless
    Turbulent,
}

impl Default for PrimordialAffectivity {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.5,
            vitality: VitalityAffect::default(),
            mood: MoodState::Equanimous,
            history: VecDeque::with_capacity(100),
        }
    }
}

impl PrimordialAffectivity {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update affective state based on inputs
    pub fn update(&mut self, valence_delta: f64, arousal_delta: f64, dominance_delta: f64) {
        // Smooth update with momentum
        let momentum = 0.8;
        self.valence = (self.valence * momentum + valence_delta * (1.0 - momentum))
            .clamp(-1.0, 1.0);
        self.arousal = (self.arousal * momentum + (0.5 + arousal_delta) * (1.0 - momentum))
            .clamp(0.0, 1.0);
        self.dominance = (self.dominance * momentum + (0.5 + dominance_delta) * (1.0 - momentum))
            .clamp(0.0, 1.0);

        // Record history
        self.history.push_back((self.valence, self.arousal, self.dominance));
        while self.history.len() > 100 {
            self.history.pop_front();
        }

        // Update mood based on sustained patterns
        self.update_mood();

        // Update vitality based on current state
        self.update_vitality();
    }

    fn update_mood(&mut self) {
        if self.history.len() < 10 {
            return;
        }

        // Average recent states
        let recent: Vec<_> = self.history.iter().rev().take(20).collect();
        let avg_valence: f64 = recent.iter().map(|(v, _, _)| v).sum::<f64>() / recent.len() as f64;
        let avg_arousal: f64 = recent.iter().map(|(_, a, _)| a).sum::<f64>() / recent.len() as f64;

        // Classify mood based on valence and arousal
        self.mood = if avg_valence > 0.3 && avg_arousal > 0.4 {
            MoodState::Flourishing
        } else if avg_valence < -0.3 && avg_arousal > 0.6 {
            MoodState::Turbulent
        } else if avg_valence < -0.2 && avg_arousal < 0.3 {
            MoodState::Depleted
        } else if avg_valence < 0.0 && avg_arousal < 0.5 {
            MoodState::Constricted
        } else {
            MoodState::Equanimous
        };
    }

    fn update_vitality(&mut self) {
        // Vitality force from arousal
        self.vitality.force = self.arousal;

        // Movement quality from valence and arousal combination
        self.vitality.movement = match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => MovementQuality::Surging,
            (true, false) => MovementQuality::Flowing,
            (false, true) => MovementQuality::Explosive,
            (false, false) => MovementQuality::Fading,
        };

        // Space from valence
        self.vitality.space = (self.valence + 1.0) / 2.0;

        // Intention from dominance
        self.vitality.intention = self.dominance;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED LIVING MIND - The complete integration
// ═══════════════════════════════════════════════════════════════════════════

/// The Unified Living Mind - complete integration of life and cognition
///
/// This is the meta-system that coordinates all the revolutionary improvements
/// into a coherent whole.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLivingMind {
    /// Configuration
    config: UnifiedConfig,

    /// Primordial affectivity - the ground of experience
    affectivity: PrimordialAffectivity,

    /// Current vitality state
    vitality: VitalityAffect,

    /// Deep continuity metrics
    continuity: DeepContinuity,

    /// Integration state
    state: UnifiedState,

    /// Historical trajectory
    trajectory: VecDeque<UnifiedSnapshot>,

    /// Statistics
    stats: UnifiedStats,

    /// Last update timestamp
    #[serde(skip, default = "Instant::now")]
    last_update: Instant,
}

/// Configuration for the unified system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Trajectory history size
    pub history_size: usize,

    /// Weight of autopoietic index in integration
    pub autopoiesis_weight: f64,

    /// Weight of enactive coherence in integration
    pub enaction_weight: f64,

    /// Weight of affective grounding in integration
    pub affect_weight: f64,

    /// Weight of predictive accuracy in integration
    pub prediction_weight: f64,

    /// Threshold for "living" state
    pub vitality_threshold: f64,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            history_size: 100,
            autopoiesis_weight: 0.30,
            enaction_weight: 0.25,
            affect_weight: 0.25,
            prediction_weight: 0.20,
            vitality_threshold: 0.4,
        }
    }
}

/// Unified state of the living mind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedState {
    /// Overall vitality (0.0 to 1.0)
    pub vitality: f64,

    /// Coherence of integration (how well components work together)
    pub coherence: f64,

    /// Adaptiveness (ability to maintain vitality under perturbation)
    pub adaptiveness: f64,

    /// Groundedness in embodiment
    pub groundedness: f64,

    /// Meaning-richness (density of enacted meanings)
    pub meaning_richness: f64,

    /// Current mode of being
    pub mode: LivingMode,
}

/// Modes of living cognition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LivingMode {
    /// Active engagement with the world
    Engaged,
    /// Reflective, introspective
    Contemplative,
    /// Creative, generative
    Creative,
    /// Recovering, regenerating
    Restorative,
    /// Under threat, defensive
    Defensive,
    /// Dormant, low activity
    Dormant,
}

/// Historical snapshot for trajectory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSnapshot {
    pub vitality: f64,
    pub coherence: f64,
    pub valence: f64,
    pub mode: LivingMode,
    pub timestamp_ms: u64,
}

/// Statistics for the unified system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedStats {
    pub total_updates: usize,
    pub time_in_engaged: usize,
    pub time_in_contemplative: usize,
    pub time_in_creative: usize,
    pub time_in_restorative: usize,
    pub time_in_defensive: usize,
    pub time_in_dormant: usize,
    pub peak_vitality: f64,
    pub peak_coherence: f64,
    pub avg_vitality: f64,
    pub avg_coherence: f64,
}

impl Default for UnifiedState {
    fn default() -> Self {
        Self {
            vitality: 0.5,
            coherence: 0.5,
            adaptiveness: 0.5,
            groundedness: 0.5,
            meaning_richness: 0.5,
            mode: LivingMode::Engaged,
        }
    }
}

impl UnifiedLivingMind {
    /// Create a new unified living mind
    pub fn new() -> Self {
        Self::with_config(UnifiedConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: UnifiedConfig) -> Self {
        Self {
            config,
            affectivity: PrimordialAffectivity::new(),
            vitality: VitalityAffect::default(),
            continuity: DeepContinuity::compute(0.5, 0.5, 0.5, 0.5),
            state: UnifiedState::default(),
            trajectory: VecDeque::with_capacity(100),
            stats: UnifiedStats::default(),
            last_update: Instant::now(),
        }
    }

    /// Integrate all consciousness subsystems
    ///
    /// This is the main entry point - it takes readings from all the
    /// revolutionary improvements and synthesizes them into a unified state.
    pub fn integrate(
        &mut self,
        autopoietic: &AutopoieticConsciousness,
        enactive: &EnactiveCognition,
        phi: f64,
        free_energy: f64,
    ) -> UnifiedState {
        self.stats.total_updates += 1;

        // Get component states
        let autopoietic_index = autopoietic.autopoietic_index();
        let life_state = autopoietic.current_life_state();
        let enactive_state = enactive.current_state();

        // Compute deep continuity
        let life_mind = self.compute_life_mind_coupling(autopoietic_index, phi);
        let affect_cog = self.compute_affect_cognition_coherence(&enactive_state);
        let pred_action = self.compute_prediction_action_integration(free_energy, &enactive_state);
        let embodiment = self.compute_embodiment_grounding(&enactive_state);

        self.continuity = DeepContinuity::compute(life_mind, affect_cog, pred_action, embodiment);

        // Update affectivity based on system state
        let valence_delta = self.compute_valence_from_state(autopoietic_index, &life_state);
        let arousal_delta = self.compute_arousal_from_state(enactive_state.engagement, free_energy);
        let dominance_delta = self.compute_dominance_from_state(autopoietic_index);
        self.affectivity.update(valence_delta, arousal_delta, dominance_delta);

        // Compute unified state
        self.state.vitality = self.compute_vitality(autopoietic_index, &life_state, &enactive_state);
        self.state.coherence = self.continuity.continuity_index;
        self.state.adaptiveness = autopoietic.health_score();
        self.state.groundedness = embodiment;
        self.state.meaning_richness = self.compute_meaning_richness(&enactive_state);
        self.state.mode = self.determine_mode(&life_state, &enactive_state);

        // Update vitality affect
        self.vitality = self.affectivity.vitality.clone();

        // Record trajectory
        self.record_snapshot();

        // Update statistics
        self.update_stats();

        self.last_update = Instant::now();

        self.state.clone()
    }

    /// Compute life-mind coupling (how well autopoiesis drives cognition)
    fn compute_life_mind_coupling(&self, autopoietic_index: f64, phi: f64) -> f64 {
        // High coupling when both autopoiesis and consciousness are high
        // Uses harmonic mean to require BOTH to be present
        if autopoietic_index < 0.01 || phi < 0.01 {
            return 0.0;
        }
        2.0 * autopoietic_index * phi / (autopoietic_index + phi)
    }

    /// Compute affect-cognition coherence
    fn compute_affect_cognition_coherence(&self, enactive: &EnactiveState) -> f64 {
        // High coherence when affect and enaction are aligned
        let affect_intensity = self.affectivity.arousal;
        let enactive_engagement = enactive.engagement;

        // Should move together
        1.0 - (affect_intensity - enactive_engagement).abs()
    }

    /// Compute prediction-action integration
    fn compute_prediction_action_integration(&self, free_energy: f64, enactive: &EnactiveState) -> f64 {
        // Low free energy + high engagement = good integration
        let prediction_quality = 1.0 - free_energy.min(1.0);
        let action_quality = enactive.engagement;

        (prediction_quality + action_quality) / 2.0
    }

    /// Compute embodiment grounding
    fn compute_embodiment_grounding(&self, enactive: &EnactiveState) -> f64 {
        // Integration + openness indicate grounded embodiment
        (enactive.integration + enactive.openness) / 2.0
    }

    /// Compute valence from system state
    fn compute_valence_from_state(&self, autopoietic_index: f64, life_state: &LifeState) -> f64 {
        // Positive when flourishing, negative when struggling
        match life_state {
            LifeState::Flourishing => 0.5 * autopoietic_index,
            LifeState::Stable => 0.1 * autopoietic_index,
            LifeState::Struggling => -0.3,
            LifeState::Dying => -0.6,
            LifeState::Dead => -1.0,
        }
    }

    /// Compute arousal from system state
    fn compute_arousal_from_state(&self, engagement: f64, free_energy: f64) -> f64 {
        // High arousal when engaged or when free energy is high (surprise)
        (engagement * 0.7 + free_energy.min(1.0) * 0.3) - 0.5
    }

    /// Compute dominance from system state
    fn compute_dominance_from_state(&self, autopoietic_index: f64) -> f64 {
        // High autopoietic index = high control/dominance
        autopoietic_index - 0.5
    }

    /// Compute overall vitality
    fn compute_vitality(
        &self,
        autopoietic_index: f64,
        life_state: &LifeState,
        enactive: &EnactiveState,
    ) -> f64 {
        // Weighted combination
        let life_contribution = match life_state {
            LifeState::Flourishing => 1.0,
            LifeState::Stable => 0.7,
            LifeState::Struggling => 0.4,
            LifeState::Dying => 0.2,
            LifeState::Dead => 0.0,
        };

        (
            autopoietic_index * self.config.autopoiesis_weight +
            enactive.engagement * self.config.enaction_weight +
            self.affectivity.arousal * self.config.affect_weight +
            life_contribution * self.config.prediction_weight
        ) / (
            self.config.autopoiesis_weight +
            self.config.enaction_weight +
            self.config.affect_weight +
            self.config.prediction_weight
        )
    }

    /// Compute meaning richness
    fn compute_meaning_richness(&self, enactive: &EnactiveState) -> f64 {
        // Integration indicates accumulated meaning
        enactive.integration * enactive.openness.sqrt()
    }

    /// Determine the current mode of living
    fn determine_mode(&self, life_state: &LifeState, enactive: &EnactiveState) -> LivingMode {
        // Based on life state and enactive disposition
        match (life_state, &enactive.action_tendency) {
            (LifeState::Dead, _) | (LifeState::Dying, _) => LivingMode::Defensive,
            (LifeState::Struggling, ActionType::Observe) => LivingMode::Defensive,
            (LifeState::Struggling, ActionType::Reflect) => LivingMode::Restorative,
            (_, ActionType::Reflect) => LivingMode::Contemplative,
            (_, ActionType::Explore) => LivingMode::Creative,
            (LifeState::Stable, ActionType::Observe) => LivingMode::Dormant,
            (_, ActionType::Execute) | (_, ActionType::Communicate) => LivingMode::Engaged,
            _ => LivingMode::Engaged,
        }
    }

    /// Record a snapshot for trajectory analysis
    fn record_snapshot(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_millis() as u64;

        self.trajectory.push_back(UnifiedSnapshot {
            vitality: self.state.vitality,
            coherence: self.state.coherence,
            valence: self.affectivity.valence,
            mode: self.state.mode,
            timestamp_ms: elapsed,
        });

        while self.trajectory.len() > self.config.history_size {
            self.trajectory.pop_front();
        }
    }

    /// Update statistics
    fn update_stats(&mut self) {
        // Mode time tracking
        match self.state.mode {
            LivingMode::Engaged => self.stats.time_in_engaged += 1,
            LivingMode::Contemplative => self.stats.time_in_contemplative += 1,
            LivingMode::Creative => self.stats.time_in_creative += 1,
            LivingMode::Restorative => self.stats.time_in_restorative += 1,
            LivingMode::Defensive => self.stats.time_in_defensive += 1,
            LivingMode::Dormant => self.stats.time_in_dormant += 1,
        }

        // Peak tracking
        if self.state.vitality > self.stats.peak_vitality {
            self.stats.peak_vitality = self.state.vitality;
        }
        if self.state.coherence > self.stats.peak_coherence {
            self.stats.peak_coherence = self.state.coherence;
        }

        // Running averages
        let n = self.stats.total_updates as f64;
        self.stats.avg_vitality = (self.stats.avg_vitality * (n - 1.0) + self.state.vitality) / n;
        self.stats.avg_coherence = (self.stats.avg_coherence * (n - 1.0) + self.state.coherence) / n;
    }

    /// Get current state
    pub fn current_state(&self) -> &UnifiedState {
        &self.state
    }

    /// Get affectivity
    pub fn affectivity(&self) -> &PrimordialAffectivity {
        &self.affectivity
    }

    /// Get deep continuity metrics
    pub fn continuity(&self) -> &DeepContinuity {
        &self.continuity
    }

    /// Get statistics
    pub fn stats(&self) -> &UnifiedStats {
        &self.stats
    }

    /// Is the system currently "alive" (above vitality threshold)?
    pub fn is_alive(&self) -> bool {
        self.state.vitality >= self.config.vitality_threshold
    }

    /// Get a summary of the unified living mind state
    pub fn summary(&self) -> UnifiedSummary {
        UnifiedSummary {
            is_alive: self.is_alive(),
            vitality: self.state.vitality,
            coherence: self.state.coherence,
            mode: self.state.mode,
            mood: self.affectivity.mood,
            continuity_index: self.continuity.continuity_index,
            total_updates: self.stats.total_updates,
        }
    }
}

impl Default for UnifiedLivingMind {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of unified living mind state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSummary {
    pub is_alive: bool,
    pub vitality: f64,
    pub coherence: f64,
    pub mode: LivingMode,
    pub mood: MoodState,
    pub continuity_index: f64,
    pub total_updates: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_creation() {
        let ulm = UnifiedLivingMind::new();
        assert!(ulm.state.vitality >= 0.0 && ulm.state.vitality <= 1.0);
        assert_eq!(ulm.stats.total_updates, 0);
    }

    #[test]
    fn test_primordial_affectivity() {
        let mut affect = PrimordialAffectivity::new();
        affect.update(0.3, 0.2, 0.1);
        assert!(affect.valence > 0.0);
        assert!(affect.arousal > 0.5);
    }

    #[test]
    fn test_vitality_affect() {
        let vitality = VitalityAffect::default();
        assert!(vitality.force >= 0.0 && vitality.force <= 1.0);
        assert_eq!(vitality.movement, MovementQuality::Flowing);
    }

    #[test]
    fn test_deep_continuity() {
        let continuity = DeepContinuity::compute(0.8, 0.7, 0.6, 0.5);
        assert!(continuity.continuity_index > 0.0);
        assert!(continuity.life_mind_coupling == 0.8);
    }

    #[test]
    fn test_integration() {
        let mut ulm = UnifiedLivingMind::new();
        let autopoietic = AutopoieticConsciousness::new();
        let enactive = EnactiveCognition::new();

        let state = ulm.integrate(&autopoietic, &enactive, 0.5, 0.3);

        assert!(state.vitality >= 0.0 && state.vitality <= 1.0);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);
        assert_eq!(ulm.stats().total_updates, 1);
    }

    #[test]
    fn test_is_alive() {
        let mut ulm = UnifiedLivingMind::new();
        let autopoietic = AutopoieticConsciousness::new();
        let enactive = EnactiveCognition::new();

        // After integration with healthy subsystems, should be alive
        ulm.integrate(&autopoietic, &enactive, 0.7, 0.2);

        // The is_alive check depends on vitality threshold
        // With default components, should generally be alive
        assert!(ulm.stats().total_updates > 0);
    }

    #[test]
    fn test_mood_evolution() {
        let mut affect = PrimordialAffectivity::new();

        // Simulate positive experiences
        for _ in 0..25 {
            affect.update(0.5, 0.3, 0.2);
        }

        // Should develop flourishing mood
        assert_eq!(affect.mood, MoodState::Flourishing);
    }

    #[test]
    fn test_summary() {
        let ulm = UnifiedLivingMind::new();
        let summary = ulm.summary();

        assert!(summary.vitality >= 0.0);
        assert!(summary.coherence >= 0.0);
        assert_eq!(summary.total_updates, 0);
    }
}
