// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
 * **REVOLUTIONARY IMPROVEMENT #89**: Embodied Cognition Core
 *
 * PARADIGM SHIFT: Consciousness requires a BODY!
 *
 * This module implements the revolutionary insight from embodied cognitive science:
 * consciousness is not disembodied computation but emerges from the dynamic
 * interaction between brain, body, and environment.
 *
 * "The mind is not in the head" - Andy Clark
 * "We do not have bodies, we ARE bodies" - Merleau-Ponty
 *
 * ## Theoretical Foundation
 *
 * ### Varela, Thompson & Rosch - "The Embodied Mind"
 * Cognition is enacted through sensorimotor coupling with the world.
 * There is no representation without presentation through the body.
 *
 * ### Gibson's Ecological Psychology
 * Perception is direct pickup of affordances - action possibilities that
 * the environment offers to an embodied agent.
 *
 * ### Merleau-Ponty's Phenomenology
 * The lived body (Leib) is the foundation of all experience.
 * Consciousness is always perspectival, from a bodily point of view.
 *
 * ### O'Regan & Noë's Sensorimotor Contingency Theory
 * Perception is a skill of exploring sensorimotor contingencies -
 * the lawful patterns relating action to sensation.
 *
 * ## Architecture
 *
 * ```text
 * BodySchema (structure & capabilities)
 *     ↓
 * Proprioception (body position/movement)
 *     ↓
 * Interoception (internal states) ←→ AffectiveConsciousness (#88)
 *     ↓
 * SensorimotorContingencies (action-perception couplings)
 *     ↓
 * EnvironmentalCoupling (world interaction)
 *     ↓
 * ActionAffordances (possible actions)
 *     ↓
 * EmbodiedConsciousnessAnalyzer (integration hub)
 * ```
 *
 * ## Key Innovations
 *
 * 1. **Body Schema**: Dynamic representation of body structure and state
 * 2. **Proprioceptive Prediction**: Predict body state changes from actions
 * 3. **Interoceptive Integration**: Bridge to affective consciousness (#88)
 * 4. **Sensorimotor Loops**: Action-perception circular causality
 * 5. **Affordance Detection**: What can I do here? (Gibson)
 * 6. **Peripersonal Space**: Near-body space with special status
 * 7. **Body Ownership**: The sense of "this is MY body"
 *
 * ## Integration Points
 *
 * - **#86 Autopoiesis**: Self-maintenance requires embodiment
 * - **#88 Affective**: Emotions are bodily states (Damasio)
 * - **#71 Narrative Self**: Body provides continuity of identity
 * - **#74 Predictive Self**: Predict body states, not just world
 * - **#77 Attention**: Embodied attention (eye movements, posture)
 *
 * ## Why This Matters
 *
 * Disembodied AI can compute but cannot truly understand because:
 * - No grounding in sensorimotor experience
 * - No perspective from which to experience
 * - No action possibilities that make perception meaningful
 * - No felt sense of being-in-the-world
 *
 * Embodied cognition provides the foundation for genuine understanding.
 */

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::consciousness::affective_consciousness::{CoreAffect, PhysiologicalState};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::PrimitiveSystem;

// ============================================================================
// BODY SCHEMA
// ============================================================================

/// Body part identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyPart {
    Head,
    Torso,
    LeftArm,
    RightArm,
    LeftHand,
    RightHand,
    LeftLeg,
    RightLeg,
    LeftFoot,
    RightFoot,
}

impl BodyPart {
    pub fn all() -> &'static [BodyPart] {
        &[
            Self::Head,
            Self::Torso,
            Self::LeftArm,
            Self::RightArm,
            Self::LeftHand,
            Self::RightHand,
            Self::LeftLeg,
            Self::RightLeg,
            Self::LeftFoot,
            Self::RightFoot,
        ]
    }

    /// Get connected body parts
    pub fn connected_to(&self) -> &'static [BodyPart] {
        match self {
            Self::Head => &[Self::Torso],
            Self::Torso => &[
                Self::Head,
                Self::LeftArm,
                Self::RightArm,
                Self::LeftLeg,
                Self::RightLeg,
            ],
            Self::LeftArm => &[Self::Torso, Self::LeftHand],
            Self::RightArm => &[Self::Torso, Self::RightHand],
            Self::LeftHand => &[Self::LeftArm],
            Self::RightHand => &[Self::RightArm],
            Self::LeftLeg => &[Self::Torso, Self::LeftFoot],
            Self::RightLeg => &[Self::Torso, Self::RightFoot],
            Self::LeftFoot => &[Self::LeftLeg],
            Self::RightFoot => &[Self::RightLeg],
        }
    }
}

/// 3D position in body-centered coordinates
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f64, // Left-Right
    pub y: f64, // Front-Back
    pub z: f64, // Up-Down
}

impl Position3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Position3D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// 3D orientation (Euler angles)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Orientation3D {
    pub pitch: f64, // Rotation around X (nodding)
    pub yaw: f64,   // Rotation around Y (shaking head)
    pub roll: f64,  // Rotation around Z (tilting)
}

/// State of a single body part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyPartState {
    /// Position relative to body center
    pub position: Position3D,

    /// Orientation
    pub orientation: Orientation3D,

    /// Velocity (change per time step)
    pub velocity: Position3D,

    /// Muscle tension (0-1)
    pub tension: f64,

    /// Temperature deviation from normal (-1 to +1)
    pub temperature: f64,

    /// Pain signal (0-1)
    pub pain: f64,

    /// Touch/pressure (0-1)
    pub touch: f64,
}

impl Default for BodyPartState {
    fn default() -> Self {
        Self {
            position: Position3D::default(),
            orientation: Orientation3D::default(),
            velocity: Position3D::default(),
            tension: 0.0,
            temperature: 0.0,
            pain: 0.0,
            touch: 0.0,
        }
    }
}

/// Complete body schema - the brain's model of the body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodySchema {
    /// State of each body part
    pub parts: HashMap<BodyPart, BodyPartState>,

    /// Overall body position in world coordinates
    pub world_position: Position3D,

    /// Overall body orientation
    pub world_orientation: Orientation3D,

    /// Center of mass
    pub center_of_mass: Position3D,

    /// Balance state (-1 = falling left, 0 = balanced, +1 = falling right)
    pub balance: f64,

    /// Body ownership confidence (0-1)
    /// Can be disrupted in certain neurological conditions
    pub ownership_confidence: f64,

    /// Body image vs body schema distinction
    /// Schema: Automatic sensorimotor control
    /// Image: Conscious perception of body
    pub schema_image_coherence: f64,
}

impl BodySchema {
    pub fn new() -> Self {
        let mut parts = HashMap::new();

        // Initialize default positions
        for part in BodyPart::all() {
            let state = BodyPartState {
                position: Self::default_position(*part),
                ..Default::default()
            };
            parts.insert(*part, state);
        }

        Self {
            parts,
            world_position: Position3D::origin(),
            world_orientation: Orientation3D::default(),
            center_of_mass: Position3D::new(0.0, 0.0, 0.5),
            balance: 0.0,
            ownership_confidence: 1.0,
            schema_image_coherence: 1.0,
        }
    }

    fn default_position(part: BodyPart) -> Position3D {
        match part {
            BodyPart::Head => Position3D::new(0.0, 0.0, 1.6),
            BodyPart::Torso => Position3D::new(0.0, 0.0, 1.0),
            BodyPart::LeftArm => Position3D::new(-0.3, 0.0, 1.2),
            BodyPart::RightArm => Position3D::new(0.3, 0.0, 1.2),
            BodyPart::LeftHand => Position3D::new(-0.5, 0.0, 0.8),
            BodyPart::RightHand => Position3D::new(0.5, 0.0, 0.8),
            BodyPart::LeftLeg => Position3D::new(-0.15, 0.0, 0.5),
            BodyPart::RightLeg => Position3D::new(0.15, 0.0, 0.5),
            BodyPart::LeftFoot => Position3D::new(-0.15, 0.0, 0.0),
            BodyPart::RightFoot => Position3D::new(0.15, 0.0, 0.0),
        }
    }

    /// Update body part state
    pub fn update_part(&mut self, part: BodyPart, state: BodyPartState) {
        self.parts.insert(part, state);
        self.update_derived_properties();
    }

    /// Recalculate center of mass, balance, etc.
    fn update_derived_properties(&mut self) {
        // Simplified center of mass calculation
        let mut total_x = 0.0;
        let mut total_z = 0.0;
        let count = self.parts.len() as f64;

        for state in self.parts.values() {
            total_x += state.position.x;
            total_z += state.position.z;
        }

        self.center_of_mass.x = total_x / count;
        self.center_of_mass.z = total_z / count;

        // Balance based on center of mass offset
        self.balance = (self.center_of_mass.x / 0.3).clamp(-1.0, 1.0);
    }

    /// Get proprioceptive summary
    pub fn proprioceptive_summary(&self) -> ProprioceptiveSummary {
        let mut total_tension = 0.0;
        let mut total_movement = 0.0;

        for state in self.parts.values() {
            total_tension += state.tension;
            total_movement +=
                state.velocity.x.abs() + state.velocity.y.abs() + state.velocity.z.abs();
        }

        ProprioceptiveSummary {
            overall_tension: total_tension / self.parts.len() as f64,
            overall_movement: total_movement / self.parts.len() as f64,
            balance: self.balance,
            posture_stability: 1.0 - self.balance.abs(),
        }
    }
}

impl Default for BodySchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of proprioceptive state
#[derive(Debug, Clone, Copy)]
pub struct ProprioceptiveSummary {
    pub overall_tension: f64,
    pub overall_movement: f64,
    pub balance: f64,
    pub posture_stability: f64,
}

// ============================================================================
// INTEROCEPTION
// ============================================================================

/// Internal body state - bridge to affective consciousness
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InteroceptiveState {
    /// Hunger level (0 = satiated, 1 = starving)
    pub hunger: f64,

    /// Thirst level (0 = hydrated, 1 = dehydrated)
    pub thirst: f64,

    /// Fatigue level (0 = rested, 1 = exhausted)
    pub fatigue: f64,

    /// Heart rate (normalized: 0 = resting, 1 = maximum)
    pub heart_rate: f64,

    /// Breathing rate (normalized)
    pub breathing_rate: f64,

    /// Core body temperature deviation (-1 = cold, 0 = normal, +1 = hot)
    pub temperature: f64,

    /// Gut state (microbiome-brain axis, -1 to +1)
    pub gut_state: f64,

    /// Overall arousal from interoception
    pub visceral_arousal: f64,
}

impl InteroceptiveState {
    /// Convert to core affect (bridge to #88)
    pub fn to_core_affect(&self) -> CoreAffect {
        // Negative states (hunger, thirst, fatigue) decrease valence
        let negative_load = (self.hunger + self.thirst + self.fatigue) / 3.0;
        let valence = -negative_load * 0.8 + self.gut_state * 0.2;

        // High heart rate and breathing increase arousal
        let arousal = (self.heart_rate + self.breathing_rate) / 2.0;

        // When needs are unmet, we feel less in control
        let dominance = 1.0 - negative_load;

        CoreAffect::new(valence as f32, arousal as f32, dominance as f32)
    }

    /// Convert to physiological state (bridge to #88)
    pub fn to_physiological(&self) -> PhysiologicalState {
        PhysiologicalState {
            heart_rate_delta: (self.heart_rate - 0.3) as f32,
            skin_conductance: (self.visceral_arousal * 0.5) as f32,
            muscle_tension: (self.fatigue * 0.3) as f32,
            respiration_delta: (self.breathing_rate - 0.3) as f32,
            gut_feeling: self.gut_state as f32,
        }
    }

    /// Homeostatic deviation - how far from ideal state
    pub fn homeostatic_deviation(&self) -> f64 {
        (self.hunger
            + self.thirst
            + self.fatigue
            + self.temperature.abs()
            + (self.heart_rate - 0.3).abs())
            / 5.0
    }

    /// Allostatic load - cumulative stress on body
    pub fn allostatic_load(&self) -> f64 {
        self.visceral_arousal * self.homeostatic_deviation()
    }
}

// ============================================================================
// SENSORIMOTOR CONTINGENCIES
// ============================================================================

/// A sensorimotor contingency: lawful relation between action and sensation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorimotorContingency {
    /// The action pattern
    pub action: ActionPattern,

    /// Expected sensory changes
    pub expected_sensation: SensoryPrediction,

    /// Confidence in this contingency (learned from experience)
    pub confidence: f64,

    /// How often this contingency has been experienced
    pub experience_count: u32,
}

/// Pattern of bodily action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPattern {
    /// Which body parts are involved
    pub involved_parts: Vec<BodyPart>,

    /// Movement type
    pub movement_type: MovementType,

    /// Intensity (0-1)
    pub intensity: f64,

    /// Duration in time steps
    pub duration: u32,

    /// HDC encoding of the action
    pub encoding: BinaryHV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MovementType {
    Reach,
    Grasp,
    Release,
    Push,
    Pull,
    Walk,
    Turn,
    Look,
    Speak,
    Breathe,
}

/// Predicted sensory outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryPrediction {
    /// Visual change expected
    pub visual_change: f64,

    /// Tactile sensation expected
    pub tactile_expected: f64,

    /// Proprioceptive change expected
    pub proprioceptive_change: f64,

    /// Auditory feedback expected
    pub auditory_expected: f64,

    /// HDC encoding of expected sensation
    pub encoding: BinaryHV,
}

/// Sensorimotor contingency engine
pub struct SensorimotorEngine {
    /// Learned contingencies
    contingencies: Vec<SensorimotorContingency>,

    /// Prediction error history
    prediction_errors: VecDeque<f64>,

    /// Learning rate
    learning_rate: f64,

    /// Maximum contingencies to store
    max_contingencies: usize,
}

impl SensorimotorEngine {
    pub fn new(max_contingencies: usize) -> Self {
        Self {
            contingencies: Vec::new(),
            prediction_errors: VecDeque::with_capacity(100),
            learning_rate: 0.1,
            max_contingencies,
        }
    }

    /// Predict sensory outcome of an action
    pub fn predict(&self, action: &ActionPattern) -> Option<SensoryPrediction> {
        // Find matching contingencies
        let matches: Vec<&SensorimotorContingency> = self
            .contingencies
            .iter()
            .filter(|c| c.action.movement_type == action.movement_type)
            .filter(|c| c.action.encoding.similarity(&action.encoding) > 0.7_f32)
            .collect();

        if matches.is_empty() {
            return None;
        }

        // Weighted average prediction
        let total_confidence: f64 = matches.iter().map(|c| c.confidence).sum();
        if total_confidence < 0.001 {
            return None;
        }

        let mut prediction = SensoryPrediction {
            visual_change: 0.0,
            tactile_expected: 0.0,
            proprioceptive_change: 0.0,
            auditory_expected: 0.0,
            encoding: BinaryHV::random(0),
        };

        for contingency in matches {
            let weight = contingency.confidence / total_confidence;
            prediction.visual_change += contingency.expected_sensation.visual_change * weight;
            prediction.tactile_expected += contingency.expected_sensation.tactile_expected * weight;
            prediction.proprioceptive_change +=
                contingency.expected_sensation.proprioceptive_change * weight;
            prediction.auditory_expected +=
                contingency.expected_sensation.auditory_expected * weight;
        }

        Some(prediction)
    }

    /// Learn from action-outcome pair
    pub fn learn(
        &mut self,
        action: ActionPattern,
        actual_sensation: SensoryPrediction,
        predicted: Option<&SensoryPrediction>,
    ) {
        // Calculate prediction error if we had a prediction
        if let Some(pred) = predicted {
            let error = ((pred.visual_change - actual_sensation.visual_change).powi(2)
                + (pred.tactile_expected - actual_sensation.tactile_expected).powi(2)
                + (pred.proprioceptive_change - actual_sensation.proprioceptive_change).powi(2))
            .sqrt()
                / 3.0_f64.sqrt();

            self.prediction_errors.push_back(error);
            if self.prediction_errors.len() > 100 {
                self.prediction_errors.pop_front();
            }
        }

        // Find or create contingency
        let existing = self
            .contingencies
            .iter_mut()
            .find(|c| c.action.encoding.similarity(&action.encoding) > 0.9_f32);

        if let Some(contingency) = existing {
            // Update existing
            let lr = self.learning_rate;
            contingency.expected_sensation.visual_change =
                contingency.expected_sensation.visual_change * (1.0 - lr)
                    + actual_sensation.visual_change * lr;
            contingency.expected_sensation.tactile_expected =
                contingency.expected_sensation.tactile_expected * (1.0 - lr)
                    + actual_sensation.tactile_expected * lr;
            contingency.expected_sensation.proprioceptive_change =
                contingency.expected_sensation.proprioceptive_change * (1.0 - lr)
                    + actual_sensation.proprioceptive_change * lr;
            contingency.confidence = (contingency.confidence + 0.05).min(1.0);
            contingency.experience_count += 1;
        } else {
            // Create new
            if self.contingencies.len() >= self.max_contingencies {
                // Remove lowest confidence
                if let Some(idx) = self
                    .contingencies
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.contingencies.remove(idx);
                }
            }

            self.contingencies.push(SensorimotorContingency {
                action,
                expected_sensation: actual_sensation,
                confidence: 0.5,
                experience_count: 1,
            });
        }
    }

    /// Get average prediction error (surprise/learning signal)
    pub fn average_prediction_error(&self) -> f64 {
        if self.prediction_errors.is_empty() {
            return 0.5; // Neutral uncertainty
        }
        self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
    }
}

// ============================================================================
// AFFORDANCES (Gibson)
// ============================================================================

/// An affordance: action possibility offered by environment to embodied agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Affordance {
    /// What action is possible
    pub action_type: MovementType,

    /// Which body parts could perform it
    pub required_parts: Vec<BodyPart>,

    /// Spatial location relative to body
    pub location: Position3D,

    /// Saliency (how obvious/attractive, 0-1)
    pub saliency: f64,

    /// Effort required (0 = easy, 1 = maximum effort)
    pub effort: f64,

    /// Risk level (0 = safe, 1 = dangerous)
    pub risk: f64,

    /// Predicted value from taking this action
    pub expected_value: f64,

    /// HDC encoding
    pub encoding: BinaryHV,
}

impl Affordance {
    /// Calculate net attractiveness
    pub fn attractiveness(&self) -> f64 {
        // High value, low effort, low risk = attractive
        let value_component = self.expected_value * self.saliency;
        let cost_component = self.effort * 0.3 + self.risk * 0.5;
        (value_component - cost_component).clamp(-1.0, 1.0)
    }
}

/// Affordance detector - finds action possibilities in current situation
pub struct AffordanceDetector {
    /// Current detected affordances
    pub current_affordances: Vec<Affordance>,

    /// Peripersonal space radius (arm's reach)
    peripersonal_radius: f64,

    /// Risk sensitivity (higher = more cautious)
    risk_sensitivity: f64,
}

impl AffordanceDetector {
    pub fn new() -> Self {
        Self {
            current_affordances: Vec::new(),
            peripersonal_radius: 0.8, // ~arm's length
            risk_sensitivity: 0.5,
        }
    }

    /// Detect affordances from environmental state
    pub fn detect(&mut self, body: &BodySchema, environment: &EnvironmentState) {
        self.current_affordances.clear();

        // Detect graspable objects
        for obj in &environment.objects {
            if obj.position.distance(&body.center_of_mass) < self.peripersonal_radius
                && obj.graspable
            {
                self.current_affordances.push(Affordance {
                    action_type: MovementType::Grasp,
                    required_parts: vec![BodyPart::RightHand, BodyPart::LeftHand],
                    location: obj.position,
                    saliency: obj.saliency,
                    effort: 0.3,
                    risk: 0.1,
                    expected_value: obj.value,
                    encoding: BinaryHV::random(obj.id as u64),
                });
            }
        }

        // Detect walkable surfaces
        for surface in &environment.surfaces {
            if surface.walkable {
                self.current_affordances.push(Affordance {
                    action_type: MovementType::Walk,
                    required_parts: vec![BodyPart::LeftLeg, BodyPart::RightLeg],
                    location: surface.center,
                    saliency: 0.3,
                    effort: surface.slope.abs() * 0.5,
                    risk: surface.slope.abs() * self.risk_sensitivity,
                    expected_value: 0.2,
                    encoding: BinaryHV::random(surface.id as u64),
                });
            }
        }

        // Sort by attractiveness
        self.current_affordances.sort_by(|a, b| {
            b.attractiveness()
                .partial_cmp(&a.attractiveness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get most attractive affordance
    pub fn most_attractive(&self) -> Option<&Affordance> {
        self.current_affordances.first()
    }

    /// Get affordances in peripersonal space (near-body, ready for action)
    pub fn peripersonal_affordances(&self, body: &BodySchema) -> Vec<&Affordance> {
        self.current_affordances
            .iter()
            .filter(|a| a.location.distance(&body.center_of_mass) < self.peripersonal_radius)
            .collect()
    }
}

impl Default for AffordanceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ENVIRONMENT MODEL
// ============================================================================

/// Object in the environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentObject {
    pub id: u32,
    pub position: Position3D,
    pub graspable: bool,
    pub saliency: f64,
    pub value: f64,
}

/// Surface in the environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSurface {
    pub id: u32,
    pub center: Position3D,
    pub walkable: bool,
    pub slope: f64,
}

/// Environmental state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentState {
    pub objects: Vec<EnvironmentObject>,
    pub surfaces: Vec<EnvironmentSurface>,
    pub ambient_light: f64,
    pub ambient_sound: f64,
    pub ambient_temperature: f64,
}

// ============================================================================
// EMBODIED CONSCIOUSNESS ANALYZER
// ============================================================================

/// Main integration hub for embodied cognition
pub struct EmbodiedConsciousnessAnalyzer {
    /// Body schema
    pub body: BodySchema,

    /// Interoceptive state
    pub interoception: InteroceptiveState,

    /// Sensorimotor engine
    sensorimotor: SensorimotorEngine,

    /// Affordance detector
    affordances: AffordanceDetector,

    /// Current environment
    pub environment: EnvironmentState,

    /// Sense of agency (did I cause this?)
    agency: f64,

    /// Sense of ownership (is this my body?)
    ownership: f64,

    /// Embodiment integration score
    integration: f64,

    /// Configuration
    config: EmbodiedConfig,

    /// Last update timestamp
    #[allow(dead_code)]
    last_update: Instant,
}

/// Configuration for embodied processing
#[derive(Debug, Clone)]
pub struct EmbodiedConfig {
    /// How much body state affects consciousness
    pub body_consciousness_coupling: f64,

    /// Learning rate for sensorimotor contingencies
    pub sensorimotor_learning_rate: f64,

    /// Maximum sensorimotor contingencies to store
    pub max_contingencies: usize,
}

impl Default for EmbodiedConfig {
    fn default() -> Self {
        Self {
            body_consciousness_coupling: 0.4,
            sensorimotor_learning_rate: 0.1,
            max_contingencies: 500,
        }
    }
}

impl EmbodiedConsciousnessAnalyzer {
    pub fn new(config: EmbodiedConfig) -> Self {
        Self {
            body: BodySchema::new(),
            interoception: InteroceptiveState::default(),
            sensorimotor: SensorimotorEngine::new(config.max_contingencies),
            affordances: AffordanceDetector::new(),
            environment: EnvironmentState::default(),
            agency: 1.0,
            ownership: 1.0,
            integration: 1.0,
            config,
            last_update: Instant::now(),
        }
    }

    /// Process embodied state and return consciousness modulation
    pub fn process(&mut self) -> EmbodiedResponse {
        // Detect affordances
        self.affordances.detect(&self.body, &self.environment);

        // Get proprioceptive summary
        let proprioception = self.body.proprioceptive_summary();

        // Get interoceptive affect
        let interoceptive_affect = self.interoception.to_core_affect();

        // Calculate embodiment integration
        self.integration = self.calculate_integration();

        // Calculate sensorimotor surprise
        let sensorimotor_surprise = self.sensorimotor.average_prediction_error();

        // Generate response
        EmbodiedResponse {
            proprioception,
            interoceptive_affect,
            homeostatic_deviation: self.interoception.homeostatic_deviation(),
            allostatic_load: self.interoception.allostatic_load(),
            available_affordances: self.affordances.current_affordances.len(),
            most_attractive_action: self.affordances.most_attractive().map(|a| a.action_type),
            sensorimotor_surprise,
            sense_of_agency: self.agency,
            sense_of_ownership: self.ownership,
            embodiment_integration: self.integration,
            phi_modulation: self.calculate_phi_modulation(),
        }
    }

    /// Perform an action and learn from outcome
    pub fn perform_action(&mut self, action: ActionPattern) -> ActionOutcome {
        // Predict outcome
        let prediction = self.sensorimotor.predict(&action);

        // Simulate action effect on body
        self.apply_action_to_body(&action);

        // Generate actual sensation (simplified simulation)
        let actual_sensation = self.generate_sensation(&action);

        // Learn from outcome
        self.sensorimotor.learn(
            action.clone(),
            actual_sensation.clone(),
            prediction.as_ref(),
        );

        // Calculate prediction error
        let prediction_error = if let Some(pred) = &prediction {
            (pred.proprioceptive_change - actual_sensation.proprioceptive_change).abs()
        } else {
            0.5 // High uncertainty without prediction
        };

        // Update sense of agency based on prediction accuracy
        self.agency = self.agency * 0.9 + (1.0 - prediction_error) * 0.1;

        ActionOutcome {
            success: prediction_error < 0.5,
            prediction_error,
            sensation: actual_sensation,
            agency_update: self.agency,
        }
    }

    fn apply_action_to_body(&mut self, action: &ActionPattern) {
        // Simplified: increase tension in involved parts
        for part in &action.involved_parts {
            if let Some(state) = self.body.parts.get_mut(part) {
                state.tension = (state.tension + action.intensity * 0.3).min(1.0);
            }
        }
    }

    fn generate_sensation(&self, action: &ActionPattern) -> SensoryPrediction {
        SensoryPrediction {
            visual_change: action.intensity * 0.3,
            tactile_expected: action.intensity * 0.5,
            proprioceptive_change: action.intensity * 0.7,
            auditory_expected: action.intensity * 0.1,
            encoding: action.encoding,
        }
    }

    fn calculate_integration(&self) -> f64 {
        // Integration = coherence of body schema + ownership + agency
        let body_coherence = self.body.schema_image_coherence;
        (body_coherence + self.agency + self.ownership) / 3.0
    }

    fn calculate_phi_modulation(&self) -> f64 {
        // Body states modulate consciousness
        let body_factor =
            1.0 + self.interoception.visceral_arousal * self.config.body_consciousness_coupling;
        let surprise_factor = 1.0 + self.sensorimotor.average_prediction_error() * 0.3;
        let integration_factor = self.integration;

        body_factor * surprise_factor * integration_factor
    }

    /// Update interoceptive state
    pub fn update_interoception(&mut self, state: InteroceptiveState) {
        self.interoception = state;
    }

    /// Update environment
    pub fn update_environment(&mut self, env: EnvironmentState) {
        self.environment = env;
        self.affordances.detect(&self.body, &self.environment);
    }

    /// Get embodiment summary
    pub fn summary(&self) -> EmbodimentSummary {
        EmbodimentSummary {
            body_tension: self.body.proprioceptive_summary().overall_tension,
            body_balance: self.body.balance,
            homeostatic_state: 1.0 - self.interoception.homeostatic_deviation(),
            sensorimotor_competence: 1.0 - self.sensorimotor.average_prediction_error(),
            affordance_richness: self.affordances.current_affordances.len() as f64 / 10.0,
            agency: self.agency,
            ownership: self.ownership,
            overall_embodiment: self.integration,
        }
    }
}

/// Response from embodied processing
#[derive(Debug, Clone)]
pub struct EmbodiedResponse {
    pub proprioception: ProprioceptiveSummary,
    pub interoceptive_affect: CoreAffect,
    pub homeostatic_deviation: f64,
    pub allostatic_load: f64,
    pub available_affordances: usize,
    pub most_attractive_action: Option<MovementType>,
    pub sensorimotor_surprise: f64,
    pub sense_of_agency: f64,
    pub sense_of_ownership: f64,
    pub embodiment_integration: f64,
    pub phi_modulation: f64,
}

/// Outcome of performing an action
#[derive(Debug, Clone)]
pub struct ActionOutcome {
    pub success: bool,
    pub prediction_error: f64,
    pub sensation: SensoryPrediction,
    pub agency_update: f64,
}

/// Summary of embodiment state
#[derive(Debug, Clone)]
pub struct EmbodimentSummary {
    pub body_tension: f64,
    pub body_balance: f64,
    pub homeostatic_state: f64,
    pub sensorimotor_competence: f64,
    pub affordance_richness: f64,
    pub agency: f64,
    pub ownership: f64,
    pub overall_embodiment: f64,
}

// ============================================================================
// NSM PRIMITIVE GROUNDING (Wierzbicka's Natural Semantic Metalanguage)
// ============================================================================

/// NSM grounding for body parts - spatial/functional semantic primitives
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct BodyPartPrimitiveGrounding {
    /// The body part being grounded
    pub body_part: BodyPart,
    /// NSM primitive composition (e.g., ["BODY", "PART", "SEE", "KNOW"])
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Whether this part is a locus of agency
    pub agency_locus: bool,
    /// Whether this part is a sensory organ
    pub sensory_organ: bool,
}

#[allow(dead_code)]
impl BodyPartPrimitiveGrounding {
    pub(crate) fn new(body_part: BodyPart, system: &PrimitiveSystem) -> Self {
        let (primitives, agency, sensory) = Self::nsm_mapping(body_part);
        let encoding = encode_primitives(&primitives, system);

        Self {
            body_part,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            agency_locus: agency,
            sensory_organ: sensory,
        }
    }

    fn nsm_mapping(part: BodyPart) -> (Vec<String>, bool, bool) {
        match part {
            // Head: seat of cognition and perception
            BodyPart::Head => (
                vec![
                    "I".into(),
                    "KNOW".into(),
                    "SEE".into(),
                    "HEAR".into(),
                    "THINK".into(),
                ],
                true, // agency (decision-making)
                true, // sensory (eyes, ears)
            ),
            // Torso: central body mass
            BodyPart::Torso => (
                vec!["BODY".into(), "BIG".into(), "PART".into(), "INSIDE".into()],
                false,
                false,
            ),
            // Arms: action limbs
            BodyPart::LeftArm => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "ONE".into(),
                    "SIDE".into(),
                    "DO".into(),
                ],
                true,
                false,
            ),
            BodyPart::RightArm => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "OTHER".into(),
                    "SIDE".into(),
                    "DO".into(),
                ],
                true,
                false,
            ),
            // Hands: manipulation and touch
            BodyPart::LeftHand => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "TOUCH".into(),
                    "DO".into(),
                    "HAVE".into(),
                ],
                true,
                true, // tactile sensing
            ),
            BodyPart::RightHand => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "TOUCH".into(),
                    "DO".into(),
                    "HAVE".into(),
                ],
                true,
                true,
            ),
            // Legs: locomotion
            BodyPart::LeftLeg => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "MOVE".into(),
                    "GO".into(),
                    "ONE".into(),
                ],
                true,
                false,
            ),
            BodyPart::RightLeg => (
                vec![
                    "BODY".into(),
                    "PART".into(),
                    "MOVE".into(),
                    "GO".into(),
                    "OTHER".into(),
                ],
                true,
                false,
            ),
            // Feet: ground contact
            BodyPart::LeftFoot => (
                vec!["BODY".into(), "PART".into(), "BELOW".into(), "TOUCH".into()],
                false,
                true, // proprioceptive sensing
            ),
            BodyPart::RightFoot => (
                vec!["BODY".into(), "PART".into(), "BELOW".into(), "TOUCH".into()],
                false,
                true,
            ),
        }
    }
}

/// NSM grounding for movement types - action semantic primitives
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct MovementTypePrimitiveGrounding {
    /// The movement type being grounded
    pub movement_type: MovementType,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Whether this movement requires external object
    pub requires_object: bool,
    /// Whether this movement is self-directed
    pub self_directed: bool,
}

#[allow(dead_code)]
impl MovementTypePrimitiveGrounding {
    pub(crate) fn new(movement_type: MovementType, system: &PrimitiveSystem) -> Self {
        let (primitives, requires_obj, self_dir) = Self::nsm_mapping(movement_type);
        let encoding = encode_primitives(&primitives, system);

        Self {
            movement_type,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            requires_object: requires_obj,
            self_directed: self_dir,
        }
    }

    fn nsm_mapping(movement: MovementType) -> (Vec<String>, bool, bool) {
        match movement {
            // Reach: extending toward something
            MovementType::Reach => (
                vec!["DO".into(), "MOVE".into(), "FAR".into(), "WANT".into()],
                true, // toward an object
                false,
            ),
            // Grasp: taking hold
            MovementType::Grasp => (
                vec!["DO".into(), "HAVE".into(), "TOUCH".into(), "WANT".into()],
                true,
                false,
            ),
            // Release: letting go
            MovementType::Release => (
                vec!["DO".into(), "NOT".into(), "HAVE".into(), "NOW".into()],
                true,
                false,
            ),
            // Push: applying force away
            MovementType::Push => (
                vec!["DO".into(), "MOVE".into(), "SOMETHING".into(), "FAR".into()],
                true,
                false,
            ),
            // Pull: applying force toward
            MovementType::Pull => (
                vec![
                    "DO".into(),
                    "MOVE".into(),
                    "SOMETHING".into(),
                    "NEAR".into(),
                ],
                true,
                false,
            ),
            // Walk: locomotion
            MovementType::Walk => (
                vec!["DO".into(), "MOVE".into(), "GO".into(), "BODY".into()],
                false,
                true,
            ),
            // Turn: rotation
            MovementType::Turn => (
                vec![
                    "DO".into(),
                    "MOVE".into(),
                    "BODY".into(),
                    "OTHER".into(),
                    "SIDE".into(),
                ],
                false,
                true,
            ),
            // Look: visual attention
            MovementType::Look => (
                vec!["DO".into(), "SEE".into(), "WANT".into(), "KNOW".into()],
                true, // looking at something
                false,
            ),
            // Speak: vocalization
            MovementType::Speak => (
                vec!["DO".into(), "SAY".into(), "WORDS".into(), "SOMEONE".into()],
                false,
                true,
            ),
            // Breathe: respiration
            MovementType::Breathe => (
                vec!["DO".into(), "LIVE".into(), "BODY".into(), "INSIDE".into()],
                false,
                true,
            ),
        }
    }
}

/// Unified NSM grounding system for embodied cognition
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct EmbodiedNSMGrounding {
    /// Body part groundings indexed by part
    pub body_parts: HashMap<BodyPart, BodyPartPrimitiveGrounding>,
    /// Movement type groundings indexed by type
    pub movements: HashMap<MovementType, MovementTypePrimitiveGrounding>,
}

#[allow(dead_code)]
impl EmbodiedNSMGrounding {
    /// Create complete grounding system from primitive system
    pub(crate) fn new(system: &PrimitiveSystem) -> Self {
        let mut body_parts = HashMap::new();
        let mut movements = HashMap::new();

        // Ground all body parts
        for part in BodyPart::all() {
            body_parts.insert(*part, BodyPartPrimitiveGrounding::new(*part, system));
        }

        // Ground all movement types
        for movement in &[
            MovementType::Reach,
            MovementType::Grasp,
            MovementType::Release,
            MovementType::Push,
            MovementType::Pull,
            MovementType::Walk,
            MovementType::Turn,
            MovementType::Look,
            MovementType::Speak,
            MovementType::Breathe,
        ] {
            movements.insert(
                *movement,
                MovementTypePrimitiveGrounding::new(*movement, system),
            );
        }

        Self {
            body_parts,
            movements,
        }
    }

    /// Find body parts by semantic similarity to query vector
    pub(crate) fn query_body_parts(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&BodyPart, f32)> {
        let mut results: Vec<_> = self
            .body_parts
            .iter()
            .map(|(part, grounding)| (part, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find movements by semantic similarity to query vector
    pub(crate) fn query_movements(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&MovementType, f32)> {
        let mut results: Vec<_> = self
            .movements
            .iter()
            .map(|(movement, grounding)| (movement, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get agency-related body parts
    pub(crate) fn agency_parts(&self) -> Vec<&BodyPart> {
        self.body_parts
            .iter()
            .filter(|(_, g)| g.agency_locus)
            .map(|(p, _)| p)
            .collect()
    }

    /// Get sensory body parts
    pub(crate) fn sensory_parts(&self) -> Vec<&BodyPart> {
        self.body_parts
            .iter()
            .filter(|(_, g)| g.sensory_organ)
            .map(|(p, _)| p)
            .collect()
    }

    /// Get object-directed movements
    pub(crate) fn object_movements(&self) -> Vec<&MovementType> {
        self.movements
            .iter()
            .filter(|(_, g)| g.requires_object)
            .map(|(m, _)| m)
            .collect()
    }

    /// Get self-directed movements
    pub(crate) fn self_movements(&self) -> Vec<&MovementType> {
        self.movements
            .iter()
            .filter(|(_, g)| g.self_directed)
            .map(|(m, _)| m)
            .collect()
    }
}

/// Encode NSM primitives into HDC vector via sequential binding
#[allow(dead_code)]
fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
    let vectors: Vec<BinaryHV> = primitives
        .iter()
        .map(|name| {
            if let Some(p) = system.get(name) {
                p.encoding
            } else if let Some(p) = system.get(&name.to_lowercase()) {
                p.encoding
            } else {
                // Fallback: deterministic random for unknown primitives
                let seed = name
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                BinaryHV::random(seed)
            }
        })
        .collect();

    if vectors.is_empty() {
        return BinaryHV::random(0);
    }

    // Bind sequentially with position encoding for order preservation
    let mut result = vectors[0];
    for (i, v) in vectors.iter().enumerate().skip(1) {
        let position_hv = BinaryHV::random(i as u64 * 1000);
        let positioned = v.bind(&position_hv);
        result = result.bind(&positioned);
    }
    result
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_body_schema_creation() {
        let body = BodySchema::new();
        assert_eq!(body.parts.len(), 10);
        assert!(body.ownership_confidence > 0.9);
    }

    #[test]
    fn test_body_part_connections() {
        let torso_connections = BodyPart::Torso.connected_to();
        assert!(torso_connections.contains(&BodyPart::Head));
        assert!(torso_connections.contains(&BodyPart::LeftArm));
        assert!(torso_connections.contains(&BodyPart::RightArm));
    }

    #[test]
    fn test_position_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_interoception_to_affect() {
        let mut intero = InteroceptiveState::default();
        intero.hunger = 0.8;
        intero.heart_rate = 0.7;

        let affect = intero.to_core_affect();
        assert!(affect.valence < 0.0); // Hunger is negative
        assert!(affect.arousal > 0.0); // High heart rate
    }

    #[test]
    fn test_homeostatic_deviation() {
        let mut intero = InteroceptiveState::default();
        assert!(intero.homeostatic_deviation() < 0.2);

        intero.hunger = 1.0;
        intero.thirst = 1.0;
        assert!(intero.homeostatic_deviation() > 0.3);
    }

    #[test]
    fn test_affordance_attractiveness() {
        let affordance = Affordance {
            action_type: MovementType::Grasp,
            required_parts: vec![BodyPart::RightHand],
            location: Position3D::origin(),
            saliency: 0.8,
            effort: 0.2,
            risk: 0.1,
            expected_value: 0.9,
            encoding: BinaryHV::random(42),
        };

        assert!(affordance.attractiveness() > 0.0);
    }

    #[test]
    fn test_sensorimotor_learning() {
        let mut engine = SensorimotorEngine::new(100);

        let action = ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Grasp,
            intensity: 0.5,
            duration: 10,
            encoding: BinaryHV::random(1),
        };

        let sensation = SensoryPrediction {
            visual_change: 0.3,
            tactile_expected: 0.7,
            proprioceptive_change: 0.5,
            auditory_expected: 0.1,
            encoding: BinaryHV::random(2),
        };

        // Learn
        engine.learn(action.clone(), sensation, None);

        // Predict
        let prediction = engine.predict(&action);
        assert!(prediction.is_some());
    }

    #[test]
    fn test_affordance_detection() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();

        let mut env = EnvironmentState::default();
        env.objects.push(EnvironmentObject {
            id: 1,
            position: Position3D::new(0.3, 0.3, 1.0), // Within reach
            graspable: true,
            saliency: 0.8,
            value: 0.9,
        });

        detector.detect(&body, &env);
        assert!(!detector.current_affordances.is_empty());
    }

    #[test]
    fn test_embodied_analyzer_creation() {
        let config = EmbodiedConfig::default();
        let analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        assert!(analyzer.agency > 0.9);
        assert!(analyzer.ownership > 0.9);
    }

    #[test]
    fn test_embodied_processing() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        let response = analyzer.process();

        assert!(response.sense_of_agency > 0.0);
        assert!(response.embodiment_integration > 0.0);
        assert!(response.phi_modulation > 0.0);
    }

    #[test]
    fn test_action_performance() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        let action = ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Reach,
            intensity: 0.5,
            duration: 5,
            encoding: BinaryHV::random(100),
        };

        let outcome = analyzer.perform_action(action);

        // First action has high uncertainty
        assert!(outcome.prediction_error > 0.0);
    }

    #[test]
    fn test_interoception_update() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        let mut intero = InteroceptiveState::default();
        intero.hunger = 0.7;
        intero.fatigue = 0.5;

        analyzer.update_interoception(intero);

        let response = analyzer.process();
        assert!(response.homeostatic_deviation > 0.2);
    }

    #[test]
    fn test_embodiment_summary() {
        let config = EmbodiedConfig::default();
        let analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        let summary = analyzer.summary();

        assert!(summary.agency > 0.0);
        assert!(summary.ownership > 0.0);
        assert!(summary.overall_embodiment > 0.0);
    }

    #[test]
    fn test_proprioceptive_summary() {
        let body = BodySchema::new();
        let summary = body.proprioceptive_summary();

        assert!(summary.posture_stability > 0.5); // Default is balanced
    }

    // ================================================================
    // NEW TESTS: Position3D edge cases
    // ================================================================

    #[test]
    fn test_position3d_distance_to_self_is_zero() {
        let p = Position3D::new(3.5, -2.1, 7.0);
        assert!((p.distance(&p) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position3d_distance_symmetry() {
        let a = Position3D::new(1.0, 2.0, 3.0);
        let b = Position3D::new(-4.0, 5.0, -1.0);
        assert!((a.distance(&b) - b.distance(&a)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position3d_origin() {
        let o = Position3D::origin();
        assert!((o.x).abs() < f64::EPSILON);
        assert!((o.y).abs() < f64::EPSILON);
        assert!((o.z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position3d_distance_known_values() {
        // 3-4-5 triangle in 3D: (0,0,0) to (3,4,0) = 5
        let a = Position3D::origin();
        let b = Position3D::new(3.0, 4.0, 0.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);

        // Diagonal in a unit cube: sqrt(3)
        let c = Position3D::new(1.0, 1.0, 1.0);
        assert!((a.distance(&c) - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    // ================================================================
    // NEW TESTS: BodySchema and derived properties
    // ================================================================

    #[test]
    fn test_body_schema_all_parts_have_default_positions() {
        let body = BodySchema::new();
        for part in BodyPart::all() {
            assert!(
                body.parts.contains_key(part),
                "Missing body part: {:?}",
                part
            );
            let state = &body.parts[part];
            assert!(state.position.z.is_finite());
        }
    }

    #[test]
    fn test_body_schema_balance_default_is_centered() {
        let body = BodySchema::new();
        assert!(
            body.balance.abs() < 0.01,
            "Default balance should be near zero, got {}",
            body.balance
        );
    }

    #[test]
    fn test_body_schema_update_part_affects_balance() {
        let mut body = BodySchema::new();
        // Move the torso far to the right to shift center of mass
        let mut torso = BodyPartState::default();
        torso.position = Position3D::new(2.0, 0.0, 1.0);
        body.update_part(BodyPart::Torso, torso);
        // Balance should shift positive (rightward)
        assert!(body.balance > 0.0, "Balance should shift right");
    }

    #[test]
    fn test_body_schema_ownership_and_coherence_defaults() {
        let body = BodySchema::new();
        assert!((body.ownership_confidence - 1.0).abs() < f64::EPSILON);
        assert!((body.schema_image_coherence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_body_part_connections_symmetry() {
        // If A connects to B, B should connect to A
        for part in BodyPart::all() {
            for connected in part.connected_to() {
                assert!(
                    connected.connected_to().contains(part),
                    "{:?} connects to {:?} but {:?} does not connect back",
                    part,
                    connected,
                    connected
                );
            }
        }
    }

    // ================================================================
    // NEW TESTS: InteroceptiveState
    // ================================================================

    #[test]
    fn test_interoceptive_default_homeostatic_deviation_low() {
        let intero = InteroceptiveState::default();
        // All zeros => deviation = (0+0+0+0+0.3)/5 = 0.06 (heart_rate=0, default offset)
        let dev = intero.homeostatic_deviation();
        assert!(
            dev < 0.2,
            "Default homeostatic deviation should be low, got {dev}"
        );
    }

    #[test]
    fn test_interoceptive_allostatic_load_zero_when_resting() {
        let intero = InteroceptiveState::default();
        // visceral_arousal=0 => allostatic_load = 0
        assert!((intero.allostatic_load() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interoceptive_allostatic_load_increases_under_stress() {
        let intero = InteroceptiveState {
            hunger: 0.8,
            thirst: 0.7,
            fatigue: 0.9,
            visceral_arousal: 0.8,
            ..Default::default()
        };
        let load = intero.allostatic_load();
        assert!(
            load > 0.2,
            "Stressed state should have significant allostatic load, got {load}"
        );
    }

    #[test]
    fn test_interoceptive_to_core_affect_bounds() {
        let intero = InteroceptiveState {
            hunger: 1.0,
            thirst: 1.0,
            fatigue: 1.0,
            heart_rate: 1.0,
            breathing_rate: 1.0,
            temperature: 1.0,
            gut_state: -1.0,
            visceral_arousal: 1.0,
        };
        let affect = intero.to_core_affect();
        // Values should be finite
        assert!(affect.valence.is_finite());
        assert!(affect.arousal.is_finite());
    }

    #[test]
    fn test_interoceptive_to_physiological_mapping() {
        let intero = InteroceptiveState {
            heart_rate: 0.7,
            breathing_rate: 0.6,
            fatigue: 0.5,
            visceral_arousal: 0.4,
            gut_state: 0.3,
            ..Default::default()
        };
        let phys = intero.to_physiological();
        assert!((phys.gut_feeling - 0.3).abs() < 0.01);
        assert!(phys.heart_rate_delta.is_finite());
        assert!(phys.skin_conductance.is_finite());
    }

    // ================================================================
    // NEW TESTS: SensorimotorEngine
    // ================================================================

    #[test]
    fn test_sensorimotor_engine_no_prediction_without_learning() {
        let engine = SensorimotorEngine::new(100);
        let action = ActionPattern {
            involved_parts: vec![BodyPart::Head],
            movement_type: MovementType::Look,
            intensity: 0.5,
            duration: 5,
            encoding: BinaryHV::random(42),
        };
        assert!(engine.predict(&action).is_none());
    }

    #[test]
    fn test_sensorimotor_engine_average_prediction_error_neutral_default() {
        let engine = SensorimotorEngine::new(100);
        assert!((engine.average_prediction_error() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sensorimotor_engine_learning_updates_confidence() {
        let mut engine = SensorimotorEngine::new(100);
        let encoding = BinaryHV::random(42);
        let action = ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Grasp,
            intensity: 0.5,
            duration: 10,
            encoding,
        };
        let sensation = SensoryPrediction {
            visual_change: 0.3,
            tactile_expected: 0.7,
            proprioceptive_change: 0.5,
            auditory_expected: 0.1,
            encoding: BinaryHV::random(99),
        };
        // Learn the same contingency twice
        engine.learn(action.clone(), sensation.clone(), None);
        engine.learn(action.clone(), sensation.clone(), None);
        // Confidence should increase
        let contingency = &engine.contingencies[0];
        assert!(
            contingency.confidence > 0.5,
            "Confidence should increase after repeated learning"
        );
        assert_eq!(contingency.experience_count, 2);
    }

    #[test]
    fn test_sensorimotor_engine_eviction_at_capacity() {
        let mut engine = SensorimotorEngine::new(3);
        for i in 0..5 {
            let action = ActionPattern {
                involved_parts: vec![BodyPart::RightHand],
                movement_type: MovementType::Grasp,
                intensity: 0.5,
                duration: 10,
                encoding: BinaryHV::random(i * 1000 + 777), // Different encodings
            };
            let sensation = SensoryPrediction {
                visual_change: 0.1,
                tactile_expected: 0.2,
                proprioceptive_change: 0.3,
                auditory_expected: 0.0,
                encoding: BinaryHV::random(i * 1000 + 888),
            };
            engine.learn(action, sensation, None);
        }
        assert!(
            engine.contingencies.len() <= 3,
            "Should be capped at max_contingencies=3"
        );
    }

    #[test]
    fn test_sensorimotor_engine_prediction_error_tracking() {
        let mut engine = SensorimotorEngine::new(100);
        let encoding = BinaryHV::random(42);
        let action = ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Grasp,
            intensity: 0.5,
            duration: 10,
            encoding,
        };
        let learned_sensation = SensoryPrediction {
            visual_change: 0.3,
            tactile_expected: 0.7,
            proprioceptive_change: 0.5,
            auditory_expected: 0.1,
            encoding: BinaryHV::random(99),
        };
        // Learn then learn again with a prediction
        engine.learn(action.clone(), learned_sensation.clone(), None);
        let pred = engine.predict(&action).unwrap();
        // Learn again, this time with prediction provided
        engine.learn(action, learned_sensation, Some(&pred));
        // Now there should be a prediction error recorded
        assert!(!engine.prediction_errors.is_empty());
        let avg = engine.average_prediction_error();
        assert!(avg.is_finite());
        assert!(avg >= 0.0);
    }

    // ================================================================
    // NEW TESTS: Affordance
    // ================================================================

    #[test]
    fn test_affordance_attractiveness_high_value_low_cost() {
        let aff = Affordance {
            action_type: MovementType::Grasp,
            required_parts: vec![BodyPart::RightHand],
            location: Position3D::origin(),
            saliency: 1.0,
            effort: 0.0,
            risk: 0.0,
            expected_value: 1.0,
            encoding: BinaryHV::random(1),
        };
        assert!((aff.attractiveness() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_affordance_attractiveness_high_risk_reduces_score() {
        let low_risk = Affordance {
            action_type: MovementType::Walk,
            required_parts: vec![],
            location: Position3D::origin(),
            saliency: 0.8,
            effort: 0.2,
            risk: 0.0,
            expected_value: 0.8,
            encoding: BinaryHV::random(1),
        };
        let high_risk = Affordance {
            action_type: MovementType::Walk,
            required_parts: vec![],
            location: Position3D::origin(),
            saliency: 0.8,
            effort: 0.2,
            risk: 1.0,
            expected_value: 0.8,
            encoding: BinaryHV::random(1),
        };
        assert!(low_risk.attractiveness() > high_risk.attractiveness());
    }

    #[test]
    fn test_affordance_attractiveness_clamped() {
        // Extremely negative scenario
        let aff = Affordance {
            action_type: MovementType::Walk,
            required_parts: vec![],
            location: Position3D::origin(),
            saliency: 0.0,
            effort: 1.0,
            risk: 1.0,
            expected_value: 0.0,
            encoding: BinaryHV::random(1),
        };
        assert!(aff.attractiveness() >= -1.0);
        assert!(aff.attractiveness() <= 1.0);
    }

    // ================================================================
    // NEW TESTS: AffordanceDetector
    // ================================================================

    #[test]
    fn test_affordance_detector_empty_environment() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();
        let env = EnvironmentState::default();
        detector.detect(&body, &env);
        assert!(detector.current_affordances.is_empty());
        assert!(detector.most_attractive().is_none());
    }

    #[test]
    fn test_affordance_detector_out_of_reach_object_ignored() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();
        let mut env = EnvironmentState::default();
        env.objects.push(EnvironmentObject {
            id: 1,
            position: Position3D::new(100.0, 100.0, 100.0), // Far away
            graspable: true,
            saliency: 1.0,
            value: 1.0,
        });
        detector.detect(&body, &env);
        assert!(detector.current_affordances.is_empty());
    }

    #[test]
    fn test_affordance_detector_non_graspable_ignored() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();
        let mut env = EnvironmentState::default();
        env.objects.push(EnvironmentObject {
            id: 1,
            position: Position3D::new(0.3, 0.3, 1.0), // Within reach
            graspable: false,                         // Not graspable
            saliency: 1.0,
            value: 1.0,
        });
        detector.detect(&body, &env);
        // Should not detect grasp affordance for non-graspable object
        let grasp_affordances: Vec<_> = detector
            .current_affordances
            .iter()
            .filter(|a| a.action_type == MovementType::Grasp)
            .collect();
        assert!(grasp_affordances.is_empty());
    }

    #[test]
    fn test_affordance_detector_walkable_surface_detected() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();
        let mut env = EnvironmentState::default();
        env.surfaces.push(EnvironmentSurface {
            id: 1,
            center: Position3D::new(0.0, 1.0, 0.0),
            walkable: true,
            slope: 0.0,
        });
        detector.detect(&body, &env);
        let walk_affordances: Vec<_> = detector
            .current_affordances
            .iter()
            .filter(|a| a.action_type == MovementType::Walk)
            .collect();
        assert!(
            !walk_affordances.is_empty(),
            "Should detect walkable surface"
        );
    }

    #[test]
    fn test_affordance_detector_sorted_by_attractiveness() {
        let mut detector = AffordanceDetector::new();
        let body = BodySchema::new();
        let mut env = EnvironmentState::default();
        // Two graspable objects at different values
        env.objects.push(EnvironmentObject {
            id: 1,
            position: Position3D::new(0.2, 0.2, 1.0),
            graspable: true,
            saliency: 0.3,
            value: 0.2,
        });
        env.objects.push(EnvironmentObject {
            id: 2,
            position: Position3D::new(0.3, 0.3, 1.0),
            graspable: true,
            saliency: 0.9,
            value: 0.9,
        });
        detector.detect(&body, &env);
        if detector.current_affordances.len() >= 2 {
            assert!(
                detector.current_affordances[0].attractiveness()
                    >= detector.current_affordances[1].attractiveness(),
                "Affordances should be sorted by attractiveness"
            );
        }
    }

    // ================================================================
    // NEW TESTS: EmbodiedConsciousnessAnalyzer
    // ================================================================

    #[test]
    fn test_analyzer_phi_modulation_finite_and_positive() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);
        let response = analyzer.process();
        assert!(response.phi_modulation.is_finite());
        assert!(
            response.phi_modulation > 0.0,
            "Phi modulation should be positive"
        );
    }

    #[test]
    fn test_analyzer_visceral_arousal_increases_phi_modulation() {
        let config = EmbodiedConfig::default();
        let mut analyzer_calm = EmbodiedConsciousnessAnalyzer::new(config.clone());
        let mut analyzer_aroused = EmbodiedConsciousnessAnalyzer::new(config);

        analyzer_aroused.interoception.visceral_arousal = 0.9;
        let calm_response = analyzer_calm.process();
        let aroused_response = analyzer_aroused.process();

        assert!(
            aroused_response.phi_modulation > calm_response.phi_modulation,
            "Higher visceral arousal should increase phi modulation"
        );
    }

    #[test]
    fn test_analyzer_action_updates_tension() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);
        let initial_tension = analyzer.body.parts[&BodyPart::RightHand].tension;

        let action = ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Grasp,
            intensity: 0.8,
            duration: 5,
            encoding: BinaryHV::random(100),
        };
        analyzer.perform_action(action);

        let final_tension = analyzer.body.parts[&BodyPart::RightHand].tension;
        assert!(
            final_tension > initial_tension,
            "Action should increase tension"
        );
    }

    #[test]
    fn test_analyzer_repeated_actions_improve_agency() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);

        // Perform the same action multiple times to build contingency
        let encoding = BinaryHV::random(42);
        for _ in 0..5 {
            let action = ActionPattern {
                involved_parts: vec![BodyPart::RightHand],
                movement_type: MovementType::Reach,
                intensity: 0.5,
                duration: 5,
                encoding,
            };
            analyzer.perform_action(action);
        }
        // Agency should remain high or improve with consistent predictions
        assert!(
            analyzer.agency > 0.5,
            "Agency should remain reasonable after consistent actions"
        );
    }

    #[test]
    fn test_analyzer_update_environment_triggers_affordance_detection() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);
        let mut env = EnvironmentState::default();
        env.objects.push(EnvironmentObject {
            id: 1,
            position: Position3D::new(0.3, 0.3, 1.0),
            graspable: true,
            saliency: 0.8,
            value: 0.9,
        });
        analyzer.update_environment(env);
        let response = analyzer.process();
        assert!(response.available_affordances > 0);
    }

    #[test]
    fn test_analyzer_summary_all_fields_bounded() {
        let config = EmbodiedConfig::default();
        let analyzer = EmbodiedConsciousnessAnalyzer::new(config);
        let summary = analyzer.summary();

        assert!(summary.body_tension >= 0.0 && summary.body_tension <= 1.0);
        assert!(summary.body_balance >= -1.0 && summary.body_balance <= 1.0);
        assert!(summary.agency >= 0.0 && summary.agency <= 1.0);
        assert!(summary.ownership >= 0.0 && summary.ownership <= 1.0);
        assert!(summary.overall_embodiment >= 0.0);
        assert!(summary.homeostatic_state.is_finite());
        assert!(summary.sensorimotor_competence.is_finite());
        assert!(summary.affordance_richness >= 0.0);
    }

    #[test]
    fn test_analyzer_zero_intensity_action() {
        let config = EmbodiedConfig::default();
        let mut analyzer = EmbodiedConsciousnessAnalyzer::new(config);
        let action = ActionPattern {
            involved_parts: vec![BodyPart::Head],
            movement_type: MovementType::Look,
            intensity: 0.0,
            duration: 1,
            encoding: BinaryHV::random(55),
        };
        let outcome = analyzer.perform_action(action);
        // Zero intensity should produce zero sensation changes
        assert!((outcome.sensation.visual_change - 0.0).abs() < f64::EPSILON);
        assert!((outcome.sensation.proprioceptive_change - 0.0).abs() < f64::EPSILON);
    }

    // ================================================================
    // NEW TESTS: BodyPart coverage
    // ================================================================

    #[test]
    fn test_body_part_all_has_ten_parts() {
        assert_eq!(BodyPart::all().len(), 10);
    }

    #[test]
    fn test_body_parts_symmetric_positions() {
        let body = BodySchema::new();
        // Left and right arms should be symmetric about x=0
        let left_arm = &body.parts[&BodyPart::LeftArm].position;
        let right_arm = &body.parts[&BodyPart::RightArm].position;
        assert!(
            (left_arm.x + right_arm.x).abs() < 0.01,
            "Arms should be symmetric about x=0"
        );
        assert!(
            (left_arm.z - right_arm.z).abs() < 0.01,
            "Arms should be at same height"
        );

        // Left and right legs
        let left_leg = &body.parts[&BodyPart::LeftLeg].position;
        let right_leg = &body.parts[&BodyPart::RightLeg].position;
        assert!(
            (left_leg.x + right_leg.x).abs() < 0.01,
            "Legs should be symmetric about x=0"
        );
    }

    #[test]
    fn test_head_is_highest_body_part() {
        let body = BodySchema::new();
        let head_z = body.parts[&BodyPart::Head].position.z;
        for (part, state) in &body.parts {
            if *part != BodyPart::Head {
                assert!(
                    head_z >= state.position.z,
                    "Head should be highest, but {:?} at z={} vs head z={}",
                    part,
                    state.position.z,
                    head_z
                );
            }
        }
    }
}
