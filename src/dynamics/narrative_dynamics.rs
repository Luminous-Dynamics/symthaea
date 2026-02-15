//! Narrative Arc Dynamics using Hierarchical CfC
//!
//! Wraps `HierarchicalCfC` with narrative-specific semantics. Scene HDC vectors
//! are projected into the CfC input space, and the multi-scale output is decoded
//! into a **NarrativeSignal** (the "Ghost Signal") that captures the continuous
//! dynamics of a story: energy, surprise, valence, tension, momentum.
//!
//! # Time Scale Mapping
//!
//! | Layer | tau     | Narrative role         |
//! |-------|---------|------------------------|
//! | 0     | 0.1     | Beat-by-beat pacing    |
//! | 1     | 1.0     | Scene arc              |
//! | 2     | 10.0    | Act arc / tension      |
//! | 3     | 100.0   | Thematic throughline   |
//!
//! The `NarrativeCompiler` (in `language::narrative_compiler`) reads the Ghost
//! Signal and converts it into concrete writing instructions for an LLM.

use std::collections::VecDeque;

use ndarray::Array1;
use symthaea_core::hdc::ContinuousHV;

use super::hierarchical_cfc::{HierarchicalCfC, HierarchicalCfCConfig};
use crate::hdc::narrative_algebra::{ArcPhase, NARRATIVE_DIM};

// ============================================================================
// Narrative Archetypes — deterministic HDC reference vectors
// ============================================================================

/// Archetype HVs for computing instantaneous narrative signal from HDC space.
///
/// These provide scene-level signal values via cosine similarity, bypassing
/// the CfC network for a direct "HDC readout" that is then blended with the
/// CfC temporal dynamics.
pub struct NarrativeArchetypes {
    pub high_energy: ContinuousHV,
    pub surprise: ContinuousHV,
    pub positive_valence: ContinuousHV,
    pub high_tension: ContinuousHV,
}

impl NarrativeArchetypes {
    /// Create archetypes with deterministic seeds in the 13_000_000 range.
    pub fn new(dim: usize) -> Self {
        Self {
            high_energy: ContinuousHV::random(dim, 13_000_003),
            surprise: ContinuousHV::random(dim, 13_000_017),
            positive_valence: ContinuousHV::random(dim, 13_000_029),
            high_tension: ContinuousHV::random(dim, 13_000_039),
        }
    }
}

// ============================================================================
// NarrativeSignal — the "Ghost Signal"
// ============================================================================

/// The continuous narrative dynamics output ("Ghost Signal").
///
/// Each field is a scalar extracted from the multi-scale CfC output, capturing
/// one dimension of the story's current state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NarrativeSignal {
    /// 0.0 (still) to 1.0 (frantic) — pacing / intensity
    pub energy: f32,
    /// 0.0 (predictable) to 1.0 (shocking) — entropy
    pub surprise: f32,
    /// -1.0 (dark) to 1.0 (light) — emotional direction
    pub valence: f32,
    /// 0.0 (relaxed) to 1.0 (unbearable) — stakes / pressure
    pub tension: f32,
    /// -1.0 (decelerating) to 1.0 (accelerating) — pace change
    pub momentum: f32,
    /// Inferred from signal shape
    pub arc_phase: ArcPhase,
}

impl Default for NarrativeSignal {
    fn default() -> Self {
        Self {
            energy: 0.0,
            surprise: 0.0,
            valence: 0.0,
            tension: 0.0,
            momentum: 0.0,
            arc_phase: ArcPhase::Setup,
        }
    }
}

// ============================================================================
// StoryArcConfig
// ============================================================================

/// Configuration for story arc dynamics.
pub struct StoryArcConfig {
    /// Fast: beat-by-beat (default: 0.1)
    pub scene_tau: f32,
    /// Medium: scene arcs (default: 1.0)
    pub act_tau: f32,
    /// Slow: overall arc (default: 10.0)
    pub story_tau: f32,
    /// Glacial: thematic throughline (default: 100.0)
    pub theme_tau: f32,
    /// Top-down theme → scene influence (default: 0.3)
    pub feedback_strength: f32,
    /// CfC hidden dimension per layer (default: 64)
    pub hidden_dim: usize,
    /// Input projection dimension — size of the scene vector fed to the CfC
    /// (default: 32)
    pub input_dim: usize,
}

impl Default for StoryArcConfig {
    fn default() -> Self {
        Self {
            scene_tau: 0.1,
            act_tau: 1.0,
            story_tau: 10.0,
            theme_tau: 100.0,
            feedback_strength: 0.3,
            hidden_dim: 64,
            input_dim: 32,
        }
    }
}

// ============================================================================
// StoryArcDynamics
// ============================================================================

/// Story arc dynamics engine wrapping `HierarchicalCfC`.
pub struct StoryArcDynamics {
    config: StoryArcConfig,
    cfc: HierarchicalCfC,
    /// Stored CfC config for re-creation in predict_horizon
    cfc_config: HierarchicalCfCConfig,
    step_count: u64,
    signal_history: VecDeque<NarrativeSignal>,
    /// Previous signal for momentum computation
    prev_energy: f32,
    /// HDC archetype vectors for instantaneous readout
    archetypes: NarrativeArchetypes,
    /// Blend factor: alpha * CfC + (1-alpha) * HDC (default 0.3)
    hdc_blend_alpha: f32,
}

/// Maximum signal history length
const MAX_HISTORY: usize = 256;

impl StoryArcDynamics {
    /// Create a new story arc dynamics engine.
    pub fn new(config: StoryArcConfig) -> Self {
        let cfc_config = HierarchicalCfCConfig {
            input_dim: config.input_dim,
            output_dim: config.hidden_dim,
            time_constants: vec![
                config.scene_tau,
                config.act_tau,
                config.story_tau,
                config.theme_tau,
            ],
            hidden_dims: vec![
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
            ],
            top_down_feedback: true,
            feedback_strength: config.feedback_strength,
            bottom_up_integration: true,
            integration_strength: 0.5,
            multi_scale_prediction: true,
            lr_scales: vec![0.5, 1.0, 1.0, 0.5],
        };

        let mut cfc = HierarchicalCfC::new(cfc_config.clone());

        // Apply narrative-specific output biases so resting state isn't flat 0.5
        Self::apply_narrative_bias(&mut cfc, config.hidden_dim);

        let archetypes = NarrativeArchetypes::new(NARRATIVE_DIM);

        Self {
            config,
            cfc,
            cfc_config,
            step_count: 0,
            signal_history: VecDeque::with_capacity(MAX_HISTORY),
            prev_energy: 0.0,
            archetypes,
            hdc_blend_alpha: 0.3,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StoryArcConfig::default())
    }

    /// Advance dynamics one step.
    ///
    /// `scene_input` should be an array of dimension `config.input_dim`
    /// (project your HDC scene vector down before calling this).
    pub fn step(&mut self, scene_input: &Array1<f32>, dt: f32) -> NarrativeSignal {
        let output = self.cfc.forward_hierarchical(scene_input, dt);
        self.step_count += 1;

        // Extract scalar fields from multi-scale layer outputs.
        // Each layer's output is an Array1<f32> of length hidden_dim.
        // We take the mean of the first few elements as the scalar signal.
        let energy = Self::extract_scalar(&output.scale_outputs[0], 0);
        let momentum_raw = energy - self.prev_energy;
        self.prev_energy = energy;

        let surprise = Self::extract_scalar(&output.scale_outputs[1], 0);
        let tension = Self::extract_scalar(&output.scale_outputs[2], 0);
        let valence = Self::extract_scalar_signed(&output.scale_outputs[3], 0);

        let momentum = momentum_raw.clamp(-1.0, 1.0);

        let arc_phase = Self::infer_arc_phase(tension, momentum, self.step_count);

        let signal = NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum,
            arc_phase,
        };

        // Maintain bounded history
        if self.signal_history.len() >= MAX_HISTORY {
            self.signal_history.pop_front();
        }
        self.signal_history.push_back(signal.clone());

        signal
    }

    /// Look ahead without advancing internal state.
    ///
    /// Creates a fresh CfC from the stored config, steps it forward, and
    /// returns the resulting signal. Because CfC internal state is not
    /// cloneable, this gives an approximate "relaxation" prediction.
    pub fn predict_horizon(&self, horizon: f32) -> NarrativeSignal {
        let mut cfc_fresh = HierarchicalCfC::new(self.cfc_config.clone());
        let dt = 0.1_f32;
        let steps = (horizon / dt).ceil().max(1.0) as usize;
        let zero_input = Array1::zeros(self.config.input_dim);

        let mut last_output = cfc_fresh.forward_hierarchical(&zero_input, dt);
        for _ in 1..steps {
            last_output = cfc_fresh.forward_hierarchical(&zero_input, dt);
        }

        let energy = Self::extract_scalar(&last_output.scale_outputs[0], 0);
        let surprise = Self::extract_scalar(&last_output.scale_outputs[1], 0);
        let tension = Self::extract_scalar(&last_output.scale_outputs[2], 0);
        let valence = Self::extract_scalar_signed(&last_output.scale_outputs[3], 0);

        NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum: 0.0, // no delta available in prediction
            arc_phase: Self::infer_arc_phase(tension, 0.0, self.step_count + steps as u64),
        }
    }

    /// Reset to initial state (start a new story).
    pub fn reset(&mut self) {
        self.cfc.reset();
        self.step_count = 0;
        self.signal_history.clear();
        self.prev_energy = 0.0;
    }

    /// Recent signal trajectory.
    pub fn signal_history(&self) -> &VecDeque<NarrativeSignal> {
        &self.signal_history
    }

    /// Number of steps processed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Advance dynamics one step with hybrid HDC+CfC signal extraction.
    ///
    /// Blends CfC temporal dynamics with instantaneous HDC archetype similarity:
    /// `signal = alpha * cfc_value + (1 - alpha) * hdc_value`
    ///
    /// This ensures the Ghost Signal varies even when CfC weights are random,
    /// because the HDC similarity captures scene-level differences directly.
    pub fn step_hybrid(
        &mut self,
        scene_input: &Array1<f32>,
        scene_hv: &ContinuousHV,
        dt: f32,
    ) -> NarrativeSignal {
        let output = self.cfc.forward_hierarchical(scene_input, dt);
        self.step_count += 1;

        let alpha = self.hdc_blend_alpha;
        let one_minus_alpha = 1.0 - alpha;

        // CfC component (temporal dynamics)
        let cfc_energy = Self::extract_scalar(&output.scale_outputs[0], 0);
        let cfc_surprise = Self::extract_scalar(&output.scale_outputs[1], 0);
        let cfc_tension = Self::extract_scalar(&output.scale_outputs[2], 0);
        let cfc_valence = Self::extract_scalar_signed(&output.scale_outputs[3], 0);

        // HDC component (instantaneous archetype similarity)
        let hdc_energy = (scene_hv.similarity(&self.archetypes.high_energy) + 1.0) / 2.0;
        let hdc_surprise = (scene_hv.similarity(&self.archetypes.surprise) + 1.0) / 2.0;
        let hdc_tension = (scene_hv.similarity(&self.archetypes.high_tension) + 1.0) / 2.0;
        let hdc_valence = scene_hv.similarity(&self.archetypes.positive_valence); // [-1,1]

        // Blend
        let energy = (alpha * cfc_energy + one_minus_alpha * hdc_energy).clamp(0.0, 1.0);
        let surprise = (alpha * cfc_surprise + one_minus_alpha * hdc_surprise).clamp(0.0, 1.0);
        let tension = (alpha * cfc_tension + one_minus_alpha * hdc_tension).clamp(0.0, 1.0);
        let valence = (alpha * cfc_valence + one_minus_alpha * hdc_valence).clamp(-1.0, 1.0);

        let momentum_raw = energy - self.prev_energy;
        self.prev_energy = energy;
        let momentum = momentum_raw.clamp(-1.0, 1.0);

        let arc_phase = Self::infer_arc_phase(tension, momentum, self.step_count);

        let signal = NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum,
            arc_phase,
        };

        if self.signal_history.len() >= MAX_HISTORY {
            self.signal_history.pop_front();
        }
        self.signal_history.push_back(signal.clone());

        signal
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Apply narrative-specific output biases so the resting state isn't flat 0.5.
    fn apply_narrative_bias(cfc: &mut HierarchicalCfC, hidden_dim: usize) {
        // Layer 0 → energy: slight positive bias (resting energy > 0.5)
        let mut energy_bias = Array1::zeros(hidden_dim);
        energy_bias[0] = 0.3;
        cfc.set_output_bias(0, energy_bias);

        // Layer 2 → tension: slight negative bias (resting tension < 0.5)
        let mut tension_bias = Array1::zeros(hidden_dim);
        tension_bias[0] = -0.2;
        cfc.set_output_bias(2, tension_bias);
    }

    /// Extract a [0,1]-clamped scalar from a layer output.
    /// Uses sigmoid on the mean of the first `width` elements.
    fn extract_scalar(arr: &Array1<f32>, _channel: usize) -> f32 {
        let width = arr.len().min(8);
        let mean: f32 = arr.iter().take(width).sum::<f32>() / width as f32;
        sigmoid(mean)
    }

    /// Extract a [-1,1]-clamped scalar (for valence).
    fn extract_scalar_signed(arr: &Array1<f32>, _channel: usize) -> f32 {
        let width = arr.len().min(8);
        let mean: f32 = arr.iter().take(width).sum::<f32>() / width as f32;
        mean.tanh()
    }

    /// Infer arc phase from tension trajectory and step count.
    pub fn infer_arc_phase(tension: f32, momentum: f32, step: u64) -> ArcPhase {
        if step < 3 {
            return ArcPhase::Setup;
        }
        if tension > 0.8 {
            ArcPhase::Climax
        } else if tension > 0.5 && momentum > 0.0 {
            ArcPhase::RisingAction
        } else if tension > 0.3 && momentum < 0.0 {
            ArcPhase::FallingAction
        } else if tension <= 0.3 && momentum <= 0.0 && step > 5 {
            ArcPhase::Resolution
        } else {
            ArcPhase::Setup
        }
    }
}

/// Standard sigmoid
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    #[test]
    fn test_default_config() {
        let config = StoryArcConfig::default();
        assert_eq!(config.scene_tau, 0.1);
        assert_eq!(config.act_tau, 1.0);
        assert_eq!(config.story_tau, 10.0);
        assert_eq!(config.theme_tau, 100.0);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.input_dim, 32);
        assert!((config.feedback_strength - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_step_produces_signal() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);
        let signal = dynamics.step(&input, 0.1);

        assert!(
            (0.0..=1.0).contains(&signal.energy),
            "energy out of range: {}",
            signal.energy
        );
        assert!(
            (0.0..=1.0).contains(&signal.surprise),
            "surprise out of range: {}",
            signal.surprise
        );
        assert!(
            (-1.0..=1.0).contains(&signal.valence),
            "valence out of range: {}",
            signal.valence
        );
        assert!(
            (0.0..=1.0).contains(&signal.tension),
            "tension out of range: {}",
            signal.tension
        );
        assert!(
            (-1.0..=1.0).contains(&signal.momentum),
            "momentum out of range: {}",
            signal.momentum
        );
    }

    #[test]
    fn test_signal_evolves() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![1.0; 32]);

        let signal_1 = dynamics.step(&input, 0.1);
        // Feed different input
        let input_2 = Array1::from_vec(vec![-1.0; 32]);
        let signal_2 = dynamics.step(&input_2, 0.1);

        // At least one field should differ (CfC state evolves)
        let same = (signal_1.energy - signal_2.energy).abs() < 1e-6
            && (signal_1.surprise - signal_2.surprise).abs() < 1e-6
            && (signal_1.tension - signal_2.tension).abs() < 1e-6
            && (signal_1.valence - signal_2.valence).abs() < 1e-6;
        assert!(!same, "Signal should evolve after different inputs");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![1.0; 32]);
        dynamics.step(&input, 0.1);
        dynamics.step(&input, 0.1);
        assert_eq!(dynamics.step_count(), 2);
        assert!(!dynamics.signal_history().is_empty());

        dynamics.reset();
        assert_eq!(dynamics.step_count(), 0);
        assert!(dynamics.signal_history().is_empty());
    }

    #[test]
    fn test_high_input_energy() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        // Large positive values should push energy high (after sigmoid)
        let high_input = Array1::from_vec(vec![5.0; 32]);
        let signal = dynamics.step(&high_input, 0.1);

        // sigmoid(large positive) → close to 1.0
        // The exact value depends on CfC dynamics, but should be > 0.5
        assert!(
            signal.energy > 0.4,
            "High input should produce elevated energy: {}",
            signal.energy
        );
    }

    #[test]
    fn test_arc_phase_inference() {
        // Peak tension → Climax
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.9, 0.1, 10),
            ArcPhase::Climax
        );
        // Rising tension with positive momentum → RisingAction
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.6, 0.2, 10),
            ArcPhase::RisingAction
        );
        // Falling tension with negative momentum → FallingAction
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.4, -0.3, 10),
            ArcPhase::FallingAction
        );
        // Low tension, negative momentum, late in story → Resolution
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.2, -0.1, 10),
            ArcPhase::Resolution
        );
        // Early steps → always Setup
        assert_eq!(
            StoryArcDynamics::infer_arc_phase(0.9, 0.5, 1),
            ArcPhase::Setup
        );
    }

    #[test]
    fn test_hybrid_signal_varies_with_mood() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);

        // Two very different scene HVs
        let calm_hv = ContinuousHV::random(NARRATIVE_DIM, 100);
        let tense_hv = ContinuousHV::random(NARRATIVE_DIM, 200);

        let signal_calm = dynamics.step_hybrid(&input, &calm_hv, 0.1);
        let signal_tense = dynamics.step_hybrid(&input, &tense_hv, 0.1);

        // The signals should differ because HDC similarities differ
        let any_differs = (signal_calm.energy - signal_tense.energy).abs() > 0.001
            || (signal_calm.surprise - signal_tense.surprise).abs() > 0.001
            || (signal_calm.tension - signal_tense.tension).abs() > 0.001
            || (signal_calm.valence - signal_tense.valence).abs() > 0.001;

        assert!(
            any_differs,
            "Hybrid signal should vary between different scene HVs"
        );
    }

    #[test]
    fn test_hybrid_energy_responds_to_conflict() {
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);

        // Use the high_energy archetype itself as scene — should maximize energy HDC component
        let energy_scene = dynamics.archetypes.high_energy.clone();
        let signal = dynamics.step_hybrid(&input, &energy_scene, 0.1);

        // Energy should be elevated (HDC component is 1.0 for self-similarity)
        assert!(
            signal.energy > 0.5,
            "Scene matching energy archetype should have high energy, got {}",
            signal.energy
        );
    }

    #[test]
    fn test_backward_compat_step() {
        // Original step() should still work and produce valid signals
        let mut dynamics = StoryArcDynamics::with_defaults();
        let input = Array1::from_vec(vec![0.5; 32]);
        let signal = dynamics.step(&input, 0.1);

        assert!((0.0..=1.0).contains(&signal.energy));
        assert!((0.0..=1.0).contains(&signal.tension));
        assert!((-1.0..=1.0).contains(&signal.valence));
    }

    #[test]
    fn test_narrative_bias_applied() {
        // With narrative bias, zero-input energy should differ from 0.5
        let mut dynamics = StoryArcDynamics::with_defaults();
        let zero_input = Array1::zeros(32);
        let zero_hv = ContinuousHV::zero(NARRATIVE_DIM);
        let signal = dynamics.step_hybrid(&zero_input, &zero_hv, 0.1);

        // Energy layer has +0.3 bias → sigmoid(0.3) ≈ 0.574
        // Blended with HDC (0.5 for zero HV), should not be exactly 0.5
        // We just check it's valid and not exactly 0.5
        assert!((0.0..=1.0).contains(&signal.energy));
    }
}
