// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CognitiveSnapshot → AestheticState: maps cognitive telemetry to visual parameters.

use crate::CognitiveSnapshot;
use crate::animation::breathing_phase;
use crate::color::Palette;
use serde::{Deserialize, Serialize};

/// All visual parameters, normalized and ready for scene generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticState {
    /// Overall brightness from consciousness level Ψ.
    pub luminosity: f32,
    /// Visual complexity from Cantor depth + Betti numbers.
    pub complexity: f32,
    /// Turbulence/distortion from prediction error.
    pub turbulence: f32,
    /// Kinetic energy from arousal.
    pub energy: f32,
    /// Color warmth from valence.
    pub warmth: f32,
    /// Fractal recursion depth (0-5).
    pub fractal_depth: u8,
    /// Color palette from neuromodulators.
    pub palette: Palette,
    /// β₀ — connected components.
    pub component_count: usize,
    /// β₁ — loops/rings.
    pub ring_count: usize,
    /// β₂ — voids.
    pub void_count: usize,
    /// Octagonal radii from 8 harmony activations.
    pub harmony_radii: [f32; 8],
    /// Layout center from thought vector PCA.
    pub layout_center: (f32, f32),
    /// Persistence diagram: birth-death pairs for components.
    pub persistence_components: Vec<[f64; 2]>,
    /// Persistence diagram: birth-death pairs for cycles.
    pub persistence_cycles: Vec<[f64; 2]>,
    /// Breathing oscillation phase [0, 1].
    pub cycle_phase: f64,
    /// Birkhoff beauty metric (filled after scene generation).
    pub aesthetic_score: f32,
}

/// Stateful engine that converts cognitive snapshots to aesthetic states.
/// Applies EMA smoothing for visual stability.
pub struct AestheticEngine {
    ema_alpha: f32,
    prev: Option<AestheticState>,
}

impl AestheticEngine {
    pub fn new() -> Self {
        Self {
            ema_alpha: 0.3, // 30% new, 70% old — smooth transitions
            prev: None,
        }
    }

    pub fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Map a cognitive snapshot to an aesthetic state.
    pub fn process(&mut self, snap: &CognitiveSnapshot) -> AestheticState {
        let raw = self.map_raw(snap);
        let smoothed = match &self.prev {
            Some(prev) => self.smooth(prev, &raw),
            None => raw,
        };
        self.prev = Some(smoothed.clone());
        smoothed
    }

    /// Direct mapping without smoothing (for testing).
    pub fn map_raw(&self, snap: &CognitiveSnapshot) -> AestheticState {
        let psi = snap.consciousness_level;

        // Luminosity: Ψ^0.7 (gamma curve — never fully dark)
        let luminosity = (psi as f32).powf(0.7);

        // Complexity: Cantor depth (0-5) + Betti sum, scaled
        let betti_sum = (snap.betti_0 + snap.betti_1 + snap.betti_2) as f32;
        let complexity = (snap.cantor_metacognitive_depth * 0.3 + betti_sum * 0.1).clamp(0.0, 1.0);

        // Turbulence: PE^0.5 (emphasize small errors)
        let turbulence = snap.prediction_error.abs().sqrt().min(1.0);

        // Energy: direct from arousal
        let energy = snap.arousal.clamp(0.0, 1.0);

        // Warmth: valence [-1,1] → [0,1]
        let warmth = (snap.valence + 1.0) / 2.0;

        // Fractal depth: clamped to 0-5
        let fractal_depth = snap.cantor_last_depth.min(5);

        // Palette from neuromodulators
        let palette = Palette::from_neuromodulators(
            snap.dopamine,
            snap.serotonin,
            snap.noradrenaline,
            snap.oxytocin,
            snap.gaba,
        );

        // Harmony radii: 0.3 + activation * 0.7
        let mut harmony_radii = [0.0f32; 8];
        for (i, r) in harmony_radii.iter_mut().enumerate() {
            let activation = snap.harmony_activations[i].clamp(0.0, 1.0);
            *r = 0.3 + activation * 0.7;
        }

        // Layout center from thought vector dims 0,1
        let cx = if !snap.thought_vector.is_empty() {
            256.0 + snap.thought_vector[0] * 64.0
        } else {
            256.0
        };
        let cy = if snap.thought_vector.len() > 1 {
            256.0 + snap.thought_vector[1] * 64.0
        } else {
            256.0
        };

        let cycle_phase = breathing_phase(snap.cycle_count);

        AestheticState {
            luminosity,
            complexity,
            turbulence,
            energy,
            warmth,
            fractal_depth,
            palette,
            component_count: snap.betti_0,
            ring_count: snap.betti_1,
            void_count: snap.betti_2,
            harmony_radii,
            layout_center: (cx.clamp(128.0, 384.0), cy.clamp(128.0, 384.0)),
            persistence_components: snap.persistence_components.clone(),
            persistence_cycles: snap.persistence_cycles.clone(),
            cycle_phase,
            aesthetic_score: 0.0, // filled after scene generation
        }
    }

    /// EMA smooth between previous and new state.
    fn smooth(&self, prev: &AestheticState, new: &AestheticState) -> AestheticState {
        let a = self.ema_alpha;
        let b = 1.0 - a;

        let harmony_radii: [f32; 8] =
            std::array::from_fn(|i| prev.harmony_radii[i] * b + new.harmony_radii[i] * a);

        AestheticState {
            luminosity: prev.luminosity * b + new.luminosity * a,
            complexity: prev.complexity * b + new.complexity * a,
            turbulence: prev.turbulence * b + new.turbulence * a,
            energy: prev.energy * b + new.energy * a,
            warmth: prev.warmth * b + new.warmth * a,
            fractal_depth: new.fractal_depth, // discrete, no smoothing
            palette: new.palette.clone(),     // palette changes are intentional
            component_count: new.component_count,
            ring_count: new.ring_count,
            void_count: new.void_count,
            harmony_radii,
            layout_center: (
                prev.layout_center.0 * b + new.layout_center.0 * a,
                prev.layout_center.1 * b + new.layout_center.1 * a,
            ),
            persistence_components: new.persistence_components.clone(),
            persistence_cycles: new.persistence_cycles.clone(),
            cycle_phase: new.cycle_phase, // always current
            aesthetic_score: 0.0,
        }
    }
}

impl Default for AestheticEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CognitiveSnapshot;

    fn make_snapshot(consciousness: f64) -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: consciousness,
            prediction_error: 0.1,
            living_mind_vitality: 0.5,
            living_mind_coherence: 0.5,
            dopamine: 0.5,
            noradrenaline: 0.3,
            serotonin: 0.5,
            acetylcholine: 0.4,
            oxytocin: 0.3,
            gaba: 0.5,
            allostatic_load: 0.2,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            cantor_metacognitive_depth: 0.6,
            cantor_last_depth: 3,
            valence: 0.2,
            arousal: 0.4,
            harmony_activations: [0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1],
            thought_vector: vec![0.5, -0.3],
            cycle_count: 100,
        }
    }

    #[test]
    fn high_psi_high_luminosity() {
        let engine = AestheticEngine::new();
        let high_raw = engine.map_raw(&make_snapshot(0.9));
        let low_raw = engine.map_raw(&make_snapshot(0.1));
        assert!(high_raw.luminosity > low_raw.luminosity);
    }

    #[test]
    fn deterministic_output() {
        let snap = make_snapshot(0.5);
        let mut e1 = AestheticEngine::new();
        let mut e2 = AestheticEngine::new();
        let s1 = e1.process(&snap);
        let s2 = e2.process(&snap);
        assert_eq!(s1.luminosity, s2.luminosity);
        assert_eq!(s1.complexity, s2.complexity);
        assert_eq!(s1.turbulence, s2.turbulence);
    }

    #[test]
    fn ema_smoothing_works() {
        let mut engine = AestheticEngine::new();
        let _ = engine.process(&make_snapshot(0.9));
        // Sudden drop
        let smoothed = engine.process(&make_snapshot(0.1));
        let raw = engine.map_raw(&make_snapshot(0.1));
        // Smoothed should be between raw(0.1) and previous(0.9)
        assert!(smoothed.luminosity > raw.luminosity);
    }

    #[test]
    fn harmony_radii_mapped() {
        let engine = AestheticEngine::new();
        let snap = make_snapshot(0.5);
        let state = engine.map_raw(&snap);
        // harmony_activations[2] = 0.7 → radius = 0.3 + 0.7*0.7 = 0.79
        assert!((state.harmony_radii[2] - 0.79).abs() < 0.01);
        // harmony_activations[7] = 0.1 → radius = 0.3 + 0.1*0.7 = 0.37
        assert!((state.harmony_radii[7] - 0.37).abs() < 0.01);
    }

    #[test]
    fn layout_center_from_thought_vector() {
        let engine = AestheticEngine::new();
        let snap = make_snapshot(0.5);
        let state = engine.map_raw(&snap);
        // thought_vector = [0.5, -0.3]
        // cx = 256 + 0.5*64 = 288, cy = 256 + (-0.3)*64 = 236.8
        assert!((state.layout_center.0 - 288.0).abs() < 1.0);
        assert!((state.layout_center.1 - 236.8).abs() < 1.0);
    }

    #[test]
    fn prediction_error_turbulence() {
        let engine = AestheticEngine::new();
        let mut snap = make_snapshot(0.5);
        snap.prediction_error = 0.0;
        let calm = engine.map_raw(&snap);
        snap.prediction_error = 0.8;
        let turbulent = engine.map_raw(&snap);
        assert!(turbulent.turbulence > calm.turbulence);
    }
}
