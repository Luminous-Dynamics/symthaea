// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canvas bridge: CognitiveLoop → AestheticEngine → SVG output.
//!
//! Parallel to broca_bridge.rs — stateful manager that converts cognitive telemetry
//! into living topology SVG on a configurable generation interval.

#[cfg(feature = "canvas")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "canvas")]
use symthaea_canvas::{
    AestheticEngine, AestheticState, CognitiveSnapshot, aesthetic_score, build_scene, render_svg,
};

/// Telemetry from the canvas pipeline, stored in CycleMetadata.
#[cfg(feature = "canvas")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CanvasTelemetry {
    /// Whether SVG was generated this cycle (respects generation_interval).
    pub generated: bool,
    /// Whether generation was gated by low consciousness.
    pub consciousness_gated: bool,
    /// Birkhoff aesthetic score of the last generated scene (0.0–1.0).
    pub aesthetic_score: f32,
    /// SVG generation time in microseconds.
    pub generation_time_us: u64,
    /// Scene node count in the last generated frame.
    pub node_count: usize,
    /// SVG byte length.
    pub svg_bytes: usize,
}

/// Stateful canvas manager — holds the AestheticEngine (EMA state) and last output.
#[cfg(feature = "canvas")]
pub(crate) struct CanvasManager {
    engine: AestheticEngine,
    /// Consciousness threshold below which we skip full scene generation.
    consciousness_threshold: f32,
    /// Generate SVG every N cycles (default: 5 → ~47Hz at 234Hz cycle rate).
    generation_interval: u32,
    /// Cycle counter for interval gating.
    cycles_since_generation: u32,
    /// Last generated SVG string.
    last_svg: Option<String>,
    /// Last aesthetic state (for telemetry extraction).
    last_state: Option<AestheticState>,
    /// Telemetry from the most recent tick.
    last_telemetry: CanvasTelemetry,
}

#[cfg(feature = "canvas")]
impl CanvasManager {
    pub fn new() -> Self {
        Self {
            engine: AestheticEngine::new(),
            consciousness_threshold: 0.0, // always generate (even dormant ember)
            generation_interval: 5,
            cycles_since_generation: 0,
            last_svg: None,
            last_state: None,
            last_telemetry: CanvasTelemetry::default(),
        }
    }

    pub fn with_generation_interval(mut self, interval: u32) -> Self {
        self.generation_interval = interval.max(1);
        self
    }

    /// Set the generation interval on an existing manager.
    pub fn set_generation_interval(&mut self, interval: u32) {
        self.generation_interval = interval.max(1);
    }

    /// Tick the canvas pipeline. Returns Some(svg) when a new frame is generated.
    pub fn tick(&mut self, snap: &CognitiveSnapshot) -> Option<&str> {
        self.cycles_since_generation += 1;

        // Interval gating
        if self.cycles_since_generation < self.generation_interval {
            self.last_telemetry = CanvasTelemetry {
                generated: false,
                consciousness_gated: false,
                ..CanvasTelemetry::default()
            };
            return None;
        }
        self.cycles_since_generation = 0;

        // Consciousness gating (if threshold set)
        if (snap.consciousness_level as f32) < self.consciousness_threshold {
            self.last_telemetry = CanvasTelemetry {
                generated: false,
                consciousness_gated: true,
                ..CanvasTelemetry::default()
            };
            return None;
        }

        let start = std::time::Instant::now();

        // Pipeline: snapshot → aesthetic state → scene → SVG
        let mut state = self.engine.process(snap);
        let scene = build_scene(&state);
        state.aesthetic_score = aesthetic_score(&state, &scene);
        let svg = render_svg(&scene, snap.consciousness_level);

        let elapsed = start.elapsed();

        self.last_telemetry = CanvasTelemetry {
            generated: true,
            consciousness_gated: false,
            aesthetic_score: state.aesthetic_score,
            generation_time_us: elapsed.as_micros() as u64,
            node_count: scene.node_count(),
            svg_bytes: svg.len(),
        };

        self.last_svg = Some(svg);
        self.last_state = Some(state);

        self.last_svg.as_deref()
    }

    /// Most recent telemetry (always valid, even when no SVG was generated).
    pub fn last_telemetry(&self) -> &CanvasTelemetry {
        &self.last_telemetry
    }

    /// Take the last generated SVG (drains it).
    pub fn take_svg(&mut self) -> Option<String> {
        self.last_svg.take()
    }

    /// Reference to last aesthetic state (for external inspection).
    pub fn last_aesthetic_state(&self) -> Option<&AestheticState> {
        self.last_state.as_ref()
    }
}

#[cfg(feature = "canvas")]
impl Default for CanvasManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a CognitiveSnapshot from CycleMetadata + neuromod bath.
///
/// Called in cycle_phase_dynamics when canvas feature is enabled.
#[cfg(feature = "canvas")]
pub fn extract_snapshot(
    metadata: &super::CycleMetadata,
    neuromod: &super::NeuromodTelemetry,
    harmony_coords: &[f32; 8],
    thought_vector: &[f32],
    betti: (usize, usize, usize),
    persistence_components: &[[f64; 2]],
    persistence_cycles: &[[f64; 2]],
    cantor_depth: f32,
    cantor_last: u8,
    cycle_count: u64,
) -> CognitiveSnapshot {
    CognitiveSnapshot {
        consciousness_level: metadata.consciousness.consciousness_level,
        prediction_error: metadata.fep.fep_surprise as f32,
        living_mind_vitality: metadata.living_mind_vitality,
        living_mind_coherence: metadata.living_mind_coherence,
        dopamine: neuromod.dopamine_effective,
        noradrenaline: neuromod.noradrenaline_effective,
        serotonin: neuromod.serotonin_effective,
        acetylcholine: neuromod.acetylcholine_effective,
        oxytocin: neuromod.neuromod_oxytocin_effective,
        gaba: neuromod.neuromod_gaba_effective,
        allostatic_load: neuromod.neuromod_allostatic_load,
        betti_0: betti.0,
        betti_1: betti.1,
        betti_2: betti.2,
        persistence_components: persistence_components.to_vec(),
        persistence_cycles: persistence_cycles.to_vec(),
        cantor_metacognitive_depth: cantor_depth,
        cantor_last_depth: cantor_last,
        valence: metadata.embodied.affective_valence,
        arousal: metadata.embodied.affective_arousal,
        harmony_activations: *harmony_coords,
        thought_vector: thought_vector.to_vec(),
        cycle_count,
    }
}
