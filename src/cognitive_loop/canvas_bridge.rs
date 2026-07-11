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
use symthaea_aesthetic::{AestheticConfig, AestheticFeedback, AestheticScore};
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
    /// EMA of recent aesthetic scores (the canvas's aesthetic expectation).
    /// Persists across non-generation cycles, unlike `aesthetic_score`.
    pub aesthetic_ema: f32,
    /// Dopamine delta applied to the bath this cycle (already scaled by
    /// `CANVAS_AESTHETIC_FEEDBACK_WEIGHT`). Zero on non-generation cycles.
    pub dopamine_delta: f32,
    /// Serotonin delta applied to the bath this cycle (already scaled).
    pub serotonin_delta: f32,
    /// Surprise signal from aesthetic feedback (already scaled; telemetry only).
    pub surprise_signal: f32,
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
    /// EMA of composite aesthetic scores — the reward-prediction baseline
    /// for `AestheticFeedback` (mirrors `AestheticTracker` in the creative path).
    score_ema: f32,
    /// Feedback scaling parameters (shared defaults with the creative path).
    feedback_config: AestheticConfig,
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
            // Neutral prior: mid-scale expectation so the first frames neither
            // spike nor crater dopamine before the EMA has data.
            score_ema: 0.5,
            feedback_config: AestheticConfig::default(),
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
                aesthetic_ema: self.score_ema,
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
                aesthetic_ema: self.score_ema,
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

        // ── Aesthetic feedback (closes the canvas loop) ────────────────
        // Same machinery as the creative path (symthaea_aesthetic::compute_feedback),
        // but scaled down by CANVAS_AESTHETIC_FEEDBACK_WEIGHT: the canvas is a
        // diagnostic view of cognitive state, not deliberate art.
        let feedback = self.compute_scaled_feedback(&state, snap);
        // Update the EMA *after* feedback so dopamine measures reward
        // prediction error against the prior expectation.
        self.score_ema = self.score_ema * (1.0 - self.feedback_config.ema_alpha)
            + state.aesthetic_score * self.feedback_config.ema_alpha;

        self.last_telemetry = CanvasTelemetry {
            generated: true,
            consciousness_gated: false,
            aesthetic_score: state.aesthetic_score,
            aesthetic_ema: self.score_ema,
            dopamine_delta: feedback.dopamine_delta,
            serotonin_delta: feedback.serotonin_delta,
            surprise_signal: feedback.surprise_signal,
            generation_time_us: elapsed.as_micros() as u64,
            node_count: scene.node_count(),
            svg_bytes: svg.len(),
        };

        self.last_svg = Some(svg);
        self.last_state = Some(state);

        self.last_svg.as_deref()
    }

    /// Map the canvas AestheticState onto a modality-level `AestheticScore`
    /// and derive bath-ready deltas, scaled by `CANVAS_AESTHETIC_FEEDBACK_WEIGHT`.
    fn compute_scaled_feedback(
        &self,
        state: &AestheticState,
        snap: &CognitiveSnapshot,
    ) -> AestheticFeedback {
        use super::thresholds::CANVAS_AESTHETIC_FEEDBACK_WEIGHT;

        let harmony_mean = snap.harmony_activations.iter().sum::<f32>() / 8.0;
        let score = AestheticScore {
            // The canvas score is a Birkhoff composite (order/complexity),
            // so it stands in for both the order and birkhoff dimensions.
            order: state.aesthetic_score.clamp(0.0, 1.0),
            complexity: state.complexity.clamp(0.0, 1.0),
            surprise: (state.aesthetic_score - self.score_ema)
                .abs()
                .clamp(0.0, 1.0),
            harmony: harmony_mean.clamp(0.0, 1.0),
            birkhoff: state.aesthetic_score.clamp(0.0, 1.0),
            composite: state.aesthetic_score.clamp(0.0, 1.0),
        };

        let mut feedback = symthaea_aesthetic::feedback::compute_feedback(
            &score,
            self.score_ema,
            &self.feedback_config,
            &snap.harmony_activations,
        );
        feedback.dopamine_delta *= CANVAS_AESTHETIC_FEEDBACK_WEIGHT;
        feedback.serotonin_delta *= CANVAS_AESTHETIC_FEEDBACK_WEIGHT;
        feedback.surprise_signal *= CANVAS_AESTHETIC_FEEDBACK_WEIGHT;
        feedback
    }

    /// Running EMA of aesthetic scores (persists across non-generation cycles).
    pub fn aesthetic_ema(&self) -> f32 {
        self.score_ema
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
/// **Status (2026-07-10): zero callers.** The live path uses
/// [`snapshot_from_cycle`] + call-site topology enrichment instead (this fn
/// predates that; CycleMetadata isn't assembled yet at the point in
/// phase_dynamics where the canvas ticks). Kept as the reference for a
/// full-metadata extraction should the tick ever move after metadata
/// assembly.
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

/// Build a `CognitiveSnapshot` from the per-cycle `CycleSnapshot` + neuromod bath.
///
/// Lighter-weight than [`extract_snapshot`]: topology fields start at dormant
/// defaults here because `CycleSnapshot` doesn't carry them — the live call
/// site in `cycle_phase_dynamics` enriches the returned snapshot with real
/// Betti numbers (cached `topological_measure` Hodge pipeline) and Cantor
/// depth (live `cantor_dream` sources) before ticking the manager
/// (2026-07-10; they were left dormant before that, hollowing out the
/// topology-driven styles). Persistence diagrams remain empty pending real
/// persistent homology. Harmony decomposition mirrors
/// `MuseManager::map_snapshot_to_musical_state`.
#[cfg(feature = "canvas")]
pub(crate) fn snapshot_from_cycle(
    cs: &super::subsystem_trait::CycleSnapshot,
    bath: &super::neuromodulators::NeuromodulatorBath,
) -> CognitiveSnapshot {
    // Decompose harmonic_coherence into 8 activations using compressed_state
    // (sigmoid preserves polarity; steepness 3 maps ±1 → ~0.05–0.95).
    let hc = (cs.harmonic_coherence as f32).clamp(0.05, 1.0);
    let mut harmony_activations = [0.0f32; 8];
    for (i, activation) in harmony_activations.iter_mut().enumerate() {
        let sigmoid = 1.0 / (1.0 + (-cs.compressed_state[i] * 3.0).exp());
        *activation = sigmoid * hc;
    }

    CognitiveSnapshot {
        consciousness_level: cs.unified_psi,
        prediction_error: cs.prediction_error,
        living_mind_vitality: cs.dissipative_health,
        living_mind_coherence: cs.coherence as f64,
        dopamine: bath.dopamine.effective(),
        noradrenaline: bath.noradrenaline.effective(),
        serotonin: bath.serotonin.effective(),
        acetylcholine: bath.acetylcholine.effective(),
        oxytocin: bath.oxytocin.effective(),
        gaba: bath.gaba.effective(),
        allostatic_load: bath.allostatic_load,
        valence: cs.valence,
        arousal: cs.arousal,
        harmony_activations,
        thought_vector: cs.compressed_state[..32].to_vec(),
        cycle_count: cs.cycle_number,
        // Topology (Betti/persistence/Cantor) uses dormant defaults here.
        ..CognitiveSnapshot::dormant()
    }
}

#[cfg(all(test, feature = "canvas"))]
mod tests {
    use super::*;

    /// A snapshot conscious enough to pass any gating, with active harmonies.
    fn awake_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.8,
            harmony_activations: [0.5; 8],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn feedback_is_scaled_below_creative_caps() {
        let mut mgr = CanvasManager::new().with_generation_interval(1);
        let snap = awake_snapshot();
        // Tick until a frame is generated (interval 1 → first tick).
        assert!(mgr.tick(&snap).is_some());
        let t = mgr.last_telemetry();
        // compute_feedback caps dopamine at +0.15 / -0.05; canvas halves it.
        assert!(
            t.dopamine_delta.abs() <= 0.15 * 0.5 + f32::EPSILON,
            "dopamine delta {} exceeds canvas cap",
            t.dopamine_delta
        );
        // Serotonin: harmony (≤1) × serotonin_scale (0.05) × ½ weight.
        assert!(
            t.serotonin_delta.abs() <= 0.05 * 0.5 + f32::EPSILON,
            "serotonin delta {} exceeds canvas cap",
            t.serotonin_delta
        );
    }

    #[test]
    fn aesthetic_ema_persists_on_gated_cycles() {
        let mut mgr = CanvasManager::new().with_generation_interval(3);
        let snap = awake_snapshot();
        // First two ticks are interval-gated: no frame, but EMA telemetry
        // must still carry the running expectation (initial prior 0.5).
        assert!(mgr.tick(&snap).is_none());
        assert!(mgr.tick(&snap).is_none());
        assert!(mgr.last_telemetry().aesthetic_ema > 0.0);
        assert_eq!(mgr.last_telemetry().dopamine_delta, 0.0);
        // Third tick generates and moves the EMA toward the actual score.
        assert!(mgr.tick(&snap).is_some());
        let t = mgr.last_telemetry();
        assert!(t.generated);
        assert!((0.0..=1.0).contains(&t.aesthetic_ema));
        assert_eq!(t.aesthetic_ema, mgr.aesthetic_ema());
    }

    #[test]
    fn snapshot_from_cycle_maps_bath_and_state() {
        let cs = crate::cognitive_loop::subsystem_trait::CycleSnapshot::default();
        let bath = crate::cognitive_loop::neuromodulators::NeuromodulatorBath::default();
        let snap = snapshot_from_cycle(&cs, &bath);
        assert_eq!(snap.dopamine, bath.dopamine.effective());
        assert_eq!(snap.serotonin, bath.serotonin.effective());
        assert_eq!(snap.thought_vector.len(), 32);
        for h in snap.harmony_activations {
            assert!((0.0..=1.0).contains(&h), "harmony activation out of range");
        }
    }
}
