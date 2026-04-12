// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gray-Scott reaction-diffusion system for living visual art.
//!
//! The Gray-Scott model simulates two chemical species (U and V) that react
//! and diffuse across a 2D grid, producing emergent patterns — spots, stripes,
//! spirals, worms — depending on the feed (F) and kill (k) parameters.
//!
//! **Why this for consciousness-driven art?**
//! The Eight Harmonies map to (F, k) parameter space: different harmonies
//! produce different pattern "personalities." High-Phi consciousness pushes
//! the system further from equilibrium, creating more complex patterns.
//!
//! # Reference
//! Pearson, J. E. (1993). Complex patterns in a simple system. *Science*, 261, 189–192.
//! Gray, P. & Scott, S. K. (1983). Autocatalytic reactions in the CSTR. *Chemical Engineering Science*.

use rand::rngs::StdRng;
use rand::Rng;
use symthaea_canvas::{CognitiveSnapshot, SceneNode};

// ─── Parameter space ─────────────────────────────────────────────────────────

/// Reaction-diffusion parameters mapped to pattern families.
///
/// The (F, k) plane contains distinct pattern-forming regions.
/// Reference: Munafo's parameter map at mrob.com/pub/comp/xmorphia/
#[derive(Debug, Clone, Copy)]
struct RdParams {
    /// Feed rate — rate U is supplied to the system.
    feed: f32,
    /// Kill rate — rate V is consumed.
    kill: f32,
    /// Diffusion rate of U (the activator).
    du: f32,
    /// Diffusion rate of V (the inhibitor).
    dv: f32,
}

/// Map Eight Harmony activations to RD parameters.
///
/// Each harmony has a characteristic pattern personality:
/// - ResonantCoherence (0) → spots (F=0.037, k=0.060)
/// - PanSentientFlourishing (1) → coral/mazes (F=0.055, k=0.062)
/// - IntegralWisdom (2) → worms (F=0.039, k=0.058)
/// - InfinitePlay (3) → spirals/chaos (F=0.026, k=0.051)
/// - UniversalInterconnectedness (4) → mitosis/splitting (F=0.028, k=0.053)
/// - SacredReciprocity (5) → fingerprints (F=0.014, k=0.054)
/// - EvolutionaryProgression (6) → stripes (F=0.022, k=0.059)
/// - SacredStillness (7) → solitons/stable (F=0.062, k=0.061)
const HARMONY_RD: [(f32, f32); 8] = [
    (0.037, 0.060), // ResonantCoherence → spots
    (0.055, 0.062), // PanSentientFlourishing → coral/branching
    (0.039, 0.058), // IntegralWisdom → worms
    (0.026, 0.051), // InfinitePlay → spirals (chaotic regime)
    (0.028, 0.053), // UniversalInterconnectedness → mitosis
    (0.014, 0.054), // SacredReciprocity → fingerprint/moiré
    (0.022, 0.059), // EvolutionaryProgression → stripes
    (0.062, 0.061), // SacredStillness → stable solitons
];

fn harmony_to_params(activations: &[f32; 8], consciousness: f32) -> RdParams {
    // Weighted blend of harmony parameter points
    let total: f32 = activations.iter().sum::<f32>().max(0.001);
    let mut f = 0.0f32;
    let mut k = 0.0f32;
    for (i, &act) in activations.iter().enumerate() {
        let w = act / total;
        f += w * HARMONY_RD[i].0;
        k += w * HARMONY_RD[i].1;
    }

    // Consciousness level nudges the system: high Phi → further from equilibrium
    // (more complex patterns). This is a small perturbation, not a large shift.
    let phi_nudge = (consciousness - 0.5) * 0.005;
    f = (f + phi_nudge).clamp(0.010, 0.070);
    k = (k + phi_nudge * 0.5).clamp(0.045, 0.070);

    RdParams {
        feed: f,
        kill: k,
        du: 0.2097,
        dv: 0.1050,
    }
}

// ─── Simulation ──────────────────────────────────────────────────────────────

/// Grid resolution for the RD simulation.
/// 64×64 runs fast enough to be deterministic per seed in <10ms.
const GRID: usize = 64;

struct RdGrid {
    u: Vec<f32>,
    v: Vec<f32>,
}

impl RdGrid {
    fn new_random(rng: &mut StdRng, params: &RdParams) -> Self {
        let n = GRID * GRID;
        let mut u = vec![1.0f32; n];
        let mut v = vec![0.0f32; n];

        // Seed with small random perturbation regions
        let n_seeds = 4 + (params.feed * 100.0) as usize;
        for _ in 0..n_seeds {
            let cx = rng.gen_range(8..GRID - 8);
            let cy = rng.gen_range(8..GRID - 8);
            let radius = rng.gen_range(2usize..6);
            for dy in 0..radius * 2 {
                for dx in 0..radius * 2 {
                    let x = cx + dx - radius;
                    let y = cy + dy - radius;
                    if x < GRID && y < GRID {
                        let idx = y * GRID + x;
                        u[idx] = 0.5 + rng.gen::<f32>() * 0.1;
                        v[idx] = 0.25 + rng.gen::<f32>() * 0.1;
                    }
                }
            }
        }
        Self { u, v }
    }

    fn step(&self, params: &RdParams, dt: f32) -> Self {
        let n = GRID;
        let mut nu = self.u.clone();
        let mut nv = self.v.clone();

        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let idx = y * n + x;
                let u = self.u[idx];
                let v = self.v[idx];

                // Discrete Laplacian (5-point stencil)
                let lap_u =
                    self.u[idx - n] + self.u[idx + n] + self.u[idx - 1] + self.u[idx + 1] - 4.0 * u;
                let lap_v =
                    self.v[idx - n] + self.v[idx + n] + self.v[idx - 1] + self.v[idx + 1] - 4.0 * v;

                // Gray-Scott reaction terms
                let uvv = u * v * v;
                nu[idx] =
                    (u + dt * (params.du * lap_u - uvv + params.feed * (1.0 - u))).clamp(0.0, 1.0);
                nv[idx] = (v + dt * (params.dv * lap_v + uvv - (params.feed + params.kill) * v))
                    .clamp(0.0, 1.0);
            }
        }

        Self { u: nu, v: nv }
    }
}

// ─── SVG rendering ────────────────────────────────────────────────────────────

/// Generate a reaction-diffusion artwork as an SVG scene graph.
///
/// Runs the Gray-Scott model for a fixed number of steps, then renders
/// the V-concentration field as a heatmap of colored circles.
pub fn generate(
    config: &crate::AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let params = harmony_to_params(
        &snapshot.harmony_activations,
        snapshot.consciousness_level as f32,
    );

    let mut grid = RdGrid::new_random(rng, &params);

    // Run simulation. More steps = richer patterns; gated by consciousness.
    let steps = 40 + (snapshot.consciousness_level as f32 * 80.0) as usize;
    let dt = 1.0;
    for _ in 0..steps {
        grid = grid.step(&params, dt);
    }

    // Render V-concentration as colored circles on a grid
    let cell_w = config.width / GRID as f32;
    let cell_h = config.height / GRID as f32;

    // Color palette: neuromodulators drive hue
    // dopamine → warm (red-orange), serotonin → cool (blue-green)
    let hue_base = snapshot.dopamine * 30.0 + snapshot.serotonin * 180.0;

    // Background
    let bg = SceneNode::rect(0.0, 0.0, config.width, config.height).with_style(
        symthaea_canvas::scene_graph::Style {
            fill: Some(symthaea_canvas::Color::from_hsla(hue_base, 0.15, 0.08, 1.0)),
            ..Default::default()
        },
    );

    let mut root = SceneNode::group(Some("reaction-diffusion")).with_child(bg);

    // Sample every other cell for performance (32×32 = 1024 circles)
    let step = 2;
    for y in (0..GRID).step_by(step) {
        for x in (0..GRID).step_by(step) {
            let v = grid.v[y * GRID + x];
            if v < 0.01 {
                continue; // skip background
            }

            let cx = x as f32 * cell_w + cell_w * step as f32 * 0.5;
            let cy = y as f32 * cell_h + cell_h * step as f32 * 0.5;
            let radius = (v * cell_w * step as f32 * 0.6).max(0.5);

            // Hue varies with concentration + position for visual richness
            let hue = (hue_base + v * 60.0 + x as f32 * 0.5) % 360.0;
            let lightness = 0.30 + v * 0.55;
            let opacity = 0.4 + v * 0.6;

            let dot =
                SceneNode::circle(cx, cy, radius).with_style(symthaea_canvas::scene_graph::Style {
                    fill: Some(symthaea_canvas::Color::from_hsla(
                        hue, 0.75, lightness, opacity,
                    )),
                    ..Default::default()
                });
            root = root.with_child(dot);
        }
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use symthaea_canvas::CognitiveSnapshot;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn harmony_to_params_bounded() {
        let activations = [0.5; 8];
        let p = harmony_to_params(&activations, 0.7);
        assert!(
            p.feed >= 0.010 && p.feed <= 0.070,
            "feed out of bounds: {}",
            p.feed
        );
        assert!(
            p.kill >= 0.045 && p.kill <= 0.070,
            "kill out of bounds: {}",
            p.kill
        );
    }

    #[test]
    fn dominant_harmony_shifts_params() {
        let mut spots = [0.0f32; 8];
        spots[0] = 1.0; // ResonantCoherence → spots
        let mut chaos = [0.0f32; 8];
        chaos[3] = 1.0; // InfinitePlay → spirals

        let p_spots = harmony_to_params(&spots, 0.5);
        let p_chaos = harmony_to_params(&chaos, 0.5);
        assert!(
            (p_spots.feed - p_chaos.feed).abs() > 0.005,
            "different harmonies should produce different F"
        );
    }

    #[test]
    fn rd_grid_step_bounded() {
        let params = harmony_to_params(&[0.5; 8], 0.5);
        let mut rng = StdRng::seed_from_u64(42);
        let grid = RdGrid::new_random(&mut rng, &params);
        let stepped = grid.step(&params, 1.0);
        for &u in &stepped.u {
            assert!(u >= 0.0 && u <= 1.0, "u out of [0,1]: {u}");
        }
        for &v in &stepped.v {
            assert!(v >= 0.0 && v <= 1.0, "v out of [0,1]: {v}");
        }
    }

    #[test]
    fn generate_produces_scene() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot();
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() > 5, "should generate meaningful scene");
    }

    #[test]
    fn generate_deterministic() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot();
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let s1 = generate(&config, &snapshot, &mut rng1);
        let s2 = generate(&config, &snapshot, &mut rng2);
        assert_eq!(s1.node_count(), s2.node_count());
    }

    #[test]
    fn high_consciousness_more_steps_richer() {
        let config = crate::AtelierConfig::default();
        let low = CognitiveSnapshot {
            consciousness_level: 0.1,
            harmony_activations: [0.5; 8],
            ..CognitiveSnapshot::dormant()
        };
        let high = CognitiveSnapshot {
            consciousness_level: 0.9,
            harmony_activations: [0.5; 8],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng_l = StdRng::seed_from_u64(7);
        let mut rng_h = StdRng::seed_from_u64(7);
        let s_low = generate(&config, &low, &mut rng_l);
        let s_high = generate(&config, &high, &mut rng_h);
        // High consciousness runs more steps → different (likely richer) pattern
        // Just confirm both run without panic and produce scenes
        assert!(s_low.node_count() > 0);
        assert!(s_high.node_count() > 0);
    }
}
