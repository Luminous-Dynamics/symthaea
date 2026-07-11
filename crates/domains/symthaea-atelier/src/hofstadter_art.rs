// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hofstadter butterfly artwork from real Harper-model spectra.
//!
//! The Hofstadter butterfly is the fractal energy spectrum of an electron on
//! a 2D square lattice in a perpendicular magnetic field (the Harper /
//! almost-Mathieu operator). At rational flux α = p/q the spectrum splits
//! into q bands; plotting energy (y) against flux (x) over all reduced
//! fractions produces the recursive butterfly.
//!
//! **Honesty contract**: every rendered point is a genuine eigenvalue of the
//! q×q Harper matrix, computed by [`HofstadterGenerator::generate_harper_slice_with_phase`]
//! from `symthaea-fractal-time-lab` — nothing is interpolated, cached, or
//! faked. Consciousness level scales only the flux-denominator cap (spectral
//! resolution), hard-bounded at [`MAX_Q_CAP`] so the number of distinct flux
//! values stays ≤ 57 and generation remains in the low-millisecond range.
//! Neuromodulators and the RNG drive the palette only; they never alter the
//! spectrum or point positions.
//!
//! # References
//! - Hofstadter, D. R. (1976). Energy levels and wave functions of Bloch
//!   electrons in rational and irrational magnetic fields. *Phys. Rev. B*,
//!   14, 2239–2249.
//! - Harper, P. G. (1955). Single band motion of conduction electrons in a
//!   uniform magnetic field. *Proc. Phys. Soc. A*, 68, 874.

use rand::Rng;
use rand::rngs::StdRng;
use std::f64::consts::PI;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};
use symthaea_fractal_time_lab::hofstadter::{HDC_DIM, HofstadterGenerator};

use crate::AtelierConfig;

/// Hard cap on the flux denominator. q ≤ 13 keeps the number of distinct
/// reduced fractions p/q at 57 (= Σ_{q=2}^{13} φ(q)) and every eigenproblem
/// at most 13×13 — well inside a low-millisecond budget.
const MAX_Q_CAP: usize = 13;

/// Minimum flux-denominator cap (dormant consciousness still shows the
/// coarse butterfly skeleton: 9 flux values at q ≤ 5).
const MIN_Q: usize = 5;

/// Boundary phases (kx = ky) sampled per flux value. The full butterfly band
/// at each rational flux is the union over Bloch phases; sampling three real
/// phases traces genuine band extent without inventing intermediate points.
const N_PHASES: usize = 3;

/// The Harper spectrum is contained in [-4, 4]; using the fixed physical
/// range keeps the vertical layout faithful across resolutions.
const ENERGY_RANGE: f64 = 4.0;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Compute real spectral points: `(flux, energy, q)` for every reduced
/// fraction p/q with 2 ≤ q ≤ `max_q`, at [`N_PHASES`] boundary phases.
fn butterfly_points(max_q: usize) -> Vec<(f64, f64, usize)> {
    let generator = HofstadterGenerator::new(HDC_DIM);
    let mut points = Vec::new();

    for q in 2..=max_q {
        for p in 1..q {
            if gcd(p, q) != 1 {
                continue;
            }
            let alpha = p as f64 / q as f64;
            for k in 0..N_PHASES {
                let phase = PI * k as f64 / (N_PHASES - 1) as f64;
                for energy in generator.generate_harper_slice_with_phase(p, q, phase, phase) {
                    points.push((alpha, energy, q));
                }
            }
        }
    }

    points
}

/// Generate a Hofstadter-butterfly artwork as an SVG scene graph.
///
/// Flux α on the x-axis, energy on the y-axis. Consciousness level scales
/// the flux-denominator cap between [`MIN_Q`] and [`MAX_Q_CAP`]; the palette
/// derives from snapshot neuromodulators (dopamine/serotonin hue base) with
/// an RNG hue offset for seed-to-seed variety.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let consciousness = snapshot.consciousness_level as f32;

    // Consciousness gates spectral resolution, hard-bounded.
    let max_q =
        MIN_Q + ((consciousness.clamp(0.0, 1.0)) * (MAX_Q_CAP - MIN_Q) as f32).round() as usize;
    let max_q = max_q.min(MAX_Q_CAP);

    let points = butterfly_points(max_q);

    // Palette: neuromodulators set the hue base; RNG adds an aesthetic
    // offset only (positions are pure physics).
    let hue_base =
        snapshot.dopamine * 60.0 + snapshot.serotonin * 240.0 + rng.r#gen::<f32>() * 40.0;
    let opacity = 0.35 + consciousness * 0.35;

    let margin = 0.06 * config.width.min(config.height);
    let plot_w = config.width - 2.0 * margin;
    let plot_h = config.height - 2.0 * margin;

    // Dark background, consistent with the attractor subsystem.
    let bg = SceneNode::rect(0.0, 0.0, config.width, config.height).with_style(Style {
        fill: Some(Color::from_hsla(250.0, 0.25, 0.06, 1.0)),
        ..Style::default()
    });

    let mut root = SceneNode::group(Some("hofstadter-butterfly")).with_child(bg);

    for &(alpha, energy, q) in &points {
        let x = margin + (alpha as f32) * plot_w;
        // Energy ∈ [-4, 4] mapped top-down (high energy at the top).
        let norm_e = ((energy + ENERGY_RANGE) / (2.0 * ENERGY_RANGE)).clamp(0.0, 1.0) as f32;
        let y = margin + (1.0 - norm_e) * plot_h;

        // Hue sweeps with energy; higher-q sub-bands render thinner, echoing
        // the true narrowing of bands deeper into the fractal.
        let hue = (hue_base + norm_e * 140.0) % 360.0;
        let radius = 0.8 + 3.0 / q as f32;

        let dot = SceneNode::circle(x, y, radius).with_style(Style {
            fill: Some(Color::from_hsl(hue, 0.85, 0.45 + norm_e * 0.25)),
            opacity: Some(opacity.min(0.9)),
            ..Style::default()
        });
        root.children.push(dot);
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_snapshot(consciousness: f64) -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: consciousness,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5; 8],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn butterfly_points_are_real_spectra() {
        // For flux 1/2 the Harper matrix at kx=ky=0 has eigenvalues {-2, 2}
        // per slice (q=2 gives 2 eigenvalues); all energies must lie in the
        // physical range [-4, 4].
        let points = butterfly_points(5);
        assert!(!points.is_empty());
        for &(alpha, energy, q) in &points {
            assert!(alpha > 0.0 && alpha < 1.0, "flux must be in (0,1)");
            assert!(
                (-ENERGY_RANGE..=ENERGY_RANGE).contains(&energy),
                "Harper eigenvalue {energy} escaped [-4,4]"
            );
            assert!((2..=5).contains(&q));
        }
    }

    #[test]
    fn flux_count_bounded_at_full_consciousness() {
        // Distinct reduced fractions with q ≤ 13 is Σφ(q) = 57 ≤ 60.
        let points = butterfly_points(MAX_Q_CAP);
        let mut fluxes: Vec<u64> = points.iter().map(|p| p.0.to_bits()).collect();
        fluxes.sort_unstable();
        fluxes.dedup();
        assert!(fluxes.len() <= 60, "flux steps {} > 60", fluxes.len());
        assert_eq!(fluxes.len(), 57);
    }

    #[test]
    fn generate_produces_nonempty_scene() {
        let config = AtelierConfig::default();
        let snapshot = test_snapshot(0.7);
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() > 10, "butterfly scene nearly empty");
    }

    #[test]
    fn consciousness_scales_resolution() {
        let config = AtelierConfig::default();
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let low = generate(&config, &test_snapshot(0.0), &mut rng1);
        let high = generate(&config, &test_snapshot(1.0), &mut rng2);
        assert!(
            high.node_count() > low.node_count(),
            "higher consciousness must add spectral resolution ({} vs {})",
            high.node_count(),
            low.node_count()
        );
    }

    #[test]
    fn generate_deterministic_same_seed() {
        let config = AtelierConfig::default();
        let snapshot = test_snapshot(0.7);
        let mut rng1 = StdRng::seed_from_u64(77);
        let mut rng2 = StdRng::seed_from_u64(77);
        let s1 = generate(&config, &snapshot, &mut rng1);
        let s2 = generate(&config, &snapshot, &mut rng2);
        let svg1 = symthaea_canvas::render_svg(&s1, snapshot.consciousness_level);
        let svg2 = symthaea_canvas::render_svg(&s2, snapshot.consciousness_level);
        assert_eq!(svg1, svg2);
    }
}
