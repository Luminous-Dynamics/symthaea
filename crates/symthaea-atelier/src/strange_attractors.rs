// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strange attractor visualization for consciousness-driven art.
//!
//! Strange attractors are chaotic dynamical systems whose trajectories
//! never repeat yet remain bounded — they trace intricate, infinitely
//! detailed geometric structures. When consciousness is the parameter,
//! the attractor's complexity reflects the depth of inner experience.
//!
//! **Attractor-Harmony mapping:**
//! - SacredStillness → Lorenz attractor near fixed point (low chaos)
//! - InfinitePlay → Clifford attractor (high chaos, unpredictable beauty)
//! - IntegralWisdom → Rössler attractor (ordered spirals)
//! - UniversalInterconnectedness → Duffing attractor (coupled oscillators)
//!
//! # References
//! - Lorenz, E. N. (1963). Deterministic nonperiodic flow. *JAS*, 20, 130–141.
//! - Clifford Pickover: `x' = sin(a*y) + c*cos(a*x)`, `y' = sin(b*x) + d*cos(b*y)`

use rand::Rng;
use rand::rngs::StdRng;
use symthaea_canvas::{CognitiveSnapshot, SceneNode};

// ─── Attractor types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum AttractorType {
    Clifford,
    Lorenz,
    Rossler,
    Duffing,
}

fn select_attractor(activations: &[f32; 8]) -> AttractorType {
    // Select based on the most dominant harmony
    let dominant = activations
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    match dominant {
        3 => AttractorType::Clifford, // InfinitePlay → chaos
        7 => AttractorType::Lorenz,   // SacredStillness → Lorenz
        2 => AttractorType::Rossler,  // IntegralWisdom → ordered spirals
        4 => AttractorType::Duffing,  // UniversalInterconnectedness → coupled
        _ => AttractorType::Clifford, // default: Clifford
    }
}

// ─── Clifford attractor ───────────────────────────────────────────────────────

/// Clifford attractor parameters.
/// `x' = sin(a*y) + c*cos(a*x)`, `y' = sin(b*x) + d*cos(b*y)`
#[derive(Debug, Clone, Copy)]
struct CliffordParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

fn harmony_to_clifford(activations: &[f32; 8], consciousness: f32) -> CliffordParams {
    // Map harmony activations to attractor parameters.
    // Parameters are carefully chosen to stay in chaotic-but-bounded regimes.
    // Reference: Sprott (1993) attractor catalog.
    let h = activations;

    // a, b in [-2.5, 2.5], c, d in [-1.5, 1.5] for bounded chaos
    let a = -1.4 + (h[0] - h[2]) * 0.8 + consciousness * 0.3;
    let b = 1.6 + (h[1] - h[3]) * 0.6 - consciousness * 0.2;
    let c = 1.0 + (h[4] - h[6]) * 0.5;
    let d = -0.6 + (h[5] - h[7]) * 0.4 + consciousness * 0.1;

    CliffordParams {
        a: a.clamp(-2.5, 2.5),
        b: b.clamp(-2.5, 2.5),
        c: c.clamp(-1.5, 1.5),
        d: d.clamp(-1.5, 1.5),
    }
}

fn iterate_clifford(params: &CliffordParams, n_points: usize, seed: u32) -> Vec<[f32; 2]> {
    let mut x = (seed as f32 * 0.001).sin() * 0.5;
    let mut y = (seed as f32 * 0.002).cos() * 0.5;
    let mut points = Vec::with_capacity(n_points);

    // Burn-in: let the orbit settle onto the attractor
    for _ in 0..200 {
        let nx = (params.a * y).sin() + params.c * (params.a * x).cos();
        let ny = (params.b * x).sin() + params.d * (params.b * y).cos();
        x = nx;
        y = ny;
    }

    for _ in 0..n_points {
        let nx = (params.a * y).sin() + params.c * (params.a * x).cos();
        let ny = (params.b * x).sin() + params.d * (params.b * y).cos();
        x = nx;
        y = ny;
        if x.is_finite() && y.is_finite() {
            points.push([x, y]);
        }
    }
    points
}

// ─── Lorenz attractor ─────────────────────────────────────────────────────────

fn iterate_lorenz(consciousness: f32, n_points: usize, seed: u32) -> Vec<[f32; 2]> {
    // Lorenz parameters: sigma, rho, beta
    // Standard chaos: sigma=10, rho=28, beta=8/3
    // Near fixed point (SacredStillness): rho < 1 (stable), rho→28 (chaotic)
    let rho = 1.0 + consciousness * 27.0; // 1 (still) to 28 (classic chaos)
    let sigma = 10.0;
    let beta = 8.0 / 3.0;

    let s = seed as f32 * 0.001;
    let mut x = s.sin() * 0.1;
    let mut y = s.cos() * 0.1;
    let mut z = rho - 1.0 + s.sin().abs() * 0.01;
    let dt = 0.01;

    let mut points = Vec::with_capacity(n_points);

    // Burn-in
    for _ in 0..300 {
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
    }

    for _ in 0..n_points {
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
        if x.is_finite() && y.is_finite() {
            // Project 3D Lorenz to 2D: use x-z plane (the butterfly wings)
            points.push([x, z]);
        }
    }
    points
}

// ─── Rössler attractor ────────────────────────────────────────────────────────

fn iterate_rossler(consciousness: f32, n_points: usize, seed: u32) -> Vec<[f32; 2]> {
    // Rössler: dx = -y-z, dy = x+a*y, dz = b+z*(x-c)
    // Standard: a=0.2, b=0.2, c=5.7 (spiral chaos)
    let a = 0.1 + consciousness * 0.15;
    let b = 0.1 + consciousness * 0.1;
    let c = 4.0 + consciousness * 2.0;

    let s = seed as f32 * 0.001;
    let mut x = 0.1 + s.sin() * 0.05;
    let mut y = 0.0;
    let mut z = 0.0;
    let dt = 0.05;

    let mut points = Vec::with_capacity(n_points);

    for _ in 0..200 {
        let dx = -y - z;
        let dy = x + a * y;
        let dz = b + z * (x - c);
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
    }

    for _ in 0..n_points {
        let dx = -y - z;
        let dy = x + a * y;
        let dz = b + z * (x - c);
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
        if x.is_finite() && y.is_finite() {
            points.push([x, y]);
        }
    }
    points
}

// ─── Duffing attractor ────────────────────────────────────────────────────────

fn iterate_duffing(consciousness: f32, n_points: usize, seed: u32) -> Vec<[f32; 2]> {
    // Duffing: dx = y, dy = -delta*y - alpha*x - beta*x^3 + gamma*cos(omega*t)
    let delta = 0.3;
    let alpha = -1.0;
    let beta = 1.0;
    let gamma = 0.3 + consciousness * 0.4;
    let omega = 1.2;

    let s = seed as f32 * 0.001;
    let mut x = s.sin();
    let mut y = 0.0;
    let mut t = 0.0f32;
    let dt = 0.05;

    let mut points = Vec::with_capacity(n_points);

    for _ in 0..200 {
        let dx = y;
        let dy = -delta * y - alpha * x - beta * x.powi(3) + gamma * (omega * t).cos();
        x += dx * dt;
        y += dy * dt;
        t += dt;
    }

    for _ in 0..n_points {
        let dx = y;
        let dy = -delta * y - alpha * x - beta * x.powi(3) + gamma * (omega * t).cos();
        x += dx * dt;
        y += dy * dt;
        t += dt;
        if x.is_finite() && y.is_finite() {
            points.push([x, y]);
        }
    }
    points
}

// ─── Rendering ───────────────────────────────────────────────────────────────

/// Normalize points to [0, config.width] × [0, config.height] viewport.
fn normalize_points(points: &[[f32; 2]], width: f32, height: f32, margin: f32) -> Vec<[f32; 2]> {
    if points.is_empty() {
        return Vec::new();
    }

    let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);

    let range_x = (max_x - min_x).max(0.001);
    let range_y = (max_y - min_y).max(0.001);

    points
        .iter()
        .map(|p| {
            [
                margin + (p[0] - min_x) / range_x * (width - 2.0 * margin),
                margin + (p[1] - min_y) / range_y * (height - 2.0 * margin),
            ]
        })
        .collect()
}

/// Generate a strange attractor visualization as an SVG scene graph.
///
/// The attractor type is selected by the dominant harmony; orbit depth
/// scales with consciousness level.
pub fn generate(
    config: &crate::AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let consciousness = snapshot.consciousness_level as f32;
    let attractor = select_attractor(&snapshot.harmony_activations);

    // More points = richer orbit trace. Consciousness gates complexity.
    let n_points = 2000 + (consciousness * 6000.0) as usize;
    let seed: u32 = rng.r#gen();

    let raw_points = match attractor {
        AttractorType::Clifford => {
            let params = harmony_to_clifford(&snapshot.harmony_activations, consciousness);
            iterate_clifford(&params, n_points, seed)
        }
        AttractorType::Lorenz => iterate_lorenz(consciousness, n_points, seed),
        AttractorType::Rossler => iterate_rossler(consciousness, n_points, seed),
        AttractorType::Duffing => iterate_duffing(consciousness, n_points, seed),
    };

    let points = normalize_points(&raw_points, config.width, config.height, 40.0);

    // Color: cycle hue through the orbit based on neuromodulators
    let hue_base = snapshot.dopamine * 60.0 + snapshot.serotonin * 240.0;
    let n = points.len();

    // Dark background
    let bg = SceneNode::rect(0.0, 0.0, config.width, config.height).with_style(
        symthaea_canvas::scene_graph::Style {
            fill: Some(symthaea_canvas::Color::from_hsla(240.0, 0.20, 0.05, 1.0)),
            ..Default::default()
        },
    );

    // Render points as tiny translucent circles; color cycles with orbit position
    let step = (n / 3000).max(1); // subsample for SVG performance
    let mut root =
        SceneNode::group(Some(&format!("{attractor:?}-attractor").to_lowercase())).with_child(bg);

    for (i, p) in points.iter().step_by(step).enumerate() {
        let t = i as f32 / (n / step) as f32;
        let hue = (hue_base + t * 180.0) % 360.0;
        let lightness = 0.45 + t * 0.30;
        let opacity = 0.15 + consciousness * 0.25;

        let dot = SceneNode::circle(p[0], p[1], 1.0 + consciousness * 0.5).with_style(
            symthaea_canvas::scene_graph::Style {
                fill: Some(symthaea_canvas::Color::from_hsla(
                    hue, 0.90, lightness, opacity,
                )),
                ..Default::default()
            },
        );
        root = root.with_child(dot);
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use symthaea_canvas::CognitiveSnapshot;

    fn test_snapshot(dominant_harmony: usize) -> CognitiveSnapshot {
        let mut harmonies = [0.1f32; 8];
        harmonies[dominant_harmony] = 0.9;
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: harmonies,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn clifford_bounded() {
        let params = harmony_to_clifford(&[0.5; 8], 0.7);
        let points = iterate_clifford(&params, 1000, 42);
        assert!(!points.is_empty());
        for p in &points {
            assert!(p[0].is_finite() && p[1].is_finite());
        }
    }

    #[test]
    fn lorenz_bounded() {
        let points = iterate_lorenz(0.7, 500, 42);
        assert!(!points.is_empty());
        for p in &points {
            assert!(p[0].is_finite() && p[1].is_finite());
        }
    }

    #[test]
    fn rossler_bounded() {
        let points = iterate_rossler(0.5, 500, 42);
        assert!(!points.is_empty());
        for p in &points {
            assert!(p[0].is_finite() && p[1].is_finite());
        }
    }

    #[test]
    fn duffing_bounded() {
        let points = iterate_duffing(0.6, 500, 42);
        assert!(!points.is_empty());
        for p in &points {
            assert!(p[0].is_finite() && p[1].is_finite());
        }
    }

    #[test]
    fn harmony_selects_attractor() {
        // InfinitePlay (idx 3) → Clifford
        assert!(matches!(
            select_attractor(&{
                let mut h = [0.1f32; 8];
                h[3] = 0.9;
                h
            }),
            AttractorType::Clifford
        ));
        // SacredStillness (idx 7) → Lorenz
        assert!(matches!(
            select_attractor(&{
                let mut h = [0.1f32; 8];
                h[7] = 0.9;
                h
            }),
            AttractorType::Lorenz
        ));
    }

    #[test]
    fn generate_clifford_produces_scene() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot(3); // InfinitePlay → Clifford
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() > 10);
    }

    #[test]
    fn generate_lorenz_produces_scene() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot(7); // SacredStillness → Lorenz
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() > 10);
    }

    #[test]
    fn normalize_points_in_viewport() {
        let points = vec![[-5.0, -3.0], [0.0, 0.0], [5.0, 3.0]];
        let norm = normalize_points(&points, 1024.0, 1024.0, 40.0);
        for p in &norm {
            assert!(p[0] >= 0.0 && p[0] <= 1024.0);
            assert!(p[1] >= 0.0 && p[1] <= 1024.0);
        }
    }

    #[test]
    fn generate_deterministic() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot(3);
        let mut rng1 = StdRng::seed_from_u64(77);
        let mut rng2 = StdRng::seed_from_u64(77);
        let s1 = generate(&config, &snapshot, &mut rng1);
        let s2 = generate(&config, &snapshot, &mut rng2);
        assert_eq!(s1.node_count(), s2.node_count());
    }
}
