// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parametric curve generation driven by the thought vector.
//!
//! The 32D thought_vector maps to Lissajous and rose curve parameters,
//! creating flowing, organic compositions. Neuromodulators affect stroke style.

use rand::Rng;
use rand::rngs::StdRng;
use std::fmt::Write;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};

use crate::AtelierConfig;

/// Generate parametric curve artwork.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let cx = config.width / 2.0;
    let cy = config.height / 2.0;
    let tv = &snapshot.thought_vector;
    let num_curves = 3 + (snapshot.consciousness_level * 5.0) as usize;

    let mut root = SceneNode::group(Some("curves"));

    for i in 0..num_curves {
        // Extract parameters from thought vector (wrap around if short)
        let freq_x = 1.0 + get_tv(tv, i * 4, rng) * 4.0;
        let freq_y = 1.0 + get_tv(tv, i * 4 + 1, rng) * 4.0;
        let phase = get_tv(tv, i * 4 + 2, rng) * std::f32::consts::TAU;
        let amplitude = config.width * 0.15 + get_tv(tv, i * 4 + 3, rng) * config.width * 0.2;

        let path_data = lissajous_path(cx, cy, freq_x, freq_y, phase, amplitude, 200);

        let hue = (i as f32 / num_curves as f32) * 360.0;
        let saturation = 0.5 + snapshot.serotonin * 0.4;
        let lightness = 0.3 + snapshot.consciousness_level as f32 * 0.3;

        let curve = SceneNode::path(path_data).with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(hue, saturation, lightness)),
            stroke_width: Some(1.0 + snapshot.dopamine * 3.0),
            opacity: Some(0.4 + (i as f32 / num_curves as f32) * 0.4),
            ..Style::default()
        });

        root.children.push(curve);
    }

    // Add rose curves if arousal is high
    if snapshot.arousal > 0.4 {
        let n_petals = 3 + (snapshot.arousal * 5.0) as u32;
        let rose_data = rose_curve_path(cx, cy, config.width * 0.3, n_petals, 300);

        let rose = SceneNode::path(rose_data).with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(300.0 + snapshot.valence * 60.0, 0.7, 0.5)),
            stroke_width: Some(0.8),
            opacity: Some(0.5),
            ..Style::default()
        });

        root.children.push(rose);
    }

    root
}

/// Get a thought vector element, or a random fallback if index is out of bounds.
fn get_tv(tv: &[f32], idx: usize, rng: &mut StdRng) -> f32 {
    if idx < tv.len() {
        (tv[idx] + 1.0) / 2.0 // normalize from [-1,1] to [0,1]
    } else {
        rng.r#gen::<f32>()
    }
}

/// Generate SVG path data for a Lissajous curve using smooth cubic Bezier segments.
///
/// Instead of straight-line polylines, this produces cubic Bezier curves (C command)
/// for smooth, organic-looking paths. Control points are computed from the
/// tangent of the parametric function at each sample point.
fn lissajous_path(
    cx: f32,
    cy: f32,
    freq_x: f32,
    freq_y: f32,
    phase: f32,
    amplitude: f32,
    steps: usize,
) -> String {
    let mut path = String::with_capacity(steps * 30);
    let steps = steps.max(4);

    // Sample points
    let points: Vec<(f32, f32)> = (0..=steps)
        .map(|i| {
            let t = (i as f32 / steps as f32) * std::f32::consts::TAU;
            let x = cx + amplitude * (freq_x * t + phase).sin();
            let y = cy + amplitude * (freq_y * t).sin();
            (x, y)
        })
        .collect();

    // Start at first point
    let _ = write!(path, "M {:.1} {:.1}", points[0].0, points[0].1);

    // Cubic Bezier through every 3 points (Catmull-Rom → Bezier conversion)
    for i in 0..points.len() - 1 {
        let p0 = if i > 0 { points[i - 1] } else { points[i] };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 < points.len() {
            points[i + 2]
        } else {
            points[i + 1]
        };

        // Catmull-Rom to cubic Bezier control points
        let tension = 6.0;
        let cp1x = p1.0 + (p2.0 - p0.0) / tension;
        let cp1y = p1.1 + (p2.1 - p0.1) / tension;
        let cp2x = p2.0 - (p3.0 - p1.0) / tension;
        let cp2y = p2.1 - (p3.1 - p1.1) / tension;

        let _ = write!(
            path,
            " C {cp1x:.1} {cp1y:.1}, {cp2x:.1} {cp2y:.1}, {:.1} {:.1}",
            p2.0, p2.1
        );
    }

    path
}

/// Generate SVG path data for a rose (rhodonea) curve.
fn rose_curve_path(cx: f32, cy: f32, radius: f32, petals: u32, steps: usize) -> String {
    let k = petals as f32;
    let mut path = String::with_capacity(steps * 16);
    for i in 0..=steps {
        let theta = (i as f32 / steps as f32) * std::f32::consts::TAU;
        let r = radius * (k * theta).cos();
        let x = cx + r * theta.cos();
        let y = cy + r * theta.sin();
        if i == 0 {
            let _ = write!(path, "M {x:.1} {y:.1}");
        } else {
            let _ = write!(path, " L {x:.1} {y:.1}");
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn lissajous_generates_path() {
        let path = lissajous_path(256.0, 256.0, 3.0, 2.0, 0.5, 100.0, 50);
        assert!(path.starts_with("M "));
        // Now uses cubic Bezier (C) instead of straight lines (L)
        assert!(
            path.contains(" C "),
            "should contain Bezier curves: {}",
            &path[..80.min(path.len())]
        );
    }

    #[test]
    fn rose_generates_path() {
        let path = rose_curve_path(256.0, 256.0, 100.0, 5, 50);
        assert!(path.starts_with("M "));
    }

    #[test]
    fn generate_with_short_thought_vector() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            thought_vector: vec![0.5], // very short
            consciousness_level: 0.5,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() >= 2);
    }

    #[test]
    fn high_arousal_adds_rose() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            arousal: 0.8,
            consciousness_level: 0.5,
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Should have curves + rose
        assert!(scene.children.len() >= 4); // 3+ Lissajous + 1 rose
    }
}
