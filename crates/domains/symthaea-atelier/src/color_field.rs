// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator-driven color field paintings.
//!
//! Maps the 8 neuromodulator levels to a gradient field across the viewport.
//! Each region's color is determined by the dominant neuromodulator,
//! creating large soft-edged abstract color compositions.

use rand::Rng;
use rand::rngs::StdRng;
use symthaea_canvas::scene_graph::{Style, Transform};
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};

use crate::AtelierConfig;

/// Neuromodulator-to-color mapping.
struct NeuroColor {
    #[allow(dead_code)]
    name: &'static str,
    level: f32,
    hue: f32,
    saturation: f32,
}

/// Generate color field artwork.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let mut root = SceneNode::group(Some("color-field"));

    let neuro_colors = [
        NeuroColor {
            name: "dopamine",
            level: snapshot.dopamine,
            hue: 45.0,
            saturation: 0.80,
        },
        NeuroColor {
            name: "serotonin",
            level: snapshot.serotonin,
            hue: 30.0,
            saturation: 0.70,
        },
        NeuroColor {
            name: "noradrenaline",
            level: snapshot.noradrenaline,
            hue: 210.0,
            saturation: 0.65,
        },
        NeuroColor {
            name: "acetylcholine",
            level: snapshot.acetylcholine,
            hue: 150.0,
            saturation: 0.60,
        },
        NeuroColor {
            name: "oxytocin",
            level: snapshot.oxytocin,
            hue: 330.0,
            saturation: 0.55,
        },
        NeuroColor {
            name: "gaba",
            level: snapshot.gaba,
            hue: 270.0,
            saturation: 0.40,
        },
    ];

    // Sort by level (highest first) to layer dominant colors on top
    let mut sorted: Vec<&NeuroColor> = neuro_colors.iter().collect();
    sorted.sort_by(|a, b| {
        b.level
            .partial_cmp(&a.level)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Create overlapping soft rectangles for each neuromodulator
    for (_i, nc) in sorted.iter().enumerate() {
        if nc.level < 0.1 {
            continue; // skip negligible neuromodulators
        }

        // Position: spread across viewport with some randomness
        let x = rng.r#gen::<f32>() * config.width * 0.4;
        let y = rng.r#gen::<f32>() * config.height * 0.4;
        let w = config.width * (0.4 + nc.level * 0.5);
        let h = config.height * (0.4 + nc.level * 0.5);

        let lightness = 0.3 + snapshot.consciousness_level as f32 * 0.3;
        let color = Color::from_hsl(nc.hue, nc.saturation, lightness);

        let rect = SceneNode::rect(x, y, w, h)
            .with_style(Style {
                fill: Some(color),
                opacity: Some(0.15 + nc.level * 0.25),
                ..Style::default()
            })
            .with_transform(Transform {
                rotate_deg: rng.r#gen::<f32>() * 30.0 - 15.0,
                ..Transform::identity()
            });

        root.children.push(rect);

        // Add a circle accent for high-level neuromodulators
        if nc.level > 0.5 {
            let accent_r = config.width * 0.05 + nc.level * config.width * 0.1;
            let accent_x = x + w * 0.5;
            let accent_y = y + h * 0.5;

            let accent = SceneNode::circle(accent_x, accent_y, accent_r).with_style(Style {
                fill: Some(Color::from_hsl(
                    nc.hue,
                    nc.saturation * 1.2,
                    lightness + 0.15,
                )),
                opacity: Some(0.3 + nc.level * 0.2),
                ..Style::default()
            });
            root.children.push(accent);
        }
    }

    // Valence overlay: warm wash for positive, cool for negative
    let valence_hue = if snapshot.valence >= 0.0 {
        30.0 // warm amber
    } else {
        220.0 // cool blue
    };
    let valence_opacity = snapshot.valence.abs() * 0.15;

    if valence_opacity > 0.02 {
        let wash = SceneNode::rect(0.0, 0.0, config.width, config.height).with_style(Style {
            fill: Some(Color::from_hsl(valence_hue, 0.5, 0.5)),
            opacity: Some(valence_opacity),
            ..Style::default()
        });
        root.children.push(wash);
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn generates_color_field() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            dopamine: 0.7,
            serotonin: 0.5,
            noradrenaline: 0.3,
            consciousness_level: 0.6,
            valence: 0.3,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.children.len() >= 2); // at least DA + 5-HT blocks
    }

    #[test]
    fn low_neuro_sparse() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            dopamine: 0.05,
            serotonin: 0.05,
            noradrenaline: 0.05,
            acetylcholine: 0.05,
            oxytocin: 0.05,
            gaba: 0.05,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Most below 0.1 threshold, so very sparse
        assert!(scene.children.len() <= 3);
    }

    #[test]
    fn negative_valence_cool_wash() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            valence: -0.8,
            dopamine: 0.5,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.children.len() >= 2); // at least DA block + valence wash
    }
}
