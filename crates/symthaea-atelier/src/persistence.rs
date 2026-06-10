// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistence-texture generation from topological persistence diagrams.
//!
//! Birth-death pairs from the consciousness topology become Voronoi seed points,
//! creating organic cellular textures. Persistence (death - birth) controls
//! circle radius, creating a visual representation of topological significance.

use rand::Rng;
use rand::rngs::StdRng;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};

use crate::AtelierConfig;

/// Generate persistence-texture artwork.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let mut root = SceneNode::group(Some("persistence"));

    // Background circles from persistence_components (birth-death pairs)
    for (i, pair) in snapshot.persistence_components.iter().enumerate() {
        let birth = pair[0] as f32;
        let death = pair[1] as f32;
        let persistence = (death - birth).abs();

        // Map birth to position, persistence to size
        let x = (birth * config.width * 0.8) + config.width * 0.1;
        let y = config.height * 0.3 + (i as f32 * config.height * 0.1);
        let radius = persistence * config.width * 0.15 + 5.0;

        let hue = birth * 200.0 + 180.0; // cool tones for components
        let opacity = 0.3 + persistence * 0.5;

        let circle = SceneNode::circle(x, y, radius).with_style(Style {
            fill: Some(Color::from_hsl(hue, 0.6, 0.4 + persistence * 0.2)),
            opacity: Some(opacity.min(0.8)),
            ..Style::default()
        });
        root.children.push(circle);
    }

    // Rings from persistence_cycles
    for (i, pair) in snapshot.persistence_cycles.iter().enumerate() {
        let birth = pair[0] as f32;
        let death = pair[1] as f32;
        let persistence = (death - birth).abs();

        let x = config.width * 0.5 + (birth - 0.5) * config.width * 0.6;
        let y = config.height * 0.6 + (i as f32 * config.height * 0.08);
        let radius = persistence * config.width * 0.12 + 10.0;

        let hue = 30.0 + birth * 60.0; // warm tones for cycles

        let ring = SceneNode::circle(x, y, radius).with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(hue, 0.7, 0.5)),
            stroke_width: Some(2.0 + persistence * 4.0),
            opacity: Some(0.5 + persistence * 0.3),
            ..Style::default()
        });
        root.children.push(ring);
    }

    // Fill with random Voronoi-style cells if topology is sparse
    let min_cells = 5;
    let current_count = root.children.len();
    if current_count < min_cells {
        for _ in current_count..min_cells {
            let x = rng.r#gen::<f32>() * config.width;
            let y = rng.r#gen::<f32>() * config.height;
            let r = 10.0 + rng.r#gen::<f32>() * 30.0;
            let hue = rng.r#gen::<f32>() * 360.0;

            let cell = SceneNode::circle(x, y, r).with_style(Style {
                fill: Some(Color::from_hsl(hue, 0.4, 0.5)),
                opacity: Some(0.3),
                ..Style::default()
            });
            root.children.push(cell);
        }
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn generates_from_empty_persistence() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot::dormant(); // empty persistence
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Should still produce cells via fallback
        assert!(scene.children.len() >= 5);
    }

    #[test]
    fn generates_from_rich_persistence() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8], [0.3, 0.9]],
            persistence_cycles: vec![[0.2, 0.6], [0.4, 0.7]],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.children.len() >= 5);
    }
}
