// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! L-system generative art driven by Eight Harmony activations.
//!
//! Each harmony influences branching angle, recursion depth, and production rules,
//! creating fractal structures that reflect the system's value basis.

use rand::rngs::StdRng;
use rand::Rng;
use std::fmt::Write;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};

use crate::AtelierConfig;

/// L-system production rule.
struct Rule {
    /// Input character.
    from: char,
    /// Replacement string.
    to: String,
}

/// Generate an L-system artwork.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let harmonies = &snapshot.harmony_activations;

    // Derive L-system parameters from harmonies
    let base_angle = 15.0 + harmonies[3] * 30.0; // InfinitePlay → wider branching
    let depth = 2 + (harmonies[6] * 4.0) as usize; // EvolutionaryProgression → deeper
    let depth = depth.min(6); // cap for performance
    let branch_ratio = 0.6 + harmonies[0] * 0.2; // ResonantCoherence → longer branches
    let initial_length = config.height * 0.25;

    // Build production rules based on active harmonies
    let rules = build_rules(harmonies, rng);
    let axiom = "F";

    // Expand L-system string
    let mut current = axiom.to_string();
    for _ in 0..depth {
        current = expand(&current, &rules);
        if current.len() > config.max_elements * 10 {
            break; // safety cap
        }
    }

    // Turtle-walk to generate SVG path
    let path_data = turtle_walk(
        &current,
        config.width / 2.0,
        config.height * 0.85,
        -90.0_f32.to_radians(), // start pointing up
        initial_length * (0.5_f32).powf(depth as f32 * 0.3),
        base_angle.to_radians(),
        branch_ratio,
    );

    let stroke_color = Color::from_hsl(
        45.0 + harmonies[1] * 60.0, // PanSentientFlourishing → hue shift
        0.6 + harmonies[4] * 0.3,   // Interconnectedness → saturation
        0.3 + snapshot.consciousness_level as f32 * 0.4,
    );

    let path_node = SceneNode::path(path_data).with_style(Style {
        fill: None,
        stroke: Some(stroke_color),
        stroke_width: Some(1.5 + snapshot.dopamine * 2.0),
        opacity: Some(0.7 + snapshot.consciousness_level as f32 * 0.3),
        ..Style::default()
    });

    SceneNode::group(Some("lsystem")).with_child(path_node)
}

fn build_rules(harmonies: &[f32; 8], rng: &mut StdRng) -> Vec<Rule> {
    let mut rules = Vec::new();

    // Base rule: F → F[+F]F[-F]F (classic branching)
    let mut base = if harmonies[3] > 0.5 {
        // InfinitePlay: add more branching variety
        "F[+F]F[-F][+F]".to_string()
    } else if harmonies[7] > 0.5 {
        // SacredStillness: minimal, meditative
        "F[+F][-F]".to_string()
    } else {
        "F[+F]F[-F]F".to_string()
    };

    // Add slight randomness
    if rng.r#gen::<f32>() < harmonies[2] {
        // IntegralWisdom: systematic structure
        base.push_str("[+F]");
    }

    rules.push(Rule {
        from: 'F',
        to: base,
    });

    rules
}

fn expand(input: &str, rules: &[Rule]) -> String {
    let mut output = String::with_capacity(input.len() * 3);
    for ch in input.chars() {
        let mut replaced = false;
        for rule in rules {
            if ch == rule.from {
                output.push_str(&rule.to);
                replaced = true;
                break;
            }
        }
        if !replaced {
            output.push(ch);
        }
    }
    output
}

fn turtle_walk(
    instructions: &str,
    start_x: f32,
    start_y: f32,
    start_angle: f32,
    step_length: f32,
    turn_angle: f32,
    branch_ratio: f32,
) -> String {
    let mut path = String::with_capacity(instructions.len() * 12);
    let mut x = start_x;
    let mut y = start_y;
    let mut angle = start_angle;
    let mut stack: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut length = step_length;
    let mut first = true;

    for ch in instructions.chars() {
        match ch {
            'F' => {
                let nx = x + angle.cos() * length;
                let ny = y + angle.sin() * length;
                if first {
                    let _ = write!(path, "M {x:.1} {y:.1} L {nx:.1} {ny:.1}");
                    first = false;
                } else {
                    let _ = write!(path, " M {x:.1} {y:.1} L {nx:.1} {ny:.1}");
                }
                x = nx;
                y = ny;
            }
            '+' => angle += turn_angle,
            '-' => angle -= turn_angle,
            '[' => {
                stack.push((x, y, angle, length));
                length *= branch_ratio;
            }
            ']' => {
                if let Some((sx, sy, sa, sl)) = stack.pop() {
                    x = sx;
                    y = sy;
                    angle = sa;
                    length = sl;
                }
            }
            _ => {}
        }
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn expand_basic() {
        let rules = vec![Rule {
            from: 'F',
            to: "F+F".to_string(),
        }];
        assert_eq!(expand("F", &rules), "F+F");
        assert_eq!(expand("F+F", &rules), "F+F+F+F");
    }

    #[test]
    fn turtle_produces_path() {
        let path = turtle_walk(
            "F+F+F+F",
            100.0,
            100.0,
            0.0,
            50.0,
            std::f32::consts::FRAC_PI_2,
            1.0,
        );
        assert!(path.contains("M "));
        assert!(path.contains("L "));
    }

    #[test]
    fn generate_produces_scene() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.6,
            harmony_activations: [0.5; 8],
            dopamine: 0.5,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        assert!(scene.node_count() >= 2);
    }

    #[test]
    fn depth_capped() {
        let config = AtelierConfig {
            max_elements: 100,
            ..Default::default()
        };
        let snapshot = CognitiveSnapshot {
            harmony_activations: [1.0; 8], // max everything
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Should not hang or OOM — depth is capped at 6
        assert!(scene.node_count() >= 2);
    }
}
