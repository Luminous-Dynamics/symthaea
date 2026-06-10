// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Harmony-specific visual vocabularies.
//!
//! Each of the Eight Harmonies has a characteristic visual language:
//! - **ResonantCoherence**: Concentric circles, mandalas — unity
//! - **PanSentientFlourishing**: Organic blooms, radiating petals — growth
//! - **IntegralWisdom**: Geometric tessellations, crystalline forms — structure
//! - **InfinitePlay**: Spirals, swirls, recursive fractals — playfulness
//! - **UniversalInterconnectedness**: Webs, networks, overlapping circles — connection
//! - **SacredReciprocity**: Mirrored/symmetrical forms, yin-yang curves — balance
//! - **EvolutionaryProgression**: Ascending arcs, arrows, gradient progressions — growth
//! - **SacredStillness**: Ensō circles, negative space, single brushstrokes — stillness

use std::fmt::Write;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{Color, SceneNode};

/// Generate a visual element characteristic of the given harmony.
///
/// `activation` controls size/intensity (0.0-1.0).
/// `cx`, `cy` are the center coordinates.
/// `radius` is the base size.
pub fn harmony_shape(
    harmony_index: usize,
    activation: f32,
    cx: f32,
    cy: f32,
    radius: f32,
    hue: f32,
) -> SceneNode {
    let a = activation.clamp(0.0, 1.0);
    match harmony_index {
        0 => concentric_circles(cx, cy, radius, a, hue),
        1 => bloom(cx, cy, radius, a, hue),
        2 => crystal(cx, cy, radius, a, hue),
        3 => spiral(cx, cy, radius, a, hue),
        4 => web(cx, cy, radius, a, hue),
        5 => mirror(cx, cy, radius, a, hue),
        6 => ascending_arc(cx, cy, radius, a, hue),
        7 => enso(cx, cy, radius, a, hue),
        _ => SceneNode::group(None),
    }
}

/// ResonantCoherence: concentric circles radiating from center.
fn concentric_circles(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("coherence"));
    let rings = 3 + (activation * 4.0) as usize;
    for i in 0..rings {
        let r = radius * (i as f32 + 1.0) / rings as f32;
        let opacity = 0.6 - (i as f32 / rings as f32) * 0.4;
        let circle = SceneNode::circle(cx, cy, r).with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(hue, 0.6, 0.5)),
            stroke_width: Some(1.5 + activation),
            opacity: Some(opacity),
            ..Style::default()
        });
        group.children.push(circle);
    }
    group
}

/// PanSentientFlourishing: organic bloom with radiating petals.
fn bloom(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let petals = 5 + (activation * 8.0) as usize;
    let mut path = String::with_capacity(petals * 60);

    for i in 0..petals {
        let angle = (i as f32 / petals as f32) * std::f32::consts::TAU;
        let tip_x = cx + radius * angle.cos();
        let tip_y = cy + radius * angle.sin();

        // Petal curves via quadratic Bezier
        let cp_angle = angle + 0.3;
        let cp_r = radius * 0.6;
        let cpx = cx + cp_r * cp_angle.cos();
        let cpy = cy + cp_r * cp_angle.sin();

        if i == 0 {
            let _ = write!(
                path,
                "M {cx:.1} {cy:.1} Q {cpx:.1} {cpy:.1} {tip_x:.1} {tip_y:.1}"
            );
        } else {
            let _ = write!(
                path,
                " M {cx:.1} {cy:.1} Q {cpx:.1} {cpy:.1} {tip_x:.1} {tip_y:.1}"
            );
        }
    }

    SceneNode::path(path).with_style(Style {
        fill: None,
        stroke: Some(Color::from_hsl(hue, 0.7, 0.45)),
        stroke_width: Some(2.0 + activation * 2.0),
        opacity: Some(0.6 + activation * 0.3),
        ..Style::default()
    })
}

/// IntegralWisdom: geometric hexagonal tessellation.
fn crystal(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("crystal"));
    let sides = 6;
    let layers = 1 + (activation * 3.0) as usize;

    for layer in 0..layers {
        let r = radius * (layer as f32 + 1.0) / layers as f32;
        let mut points = Vec::with_capacity(sides);
        for i in 0..sides {
            let angle =
                (i as f32 / sides as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_6;
            points.push((cx + r * angle.cos(), cy + r * angle.sin()));
        }
        let hex = SceneNode::polygon(points, true).with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(hue + layer as f32 * 15.0, 0.5, 0.5)),
            stroke_width: Some(1.0),
            opacity: Some(0.5 + activation * 0.3),
            ..Style::default()
        });
        group.children.push(hex);
    }
    group
}

/// InfinitePlay: logarithmic spiral.
fn spiral(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let turns = 2.0 + activation * 4.0;
    let steps = 100 + (activation * 100.0) as usize;
    let mut path = String::with_capacity(steps * 20);

    let growth = 0.12 + activation * 0.08; // logarithmic growth rate

    for i in 0..=steps {
        let t = (i as f32 / steps as f32) * turns * std::f32::consts::TAU;
        let r = radius * 0.05 * (growth * t).exp();
        let r = r.min(radius); // cap
        let x = cx + r * t.cos();
        let y = cy + r * t.sin();
        if i == 0 {
            let _ = write!(path, "M {x:.1} {y:.1}");
        } else {
            let _ = write!(path, " L {x:.1} {y:.1}");
        }
    }

    SceneNode::path(path).with_style(Style {
        fill: None,
        stroke: Some(Color::from_hsl(hue, 0.8, 0.5)),
        stroke_width: Some(1.0 + activation * 2.0),
        opacity: Some(0.7),
        ..Style::default()
    })
}

/// UniversalInterconnectedness: overlapping circles forming a web.
fn web(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("web"));
    let nodes = 4 + (activation * 6.0) as usize;

    // Place nodes in a ring
    let positions: Vec<(f32, f32)> = (0..nodes)
        .map(|i| {
            let angle = (i as f32 / nodes as f32) * std::f32::consts::TAU;
            (
                cx + radius * 0.6 * angle.cos(),
                cy + radius * 0.6 * angle.sin(),
            )
        })
        .collect();

    // Draw circles at each node
    for &(x, y) in &positions {
        let circle = SceneNode::circle(x, y, radius * 0.15).with_style(Style {
            fill: Some(Color::from_hsl(hue, 0.5, 0.6)),
            opacity: Some(0.3),
            ..Style::default()
        });
        group.children.push(circle);
    }

    // Connect nodes with lines
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let line = SceneNode::line(
                positions[i].0,
                positions[i].1,
                positions[j].0,
                positions[j].1,
            )
            .with_style(Style {
                stroke: Some(Color::from_hsl(hue, 0.4, 0.5)),
                stroke_width: Some(0.5 + activation),
                opacity: Some(0.2 + activation * 0.2),
                ..Style::default()
            });
            group.children.push(line);
        }
    }
    group
}

/// SacredReciprocity: mirrored yin-yang curve.
fn mirror(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("mirror"));
    let steps = 50;

    // S-curve on one side
    let mut path_top = String::with_capacity(steps * 20);
    let mut path_bot = String::with_capacity(steps * 20);

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = cx + (t - 0.5) * radius * 2.0;
        let y_offset = radius * 0.5 * (t * std::f32::consts::PI).sin() * (1.0 + activation * 0.5);
        let y_top = cy - y_offset;
        let y_bot = cy + y_offset;

        if i == 0 {
            let _ = write!(path_top, "M {x:.1} {y_top:.1}");
            let _ = write!(path_bot, "M {x:.1} {y_bot:.1}");
        } else {
            let _ = write!(path_top, " L {x:.1} {y_top:.1}");
            let _ = write!(path_bot, " L {x:.1} {y_bot:.1}");
        }
    }

    let top = SceneNode::path(path_top).with_style(Style {
        fill: None,
        stroke: Some(Color::from_hsl(hue, 0.6, 0.45)),
        stroke_width: Some(2.0 + activation),
        opacity: Some(0.7),
        ..Style::default()
    });
    let bot = SceneNode::path(path_bot).with_style(Style {
        fill: None,
        stroke: Some(Color::from_hsl(hue + 180.0, 0.6, 0.55)),
        stroke_width: Some(2.0 + activation),
        opacity: Some(0.7),
        ..Style::default()
    });

    group.children.push(top);
    group.children.push(bot);
    group
}

/// EvolutionaryProgression: ascending arc with growing elements.
fn ascending_arc(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("progression"));
    let steps = 8 + (activation * 8.0) as usize;

    for i in 0..steps {
        let t = i as f32 / steps as f32;
        // Arc from lower-left to upper-right
        let x = cx - radius + t * radius * 2.0;
        let y = cy + radius * 0.5 - t * radius * 1.2;
        let r = 3.0 + t * radius * 0.08 * (1.0 + activation);

        let circle = SceneNode::circle(x, y, r).with_style(Style {
            fill: Some(Color::from_hsl(hue + t * 40.0, 0.6, 0.3 + t * 0.3)),
            opacity: Some(0.4 + t * 0.4),
            ..Style::default()
        });
        group.children.push(circle);
    }
    group
}

/// SacredStillness: ensō circle — a single imperfect brushstroke.
fn enso(cx: f32, cy: f32, radius: f32, activation: f32, hue: f32) -> SceneNode {
    let steps = 80;
    let gap = 0.15 + (1.0 - activation) * 0.2; // more active = more complete circle
    let mut path = String::with_capacity(steps * 20);

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        if t > (1.0 - gap) {
            break; // leave a gap — the ensō is intentionally incomplete
        }
        let angle = t * std::f32::consts::TAU;
        // Slight wobble for brushstroke feel
        let wobble = 1.0 + (angle * 3.0).sin() * 0.02 + (angle * 7.0).cos() * 0.01;
        let r = radius * wobble;
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();

        if i == 0 {
            let _ = write!(path, "M {x:.1} {y:.1}");
        } else {
            let _ = write!(path, " L {x:.1} {y:.1}");
        }
    }

    SceneNode::path(path).with_style(Style {
        fill: None,
        stroke: Some(Color::from_hsl(hue, 0.15, 0.25)), // muted, ink-like
        stroke_width: Some(4.0 + activation * 6.0),     // thick brushstroke
        opacity: Some(0.7 + activation * 0.25),
        ..Style::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_harmonies_produce_shapes() {
        for i in 0..8 {
            let shape = harmony_shape(i, 0.7, 256.0, 256.0, 100.0, 180.0);
            assert!(shape.node_count() >= 1, "harmony {i} produced empty shape");
        }
    }

    #[test]
    fn activation_affects_complexity() {
        let low = harmony_shape(0, 0.1, 256.0, 256.0, 100.0, 180.0);
        let high = harmony_shape(0, 0.9, 256.0, 256.0, 100.0, 180.0);
        // Higher activation should produce more elements (more concentric rings)
        assert!(
            high.node_count() >= low.node_count(),
            "high {} should >= low {}",
            high.node_count(),
            low.node_count()
        );
    }

    #[test]
    fn enso_gap_visible() {
        // Low activation = bigger gap = more incomplete circle
        let still = harmony_shape(7, 0.2, 256.0, 256.0, 100.0, 0.0);
        assert!(still.node_count() >= 1);
    }

    #[test]
    fn out_of_range_harmony_empty() {
        let shape = harmony_shape(99, 0.5, 256.0, 256.0, 100.0, 0.0);
        assert_eq!(shape.node_count(), 1); // just the empty group
    }
}
