// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistence-diagram artwork from the snapshot's real topological data.
//!
//! A persistence diagram plots each topological feature as a point
//! (birth, death) above the diagonal y = x; distance from the diagonal is
//! the feature's lifetime (persistence). This module renders the snapshot's
//! genuine `persistence_components` (H0) and `persistence_cycles` (H1)
//! birth/death pairs as a proper diagram — diagonal line, axes, points sized
//! and colored by lifetime — with a barcode band below (each pair as a
//! horizontal [birth, death] bar).
//!
//! **Empty-diagram behavior (documented degradation)**: the live cognitive
//! loop currently delivers real Betti numbers but *empty* persistence
//! diagrams. When both diagrams are empty this module renders the
//! empty-diagram aesthetic (axes + diagonal + infinity line) and, if Betti
//! counts are nonzero, marks the *essential classes* they represent on the
//! infinity line: Betti numbers count features that never die, which is the
//! standard "death = ∞" row of a persistence diagram. Their **x positions
//! (births) are a placeholder layout** (evenly spaced — birth times are not
//! known from Betti counts alone); their deaths (∞) and their count are the
//! real data. No finite birth/death pairs are ever invented.
//!
//! # References
//! - Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological
//!   persistence and simplification. *Discrete Comput. Geom.*, 28, 511–533.
//! - Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
//!   Introduction*. AMS.

use rand::Rng;
use rand::rngs::StdRng;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};

use crate::AtelierConfig;

/// Plot frame in viewport-relative coordinates.
struct Frame {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    /// y of the "death = ∞" line (essential classes).
    infinity_y: f32,
    /// Barcode band vertical extent.
    band_top: f32,
    band_bottom: f32,
}

impl Frame {
    fn new(config: &AtelierConfig) -> Self {
        let w = config.width;
        let h = config.height;
        Self {
            left: 0.12 * w,
            right: 0.90 * w,
            top: 0.14 * h,
            bottom: 0.68 * h,
            infinity_y: 0.09 * h,
            band_top: 0.76 * h,
            band_bottom: 0.94 * h,
        }
    }

    /// Map a filtration value in [0, vmax] to plot x.
    fn x(&self, v: f32, vmax: f32) -> f32 {
        self.left + (v / vmax).clamp(0.0, 1.0) * (self.right - self.left)
    }

    /// Map a filtration value in [0, vmax] to plot y (larger = higher).
    fn y(&self, v: f32, vmax: f32) -> f32 {
        self.bottom - (v / vmax).clamp(0.0, 1.0) * (self.bottom - self.top)
    }
}

fn frame_line(x1: f32, y1: f32, x2: f32, y2: f32, color: Color, width: f32) -> SceneNode {
    SceneNode::line(x1, y1, x2, y2).with_style(Style {
        stroke: Some(color),
        stroke_width: Some(width),
        opacity: Some(0.8),
        ..Style::default()
    })
}

/// Generate a persistence-diagram artwork from the snapshot's real
/// birth/death pairs (see module docs for the empty-diagram degradation).
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let frame = Frame::new(config);
    let consciousness = snapshot.consciousness_level as f32;

    // Palette: neuromodulators set the hue bases; RNG adds an aesthetic
    // offset only — positions and sizes come from the topological data.
    let hue_jitter = rng.r#gen::<f32>() * 24.0;
    let cool_hue = 180.0 + snapshot.serotonin * 60.0 + hue_jitter; // H0
    let warm_hue = 20.0 + snapshot.dopamine * 50.0 + hue_jitter; // H1
    let axis_color = Color::from_hsl(220.0, 0.15, 0.55);

    // Normalization: scale by the largest finite death/birth seen, floored
    // at 1.0 so canonical [0,1]-valued diagrams keep their true proportions.
    let vmax = snapshot
        .persistence_components
        .iter()
        .chain(snapshot.persistence_cycles.iter())
        .flat_map(|p| p.iter().copied())
        .fold(1.0f64, f64::max) as f32;

    let bg = SceneNode::rect(0.0, 0.0, config.width, config.height).with_style(Style {
        fill: Some(Color::from_hsla(230.0, 0.22, 0.07, 1.0)),
        ..Style::default()
    });

    let mut root = SceneNode::group(Some("persistence-diagram")).with_child(bg);

    // Axes: birth (x) and death (y).
    root.children.push(frame_line(
        frame.left,
        frame.bottom,
        frame.right,
        frame.bottom,
        axis_color,
        1.5,
    ));
    root.children.push(frame_line(
        frame.left,
        frame.bottom,
        frame.left,
        frame.top,
        axis_color,
        1.5,
    ));

    // The diagonal y = x: features are born and die at equal filtration
    // values here; everything meaningful lives above it.
    root.children.push(frame_line(
        frame.left,
        frame.bottom,
        frame.right,
        frame.top,
        Color::from_hsl(220.0, 0.20, 0.40),
        1.0,
    ));

    // Infinity line (death = ∞ row for essential classes).
    root.children.push(
        SceneNode::line(frame.left, frame.infinity_y, frame.right, frame.infinity_y).with_style(
            Style {
                stroke: Some(Color::from_hsl(220.0, 0.15, 0.35)),
                stroke_width: Some(0.8),
                opacity: Some(0.6),
                ..Style::default()
            },
        ),
    );

    // ── Finite pairs: the real diagrams ─────────────────────────────────
    // H0 components: filled circles, cool hue, sized by lifetime.
    for pair in &snapshot.persistence_components {
        let (birth, death) = (pair[0] as f32, pair[1] as f32);
        let lifetime = (death - birth).abs() / vmax;
        let dot = SceneNode::circle(
            frame.x(birth, vmax),
            frame.y(death, vmax),
            3.0 + lifetime * 14.0,
        )
        .with_style(Style {
            fill: Some(Color::from_hsl(
                (cool_hue + lifetime * 40.0) % 360.0,
                0.75,
                0.45 + lifetime * 0.25,
            )),
            opacity: Some((0.5 + lifetime * 0.4).min(0.9)),
            ..Style::default()
        });
        root.children.push(dot);
    }

    // H1 cycles: rings, warm hue, sized by lifetime.
    for pair in &snapshot.persistence_cycles {
        let (birth, death) = (pair[0] as f32, pair[1] as f32);
        let lifetime = (death - birth).abs() / vmax;
        let ring = SceneNode::circle(
            frame.x(birth, vmax),
            frame.y(death, vmax),
            4.0 + lifetime * 14.0,
        )
        .with_style(Style {
            fill: None,
            stroke: Some(Color::from_hsl(
                (warm_hue + lifetime * 30.0) % 360.0,
                0.80,
                0.55,
            )),
            stroke_width: Some(1.5 + lifetime * 3.0),
            opacity: Some((0.55 + lifetime * 0.35).min(0.9)),
            ..Style::default()
        });
        root.children.push(ring);
    }

    // ── Essential classes on the ∞ line (Betti counts) ──────────────────
    // Only drawn when the finite diagrams are empty (the live-loop case):
    // count and death (∞) are real; the x layout is a documented
    // placeholder (evenly spaced — births unknown from Betti counts).
    if snapshot.persistence_components.is_empty() && snapshot.persistence_cycles.is_empty() {
        let total = snapshot.betti_0 + snapshot.betti_1 + snapshot.betti_2;
        if total > 0 {
            let span = frame.right - frame.left;
            let mut slot = 0usize;
            let mut next_x = |slot: &mut usize| {
                let x = frame.left + span * (*slot as f32 + 1.0) / (total as f32 + 1.0);
                *slot += 1;
                x
            };
            for _ in 0..snapshot.betti_0 {
                root.children.push(
                    SceneNode::circle(next_x(&mut slot), frame.infinity_y, 6.0).with_style(Style {
                        fill: Some(Color::from_hsl(cool_hue % 360.0, 0.7, 0.55)),
                        opacity: Some(0.85),
                        ..Style::default()
                    }),
                );
            }
            for _ in 0..snapshot.betti_1 {
                root.children.push(
                    SceneNode::circle(next_x(&mut slot), frame.infinity_y, 7.0).with_style(Style {
                        fill: None,
                        stroke: Some(Color::from_hsl(warm_hue % 360.0, 0.75, 0.55)),
                        stroke_width: Some(2.0),
                        opacity: Some(0.85),
                        ..Style::default()
                    }),
                );
            }
            for _ in 0..snapshot.betti_2 {
                let x = next_x(&mut slot);
                root.children.push(
                    SceneNode::rect(x - 5.0, frame.infinity_y - 5.0, 10.0, 10.0).with_style(
                        Style {
                            fill: Some(Color::from_hsl(290.0 + hue_jitter, 0.6, 0.55)),
                            opacity: Some(0.85),
                            ..Style::default()
                        },
                    ),
                );
            }
        }
    }

    // ── Barcode band: each finite pair as a [birth, death] bar ──────────
    let bars: Vec<(f32, f32, bool)> = snapshot
        .persistence_components
        .iter()
        .map(|p| (p[0] as f32, p[1] as f32, true))
        .chain(
            snapshot
                .persistence_cycles
                .iter()
                .map(|p| (p[0] as f32, p[1] as f32, false)),
        )
        .collect();
    if !bars.is_empty() {
        let band_h = frame.band_bottom - frame.band_top;
        let row_h = (band_h / bars.len() as f32).min(0.04 * config.height);
        for (i, &(birth, death, is_component)) in bars.iter().enumerate() {
            let lifetime = (death - birth).abs() / vmax;
            let y = frame.band_top + (i as f32 + 0.5) * row_h;
            let hue = if is_component { cool_hue } else { warm_hue };
            root.children.push(
                SceneNode::line(frame.x(birth, vmax), y, frame.x(death, vmax), y).with_style(
                    Style {
                        stroke: Some(Color::from_hsl(hue % 360.0, 0.7, 0.5 + lifetime * 0.2)),
                        stroke_width: Some((row_h * 0.6).clamp(1.5, 8.0)),
                        opacity: Some(0.55 + consciousness * 0.3),
                        ..Style::default()
                    },
                ),
            );
        }
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rich_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            dopamine: 0.6,
            serotonin: 0.5,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8], [0.3, 0.9]],
            persistence_cycles: vec![[0.2, 0.6], [0.4, 0.7]],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn rich_diagram_renders_pairs_and_barcode() {
        let config = AtelierConfig::default();
        let snapshot = rich_snapshot();
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // bg + 2 axes + diagonal + ∞ line + 5 diagram points + 5 barcode
        // bars = 15 children (no essential markers: finite pairs present).
        assert_eq!(scene.children.len(), 15);
    }

    #[test]
    fn empty_diagram_degrades_to_axes_plus_betti_markers() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            betti_0: 3,
            betti_1: 2,
            betti_2: 1,
            persistence_components: vec![],
            persistence_cycles: vec![],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // bg + 2 axes + diagonal + ∞ line + (3+2+1) essential markers = 11.
        assert_eq!(scene.children.len(), 11);
    }

    #[test]
    fn fully_empty_snapshot_still_produces_scene() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            betti_0: 0,
            betti_1: 0,
            betti_2: 0,
            persistence_components: vec![],
            persistence_cycles: vec![],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Empty-diagram aesthetic: bg + axes + diagonal + ∞ line.
        assert!(scene.node_count() > 1);
        assert!(scene.children.len() >= 5);
    }

    #[test]
    fn generate_deterministic_same_seed() {
        let config = AtelierConfig::default();
        let snapshot = rich_snapshot();
        let mut rng1 = StdRng::seed_from_u64(77);
        let mut rng2 = StdRng::seed_from_u64(77);
        let s1 = generate(&config, &snapshot, &mut rng1);
        let s2 = generate(&config, &snapshot, &mut rng2);
        let svg1 = symthaea_canvas::render_svg(&s1, snapshot.consciousness_level);
        let svg2 = symthaea_canvas::render_svg(&s2, snapshot.consciousness_level);
        assert_eq!(svg1, svg2);
    }
}
