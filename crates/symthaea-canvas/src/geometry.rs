// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AestheticState → SceneNode tree: topology-driven visual geometry.
//!
//! Layers (back to front):
//! 1. Background field (radial gradient)
//! 2. Harmonic octagon (8-spoke mandala)
//! 3. Topological masses (β₀ connected components)
//! 4. Orbital rings (β₁ loops)
//! 5. Void portals (β₂ voids)
//! 6. Fractal detail (recursive subdivision)
//! 7. Turbulence overlay (prediction error distortion)
//! 8. Center bloom (consciousness beacon)
//! 9. Energy particles (arousal-driven)

use std::f32::consts::PI;

use crate::aesthetic_engine::AestheticState;
use crate::animation::layer_opacity;
use crate::color::Color;
use crate::scene_graph::{FilterType, GradientStop, NodeKind, SceneNode, Style, Transform};
use crate::svg_renderer::{VIEWPORT_H, VIEWPORT_W};

/// Build the complete scene graph from an aesthetic state.
pub fn build_scene(state: &AestheticState) -> SceneNode {
    let psi = state.luminosity as f64;
    let mut root = SceneNode::group(Some("canvas"));

    // Layer 1: Background (always present)
    root.children.push(build_background(state));

    // Layer 7: Turbulence filter definition (threshold 0.1)
    let turb_opacity = layer_opacity(psi, 0.1, 0.15);
    if turb_opacity > 0.0 {
        root.children.push(build_turbulence_filter(state));
    }

    // Layer 2: Harmonic octagon (threshold 0.15)
    let oct_opacity = layer_opacity(psi, 0.15, 0.15);
    if oct_opacity > 0.0 {
        root.children
            .push(build_harmonic_octagon(state, oct_opacity));
    }

    // Layer 3: Topological masses (threshold 0.2)
    let mass_opacity = layer_opacity(psi, 0.2, 0.15);
    if mass_opacity > 0.0 && state.component_count > 0 {
        root.children
            .push(build_topological_masses(state, mass_opacity));
    }

    // Layer 4: Orbital rings (threshold 0.3)
    let ring_opacity = layer_opacity(psi, 0.3, 0.15);
    if ring_opacity > 0.0 && state.ring_count > 0 {
        root.children.push(build_orbital_rings(state, ring_opacity));
    }

    // Layer 4.5: Persistence arcs (threshold 0.35)
    let persist_opacity = layer_opacity(psi, 0.35, 0.15);
    if persist_opacity > 0.0
        && (!state.persistence_components.is_empty() || !state.persistence_cycles.is_empty())
    {
        root.children
            .push(build_persistence_arcs(state, persist_opacity));
    }

    // Layer 5: Void portals (threshold 0.4)
    let void_opacity = layer_opacity(psi, 0.4, 0.15);
    if void_opacity > 0.0 && state.void_count > 0 {
        root.children.push(build_void_portals(state, void_opacity));
    }

    // Layer 6: Fractal detail (threshold 0.5)
    let frac_opacity = layer_opacity(psi, 0.5, 0.2);
    if frac_opacity > 0.0 && state.fractal_depth > 0 {
        root.children
            .push(build_fractal_detail(state, frac_opacity));
    }

    // Layer 8: Center bloom (always present — warm amber at low Ψ, gold at high)
    root.children.push(build_center_bloom(state));

    // Layer 9: Energy particles (threshold 0.3 arousal)
    if state.energy > 0.3 {
        let particle_opacity = layer_opacity(psi, 0.3, 0.2);
        if particle_opacity > 0.0 {
            root.children
                .push(build_energy_particles(state, particle_opacity));
        }
    }

    root
}

/// Layer 1: Dark background with consciousness-responsive radial gradient.
fn build_background(state: &AestheticState) -> SceneNode {
    let psi = state.luminosity;
    // HSL(220, 15%, 5% + Ψ*15%)
    let bg_color = Color::from_hsl(220.0, 0.15, 0.05 + psi * 0.15);
    let bg_center = Color::from_hsl(220.0, 0.2, 0.08 + psi * 0.2);

    let mut group = SceneNode::group(Some("background"));

    // Gradient definition
    group.children.push(SceneNode {
        kind: NodeKind::RadialGradient {
            id: "bg-grad".to_string(),
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: bg_center,
                },
                GradientStop {
                    offset: 1.0,
                    color: bg_color,
                },
            ],
        },
        transform: Transform::identity(),
        style: Style::default(),
        children: Vec::new(),
    });

    // Full-viewport rect with gradient fill
    group.children.push(
        SceneNode::rect(0.0, 0.0, VIEWPORT_W, VIEWPORT_H).with_style(Style {
            fill_url: Some("bg-grad".to_string()),
            ..Style::default()
        }),
    );

    group
}

/// Layer 2: 8-spoke mandala with harmony-driven radii.
fn build_harmonic_octagon(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("harmonics"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let base_radius = 120.0;

    // Build polygon from 8 spoke endpoints
    let mut points = Vec::with_capacity(8);
    for i in 0..8 {
        let angle = (i as f32) * PI / 4.0 - PI / 2.0; // start from top
        let r = base_radius * state.harmony_radii[i];
        points.push((cx + r * angle.cos(), cy + r * angle.sin()));
    }

    let poly = SceneNode::polygon(points.clone(), true).with_style(Style {
        fill: Some(state.palette.primary.with_alpha(0.15)),
        stroke: Some(state.palette.primary.with_alpha(0.6)),
        stroke_width: Some(1.5),
        ..Style::default()
    });
    group.children.push(poly);

    // Spoke lines from center to each vertex
    for (px, py) in &points {
        let spoke = SceneNode::line(cx, cy, *px, *py).with_style(Style {
            stroke: Some(state.palette.primary.with_alpha(0.2)),
            stroke_width: Some(0.5),
            ..Style::default()
        });
        group.children.push(spoke);
    }

    group
}

/// Layer 3: Topological masses (β₀ connected components).
fn build_topological_masses(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("masses"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let count = state.component_count.min(8); // cap at 8 for visual sanity
    let phase = state.cycle_phase as f32;

    for i in 0..count {
        let angle = (i as f32 / count as f32) * 2.0 * PI + phase * 0.5;
        let dist = 60.0 + (i as f32) * 15.0;
        let x = cx + dist * angle.cos();
        let y = cy + dist * angle.sin();
        let r = 8.0 + state.luminosity * 12.0;

        let mass = SceneNode::circle(x, y, r).with_style(Style {
            fill: Some(state.palette.secondary.with_alpha(0.4)),
            stroke: Some(state.palette.secondary.with_alpha(0.7)),
            stroke_width: Some(1.0),
            ..Style::default()
        });
        group.children.push(mass);
    }

    group
}

/// Layer 4: Orbital rings (β₁ loops).
fn build_orbital_rings(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("rings"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let count = state.ring_count.min(5);

    for i in 0..count {
        let rx = 50.0 + (i as f32) * 30.0;
        let ry = rx * 0.6; // elliptical
        let rotation = (i as f32) * 30.0 + state.cycle_phase as f32 * 60.0;

        let ring = SceneNode::ellipse(cx, cy, rx, ry)
            .with_style(Style {
                fill: None,
                stroke: Some(state.palette.accent.with_alpha(0.4)),
                stroke_width: Some(1.0),
                ..Style::default()
            })
            .with_transform(Transform {
                translate_x: 0.0,
                translate_y: 0.0,
                rotate_deg: rotation,
                scale: 1.0,
            });
        group.children.push(ring);
    }

    group
}

/// Layer 4.5: Persistence diagram arcs — birth-death pairs as radial arcs.
///
/// Each persistence pair [birth, death] becomes an arc whose inner radius = birth
/// and outer radius = death (scaled to viewport). Long-lived features (large death-birth)
/// are more prominent. Components use the secondary color, cycles use the accent.
fn build_persistence_arcs(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("persistence"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let max_radius = 100.0;

    // Components: secondary color, distributed around the top half
    let comp_count = state.persistence_components.len().min(8);
    for (i, pair) in state.persistence_components.iter().take(8).enumerate() {
        let birth = pair[0] as f32;
        let death = pair[1] as f32;
        let lifetime = (death - birth).max(0.01);
        let r_inner = 30.0 + birth * max_radius;
        let r_outer = 30.0 + death * max_radius;
        let angle_start = -PI / 2.0 + (i as f32 / comp_count.max(1) as f32) * PI;
        let arc_sweep = (lifetime * PI).min(PI / 3.0);

        // Draw as a pair of concentric arcs connected at ends
        let mid_r = (r_inner + r_outer) / 2.0;
        let width = (r_outer - r_inner).max(1.0);

        let x1 = cx + mid_r * angle_start.cos();
        let y1 = cy + mid_r * angle_start.sin();
        let x2 = cx + mid_r * (angle_start + arc_sweep).cos();
        let y2 = cy + mid_r * (angle_start + arc_sweep).sin();

        let arc = SceneNode::line(x1, y1, x2, y2).with_style(Style {
            stroke: Some(state.palette.secondary.with_alpha(0.3 + lifetime * 0.4)),
            stroke_width: Some(width.min(6.0)),
            ..Style::default()
        });
        group.children.push(arc);
    }

    // Cycles: accent color, bottom half
    let cycle_count = state.persistence_cycles.len().min(5);
    for (i, pair) in state.persistence_cycles.iter().take(5).enumerate() {
        let birth = pair[0] as f32;
        let death = pair[1] as f32;
        let lifetime = (death - birth).max(0.01);
        let r = 30.0 + ((birth + death) / 2.0) * max_radius;
        let angle = PI / 2.0 + (i as f32 / cycle_count.max(1) as f32) * PI;

        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        let size = 3.0 + lifetime * 8.0;

        // Cycle features as small rings (echoing β₁ semantics)
        let ring = SceneNode::circle(x, y, size).with_style(Style {
            fill: None,
            stroke: Some(state.palette.accent.with_alpha(0.3 + lifetime * 0.5)),
            stroke_width: Some(1.5),
            ..Style::default()
        });
        group.children.push(ring);
    }

    group
}

/// Layer 5: Void portals (β₂ — dark circles with luminous rims).
fn build_void_portals(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("voids"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let count = state.void_count.min(3);

    for i in 0..count {
        let angle = (i as f32 / count.max(1) as f32) * 2.0 * PI;
        let dist = 80.0;
        let x = cx + dist * angle.cos();
        let y = cy + dist * angle.sin();

        // Dark interior
        let void_inner = SceneNode::circle(x, y, 15.0).with_style(Style {
            fill: Some(Color::rgba(0.0, 0.0, 0.05, 0.9)),
            ..Style::default()
        });
        group.children.push(void_inner);

        // Luminous rim
        let void_rim = SceneNode::circle(x, y, 18.0).with_style(Style {
            fill: None,
            stroke: Some(state.palette.accent.with_alpha(0.6)),
            stroke_width: Some(2.0),
            ..Style::default()
        });
        group.children.push(void_rim);
    }

    group
}

/// Layer 6: Fractal detail — recursive subdivision of harmonic spokes.
fn build_fractal_detail(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("fractals"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let depth = state.fractal_depth.min(5) as usize;
    let base_r = 120.0;

    // Fractal on each spoke
    for i in 0..8 {
        let angle = (i as f32) * PI / 4.0 - PI / 2.0;
        let r = base_r * state.harmony_radii[i];
        let end_x = cx + r * angle.cos();
        let end_y = cy + r * angle.sin();
        fractal_branch(
            &mut group,
            end_x,
            end_y,
            angle,
            r * 0.3,
            depth,
            &state.palette.primary.with_alpha(0.3 * opacity),
        );
    }

    group
}

fn fractal_branch(
    parent: &mut SceneNode,
    x: f32,
    y: f32,
    angle: f32,
    length: f32,
    depth: usize,
    color: &Color,
) {
    if depth == 0 || length < 2.0 {
        return;
    }

    let spread = PI / 6.0;
    for &offset in &[-spread, spread] {
        let a = angle + offset;
        let ex = x + length * a.cos();
        let ey = y + length * a.sin();

        let branch = SceneNode::line(x, y, ex, ey).with_style(Style {
            stroke: Some(*color),
            stroke_width: Some((depth as f32) * 0.3),
            ..Style::default()
        });
        parent.children.push(branch);

        fractal_branch(parent, ex, ey, a, length * 0.6, depth - 1, color);
    }
}

/// Layer 7: Turbulence SVG filter.
fn build_turbulence_filter(state: &AestheticState) -> SceneNode {
    SceneNode {
        kind: NodeKind::Filter {
            id: "turb-filter".to_string(),
            filter_type: FilterType::Turbulence {
                base_frequency: 0.01 + state.turbulence * 0.04,
                num_octaves: 2,
                scale: state.turbulence * 15.0,
            },
        },
        transform: Transform::identity(),
        style: Style::default(),
        children: Vec::new(),
    }
}

/// Layer 8: Center bloom — consciousness beacon.
fn build_center_bloom(state: &AestheticState) -> SceneNode {
    let mut group = SceneNode::group(Some("bloom"));
    let (cx, cy) = state.layout_center;
    let psi = state.luminosity;

    // Breathing modulation
    let breath = state.cycle_phase as f32;
    let breath_scale = 0.9 + breath * 0.2; // 0.9 to 1.1

    // Color: gold at high Ψ, warm amber at low Ψ
    let bloom_color = if psi > 0.5 {
        // Gold: rgba(232, 197, 71, Ψ*0.5)
        Color::rgba(232.0 / 255.0, 197.0 / 255.0, 71.0 / 255.0, psi * 0.5)
    } else {
        // Warm amber: rgba(196, 149, 106, 0.15)
        Color::rgba(
            196.0 / 255.0,
            149.0 / 255.0,
            106.0 / 255.0,
            0.15 + psi * 0.2,
        )
    };

    // Outer glow
    let glow_r = 40.0 * breath_scale;
    let glow = SceneNode::circle(cx, cy, glow_r).with_style(Style {
        fill: Some(bloom_color.with_alpha(bloom_color.a * 0.4)),
        ..Style::default()
    });
    group.children.push(glow);

    // Inner core
    let core_r = 12.0 * breath_scale;
    let core = SceneNode::circle(cx, cy, core_r).with_style(Style {
        fill: Some(bloom_color),
        ..Style::default()
    });
    group.children.push(core);

    group
}

/// Layer 9: Energy particles — orbiting dots when arousal > 0.3.
fn build_energy_particles(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("particles"));
    group.style.opacity = Some(opacity);

    let (cx, cy) = state.layout_center;
    let count = ((state.energy - 0.3) * 15.0).round() as usize; // 0-10 particles
    let count = count.min(10);
    let phase = state.cycle_phase as f32;

    for i in 0..count {
        let angle = (i as f32 / count.max(1) as f32) * 2.0 * PI + phase * 4.0;
        let dist = 90.0 + (i as f32) * 8.0;
        let x = cx + dist * angle.cos();
        let y = cy + dist * angle.sin();

        let particle = SceneNode::circle(x, y, 2.0).with_style(Style {
            fill: Some(state.palette.accent.with_alpha(0.6)),
            ..Style::default()
        });
        group.children.push(particle);
    }

    group
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Palette;

    fn make_state(psi: f32) -> AestheticState {
        AestheticState {
            luminosity: psi,
            complexity: 0.5,
            turbulence: 0.2,
            energy: 0.5,
            warmth: 0.5,
            fractal_depth: 3,
            palette: Palette::default(),
            component_count: 3,
            ring_count: 2,
            void_count: 1,
            harmony_radii: [0.6; 8],
            layout_center: (256.0, 256.0),
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            cycle_phase: 0.5,
            aesthetic_score: 0.0,
        }
    }

    #[test]
    fn low_psi_minimal_scene() {
        let state = make_state(0.05);
        let scene = build_scene(&state);
        // Should have background + bloom only (no octagon, no masses, etc.)
        let child_ids: Vec<_> = scene
            .children
            .iter()
            .filter_map(|c| {
                if let NodeKind::Group { id } = &c.kind {
                    id.clone()
                } else {
                    None
                }
            })
            .collect();
        assert!(child_ids.contains(&"background".to_string()));
        assert!(child_ids.contains(&"bloom".to_string()));
        assert!(!child_ids.contains(&"harmonics".to_string()));
        assert!(!child_ids.contains(&"masses".to_string()));
    }

    #[test]
    fn high_psi_full_scene() {
        let state = make_state(0.9);
        let scene = build_scene(&state);
        let child_ids: Vec<_> = scene
            .children
            .iter()
            .filter_map(|c| {
                if let NodeKind::Group { id } = &c.kind {
                    id.clone()
                } else {
                    None
                }
            })
            .collect();
        assert!(child_ids.contains(&"harmonics".to_string()));
        assert!(child_ids.contains(&"masses".to_string()));
        assert!(child_ids.contains(&"rings".to_string()));
        assert!(child_ids.contains(&"voids".to_string()));
        assert!(child_ids.contains(&"bloom".to_string()));
    }

    #[test]
    fn betti_0_affects_mass_count() {
        let mut state = make_state(0.8);
        state.component_count = 5;
        let scene = build_scene(&state);
        let masses = scene
            .children
            .iter()
            .find(|c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("masses")));
        assert!(masses.is_some());
        assert_eq!(masses.unwrap().children.len(), 5);
    }

    #[test]
    fn betti_1_affects_ring_count() {
        let mut state = make_state(0.8);
        state.ring_count = 3;
        let scene = build_scene(&state);
        let rings = scene
            .children
            .iter()
            .find(|c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("rings")));
        assert!(rings.is_some());
        assert_eq!(rings.unwrap().children.len(), 3);
    }

    #[test]
    fn zero_betti_no_masses() {
        let mut state = make_state(0.8);
        state.component_count = 0;
        let scene = build_scene(&state);
        let has_masses = scene
            .children
            .iter()
            .any(|c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("masses")));
        assert!(!has_masses);
    }

    #[test]
    fn particles_only_at_high_arousal() {
        let mut state = make_state(0.8);
        state.energy = 0.1;
        let scene = build_scene(&state);
        let has_particles = scene.children.iter().any(
            |c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("particles")),
        );
        assert!(!has_particles);

        state.energy = 0.7;
        let scene2 = build_scene(&state);
        let has_particles2 = scene2.children.iter().any(
            |c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("particles")),
        );
        assert!(has_particles2);
    }

    #[test]
    fn node_count_reasonable() {
        let state = make_state(0.9);
        let scene = build_scene(&state);
        let count = scene.node_count();
        // Should be meaningful but not excessive
        assert!(count > 10, "too few nodes: {count}");
        assert!(count < 500, "too many nodes: {count}");
    }
}
