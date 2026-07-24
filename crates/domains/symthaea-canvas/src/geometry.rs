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

    // Semantic field layers: complexity, attention, and physiological constraint.
    let semantic_opacity = layer_opacity(psi, 0.15, 0.2);
    if semantic_opacity > 0.0 && state.complexity > 0.05 {
        root.children
            .push(build_complexity_contours(state, semantic_opacity));
    }
    if semantic_opacity > 0.0 && state.attention > 0.05 {
        root.children
            .push(build_attention_halo(state, semantic_opacity));
    }
    if semantic_opacity > 0.0 && state.allostatic_load > 0.05 {
        root.children
            .push(build_allostatic_boundary(state, semantic_opacity));
    }

    // Layer 7: Turbulence filter definition (threshold 0.1)
    let turb_opacity = layer_opacity(psi, 0.1, 0.15);
    if turb_opacity > 0.0 && state.turbulence > 0.01 {
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
    // Valence warmth shifts the field from indigo/cobalt toward violet/amber.
    let field_hue = 240.0 - state.warmth * 75.0;
    let center_hue = 230.0 - state.warmth * 105.0;
    let bg_color = Color::from_hsl(field_hue, 0.15 + state.warmth * 0.05, 0.05 + psi * 0.15);
    let bg_center = Color::from_hsl(center_hue, 0.2 + state.warmth * 0.1, 0.08 + psi * 0.2);

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

/// Complexity is rendered as nested structural contours rather than hidden metadata.
fn build_complexity_contours(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("complexity"));
    group.style.opacity = Some(opacity * (0.25 + state.complexity * 0.5));

    let (cx, cy) = state.layout_center;
    let count = (1 + (state.complexity * 5.0).round() as usize).min(6);
    for i in 0..count {
        let radius = 34.0 + i as f32 * 22.0;
        group.children.push(
            SceneNode::circle(cx, cy, radius).with_style(Style {
                fill: None,
                stroke: Some(
                    state
                        .palette
                        .ambient
                        .with_alpha(0.12 + state.complexity * 0.18),
                ),
                stroke_width: Some(0.4 + state.complexity * 0.8),
                ..Style::default()
            }),
        );
    }

    group
}

/// Acetylcholine sharpens and expands the attentional aperture.
fn build_attention_halo(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("attention"));
    group.style.opacity = Some(opacity * (0.2 + state.attention * 0.6));
    let (cx, cy) = state.layout_center;
    let radius = 20.0 + state.attention * 34.0;
    group.children.push(
        SceneNode::circle(cx, cy, radius).with_style(Style {
            fill: None,
            stroke: Some(
                state
                    .palette
                    .accent
                    .with_alpha(0.2 + state.attention * 0.45),
            ),
            stroke_width: Some(0.5 + state.attention * 2.0),
            ..Style::default()
        }),
    );
    group
}

/// Allostatic load contracts the available visual field into a visible boundary.
fn build_allostatic_boundary(state: &AestheticState, opacity: f32) -> SceneNode {
    let mut group = SceneNode::group(Some("allostatic-boundary"));
    group.style.opacity = Some(opacity * state.allostatic_load);
    let inset = 12.0 + state.allostatic_load * 76.0;
    let width = (VIEWPORT_W - inset * 2.0).max(0.0);
    let height = (VIEWPORT_H - inset * 2.0).max(0.0);
    let mut boundary = SceneNode::rect(inset, inset, width, height);
    if let NodeKind::Rect { rx, .. } = &mut boundary.kind {
        *rx = 8.0 + state.allostatic_load * 24.0;
    }
    boundary.style = Style {
        fill: None,
        stroke: Some(
            Color::from_hsl(8.0, 0.65, 0.5).with_alpha(0.25 + state.allostatic_load * 0.5),
        ),
        stroke_width: Some(0.75 + state.allostatic_load * 2.25),
        ..Style::default()
    };
    group.children.push(boundary);
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
        fill: Some(
            state
                .palette
                .primary
                .with_alpha(0.08 + state.coherence * 0.12),
        ),
        stroke: Some(
            state
                .palette
                .primary
                .with_alpha(0.35 + state.coherence * 0.45),
        ),
        stroke_width: Some(0.75 + state.coherence * 1.5),
        ..Style::default()
    });
    group.children.push(poly);

    // Spoke lines from center to each vertex
    for (px, py) in &points {
        let spoke = SceneNode::line(cx, cy, *px, *py).with_style(Style {
            stroke: Some(
                state
                    .palette
                    .primary
                    .with_alpha(0.12 + state.coherence * 0.22),
            ),
            stroke_width: Some(0.35 + state.coherence * 0.45),
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
    apply_turbulence(&mut group, state);

    let (cx, cy) = state.layout_center;
    let count = state.component_count.min(8); // cap at 8 for visual sanity
    let phase = state.cycle_phase as f32;

    for i in 0..count {
        let angle = (i as f32 / count as f32) * 2.0 * PI + phase * 0.5;
        let dist = (60.0 + (i as f32) * 15.0) * (0.9 + state.vitality * 0.2);
        let x = cx + dist * angle.cos();
        let y = cy + dist * angle.sin();
        let r = (8.0 + state.luminosity * 12.0) * (0.75 + state.vitality * 0.5);

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
    apply_turbulence(&mut group, state);

    let (cx, cy) = state.layout_center;
    let count = state.ring_count.min(5);

    for i in 0..count {
        let rx = 50.0 + (i as f32) * 30.0;
        let ry = rx * (0.45 + state.coherence * 0.45); // coherent states approach circular orbits
        let rotation = (i as f32) * 30.0 + state.cycle_phase as f32 * 60.0;

        let ring = SceneNode::ellipse(cx, cy, rx, ry)
            .with_style(Style {
                fill: None,
                stroke: Some(state.palette.accent.with_alpha(0.2 + state.coherence * 0.4)),
                stroke_width: Some(0.6 + state.coherence * 1.0),
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
    apply_turbulence(&mut group, state);

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
    apply_turbulence(&mut group, state);

    let (cx, cy) = state.layout_center;
    let attention_gain = 0.5 + state.attention * 0.5;
    let depth = ((state.fractal_depth.min(5) as f32) * attention_gain)
        .round()
        .clamp(1.0, 5.0) as usize;
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

fn apply_turbulence(group: &mut SceneNode, state: &AestheticState) {
    if state.turbulence > 0.01 {
        group.style.filter = Some("turb-filter".to_string());
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
    let breath_amplitude = 0.04 + state.vitality * 0.14;
    let breath_scale = 1.0 + (breath * 2.0 - 1.0) * breath_amplitude;

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
    apply_turbulence(&mut group, state);

    let (cx, cy) = state.layout_center;
    let count = ((state.energy - 0.3) * 15.0).round() as usize; // 0-10 particles
    let count = count.min(10);
    let phase = state.cycle_phase as f32;

    for i in 0..count {
        let angle = (i as f32 / count.max(1) as f32) * 2.0 * PI + phase * 4.0;
        let dist = 90.0 + (i as f32) * 8.0;
        let x = cx + dist * angle.cos();
        let y = cy + dist * angle.sin();

        let particle = SceneNode::circle(x, y, 1.5 + state.attention).with_style(Style {
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
            vitality: 0.5,
            coherence: 0.5,
            attention: 0.5,
            allostatic_load: 0.2,
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
    fn semantic_channels_have_distinct_visual_effects() {
        let mut low = make_state(0.9);
        low.vitality = 0.0;
        low.coherence = 0.0;
        low.attention = 0.1;
        low.allostatic_load = 0.1;
        low.complexity = 0.1;
        low.warmth = 0.0;

        let mut high = low.clone();
        high.vitality = 1.0;
        high.coherence = 1.0;
        high.attention = 1.0;
        high.allostatic_load = 1.0;
        high.complexity = 1.0;
        high.warmth = 1.0;

        let low_scene = build_scene(&low);
        let high_scene = build_scene(&high);

        let group = |scene: &SceneNode, wanted: &str| {
            scene
                .children
                .iter()
                .find(|node| {
                    matches!(
                        &node.kind,
                        NodeKind::Group { id } if id.as_deref() == Some(wanted)
                    )
                })
                .unwrap_or_else(|| panic!("missing group {wanted}"))
        };

        assert!(
            group(&high_scene, "complexity").children.len()
                > group(&low_scene, "complexity").children.len()
        );

        let halo_radius = |scene: &SceneNode| match group(scene, "attention").children[0].kind {
            NodeKind::Circle { r, .. } => r,
            _ => panic!("attention halo must be a circle"),
        };
        assert!(halo_radius(&high_scene) > halo_radius(&low_scene));

        let boundary_inset =
            |scene: &SceneNode| match group(scene, "allostatic-boundary").children[0].kind {
                NodeKind::Rect { x, .. } => x,
                _ => panic!("allostatic boundary must be a rect"),
            };
        assert!(boundary_inset(&high_scene) > boundary_inset(&low_scene));

        let bloom_radius = |scene: &SceneNode| match group(scene, "bloom").children[0].kind {
            NodeKind::Circle { r, .. } => r,
            _ => panic!("bloom must be a circle"),
        };
        assert!(bloom_radius(&high_scene) > bloom_radius(&low_scene));

        let ring_ry = |scene: &SceneNode| match group(scene, "rings").children[0].kind {
            NodeKind::Ellipse { ry, .. } => ry,
            _ => panic!("ring must be an ellipse"),
        };
        assert!(ring_ry(&high_scene) > ring_ry(&low_scene));

        let background_center =
            |scene: &SceneNode| match &group(scene, "background").children[0].kind {
                NodeKind::RadialGradient { stops, .. } => stops[0].color,
                _ => panic!("background must define a radial gradient"),
            };
        assert_ne!(
            background_center(&high_scene),
            background_center(&low_scene)
        );
    }

    #[test]
    fn prediction_error_filter_is_applied_to_artwork() {
        let mut state = make_state(0.9);
        state.turbulence = 0.8;
        let scene = build_scene(&state);
        let masses = scene
            .children
            .iter()
            .find(|c| matches!(&c.kind, NodeKind::Group { id } if id.as_deref() == Some("masses")))
            .expect("masses layer");
        assert_eq!(masses.style.filter.as_deref(), Some("turb-filter"));
        assert!(scene.children.iter().any(|node| matches!(
            &node.kind,
            NodeKind::Filter { id, .. } if id == "turb-filter"
        )));
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
