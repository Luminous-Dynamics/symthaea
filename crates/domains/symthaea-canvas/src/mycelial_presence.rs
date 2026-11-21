// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Abstract Mycelial Presence adapter for the renderer-neutral embodiment contract.
//!
//! This is a deliberately non-humanoid first embodiment. Its geometry responds
//! only to explicit semantic snapshot fields; missing signals remain absent.

use std::f32::consts::TAU;

use crate::color::Color;
use crate::embodiment::{EmbodimentActivity, EmbodimentSnapshot, SpeechOutputState};
use crate::mycelial_model::{PresenceMode, SpatialPoint3, build_mycelial_model};
use crate::scene_graph::{FilterType, GradientStop, NodeKind, SceneNode, Style, Transform};
use crate::svg_renderer::{SvgRenderOptions, VIEWPORT_H, VIEWPORT_W, render_svg_with_options};

const CENTER_X: f32 = VIEWPORT_W / 2.0;
const CENTER_Y: f32 = VIEWPORT_H / 2.0;

fn build_activity_motif(activity: EmbodimentActivity, color: Color) -> SceneNode {
    // Every state glyph sits in the same compact seal above the organism.
    // Geometry, not hue, carries the primary distinction.
    let glyph_x = CENTER_X;
    let glyph_y = CENTER_Y - 132.0;
    let mut motif = SceneNode::group(Some(match activity {
        EmbodimentActivity::Listening => "motif-listening",
        EmbodimentActivity::Perceiving => "motif-perceiving",
        EmbodimentActivity::Retrieving => "motif-retrieving",
        EmbodimentActivity::Associating => "motif-associating",
        EmbodimentActivity::Predicting => "motif-predicting",
        EmbodimentActivity::Planning => "motif-planning",
        EmbodimentActivity::ExecutingTool => "motif-executing-tool",
        EmbodimentActivity::AwaitingTool => "motif-awaiting-tool",
        EmbodimentActivity::Interrupted => "motif-interrupted",
        EmbodimentActivity::Fault => "motif-fault",
        EmbodimentActivity::Idle => "motif-idle",
    }));

    motif
        .children
        .push(SceneNode::circle(glyph_x, glyph_y, 27.0).with_style(Style {
            fill: Some(Color::from_hsla(224.0, 0.42, 0.075, 0.94)),
            stroke: Some(color.with_alpha(0.48)),
            stroke_width: Some(1.1),
            ..Style::default()
        }));
    motif
        .children
        .push(SceneNode::circle(glyph_x, glyph_y, 22.0).with_style(Style {
            fill: None,
            stroke: Some(color.with_alpha(0.24)),
            stroke_width: Some(0.7),
            ..Style::default()
        }));

    let thin = Style {
        fill: None,
        stroke: Some(color.with_alpha(0.92)),
        stroke_width: Some(1.55),
        ..Style::default()
    };
    let dot = |x: f32, y: f32, radius: f32| {
        SceneNode::circle(x, y, radius).with_style(Style {
            fill: Some(color.with_alpha(0.96)),
            stroke: Some(Color::from_hsla(190.0, 0.34, 0.96, 0.72)),
            stroke_width: Some(0.55),
            ..Style::default()
        })
    };

    match activity {
        EmbodimentActivity::Idle => {}
        EmbodimentActivity::Listening => {
            for index in 0..3 {
                let spread = index as f32 * 4.5;
                motif.children.push(
                    SceneNode::ellipse(glyph_x, glyph_y, 7.0 + spread, 4.0 + spread * 0.48)
                        .with_style(thin.clone()),
                );
            }
        }
        EmbodimentActivity::Perceiving => {
            for offset in [-12.0, 0.0, 12.0] {
                motif.children.push(dot(glyph_x + offset, glyph_y, 2.8));
            }
            motif.children.push(
                SceneNode::line(glyph_x - 15.0, glyph_y, glyph_x + 15.0, glyph_y)
                    .with_style(thin.clone()),
            );
        }
        EmbodimentActivity::Retrieving => {
            motif
                .children
                .push(SceneNode::ellipse(glyph_x, glyph_y, 14.0, 9.0).with_style(thin.clone()));
            for index in 0..4 {
                let angle = index as f32 * TAU / 4.0;
                motif.children.push(dot(
                    glyph_x + angle.cos() * 14.0,
                    glyph_y + angle.sin() * 9.0,
                    2.5,
                ));
            }
            motif.children.push(dot(glyph_x, glyph_y, 3.0));
        }
        EmbodimentActivity::Associating => {
            motif.children.push(
                SceneNode::path(format!(
                    "M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}",
                    glyph_x - 15.0,
                    glyph_y + 4.0,
                    glyph_x - 7.0,
                    glyph_y - 10.0,
                    glyph_x + 7.0,
                    glyph_y - 10.0,
                    glyph_x + 15.0,
                    glyph_y + 4.0
                ))
                .with_style(thin.clone()),
            );
            motif.children.push(dot(glyph_x - 15.0, glyph_y + 4.0, 2.8));
            motif.children.push(dot(glyph_x + 15.0, glyph_y + 4.0, 2.8));
            motif.children.push(dot(glyph_x, glyph_y - 6.0, 2.2));
        }
        EmbodimentActivity::Predicting => {
            for offset in [-12.0, 0.0, 12.0] {
                motif.children.push(
                    SceneNode::path(format!(
                        "M {:.1} {:.1} L {:.1} {:.1} L {:.1} {:.1}",
                        glyph_x,
                        glyph_y + 10.0,
                        glyph_x + offset * 0.45,
                        glyph_y - 3.0,
                        glyph_x + offset,
                        glyph_y - 15.0
                    ))
                    .with_style(thin.clone()),
                );
                motif
                    .children
                    .push(dot(glyph_x + offset, glyph_y - 15.0, 2.4));
            }
        }
        EmbodimentActivity::Planning => {
            let points = (0..6)
                .map(|index| {
                    let angle = index as f32 * TAU / 6.0 - TAU / 4.0;
                    (glyph_x + angle.cos() * 14.0, glyph_y + angle.sin() * 12.0)
                })
                .collect();
            motif
                .children
                .push(SceneNode::polygon(points, true).with_style(Style {
                    fill: Some(color.with_alpha(0.12)),
                    stroke: Some(color.with_alpha(0.96)),
                    stroke_width: Some(1.35),
                    ..Style::default()
                }));
            motif.children.push(dot(glyph_x, glyph_y, 2.8));
        }
        EmbodimentActivity::ExecutingTool => {
            motif.children.push(
                SceneNode::line(glyph_x, glyph_y + 12.0, glyph_x, glyph_y - 3.0)
                    .with_style(thin.clone()),
            );
            for offset in [-12.0, 0.0, 12.0] {
                let x = glyph_x + offset;
                motif.children.push(
                    SceneNode::line(x, glyph_y - 3.0, x, glyph_y - 11.0).with_style(thin.clone()),
                );
                motif.children.push(
                    SceneNode::rect(x - 3.0, glyph_y - 16.0, 6.0, 6.0).with_style(Style {
                        fill: Some(color.with_alpha(0.20)),
                        stroke: Some(color.with_alpha(0.98)),
                        stroke_width: Some(1.1),
                        ..Style::default()
                    }),
                );
            }
        }
        EmbodimentActivity::AwaitingTool => {
            motif
                .children
                .push(SceneNode::circle(glyph_x, glyph_y, 12.0).with_style(thin.clone()));
            motif.children.push(dot(glyph_x, glyph_y, 3.2));
        }
        EmbodimentActivity::Interrupted => {
            motif.children.push(
                SceneNode::rect(glyph_x - 7.0, glyph_y - 12.0, 4.0, 24.0).with_style(Style {
                    fill: Some(color.with_alpha(0.94)),
                    ..Style::default()
                }),
            );
            motif.children.push(
                SceneNode::rect(glyph_x + 3.0, glyph_y - 12.0, 4.0, 24.0).with_style(Style {
                    fill: Some(color.with_alpha(0.94)),
                    ..Style::default()
                }),
            );
        }
        EmbodimentActivity::Fault => {
            motif.children.push(
                SceneNode::line(glyph_x - 9.0, glyph_y - 9.0, glyph_x + 9.0, glyph_y + 9.0)
                    .with_style(thin.clone()),
            );
            motif.children.push(
                SceneNode::line(glyph_x + 9.0, glyph_y - 9.0, glyph_x - 9.0, glyph_y + 9.0)
                    .with_style(thin),
            );
        }
    }

    motif
}

fn blur_filter(id: &str, std_dev: f32) -> SceneNode {
    SceneNode {
        kind: NodeKind::Filter {
            id: id.to_string(),
            filter_type: FilterType::Blur { std_dev },
        },
        transform: crate::scene_graph::Transform::identity(),
        style: Style::default(),
        children: Vec::new(),
    }
}

fn build_starfield() -> SceneNode {
    let mut stars = SceneNode::group(Some("starfield"));
    // Golden-angle placement avoids a rigid grid while remaining deterministic.
    let golden_angle = 2.399_963_1_f32;
    for index in 0..38 {
        let angle = index as f32 * golden_angle + 0.37;
        let radius = 184.0 + ((index * 47) % 112) as f32;
        let x = CENTER_X + angle.cos() * radius;
        let y = CENTER_Y + angle.sin() * radius;
        let size = 0.7 + ((index * 13) % 4) as f32 * 0.34;
        let alpha = 0.14 + ((index * 7) % 5) as f32 * 0.045;
        let tint = if index % 7 == 0 {
            Color::from_hsl(42.0, 0.72, 0.78)
        } else if index % 3 == 0 {
            Color::from_hsl(276.0, 0.52, 0.76)
        } else {
            Color::from_hsl(188.0, 0.58, 0.76)
        };
        stars
            .children
            .push(SceneNode::circle(x, y, size).with_style(Style {
                fill: Some(tint.with_alpha(alpha)),
                css_class: Some(format!("spore-phase-{}", index % 5)),
                ..Style::default()
            }));
    }
    stars
}

fn build_core(speech_weight: f32, activity_color: Color) -> SceneNode {
    let mut core = SceneNode::group(Some("bloom"));

    core.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 76.0).with_style(Style {
            fill: Some(activity_color.with_alpha(0.28)),
            filter: Some("mycelium-soft-glow".to_string()),
            ..Style::default()
        }),
    );
    core.children.push(
        SceneNode::ellipse(CENTER_X, CENTER_Y, 58.0, 44.0).with_style(Style {
            fill: Some(Color::from_hsla(187.0, 0.72, 0.32, 0.16)),
            stroke: Some(activity_color.with_alpha(0.36)),
            stroke_width: Some(1.2),
            ..Style::default()
        }),
    );

    for index in 0..10 {
        let angle = index as f32 * TAU / 10.0 + 0.18;
        let x = CENTER_X + angle.cos() * 40.0;
        let y = CENTER_Y + angle.sin() * 30.0;
        let radius = if index % 2 == 0 { 3.3 } else { 2.4 };
        let cell_color = if index % 3 == 0 {
            Color::from_hsl(44.0, 0.82, 0.72)
        } else {
            activity_color
        };
        core.children
            .push(SceneNode::circle(x, y, radius).with_style(Style {
                fill: Some(cell_color.with_alpha(0.86)),
                stroke: Some(Color::from_hsla(186.0, 0.50, 0.94, 0.55)),
                stroke_width: Some(0.55),
                ..Style::default()
            }));
    }

    core.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 27.0 + speech_weight * 5.0).with_style(Style {
            fill_url: Some("mycelium-core".to_string()),
            stroke: Some(Color::from_hsl(182.0, 0.82, 0.84).with_alpha(0.80)),
            stroke_width: Some(1.7),
            ..Style::default()
        }),
    );
    core.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 16.0 + speech_weight * 2.0).with_style(Style {
            fill: Some(Color::from_hsl(43.0, 0.88, 0.78)),
            opacity: Some(0.72),
            ..Style::default()
        }),
    );
    core.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 5.5 + speech_weight).with_style(Style {
            fill: Some(Color::from_hsl(48.0, 0.98, 0.92)),
            stroke: Some(Color::from_hsl(190.0, 0.48, 0.98).with_alpha(0.82)),
            stroke_width: Some(0.75),
            ..Style::default()
        }),
    );
    core
}

/// Build the Mycelial Presence scene graph from explicit embodiment signals.
///
/// Absent activity defaults to a quiet visual presentation, not an inferred
/// runtime state. Uncertainty and speech modulation are rendered only when the
/// corresponding observed signal is present.
pub fn build_mycelial_presence(snapshot: &EmbodimentSnapshot) -> SceneNode {
    build_mycelial_presence_with_mode(snapshot, PresenceMode::Ambient)
}

/// Build the same canonical organism in a user-selected presentation mode.
pub fn build_mycelial_presence_with_mode(
    snapshot: &EmbodimentSnapshot,
    mode: PresenceMode,
) -> SceneNode {
    let model = build_mycelial_model(snapshot, mode);
    let activity = model
        .activity
        .as_ref()
        .map(|observed| observed.value)
        .unwrap_or(EmbodimentActivity::Idle);
    let activity_color = color_for_activity(activity);

    let mut root = SceneNode::group(Some("mycelial-presence"));
    root.children.push(radial_gradient(
        "mycelium-background",
        Color::from_hsl(231.0, 0.40, 0.065),
        Color::from_hsl(228.0, 0.48, 0.014),
    ));
    root.children.push(radial_gradient(
        "mycelium-core",
        Color::from_hsl(43.0, 0.88, 0.86),
        activity_color,
    ));
    root.children.push(radial_gradient(
        "mycelium-nebula",
        Color::from_hsla(274.0, 0.62, 0.30, 0.28),
        Color::from_hsla(274.0, 0.62, 0.08, 0.0),
    ));
    root.children.push(radial_gradient(
        "mycelium-nebula-cyan",
        Color::from_hsla(190.0, 0.76, 0.28, 0.22),
        Color::from_hsla(190.0, 0.76, 0.08, 0.0),
    ));
    root.children.push(blur_filter("mycelium-soft-glow", 12.0));
    root.children.push(
        SceneNode::rect(0.0, 0.0, VIEWPORT_W, VIEWPORT_H).with_style(Style {
            fill_url: Some("mycelium-background".to_string()),
            ..Style::default()
        }),
    );
    root.children.push(
        SceneNode::ellipse(CENTER_X - 38.0, CENTER_Y + 28.0, 208.0, 152.0).with_style(Style {
            fill_url: Some("mycelium-nebula".to_string()),
            opacity: Some(0.52),
            ..Style::default()
        }),
    );
    root.children.push(
        SceneNode::ellipse(CENTER_X + 104.0, CENTER_Y - 92.0, 176.0, 126.0).with_style(Style {
            fill_url: Some("mycelium-nebula-cyan".to_string()),
            opacity: Some(0.44),
            ..Style::default()
        }),
    );
    root.children.push(build_starfield());

    let mut hyphae = SceneNode::group(Some("hyphae"));
    hyphae.style.opacity = Some(if activity == EmbodimentActivity::Idle {
        0.70
    } else {
        0.96
    });
    // Preserve the signature teal identity while letting the current runtime
    // activity tint the organism's branches in a restrained, coherent way.
    let base_color = Color::from_hsl(184.0, 0.82, 0.62).lerp(&activity_color, 0.18);
    for filament in &model.filaments {
        let (start_x, start_y) = project_spatial(filament.start);
        let (control_a_x, control_a_y) = project_spatial(filament.control_a);
        let (control_b_x, control_b_y) = project_spatial(filament.control_b);
        let (end_x, end_y) = project_spatial(filament.end);
        let path = format!(
            "M {:.2} {:.2} C {:.2} {:.2}, {:.2} {:.2}, {:.2} {:.2}",
            start_x, start_y, control_a_x, control_a_y, control_b_x, control_b_y, end_x, end_y,
        );
        let generation_factor = (filament.generation as f32 / 4.0).clamp(0.0, 1.0);
        let branch_color = base_color.lerp(
            &Color::from_hsl(274.0, 0.58, 0.70),
            generation_factor * 0.72,
        );

        if filament.generation == 0 {
            hyphae
                .children
                .push(SceneNode::path(path.clone()).with_style(Style {
                    fill: None,
                    stroke: Some(branch_color.with_alpha(0.16)),
                    stroke_width: Some(filament.radius_start * 5.0),
                    filter: Some("mycelium-soft-glow".to_string()),
                    ..Style::default()
                }));
        }

        hyphae
            .children
            .push(SceneNode::path(path).with_style(Style {
                fill: None,
                stroke: Some(branch_color.with_alpha(0.44 + (1.0 - generation_factor) * 0.42)),
                stroke_width: Some((filament.radius_start + filament.radius_end) * 0.72),
                ..Style::default()
            }));

        if filament.terminal {
            hyphae.children.push(
                SceneNode::circle(end_x, end_y, filament.radius_end * 1.8).with_style(Style {
                    fill: Some(branch_color.with_alpha(0.22)),
                    ..Style::default()
                }),
            );
            hyphae.children.push(
                SceneNode::circle(end_x, end_y, filament.radius_end * 0.9).with_style(Style {
                    fill: Some(branch_color.with_alpha(0.94)),
                    stroke: Some(Color::from_hsla(186.0, 0.36, 0.96, 0.66)),
                    stroke_width: Some(0.5),
                    ..Style::default()
                }),
            );
        }
    }
    root.children.push(hyphae);

    let mut rings = SceneNode::group(Some("rings"));
    for ring in 0..3 {
        let offset = ring as f32 * 17.0;
        rings.children.push(
            SceneNode::ellipse(CENTER_X, CENTER_Y, 58.0 + offset, 43.0 + offset).with_style(
                Style {
                    fill: None,
                    stroke: Some(activity_color.with_alpha(0.10 + ring as f32 * 0.035)),
                    stroke_width: Some(0.8 + ring as f32 * 0.25),
                    ..Style::default()
                },
            ),
        );
    }
    root.children.push(rings);

    let mut orbital_cells = SceneNode::group(Some("orbital-cells"));
    let cell_scale = match model.mode {
        PresenceMode::Ambient => 1.0,
        PresenceMode::Cognitive => 1.22,
        PresenceMode::Conversational => 0.86,
    };
    for cell in &model.orbital_cells {
        let (x, y) = project_spatial(cell.position);
        orbital_cells.children.push(
            SceneNode::circle(x, y, cell.radius * cell_scale).with_style(Style {
                fill: Some(activity_color.with_alpha(0.46)),
                stroke: Some(Color::from_hsla(187.0, 0.46, 0.94, 0.42)),
                stroke_width: Some(0.45),
                ..Style::default()
            }),
        );
    }
    root.children.push(orbital_cells);

    if let Some(gaze) = &model.gaze {
        let (dx, dy) = match gaze.value {
            crate::embodiment::GazeTarget::ConversationPartner => (0.0, -1.0),
            crate::embodiment::GazeTarget::Content => (-0.82, -0.42),
            crate::embodiment::GazeTarget::Offscreen => (0.82, -0.34),
            crate::embodiment::GazeTarget::Neutral => (0.0, 1.0),
        };
        let mut gaze_marker = SceneNode::group(Some("gaze-indicator"));
        gaze_marker.children.push(
            SceneNode::line(
                CENTER_X + dx * 35.0,
                CENTER_Y + dy * 28.0,
                CENTER_X + dx * 77.0,
                CENTER_Y + dy * 62.0,
            )
            .with_style(Style {
                fill: None,
                stroke: Some(Color::from_hsl(188.0, 0.72, 0.82).with_alpha(0.68)),
                stroke_width: Some(1.15),
                ..Style::default()
            }),
        );
        gaze_marker.children.push(
            SceneNode::circle(CENTER_X + dx * 77.0, CENTER_Y + dy * 62.0, 3.0).with_style(Style {
                fill: Some(Color::from_hsl(188.0, 0.72, 0.86)),
                ..Style::default()
            }),
        );
        root.children.push(gaze_marker);
    }

    if model.mode == PresenceMode::Cognitive {
        let mut structure_frame = SceneNode::group(Some("signal-inspector-guide-rings"));
        for radius in [132.0, 145.0, 158.0] {
            structure_frame.children.push(
                SceneNode::circle(CENTER_X, CENTER_Y, radius).with_style(Style {
                    fill: None,
                    stroke: Some(Color::from_hsl(187.0, 0.50, 0.76).with_alpha(0.12)),
                    stroke_width: Some(0.65),
                    ..Style::default()
                }),
            );
        }
        root.children.push(structure_frame);
    }

    if model
        .speech_output
        .as_ref()
        .is_some_and(|observed| observed.value == SpeechOutputState::Playing)
    {
        let mut speech_ring = SceneNode::group(Some("speech-output"));
        speech_ring.children.push(
            SceneNode::ellipse(CENTER_X, CENTER_Y, 75.0, 58.0).with_style(Style {
                fill: None,
                stroke: Some(activity_color.with_alpha(0.44)),
                stroke_width: Some(1.4),
                ..Style::default()
            }),
        );
        root.children.push(speech_ring);
    }

    if activity != EmbodimentActivity::Idle {
        let mut activity_halo = SceneNode::group(Some("activity-halo"));
        activity_halo.style.opacity = Some(0.25);
        activity_halo
            .children
            .push(
                SceneNode::circle(CENTER_X, CENTER_Y, 104.0).with_style(Style {
                    fill: None,
                    stroke: Some(activity_color),
                    stroke_width: Some(2.0),
                    ..Style::default()
                }),
            );
        root.children.push(activity_halo);
        root.children
            .push(build_activity_motif(activity, activity_color));
    }

    if let Some(uncertainty) = &model.uncertainty {
        let level = finite_unit(uncertainty.value);
        if level > 0.0 {
            let mut uncertainty_halo = SceneNode::group(Some("uncertainty-halo"));
            uncertainty_halo.style.opacity = Some(0.2 + level * 0.55);
            uncertainty_halo.children.push(
                SceneNode::circle(CENTER_X, CENTER_Y, 121.0).with_style(Style {
                    fill: None,
                    stroke: Some(Color::from_hsl(38.0, 0.92, 0.68)),
                    stroke_width: Some(0.8 + level * 2.4),
                    ..Style::default()
                }),
            );
            root.children.push(uncertainty_halo);
        }
    }

    let speech_weight = model
        .viseme
        .as_ref()
        .map(|observed| finite_unit(observed.value.weight))
        .unwrap_or(0.0);

    root.children
        .push(build_core(speech_weight, activity_color));

    let mut particles = SceneNode::group(Some("particles"));
    for particle in 0..12 {
        let angle = particle as f32 * TAU / 12.0 + 0.13;
        let radius = 145.0 + (particle % 3) as f32 * 15.0;
        let x = CENTER_X + angle.cos() * radius;
        let y = CENTER_Y + angle.sin() * radius;
        particles.children.push(
            SceneNode::circle(x, y, 1.8 + (particle % 2) as f32 * 0.7).with_style(Style {
                fill: Some(activity_color.with_alpha(0.55)),
                ..Style::default()
            }),
        );
    }
    root.children.push(particles);

    root
}

/// Render a self-contained SVG Mycelial Presence.
pub fn render_mycelial_presence(
    snapshot: &EmbodimentSnapshot,
    options: SvgRenderOptions,
) -> String {
    render_mycelial_presence_with_mode(snapshot, PresenceMode::Ambient, options)
}

/// Render a selected Mycelial Presence mode through the SVG compatibility path.
pub fn render_mycelial_presence_with_mode(
    snapshot: &EmbodimentSnapshot,
    mode: PresenceMode,
    options: SvgRenderOptions,
) -> String {
    let scene = build_mycelial_presence_with_mode(snapshot, mode);
    render_svg_with_options(&scene, 1.0, options)
}

fn radial_gradient(id: &str, center: Color, edge: Color) -> SceneNode {
    SceneNode {
        kind: NodeKind::RadialGradient {
            id: id.to_string(),
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: center,
                },
                GradientStop {
                    offset: 1.0,
                    color: edge,
                },
            ],
        },
        transform: Transform::identity(),
        style: Style::default(),
        children: Vec::new(),
    }
}

fn color_for_activity(activity: EmbodimentActivity) -> Color {
    let (hue, saturation, lightness) = match activity {
        EmbodimentActivity::Idle => (188.0, 0.62, 0.53),
        EmbodimentActivity::Listening => (184.0, 0.86, 0.62),
        EmbodimentActivity::Perceiving => (205.0, 0.82, 0.62),
        EmbodimentActivity::Retrieving => (158.0, 0.72, 0.57),
        EmbodimentActivity::Associating => (151.0, 0.78, 0.61),
        EmbodimentActivity::Predicting => (43.0, 0.88, 0.64),
        EmbodimentActivity::Planning => (275.0, 0.66, 0.65),
        EmbodimentActivity::ExecutingTool => (27.0, 0.90, 0.60),
        EmbodimentActivity::AwaitingTool => (263.0, 0.70, 0.64),
        EmbodimentActivity::Interrupted => (7.0, 0.78, 0.59),
        EmbodimentActivity::Fault => (0.0, 0.86, 0.58),
    };
    Color::from_hsl(hue, saturation, lightness)
}

fn project_spatial(point: SpatialPoint3) -> (f32, f32) {
    // Mild orthographic depth projection retains the 3D canopy's parallax cues.
    (
        CENTER_X + point.x + point.z * 0.16,
        CENTER_Y + point.y - point.z * 0.28,
    )
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::{
        EmbodimentActivity, Observed, SignalProvenance, SpeechOutputState, Viseme, VisemeSignal,
    };

    fn observed<T>(value: T, sequence: u64) -> Observed<T> {
        Observed {
            value,
            provenance: SignalProvenance::RuntimeEvent,
            sequence,
            monotonic_ns: sequence * 1_000,
        }
    }

    #[test]
    fn presence_uses_layered_glow_nebula_and_deterministic_starfield() {
        let scene = build_mycelial_presence(&EmbodimentSnapshot::default());
        let ids: Vec<&str> = scene
            .children
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Group { id: Some(id) } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert!(ids.contains(&"starfield"));
        assert!(ids.contains(&"bloom"));
        assert!(ids.contains(&"orbital-cells"));
        assert!(
            scene
                .children
                .iter()
                .any(|node| matches!(node.kind, NodeKind::Filter { .. }))
        );
    }

    #[test]
    fn all_activity_glyphs_share_the_same_visual_anchor() {
        let activities = [
            EmbodimentActivity::Listening,
            EmbodimentActivity::Perceiving,
            EmbodimentActivity::Retrieving,
            EmbodimentActivity::Associating,
            EmbodimentActivity::Predicting,
            EmbodimentActivity::Planning,
            EmbodimentActivity::ExecutingTool,
            EmbodimentActivity::AwaitingTool,
            EmbodimentActivity::Interrupted,
            EmbodimentActivity::Fault,
        ];
        for activity in activities {
            let motif = build_activity_motif(activity, color_for_activity(activity));
            assert!(matches!(
                motif.children.first().map(|node| &node.kind),
                Some(NodeKind::Circle { cx, cy, r })
                    if (*cx - CENTER_X).abs() < f32::EPSILON
                        && (*cy - (CENTER_Y - 132.0)).abs() < f32::EPSILON
                        && (*r - 27.0).abs() < f32::EPSILON
            ));
        }
    }

    #[test]
    fn full_motion_svg_has_staggered_spores_and_biomimetic_hyphae_pulse() {
        let svg =
            render_mycelial_presence(&EmbodimentSnapshot::default(), SvgRenderOptions::default());
        assert!(svg.contains("spore-phase-0"));
        assert!(svg.contains("spore-phase-4"));
        assert!(svg.contains("hyphae-pulse"));
        assert!(svg.contains("mycelium-soft-glow"));
    }

    #[test]
    fn primary_hyphae_use_deterministic_non_uniform_growth() {
        let first =
            render_mycelial_presence(&EmbodimentSnapshot::default(), SvgRenderOptions::default());
        let second =
            render_mycelial_presence(&EmbodimentSnapshot::default(), SvgRenderOptions::default());
        assert_eq!(first, second);
        assert!(first.matches("filter=\"url(#mycelium-soft-glow)\"").count() < 16);
    }

    #[test]
    fn idle_presence_renders_as_a_non_humanoid_scene() {
        let snapshot = EmbodimentSnapshot::default();
        let scene = build_mycelial_presence(&snapshot);
        assert!(scene.node_count() > 100);
        assert!(matches!(scene.kind, NodeKind::Group { .. }));

        let svg = render_mycelial_presence(&snapshot, SvgRenderOptions::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("mycelial-presence"));
        assert!(svg.contains("hyphae"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn associative_activity_and_playback_change_the_visual_scene() {
        let idle = EmbodimentSnapshot::default();
        let mut speaking = EmbodimentSnapshot::default();
        speaking.activity = Some(observed(EmbodimentActivity::Associating, 1));
        speaking.speech_output = Some(observed(SpeechOutputState::Playing, 2));
        speaking.viseme = Some(observed(
            VisemeSignal {
                viseme: Viseme::Aa,
                weight: 0.8,
            },
            3,
        ));

        let idle_svg = render_mycelial_presence(&idle, SvgRenderOptions::default());
        let speaking_svg = render_mycelial_presence(&speaking, SvgRenderOptions::default());
        assert_ne!(idle_svg, speaking_svg);
        assert!(speaking_svg.contains("activity-halo"));
    }

    #[test]
    fn viseme_modulates_the_core_only_during_confirmed_playback() {
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.activity = Some(observed(EmbodimentActivity::Associating, 1));
        snapshot.viseme = Some(observed(
            VisemeSignal {
                viseme: Viseme::Aa,
                weight: 0.9,
            },
            2,
        ));

        let queued = render_mycelial_presence(&snapshot, SvgRenderOptions::default());
        snapshot.speech_output = Some(observed(SpeechOutputState::Queued, 3));
        let still_queued = render_mycelial_presence(&snapshot, SvgRenderOptions::default());
        assert_eq!(queued, still_queued);

        snapshot.speech_output = Some(observed(SpeechOutputState::Playing, 4));
        let playing = render_mycelial_presence(&snapshot, SvgRenderOptions::default());
        assert_ne!(still_queued, playing);
    }

    #[test]
    fn semantic_activity_motifs_are_geometrically_distinct() {
        let mut associating = EmbodimentSnapshot::default();
        associating.activity = Some(observed(EmbodimentActivity::Associating, 1));
        let mut predicting = EmbodimentSnapshot::default();
        predicting.activity = Some(observed(EmbodimentActivity::Predicting, 1));

        let associating_scene = build_mycelial_presence(&associating);
        let predicting_scene = build_mycelial_presence(&predicting);
        let motif_id = |scene: &SceneNode| {
            scene.children.iter().find_map(|child| match &child.kind {
                NodeKind::Group { id }
                    if id
                        .as_deref()
                        .is_some_and(|value| value.starts_with("motif-")) =>
                {
                    id.clone()
                }
                _ => None,
            })
        };

        assert_eq!(
            motif_id(&associating_scene).as_deref(),
            Some("motif-associating")
        );
        assert_eq!(
            motif_id(&predicting_scene).as_deref(),
            Some("motif-predicting")
        );
        fn motif_node(scene: &SceneNode) -> Option<&SceneNode> {
            scene.children.iter().find(|child| match &child.kind {
                NodeKind::Group { id } => id
                    .as_deref()
                    .is_some_and(|value| value.starts_with("motif-")),
                _ => false,
            })
        }
        assert_ne!(
            motif_node(&associating_scene).unwrap().node_count(),
            motif_node(&predicting_scene).unwrap().node_count()
        );
    }

    #[test]
    fn uncertainty_is_rendered_only_when_explicitly_present() {
        let absent = EmbodimentSnapshot::default();
        let mut present = EmbodimentSnapshot::default();
        present.uncertainty = Some(observed(0.7_f32, 1));

        let absent_svg = render_mycelial_presence(&absent, SvgRenderOptions::default());
        let present_svg = render_mycelial_presence(&present, SvgRenderOptions::default());
        assert!(!absent_svg.contains("uncertainty-halo"));
        assert!(present_svg.contains("uncertainty-halo"));
    }

    #[test]
    fn zero_uncertainty_does_not_create_an_uncertainty_halo() {
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.uncertainty = Some(observed(0.0_f32, 1));
        let scene = build_mycelial_presence(&snapshot);
        assert!(!scene.children.iter().any(|child| {
            matches!(
                &child.kind,
                NodeKind::Group { id: Some(id) } if id == "uncertainty-halo"
            )
        }));
    }

    #[test]
    fn reduced_motion_is_respected_by_svg_renderer() {
        let snapshot = EmbodimentSnapshot::default();
        let svg = render_mycelial_presence(
            &snapshot,
            SvgRenderOptions {
                motion: crate::animation::MotionPreference::Reduced,
                ..SvgRenderOptions::default()
            },
        );
        assert!(svg.contains("animation: none"));
    }
    #[test]
    fn renderer_modes_share_one_model_but_offer_distinct_presentations() {
        let snapshot = EmbodimentSnapshot::default();
        let ambient = build_mycelial_presence_with_mode(&snapshot, PresenceMode::Ambient);
        let cognitive = build_mycelial_presence_with_mode(&snapshot, PresenceMode::Cognitive);
        let conversational =
            build_mycelial_presence_with_mode(&snapshot, PresenceMode::Conversational);
        assert_ne!(ambient.node_count(), cognitive.node_count());
        assert_ne!(ambient.node_count(), conversational.node_count());
        let cognitive_ids: Vec<&str> = cognitive
            .children
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Group { id: Some(id) } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert!(cognitive_ids.contains(&"signal-inspector-guide-rings"));
    }

    #[test]
    fn observed_gaze_and_confirmed_playback_have_separate_visual_layers() {
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.gaze = Some(observed(crate::embodiment::GazeTarget::Content, 3));
        let with_gaze = build_mycelial_presence_with_mode(&snapshot, PresenceMode::Conversational);
        assert!(with_gaze.children.iter().any(|node| matches!(
            &node.kind,
            NodeKind::Group { id: Some(id) } if id == "gaze-indicator"
        )));

        snapshot.speech_output = Some(observed(SpeechOutputState::Playing, 4));
        let playing = build_mycelial_presence_with_mode(&snapshot, PresenceMode::Conversational);
        assert!(playing.children.iter().any(|node| matches!(
            &node.kind,
            NodeKind::Group { id: Some(id) } if id == "speech-output"
        )));
    }
}
