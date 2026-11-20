// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Abstract Mycelial Presence adapter for the renderer-neutral embodiment contract.
//!
//! This is a deliberately non-humanoid first embodiment. Its geometry responds
//! only to explicit semantic snapshot fields; missing signals remain absent.

use std::f32::consts::TAU;

use crate::color::Color;
use crate::embodiment::{EmbodimentActivity, EmbodimentSnapshot};
use crate::scene_graph::{GradientStop, NodeKind, SceneNode, Style, Transform};
use crate::svg_renderer::{
    SvgRenderOptions, VIEWPORT_H, VIEWPORT_W, render_svg_with_options,
};

const CENTER_X: f32 = VIEWPORT_W / 2.0;
const CENTER_Y: f32 = VIEWPORT_H / 2.0;
const BRANCH_DEPTH: usize = 3;
const PRIMARY_BRANCHES: usize = 8;


fn build_activity_motif(activity: EmbodimentActivity, color: Color) -> SceneNode {
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
    let thin = Style {
        fill: None,
        stroke: Some(color.with_alpha(0.84)),
        stroke_width: Some(1.6),
        ..Style::default()
    };
    let dot = |x: f32, y: f32, radius: f32| {
        SceneNode::circle(x, y, radius).with_style(Style {
            fill: Some(color.with_alpha(0.92)),
            ..Style::default()
        })
    };

    match activity {
        EmbodimentActivity::Idle => {}
        EmbodimentActivity::Listening => {
            for index in 0..3 {
                let spread = index as f32 * 8.0;
                motif.children.push(
                    SceneNode::ellipse(
                        CENTER_X,
                        CENTER_Y - 132.0,
                        17.0 + spread,
                        5.0 + spread * 0.38,
                    )
                    .with_style(thin.clone()),
                );
            }
        }
        EmbodimentActivity::Perceiving => {
            for offset in [-24.0, 0.0, 24.0] {
                motif.children.push(dot(CENTER_X + offset, CENTER_Y - 132.0, 4.2));
            }
            motif.children.push(
                SceneNode::line(
                    CENTER_X - 24.0,
                    CENTER_Y - 132.0,
                    CENTER_X + 24.0,
                    CENTER_Y - 132.0,
                )
                .with_style(thin.clone()),
            );
        }
        EmbodimentActivity::Retrieving => {
            let orbit = SceneNode::ellipse(CENTER_X, CENTER_Y, 104.0, 76.0)
                .with_style(thin.clone());
            motif.children.push(orbit);
            motif.children.push(dot(CENTER_X, CENTER_Y - 76.0, 4.0));
            motif.children.push(dot(CENTER_X - 90.0, CENTER_Y + 38.0, 4.0));
            motif.children.push(dot(CENTER_X + 90.0, CENTER_Y + 38.0, 4.0));
        }
        EmbodimentActivity::Associating => {
            motif.children.push(
                SceneNode::path(format!(
                    "M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}",
                    CENTER_X - 92.0,
                    CENTER_Y - 72.0,
                    CENTER_X - 38.0,
                    CENTER_Y - 112.0,
                    CENTER_X + 38.0,
                    CENTER_Y - 112.0,
                    CENTER_X + 92.0,
                    CENTER_Y - 72.0
                ))
                .with_style(thin.clone()),
            );
            motif.children.push(dot(CENTER_X - 92.0, CENTER_Y - 72.0, 4.0));
            motif.children.push(dot(CENTER_X + 92.0, CENTER_Y - 72.0, 4.0));
        }
        EmbodimentActivity::Predicting => {
            for offset in [-34.0, 0.0, 34.0] {
                motif.children.push(
                    SceneNode::path(format!(
                        "M {:.1} {:.1} L {:.1} {:.1} L {:.1} {:.1}",
                        CENTER_X,
                        CENTER_Y - 96.0,
                        CENTER_X + offset * 0.45,
                        CENTER_Y - 128.0,
                        CENTER_X + offset,
                        CENTER_Y - 148.0
                    ))
                    .with_style(thin.clone()),
                );
                motif.children.push(dot(
                    CENTER_X + offset,
                    CENTER_Y - 148.0,
                    3.2,
                ));
            }
        }
        EmbodimentActivity::Planning => {
            let points = (0..6)
                .map(|index| {
                    let angle = index as f32 * TAU / 6.0 - TAU / 4.0;
                    (
                        CENTER_X + angle.cos() * 25.0,
                        CENTER_Y - 130.0 + angle.sin() * 18.0,
                    )
                })
                .collect();
            motif.children.push(
                SceneNode::polygon(points, true).with_style(Style {
                    fill: Some(color.with_alpha(0.12)),
                    stroke: Some(color.with_alpha(0.90)),
                    stroke_width: Some(1.5),
                    ..Style::default()
                }),
            );
            motif.children.push(dot(CENTER_X, CENTER_Y - 130.0, 3.0));
        }
        EmbodimentActivity::ExecutingTool => {
            for offset in [-42.0, 0.0, 42.0] {
                let x = CENTER_X + offset;
                motif.children.push(
                    SceneNode::line(CENTER_X, CENTER_Y - 86.0, x, CENTER_Y - 132.0)
                        .with_style(thin.clone()),
                );
                motif.children.push(
                    SceneNode::rect(x - 5.0, CENTER_Y - 137.0, 10.0, 10.0)
                        .with_style(Style {
                            fill: Some(color.with_alpha(0.16)),
                            stroke: Some(color.with_alpha(0.94)),
                            stroke_width: Some(1.3),
                            ..Style::default()
                        }),
                );
            }
        }
        EmbodimentActivity::AwaitingTool => {
            motif.children.push(
                SceneNode::circle(CENTER_X, CENTER_Y - 132.0, 11.0)
                    .with_style(thin.clone()),
            );
            motif.children.push(dot(CENTER_X, CENTER_Y - 132.0, 2.8));
        }
        EmbodimentActivity::Interrupted => {
            motif.children.push(
                SceneNode::line(
                    CENTER_X - 30.0,
                    CENTER_Y - 132.0,
                    CENTER_X - 7.0,
                    CENTER_Y - 132.0,
                )
                .with_style(thin.clone()),
            );
            motif.children.push(
                SceneNode::line(
                    CENTER_X + 7.0,
                    CENTER_Y - 132.0,
                    CENTER_X + 30.0,
                    CENTER_Y - 132.0,
                )
                .with_style(thin.clone()),
            );
            motif.children.push(dot(CENTER_X, CENTER_Y - 132.0, 3.0));
        }
        EmbodimentActivity::Fault => {
            motif.children.push(
                SceneNode::line(
                    CENTER_X - 18.0,
                    CENTER_Y - 150.0,
                    CENTER_X + 18.0,
                    CENTER_Y - 114.0,
                )
                .with_style(thin.clone()),
            );
            motif.children.push(
                SceneNode::line(
                    CENTER_X + 18.0,
                    CENTER_Y - 150.0,
                    CENTER_X - 18.0,
                    CENTER_Y - 114.0,
                )
                .with_style(thin),
            );
        }
    }

    motif
}

/// Build the Mycelial Presence scene graph from explicit embodiment signals.
///
/// Absent activity defaults to a quiet visual presentation, not an inferred
/// runtime state. Uncertainty and speech modulation are rendered only when the
/// corresponding observed signal is present.
pub fn build_mycelial_presence(snapshot: &EmbodimentSnapshot) -> SceneNode {
    let activity = snapshot
        .current_activity()
        .unwrap_or(EmbodimentActivity::Idle);
    let activity_color = color_for_activity(activity);

    let mut root = SceneNode::group(Some("mycelial-presence"));
    root.children.push(radial_gradient(
        "mycelium-background",
        Color::from_hsl(222.0, 0.34, 0.025),
        Color::from_hsl(210.0, 0.42, 0.10),
    ));
    root.children.push(radial_gradient(
        "mycelium-core",
        Color::from_hsl(43.0, 0.88, 0.78),
        activity_color,
    ));
    root.children.push(
        SceneNode::rect(0.0, 0.0, VIEWPORT_W, VIEWPORT_H).with_style(Style {
            fill_url: Some("mycelium-background".to_string()),
            ..Style::default()
        }),
    );

    let mut hyphae = SceneNode::group(Some("hyphae"));
    hyphae.style.opacity = Some(if activity == EmbodimentActivity::Idle {
        0.70
    } else {
        0.96
    });
    let base_color = Color::from_hsl(184.0, 0.82, 0.62);
    for branch in 0..PRIMARY_BRANCHES {
        let angle = (branch as f32 / PRIMARY_BRANCHES as f32) * TAU - TAU / 4.0;
        add_branch(
            &mut hyphae,
            CENTER_X,
            CENTER_Y,
            angle,
            112.0,
            BRANCH_DEPTH,
            branch,
            base_color,
        );
    }
    root.children.push(hyphae);

    let mut rings = SceneNode::group(Some("rings"));
    for ring in 0..3 {
        let offset = ring as f32 * 17.0;
        rings.children.push(
            SceneNode::ellipse(
                CENTER_X,
                CENTER_Y,
                58.0 + offset,
                43.0 + offset,
            )
            .with_style(Style {
                fill: None,
                stroke: Some(activity_color.with_alpha(0.10 + ring as f32 * 0.035)),
                stroke_width: Some(0.8 + ring as f32 * 0.25),
                ..Style::default()
            }),
        );
    }
    root.children.push(rings);

    if activity != EmbodimentActivity::Idle {
        let mut activity_halo = SceneNode::group(Some("activity-halo"));
        activity_halo.style.opacity = Some(0.25);
        activity_halo.children.push(
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

    if let Some(uncertainty) = &snapshot.uncertainty {
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

    let speech_weight = snapshot
        .viseme
        .as_ref()
        .filter(|_| {
            snapshot.current_speech_output() == Some(crate::embodiment::SpeechOutputState::Playing)
        })
        .map(|observed| finite_unit(observed.value.weight))
        .unwrap_or(0.0);

    let mut bloom = SceneNode::group(Some("bloom"));
    bloom.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 25.0 + speech_weight * 7.0)
            .with_style(Style {
                fill_url: Some("mycelium-core".to_string()),
                stroke: Some(Color::from_hsl(178.0, 0.92, 0.78).with_alpha(0.78)),
                stroke_width: Some(1.5),
                ..Style::default()
            }),
    );
    bloom.children.push(
        SceneNode::circle(CENTER_X, CENTER_Y, 8.0 + speech_weight * 2.0).with_style(Style {
            fill: Some(Color::from_hsl(48.0, 0.95, 0.88)),
            opacity: Some(0.90),
            ..Style::default()
        }),
    );
    root.children.push(bloom);

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
    let scene = build_mycelial_presence(snapshot);
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

#[allow(clippy::too_many_arguments)]
fn add_branch(
    parent: &mut SceneNode,
    x: f32,
    y: f32,
    angle: f32,
    length: f32,
    depth: usize,
    seed: usize,
    base_color: Color,
) {
    let phase = ((seed.wrapping_mul(37) + depth.wrapping_mul(19)) % 9) as f32 - 4.0;
    let bend = phase * 0.018;
    let angle_a = angle + bend;
    let angle_b = angle + bend * 1.4;
    let end_x = x + angle.cos() * length;
    let end_y = y + angle.sin() * length;
    let control_1_x = x + angle_a.cos() * length * 0.34;
    let control_1_y = y + angle_a.sin() * length * 0.34;
    let control_2_x = x + angle_b.cos() * length * 0.76;
    let control_2_y = y + angle_b.sin() * length * 0.76;
    let path = format!(
        concat!(
            "M {x:.2} {y:.2} C {control_1_x:.2} {control_1_y:.2}, ",
            "{control_2_x:.2} {control_2_y:.2}, {end_x:.2} {end_y:.2}"
        )
    );
    let depth_factor = depth as f32 / BRANCH_DEPTH as f32;
    let branch_color = base_color.lerp(
        &Color::from_hsl(274.0, 0.58, 0.70),
        1.0 - depth_factor,
    );
    parent.children.push(SceneNode::path(path).with_style(Style {
        fill: None,
        stroke: Some(branch_color.with_alpha(0.40 + depth_factor * 0.42)),
        stroke_width: Some(0.7 + depth as f32 * 0.55),
        ..Style::default()
    }));

    if depth == 0 {
        parent.children.push(
            SceneNode::circle(end_x, end_y, 2.1).with_style(Style {
                fill: Some(branch_color.with_alpha(0.90)),
                ..Style::default()
            }),
        );
        return;
    }

    let child_length = length * 0.66;
    let fork = 0.42 + ((seed % 3) as f32 * 0.035);
    add_branch(
        parent,
        end_x,
        end_y,
        angle - fork,
        child_length,
        depth - 1,
        seed.wrapping_mul(2).wrapping_add(1),
        base_color,
    );
    add_branch(
        parent,
        end_x,
        end_y,
        angle + fork,
        child_length,
        depth - 1,
        seed.wrapping_mul(2).wrapping_add(2),
        base_color,
    );
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
                NodeKind::Group { id } if id.as_deref().is_some_and(|value| value.starts_with("motif-")) => {
                    id.clone()
                }
                _ => None,
            })
        };

        assert_eq!(motif_id(&associating_scene).as_deref(), Some("motif-associating"));
        assert_eq!(motif_id(&predicting_scene).as_deref(), Some("motif-predicting"));
        assert_ne!(
            associating_scene.children.last().unwrap().node_count(),
            predicting_scene.children.last().unwrap().node_count()
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
        assert!(!scene
            .children
            .iter()
            .any(|child| {
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
}
