// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SceneNode → self-contained animated SVG string.

use std::fmt::Write;

use crate::scene_graph::{FilterType, GradientStop, NodeKind, SceneNode, Style};

/// Viewport dimensions for the generated SVG.
pub const VIEWPORT_W: f32 = 512.0;
pub const VIEWPORT_H: f32 = 512.0;

/// Render a SceneNode tree to a self-contained animated SVG string.
pub fn render_svg(root: &SceneNode, consciousness: f64) -> String {
    let mut buf = String::with_capacity(4096);
    write_header(&mut buf, consciousness);
    write_css_animations(&mut buf, consciousness);
    write_node(&mut buf, root, 1);
    buf.push_str("</svg>\n");
    buf
}

fn write_header(buf: &mut String, _consciousness: f64) {
    let _ = write!(
        buf,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWPORT_W} {VIEWPORT_H}" width="{VIEWPORT_W}" height="{VIEWPORT_H}">"#,
    );
    buf.push('\n');
}

/// Inject CSS `@keyframes` for breathing bloom, orbital rotation, and particle orbits.
fn write_css_animations(buf: &mut String, consciousness: f64) {
    let bloom_min = if consciousness > 0.5 { 0.85 } else { 0.9 };
    let bloom_max = if consciousness > 0.5 { 1.15 } else { 1.1 };
    let breath_dur = 2.0; // seconds
    let orbit_dur = 8.0;
    let particle_dur = 3.0;

    let _ = write!(
        buf,
        r##"  <style>
    @keyframes breathe {{
      0%, 100% {{ transform: scale({bloom_min:.2}); }}
      50% {{ transform: scale({bloom_max:.2}); }}
    }}
    @keyframes orbit {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
    @keyframes particle-orbit {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
    #bloom {{ animation: breathe {breath_dur:.1}s ease-in-out infinite; transform-origin: center; }}
    #rings > ellipse {{ animation: orbit {orbit_dur:.1}s linear infinite; transform-origin: center; }}
    #particles > circle {{ animation: particle-orbit {particle_dur:.1}s linear infinite; transform-origin: center; }}
  </style>
"##,
    );
}

fn write_node(buf: &mut String, node: &SceneNode, depth: usize) {
    let indent = "  ".repeat(depth);
    match &node.kind {
        NodeKind::Group { id } => {
            let _ = write!(buf, "{indent}<g");
            if let Some(id) = id {
                let _ = write!(buf, r#" id="{id}""#);
            }
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str(">\n");
            for child in &node.children {
                write_node(buf, child, depth + 1);
            }
            let _ = writeln!(buf, "{indent}</g>");
        }
        NodeKind::Circle { cx, cy, r } => {
            let _ = write!(
                buf,
                r#"{indent}<circle cx="{cx:.1}" cy="{cy:.1}" r="{r:.1}""#
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            let _ = write!(
                buf,
                r#"{indent}<ellipse cx="{cx:.1}" cy="{cy:.1}" rx="{rx:.1}" ry="{ry:.1}""#,
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            let _ = write!(
                buf,
                r#"{indent}<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}""#,
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
        NodeKind::Polygon { points, closed } => {
            let tag = if *closed { "polygon" } else { "polyline" };
            let _ = write!(buf, "{indent}<{tag} points=\"");
            for (i, (x, y)) in points.iter().enumerate() {
                if i > 0 {
                    buf.push(' ');
                }
                let _ = write!(buf, "{x:.1},{y:.1}");
            }
            buf.push('"');
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
        NodeKind::Rect { x, y, w, h, rx } => {
            let _ = write!(
                buf,
                r#"{indent}<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}""#,
            );
            if *rx > 0.0 {
                let _ = write!(buf, r#" rx="{rx:.1}""#);
            }
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
        NodeKind::RadialGradient { id, stops } => {
            let _ = writeln!(buf, r#"{indent}<defs>"#);
            let _ = write!(buf, r#"{indent}  <radialGradient id="{id}">"#);
            buf.push('\n');
            for stop in stops {
                write_gradient_stop(buf, stop, depth + 2);
            }
            let _ = writeln!(buf, "{indent}  </radialGradient>");
            let _ = writeln!(buf, "{indent}</defs>");
        }
        NodeKind::Filter { id, filter_type } => {
            let _ = writeln!(buf, "{indent}<defs>");
            let _ = writeln!(buf, r#"{indent}  <filter id="{id}">"#);
            match filter_type {
                FilterType::Turbulence {
                    base_frequency,
                    num_octaves,
                    scale,
                } => {
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feTurbulence type="turbulence" baseFrequency="{base_frequency:.4}" numOctaves="{num_octaves}" result="turb"/>"#,
                    );
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feDisplacementMap in="SourceGraphic" in2="turb" scale="{scale:.1}"/>"#,
                    );
                }
                FilterType::Blur { std_dev } => {
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feGaussianBlur stdDeviation="{std_dev:.1}"/>"#,
                    );
                }
            }
            let _ = writeln!(buf, "{indent}  </filter>");
            let _ = writeln!(buf, "{indent}</defs>");
        }
        NodeKind::UseFilter { filter_id } => {
            // Applied via style.filter on other nodes — this is a no-op placeholder
            let _ = writeln!(buf, r#"{indent}<!-- filter ref: {filter_id} -->"#);
        }
        NodeKind::Path { d } => {
            let _ = write!(buf, r#"{indent}<path d="{d}""#);
            write_transform(buf, node);
            write_style_attrs(buf, &node.style);
            buf.push_str("/>\n");
        }
    }
}

fn write_transform(buf: &mut String, node: &SceneNode) {
    if !node.transform.is_identity() {
        let _ = write!(buf, r#" transform="{}""#, node.transform.to_svg());
    }
}

fn write_style_attrs(buf: &mut String, style: &Style) {
    if let Some(url) = &style.fill_url {
        let _ = write!(buf, r#" fill="url(#{url})""#);
    } else if let Some(fill) = &style.fill {
        let _ = write!(buf, r#" fill="{}""#, fill.to_css());
    }
    if let Some(stroke) = &style.stroke {
        let _ = write!(buf, r#" stroke="{}""#, stroke.to_css());
    }
    if let Some(sw) = style.stroke_width {
        let _ = write!(buf, r#" stroke-width="{sw:.2}""#);
    }
    if let Some(op) = style.opacity {
        let _ = write!(buf, r#" opacity="{op:.2}""#);
    }
    if let Some(f) = &style.filter {
        let _ = write!(buf, r#" filter="url(#{f})""#);
    }
    if let Some(cls) = &style.css_class {
        let _ = write!(buf, r#" class="{cls}""#);
    }
}

fn write_gradient_stop(buf: &mut String, stop: &GradientStop, depth: usize) {
    let indent = "  ".repeat(depth);
    let _ = writeln!(
        buf,
        r#"{indent}<stop offset="{:.0}%" stop-color="{}"/>"#,
        stop.offset * 100.0,
        stop.color.to_css(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::scene_graph::Transform;

    #[test]
    fn renders_svg_wrapper() {
        let root = SceneNode::group(Some("root"));
        let svg = render_svg(&root, 0.5);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("viewBox"));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn renders_circle() {
        let root = SceneNode::group(None).with_child(SceneNode::circle(100.0, 200.0, 50.0));
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains(r#"cx="100.0""#));
        assert!(svg.contains(r#"cy="200.0""#));
        assert!(svg.contains(r#"r="50.0""#));
    }

    #[test]
    fn renders_fill_color() {
        let c = SceneNode::circle(0.0, 0.0, 10.0).with_style(Style {
            fill: Some(Color::rgb(1.0, 0.0, 0.0)),
            ..Style::default()
        });
        let root = SceneNode::group(None).with_child(c);
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains("fill=\"#ff0000\""));
    }

    #[test]
    fn renders_transform() {
        let c = SceneNode::circle(0.0, 0.0, 10.0).with_transform(Transform {
            translate_x: 256.0,
            translate_y: 256.0,
            rotate_deg: 0.0,
            scale: 1.0,
        });
        let root = SceneNode::group(None).with_child(c);
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains("translate(256.0,256.0)"));
    }

    #[test]
    fn renders_polygon() {
        let p = SceneNode::polygon(vec![(0.0, 0.0), (100.0, 0.0), (50.0, 86.6)], true);
        let root = SceneNode::group(None).with_child(p);
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("points="));
    }

    #[test]
    fn svg_no_unclosed_tags() {
        let root = SceneNode::group(Some("test"))
            .with_child(SceneNode::circle(10.0, 10.0, 5.0))
            .with_child(SceneNode::rect(0.0, 0.0, 100.0, 100.0));
        let svg = render_svg(&root, 0.5);
        // Every <g must have </g>
        let open_g = svg.matches("<g").count();
        let close_g = svg.matches("</g>").count();
        assert_eq!(open_g, close_g, "unclosed <g> tags");
    }

    #[test]
    fn renders_ellipse() {
        let e = SceneNode::ellipse(256.0, 256.0, 100.0, 60.0);
        let root = SceneNode::group(None).with_child(e);
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains("<ellipse"));
        assert!(svg.contains(r#"rx="100.0""#));
    }

    #[test]
    fn renders_gradient() {
        let grad = SceneNode {
            kind: NodeKind::RadialGradient {
                id: "bg".to_string(),
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: Color::rgb(0.0, 0.0, 0.0),
                    },
                    GradientStop {
                        offset: 1.0,
                        color: Color::rgb(0.1, 0.1, 0.2),
                    },
                ],
            },
            transform: crate::scene_graph::Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        };
        let root = SceneNode::group(None).with_child(grad);
        let svg = render_svg(&root, 0.5);
        assert!(svg.contains("radialGradient"));
        assert!(svg.contains(r#"id="bg""#));
        assert!(svg.contains("<stop"));
    }
}
