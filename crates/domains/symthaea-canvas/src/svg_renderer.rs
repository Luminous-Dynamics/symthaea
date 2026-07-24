// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SceneNode → self-contained animated SVG string.

use std::fmt::Write;

use crate::animation::{FrameContext, MotionPreference};
use crate::scene_graph::{FilterType, GradientStop, NodeKind, SceneNode, Style};

/// Viewport dimensions for the generated SVG.
pub const VIEWPORT_W: f32 = 512.0;
pub const VIEWPORT_H: f32 = 512.0;

/// Temporal options for self-contained SVG animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgRenderOptions {
    pub elapsed_seconds: f64,
    pub motion: MotionPreference,
    pub instance_id: Option<u64>,
}

impl SvgRenderOptions {
    pub fn from_frame(frame: FrameContext) -> Self {
        Self {
            elapsed_seconds: frame.elapsed_seconds,
            motion: frame.motion,
            instance_id: frame.instance_id,
        }
    }
}

impl Default for SvgRenderOptions {
    fn default() -> Self {
        Self {
            elapsed_seconds: 0.0,
            motion: MotionPreference::Full,
            instance_id: None,
        }
    }
}

/// Render a SceneNode tree to a self-contained animated SVG string.
pub fn render_svg(root: &SceneNode, consciousness: f64) -> String {
    render_svg_with_options(root, consciousness, SvgRenderOptions::default())
}

/// Render with an explicit timeline so replacing the SVG does not restart motion.
pub fn render_svg_with_options(
    root: &SceneNode,
    consciousness: f64,
    options: SvgRenderOptions,
) -> String {
    let mut buf = String::with_capacity(4096);
    write_header(&mut buf, consciousness);
    write_css_animations(&mut buf, consciousness, options);
    write_node(&mut buf, root, 1, options);
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
fn write_css_animations(buf: &mut String, consciousness: f64, options: SvgRenderOptions) {
    let consciousness = if consciousness.is_finite() {
        consciousness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bloom_min = if consciousness > 0.5 { 0.85 } else { 0.9 };
    let bloom_max = if consciousness > 0.5 { 1.15 } else { 1.1 };
    let breath_dur = 2.0; // seconds
    let orbit_dur = 8.0;
    let particle_dur = 3.0;

    if options.motion == MotionPreference::Reduced {
        let bloom_id = qualified_id(options, "bloom");
        let rings_id = qualified_id(options, "rings");
        let particles_id = qualified_id(options, "particles");
        let _ = writeln!(
            buf,
            "  <style>#{bloom_id}, #{rings_id} > ellipse, #{particles_id} > circle {{ animation: none; }}</style>"
        );
        return;
    }

    let elapsed = if options.elapsed_seconds.is_finite() {
        options.elapsed_seconds.max(0.0)
    } else {
        0.0
    };
    let breath_delay = -(elapsed % breath_dur);
    let orbit_delay = -(elapsed % orbit_dur);
    let particle_delay = -(elapsed % particle_dur);
    let bloom_id = qualified_id(options, "bloom");
    let rings_id = qualified_id(options, "rings");
    let particles_id = qualified_id(options, "particles");
    let breathe_name = qualified_id(options, "breathe");
    let orbit_name = qualified_id(options, "orbit");
    let particle_orbit_name = qualified_id(options, "particle-orbit");

    let _ = write!(
        buf,
        r##"  <style>
    @keyframes {breathe_name} {{
      0%, 100% {{ transform: scale({bloom_min:.2}); }}
      50% {{ transform: scale({bloom_max:.2}); }}
    }}
    @keyframes {orbit_name} {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
    @keyframes {particle_orbit_name} {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
    #{bloom_id} {{ animation: {breathe_name} {breath_dur:.1}s ease-in-out infinite; animation-delay: {breath_delay:.3}s; transform-origin: center; }}
    #{rings_id} > ellipse {{ animation: {orbit_name} {orbit_dur:.1}s linear infinite; animation-delay: {orbit_delay:.3}s; transform-origin: center; }}
    #{particles_id} > circle {{ animation: {particle_orbit_name} {particle_dur:.1}s linear infinite; animation-delay: {particle_delay:.3}s; transform-origin: center; }}
    @media (prefers-reduced-motion: reduce) {{
      #{bloom_id}, #{rings_id} > ellipse, #{particles_id} > circle {{ animation: none; }}
    }}
  </style>
"##,
    );
}

fn write_node(buf: &mut String, node: &SceneNode, depth: usize, options: SvgRenderOptions) {
    let indent = "  ".repeat(depth);
    match &node.kind {
        NodeKind::Group { id } => {
            let _ = write!(buf, "{indent}<g");
            if let Some(id) = id {
                let _ = write!(
                    buf,
                    r#" id="{}""#,
                    escape_xml_attr(&qualified_id(options, id))
                );
            }
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str(">\n");
            for child in &node.children {
                write_node(buf, child, depth + 1, options);
            }
            let _ = writeln!(buf, "{indent}</g>");
        }
        NodeKind::Circle { cx, cy, r } => {
            let (cx, cy, r) = (finite(*cx, 0.0), finite(*cy, 0.0), nonnegative(*r));
            let _ = write!(
                buf,
                r#"{indent}<circle cx="{cx:.1}" cy="{cy:.1}" r="{r:.1}""#
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            let (cx, cy, rx, ry) = (
                finite(*cx, 0.0),
                finite(*cy, 0.0),
                nonnegative(*rx),
                nonnegative(*ry),
            );
            let _ = write!(
                buf,
                r#"{indent}<ellipse cx="{cx:.1}" cy="{cy:.1}" rx="{rx:.1}" ry="{ry:.1}""#,
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            let (x1, y1, x2, y2) = (
                finite(*x1, 0.0),
                finite(*y1, 0.0),
                finite(*x2, 0.0),
                finite(*y2, 0.0),
            );
            let _ = write!(
                buf,
                r#"{indent}<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}""#,
            );
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
        NodeKind::Polygon { points, closed } => {
            let tag = if *closed { "polygon" } else { "polyline" };
            let _ = write!(buf, "{indent}<{tag} points=\"");
            for (i, (x, y)) in points.iter().enumerate() {
                if i > 0 {
                    buf.push(' ');
                }
                let _ = write!(buf, "{:.1},{:.1}", finite(*x, 0.0), finite(*y, 0.0));
            }
            buf.push('"');
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
        NodeKind::Rect { x, y, w, h, rx } => {
            let (x, y, w, h, rx) = (
                finite(*x, 0.0),
                finite(*y, 0.0),
                nonnegative(*w),
                nonnegative(*h),
                nonnegative(*rx),
            );
            let _ = write!(
                buf,
                r#"{indent}<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}""#,
            );
            if rx > 0.0 {
                let _ = write!(buf, r#" rx="{rx:.1}""#);
            }
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
        NodeKind::RadialGradient { id, stops } => {
            let _ = writeln!(buf, r#"{indent}<defs>"#);
            let _ = write!(
                buf,
                r#"{indent}  <radialGradient id="{}">"#,
                escape_xml_attr(&qualified_id(options, id))
            );
            buf.push('\n');
            for stop in stops {
                write_gradient_stop(buf, stop, depth + 2);
            }
            let _ = writeln!(buf, "{indent}  </radialGradient>");
            let _ = writeln!(buf, "{indent}</defs>");
        }
        NodeKind::Filter { id, filter_type } => {
            let _ = writeln!(buf, "{indent}<defs>");
            let _ = writeln!(
                buf,
                r#"{indent}  <filter id="{}">"#,
                escape_xml_attr(&qualified_id(options, id))
            );
            match filter_type {
                FilterType::Turbulence {
                    base_frequency,
                    num_octaves,
                    scale,
                } => {
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feTurbulence type="turbulence" baseFrequency="{:.4}" numOctaves="{num_octaves}" result="turb"/>"#,
                        nonnegative(*base_frequency),
                    );
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feDisplacementMap in="SourceGraphic" in2="turb" scale="{:.1}"/>"#,
                        nonnegative(*scale),
                    );
                }
                FilterType::Blur { std_dev } => {
                    let _ = writeln!(
                        buf,
                        r#"{indent}    <feGaussianBlur stdDeviation="{:.1}"/>"#,
                        nonnegative(*std_dev),
                    );
                }
            }
            let _ = writeln!(buf, "{indent}  </filter>");
            let _ = writeln!(buf, "{indent}</defs>");
        }
        NodeKind::UseFilter { filter_id } => {
            // Applied via style.filter on other nodes — this is a no-op placeholder
            let _ = writeln!(
                buf,
                r#"{indent}<!-- filter ref: {} -->"#,
                escape_xml_text(&qualified_id(options, filter_id))
            );
        }
        NodeKind::Path { d } => {
            let _ = write!(buf, r#"{indent}<path d="{}""#, escape_xml_attr(d));
            write_transform(buf, node);
            write_style_attrs(buf, &node.style, options);
            buf.push_str("/>\n");
        }
    }
}

fn write_transform(buf: &mut String, node: &SceneNode) {
    let transform = node.transform.to_svg();
    if !transform.is_empty() {
        let _ = write!(buf, r#" transform="{}""#, transform);
    }
}

fn write_style_attrs(buf: &mut String, style: &Style, options: SvgRenderOptions) {
    if let Some(url) = &style.fill_url {
        let _ = write!(
            buf,
            r#" fill="url(#{})""#,
            escape_xml_attr(&qualified_id(options, url))
        );
    } else if let Some(fill) = &style.fill {
        let _ = write!(buf, r#" fill="{}""#, fill.to_css());
    }
    if let Some(stroke) = &style.stroke {
        let _ = write!(buf, r#" stroke="{}""#, stroke.to_css());
    }
    if let Some(sw) = style.stroke_width {
        let _ = write!(buf, r#" stroke-width="{:.2}""#, nonnegative(sw));
    }
    if let Some(op) = style.opacity {
        let _ = write!(buf, r#" opacity="{:.2}""#, unit(op));
    }
    if let Some(f) = &style.filter {
        let _ = write!(
            buf,
            r#" filter="url(#{})""#,
            escape_xml_attr(&qualified_id(options, f))
        );
    }
    if let Some(cls) = &style.css_class {
        let _ = write!(buf, r#" class="{}""#, escape_xml_attr(cls));
    }
}

fn write_gradient_stop(buf: &mut String, stop: &GradientStop, depth: usize) {
    let indent = "  ".repeat(depth);
    let _ = writeln!(
        buf,
        r#"{indent}<stop offset="{:.0}%" stop-color="{}"/>"#,
        unit(stop.offset) * 100.0,
        stop.color.to_css(),
    );
}

fn qualified_id(options: SvgRenderOptions, id: &str) -> String {
    match options.instance_id {
        Some(instance_id) => format!("canvas-{instance_id}-{id}"),
        None => id.to_string(),
    }
}

fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn nonnegative(value: f32) -> f32 {
    finite(value, 0.0).max(0.0)
}

fn unit(value: f32) -> f32 {
    finite(value, 0.0).clamp(0.0, 1.0)
}

fn escape_xml_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace("--", "—")
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
    #[test]
    fn explicit_timeline_sets_negative_animation_delays() {
        let root = SceneNode::group(Some("root"));
        let svg = render_svg_with_options(
            &root,
            0.5,
            SvgRenderOptions {
                elapsed_seconds: 3.25,
                motion: MotionPreference::Full,
                instance_id: None,
            },
        );
        assert!(svg.contains("animation-delay: -1.250s"));
        assert!(svg.contains("prefers-reduced-motion"));
    }

    #[test]
    fn reduced_motion_disables_animation() {
        let root = SceneNode::group(Some("root"));
        let svg = render_svg_with_options(
            &root,
            0.5,
            SvgRenderOptions {
                elapsed_seconds: 1.0,
                motion: MotionPreference::Reduced,
                instance_id: None,
            },
        );
        assert!(svg.contains("animation: none"));
        assert!(!svg.contains("@keyframes"));
    }

    #[test]
    fn namespaces_ids_references_selectors_and_keyframes() {
        let mut root = SceneNode::group(Some("bloom"));
        root.children.push(SceneNode {
            kind: NodeKind::RadialGradient {
                id: "bg-grad".into(),
                stops: vec![],
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: vec![],
        });
        root.children
            .push(SceneNode::rect(0.0, 0.0, 10.0, 10.0).with_style(Style {
                fill_url: Some("bg-grad".into()),
                ..Style::default()
            }));
        let svg = render_svg_with_options(
            &root,
            0.5,
            SvgRenderOptions {
                elapsed_seconds: 0.0,
                motion: MotionPreference::Full,
                instance_id: Some(42),
            },
        );
        assert!(svg.contains(r#"id="canvas-42-bloom""#));
        assert!(svg.contains(r#"id="canvas-42-bg-grad""#));
        assert!(svg.contains("url(#canvas-42-bg-grad)"));
        assert!(svg.contains("#canvas-42-bloom"));
        assert!(svg.contains("@keyframes canvas-42-breathe"));
    }

    #[test]
    fn escapes_untrusted_attributes() {
        let mut group = SceneNode::group(Some("root\" onload=\"alert(1)"));
        group.style.css_class = Some("x\" onclick=\"bad()".into());
        group
            .children
            .push(SceneNode::path("M 0 0 \"/><script>bad()</script>"));
        let svg = render_svg(&group, 0.5);
        assert!(!svg.contains("<script>"));
        assert!(!svg.contains(" onload=\"alert"));
        assert!(svg.contains("&quot;"));
        assert!(svg.contains("&lt;script&gt;"));
    }

    #[test]
    fn non_finite_geometry_is_not_serialized() {
        let root = SceneNode::group(None).with_child(SceneNode::circle(
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ));
        let svg = render_svg(&root, f64::NAN);
        assert!(!svg.contains("NaN"));
        assert!(!svg.contains("inf"));
        assert!(svg.contains(r#"cx="0.0""#));
        assert!(svg.contains(r#"r="0.0""#));
    }
}
