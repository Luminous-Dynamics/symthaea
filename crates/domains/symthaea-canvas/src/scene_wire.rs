// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, renderer-neutral scene wire representation.
//!
//! This is the intended network boundary for the cognitive canvas. It carries
//! semantic geometry rather than SVG/XML or backend-specific GPU buffers.
//! Consumers can render it with WebGPU, SVG, Canvas2D, native GPU APIs, etc.

use serde::{Deserialize, Serialize};

use crate::color::Color;
use crate::scene_graph::{NodeKind, SceneNode, Style, Transform};

const MAX_SCENE_NODES: usize = 256;
const MAX_SCENE_DEPTH: usize = 24;
const MAX_POLYGON_POINTS: usize = 128;
const MAX_PATH_BYTES: usize = 12 * 1024;
const MAX_SCENE_BYTES: usize = 128 * 1024;
const MAX_ABS_COORDINATE: f32 = 1_000_000.0;
const MAX_ABS_SCALE: f32 = 8.0;
const MIN_RENDERABLE_LINE_LENGTH: f64 = 1e-6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct WireTransform {
    pub translate_x: f32,
    pub translate_y: f32,
    pub rotate_deg: f32,
    pub scale: f32,
}

impl Default for WireTransform {
    fn default() -> Self {
        Self {
            translate_x: 0.0,
            translate_y: 0.0,
            rotate_deg: 0.0,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireStyle {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: Option<f32>,
    pub opacity: f32,
}

impl Default for WireStyle {
    fn default() -> Self {
        Self {
            fill: None,
            stroke: None,
            stroke_width: None,
            opacity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WirePrimitive {
    Group,
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    },
    Polygon {
        points: Vec<[f32; 2]>,
        closed: bool,
    },
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rx: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireNode {
    pub primitive: WirePrimitive,
    pub transform: WireTransform,
    pub style: WireStyle,
    pub children: Vec<WireNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteScene {
    pub version: u16,
    pub root: WireNode,
}

impl RemoteScene {
    pub const VERSION: u16 = 1;

    /// Compile a SceneNode into a bounded semantic wire scene.
    ///
    /// Unsupported effect/path definitions are omitted deliberately. The
    /// resulting document is self-contained and backend-neutral.
    pub fn from_scene(scene: &SceneNode) -> Self {
        let mut node_count = 0usize;
        let gradients = collect_first_gradient_colors(scene);
        let root = compile_node(scene, 0, &mut node_count, &gradients);
        let root = root.unwrap_or_else(|| WireNode {
            primitive: WirePrimitive::Group,
            transform: WireTransform::default(),
            style: WireStyle::default(),
            children: Vec::new(),
        });
        let mut remote = Self {
            version: Self::VERSION,
            root,
        };

        // Serialization size is part of the protocol contract. Never emit a
        // partially serialized scene; replace it atomically with an inert root.
        if remote.is_supported() {
            remote
        } else {
            remote.root = WireNode {
                primitive: WirePrimitive::Group,
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: Vec::new(),
            };
            remote
        }
    }

    /// Reconstruct a native SceneNode from the bounded wire representation.
    pub fn to_scene_node(&self) -> SceneNode {
        if !self.is_supported() {
            return SceneNode::group(None);
        }
        wire_to_scene_node(&self.root)
    }

    pub fn serialized_len(&self) -> usize {
        let mut writer = BoundedCountWriter::new(MAX_SCENE_BYTES);
        if serde_json::to_writer(&mut writer, self).is_err() {
            MAX_SCENE_BYTES + 1
        } else {
            writer.len()
        }
    }

    pub fn is_supported(&self) -> bool {
        if self.version != Self::VERSION {
            return false;
        }

        // Validate the bounded graph before serializing it. This keeps a
        // caller-supplied oversized object from forcing a large temporary JSON
        // allocation merely to discover that it is structurally invalid.
        let mut count = 0usize;
        if !validate_node(&self.root, 0, &mut count, 1.0) {
            return false;
        }

        self.serialized_len() <= MAX_SCENE_BYTES
    }

    pub fn is_within_budget(&self) -> bool {
        self.serialized_len() <= MAX_SCENE_BYTES
    }
}

struct BoundedCountWriter {
    written: usize,
    limit: usize,
}

impl BoundedCountWriter {
    fn new(limit: usize) -> Self {
        Self { written: 0, limit }
    }

    fn len(&self) -> usize {
        self.written
    }
}

impl std::io::Write for BoundedCountWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let Some(next) = self.written.checked_add(buf.len()) else {
            self.written = self.limit.saturating_add(1);
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "serialized scene size overflow",
            ));
        };
        if next > self.limit {
            self.written = self.limit.saturating_add(1);
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "serialized scene exceeds bounded size",
            ));
        }
        self.written = next;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn compile_node(
    node: &SceneNode,
    depth: usize,
    count: &mut usize,
    gradients: &std::collections::HashMap<&str, Color>,
) -> Option<WireNode> {
    if depth > MAX_SCENE_DEPTH || *count >= MAX_SCENE_NODES {
        return None;
    }
    *count += 1;

    let primitive = match &node.kind {
        NodeKind::Group { .. } => WirePrimitive::Group,
        NodeKind::Circle { cx, cy, r } => WirePrimitive::Circle {
            cx: bounded(*cx),
            cy: bounded(*cy),
            r: nonnegative(*r),
        },
        NodeKind::Ellipse { cx, cy, rx, ry } => WirePrimitive::Ellipse {
            cx: bounded(*cx),
            cy: bounded(*cy),
            rx: nonnegative(*rx),
            ry: nonnegative(*ry),
        },
        NodeKind::Line { x1, y1, x2, y2 } => WirePrimitive::Line {
            x1: bounded(*x1),
            y1: bounded(*y1),
            x2: bounded(*x2),
            y2: bounded(*y2),
        },
        NodeKind::Polygon { points, closed } => {
            if points.len() < 2 || points.len() > MAX_POLYGON_POINTS {
                return None;
            }
            WirePrimitive::Polygon {
                points: points.iter().map(|&(x, y)| [bounded(x), bounded(y)]).collect(),
                closed: *closed,
            }
        }
        NodeKind::Rect { x, y, w, h, rx } => WirePrimitive::Rect {
            x: bounded(*x),
            y: bounded(*y),
            w: nonnegative(*w),
            h: nonnegative(*h),
            rx: nonnegative(*rx),
        },
        NodeKind::Path { d } => {
            if d.len() > MAX_PATH_BYTES {
                return None;
            }
            return None;
        }
        NodeKind::RadialGradient { .. } | NodeKind::Filter { .. } | NodeKind::UseFilter { .. } => {
            return None;
        }
    };

    let transform = node.transform;
    let wire_transform = WireTransform {
        translate_x: bounded(transform.translate_x),
        translate_y: bounded(transform.translate_y),
        rotate_deg: bounded(transform.rotate_deg),
        scale: bounded(transform.scale).clamp(-MAX_ABS_SCALE, MAX_ABS_SCALE),
    };

    let wire_style = WireStyle {
        fill: node
            .style
            .fill
            .or_else(|| node.style.fill_url.as_deref().and_then(|id| gradients.get(id).copied()))
            .map(|c| c.sanitized()),
        stroke: node.style.stroke.map(|c| c.sanitized()),
        stroke_width: node.style.stroke_width.map(|v| nonnegative(v)),
        opacity: node
            .style
            .opacity
            .map(|v| finite(v, 0.0).clamp(0.0, 1.0))
            .unwrap_or(1.0),
    };

    let children = node
        .children
        .iter()
        .filter_map(|child| compile_node(child, depth + 1, count, gradients))
        .collect();

    Some(WireNode {
        primitive,
        transform: wire_transform,
        style: wire_style,
        children,
    })
}

fn validate_node(
    node: &WireNode,
    depth: usize,
    count: &mut usize,
    parent_scale_abs: f64,
) -> bool {
    if depth > MAX_SCENE_DEPTH || *count >= MAX_SCENE_NODES {
        return false;
    }
    *count += 1;

    let primitive_valid = match &node.primitive {
        WirePrimitive::Group => true,
        WirePrimitive::Circle { cx, cy, r } => {
            valid_coordinate(*cx)
                && valid_coordinate(*cy)
                && valid_positive(*r)
        }
        WirePrimitive::Ellipse { cx, cy, rx, ry } => {
            valid_coordinate(*cx)
                && valid_coordinate(*cy)
                && valid_positive(*rx)
                && valid_positive(*ry)
        }
        WirePrimitive::Line { x1, y1, x2, y2 } => {
            valid_coordinate(*x1)
                && valid_coordinate(*y1)
                && valid_coordinate(*x2)
                && valid_coordinate(*y2)
                && line_is_renderable(
                    *x1,
                    *y1,
                    *x2,
                    *y2,
                    parent_scale_abs * f64::from(node.transform.scale.abs()),
                )
        }
        WirePrimitive::Polygon { points, closed } => {
            ((!*closed && points.len() >= 2) || (*closed && points.len() >= 3))
                && points.len() <= MAX_POLYGON_POINTS
                && points
                    .iter()
                    .all(|point| valid_coordinate(point[0]) && valid_coordinate(point[1]))
                && (!*closed || is_valid_closed_polygon(points))
        }
        WirePrimitive::Rect { x, y, w, h, rx } => {
            valid_coordinate(*x)
                && valid_coordinate(*y)
                && valid_positive(*w)
                && valid_positive(*h)
                && valid_nonnegative(*rx)
                && *rx <= w.min(*h) * 0.5
        }
    };

    if !primitive_valid
        || !node.transform.translate_x.is_finite()
        || !node.transform.translate_y.is_finite()
        || !node.transform.rotate_deg.is_finite()
        || !node.transform.scale.is_finite()
        || node.transform.translate_x.abs() > MAX_ABS_COORDINATE
        || node.transform.translate_y.abs() > MAX_ABS_COORDINATE
        || node.transform.rotate_deg.abs() > MAX_ABS_COORDINATE
        || node.transform.scale.abs() > MAX_ABS_SCALE
        || !node.style.opacity.is_finite()
        || !(0.0..=1.0).contains(&node.style.opacity)
        || node.style.stroke_width.is_some_and(|width| {
            !width.is_finite() || width.is_sign_negative() || width > MAX_ABS_COORDINATE
        })
        || node
            .style
            .fill
            .is_some_and(|color| !valid_color(color))
        || node
            .style
            .stroke
            .is_some_and(|color| !valid_color(color))
    {
        return false;
    }

    let local_scale_abs = f64::from(node.transform.scale.abs());
    let combined_scale_abs = parent_scale_abs * local_scale_abs;
    if !combined_scale_abs.is_finite() {
        return false;
    }

    node.children
        .iter()
        .all(|child| validate_node(child, depth + 1, count, combined_scale_abs))
}

fn is_valid_closed_polygon(points: &[[f32; 2]]) -> bool {
    let mut normalized = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if normalized
            .last()
            .is_none_or(|&last: &usize| points[last] != *point)
        {
            normalized.push(index);
        }
    }
    if normalized.len() > 1 && points[*normalized.first().unwrap()] == points[*normalized.last().unwrap()] {
        normalized.pop();
    }
    if normalized.len() < 3 {
        return false;
    }
    let area2 = normalized
        .iter()
        .enumerate()
        .map(|(index, &a)| {
            let b = normalized[(index + 1) % normalized.len()];
            f64::from(points[a][0]) * f64::from(points[b][1])
                - f64::from(points[a][1]) * f64::from(points[b][0])
        })
        .sum::<f64>();
    area2.is_finite()
        && area2.abs() > 1e-6
        && is_simple_polygon(points, &normalized)
}

fn is_simple_polygon(points: &[[f32; 2]], polygon: &[usize]) -> bool {
    let n = polygon.len();
    if n < 4 {
        return true;
    }

    for i in 0..n {
        let a = points[polygon[i]];
        let b = points[polygon[(i + 1) % n]];
        for j in (i + 1)..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let c = points[polygon[j]];
            let d = points[polygon[(j + 1) % n]];
            if segments_intersect_or_touch(a, b, c, d) {
                return false;
            }
        }
    }
    true
}

fn segments_intersect_or_touch(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
) -> bool {
    const EPSILON: f64 = 1e-6;
    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);
    if !ab_c.is_finite() || !ab_d.is_finite() || !cd_a.is_finite() || !cd_b.is_finite() {
        return false;
    }
    let proper = ((ab_c > EPSILON && ab_d < -EPSILON)
        || (ab_c < -EPSILON && ab_d > EPSILON))
        && ((cd_a > EPSILON && cd_b < -EPSILON)
            || (cd_a < -EPSILON && cd_b > EPSILON));
    if proper {
        return true;
    }
    (ab_c.abs() <= EPSILON && point_on_segment(a, b, c, EPSILON))
        || (ab_d.abs() <= EPSILON && point_on_segment(a, b, d, EPSILON))
        || (cd_a.abs() <= EPSILON && point_on_segment(c, d, a, EPSILON))
        || (cd_b.abs() <= EPSILON && point_on_segment(c, d, b, EPSILON))
}

fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
    (f64::from(b[0]) - f64::from(a[0])) * (f64::from(c[1]) - f64::from(a[1]))
        - (f64::from(b[1]) - f64::from(a[1])) * (f64::from(c[0]) - f64::from(a[0]))
}

fn point_on_segment(
    a: [f32; 2],
    b: [f32; 2],
    point: [f32; 2],
    epsilon: f64,
) -> bool {
    f64::from(point[0]) >= f64::from(a[0].min(b[0])) - epsilon
        && f64::from(point[0]) <= f64::from(a[0].max(b[0])) + epsilon
        && f64::from(point[1]) >= f64::from(a[1].min(b[1])) - epsilon
        && f64::from(point[1]) <= f64::from(a[1].max(b[1])) + epsilon
}

fn collect_first_gradient_colors(root: &SceneNode) -> std::collections::HashMap<&str, Color> {
    let mut colors = std::collections::HashMap::new();
    let mut stack = vec![(root, 0usize)];
    let mut visited = 0usize;

    while let Some((node, depth)) = stack.pop() {
        if depth > MAX_SCENE_DEPTH || visited >= MAX_SCENE_NODES {
            break;
        }
        visited += 1;
        if let NodeKind::RadialGradient { id, stops } = &node.kind {
            if let Some(stop) = stops.first() {
                colors.entry(id.as_str()).or_insert(stop.color);
            }
        }
        for child in node.children.iter().rev() {
            if visited.saturating_add(stack.len()) >= MAX_SCENE_NODES {
                break;
            }
            stack.push((child, depth + 1));
        }
    }
    colors
}
fn wire_to_scene_node(node: &WireNode) -> SceneNode {
    let transform = Transform {
        translate_x: node.transform.translate_x,
        translate_y: node.transform.translate_y,
        rotate_deg: node.transform.rotate_deg,
        scale: node.transform.scale,
    };
    let style = Style {
        fill: node.style.fill,
        fill_url: None,
        stroke: node.style.stroke,
        stroke_width: node.style.stroke_width,
        opacity: Some(node.style.opacity),
        filter: None,
        css_class: None,
    };
    let kind = match &node.primitive {
        WirePrimitive::Group => NodeKind::Group { id: None },
        WirePrimitive::Circle { cx, cy, r } => NodeKind::Circle { cx: *cx, cy: *cy, r: *r },
        WirePrimitive::Ellipse { cx, cy, rx, ry } => NodeKind::Ellipse {
            cx: *cx,
            cy: *cy,
            rx: *rx,
            ry: *ry,
        },
        WirePrimitive::Line { x1, y1, x2, y2 } => NodeKind::Line {
            x1: *x1,
            y1: *y1,
            x2: *x2,
            y2: *y2,
        },
        WirePrimitive::Polygon { points, closed } => NodeKind::Polygon {
            points: points.iter().map(|p| (p[0], p[1])).collect(),
            closed: *closed,
        },
        WirePrimitive::Rect { x, y, w, h, rx } => NodeKind::Rect {
            x: *x,
            y: *y,
            w: *w,
            h: *h,
            rx: *rx,
        },
    };
    SceneNode {
        kind,
        transform,
        style,
        children: node.children.iter().map(wire_to_scene_node).collect(),
    }
}

fn valid_coordinate(value: f32) -> bool {
    value.is_finite() && value.abs() <= MAX_ABS_COORDINATE
}

fn valid_nonnegative(value: f32) -> bool {
    valid_coordinate(value) && !value.is_sign_negative()
}

fn valid_positive(value: f32) -> bool {
    valid_nonnegative(value) && value > 0.0
}

fn line_is_renderable(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    effective_scale_abs: f64,
) -> bool {
    let dx = f64::from(x2) - f64::from(x1);
    let dy = f64::from(y2) - f64::from(y1);
    let length = dx.mul_add(dx, dy * dy).sqrt();
    let rendered_length = length * effective_scale_abs;
    rendered_length.is_finite() && rendered_length > MIN_RENDERABLE_LINE_LENGTH
}

fn valid_color(color: Color) -> bool {
    [color.r, color.g, color.b, color.a]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
}

fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn bounded(value: f32) -> f32 {
    finite(value, 0.0).clamp(-MAX_ABS_COORDINATE, MAX_ABS_COORDINATE)
}

fn nonnegative(value: f32) -> f32 {
    bounded(value).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_graph::{Style, Transform};

    #[test]
    fn wire_scene_roundtrips() {
        let scene = SceneNode::group(None)
            .with_child(
                SceneNode::circle(128.0, 128.0, 32.0).with_style(Style {
                    fill: Some(Color::rgba(0.2, 0.4, 0.8, 0.7)),
                    opacity: Some(0.9),
                    ..Style::default()
                }),
            )
            .with_child(
                SceneNode::rect(20.0, 30.0, 100.0, 40.0).with_transform(Transform {
                    translate_x: 4.0,
                    rotate_deg: 15.0,
                    ..Transform::identity()
                }),
            );

        let wire = RemoteScene::from_scene(&scene);
        let encoded = serde_json::to_vec(&wire).expect("wire JSON");
        assert!(encoded.len() <= MAX_SCENE_BYTES);
        let decoded: RemoteScene = serde_json::from_slice(&encoded).expect("wire decode");
        assert_eq!(decoded.version, RemoteScene::VERSION);
        assert_eq!(decoded, wire);
    }

    #[test]
    fn gradient_fill_is_flattened_to_first_stop() {
        let gradient = SceneNode {
            kind: NodeKind::RadialGradient {
                id: "bg".to_string(),
                stops: vec![
                    crate::scene_graph::GradientStop {
                        offset: 0.0,
                        color: Color::rgb(0.1, 0.2, 0.3),
                    },
                    crate::scene_graph::GradientStop {
                        offset: 1.0,
                        color: Color::rgb(0.9, 0.8, 0.7),
                    },
                ],
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: vec![],
        };
        let rect = SceneNode::rect(0.0, 0.0, 10.0, 10.0).with_style(Style {
            fill_url: Some("bg".to_string()),
            ..Style::default()
        });
        let root = SceneNode::group(None)
            .with_child(gradient)
            .with_child(rect);
        let wire = RemoteScene::from_scene(&root);
        assert_eq!(wire.root.children.len(), 1);
        match &wire.root.children[0].primitive {
            WirePrimitive::Rect { .. } => {}
            _ => panic!("expected rectangle child"),
        }
        match &wire.root.children[0].style.fill {
            Some(color) => assert_eq!(*color, Color::rgb(0.1, 0.2, 0.3)),
            None => panic!("gradient fill was not flattened"),
        }
    }

    #[test]
    fn unsupported_protocol_version_is_rejected() {
        let mut scene = RemoteScene::from_scene(&SceneNode::circle(1.0, 1.0, 1.0));
        scene.version = RemoteScene::VERSION + 1;
        assert!(!scene.is_supported());
    }

    #[test]
    fn degenerate_primitives_are_rejected() {
        let zero_circle = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Circle {
                    cx: 0.0,
                    cy: 0.0,
                    r: 0.0,
                },
                transform: WireTransform::default(),
                style: WireStyle {
                    fill: Some(Color::rgb(1.0, 0.0, 0.0)),
                    ..WireStyle::default()
                },
                children: Vec::new(),
            },
        };
        assert!(!zero_circle.is_supported());

        let zero_ellipse = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Ellipse {
                    cx: 0.0,
                    cy: 0.0,
                    rx: 0.0,
                    ry: 1.0,
                },
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: Vec::new(),
            },
        };
        assert!(!zero_ellipse.is_supported());

        let zero_line = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Line {
                    x1: 4.0,
                    y1: 4.0,
                    x2: 4.0,
                    y2: 4.0,
                },
                transform: WireTransform::default(),
                style: WireStyle {
                    stroke: Some(Color::rgb(1.0, 1.0, 1.0)),
                    ..WireStyle::default()
                },
                children: Vec::new(),
            },
        };
        assert!(!zero_line.is_supported());

        let tiny_line = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1e-7,
                    y2: 0.0,
                },
                transform: WireTransform::default(),
                style: WireStyle {
                    stroke: Some(Color::rgb(1.0, 1.0, 1.0)),
                    ..WireStyle::default()
                },
                children: Vec::new(),
            },
        };
        assert!(!tiny_line.is_supported());

        let zero_rect = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 10.0,
                    rx: 0.0,
                },
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: Vec::new(),
            },
        };
        assert!(!zero_rect.is_supported());
    }

    #[test]
    fn externally_constructed_line_rejects_non_renderable_nested_scale() {
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Group,
                transform: WireTransform {
                    scale: 1e-7,
                    ..WireTransform::default()
                },
                style: WireStyle::default(),
                children: vec![WireNode {
                    primitive: WirePrimitive::Line {
                        x1: 0.0,
                        y1: 0.0,
                        x2: 1.0,
                        y2: 0.0,
                    },
                    transform: WireTransform::default(),
                    style: WireStyle {
                        stroke: Some(Color::rgb(1.0, 1.0, 1.0)),
                        ..WireStyle::default()
                    },
                    children: vec![],
                }],
            },
        };
        assert!(
            !scene.is_supported(),
            "line renderability must account for cumulative scale"
        );
    }

    #[test]
    fn externally_constructed_rect_radius_must_fit_shape() {
        let mut scene = RemoteScene::from_scene(&SceneNode::rect(0.0, 0.0, 10.0, 20.0));
        scene.root = WireNode {
            primitive: WirePrimitive::Rect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 20.0,
                rx: 6.0,
            },
            transform: WireTransform::default(),
            style: WireStyle::default(),
            children: Vec::new(),
        };
        assert!(
            !scene.is_supported(),
            "wire rect radius must match GPU reconstruction clamping semantics"
        );
        assert_eq!(scene.to_scene_node().children.len(), 0);
    }

    #[test]
    fn oversized_compiled_scene_falls_back_to_inert_root() {
        let polygon = SceneNode::polygon(
            (0..MAX_POLYGON_POINTS)
                .map(|i| (i as f32 * 1000.0, (i as f32 + 1.0) * -1000.0))
                .collect(),
            true,
        );
        let mut root = SceneNode::group(None);
        for _ in 0..255 {
            root.children.push(polygon.clone());
        }

        let wire = RemoteScene::from_scene(&root);
        assert_eq!(wire.version, RemoteScene::VERSION);
        assert!(matches!(wire.root.primitive, WirePrimitive::Group));
        assert!(
            wire.root.children.is_empty(),
            "oversized wire scene must be replaced atomically"
        );
        assert!(wire.serialized_len() <= MAX_SCENE_BYTES);
    }

    #[test]
    fn hostile_scene_is_bounded_and_effect_free() {
        let mut root = SceneNode::group(None);
        for _ in 0..600 {
            root.children
                .push(SceneNode::path("M 0 0 L 1 1"));
        }
        root.children.push(SceneNode {
            kind: NodeKind::Filter {
                id: "bad".into(),
                filter_type: crate::scene_graph::FilterType::Blur { std_dev: 99.0 },
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: vec![],
        });

        let wire = RemoteScene::from_scene(&root);
        assert!(wire.is_within_budget());
        assert_eq!(wire.root.children.len(), 0);
    }

    #[test]
    fn non_finite_values_fail_closed_after_sanitization() {
        let scene = SceneNode::circle(f32::NAN, f32::INFINITY, f32::NEG_INFINITY);
        let wire = RemoteScene::from_scene(&scene);
        assert!(matches!(wire.root.primitive, WirePrimitive::Group));
        assert!(
            wire.root.children.is_empty(),
            "sanitized degenerate primitive must be replaced by an inert root"
        );
        assert!(wire.is_supported());
    }

    #[test]
    fn untrusted_scene_reconstruction_fails_closed() {
        let root = WireNode {
            primitive: WirePrimitive::Group,
            transform: WireTransform::default(),
            style: WireStyle::default(),
            children: (0..300)
                .map(|_| WireNode {
                    primitive: WirePrimitive::Group,
                    transform: WireTransform::default(),
                    style: WireStyle::default(),
                    children: vec![],
                })
                .collect(),
        };
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root,
        };
        let reconstructed = scene.to_scene_node();
        assert!(matches!(reconstructed.kind, NodeKind::Group { .. }));
        assert!(reconstructed.children.is_empty());
    }

    #[test]
    fn externally_constructed_scene_rejects_node_budget_overrun() {
        let root = WireNode {
            primitive: WirePrimitive::Group,
            transform: WireTransform::default(),
            style: WireStyle::default(),
            children: (0..300)
                .map(|_| WireNode {
                    primitive: WirePrimitive::Group,
                    transform: WireTransform::default(),
                    style: WireStyle::default(),
                    children: vec![],
                })
                .collect(),
        };
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root,
        };
        assert!(!scene.is_supported());
    }

    #[test]
    fn compiled_invalid_closed_polygon_falls_back_to_inert_root() {
        let scene = SceneNode::polygon(
            vec![
                (20.0, 20.0),
                (140.0, 140.0),
                (20.0, 140.0),
                (140.0, 20.0),
            ],
            true,
        );
        let wire = RemoteScene::from_scene(&scene);
        assert!(matches!(wire.root.primitive, WirePrimitive::Group));
        assert!(wire.root.children.is_empty());
        assert!(wire.is_supported());
    }

    #[test]
    fn externally_constructed_closed_polygon_requires_three_vertices() {
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Polygon {
                    points: vec![[0.0, 0.0], [10.0, 10.0]],
                    closed: true,
                },
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: vec![],
            },
        };
        assert!(!scene.is_supported());
    }

    #[test]
    fn externally_constructed_closed_polygon_rejects_collapsed_distinct_vertices() {
        for points in [
            vec![[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]],
            vec![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        ] {
            let scene = RemoteScene {
                version: RemoteScene::VERSION,
                root: WireNode {
                    primitive: WirePrimitive::Polygon { points, closed: true },
                    transform: WireTransform::default(),
                    style: WireStyle::default(),
                    children: vec![],
                },
            };
            assert!(!scene.is_supported());
        }
    }

    #[test]
    fn externally_constructed_scene_rejects_invalid_closed_polygon_topology() {
        for points in [
            vec![
                [20.0, 20.0],
                [140.0, 140.0],
                [20.0, 140.0],
                [140.0, 20.0],
            ],
            vec![
                [20.0, 20.0],
                [140.0, 20.0],
                [140.0, 140.0],
                [20.0, 140.0],
                [80.0, 20.0],
            ],
            vec![
                [20.0, 20.0],
                [140.0, 20.0],
                [140.0, 140.0],
                [20.0, 20.0],
                [20.0, 140.0],
            ],
        ] {
            let scene = RemoteScene {
                version: RemoteScene::VERSION,
                root: WireNode {
                    primitive: WirePrimitive::Polygon { points, closed: true },
                    transform: WireTransform::default(),
                    style: WireStyle::default(),
                    children: vec![],
                },
            };
            assert!(!scene.is_supported());
        }
    }

    #[test]
    fn externally_constructed_scene_rejects_out_of_bounds_primitives() {
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Circle {
                    cx: MAX_ABS_COORDINATE + 1.0,
                    cy: 0.0,
                    r: 1.0,
                },
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: vec![],
            },
        };
        assert!(!scene.is_supported());
    }

    #[test]
    fn externally_constructed_scene_rejects_large_coordinate_self_intersection() {
        let points = vec![
            [900_000.0, 900_000.0],
            [-900_000.0, -900_000.0],
            [900_000.0, -900_000.0],
            [-900_000.0, 900_000.0],
        ];
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Polygon { points, closed: true },
                transform: WireTransform::default(),
                style: WireStyle::default(),
                children: vec![],
            },
        };
        assert!(!scene.is_supported());
    }

    #[test]
    fn externally_constructed_scene_rejects_invalid_colors() {
        let scene = RemoteScene {
            version: RemoteScene::VERSION,
            root: WireNode {
                primitive: WirePrimitive::Group,
                transform: WireTransform::default(),
                style: WireStyle {
                    fill: Some(Color::rgba(2.0, 0.0, 0.0, 1.0)),
                    ..WireStyle::default()
                },
                children: vec![],
            },
        };
        assert!(!scene.is_supported());
    }

    #[test]
    fn nested_scene_respects_depth_bound() {
        let mut root = SceneNode::group(None);
        let mut cursor = &mut root;
        for _ in 0..100 {
            cursor.children.push(SceneNode::group(None));
            cursor = cursor.children.last_mut().unwrap();
        }

        let wire = RemoteScene::from_scene(&root);
        assert!(wire.serialized_len() <= MAX_SCENE_BYTES);
    }
}
