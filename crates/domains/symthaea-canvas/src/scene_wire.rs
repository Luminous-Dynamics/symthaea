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
use crate::scene_graph::{NodeKind, SceneNode};

const MAX_SCENE_NODES: usize = 256;
const MAX_SCENE_DEPTH: usize = 24;
const MAX_POLYGON_POINTS: usize = 128;
const MAX_PATH_BYTES: usize = 12 * 1024;
const MAX_SCENE_BYTES: usize = 128 * 1024;
const MAX_ABS_COORDINATE: f32 = 1_000_000.0;
const MAX_ABS_SCALE: f32 = 8.0;

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
        let root = compile_node(scene, 0, &mut node_count);
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
        if serde_json::to_vec(&remote)
            .map(|bytes| bytes.len() <= MAX_SCENE_BYTES)
            .unwrap_or(false)
        {
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

    pub fn serialized_len(&self) -> usize {
        serde_json::to_vec(self).map(|bytes| bytes.len()).unwrap_or(MAX_SCENE_BYTES + 1)
    }

    pub fn is_within_budget(&self) -> bool {
        self.serialized_len() <= MAX_SCENE_BYTES
    }
}

fn compile_node(node: &SceneNode, depth: usize, count: &mut usize) -> Option<WireNode> {
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
        fill: node.style.fill.map(|c| c.sanitized()),
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
        .filter_map(|child| compile_node(child, depth + 1, count))
        .collect();

    Some(WireNode {
        primitive,
        transform: wire_transform,
        style: wire_style,
        children,
    })
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
    fn non_finite_values_are_sanitized() {
        let scene = SceneNode::circle(f32::NAN, f32::INFINITY, f32::NEG_INFINITY);
        let wire = RemoteScene::from_scene(&scene);
        match wire.root.primitive {
            WirePrimitive::Circle { cx, cy, r } => {
                assert_eq!(cx, 0.0);
                assert_eq!(cy, 0.0);
                assert_eq!(r, 0.0);
            }
            _ => panic!("unexpected primitive"),
        }
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
