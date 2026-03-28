// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Declarative scene graph: SceneNode tree that the SVG renderer consumes.

use crate::color::Color;
use serde::{Deserialize, Serialize};

/// A node in the scene graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    pub kind: NodeKind,
    pub transform: Transform,
    pub style: Style,
    pub children: Vec<SceneNode>,
}

/// The geometric primitive or group this node represents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// Invisible container for grouping children.
    Group { id: Option<String> },
    /// Circle at (cx, cy) with radius r.
    Circle { cx: f32, cy: f32, r: f32 },
    /// Ellipse at (cx, cy) with radii rx, ry.
    Ellipse { cx: f32, cy: f32, rx: f32, ry: f32 },
    /// Line from (x1,y1) to (x2,y2).
    Line { x1: f32, y1: f32, x2: f32, y2: f32 },
    /// Polyline/polygon from a list of (x, y) points.
    Polygon {
        points: Vec<(f32, f32)>,
        closed: bool,
    },
    /// Rectangle at (x, y) with width/height.
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rx: f32,
    },
    /// SVG radial gradient definition (used in <defs>).
    RadialGradient {
        id: String,
        stops: Vec<GradientStop>,
    },
    /// SVG filter definition.
    Filter { id: String, filter_type: FilterType },
    /// Reference a defined gradient/filter (via url(#id)).
    UseFilter { filter_id: String },
    /// SVG path element with raw path data string (e.g., "M 0 0 L 10 10 C ...").
    /// Used by generative art systems (atelier) for arbitrary curves and L-system output.
    Path { d: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// SVG feTurbulence + feDisplacementMap.
    Turbulence {
        base_frequency: f32,
        num_octaves: u8,
        scale: f32,
    },
    /// Gaussian blur.
    Blur { std_dev: f32 },
}

/// 2D affine transform.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Transform {
    pub translate_x: f32,
    pub translate_y: f32,
    pub rotate_deg: f32,
    pub scale: f32,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            translate_x: 0.0,
            translate_y: 0.0,
            rotate_deg: 0.0,
            scale: 1.0,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.translate_x == 0.0
            && self.translate_y == 0.0
            && self.rotate_deg == 0.0
            && (self.scale - 1.0).abs() < 1e-6
    }

    /// Emit SVG transform attribute value.
    pub fn to_svg(&self) -> String {
        let mut parts = Vec::new();
        if self.translate_x != 0.0 || self.translate_y != 0.0 {
            parts.push(format!(
                "translate({:.1},{:.1})",
                self.translate_x, self.translate_y
            ));
        }
        if self.rotate_deg != 0.0 {
            parts.push(format!("rotate({:.1})", self.rotate_deg));
        }
        if (self.scale - 1.0).abs() > 1e-6 {
            parts.push(format!("scale({:.3})", self.scale));
        }
        parts.join(" ")
    }
}

/// Visual style applied to a node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Style {
    pub fill: Option<Color>,
    /// Fill referencing a gradient/pattern by id: `url(#id)`.
    pub fill_url: Option<String>,
    pub stroke: Option<Color>,
    pub stroke_width: Option<f32>,
    pub opacity: Option<f32>,
    pub filter: Option<String>,
    pub css_class: Option<String>,
}

impl SceneNode {
    pub fn group(id: Option<&str>) -> Self {
        Self {
            kind: NodeKind::Group {
                id: id.map(String::from),
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn circle(cx: f32, cy: f32, r: f32) -> Self {
        Self {
            kind: NodeKind::Circle { cx, cy, r },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> Self {
        Self {
            kind: NodeKind::Ellipse { cx, cy, rx, ry },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            kind: NodeKind::Rect {
                x,
                y,
                w,
                h,
                rx: 0.0,
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn polygon(points: Vec<(f32, f32)>, closed: bool) -> Self {
        Self {
            kind: NodeKind::Polygon { points, closed },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn line(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            kind: NodeKind::Line { x1, y1, x2, y2 },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn path(d: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Path { d: d.into() },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_child(mut self, child: SceneNode) -> Self {
        self.children.push(child);
        self
    }

    /// Total node count (self + all descendants).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_empty() {
        let n = SceneNode::group(None);
        assert_eq!(n.node_count(), 1);
    }

    #[test]
    fn node_count_nested() {
        let tree = SceneNode::group(Some("root"))
            .with_child(SceneNode::circle(0.0, 0.0, 10.0))
            .with_child(SceneNode::group(None).with_child(SceneNode::circle(5.0, 5.0, 3.0)));
        assert_eq!(tree.node_count(), 4);
    }

    #[test]
    fn transform_identity_svg() {
        let t = Transform::identity();
        assert!(t.is_identity());
        assert!(t.to_svg().is_empty());
    }

    #[test]
    fn transform_translate_svg() {
        let t = Transform {
            translate_x: 10.0,
            translate_y: 20.0,
            ..Transform::identity()
        };
        assert!(t.to_svg().contains("translate(10.0,20.0)"));
    }
}
