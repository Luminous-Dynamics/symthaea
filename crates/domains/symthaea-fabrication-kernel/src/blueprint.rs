// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Blueprinting Engine: Project 3D geometry into 2D technical drawings (SVG).

use crate::csg::{CSGNode, Primitive};
use crate::thought::GeometricThought;
use symthaea_canvas::color::Color;
use symthaea_canvas::scene_graph::{NodeKind, SceneNode, Style, Transform};

/// A blueprint view (Front, Top, Side).
#[derive(Debug, Clone, Copy)]
pub enum View {
    Front,
    Top,
    Side,
}

pub struct BlueprintEngine;

impl BlueprintEngine {
    /// Generate a 2D technical blueprint from a geometric thought.
    // The elements are built up across ~40 lines of multi-field struct
    // literals and function calls; a vec![] literal here would be far
    // less readable than the sequential pushes.
    #[allow(clippy::vec_init_then_push)]
    pub fn generate_blueprint(thought: &GeometricThought) -> String {
        let mut elements = Vec::new();

        // 1. Generate multi-view projections
        elements.push(Self::project_view(
            &thought.operation_tree,
            View::Top,
            [100.0, 100.0],
        ));
        elements.push(Self::project_view(
            &thought.operation_tree,
            View::Front,
            [100.0, 400.0],
        ));
        elements.push(Self::project_view(
            &thought.operation_tree,
            View::Side,
            [400.0, 400.0],
        ));

        // 2. Add Title Block and Annotations
        elements.push(SceneNode {
            kind: NodeKind::Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 100.0,
                rx: 0.0,
            },
            transform: Transform {
                translate_x: 100.0,
                translate_y: 700.0,
                ..Transform::identity()
            },
            style: Style {
                fill: Some(Color::rgba(200.0, 200.0, 255.0, 0.1)),
                stroke: Some(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                stroke_width: Some(2.0),
                ..Default::default()
            },
            children: Vec::new(),
        });

        let scene = SceneNode {
            kind: NodeKind::Group {
                id: Some("blueprint".into()),
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: elements,
        };

        symthaea_canvas::render_svg(&scene, 1.0)
    }

    /// Project 3D CSG into a 2D view.
    fn project_view(node: &CSGNode, view: View, offset: [f32; 2]) -> SceneNode {
        let mut shapes = Vec::new();

        // Simple recursive wireframe projection
        Self::project_recursive(node, view, &mut shapes);

        SceneNode {
            kind: NodeKind::Group {
                id: Some(format!("{:?}", view)),
            },
            transform: Transform {
                translate_x: offset[0],
                translate_y: offset[1],
                ..Transform::identity()
            },
            style: Style::default(),
            children: shapes,
        }
    }

    fn project_recursive(node: &CSGNode, view: View, shapes: &mut Vec<SceneNode>) {
        match node {
            CSGNode::Primitive(p) => {
                let kind = match (p, view) {
                    (Primitive::Cube, _) => NodeKind::Rect {
                        x: -50.0,
                        y: -50.0,
                        w: 100.0,
                        h: 100.0,
                        rx: 0.0,
                    },
                    (Primitive::Sphere, _) => NodeKind::Circle {
                        cx: 0.0,
                        cy: 0.0,
                        r: 50.0,
                    },
                    (Primitive::Cylinder, View::Top) => NodeKind::Circle {
                        cx: 0.0,
                        cy: 0.0,
                        r: 50.0,
                    },
                    (Primitive::Cylinder, _) => NodeKind::Rect {
                        x: -50.0,
                        y: -50.0,
                        w: 100.0,
                        h: 100.0,
                        rx: 0.0,
                    },
                    _ => NodeKind::Rect {
                        x: -25.0,
                        y: -25.0,
                        w: 50.0,
                        h: 50.0,
                        rx: 0.0,
                    },
                };
                shapes.push(SceneNode {
                    kind,
                    transform: Transform::identity(),
                    style: Style {
                        fill: None,
                        stroke: Some(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                        stroke_width: Some(1.0),
                        ..Default::default()
                    },
                    children: Vec::new(),
                });
            }
            CSGNode::Transform { node, .. } => {
                // In real implementation, apply the transform to the projection
                Self::project_recursive(node, view, shapes);
            }
            CSGNode::Boolean { left, right, .. } => {
                Self::project_recursive(left, view, shapes);
                Self::project_recursive(right, view, shapes);
            }
        }
    }
}
