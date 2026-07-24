// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constructive Solid Geometry (CSG) tree types
//!
//! Represents boolean operations (union, subtract, intersect) on geometric primitives.

use serde::{Deserialize, Serialize};

/// A CSG operation tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CSGNode {
    /// A geometric primitive
    Primitive(Primitive),
    /// A transformed sub-tree
    Transform {
        node: Box<CSGNode>,
        transform: Transform3D,
    },
    /// A boolean operation on two sub-trees
    Boolean {
        op: BooleanOp,
        left: Box<CSGNode>,
        right: Box<CSGNode>,
    },
}

/// Geometric primitive shapes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Primitive {
    /// Unit cube centered at origin (1x1x1)
    Cube,
    /// Unit cylinder along Z axis (radius=0.5, height=1.0)
    Cylinder,
    /// Unit sphere at origin (radius=0.5)
    Sphere,
    /// Unit cone along Z axis (base radius=0.5, height=1.0)
    Cone,
    /// Torus at origin (major_radius=0.5, minor_radius=0.2)
    Torus,
}

/// Boolean operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BooleanOp {
    Union,
    Subtract,
    Intersect,
}

/// 3D affine transform (scale, rotate, translate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform3D {
    pub scale: [f32; 3],
    pub rotate: [f32; 3], // Euler angles in radians (X, Y, Z)
    pub translate: [f32; 3],
}

impl Default for Transform3D {
    fn default() -> Self {
        Self {
            scale: [1.0, 1.0, 1.0],
            rotate: [0.0, 0.0, 0.0],
            translate: [0.0, 0.0, 0.0],
        }
    }
}

impl Transform3D {
    fn rotate_vector(&self, mut p: [f32; 3]) -> [f32; 3] {
        // Rotate X
        let (sx, cx) = self.rotate[0].sin_cos();
        let y = p[1] * cx - p[2] * sx;
        let z = p[1] * sx + p[2] * cx;
        p[1] = y;
        p[2] = z;

        // Rotate Y
        let (sy, cy) = self.rotate[1].sin_cos();
        let x = p[0] * cy + p[2] * sy;
        let z = -p[0] * sy + p[2] * cy;
        p[0] = x;
        p[2] = z;

        // Rotate Z
        let (sz, cz) = self.rotate[2].sin_cos();
        let x = p[0] * cz - p[1] * sz;
        let y = p[0] * sz + p[1] * cz;
        p[0] = x;
        p[1] = y;

        p
    }

    /// Apply transform to a point: scale → rotate → translate.
    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let scaled = [
            point[0] * self.scale[0],
            point[1] * self.scale[1],
            point[2] * self.scale[2],
        ];
        let mut p = self.rotate_vector(scaled);
        p[0] += self.translate[0];
        p[1] += self.translate[1];
        p[2] += self.translate[2];
        p
    }

    /// Transform a normal with the inverse-transpose of the linear transform.
    ///
    /// Returns `None` for a singular or non-finite transform because a normal
    /// is undefined when any scale axis collapses to zero.
    pub fn apply_normal(&self, normal: [f32; 3]) -> Option<[f32; 3]> {
        const MIN_SCALE: f32 = 1.0e-12;
        if self
            .scale
            .iter()
            .any(|s| !s.is_finite() || s.abs() <= MIN_SCALE)
            || self.rotate.iter().any(|r| !r.is_finite())
        {
            return None;
        }

        let inverse_scaled = [
            normal[0] / self.scale[0],
            normal[1] / self.scale[1],
            normal[2] / self.scale[2],
        ];
        let mut transformed = self.rotate_vector(inverse_scaled);
        let len = (transformed[0] * transformed[0]
            + transformed[1] * transformed[1]
            + transformed[2] * transformed[2])
            .sqrt();
        if !len.is_finite() || len <= 1.0e-12 {
            return None;
        }
        transformed[0] /= len;
        transformed[1] /= len;
        transformed[2] /= len;
        Some(transformed)
    }

    /// True when the linear transform reverses orientation.
    pub fn reverses_orientation(&self) -> bool {
        self.scale[0] * self.scale[1] * self.scale[2] < 0.0
    }
}

impl CSGNode {
    /// Create a primitive node
    pub fn cube() -> Self {
        CSGNode::Primitive(Primitive::Cube)
    }
    pub fn cylinder() -> Self {
        CSGNode::Primitive(Primitive::Cylinder)
    }
    pub fn sphere() -> Self {
        CSGNode::Primitive(Primitive::Sphere)
    }
    pub fn cone() -> Self {
        CSGNode::Primitive(Primitive::Cone)
    }
    pub fn torus() -> Self {
        CSGNode::Primitive(Primitive::Torus)
    }

    /// Wrap in a transform
    pub fn with_transform(self, transform: Transform3D) -> Self {
        CSGNode::Transform {
            node: Box::new(self),
            transform,
        }
    }

    /// Boolean union with another node
    pub fn union(self, other: CSGNode) -> Self {
        CSGNode::Boolean {
            op: BooleanOp::Union,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    /// Boolean subtraction (self - other)
    pub fn subtract(self, other: CSGNode) -> Self {
        CSGNode::Boolean {
            op: BooleanOp::Subtract,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    /// Boolean intersection
    pub fn intersect(self, other: CSGNode) -> Self {
        CSGNode::Boolean {
            op: BooleanOp::Intersect,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    /// Count total nodes in the tree
    pub fn node_count(&self) -> usize {
        match self {
            CSGNode::Primitive(_) => 1,
            CSGNode::Transform { node, .. } => 1 + node.node_count(),
            CSGNode::Boolean { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let t = Transform3D::default();
        let p = t.apply([1.0, 2.0, 3.0]);
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!((p[1] - 2.0).abs() < 1e-6);
        assert!((p[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_transform() {
        let t = Transform3D {
            scale: [2.0, 3.0, 4.0],
            ..Default::default()
        };
        let p = t.apply([1.0, 1.0, 1.0]);
        assert!((p[0] - 2.0).abs() < 1e-6);
        assert!((p[1] - 3.0).abs() < 1e-6);
        assert!((p[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_count() {
        let tree = CSGNode::cube()
            .union(CSGNode::sphere())
            .subtract(CSGNode::cylinder());
        assert_eq!(tree.node_count(), 5); // 3 primitives + 2 booleans
    }

    #[test]
    fn test_serde_roundtrip() {
        let tree = CSGNode::cube().with_transform(Transform3D {
            scale: [0.1, 0.05, 0.02],
            ..Default::default()
        });
        let json = serde_json::to_string(&tree).unwrap();
        let parsed: CSGNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_count(), 2);
    }
}

impl CSGNode {
    pub fn scale(self, x: f64, y: f64, z: f64) -> Self {
        self.with_transform(Transform3D {
            scale: [x as f32, y as f32, z as f32],
            ..Default::default()
        })
    }

    pub fn rotate(self, x: f64, y: f64, z: f64) -> Self {
        self.with_transform(Transform3D {
            rotate: [x as f32, y as f32, z as f32],
            ..Default::default()
        })
    }

    pub fn translate(self, x: f64, y: f64, z: f64) -> Self {
        self.with_transform(Transform3D {
            translate: [x as f32, y as f32, z as f32],
            ..Default::default()
        })
    }
}
