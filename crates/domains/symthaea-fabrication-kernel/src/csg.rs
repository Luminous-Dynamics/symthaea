// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constructive Solid Geometry (CSG) tree types
//!
//! Represents boolean operations (union, subtract, intersect) on geometric primitives.

use serde::{Deserialize, Serialize};


/// Structural and numeric budgets for validating an untrusted CSG operation tree.
///
/// These limits bound the tree walk and reject transforms whose conservative
/// coordinate envelope cannot be represented by the mesh's f32 coordinates.
/// They do not bound the full CPU or memory cost of downstream boolean meshing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsgValidationLimits {
    pub max_nodes: usize,
    pub max_depth: usize,
}

impl Default for CsgValidationLimits {
    fn default() -> Self {
        Self {
            max_nodes: 4_096,
            max_depth: 64,
        }
    }
}

/// Summary returned after a bounded CSG preflight succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsgValidationSummary {
    pub node_count: usize,
    pub max_depth: usize,
}

/// Fail-closed reasons why a CSG tree cannot enter the bounded resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgValidationError {
    InvalidLimits,
    NodeBudgetExceeded { limit: usize },
    DepthBudgetExceeded { limit: usize },
    NonFiniteTransform,
    SingularScale,
    NumericRangeExceeded,
}

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

    /// Count total nodes without recursively consuming the Rust call stack.
    pub fn node_count(&self) -> usize {
        let mut pending = vec![self];
        let mut count = 0usize;
        while let Some(node) = pending.pop() {
            count = count.saturating_add(1);
            match node {
                CSGNode::Primitive(_) => {}
                CSGNode::Transform { node, .. } => pending.push(node),
                CSGNode::Boolean { left, right, .. } => {
                    pending.push(left);
                    pending.push(right);
                }
            }
        }
        count
    }

    /// Validate tree structure and transform numerics before recursive meshing.
    ///
    /// The iterative walk avoids recursing on attacker-controlled tree depth.
    /// A conservative origin-centred radius envelope starts at 2 mm, which
    /// contains every current unit primitive, then expands through each
    /// scale/rotation/translation transform. Negative scales remain permitted
    /// because the mesh backend explicitly reverses winding for reflections.
    pub fn validate_bounded(
        &self,
        limits: CsgValidationLimits,
    ) -> Result<CsgValidationSummary, CsgValidationError> {
        const MIN_ABS_SCALE: f32 = 1.0e-12;
        const INITIAL_RADIUS_MM: f64 = 2.0;

        if limits.max_nodes == 0 || limits.max_depth == 0 {
            return Err(CsgValidationError::InvalidLimits);
        }

        // radius is a conservative upper bound on the distance of any point
        // from the origin after the transforms leading to this node.
        let mut pending = vec![(self, 1usize, INITIAL_RADIUS_MM)];
        let mut node_count = 0usize;
        let mut observed_max_depth = 0usize;

        while let Some((node, depth, radius)) = pending.pop() {
            if depth > limits.max_depth {
                return Err(CsgValidationError::DepthBudgetExceeded {
                    limit: limits.max_depth,
                });
            }

            node_count = node_count.saturating_add(1);
            if node_count > limits.max_nodes {
                return Err(CsgValidationError::NodeBudgetExceeded {
                    limit: limits.max_nodes,
                });
            }
            observed_max_depth = observed_max_depth.max(depth);

            match node {
                CSGNode::Primitive(_) => {}
                CSGNode::Transform { node, transform } => {
                    if transform.scale.iter().any(|value| !value.is_finite())
                        || transform.rotate.iter().any(|value| !value.is_finite())
                        || transform.translate.iter().any(|value| !value.is_finite())
                    {
                        return Err(CsgValidationError::NonFiniteTransform);
                    }

                    if transform
                        .scale
                        .iter()
                        .any(|value| value.abs() <= MIN_ABS_SCALE)
                    {
                        return Err(CsgValidationError::SingularScale);
                    }

                    let maximum_scale = transform
                        .scale
                        .iter()
                        .map(|value| f64::from(value.abs()))
                        .fold(0.0_f64, f64::max);
                    let translation_radius = transform
                        .translate
                        .iter()
                        .map(|value| {
                            let value = f64::from(*value);
                            value * value
                        })
                        .sum::<f64>()
                        .sqrt();
                    let next_radius = radius * maximum_scale + translation_radius;
                    if !next_radius.is_finite() || next_radius > f64::from(f32::MAX) {
                        return Err(CsgValidationError::NumericRangeExceeded);
                    }

                    pending.push((node, depth + 1, next_radius));
                }
                CSGNode::Boolean { left, right, .. } => {
                    pending.push((left, depth + 1, radius));
                    pending.push((right, depth + 1, radius));
                }
            }
        }

        Ok(CsgValidationSummary {
            node_count,
            max_depth: observed_max_depth,
        })
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

    #[test]
    fn bounded_preflight_accepts_current_primitive_and_boolean_trees() {
        let tree = CSGNode::cube()
            .with_transform(Transform3D::default())
            .union(CSGNode::sphere().translate(10.0, 0.0, 0.0));
        let report = tree.validate_bounded(CsgValidationLimits::default()).unwrap();
        assert_eq!(report.node_count, 5);
        assert_eq!(report.max_depth, 3);
    }

    #[test]
    fn bounded_preflight_rejects_non_finite_transform_components() {
        let tree = CSGNode::cube().translate(f64::INFINITY, 0.0, 0.0);
        assert_eq!(
            tree.validate_bounded(CsgValidationLimits::default()),
            Err(CsgValidationError::NonFiniteTransform)
        );
    }

    #[test]
    fn bounded_preflight_rejects_singular_scale() {
        let tree = CSGNode::cube().scale(1.0, 0.0, 1.0);
        assert_eq!(
            tree.validate_bounded(CsgValidationLimits::default()),
            Err(CsgValidationError::SingularScale)
        );
    }

    #[test]
    fn bounded_preflight_enforces_node_and_depth_budgets() {
        let tree = CSGNode::cube()
            .union(CSGNode::sphere())
            .union(CSGNode::cylinder());
        assert_eq!(
            tree.validate_bounded(CsgValidationLimits {
                max_nodes: 4,
                max_depth: 64,
            }),
            Err(CsgValidationError::NodeBudgetExceeded { limit: 4 })
        );

        let deep_tree = (0..4).fold(CSGNode::cube(), |node, _| {
            node.with_transform(Transform3D::default())
        });
        assert_eq!(
            deep_tree.validate_bounded(CsgValidationLimits {
                max_nodes: 64,
                max_depth: 4,
            }),
            Err(CsgValidationError::DepthBudgetExceeded { limit: 4 })
        );
    }

    #[test]
    fn bounded_preflight_rejects_unrepresentable_nested_scale_envelope() {
        let tree = CSGNode::cube()
            .scale(1.0e20, 1.0, 1.0)
            .scale(1.0e20, 1.0, 1.0);
        assert_eq!(
            tree.validate_bounded(CsgValidationLimits::default()),
            Err(CsgValidationError::NumericRangeExceeded)
        );
    }

    #[test]
    fn bounded_preflight_rejects_zero_limits() {
        assert_eq!(
            CSGNode::cube().validate_bounded(CsgValidationLimits {
                max_nodes: 0,
                max_depth: 1,
            }),
            Err(CsgValidationError::InvalidLimits)
        );
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
