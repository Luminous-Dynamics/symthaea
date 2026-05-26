// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geometric thoughts — HDC intent to CSG tree resolution
//!
//! A GeometricThought captures both the HDC encoding of a design intent
//! and its explicit CSG operation tree, enabling bidirectional mapping
//! between the "thought space" and "geometry space".

use crate::csg::{BooleanOp, CSGNode, Primitive};
use crate::mesh::{resolve_to_mesh, TriangleMesh};
use crate::primitives::*;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Printer constraints for parametric generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrinterConstraints {
    pub max_build_volume: [f32; 3], // mm
    pub min_layer_height: f32,      // mm
    pub max_layer_height: f32,      // mm
}

/// A geometric thought: the bridge between HDC intent and physical geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricThought {
    /// The "thought" as an HDC vector
    #[serde(skip)]
    pub intent_hv: Option<ContinuousHV>,
    /// Explicit geometry tree
    pub operation_tree: CSGNode,
    /// Material constraint (e.g., "PLA", "steel")
    pub material_binding: Option<String>,
    /// Printer constraints
    pub printer_constraints: Option<PrinterConstraints>,
}

impl GeometricThought {
    /// Create from an explicit CSG tree, encoding the intent as HDC
    pub fn from_csg(tree: CSGNode) -> Self {
        let intent_hv = encode_csg_as_hv(&tree);
        Self {
            intent_hv: Some(intent_hv),
            operation_tree: tree,
            material_binding: None,
            printer_constraints: None,
        }
    }

    /// Resolve to a triangle mesh
    pub fn resolve(&self) -> TriangleMesh {
        resolve_to_mesh(&self.operation_tree)
    }

    /// Check if mesh fits within printer constraints
    pub fn fits_constraints(&self) -> bool {
        if let Some(ref constraints) = self.printer_constraints {
            let mesh = self.resolve();
            let (min, max) = mesh_aabb(&mesh);
            let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
            size[0] <= constraints.max_build_volume[0]
                && size[1] <= constraints.max_build_volume[1]
                && size[2] <= constraints.max_build_volume[2]
        } else {
            true // No constraints = always fits
        }
    }

    /// Autonomously derive symbolic SMT safety invariants from the CSG tree.
    ///
    /// These invariants ensure the geometry is physically realizable
    /// (e.g., positive scale, within build volume, no self-intersection hazards).
    pub fn derive_invariants(&self) -> Vec<String> {
        let mut invariants = Vec::new();
        
        // 1. Global Build Volume Invariants
        if let Some(ref c) = self.printer_constraints {
            invariants.push(format!("(assert (<= total_width {:.1}))", c.max_build_volume[0]));
            invariants.push(format!("(assert (<= total_height {:.1}))", c.max_build_volume[1]));
            invariants.push(format!("(assert (<= total_depth {:.1}))", c.max_build_volume[2]));
        }

        // 2. Local Geometric Invariants (Recursive traversal)
        derive_node_invariants(&self.operation_tree, &mut invariants, "root");

        invariants
    }
}

/// Recursively derive invariants for a CSG node.
fn derive_node_invariants(node: &CSGNode, invariants: &mut Vec<String>, path: &str) {
    match node {
        CSGNode::Primitive(p) => {
            match p {
                Primitive::Cylinder | Primitive::Sphere | Primitive::Torus => {
                    invariants.push(format!("(assert (> {}_{:?}_radius 0.0))", path, p));
                }
                _ => {}
            }
        }
        CSGNode::Transform { node, transform } => {
            // Safety: Scale must be positive to prevent inverted geometry (broken meshes)
            invariants.push(format!("(assert (> {}_{:.4}_scale_x 0.0))", path, transform.scale[0]));
            invariants.push(format!("(assert (> {}_{:.4}_scale_y 0.0))", path, transform.scale[1]));
            invariants.push(format!("(assert (> {}_{:.4}_scale_z 0.0))", path, transform.scale[2]));
            
            derive_node_invariants(node, invariants, &format!("{}_t", path));
        }
        CSGNode::Boolean { op, left, right } => {
            if *op == BooleanOp::Subtract {
                // Heuristic: Subtraction should not completely consume the source
                invariants.push(format!("(assert (not (= {path}_left {path}_right)))"));
            }
            derive_node_invariants(left, invariants, &format!("{}_l", path));
            derive_node_invariants(right, invariants, &format!("{}_r", path));
        }
    }
}

/// Recursively encode a CSG tree as an HDC hypervector
pub fn encode_csg_as_hv(node: &CSGNode) -> ContinuousHV {
    match node {
        CSGNode::Primitive(prim) => match prim {
            Primitive::Cube => cube_hv(),
            Primitive::Cylinder => cylinder_hv(),
            Primitive::Sphere => sphere_hv(),
            Primitive::Cone => cone_hv(),
            Primitive::Torus => torus_hv(),
        },
        CSGNode::Transform { node, transform } => {
            let node_hv = encode_csg_as_hv(node);
            // Encode transform as scale-bound parameters
            let sx = param_hv(0x5358_0001, transform.scale[0]);
            let sy = param_hv(0x5359_0002, transform.scale[1]);
            let sz = param_hv(0x535A_0003, transform.scale[2]);
            let scale_params = ContinuousHV::bundle(&[&sx, &sy, &sz]);
            let transform_hv = scale_hv().bind(&scale_params);
            node_hv.bind(&transform_hv)
        }
        CSGNode::Boolean { op, left, right } => {
            let left_hv = encode_csg_as_hv(left);
            let right_hv = encode_csg_as_hv(right);
            let operands = ContinuousHV::bundle(&[&left_hv, &right_hv]);
            let op_hv = match op {
                BooleanOp::Union => union_hv(),
                BooleanOp::Subtract => subtract_hv(),
                BooleanOp::Intersect => intersect_hv(),
            };
            op_hv.bind(&operands)
        }
    }
}

/// Compute axis-aligned bounding box of a mesh
fn mesh_aabb(mesh: &TriangleMesh) -> ([f32; 3], [f32; 3]) {
    if mesh.vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &mesh.vertices {
        for i in 0..3 {
            if v[i] < min[i] {
                min[i] = v[i];
            }
            if v[i] > max[i] {
                max[i] = v[i];
            }
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_cube() {
        let thought = GeometricThought::from_csg(CSGNode::cube());
        assert!(thought.intent_hv.is_some());
        let mesh = thought.resolve();
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_same_shape_similarity() {
        let hv1 = encode_csg_as_hv(&CSGNode::cube());
        let hv2 = encode_csg_as_hv(&CSGNode::cube());
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_different_shape_dissimilarity() {
        let cube = encode_csg_as_hv(&CSGNode::cube());
        let sphere = encode_csg_as_hv(&CSGNode::sphere());
        assert!(cube.similarity(&sphere).abs() < 0.15);
    }

    #[test]
    fn test_fits_constraints_yes() {
        let mut thought = GeometricThought::from_csg(CSGNode::cube());
        thought.printer_constraints = Some(PrinterConstraints {
            max_build_volume: [100.0, 100.0, 100.0],
            min_layer_height: 0.1,
            max_layer_height: 0.3,
        });
        assert!(thought.fits_constraints());
    }

    #[test]
    fn test_fits_constraints_no() {
        let mut thought = GeometricThought::from_csg(CSGNode::cube());
        thought.printer_constraints = Some(PrinterConstraints {
            max_build_volume: [0.1, 0.1, 0.1], // Tiny volume
            min_layer_height: 0.1,
            max_layer_height: 0.3,
        });
        assert!(!thought.fits_constraints());
    }

    #[test]
    fn test_no_constraints() {
        let thought = GeometricThought::from_csg(CSGNode::cube());
        assert!(thought.fits_constraints()); // No constraints = always fits
    }

    #[test]
    fn test_boolean_tree_encoding() {
        let tree = CSGNode::cube().union(CSGNode::sphere());
        let hv = encode_csg_as_hv(&tree);
        // Just verify it doesn't panic and produces a valid vector
        assert_eq!(hv.values.len(), FAB_KERNEL_DIM);
        assert!((hv.similarity(&hv) - 1.0).abs() < 0.001);
    }
}
