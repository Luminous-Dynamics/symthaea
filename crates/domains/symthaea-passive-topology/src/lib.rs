// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural positive/negative-space descriptors for passive design search.
//!
//! A CSG subtraction is treated as *negative-space intent*, not as proof that a
//! physical cavity exists. Actual enclosure, connectivity, flow accessibility,
//! wall thickness, and printability remain downstream geometric/physics facts.

use symthaea_fabrication_kernel::csg::{BooleanOp, CSGNode, Primitive};

/// Deterministic description of where a CSG design explicitly removes space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSpaceSignature {
    /// Canonical digest of the complete CSG tree.
    pub digest: [u8; 32],
    /// Number of CSG nodes in the complete tree.
    pub node_count: usize,
    /// Number of primitive nodes in the complete tree.
    pub primitive_count: usize,
    /// Number of explicit subtraction operations.
    pub void_cut_count: usize,
    /// Number of tree nodes belonging to subtraction right-hand sides.
    pub void_node_count: usize,
    /// Number of explicit intersection operations.
    pub intersection_count: usize,
    /// Number of transforms in the complete tree.
    pub transform_count: usize,
    /// Deepest subtraction right-hand-side root in the tree.
    pub max_void_depth: usize,
}

impl NegativeSpaceSignature {
    pub fn from_csg(node: &CSGNode) -> Self {
        let digest = hash_node(node);
        let mut stats = Stats::default();
        visit(node, 1, false, &mut stats);

        Self {
            digest: *digest.as_bytes(),
            node_count: stats.node_count,
            primitive_count: stats.primitive_count,
            void_cut_count: stats.void_cut_count,
            void_node_count: stats.void_node_count,
            intersection_count: stats.intersection_count,
            transform_count: stats.transform_count,
            max_void_depth: stats.max_void_depth,
        }
    }

    /// True when the design explicitly expresses at least one removed-space
    /// operation. This is an intent signal, not a cavity certification.
    pub fn declares_negative_space(&self) -> bool {
        self.void_cut_count > 0
    }
}

#[derive(Default)]
struct Stats {
    node_count: usize,
    primitive_count: usize,
    void_cut_count: usize,
    void_node_count: usize,
    intersection_count: usize,
    transform_count: usize,
    max_void_depth: usize,
}

fn visit(node: &CSGNode, depth: usize, in_void_subtree: bool, out: &mut Stats) {
    out.node_count += 1;

    if in_void_subtree {
        out.void_node_count += 1;
        out.max_void_depth = out.max_void_depth.max(depth);
    }

    match node {
        CSGNode::Primitive(_) => {
            out.primitive_count += 1;
        }
        CSGNode::Transform { node, .. } => {
            out.transform_count += 1;
            visit(node, depth + 1, in_void_subtree, out);
        }
        CSGNode::Boolean { op, left, right } => {
            if *op == BooleanOp::Subtract {
                out.void_cut_count += 1;
                visit(left, depth + 1, in_void_subtree, out);
                visit(right, depth + 1, true, out);
            } else {
                if *op == BooleanOp::Intersect {
                    out.intersection_count += 1;
                }
                visit(left, depth + 1, in_void_subtree, out);
                visit(right, depth + 1, in_void_subtree, out);
            }
        }
    }
}

fn hash_node(node: &CSGNode) -> blake3::Hash {
    match node {
        CSGNode::Primitive(primitive) => {
            let mut h = blake3::Hasher::new();
            h.update(b"primitive:");
            h.update(&[match primitive {
                Primitive::Cube => 0,
                Primitive::Cylinder => 1,
                Primitive::Sphere => 2,
                Primitive::Cone => 3,
                Primitive::Torus => 4,
            }]);
            h.finalize()
        }
        CSGNode::Transform { node, transform } => {
            let mut h = blake3::Hasher::new();
            h.update(b"transform:");
            for value in transform.scale {
                h.update(&value.to_le_bytes());
            }
            for value in transform.rotate {
                h.update(&value.to_le_bytes());
            }
            for value in transform.translate {
                h.update(&value.to_le_bytes());
            }
            h.update(hash_node(node).as_bytes());
            h.finalize()
        }
        CSGNode::Boolean { op, left, right } => {
            let left_hash = hash_node(left);
            let right_hash = hash_node(right);
            let (left_bytes, right_bytes) = (left_hash.as_bytes(), right_hash.as_bytes());

            let mut h = blake3::Hasher::new();
            h.update(b"boolean:");
            h.update(&[match op {
                BooleanOp::Union => 0,
                BooleanOp::Subtract => 1,
                BooleanOp::Intersect => 2,
            }]);

            if matches!(op, BooleanOp::Union | BooleanOp::Intersect)
                && right_bytes < left_bytes
            {
                h.update(right_bytes);
                h.update(left_bytes);
            } else {
                h.update(left_bytes);
                h.update(right_bytes);
            }
            h.finalize()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_only_design_has_no_negative_space_intent() {
        let signature = NegativeSpaceSignature::from_csg(&CSGNode::cube());
        assert!(!signature.declares_negative_space());
        assert_eq!(signature.void_cut_count, 0);
        assert_eq!(signature.void_node_count, 0);
        assert_eq!(signature.node_count, 1);
    }

    #[test]
    fn subtraction_is_explicit_negative_space_intent() {
        let signature =
            NegativeSpaceSignature::from_csg(&CSGNode::cube().subtract(CSGNode::cylinder()));

        assert!(signature.declares_negative_space());
        assert_eq!(signature.void_cut_count, 1);
        assert_eq!(signature.void_node_count, 1);
        assert_eq!(signature.primitive_count, 2);
        assert_eq!(signature.max_void_depth, 2);
    }

    #[test]
    fn nested_cut_tracks_the_removed_subtree() {
        let tool = CSGNode::cylinder().subtract(CSGNode::sphere());
        let signature = NegativeSpaceSignature::from_csg(&CSGNode::cube().subtract(tool));

        assert_eq!(signature.void_cut_count, 2);
        assert!(signature.void_node_count >= 2);
        assert!(signature.max_void_depth >= 2);
    }

    #[test]
    fn commutative_operations_have_stable_identity() {
        let a = CSGNode::cube().union(CSGNode::sphere());
        let b = CSGNode::sphere().union(CSGNode::cube());
        assert_eq!(
            NegativeSpaceSignature::from_csg(&a).digest,
            NegativeSpaceSignature::from_csg(&b).digest
        );
    }

    #[test]
    fn subtraction_operand_order_is_semantically_significant() {
        let a = CSGNode::cube().subtract(CSGNode::sphere());
        let b = CSGNode::sphere().subtract(CSGNode::cube());
        assert_ne!(
            NegativeSpaceSignature::from_csg(&a).digest,
            NegativeSpaceSignature::from_csg(&b).digest
        );
    }
}
