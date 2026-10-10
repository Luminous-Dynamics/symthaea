// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Multi-objective search primitives for passive engineering.
//!
//! This crate deliberately keeps search mathematics separate from physics.
//! Physics evaluators produce objective observations; this crate preserves the
//! non-dominated frontier and a deterministic structural identity for CSG
//! candidates.

/// Whether larger or smaller values are preferred for one objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

/// A finite multi-objective observation.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveVector {
    pub values: Vec<f64>,
}

impl ObjectiveVector {
    pub fn new(values: Vec<f64>) -> Option<Self> {
        if values.iter().all(|value| value.is_finite()) {
            Some(Self { values })
        } else {
            None
        }
    }

    /// True when this point is no worse in every objective and better in at
    /// least one objective.
    pub fn dominates(
        &self,
        other: &Self,
        directions: &[ObjectiveDirection],
    ) -> bool {
        if self.values.len() != other.values.len() || self.values.len() != directions.len() {
            return false;
        }

        let mut strictly_better = false;
        for ((a, b), direction) in self
            .values
            .iter()
            .zip(&other.values)
            .zip(directions)
        {
            let (better_or_equal, better) = match direction {
                ObjectiveDirection::Maximize => (a >= b, a > b),
                ObjectiveDirection::Minimize => (a <= b, a < b),
            };
            if !better_or_equal {
                return false;
            }
            strictly_better |= better;
        }
        strictly_better
    }
}

/// A design payload paired with its objective observation.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoCandidate<T> {
    pub payload: T,
    pub objectives: ObjectiveVector,
}

/// Result of attempting to add a candidate to the frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    Inserted { removed: usize },
    Dominated,
    Invalid,
}

/// A deterministic archive of currently non-dominated candidates.
///
/// Equal objective vectors are retained deliberately: two geometrically
/// different designs can have identical measured objectives and still be
/// valuable for diversity, manufacturability, or future re-evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoArchive<T> {
    directions: Vec<ObjectiveDirection>,
    entries: Vec<ParetoCandidate<T>>,
}

impl<T: Clone + PartialEq> ParetoArchive<T> {
    pub fn new(directions: Vec<ObjectiveDirection>) -> Self {
        Self {
            directions,
            entries: Vec::new(),
        }
    }

    pub fn directions(&self) -> &[ObjectiveDirection] {
        &self.directions
    }

    pub fn entries(&self) -> &[ParetoCandidate<T>] {
        &self.entries
    }

    pub fn insert(&mut self, candidate: ParetoCandidate<T>) -> InsertOutcome {
        if candidate.objectives.values.len() != self.directions.len()
            || candidate.objectives.values.iter().any(|v| !v.is_finite())
        {
            return InsertOutcome::Invalid;
        }

        if self
            .entries
            .iter()
            .any(|existing| existing.objectives.dominates(&candidate.objectives, &self.directions))
        {
            return InsertOutcome::Dominated;
        }

        let before = self.entries.len();
        self.entries.retain(|existing| {
            !candidate
                .objectives
                .dominates(&existing.objectives, &self.directions)
        });
        let removed = before - self.entries.len();
        self.entries.push(candidate);
        InsertOutcome::Inserted { removed }
    }
}

/// Stable structural fingerprint of a CSG tree.
///
/// The fingerprint captures both topology and exact transform parameters. For
/// union/intersection nodes, child fingerprints are sorted so equivalent
/// commutative trees receive the same structural identity. Subtraction keeps
/// operand order because it is not commutative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsgFingerprint {
    pub digest: [u8; 32],
    pub node_count: usize,
    pub primitive_count: usize,
    pub transform_count: usize,
    pub boolean_count: usize,
    pub max_depth: usize,
}

impl CsgFingerprint {
    pub fn from_csg(node: &symthaea_fabrication_kernel::csg::CSGNode) -> Self {
        let stats = stats(node);
        let hash = hash_node(node);
        Self {
            digest: *hash.as_bytes(),
            node_count: stats.node_count,
            primitive_count: stats.primitive_count,
            transform_count: stats.transform_count,
            boolean_count: stats.boolean_count,
            max_depth: stats.max_depth,
        }
    }
}

#[derive(Default)]
struct Stats {
    node_count: usize,
    primitive_count: usize,
    transform_count: usize,
    boolean_count: usize,
    max_depth: usize,
}

fn stats(node: &symthaea_fabrication_kernel::csg::CSGNode) -> Stats {
    fn visit(
        node: &symthaea_fabrication_kernel::csg::CSGNode,
        depth: usize,
        out: &mut Stats,
    ) {
        out.node_count += 1;
        out.max_depth = out.max_depth.max(depth);
        match node {
            symthaea_fabrication_kernel::csg::CSGNode::Primitive(_) => {
                out.primitive_count += 1;
            }
            symthaea_fabrication_kernel::csg::CSGNode::Transform { node, .. } => {
                out.transform_count += 1;
                visit(node, depth + 1, out);
            }
            symthaea_fabrication_kernel::csg::CSGNode::Boolean { left, right, .. } => {
                out.boolean_count += 1;
                visit(left, depth + 1, out);
                visit(right, depth + 1, out);
            }
        }
    }

    let mut out = Stats::default();
    visit(node, 1, &mut out);
    out
}

fn hash_node(node: &symthaea_fabrication_kernel::csg::CSGNode) -> blake3::Hash {
    use symthaea_fabrication_kernel::csg::{BooleanOp, CSGNode, Primitive};

    match node {
        CSGNode::Primitive(primitive) => {
            let mut h = blake3::Hasher::new();
            h.update(b"primitive:");
            h.update(format!("{primitive:?}").as_bytes());
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
            let mut children = [left_hash.as_bytes(), right_hash.as_bytes()];
            if matches!(op, BooleanOp::Union | BooleanOp::Intersect)
                && children[1] < children[0]
            {
                children.swap(0, 1);
            }

            let mut h = blake3::Hasher::new();
            h.update(b"boolean:");
            h.update(format!("{op:?}").as_bytes());
            h.update(children[0]);
            h.update(children[1]);
            h.finalize()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::CSGNode;

    #[test]
    fn dominance_requires_strict_improvement() {
        let dirs = [ObjectiveDirection::Maximize, ObjectiveDirection::Maximize];
        let a = ObjectiveVector::new(vec![0.9, 0.5]).unwrap();
        let b = ObjectiveVector::new(vec![0.8, 0.5]).unwrap();
        let c = ObjectiveVector::new(vec![0.9, 0.5]).unwrap();
        assert!(a.dominates(&b, &dirs));
        assert!(!a.dominates(&c, &dirs));
    }

    #[test]
    fn archive_removes_dominated_points_and_keeps_equal_points() {
        let mut archive = ParetoArchive::new(vec![
            ObjectiveDirection::Maximize,
            ObjectiveDirection::Maximize,
        ]);

        assert_eq!(
            archive.insert(ParetoCandidate {
                payload: "a",
                objectives: ObjectiveVector::new(vec![0.7, 0.7]).unwrap(),
            }),
            InsertOutcome::Inserted { removed: 0 }
        );
        assert_eq!(
            archive.insert(ParetoCandidate {
                payload: "b",
                objectives: ObjectiveVector::new(vec![0.8, 0.6]).unwrap(),
            }),
            InsertOutcome::Inserted { removed: 0 }
        );
        assert_eq!(
            archive.insert(ParetoCandidate {
                payload: "c",
                objectives: ObjectiveVector::new(vec![0.9, 0.9]).unwrap(),
            }),
            InsertOutcome::Inserted { removed: 2 }
        );
        assert_eq!(
            archive.insert(ParetoCandidate {
                payload: "d",
                objectives: ObjectiveVector::new(vec![0.9, 0.9]).unwrap(),
            }),
            InsertOutcome::Inserted { removed: 0 }
        );
        assert_eq!(archive.entries().len(), 2);
    }

    #[test]
    fn invalid_vectors_are_rejected() {
        let mut archive = ParetoArchive::<&'static str>::new(vec![ObjectiveDirection::Maximize]);
        let candidate = ParetoCandidate {
            payload: "bad",
            objectives: ObjectiveVector {
                values: vec![f64::NAN],
            },
        };
        assert_eq!(archive.insert(candidate), InsertOutcome::Invalid);
    }

    #[test]
    fn csg_fingerprint_is_deterministic() {
        let a = CsgFingerprint::from_csg(&CSGNode::cube());
        let b = CsgFingerprint::from_csg(&CSGNode::cube());
        assert_eq!(a, b);
        assert_eq!(a.node_count, 1);
        assert_eq!(a.primitive_count, 1);
        assert_eq!(a.max_depth, 1);
    }

    #[test]
    fn union_fingerprint_is_commutative_but_subtract_is_not() {
        let left = CSGNode::cube().union(CSGNode::sphere());
        let right = CSGNode::sphere().union(CSGNode::cube());
        assert_eq!(
            CsgFingerprint::from_csg(&left).digest,
            CsgFingerprint::from_csg(&right).digest
        );

        let a = CSGNode::cube();
        let b = CSGNode::sphere();
        let subtract_left = CsgFingerprint::from_csg(&a.clone().subtract(b.clone())).digest;
        let subtract_right = CsgFingerprint::from_csg(&b.subtract(a)).digest;
        assert_ne!(subtract_left, subtract_right);
    }
}
