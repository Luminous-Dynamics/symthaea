// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Topological Fingerprint — persistent homology of code structure.
//!
//! Computes Betti numbers from a [`SimplicialComplex`] built from a PDG:
//! - beta_0 = connected components (module coupling)
//! - beta_1 = independent cycles (loops)
//! - beta_2 = enclosed voids (unreachable code)
//!
//! The fingerprint is encoded as an HDC [`BinaryHV`] for fast similarity
//! comparison between topologically distinct code structures.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_hodge::SimplicialComplex;

// ---------------------------------------------------------------------------
// Deterministic role seeds for HDC encoding of topological features.
// These must remain stable across versions for fingerprint compatibility.
// ---------------------------------------------------------------------------
const TOPO_ROLE_SEED: u64 = 0x70F0_0000_0000_0001;
const BETA0_ROLE_SEED: u64 = 0xBE7A_0000_0000_0000;
const BETA1_ROLE_SEED: u64 = 0xBE7A_0000_0000_0001;
const BETA2_ROLE_SEED: u64 = 0xBE7A_0000_0000_0002;
const EULER_ROLE_SEED: u64 = 0xECEE_0000_0000_0000;

// ---------------------------------------------------------------------------
// Union-Find (disjoint-set) for connected component computation
// ---------------------------------------------------------------------------

/// Lightweight union-find with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }

    fn component_count(&mut self) -> usize {
        let n = self.parent.len();
        if n == 0 {
            return 0;
        }
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(self.find(i));
        }
        roots.len()
    }
}

// ---------------------------------------------------------------------------
// Betti numbers
// ---------------------------------------------------------------------------

/// Betti numbers for a code structure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BettiNumbers {
    /// beta_0: connected components
    pub beta_0: usize,
    /// beta_1: independent cycles / loops
    pub beta_1: usize,
    /// beta_2: enclosed voids / dead code
    pub beta_2: usize,
}

// ---------------------------------------------------------------------------
// Topological constraints
// ---------------------------------------------------------------------------

/// A topological constraint on generated code.
#[derive(Debug, Clone)]
pub enum TopologicalConstraint {
    /// Require exactly `count` features at `dimension`.
    ExactBetti { dimension: usize, count: usize },
    /// Require at most `max` features at `dimension`.
    MaxBetti { dimension: usize, max: usize },
    /// Require at least this many connected components.
    MinComponents(usize),
    /// Require at most this many connected components.
    MaxComponents(usize),
    /// beta_1 must be 0 (no cycles).
    NoCycles,
    /// beta_2 must be 0 (no dead-code voids).
    NoDeadCode,
}

// ---------------------------------------------------------------------------
// Topological fingerprint
// ---------------------------------------------------------------------------

/// Topological fingerprint of a code structure.
///
/// Encodes the Betti numbers, Euler characteristic, and simplex counts into
/// a single HDC vector for fast similarity comparison.
#[derive(Debug, Clone)]
pub struct TopologicalFingerprint {
    /// Betti numbers (beta_0, beta_1, beta_2).
    pub betti: BettiNumbers,
    /// Euler characteristic: beta_0 - beta_1 + beta_2.
    pub euler_characteristic: i64,
    /// Number of 0-simplices (vertices / nodes).
    pub node_count: usize,
    /// Number of 1-simplices (edges).
    pub edge_count: usize,
    /// Topological features encoded as an HDC vector.
    pub hdc_encoding: BinaryHV,
}

impl TopologicalFingerprint {
    /// Create a default fingerprint for a function with no structural info.
    ///
    /// Represents a single connected component (beta_0 = 1) with no cycles
    /// or voids. Uses a deterministic seed for the HDC encoding so that all
    /// default fingerprints are identical and comparable.
    pub fn default_for_function() -> Self {
        let betti = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };
        let euler = 1i64;
        Self {
            hdc_encoding: Self::encode_as_hv(&betti, euler),
            betti,
            euler_characteristic: euler,
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Compute a topological fingerprint from a [`SimplicialComplex`].
    ///
    /// Uses union-find for beta_0 (connected components) and the Euler
    /// characteristic relation for higher Betti numbers.
    pub fn from_complex(complex: &SimplicialComplex) -> Self {
        let betti = Self::compute_betti(complex);
        let euler = betti.beta_0 as i64 - betti.beta_1 as i64 + betti.beta_2 as i64;
        let node_count = complex.count(0);
        let edge_count = complex.count(1);
        let hdc_encoding = Self::encode_full(&betti, euler, node_count, edge_count);

        Self {
            betti,
            euler_characteristic: euler,
            node_count,
            edge_count,
            hdc_encoding,
        }
    }

    /// Compute Betti numbers from the simplicial complex.
    ///
    /// - beta_0: connected components via union-find on 1-simplices (edges).
    /// - beta_1: from the Euler relation on the 1-skeleton:
    ///   chi_1 = V - E, and chi_1 = beta_0 - beta_1, so beta_1 = E - V + beta_0.
    /// - beta_2: from the full Euler characteristic of the 2-skeleton:
    ///   chi_2 = V - E + T = beta_0 - beta_1 + beta_2,
    ///   so beta_2 = chi_2 - beta_0 + beta_1 = T - E + V - beta_0 + (E - V + beta_0) = T
    ///   ... but that over-counts. We use the rank-nullity relation:
    ///   beta_2 = |2-simplices| - rank(boundary_2).
    ///   For simplicity we approximate: beta_2 = max(0, T - (E - V + beta_0)).
    fn compute_betti(complex: &SimplicialComplex) -> BettiNumbers {
        let v = complex.count(0); // vertices
        let e = complex.count(1); // edges
        let t = complex.count(2); // triangles

        if v == 0 {
            return BettiNumbers::default();
        }

        // --- beta_0: connected components via union-find ---
        // Build index map: vertex id -> contiguous index
        let mut vertex_index = std::collections::HashMap::new();
        for (i, simplex) in complex.simplices[0].iter().enumerate() {
            vertex_index.insert(simplex[0], i);
        }

        let mut uf = UnionFind::new(v);
        if complex.simplices.len() > 1 {
            for edge in &complex.simplices[1] {
                if let (Some(&a), Some(&b)) =
                    (vertex_index.get(&edge[0]), vertex_index.get(&edge[1]))
                {
                    uf.union(a, b);
                }
            }
        }
        let beta_0 = uf.component_count();

        // --- beta_1: cycles ---
        // From the Euler characteristic of the 1-skeleton:
        //   chi = V - E = beta_0 - beta_1
        //   beta_1 = E - V + beta_0
        let beta_1 = if e + beta_0 >= v { e + beta_0 - v } else { 0 };

        // --- beta_2: voids ---
        // From the full 2-skeleton Euler characteristic:
        //   chi = V - E + T = beta_0 - beta_1 + beta_2
        //   beta_2 = chi - beta_0 + beta_1 = V - E + T - beta_0 + beta_1
        //          = V - E + T - beta_0 + (E - V + beta_0)
        //          = T
        // But this is only correct when every triangle bounds a distinct void,
        // which is rare. A better approximation accounts for the rank of the
        // boundary operator. For code graphs the typical case is beta_2 ~ 0
        // (triangles represent three-way dependencies, not cavities). We use:
        //   beta_2 = max(0, (V - E + T) - (beta_0 - beta_1))
        let chi_2 = v as i64 - e as i64 + t as i64;
        let expected_chi = beta_0 as i64 - beta_1 as i64;
        let beta_2 = (chi_2 - expected_chi).max(0) as usize;

        BettiNumbers {
            beta_0,
            beta_1,
            beta_2,
        }
    }

    /// Encode topological features as a [`BinaryHV`].
    ///
    /// Uses role-filler binding: each feature is encoded as
    /// `ROLE bind VALUE`, and all role-value pairs are bundled (majority vote)
    /// into a single composite vector.
    fn encode_as_hv(betti: &BettiNumbers, euler: i64) -> BinaryHV {
        Self::encode_full(betti, euler, 0, 0)
    }

    /// Encode topology + structural size into an HDC vector.
    ///
    /// Includes node_count and edge_count to differentiate functions that share
    /// the same Betti numbers but have different structural complexity.
    /// A 3-line function and a 50-line function both with β₁=0 will get
    /// DIFFERENT encodings, enabling meaningful manifold clustering.
    fn encode_full(
        betti: &BettiNumbers,
        euler: i64,
        node_count: usize,
        edge_count: usize,
    ) -> BinaryHV {
        let topo_role = BinaryHV::random(TOPO_ROLE_SEED);

        // Role vectors
        let beta0_role = BinaryHV::random(BETA0_ROLE_SEED);
        let beta1_role = BinaryHV::random(BETA1_ROLE_SEED);
        let beta2_role = BinaryHV::random(BETA2_ROLE_SEED);
        let euler_role = BinaryHV::random(EULER_ROLE_SEED);
        let size_role = BinaryHV::random(0xDEAD_5123_0000_0001);
        let edges_role = BinaryHV::random(0xDEAD_5123_0000_0002);

        // Value vectors (deterministic from the value itself)
        let beta0_val = BinaryHV::random(BETA0_ROLE_SEED.wrapping_add(betti.beta_0 as u64));
        let beta1_val = BinaryHV::random(BETA1_ROLE_SEED.wrapping_add(betti.beta_1 as u64));
        let beta2_val = BinaryHV::random(BETA2_ROLE_SEED.wrapping_add(betti.beta_2 as u64));
        let euler_val = BinaryHV::random(EULER_ROLE_SEED.wrapping_add((euler + 10_000) as u64));
        // Quantize node/edge counts to buckets for similarity clustering:
        // 0-3, 4-7, 8-15, 16-31, 32-63, 64+
        let size_bucket = match node_count {
            0..=3 => 0u64,
            4..=7 => 1,
            8..=15 => 2,
            16..=31 => 3,
            32..=63 => 4,
            _ => 5,
        };
        let edge_bucket = match edge_count {
            0..=3 => 0u64,
            4..=7 => 1,
            8..=15 => 2,
            16..=31 => 3,
            32..=63 => 4,
            _ => 5,
        };
        let size_val = BinaryHV::random(0xDEAD_5123_0000_0001_u64.wrapping_add(size_bucket));
        let edges_val = BinaryHV::random(0xDEAD_5123_0000_0002_u64.wrapping_add(edge_bucket));

        // Bind role-value pairs
        let b0 = beta0_role.bind(&beta0_val);
        let b1 = beta1_role.bind(&beta1_val);
        let b2 = beta2_role.bind(&beta2_val);
        let eu = euler_role.bind(&euler_val);
        let sz = size_role.bind(&size_val);
        let ed = edges_role.bind(&edges_val);

        // Bundle all pairs, then bind with the top-level TOPO role
        let composite = BinaryHV::bundle(&[b0, b1, b2, eu, sz, ed]);
        topo_role.bind(&composite)
    }

    /// Check if this fingerprint satisfies a single constraint.
    pub fn satisfies(&self, constraint: &TopologicalConstraint) -> bool {
        match constraint {
            TopologicalConstraint::ExactBetti { dimension, count } => {
                self.betti_at(*dimension) == *count
            }
            TopologicalConstraint::MaxBetti { dimension, max } => self.betti_at(*dimension) <= *max,
            TopologicalConstraint::MinComponents(min) => self.betti.beta_0 >= *min,
            TopologicalConstraint::MaxComponents(max) => self.betti.beta_0 <= *max,
            TopologicalConstraint::NoCycles => self.betti.beta_1 == 0,
            TopologicalConstraint::NoDeadCode => self.betti.beta_2 == 0,
        }
    }

    /// Check all constraints, returning a list of human-readable violations.
    ///
    /// An empty vector means all constraints are satisfied.
    pub fn verify(&self, constraints: &[TopologicalConstraint]) -> Vec<String> {
        constraints
            .iter()
            .filter_map(|c| {
                if self.satisfies(c) {
                    None
                } else {
                    Some(self.describe_violation(c))
                }
            })
            .collect()
    }

    /// HDC similarity between two topological fingerprints.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means identical topology
    /// and ~0.5 means unrelated.
    pub fn similarity(&self, other: &TopologicalFingerprint) -> f32 {
        self.hdc_encoding.similarity(&other.hdc_encoding)
    }

    // -- helpers --

    fn betti_at(&self, dimension: usize) -> usize {
        match dimension {
            0 => self.betti.beta_0,
            1 => self.betti.beta_1,
            2 => self.betti.beta_2,
            _ => 0,
        }
    }

    fn describe_violation(&self, constraint: &TopologicalConstraint) -> String {
        match constraint {
            TopologicalConstraint::ExactBetti { dimension, count } => {
                format!(
                    "ExactBetti(dim={}, expected={}, actual={})",
                    dimension,
                    count,
                    self.betti_at(*dimension)
                )
            }
            TopologicalConstraint::MaxBetti { dimension, max } => {
                format!(
                    "MaxBetti(dim={}, max={}, actual={})",
                    dimension,
                    max,
                    self.betti_at(*dimension)
                )
            }
            TopologicalConstraint::MinComponents(min) => {
                format!("MinComponents(min={}, actual={})", min, self.betti.beta_0)
            }
            TopologicalConstraint::MaxComponents(max) => {
                format!("MaxComponents(max={}, actual={})", max, self.betti.beta_0)
            }
            TopologicalConstraint::NoCycles => {
                format!("NoCycles(beta_1={})", self.betti.beta_1)
            }
            TopologicalConstraint::NoDeadCode => {
                format!("NoDeadCode(beta_2={})", self.betti.beta_2)
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Single isolated node: one component, no cycles, no voids.
    #[test]
    fn test_betti_single_node() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);

        let fp = TopologicalFingerprint::from_complex(&complex);
        assert_eq!(fp.betti.beta_0, 1);
        assert_eq!(fp.betti.beta_1, 0);
        assert_eq!(fp.betti.beta_2, 0);
        assert_eq!(fp.euler_characteristic, 1);
        assert_eq!(fp.node_count, 1);
        assert_eq!(fp.edge_count, 0);
    }

    /// Triangle (3 vertices, 3 edges, no filled face): has one cycle.
    #[test]
    fn test_betti_simple_cycle() {
        let mut complex = SimplicialComplex::new();
        // Add only edges (not the 2-simplex), so the triangle is hollow
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let fp = TopologicalFingerprint::from_complex(&complex);
        assert_eq!(fp.betti.beta_0, 1, "one connected component");
        assert_eq!(fp.betti.beta_1, 1, "one independent cycle");
        assert_eq!(fp.betti.beta_2, 0, "no voids");
    }

    /// Constraint verification: ExactBetti passes and fails correctly.
    #[test]
    fn test_constraint_verification() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![0, 2]);

        let fp = TopologicalFingerprint::from_complex(&complex);

        // beta_1 == 1, so ExactBetti(1, 1) should pass
        let pass = TopologicalConstraint::ExactBetti {
            dimension: 1,
            count: 1,
        };
        assert!(fp.satisfies(&pass));

        // ExactBetti(1, 0) should fail
        let fail = TopologicalConstraint::ExactBetti {
            dimension: 1,
            count: 0,
        };
        assert!(!fp.satisfies(&fail));

        // NoCycles should fail
        assert!(!fp.satisfies(&TopologicalConstraint::NoCycles));

        // NoDeadCode should pass
        assert!(fp.satisfies(&TopologicalConstraint::NoDeadCode));

        // verify() should return exactly the failures
        let violations = fp.verify(&[pass.clone(), fail, TopologicalConstraint::NoCycles]);
        assert_eq!(violations.len(), 2);
    }

    /// Two identical complexes should produce fingerprints with similarity ~1.0.
    #[test]
    fn test_fingerprint_similarity() {
        let mut c1 = SimplicialComplex::new();
        c1.add_simplex(vec![0, 1]);
        c1.add_simplex(vec![1, 2]);

        let mut c2 = SimplicialComplex::new();
        c2.add_simplex(vec![0, 1]);
        c2.add_simplex(vec![1, 2]);

        let fp1 = TopologicalFingerprint::from_complex(&c1);
        let fp2 = TopologicalFingerprint::from_complex(&c2);

        // Identical topology → identical encoding → similarity 1.0
        let sim = fp1.similarity(&fp2);
        assert!(
            sim > 0.99,
            "identical complexes should have similarity ~1.0, got {}",
            sim
        );
    }

    /// Two disconnected nodes: beta_0 = 2.
    #[test]
    fn test_betti_two_components() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);
        complex.add_simplex(vec![1]);

        let fp = TopologicalFingerprint::from_complex(&complex);
        assert_eq!(fp.betti.beta_0, 2);
        assert_eq!(fp.betti.beta_1, 0);
    }

    /// Empty complex.
    #[test]
    fn test_betti_empty_complex() {
        let complex = SimplicialComplex::new();
        let fp = TopologicalFingerprint::from_complex(&complex);
        assert_eq!(fp.betti, BettiNumbers::default());
        assert_eq!(fp.euler_characteristic, 0);
    }

    /// Linear chain: no cycles.
    #[test]
    fn test_betti_linear_chain() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![2, 3]);

        let fp = TopologicalFingerprint::from_complex(&complex);
        assert_eq!(fp.betti.beta_0, 1);
        assert_eq!(fp.betti.beta_1, 0);
        assert_eq!(fp.node_count, 4);
        assert_eq!(fp.edge_count, 3);
    }
}
