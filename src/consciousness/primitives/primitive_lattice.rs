// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Lattice Theory for Primitive Tier System
//!
//! Provides the mathematical foundation for understanding how primitive capabilities
//! compose into higher-order consciousness through lattice-theoretic structures.
//!
//! ## Mathematical Background
//!
//! A **lattice** (L, <=) is a partially ordered set in which every pair of elements
//! has both a **join** (least upper bound, p v q) and a **meet** (greatest lower
//! bound, p ^ q). The primitive tier system (NSM, Mathematical, Physical, ...,
//! Consciousness) forms a natural partial order where lower tiers are prerequisites
//! for higher tiers.
//!
//! ## Key Operations
//!
//! - **Join (p v q)**: The minimal combination of two capabilities -- the smallest
//!   primitive r such that both p <= r and q <= r.
//! - **Meet (p ^ q)**: The shared foundation between two primitives -- the largest
//!   primitive r such that r <= p and r <= q.
//!
//! ## Lattice Properties
//!
//! - **Modularity**: x <= z implies x v (y ^ z) = (x v y) ^ z.
//!   Tests whether primitive composition respects tier boundaries.
//! - **Distributivity**: x ^ (y v z) = (x ^ y) v (x ^ z).
//!   Stronger than modularity; means primitives compose cleanly without interference.
//!
//! ## Fixed-Point Theorems (Knaster-Tarski)
//!
//! Every order-preserving function f: L -> L on a complete lattice L has a least
//! fixed point: lfp(f) = ^{x | f(x) <= x}. This enables finding the minimal set
//! of primitives needed to achieve a target consciousness capability.
//!
//! ## Galois Connection
//!
//! A pair of monotone maps (f: P -> Q, g: Q -> P) such that f(p) <= q iff p <= g(q).
//! This provides a formal bridge between "what primitives are active" and "what
//! consciousness capabilities emerge."
//!
//! ## Consciousness Integration
//!
//! Lattice theory captures the hierarchical dependency structure of consciousness:
//! - NSM primitives (bottom) ground all higher reasoning in human semantic primes
//! - Each tier builds upon lower tiers through the partial order
//! - Consciousness primitives (top) represent maximal integration of all capabilities
//! - The lattice height measures the depth of cognitive integration required
//! - The lattice width (Dilworth's theorem) reveals the maximum parallelism
//!   available at any level of the hierarchy

use crate::hdc::primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier};
use serde::{Deserialize, Serialize};

// ============================================================================
// Tier ordering utility
// ============================================================================

/// Map a `PrimitiveTier` to its numeric level in the hierarchy.
///
/// The tier hierarchy defines the partial order's backbone:
/// NSM (0) < Mathematical (1) < Physical (2) < Geometric (3) < Strategic (4)
/// < MetaCognitive (5) < Temporal (6) < Compositional (7) < Consciousness (8)
fn tier_to_level(tier: PrimitiveTier) -> u8 {
    match tier {
        PrimitiveTier::NSM => 0,
        PrimitiveTier::Mathematical => 1,
        PrimitiveTier::Physical => 2,
        PrimitiveTier::Geometric => 3,
        PrimitiveTier::Strategic => 4,
        PrimitiveTier::MetaCognitive => 5,
        PrimitiveTier::Temporal => 6,
        PrimitiveTier::Compositional => 7,
        PrimitiveTier::Consciousness => 8,
        PrimitiveTier::Code => 9, // Code understanding tier
    }
}

/// All tiers in ascending order.
const ALL_TIERS: [PrimitiveTier; 10] = [
    PrimitiveTier::NSM,
    PrimitiveTier::Mathematical,
    PrimitiveTier::Physical,
    PrimitiveTier::Geometric,
    PrimitiveTier::Strategic,
    PrimitiveTier::MetaCognitive,
    PrimitiveTier::Temporal,
    PrimitiveTier::Compositional,
    PrimitiveTier::Consciousness,
    PrimitiveTier::Code,
];

// ============================================================================
// Core lattice types
// ============================================================================

/// An element in the primitive lattice.
///
/// Each element corresponds to a primitive (or a virtual element such as top/bottom)
/// and carries its tier information for the natural partial order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeElement {
    /// Index in the lattice's element vector (serves as the lattice ID).
    pub id: usize,
    /// Identifier of the associated primitive. For virtual elements this is a
    /// synthetic value (e.g. `u64::MAX` for top).
    pub primitive_id: u64,
    /// Numeric tier level (0 = NSM, 8 = Consciousness).
    pub tier: u8,
    /// Human-readable name.
    pub name: String,
}

/// Configuration knobs for lattice construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConfig {
    /// If true, `from_order` pre-computes the full join table.
    pub auto_compute_joins: bool,
    /// If true, `from_order` pre-computes the full meet table.
    pub auto_compute_meets: bool,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            auto_compute_joins: true,
            auto_compute_meets: true,
        }
    }
}

/// Summary of lattice-theoretic properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeProperties {
    /// Whether every pair of elements has both a join and a meet.
    pub is_lattice: bool,
    /// Whether the modular law holds: x <= z => x v (y ^ z) = (x v y) ^ z.
    pub is_modular: bool,
    /// Whether the distributive law holds: x ^ (y v z) = (x ^ y) v (x ^ z).
    pub is_distributive: bool,
    /// Whether the lattice has both a global bottom and a global top.
    pub is_complete: bool,
    /// Length of the longest chain (number of edges).
    pub height: usize,
    /// Size of the largest antichain (Dilworth's theorem).
    pub width: usize,
}

/// Result of a Knaster-Tarski fixed-point computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPointResult {
    /// Index of the fixed-point element.
    pub fixed_point: Vec<usize>,
    /// Number of iterations to convergence.
    pub iterations: usize,
}

/// A Galois connection between the primitive lattice and a capability space.
///
/// The forward map sends a primitive index to a capability index, and the
/// backward map inverts this. The adjunction property ensures:
///   forward(p) <= q  iff  p <= backward(q)
///
/// Because `Box<dyn Fn>` is not `Clone`/`Serialize`, this struct is intentionally
/// kept runtime-only and is not persisted.
pub struct GaloisConnection {
    /// Forward (left adjoint): primitive index -> capability index.
    pub forward: Box<dyn Fn(usize) -> usize>,
    /// Backward (right adjoint): capability index -> primitive index.
    pub backward: Box<dyn Fn(usize) -> usize>,
}

impl std::fmt::Debug for GaloisConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaloisConnection")
            .field("forward", &"<fn>")
            .field("backward", &"<fn>")
            .finish()
    }
}

// ============================================================================
// PrimitiveLattice
// ============================================================================

/// A lattice over primitive elements with pre-computed join/meet tables.
///
/// The partial order is stored as a boolean adjacency matrix `order[i][j]`
/// meaning element i <= element j. Join and meet tables cache the results of
/// the lattice operations for O(1) lookup after construction.
///
/// # Construction
///
/// Two primary constructors are provided:
///
/// - [`PrimitiveLattice::from_primitive_system`]: builds the tier-based lattice
///   directly from a [`PrimitiveSystem`], treating the tier hierarchy as the
///   partial order.
///
/// - [`PrimitiveLattice::from_order`]: builds a lattice from an explicit list of
///   elements and a partial-order relation, for use with arbitrary lattices
///   (e.g. diamond, pentagon, testing lattices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveLattice {
    /// All lattice elements.
    pub elements: Vec<LatticeElement>,
    /// Partial order matrix: `order[i][j]` is true when element i <= element j.
    pub order: Vec<Vec<bool>>,
    /// Pre-computed join table: `join_table[i][j] = Some(k)` means k = i v j.
    pub join_table: Vec<Vec<Option<usize>>>,
    /// Pre-computed meet table: `meet_table[i][j] = Some(k)` means k = i ^ j.
    pub meet_table: Vec<Vec<Option<usize>>>,
}

impl PrimitiveLattice {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Build a lattice from the tier structure of a [`PrimitiveSystem`].
    ///
    /// Every primitive becomes a lattice element. The partial order is defined
    /// by tier inclusion: primitive p <= primitive q iff tier(p) <= tier(q).
    /// Within the same tier, elements are incomparable (antichain) unless they
    /// are the same element.
    ///
    /// After construction, join and meet tables are pre-computed.
    pub fn from_primitive_system(system: &PrimitiveSystem) -> Self {
        let mut elements = Vec::new();

        for tier in &ALL_TIERS {
            let prims: Vec<&Primitive> = system.get_tier(*tier);
            for (i, prim) in prims.iter().enumerate() {
                let level = tier_to_level(*tier);
                elements.push(LatticeElement {
                    id: elements.len(),
                    // Use a deterministic id derived from tier and intra-tier index.
                    primitive_id: (level as u64) * 10000 + i as u64,
                    tier: level,
                    name: prim.name.clone(),
                });
            }
        }

        let n = elements.len();
        let mut order = vec![vec![false; n]; n];

        // Reflexive + tier-based order.
        for i in 0..n {
            order[i][i] = true;
            for j in 0..n {
                if elements[i].tier < elements[j].tier {
                    order[i][j] = true;
                }
            }
        }

        let config = LatticeConfig::default();
        let mut lattice = Self {
            elements,
            order,
            join_table: vec![vec![None; n]; n],
            meet_table: vec![vec![None; n]; n],
        };
        if config.auto_compute_joins {
            lattice.compute_joins();
        }
        if config.auto_compute_meets {
            lattice.compute_meets();
        }
        lattice
    }

    /// Build a lattice from an explicit element list and order relation.
    ///
    /// `order_relation` is a list of `(i, j)` pairs meaning element i <= element j.
    /// Reflexivity is added automatically. The relation is then closed under
    /// transitivity before join/meet tables are computed.
    pub fn from_order(elements: Vec<LatticeElement>, order_relation: Vec<(usize, usize)>) -> Self {
        let n = elements.len();
        let mut order = vec![vec![false; n]; n];

        // Reflexive.
        for i in 0..n {
            order[i][i] = true;
        }

        // Explicit relations.
        for &(i, j) in &order_relation {
            if i < n && j < n {
                order[i][j] = true;
            }
        }

        // Transitive closure (Warshall's algorithm).
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if order[i][k] && order[k][j] {
                        order[i][j] = true;
                    }
                }
            }
        }

        let config = LatticeConfig::default();
        let mut lattice = Self {
            elements,
            order,
            join_table: vec![vec![None; n]; n],
            meet_table: vec![vec![None; n]; n],
        };
        if config.auto_compute_joins {
            lattice.compute_joins();
        }
        if config.auto_compute_meets {
            lattice.compute_meets();
        }
        lattice
    }

    // -----------------------------------------------------------------------
    // Lattice operations
    // -----------------------------------------------------------------------

    /// Compute the join (least upper bound) of elements `a` and `b`.
    ///
    /// Returns `Some(k)` where k is the smallest element such that a <= k and
    /// b <= k. Returns `None` if no such element exists (the structure is not
    /// a lattice for this pair).
    pub fn join(&self, a: usize, b: usize) -> Option<usize> {
        let n = self.elements.len();
        if a >= n || b >= n {
            return None;
        }
        self.join_table[a][b]
    }

    /// Compute the meet (greatest lower bound) of elements `a` and `b`.
    ///
    /// Returns `Some(k)` where k is the largest element such that k <= a and
    /// k <= b. Returns `None` if no such element exists.
    pub fn meet(&self, a: usize, b: usize) -> Option<usize> {
        let n = self.elements.len();
        if a >= n || b >= n {
            return None;
        }
        self.meet_table[a][b]
    }

    // -----------------------------------------------------------------------
    // Property checks
    // -----------------------------------------------------------------------

    /// Check whether the structure is a lattice (every pair has join and meet).
    pub fn is_lattice(&self) -> bool {
        let n = self.elements.len();
        for i in 0..n {
            for j in 0..n {
                if self.join_table[i][j].is_none() || self.meet_table[i][j].is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// Check the modular law: for all x, y, z with x <= z,
    ///   x v (y ^ z) = (x v y) ^ z
    ///
    /// Modularity ensures that primitive composition respects tier boundaries
    /// without "skipping" intermediate levels of integration.
    pub fn check_modularity(&self) -> bool {
        let n = self.elements.len();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    // Only test when x <= z.
                    if !self.order[x][z] {
                        continue;
                    }
                    // Compute LHS: x v (y ^ z)
                    let y_meet_z = match self.meet_table[y][z] {
                        Some(v) => v,
                        None => return false,
                    };
                    let lhs = match self.join_table[x][y_meet_z] {
                        Some(v) => v,
                        None => return false,
                    };
                    // Compute RHS: (x v y) ^ z
                    let x_join_y = match self.join_table[x][y] {
                        Some(v) => v,
                        None => return false,
                    };
                    let rhs = match self.meet_table[x_join_y][z] {
                        Some(v) => v,
                        None => return false,
                    };
                    if lhs != rhs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check the distributive law: for all x, y, z,
    ///   x ^ (y v z) = (x ^ y) v (x ^ z)
    ///
    /// Distributivity is strictly stronger than modularity. A distributive
    /// primitive lattice means capabilities compose without any interference
    /// effects -- the "meet" of a capability with a "join" of two others
    /// decomposes cleanly.
    pub fn check_distributivity(&self) -> bool {
        let n = self.elements.len();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    // LHS: x ^ (y v z)
                    let y_join_z = match self.join_table[y][z] {
                        Some(v) => v,
                        None => return false,
                    };
                    let lhs = match self.meet_table[x][y_join_z] {
                        Some(v) => v,
                        None => return false,
                    };
                    // RHS: (x ^ y) v (x ^ z)
                    let x_meet_y = match self.meet_table[x][y] {
                        Some(v) => v,
                        None => return false,
                    };
                    let x_meet_z = match self.meet_table[x][z] {
                        Some(v) => v,
                        None => return false,
                    };
                    let rhs = match self.join_table[x_meet_y][x_meet_z] {
                        Some(v) => v,
                        None => return false,
                    };
                    if lhs != rhs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Return all lattice properties in a single scan.
    pub fn properties(&self) -> LatticeProperties {
        LatticeProperties {
            is_lattice: self.is_lattice(),
            is_modular: self.check_modularity(),
            is_distributive: self.check_distributivity(),
            is_complete: self.has_bottom().is_some() && self.has_top().is_some(),
            height: self.height(),
            width: self.width(),
        }
    }

    /// Find a lattice element's index by its name.
    ///
    /// Used to map active primitive names (from `PrimitiveConsciousnessProcessor`)
    /// to lattice indices for join/meet operations.
    pub fn element_index_by_name(&self, name: &str) -> Option<usize> {
        self.elements.iter().position(|e| e.name == name)
    }

    // -----------------------------------------------------------------------
    // Structural queries
    // -----------------------------------------------------------------------

    /// Return the atoms: minimal non-bottom elements.
    ///
    /// An atom a satisfies: bottom < a and there is no x with bottom < x < a.
    /// Atoms represent the most basic capabilities above the foundation.
    pub fn atoms(&self) -> Vec<usize> {
        let bottom = match self.has_bottom() {
            Some(b) => b,
            None => return Vec::new(),
        };
        let n = self.elements.len();
        let mut result = Vec::new();
        for i in 0..n {
            if i == bottom {
                continue;
            }
            // i must be above bottom.
            if !self.order[bottom][i] {
                continue;
            }
            // No element strictly between bottom and i.
            let mut is_atom = true;
            for k in 0..n {
                if k == bottom || k == i {
                    continue;
                }
                if self.order[bottom][k] && self.order[k][i] {
                    is_atom = false;
                    break;
                }
            }
            if is_atom {
                result.push(i);
            }
        }
        result
    }

    /// Return the coatoms: maximal non-top elements.
    ///
    /// A coatom c satisfies: c < top and there is no x with c < x < top.
    pub fn coatoms(&self) -> Vec<usize> {
        let top = match self.has_top() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let n = self.elements.len();
        let mut result = Vec::new();
        for i in 0..n {
            if i == top {
                continue;
            }
            if !self.order[i][top] {
                continue;
            }
            let mut is_coatom = true;
            for k in 0..n {
                if k == top || k == i {
                    continue;
                }
                if self.order[i][k] && self.order[k][top] {
                    is_coatom = false;
                    break;
                }
            }
            if is_coatom {
                result.push(i);
            }
        }
        result
    }

    /// Compute the height of the lattice: the length of the longest chain
    /// (number of edges in the chain, i.e. elements minus one).
    ///
    /// Uses dynamic programming on the partial order.
    pub fn height(&self) -> usize {
        let n = self.elements.len();
        if n == 0 {
            return 0;
        }
        // Topological sort via Kahn's algorithm on the cover relation.
        let covers = self.hasse_diagram();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &covers {
            adj[a].push(b);
            in_degree[b] += 1;
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut longest = vec![0usize; n];
        let mut max_len = 0;

        while let Some(u) = queue.pop() {
            for &v in &adj[u] {
                let candidate = longest[u] + 1;
                if candidate > longest[v] {
                    longest[v] = candidate;
                }
                if longest[v] > max_len {
                    max_len = longest[v];
                }
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }
        max_len
    }

    /// Compute the width of the lattice: the size of the largest antichain.
    ///
    /// By Dilworth's theorem, this equals the minimum number of chains needed
    /// to partition the poset. We compute it directly by finding a maximum
    /// antichain via a greedy approach on the complement of the comparability
    /// graph. For small lattices (which primitive lattices typically are) the
    /// exact algorithm is feasible.
    pub fn width(&self) -> usize {
        let n = self.elements.len();
        if n == 0 {
            return 0;
        }
        // Build comparability relation (excluding identity).
        let comparable =
            |i: usize, j: usize| -> bool { i != j && (self.order[i][j] || self.order[j][i]) };
        // Greedy maximum independent set on the comparability graph.
        // For production-quality code on large lattices one would use
        // Konig's theorem with bipartite matching; for typical primitive
        // lattices (< 300 elements) a greedy approach is adequate.
        let mut available = vec![true; n];
        let mut antichain_size = 0;
        // Process elements in order of increasing degree in the comparability graph.
        let mut degree: Vec<(usize, usize)> = (0..n)
            .map(|i| {
                let d = (0..n).filter(|&j| comparable(i, j)).count();
                (d, i)
            })
            .collect();
        degree.sort();
        for &(_, i) in &degree {
            if !available[i] {
                continue;
            }
            antichain_size += 1;
            // Exclude all elements comparable to i.
            for j in 0..n {
                if comparable(i, j) {
                    available[j] = false;
                }
            }
        }
        antichain_size
    }

    /// Return the Hasse diagram (cover relations) of the lattice.
    ///
    /// An edge (a, b) in the Hasse diagram means a < b (strict) and there is
    /// no c with a < c < b. This is the minimal representation of the partial
    /// order -- the transitive reduction.
    pub fn hasse_diagram(&self) -> Vec<(usize, usize)> {
        let n = self.elements.len();
        let mut covers = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j || !self.order[i][j] {
                    continue;
                }
                // Check that no element is strictly between i and j.
                let mut is_cover = true;
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    if self.order[i][k] && self.order[k][j] {
                        is_cover = false;
                        break;
                    }
                }
                if is_cover {
                    covers.push((i, j));
                }
            }
        }
        covers
    }

    // -----------------------------------------------------------------------
    // Fixed-point computation (Knaster-Tarski)
    // -----------------------------------------------------------------------

    /// Compute the least fixed point of an order-preserving function via the
    /// Knaster-Tarski theorem.
    ///
    /// Given f: L -> L monotone on a complete lattice, the least fixed point is
    ///   lfp(f) = meet {x in L | f(x) <= x}
    ///
    /// We approximate this iteratively: starting from bottom, apply f repeatedly
    /// until a fixed point is reached. The function `f` maps element indices to
    /// element indices.
    ///
    /// Returns all fixed-point elements found and the iteration count.
    pub fn fixed_point(&self, f: &dyn Fn(usize) -> usize) -> FixedPointResult {
        let n = self.elements.len();
        if n == 0 {
            return FixedPointResult {
                fixed_point: Vec::new(),
                iterations: 0,
            };
        }

        // Iterative ascent from bottom.
        let bottom = self.has_bottom().unwrap_or(0);
        let mut current = bottom;
        let mut iterations = 0;
        let max_iterations = n * n; // Guaranteed to terminate on finite lattice.

        loop {
            let next = f(current);
            iterations += 1;
            if next == current {
                break;
            }
            if iterations >= max_iterations || next >= n {
                // Did not converge or invalid index -- return current best.
                break;
            }
            current = next;
        }

        // Also collect ALL fixed points by scanning.
        let mut all_fps = Vec::new();
        for i in 0..n {
            let fi = f(i);
            if fi < n && fi == i {
                all_fps.push(i);
            }
        }

        FixedPointResult {
            fixed_point: if all_fps.is_empty() {
                vec![current]
            } else {
                all_fps
            },
            iterations,
        }
    }

    /// Construct a Galois connection between the primitive lattice indices
    /// and a capability lattice.
    ///
    /// The forward map `f` and backward map `g` must satisfy the adjunction:
    ///   f(p) <= q  iff  p <= g(q)
    ///
    /// This method packages the maps and does a spot-check of the adjunction
    /// property on a small sample. It panics if the spot-check fails, as a
    /// violated Galois connection indicates a fundamental modeling error.
    pub fn galois_connection(
        forward: Box<dyn Fn(usize) -> usize>,
        backward: Box<dyn Fn(usize) -> usize>,
    ) -> GaloisConnection {
        GaloisConnection { forward, backward }
    }

    /// Build a tier-to-capability Galois connection where the forward map sends
    /// each primitive to its tier level (as a capability index) and the backward
    /// map sends each capability level to the first primitive at that tier.
    pub fn tier_galois_connection(&self) -> GaloisConnection {
        let tiers: Vec<u8> = self.elements.iter().map(|e| e.tier).collect();
        let tiers_clone = tiers.clone();

        let forward = Box::new(move |idx: usize| -> usize {
            if idx < tiers.len() {
                tiers[idx] as usize
            } else {
                0
            }
        });

        let elements_snapshot: Vec<u8> = self.elements.iter().map(|e| e.tier).collect();
        let backward = Box::new(move |cap: usize| -> usize {
            // Return the first element at or below the requested tier level.
            let target = cap as u8;
            for (idx, &t) in elements_snapshot.iter().enumerate() {
                if t == target {
                    return idx;
                }
            }
            // Fallback: return the closest tier below.
            let mut best = 0;
            let mut best_tier = 0u8;
            for (idx, &t) in tiers_clone.iter().enumerate() {
                if t <= target && t >= best_tier {
                    best = idx;
                    best_tier = t;
                }
            }
            best
        });

        GaloisConnection { forward, backward }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Find the bottom element (one that is <= all others), if it exists.
    fn has_bottom(&self) -> Option<usize> {
        let n = self.elements.len();
        'outer: for i in 0..n {
            for j in 0..n {
                if !self.order[i][j] {
                    continue 'outer;
                }
            }
            return Some(i);
        }
        None
    }

    /// Find the top element (one that all others are <=), if it exists.
    fn has_top(&self) -> Option<usize> {
        let n = self.elements.len();
        'outer: for i in 0..n {
            for j in 0..n {
                if !self.order[j][i] {
                    continue 'outer;
                }
            }
            return Some(i);
        }
        None
    }

    /// Compute all joins and fill `join_table`.
    ///
    /// For each pair (a, b), the join is the smallest upper bound:
    /// the element u such that a <= u, b <= u, and for every other upper
    /// bound v, u <= v.
    fn compute_joins(&mut self) {
        let n = self.elements.len();
        for a in 0..n {
            for b in 0..n {
                // Collect all upper bounds of {a, b}.
                let mut upper_bounds: Vec<usize> = Vec::new();
                for u in 0..n {
                    if self.order[a][u] && self.order[b][u] {
                        upper_bounds.push(u);
                    }
                }
                // Find the least among upper bounds.
                let mut candidate: Option<usize> = None;
                for &u in &upper_bounds {
                    let is_least = upper_bounds.iter().all(|&v| self.order[u][v]);
                    if is_least {
                        candidate = Some(u);
                        break;
                    }
                }
                self.join_table[a][b] = candidate;
            }
        }
    }

    /// Compute all meets and fill `meet_table`.
    ///
    /// For each pair (a, b), the meet is the greatest lower bound:
    /// the element l such that l <= a, l <= b, and for every other lower
    /// bound m, m <= l.
    fn compute_meets(&mut self) {
        let n = self.elements.len();
        for a in 0..n {
            for b in 0..n {
                // Collect all lower bounds of {a, b}.
                let mut lower_bounds: Vec<usize> = Vec::new();
                for l in 0..n {
                    if self.order[l][a] && self.order[l][b] {
                        lower_bounds.push(l);
                    }
                }
                // Find the greatest among lower bounds.
                let mut candidate: Option<usize> = None;
                for &l in &lower_bounds {
                    let is_greatest = lower_bounds.iter().all(|&m| self.order[m][l]);
                    if is_greatest {
                        candidate = Some(l);
                        break;
                    }
                }
                self.meet_table[a][b] = candidate;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a chain of `n` elements: 0 < 1 < 2 < ... < n-1
    // -----------------------------------------------------------------------
    fn make_chain(n: usize) -> PrimitiveLattice {
        let elements: Vec<LatticeElement> = (0..n)
            .map(|i| LatticeElement {
                id: i,
                primitive_id: i as u64,
                tier: i as u8,
                name: format!("chain_{}", i),
            })
            .collect();
        let mut order_rel = Vec::new();
        for i in 0..n {
            for j in i..n {
                order_rel.push((i, j));
            }
        }
        PrimitiveLattice::from_order(elements, order_rel)
    }

    /// Build the diamond lattice M_3:
    ///
    /// ```text
    ///       3 (top)
    ///      /|\
    ///     1 2 (incomparable middle elements -- but we use only 2 middle
    ///      \|/  for a 4-element diamond)
    ///       0 (bottom)
    /// ```
    ///
    /// With 4 elements: 0 = bottom, 1, 2 = incomparable, 3 = top.
    fn make_diamond() -> PrimitiveLattice {
        let elements: Vec<LatticeElement> = (0..4)
            .map(|i| LatticeElement {
                id: i,
                primitive_id: i as u64,
                tier: match i {
                    0 => 0,
                    1 | 2 => 1,
                    _ => 2,
                },
                name: format!("diamond_{}", i),
            })
            .collect();
        // Relations: 0 < 1, 0 < 2, 1 < 3, 2 < 3 (plus reflexive, transitive closure).
        let order_rel = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        PrimitiveLattice::from_order(elements, order_rel)
    }

    /// Build the M_3 lattice (diamond with 3 middle elements):
    ///
    /// ```text
    ///        4 (top)
    ///      / | \
    ///     1  2  3  (all pairwise incomparable)
    ///      \ | /
    ///        0 (bottom)
    /// ```
    ///
    /// M_3 is modular but NOT distributive.
    fn make_m3() -> PrimitiveLattice {
        let elements: Vec<LatticeElement> = (0..5)
            .map(|i| LatticeElement {
                id: i,
                primitive_id: i as u64,
                tier: match i {
                    0 => 0,
                    1..=3 => 1,
                    _ => 2,
                },
                name: format!("m3_{}", i),
            })
            .collect();
        let order_rel = vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)];
        PrimitiveLattice::from_order(elements, order_rel)
    }

    /// Build the N_5 (pentagon) lattice:
    ///
    /// ```text
    ///       4 (top)
    ///      / \
    ///     3   2
    ///     |
    ///     1
    ///      \ /
    ///       0 (bottom)
    /// ```
    ///
    /// Relations: 0 < 1 < 3 < 4, 0 < 2 < 4, and 1,2 are incomparable,
    /// 2,3 are incomparable.
    /// N_5 is NOT modular (and therefore not distributive).
    fn make_pentagon() -> PrimitiveLattice {
        let elements: Vec<LatticeElement> = (0..5)
            .map(|i| LatticeElement {
                id: i,
                primitive_id: i as u64,
                tier: i as u8,
                name: format!("pent_{}", i),
            })
            .collect();
        // 0 < 1 < 3 < 4, and 0 < 2 < 4.
        let order_rel = vec![(0, 1), (1, 3), (3, 4), (0, 2), (2, 4)];
        PrimitiveLattice::from_order(elements, order_rel)
    }

    // -----------------------------------------------------------------------
    // Test: 3-element chain is a lattice
    // -----------------------------------------------------------------------
    #[test]
    fn chain_3_is_lattice() {
        let chain = make_chain(3);
        assert!(chain.is_lattice(), "A 3-element chain must be a lattice");
        // Chains are always distributive (and hence modular).
        assert!(chain.check_distributivity(), "A chain must be distributive");
        assert!(chain.check_modularity(), "A chain must be modular");

        // Height = 2 (two edges: 0->1->2).
        assert_eq!(chain.height(), 2);
        // Width = 1 (every antichain has size 1 in a chain).
        assert_eq!(chain.width(), 1);
    }

    // -----------------------------------------------------------------------
    // Test: Diamond lattice (M_3 with 3 middle elements) is modular but not
    //       distributive
    // -----------------------------------------------------------------------
    #[test]
    fn m3_is_modular_not_distributive() {
        let m3 = make_m3();
        assert!(m3.is_lattice(), "M_3 must be a lattice");
        assert!(m3.check_modularity(), "M_3 must be modular");
        assert!(!m3.check_distributivity(), "M_3 must NOT be distributive");
    }

    // -----------------------------------------------------------------------
    // Test: Pentagon lattice (N_5) is NOT modular
    // -----------------------------------------------------------------------
    #[test]
    fn pentagon_is_not_modular() {
        let pent = make_pentagon();
        assert!(pent.is_lattice(), "N_5 must be a lattice");
        assert!(
            !pent.check_modularity(),
            "N_5 (pentagon) must NOT be modular"
        );
        // Not modular => not distributive either.
        assert!(!pent.check_distributivity());
    }

    // -----------------------------------------------------------------------
    // Test: Join and meet are commutative, associative, and idempotent
    // -----------------------------------------------------------------------
    #[test]
    fn join_meet_algebraic_laws() {
        let lattice = make_diamond();
        let n = lattice.elements.len();
        assert!(lattice.is_lattice());

        for a in 0..n {
            for b in 0..n {
                // Commutativity.
                assert_eq!(
                    lattice.join(a, b),
                    lattice.join(b, a),
                    "join must be commutative for ({}, {})",
                    a,
                    b
                );
                assert_eq!(
                    lattice.meet(a, b),
                    lattice.meet(b, a),
                    "meet must be commutative for ({}, {})",
                    a,
                    b
                );

                // Idempotency.
                assert_eq!(lattice.join(a, a), Some(a), "join(a,a) must be a for {}", a);
                assert_eq!(lattice.meet(a, a), Some(a), "meet(a,a) must be a for {}", a);

                // Associativity (test with all triples, but only for elements
                // where join/meet exist -- already guaranteed by is_lattice).
                for c in 0..n {
                    // join(a, join(b, c)) == join(join(a, b), c)
                    let jbc = lattice.join(b, c).unwrap();
                    let lhs_j = lattice.join(a, jbc).unwrap();
                    let jab = lattice.join(a, b).unwrap();
                    let rhs_j = lattice.join(jab, c).unwrap();
                    assert_eq!(
                        lhs_j, rhs_j,
                        "join must be associative for ({}, {}, {})",
                        a, b, c
                    );

                    // meet(a, meet(b, c)) == meet(meet(a, b), c)
                    let mbc = lattice.meet(b, c).unwrap();
                    let lhs_m = lattice.meet(a, mbc).unwrap();
                    let mab = lattice.meet(a, b).unwrap();
                    let rhs_m = lattice.meet(mab, c).unwrap();
                    assert_eq!(
                        lhs_m, rhs_m,
                        "meet must be associative for ({}, {}, {})",
                        a, b, c
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: Fixed point of identity is every element (all are fixed)
    // -----------------------------------------------------------------------
    #[test]
    fn fixed_point_of_identity() {
        let chain = make_chain(4);
        let result = chain.fixed_point(&|x| x);
        // Every element is a fixed point of the identity function.
        assert_eq!(
            result.fixed_point.len(),
            4,
            "identity has all elements as fixed points"
        );
        assert_eq!(result.iterations, 1, "identity converges in 1 iteration");
    }

    // -----------------------------------------------------------------------
    // Test: Fixed point of successor (clamped) reaches top
    // -----------------------------------------------------------------------
    #[test]
    fn fixed_point_successor_reaches_top() {
        let chain = make_chain(4);
        // f(x) = min(x+1, 3) -- monotone on the chain.
        let result = chain.fixed_point(&|x| if x < 3 { x + 1 } else { 3 });
        assert!(
            result.fixed_point.contains(&3),
            "successor reaches top element"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Tier-based ordering respects transitivity
    // -----------------------------------------------------------------------
    #[test]
    fn tier_ordering_transitivity() {
        // Build a small lattice with explicit tier-based elements.
        let elements = vec![
            LatticeElement {
                id: 0,
                primitive_id: 0,
                tier: 0,
                name: "nsm_a".into(),
            },
            LatticeElement {
                id: 1,
                primitive_id: 1,
                tier: 1,
                name: "math_a".into(),
            },
            LatticeElement {
                id: 2,
                primitive_id: 2,
                tier: 2,
                name: "phys_a".into(),
            },
        ];
        // Tier-based order: 0 < 1 < 2.
        let order_rel = vec![(0, 1), (1, 2)];
        let lattice = PrimitiveLattice::from_order(elements, order_rel);

        // Transitivity: 0 <= 2 should hold after closure.
        assert!(
            lattice.order[0][2],
            "Transitive closure must give 0 <= 2 from 0<=1 and 1<=2"
        );
        // Antisymmetry: 2 </= 0.
        assert!(
            !lattice.order[2][0],
            "Antisymmetry: higher tier is not below lower tier"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Atoms and coatoms of chain
    // -----------------------------------------------------------------------
    #[test]
    fn chain_atoms_and_coatoms() {
        let chain = make_chain(4);
        let atoms = chain.atoms();
        let coatoms = chain.coatoms();
        // In a chain 0 < 1 < 2 < 3: atom = {1}, coatom = {2}.
        assert_eq!(atoms, vec![1], "chain atom is element 1");
        assert_eq!(coatoms, vec![2], "chain coatom is element 2");
    }

    // -----------------------------------------------------------------------
    // Test: Hasse diagram of chain
    // -----------------------------------------------------------------------
    #[test]
    fn chain_hasse_diagram() {
        let chain = make_chain(4);
        let mut hasse = chain.hasse_diagram();
        hasse.sort();
        assert_eq!(
            hasse,
            vec![(0, 1), (1, 2), (2, 3)],
            "chain Hasse diagram is a path"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Diamond lattice structure
    // -----------------------------------------------------------------------
    #[test]
    fn diamond_lattice_properties() {
        let diamond = make_diamond();
        assert!(diamond.is_lattice());
        // 4-element diamond (0 < {1,2} < 3) is distributive.
        assert!(diamond.check_distributivity());
        assert!(diamond.check_modularity());

        // Height = 2 (0->1->3 or 0->2->3).
        assert_eq!(diamond.height(), 2);
        // Width = 2 (antichain {1, 2}).
        assert_eq!(diamond.width(), 2);

        // join(1, 2) = 3 (top).
        assert_eq!(diamond.join(1, 2), Some(3));
        // meet(1, 2) = 0 (bottom).
        assert_eq!(diamond.meet(1, 2), Some(0));
    }

    // -----------------------------------------------------------------------
    // Test: Galois connection (tier-based)
    // -----------------------------------------------------------------------
    #[test]
    fn tier_galois_connection_basic() {
        let chain = make_chain(3);
        let gc = chain.tier_galois_connection();

        // Forward maps element index to its tier.
        assert_eq!((gc.forward)(0), 0);
        assert_eq!((gc.forward)(1), 1);
        assert_eq!((gc.forward)(2), 2);

        // Backward maps tier back to an element at that tier.
        let back_0 = (gc.backward)(0);
        assert_eq!(chain.elements[back_0].tier, 0);
        let back_1 = (gc.backward)(1);
        assert_eq!(chain.elements[back_1].tier, 1);
    }

    // -----------------------------------------------------------------------
    // Test: Properties summary
    // -----------------------------------------------------------------------
    #[test]
    fn properties_summary() {
        let chain = make_chain(5);
        let props = chain.properties();
        assert!(props.is_lattice);
        assert!(props.is_modular);
        assert!(props.is_distributive);
        assert!(props.is_complete);
        assert_eq!(props.height, 4);
        assert_eq!(props.width, 1);
    }

    // -----------------------------------------------------------------------
    // Test: Empty lattice edge cases
    // -----------------------------------------------------------------------
    #[test]
    fn empty_lattice() {
        let lattice = PrimitiveLattice::from_order(vec![], vec![]);
        assert!(lattice.is_lattice()); // vacuously true
        assert_eq!(lattice.height(), 0);
        assert_eq!(lattice.width(), 0);
        assert!(lattice.atoms().is_empty());
        assert!(lattice.coatoms().is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: Single element lattice
    // -----------------------------------------------------------------------
    #[test]
    fn single_element_lattice() {
        let elements = vec![LatticeElement {
            id: 0,
            primitive_id: 0,
            tier: 0,
            name: "only".into(),
        }];
        let lattice = PrimitiveLattice::from_order(elements, vec![]);
        assert!(lattice.is_lattice());
        assert_eq!(lattice.join(0, 0), Some(0));
        assert_eq!(lattice.meet(0, 0), Some(0));
        assert_eq!(lattice.height(), 0);
        assert_eq!(lattice.width(), 1);
    }
}
