// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Program Manifold — fiber bundle of implementations over specifications.
//!
//! The manifold organizes code implementations as a **fiber bundle**:
//! - **Base space**: specification encodings (algorithm pattern + type signature
//!   + topological fingerprint), each represented as a [`BinaryHV`].
//! - **Fiber**: the set of concrete implementations that share the same base
//!   point (i.e., satisfy the same specification).
//!
//! This structure enables:
//! - Finding the closest known specification to a new query.
//! - Retrieving all known implementations of a specification.
//! - Interpolating between implementations (weighted HDC bundle).
//! - Computing tangent vectors (semantic diff between implementations).

use symthaea_core::hdc::binary_hv::BinaryHV;

use crate::topology::TopologicalFingerprint;

/// Similarity threshold for assigning a new point to an existing fiber.
///
/// BinaryHV similarity is in [0, 1] where 0.5 is orthogonal (random).
/// A threshold of 0.65 means "clearly more similar than chance".
/// Similarity threshold for assigning a new point to an existing fiber.
/// Random BinaryHV similarity is ~0.5; 0.65 means "clearly above chance".
/// With size-aware encoding, functions of different structural complexity
/// will fall below this threshold and create separate fibers.
const FIBER_ASSIGNMENT_THRESHOLD: f32 = 0.65;

// ---------------------------------------------------------------------------
// Fiber point — a single concrete implementation
// ---------------------------------------------------------------------------

/// A point in the fiber: a concrete implementation of a specification.
#[derive(Debug, Clone)]
pub struct FiberPoint {
    /// Human-readable name (e.g., "quicksort_v3").
    pub name: String,
    /// HDC encoding of the implementation (from program algebra).
    pub encoding: BinaryHV,
    /// Topological fingerprint of the implementation's structure.
    pub fingerprint: TopologicalFingerprint,
    /// Quality score in [0.0, 1.0] (from compilation/test success rate).
    pub quality: f32,
    /// Original source code snippet (for retrieval-based slot filling).
    ///
    /// When filling skeleton slots, we retrieve real code from the nearest
    /// fiber rather than decoding HDC vectors to tokens. This sidesteps the
    /// lossy encoding problem entirely — the HDC similarity finds the right
    /// fiber, and the source provides the actual expressions.
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// Fiber — collection of implementations sharing a specification
// ---------------------------------------------------------------------------

/// A fiber: collection of implementations over a common base point.
///
/// The base point is the specification encoding. The centroid is the
/// majority-vote bundle of all implementation encodings — it represents
/// the "average" or "typical" implementation of the spec.
#[derive(Debug, Clone)]
pub struct Fiber {
    /// The specification / pattern encoding (base point).
    pub base_encoding: BinaryHV,
    /// Concrete implementations.
    pub points: Vec<FiberPoint>,
    /// Majority-vote centroid of all point encodings.
    pub centroid: BinaryHV,
}

impl Fiber {
    /// Create a new fiber with the given base (specification) encoding.
    fn new(base: BinaryHV) -> Self {
        Self {
            base_encoding: base,
            points: Vec::new(),
            centroid: BinaryHV::zero(),
        }
    }

    /// Add an implementation point and recompute the centroid.
    fn add_point(&mut self, point: FiberPoint) {
        self.points.push(point);
        self.recompute_centroid();
    }

    /// Recompute centroid as the majority-vote bundle of all point encodings.
    fn recompute_centroid(&mut self) {
        if self.points.is_empty() {
            self.centroid = BinaryHV::zero();
            return;
        }
        if self.points.len() == 1 {
            self.centroid = self.points[0].encoding;
            return;
        }

        let encodings: Vec<BinaryHV> = self.points.iter().map(|p| p.encoding).collect();
        self.centroid = BinaryHV::bundle(&encodings);
    }

    /// Number of implementations in this fiber.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether this fiber has no implementations.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Best implementation by quality score.
    pub fn best(&self) -> Option<&FiberPoint> {
        self.points.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Average quality across all implementations.
    pub fn average_quality(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.points.iter().map(|p| p.quality).sum();
        sum / self.points.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Program Manifold
// ---------------------------------------------------------------------------

/// The program manifold: maps specifications to fibers of implementations.
///
/// This is a discrete approximation to a fiber bundle where:
/// - The base space is the set of specification encodings.
/// - Each fiber is the set of concrete implementations of that spec.
/// - Navigation is via HDC similarity in the base space.
#[derive(Debug, Clone, Default)]
pub struct ProgramManifold {
    fibers: Vec<Fiber>,
}

impl ProgramManifold {
    /// Create an empty program manifold.
    pub fn new() -> Self {
        Self { fibers: Vec::new() }
    }

    /// Add an implementation to the manifold.
    ///
    /// Finds the nearest existing fiber whose centroid similarity exceeds
    /// [`FIBER_ASSIGNMENT_THRESHOLD`]. If no fiber is close enough, creates
    /// a new fiber with the encoding as its base point.
    pub fn insert(
        &mut self,
        name: &str,
        encoding: BinaryHV,
        fingerprint: TopologicalFingerprint,
        quality: f32,
    ) {
        self.insert_with_source(name, encoding, fingerprint, quality, None);
    }

    /// Insert a function with its source code for retrieval-based filling.
    pub fn insert_with_source(
        &mut self,
        name: &str,
        encoding: BinaryHV,
        fingerprint: TopologicalFingerprint,
        quality: f32,
        source: Option<String>,
    ) {
        let point = FiberPoint {
            name: name.to_string(),
            encoding,
            fingerprint,
            quality: quality.clamp(0.0, 1.0),
            source,
        };

        // Find the nearest fiber
        let best = self.nearest_fiber_index(&encoding);

        match best {
            Some(idx) => {
                self.fibers[idx].add_point(point);
            }
            None => {
                // No fiber close enough — create a new one
                let mut fiber = Fiber::new(encoding);
                fiber.add_point(point);
                self.fibers.push(fiber);
            }
        }
    }

    /// Find the nearest fiber to a specification encoding.
    ///
    /// Returns `None` if the manifold is empty.
    pub fn nearest_fiber(&self, spec: &BinaryHV) -> Option<&Fiber> {
        if self.fibers.is_empty() {
            return None;
        }

        self.fibers.iter().max_by(|a, b| {
            let sa = spec.similarity(&a.centroid);
            let sb = spec.similarity(&b.centroid);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the centroid of the nearest fiber (the "abstract" implementation).
    pub fn nearest_centroid(&self, spec: &BinaryHV) -> Option<&BinaryHV> {
        self.nearest_fiber(spec).map(|f| &f.centroid)
    }

    /// Interpolate between two implementation encodings.
    ///
    /// Uses weighted bundle: `alpha` weight on `a`, `(1 - alpha)` on `b`.
    /// At alpha=0.0 the result matches `b`; at alpha=1.0 it matches `a`.
    pub fn interpolate(a: &BinaryHV, b: &BinaryHV, alpha: f32) -> BinaryHV {
        let alpha = alpha.clamp(0.0, 1.0);
        BinaryHV::weighted_bundle(&[*a, *b], &[alpha, 1.0 - alpha])
    }

    /// Compute the tangent vector between two implementations.
    ///
    /// In HDC, the XOR (bind) of two vectors gives the "difference" —
    /// bits that differ represent the semantic delta between the two
    /// programs. The tangent is the self-inverse binding: `from XOR to`.
    pub fn tangent(from: &BinaryHV, to: &BinaryHV) -> BinaryHV {
        from.bind(to)
    }

    /// Number of fibers (specification clusters) in the manifold.
    pub fn fiber_count(&self) -> usize {
        self.fibers.len()
    }

    /// Total implementations across all fibers.
    pub fn total_points(&self) -> usize {
        self.fibers.iter().map(|f| f.points.len()).sum()
    }

    /// Iterate over all fibers.
    pub fn fibers(&self) -> &[Fiber] {
        &self.fibers
    }

    // -- internal helpers --

    /// Find the index of the nearest fiber above the assignment threshold.
    fn nearest_fiber_index(&self, encoding: &BinaryHV) -> Option<usize> {
        if self.fibers.is_empty() {
            return None;
        }

        let mut best_idx = None;
        let mut best_sim = FIBER_ASSIGNMENT_THRESHOLD;

        for (i, fiber) in self.fibers.iter().enumerate() {
            let sim = encoding.similarity(&fiber.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        best_idx
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_hodge::SimplicialComplex;

    /// Helper: create a minimal TopologicalFingerprint for testing.
    fn dummy_fingerprint() -> TopologicalFingerprint {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(vec![0]);
        TopologicalFingerprint::from_complex(&complex)
    }

    /// Insert a point and query — should find it in the nearest fiber.
    #[test]
    fn test_manifold_insert_and_lookup() {
        let mut manifold = ProgramManifold::new();
        let enc = BinaryHV::random(100);
        manifold.insert("impl_a", enc, dummy_fingerprint(), 0.9);

        assert_eq!(manifold.fiber_count(), 1);
        assert_eq!(manifold.total_points(), 1);

        // Query with the same encoding — should find the fiber
        let nearest = manifold.nearest_fiber(&enc).unwrap();
        assert_eq!(nearest.points.len(), 1);
        assert_eq!(nearest.points[0].name, "impl_a");
    }

    /// Two dissimilar implementations should land in separate fibers.
    #[test]
    fn test_manifold_separate_fibers() {
        let mut manifold = ProgramManifold::new();

        // Use very different seeds to get dissimilar encodings
        let enc_a = BinaryHV::random(1);
        let enc_b = BinaryHV::random(2);

        // Random BinaryHVs have similarity ~0.5, well below the 0.65 threshold
        let sim = enc_a.similarity(&enc_b);
        assert!(
            sim < FIBER_ASSIGNMENT_THRESHOLD,
            "random HVs should be below threshold, got {}",
            sim
        );

        manifold.insert("impl_a", enc_a, dummy_fingerprint(), 0.8);
        manifold.insert("impl_b", enc_b, dummy_fingerprint(), 0.7);

        assert_eq!(
            manifold.fiber_count(),
            2,
            "dissimilar points → separate fibers"
        );
        assert_eq!(manifold.total_points(), 2);
    }

    /// Similar implementations should cluster into the same fiber.
    #[test]
    fn test_manifold_same_fiber() {
        let mut manifold = ProgramManifold::new();

        // Same encoding → should cluster
        let enc = BinaryHV::random(42);
        manifold.insert("impl_a", enc, dummy_fingerprint(), 0.8);
        manifold.insert("impl_b", enc, dummy_fingerprint(), 0.9);

        assert_eq!(
            manifold.fiber_count(),
            1,
            "identical encodings → same fiber"
        );
        assert_eq!(manifold.total_points(), 2);
    }

    /// Tangent vector: XOR of a and b, applied again, recovers the original.
    #[test]
    fn test_tangent_vector() {
        let a = BinaryHV::random(10);
        let b = BinaryHV::random(20);

        let tangent = ProgramManifold::tangent(&a, &b);

        // XOR is self-inverse: a XOR (a XOR b) = b
        let recovered = a.bind(&tangent);
        assert_eq!(
            recovered, b,
            "tangent applied to source should yield target"
        );
    }

    /// Interpolation at extremes should match the endpoints.
    #[test]
    fn test_interpolation_extremes() {
        let a = BinaryHV::random(100);
        let b = BinaryHV::random(200);

        // alpha=1.0 → should match a
        let result_a = ProgramManifold::interpolate(&a, &b, 1.0);
        assert!(
            result_a.similarity(&a) > 0.99,
            "alpha=1.0 should yield a, sim={}",
            result_a.similarity(&a)
        );

        // alpha=0.0 → should match b
        let result_b = ProgramManifold::interpolate(&a, &b, 0.0);
        assert!(
            result_b.similarity(&b) > 0.99,
            "alpha=0.0 should yield b, sim={}",
            result_b.similarity(&b)
        );
    }

    /// Empty manifold queries return None.
    #[test]
    fn test_empty_manifold() {
        let manifold = ProgramManifold::new();
        let query = BinaryHV::random(1);

        assert!(manifold.nearest_fiber(&query).is_none());
        assert!(manifold.nearest_centroid(&query).is_none());
        assert_eq!(manifold.fiber_count(), 0);
        assert_eq!(manifold.total_points(), 0);
    }
}
