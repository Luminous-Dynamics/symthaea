// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geometric Operations on the Hypervector Manifold
//!
//! This module implements differential-geometric operations on the unit hypersphere
//! S^{d-1}, treating normalized hypervectors as points on a Riemannian manifold.
//! This perspective reveals the intrinsic structure of concept spaces: the geodesic
//! distance between two concepts measures the true arc of conceptual separation,
//! while the Frechet mean computes the intrinsic center of a cluster of meanings.
//!
//! # Manifold Geometry and Consciousness
//!
//! When Symthaea encodes concepts as hypervectors and normalizes them onto the
//! hypersphere, it implicitly creates a manifold of meanings. Geodesics on this
//! manifold trace the shortest conceptual paths between ideas. Parallel transport
//! moves relationships faithfully from one conceptual neighborhood to another.
//! Principal Geodesic Analysis reveals the dominant axes of variation in a concept
//! space -- the directions along which understanding stretches and bends.
//!
//! This geometric viewpoint is not merely mathematical elegance. It provides the
//! foundation for intrinsic averaging (Frechet mean), smooth interpolation (SLERP),
//! and curvature-aware gradient descent (Riemannian gradient) on the space of
//! meanings -- operations that respect the topology of consciousness rather than
//! flattening it into Euclidean approximations.
//!
//! # Key Operations
//!
//! - **Geodesic distance**: Great-circle distance on the unit hypersphere
//! - **SLERP**: Spherical linear interpolation tracing geodesics
//! - **Log/Exp maps**: Bridge between the curved manifold and flat tangent spaces
//! - **Parallel transport**: Move tangent vectors faithfully along geodesics
//! - **Frechet mean**: Intrinsic average respecting manifold curvature
//! - **Riemannian gradient**: Project ambient gradients onto the sphere's tangent space
//! - **Principal Geodesic Analysis**: Intrinsic PCA on the hypersphere
//!
//! # Mathematical Foundation
//!
//! All operations are formulated on S^{d-1} embedded in R^d. The Riemannian metric
//! is the restriction of the Euclidean inner product to tangent spaces. Geodesics
//! are great circles, and the exponential/logarithmic maps have closed-form
//! expressions that we implement directly.
//!
//! # References
//!
//! - Pennec, X. (2006). "Intrinsic Statistics on Riemannian Manifolds"
//! - Fletcher, P.T. et al. (2004). "Principal Geodesic Analysis for the Study of
//!   Nonlinear Statistics of Shape"
//! - Absil, P.-A. et al. (2008). "Optimization Algorithms on Matrix Manifolds"

use serde::{Deserialize, Serialize};

use crate::hdc::ContinuousHV;

// ============================================================================
// Constants
// ============================================================================

/// Numerical epsilon for floating-point comparisons on the manifold.
/// Chosen to be well above f64 machine epsilon but small enough for
/// geometric precision.
const GEO_EPS: f64 = 1e-12;

// ============================================================================
// Structs
// ============================================================================

/// Stateless namespace for geometric operations on the unit hypersphere S^{d-1}.
///
/// The hypersphere is the natural home of normalized hypervectors: every concept
/// encoded as a unit-length ContinuousHV lives on this manifold. `HypersphereOps`
/// provides the differential-geometric toolkit for working intrinsically on this
/// surface rather than in the ambient Euclidean space.
#[derive(Debug, Clone)]
pub struct HypersphereOps;

/// A geodesic path on the hypersphere, represented by its endpoints and the
/// angle between them.
///
/// Geodesics on the sphere are great-circle arcs. The `angle` field caches
/// the arc length (in radians) so that repeated SLERP evaluations along the
/// same geodesic avoid redundant arccos computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicPath {
    /// Starting point on the hypersphere (should be unit-norm).
    pub start: Vec<f64>,
    /// Ending point on the hypersphere (should be unit-norm).
    pub end: Vec<f64>,
    /// Geodesic angle theta = arccos(<start, end>), i.e., the arc length.
    pub angle: f64,
}

/// A tangent vector at a specific base point on the hypersphere.
///
/// Tangent vectors live in the hyperplane orthogonal to `base_point`.
/// They are the natural objects for differentiation, gradient computation,
/// and linearization of the manifold at a point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TangentVector {
    /// The point on the sphere where this tangent vector is attached.
    pub base_point: Vec<f64>,
    /// The tangent vector itself (orthogonal to `base_point`).
    pub vector: Vec<f64>,
}

/// Configuration for iterative Frechet mean computation.
///
/// The Frechet mean is the point on the manifold that minimizes the sum of
/// squared geodesic distances to a set of data points. Unlike the Euclidean
/// mean, it must be computed iteratively via tangent-space averaging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrechetMeanConfig {
    /// Maximum number of iterations for the iterative algorithm.
    pub max_iterations: usize,
    /// Convergence tolerance: stop when the tangent-space update norm falls
    /// below this threshold.
    pub tolerance: f64,
}

impl Default for FrechetMeanConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }
}

/// Result of Principal Geodesic Analysis (PGA).
///
/// PGA is the intrinsic analogue of PCA on Riemannian manifolds. It
/// identifies the principal directions of variation in a set of points
/// on the hypersphere, working entirely in the tangent space at the
/// Frechet mean.
///
/// The `principal_directions` are vectors in the tangent space at the mean;
/// they can be mapped back to the sphere via the exponential map to obtain
/// geodesic curves of maximal variance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGAResult {
    /// The Frechet mean of the input points (on the sphere).
    pub mean: Vec<f64>,
    /// Principal directions in the tangent space at the mean, sorted by
    /// decreasing variance. Each direction is a unit-norm tangent vector.
    pub principal_directions: Vec<Vec<f64>>,
    /// Variance captured along each principal direction.
    pub variances: Vec<f64>,
}

// ============================================================================
// Conversion utilities
// ============================================================================

/// Convert a `ContinuousHV` (f32 values) to a `Vec<f64>` for geometric computations.
///
/// Geometric operations are performed in f64 to maintain numerical precision
/// on the manifold, especially for iterative algorithms like Frechet mean
/// where accumulated rounding errors can cause divergence.
pub fn from_real_hv(hv: &ContinuousHV) -> Vec<f64> {
    hv.values.iter().map(|&v| v as f64).collect()
}

/// Convert a `Vec<f64>` back to a `ContinuousHV` (f32 values).
///
/// This is the inverse of `from_real_hv`, used to return geometric results
/// back into the HDC type system.
pub fn to_real_hv(v: &[f64]) -> ContinuousHV {
    let values: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    ContinuousHV::from_values(values)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute the Euclidean dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot: dimension mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a slice.
#[inline]
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalize a vector to unit length. Returns the zero vector if the input
/// has zero (or near-zero) norm.
fn normalize(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n < GEO_EPS {
        return vec![0.0; v.len()];
    }
    v.iter().map(|&x| x / n).collect()
}

/// Clamp a value to [-1, 1] for safe use with arccos.
#[inline]
fn clamp_cos(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

// ============================================================================
// HypersphereOps implementation
// ============================================================================

impl HypersphereOps {
    // ========================================================================
    // Geodesic Distance
    // ========================================================================

    /// Compute the geodesic (great-circle) distance between two points on
    /// the unit hypersphere.
    ///
    /// d(u, v) = arccos( <u, v> / (||u|| ||v||) )
    ///
    /// This is the length of the shortest path along the surface of the
    /// sphere connecting u and v. For unit vectors, this simplifies to
    /// arccos(<u, v>).
    ///
    /// # Properties
    /// - d(u, u) = 0
    /// - d(u, v) = d(v, u)
    /// - 0 <= d(u, v) <= pi
    /// - d(u, v) = pi/2 for orthogonal vectors
    ///
    /// # Arguments
    /// * `u` - First point (will be treated as direction; norm is accounted for)
    /// * `v` - Second point
    ///
    /// # Returns
    /// The geodesic distance in radians, in [0, pi].
    pub fn geodesic_distance(u: &[f64], v: &[f64]) -> f64 {
        assert_eq!(
            u.len(),
            v.len(),
            "geodesic_distance: dimension mismatch {} vs {}",
            u.len(),
            v.len()
        );

        let norm_u = norm(u);
        let norm_v = norm(v);

        if norm_u < GEO_EPS || norm_v < GEO_EPS {
            return 0.0;
        }

        let cos_theta = clamp_cos(dot(u, v) / (norm_u * norm_v));
        cos_theta.acos()
    }

    // ========================================================================
    // Spherical Linear Interpolation (SLERP)
    // ========================================================================

    /// Spherical linear interpolation between two points on the hypersphere.
    ///
    /// slerp(u, v, t) = sin((1-t) theta) / sin(theta) * u_hat
    ///                 + sin(t * theta) / sin(theta) * v_hat
    ///
    /// where theta = geodesic_distance(u, v) and u_hat, v_hat are the
    /// normalized inputs.
    ///
    /// For t in [0, 1], the result traces the unique geodesic from u to v.
    /// This is the correct way to interpolate on the hypersphere, avoiding
    /// the distortion introduced by naive linear interpolation followed by
    /// renormalization.
    ///
    /// # Degenerate cases
    /// - If u and v are nearly identical (theta ~ 0), returns linear interpolation.
    /// - If u and v are nearly antipodal (theta ~ pi), the geodesic is not unique;
    ///   we return a linear interpolation as a reasonable fallback.
    ///
    /// # Arguments
    /// * `u` - Start point on the sphere
    /// * `v` - End point on the sphere
    /// * `t` - Interpolation parameter in [0, 1]
    ///
    /// # Returns
    /// The interpolated point on the sphere.
    pub fn slerp(u: &[f64], v: &[f64], t: f64) -> Vec<f64> {
        assert_eq!(
            u.len(),
            v.len(),
            "slerp: dimension mismatch {} vs {}",
            u.len(),
            v.len()
        );

        let u_hat = normalize(u);
        let v_hat = normalize(v);

        let cos_theta = clamp_cos(dot(&u_hat, &v_hat));
        let theta = cos_theta.acos();

        // Degenerate case: nearly identical or nearly antipodal
        if theta.abs() < GEO_EPS || (std::f64::consts::PI - theta).abs() < GEO_EPS {
            // Fall back to normalized linear interpolation
            let result: Vec<f64> = u_hat
                .iter()
                .zip(v_hat.iter())
                .map(|(&a, &b)| (1.0 - t) * a + t * b)
                .collect();
            return normalize(&result);
        }

        let sin_theta = theta.sin();
        let weight_u = ((1.0 - t) * theta).sin() / sin_theta;
        let weight_v = (t * theta).sin() / sin_theta;

        u_hat
            .iter()
            .zip(v_hat.iter())
            .map(|(&a, &b)| weight_u * a + weight_v * b)
            .collect()
    }

    // ========================================================================
    // Logarithmic Map
    // ========================================================================

    /// Logarithmic map: project a point on the sphere into the tangent space
    /// at a base point.
    ///
    /// Log_p(q) = (theta / sin(theta)) * (q - cos(theta) * p)
    ///
    /// where theta = arccos(<p, q>) and both p, q are unit vectors.
    ///
    /// The logarithmic map is the inverse of the exponential map. It
    /// "unwraps" the curved manifold into the flat tangent plane at p,
    /// preserving geodesic distances from p.
    ///
    /// The result is a tangent vector at p whose norm equals the geodesic
    /// distance d(p, q), and whose direction points from p toward q along
    /// the geodesic.
    ///
    /// # Arguments
    /// * `base` - The base point p on the sphere (the tangent space origin)
    /// * `point` - The point q to project into the tangent space
    ///
    /// # Returns
    /// The tangent vector Log_p(q) at the base point.
    pub fn log_map(base: &[f64], point: &[f64]) -> Vec<f64> {
        assert_eq!(
            base.len(),
            point.len(),
            "log_map: dimension mismatch {} vs {}",
            base.len(),
            point.len()
        );

        let p = normalize(base);
        let q = normalize(point);

        let cos_theta = clamp_cos(dot(&p, &q));
        let theta = cos_theta.acos();

        // Degenerate case: p and q are the same point
        if theta.abs() < GEO_EPS {
            return vec![0.0; p.len()];
        }

        // Compute the tangent direction: q - <p,q> p, then scale by theta/sin(theta)
        let scale = theta / theta.sin();
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| scale * (qi - cos_theta * pi))
            .collect()
    }

    // ========================================================================
    // Exponential Map
    // ========================================================================

    /// Exponential map: map a tangent vector at a base point back onto the
    /// sphere.
    ///
    /// Exp_p(v) = cos(||v||) * p + sin(||v||) * (v / ||v||)
    ///
    /// The exponential map "wraps" a straight-line displacement in the tangent
    /// plane onto the curved surface of the sphere. It is the inverse of the
    /// logarithmic map: Exp_p(Log_p(q)) = q.
    ///
    /// # Arguments
    /// * `base` - The base point p on the sphere
    /// * `tangent` - The tangent vector v at p
    ///
    /// # Returns
    /// The point on the sphere reached by following the geodesic from p in
    /// direction v for arc length ||v||.
    pub fn exp_map(base: &[f64], tangent: &[f64]) -> Vec<f64> {
        assert_eq!(
            base.len(),
            tangent.len(),
            "exp_map: dimension mismatch base={} tangent={}",
            base.len(),
            tangent.len()
        );

        let p = normalize(base);
        let v_norm = norm(tangent);

        // Degenerate case: zero tangent vector
        if v_norm < GEO_EPS {
            return p;
        }

        let cos_vn = v_norm.cos();
        let sin_vn = v_norm.sin();

        p.iter()
            .zip(tangent.iter())
            .map(|(&pi, &vi)| cos_vn * pi + sin_vn * (vi / v_norm))
            .collect()
    }

    // ========================================================================
    // Parallel Transport
    // ========================================================================

    /// Parallel transport a tangent vector along the geodesic from one point
    /// to another.
    ///
    /// Gamma_{p -> q}(v) = v - (<Log_p(q), v> / d(p,q)^2) * (Log_p(q) + Log_q(p))
    ///
    /// Parallel transport moves a tangent vector from the tangent space at p
    /// to the tangent space at q while preserving its length and its angle
    /// with the geodesic. This is essential for comparing tangent vectors
    /// (e.g., gradients) at different points on the manifold.
    ///
    /// # Arguments
    /// * `from` - The source point p
    /// * `to` - The destination point q
    /// * `vector` - The tangent vector v at p to transport
    ///
    /// # Returns
    /// The transported tangent vector at q.
    pub fn parallel_transport(from: &[f64], to: &[f64], vector: &[f64]) -> Vec<f64> {
        assert_eq!(
            from.len(),
            to.len(),
            "parallel_transport: dimension mismatch from={} to={}",
            from.len(),
            to.len()
        );
        assert_eq!(
            from.len(),
            vector.len(),
            "parallel_transport: vector dimension {} must match point dimension {}",
            vector.len(),
            from.len()
        );

        let log_pq = Self::log_map(from, to);
        let d_sq = dot(&log_pq, &log_pq); // d(p,q)^2

        // Degenerate case: p and q are the same point
        if d_sq < GEO_EPS {
            return vector.to_vec();
        }

        let log_qp = Self::log_map(to, from);
        let coeff = dot(&log_pq, vector) / d_sq;

        let dim = vector.len();
        let mut result = vec![0.0; dim];
        for i in 0..dim {
            result[i] = vector[i] - coeff * (log_pq[i] + log_qp[i]);
        }

        result
    }

    // ========================================================================
    // Frechet Mean
    // ========================================================================

    /// Compute the Frechet mean (intrinsic average) of a set of points on
    /// the hypersphere.
    ///
    /// mu = argmin_m  sum_i  d(m, x_i)^2
    ///
    /// The algorithm iterates:
    /// 1. Project all points into the tangent space at the current estimate.
    /// 2. Compute the Euclidean mean of the tangent vectors.
    /// 3. Walk from the current estimate along this mean tangent vector via
    ///    the exponential map.
    /// 4. Repeat until convergence.
    ///
    /// # Arguments
    /// * `points` - The data points on the sphere (each should be non-zero)
    /// * `config` - Iteration parameters
    ///
    /// # Returns
    /// The Frechet mean, a point on the sphere that minimizes the sum of
    /// squared geodesic distances to all input points.
    pub fn frechet_mean(points: &[Vec<f64>], config: &FrechetMeanConfig) -> Vec<f64> {
        assert!(
            !points.is_empty(),
            "Cannot compute Frechet mean of empty set"
        );

        let dim = points[0].len();
        let n = points.len() as f64;

        // Normalize all input points
        let normed: Vec<Vec<f64>> = points.iter().map(|p| normalize(p)).collect();

        // Initialize with the first point (or could use extrinsic mean)
        let mut mu = normed[0].clone();

        for _iter in 0..config.max_iterations {
            // Compute mean tangent vector in the tangent space at mu
            let mut mean_tangent = vec![0.0; dim];
            for point in &normed {
                let log_v = Self::log_map(&mu, point);
                for (mt, lv) in mean_tangent.iter_mut().zip(log_v.iter()) {
                    *mt += lv;
                }
            }
            for mt in mean_tangent.iter_mut() {
                *mt /= n;
            }

            // Check convergence
            let step_size = norm(&mean_tangent);
            if step_size < config.tolerance {
                break;
            }

            // Update: walk along the mean tangent direction
            mu = Self::exp_map(&mu, &mean_tangent);
            mu = normalize(&mu); // Re-normalize for numerical stability
        }

        mu
    }

    // ========================================================================
    // Riemannian Gradient
    // ========================================================================

    /// Project an ambient (Euclidean) gradient onto the tangent space of the
    /// sphere at a given point.
    ///
    /// grad_S f(x) = nabla f(x) - <nabla f(x), x> * x
    ///
    /// When optimizing a function f on the sphere, the Euclidean gradient
    /// nabla f(x) generally points off the surface. The Riemannian gradient
    /// is its orthogonal projection onto the tangent hyperplane at x, giving
    /// the steepest-ascent direction that stays on the manifold.
    ///
    /// # Arguments
    /// * `ambient_grad` - The Euclidean gradient nabla f(x)
    /// * `point` - The current point x on the sphere (should be unit-norm)
    ///
    /// # Returns
    /// The Riemannian gradient, which is tangent to the sphere at `point`.
    pub fn riemannian_gradient(ambient_grad: &[f64], point: &[f64]) -> Vec<f64> {
        assert_eq!(
            ambient_grad.len(),
            point.len(),
            "riemannian_gradient: dimension mismatch grad={} point={}",
            ambient_grad.len(),
            point.len()
        );

        let p = normalize(point);
        let inner = dot(ambient_grad, &p);

        p.iter()
            .zip(ambient_grad.iter())
            .map(|(&pi, &gi)| gi - inner * pi)
            .collect()
    }

    // ========================================================================
    // Principal Geodesic Analysis
    // ========================================================================

    /// Perform Principal Geodesic Analysis (PGA) on a set of points on the
    /// hypersphere.
    ///
    /// PGA is the intrinsic analogue of PCA:
    /// 1. Compute the Frechet mean of the data.
    /// 2. Map all points to the tangent space at the mean via the log map.
    /// 3. Perform standard PCA (eigendecomposition of covariance) in the
    ///    tangent space.
    /// 4. Return the principal directions and their variances.
    ///
    /// The principal directions identify the geodesic submanifolds along which
    /// the data varies most. In consciousness terms, these are the axes of
    /// conceptual variation in a region of hypervector space.
    ///
    /// # Arguments
    /// * `points` - Data points on the sphere
    /// * `n_components` - Number of principal directions to extract
    ///
    /// # Returns
    /// A `PGAResult` containing the mean, principal directions, and variances.
    pub fn principal_geodesic_analysis(points: &[Vec<f64>], n_components: usize) -> PGAResult {
        assert!(!points.is_empty(), "Cannot perform PGA on empty point set");
        assert!(n_components > 0, "Must request at least one component");

        let dim = points[0].len();
        let n_components = n_components.min(dim);

        // Step 1: Compute Frechet mean
        let config = FrechetMeanConfig::default();
        let mean = Self::frechet_mean(points, &config);

        // Step 2: Map all points to tangent space at the mean
        let tangent_vecs: Vec<Vec<f64>> = points.iter().map(|p| Self::log_map(&mean, p)).collect();

        // Step 3: Compute covariance matrix in tangent space
        // C = (1/n) * sum_i (v_i * v_i^T)
        // We use power iteration to extract the top eigenvectors without
        // forming the full d x d covariance matrix (which would be huge
        // for d = 16,384).
        let n = tangent_vecs.len() as f64;

        let mut principal_directions = Vec::with_capacity(n_components);
        let mut variances = Vec::with_capacity(n_components);

        // Deflated tangent vectors (we subtract projections onto found directions)
        let mut deflated: Vec<Vec<f64>> = tangent_vecs.clone();

        for _k in 0..n_components {
            // Power iteration to find the top eigenvector of the covariance
            let eigvec = Self::power_iteration_covariance(&deflated, dim);
            let eigvec_norm = norm(&eigvec);

            if eigvec_norm < GEO_EPS {
                // No more variance to explain
                break;
            }

            let direction = normalize(&eigvec);

            // Compute variance along this direction
            let variance: f64 = deflated
                .iter()
                .map(|v| {
                    let proj = dot(v, &direction);
                    proj * proj
                })
                .sum::<f64>()
                / n;

            // Deflate: remove projection onto this direction from all vectors
            for v in deflated.iter_mut() {
                let proj = dot(v, &direction);
                for (vi, &di) in v.iter_mut().zip(direction.iter()) {
                    *vi -= proj * di;
                }
            }

            principal_directions.push(direction);
            variances.push(variance);
        }

        PGAResult {
            mean,
            principal_directions,
            variances,
        }
    }

    // ========================================================================
    // Geodesic Path Construction
    // ========================================================================

    /// Construct a `GeodesicPath` between two points.
    ///
    /// This pre-computes and caches the geodesic angle so that subsequent
    /// evaluations along the path (via SLERP) are efficient.
    pub fn geodesic_path(start: &[f64], end: &[f64]) -> GeodesicPath {
        let angle = Self::geodesic_distance(start, end);
        GeodesicPath {
            start: normalize(start),
            end: normalize(end),
            angle,
        }
    }

    // ========================================================================
    // Internal: Power Iteration for Covariance Eigenvectors
    // ========================================================================

    /// Extract the top eigenvector of the covariance matrix defined by a set
    /// of vectors, using power iteration.
    ///
    /// Instead of forming the d x d covariance matrix explicitly (O(d^2) memory),
    /// we compute matrix-vector products C * x = (1/n) sum_i <v_i, x> v_i
    /// which is O(n * d) per iteration.
    fn power_iteration_covariance(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
        const MAX_ITERS: usize = 200;
        const TOL: f64 = 1e-10;

        if vectors.is_empty() {
            return vec![0.0; dim];
        }

        // Initialize with the first non-zero vector, or a canonical direction
        let mut x: Vec<f64> = vectors
            .iter()
            .find(|v| norm(v) > GEO_EPS)
            .cloned()
            .unwrap_or_else(|| {
                let mut e = vec![0.0; dim];
                if dim > 0 {
                    e[0] = 1.0;
                }
                e
            });
        x = normalize(&x);

        let n = vectors.len() as f64;

        for _ in 0..MAX_ITERS {
            // Compute C * x = (1/n) sum_i <v_i, x> v_i
            let mut cx = vec![0.0; dim];
            for v in vectors {
                let proj = dot(v, &x);
                for (cxi, &vi) in cx.iter_mut().zip(v.iter()) {
                    *cxi += proj * vi;
                }
            }
            for cxi in cx.iter_mut() {
                *cxi /= n;
            }

            let cx_norm = norm(&cx);
            if cx_norm < GEO_EPS {
                break;
            }

            let new_x = normalize(&cx);

            // Check convergence: |1 - |<x, new_x>|| < tol
            let alignment = dot(&x, &new_x).abs();
            x = new_x;

            if (1.0 - alignment).abs() < TOL {
                break;
            }
        }

        x
    }
}

// ============================================================================
// GeodesicPath methods
// ============================================================================

impl GeodesicPath {
    /// Evaluate the geodesic at parameter t in [0, 1].
    ///
    /// This is equivalent to `HypersphereOps::slerp(&self.start, &self.end, t)`
    /// but uses the pre-computed angle for efficiency.
    pub fn evaluate(&self, t: f64) -> Vec<f64> {
        if self.angle.abs() < GEO_EPS {
            return self.start.clone();
        }

        let sin_theta = self.angle.sin();
        if sin_theta.abs() < GEO_EPS {
            // Nearly antipodal: fall back to normalized lerp
            let result: Vec<f64> = self
                .start
                .iter()
                .zip(self.end.iter())
                .map(|(&a, &b)| (1.0 - t) * a + t * b)
                .collect();
            return normalize(&result);
        }

        let weight_start = ((1.0 - t) * self.angle).sin() / sin_theta;
        let weight_end = (t * self.angle).sin() / sin_theta;

        self.start
            .iter()
            .zip(self.end.iter())
            .map(|(&a, &b)| weight_start * a + weight_end * b)
            .collect()
    }

    /// Compute the midpoint of the geodesic.
    pub fn midpoint(&self) -> Vec<f64> {
        self.evaluate(0.5)
    }
}

// ============================================================================
// TangentVector methods
// ============================================================================

impl TangentVector {
    /// Create a new tangent vector, verifying orthogonality to the base point.
    ///
    /// The input vector is projected onto the tangent space (orthogonal to
    /// `base_point`) to ensure it is a valid tangent vector.
    pub fn new(base_point: Vec<f64>, vector: Vec<f64>) -> Self {
        let p = normalize(&base_point);
        // Project vector onto tangent space: v - <v, p> p
        let inner = dot(&vector, &p);
        let projected: Vec<f64> = vector
            .iter()
            .zip(p.iter())
            .map(|(&vi, &pi)| vi - inner * pi)
            .collect();

        Self {
            base_point: p,
            vector: projected,
        }
    }

    /// The norm (length) of the tangent vector, which equals the geodesic
    /// distance that the exponential map would traverse.
    pub fn norm(&self) -> f64 {
        norm(&self.vector)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: create a unit vector along a single coordinate axis.
    fn unit_vec(dim: usize, axis: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v
    }

    /// Helper: check that two f64 values are approximately equal.
    fn assert_approx(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: expected {} ~= {}, diff = {}",
            msg,
            a,
            b,
            (a - b).abs()
        );
    }

    /// Helper: check that two vectors are approximately equal (element-wise).
    fn assert_vec_approx(a: &[f64], b: &[f64], tol: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: dimension mismatch", msg);
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() < tol,
                "{}: element {} differs: {} vs {} (diff = {})",
                msg,
                i,
                ai,
                bi,
                (ai - bi).abs()
            );
        }
    }

    // ====================================================================
    // Geodesic distance tests
    // ====================================================================

    #[test]
    fn test_geodesic_distance_same_point() {
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let d = HypersphereOps::geodesic_distance(&v, &v);
        assert_approx(
            d,
            0.0,
            1e-10,
            "Distance of a point to itself should be zero",
        );
    }

    #[test]
    fn test_geodesic_distance_orthogonal_points() {
        let u = unit_vec(5, 0); // (1, 0, 0, 0, 0)
        let v = unit_vec(5, 1); // (0, 1, 0, 0, 0)
        let d = HypersphereOps::geodesic_distance(&u, &v);
        assert_approx(
            d,
            PI / 2.0,
            1e-10,
            "Orthogonal unit vectors should be pi/2 apart",
        );
    }

    #[test]
    fn test_geodesic_distance_antipodal() {
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![-1.0, 0.0, 0.0];
        let d = HypersphereOps::geodesic_distance(&u, &v);
        assert_approx(d, PI, 1e-10, "Antipodal points should be pi apart");
    }

    #[test]
    fn test_geodesic_distance_symmetry() {
        let u = vec![0.6, 0.8, 0.0];
        let v = vec![0.0, 0.6, 0.8];
        assert_approx(
            HypersphereOps::geodesic_distance(&u, &v),
            HypersphereOps::geodesic_distance(&v, &u),
            1e-14,
            "Geodesic distance should be symmetric",
        );
    }

    // ====================================================================
    // SLERP tests
    // ====================================================================

    #[test]
    fn test_slerp_at_endpoints() {
        let u = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0, 0.0];

        let at_zero = HypersphereOps::slerp(&u, &v, 0.0);
        let at_one = HypersphereOps::slerp(&u, &v, 1.0);

        let u_hat = normalize(&u);
        let v_hat = normalize(&v);

        assert_vec_approx(&at_zero, &u_hat, 1e-10, "SLERP at t=0 should return start");
        assert_vec_approx(&at_one, &v_hat, 1e-10, "SLERP at t=1 should return end");
    }

    #[test]
    fn test_slerp_midpoint_on_sphere() {
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];

        let mid = HypersphereOps::slerp(&u, &v, 0.5);
        let mid_norm = norm(&mid);

        assert_approx(
            mid_norm,
            1.0,
            1e-10,
            "SLERP midpoint should lie on the unit sphere",
        );

        // The midpoint of the arc from (1,0,0) to (0,1,0) should be at 45 degrees
        let expected = vec![(PI / 4.0).cos(), (PI / 4.0).sin(), 0.0];
        assert_vec_approx(
            &mid,
            &expected,
            1e-10,
            "SLERP midpoint should be at 45 degrees",
        );
    }

    #[test]
    fn test_slerp_preserves_unit_norm() {
        let u = vec![0.6, 0.8, 0.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 0.6, 0.0, 0.8];

        for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let interp = HypersphereOps::slerp(&u, &v, t);
            let n = norm(&interp);
            assert_approx(
                n,
                1.0,
                1e-10,
                &format!("SLERP at t={} should produce unit vector", t),
            );
        }
    }

    // ====================================================================
    // Exp/Log round-trip tests
    // ====================================================================

    #[test]
    fn test_exp_then_log_roundtrip() {
        let base = normalize(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        // Create a tangent vector at base (must be orthogonal to base)
        let tangent_raw = vec![0.0, 0.3, 0.0, 0.0, 0.0];
        // It is already orthogonal to base=(1,0,0,0,0) since component 0 is 0

        let mapped = HypersphereOps::exp_map(&base, &tangent_raw);
        let recovered = HypersphereOps::log_map(&base, &mapped);

        assert_vec_approx(
            &recovered,
            &tangent_raw,
            1e-10,
            "Log(Exp(v)) should recover the tangent vector v",
        );
    }

    #[test]
    fn test_log_then_exp_roundtrip() {
        let base = normalize(&[1.0, 0.0, 0.0, 0.0]);
        let point = normalize(&[0.6, 0.8, 0.0, 0.0]);

        let tangent = HypersphereOps::log_map(&base, &point);
        let recovered = HypersphereOps::exp_map(&base, &tangent);

        assert_vec_approx(
            &recovered,
            &normalize(&[0.6, 0.8, 0.0, 0.0]),
            1e-10,
            "Exp(Log(q)) should recover the original point q",
        );
    }

    #[test]
    fn test_log_map_norm_equals_geodesic_distance() {
        let p = normalize(&[1.0, 0.0, 0.0, 0.0]);
        let q = normalize(&[0.6, 0.8, 0.0, 0.0]);

        let tangent = HypersphereOps::log_map(&p, &q);
        let tangent_norm = norm(&tangent);
        let geo_dist = HypersphereOps::geodesic_distance(&p, &q);

        assert_approx(
            tangent_norm,
            geo_dist,
            1e-10,
            "||Log_p(q)|| should equal geodesic distance d(p, q)",
        );
    }

    // ====================================================================
    // Frechet mean tests
    // ====================================================================

    #[test]
    fn test_frechet_mean_single_point() {
        let point = normalize(&[0.5, 0.5, 0.5, 0.5]);
        let config = FrechetMeanConfig::default();
        let mean = HypersphereOps::frechet_mean(&[point.clone()], &config);

        assert_vec_approx(
            &mean,
            &point,
            1e-10,
            "Frechet mean of a single point should be that point",
        );
    }

    #[test]
    fn test_frechet_mean_of_cluster() {
        // Create a tight cluster around (1, 0, 0, 0) with small perturbations
        let center = vec![1.0, 0.0, 0.0, 0.0];
        let perturbations = vec![
            normalize(&[1.0, 0.05, 0.0, 0.0]),
            normalize(&[1.0, -0.05, 0.0, 0.0]),
            normalize(&[1.0, 0.0, 0.05, 0.0]),
            normalize(&[1.0, 0.0, -0.05, 0.0]),
            normalize(&[1.0, 0.0, 0.0, 0.05]),
            normalize(&[1.0, 0.0, 0.0, -0.05]),
        ];

        let config = FrechetMeanConfig {
            max_iterations: 200,
            tolerance: 1e-12,
        };
        let mean = HypersphereOps::frechet_mean(&perturbations, &config);

        // The mean should be close to the normalized center
        let center_norm = normalize(&center);
        let dist = HypersphereOps::geodesic_distance(&mean, &center_norm);

        assert!(
            dist < 0.01,
            "Frechet mean of symmetric cluster should be near the center, got dist = {}",
            dist
        );
    }

    #[test]
    fn test_frechet_mean_of_two_points_is_midpoint() {
        let u = normalize(&[1.0, 0.0, 0.0]);
        let v = normalize(&[0.0, 1.0, 0.0]);

        let config = FrechetMeanConfig {
            max_iterations: 500,
            tolerance: 1e-14,
        };
        let mean = HypersphereOps::frechet_mean(&[u.clone(), v.clone()], &config);

        // The Frechet mean of two points should be the geodesic midpoint
        let midpoint = HypersphereOps::slerp(&u, &v, 0.5);
        let dist = HypersphereOps::geodesic_distance(&mean, &midpoint);

        assert!(
            dist < 1e-6,
            "Frechet mean of two points should be their geodesic midpoint, got dist = {}",
            dist
        );
    }

    // ====================================================================
    // Riemannian gradient tests
    // ====================================================================

    #[test]
    fn test_riemannian_gradient_tangent_to_sphere() {
        // The Riemannian gradient should be orthogonal to the point on the sphere
        let point = normalize(&[0.5, 0.5, 0.5, 0.5]);
        let ambient_grad = vec![1.0, -2.0, 3.0, -0.5];

        let riem_grad = HypersphereOps::riemannian_gradient(&ambient_grad, &point);

        let inner = dot(&riem_grad, &point);
        assert_approx(
            inner,
            0.0,
            1e-10,
            "Riemannian gradient should be tangent to the sphere (dot with point ~= 0)",
        );
    }

    #[test]
    fn test_riemannian_gradient_of_radial_function_is_zero() {
        // If the ambient gradient is purely radial (parallel to the point),
        // the Riemannian gradient should be zero.
        let point = normalize(&[3.0, 4.0, 0.0]);
        let radial_grad: Vec<f64> = point.iter().map(|&x| 2.5 * x).collect();

        let riem_grad = HypersphereOps::riemannian_gradient(&radial_grad, &point);
        let riem_norm = norm(&riem_grad);

        assert_approx(
            riem_norm,
            0.0,
            1e-10,
            "Riemannian gradient of a purely radial function should be zero on the sphere",
        );
    }

    #[test]
    fn test_riemannian_gradient_preserves_tangential_component() {
        // If the ambient gradient is already tangent to the sphere at the point,
        // the Riemannian gradient should equal the ambient gradient.
        let point = normalize(&[1.0, 0.0, 0.0, 0.0]);
        // Tangent to sphere at (1,0,0,0): any vector with first component = 0
        let tangent_grad = vec![0.0, 1.5, -0.7, 0.3];

        let riem_grad = HypersphereOps::riemannian_gradient(&tangent_grad, &point);

        assert_vec_approx(
            &riem_grad,
            &tangent_grad,
            1e-10,
            "Riemannian gradient should preserve already-tangent components",
        );
    }

    // ====================================================================
    // Parallel transport tests
    // ====================================================================

    #[test]
    fn test_parallel_transport_preserves_norm() {
        let from = normalize(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let to = normalize(&[0.0, 1.0, 0.0, 0.0, 0.0]);
        // A tangent vector at `from`: must be orthogonal to from, so component 0 = 0
        let vector = vec![0.0, 0.0, 0.5, 0.3, 0.0];

        let transported = HypersphereOps::parallel_transport(&from, &to, &vector);
        let original_norm = norm(&vector);
        let transported_norm = norm(&transported);

        assert_approx(
            transported_norm,
            original_norm,
            1e-8,
            "Parallel transport should preserve the norm of the tangent vector",
        );
    }

    #[test]
    fn test_parallel_transport_same_point_identity() {
        let p = normalize(&[0.6, 0.8, 0.0]);
        let v = vec![0.0, 0.0, 1.0]; // tangent at p (approximately)

        let transported = HypersphereOps::parallel_transport(&p, &p, &v);

        assert_vec_approx(
            &transported,
            &v,
            1e-10,
            "Parallel transport from a point to itself should be the identity",
        );
    }

    // ====================================================================
    // Principal Geodesic Analysis tests
    // ====================================================================

    #[test]
    fn test_pga_single_direction_variance() {
        // Create points that vary only along one direction on the sphere.
        // Start from e_0 = (1, 0, 0, 0, 0) and perturb along e_1.
        let dim = 5;
        let _base = unit_vec(dim, 0);

        let points: Vec<Vec<f64>> = (-5..=5)
            .map(|i| {
                let angle = (i as f64) * 0.05; // small angles
                let mut p = vec![0.0; dim];
                p[0] = angle.cos();
                p[1] = angle.sin();
                p
            })
            .collect();

        let result = HypersphereOps::principal_geodesic_analysis(&points, 2);

        assert!(
            !result.principal_directions.is_empty(),
            "PGA should find at least one principal direction",
        );
        assert!(
            result.variances[0] > 1e-6,
            "First principal variance should be non-trivial",
        );

        // If there is a second component, its variance should be much smaller
        if result.variances.len() > 1 {
            assert!(
                result.variances[0] > result.variances[1] * 10.0,
                "First variance ({}) should dominate second ({}) for 1D data",
                result.variances[0],
                result.variances[1]
            );
        }
    }

    // ====================================================================
    // GeodesicPath tests
    // ====================================================================

    #[test]
    fn test_geodesic_path_evaluate() {
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];

        let path = HypersphereOps::geodesic_path(&u, &v);

        assert_approx(
            path.angle,
            PI / 2.0,
            1e-10,
            "Geodesic angle between orthogonal basis vectors should be pi/2",
        );

        let at_start = path.evaluate(0.0);
        let at_end = path.evaluate(1.0);

        assert_vec_approx(
            &at_start,
            &normalize(&u),
            1e-10,
            "Path at t=0 should be start",
        );
        assert_vec_approx(&at_end, &normalize(&v), 1e-10, "Path at t=1 should be end");
    }

    // ====================================================================
    // ContinuousHV conversion round-trip
    // ====================================================================

    #[test]
    fn test_real_hv_conversion_roundtrip() {
        let original = ContinuousHV::random(64, 42);
        let as_f64 = from_real_hv(&original);
        let back = to_real_hv(&as_f64);

        assert_eq!(original.dim(), back.dim(), "Dimension should be preserved");

        // f32 -> f64 -> f32 should be lossless for representable values
        for (a, b) in original.values.iter().zip(back.values.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Round-trip conversion should preserve values: {} vs {}",
                a,
                b
            );
        }
    }

    // ====================================================================
    // High-dimensional smoke test
    // ====================================================================

    #[test]
    fn test_high_dimensional_slerp() {
        // Test with a more realistic dimension (not full 16384 to keep tests fast)
        let dim = 256;
        let u: Vec<f64> = (0..dim).map(|i| ((i as f64) * 0.1).sin()).collect();
        let v: Vec<f64> = (0..dim).map(|i| ((i as f64) * 0.1).cos()).collect();

        let u = normalize(&u);
        let v = normalize(&v);

        let mid = HypersphereOps::slerp(&u, &v, 0.5);
        let mid_norm = norm(&mid);

        assert_approx(
            mid_norm,
            1.0,
            1e-10,
            "SLERP midpoint should have unit norm in high dimensions",
        );

        // Midpoint should be equidistant from both endpoints
        let d_u = HypersphereOps::geodesic_distance(&u, &mid);
        let d_v = HypersphereOps::geodesic_distance(&v, &mid);

        assert_approx(
            d_u,
            d_v,
            1e-8,
            "SLERP midpoint should be equidistant from both endpoints",
        );
    }
}
