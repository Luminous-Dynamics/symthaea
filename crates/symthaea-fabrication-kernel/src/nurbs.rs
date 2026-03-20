//! NURBS curves and surfaces with adaptive tessellation
//!
//! Implements Non-Uniform Rational B-Spline (NURBS) geometry following
//! the De Boor algorithm for evaluation and Cox-de Boor recursion for
//! basis functions. Adaptive tessellation subdivides until a flatness
//! tolerance is met, producing triangle meshes suitable for fabrication.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Basis functions (Cox-de Boor recursion)
// ---------------------------------------------------------------------------

/// Evaluate the `i`-th B-spline basis function of given `degree` at parameter `t`.
///
/// Uses the Cox-de Boor recursion formula. The knot vector must contain at
/// least `i + degree + 2` entries.
pub fn basis_function(knots: &[f32], i: usize, degree: u32, t: f32) -> f32 {
    if degree == 0 {
        // Half-open interval [knots[i], knots[i+1])
        // Special case: if t == last knot, include the last span
        if i + 1 < knots.len() {
            let last_knot = knots[knots.len() - 1];
            if t >= knots[i] && (t < knots[i + 1] || (t == last_knot && t == knots[i + 1])) {
                return 1.0;
            }
        }
        return 0.0;
    }

    let mut result = 0.0;

    // Left term
    let denom_left = knots[i + degree as usize] - knots[i];
    if denom_left.abs() > 1e-10 {
        result += (t - knots[i]) / denom_left * basis_function(knots, i, degree - 1, t);
    }

    // Right term
    if i + degree as usize + 1 < knots.len() {
        let denom_right = knots[i + degree as usize + 1] - knots[i + 1];
        if denom_right.abs() > 1e-10 {
            result += (knots[i + degree as usize + 1] - t) / denom_right
                * basis_function(knots, i + 1, degree - 1, t);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// NurbsCurve
// ---------------------------------------------------------------------------

/// A Non-Uniform Rational B-Spline curve in 3D.
///
/// The curve is defined by `n+1` control points, a weight per control point,
/// a knot vector of length `n + degree + 2`, and a polynomial degree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NurbsCurve {
    /// Polynomial degree (1 = linear, 2 = quadratic, 3 = cubic, ...)
    pub degree: u32,
    /// Control points in 3D
    pub control_points: Vec<[f32; 3]>,
    /// Per-control-point weights (rational component)
    pub weights: Vec<f32>,
    /// Knot vector — length must equal `control_points.len() + degree + 1`
    pub knots: Vec<f32>,
}

impl NurbsCurve {
    /// Find the knot span index for parameter `t`.
    ///
    /// Returns the index `i` such that `knots[i] <= t < knots[i+1]`,
    /// clamped to the valid range `[degree, n]` where `n = control_points.len() - 1`.
    pub fn find_knot_span(&self, t: f32) -> usize {
        let n = self.control_points.len() - 1;
        let p = self.degree as usize;

        // Clamp to domain boundaries
        if t >= self.knots[n + 1] {
            return n;
        }
        if t <= self.knots[p] {
            return p;
        }

        // Binary search
        let mut low = p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while t < self.knots[mid] || t >= self.knots[mid + 1] {
            if t < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    /// Evaluate the curve at parameter `t` using De Boor's algorithm.
    ///
    /// The parameter should lie within the knot domain
    /// `[knots[degree], knots[n+1]]`.
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let n = self.control_points.len();
        let p = self.degree as usize;

        // Clamp t to the valid domain
        let t_min = self.knots[p];
        let t_max = self.knots[n];
        let t = t.clamp(t_min, t_max);

        // Rational evaluation: compute weighted sum of basis functions
        let mut numerator = [0.0f32; 3];
        let mut denominator = 0.0f32;

        for i in 0..n {
            let basis = basis_function(&self.knots, i, self.degree, t);
            let w = self.weights[i];
            let bw = basis * w;
            numerator[0] += bw * self.control_points[i][0];
            numerator[1] += bw * self.control_points[i][1];
            numerator[2] += bw * self.control_points[i][2];
            denominator += bw;
        }

        if denominator.abs() < 1e-10 {
            return self.control_points[0];
        }

        [
            numerator[0] / denominator,
            numerator[1] / denominator,
            numerator[2] / denominator,
        ]
    }

    /// Adaptively tessellate the curve to a polyline with the given flatness
    /// `tolerance`.
    ///
    /// Subdivides spans where the midpoint deviation from a straight line
    /// exceeds `tolerance`. Returns the list of sampled 3D points.
    pub fn tessellate(&self, tolerance: f32) -> Vec<[f32; 3]> {
        let p = self.degree as usize;
        let n = self.control_points.len();
        let t_min = self.knots[p];
        let t_max = self.knots[n];

        let mut result = Vec::new();
        result.push(self.evaluate(t_min));
        self.tessellate_recursive(t_min, t_max, tolerance, &mut result, 0);
        result
    }

    fn tessellate_recursive(
        &self,
        t0: f32,
        t1: f32,
        tolerance: f32,
        result: &mut Vec<[f32; 3]>,
        depth: usize,
    ) {
        const MAX_DEPTH: usize = 12;

        if depth >= MAX_DEPTH || (t1 - t0) < 1e-7 {
            result.push(self.evaluate(t1));
            return;
        }

        let t_mid = (t0 + t1) * 0.5;
        let p0 = self.evaluate(t0);
        let p1 = self.evaluate(t1);
        let p_mid = self.evaluate(t_mid);

        // Midpoint of the chord
        let chord_mid = [
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5,
        ];

        let dx = p_mid[0] - chord_mid[0];
        let dy = p_mid[1] - chord_mid[1];
        let dz = p_mid[2] - chord_mid[2];
        let deviation = (dx * dx + dy * dy + dz * dz).sqrt();

        if deviation > tolerance {
            self.tessellate_recursive(t0, t_mid, tolerance, result, depth + 1);
            self.tessellate_recursive(t_mid, t1, tolerance, result, depth + 1);
        } else {
            result.push(self.evaluate(t1));
        }
    }

    /// Estimate the arc length of the curve by sampling `segments` uniform
    /// chords and summing their lengths.
    pub fn length_estimate(&self, segments: usize) -> f32 {
        let p = self.degree as usize;
        let n = self.control_points.len();
        let t_min = self.knots[p];
        let t_max = self.knots[n];
        let segments = segments.max(1);

        let mut total = 0.0f32;
        let mut prev = self.evaluate(t_min);
        for i in 1..=segments {
            let t = t_min + (t_max - t_min) * (i as f32) / (segments as f32);
            let cur = self.evaluate(t);
            let dx = cur[0] - prev[0];
            let dy = cur[1] - prev[1];
            let dz = cur[2] - prev[2];
            total += (dx * dx + dy * dy + dz * dz).sqrt();
            prev = cur;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// NurbsSurface
// ---------------------------------------------------------------------------

/// A NURBS surface defined as a tensor-product patch.
///
/// The surface is parameterized over `(u, v)` with independent degree,
/// control net, weights, and knot vectors in each direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NurbsSurface {
    pub degree_u: u32,
    pub degree_v: u32,
    /// Control net: `control_points[i][j]` for the i-th row, j-th column
    pub control_points: Vec<Vec<[f32; 3]>>,
    /// Weights matching the control net layout
    pub weights: Vec<Vec<f32>>,
    /// Knot vector in the u-direction
    pub knots_u: Vec<f32>,
    /// Knot vector in the v-direction
    pub knots_v: Vec<f32>,
}

impl NurbsSurface {
    /// Evaluate the surface at parameters `(u, v)`.
    pub fn evaluate(&self, u: f32, v: f32) -> [f32; 3] {
        let rows = self.control_points.len();
        let cols = if rows > 0 { self.control_points[0].len() } else { 0 };
        let pu = self.degree_u as usize;
        let pv = self.degree_v as usize;

        // Clamp parameters
        let u = u.clamp(self.knots_u[pu], self.knots_u[rows]);
        let v = v.clamp(self.knots_v[pv], self.knots_v[cols]);

        let mut numerator = [0.0f32; 3];
        let mut denominator = 0.0f32;

        for i in 0..rows {
            let bu = basis_function(&self.knots_u, i, self.degree_u, u);
            for j in 0..cols {
                let bv = basis_function(&self.knots_v, j, self.degree_v, v);
                let w = self.weights[i][j];
                let bw = bu * bv * w;
                numerator[0] += bw * self.control_points[i][j][0];
                numerator[1] += bw * self.control_points[i][j][1];
                numerator[2] += bw * self.control_points[i][j][2];
                denominator += bw;
            }
        }

        if denominator.abs() < 1e-10 {
            return self.control_points[0][0];
        }

        [
            numerator[0] / denominator,
            numerator[1] / denominator,
            numerator[2] / denominator,
        ]
    }

    /// Compute the surface normal at `(u, v)` via finite-difference partial
    /// derivatives and cross product.
    pub fn normal_at(&self, u: f32, v: f32) -> [f32; 3] {
        let eps = 1e-4f32;
        let p = self.evaluate(u, v);

        // Partial derivative with respect to u
        let pu_plus = self.evaluate(u + eps, v);
        let pu_minus = self.evaluate(u - eps, v);
        let du = [
            (pu_plus[0] - pu_minus[0]) / (2.0 * eps),
            (pu_plus[1] - pu_minus[1]) / (2.0 * eps),
            (pu_plus[2] - pu_minus[2]) / (2.0 * eps),
        ];

        // Partial derivative with respect to v
        let pv_plus = self.evaluate(u, v + eps);
        let pv_minus = self.evaluate(u, v - eps);
        let dv = [
            (pv_plus[0] - pv_minus[0]) / (2.0 * eps),
            (pv_plus[1] - pv_minus[1]) / (2.0 * eps),
            (pv_plus[2] - pv_minus[2]) / (2.0 * eps),
        ];

        // Cross product du x dv
        let mut n = [
            du[1] * dv[2] - du[2] * dv[1],
            du[2] * dv[0] - du[0] * dv[2],
            du[0] * dv[1] - du[1] * dv[0],
        ];

        // Normalize
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-10 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            n = [0.0, 0.0, 1.0]; // fallback
        }

        n
    }

    /// Adaptively tessellate the surface into a [`TriangleMesh`].
    ///
    /// The surface is initially sampled on a grid and then refined where the
    /// deviation between the true surface point and the bilinear interpolation
    /// of the grid exceeds `tolerance`.
    pub fn tessellate(&self, tolerance: f32) -> crate::mesh::TriangleMesh {
        let rows = self.control_points.len();
        let cols = if rows > 0 { self.control_points[0].len() } else { 0 };
        let pu = self.degree_u as usize;
        let pv = self.degree_v as usize;

        let u_min = self.knots_u[pu];
        let u_max = self.knots_u[rows];
        let v_min = self.knots_v[pv];
        let v_max = self.knots_v[cols];

        // Determine initial grid resolution
        let initial_u_steps = ((rows + 1) * 4).max(8);
        let initial_v_steps = ((cols + 1) * 4).max(8);

        // Build initial grid
        let mut grid = TessGrid::new(self, initial_u_steps, initial_v_steps, u_min, u_max, v_min, v_max);

        // Adaptive refinement
        grid.refine(self, tolerance);

        // Convert grid to triangle mesh
        grid.to_triangle_mesh(self)
    }
}

// ---------------------------------------------------------------------------
// Adaptive tessellation grid helper
// ---------------------------------------------------------------------------

struct TessGrid {
    u_params: Vec<f32>,
    v_params: Vec<f32>,
    /// Cached surface points: points[u_idx][v_idx]
    points: Vec<Vec<[f32; 3]>>,
}

impl TessGrid {
    fn new(
        surface: &NurbsSurface,
        u_steps: usize,
        v_steps: usize,
        u_min: f32,
        u_max: f32,
        v_min: f32,
        v_max: f32,
    ) -> Self {
        let u_params: Vec<f32> = (0..=u_steps)
            .map(|i| u_min + (u_max - u_min) * (i as f32) / (u_steps as f32))
            .collect();
        let v_params: Vec<f32> = (0..=v_steps)
            .map(|j| v_min + (v_max - v_min) * (j as f32) / (v_steps as f32))
            .collect();

        let mut points = Vec::with_capacity(u_params.len());
        for &u in &u_params {
            let mut row = Vec::with_capacity(v_params.len());
            for &v in &v_params {
                row.push(surface.evaluate(u, v));
            }
            points.push(row);
        }

        Self {
            u_params,
            v_params,
            points,
        }
    }

    /// One pass of adaptive refinement: insert midpoints where deviation is
    /// too large. Limited to one level to avoid exponential blowup.
    fn refine(&mut self, surface: &NurbsSurface, tolerance: f32) {
        // Refine in u-direction
        let mut new_u = Vec::new();
        for i in 0..self.u_params.len() - 1 {
            let u_mid = (self.u_params[i] + self.u_params[i + 1]) * 0.5;
            // Check deviation at one v sample point
            let v_mid_idx = self.v_params.len() / 2;
            let v = self.v_params[v_mid_idx];
            let actual = surface.evaluate(u_mid, v);
            let interp = midpoint_3d(self.points[i][v_mid_idx], self.points[i + 1][v_mid_idx]);
            if dist_3d(actual, interp) > tolerance {
                new_u.push((i + 1, u_mid));
            }
        }
        // Insert in reverse order to preserve indices
        for (idx, u) in new_u.into_iter().rev() {
            self.u_params.insert(idx, u);
            let mut row = Vec::with_capacity(self.v_params.len());
            for &v in &self.v_params {
                row.push(surface.evaluate(u, v));
            }
            self.points.insert(idx, row);
        }

        // Refine in v-direction
        let mut new_v = Vec::new();
        for j in 0..self.v_params.len() - 1 {
            let v_mid = (self.v_params[j] + self.v_params[j + 1]) * 0.5;
            let u_mid_idx = self.u_params.len() / 2;
            let u = self.u_params[u_mid_idx];
            let actual = surface.evaluate(u, v_mid);
            let interp = midpoint_3d(self.points[u_mid_idx][j], self.points[u_mid_idx][j + 1]);
            if dist_3d(actual, interp) > tolerance {
                new_v.push((j + 1, v_mid));
            }
        }
        for (idx, v) in new_v.into_iter().rev() {
            self.v_params.insert(idx, v);
        }
        // Re-evaluate rows whose length no longer matches v_params
        for (ui, row) in self.points.iter_mut().enumerate() {
            if row.len() != self.v_params.len() {
                let u = self.u_params[ui];
                *row = self.v_params.iter().map(|&v| surface.evaluate(u, v)).collect();
            }
        }
    }

    fn to_triangle_mesh(&self, surface: &NurbsSurface) -> crate::mesh::TriangleMesh {
        let u_count = self.u_params.len();
        let v_count = self.v_params.len();

        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        // Flatten grid points into vertex buffer
        for i in 0..u_count {
            for j in 0..v_count {
                vertices.push(self.points[i][j]);
                normals.push(surface.normal_at(self.u_params[i], self.v_params[j]));
            }
        }

        // Triangulate grid quads
        for i in 0..u_count - 1 {
            for j in 0..v_count - 1 {
                let a = (i * v_count + j) as u32;
                let b = (i * v_count + j + 1) as u32;
                let c = ((i + 1) * v_count + j + 1) as u32;
                let d = ((i + 1) * v_count + j) as u32;
                indices.push([a, d, b]);
                indices.push([b, d, c]);
            }
        }

        crate::mesh::TriangleMesh {
            vertices,
            normals,
            indices,
        }
    }
}

fn midpoint_3d(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn dist_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl NurbsCurve {
    /// Create a line segment from `a` to `b` (degree-1 NURBS).
    pub fn line(a: [f32; 3], b: [f32; 3]) -> Self {
        Self {
            degree: 1,
            control_points: vec![a, b],
            weights: vec![1.0, 1.0],
            knots: vec![0.0, 0.0, 1.0, 1.0],
        }
    }

    /// Approximate a quarter-circle arc in the XY plane with a rational
    /// quadratic NURBS (degree 2). Radius `r`, centered at origin, from
    /// `(r, 0, 0)` to `(0, r, 0)`.
    pub fn quarter_circle(r: f32) -> Self {
        let w_mid = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)
        Self {
            degree: 2,
            control_points: vec![[r, 0.0, 0.0], [r, r, 0.0], [0.0, r, 0.0]],
            weights: vec![1.0, w_mid, 1.0],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    }
}

impl NurbsSurface {
    /// Create a flat bilinear patch (degree 1x1) from four corners.
    pub fn bilinear(corners: [[f32; 3]; 4]) -> Self {
        Self {
            degree_u: 1,
            degree_v: 1,
            control_points: vec![
                vec![corners[0], corners[1]],
                vec![corners[2], corners[3]],
            ],
            weights: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_curve_evaluation() {
        let curve = NurbsCurve::line([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let p0 = curve.evaluate(0.0);
        let p_mid = curve.evaluate(0.5);
        let p1 = curve.evaluate(1.0);

        assert!((p0[0]).abs() < 1e-5);
        assert!((p_mid[0] - 5.0).abs() < 1e-4);
        assert!((p1[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_circle_approximation() {
        let curve = NurbsCurve::quarter_circle(1.0);
        // All evaluated points should lie approximately on the unit circle
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let p = curve.evaluate(t);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 0.01,
                "Point at t={} has radius {}, expected ~1.0",
                t,
                r
            );
        }
    }

    #[test]
    fn test_surface_evaluation_bilinear() {
        let surface = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let center = surface.evaluate(0.5, 0.5);
        assert!((center[0] - 0.5).abs() < 0.05);
        assert!((center[1] - 0.5).abs() < 0.05);
        assert!(center[2].abs() < 0.05);
    }

    #[test]
    fn test_surface_normal_flat() {
        let surface = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let n = surface.normal_at(0.5, 0.5);
        // Should point in z-direction for a flat XY patch
        assert!(n[2].abs() > 0.9, "Normal z component should dominate: {:?}", n);
    }

    #[test]
    fn test_tessellation_triangle_count() {
        let surface = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let mesh = surface.tessellate(0.01);
        // A bilinear patch should produce a reasonable number of triangles
        assert!(
            mesh.triangle_count() >= 2,
            "Expected at least 2 triangles, got {}",
            mesh.triangle_count()
        );
        // Vertices and normals should match
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
    }

    #[test]
    fn test_basis_function_partition_of_unity() {
        // For a clamped knot vector, basis functions should sum to 1.0
        // at any interior parameter value.
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // degree=3, 4 CPs
        let degree = 3u32;
        let n_basis = 4; // knots.len() - degree - 1

        for step in 1..20 {
            let t = step as f32 / 20.0;
            let sum: f32 = (0..n_basis).map(|i| basis_function(&knots, i, degree, t)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Basis functions at t={} sum to {}, expected 1.0",
                t,
                sum
            );
        }
    }

    #[test]
    fn test_knot_span_finding() {
        let curve = NurbsCurve {
            degree: 2,
            control_points: vec![[0.0; 3]; 5],
            weights: vec![1.0; 5],
            knots: vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        };

        assert_eq!(curve.find_knot_span(0.0), 2);
        assert_eq!(curve.find_knot_span(0.25), 2);
        assert_eq!(curve.find_knot_span(0.5), 3);
        assert_eq!(curve.find_knot_span(0.75), 3);
        // At end, should clamp to n
        assert_eq!(curve.find_knot_span(1.0), 4);
    }

    #[test]
    fn test_length_estimate_line() {
        let curve = NurbsCurve::line([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        let len = curve.length_estimate(100);
        assert!(
            (len - 5.0).abs() < 0.01,
            "Line length should be 5.0, got {}",
            len
        );
    }

    #[test]
    fn test_degenerate_single_point_curve() {
        // A degree-1 curve with both control points at the same location
        let curve = NurbsCurve::line([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]);
        let p = curve.evaluate(0.5);
        assert!((p[0] - 5.0).abs() < 1e-5);
        assert!((p[1] - 5.0).abs() < 1e-5);
        assert!((p[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_tessellate_line_curve() {
        let curve = NurbsCurve::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let pts = curve.tessellate(0.001);
        // A line should tessellate to very few points (start + end minimum)
        assert!(pts.len() >= 2);
        // First and last should be endpoints
        assert!((pts[0][0]).abs() < 1e-4);
        assert!((pts.last().unwrap()[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tessellate_circle_more_points_than_line() {
        let line = NurbsCurve::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let circle = NurbsCurve::quarter_circle(1.0);

        let line_pts = line.tessellate(0.001);
        let circle_pts = circle.tessellate(0.001);

        // A curved shape should require more tessellation points than a line
        assert!(
            circle_pts.len() >= line_pts.len(),
            "Circle ({} pts) should need at least as many points as line ({} pts)",
            circle_pts.len(),
            line_pts.len()
        );
    }

    #[test]
    fn test_basis_function_zero_outside_support() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        // Basis function 0 (degree 2) has support on [0, 1]
        let val = basis_function(&knots, 0, 2, 2.5);
        assert!(val.abs() < 1e-6, "Basis 0 at t=2.5 should be ~0, got {}", val);
    }

    #[test]
    fn test_cubic_curve_smoothness() {
        // Cubic NURBS through 5 control points — ensure evaluation is smooth
        let curve = NurbsCurve {
            degree: 3,
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [5.0, 2.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            weights: vec![1.0; 5],
            knots: vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        };

        let mut prev = curve.evaluate(0.0);
        let steps = 100;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let cur = curve.evaluate(t);
            let d = dist_3d(prev, cur);
            // Adjacent samples should be close (no jumps)
            assert!(d < 0.5, "Jump of {} between t={} and t={}", d, (i - 1) as f32 / steps as f32, t);
            prev = cur;
        }
    }
}
