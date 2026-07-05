// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NURBS curves and surfaces
//!
//! Non-Uniform Rational B-Spline (NURBS) geometry representation with
//! evaluation via Cox-de Boor recursion and De Boor's algorithm,
//! tessellation to triangle meshes, and convenience constructors.

use crate::mesh::TriangleMesh;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A NURBS curve in 3-space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NurbsCurve {
    pub degree: u32,
    pub control_points: Vec<[f32; 3]>,
    pub weights: Vec<f32>,
    pub knots: Vec<f32>,
}

/// A NURBS surface in 3-space (tensor-product patch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NurbsSurface {
    pub degree_u: u32,
    pub degree_v: u32,
    pub control_points: Vec<Vec<[f32; 3]>>,
    pub weights: Vec<Vec<f32>>,
    pub knots_u: Vec<f32>,
    pub knots_v: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Cox-de Boor recursion for B-spline basis function N_{i,p}(t).
///
/// Returns the value of the i-th basis function of the given `degree`
/// evaluated at parameter `t` over the supplied `knots` vector.
pub fn basis_function(knots: &[f32], i: usize, degree: u32, t: f32) -> f32 {
    if degree == 0 {
        // Base case: characteristic function of [knots[i], knots[i+1])
        if i + 1 >= knots.len() {
            return 0.0;
        }
        let t_i = knots[i];
        let t_i1 = knots[i + 1];
        // Degenerate (zero-length) span
        if (t_i1 - t_i).abs() < 1e-10 {
            return 0.0;
        }
        // Standard half-open interval, but include the right endpoint
        // if this is the last non-degenerate span (clamped knot vector convention).
        let is_last_nondegenerate = {
            let mut last = true;
            for k in (i + 1)..knots.len().saturating_sub(1) {
                if (knots[k + 1] - knots[k]).abs() > 1e-10 {
                    last = false;
                    break;
                }
            }
            last
        };
        if is_last_nondegenerate {
            return if t >= t_i && t <= t_i1 { 1.0 } else { 0.0 };
        }
        return if t >= t_i && t < t_i1 { 1.0 } else { 0.0 };
    }

    let mut result = 0.0;

    // Left term:  (t - t_i) / (t_{i+p} - t_i)  *  N_{i, p-1}(t)
    let denom_left = knots.get(i + degree as usize).copied().unwrap_or(0.0)
        - knots.get(i).copied().unwrap_or(0.0);
    if denom_left.abs() > 1e-10 {
        let left = (t - knots[i]) / denom_left;
        result += left * basis_function(knots, i, degree - 1, t);
    }

    // Right term: (t_{i+p+1} - t) / (t_{i+p+1} - t_{i+1})  *  N_{i+1, p-1}(t)
    let denom_right = knots.get(i + degree as usize + 1).copied().unwrap_or(0.0)
        - knots.get(i + 1).copied().unwrap_or(0.0);
    if denom_right.abs() > 1e-10
        && let Some(&k) = knots.get(i + degree as usize + 1)
    {
        let right = (k - t) / denom_right;
        result += right * basis_function(knots, i + 1, degree - 1, t);
    }

    result
}

// ---------------------------------------------------------------------------
// NurbsCurve
// ---------------------------------------------------------------------------

impl NurbsCurve {
    /// Find the knot span index such that `knots[span] <= t < knots[span+1]`.
    ///
    /// Clamps to the valid range `[degree, n]` where `n = control_points.len() - 1`.
    pub fn find_knot_span(&self, t: f32) -> usize {
        let n = self.control_points.len();
        if n == 0 {
            return 0;
        }
        let p = self.degree as usize;
        let last = n; // n = number of control points, last valid span index

        // Malformed/short knot vector (e.g. parsed from an untrusted CAD
        // file with a truncated knot group) -- there's no valid span to
        // report; return a safe in-bounds default rather than indexing out
        // of range below.
        if self.knots.len() <= last {
            return last.saturating_sub(1);
        }

        // Clamp to domain
        if t >= self.knots[last] {
            return last - 1;
        }
        if p < self.knots.len() && t <= self.knots[p] {
            return p;
        }

        // Linear search (sufficient for typical knot vectors)
        for i in p..last {
            if t >= self.knots[i] && t < self.knots[i + 1] {
                return i;
            }
        }
        last - 1
    }

    /// Evaluate the NURBS curve at parameter `t` using rational basis functions (De Boor).
    ///
    /// The parameter `t` should lie within the knot domain.
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let n = self.control_points.len();
        if n == 0 {
            return [0.0; 3];
        }

        let mut point = [0.0f32; 3];
        let mut w_sum = 0.0f32;

        for i in 0..n {
            let basis = basis_function(&self.knots, i, self.degree, t);
            let wi = self.weights.get(i).copied().unwrap_or(1.0);
            let bw = basis * wi;
            point[0] += bw * self.control_points[i][0];
            point[1] += bw * self.control_points[i][1];
            point[2] += bw * self.control_points[i][2];
            w_sum += bw;
        }

        if w_sum.abs() > 1e-10 {
            point[0] /= w_sum;
            point[1] /= w_sum;
            point[2] /= w_sum;
        }

        point
    }

    /// Uniformly sample the curve at `num_points` positions along the knot domain.
    pub fn tessellate(&self, num_points: usize) -> Vec<[f32; 3]> {
        if num_points == 0 || self.control_points.is_empty() {
            return Vec::new();
        }

        // Same defensive `.get()`/fallback pattern as `NurbsSurface::tessellate`
        // -- a malformed/short knot vector (e.g. from an untrusted CAD file)
        // must not panic here.
        let t_start = self.knots.get(self.degree as usize).copied().unwrap_or(0.0);
        let t_end = self
            .knots
            .get(self.control_points.len())
            .copied()
            .unwrap_or_else(|| self.knots.last().copied().unwrap_or(1.0));

        if num_points == 1 {
            return vec![self.evaluate(t_start)];
        }

        (0..num_points)
            .map(|i| {
                let frac = i as f32 / (num_points - 1) as f32;
                let t = t_start + frac * (t_end - t_start);
                self.evaluate(t)
            })
            .collect()
    }

    /// Convenience constructor for a degree-1 line from `start` to `end`.
    pub fn line(start: [f32; 3], end: [f32; 3]) -> Self {
        Self {
            degree: 1,
            control_points: vec![start, end],
            weights: vec![1.0, 1.0],
            knots: vec![0.0, 0.0, 1.0, 1.0],
        }
    }

    /// Estimate the arc length by summing chord lengths over `segments` uniform samples.
    pub fn length_estimate(&self, segments: usize) -> f32 {
        if segments == 0 || self.control_points.is_empty() {
            return 0.0;
        }
        let pts = self.tessellate(segments + 1);
        let mut length = 0.0f32;
        for i in 1..pts.len() {
            let dx = pts[i][0] - pts[i - 1][0];
            let dy = pts[i][1] - pts[i - 1][1];
            let dz = pts[i][2] - pts[i - 1][2];
            length += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        length
    }
}

// ---------------------------------------------------------------------------
// NurbsSurface
// ---------------------------------------------------------------------------

impl NurbsSurface {
    /// Evaluate the NURBS surface at parameters `(u, v)`.
    pub fn evaluate(&self, u: f32, v: f32) -> [f32; 3] {
        let rows = self.control_points.len();
        if rows == 0 {
            return [0.0; 3];
        }
        let cols = self.control_points[0].len();

        let mut point = [0.0f32; 3];
        let mut w_sum = 0.0f32;

        for i in 0..rows {
            let basis_u = basis_function(&self.knots_u, i, self.degree_u, u);
            for j in 0..cols {
                let basis_v = basis_function(&self.knots_v, j, self.degree_v, v);
                let wi = self
                    .weights
                    .get(i)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(1.0);
                let bw = basis_u * basis_v * wi;
                point[0] += bw * self.control_points[i][j][0];
                point[1] += bw * self.control_points[i][j][1];
                point[2] += bw * self.control_points[i][j][2];
                w_sum += bw;
            }
        }

        if w_sum.abs() > 1e-10 {
            point[0] /= w_sum;
            point[1] /= w_sum;
            point[2] /= w_sum;
        }

        point
    }

    /// Tessellate the surface into a [`TriangleMesh`] by evaluating on a
    /// `u_steps × v_steps` grid and connecting adjacent samples into triangle pairs.
    pub fn tessellate(&self, u_steps: usize, v_steps: usize) -> TriangleMesh {
        if u_steps < 2 || v_steps < 2 || self.control_points.is_empty() {
            return TriangleMesh::empty();
        }

        let rows = self.control_points.len();
        let cols = self.control_points[0].len();
        let du = self.degree_u as usize;
        let dv = self.degree_v as usize;

        let u_start = self.knots_u.get(du).copied().unwrap_or(0.0);
        let u_end = self
            .knots_u
            .get(rows)
            .copied()
            .unwrap_or_else(|| self.knots_u.last().copied().unwrap_or(1.0));
        let v_start = self.knots_v.get(dv).copied().unwrap_or(0.0);
        let v_end = self
            .knots_v
            .get(cols)
            .copied()
            .unwrap_or_else(|| self.knots_v.last().copied().unwrap_or(1.0));

        // Evaluate grid
        let mut vertices = Vec::with_capacity(u_steps * v_steps);
        for ui in 0..u_steps {
            let u = u_start + (ui as f32 / (u_steps - 1) as f32) * (u_end - u_start);
            for vi in 0..v_steps {
                let v = v_start + (vi as f32 / (v_steps - 1) as f32) * (v_end - v_start);
                vertices.push(self.evaluate(u, v));
            }
        }

        // Compute per-vertex normals via central differences (approximate)
        let mut normals = Vec::with_capacity(vertices.len());
        for ui in 0..u_steps {
            for vi in 0..v_steps {
                let idx = ui * v_steps + vi;
                // du tangent
                let du = if ui + 1 < u_steps {
                    let next = (ui + 1) * v_steps + vi;
                    [
                        vertices[next][0] - vertices[idx][0],
                        vertices[next][1] - vertices[idx][1],
                        vertices[next][2] - vertices[idx][2],
                    ]
                } else if ui > 0 {
                    let prev = (ui - 1) * v_steps + vi;
                    [
                        vertices[idx][0] - vertices[prev][0],
                        vertices[idx][1] - vertices[prev][1],
                        vertices[idx][2] - vertices[prev][2],
                    ]
                } else {
                    [1.0, 0.0, 0.0]
                };
                // dv tangent
                let dv = if vi + 1 < v_steps {
                    let next = idx + 1;
                    [
                        vertices[next][0] - vertices[idx][0],
                        vertices[next][1] - vertices[idx][1],
                        vertices[next][2] - vertices[idx][2],
                    ]
                } else if vi > 0 {
                    let prev = idx - 1;
                    [
                        vertices[idx][0] - vertices[prev][0],
                        vertices[idx][1] - vertices[prev][1],
                        vertices[idx][2] - vertices[prev][2],
                    ]
                } else {
                    [0.0, 1.0, 0.0]
                };
                // Cross product
                let mut n = [
                    du[1] * dv[2] - du[2] * dv[1],
                    du[2] * dv[0] - du[0] * dv[2],
                    du[0] * dv[1] - du[1] * dv[0],
                ];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-10 {
                    n[0] /= len;
                    n[1] /= len;
                    n[2] /= len;
                } else {
                    n = [0.0, 0.0, 1.0];
                }
                normals.push(n);
            }
        }

        // Build triangles
        let mut indices = Vec::with_capacity((u_steps - 1) * (v_steps - 1) * 2);
        for ui in 0..(u_steps - 1) {
            for vi in 0..(v_steps - 1) {
                let a = (ui * v_steps + vi) as u32;
                let b = (ui * v_steps + vi + 1) as u32;
                let c = ((ui + 1) * v_steps + vi) as u32;
                let d = ((ui + 1) * v_steps + vi + 1) as u32;
                indices.push([a, c, b]);
                indices.push([b, c, d]);
            }
        }

        TriangleMesh {
            vertices,
            normals,
            indices,
        }
    }

    /// Convenience constructor for a bilinear (degree 1×1) surface from four corners.
    ///
    /// Corner ordering: `[bottom-left, bottom-right, top-left, top-right]`.
    pub fn bilinear(corners: [[f32; 3]; 4]) -> Self {
        Self {
            degree_u: 1,
            degree_v: 1,
            control_points: vec![vec![corners[0], corners[1]], vec![corners[2], corners[3]]],
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

    const EPS: f32 = 1e-4;

    fn approx_eq(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    #[test]
    fn line_evaluation() {
        let line = NurbsCurve::line([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let p0 = line.evaluate(0.0);
        let p1 = line.evaluate(1.0);
        let pm = line.evaluate(0.5);
        assert!(approx_eq(p0, [0.0, 0.0, 0.0], EPS));
        assert!(approx_eq(p1, [10.0, 0.0, 0.0], EPS));
        assert!(approx_eq(pm, [5.0, 0.0, 0.0], EPS));
    }

    #[test]
    fn circle_approx_quarter() {
        // Degree-2 rational quarter circle in XY plane, radius 1
        let w = (2.0f32).sqrt() / 2.0;
        let curve = NurbsCurve {
            degree: 2,
            control_points: vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            weights: vec![1.0, w, 1.0],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        };
        let start = curve.evaluate(0.0);
        let end = curve.evaluate(1.0);
        let mid = curve.evaluate(0.5);
        assert!(approx_eq(start, [1.0, 0.0, 0.0], EPS));
        assert!(approx_eq(end, [0.0, 1.0, 0.0], EPS));
        // Midpoint of quarter circle should be at (cos45, sin45, 0)
        let expected_mid = [(2.0f32).sqrt() / 2.0, (2.0f32).sqrt() / 2.0, 0.0];
        assert!(approx_eq(mid, expected_mid, 0.01));
        // Radius at midpoint should be ~1.0
        let r = (mid[0] * mid[0] + mid[1] * mid[1]).sqrt();
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn surface_evaluation() {
        let surf = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let center = surf.evaluate(0.5, 0.5);
        assert!(approx_eq(center, [0.5, 0.5, 0.0], EPS));
    }

    #[test]
    fn bilinear_corners() {
        let corners = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
        ];
        let surf = NurbsSurface::bilinear(corners);
        assert!(approx_eq(surf.evaluate(0.0, 0.0), corners[0], EPS));
        assert!(approx_eq(surf.evaluate(0.0, 1.0), corners[1], EPS));
        assert!(approx_eq(surf.evaluate(1.0, 0.0), corners[2], EPS));
        assert!(approx_eq(surf.evaluate(1.0, 1.0), corners[3], EPS));
    }

    #[test]
    fn tessellation_triangle_count() {
        let surf = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let mesh = surf.tessellate(5, 5);
        // (5-1)*(5-1)*2 = 32 triangles
        assert_eq!(mesh.triangle_count(), 32);
        assert_eq!(mesh.vertices.len(), 25);
        assert_eq!(mesh.normals.len(), 25);
    }

    #[test]
    fn basis_partition_of_unity() {
        // For any t in the domain, the sum of all basis functions should be 1
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let degree = 2u32;
        let n_basis = knots.len() - degree as usize - 1; // 4
        for step in 0..=10 {
            let t = step as f32 / 10.0;
            let sum: f32 = (0..n_basis)
                .map(|i| basis_function(&knots, i, degree, t))
                .sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Partition of unity failed at t={}: sum={}",
                t,
                sum
            );
        }
    }

    #[test]
    fn knot_span_finding() {
        let curve = NurbsCurve {
            degree: 2,
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            weights: vec![1.0; 4],
            knots: vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        };
        assert_eq!(curve.find_knot_span(0.0), 2);
        assert_eq!(curve.find_knot_span(0.25), 2);
        assert_eq!(curve.find_knot_span(0.5), 3);
        assert_eq!(curve.find_knot_span(0.75), 3);
        // At the end, should clamp
        assert_eq!(curve.find_knot_span(1.0), 3);
    }

    #[test]
    fn length_positive() {
        let line = NurbsCurve::line([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        let len = line.length_estimate(100);
        assert!((len - 5.0).abs() < 0.01, "Expected ~5.0, got {}", len);
    }

    #[test]
    fn empty_curve() {
        let curve = NurbsCurve {
            degree: 1,
            control_points: vec![],
            weights: vec![],
            knots: vec![],
        };
        let p = curve.evaluate(0.5);
        assert_eq!(p, [0.0, 0.0, 0.0]);
        let pts = curve.tessellate(10);
        assert!(pts.is_empty());
        assert_eq!(curve.length_estimate(10), 0.0);
    }

    #[test]
    fn serde_roundtrip() {
        let curve = NurbsCurve::line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let json = serde_json::to_string(&curve).unwrap();
        let restored: NurbsCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.degree, curve.degree);
        assert_eq!(restored.control_points.len(), curve.control_points.len());

        let surf = NurbsSurface::bilinear([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let json2 = serde_json::to_string(&surf).unwrap();
        let restored2: NurbsSurface = serde_json::from_str(&json2).unwrap();
        assert_eq!(restored2.degree_u, surf.degree_u);
    }

    #[test]
    fn tessellate_curve_uniform() {
        let line = NurbsCurve::line([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let pts = line.tessellate(11);
        assert_eq!(pts.len(), 11);
        for (i, p) in pts.iter().enumerate() {
            let expected_x = i as f32;
            assert!(
                (p[0] - expected_x).abs() < EPS,
                "Point {}: expected x={}, got {}",
                i,
                expected_x,
                p[0]
            );
        }
    }

    /// Regression test: a curve with non-empty control points but an empty
    /// knot vector (e.g. `step_import.rs` parsing a STEP file with a
    /// truncated/malformed knot group via
    /// `knots: knots.last().cloned().unwrap_or_default()`) must not panic
    /// when tessellated or when `find_knot_span` is queried.
    #[test]
    fn tessellate_with_empty_knots_does_not_panic() {
        let curve = NurbsCurve {
            degree: 3,
            control_points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            weights: vec![1.0, 1.0, 1.0],
            knots: vec![],
        };
        let pts = curve.tessellate(5);
        assert_eq!(pts.len(), 5);
        let span = curve.find_knot_span(0.5);
        assert!(span < curve.control_points.len());
    }

    /// Same as above, but with a knot vector that's non-empty yet shorter
    /// than `control_points.len()` (a partially-truncated parse, rather
    /// than a fully empty one).
    #[test]
    fn tessellate_with_short_knots_does_not_panic() {
        let curve = NurbsCurve {
            degree: 2,
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            weights: vec![1.0; 4],
            knots: vec![0.0, 0.0, 0.0], // too short for 4 control points
        };
        let pts = curve.tessellate(5);
        assert_eq!(pts.len(), 5);
        let span = curve.find_knot_span(0.5);
        assert!(span < curve.control_points.len());
    }
}
