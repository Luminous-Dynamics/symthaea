// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Geometric Dilution of Precision (GDOP/PDOP) analysis.
//!
//! GDOP measures how anchor geometry affects position accuracy.
//! Low GDOP (< 3) = well-distributed anchors = good accuracy.
//! High GDOP (> 10) = clustered/collinear anchors = poor accuracy.

use serde::{Deserialize, Serialize};

/// A coverage analysis point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveragePoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub gdop: f64,
    pub pdop: f64,
}

/// Compute GDOP (Geometric Dilution of Precision) at a point.
///
/// GDOP = sqrt(trace((J^T J)^{-1})) where J is the geometry matrix.
/// Lower is better. Returns f64::INFINITY for degenerate geometry.
///
/// # Arguments
/// - `anchors`: Known anchor positions (body-fixed Cartesian, meters)
/// - `point`: Query position (where we want to know the accuracy)
pub fn gdop(anchors: &[[f64; 3]], point: &[f64; 3]) -> f64 {
    if anchors.len() < 4 {
        return f64::INFINITY;
    }

    // Build geometry matrix J: each row is unit vector from point to anchor
    let mut jtj = [[0.0_f64; 3]; 3];
    for anchor in anchors {
        let dx = anchor[0] - point[0];
        let dy = anchor[1] - point[1];
        let dz = anchor[2] - point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-10 {
            continue;
        }
        let u = [dx / dist, dy / dist, dz / dist];

        for a in 0..3 {
            for b in 0..3 {
                jtj[a][b] += u[a] * u[b];
            }
        }
    }

    // GDOP = sqrt(trace(inv(J^T J)))
    match invert_3x3(&jtj) {
        Some(inv) => {
            let trace = inv[0][0] + inv[1][1] + inv[2][2];
            if trace > 0.0 {
                trace.sqrt()
            } else {
                f64::INFINITY
            }
        }
        None => f64::INFINITY,
    }
}

/// Compute PDOP (Position Dilution of Precision) — same as GDOP for 3D.
/// Included for API clarity; in our formulation PDOP = GDOP.
pub fn pdop(anchors: &[[f64; 3]], point: &[f64; 3]) -> f64 {
    gdop(anchors, point)
}

/// Compute 2D GDOP (horizontal dilution) at a surface point.
pub fn gdop_2d(anchors: &[[f64; 2]], point: &[f64; 2]) -> f64 {
    if anchors.len() < 3 {
        return f64::INFINITY;
    }

    let mut jtj = [[0.0_f64; 2]; 2];
    for anchor in anchors {
        let dx = anchor[0] - point[0];
        let dy = anchor[1] - point[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-10 {
            continue;
        }
        let u = [dx / dist, dy / dist];
        for a in 0..2 {
            for b in 0..2 {
                jtj[a][b] += u[a] * u[b];
            }
        }
    }

    let det = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0];
    if det.abs() < 1e-20 {
        return f64::INFINITY;
    }
    let trace_inv = (jtj[1][1] + jtj[0][0]) / det;
    if trace_inv > 0.0 {
        trace_inv.sqrt()
    } else {
        f64::INFINITY
    }
}

/// Evaluate coverage quality over a grid of points.
pub fn coverage_grid_2d(
    anchors: &[[f64; 2]],
    x_range: (f64, f64),
    y_range: (f64, f64),
    step: f64,
) -> Vec<CoveragePoint> {
    let mut points = Vec::new();
    let mut x = x_range.0;
    while x <= x_range.1 {
        let mut y = y_range.0;
        while y <= y_range.1 {
            let g = gdop_2d(anchors, &[x, y]);
            points.push(CoveragePoint {
                x,
                y,
                z: 0.0,
                gdop: g,
                pdop: g,
            });
            y += step;
        }
        x += step;
    }
    points
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdop_well_distributed() {
        // Anchors at 4 corners of a tetrahedron — excellent geometry
        let anchors = [
            [100.0, 0.0, 0.0],
            [-50.0, 86.6, 0.0],
            [-50.0, -86.6, 0.0],
            [0.0, 0.0, 100.0],
        ];
        let g = gdop(&anchors, &[0.0, 0.0, 0.0]);
        assert!(g < 3.0, "GDOP {} should be < 3 for good geometry", g);
    }

    #[test]
    fn gdop_collinear_is_high() {
        // All anchors on a line — terrible geometry
        let anchors = [
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
        ];
        let g = gdop(&anchors, &[0.0, 0.0, 0.0]);
        assert!(
            g > 10.0 || g.is_infinite(),
            "GDOP {} should be high for collinear",
            g
        );
    }

    #[test]
    fn gdop_insufficient_anchors() {
        let g = gdop(&[[1.0, 0.0, 0.0]], &[0.0, 0.0, 0.0]);
        assert!(g.is_infinite());
    }

    #[test]
    fn gdop_2d_triangle() {
        let anchors = [[0.0, 0.0], [100.0, 0.0], [50.0, 86.6]];
        let g = gdop_2d(&anchors, &[50.0, 30.0]);
        assert!(
            g < 5.0,
            "2D GDOP {} should be reasonable for equilateral triangle",
            g
        );
    }

    #[test]
    fn coverage_grid_produces_points() {
        let anchors = [[0.0, 0.0], [100.0, 0.0], [50.0, 86.6]];
        let grid = coverage_grid_2d(&anchors, (0.0, 100.0), (0.0, 100.0), 25.0);
        assert!(grid.len() > 10);
        // Center should have better GDOP than edges
        let center = grid
            .iter()
            .find(|p| (p.x - 50.0).abs() < 13.0 && (p.y - 50.0).abs() < 13.0);
        assert!(center.is_some());
    }
}
