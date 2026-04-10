// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core positioning algorithm: weighted nonlinear least-squares trilateration.
//!
//! Given N anchors at known positions and N range measurements with uncertainties,
//! compute the most likely position and its 3×3 covariance matrix.
//!
//! Method: Iterative Gauss-Newton with trust-weighted residuals.
//! Follows the solver pattern from orbital-mechanics orbit_determination.rs.

use serde::{Deserialize, Serialize};

/// Maximum iterations for Gauss-Newton convergence.
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold in meters.
const CONVERGENCE_M: f64 = 1e-6;

/// A 3D position estimate with uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEstimate {
    /// Estimated position (body-fixed Cartesian, meters).
    pub position: [f64; 3],
    /// 3×3 covariance matrix (row-major, meters²).
    pub covariance: [f64; 9],
    /// 1-sigma position uncertainty (RSS of diagonal), meters.
    pub sigma_m: f64,
    /// Number of anchors used.
    pub anchor_count: usize,
    /// Number of iterations to converge.
    pub iterations: usize,
    /// Post-fit residual RMS (meters).
    pub residual_rms: f64,
}

/// A 2D position estimate (surface positioning).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEstimate2D {
    pub position: [f64; 2],
    pub covariance: [f64; 4],
    pub sigma_m: f64,
    pub anchor_count: usize,
    pub iterations: usize,
    pub residual_rms: f64,
}

/// Errors during trilateration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrilaterationError {
    /// Need at least 3 anchors for 2D, 4 for 3D.
    InsufficientAnchors { have: usize, need: usize },
    /// Input arrays have mismatched lengths.
    LengthMismatch,
    /// Anchors are nearly collinear (singular Jacobian).
    DegenerateGeometry,
    /// Failed to converge within MAX_ITERATIONS.
    DidNotConverge { iterations: usize, residual: f64 },
    /// Negative or zero range measurement.
    InvalidRange { index: usize, value: f64 },
}

impl std::fmt::Display for TrilaterationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientAnchors { have, need } => {
                write!(f, "Need {} anchors, have {}", need, have)
            }
            Self::LengthMismatch => write!(f, "Input arrays have different lengths"),
            Self::DegenerateGeometry => write!(f, "Anchors are nearly collinear"),
            Self::DidNotConverge {
                iterations,
                residual,
            } => write!(
                f,
                "Did not converge after {} iterations (residual: {:.3}m)",
                iterations, residual
            ),
            Self::InvalidRange { index, value } => {
                write!(f, "Invalid range at index {}: {}", index, value)
            }
        }
    }
}

/// Trilaterate a 3D position from range measurements to known anchors.
///
/// # Arguments
/// - `anchors`: Known positions in body-fixed Cartesian (meters), shape (N, 3)
/// - `ranges`: Measured distances to each anchor (meters)
/// - `sigmas`: Measurement uncertainties (1-sigma, meters)
/// - `trust_weights`: Per-anchor trust weights (0.0-1.0). Applied as: effective_sigma = sigma / sqrt(trust)
///
/// # Returns
/// `PositionEstimate` with position, 3×3 covariance, and diagnostics.
pub fn trilaterate_3d(
    anchors: &[[f64; 3]],
    ranges: &[f64],
    sigmas: &[f64],
    trust_weights: &[f64],
) -> Result<PositionEstimate, TrilaterationError> {
    let n = anchors.len();
    if n < 4 {
        return Err(TrilaterationError::InsufficientAnchors { have: n, need: 4 });
    }
    if ranges.len() != n || sigmas.len() != n || trust_weights.len() != n {
        return Err(TrilaterationError::LengthMismatch);
    }
    for (i, &r) in ranges.iter().enumerate() {
        if r <= 0.0 {
            return Err(TrilaterationError::InvalidRange { index: i, value: r });
        }
    }

    // Effective weights: w_i = trust_i / sigma_i²
    let weights: Vec<f64> = (0..n)
        .map(|i| trust_weights[i].max(0.01) / (sigmas[i] * sigmas[i]))
        .collect();

    // Initial guess: weighted centroid of anchors
    let total_w: f64 = weights.iter().sum();
    let mut pos = [0.0_f64; 3];
    for i in 0..n {
        for j in 0..3 {
            pos[j] += weights[i] * anchors[i][j];
        }
    }
    for j in 0..3 {
        pos[j] /= total_w;
    }

    // Gauss-Newton iteration
    let mut iterations = 0;
    let mut residual_rms = f64::MAX;

    for iter in 0..MAX_ITERATIONS {
        iterations = iter + 1;

        // Compute residuals and Jacobian
        let mut jtw_j = [[0.0_f64; 3]; 3]; // J^T W J (3×3)
        let mut jtw_r = [0.0_f64; 3]; // J^T W r (3×1)
        let mut sum_sq_residual = 0.0;

        for i in 0..n {
            let dx = pos[0] - anchors[i][0];
            let dy = pos[1] - anchors[i][1];
            let dz = pos[2] - anchors[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < 1e-10 {
                continue; // Skip if on top of anchor
            }

            let residual = dist - ranges[i];
            sum_sq_residual += weights[i] * residual * residual;

            // Jacobian row: d(dist)/d(pos) = [dx/dist, dy/dist, dz/dist]
            let j_row = [dx / dist, dy / dist, dz / dist];

            // Accumulate J^T W J and J^T W r
            for a in 0..3 {
                jtw_r[a] += weights[i] * j_row[a] * residual;
                for b in 0..3 {
                    jtw_j[a][b] += weights[i] * j_row[a] * j_row[b];
                }
            }
        }

        residual_rms = (sum_sq_residual / n as f64).sqrt();

        // Solve 3×3 system: jtw_j * delta = jtw_r
        let delta = match solve_3x3(&jtw_j, &jtw_r) {
            Some(d) => d,
            None => return Err(TrilaterationError::DegenerateGeometry),
        };

        // Update position
        pos[0] -= delta[0];
        pos[1] -= delta[1];
        pos[2] -= delta[2];

        // Check convergence
        let step = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if step < CONVERGENCE_M {
            break;
        }
    }

    if iterations == MAX_ITERATIONS && residual_rms > 1.0 {
        return Err(TrilaterationError::DidNotConverge {
            iterations,
            residual: residual_rms,
        });
    }

    // Compute covariance: C = (J^T W J)^{-1}
    let mut jtw_j_final = [[0.0_f64; 3]; 3];
    for i in 0..n {
        let dx = pos[0] - anchors[i][0];
        let dy = pos[1] - anchors[i][1];
        let dz = pos[2] - anchors[i][2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-10 {
            continue;
        }
        let j_row = [dx / dist, dy / dist, dz / dist];
        for a in 0..3 {
            for b in 0..3 {
                jtw_j_final[a][b] += weights[i] * j_row[a] * j_row[b];
            }
        }
    }

    let covariance = invert_3x3(&jtw_j_final).unwrap_or([[1e6; 3]; 3]);
    let sigma = (covariance[0][0] + covariance[1][1] + covariance[2][2]).sqrt();

    Ok(PositionEstimate {
        position: pos,
        covariance: [
            covariance[0][0],
            covariance[0][1],
            covariance[0][2],
            covariance[1][0],
            covariance[1][1],
            covariance[1][2],
            covariance[2][0],
            covariance[2][1],
            covariance[2][2],
        ],
        sigma_m: sigma,
        anchor_count: n,
        iterations,
        residual_rms,
    })
}

/// Trilaterate a 2D position (surface positioning, known altitude).
pub fn trilaterate_2d(
    anchors: &[[f64; 2]],
    ranges: &[f64],
    sigmas: &[f64],
    trust_weights: &[f64],
) -> Result<PositionEstimate2D, TrilaterationError> {
    let n = anchors.len();
    if n < 3 {
        return Err(TrilaterationError::InsufficientAnchors { have: n, need: 3 });
    }
    if ranges.len() != n || sigmas.len() != n || trust_weights.len() != n {
        return Err(TrilaterationError::LengthMismatch);
    }
    for (i, &r) in ranges.iter().enumerate() {
        if r <= 0.0 {
            return Err(TrilaterationError::InvalidRange { index: i, value: r });
        }
    }

    let weights: Vec<f64> = (0..n)
        .map(|i| trust_weights[i].max(0.01) / (sigmas[i] * sigmas[i]))
        .collect();

    let total_w: f64 = weights.iter().sum();
    let mut pos = [0.0_f64; 2];
    for i in 0..n {
        pos[0] += weights[i] * anchors[i][0];
        pos[1] += weights[i] * anchors[i][1];
    }
    pos[0] /= total_w;
    pos[1] /= total_w;

    let mut iterations = 0;
    let mut residual_rms = f64::MAX;

    for iter in 0..MAX_ITERATIONS {
        iterations = iter + 1;
        let mut jtw_j = [[0.0_f64; 2]; 2];
        let mut jtw_r = [0.0_f64; 2];
        let mut sum_sq = 0.0;

        for i in 0..n {
            let dx = pos[0] - anchors[i][0];
            let dy = pos[1] - anchors[i][1];
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 1e-10 {
                continue;
            }

            let residual = dist - ranges[i];
            sum_sq += weights[i] * residual * residual;
            let j_row = [dx / dist, dy / dist];

            for a in 0..2 {
                jtw_r[a] += weights[i] * j_row[a] * residual;
                for b in 0..2 {
                    jtw_j[a][b] += weights[i] * j_row[a] * j_row[b];
                }
            }
        }

        residual_rms = (sum_sq / n as f64).sqrt();

        let det = jtw_j[0][0] * jtw_j[1][1] - jtw_j[0][1] * jtw_j[1][0];
        if det.abs() < 1e-20 {
            return Err(TrilaterationError::DegenerateGeometry);
        }
        let delta = [
            (jtw_j[1][1] * jtw_r[0] - jtw_j[0][1] * jtw_r[1]) / det,
            (jtw_j[0][0] * jtw_r[1] - jtw_j[1][0] * jtw_r[0]) / det,
        ];

        pos[0] -= delta[0];
        pos[1] -= delta[1];

        if (delta[0] * delta[0] + delta[1] * delta[1]).sqrt() < CONVERGENCE_M {
            break;
        }
    }

    // Final covariance
    let mut jtw_j_final = [[0.0_f64; 2]; 2];
    for i in 0..n {
        let dx = pos[0] - anchors[i][0];
        let dy = pos[1] - anchors[i][1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-10 {
            continue;
        }
        let j_row = [dx / dist, dy / dist];
        for a in 0..2 {
            for b in 0..2 {
                jtw_j_final[a][b] += weights[i] * j_row[a] * j_row[b];
            }
        }
    }
    let det = jtw_j_final[0][0] * jtw_j_final[1][1] - jtw_j_final[0][1] * jtw_j_final[1][0];
    let cov = if det.abs() > 1e-20 {
        [
            jtw_j_final[1][1] / det,
            -jtw_j_final[0][1] / det,
            -jtw_j_final[1][0] / det,
            jtw_j_final[0][0] / det,
        ]
    } else {
        [1e6, 0.0, 0.0, 1e6]
    };
    let sigma = (cov[0] + cov[3]).sqrt();

    Ok(PositionEstimate2D {
        position: pos,
        covariance: cov,
        sigma_m: sigma,
        anchor_count: n,
        iterations,
        residual_rms,
    })
}

// ============================================================================
// LINEAR ALGEBRA HELPERS (3×3)
// ============================================================================

fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let inv = invert_3x3(a)?;
    Some([
        inv[0][0] * b[0] + inv[0][1] * b[1] + inv[0][2] * b[2],
        inv[1][0] * b[0] + inv[1][1] * b[1] + inv[1][2] * b[2],
        inv[2][0] * b[0] + inv[2][1] * b[1] + inv[2][2] * b[2],
    ])
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilaterate_3d_basic() {
        // 4 anchors at corners of a cube, target at origin
        let anchors = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [-100.0, 0.0, 0.0],
        ];
        let ranges = [100.0, 100.0, 100.0, 100.0];
        let sigmas = [1.0, 1.0, 1.0, 1.0];
        let trusts = [1.0, 1.0, 1.0, 1.0];

        let est = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts).unwrap();
        assert!(est.position[0].abs() < 1.0, "x={}", est.position[0]);
        assert!(est.position[1].abs() < 1.0, "y={}", est.position[1]);
        assert!(est.position[2].abs() < 1.0, "z={}", est.position[2]);
        assert!(est.iterations <= 10);
    }

    #[test]
    fn trilaterate_3d_off_center() {
        // Target at (50, 30, -20)
        let target: [f64; 3] = [50.0, 30.0, -20.0];
        let anchors: [[f64; 3]; 5] = [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [100.0, 100.0, 0.0],
        ];
        let ranges: Vec<f64> = anchors
            .iter()
            .map(|a| {
                let dx: f64 = a[0] - target[0];
                let dy: f64 = a[1] - target[1];
                let dz: f64 = a[2] - target[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        let sigmas = vec![1.0; 5];
        let trusts = vec![1.0; 5];

        let est = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts).unwrap();
        assert!((est.position[0] - 50.0).abs() < 0.1);
        assert!((est.position[1] - 30.0).abs() < 0.1);
        assert!((est.position[2] - (-20.0)).abs() < 0.1);
    }

    #[test]
    fn trilaterate_3d_trust_weighting() {
        // One anchor has wrong range but low trust — should be downweighted
        let anchors = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [-100.0, 0.0, 0.0],
            [50.0, 50.0, 50.0], // Bad anchor
        ];
        let ranges = vec![100.0, 100.0, 100.0, 100.0, 200.0]; // Last one is wrong
        let sigmas = vec![1.0; 5];
        let trusts = vec![1.0, 1.0, 1.0, 1.0, 0.01]; // Low trust on bad anchor

        let est = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts).unwrap();
        // Should still be near origin despite bad measurement
        let error =
            (est.position[0].powi(2) + est.position[1].powi(2) + est.position[2].powi(2)).sqrt();
        assert!(
            error < 10.0,
            "Error {} too large with low-trust bad anchor",
            error
        );

        // With high trust on bad anchor, error should be larger
        let trusts_high = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let est_high = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts_high).unwrap();
        let error_high = (est_high.position[0].powi(2)
            + est_high.position[1].powi(2)
            + est_high.position[2].powi(2))
        .sqrt();
        assert!(
            error_high > error,
            "High trust on bad anchor should give worse result"
        );
    }

    #[test]
    fn trilaterate_2d_basic() {
        let anchors: [[f64; 2]; 3] = [[0.0, 0.0], [100.0, 0.0], [50.0, 86.6]]; // equilateral triangle
        let target: [f64; 2] = [40.0, 30.0];
        let ranges: Vec<f64> = anchors
            .iter()
            .map(|a| {
                let dx: f64 = a[0] - target[0];
                let dy: f64 = a[1] - target[1];
                (dx.powi(2) + dy.powi(2)).sqrt()
            })
            .collect();
        let sigmas = vec![1.0; 3];
        let trusts = vec![1.0; 3];

        let est = trilaterate_2d(&anchors, &ranges, &sigmas, &trusts).unwrap();
        assert!((est.position[0] - 40.0).abs() < 0.1);
        assert!((est.position[1] - 30.0).abs() < 0.1);
    }

    #[test]
    fn insufficient_anchors_3d() {
        let result = trilaterate_3d(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
        );
        assert!(matches!(
            result,
            Err(TrilaterationError::InsufficientAnchors { .. })
        ));
    }

    #[test]
    fn insufficient_anchors_2d() {
        let result = trilaterate_2d(
            &[[0.0, 0.0], [1.0, 0.0]],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
        );
        assert!(matches!(
            result,
            Err(TrilaterationError::InsufficientAnchors { .. })
        ));
    }

    #[test]
    fn negative_range_rejected() {
        let result = trilaterate_3d(
            &[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[1.0, -1.0, 1.0, 1.0],
            &[1.0; 4],
            &[1.0; 4],
        );
        assert!(matches!(
            result,
            Err(TrilaterationError::InvalidRange { index: 1, .. })
        ));
    }

    #[test]
    fn covariance_is_positive() {
        let anchors = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [-100.0, 0.0, 0.0],
        ];
        let ranges = [100.0, 100.0, 100.0, 100.0];
        let est = trilaterate_3d(&anchors, &ranges, &[1.0; 4], &[1.0; 4]).unwrap();
        // Diagonal elements should be positive
        assert!(est.covariance[0] > 0.0); // C[0][0]
        assert!(est.covariance[4] > 0.0); // C[1][1]
        assert!(est.covariance[8] > 0.0); // C[2][2]
    }

    #[test]
    fn invert_3x3_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&id).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }
}
