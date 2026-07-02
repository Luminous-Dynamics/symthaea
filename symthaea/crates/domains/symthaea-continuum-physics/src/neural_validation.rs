// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Validation: testing the I×D theorem against real EEG data.
//!
//! Our prediction: eigenvalue spectra of EEG covariance matrices should
//! be more distributed during wakefulness (high consciousness) than
//! during deep sleep (low consciousness).
//!
//! Uses the Sleep-EDF dataset (PhysioNet) with expert sleep stage annotations.
//!
//! This module doesn't parse EDF directly (that requires std::io which
//! complicates the crate structure). Instead, it provides the analysis
//! functions that operate on pre-extracted signal matrices.

/// Compute the eigenvalue concentration ratio from a covariance matrix.
///
/// Concentration = λ₁ / Σλ — fraction of variance in the first principal component.
/// Low concentration = distributed (more conscious).
/// High concentration = concentrated (less conscious).
pub fn eigenvalue_concentration(covariance_matrix: &[f64], n: usize) -> (f64, Vec<f64>) {
    let eigenvalues = symmetric_eigenvalues(covariance_matrix, n);
    let total: f64 = eigenvalues.iter().sum();
    let concentration = if total > 1e-15 {
        eigenvalues[0] / total // First (largest) eigenvalue / total
    } else {
        1.0
    };
    (concentration, eigenvalues)
}

/// Compute effective dimensionality from eigenvalues.
/// d_eff = (Σλ)² / Σλ² — the "participation ratio."
/// Equals n for uniform distribution, 1 for single dominant component.
pub fn effective_dimensionality(eigenvalues: &[f64]) -> f64 {
    let sum: f64 = eigenvalues.iter().sum();
    let sum_sq: f64 = eigenvalues.iter().map(|e| e * e).sum();
    if sum_sq < 1e-30 {
        return 0.0;
    }
    sum * sum / sum_sq
}

/// Compute covariance matrix from multi-channel signal data.
///
/// `signals[channel * n_samples + sample]` — row-major, channels × samples.
pub fn compute_covariance(signals: &[f64], n_channels: usize, n_samples: usize) -> Vec<f64> {
    // Compute means
    let means: Vec<f64> = (0..n_channels)
        .map(|ch| {
            let offset = ch * n_samples;
            signals[offset..offset + n_samples].iter().sum::<f64>() / n_samples as f64
        })
        .collect();

    // Compute covariance
    let mut cov = vec![0.0; n_channels * n_channels];
    for i in 0..n_channels {
        for j in i..n_channels {
            let mut sum = 0.0;
            for s in 0..n_samples {
                sum += (signals[i * n_samples + s] - means[i])
                    * (signals[j * n_samples + s] - means[j]);
            }
            let c = sum / (n_samples - 1) as f64;
            cov[i * n_channels + j] = c;
            cov[j * n_channels + i] = c;
        }
    }
    cov
}

/// Compute I and D from multi-channel EEG signals.
///
/// I = mean pairwise correlation (integration across channels)
/// D = variance of channel powers (differentiation of channel activity)
pub fn compute_eeg_id(signals: &[f64], n_channels: usize, n_samples: usize) -> (f64, f64) {
    // Integration: mean pairwise correlation
    let mut integration = 0.0;
    let mut n_pairs = 0usize;

    let traces: Vec<&[f64]> = (0..n_channels)
        .map(|ch| &signals[ch * n_samples..(ch + 1) * n_samples])
        .collect();

    for i in 0..n_channels {
        for j in (i + 1)..n_channels {
            integration += pearson_abs(traces[i], traces[j]);
            n_pairs += 1;
        }
    }
    integration /= n_pairs.max(1) as f64;

    // Differentiation: variance of channel powers (RMS voltage)
    let powers: Vec<f64> = traces
        .iter()
        .map(|tr| (tr.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt())
        .collect();
    let mean_power = powers.iter().sum::<f64>() / n_channels as f64;
    let differentiation =
        powers.iter().map(|p| (p - mean_power).powi(2)).sum::<f64>() / n_channels as f64;

    (integration, differentiation)
}

/// Sleep stage classification based on our theory.
///
/// Prediction: Wake has lowest eigenvalue concentration (most distributed),
/// Deep sleep (N3/N4) has highest concentration (most concentrated).
///
/// Returns predicted consciousness level [0, 1] from eigenvalue spectrum.
pub fn consciousness_from_eigenvalue_spectrum(eigenvalues: &[f64]) -> f64 {
    let d_eff = effective_dimensionality(eigenvalues);
    let n = eigenvalues.len() as f64;
    // Normalize: d_eff ranges from 1 (minimal) to n (maximal)
    if n <= 1.0 {
        return 0.0;
    }
    (d_eff - 1.0) / (n - 1.0)
}

/// Symmetric eigendecomposition (Jacobi method, returns sorted descending).
fn symmetric_eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].abs()).collect();
    eigenvalues.sort_by(|a, b| b.total_cmp(a));
    eigenvalues
}

fn pearson_abs(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let (dx, dy) = (xi - mx, yi - my);
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 {
        return 0.0;
    }
    (cov / (vx * vy).sqrt()).abs().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concentration_identity() {
        // Identity covariance: all eigenvalues equal → concentration = 1/n
        let n = 4;
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = 1.0;
        }
        let (conc, _) = eigenvalue_concentration(&cov, n);
        assert!(
            (conc - 0.25).abs() < 0.01,
            "Identity concentration = {:.4}",
            conc
        );
    }

    #[test]
    fn test_concentration_single_dominant() {
        // One dominant eigenvalue → concentration ≈ 1
        let n = 4;
        let mut cov = vec![0.0; n * n];
        cov[0] = 100.0; // Dominant
        for i in 1..n {
            cov[i * n + i] = 0.01;
        }
        let (conc, _) = eigenvalue_concentration(&cov, n);
        assert!(conc > 0.95, "Dominant concentration = {:.4}", conc);
    }

    #[test]
    fn test_effective_dimensionality() {
        // Uniform: d_eff = n
        let evals = vec![1.0; 5];
        let d = effective_dimensionality(&evals);
        assert!((d - 5.0).abs() < 0.01, "Uniform d_eff = {:.2}", d);

        // Single dominant: d_eff ≈ 1
        let evals2 = vec![100.0, 0.01, 0.01, 0.01, 0.01];
        let d2 = effective_dimensionality(&evals2);
        assert!(d2 < 1.5, "Dominant d_eff = {:.2}", d2);
    }

    #[test]
    fn test_covariance_symmetric() {
        let signals = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        let cov = compute_covariance(&signals, 2, 4);
        assert!((cov[0 * 2 + 1] - cov[1 * 2 + 0]).abs() < 1e-14);
    }

    #[test]
    fn test_consciousness_from_spectrum_ranges() {
        // Uniform → high consciousness
        let uniform = vec![1.0; 10];
        let c1 = consciousness_from_eigenvalue_spectrum(&uniform);
        assert!(c1 > 0.9, "Uniform spectrum = high consciousness: {:.4}", c1);

        // Single dominant → low consciousness
        let concentrated = vec![100.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let c2 = consciousness_from_eigenvalue_spectrum(&concentrated);
        assert!(
            c2 < 0.2,
            "Concentrated spectrum = low consciousness: {:.4}",
            c2
        );

        // Prediction: c1 > c2 (distributed > concentrated)
        assert!(c1 > c2);
    }
}
