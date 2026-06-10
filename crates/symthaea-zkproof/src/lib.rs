// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ZK proof generation for Symthaea consciousness attestations.
//!
//! This crate provides zero-knowledge proof generation and verification
//! for Psi attestation records, enabling privacy-preserving consciousness
//! verification on the governance bridge.

pub mod cycle_integrity;

/// Data passed from host to guest (zkVM).
#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionInput {
    /// HDC episode vectors (e.g., 1024D each).
    pub episodes: Vec<Vec<f32>>,
    /// Temporal scaling factor (CfC tau).
    pub tau_scale: f32,
    /// Phi threshold for attestation pass/fail.
    pub threshold: f32,
}

/// Data committed by the guest as public output.
#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionOutput {
    /// Mean Phi across all episodes.
    pub average_phi: f32,
    /// The tau_scale that was used.
    pub tau_scale: f32,
    /// Number of episodes processed.
    pub episode_count: u32,
}

/// Compute the average Phi across episodes using a simplified Laplacian
/// connectivity proxy (variance-based coherence scaled by tau).
///
/// This is the pure-logic kernel that runs inside the zkVM guest.
pub fn compute_average_phi(input: &EvolutionInput) -> EvolutionOutput {
    let n = input.episodes.len() as f32;
    let mut total_phi = 0.0_f32;

    for episode in &input.episodes {
        if episode.is_empty() {
            continue;
        }
        let mean = episode.iter().sum::<f32>() / episode.len() as f32;
        let variance = episode.iter().map(|x| (x - mean).powi(2)).sum::<f32>();
        let phi = variance * input.tau_scale;
        total_phi += phi;
    }

    let average_phi = if n > 0.0 { total_phi / n } else { 0.0 };

    EvolutionOutput {
        average_phi,
        tau_scale: input.tau_scale,
        episode_count: input.episodes.len() as u32,
    }
}

/// Check whether an [`EvolutionOutput`] meets the attestation threshold.
pub fn attestation_passes(output: &EvolutionOutput, threshold: f32) -> bool {
    output.average_phi >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolution_input_construction() {
        let input = EvolutionInput {
            episodes: vec![vec![1.0, 2.0, 3.0]],
            tau_scale: 0.5,
            threshold: 0.1,
        };
        assert_eq!(input.episodes.len(), 1);
        assert_eq!(input.tau_scale, 0.5);
        assert_eq!(input.threshold, 0.1);
    }

    #[test]
    fn compute_phi_empty_episodes() {
        let input = EvolutionInput {
            episodes: vec![],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert_eq!(output.average_phi, 0.0);
        assert_eq!(output.episode_count, 0);
        assert_eq!(output.tau_scale, 1.0);
    }

    #[test]
    fn compute_phi_single_uniform_episode() {
        let input = EvolutionInput {
            episodes: vec![vec![5.0; 100]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert!(
            output.average_phi.abs() < 1e-9,
            "uniform episode should have zero phi, got {}",
            output.average_phi
        );
        assert_eq!(output.episode_count, 1);
    }

    #[test]
    fn compute_phi_single_varied_episode() {
        // [0, 1] => mean=0.5, var = (0.25 + 0.25) = 0.5
        // phi = 0.5 * tau_scale(2.0) = 1.0
        let input = EvolutionInput {
            episodes: vec![vec![0.0, 1.0]],
            tau_scale: 2.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert!(
            (output.average_phi - 1.0).abs() < 1e-6,
            "expected phi=1.0, got {}",
            output.average_phi
        );
    }

    #[test]
    fn compute_phi_multiple_episodes_averaged() {
        let input = EvolutionInput {
            episodes: vec![vec![0.0, 1.0], vec![5.0, 5.0, 5.0, 5.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert!(
            (output.average_phi - 0.25).abs() < 1e-6,
            "expected 0.25, got {}",
            output.average_phi
        );
        assert_eq!(output.episode_count, 2);
    }

    #[test]
    fn compute_phi_tau_scale_multiplier() {
        let base = EvolutionInput {
            episodes: vec![vec![0.0, 1.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let scaled = EvolutionInput {
            tau_scale: 3.0,
            ..base.clone()
        };
        let out_base = compute_average_phi(&base);
        let out_scaled = compute_average_phi(&scaled);
        assert!(
            (out_scaled.average_phi - 3.0 * out_base.average_phi).abs() < 1e-6,
            "phi should scale linearly with tau: base={}, scaled={}",
            out_base.average_phi,
            out_scaled.average_phi
        );
    }

    #[test]
    fn attestation_pass_above_threshold() {
        let output = EvolutionOutput {
            average_phi: 0.75,
            tau_scale: 1.0,
            episode_count: 10,
        };
        assert!(attestation_passes(&output, 0.5));
        assert!(attestation_passes(&output, 0.75));
        assert!(!attestation_passes(&output, 0.76));
    }

    #[test]
    fn compute_phi_empty_episode_vector_skipped() {
        let input = EvolutionInput {
            episodes: vec![vec![], vec![0.0, 1.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert!((output.average_phi - 0.25).abs() < 1e-6);
        assert_eq!(output.episode_count, 2);
    }

    #[test]
    fn test_compute_phi_nan_input() {
        let input = EvolutionInput {
            episodes: vec![vec![f32::NAN, 1.0, 2.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        // Should not panic; NaN propagates through arithmetic
        assert!(output.average_phi.is_nan() || output.average_phi >= 0.0);
    }

    #[test]
    fn test_compute_phi_inf_input() {
        let input = EvolutionInput {
            episodes: vec![vec![f32::INFINITY, 1.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        // Should not panic; Inf propagates
        assert!(output.average_phi.is_infinite() || output.average_phi.is_nan());
    }

    #[test]
    fn test_compute_phi_negative_values() {
        let input = EvolutionInput {
            episodes: vec![vec![-1.0, -2.0, -3.0]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        // Variance is always non-negative, so phi should be >= 0
        assert!(
            output.average_phi >= 0.0,
            "phi with negative values should be >= 0"
        );
    }

    #[test]
    fn test_compute_phi_very_large_values() {
        let input = EvolutionInput {
            episodes: vec![vec![1e30, -1e30]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        // Should not panic; large values produce large variance
        assert!(output.average_phi.is_finite() || output.average_phi.is_infinite());
    }

    #[test]
    fn test_attestation_at_exact_threshold() {
        let output = EvolutionOutput {
            average_phi: 0.5,
            tau_scale: 1.0,
            episode_count: 1,
        };
        // phi == threshold should pass (>=)
        assert!(attestation_passes(&output, 0.5));
        // phi just below threshold should fail
        let output_below = EvolutionOutput {
            average_phi: 0.4999,
            tau_scale: 1.0,
            episode_count: 1,
        };
        assert!(!attestation_passes(&output_below, 0.5));
    }

    #[test]
    fn test_compute_phi_deterministic() {
        let input = EvolutionInput {
            episodes: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]],
            tau_scale: 0.7,
            threshold: 0.0,
        };
        let out1 = compute_average_phi(&input);
        let out2 = compute_average_phi(&input);
        assert_eq!(
            out1.average_phi, out2.average_phi,
            "Same input must produce same output"
        );
    }

    #[test]
    fn test_evolution_output_episode_count() {
        let input = EvolutionInput {
            episodes: vec![vec![1.0], vec![], vec![2.0, 3.0], vec![]],
            tau_scale: 1.0,
            threshold: 0.0,
        };
        let output = compute_average_phi(&input);
        assert_eq!(
            output.episode_count, 4,
            "episode_count should match total episodes including empty"
        );
    }
}
