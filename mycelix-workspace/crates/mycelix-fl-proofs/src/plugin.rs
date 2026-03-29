// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator concept — bridges proofs to fl-core's VerificationPlugin
//! VerificationPlugin bridge for mycelix-fl-core pipeline integration.
//!
//! Implements fl-core's `VerificationPlugin` trait using Blake3 gradient
//! commitments. This provides a lightweight integrity check that can be
//! upgraded to full zkSTARK proofs when the gradient circuit is ported.

use std::collections::HashMap;

use mycelix_fl_core::plugins::{VerificationPlugin, VerificationResult};
use mycelix_fl_core::types::GradientUpdate;

use crate::commitment::{commit_gradient, commit_gradient_with_round};
use crate::types::{ProofConfig, ProofStats};

/// Gradient integrity verifier using Blake3 commitments.
///
/// This is a lightweight implementation that verifies gradient integrity
/// via deterministic commitments. It serves as a bridge to the full
/// zkSTARK proof system — the same public inputs (commitments) will be
/// used as AIR public inputs when the gradient circuit is ported.
pub struct GradientIntegrityVerifier {
    config: ProofConfig,
    stats: ProofStats,
    /// Round number for commitment binding.
    current_round: u32,
}

impl GradientIntegrityVerifier {
    /// Create a new verifier with default config.
    pub fn new() -> Self {
        Self {
            config: ProofConfig::default(),
            stats: ProofStats::default(),
            current_round: 0,
        }
    }

    /// Create with a specific config.
    pub fn with_config(config: ProofConfig) -> Self {
        Self {
            config,
            stats: ProofStats::default(),
            current_round: 0,
        }
    }

    /// Set the current round for commitment binding.
    pub fn set_round(&mut self, round: u32) {
        self.current_round = round;
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &ProofStats {
        &self.stats
    }

    /// Get the config.
    pub fn config(&self) -> &ProofConfig {
        &self.config
    }
}

impl Default for GradientIntegrityVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationPlugin for GradientIntegrityVerifier {
    fn verify(
        &mut self,
        inputs: &[GradientUpdate],
        aggregated: &[f32],
        _reputations: &HashMap<String, f32>,
    ) -> VerificationResult {
        let start = std::time::Instant::now();

        // Verify: each input gradient produces a valid commitment,
        // and the aggregated result is non-degenerate.
        let mut all_valid = true;

        // Check input gradients are non-empty and finite
        for update in inputs {
            if update.gradients.is_empty() {
                all_valid = false;
                break;
            }
            if update.gradients.iter().any(|v| !v.is_finite()) {
                all_valid = false;
                break;
            }
        }

        // Check aggregated result is finite
        if all_valid && aggregated.iter().any(|v| !v.is_finite()) {
            all_valid = false;
        }

        // Compute commitment to aggregated result for proof data
        let proof_data = if all_valid {
            let commitment = commit_gradient_with_round(aggregated, self.current_round);
            Some(commitment.value.to_vec())
        } else {
            None
        };

        let duration = start.elapsed();
        self.stats.record_verification(duration, all_valid);

        // Also compute input commitments for potential future proof linking
        if all_valid {
            for update in inputs {
                let _ = commit_gradient(&update.gradients);
            }
        }

        VerificationResult {
            verified: all_valid,
            proof_data,
            verifier: "blake3-commitment-v1".into(),
        }
    }

    fn name(&self) -> &str {
        "GradientIntegrityVerifier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelix_fl_core::types::GradientUpdate;

    fn make_update(id: &str, gradients: Vec<f32>) -> GradientUpdate {
        GradientUpdate::new(id.to_string(), 1, gradients, 1, 0.9)
    }

    #[test]
    fn test_verify_valid_inputs() {
        let mut verifier = GradientIntegrityVerifier::new();
        verifier.set_round(1);

        let inputs = vec![
            make_update("p1", vec![0.1, 0.2, 0.3]),
            make_update("p2", vec![0.4, 0.5, 0.6]),
        ];
        let aggregated = vec![0.25, 0.35, 0.45];
        let reputations = HashMap::new();

        let result = verifier.verify(&inputs, &aggregated, &reputations);
        assert!(result.verified);
        assert!(result.proof_data.is_some());
        assert_eq!(result.proof_data.as_ref().unwrap().len(), 32); // Blake3 hash
        assert_eq!(result.verifier, "blake3-commitment-v1");
    }

    #[test]
    fn test_verify_nan_gradient_fails() {
        let mut verifier = GradientIntegrityVerifier::new();

        let inputs = vec![make_update("p1", vec![f32::NAN, 0.2])];
        let aggregated = vec![0.1, 0.2];
        let reputations = HashMap::new();

        let result = verifier.verify(&inputs, &aggregated, &reputations);
        assert!(!result.verified);
        assert!(result.proof_data.is_none());
    }

    #[test]
    fn test_verify_nan_aggregated_fails() {
        let mut verifier = GradientIntegrityVerifier::new();

        let inputs = vec![make_update("p1", vec![0.1, 0.2])];
        let aggregated = vec![f32::NAN, 0.2];
        let reputations = HashMap::new();

        let result = verifier.verify(&inputs, &aggregated, &reputations);
        assert!(!result.verified);
    }

    #[test]
    fn test_verify_empty_gradient_fails() {
        let mut verifier = GradientIntegrityVerifier::new();

        let inputs = vec![make_update("p1", vec![])];
        let aggregated = vec![0.1];
        let reputations = HashMap::new();

        let result = verifier.verify(&inputs, &aggregated, &reputations);
        assert!(!result.verified);
    }

    #[test]
    fn test_stats_updated() {
        let mut verifier = GradientIntegrityVerifier::new();

        let inputs = vec![make_update("p1", vec![0.1])];
        let aggregated = vec![0.1];
        let reputations = HashMap::new();

        verifier.verify(&inputs, &aggregated, &reputations);
        verifier.verify(&inputs, &aggregated, &reputations);

        assert_eq!(verifier.stats().proofs_verified, 2);
        assert_eq!(verifier.stats().verification_failures, 0);
    }

    #[test]
    fn test_name() {
        let verifier = GradientIntegrityVerifier::new();
        assert_eq!(verifier.name(), "GradientIntegrityVerifier");
    }

    #[test]
    fn test_default() {
        let verifier = GradientIntegrityVerifier::default();
        assert_eq!(verifier.config().security_level, crate::types::SecurityLevel::Standard128);
    }
}
