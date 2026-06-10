// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ZK Proof Bridge for Federated Learning
//!
//! Integrates zero-knowledge proofs with the FL pipeline to provide
//! cryptographic guarantees of gradient quality without revealing gradient values.
//!
//! # Overview
//!
//! This bridge connects `GradientProver` with `FLCoordinator` to:
//! 1. Generate proofs for gradient submissions
//! 2. Verify proofs during aggregation
//! 3. Filter out participants with invalid proofs (Byzantine detection)
//!
//! # Privacy Guarantees
//!
//! - Gradient values never leave the prover
//! - Only hash commitments are shared
//! - Verifier learns only validity, not actual values
//!
//! # Feature Requirements
//!
//! This module requires either `risc0` or `simulation` feature to be enabled.
//! - Production: `--features risc0`
//! - Testing: `--features simulation`
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::fl::{ZKProofFLBridge, ProvenGradientUpdate};
//! use mycelix_sdk::zkproof::GradientProver;
//!
//! let bridge = ZKProofFLBridge::new();
//!
//! // Submit gradient with proof
//! let proven_update = bridge.submit_with_proof(
//!     "client-1",
//!     &gradient,
//!     &model_hash,
//!     5,
//!     0.01,
//!     1,
//! )?;
//!
//! // Verify and aggregate only valid proofs
//! let result = bridge.aggregate_verified()?;
//! ```

#[cfg(any(feature = "simulation", feature = "risc0"))]
use crate::zkproof::{GradientProofReceipt, GradientProver};

use super::aggregation::{fedavg, trimmed_mean};
use super::types::{AggregatedGradient, GradientUpdate};

/// A gradient update bundled with its ZK proof
#[cfg(any(feature = "simulation", feature = "risc0"))]
#[derive(Debug, Clone)]
pub struct ProvenGradientUpdate {
    /// The gradient update
    pub update: GradientUpdate,
    /// ZK proof of gradient quality
    pub proof: GradientProofReceipt,
    /// Whether the proof has been verified
    pub verified: bool,
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl ProvenGradientUpdate {
    /// Create a new proven update (unverified)
    pub fn new(update: GradientUpdate, proof: GradientProofReceipt) -> Self {
        Self {
            update,
            proof,
            verified: false,
        }
    }

    /// Check if the proof indicates valid gradient
    pub fn is_valid(&self) -> bool {
        self.proof.is_valid()
    }

    /// Get the gradient hash commitment
    pub fn gradient_hash(&self) -> &[u8; 32] {
        self.proof.gradient_hash()
    }
}

/// Statistics from the ZK-FL bridge
#[derive(Debug, Clone, Default)]
pub struct ZKFLStats {
    /// Total submissions received
    pub total_submissions: usize,
    /// Submissions with valid proofs
    pub valid_proofs: usize,
    /// Submissions with invalid proofs (Byzantine)
    pub invalid_proofs: usize,
    /// Proofs that failed verification
    pub verification_failures: usize,
    /// Average proof generation time in ms
    pub avg_proof_time_ms: u64,
}

/// Aggregation method for verified gradients
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum VerifiedAggregationMethod {
    /// Simple averaging of all verified gradients
    #[default]
    FedAvg,
    /// Trimmed mean (exclude extreme values)
    TrimmedMean {
        /// Ratio of extreme values to exclude.
        trim_ratio: f64,
    },
    /// Weighted by proof generation time (faster = more weight)
    TimeWeighted,
}

/// Result of verified aggregation
#[derive(Debug, Clone)]
pub struct VerifiedAggregationResult {
    /// Aggregated gradient
    pub gradient: AggregatedGradient,
    /// Number of gradients included (valid proofs)
    pub included_count: usize,
    /// Number of gradients excluded (invalid proofs)
    pub excluded_count: usize,
    /// Participant IDs with invalid proofs
    pub excluded_participants: Vec<String>,
    /// Hash commitments of included gradients
    pub included_hashes: Vec<[u8; 32]>,
}

/// ZK Proof Bridge for Federated Learning
///
/// Manages the integration of ZK proofs with the FL coordinator.
///
/// Requires `risc0` or `simulation` feature to be enabled.
#[cfg(any(feature = "simulation", feature = "risc0"))]
#[derive(Debug)]
pub struct ZKProofFLBridge {
    /// The gradient prover
    prover: GradientProver,
    /// Current round number
    round: u32,
    /// Model hash for the current round
    model_hash: [u8; 32],
    /// Pending updates awaiting aggregation
    pending_updates: Vec<ProvenGradientUpdate>,
    /// Statistics
    stats: ZKFLStats,
    /// Aggregation method
    aggregation_method: VerifiedAggregationMethod,
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl ZKProofFLBridge {
    /// Create a new bridge with simulation mode prover
    pub fn new() -> Self {
        Self {
            prover: GradientProver::for_federated_learning(),
            round: 0,
            model_hash: [0u8; 32],
            pending_updates: Vec::new(),
            stats: ZKFLStats::default(),
            aggregation_method: VerifiedAggregationMethod::default(),
        }
    }

    /// Create with a custom prover
    pub fn with_prover(prover: GradientProver) -> Self {
        Self {
            prover,
            round: 0,
            model_hash: [0u8; 32],
            pending_updates: Vec::new(),
            stats: ZKFLStats::default(),
            aggregation_method: VerifiedAggregationMethod::default(),
        }
    }

    /// Set the aggregation method
    pub fn with_aggregation(mut self, method: VerifiedAggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Start a new round
    pub fn start_round(&mut self, round: u32, model_hash: [u8; 32]) {
        self.round = round;
        self.model_hash = model_hash;
        self.pending_updates.clear();
    }

    /// Submit a gradient with ZK proof
    ///
    /// Generates a proof of gradient quality and bundles it with the update.
    pub fn submit_with_proof(
        &mut self,
        participant_id: &str,
        gradient: &[f32],
        epochs: u32,
        learning_rate: f32,
        batch_size: usize,
        loss: f64,
    ) -> Result<ProvenGradientUpdate, ZKFLError> {
        // Generate proof
        let proof = self
            .prover
            .prove_gradient_quality(
                gradient,
                &self.model_hash,
                epochs,
                learning_rate,
                participant_id,
                self.round,
            )
            .map_err(|e| ZKFLError::ProofGenerationFailed(e.to_string()))?;

        // Update stats
        self.stats.total_submissions += 1;
        self.stats.avg_proof_time_ms = (self.stats.avg_proof_time_ms
            * (self.stats.total_submissions - 1) as u64
            + proof.generation_time_ms)
            / self.stats.total_submissions as u64;

        if proof.is_valid() {
            self.stats.valid_proofs += 1;
        } else {
            self.stats.invalid_proofs += 1;
        }

        // Convert gradient to f64 for storage
        let gradient_f64: Vec<f64> = gradient.iter().map(|&g| g as f64).collect();

        // Create update
        let update = GradientUpdate::new(
            participant_id.to_string(),
            self.round as u64,
            gradient_f64,
            batch_size,
            loss,
        );

        let proven = ProvenGradientUpdate::new(update, proof);
        self.pending_updates.push(proven.clone());

        Ok(proven)
    }

    /// Submit a pre-computed gradient update with proof
    pub fn submit_proven(&mut self, update: GradientUpdate, proof: GradientProofReceipt) {
        self.stats.total_submissions += 1;
        if proof.is_valid() {
            self.stats.valid_proofs += 1;
        } else {
            self.stats.invalid_proofs += 1;
        }
        self.pending_updates
            .push(ProvenGradientUpdate::new(update, proof));
    }

    /// Verify all pending proofs
    pub fn verify_all(&mut self) -> Vec<String> {
        let mut failed = Vec::new();

        for update in &mut self.pending_updates {
            match self.prover.verify(&update.proof) {
                Ok(valid) => {
                    update.verified = true;
                    if !valid {
                        failed.push(update.update.participant_id.clone());
                        self.stats.verification_failures += 1;
                    }
                }
                Err(_) => {
                    failed.push(update.update.participant_id.clone());
                    self.stats.verification_failures += 1;
                }
            }
        }

        failed
    }

    /// Aggregate only verified valid gradients
    pub fn aggregate_verified(&self) -> Result<VerifiedAggregationResult, ZKFLError> {
        // Filter to valid, verified updates
        let valid_updates: Vec<&ProvenGradientUpdate> = self
            .pending_updates
            .iter()
            .filter(|u| u.proof.is_valid())
            .collect();

        let excluded: Vec<String> = self
            .pending_updates
            .iter()
            .filter(|u| !u.proof.is_valid())
            .map(|u| u.update.participant_id.clone())
            .collect();

        if valid_updates.is_empty() {
            return Err(ZKFLError::NoValidGradients);
        }

        // Extract updates for aggregation
        let updates: Vec<GradientUpdate> = valid_updates.iter().map(|u| u.update.clone()).collect();

        // Helper to create AggregatedGradient from raw gradients
        let make_aggregated =
            |gradients: Vec<f64>, method: super::AggregationMethod| AggregatedGradient {
                gradients,
                model_version: self.round as u64,
                participant_count: updates.len(),
                excluded_count: excluded.len(),
                aggregation_method: method,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            };

        // Aggregate based on method
        let aggregated = match self.aggregation_method {
            VerifiedAggregationMethod::FedAvg => {
                let gradients =
                    fedavg(&updates).map_err(|e| ZKFLError::AggregationFailed(e.to_string()))?;
                make_aggregated(gradients, super::AggregationMethod::FedAvg)
            }
            VerifiedAggregationMethod::TrimmedMean { trim_ratio } => {
                let gradients = trimmed_mean(&updates, trim_ratio)
                    .map_err(|e| ZKFLError::AggregationFailed(e.to_string()))?;
                make_aggregated(gradients, super::AggregationMethod::TrimmedMean)
            }
            VerifiedAggregationMethod::TimeWeighted => {
                // Weight by inverse of proof time (faster = higher weight)
                let weights: Vec<f64> = valid_updates
                    .iter()
                    .map(|u| 1.0 / (u.proof.generation_time_ms.max(1) as f64))
                    .collect();

                // Normalize weights
                let total: f64 = weights.iter().sum();
                let norm_weights: Vec<f64> = weights.iter().map(|w| w / total).collect();

                // Weighted average
                let dim = updates[0].gradients.len();
                let mut aggregated_gradients = vec![0.0; dim];

                for (update, weight) in updates.iter().zip(norm_weights.iter()) {
                    for (i, g) in update.gradients.iter().enumerate() {
                        aggregated_gradients[i] += g * weight;
                    }
                }

                make_aggregated(
                    aggregated_gradients,
                    super::AggregationMethod::TrustWeighted,
                )
            }
        };

        // Collect included hashes
        let included_hashes: Vec<[u8; 32]> =
            valid_updates.iter().map(|u| *u.gradient_hash()).collect();

        Ok(VerifiedAggregationResult {
            gradient: aggregated,
            included_count: valid_updates.len(),
            excluded_count: excluded.len(),
            excluded_participants: excluded,
            included_hashes,
        })
    }

    /// Get Byzantine fraction based on proof validity
    pub fn byzantine_fraction(&self) -> f64 {
        if self.stats.total_submissions == 0 {
            return 0.0;
        }
        self.stats.invalid_proofs as f64 / self.stats.total_submissions as f64
    }

    /// Get current statistics
    pub fn stats(&self) -> &ZKFLStats {
        &self.stats
    }

    /// Get pending updates
    pub fn pending_updates(&self) -> &[ProvenGradientUpdate] {
        &self.pending_updates
    }

    /// Get current round
    pub fn round(&self) -> u32 {
        self.round
    }

    /// Clear all pending updates
    pub fn clear(&mut self) {
        self.pending_updates.clear();
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ZKFLStats::default();
    }
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl Default for ZKProofFLBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from the ZK-FL bridge
#[derive(Debug, Clone)]
pub enum ZKFLError {
    /// Proof generation failed
    ProofGenerationFailed(String),
    /// Proof verification failed
    VerificationFailed(String),
    /// No valid gradients to aggregate
    NoValidGradients,
    /// Aggregation failed
    AggregationFailed(String),
    /// Round mismatch
    RoundMismatch {
        /// Expected round number.
        expected: u32,
        /// Actual round number received.
        got: u32,
    },
}

impl std::fmt::Display for ZKFLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZKFLError::ProofGenerationFailed(e) => write!(f, "Proof generation failed: {}", e),
            ZKFLError::VerificationFailed(e) => write!(f, "Verification failed: {}", e),
            ZKFLError::NoValidGradients => write!(f, "No valid gradients to aggregate"),
            ZKFLError::AggregationFailed(e) => write!(f, "Aggregation failed: {}", e),
            ZKFLError::RoundMismatch { expected, got } => {
                write!(f, "Round mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for ZKFLError {}

#[cfg(all(test, any(feature = "simulation", feature = "risc0")))]
mod tests {
    use super::*;

    fn sample_gradient(size: usize, scale: f32) -> Vec<f32> {
        (0..size)
            .map(|i| (i as f32 / size as f32 * std::f32::consts::PI * 2.0).sin() * scale)
            .collect()
    }

    #[test]
    fn test_submit_with_proof() {
        let mut bridge = ZKProofFLBridge::new();
        bridge.start_round(1, [0x42u8; 32]);

        let gradient = sample_gradient(1000, 0.5);
        let result = bridge.submit_with_proof("client-1", &gradient, 5, 0.01, 32, 0.5);

        assert!(result.is_ok());
        let proven = result.unwrap();
        assert!(proven.is_valid());
        assert_eq!(bridge.stats().total_submissions, 1);
        assert_eq!(bridge.stats().valid_proofs, 1);
    }

    #[test]
    fn test_invalid_gradient_detected() {
        let mut bridge = ZKProofFLBridge::new();
        bridge.start_round(1, [0x42u8; 32]);

        // Zero gradient should fail quality check
        let gradient = vec![0.0f32; 1000];
        let result = bridge.submit_with_proof("client-1", &gradient, 5, 0.01, 32, 0.5);

        assert!(result.is_ok());
        let proven = result.unwrap();
        assert!(!proven.is_valid()); // Should be invalid
        assert_eq!(bridge.stats().invalid_proofs, 1);
    }

    #[test]
    fn test_aggregate_verified() {
        let mut bridge = ZKProofFLBridge::new();
        bridge.start_round(1, [0x42u8; 32]);

        // Submit 3 valid gradients
        for i in 0..3 {
            let gradient = sample_gradient(100, 0.3 + i as f32 * 0.1);
            bridge
                .submit_with_proof(&format!("client-{}", i), &gradient, 5, 0.01, 32, 0.5)
                .unwrap();
        }

        // Submit 1 invalid gradient
        let bad_gradient = vec![0.0f32; 100];
        bridge
            .submit_with_proof("bad-client", &bad_gradient, 5, 0.01, 32, 0.5)
            .unwrap();

        let result = bridge.aggregate_verified().unwrap();

        assert_eq!(result.included_count, 3);
        assert_eq!(result.excluded_count, 1);
        assert!(result
            .excluded_participants
            .contains(&"bad-client".to_string()));
    }

    #[test]
    fn test_byzantine_fraction() {
        let mut bridge = ZKProofFLBridge::new();
        bridge.start_round(1, [0x42u8; 32]);

        // 2 valid
        for i in 0..2 {
            let gradient = sample_gradient(100, 0.5);
            bridge
                .submit_with_proof(&format!("good-{}", i), &gradient, 5, 0.01, 32, 0.5)
                .unwrap();
        }

        // 1 invalid
        let bad_gradient = vec![0.0f32; 100];
        bridge
            .submit_with_proof("bad", &bad_gradient, 5, 0.01, 32, 0.5)
            .unwrap();

        let byzantine_frac = bridge.byzantine_fraction();
        assert!((byzantine_frac - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_rounds() {
        let mut bridge = ZKProofFLBridge::new();

        // Round 1
        bridge.start_round(1, [0x01u8; 32]);
        let g1 = sample_gradient(100, 0.5);
        bridge
            .submit_with_proof("client-1", &g1, 5, 0.01, 32, 0.5)
            .unwrap();
        let result1 = bridge.aggregate_verified().unwrap();
        assert_eq!(result1.included_count, 1);

        // Round 2
        bridge.start_round(2, [0x02u8; 32]);
        assert!(bridge.pending_updates().is_empty()); // Cleared
        let g2 = sample_gradient(100, 0.4);
        bridge
            .submit_with_proof("client-2", &g2, 5, 0.01, 32, 0.5)
            .unwrap();
        let result2 = bridge.aggregate_verified().unwrap();
        assert_eq!(result2.included_count, 1);
    }

    #[test]
    fn test_time_weighted_aggregation() {
        let mut bridge =
            ZKProofFLBridge::new().with_aggregation(VerifiedAggregationMethod::TimeWeighted);
        bridge.start_round(1, [0x42u8; 32]);

        for i in 0..3 {
            let gradient = sample_gradient(100, 0.5);
            bridge
                .submit_with_proof(&format!("client-{}", i), &gradient, 5, 0.01, 32, 0.5)
                .unwrap();
        }

        let result = bridge.aggregate_verified().unwrap();
        assert_eq!(
            result.gradient.aggregation_method,
            crate::fl::AggregationMethod::TrustWeighted
        );
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let mut bridge = ZKProofFLBridge::new()
            .with_aggregation(VerifiedAggregationMethod::TrimmedMean { trim_ratio: 0.1 });
        bridge.start_round(1, [0x42u8; 32]);

        for i in 0..5 {
            let gradient = sample_gradient(100, 0.3 + i as f32 * 0.1);
            bridge
                .submit_with_proof(&format!("client-{}", i), &gradient, 5, 0.01, 32, 0.5)
                .unwrap();
        }

        let result = bridge.aggregate_verified().unwrap();
        assert_eq!(result.included_count, 5);
    }

    #[test]
    fn test_no_valid_gradients_error() {
        let mut bridge = ZKProofFLBridge::new();
        bridge.start_round(1, [0x42u8; 32]);

        // Only invalid gradients
        let bad_gradient = vec![0.0f32; 100];
        bridge
            .submit_with_proof("bad-1", &bad_gradient, 5, 0.01, 32, 0.5)
            .unwrap();
        bridge
            .submit_with_proof("bad-2", &bad_gradient, 5, 0.01, 32, 0.5)
            .unwrap();

        let result = bridge.aggregate_verified();
        assert!(matches!(result, Err(ZKFLError::NoValidGradients)));
    }
}
