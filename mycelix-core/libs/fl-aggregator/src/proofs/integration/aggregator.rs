//! Aggregator Integration
//!
//! Integrates gradient proofs with the FL aggregator for verified submissions.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{VerifiedGradientSubmission, ProofVerifyingSubmitter};
//!
//! // Create a verified submission with proof
//! let submission = VerifiedGradientSubmission::new(
//!     node_id,
//!     gradient,
//!     proof_bundle,
//! );
//!
//! // Verify before processing
//! if submission.verify().is_ok() {
//!     aggregator.submit(submission);
//! }
//! ```

use crate::proofs::{
    GradientIntegrityProof, ProofConfig, ProofResult, VerificationResult,
    compute_gradient_commitment,
};
use crate::{Gradient, NodeId};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Bundle of proofs for a gradient submission
#[derive(Clone)]
pub struct GradientProofBundle {
    /// Proof that gradient is well-formed and bounded
    pub integrity_proof: GradientIntegrityProof,
}

impl GradientProofBundle {
    /// Create a new proof bundle
    pub fn new(integrity_proof: GradientIntegrityProof) -> Self {
        Self { integrity_proof }
    }

    /// Generate proofs for a gradient
    pub fn generate(
        gradient: &[f32],
        max_norm: f32,
        config: ProofConfig,
    ) -> ProofResult<Self> {
        let proof = GradientIntegrityProof::generate(
            gradient,
            max_norm,
            config,
        )?;

        Ok(Self::new(proof))
    }

    /// Verify all proofs in the bundle
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        self.integrity_proof.verify()
    }

    /// Get the gradient commitment from the proof
    pub fn gradient_commitment(&self) -> [u8; 32] {
        self.integrity_proof.public_inputs().commitment
    }

    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        self.integrity_proof.size()
    }
}

/// A gradient submission with cryptographic proof
#[derive(Clone)]
pub struct VerifiedGradientSubmission {
    /// Node that submitted
    pub node_id: NodeId,

    /// The gradient data
    pub gradient: Gradient,

    /// Submission timestamp
    pub timestamp: Instant,

    /// Cryptographic proofs
    pub proofs: Option<GradientProofBundle>,

    /// Whether proofs have been verified
    verified: bool,
}

impl VerifiedGradientSubmission {
    /// Create a new verified submission without proofs
    pub fn new(node_id: NodeId, gradient: Gradient) -> Self {
        Self {
            node_id,
            gradient,
            timestamp: Instant::now(),
            proofs: None,
            verified: false,
        }
    }

    /// Create a submission with proofs
    pub fn with_proofs(
        node_id: NodeId,
        gradient: Gradient,
        proofs: GradientProofBundle,
    ) -> Self {
        Self {
            node_id,
            gradient,
            timestamp: Instant::now(),
            proofs: Some(proofs),
            verified: false,
        }
    }

    /// Generate proofs for this submission
    pub fn generate_proofs(
        &mut self,
        max_norm: f32,
        config: ProofConfig,
    ) -> ProofResult<()> {
        let gradient_slice = self.gradient.as_slice()
            .ok_or_else(|| crate::proofs::ProofError::InvalidPublicInputs(
                "Gradient is not contiguous".to_string()
            ))?;

        let bundle = GradientProofBundle::generate(
            gradient_slice,
            max_norm,
            config,
        )?;

        self.proofs = Some(bundle);
        self.verified = false;

        Ok(())
    }

    /// Verify the proofs
    pub fn verify(&mut self) -> ProofResult<VerificationResult> {
        let proofs = self.proofs.as_ref().ok_or_else(|| {
            crate::proofs::ProofError::InvalidPublicInputs("No proofs attached".to_string())
        })?;

        let result = proofs.verify()?;
        self.verified = result.valid;

        // Also verify the commitment matches the gradient
        if result.valid {
            let gradient_slice = self.gradient.as_slice()
                .ok_or_else(|| crate::proofs::ProofError::InvalidPublicInputs(
                    "Gradient is not contiguous".to_string()
                ))?;

            let expected_commitment = compute_gradient_commitment(gradient_slice);
            let actual_commitment = proofs.gradient_commitment();

            if expected_commitment != actual_commitment {
                self.verified = false;
                return Ok(VerificationResult::failure(
                    crate::proofs::ProofType::GradientIntegrity,
                    result.verification_time,
                    "Gradient commitment mismatch",
                ));
            }
        }

        Ok(result)
    }

    /// Check if proofs have been verified
    pub fn is_verified(&self) -> bool {
        self.verified
    }

    /// Check if proofs are attached
    pub fn has_proofs(&self) -> bool {
        self.proofs.is_some()
    }

    /// Get the proof bundle if present
    pub fn proofs(&self) -> Option<&GradientProofBundle> {
        self.proofs.as_ref()
    }
}

/// Trait for submitters that verify proofs
pub trait ProofVerifyingSubmitter {
    /// Submit a verified gradient
    fn submit_verified(
        &mut self,
        submission: VerifiedGradientSubmission,
    ) -> ProofResult<()>;

    /// Check if proof verification is required
    fn requires_proofs(&self) -> bool;

    /// Get proof configuration
    fn proof_config(&self) -> ProofConfig;
}

/// Configuration for proof-verified aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedAggregationConfig {
    /// Whether proofs are required
    pub require_proofs: bool,

    /// Maximum allowed gradient norm
    pub max_gradient_norm: f32,

    /// Proof configuration
    #[serde(skip)]
    pub proof_config: ProofConfig,
}

impl Default for VerifiedAggregationConfig {
    fn default() -> Self {
        Self {
            require_proofs: false,
            max_gradient_norm: 10.0,
            proof_config: ProofConfig::default(),
        }
    }
}

impl VerifiedAggregationConfig {
    /// Enable proof requirement
    pub fn with_proofs(mut self) -> Self {
        self.require_proofs = true;
        self
    }

    /// Set max gradient norm
    pub fn with_max_norm(mut self, norm: f32) -> Self {
        self.max_gradient_norm = norm;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;
    use ndarray::Array1;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_verified_submission_creation() {
        let gradient: Gradient = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let submission = VerifiedGradientSubmission::new("node1".to_string(), gradient);

        assert!(!submission.has_proofs());
        assert!(!submission.is_verified());
    }

    #[test]
    fn test_proof_bundle_generation() {
        let gradient = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let bundle = GradientProofBundle::generate(
            &gradient,
            10.0,
            test_config(),
        ).unwrap();

        let result = bundle.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_verified_submission_with_proofs() {
        let gradient: Gradient = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let node_id = "node1".to_string();

        let bundle = GradientProofBundle::generate(
            gradient.as_slice().unwrap(),
            10.0,
            test_config(),
        ).unwrap();

        let mut submission = VerifiedGradientSubmission::with_proofs(
            node_id,
            gradient,
            bundle,
        );

        assert!(submission.has_proofs());
        assert!(!submission.is_verified());

        let result = submission.verify().unwrap();
        assert!(result.valid);
        assert!(submission.is_verified());
    }

    #[test]
    fn test_generate_proofs_on_submission() {
        let gradient: Gradient = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let mut submission = VerifiedGradientSubmission::new("node1".to_string(), gradient);

        assert!(!submission.has_proofs());

        submission.generate_proofs(10.0, test_config()).unwrap();

        assert!(submission.has_proofs());

        let result = submission.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_verified_aggregation_config() {
        let config = VerifiedAggregationConfig::default()
            .with_proofs()
            .with_max_norm(5.0);

        assert!(config.require_proofs);
        assert_eq!(config.max_gradient_norm, 5.0);
    }
}
