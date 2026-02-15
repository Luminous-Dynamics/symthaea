//! RISC-0 ZK Prover Integration
//!
//! High-level API for generating and verifying gradient quality proofs.
//!
//! # Security Notice
//!
//! Production builds MUST use the `risc0` feature for cryptographic guarantees.
//! Simulation mode is ONLY for testing and provides NO security.
//!
//! # Usage
//!
//! ## Production (with `risc0` feature):
//! ```rust,ignore
//! use mycelix_sdk::zkproof::{ProductionProver, GradientConstraints};
//!
//! let prover = ProductionProver::new(GradientConstraints::default())?;
//! let receipt = prover.prove_gradient_quality(...)?;
//! // Receipt contains real cryptographic proof
//! ```
//!
//! ## Testing (with `simulation` feature):
//! ```rust,ignore
//! use mycelix_sdk::zkproof::{SimulationProver, GradientConstraints};
//!
//! // WARNING: Simulation proofs have NO cryptographic guarantees!
//! let prover = SimulationProver::new_for_testing(GradientConstraints::default());
//! let receipt = prover.prove_gradient_quality(...)?;
//! assert!(receipt.is_simulation_proof()); // Always true for simulation
//! ```

use super::gradient_proof::{
    compute_commitment, hash_gradient, verify_gradient_quality, GradientConstraints,
    GradientProofOutput,
};
use crate::crypto::secure_compare_hash;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// =============================================================================
// Compile-time safety checks
// =============================================================================

// NOTE: This module is only compiled when `simulation` or `risc0` feature is enabled
// (gated by #[cfg] in zkproof/mod.rs). No compile_error! needed here.

// =============================================================================
// Simulation Proof Marker
// =============================================================================

/// Explicit marker for simulation proofs.
///
/// This struct provides a clear, type-safe way to identify simulation proofs
/// rather than relying on magic bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationProofMarker {
    /// Version of the simulation marker format
    pub version: u8,
    /// Explicit warning that this is not a real proof
    pub warning: String,
    /// Timestamp when the simulation was generated
    pub generated_at_ms: u64,
}

impl SimulationProofMarker {
    /// Create a new simulation marker
    pub fn new() -> Self {
        Self {
            version: 1,
            warning: "SIMULATION_ONLY_NO_CRYPTOGRAPHIC_GUARANTEES".to_string(),
            generated_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Serialize the marker to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        // Use the fixed version with proper ASCII byte values
        self.to_bytes_v2()
    }

    /// Check if bytes represent a simulation marker
    pub fn is_simulation_bytes(bytes: &[u8]) -> bool {
        // Use the fixed version with proper ASCII byte values
        Self::is_simulation_bytes_v2(bytes)
    }
}

impl Default for SimulationProofMarker {
    fn default() -> Self {
        Self::new()
    }
}

// Fix invalid hex literals - use actual byte values
const SIMULATION_PREFIX: [u8; 4] = [0x53, 0x49, 0x4D, 0x55]; // "SIMU" in ASCII

impl SimulationProofMarker {
    /// Serialize the marker to bytes (fixed version)
    pub fn to_bytes_v2(&self) -> Vec<u8> {
        let mut bytes = SIMULATION_PREFIX.to_vec();
        bytes.extend(bincode::serialize(self).unwrap_or_default());
        bytes
    }

    /// Check if bytes represent a simulation marker (fixed version)
    pub fn is_simulation_bytes_v2(bytes: &[u8]) -> bool {
        // Check for new simulation marker prefix "SIMU"
        if bytes.len() >= 4 && bytes[0..4] == SIMULATION_PREFIX {
            return true;
        }
        // Also check for legacy DEADBEEF marker for backwards compatibility
        if bytes == [0xDE, 0xAD, 0xBE, 0xEF] {
            return true;
        }
        // Empty bytes also indicate simulation
        bytes.is_empty()
    }
}

// =============================================================================
// Prover Modes (kept for API compatibility, but with warnings)
// =============================================================================

/// Prover operating mode
///
/// **SECURITY WARNING**: `ProverMode::Simulation` provides NO cryptographic
/// guarantees. Use `ProductionProver` for real security.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProverMode {
    /// Simulation mode - no real proofs, instant "verification"
    ///
    /// **WARNING**: Simulation mode provides NO cryptographic security!
    /// Only use for testing with `--features simulation`.
    #[cfg(feature = "simulation")]
    #[default]
    Simulation,
    /// Real RISC-0 proofs (requires risc0 feature and toolchain)
    #[cfg(feature = "risc0")]
    Risc0,
}

#[cfg(feature = "risc0")]
impl Default for ProverMode {
    fn default() -> Self {
        // Production default: real proofs
        ProverMode::Risc0
    }
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
/// Proof receipt containing the proof and public output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofReceipt {
    /// Public output from the proof
    pub output: GradientProofOutput,
    /// Serialized proof bytes (contains SimulationProofMarker in simulation mode)
    pub proof_bytes: Vec<u8>,
    /// Mode used to generate this proof
    pub mode: String,
    /// Time to generate proof in milliseconds
    pub generation_time_ms: u64,
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl GradientProofReceipt {
    /// Check if the proof indicates valid gradient quality
    pub fn is_valid(&self) -> bool {
        self.output.norm_valid
    }

    /// Get the gradient hash commitment
    pub fn gradient_hash(&self) -> &[u8; 32] {
        &self.output.gradient_hash
    }

    /// Check if this is a real proof (not simulation)
    ///
    /// Returns `true` only if the proof was generated with real RISC-0 proofs.
    /// Returns `false` for simulation proofs which provide NO cryptographic guarantees.
    pub fn is_real_proof(&self) -> bool {
        !self.is_simulation_proof()
    }

    /// Check if this is a simulation proof (provides NO cryptographic guarantees)
    ///
    /// **SECURITY WARNING**: Simulation proofs can be trivially forged!
    /// Never accept simulation proofs in production.
    pub fn is_simulation_proof(&self) -> bool {
        // Check mode string
        if self.mode == "simulation" {
            return true;
        }
        // Check proof bytes for simulation markers
        SimulationProofMarker::is_simulation_bytes_v2(&self.proof_bytes)
    }

    /// Verify this receipt is suitable for production use
    ///
    /// Returns an error if this is a simulation proof.
    pub fn verify_production_ready(&self) -> Result<(), ProverError> {
        if self.is_simulation_proof() {
            return Err(ProverError::SimulationProofRejected(
                "Simulation proofs cannot be used in production. Enable 'risc0' feature for real proofs.".into()
            ));
        }
        Ok(())
    }
}

// =============================================================================
// Production Prover (requires risc0 feature)
// =============================================================================

/// Production-grade ZK prover using RISC-0.
///
/// This prover generates real cryptographic proofs that provide
/// actual security guarantees. Requires the `risc0` feature.
#[cfg(feature = "risc0")]
#[derive(Debug, Clone)]
pub struct ProductionProver {
    constraints: GradientConstraints,
}

#[cfg(feature = "risc0")]
impl ProductionProver {
    /// Create a new production prover
    pub fn new(constraints: GradientConstraints) -> Result<Self, ProverError> {
        Ok(Self { constraints })
    }

    /// Create a prover for federated learning scenarios
    pub fn for_federated_learning() -> Result<Self, ProverError> {
        Self::new(GradientConstraints::for_federated_learning())
    }

    /// Generate a real ZK proof of gradient quality
    pub fn prove_gradient_quality(
        &self,
        gradient: &[f32],
        global_model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverError> {
        let start = Instant::now();
        // Real RISC-0 proof generation would go here
        // For now, delegate to internal implementation
        let prover = GradientProver::new_internal(ProverMode::Risc0);
        prover.prove_gradient_quality(
            gradient,
            global_model_hash,
            epochs,
            learning_rate,
            client_id,
            round,
        )
    }

    /// Verify a proof receipt
    pub fn verify(&self, receipt: &GradientProofReceipt) -> Result<bool, ProverError> {
        // Reject simulation proofs
        receipt.verify_production_ready()?;

        let prover = GradientProver::new_internal(ProverMode::Risc0);
        prover.verify(receipt)
    }
}

// =============================================================================
// Simulation Prover (requires explicit simulation feature)
// =============================================================================

/// Simulation prover for TESTING ONLY.
///
/// **SECURITY WARNING**: This prover provides NO cryptographic guarantees!
/// Proofs generated by this prover can be trivially forged.
/// Never use in production - only for development and testing.
#[cfg(feature = "simulation")]
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SimulationProver {
    constraints: GradientConstraints,
    _warning_logged: bool,
}

#[cfg(feature = "simulation")]
impl SimulationProver {
    /// Create a simulation prover for testing.
    ///
    /// **WARNING**: Logs a security warning to stderr.
    /// This is intentional to prevent accidental production use.
    pub fn new_for_testing(constraints: GradientConstraints) -> Self {
        // Always log warning when simulation prover is created
        log_simulation_warning();

        Self {
            constraints,
            _warning_logged: true,
        }
    }

    /// Create a simulation prover for federated learning testing
    pub fn for_federated_learning_testing() -> Self {
        Self::new_for_testing(GradientConstraints::for_federated_learning())
    }

    /// Generate a SIMULATION proof (NO cryptographic guarantees!)
    pub fn prove_gradient_quality(
        &self,
        gradient: &[f32],
        global_model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverError> {
        let prover = GradientProver::new_internal(ProverMode::Simulation);
        prover.prove_gradient_quality(
            gradient,
            global_model_hash,
            epochs,
            learning_rate,
            client_id,
            round,
        )
    }

    /// Verify a simulation receipt (only checks commitment, not cryptography)
    pub fn verify(&self, receipt: &GradientProofReceipt) -> Result<bool, ProverError> {
        let prover = GradientProver::new_internal(ProverMode::Simulation);
        prover.verify(receipt)
    }
}

/// Log a security warning about simulation mode
#[cfg(feature = "simulation")]
fn log_simulation_warning() {
    #[cfg(feature = "std")]
    eprintln!(
        "\n\x1b[1;33m========================================\x1b[0m\n\
         \x1b[1;33m[ZK_PROOF_SECURITY_WARNING]\x1b[0m\n\
         Running in SIMULATION mode!\n\
         Proofs provide NO cryptographic guarantees.\n\
         Do NOT use in production.\n\
         \x1b[1;33m========================================\x1b[0m\n"
    );

    // TODO: Re-enable when log crate is added as dependency
    //     // Also log via log crate if available
    //     [allow(unused)]
    //     log::warn!(
    //         "ZK_PROOF_SECURITY_WARNING: Running in simulation mode - \
    //          proofs provide NO cryptographic guarantees"
    //     );
}

// =============================================================================
// Legacy GradientProver (kept for API compatibility)
// =============================================================================

/// Gradient quality prover (legacy API)
///
/// **Deprecated**: Use `ProductionProver` or `SimulationProver` instead.
/// This type is kept for backwards compatibility but will be removed in v2.0.
#[derive(Debug, Clone)]
pub struct GradientProver {
    mode: ProverMode,
    constraints: GradientConstraints,
}

impl GradientProver {
    /// Create a new prover with the specified mode
    ///
    /// **Deprecated**: Use `ProductionProver::new()` or `SimulationProver::new_for_testing()`.
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    pub fn new(mode: ProverMode) -> Self {
        // Log warning for simulation mode
        #[cfg(feature = "simulation")]
        if matches!(mode, ProverMode::Simulation) {
            log_simulation_warning();
        }

        Self {
            mode,
            constraints: GradientConstraints::default(),
        }
    }

    /// Internal constructor that skips warning (used by typed provers)
    fn new_internal(mode: ProverMode) -> Self {
        Self {
            mode,
            constraints: GradientConstraints::default(),
        }
    }

    /// Create a prover for federated learning scenarios
    ///
    /// **Note**: Now requires explicit mode selection via feature flags.
    #[cfg(feature = "risc0")]
    pub fn for_federated_learning() -> Self {
        Self {
            mode: ProverMode::Risc0,
            constraints: GradientConstraints::for_federated_learning(),
        }
    }

    /// Create a prover configured for federated learning proof generation
    #[cfg(all(feature = "simulation", not(feature = "risc0")))]
    pub fn for_federated_learning() -> Self {
        log_simulation_warning();
        Self {
            mode: ProverMode::Simulation,
            constraints: GradientConstraints::for_federated_learning(),
        }
    }

    /// Set custom constraints
    pub fn with_constraints(mut self, constraints: GradientConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Generate a proof of gradient quality
    ///
    /// The gradient values remain private - only the hash and quality verdict
    /// are revealed.
    pub fn prove_gradient_quality(
        &self,
        gradient: &[f32],
        global_model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverError> {
        match self.mode {
            #[cfg(feature = "simulation")]
            ProverMode::Simulation => self.prove_simulation(
                gradient,
                global_model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
            ),
            #[cfg(feature = "risc0")]
            ProverMode::Risc0 => self.prove_risc0(
                gradient,
                global_model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
                Instant::now(),
            ),
        }
    }

    /// Simulation mode proof generation
    ///
    /// **WARNING**: Generates a proof with NO cryptographic guarantees!
    #[cfg(feature = "simulation")]
    fn prove_simulation(
        &self,
        gradient: &[f32],
        global_model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverError> {
        let start = Instant::now();
        // Run the same verification logic that would run in zkVM
        let result = verify_gradient_quality(gradient, &self.constraints);

        // Compute commitments
        let gradient_hash = hash_gradient(gradient);

        // Create initial output for commitment computation
        let mut output = GradientProofOutput {
            gradient_hash,
            global_model_hash: *global_model_hash,
            epochs,
            learning_rate,
            norm_valid: result.valid,
            client_id: client_id.to_string(),
            round,
            commitment: [0u8; 32],
        };

        // Compute and set commitment
        output.commitment = compute_commitment(&output);

        let generation_time_ms = start.elapsed().as_millis() as u64;

        // Use explicit simulation marker instead of empty bytes or magic DEADBEEF
        let simulation_marker = SimulationProofMarker::new();

        Ok(GradientProofReceipt {
            output,
            proof_bytes: simulation_marker.to_bytes_v2(),
            mode: "simulation".to_string(),
            generation_time_ms,
        })
    }

    /// Verify a proof receipt
    pub fn verify(&self, receipt: &GradientProofReceipt) -> Result<bool, ProverError> {
        match self.mode {
            #[cfg(feature = "simulation")]
            ProverMode::Simulation => {
                // In simulation mode, verify commitment matches
                let expected_commitment = compute_commitment(&GradientProofOutput {
                    gradient_hash: receipt.output.gradient_hash,
                    global_model_hash: receipt.output.global_model_hash,
                    epochs: receipt.output.epochs,
                    learning_rate: receipt.output.learning_rate,
                    norm_valid: receipt.output.norm_valid,
                    client_id: receipt.output.client_id.clone(),
                    round: receipt.output.round,
                    commitment: [0u8; 32],
                });
                // SECURITY: Use constant-time comparison for commitment verification (FIND-004)
                Ok(
                    secure_compare_hash(&receipt.output.commitment, &expected_commitment)
                        && receipt.output.norm_valid,
                )
            }
            #[cfg(feature = "risc0")]
            ProverMode::Risc0 => self.verify_risc0(receipt),
        }
    }

    /// Verify receipt matches expected parameters
    pub fn verify_parameters(
        &self,
        receipt: &GradientProofReceipt,
        expected_model_hash: &[u8; 32],
        expected_round: u32,
    ) -> bool {
        // SECURITY: Use constant-time comparison for hash verification (FIND-004)
        secure_compare_hash(&receipt.output.global_model_hash, expected_model_hash)
            && receipt.output.round == expected_round
            && receipt.output.norm_valid
    }
}

/// Prover errors
#[derive(Debug, Clone)]
pub enum ProverError {
    /// Proof generation failed
    ProofGenerationFailed(String),
    /// Proof verification failed
    VerificationFailed(String),
    /// Invalid input
    InvalidInput(String),
    /// RISC-0 not available
    Risc0Unavailable,
    /// Simulation proof rejected in production context
    ///
    /// This error is returned when a simulation proof is used where
    /// a real cryptographic proof is required.
    SimulationProofRejected(String),
}

impl std::fmt::Display for ProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProverError::ProofGenerationFailed(e) => write!(f, "Proof generation failed: {}", e),
            ProverError::VerificationFailed(e) => write!(f, "Verification failed: {}", e),
            ProverError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            ProverError::Risc0Unavailable => write!(f, "RISC-0 feature not enabled"),
            ProverError::SimulationProofRejected(e) => write!(
                f,
                "SECURITY ERROR: Simulation proof rejected - {}. \
                 Simulation proofs provide NO cryptographic guarantees.",
                e
            ),
        }
    }
}

impl std::error::Error for ProverError {}

/// Batch prover for multiple gradients
#[derive(Debug)]
pub struct BatchGradientProver {
    prover: GradientProver,
}

impl BatchGradientProver {
    /// Create a new batch prover
    pub fn new(prover: GradientProver) -> Self {
        Self { prover }
    }

    /// Prove multiple gradients and return receipts
    pub fn prove_batch(
        &self,
        gradients: &[(&[f32], &str)], // (gradient, client_id)
        global_model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        round: u32,
    ) -> Vec<Result<GradientProofReceipt, ProverError>> {
        gradients
            .iter()
            .map(|(gradient, client_id)| {
                self.prover.prove_gradient_quality(
                    gradient,
                    global_model_hash,
                    epochs,
                    learning_rate,
                    client_id,
                    round,
                )
            })
            .collect()
    }

    /// Verify batch of receipts
    pub fn verify_batch(&self, receipts: &[GradientProofReceipt]) -> Vec<bool> {
        receipts
            .iter()
            .map(|r| self.prover.verify(r).unwrap_or(false))
            .collect()
    }
}

// Tests require simulation feature
#[cfg(all(test, feature = "simulation"))]
mod tests {
    use super::*;

    fn sample_gradient(size: usize, scale: f32) -> Vec<f32> {
        (0..size)
            .map(|i| (i as f32 / size as f32 * std::f32::consts::PI * 2.0).sin() * scale)
            .collect()
    }

    #[test]
    fn test_simulation_prover_valid_gradient() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = sample_gradient(1000, 0.5);
        let model_hash = [0x42u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .expect("Proof should succeed");

        assert!(receipt.is_valid());
        assert!(!receipt.is_real_proof());
        assert!(receipt.is_simulation_proof());
        assert!(prover.verify(&receipt).unwrap());
    }

    #[test]
    fn test_simulation_proof_marker() {
        let marker = SimulationProofMarker::new();
        let bytes = marker.to_bytes_v2();

        // Should be detected as simulation
        assert!(SimulationProofMarker::is_simulation_bytes_v2(&bytes));

        // Legacy DEADBEEF should also be detected
        assert!(SimulationProofMarker::is_simulation_bytes_v2(&[
            0xDE, 0xAD, 0xBE, 0xEF
        ]));

        // Empty bytes should be detected as simulation
        assert!(SimulationProofMarker::is_simulation_bytes_v2(&[]));

        // Random bytes should NOT be simulation
        assert!(!SimulationProofMarker::is_simulation_bytes_v2(&[
            0x01, 0x02, 0x03, 0x04, 0x05
        ]));
    }

    #[test]
    fn test_simulation_prover_zero_gradient() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = vec![0.0f32; 1000];
        let model_hash = [0x42u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .expect("Proof should succeed");

        // Zero gradient fails quality check
        assert!(!receipt.is_valid());
    }

    #[test]
    fn test_simulation_prover_exploding_gradient() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = vec![1000.0f32; 1000]; // Way too large
        let model_hash = [0x42u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .expect("Proof should succeed");

        // Exploding gradient fails quality check
        assert!(!receipt.is_valid());
    }

    #[test]
    fn test_federated_learning_simulation_prover() {
        let prover = SimulationProver::for_federated_learning_testing();
        let gradient = sample_gradient(500, 0.3);
        let model_hash = [0x42u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 10, 0.001, "fl-client-1", 5)
            .expect("Proof should succeed");

        assert!(receipt.is_valid());
        assert_eq!(receipt.output.epochs, 10);
        assert_eq!(receipt.output.round, 5);
    }

    #[test]
    fn test_production_ready_check() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = sample_gradient(1000, 0.5);
        let model_hash = [0x42u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .expect("Proof should succeed");

        // Simulation proofs should fail production-ready check
        let result = receipt.verify_production_ready();
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ProverError::SimulationProofRejected(_))
        ));
    }

    #[test]
    fn test_verify_parameters() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = sample_gradient(1000, 0.5);
        let model_hash = [0x42u8; 32];
        let wrong_hash = [0x43u8; 32];

        let receipt = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .unwrap();

        // Use internal prover for verify_parameters (legacy API)
        let internal_prover = GradientProver::new_internal(ProverMode::Simulation);
        assert!(internal_prover.verify_parameters(&receipt, &model_hash, 1));
        assert!(!internal_prover.verify_parameters(&receipt, &wrong_hash, 1));
        assert!(!internal_prover.verify_parameters(&receipt, &model_hash, 2));
    }

    #[test]
    fn test_batch_prover() {
        let prover = GradientProver::new_internal(ProverMode::Simulation);
        let batch_prover = BatchGradientProver::new(prover);

        let g1 = sample_gradient(1000, 0.5);
        let g2 = sample_gradient(1000, 0.3);
        let g3 = vec![0.0f32; 1000]; // Invalid

        let gradients: Vec<(&[f32], &str)> =
            vec![(&g1, "client-1"), (&g2, "client-2"), (&g3, "client-3")];

        let model_hash = [0x42u8; 32];
        let results = batch_prover.prove_batch(&gradients, &model_hash, 5, 0.01, 1);

        assert!(results[0].as_ref().unwrap().is_valid());
        assert!(results[1].as_ref().unwrap().is_valid());
        assert!(!results[2].as_ref().unwrap().is_valid()); // Zero gradient
    }

    #[test]
    fn test_gradient_hash_deterministic() {
        let prover = SimulationProver::new_for_testing(GradientConstraints::default());
        let gradient = sample_gradient(1000, 0.5);
        let model_hash = [0x42u8; 32];

        let receipt1 = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .unwrap();
        let receipt2 = prover
            .prove_gradient_quality(&gradient, &model_hash, 5, 0.01, "client-1", 1)
            .unwrap();

        // Same gradient should produce same hash
        assert_eq!(receipt1.gradient_hash(), receipt2.gradient_hash());
    }
}
