//! Proof Simulation Mode
//!
//! Provides dry-run proof generation for testing and validation without
//! the computational overhead of actual STARK proof generation.
//!
//! ## Features
//!
//! - **Dry-run validation**: Validate inputs without generating proofs
//! - **Cost estimation**: Estimate proof generation time and size
//! - **Mock proofs**: Generate deterministic mock proofs for testing
//! - **Simulation statistics**: Track simulated operations
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::simulation::{ProofSimulator, SimulationMode};
//!
//! // Create simulator with estimates
//! let simulator = ProofSimulator::new(SimulationMode::Estimate);
//!
//! // Validate a range proof request
//! let result = simulator.simulate_range_proof(42, 0, 100)?;
//! println!("Estimated time: {:.2}ms", result.estimated_time_ms);
//! println!("Estimated size: {} bytes", result.estimated_size_bytes);
//!
//! // Generate mock proof for testing
//! let mock_simulator = ProofSimulator::new(SimulationMode::Mock);
//! let mock_result = mock_simulator.simulate_range_proof(42, 0, 100)?;
//! assert!(mock_result.mock_proof.is_some());
//! ```
//!
//! ## Test Integration
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::simulation::MockProofGenerator;
//!
//! // In tests, use mock generator for speed
//! let generator = MockProofGenerator::new();
//! let mock_proof = generator.mock_range_proof(42, 0, 100);
//!
//! // Mock proofs verify as valid in simulation mode
//! let verifier = MockProofVerifier::new();
//! assert!(verifier.verify(&mock_proof).is_valid);
//! ```

use std::collections::HashMap;
use std::time::Instant;

use crate::proofs::{
    ProofConfig, ProofResult, ProofType, SecurityLevel,
    ProofAssuranceLevel, ProofProposalType, ProofVoterProfile,
};
use serde::{Deserialize, Serialize};

/// Simulation mode determines what actions are taken
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimulationMode {
    /// Validate inputs and estimate costs, no proof generation
    #[default]
    Estimate,
    /// Generate mock proofs for testing
    Mock,
    /// Validate only, no estimation
    ValidateOnly,
}

/// Result of a simulated proof operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Whether the inputs are valid
    pub inputs_valid: bool,
    /// Validation errors if any
    pub validation_errors: Vec<String>,
    /// Estimated generation time in milliseconds
    pub estimated_time_ms: f64,
    /// Estimated proof size in bytes
    pub estimated_size_bytes: usize,
    /// Estimated verification time in milliseconds
    pub estimated_verify_time_ms: f64,
    /// The proof type
    pub proof_type: ProofType,
    /// Security level used
    pub security_level: SecurityLevel,
    /// Mock proof bytes (if SimulationMode::Mock)
    #[serde(skip)]
    pub mock_proof: Option<Vec<u8>>,
    /// Time taken for simulation
    pub simulation_time_ms: f64,
}

impl SimulationResult {
    /// Check if simulation succeeded
    pub fn is_valid(&self) -> bool {
        self.inputs_valid && self.validation_errors.is_empty()
    }
}

/// Cost estimates for proof operations
#[derive(Debug, Clone, Copy)]
pub struct CostEstimates {
    /// Base generation time in ms
    pub base_time_ms: f64,
    /// Time per constraint
    pub time_per_constraint_ms: f64,
    /// Base proof size in bytes
    pub base_size_bytes: usize,
    /// Size per constraint
    pub size_per_constraint: usize,
    /// Verification time in ms
    pub verify_time_ms: f64,
}

impl CostEstimates {
    /// Default estimates for Range proofs
    pub fn range() -> Self {
        Self {
            base_time_ms: 30.0,
            time_per_constraint_ms: 0.3,
            base_size_bytes: 12000,
            size_per_constraint: 50,
            verify_time_ms: 5.0,
        }
    }

    /// Default estimates for Membership proofs
    pub fn membership() -> Self {
        Self {
            base_time_ms: 50.0,
            time_per_constraint_ms: 0.5,
            base_size_bytes: 18000,
            size_per_constraint: 80,
            verify_time_ms: 8.0,
        }
    }

    /// Default estimates for Gradient proofs
    pub fn gradient() -> Self {
        Self {
            base_time_ms: 80.0,
            time_per_constraint_ms: 0.8,
            base_size_bytes: 22000,
            size_per_constraint: 100,
            verify_time_ms: 12.0,
        }
    }

    /// Default estimates for Identity proofs
    pub fn identity() -> Self {
        Self {
            base_time_ms: 40.0,
            time_per_constraint_ms: 0.4,
            base_size_bytes: 15000,
            size_per_constraint: 60,
            verify_time_ms: 6.0,
        }
    }

    /// Default estimates for Vote proofs
    pub fn vote() -> Self {
        Self {
            base_time_ms: 35.0,
            time_per_constraint_ms: 0.35,
            base_size_bytes: 14000,
            size_per_constraint: 55,
            verify_time_ms: 5.5,
        }
    }

    /// Estimate total time for given constraint count
    pub fn estimate_time(&self, constraints: usize) -> f64 {
        self.base_time_ms + (constraints as f64 * self.time_per_constraint_ms)
    }

    /// Estimate total size for given constraint count
    pub fn estimate_size(&self, constraints: usize) -> usize {
        self.base_size_bytes + (constraints * self.size_per_constraint)
    }

    /// Apply security level multiplier
    pub fn with_security(&self, level: SecurityLevel) -> Self {
        let multiplier = match level {
            SecurityLevel::Standard96 => 0.7,
            SecurityLevel::Standard128 => 1.0,
            SecurityLevel::High256 => 1.8,
        };
        Self {
            base_time_ms: self.base_time_ms * multiplier,
            time_per_constraint_ms: self.time_per_constraint_ms * multiplier,
            base_size_bytes: (self.base_size_bytes as f64 * multiplier) as usize,
            size_per_constraint: (self.size_per_constraint as f64 * multiplier) as usize,
            verify_time_ms: self.verify_time_ms * multiplier,
        }
    }
}

/// Proof simulator for dry-run operations
pub struct ProofSimulator {
    mode: SimulationMode,
    config: ProofConfig,
    estimates: HashMap<ProofType, CostEstimates>,
    stats: SimulationStats,
}

impl ProofSimulator {
    /// Create a new simulator
    pub fn new(mode: SimulationMode) -> Self {
        let mut estimates = HashMap::new();
        estimates.insert(ProofType::Range, CostEstimates::range());
        estimates.insert(ProofType::Membership, CostEstimates::membership());
        estimates.insert(ProofType::GradientIntegrity, CostEstimates::gradient());
        estimates.insert(ProofType::IdentityAssurance, CostEstimates::identity());
        estimates.insert(ProofType::VoteEligibility, CostEstimates::vote());

        Self {
            mode,
            config: ProofConfig::default(),
            estimates,
            stats: SimulationStats::default(),
        }
    }

    /// Set proof configuration
    pub fn with_config(mut self, config: ProofConfig) -> Self {
        self.config = config;
        self
    }

    /// Override cost estimates for a proof type
    pub fn with_estimates(mut self, proof_type: ProofType, estimates: CostEstimates) -> Self {
        self.estimates.insert(proof_type, estimates);
        self
    }

    /// Get simulation statistics
    pub fn stats(&self) -> &SimulationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimulationStats::default();
    }

    // ========================================================================
    // Range Proof Simulation
    // ========================================================================

    /// Simulate a range proof
    pub fn simulate_range_proof(
        &mut self,
        value: u64,
        min: u64,
        max: u64,
    ) -> ProofResult<SimulationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Validate inputs
        if min > max {
            errors.push(format!("Invalid range: min ({}) > max ({})", min, max));
        }
        if value < min || value > max {
            errors.push(format!("Value {} not in range [{}, {}]", value, min, max));
        }

        // Estimate costs (64 constraints for bit decomposition)
        let estimates = self.get_estimates(ProofType::Range);
        let constraints = 64; // Standard bit decomposition

        let result = self.build_result(
            ProofType::Range,
            errors.is_empty(),
            errors,
            estimates,
            constraints,
            start,
        );

        self.stats.record(ProofType::Range, result.is_valid());
        Ok(result)
    }

    // ========================================================================
    // Membership Proof Simulation
    // ========================================================================

    /// Simulate a membership proof
    pub fn simulate_membership_proof(
        &mut self,
        _leaf: &[u8; 32],
        _merkle_root: &[u8; 32],
        path_length: usize,
    ) -> ProofResult<SimulationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Validate inputs
        if path_length == 0 {
            errors.push("Path length must be > 0".to_string());
        }
        if path_length > 256 {
            errors.push(format!("Path length {} exceeds maximum (256)", path_length));
        }

        // Estimate costs based on tree depth
        let estimates = self.get_estimates(ProofType::Membership);
        let constraints = path_length * 512; // ~512 constraints per hash

        let result = self.build_result(
            ProofType::Membership,
            errors.is_empty(),
            errors,
            estimates,
            constraints,
            start,
        );

        self.stats.record(ProofType::Membership, result.is_valid());
        Ok(result)
    }

    // ========================================================================
    // Gradient Proof Simulation
    // ========================================================================

    /// Simulate a gradient integrity proof
    pub fn simulate_gradient_proof(
        &mut self,
        gradient_len: usize,
        max_norm: f32,
    ) -> ProofResult<SimulationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Validate inputs
        if gradient_len == 0 {
            errors.push("Gradient length must be > 0".to_string());
        }
        if gradient_len > 10_000_000 {
            errors.push(format!(
                "Gradient length {} exceeds maximum (10M)",
                gradient_len
            ));
        }
        if max_norm <= 0.0 {
            errors.push(format!("Max norm must be > 0, got {}", max_norm));
        }
        if max_norm.is_nan() || max_norm.is_infinite() {
            errors.push("Max norm must be finite".to_string());
        }

        // Estimate costs based on gradient size
        let estimates = self.get_estimates(ProofType::GradientIntegrity);
        let constraints = gradient_len * 4; // ~4 constraints per element

        let result = self.build_result(
            ProofType::GradientIntegrity,
            errors.is_empty(),
            errors,
            estimates,
            constraints,
            start,
        );

        self.stats.record(ProofType::GradientIntegrity, result.is_valid());
        Ok(result)
    }

    // ========================================================================
    // Identity Proof Simulation
    // ========================================================================

    /// Simulate an identity assurance proof
    pub fn simulate_identity_proof(
        &mut self,
        did: &str,
        factor_count: usize,
        min_level: ProofAssuranceLevel,
    ) -> ProofResult<SimulationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Validate inputs
        if did.is_empty() {
            errors.push("DID cannot be empty".to_string());
        }
        if !did.starts_with("did:") {
            errors.push(format!("Invalid DID format: {}", did));
        }
        if factor_count == 0 {
            errors.push("At least one identity factor required".to_string());
        }
        if factor_count > 20 {
            errors.push(format!("Too many factors: {} (max 20)", factor_count));
        }

        // Check if min_level is achievable
        let min_factors_needed = match min_level {
            ProofAssuranceLevel::E0 => 0,
            ProofAssuranceLevel::E1 => 1,
            ProofAssuranceLevel::E2 => 2,
            ProofAssuranceLevel::E3 => 3,
            ProofAssuranceLevel::E4 => 4,
        };
        if factor_count < min_factors_needed {
            errors.push(format!(
                "Level {:?} requires at least {} factors, have {}",
                min_level, min_factors_needed, factor_count
            ));
        }

        // Estimate costs
        let estimates = self.get_estimates(ProofType::IdentityAssurance);
        let constraints = 9 + factor_count * 16; // Base + per-factor

        let result = self.build_result(
            ProofType::IdentityAssurance,
            errors.is_empty(),
            errors,
            estimates,
            constraints,
            start,
        );

        self.stats.record(ProofType::IdentityAssurance, result.is_valid());
        Ok(result)
    }

    // ========================================================================
    // Vote Proof Simulation
    // ========================================================================

    /// Simulate a vote eligibility proof
    pub fn simulate_vote_proof(
        &mut self,
        voter: &ProofVoterProfile,
        proposal_type: ProofProposalType,
    ) -> ProofResult<SimulationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Validate voter profile
        if voter.did.is_empty() {
            errors.push("Voter DID cannot be empty".to_string());
        }
        if voter.assurance_level > 4 {
            errors.push(format!(
                "Invalid assurance level: {} (max 4)",
                voter.assurance_level
            ));
        }
        if voter.matl_score < 0.0 || voter.matl_score > 1.0 {
            errors.push(format!(
                "MATL score must be [0, 1], got {}",
                voter.matl_score
            ));
        }
        if voter.participation_rate < 0.0 || voter.participation_rate > 1.0 {
            errors.push(format!(
                "Participation rate must be [0, 1], got {}",
                voter.participation_rate
            ));
        }

        // Check eligibility for proposal type
        match proposal_type {
            ProofProposalType::Standard => {
                // Minimal requirements
            }
            ProofProposalType::Constitutional => {
                if !voter.has_humanity_proof {
                    errors.push("Constitutional votes require humanity proof".to_string());
                }
                if voter.stake < 500.0 {
                    errors.push(format!(
                        "Constitutional votes require stake >= 500, have {}",
                        voter.stake
                    ));
                }
                if voter.assurance_level < 2 {
                    errors.push(format!(
                        "Constitutional votes require assurance >= E2, have E{}",
                        voter.assurance_level
                    ));
                }
            }
            ProofProposalType::ModelGovernance => {
                if voter.fl_contributions < 10 {
                    errors.push(format!(
                        "Model governance requires >= 10 FL contributions, have {}",
                        voter.fl_contributions
                    ));
                }
                if voter.matl_score < 0.5 {
                    errors.push(format!(
                        "Model governance requires MATL >= 0.5, have {}",
                        voter.matl_score
                    ));
                }
            }
            ProofProposalType::Emergency => {
                if voter.assurance_level < 3 {
                    errors.push(format!(
                        "Emergency votes require assurance >= E3, have E{}",
                        voter.assurance_level
                    ));
                }
                if voter.stake < 1000.0 {
                    errors.push(format!(
                        "Emergency votes require stake >= 1000, have {}",
                        voter.stake
                    ));
                }
            }
            ProofProposalType::Treasury => {
                if voter.stake < 500.0 {
                    errors.push(format!(
                        "Treasury votes require stake >= 500, have {}",
                        voter.stake
                    ));
                }
                if voter.participation_rate < 0.3 {
                    errors.push(format!(
                        "Treasury votes require participation >= 0.3, have {}",
                        voter.participation_rate
                    ));
                }
            }
            ProofProposalType::Membership => {
                if voter.account_age_days < 30 {
                    errors.push(format!(
                        "Membership votes require account age >= 30 days, have {}",
                        voter.account_age_days
                    ));
                }
            }
        }

        // Estimate costs
        let estimates = self.get_estimates(ProofType::VoteEligibility);
        let constraints = 7 * 32; // 7 requirements, ~32 constraints each

        let result = self.build_result(
            ProofType::VoteEligibility,
            errors.is_empty(),
            errors,
            estimates,
            constraints,
            start,
        );

        self.stats.record(ProofType::VoteEligibility, result.is_valid());
        Ok(result)
    }

    // ========================================================================
    // Batch Simulation
    // ========================================================================

    /// Simulate a batch of proofs
    pub fn simulate_batch(&mut self, requests: &[ProofRequest]) -> BatchSimulationResult {
        let start = Instant::now();
        let mut results = Vec::with_capacity(requests.len());
        let mut total_time = 0.0;
        let mut total_size = 0;
        let mut valid_count = 0;

        for request in requests {
            let result = match request {
                ProofRequest::Range { value, min, max } => {
                    self.simulate_range_proof(*value, *min, *max).ok()
                }
                ProofRequest::Membership {
                    leaf,
                    root,
                    path_len,
                } => self.simulate_membership_proof(leaf, root, *path_len).ok(),
                ProofRequest::Gradient { len, max_norm } => {
                    self.simulate_gradient_proof(*len, *max_norm).ok()
                }
                ProofRequest::Identity {
                    did,
                    factor_count,
                    min_level,
                } => self.simulate_identity_proof(did, *factor_count, *min_level).ok(),
                ProofRequest::Vote { voter, proposal } => {
                    self.simulate_vote_proof(voter, *proposal).ok()
                }
            };

            if let Some(r) = result {
                if r.is_valid() {
                    valid_count += 1;
                }
                total_time += r.estimated_time_ms;
                total_size += r.estimated_size_bytes;
                results.push(r);
            }
        }

        BatchSimulationResult {
            total_count: requests.len(),
            valid_count,
            invalid_count: requests.len() - valid_count,
            estimated_total_time_ms: total_time,
            estimated_total_size_bytes: total_size,
            simulation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            results,
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn get_estimates(&self, proof_type: ProofType) -> CostEstimates {
        self.estimates
            .get(&proof_type)
            .copied()
            .unwrap_or(CostEstimates::range())
            .with_security(self.config.security_level)
    }

    fn build_result(
        &self,
        proof_type: ProofType,
        inputs_valid: bool,
        errors: Vec<String>,
        estimates: CostEstimates,
        constraints: usize,
        start: Instant,
    ) -> SimulationResult {
        let mock_proof = if self.mode == SimulationMode::Mock && inputs_valid {
            Some(self.generate_mock_proof(proof_type, constraints))
        } else {
            None
        };

        SimulationResult {
            inputs_valid,
            validation_errors: errors,
            estimated_time_ms: estimates.estimate_time(constraints),
            estimated_size_bytes: estimates.estimate_size(constraints),
            estimated_verify_time_ms: estimates.verify_time_ms,
            proof_type,
            security_level: self.config.security_level,
            mock_proof,
            simulation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn generate_mock_proof(&self, proof_type: ProofType, size_hint: usize) -> Vec<u8> {
        // Generate deterministic mock proof
        let mut proof = Vec::with_capacity(1024);

        // Magic header
        proof.extend_from_slice(b"MOCK");

        // Proof type
        proof.push(proof_type as u8);

        // Security level
        proof.push(self.config.security_level as u8);

        // Size hint
        proof.extend_from_slice(&(size_hint as u32).to_le_bytes());

        // Deterministic "proof data" - just zeros padded to estimated size
        let padding = self
            .get_estimates(proof_type)
            .estimate_size(size_hint)
            .saturating_sub(proof.len());
        proof.resize(proof.len() + padding.min(50000), 0);

        proof
    }
}

impl Default for ProofSimulator {
    fn default() -> Self {
        Self::new(SimulationMode::default())
    }
}

/// Proof request for batch simulation
#[derive(Debug, Clone)]
pub enum ProofRequest {
    Range { value: u64, min: u64, max: u64 },
    Membership { leaf: [u8; 32], root: [u8; 32], path_len: usize },
    Gradient { len: usize, max_norm: f32 },
    Identity { did: String, factor_count: usize, min_level: ProofAssuranceLevel },
    Vote { voter: ProofVoterProfile, proposal: ProofProposalType },
}

/// Batch simulation result
#[derive(Debug, Clone)]
pub struct BatchSimulationResult {
    pub total_count: usize,
    pub valid_count: usize,
    pub invalid_count: usize,
    pub estimated_total_time_ms: f64,
    pub estimated_total_size_bytes: usize,
    pub simulation_time_ms: f64,
    pub results: Vec<SimulationResult>,
}

/// Simulation statistics
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    pub simulations_run: usize,
    pub valid_inputs: usize,
    pub invalid_inputs: usize,
    pub by_type: HashMap<ProofType, TypeStats>,
}

impl SimulationStats {
    fn record(&mut self, proof_type: ProofType, valid: bool) {
        self.simulations_run += 1;
        if valid {
            self.valid_inputs += 1;
        } else {
            self.invalid_inputs += 1;
        }

        let type_stats = self.by_type.entry(proof_type).or_default();
        type_stats.count += 1;
        if valid {
            type_stats.valid += 1;
        } else {
            type_stats.invalid += 1;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeStats {
    pub count: usize,
    pub valid: usize,
    pub invalid: usize,
}

/// Mock proof verifier for testing
pub struct MockProofVerifier;

impl MockProofVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Check if bytes are a valid mock proof
    pub fn is_mock_proof(&self, data: &[u8]) -> bool {
        data.len() >= 4 && &data[0..4] == b"MOCK"
    }

    /// Verify a mock proof (always succeeds for valid mock proofs)
    pub fn verify(&self, data: &[u8]) -> MockVerificationResult {
        if !self.is_mock_proof(data) {
            return MockVerificationResult {
                is_valid: false,
                is_mock: false,
                proof_type: None,
            };
        }

        let proof_type = if data.len() > 4 {
            Some(match data[4] {
                0 => ProofType::Range,
                1 => ProofType::GradientIntegrity,
                2 => ProofType::IdentityAssurance,
                3 => ProofType::VoteEligibility,
                4 => ProofType::Membership,
                _ => return MockVerificationResult {
                    is_valid: false,
                    is_mock: true,
                    proof_type: None,
                },
            })
        } else {
            None
        };

        MockVerificationResult {
            is_valid: true,
            is_mock: true,
            proof_type,
        }
    }
}

impl Default for MockProofVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock verification result
#[derive(Debug, Clone)]
pub struct MockVerificationResult {
    pub is_valid: bool,
    pub is_mock: bool,
    pub proof_type: Option<ProofType>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_mode_default() {
        let mode = SimulationMode::default();
        assert_eq!(mode, SimulationMode::Estimate);
    }

    #[test]
    fn test_simulate_range_proof_valid() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);
        let result = simulator.simulate_range_proof(42, 0, 100).unwrap();

        assert!(result.is_valid());
        assert!(result.validation_errors.is_empty());
        assert!(result.estimated_time_ms > 0.0);
        assert!(result.estimated_size_bytes > 0);
    }

    #[test]
    fn test_simulate_range_proof_invalid() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        // Value out of range
        let result = simulator.simulate_range_proof(150, 0, 100).unwrap();
        assert!(!result.is_valid());
        assert!(!result.validation_errors.is_empty());

        // Invalid range
        let result = simulator.simulate_range_proof(50, 100, 0).unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_simulate_gradient_proof_valid() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);
        let result = simulator.simulate_gradient_proof(1000, 5.0).unwrap();

        assert!(result.is_valid());
        assert!(result.estimated_time_ms > 0.0);
    }

    #[test]
    fn test_simulate_gradient_proof_invalid() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        // Zero length
        let result = simulator.simulate_gradient_proof(0, 5.0).unwrap();
        assert!(!result.is_valid());

        // Invalid norm
        let result = simulator.simulate_gradient_proof(100, -1.0).unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_simulate_identity_proof() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        let result = simulator
            .simulate_identity_proof("did:mycelix:test", 3, ProofAssuranceLevel::E2)
            .unwrap();
        assert!(result.is_valid());

        // Not enough factors for level
        let result = simulator
            .simulate_identity_proof("did:mycelix:test", 1, ProofAssuranceLevel::E4)
            .unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_simulate_vote_proof() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        let voter = ProofVoterProfile {
            did: "did:mycelix:voter".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 1000.0,
            account_age_days: 90,
            participation_rate: 0.5,
            has_humanity_proof: true,
            fl_contributions: 20,
        };

        // Valid for standard
        let result = simulator
            .simulate_vote_proof(&voter, ProofProposalType::Standard)
            .unwrap();
        assert!(result.is_valid());

        // Valid for constitutional
        let result = simulator
            .simulate_vote_proof(&voter, ProofProposalType::Constitutional)
            .unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_simulate_vote_proof_ineligible() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        let voter = ProofVoterProfile {
            did: "did:mycelix:voter".to_string(),
            assurance_level: 1,
            matl_score: 0.3,
            stake: 100.0,
            account_age_days: 10,
            participation_rate: 0.1,
            has_humanity_proof: false,
            fl_contributions: 2,
        };

        // Ineligible for constitutional
        let result = simulator
            .simulate_vote_proof(&voter, ProofProposalType::Constitutional)
            .unwrap();
        assert!(!result.is_valid());
        assert!(result.validation_errors.len() >= 2); // Multiple failures
    }

    #[test]
    fn test_mock_proof_generation() {
        let mut simulator = ProofSimulator::new(SimulationMode::Mock);
        let result = simulator.simulate_range_proof(42, 0, 100).unwrap();

        assert!(result.is_valid());
        assert!(result.mock_proof.is_some());

        let mock = result.mock_proof.unwrap();
        assert!(&mock[0..4] == b"MOCK");
    }

    #[test]
    fn test_mock_proof_verification() {
        let mut simulator = ProofSimulator::new(SimulationMode::Mock);
        let result = simulator.simulate_range_proof(42, 0, 100).unwrap();
        let mock = result.mock_proof.unwrap();

        let verifier = MockProofVerifier::new();
        let verification = verifier.verify(&mock);

        assert!(verification.is_valid);
        assert!(verification.is_mock);
        assert_eq!(verification.proof_type, Some(ProofType::Range));
    }

    #[test]
    fn test_batch_simulation() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        let requests = vec![
            ProofRequest::Range {
                value: 42,
                min: 0,
                max: 100,
            },
            ProofRequest::Range {
                value: 150,
                min: 0,
                max: 100,
            }, // Invalid
            ProofRequest::Gradient {
                len: 1000,
                max_norm: 5.0,
            },
        ];

        let result = simulator.simulate_batch(&requests);

        assert_eq!(result.total_count, 3);
        assert_eq!(result.valid_count, 2);
        assert_eq!(result.invalid_count, 1);
        assert!(result.estimated_total_time_ms > 0.0);
    }

    #[test]
    fn test_simulation_stats() {
        let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

        simulator.simulate_range_proof(42, 0, 100).unwrap();
        simulator.simulate_range_proof(150, 0, 100).unwrap(); // Invalid
        simulator.simulate_gradient_proof(1000, 5.0).unwrap();

        let stats = simulator.stats();
        assert_eq!(stats.simulations_run, 3);
        assert_eq!(stats.valid_inputs, 2);
        assert_eq!(stats.invalid_inputs, 1);
    }

    #[test]
    fn test_security_level_estimates() {
        let base = CostEstimates::range();
        let high = base.with_security(SecurityLevel::High256);

        assert!(high.base_time_ms > base.base_time_ms);
        assert!(high.base_size_bytes > base.base_size_bytes);
    }

    #[test]
    fn test_cost_estimation() {
        let estimates = CostEstimates::range();

        let time_100 = estimates.estimate_time(100);
        let time_1000 = estimates.estimate_time(1000);

        assert!(time_1000 > time_100);
        assert!((time_1000 - time_100 - 900.0 * estimates.time_per_constraint_ms).abs() < 0.01);
    }
}
