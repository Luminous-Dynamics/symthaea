//! Generic Zero-Knowledge Proof System Abstraction
//!
//! This module provides a pluggable proof system architecture that supports
//! multiple ZK backends (simulation, RISC-0, future SP1/Halo2) with a unified API.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        ProofSystem Trait                            │
//! │  - prove(&self, statement, witness) -> Receipt                      │
//! │  - verify(&self, receipt) -> bool                                   │
//! │  - backend_name() -> &str                                           │
//! └─────────────────────────────────────────────────────────────────────┘
//!                                    │
//!          ┌─────────────────────────┼─────────────────────────┐
//!          ▼                         ▼                         ▼
//!  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
//!  │  Simulation   │        │    RISC-0     │        │    Future     │
//!  │   Backend     │        │    Backend    │        │    Backend    │
//!  │ (testing only)│        │ (production)  │        │  (SP1, etc.)  │
//!  └───────────────┘        └───────────────┘        └───────────────┘
//! ```
//!
//! # Security Model
//!
//! - **Simulation**: NO cryptographic guarantees - use only for testing
//! - **RISC-0**: 128-bit computational soundness via zkSTARKs
//! - All backends share the same API, but only production backends provide security
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::zkproof::proof_system::*;
//!
//! // Create a proof system (automatically selects best available backend)
//! let system = ProofSystemBuilder::new()
//!     .with_timeout(Duration::from_secs(300))
//!     .build()?;
//!
//! // Generate a proof
//! let receipt = system.prove(&statement, &witness)?;
//!
//! // Verify the proof
//! assert!(system.verify(&receipt)?);
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Core Traits
// ============================================================================

/// A statement that can be proven in zero-knowledge.
///
/// Statements define WHAT is being proven without revealing HOW.
/// The witness (private input) provides the secret knowledge that
/// satisfies the statement.
pub trait ProofStatement: Clone + Send + Sync {
    /// Human-readable description of what this statement proves
    fn description(&self) -> String;

    /// Unique identifier for this statement type
    fn statement_type_id(&self) -> &'static str;

    /// Serialize statement for proof generation
    fn to_bytes(&self) -> Vec<u8>;
}

/// Private witness data that satisfies a statement.
///
/// Witnesses contain the secret information that proves a statement is true.
/// They are NEVER revealed - only the proof that they exist is shared.
pub trait ProofWitness: Clone + Send + Sync {
    /// The statement type this witness satisfies
    type Statement: ProofStatement;

    /// Evaluate if this witness satisfies the given statement
    fn satisfies(&self, statement: &Self::Statement) -> bool;

    /// Serialize witness for proof generation (SENSITIVE - handle with care)
    fn to_bytes(&self) -> Vec<u8>;
}

/// A zero-knowledge proof receipt.
///
/// Contains the proof bytes and public outputs. The receipt can be
/// verified without access to the original witness.
pub trait ProofReceipt: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// The statement type this receipt proves
    type Statement: ProofStatement;

    /// Get the public output from the proof
    fn public_output(&self) -> &ProofOutput;

    /// Get the raw proof bytes
    fn proof_bytes(&self) -> &[u8];

    /// Check if this is a simulation proof (NO cryptographic guarantees)
    fn is_simulation(&self) -> bool;

    /// Check if the statement was satisfied
    fn statement_satisfied(&self) -> bool;
}

/// Public output from a proof.
///
/// This is the verifiable, public information that results from
/// proof generation. It does NOT include any private witness data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOutput {
    /// Whether the statement was satisfied
    pub satisfied: bool,
    /// Commitment to the witness (hides actual values)
    pub witness_commitment: [u8; 32],
    /// Timestamp when proof was generated
    pub timestamp: u64,
    /// Statement type identifier
    pub statement_type: String,
    /// Additional public data (statement-specific)
    pub public_data: Vec<u8>,
}

impl ProofOutput {
    /// Create a new proof output
    pub fn new(
        satisfied: bool,
        witness_commitment: [u8; 32],
        statement_type: impl Into<String>,
    ) -> Self {
        Self {
            satisfied,
            witness_commitment,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            statement_type: statement_type.into(),
            public_data: Vec::new(),
        }
    }

    /// Add public data to the output
    pub fn with_public_data(mut self, data: Vec<u8>) -> Self {
        self.public_data = data;
        self
    }
}

// ============================================================================
// Proof System Trait
// ============================================================================

/// Error types for proof system operations
#[derive(Debug, Clone, PartialEq)]
pub enum ProofSystemError {
    /// Proof generation failed
    ProofGenerationFailed(String),
    /// Proof verification failed
    VerificationFailed(String),
    /// Backend not available
    BackendUnavailable(String),
    /// Invalid statement or witness
    InvalidInput(String),
    /// Timeout during proof generation
    Timeout(Duration),
    /// Simulation proof rejected in production context
    SimulationRejected(String),
    /// Serialization/deserialization error
    SerializationError(String),
}

impl std::fmt::Display for ProofSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProofGenerationFailed(e) => write!(f, "Proof generation failed: {}", e),
            Self::VerificationFailed(e) => write!(f, "Verification failed: {}", e),
            Self::BackendUnavailable(e) => write!(f, "Backend unavailable: {}", e),
            Self::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            Self::Timeout(d) => write!(f, "Proof generation timed out after {:?}", d),
            Self::SimulationRejected(e) => write!(f, "Simulation proof rejected: {}", e),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for ProofSystemError {}

/// Statistics about proof generation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofStats {
    /// Time to generate proof in milliseconds
    pub generation_time_ms: u64,
    /// Size of proof in bytes
    pub proof_size_bytes: usize,
    /// Backend used for proof generation
    pub backend: String,
    /// Whether proof was generated in simulation mode
    pub is_simulation: bool,
}

/// A zero-knowledge proof system.
///
/// This trait defines the interface for generating and verifying
/// zero-knowledge proofs. Different backends implement this trait
/// to provide various security/performance tradeoffs.
pub trait ProofSystem: Send + Sync {
    /// The statement type this system proves
    type Statement: ProofStatement;
    /// The witness type this system uses
    type Witness: ProofWitness<Statement = Self::Statement>;
    /// The receipt type this system produces
    type Receipt: ProofReceipt<Statement = Self::Statement>;

    /// Get the backend name (e.g., "simulation", "risc0", "sp1")
    fn backend_name(&self) -> &'static str;

    /// Check if this is a simulation backend (NO cryptographic guarantees)
    fn is_simulation(&self) -> bool;

    /// Generate a proof that witness satisfies statement
    fn prove(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<Self::Receipt, ProofSystemError>;

    /// Verify a proof receipt
    fn verify(&self, receipt: &Self::Receipt) -> Result<bool, ProofSystemError>;

    /// Get statistics from the last proof operation
    fn last_proof_stats(&self) -> Option<ProofStats>;
}

// ============================================================================
// Backend Configuration
// ============================================================================

/// Configuration for proof system backends
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Timeout for proof generation
    pub timeout: Duration,
    /// URL for external prover service (if applicable)
    pub prover_url: Option<String>,
    /// Whether to allow simulation mode
    pub allow_simulation: bool,
    /// Maximum proof size in bytes
    pub max_proof_size: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            prover_url: std::env::var("RISC0_PROVER_URL").ok(),
            allow_simulation: cfg!(feature = "simulation"),
            max_proof_size: 10 * 1024 * 1024, // 10 MB
        }
    }
}

/// Available proof system backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// Simulation backend - NO cryptographic security
    Simulation,
    /// RISC-0 zkSTARK backend
    Risc0,
    /// RISC-0 via Bonsai cloud service
    Risc0Bonsai,
    // Future backends:
    // Sp1,
    // Halo2,
}

impl BackendType {
    /// Check if this backend provides cryptographic security
    pub fn is_secure(&self) -> bool {
        !matches!(self, Self::Simulation)
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Simulation => "simulation",
            Self::Risc0 => "risc0",
            Self::Risc0Bonsai => "risc0-bonsai",
        }
    }
}

// ============================================================================
// Generic Receipt Implementation
// ============================================================================

/// A generic proof receipt that works with any statement type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericReceipt<S: ProofStatement> {
    /// Public output from the proof
    pub output: ProofOutput,
    /// Raw proof bytes
    pub proof_bytes: Vec<u8>,
    /// Backend that generated this proof
    pub backend: BackendType,
    /// Proof generation statistics
    pub stats: ProofStats,
    /// The original statement (for verification context)
    #[serde(skip)]
    pub _statement: std::marker::PhantomData<S>,
}

impl<S: ProofStatement> GenericReceipt<S> {
    /// Create a new receipt
    pub fn new(
        output: ProofOutput,
        proof_bytes: Vec<u8>,
        backend: BackendType,
        stats: ProofStats,
    ) -> Self {
        Self {
            output,
            proof_bytes,
            backend,
            stats,
            _statement: std::marker::PhantomData,
        }
    }
}

impl<S: ProofStatement + Serialize + for<'de> Deserialize<'de>> ProofReceipt for GenericReceipt<S> {
    type Statement = S;

    fn public_output(&self) -> &ProofOutput {
        &self.output
    }

    fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    fn is_simulation(&self) -> bool {
        matches!(self.backend, BackendType::Simulation)
    }

    fn statement_satisfied(&self) -> bool {
        self.output.satisfied
    }
}

// ============================================================================
// Simulation Backend
// ============================================================================

/// Simulation marker for proof bytes
const SIMULATION_MARKER: &[u8; 4] = b"SIMU";

/// Simulation proof system backend.
///
/// **SECURITY WARNING**: This backend provides NO cryptographic guarantees!
/// Proofs can be trivially forged. Use only for testing.
#[derive(Debug, Clone)]
pub struct SimulationBackend {
    _config: BackendConfig,
    last_stats: std::sync::Arc<std::sync::Mutex<Option<ProofStats>>>,
}

impl SimulationBackend {
    /// Create a new simulation backend
    pub fn new(config: BackendConfig) -> Self {
        Self {
            _config: config,
            last_stats: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(BackendConfig::default())
    }

    /// Get statistics from the last proof (non-generic version)
    pub fn get_last_stats(&self) -> Option<ProofStats> {
        self.last_stats.lock().ok().and_then(|s| s.clone())
    }

    /// Verify a receipt by checking if it starts with simulation marker
    ///
    /// Returns Ok(true) if the proof structure is valid (regardless of statement satisfaction).
    /// Check `receipt.output.satisfied` to see if the statement was satisfied.
    pub fn verify_simulation_receipt<S: ProofStatement>(
        &self,
        receipt: &GenericReceipt<S>,
    ) -> Result<bool, ProofSystemError> {
        // Just check if it's a valid simulation proof structure
        Ok(receipt.proof_bytes.starts_with(SIMULATION_MARKER))
    }

    /// Update the last stats (for use by wrappers)
    pub fn set_last_stats(&self, stats: ProofStats) {
        if let Ok(mut last) = self.last_stats.lock() {
            *last = Some(stats);
        }
    }
}

impl<S, W> ProofSystemBackend<S, W> for SimulationBackend
where
    S: ProofStatement + Serialize + for<'de> Deserialize<'de>,
    W: ProofWitness<Statement = S>,
{
    type Receipt = GenericReceipt<S>;

    fn prove(&self, statement: &S, witness: &W) -> Result<Self::Receipt, ProofSystemError> {
        let start = Instant::now();

        // Evaluate statement satisfaction
        let satisfied = witness.satisfies(statement);

        // Compute witness commitment (simulation uses simple hash)
        let witness_bytes = witness.to_bytes();
        let witness_commitment = compute_simple_commitment(&witness_bytes);

        // Create output
        let output = ProofOutput::new(satisfied, witness_commitment, statement.statement_type_id());

        // Create simulation proof bytes
        let mut proof_bytes = SIMULATION_MARKER.to_vec();
        proof_bytes.extend(bincode::serialize(&output).unwrap_or_default());

        let generation_time_ms = start.elapsed().as_millis() as u64;
        let stats = ProofStats {
            generation_time_ms,
            proof_size_bytes: proof_bytes.len(),
            backend: "simulation".to_string(),
            is_simulation: true,
        };

        // Store stats
        if let Ok(mut last) = self.last_stats.lock() {
            *last = Some(stats.clone());
        }

        Ok(GenericReceipt::new(
            output,
            proof_bytes,
            BackendType::Simulation,
            stats,
        ))
    }

    fn verify(&self, receipt: &Self::Receipt) -> Result<bool, ProofSystemError> {
        // Simulation verification just checks the marker and satisfaction
        if !receipt.proof_bytes.starts_with(SIMULATION_MARKER) {
            return Ok(false);
        }
        Ok(receipt.output.satisfied)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Simulation
    }

    fn last_stats(&self) -> Option<ProofStats> {
        self.last_stats.lock().ok().and_then(|s| s.clone())
    }
}

/// Backend-specific trait (internal implementation detail)
pub trait ProofSystemBackend<S: ProofStatement, W: ProofWitness<Statement = S>>:
    Send + Sync
{
    /// Receipt type produced by this backend
    type Receipt: ProofReceipt<Statement = S>;

    /// Generate a proof
    fn prove(&self, statement: &S, witness: &W) -> Result<Self::Receipt, ProofSystemError>;

    /// Verify a receipt
    fn verify(&self, receipt: &Self::Receipt) -> Result<bool, ProofSystemError>;

    /// Get backend type
    fn backend_type(&self) -> BackendType;

    /// Get last proof stats
    fn last_stats(&self) -> Option<ProofStats>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute a simple commitment for simulation mode
fn compute_simple_commitment(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(b"mycelix-proof-commitment-v1");
    hasher.update(data);
    hasher.finalize().into()
}

/// Check if proof bytes represent a simulation proof
pub fn is_simulation_proof(proof_bytes: &[u8]) -> bool {
    proof_bytes.starts_with(SIMULATION_MARKER)
        || proof_bytes.starts_with(b"SIMU")
        || proof_bytes == [0xDE, 0xAD, 0xBE, 0xEF] // Legacy marker
        || proof_bytes.is_empty()
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for creating proof systems with the appropriate backend.
#[derive(Debug, Clone)]
pub struct ProofSystemBuilder {
    config: BackendConfig,
    preferred_backend: Option<BackendType>,
}

impl ProofSystemBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: BackendConfig::default(),
            preferred_backend: None,
        }
    }

    /// Set the timeout for proof generation
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the prover URL for external services
    pub fn with_prover_url(mut self, url: impl Into<String>) -> Self {
        self.config.prover_url = Some(url.into());
        self
    }

    /// Set the preferred backend type
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Allow simulation mode (default based on feature flags)
    pub fn allow_simulation(mut self, allow: bool) -> Self {
        self.config.allow_simulation = allow;
        self
    }

    /// Build a simulation backend (for testing)
    #[cfg(feature = "simulation")]
    pub fn build_simulation(&self) -> SimulationBackend {
        SimulationBackend::new(self.config.clone())
    }

    /// Get the best available backend type based on configuration
    pub fn best_available_backend(&self) -> BackendType {
        if let Some(preferred) = self.preferred_backend {
            return preferred;
        }

        // Check for RISC-0 availability
        #[cfg(feature = "risc0")]
        {
            if self.config.prover_url.is_some() || std::env::var("BONSAI_API_KEY").is_ok() {
                return BackendType::Risc0Bonsai;
            }
            return BackendType::Risc0;
        }

        // Fall back to simulation
        #[cfg(feature = "simulation")]
        {
            BackendType::Simulation
        }

        #[cfg(not(any(feature = "simulation", feature = "risc0")))]
        {
            panic!("No proof backend available. Enable 'simulation' or 'risc0' feature.");
        }
    }
}

impl Default for ProofSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test statement
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestStatement {
        threshold: u64,
    }

    impl ProofStatement for TestStatement {
        fn description(&self) -> String {
            format!("Value exceeds {}", self.threshold)
        }

        fn statement_type_id(&self) -> &'static str {
            "test-threshold"
        }

        fn to_bytes(&self) -> Vec<u8> {
            bincode::serialize(self).unwrap_or_default()
        }
    }

    // Test witness
    #[derive(Clone, Debug)]
    struct TestWitness {
        value: u64,
    }

    impl ProofWitness for TestWitness {
        type Statement = TestStatement;

        fn satisfies(&self, statement: &Self::Statement) -> bool {
            self.value > statement.threshold
        }

        fn to_bytes(&self) -> Vec<u8> {
            self.value.to_le_bytes().to_vec()
        }
    }

    #[test]
    fn test_simulation_backend_satisfies() {
        let backend = SimulationBackend::new_default();
        let statement = TestStatement { threshold: 50 };
        let witness = TestWitness { value: 100 };

        let receipt: GenericReceipt<TestStatement> = backend.prove(&statement, &witness).unwrap();

        assert!(receipt.output.satisfied);
        assert!(receipt.is_simulation());
        assert!(backend.verify_simulation_receipt(&receipt).unwrap());
    }

    #[test]
    fn test_simulation_backend_not_satisfies() {
        let backend = SimulationBackend::new_default();
        let statement = TestStatement { threshold: 50 };
        let witness = TestWitness { value: 30 };

        let receipt: GenericReceipt<TestStatement> = backend.prove(&statement, &witness).unwrap();

        assert!(!receipt.output.satisfied);
        assert!(backend.verify_simulation_receipt(&receipt).unwrap()); // Verification succeeds, but statement not satisfied
    }

    #[test]
    fn test_proof_stats() {
        let backend = SimulationBackend::new_default();
        let statement = TestStatement { threshold: 50 };
        let witness = TestWitness { value: 100 };

        let _receipt: GenericReceipt<TestStatement> = backend.prove(&statement, &witness).unwrap();

        let stats = backend.get_last_stats().unwrap();
        assert!(stats.generation_time_ms >= 0);
        assert!(stats.proof_size_bytes > 0);
        assert_eq!(stats.backend, "simulation");
        assert!(stats.is_simulation);
    }

    #[test]
    fn test_is_simulation_proof() {
        assert!(is_simulation_proof(b"SIMU"));
        assert!(is_simulation_proof(&[0xDE, 0xAD, 0xBE, 0xEF]));
        assert!(is_simulation_proof(&[]));
        assert!(!is_simulation_proof(b"REAL_PROOF_DATA"));
    }

    #[test]
    fn test_builder() {
        let builder = ProofSystemBuilder::new()
            .with_timeout(Duration::from_secs(60))
            .allow_simulation(true);

        let backend_type = builder.best_available_backend();
        // Should be simulation in test mode
        assert_eq!(backend_type, BackendType::Simulation);
    }

    #[test]
    fn test_backend_type() {
        assert!(!BackendType::Simulation.is_secure());
        assert!(BackendType::Risc0.is_secure());
        assert!(BackendType::Risc0Bonsai.is_secure());

        assert_eq!(BackendType::Simulation.name(), "simulation");
        assert_eq!(BackendType::Risc0.name(), "risc0");
    }
}
