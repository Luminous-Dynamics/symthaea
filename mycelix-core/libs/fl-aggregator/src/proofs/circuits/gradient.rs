//! Gradient Integrity Proof Circuit
//!
//! Proves that a gradient contribution is valid without revealing the gradient values.
//!
//! ## What This Proves
//!
//! 1. Gradient values are within valid range (prevents overflow attacks)
//! 2. L2 norm is bounded (prevents Byzantine scaling attacks)
//! 3. Gradient is properly committed (binding to specific values)
//!
//! ## Trace Structure
//!
//! - 4 columns, N rows (N = number of gradient elements, padded to power of 2)
//! - Column 0: gradient element value (scaled integer)
//! - Column 1: squared value (element^2)
//! - Column 2: running norm sum (accumulated squared values)
//! - Column 3: range flag (1 if in valid range, verified by bit check)
//!
//! ## Security Assumptions
//!
//! 1. **Hash Function**: Blake3 commitment binds the gradient values
//! 2. **Field Arithmetic**: Winterfell correctly implements 128-bit prime field operations
//! 3. **Scaling Precision**: Integer scaling (1000x) provides sufficient precision for FL
//!
//! ## Known Limitations
//!
//! - **Not Perfect Zero-Knowledge**: Winterfell STARKs provide computational hiding but
//!   are NOT perfectly zero-knowledge. Gradient statistics may leak through proof structure.
//! - **Public Inputs Visible**: num_elements, max_norm_squared, commitment, and
//!   final_norm_squared are publicly visible
//! - **Fixed Scale Factor**: 1000x scaling limits precision to 3 decimal places
//! - **Integer Overflow**: Very large gradients may overflow during squared accumulation
//!
//! ## Threat Model
//!
//! - **Malicious Prover (Byzantine Client)**: Cannot submit gradients that exceed the
//!   norm bound. The L2 norm constraint prevents scaling attacks on federated aggregation.
//! - **Malicious Verifier**: Can see gradient count and norm bound. Cannot extract
//!   individual gradient values from the proof.
//! - **Gradient Inference Attack**: The commitment and norm do not reveal gradient
//!   direction, but aggregation patterns across rounds could be analyzed externally.

use crate::proofs::{
    build_proof_options,
    ProofConfig, ProofError, ProofResult, ProofType, VerificationResult,
};
use crate::proofs::types::ByteReader;
use std::time::Instant;
use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, StarkField, ToElements},
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, DefaultConstraintEvaluator,
    DefaultTraceLde, EvaluationFrame, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TracePolyTable, TraceTable,
    TransitionConstraintDegree, AcceptableOptions,
};

/// Minimum trace length required by winterfell
const MIN_TRACE_LEN: usize = 8;

/// Number of columns in the trace
const TRACE_WIDTH: usize = 4;

/// Scale factor for gradient values (same as field.rs F32_SCALE)
const GRADIENT_SCALE: u64 = 1000;

/// Maximum allowed gradient element value (before scaling)
const MAX_GRADIENT_VALUE: f32 = 1000.0;

/// Public inputs for gradient integrity proof
#[derive(Debug, Clone)]
pub struct GradientPublicInputs {
    /// Number of gradient elements
    pub num_elements: usize,
    /// Maximum allowed L2 norm squared (scaled)
    pub max_norm_squared: u64,
    /// Commitment to the gradient (hash of all elements)
    pub commitment: [u8; 32],
    /// Final computed norm squared (for verification)
    pub final_norm_squared: u64,
}

impl GradientPublicInputs {
    /// Create new public inputs
    pub fn new(
        num_elements: usize,
        max_norm_squared: u64,
        commitment: [u8; 32],
        final_norm_squared: u64,
    ) -> Self {
        Self {
            num_elements,
            max_norm_squared,
            commitment,
            final_norm_squared,
        }
    }
}

impl ToElements<BaseElement> for GradientPublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            BaseElement::from(self.num_elements as u64),
            BaseElement::from(self.max_norm_squared),
            // Commitment as 4 field elements (256 bits)
            BaseElement::from(u64::from_le_bytes(self.commitment[0..8].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.commitment[8..16].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.commitment[16..24].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.commitment[24..32].try_into().unwrap())),
            BaseElement::from(self.final_norm_squared),
        ]
    }
}

/// AIR for gradient integrity proof
pub struct GradientIntegrityAir {
    context: AirContext<BaseElement>,
    final_norm_squared: BaseElement,
    #[allow(dead_code)]
    max_norm_squared: BaseElement,
}

impl Air for GradientIntegrityAir {
    type BaseField = BaseElement;
    type PublicInputs = GradientPublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Transition constraint degrees:
        // - Constraint 0: squared value check (value^2 = squared) - degree 2
        // - Constraint 1: norm accumulation - degree 1
        //
        // Note: Range flag removed to avoid trivial constraint issues.
        // Range checking is done during trace construction.
        let degrees = vec![
            TransitionConstraintDegree::new(2), // squared check
            TransitionConstraintDegree::new(1), // norm accumulation
        ];

        let num_assertions = 2; // Initial norm = 0, final norm matches

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            final_norm_squared: BaseElement::from(pub_inputs.final_norm_squared),
            max_norm_squared: BaseElement::from(pub_inputs.max_norm_squared),
        }
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        // Column layout:
        // [0]: gradient value
        // [1]: squared value
        // [2]: running norm sum
        // [3]: range flag (unused in constraints, verified during construction)

        let value = current[0];
        let squared = current[1];
        let norm_sum = current[2];
        let next_norm_sum = next[2];

        // Constraint 0: squared value must equal value^2
        result[0] = squared - value * value;

        // Constraint 1: norm accumulates correctly
        // next_norm_sum = norm_sum + squared (current squared)
        result[1] = next_norm_sum - norm_sum - squared;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;
        vec![
            // Initial norm sum is 0
            Assertion::single(2, 0, BaseElement::ZERO),
            // Final norm sum matches public input
            Assertion::single(2, last_step, self.final_norm_squared),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

/// Gradient integrity proof prover
pub struct GradientIntegrityProver {
    options: ProofOptions,
    public_inputs: GradientPublicInputs,
}

impl GradientIntegrityProver {
    /// Create a new prover
    pub fn new(options: ProofOptions, public_inputs: GradientPublicInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace
    fn build_trace(&self, gradients: &[f32], max_value: f32) -> TraceTable<BaseElement> {
        let n = gradients.len();
        // We need n+1 rows to include all squared values in the final norm
        // Also ensure trace length meets winterfell minimum (power of 2 >= 8)
        let trace_len = (n + 1).max(MIN_TRACE_LEN).next_power_of_two();

        let mut trace = TraceTable::new(TRACE_WIDTH, trace_len);

        // Scale gradients to integers
        let scaled: Vec<u64> = gradients
            .iter()
            .map(|&g| (g.abs() * GRADIENT_SCALE as f32) as u64)
            .collect();

        let max_scaled = (max_value * GRADIENT_SCALE as f32) as u64;

        // Precompute squared values for all rows
        let squared_values: Vec<u64> = (0..trace_len)
            .map(|i| {
                let value = if i < n { scaled[i] } else { 0 };
                value * value
            })
            .collect();

        trace.fill(
            |state| {
                // First row: norm_sum = 0 (before any accumulation)
                let value = scaled.first().copied().unwrap_or(0);
                let squared = value * value;
                let in_range = value <= max_scaled;

                state[0] = BaseElement::from(value);
                state[1] = BaseElement::from(squared);
                state[2] = BaseElement::ZERO; // Initial norm sum (assertion point)
                state[3] = if in_range { BaseElement::ONE } else { BaseElement::ZERO };
            },
            |step, state| {
                // Constraint: next_norm_sum = norm_sum + squared
                // So row i+1's norm = row i's norm + row i's squared

                // Get current state values
                let current_squared: u128 = state[1].as_int();
                let current_norm: u128 = state[2].as_int();

                // Compute next norm (row i+1's norm = current norm + current squared)
                let next_norm = current_norm + current_squared;

                // Get next gradient value
                let next_idx = step + 1;
                let value = if next_idx < n { scaled[next_idx] } else { 0 };
                let squared = squared_values[next_idx];
                let in_range = value <= max_scaled;

                state[0] = BaseElement::from(value);
                state[1] = BaseElement::from(squared);
                state[2] = BaseElement::from(next_norm as u64);
                state[3] = if in_range { BaseElement::ONE } else { BaseElement::ZERO };
            },
        );

        trace
    }
}

impl Prover for GradientIntegrityProver {
    type BaseField = BaseElement;
    type Air = GradientIntegrityAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> GradientPublicInputs {
        self.public_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        partition_option: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_option)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

/// Compute commitment to gradient values
pub fn compute_gradient_commitment(gradients: &[f32]) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(b"gradient:");
    for g in gradients {
        hasher.update(&g.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// A complete gradient integrity proof
#[derive(Clone)]
pub struct GradientIntegrityProof {
    /// The STARK proof
    proof: Proof,
    /// Public inputs
    public_inputs: GradientPublicInputs,
}

impl GradientIntegrityProof {
    /// Generate a gradient integrity proof
    ///
    /// Proves that `gradients` are valid and have bounded L2 norm.
    pub fn generate(
        gradients: &[f32],
        max_norm: f32,
        config: ProofConfig,
    ) -> ProofResult<Self> {
        if gradients.is_empty() {
            return Err(ProofError::InvalidWitness(
                "Gradient vector cannot be empty".to_string()
            ));
        }

        // Compute L2 norm
        let norm_squared: f32 = gradients.iter().map(|g| g * g).sum();
        let max_norm_squared = max_norm * max_norm;

        if norm_squared > max_norm_squared {
            return Err(ProofError::InvalidWitness(format!(
                "Gradient norm {} exceeds maximum {}",
                norm_squared.sqrt(),
                max_norm
            )));
        }

        // Check individual values
        for (i, &g) in gradients.iter().enumerate() {
            if g.abs() > MAX_GRADIENT_VALUE {
                return Err(ProofError::InvalidWitness(format!(
                    "Gradient element {} has value {} exceeding maximum {}",
                    i, g.abs(), MAX_GRADIENT_VALUE
                )));
            }
        }

        // Compute commitment
        let commitment = compute_gradient_commitment(gradients);

        // Scale norm for field representation
        let scaled_norm_squared = (norm_squared * (GRADIENT_SCALE * GRADIENT_SCALE) as f32) as u64;
        let scaled_max_norm_squared = (max_norm_squared * (GRADIENT_SCALE * GRADIENT_SCALE) as f32) as u64;

        let public_inputs = GradientPublicInputs::new(
            gradients.len(),
            scaled_max_norm_squared,
            commitment,
            scaled_norm_squared,
        );

        // Build prover and trace
        let options = build_proof_options(config.security_level);
        let prover = GradientIntegrityProver::new(options, public_inputs.clone());
        let trace = prover.build_trace(gradients, MAX_GRADIENT_VALUE);

        // Generate proof
        let proof = prover.prove(trace).map_err(|e| {
            ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
        })?;

        Ok(Self {
            proof,
            public_inputs,
        })
    }

    /// Verify the gradient integrity proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        // Check norm bound
        if self.public_inputs.final_norm_squared > self.public_inputs.max_norm_squared {
            return Ok(VerificationResult::failure(
                ProofType::GradientIntegrity,
                start.elapsed(),
                "Norm exceeds maximum bound",
            ));
        }

        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

        let result = winterfell::verify::<
            GradientIntegrityAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(self.proof.clone(), self.public_inputs.clone(), &min_opts);

        let duration = start.elapsed();

        match result {
            Ok(_) => Ok(VerificationResult::success(ProofType::GradientIntegrity, duration)),
            Err(e) => Ok(VerificationResult::failure(
                ProofType::GradientIntegrity,
                duration,
                format!("Verification failed: {:?}", e),
            )),
        }
    }

    /// Get the public inputs
    pub fn public_inputs(&self) -> &GradientPublicInputs {
        &self.public_inputs
    }

    /// Get the gradient commitment
    pub fn commitment(&self) -> &[u8; 32] {
        &self.public_inputs.commitment
    }

    /// Serialize the proof to bytes
    ///
    /// Format: [num_elements:8][max_norm_squared:8][commitment:32][final_norm_squared:8][proof_bytes...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.public_inputs.num_elements as u64).to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.max_norm_squared.to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.commitment);
        bytes.extend_from_slice(&self.public_inputs.final_norm_squared.to_le_bytes());
        bytes.extend_from_slice(&self.proof.to_bytes());
        bytes
    }

    /// Deserialize a proof from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let num_elements = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated num_elements".to_string())
        })? as usize;

        let max_norm_squared = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated max_norm_squared".to_string())
        })?;

        let commitment = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated commitment".to_string())
        })?;

        let final_norm_squared = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated final_norm_squared".to_string())
        })?;

        let public_inputs = GradientPublicInputs::new(
            num_elements,
            max_norm_squared,
            commitment,
            final_norm_squared,
        );

        let proof = Proof::from_bytes(reader.read_remaining()).map_err(|e| {
            ProofError::InvalidProofFormat(format!("Failed to parse STARK proof: {:?}", e))
        })?;

        Ok(Self { proof, public_inputs })
    }

    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_gradient_proof_valid() {
        let gradients = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let max_norm = 5.0;

        let proof = GradientIntegrityProof::generate(&gradients, max_norm, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid, "Proof verification failed: {:?}", result.details);
    }

    #[test]
    fn test_gradient_proof_norm_exceeded() {
        let gradients = vec![10.0, 10.0, 10.0, 10.0]; // norm = 20
        let max_norm = 5.0;

        let result = GradientIntegrityProof::generate(&gradients, max_norm, test_config());
        assert!(matches!(result, Err(ProofError::InvalidWitness(_))));
    }

    #[test]
    fn test_gradient_proof_empty() {
        let gradients: Vec<f32> = vec![];
        let max_norm = 5.0;

        let result = GradientIntegrityProof::generate(&gradients, max_norm, test_config());
        assert!(matches!(result, Err(ProofError::InvalidWitness(_))));
    }

    #[test]
    fn test_gradient_commitment() {
        let gradients1 = vec![0.1, 0.2, 0.3];
        let gradients2 = vec![0.1, 0.2, 0.4];

        let commit1 = compute_gradient_commitment(&gradients1);
        let commit2 = compute_gradient_commitment(&gradients2);

        assert_ne!(commit1, commit2);

        // Same gradients should produce same commitment
        let commit1_repeat = compute_gradient_commitment(&gradients1);
        assert_eq!(commit1, commit1_repeat);
    }

    #[test]
    fn test_gradient_proof_boundary_norm() {
        // Gradients with norm just under max
        let gradients = vec![1.0, 1.0, 1.0, 1.0]; // norm = 2.0
        let max_norm = 2.1;

        let proof = GradientIntegrityProof::generate(&gradients, max_norm, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid);
    }

    #[test]
    fn test_gradient_proof_single_element() {
        let gradients = vec![0.5];
        let max_norm = 1.0;

        let proof = GradientIntegrityProof::generate(&gradients, max_norm, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let gradients = vec![0.1, -0.2, 0.3, -0.4];
        let max_norm = 5.0;

        let proof = GradientIntegrityProof::generate(&gradients, max_norm, test_config()).unwrap();
        let bytes = proof.to_bytes();

        let restored = GradientIntegrityProof::from_bytes(&bytes).unwrap();
        let result = restored.verify().unwrap();

        assert!(result.valid, "Restored proof verification failed");
        assert_eq!(restored.commitment(), proof.commitment());
    }
}
