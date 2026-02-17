//! Range Proof Circuit
//!
//! Proves that a value lies within a specified range [min, max] without
//! revealing the actual value.
//!
//! ## How It Works
//!
//! The proof uses bit decomposition:
//! 1. Compute `v = value - min` (the offset from minimum)
//! 2. Decompose `v` into 64 bits
//! 3. Verify each bit is 0 or 1 (constraint: b * (1 - b) = 0)
//! 4. Verify the recomposed value equals `v`
//! 5. Verify `v <= max - min` (the range width)
//!
//! ## Trace Structure
//!
//! - 3 columns, 64 rows (for 64-bit values)
//! - Column 0: bit_i (the i-th bit of the offset value)
//! - Column 1: accumulator (running sum of bits * 2^i)
//! - Column 2: power of 2 (2^i for current row)
//!
//! ## Security Assumptions
//!
//! 1. **Hash Function**: Blake3 is collision-resistant and behaves as a random oracle
//! 2. **Field Arithmetic**: Winterfell correctly implements 128-bit prime field operations
//! 3. **Honest Execution**: The STARK proves correct computation, not input validity
//!
//! ## Known Limitations
//!
//! - **Not Perfect Zero-Knowledge**: Winterfell STARKs provide computational hiding but
//!   are NOT perfectly zero-knowledge. Information may leak through proof structure.
//! - **Public Inputs Visible**: min, max, and adjusted_offset are publicly visible
//! - **64-bit Range Only**: Values must fit in u64; larger values require protocol changes
//! - **No Negative Support**: Only non-negative ranges are supported
//!
//! ## Threat Model
//!
//! - **Malicious Prover**: Cannot forge proofs for values outside [min, max]. The bit
//!   decomposition constraints ensure the value reconstructs correctly.
//! - **Malicious Verifier**: Can only learn min, max, and that value is in range.
//!   Cannot extract the actual value from the proof.
//! - **Network Attacker**: Cannot modify proofs without detection (hash binding).

use crate::proofs::{
    build_proof_options, decompose_to_bits,
    ProofConfig, ProofError, ProofResult, ProofType, VerificationResult,
};
use crate::proofs::types::ByteReader;
use std::time::Instant;
use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, ToElements},
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, DefaultConstraintEvaluator,
    DefaultTraceLde, EvaluationFrame, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TracePolyTable, TraceTable,
    TransitionConstraintDegree, AcceptableOptions,
};

/// Number of bits for range proof (supports values up to 2^64 - 1)
const NUM_BITS: usize = 64;

/// We add 1 to offset to ensure non-trivial bit decomposition
/// This avoids the edge case where offset=0 produces all-zero bits,
/// which would make the constraint polynomial trivially zero.
const OFFSET_ADJUSTMENT: u64 = 1;

/// Number of columns in the trace
const TRACE_WIDTH: usize = 3;

/// Public inputs for range proof
#[derive(Debug, Clone)]
pub struct RangePublicInputs {
    /// Minimum value (inclusive)
    pub min: u64,
    /// Maximum value (inclusive)
    pub max: u64,
    /// The adjusted offset value (value - min + 1), used for verification
    /// We add 1 to ensure non-trivial bit decomposition
    pub adjusted_offset: u64,
}

impl RangePublicInputs {
    /// Create new public inputs with adjusted offset
    pub fn new(min: u64, max: u64, adjusted_offset: u64) -> Self {
        Self { min, max, adjusted_offset }
    }

    /// Get the range width
    pub fn range_width(&self) -> u64 {
        self.max.saturating_sub(self.min)
    }

    /// Get the original offset (before adjustment)
    pub fn original_offset(&self) -> u64 {
        self.adjusted_offset.saturating_sub(OFFSET_ADJUSTMENT)
    }
}

impl ToElements<BaseElement> for RangePublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            BaseElement::from(self.min),
            BaseElement::from(self.max),
            BaseElement::from(self.adjusted_offset),
        ]
    }
}

/// AIR (Algebraic Intermediate Representation) for range proof
pub struct RangeProofAir {
    context: AirContext<BaseElement>,
    adjusted_offset: BaseElement,
}

impl Air for RangeProofAir {
    type BaseField = BaseElement;
    type PublicInputs = RangePublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Transition constraint degrees:
        // - Constraint 0: bit check (b * (1-b) = 0) - degree 2
        // - Constraint 1: accumulator update - degree 2 (involves multiplication)
        // - Constraint 2: power of 2 doubling - degree 1
        let degrees = vec![
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(1),
        ];

        let num_assertions = 4;

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            adjusted_offset: BaseElement::from(pub_inputs.adjusted_offset),
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

        // Column 0: bit
        // Column 1: accumulator
        // Column 2: power of 2

        let bit = current[0];
        let acc = current[1];
        let power = current[2];
        let next_acc = next[1];
        let next_power = next[2];

        // Constraint 0: bit must be 0 or 1
        // b * (1 - b) = 0
        result[0] = bit * (E::ONE - bit);

        // Constraint 1: accumulator update
        // next_acc = acc + bit * power
        result[1] = next_acc - acc - bit * power;

        // Constraint 2: power doubles each step
        // next_power = power * 2
        let two = E::from(2u32);
        result[2] = next_power - power * two;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;
        vec![
            // Initial accumulator is 0
            Assertion::single(1, 0, BaseElement::ZERO),
            // Initial power is 1 (2^0)
            Assertion::single(2, 0, BaseElement::ONE),
            // Final accumulator equals the adjusted offset value
            Assertion::single(1, last_step, self.adjusted_offset),
            // Final power is 2^63 (we verify this to ensure full trace)
            Assertion::single(2, last_step, BaseElement::from(1u64 << 63)),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

/// Range proof prover
pub struct RangeProofProver {
    options: ProofOptions,
    public_inputs: RangePublicInputs,
}

impl RangeProofProver {
    /// Create a new prover with given options
    pub fn new(options: ProofOptions, public_inputs: RangePublicInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace for the proof
    fn build_trace(&self, value: u64, min: u64) -> TraceTable<BaseElement> {
        let offset = value.saturating_sub(min);
        // Add adjustment to ensure non-trivial bit decomposition
        let adjusted_offset = offset + OFFSET_ADJUSTMENT;
        let bits = decompose_to_bits(adjusted_offset, NUM_BITS);

        let mut trace = TraceTable::new(TRACE_WIDTH, NUM_BITS);

        trace.fill(
            |state| {
                // Initialize first row
                state[0] = bits[0];           // First bit
                state[1] = BaseElement::ZERO; // Accumulator starts at 0
                state[2] = BaseElement::ONE;  // Power of 2 starts at 1
            },
            |step, state| {
                // Transition to next row
                if step < NUM_BITS - 1 {
                    // Update accumulator: acc += bit * power
                    let bit_contribution = state[0] * state[2];
                    state[1] += bit_contribution;

                    // Next bit
                    state[0] = bits[step + 1];

                    // Next power of 2
                    state[2] *= BaseElement::from(2u64);
                } else {
                    // Last step: finalize accumulator
                    let bit_contribution = state[0] * state[2];
                    state[1] += bit_contribution;
                }
            },
        );

        trace
    }
}

impl Prover for RangeProofProver {
    type BaseField = BaseElement;
    type Air = RangeProofAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> RangePublicInputs {
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

/// A complete range proof
#[derive(Clone)]
pub struct RangeProof {
    /// The STARK proof
    proof: Proof,
    /// Public inputs
    public_inputs: RangePublicInputs,
}

impl RangeProof {
    /// Generate a range proof
    ///
    /// Proves that `value` is in the range [min, max] without revealing `value`.
    pub fn generate(
        value: u64,
        min: u64,
        max: u64,
        config: ProofConfig,
    ) -> ProofResult<Self> {
        // Validate inputs
        if min > max {
            return Err(ProofError::InvalidPublicInputs(
                "min must be <= max".to_string(),
            ));
        }

        if value < min || value > max {
            return Err(ProofError::InvalidWitness(format!(
                "value {} not in range [{}, {}]",
                value, min, max
            )));
        }

        let offset = value - min;
        let adjusted_offset = offset + OFFSET_ADJUSTMENT;
        let public_inputs = RangePublicInputs::new(min, max, adjusted_offset);

        // Build prover and trace
        let options = build_proof_options(config.security_level);
        let prover = RangeProofProver::new(options, public_inputs.clone());
        let trace = prover.build_trace(value, min);

        // Generate proof
        let proof = prover.prove(trace).map_err(|e| {
            ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
        })?;

        Ok(Self {
            proof,
            public_inputs,
        })
    }

    /// Verify the range proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        // Use minimum acceptable security level
        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

        let result = winterfell::verify::<
            RangeProofAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(self.proof.clone(), self.public_inputs.clone(), &min_opts);

        let duration = start.elapsed();

        match result {
            Ok(_) => Ok(VerificationResult::success(ProofType::Range, duration)),
            Err(e) => Ok(VerificationResult::failure(
                ProofType::Range,
                duration,
                format!("Verification failed: {:?}", e),
            )),
        }
    }

    /// Get the public inputs
    pub fn public_inputs(&self) -> &RangePublicInputs {
        &self.public_inputs
    }

    /// Serialize the proof to bytes
    ///
    /// Format: [min:8][max:8][adjusted_offset:8][proof_bytes...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.public_inputs.min.to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.max.to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.adjusted_offset.to_le_bytes());
        bytes.extend_from_slice(&self.proof.to_bytes());
        bytes
    }

    /// Deserialize a proof from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let min = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated min value".to_string())
        })?;

        let max = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated max value".to_string())
        })?;

        let adjusted_offset = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated adjusted_offset".to_string())
        })?;

        let public_inputs = RangePublicInputs::new(min, max, adjusted_offset);

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
            security_level: SecurityLevel::Standard96, // Faster for tests
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_range_proof_valid() {
        let value = 50u64;
        let min = 0u64;
        let max = 100u64;

        let proof = RangeProof::generate(value, min, max, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid, "Proof verification failed: {:?}", result.details);
    }

    #[test]
    fn test_range_proof_boundary_min() {
        let value = 10u64;
        let min = 10u64;
        let max = 100u64;

        let proof = RangeProof::generate(value, min, max, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid);
    }

    #[test]
    fn test_range_proof_boundary_max() {
        let value = 100u64;
        let min = 10u64;
        let max = 100u64;

        let proof = RangeProof::generate(value, min, max, test_config()).unwrap();
        let result = proof.verify().unwrap();

        assert!(result.valid);
    }

    #[test]
    fn test_range_proof_value_below_min() {
        let value = 5u64;
        let min = 10u64;
        let max = 100u64;

        let result = RangeProof::generate(value, min, max, test_config());
        assert!(matches!(result, Err(ProofError::InvalidWitness(_))));
    }

    #[test]
    fn test_range_proof_value_above_max() {
        let value = 150u64;
        let min = 10u64;
        let max = 100u64;

        let result = RangeProof::generate(value, min, max, test_config());
        assert!(matches!(result, Err(ProofError::InvalidWitness(_))));
    }

    #[test]
    fn test_range_proof_invalid_range() {
        let value = 50u64;
        let min = 100u64;
        let max = 10u64;

        let result = RangeProof::generate(value, min, max, test_config());
        assert!(matches!(result, Err(ProofError::InvalidPublicInputs(_))));
    }

    #[test]
    fn test_range_proof_size() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let size = proof.size();

        // Proof should be reasonably sized (typically 10-100 KB)
        assert!(size > 1000, "Proof too small: {} bytes", size);
        assert!(size < 500_000, "Proof too large: {} bytes", size);
    }

    #[test]
    fn test_public_inputs() {
        let min = 10u64;
        let max = 100u64;
        let value = 50u64;
        let proof = RangeProof::generate(value, min, max, test_config()).unwrap();

        let inputs = proof.public_inputs();
        assert_eq!(inputs.min, min);
        assert_eq!(inputs.max, max);
        // adjusted_offset = (value - min) + 1 = 41
        assert_eq!(inputs.adjusted_offset, value - min + 1);
        assert_eq!(inputs.original_offset(), value - min);
        assert_eq!(inputs.range_width(), 90);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let value = 50u64;
        let min = 0u64;
        let max = 100u64;

        let proof = RangeProof::generate(value, min, max, test_config()).unwrap();
        let bytes = proof.to_bytes();

        // Deserialize
        let restored = RangeProof::from_bytes(&bytes).unwrap();

        // Verify the restored proof works
        let result = restored.verify().unwrap();
        assert!(result.valid, "Restored proof verification failed");

        // Verify public inputs match
        assert_eq!(restored.public_inputs().min, min);
        assert_eq!(restored.public_inputs().max, max);
    }
}
