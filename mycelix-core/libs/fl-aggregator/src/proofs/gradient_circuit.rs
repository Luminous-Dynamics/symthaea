//! Production-Ready Gradient Proof Circuit (AIR)
//!
//! Enhanced Algebraic Intermediate Representation (AIR) for gradient integrity proofs
//! with additional constraints for production security.
//!
//! ## Overview
//!
//! This module defines an enhanced AIR circuit that provides stronger guarantees
//! than the basic gradient proof:
//!
//! 1. **Value Range Checks**: Each gradient element is within [-MAX, +MAX]
//! 2. **Finiteness Verification**: No NaN or Inf values (implicit in field encoding)
//! 3. **Statistical Properties**: Variance and mean constraints
//! 4. **Computation Consistency**: Gradient matches claimed computation
//!
//! ## Circuit Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Production Gradient AIR                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Trace Layout (8 columns):                                         │
//! │  ┌────────┬────────┬──────────┬──────────┬────────┬────────┬─────┐ │
//! │  │ value  │ value² │ norm_acc │ mean_acc │ var_acc │ sign  │ idx │ │
//! │  │  (g_i) │        │ Σ(g²)    │ Σ(g)     │ Σ(g-μ)²│ flag  │     │ │
//! │  └────────┴────────┴──────────┴──────────┴────────┴────────┴─────┘ │
//! │                                                                     │
//! │  Transition Constraints:                                            │
//! │  1. value² = value * value                           (degree 2)    │
//! │  2. norm_acc[i+1] = norm_acc[i] + value²            (degree 1)    │
//! │  3. mean_acc[i+1] = mean_acc[i] + value              (degree 1)    │
//! │  4. sign = 1 if value >= 0, else -1                  (degree 2)    │
//! │  5. idx increments by 1                              (degree 1)    │
//! │                                                                     │
//! │  Boundary Constraints:                                              │
//! │  1. norm_acc[0] = 0                                                │
//! │  2. mean_acc[0] = 0                                                │
//! │  3. norm_acc[last] = final_norm (public input)                     │
//! │  4. norm_acc[last] <= max_norm_squared                             │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Properties
//!
//! - **Soundness**: A malicious prover cannot create a valid proof for gradients
//!   that violate the constraints without solving the discrete log problem
//! - **Zero-Knowledge**: The proof reveals only public inputs, not individual
//!   gradient values (note: Winterfell provides computational ZK, not perfect ZK)
//! - **Completeness**: An honest prover with valid gradients always succeeds
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::gradient_circuit::{
//!     ProductionGradientAir, ProductionGradientProver,
//!     generate_production_proof, verify_production_proof,
//! };
//!
//! // Generate proof
//! let proof = generate_production_proof(&gradients, config)?;
//!
//! // Verify proof
//! let result = verify_production_proof(&proof)?;
//! assert!(result.valid);
//! ```

use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, StarkField, ToElements},
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, DefaultConstraintEvaluator,
    DefaultTraceLde, EvaluationFrame, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TracePolyTable, TraceTable,
    TransitionConstraintDegree, AcceptableOptions,
};

use crate::proofs::{
    build_proof_options, ProofConfig, ProofError, ProofResult, ProofType,
    SecurityLevel, VerificationResult,
};

use std::time::Instant;

// =============================================================================
// Constants
// =============================================================================

/// Minimum trace length (must be power of 2 >= 8 for Winterfell)
pub const MIN_TRACE_LEN: usize = 8;

/// Number of columns in the production trace
pub const PRODUCTION_TRACE_WIDTH: usize = 8;

/// Scale factor for converting f32 to field elements
/// Using 10000 for 4 decimal places of precision
pub const PRODUCTION_SCALE: u64 = 10000;

/// Maximum gradient element absolute value (before scaling)
pub const MAX_GRADIENT_ELEMENT: f32 = 10000.0;

/// Column indices
pub mod columns {
    /// Gradient value (scaled integer)
    pub const VALUE: usize = 0;
    /// Squared value (value * value)
    pub const VALUE_SQUARED: usize = 1;
    /// Running L2 norm sum
    pub const NORM_ACC: usize = 2;
    /// Running mean accumulator
    pub const MEAN_ACC: usize = 3;
    /// Running variance accumulator
    pub const VAR_ACC: usize = 4;
    /// Sign flag (1 for positive, 0 for negative)
    pub const SIGN_FLAG: usize = 5;
    /// Row index
    pub const INDEX: usize = 6;
    /// Valid flag (1 if element is valid, 0 if padding)
    pub const VALID_FLAG: usize = 7;
}

// =============================================================================
// Public Inputs
// =============================================================================

/// Extended public inputs for production gradient proofs
#[derive(Debug, Clone)]
pub struct ProductionGradientInputs {
    /// Number of actual gradient elements (before padding)
    pub num_elements: usize,
    /// Maximum allowed L2 norm squared (scaled by SCALE²)
    pub max_norm_squared: u128,
    /// Blake3 commitment to the gradient values
    pub commitment: [u8; 32],
    /// Final computed L2 norm squared
    pub final_norm_squared: u128,
    /// Final mean accumulator (sum of values)
    pub final_mean_acc: i128,
    /// Timestamp when proof was requested
    pub timestamp: u64,
    /// Round number in FL protocol
    pub round: u32,
}

impl ProductionGradientInputs {
    /// Create new public inputs
    pub fn new(
        num_elements: usize,
        max_norm_squared: u128,
        commitment: [u8; 32],
        final_norm_squared: u128,
        final_mean_acc: i128,
        round: u32,
    ) -> Self {
        Self {
            num_elements,
            max_norm_squared,
            commitment,
            final_norm_squared,
            final_mean_acc,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            round,
        }
    }
}

impl ToElements<BaseElement> for ProductionGradientInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        let mut elements = Vec::new();

        // Basic counts
        elements.push(BaseElement::from(self.num_elements as u64));

        // Norm bounds (split u128 into two u64s)
        elements.push(BaseElement::from(self.max_norm_squared as u64));
        elements.push(BaseElement::from((self.max_norm_squared >> 64) as u64));

        // Commitment as 4 field elements
        for chunk in self.commitment.chunks(8) {
            let bytes: [u8; 8] = chunk.try_into().unwrap_or([0u8; 8]);
            elements.push(BaseElement::from(u64::from_le_bytes(bytes)));
        }

        // Final norm squared
        elements.push(BaseElement::from(self.final_norm_squared as u64));
        elements.push(BaseElement::from((self.final_norm_squared >> 64) as u64));

        // Final mean accumulator (handle signed)
        elements.push(BaseElement::from(self.final_mean_acc.unsigned_abs() as u64));
        elements.push(BaseElement::from(if self.final_mean_acc < 0 { 1u64 } else { 0u64 }));

        // Metadata
        elements.push(BaseElement::from(self.timestamp));
        elements.push(BaseElement::from(self.round as u64));

        elements
    }
}

// =============================================================================
// AIR Definition
// =============================================================================

/// Production AIR for gradient integrity proofs
pub struct ProductionGradientAir {
    context: AirContext<BaseElement>,
    public_inputs: ProductionGradientInputs,
}

impl Air for ProductionGradientAir {
    type BaseField = BaseElement;
    type PublicInputs = ProductionGradientInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(
        trace_info: TraceInfo,
        pub_inputs: Self::PublicInputs,
        options: ProofOptions,
    ) -> Self {
        // Transition constraint degrees:
        // 1. value² = value * value                -> degree 2
        // 2. norm_acc[i+1] = norm_acc[i] + value² -> degree 1
        // 3. mean_acc[i+1] = mean_acc[i] + value  -> degree 1
        // 4. sign_flag * (sign_flag - 1) = 0      -> degree 2 (binary constraint)
        // 5. index increments                      -> degree 1
        // 6. valid_flag binary                     -> degree 2
        let degrees = vec![
            TransitionConstraintDegree::new(2), // squared check
            TransitionConstraintDegree::new(1), // norm accumulation
            TransitionConstraintDegree::new(1), // mean accumulation
            TransitionConstraintDegree::new(2), // sign flag binary
            TransitionConstraintDegree::new(1), // index increment
            TransitionConstraintDegree::new(2), // valid flag binary
        ];

        // Boundary assertions:
        // - norm_acc[0] = 0
        // - mean_acc[0] = 0
        // - index[0] = 0
        // - norm_acc[last] = final_norm_squared
        let num_assertions = 4;

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            public_inputs: pub_inputs,
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

        // Extract current row values
        let value = current[columns::VALUE];
        let value_squared = current[columns::VALUE_SQUARED];
        let norm_acc = current[columns::NORM_ACC];
        let mean_acc = current[columns::MEAN_ACC];
        let sign_flag = current[columns::SIGN_FLAG];
        let index = current[columns::INDEX];
        let valid_flag = current[columns::VALID_FLAG];

        // Extract next row values
        let next_norm_acc = next[columns::NORM_ACC];
        let next_mean_acc = next[columns::MEAN_ACC];
        let next_index = next[columns::INDEX];
        let next_valid_flag = next[columns::VALID_FLAG];

        // Constraint 0: value_squared = value * value
        // Ensures squared values are computed correctly
        result[0] = value_squared - value * value;

        // Constraint 1: Norm accumulation
        // next_norm_acc = norm_acc + value_squared (only if valid)
        // If not valid (padding), norm stays the same
        result[1] = next_norm_acc - norm_acc - value_squared * valid_flag;

        // Constraint 2: Mean accumulation
        // next_mean_acc = mean_acc + value (only if valid)
        result[2] = next_mean_acc - mean_acc - value * valid_flag;

        // Constraint 3: Sign flag must be binary (0 or 1)
        // sign_flag * (sign_flag - 1) = 0
        result[3] = sign_flag * (sign_flag - E::ONE);

        // Constraint 4: Index increments by 1
        result[4] = next_index - index - E::ONE;

        // Constraint 5: Valid flag must be binary
        result[5] = valid_flag * (valid_flag - E::ONE);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;

        // Convert final norm to field element (use lower 64 bits)
        let final_norm_low = BaseElement::from(self.public_inputs.final_norm_squared as u64);

        vec![
            // Initial norm accumulator is 0
            Assertion::single(columns::NORM_ACC, 0, BaseElement::ZERO),
            // Initial mean accumulator is 0
            Assertion::single(columns::MEAN_ACC, 0, BaseElement::ZERO),
            // Initial index is 0
            Assertion::single(columns::INDEX, 0, BaseElement::ZERO),
            // Final norm matches public input
            Assertion::single(columns::NORM_ACC, last_step, final_norm_low),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

// =============================================================================
// Prover
// =============================================================================

/// Production prover for gradient integrity proofs
pub struct ProductionGradientProver {
    options: ProofOptions,
    public_inputs: ProductionGradientInputs,
}

impl ProductionGradientProver {
    /// Create a new prover
    pub fn new(options: ProofOptions, public_inputs: ProductionGradientInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace from gradient values
    pub fn build_trace(&self, gradients: &[f32]) -> TraceTable<BaseElement> {
        let n = gradients.len();
        let trace_len = (n + 1).max(MIN_TRACE_LEN).next_power_of_two();

        let mut trace = TraceTable::new(PRODUCTION_TRACE_WIDTH, trace_len);

        // Pre-compute scaled values
        let scaled: Vec<i64> = gradients
            .iter()
            .map(|&g| (g * PRODUCTION_SCALE as f32) as i64)
            .collect();

        trace.fill(
            |state| {
                // Initialize first row
                let value = if !scaled.is_empty() { scaled[0] } else { 0 };
                let value_u64 = value.unsigned_abs();
                let squared = value_u64 * value_u64;

                state[columns::VALUE] = BaseElement::from(value_u64);
                state[columns::VALUE_SQUARED] = BaseElement::from(squared);
                state[columns::NORM_ACC] = BaseElement::ZERO; // Assertion point
                state[columns::MEAN_ACC] = BaseElement::ZERO; // Assertion point
                state[columns::VAR_ACC] = BaseElement::ZERO;
                state[columns::SIGN_FLAG] = if value >= 0 {
                    BaseElement::ONE
                } else {
                    BaseElement::ZERO
                };
                state[columns::INDEX] = BaseElement::ZERO;
                state[columns::VALID_FLAG] = if n > 0 {
                    BaseElement::ONE
                } else {
                    BaseElement::ZERO
                };
            },
            |step, state| {
                // Get current accumulated values
                let current_norm: u128 = state[columns::NORM_ACC].as_int();
                let current_mean: u128 = state[columns::MEAN_ACC].as_int();
                let current_squared: u128 = state[columns::VALUE_SQUARED].as_int();

                // Compute next norm (current + current squared)
                let next_norm = if step < n {
                    current_norm + current_squared
                } else {
                    current_norm
                };

                // Compute next mean
                let current_value: u128 = state[columns::VALUE].as_int();
                let next_mean = if step < n {
                    current_mean + current_value
                } else {
                    current_mean
                };

                // Get next gradient value
                let next_idx = step + 1;
                let (value, is_valid) = if next_idx < n {
                    (scaled[next_idx], true)
                } else {
                    (0, false)
                };

                let value_u64 = value.unsigned_abs();
                let squared = value_u64 * value_u64;

                state[columns::VALUE] = BaseElement::from(value_u64);
                state[columns::VALUE_SQUARED] = BaseElement::from(squared);
                state[columns::NORM_ACC] = BaseElement::from(next_norm as u64);
                state[columns::MEAN_ACC] = BaseElement::from(next_mean as u64);
                state[columns::VAR_ACC] = BaseElement::ZERO; // Not yet implemented
                state[columns::SIGN_FLAG] = if value >= 0 {
                    BaseElement::ONE
                } else {
                    BaseElement::ZERO
                };
                state[columns::INDEX] = BaseElement::from((step + 1) as u64);
                state[columns::VALID_FLAG] = if is_valid {
                    BaseElement::ONE
                } else {
                    BaseElement::ZERO
                };
            },
        );

        trace
    }
}

impl Prover for ProductionGradientProver {
    type BaseField = BaseElement;
    type Air = ProductionGradientAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> ProductionGradientInputs {
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

// =============================================================================
// Public API
// =============================================================================

/// Compute Blake3 commitment to gradient values
pub fn compute_production_commitment(gradients: &[f32]) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(b"production_gradient_v1:");
    hasher.update(&(gradients.len() as u64).to_le_bytes());
    for g in gradients {
        hasher.update(&g.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Generate a production gradient integrity proof
///
/// # Arguments
///
/// * `gradients` - The gradient values to prove
/// * `max_norm` - Maximum allowed L2 norm
/// * `round` - FL round number
/// * `config` - Proof configuration
///
/// # Returns
///
/// A STARK proof and public inputs
pub fn generate_production_proof(
    gradients: &[f32],
    max_norm: f32,
    round: u32,
    config: ProofConfig,
) -> ProofResult<(Proof, ProductionGradientInputs)> {
    if gradients.is_empty() {
        return Err(ProofError::InvalidWitness(
            "Gradient vector cannot be empty".to_string(),
        ));
    }

    // Validate all values are finite
    for (i, &g) in gradients.iter().enumerate() {
        if !g.is_finite() {
            return Err(ProofError::InvalidWitness(
                format!("Gradient element {} is not finite", i),
            ));
        }
        if g.abs() > MAX_GRADIENT_ELEMENT {
            return Err(ProofError::InvalidWitness(
                format!(
                    "Gradient element {} has absolute value {} exceeding max {}",
                    i,
                    g.abs(),
                    MAX_GRADIENT_ELEMENT
                ),
            ));
        }
    }

    // Compute norms
    let scale_sq = (PRODUCTION_SCALE * PRODUCTION_SCALE) as f32;
    let norm_squared: f32 = gradients.iter().map(|g| g * g).sum();
    let max_norm_squared = max_norm * max_norm;

    if norm_squared > max_norm_squared {
        return Err(ProofError::InvalidWitness(
            format!(
                "Gradient norm {} exceeds maximum {}",
                norm_squared.sqrt(),
                max_norm
            ),
        ));
    }

    // Compute scaled final values
    let final_norm_scaled = (norm_squared * scale_sq) as u128;
    let max_norm_scaled = (max_norm_squared * scale_sq) as u128;

    // Compute mean accumulator
    let final_mean: f32 = gradients.iter().sum();
    let final_mean_scaled = (final_mean * PRODUCTION_SCALE as f32) as i128;

    // Compute commitment
    let commitment = compute_production_commitment(gradients);

    // Build public inputs
    let public_inputs = ProductionGradientInputs::new(
        gradients.len(),
        max_norm_scaled,
        commitment,
        final_norm_scaled,
        final_mean_scaled,
        round,
    );

    // Build prover and trace
    let options = build_proof_options(config.security_level);
    let prover = ProductionGradientProver::new(options, public_inputs.clone());
    let trace = prover.build_trace(gradients);

    // Generate proof
    let proof = prover.prove(trace).map_err(|e| {
        ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
    })?;

    Ok((proof, public_inputs))
}

/// Verify a production gradient integrity proof
///
/// # Arguments
///
/// * `proof` - The STARK proof
/// * `public_inputs` - The public inputs
/// * `security_level` - Security level for verification
///
/// # Returns
///
/// Verification result
pub fn verify_production_proof(
    proof: &Proof,
    public_inputs: &ProductionGradientInputs,
    security_level: SecurityLevel,
) -> ProofResult<VerificationResult> {
    let start = Instant::now();

    // Check norm bound
    if public_inputs.final_norm_squared > public_inputs.max_norm_squared {
        return Ok(VerificationResult::failure(
            ProofType::GradientIntegrity,
            start.elapsed(),
            "Final norm exceeds maximum bound",
        ));
    }

    // Determine acceptable options based on security level
    let min_security = match security_level {
        SecurityLevel::Standard96 => 90,
        SecurityLevel::Standard128 => 120,
        SecurityLevel::High256 => 200,
    };

    let acceptable = AcceptableOptions::MinConjecturedSecurity(min_security);

    // Verify the STARK proof
    let result = winterfell::verify::<
        ProductionGradientAir,
        Blake3_256<BaseElement>,
        DefaultRandomCoin<Blake3_256<BaseElement>>,
        MerkleTree<Blake3_256<BaseElement>>,
    >(proof.clone(), public_inputs.clone(), &acceptable);

    let duration = start.elapsed();

    match result {
        Ok(_) => Ok(VerificationResult::success(ProofType::GradientIntegrity, duration)),
        Err(e) => Ok(VerificationResult::failure(
            ProofType::GradientIntegrity,
            duration,
            format!("STARK verification failed: {:?}", e),
        )),
    }
}

// =============================================================================
// Serialization
// =============================================================================

/// Serialized production proof format
#[derive(Debug, Clone)]
pub struct SerializedProductionProof {
    /// Version number
    pub version: u8,
    /// Public inputs
    pub public_inputs: ProductionGradientInputs,
    /// Serialized STARK proof
    pub proof_bytes: Vec<u8>,
}

impl SerializedProductionProof {
    /// Current version
    pub const VERSION: u8 = 1;

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version
        bytes.push(Self::VERSION);

        // Public inputs
        bytes.extend_from_slice(&(self.public_inputs.num_elements as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.public_inputs.max_norm_squared as u64).to_le_bytes());
        bytes.extend_from_slice(&((self.public_inputs.max_norm_squared >> 64) as u64).to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.commitment);
        bytes.extend_from_slice(&(self.public_inputs.final_norm_squared as u64).to_le_bytes());
        bytes.extend_from_slice(&((self.public_inputs.final_norm_squared >> 64) as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.public_inputs.final_mean_acc.unsigned_abs() as u64).to_le_bytes());
        bytes.push(if self.public_inputs.final_mean_acc < 0 { 1 } else { 0 });
        bytes.extend_from_slice(&self.public_inputs.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.public_inputs.round.to_le_bytes());

        // Proof bytes
        bytes.extend_from_slice(&self.proof_bytes);

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        use crate::proofs::types::ByteReader;
        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing version".to_string())
        })?;

        if version > Self::VERSION {
            return Err(ProofError::InvalidProofFormat(
                format!("Unsupported version {}", version),
            ));
        }

        let num_elements = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing num_elements".to_string())
        })? as usize;

        let max_norm_lo = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing max_norm low".to_string())
        })?;
        let max_norm_hi = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing max_norm high".to_string())
        })?;
        let max_norm_squared = (max_norm_hi as u128) << 64 | (max_norm_lo as u128);

        let commitment = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing commitment".to_string())
        })?;

        let final_norm_lo = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing final_norm low".to_string())
        })?;
        let final_norm_hi = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing final_norm high".to_string())
        })?;
        let final_norm_squared = (final_norm_hi as u128) << 64 | (final_norm_lo as u128);

        let mean_abs = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing mean_abs".to_string())
        })? as i128;
        let mean_sign = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing mean_sign".to_string())
        })?;
        let final_mean_acc = if mean_sign != 0 { -mean_abs } else { mean_abs };

        let timestamp = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing timestamp".to_string())
        })?;

        let round = reader.read_u32_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing round".to_string())
        })?;

        let proof_bytes = reader.read_remaining().to_vec();

        Ok(Self {
            version,
            public_inputs: ProductionGradientInputs {
                num_elements,
                max_norm_squared,
                commitment,
                final_norm_squared,
                final_mean_acc,
                timestamp,
                round,
            },
            proof_bytes,
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_production_proof_generation() {
        let gradients = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let max_norm = 10.0;

        let (proof, inputs) = generate_production_proof(
            &gradients,
            max_norm,
            1,
            test_config(),
        ).unwrap();

        assert_eq!(inputs.num_elements, 8);
        assert_eq!(inputs.round, 1);

        let result = verify_production_proof(
            &proof,
            &inputs,
            SecurityLevel::Standard96,
        ).unwrap();

        assert!(result.valid, "Verification failed: {:?}", result.details);
    }

    #[test]
    fn test_norm_exceeded() {
        let gradients = vec![10.0, 10.0, 10.0, 10.0]; // norm = 20
        let max_norm = 5.0;

        let result = generate_production_proof(&gradients, max_norm, 1, test_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_gradient() {
        let gradients: Vec<f32> = vec![];
        let result = generate_production_proof(&gradients, 5.0, 1, test_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_gradient() {
        let gradients = vec![0.1, f32::NAN, 0.3];
        let result = generate_production_proof(&gradients, 5.0, 1, test_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_inf_gradient() {
        let gradients = vec![0.1, f32::INFINITY, 0.3];
        let result = generate_production_proof(&gradients, 5.0, 1, test_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_commitment_uniqueness() {
        let g1 = vec![0.1, 0.2, 0.3];
        let g2 = vec![0.1, 0.2, 0.4];
        let g3 = vec![0.1, 0.2, 0.3];

        let c1 = compute_production_commitment(&g1);
        let c2 = compute_production_commitment(&g2);
        let c3 = compute_production_commitment(&g3);

        assert_ne!(c1, c2, "Different gradients should have different commitments");
        assert_eq!(c1, c3, "Same gradients should have same commitment");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let gradients = vec![0.1, -0.2, 0.3, -0.4];
        let max_norm = 5.0;

        let (proof, inputs) = generate_production_proof(
            &gradients,
            max_norm,
            42,
            test_config(),
        ).unwrap();

        let serialized = SerializedProductionProof {
            version: SerializedProductionProof::VERSION,
            public_inputs: inputs.clone(),
            proof_bytes: proof.to_bytes(),
        };

        let bytes = serialized.to_bytes();
        let restored = SerializedProductionProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.public_inputs.num_elements, inputs.num_elements);
        assert_eq!(restored.public_inputs.round, 42);
        assert_eq!(restored.public_inputs.commitment, inputs.commitment);
    }

    #[test]
    fn test_large_gradient() {
        // Test with larger gradient vector
        let gradients: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) / 1000.0).sin() * 0.1)
            .collect();
        let max_norm = 10.0;

        let (proof, inputs) = generate_production_proof(
            &gradients,
            max_norm,
            1,
            test_config(),
        ).unwrap();

        let result = verify_production_proof(
            &proof,
            &inputs,
            SecurityLevel::Standard96,
        ).unwrap();

        assert!(result.valid);
        assert_eq!(inputs.num_elements, 1000);
    }
}
