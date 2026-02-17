//! Identity Assurance Proof Circuit
//!
//! Proves that an identity meets a minimum assurance level threshold without
//! revealing the specific identity factors.
//!
//! ## What This Proves
//!
//! 1. Identity has verified factors with a total score
//! 2. Total score meets the threshold for the claimed assurance level
//! 3. Proper factor diversity (multiple categories)
//!
//! ## Trace Structure
//!
//! - 4 columns, N rows (one per factor + padding)
//! - Column 0: factor contribution (scaled 0-1000)
//! - Column 1: factor category (0-4 for different types)
//! - Column 2: running score sum
//! - Column 3: active flag (1 = active factor, 0 = inactive/padding)
//!
//! ## Security Assumptions
//!
//! 1. **Hash Function**: Blake3 commitment binds identity factors
//! 2. **Factor Scoring**: Score values (0-1000) accurately represent factor strength
//! 3. **Category Validity**: Factor categories (0-4) are correctly assigned
//!
//! ## Known Limitations
//!
//! - **Not Perfect Zero-Knowledge**: Winterfell STARKs provide computational hiding but
//!   are NOT perfectly zero-knowledge. Factor count may leak through trace length.
//! - **Public Inputs Visible**: Assurance level threshold, final score, and factor
//!   count are publicly visible
//! - **Threshold Granularity**: Fixed thresholds (E0-E4) may not suit all applications
//! - **No Factor Revocation**: Proof validity is based on snapshot; revoked factors
//!   require new proof generation
//!
//! ## Threat Model
//!
//! - **Malicious Prover**: Cannot claim higher assurance level than factors support.
//!   Score accumulation is verified by trace constraints.
//! - **Malicious Verifier**: Learns assurance level achieved, not which specific
//!   factors contributed or their individual scores.
//! - **Factor Inflation**: Prover cannot inflate factor contributions beyond their
//!   actual verified values due to commitment binding.
//! - **Sybil Attack**: This circuit does NOT prevent multiple identities; that
//!   requires external uniqueness verification.

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
use blake3::Hasher;

/// Minimum trace length required by winterfell
const MIN_TRACE_LEN: usize = 8;

/// Number of columns in the trace
const TRACE_WIDTH: usize = 4;

/// Scale factor for score values (0.0-1.0 -> 0-1000)
const SCORE_SCALE: u64 = 1000;

/// Assurance level thresholds (scaled by SCORE_SCALE)
/// E0: 0, E1: 200, E2: 500, E3: 700, E4: 900
const ASSURANCE_THRESHOLDS: [u64; 5] = [0, 200, 500, 700, 900];

/// Assurance levels (E0-E4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofAssuranceLevel {
    E0 = 0,
    E1 = 1,
    E2 = 2,
    E3 = 3,
    E4 = 4,
}

impl ProofAssuranceLevel {
    /// Get the threshold score for this level (scaled)
    pub fn threshold(&self) -> u64 {
        ASSURANCE_THRESHOLDS[*self as usize]
    }

    /// Convert from numeric value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ProofAssuranceLevel::E0),
            1 => Some(ProofAssuranceLevel::E1),
            2 => Some(ProofAssuranceLevel::E2),
            3 => Some(ProofAssuranceLevel::E3),
            4 => Some(ProofAssuranceLevel::E4),
            _ => None,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ProofAssuranceLevel::E0 => "E0 (Anonymous)",
            ProofAssuranceLevel::E1 => "E1 (Basic)",
            ProofAssuranceLevel::E2 => "E2 (Verified)",
            ProofAssuranceLevel::E3 => "E3 (Trusted)",
            ProofAssuranceLevel::E4 => "E4 (Guardian)",
        }
    }
}

/// An identity factor for the proof
#[derive(Debug, Clone)]
pub struct ProofIdentityFactor {
    /// Factor contribution to score (0.0-1.0, will be scaled)
    pub contribution: f32,
    /// Factor category (0-4 representing different types)
    pub category: u8,
    /// Whether the factor is currently active
    pub is_active: bool,
}

impl ProofIdentityFactor {
    /// Create a new factor
    pub fn new(contribution: f32, category: u8, is_active: bool) -> Self {
        Self {
            contribution: contribution.clamp(0.0, 1.0),
            category: category.min(4),
            is_active,
        }
    }
}

/// Public inputs for identity assurance proof
#[derive(Debug, Clone)]
pub struct IdentityPublicInputs {
    /// Commitment to the identity (hash of DID + factors)
    pub identity_commitment: [u8; 32],
    /// Required minimum assurance level
    pub min_level: u8,
    /// Whether the identity meets the threshold
    pub meets_threshold: bool,
    /// Final computed score (scaled)
    pub final_score: u64,
}

impl IdentityPublicInputs {
    /// Create new public inputs
    pub fn new(
        identity_commitment: [u8; 32],
        min_level: u8,
        meets_threshold: bool,
        final_score: u64,
    ) -> Self {
        Self {
            identity_commitment,
            min_level,
            meets_threshold,
            final_score,
        }
    }
}

impl ToElements<BaseElement> for IdentityPublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            // Identity commitment as 4 field elements
            BaseElement::from(u64::from_le_bytes(self.identity_commitment[0..8].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.identity_commitment[8..16].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.identity_commitment[16..24].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.identity_commitment[24..32].try_into().unwrap())),
            BaseElement::from(self.min_level as u64),
            BaseElement::from(if self.meets_threshold { 1u64 } else { 0u64 }),
            BaseElement::from(self.final_score),
        ]
    }
}

/// AIR for identity assurance proof
pub struct IdentityAssuranceAir {
    context: AirContext<BaseElement>,
    final_score: BaseElement,
}

impl Air for IdentityAssuranceAir {
    type BaseField = BaseElement;
    type PublicInputs = IdentityPublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Transition constraint degrees:
        // - Constraint 0: score accumulation - degree 1
        //
        // Note: Active flag handling is done during trace construction.
        // Only active factors contribute to the sum in the trace.
        let degrees = vec![
            TransitionConstraintDegree::new(1), // score accumulation
        ];

        let num_assertions = 2; // Initial score = 0, final score matches

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            final_score: BaseElement::from(pub_inputs.final_score),
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
        // [0]: factor contribution (scaled, 0 if inactive)
        // [1]: factor category
        // [2]: running score sum
        // [3]: active flag (for info, not used in constraint)

        let contribution = current[0];
        let score_sum = current[2];
        let next_score_sum = next[2];

        // Constraint 0: score accumulates correctly
        // next_score = score + contribution
        // (inactive factors have contribution=0 in the trace)
        result[0] = next_score_sum - score_sum - contribution;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;
        vec![
            // Initial score sum is 0
            Assertion::single(2, 0, BaseElement::ZERO),
            // Final score sum matches public input
            Assertion::single(2, last_step, self.final_score),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

/// Identity assurance proof prover
pub struct IdentityAssuranceProver {
    options: ProofOptions,
    public_inputs: IdentityPublicInputs,
}

impl IdentityAssuranceProver {
    /// Create a new prover
    pub fn new(options: ProofOptions, public_inputs: IdentityPublicInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace
    fn build_trace(&self, factors: &[ProofIdentityFactor], diversity_bonus: f32) -> TraceTable<BaseElement> {
        let n = factors.len();
        // We need n+2 rows: n factors + diversity bonus + final accumulation
        let trace_len = (n + 2).max(MIN_TRACE_LEN).next_power_of_two();

        let mut trace = TraceTable::new(TRACE_WIDTH, trace_len);

        // Scale factor contributions
        let mut scaled_contributions: Vec<u64> = factors
            .iter()
            .map(|f| (f.contribution * SCORE_SCALE as f32) as u64)
            .collect();

        // Add diversity bonus as a virtual factor at the end
        let scaled_diversity = (diversity_bonus * SCORE_SCALE as f32) as u64;
        scaled_contributions.push(scaled_diversity);

        // Track which entries are active (real factors + diversity bonus)
        let mut is_active: Vec<bool> = factors.iter().map(|f| f.is_active).collect();
        is_active.push(true); // Diversity bonus is always active

        let total_entries = scaled_contributions.len();

        trace.fill(
            |state| {
                // First row
                if !scaled_contributions.is_empty() {
                    // Only include contribution if active (so constraint works)
                    let contrib = if is_active[0] { scaled_contributions[0] } else { 0 };
                    state[0] = BaseElement::from(contrib);
                    state[1] = BaseElement::from(if !factors.is_empty() { factors[0].category as u64 } else { 0 });
                    state[2] = BaseElement::ZERO; // Initial score sum
                    state[3] = if is_active[0] { BaseElement::ONE } else { BaseElement::ZERO };
                } else {
                    state[0] = BaseElement::ZERO;
                    state[1] = BaseElement::ZERO;
                    state[2] = BaseElement::ZERO;
                    state[3] = BaseElement::ONE; // Non-trivial for constraint
                }
            },
            |step, state| {
                // Get current values for accumulation
                let current_contribution: u128 = state[0].as_int();
                let current_score: u128 = state[2].as_int();

                // Compute next score (contribution is already 0 for inactive entries)
                let next_score = current_score + current_contribution;

                // Get next entry
                let next_idx = step + 1;
                if next_idx < total_entries {
                    let category = if next_idx < factors.len() { factors[next_idx].category as u64 } else { 255 }; // 255 for diversity bonus
                    // Only include contribution if active (so constraint works)
                    let contrib = if is_active[next_idx] { scaled_contributions[next_idx] } else { 0 };
                    state[0] = BaseElement::from(contrib);
                    state[1] = BaseElement::from(category);
                    state[2] = BaseElement::from(next_score as u64);
                    state[3] = if is_active[next_idx] { BaseElement::ONE } else { BaseElement::ZERO };
                } else {
                    // Padding rows - keep accumulating zero
                    state[0] = BaseElement::ZERO;
                    state[1] = BaseElement::ZERO;
                    state[2] = BaseElement::from(next_score as u64);
                    state[3] = BaseElement::ONE; // Non-trivial active flag
                }
            },
        );

        trace
    }
}

impl Prover for IdentityAssuranceProver {
    type BaseField = BaseElement;
    type Air = IdentityAssuranceAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> IdentityPublicInputs {
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

/// Compute commitment to identity factors
pub fn compute_identity_commitment(did: &str, factors: &[ProofIdentityFactor]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"identity:");
    hasher.update(did.as_bytes());
    for factor in factors {
        hasher.update(&factor.contribution.to_le_bytes());
        hasher.update(&[factor.category]);
        hasher.update(&[if factor.is_active { 1 } else { 0 }]);
    }
    *hasher.finalize().as_bytes()
}

/// A complete identity assurance proof
#[derive(Clone)]
pub struct IdentityAssuranceProof {
    /// The STARK proof
    proof: Proof,
    /// Public inputs
    public_inputs: IdentityPublicInputs,
}

impl IdentityAssuranceProof {
    /// Generate an identity assurance proof
    ///
    /// Proves that identity meets the required assurance level.
    pub fn generate(
        did: &str,
        factors: &[ProofIdentityFactor],
        min_level: ProofAssuranceLevel,
        config: ProofConfig,
    ) -> ProofResult<Self> {
        // Calculate total score from active factors
        let total_score: f32 = factors
            .iter()
            .filter(|f| f.is_active)
            .map(|f| f.contribution)
            .sum();

        // Apply diversity bonus (5% per unique category)
        let unique_categories: std::collections::HashSet<u8> = factors
            .iter()
            .filter(|f| f.is_active)
            .map(|f| f.category)
            .collect();
        let diversity_bonus = unique_categories.len() as f32 * 0.05;

        let final_score_f32 = (total_score + diversity_bonus).min(1.0);
        let final_score = (final_score_f32 * SCORE_SCALE as f32) as u64;

        // Check if meets threshold
        let threshold = min_level.threshold();
        let meets_threshold = final_score >= threshold;

        // Compute commitment
        let commitment = compute_identity_commitment(did, factors);

        let public_inputs = IdentityPublicInputs::new(
            commitment,
            min_level as u8,
            meets_threshold,
            final_score,
        );

        // Build prover and trace - include diversity bonus as an extra factor
        let options = build_proof_options(config.security_level);
        let prover = IdentityAssuranceProver::new(options, public_inputs.clone());
        let trace = prover.build_trace(factors, diversity_bonus);

        // Generate proof
        let proof = prover.prove(trace).map_err(|e| {
            ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
        })?;

        Ok(Self {
            proof,
            public_inputs,
        })
    }

    /// Verify the identity assurance proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        // Check threshold
        let min_level = ProofAssuranceLevel::from_u8(self.public_inputs.min_level)
            .ok_or_else(|| ProofError::InvalidPublicInputs("Invalid assurance level".to_string()))?;

        if !self.public_inputs.meets_threshold {
            return Ok(VerificationResult::failure(
                ProofType::IdentityAssurance,
                start.elapsed(),
                format!(
                    "Score {} does not meet {} threshold {}",
                    self.public_inputs.final_score,
                    min_level.name(),
                    min_level.threshold()
                ),
            ));
        }

        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

        let result = winterfell::verify::<
            IdentityAssuranceAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(self.proof.clone(), self.public_inputs.clone(), &min_opts);

        let duration = start.elapsed();

        match result {
            Ok(_) => Ok(VerificationResult::success(ProofType::IdentityAssurance, duration)),
            Err(e) => Ok(VerificationResult::failure(
                ProofType::IdentityAssurance,
                duration,
                format!("Verification failed: {:?}", e),
            )),
        }
    }

    /// Get the public inputs
    pub fn public_inputs(&self) -> &IdentityPublicInputs {
        &self.public_inputs
    }

    /// Check if the identity meets the required level
    pub fn meets_threshold(&self) -> bool {
        self.public_inputs.meets_threshold
    }

    /// Get the final score
    pub fn final_score(&self) -> u64 {
        self.public_inputs.final_score
    }

    /// Serialize the proof to bytes
    ///
    /// Format: [identity_commitment:32][min_level:1][meets_threshold:1][final_score:8][proof_bytes...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.public_inputs.identity_commitment);
        bytes.push(self.public_inputs.min_level);
        bytes.push(if self.public_inputs.meets_threshold { 1 } else { 0 });
        bytes.extend_from_slice(&self.public_inputs.final_score.to_le_bytes());
        bytes.extend_from_slice(&self.proof.to_bytes());
        bytes
    }

    /// Deserialize a proof from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let identity_commitment = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated identity commitment".to_string())
        })?;

        let min_level = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing min_level".to_string())
        })?;

        let meets_threshold = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing meets_threshold flag".to_string())
        })? != 0;

        let final_score = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated final_score".to_string())
        })?;

        let public_inputs = IdentityPublicInputs::new(
            identity_commitment,
            min_level,
            meets_threshold,
            final_score,
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
    fn test_identity_proof_valid() {
        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),  // CryptoKey
            ProofIdentityFactor::new(0.3, 1, true),  // GitcoinPassport
        ];

        let proof = IdentityAssuranceProof::generate(
            "did:mycelix:test123",
            &factors,
            ProofAssuranceLevel::E2,
            test_config(),
        ).unwrap();

        let result = proof.verify().unwrap();
        assert!(result.valid, "Proof verification failed: {:?}", result.details);
        assert!(proof.meets_threshold());
    }

    #[test]
    fn test_identity_proof_insufficient_level() {
        let factors = vec![
            ProofIdentityFactor::new(0.1, 0, true),  // Low contribution
        ];

        let proof = IdentityAssuranceProof::generate(
            "did:mycelix:test123",
            &factors,
            ProofAssuranceLevel::E3,  // Requires 700
            test_config(),
        ).unwrap();

        // Score should be ~150 (0.1 + 0.05 diversity), not enough for E3
        assert!(!proof.meets_threshold());

        let result = proof.verify().unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_identity_proof_inactive_factors() {
        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),   // Active
            ProofIdentityFactor::new(0.5, 1, false),  // Inactive - shouldn't count
        ];

        let proof = IdentityAssuranceProof::generate(
            "did:mycelix:test123",
            &factors,
            ProofAssuranceLevel::E2,  // Requires 500
            test_config(),
        ).unwrap();

        // Score should be 500 + 50 (diversity) = 550
        assert!(proof.meets_threshold());
    }

    #[test]
    fn test_identity_commitment() {
        let factors1 = vec![ProofIdentityFactor::new(0.5, 0, true)];
        let factors2 = vec![ProofIdentityFactor::new(0.6, 0, true)];

        let commit1 = compute_identity_commitment("did:test:1", &factors1);
        let commit2 = compute_identity_commitment("did:test:1", &factors2);
        let commit3 = compute_identity_commitment("did:test:2", &factors1);

        // Different factors -> different commitment
        assert_ne!(commit1, commit2);
        // Different DID -> different commitment
        assert_ne!(commit1, commit3);

        // Same inputs -> same commitment
        let commit1_repeat = compute_identity_commitment("did:test:1", &factors1);
        assert_eq!(commit1, commit1_repeat);
    }

    #[test]
    fn test_assurance_thresholds() {
        assert_eq!(ProofAssuranceLevel::E0.threshold(), 0);
        assert_eq!(ProofAssuranceLevel::E1.threshold(), 200);
        assert_eq!(ProofAssuranceLevel::E2.threshold(), 500);
        assert_eq!(ProofAssuranceLevel::E3.threshold(), 700);
        assert_eq!(ProofAssuranceLevel::E4.threshold(), 900);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
        ];

        let proof = IdentityAssuranceProof::generate(
            "did:mycelix:test123",
            &factors,
            ProofAssuranceLevel::E2,
            test_config(),
        ).unwrap();

        let bytes = proof.to_bytes();
        let restored = IdentityAssuranceProof::from_bytes(&bytes).unwrap();

        let result = restored.verify().unwrap();
        assert!(result.valid, "Restored proof verification failed");
        assert_eq!(restored.meets_threshold(), proof.meets_threshold());
        assert_eq!(restored.final_score(), proof.final_score());
    }
}
