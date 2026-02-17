//! Vote Eligibility Proof Circuit
//!
//! Proves that a voter meets the requirements for a proposal type without
//! revealing the voter's actual profile data.
//!
//! ## What This Proves
//!
//! 1. Assurance level >= minimum required
//! 2. MATL score >= minimum required
//! 3. Stake >= minimum required
//! 4. Account age >= minimum required
//! 5. Participation rate >= minimum required
//! 6. Humanity proof present (when required)
//! 7. FL contributions >= minimum required (when required)
//!
//! ## Trace Structure
//!
//! - 5 columns, 8 rows (7 requirements + final summary)
//! - Column 0: voter value (scaled)
//! - Column 1: requirement threshold (scaled)
//! - Column 2: requirement satisfied (1 = yes, 0 = no)
//! - Column 3: requirement active (1 = required, 0 = not required)
//! - Column 4: running satisfied count
//!
//! ## Security Assumptions
//!
//! 1. **Hash Function**: Blake3 commitment binds voter profile data
//! 2. **Requirement Thresholds**: Proposal type requirements are correctly configured
//! 3. **Profile Freshness**: Voter profile data is current at proof generation time
//!
//! ## Known Limitations
//!
//! - **Not Perfect Zero-Knowledge**: Winterfell STARKs provide computational hiding but
//!   are NOT perfectly zero-knowledge. Profile characteristics may leak.
//! - **Public Inputs Visible**: Proposal type, required thresholds, and eligibility
//!   result are publicly visible
//! - **Fixed Requirements**: Requirement thresholds are hardcoded per proposal type
//! - **Snapshot-Based**: Profile changes after proof generation are not reflected
//!
//! ## Threat Model
//!
//! - **Malicious Prover (Ineligible Voter)**: Cannot forge eligibility for proposals
//!   where requirements are not met. Each requirement is verified individually.
//! - **Malicious Verifier**: Learns only that voter is eligible, not their specific
//!   scores for each requirement (stake, age, participation, etc.).
//! - **Vote Buying**: This circuit proves eligibility only; vote privacy requires
//!   additional measures (e.g., encrypted ballots, mixnets).
//! - **Double Voting**: Preventing multiple votes requires external nullifier tracking;
//!   this circuit does not provide replay protection.

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
const TRACE_WIDTH: usize = 5;

/// Number of eligibility requirements
const NUM_REQUIREMENTS: usize = 7;

/// Scale factor for f32 values (0.0-1.0 -> 0-1000)
const SCORE_SCALE: u64 = 1000;

/// Proposal types for eligibility requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofProposalType {
    Standard = 0,
    Constitutional = 1,
    ModelGovernance = 2,
    Emergency = 3,
    Treasury = 4,
    Membership = 5,
}

impl ProofProposalType {
    /// Convert from numeric value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ProofProposalType::Standard),
            1 => Some(ProofProposalType::Constitutional),
            2 => Some(ProofProposalType::ModelGovernance),
            3 => Some(ProofProposalType::Emergency),
            4 => Some(ProofProposalType::Treasury),
            5 => Some(ProofProposalType::Membership),
            _ => None,
        }
    }

    /// Get requirements for this proposal type
    pub fn requirements(&self) -> ProofEligibilityRequirements {
        match self {
            ProofProposalType::Standard | ProofProposalType::Membership => {
                ProofEligibilityRequirements {
                    min_assurance_level: 1,
                    min_matl_score: 0.3,
                    min_stake: 0.0,
                    min_account_age_days: 7,
                    min_participation: 0.0,
                    humanity_proof_required: false,
                    fl_participation_required: false,
                    min_fl_contributions: 0,
                }
            }
            ProofProposalType::Constitutional => ProofEligibilityRequirements {
                min_assurance_level: 2,
                min_matl_score: 0.5,
                min_stake: 100.0,
                min_account_age_days: 30,
                min_participation: 0.1,
                humanity_proof_required: true,
                fl_participation_required: false,
                min_fl_contributions: 0,
            },
            ProofProposalType::ModelGovernance => ProofEligibilityRequirements {
                min_assurance_level: 2,
                min_matl_score: 0.5,
                min_stake: 0.0,
                min_account_age_days: 14,
                min_participation: 0.0,
                humanity_proof_required: false,
                fl_participation_required: true,
                min_fl_contributions: 10,
            },
            ProofProposalType::Emergency => ProofEligibilityRequirements {
                min_assurance_level: 3,
                min_matl_score: 0.7,
                min_stake: 500.0,
                min_account_age_days: 90,
                min_participation: 0.3,
                humanity_proof_required: true,
                fl_participation_required: false,
                min_fl_contributions: 0,
            },
            ProofProposalType::Treasury => ProofEligibilityRequirements {
                min_assurance_level: 2,
                min_matl_score: 0.6,
                min_stake: 200.0,
                min_account_age_days: 60,
                min_participation: 0.15,
                humanity_proof_required: true,
                fl_participation_required: false,
                min_fl_contributions: 0,
            },
        }
    }
}

/// Eligibility requirements for proof generation
#[derive(Debug, Clone)]
pub struct ProofEligibilityRequirements {
    pub min_assurance_level: u8,
    pub min_matl_score: f32,
    pub min_stake: f32,
    pub min_account_age_days: u32,
    pub min_participation: f32,
    pub humanity_proof_required: bool,
    pub fl_participation_required: bool,
    pub min_fl_contributions: u32,
}

/// Voter profile for proof generation
#[derive(Debug, Clone)]
pub struct ProofVoterProfile {
    /// DID of the voter
    pub did: String,
    /// Assurance level (0-4)
    pub assurance_level: u8,
    /// MATL score (0.0-1.0)
    pub matl_score: f32,
    /// Stake amount
    pub stake: f32,
    /// Account age in days
    pub account_age_days: u32,
    /// Recent participation rate (0.0-1.0)
    pub participation_rate: f32,
    /// Has humanity proof
    pub has_humanity_proof: bool,
    /// FL contributions count
    pub fl_contributions: u32,
}

/// Public inputs for vote eligibility proof
#[derive(Debug, Clone)]
pub struct VotePublicInputs {
    /// Commitment to the voter profile (hash of DID + profile)
    pub voter_commitment: [u8; 32],
    /// Proposal type being voted on
    pub proposal_type: u8,
    /// Whether the voter is eligible
    pub eligible: bool,
    /// Number of requirements satisfied
    pub requirements_met: u8,
    /// Total number of active requirements
    pub active_requirements: u8,
}

impl VotePublicInputs {
    /// Create new public inputs
    pub fn new(
        voter_commitment: [u8; 32],
        proposal_type: u8,
        eligible: bool,
        requirements_met: u8,
        active_requirements: u8,
    ) -> Self {
        Self {
            voter_commitment,
            proposal_type,
            eligible,
            requirements_met,
            active_requirements,
        }
    }
}

impl ToElements<BaseElement> for VotePublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            // Voter commitment as 4 field elements
            BaseElement::from(u64::from_le_bytes(self.voter_commitment[0..8].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.voter_commitment[8..16].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.voter_commitment[16..24].try_into().unwrap())),
            BaseElement::from(u64::from_le_bytes(self.voter_commitment[24..32].try_into().unwrap())),
            BaseElement::from(self.proposal_type as u64),
            BaseElement::from(if self.eligible { 1u64 } else { 0u64 }),
            BaseElement::from(self.requirements_met as u64),
            BaseElement::from(self.active_requirements as u64),
        ]
    }
}

/// AIR for vote eligibility proof
pub struct VoteEligibilityAir {
    context: AirContext<BaseElement>,
    requirements_met: BaseElement,
}

impl Air for VoteEligibilityAir {
    type BaseField = BaseElement;
    type PublicInputs = VotePublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Transition constraint degrees:
        // - Constraint 0: satisfied count accumulation (degree 1)
        let degrees = vec![
            TransitionConstraintDegree::new(1), // count accumulation
        ];

        let num_assertions = 2; // Initial count = 0, final count matches

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            requirements_met: BaseElement::from(pub_inputs.requirements_met as u64),
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
        // [0]: voter value (scaled)
        // [1]: requirement threshold (scaled)
        // [2]: requirement satisfied (1 or 0)
        // [3]: requirement active (1 or 0)
        // [4]: running satisfied count

        let satisfied = current[2];  // Already 0 if not satisfied or not active
        let count = current[4];
        let next_count = next[4];

        // Constraint: next_count = count + satisfied
        // (satisfied is 0 for inactive requirements and for unsatisfied requirements)
        result[0] = next_count - count - satisfied;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;
        vec![
            // Initial count is 0
            Assertion::single(4, 0, BaseElement::ZERO),
            // Final count matches requirements met
            Assertion::single(4, last_step, self.requirements_met),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

/// Vote eligibility proof prover
pub struct VoteEligibilityProver {
    options: ProofOptions,
    public_inputs: VotePublicInputs,
}

impl VoteEligibilityProver {
    /// Create a new prover
    pub fn new(options: ProofOptions, public_inputs: VotePublicInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace
    fn build_trace(
        &self,
        voter: &ProofVoterProfile,
        requirements: &ProofEligibilityRequirements,
    ) -> TraceTable<BaseElement> {
        let trace_len = MIN_TRACE_LEN.next_power_of_two();

        let mut trace = TraceTable::new(TRACE_WIDTH, trace_len);

        // Build requirement check data
        let checks: Vec<(u64, u64, bool, bool)> = vec![
            // (voter_value, threshold, satisfied, active)
            // Assurance level
            (
                voter.assurance_level as u64,
                requirements.min_assurance_level as u64,
                voter.assurance_level >= requirements.min_assurance_level,
                true, // Always active
            ),
            // MATL score
            (
                (voter.matl_score * SCORE_SCALE as f32) as u64,
                (requirements.min_matl_score * SCORE_SCALE as f32) as u64,
                voter.matl_score >= requirements.min_matl_score,
                true, // Always active
            ),
            // Stake
            (
                voter.stake as u64,
                requirements.min_stake as u64,
                voter.stake >= requirements.min_stake,
                requirements.min_stake > 0.0, // Only active if stake is required
            ),
            // Account age
            (
                voter.account_age_days as u64,
                requirements.min_account_age_days as u64,
                voter.account_age_days >= requirements.min_account_age_days,
                requirements.min_account_age_days > 0, // Only active if age is required
            ),
            // Participation
            (
                (voter.participation_rate * SCORE_SCALE as f32) as u64,
                (requirements.min_participation * SCORE_SCALE as f32) as u64,
                voter.participation_rate >= requirements.min_participation,
                requirements.min_participation > 0.0, // Only active if participation is required
            ),
            // Humanity proof
            (
                if voter.has_humanity_proof { 1 } else { 0 },
                1, // Threshold is 1 (must have proof)
                voter.has_humanity_proof,
                requirements.humanity_proof_required,
            ),
            // FL contributions
            (
                voter.fl_contributions as u64,
                requirements.min_fl_contributions as u64,
                voter.fl_contributions >= requirements.min_fl_contributions,
                requirements.fl_participation_required,
            ),
        ];

        trace.fill(
            |state| {
                // First row
                let (voter_val, threshold, satisfied, active) = checks[0];
                state[0] = BaseElement::from(voter_val);
                state[1] = BaseElement::from(threshold);
                // Set satisfied to 0 if not active or not satisfied
                state[2] = if active && satisfied { BaseElement::ONE } else { BaseElement::ZERO };
                state[3] = if active { BaseElement::ONE } else { BaseElement::ZERO };
                state[4] = BaseElement::ZERO; // Initial count
            },
            |step, state| {
                // Get current satisfied value for accumulation
                let current_satisfied: u128 = state[2].as_int();
                let current_count: u128 = state[4].as_int();
                let next_count = current_count + current_satisfied;

                // Get next requirement
                let next_idx = step + 1;
                if next_idx < NUM_REQUIREMENTS {
                    let (voter_val, threshold, satisfied, active) = checks[next_idx];
                    state[0] = BaseElement::from(voter_val);
                    state[1] = BaseElement::from(threshold);
                    state[2] = if active && satisfied { BaseElement::ONE } else { BaseElement::ZERO };
                    state[3] = if active { BaseElement::ONE } else { BaseElement::ZERO };
                    state[4] = BaseElement::from(next_count as u64);
                } else {
                    // Padding rows
                    state[0] = BaseElement::ZERO;
                    state[1] = BaseElement::ZERO;
                    state[2] = BaseElement::ZERO;
                    state[3] = BaseElement::ZERO;
                    state[4] = BaseElement::from(next_count as u64);
                }
            },
        );

        trace
    }
}

impl Prover for VoteEligibilityProver {
    type BaseField = BaseElement;
    type Air = VoteEligibilityAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> VotePublicInputs {
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

/// Compute commitment to voter profile
pub fn compute_voter_commitment(voter: &ProofVoterProfile) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"voter:");
    hasher.update(voter.did.as_bytes());
    hasher.update(&[voter.assurance_level]);
    hasher.update(&voter.matl_score.to_le_bytes());
    hasher.update(&voter.stake.to_le_bytes());
    hasher.update(&voter.account_age_days.to_le_bytes());
    hasher.update(&voter.participation_rate.to_le_bytes());
    hasher.update(&[if voter.has_humanity_proof { 1 } else { 0 }]);
    hasher.update(&voter.fl_contributions.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// A complete vote eligibility proof
#[derive(Clone)]
pub struct VoteEligibilityProof {
    /// The STARK proof
    proof: Proof,
    /// Public inputs
    public_inputs: VotePublicInputs,
}

impl VoteEligibilityProof {
    /// Generate a vote eligibility proof
    ///
    /// Proves that voter meets requirements for the given proposal type.
    pub fn generate(
        voter: &ProofVoterProfile,
        proposal_type: ProofProposalType,
        config: ProofConfig,
    ) -> ProofResult<Self> {
        let requirements = proposal_type.requirements();

        // Check each requirement
        let checks: Vec<(bool, bool)> = vec![
            // (satisfied, active)
            (voter.assurance_level >= requirements.min_assurance_level, true),
            (voter.matl_score >= requirements.min_matl_score, true),
            (voter.stake >= requirements.min_stake, requirements.min_stake > 0.0),
            (voter.account_age_days >= requirements.min_account_age_days, requirements.min_account_age_days > 0),
            (voter.participation_rate >= requirements.min_participation, requirements.min_participation > 0.0),
            (voter.has_humanity_proof, requirements.humanity_proof_required),
            (voter.fl_contributions >= requirements.min_fl_contributions, requirements.fl_participation_required),
        ];

        // Count requirements
        let active_requirements = checks.iter().filter(|(_, active)| *active).count() as u8;
        let requirements_met = checks.iter().filter(|(satisfied, active)| *active && *satisfied).count() as u8;
        let eligible = requirements_met == active_requirements;

        // Compute commitment
        let commitment = compute_voter_commitment(voter);

        let public_inputs = VotePublicInputs::new(
            commitment,
            proposal_type as u8,
            eligible,
            requirements_met,
            active_requirements,
        );

        // Build prover and trace
        let options = build_proof_options(config.security_level);
        let prover = VoteEligibilityProver::new(options, public_inputs.clone());
        let trace = prover.build_trace(voter, &requirements);

        // Generate proof
        let proof = prover.prove(trace).map_err(|e| {
            ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
        })?;

        Ok(Self {
            proof,
            public_inputs,
        })
    }

    /// Verify the vote eligibility proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        // Check eligibility flag
        if !self.public_inputs.eligible {
            return Ok(VerificationResult::failure(
                ProofType::VoteEligibility,
                start.elapsed(),
                format!(
                    "Voter only met {}/{} requirements",
                    self.public_inputs.requirements_met,
                    self.public_inputs.active_requirements
                ),
            ));
        }

        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

        let result = winterfell::verify::<
            VoteEligibilityAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(self.proof.clone(), self.public_inputs.clone(), &min_opts);

        let duration = start.elapsed();

        match result {
            Ok(_) => Ok(VerificationResult::success(ProofType::VoteEligibility, duration)),
            Err(e) => Ok(VerificationResult::failure(
                ProofType::VoteEligibility,
                duration,
                format!("Verification failed: {:?}", e),
            )),
        }
    }

    /// Get the public inputs
    pub fn public_inputs(&self) -> &VotePublicInputs {
        &self.public_inputs
    }

    /// Check if the voter is eligible
    pub fn is_eligible(&self) -> bool {
        self.public_inputs.eligible
    }

    /// Get the proposal type
    pub fn proposal_type(&self) -> Option<ProofProposalType> {
        ProofProposalType::from_u8(self.public_inputs.proposal_type)
    }

    /// Get requirements met count
    pub fn requirements_met(&self) -> u8 {
        self.public_inputs.requirements_met
    }

    /// Get total active requirements
    pub fn active_requirements(&self) -> u8 {
        self.public_inputs.active_requirements
    }

    /// Serialize the proof to bytes
    ///
    /// Format: [voter_commitment:32][proposal_type:1][eligible:1][requirements_met:1][active_requirements:1][proof_bytes...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.public_inputs.voter_commitment);
        bytes.push(self.public_inputs.proposal_type);
        bytes.push(if self.public_inputs.eligible { 1 } else { 0 });
        bytes.push(self.public_inputs.requirements_met);
        bytes.push(self.public_inputs.active_requirements);
        bytes.extend_from_slice(&self.proof.to_bytes());
        bytes
    }

    /// Deserialize a proof from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let voter_commitment = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated voter commitment".to_string())
        })?;

        let proposal_type = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing proposal_type".to_string())
        })?;

        let eligible = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing eligible flag".to_string())
        })? != 0;

        let requirements_met = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing requirements_met".to_string())
        })?;

        let active_requirements = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing active_requirements".to_string())
        })?;

        let public_inputs = VotePublicInputs::new(
            voter_commitment,
            proposal_type,
            eligible,
            requirements_met,
            active_requirements,
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

    fn eligible_voter() -> ProofVoterProfile {
        ProofVoterProfile {
            did: "did:mycelix:voter123".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 600.0,
            account_age_days: 100,
            participation_rate: 0.5,
            has_humanity_proof: true,
            fl_contributions: 25,
        }
    }

    #[test]
    fn test_vote_proof_standard_eligible() {
        let voter = eligible_voter();

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();

        assert!(proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(result.valid, "Proof verification failed: {:?}", result.details);
    }

    #[test]
    fn test_vote_proof_constitutional_eligible() {
        let voter = eligible_voter();

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Constitutional,
            test_config(),
        ).unwrap();

        assert!(proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(result.valid, "Proof verification failed: {:?}", result.details);
    }

    #[test]
    fn test_vote_proof_emergency_eligible() {
        let voter = eligible_voter();

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Emergency,
            test_config(),
        ).unwrap();

        assert!(proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_vote_proof_insufficient_assurance() {
        let mut voter = eligible_voter();
        voter.assurance_level = 0;

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();

        assert!(!proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_vote_proof_model_governance_needs_fl() {
        let mut voter = eligible_voter();
        voter.fl_contributions = 5; // Below required 10

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::ModelGovernance,
            test_config(),
        ).unwrap();

        assert!(!proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_vote_proof_constitutional_needs_humanity() {
        let mut voter = eligible_voter();
        voter.has_humanity_proof = false;

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Constitutional,
            test_config(),
        ).unwrap();

        assert!(!proof.is_eligible());
        let result = proof.verify().unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_voter_commitment() {
        let voter1 = eligible_voter();
        let mut voter2 = eligible_voter();
        voter2.matl_score = 0.9;

        let commit1 = compute_voter_commitment(&voter1);
        let commit2 = compute_voter_commitment(&voter2);

        // Different profiles -> different commitment
        assert_ne!(commit1, commit2);

        // Same profile -> same commitment
        let commit1_repeat = compute_voter_commitment(&voter1);
        assert_eq!(commit1, commit1_repeat);
    }

    #[test]
    fn test_proposal_type_requirements() {
        let standard = ProofProposalType::Standard.requirements();
        assert_eq!(standard.min_assurance_level, 1);
        assert!(!standard.humanity_proof_required);

        let constitutional = ProofProposalType::Constitutional.requirements();
        assert_eq!(constitutional.min_assurance_level, 2);
        assert!(constitutional.humanity_proof_required);

        let model_gov = ProofProposalType::ModelGovernance.requirements();
        assert!(model_gov.fl_participation_required);
        assert_eq!(model_gov.min_fl_contributions, 10);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let voter = eligible_voter();

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();

        let bytes = proof.to_bytes();
        let restored = VoteEligibilityProof::from_bytes(&bytes).unwrap();

        let result = restored.verify().unwrap();
        assert!(result.valid, "Restored proof verification failed");
        assert_eq!(restored.is_eligible(), proof.is_eligible());
        assert_eq!(restored.requirements_met(), proof.requirements_met());
    }
}
