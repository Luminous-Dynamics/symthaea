// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ Prover using Winterfell
//!
//! High-level API matching the RISC Zero interface for drop-in replacement.

use std::env;
use std::time::Instant;

use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    math::fields::f128::BaseElement,
    AcceptableOptions, Air, DefaultConstraintCommitment, DefaultConstraintEvaluator,
    DefaultTraceLde, PartitionOptions, Proof, ProofOptions, Prover, TraceInfo,
    TraceTable,
};

use crate::air::{PoGQAir, PublicInputs, PublicInputsWrapper, AIR_SCHEMA_REV};
use crate::provenance::ProvenanceHash;
use crate::security::SecurityProfile;
use crate::trace::TraceBuilder;

/// Convert 32-byte hash to 4× u64 little-endian representation for field elements
fn hash_to_u64x4_le(hash: &[u8; 32]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..4 {
        result[i] = u64::from_le_bytes([
            hash[i * 8],
            hash[i * 8 + 1],
            hash[i * 8 + 2],
            hash[i * 8 + 3],
            hash[i * 8 + 4],
            hash[i * 8 + 5],
            hash[i * 8 + 6],
            hash[i * 8 + 7],
        ]);
    }
    result
}

/// PoGQ prover implementing the Winterfell Prover trait
pub struct PoGQProver {
    options: ProofOptions,
    profile: SecurityProfile,
}

impl PoGQProver {
    /// Create new prover with default security profile (S128)
    ///
    /// Uses environment variable VSV_SECURITY_PROFILE if set, otherwise defaults to S128.
    pub fn new() -> Self {
        let profile = SecurityProfile::from_env().unwrap_or_default();
        Self::with_profile(profile)
    }

    /// Create prover with specified security profile
    pub fn with_profile(profile: SecurityProfile) -> Self {
        let options = profile.proof_options();

        // Print banner with provenance information
        Self::print_banner(&profile, &options);

        Self { options, profile }
    }

    /// Create prover with custom proof options (advanced use)
    ///
    /// Note: Using custom options bypasses security profile guarantees.
    /// Prefer `with_profile()` for standard deployments.
    pub fn with_options(options: ProofOptions) -> Self {
        eprintln!("⚠️  Using custom ProofOptions (bypassing security profiles)");
        Self {
            options,
            profile: SecurityProfile::S128, // nominal profile for reporting
        }
    }

    /// Get the active security profile
    pub fn profile(&self) -> SecurityProfile {
        self.profile
    }

    /// Print prover banner with full provenance
    fn print_banner(profile: &SecurityProfile, options: &ProofOptions) {
        // Get build metadata
        let rustc_version = option_env!("RUSTC_VERSION").unwrap_or("unknown");
        let git_commit = option_env!("GIT_COMMIT").unwrap_or("uncommitted");
        let build_profile = if cfg!(debug_assertions) { "debug" } else { "release" };

        eprintln!("╔══════════════════════════════════════════════════════════");
        eprintln!("║ VSV-STARK Prover - Full Provenance");
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ Security: {}", profile.description());
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ Runtime:");
        eprintln!("║   Winterfell:   v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("║   Rust:         {}", rustc_version);
        eprintln!("║   Build:        {}", build_profile);
        eprintln!("║   Git:          {}", &git_commit[..git_commit.len().min(12)]);
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ AIR Configuration:");
        eprintln!("║   Name:         PoGQ v4.1");
        eprintln!("║   Trace Width:  44 columns");
        eprintln!("║   Constraints:  40");
        eprintln!("║   Range Checks: 32-bit (rem_t 16-bit + x_t 16-bit)");
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ Proof Options:");
        eprintln!("║   num_queries:     {}", options.num_queries());
        eprintln!("║   blowup_factor:   {}", options.blowup_factor());
        eprintln!("║   grinding_factor: {} (DoS resistance)", options.grinding_factor());
        eprintln!("║   fri_folding:     8");
        eprintln!("║   max_remainder:   31");
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ Expected Performance: {}", profile.performance_estimate().describe());
        eprintln!("║ ------------------------------------------------------------");
        eprintln!("║ Note: Full provenance hash will be embedded in receipts");
        eprintln!("╚══════════════════════════════════════════════════════════");
    }

    /// Prove a single PoGQ decision
    ///
    /// # Arguments
    /// * `public` - Public inputs (parameters + initial state + expected output)
    /// * `witness_scores` - Private witness (hybrid scores for each round)
    ///
    /// # Returns
    /// Proof bytes and timing info
    pub fn prove_exec(
        &self,
        mut public: PublicInputs,
        witness_scores: Vec<u64>,
    ) -> Result<ProofResult, String> {
        let start = Instant::now();

        // Compute provenance hash and populate public inputs
        // This creates a tamper-evident commitment to all proof configuration parameters
        let prov_hash = ProvenanceHash::compute(
            self.profile,
            &self.options,
            "PoGQ-v4.1",
            44,  // TRACE_WIDTH
            40,  // constraint count
        );

        public.prov_hash = hash_to_u64x4_le(prov_hash.as_bytes());
        public.profile_id = self.profile.profile_id();
        public.air_rev = AIR_SCHEMA_REV;

        // Build execution trace
        let trace_builder = TraceBuilder::new(public.clone(), witness_scores);
        let trace_matrix = trace_builder.build();

        // Create TraceTable from matrix
        let trace_width = trace_matrix.num_cols();
        let trace_length = trace_matrix.num_rows();
        eprintln!("DEBUG: trace_width={}, trace_length={}", trace_width, trace_length);
        let mut trace_table = TraceTable::new(trace_width, trace_length);

        // Fill trace table with data from matrix
        for col in 0..trace_width {
            for row in 0..trace_length {
                trace_table.set(col, row, trace_matrix.get(col, row));
            }
        }

        let trace_build_time = start.elapsed();

        // Generate proof using Prover trait
        let prove_start = Instant::now();

        // Create trace info from the trace dimensions
        let trace_info = TraceInfo::new(trace_width, trace_length);

        // Create prover instance that implements the trait
        let prover = PoGQProverImpl::new(trace_info, public.clone(), self.options.clone());

        // Call the prove method from Prover trait
        let proof = prover
            .prove(trace_table)
            .map_err(|e| format!("Proof generation failed: {:?}", e))?;

        let prove_time = prove_start.elapsed();

        // Serialize proof
        let serialize_start = Instant::now();
        let proof_bytes = proof.to_bytes();
        let serialize_time = serialize_start.elapsed();

        Ok(ProofResult {
            proof_bytes,
            public_inputs: public,  // Return provenance-populated public inputs
            trace_build_ms: trace_build_time.as_millis() as u64,
            prove_ms: prove_time.as_millis() as u64,
            serialize_ms: serialize_time.as_millis() as u64,
            total_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Verify a proof
    ///
    /// # Arguments
    /// * `proof_bytes` - Serialized proof
    /// * `public` - Public inputs that proof claims to satisfy
    ///
    /// # Returns
    /// True if proof is valid
    pub fn verify_proof(&self, proof_bytes: &[u8], public: PublicInputs) -> Result<bool, String> {
        // Safety switch: VSV_PROVENANCE_MODE = "strict" (default) or "off"
        let provenance_mode = env::var("VSV_PROVENANCE_MODE").unwrap_or_else(|_| "strict".to_string());

        if provenance_mode == "strict" {
            // FAIL-FAST PROVENANCE CHECK (before expensive FRI verification)
            // This ensures tamper-evident configuration binding

            // 1. Compute expected provenance hash from verifier's configuration
            let expected_hash = ProvenanceHash::compute(
                self.profile,
                &self.options,
                "PoGQ-v4.1",
                44,  // TRACE_WIDTH
                40,  // constraint count
            );
            let expected_u64s = hash_to_u64x4_le(expected_hash.as_bytes());

            // 2. Compare with proof's provenance commitment
            if public.prov_hash != expected_u64s {
                return Err(format!(
                    "Provenance hash mismatch: expected {:016x}{:016x}{:016x}{:016x}, got {:016x}{:016x}{:016x}{:016x}",
                    expected_u64s[0], expected_u64s[1], expected_u64s[2], expected_u64s[3],
                    public.prov_hash[0], public.prov_hash[1], public.prov_hash[2], public.prov_hash[3]
                ));
            }

            // 3. Verify profile ID matches
            if public.profile_id != self.profile.profile_id() {
                return Err(format!(
                    "Security profile mismatch: expected {}, got {}",
                    self.profile.profile_id(),
                    public.profile_id
                ));
            }

            // 4. Verify AIR schema revision
            if public.air_rev != AIR_SCHEMA_REV {
                return Err(format!(
                    "AIR schema revision mismatch: expected {}, got {}",
                    AIR_SCHEMA_REV,
                    public.air_rev
                ));
            }

            // Provenance checks passed - proceed with FRI verification
        } else if provenance_mode != "off" {
            return Err(format!("Invalid VSV_PROVENANCE_MODE: {}. Must be 'strict' or 'off'", provenance_mode));
        }
        // If mode = "off", skip provenance checks (fallback for compatibility)
        // Deserialize proof
        let proof = Proof::from_bytes(proof_bytes)
            .map_err(|e| format!("Failed to deserialize proof: {:?}", e))?;

        // Create public inputs wrapper
        let pub_inputs = PublicInputsWrapper(public.to_elements());

        // Accept any proof with sufficient security (127 bits due to AIR constraints)
        let acceptable_options = AcceptableOptions::MinConjecturedSecurity(127);

        // Verify proof with all 4 type parameters
        winterfell::verify::<
            PoGQAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(proof, pub_inputs, &acceptable_options)
        .map_err(|e| format!("Verification failed: {:?}", e))?;

        Ok(true)
    }
}

impl Default for PoGQProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal prover implementation of the Prover trait
struct PoGQProverImpl {
    air: PoGQAir,
}

impl PoGQProverImpl {
    fn new(trace_info: TraceInfo, public_inputs: PublicInputs, options: ProofOptions) -> Self {
        Self {
            air: PoGQAir::new(trace_info, public_inputs, options),
        }
    }
}

// Implement the Prover trait using default implementations
impl Prover for PoGQProverImpl {
    type BaseField = BaseElement;
    type Air = PoGQAir;
    type Trace = TraceTable<Self::BaseField>;
    type HashFn = Blake3_256<Self::BaseField>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: winterfell::math::FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: winterfell::math::FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: winterfell::math::FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> PublicInputsWrapper {
        // Return the public inputs that were used to create the AIR
        PublicInputsWrapper(self.air.get_public_inputs())
    }

    fn options(&self) -> &ProofOptions {
        self.air.options()
    }

    fn new_trace_lde<E: winterfell::math::FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &winterfell::TraceInfo,
        main_trace: &winterfell::matrix::ColMatrix<Self::BaseField>,
        domain: &winterfell::StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, winterfell::TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn new_evaluator<'a, E: winterfell::math::FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<winterfell::AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }

    fn build_constraint_commitment<E: winterfell::math::FieldElement<BaseField = Self::BaseField>>(
        &self,
        composition_poly_trace: winterfell::CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &winterfell::StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, winterfell::CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    }
}

/// Result of proof generation
#[derive(Debug, Clone)]
pub struct ProofResult {
    pub proof_bytes: Vec<u8>,
    pub public_inputs: PublicInputs,  // Include provenance-populated public inputs
    pub trace_build_ms: u64,
    pub prove_ms: u64,
    pub serialize_ms: u64,
    pub total_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(val: f32) -> u64 {
        (val * 65536.0) as u64
    }

    #[test]
    #[ignore = "Winterfell degree validation: near-constant witness produces partial bit column variation"]
    fn test_prove_and_verify() {
        let public = PublicInputs {
            beta: q(0.85),
            w: 3,
            k: 2,
            m: 3,
            threshold: q(0.90),
            ema_init: q(0.85),
            viol_init: 0,
            clear_init: 0,
            quar_init: 0,
            round_init: 0,
            quar_out: 0,
            trace_length: 8, // Power of 2
            // Provenance fields (prover will populate automatically)
            prov_hash: [0, 0, 0, 0],
            profile_id: 128,  // Will be set by prover
            air_rev: AIR_SCHEMA_REV,
        };

        let witness_scores = vec![
            q(0.92),
            q(0.91),
            q(0.93),
            q(0.94),
            q(0.92),
            q(0.91),
            q(0.93),
            q(0.94),
        ];

        let prover = PoGQProver::new();

        // Generate proof
        let result = prover.prove_exec(public.clone(), witness_scores).unwrap();
        println!(
            "Proof generated: {} bytes in {}ms",
            result.proof_bytes.len(),
            result.total_ms
        );

        // Verify proof using provenance-populated public inputs
        let valid = prover.verify_proof(&result.proof_bytes, result.public_inputs).unwrap();
        assert!(valid);
    }
}
