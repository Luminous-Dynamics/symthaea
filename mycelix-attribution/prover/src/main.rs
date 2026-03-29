// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! mycelix-attribution-prover — Generates ZK-STARK proofs for usage attestations.
//!
//! Builds a 9-column × 16-row trace and generates a Winterfell STARK proof
//! proving knowledge of a valid usage scale without revealing it.

use clap::Parser;
use mycelix_attribution_stark_common::{
    commitment_to_limbs, compute_witness_commitment, default_proof_options, hash_to_u64_pair,
    scale_to_u8, PublicInputs, UsageAttestationAir, TRACE_LENGTH, TRACE_WIDTH,
};
use serde::Serialize;
use winter_crypto::hashers::Blake3_256;
use winter_math::fields::f128::BaseElement;
use winter_math::FieldElement;
use winterfell::{
    crypto::{DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions,
    ProofOptions, Prover, StarkDomain, TraceInfo, TraceTable,
};

// ── CLI Arguments ───────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "mycelix-attribution-prover")]
#[command(about = "Generate ZK-STARK proofs for usage attestations")]
struct Args {
    /// Dependency ID (e.g. "crate:serde:1.0")
    #[arg(long)]
    dependency_id: String,

    /// User DID (e.g. "did:mycelix:abc123")
    #[arg(long)]
    user_did: String,

    /// Usage scale: small, medium, large, enterprise
    #[arg(long)]
    usage_scale: String,

    /// Organization name (optional)
    #[arg(long, default_value = "")]
    organization: String,

    /// Attestation ID (default: generated from dependency_id + user_did)
    #[arg(long)]
    id: Option<String>,

    /// Original action hash from DHT (hex-encoded, for verification pipeline)
    #[arg(long)]
    original_action_hash: Option<String>,
}

// ── Output Type ─────────────────────────────────────────────────────

#[derive(Serialize, Debug)]
struct AttestationOutput {
    id: String,
    original_action_hash: String,
    dependency_id: String,
    user_did: String,
    witness_commitment: String,
    proof_bytes: String,
    public_inputs: Vec<u64>,
    proof_size_bytes: usize,
}

// ── Prover Implementation ───────────────────────────────────────────

struct AttestationProver {
    options: ProofOptions,
    pub_inputs: PublicInputs,
}

impl AttestationProver {
    fn new(pub_inputs: PublicInputs) -> Self {
        Self {
            options: default_proof_options(),
            pub_inputs,
        }
    }
}

impl Prover for AttestationProver {
    type BaseField = BaseElement;
    type Air = UsageAttestationAir;
    type Trace = TraceTable<Self::BaseField>;
    type HashFn = Blake3_256<Self::BaseField>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> PublicInputs {
        self.pub_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, winterfell::TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<winterfell::AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        composition_poly_trace: winterfell::CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (
        Self::ConstraintCommitment<E>,
        winterfell::CompositionPoly<E>,
    ) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    }
}

/// Build a 9-column × 16-row trace.
///
/// Column 0 (scale) varies across rows with valid values [1-4] to create
/// a non-trivial polynomial, ensuring the degree-4 range check constraint
/// produces a non-zero polynomial on the extended evaluation domain.
/// Row 0 holds the prover's actual scale; remaining rows cycle through 1-4.
/// Columns 1-4 are constant (hash values), bound by boundary assertions.
/// Columns 5-8 are constant (witness commitment limbs), bound by boundary assertions.
fn build_trace(
    scale: u8,
    dep_hash_lo: u64,
    dep_hash_hi: u64,
    did_hash_lo: u64,
    did_hash_hi: u64,
    commitment_limbs: &[u64; 4],
) -> TraceTable<BaseElement> {
    // Valid scale values for padding rows (cycle through 1,2,3,4)
    let scale_cycle: [u64; 4] = [1, 2, 3, 4];

    let cl = *commitment_limbs;
    let mut trace = TraceTable::new(TRACE_WIDTH, TRACE_LENGTH);
    trace.fill(
        |state| {
            // Row 0: actual scale + hash values + commitment limbs
            state[0] = BaseElement::from(scale as u64);
            state[1] = BaseElement::from(dep_hash_lo);
            state[2] = BaseElement::from(dep_hash_hi);
            state[3] = BaseElement::from(did_hash_lo);
            state[4] = BaseElement::from(did_hash_hi);
            state[5] = BaseElement::from(cl[0]);
            state[6] = BaseElement::from(cl[1]);
            state[7] = BaseElement::from(cl[2]);
            state[8] = BaseElement::from(cl[3]);
        },
        |step, state| {
            // Rows 1-7: cycle scale through valid values, keep other columns constant
            state[0] = BaseElement::from(scale_cycle[step % 4]);
            // Columns 1-8 retain previous values (constant)
        },
    );
    trace
}

fn generate_proof(
    scale: u8,
    dep_id: &str,
    user_did: &str,
    organization: &str,
) -> Result<(Vec<u8>, PublicInputs), String> {
    let (dep_lo, dep_hi) = hash_to_u64_pair(dep_id.as_bytes());
    let (did_lo, did_hi) = hash_to_u64_pair(user_did.as_bytes());

    let commitment = compute_witness_commitment(scale, dep_id, user_did, organization);
    let commitment_limbs = commitment_to_limbs(&commitment);

    let pub_inputs = PublicInputs {
        dep_hash_lo: dep_lo,
        dep_hash_hi: dep_hi,
        did_hash_lo: did_lo,
        did_hash_hi: did_hi,
        witness_commitment: commitment_limbs,
    };

    let trace = build_trace(scale, dep_lo, dep_hi, did_lo, did_hi, &commitment_limbs);

    let prover = AttestationProver::new(pub_inputs.clone());
    let proof = prover
        .prove(trace)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    let proof_bytes = proof.to_bytes();

    Ok((proof_bytes, pub_inputs))
}

fn main() {
    let args = Args::parse();

    let scale = scale_to_u8(&args.usage_scale).unwrap_or_else(|| {
        eprintln!(
            "Invalid usage_scale: '{}'. Expected: small, medium, large, enterprise",
            args.usage_scale
        );
        std::process::exit(1);
    });

    eprintln!("Generating ZK-STARK proof...");
    eprintln!("  dependency: {}", args.dependency_id);
    eprintln!("  user: {}", args.user_did);
    eprintln!("  scale: {} ({})", args.usage_scale, scale);

    let commitment = compute_witness_commitment(
        scale,
        &args.dependency_id,
        &args.user_did,
        &args.organization,
    );

    let (proof_bytes, pub_inputs) = generate_proof(
        scale,
        &args.dependency_id,
        &args.user_did,
        &args.organization,
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed: {}", e);
        std::process::exit(1);
    });

    let public_inputs_vec = vec![
        pub_inputs.dep_hash_lo,
        pub_inputs.dep_hash_hi,
        pub_inputs.did_hash_lo,
        pub_inputs.did_hash_hi,
        pub_inputs.witness_commitment[0],
        pub_inputs.witness_commitment[1],
        pub_inputs.witness_commitment[2],
        pub_inputs.witness_commitment[3],
    ];

    // Generate attestation ID from dep+did if not provided
    let attest_id = args.id.unwrap_or_else(|| {
        let hash = blake3::hash(format!("{}:{}", args.dependency_id, args.user_did).as_bytes());
        format!("attest-{}", &hex::encode(hash.as_bytes())[..16])
    });

    let output = AttestationOutput {
        id: attest_id,
        original_action_hash: args.original_action_hash.unwrap_or_default(),
        dependency_id: args.dependency_id,
        user_did: args.user_did,
        witness_commitment: hex::encode(commitment),
        proof_bytes: hex::encode(&proof_bytes),
        public_inputs: public_inputs_vec,
        proof_size_bytes: proof_bytes.len(),
    };

    eprintln!("  proof size: {} bytes", output.proof_size_bytes);
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use winterfell::verify;

    #[test]
    fn test_valid_proof_generation() {
        let (proof_bytes, pub_inputs) = generate_proof(
            2, // Medium
            "crate:serde:1.0",
            "did:mycelix:abc",
            "Acme Corp",
        )
        .unwrap();

        assert!(!proof_bytes.is_empty());
        assert_ne!(pub_inputs.dep_hash_lo, 0);
        assert_ne!(pub_inputs.did_hash_lo, 0);
    }

    #[test]
    fn test_roundtrip_prove_verify() {
        let (proof_bytes, pub_inputs) = generate_proof(
            3, // Large
            "npm:react:18.0",
            "did:mycelix:xyz",
            "",
        )
        .unwrap();

        let proof = winterfell::Proof::from_bytes(&proof_bytes).unwrap();

        let result = verify::<
            UsageAttestationAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(
            proof,
            pub_inputs,
            &winterfell::AcceptableOptions::OptionSet(vec![default_proof_options()]),
        );

        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_invalid_scale_rejected() {
        assert!(scale_to_u8("huge").is_none());
        assert!(scale_to_u8("").is_none());
    }
}
