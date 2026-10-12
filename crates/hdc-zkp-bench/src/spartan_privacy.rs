// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Smoke tests for the *separate* zero-knowledge-capable Iron Spartan backend.
//!
//! The production HDC benchmark still uses the standard Binius64 backend, which
//! the vendored architecture notes explicitly describe as non-zero-knowledge.
//! This test is only a tiny integration proof showing how a private Boolean
//! relation can be expressed and proved with Iron Spartan. It is not a migration
//! of the large packed-word HDC circuits and not an independent crypto audit.
//! It checks statement acceptance/binding and Boolean-witness rejection only; it
//! does not test whether proofs statistically hide private witness values. The
//! fixed-seed StdRng below is test-only and must never be reused for production
//! proof randomness.
//!
//! The Iron Spartan verifier currently declares SECURITY_BITS = 96; production
//! use requires an explicit security-parameter review before making stronger
//! confidentiality or deployment claims.

use binius_field::{arch::OptimalPackedB128, BinaryField128bGhash as B128, Field};
use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
use binius_spartan_frontend::{
    circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
    compiler::compile,
};
use binius_spartan_prover::Prover;
use binius_spartan_verifier::{config::StdChallenger, Verifier};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use rand::{rngs::StdRng, SeedableRng};

/// Add private witness bits and constrain their public XOR result.
///
/// Booleanity constraints are necessary: an arbitrary GF(2^128) element is not
/// automatically a bit. In characteristic two, the relation is a + b = c.
fn xor_bit_relation<B>(
    builder: &mut B,
    input_a: B128,
    input_b: B128,
    public_output: B::Wire,
) where
    B: CircuitBuilder<Field = B128>,
{
    let a = builder.hint::<_, 0, 1>([], move |_| [input_a])[0];
    let b = builder.hint::<_, 0, 1>([], move |_| [input_b])[0];

    let a_squared = builder.mul(a, a);
    builder.assert_eq(a_squared, a);
    let b_squared = builder.mul(b, b);
    builder.assert_eq(b_squared, b);

    let xor = builder.add(a, b);
    builder.assert_eq(xor, public_output);
}

#[test]
fn iron_spartan_proves_private_boolean_xor_and_rejects_non_bits() {
    let a_value = B128::ONE;
    let b_value = B128::ZERO;
    let output_value = a_value + b_value;

    // Construct the statement with one public output and two private hint wires.
    let mut constraint_builder = ConstraintBuilder::<B128>::new();
    let output_wire = constraint_builder.alloc_inout();
    xor_bit_relation(&mut constraint_builder, a_value, b_value, output_wire);
    let (constraint_system, layout) = compile(constraint_builder);

    let compression = StdCompression::default();
    let verifier = Verifier::<B128, StdDigest, _>::setup(
        constraint_system,
        1,
        compression.clone(),
    )
    .expect("Iron Spartan verifier setup failed");
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(
        verifier.clone(),
        ParallelCompressionAdaptor::new(compression),
    )
    .expect("Iron Spartan prover setup failed");

    // Match the verifier's padded witness layout, including its blinding wires.
    let cs = verifier.constraint_system();
    let layout = layout.with_blinding(cs.blinding_info().clone());

    let mut witness_generator = WitnessGenerator::new(&layout);
    let output_assigned = witness_generator.write_inout(output_wire, output_value);
    xor_bit_relation(
        &mut witness_generator,
        a_value,
        b_value,
        output_assigned,
    );
    let witness = witness_generator
        .build()
        .expect("valid Boolean XOR witness should be constructible");
    cs.validate(&witness);

    let public = &witness[..1 << cs.log_public()];
    let mut rng = StdRng::seed_from_u64(0x5359_4d54);
    let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
    prover
        .prove(&witness, &mut rng, &mut prover_transcript)
        .expect("Iron Spartan proof generation failed");

    let proof = prover_transcript.finalize();

    // Verify the original public statement from the serialized proof bytes.
    let mut verifier_transcript =
        VerifierTranscript::new(StdChallenger::default(), proof.clone());
    verifier
        .verify(public, &mut verifier_transcript)
        .expect("Iron Spartan rejected a valid public XOR statement");
    verifier_transcript
        .finalize()
        .expect("Iron Spartan transcript finalization failed");

    // A proof for output 1 must not verify if the public output is changed to 0.
    // The verifier absorbs public inputs into Fiat-Shamir, so this mutation
    // must invalidate the proof rather than silently rebind it to a new claim.
    let tampered_public = [output_value + B128::ONE];
    assert_eq!(tampered_public[0], B128::ZERO);
    let mut tampered_transcript =
        VerifierTranscript::new(StdChallenger::default(), proof);
    assert!(
        verifier.verify(&tampered_public, &mut tampered_transcript).is_err(),
        "Iron Spartan must reject a proof when its public XOR output is altered"
    );

    // An arbitrary field element must not pass as a Boolean witness bit.
    let bad_a = B128::new(2);
    let bad_b = B128::ZERO;
    let mut invalid_generator = WitnessGenerator::new(&layout);
    let invalid_output = invalid_generator.write_inout(output_wire, bad_a + bad_b);
    xor_bit_relation(
        &mut invalid_generator,
        bad_a,
        bad_b,
        invalid_output,
    );
    assert!(
        invalid_generator.build().is_err(),
        "the bitness constraint must reject a non-Boolean field element"
    );
}
