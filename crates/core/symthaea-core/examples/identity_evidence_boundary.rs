// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demonstrate the identifiability boundary between vector similarity and identity evidence.
//!
//! Each query vector is evaluated as a similarity-only input and as an input accompanied
//! by a registry-issued receipt. Its hypothetical unregistered-identity counterpart has
//! the exact same vector. Similarity-only outputs must be identical for both semantics;
//! only the separate evidence channel can support an identity claim.

use std::error::Error;

use symthaea_core::hdc::identity_evidence::{
    IdentityResolution, ObservationIdentityEvidence, PatternRegistry, ResolutionPolicy,
    SimilarityResolution,
};
use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 4_096;
const PROTOTYPE_COUNT: usize = 64;
const QUERIES_PER_NOISE: usize = 256;
const NOISE_RATES: [f64; 3] = [0.05, 0.10, 0.20];
const SEEDS: [u64; 5] = [17, 29, 41, 53, 67];

#[derive(Default)]
struct Counts {
    vector_candidate_only: usize,
    vector_ambiguous: usize,
    vector_no_candidate: usize,
    nearest_correct: usize,
    registration_only_evidence_required: usize,
    registration_only_conflict: usize,
    receipt_verified: usize,
    receipt_conflict: usize,
    identical_counterfactual_vectors: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let policy = ResolutionPolicy::new(0.40, 0.02)?;
    println!("# protocol=identity-evidence-boundary-v1");
    println!("# policy=min_similarity:0.40,min_top_two_margin:0.02");
    println!("# identity_claim_requires_receipt=true");
    println!("# receipt_is_cryptographic_attestation=false");
    println!(
        "seed,noise_fraction,dimension,prototype_count,queries,vector_candidate_only,vector_ambiguous,vector_no_candidate,nearest_identity_top1,registration_only_evidence_required,registration_only_conflict,observation_verified,observation_conflict,identical_counterfactual_vectors"
    );

    for seed in SEEDS {
        let mut registry = PatternRegistry::new(format!("registry-seed-{seed}"))?;
        let mut receipt_by_index = Vec::with_capacity(PROTOTYPE_COUNT);
        let mut prototype_by_index = Vec::with_capacity(PROTOTYPE_COUNT);

        for index in 0..PROTOTYPE_COUNT {
            let prototype = PatternVector::seeded_random(
                DIMENSION,
                derive_seed(seed, 0x5052_4f54, index as u64),
            )?;
            let receipt = registry.register(format!("entity-{index:04}"), prototype.clone())?;
            receipt_by_index.push(receipt);
            prototype_by_index.push(prototype);
        }

        for (noise_index, noise_fraction) in NOISE_RATES.iter().copied().enumerate() {
            let mut counts = Counts::default();
            for sample_index in 0..QUERIES_PER_NOISE {
                let identity_index =
                    (sample_index * 37 + noise_index * 19 + seed as usize) % PROTOTYPE_COUNT;
                let query = prototype_by_index[identity_index].with_noise(
                    noise_fraction,
                    derive_seed(seed, noise_index as u64, sample_index as u64),
                )?;

                // Same observation, no identity capability: report resemblance only.
                let similarity_only = registry.resolve_similarity_only(&query, policy)?;
                let nearest_identity = match &similarity_only {
                    SimilarityResolution::CandidateOnly { candidate, .. } => {
                        counts.vector_candidate_only += 1;
                        candidate.identity_id.as_str()
                    }
                    SimilarityResolution::Ambiguous { candidates, .. } => {
                        counts.vector_ambiguous += 1;
                        candidates
                            .first()
                            .ok_or("ambiguous result has no candidates")?
                            .identity_id
                            .as_str()
                    }
                    SimilarityResolution::NoCandidate { nearest } => {
                        counts.vector_no_candidate += 1;
                        nearest.identity_id.as_str()
                    }
                };
                if nearest_identity == format!("entity-{identity_index:04}") {
                    counts.nearest_correct += 1;
                }

                // Enrollment alone proves the prototype is registered, not that
                // this particular observation belongs to that identity.
                match registry.resolve_with_receipt(
                    &query,
                    &receipt_by_index[identity_index],
                    policy,
                )? {
                    IdentityResolution::EvidenceRequired { .. } => {
                        counts.registration_only_evidence_required += 1;
                    }
                    IdentityResolution::EvidenceConflict { .. } => {
                        counts.registration_only_conflict += 1;
                    }
                    IdentityResolution::Verified { .. } => {
                        return Err("registration-only path must never verify identity".into());
                    }
                }

                // Synthetic ground truth plays the role of an already-authenticated
                // independent source. This does not authenticate real identities.
                let observation_id = format!(
                    "seed-{seed}/noise-{noise_index}/entity-{identity_index}/sample-{sample_index}"
                );
                let observation_evidence = ObservationIdentityEvidence::from_authenticated_source(
                    &receipt_by_index[identity_index],
                    &observation_id,
                    &query,
                    "synthetic-known-query-oracle",
                )?;
                match registry.resolve_with_observation_evidence(
                    &query,
                    &observation_id,
                    &receipt_by_index[identity_index],
                    &observation_evidence,
                    policy,
                )? {
                    IdentityResolution::Verified { .. } => counts.receipt_verified += 1,
                    IdentityResolution::EvidenceConflict { .. } => counts.receipt_conflict += 1,
                    IdentityResolution::EvidenceRequired { .. } => {
                        return Err("observation evidence was supplied but was not consumed".into());
                    }
                }

                // Counterfactual semantic control: exactly this vector is also
                // assigned an unregistered identity. Its vector-only result cannot
                // change, because no observable feature changed.
                let counterfactual_unknown_vector = query.clone();
                let counterfactual_result =
                    registry.resolve_similarity_only(&counterfactual_unknown_vector, policy)?;
                if counterfactual_unknown_vector == query
                    && counterfactual_result == similarity_only
                {
                    counts.identical_counterfactual_vectors += 1;
                }
            }

            println!(
                "{seed},{noise_fraction:.2},{DIMENSION},{PROTOTYPE_COUNT},{QUERIES_PER_NOISE},{},{},{},{},{},{},{},{},{}",
                counts.vector_candidate_only,
                counts.vector_ambiguous,
                counts.vector_no_candidate,
                counts.nearest_correct,
                counts.registration_only_evidence_required,
                counts.registration_only_conflict,
                counts.receipt_verified,
                counts.receipt_conflict,
                counts.identical_counterfactual_vectors
            );
        }
    }

    Ok(())
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_seed(seed: u64, first: u64, second: u64) -> u64 {
    mix64(seed ^ mix64(first) ^ mix64(second.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}
