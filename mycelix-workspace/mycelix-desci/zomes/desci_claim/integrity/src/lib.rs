// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DeSci Claim Integrity Zome
//! Defines entry types for scientific claims and reviews.

use hdi::prelude::*;
extern crate holochain_serialized_bytes;
use mycelix_zkp_core::circuits::review_integrity::{ReviewIntegrityAir, ReviewPublicInputs};
use mycelix_zkp_core::winterfell::{
    AcceptableOptions, BatchingMethod, FieldExtension, ProofOptions,
};

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DesciClaim {
    pub content_hash: [u8; 32],
    pub creator: AgentPubKey,
    pub created_at: Timestamp,
    /// LEM Cube classification (E, N, M)
    pub lem_e: u8, // 0-4
    pub lem_n: u8, // 0-3
    pub lem_m: u8, // 0-3
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Review {
    pub claim_address: ActionHash,
    pub reviewer: AgentPubKey,
    pub score: u8, // 1-10
    /// STARK proof of integrity (Winterfell)
    pub proof_bytes: Vec<u8>,
    pub expertise_commitment: [u8; 32],
    pub review_commitment: [u8; 32],
    pub paper_id_comm: [u8; 32],
    pub author_id_hash: [u8; 32],
    pub min_expertise: u64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    DesciClaim(DesciClaim),
    Review(Review),
}

#[hdk_link_types]
pub enum LinkTypes {
    ClaimToReview,
    AgentToClaim,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::DesciClaim(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Review(review) => {
                    // STARK verification — enforcing the Epistemic Membrane
                    let pub_inputs = ReviewPublicInputs {
                        min_expertise: review.min_expertise,
                        paper_id_comm: review.paper_id_comm,
                        author_id_hash: review.author_id_hash,
                        reviewer_commitment: review.expertise_commitment,
                    };

                    let proof =
                        mycelix_zkp_core::winterfell::Proof::from_bytes(&review.proof_bytes)
                            .map_err(|e| {
                                wasm_error!(WasmErrorInner::Guest(format!(
                                    "Proof deserialization failed: {:?}",
                                    e
                                )))
                            })?;

                    // Default options used for verification on sovereign nodes
                    let options = ProofOptions::new(
                        28,
                        8,
                        0,
                        FieldExtension::None,
                        8,
                        31,
                        BatchingMethod::Linear,
                        BatchingMethod::Linear,
                    );
                    let acceptable = AcceptableOptions::OptionSet(vec![options]);

                    match mycelix_zkp_core::winterfell::verify::<
                        ReviewIntegrityAir,
                        mycelix_zkp_core::winterfell::crypto::hashers::Rp64_256,
                        mycelix_zkp_core::winterfell::crypto::DefaultRandomCoin<
                            mycelix_zkp_core::winterfell::crypto::hashers::Rp64_256,
                        >,
                        mycelix_zkp_core::winterfell::crypto::MerkleTree<
                            mycelix_zkp_core::winterfell::crypto::hashers::Rp64_256,
                        >,
                    >(proof, pub_inputs, &acceptable)
                    {
                        Ok(_) => Ok(ValidateCallbackResult::Valid),
                        Err(e) => Ok(ValidateCallbackResult::Invalid(format!(
                            "STARK Review Integrity Verification Failed: {:?}",
                            e
                        ))),
                    }
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
