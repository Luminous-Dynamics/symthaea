// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DeSci Claim Coordinator Zome
//! Handles creation of claims and submission of ZK-verified reviews.

use desci_claim_integrity::*;
use hdk::prelude::*;

#[hdk_extern]
pub fn create_claim(claim: DesciClaim) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::DesciClaim(claim.clone()))?;

    // Link from agent to claim
    create_link(
        agent_info()?.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToClaim,
        (),
    )?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn submit_review(review: Review) -> ExternResult<ActionHash> {
    // 1. Structural verification of the review
    if review.score < 1 || review.score > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Score must be between 1 and 10".into()
        )));
    }

    // 2. Commit the review to the source chain and DHT
    let action_hash = create_entry(EntryTypes::Review(review.clone()))?;

    // 3. Link from claim to review
    create_link(
        review.claim_address,
        action_hash.clone(),
        LinkTypes::ClaimToReview,
        (),
    )?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_claim_reviews(claim_hash: ActionHash) -> ExternResult<Vec<Review>> {
    let links = get_links(
        LinkQuery::try_new(claim_hash, LinkTypes::ClaimToReview)?,
        GetStrategy::Network,
    )?;

    let mut reviews = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid review target".into())))?;

        let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Review not found".into())
        ))?;

        let review: Review = record
            .entry()
            .to_app_option()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Serialization error: {:?}",
                    e
                )))
            })?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Not a review".into())))?;
        reviews.push(review);
    }

    Ok(reviews)
}
