// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Translation Coordinator Zome
//!
//! Community-driven translation with trust-weighted verification.
//!
//! ## Sybil Resistance (per DHT guardrail)
//!
//! Votes are weighted by the voter's consciousness tier, NOT simple count.
//! Verification threshold: weighted approval score >= 2000 (equivalent to
//! ~3 Citizen-tier votes or ~2 Steward-tier votes). This prevents low-trust
//! Sybil accounts from gaming translations.
//!
//! ## Auto-Credential
//!
//! At 50+ verified translations for a language, the translator earns a
//! "Verified Educational Translator" W3C credential.

use hdk::prelude::*;
use mycelix_zome_helpers as _;
use translation_integrity::*;

// ============== Helper Functions ==============

fn get_agent_tier(agent: AgentPubKey) -> ExternResult<f64> {
    let response = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from("identity_bridge"),
        FunctionName::from("get_agent_tier_score"),
        None,
        agent,
    )?;

    match response {
        ZomeCallResponse::Ok(bytes) => {
            let score: f64 = bytes.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            Ok(score)
        }
        _ => Ok(0.0),
    }
}

// ============== Extern Functions ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct TranslationProposalInput {
    pub source_lang: String,
    pub target_lang: String,
    pub context_key: String,
    pub source_text: String,
    pub proposed_translation: String,
}

#[hdk_extern]
pub fn propose_translation(input: TranslationProposalInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;
    let proposal = TranslationProposal {
        source_lang: input.source_lang,
        target_lang: input.target_lang,
        context_key: input.context_key,
        source_text: input.source_text,
        proposed_translation: input.proposed_translation,
        proposer: agent_info()?.agent_initial_pubkey,
        proposed_at: now.as_micros() as i64,
    };

    create_entry(EntryTypes::TranslationProposal(proposal))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TranslationVoteInput {
    pub proposal_hash: ActionHash,
    pub approved: bool,
    pub native_speaker_attestation: bool,
}

#[hdk_extern]
pub fn vote_on_translation(input: TranslationVoteInput) -> ExternResult<ActionHash> {
    let voter = agent_info()?.agent_initial_pubkey;
    let tier_score = get_agent_tier(voter.clone())?;

    let vote = TranslationVote {
        proposal_hash: input.proposal_hash,
        voter,
        approved: input.approved,
        native_speaker_attestation: input.native_speaker_attestation,
        voter_consciousness_permille: (tier_score * 1000.0) as u16,
        voted_at: (sys_time()?.as_micros() / 1000) as i64,
    };

    create_entry(EntryTypes::TranslationVote(vote))
}
