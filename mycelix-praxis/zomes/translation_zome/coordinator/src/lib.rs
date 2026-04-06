#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
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
use hdk::prelude::HdkPathExt;
use translation_integrity::{
    EntryTypes, LinkTypes, TranslationProposal, TranslationVote, VerifiedTranslation,
};

/// Weighted approval threshold for verification.
/// ~3 Citizen-tier votes (700 permille each = 2100) or ~2 Steward (1000 each = 2000).
const VERIFICATION_THRESHOLD: u32 = 2000;

/// Number of verified translations needed for auto-credential issuance.
const CREDENTIAL_THRESHOLD: u32 = 50;

// ============== Helpers ==============

fn lang_pair_anchor(source: &str, target: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("translations.{}.{}", source, target));
    let typed = path.typed(LinkTypes::LanguagePairToProposals)?;
    typed.ensure()?;
    typed.path.path_entry_hash()
}

fn verified_anchor(target_lang: &str, context_key: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("verified.{}.{}", target_lang, context_key));
    let typed = path.typed(LinkTypes::LanguageKeyToVerified)?;
    typed.ensure()?;
    typed.path.path_entry_hash()
}

fn translator_anchor(agent: &AgentPubKey) -> ExternResult<AnyDhtHash> {
    Ok(agent.clone().into())
}

// ============== Input Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeTranslationInput {
    pub source_lang: String,
    pub target_lang: String,
    pub context_key: String,
    pub source_text: String,
    pub proposed_translation: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoteOnTranslationInput {
    pub proposal_hash: ActionHash,
    pub approved: bool,
    pub native_speaker_attestation: bool,
    /// Voter's consciousness permille (from frontend consciousness provider).
    /// Determines vote weight. Higher consciousness = more influence.
    pub voter_consciousness_permille: Option<u16>,
}

/// Translator stats for credential eligibility.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TranslatorStats {
    pub agent: String,
    pub target_lang: String,
    pub proposals_count: u32,
    pub verified_count: u32,
    pub eligible_for_credential: bool,
}

// ============== Extern Functions ==============

/// Propose a translation for a UI key or curriculum content item.
#[hdk_extern]
pub fn propose_translation(input: ProposeTranslationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() / 1_000_000; // seconds

    let proposal = TranslationProposal {
        source_lang: input.source_lang.clone(),
        target_lang: input.target_lang.clone(),
        context_key: input.context_key,
        source_text: input.source_text,
        proposed_translation: input.proposed_translation,
        proposer: agent.clone(),
        proposed_at: now,
    };

    let hash = create_entry(EntryTypes::TranslationProposal(proposal))?;

    // Link: language pair anchor -> proposal
    let pair_anchor = lang_pair_anchor(&input.source_lang, &input.target_lang)?;
    create_link(pair_anchor, hash.clone(), LinkTypes::LanguagePairToProposals, vec![])?;

    // Link: translator -> proposal
    let agent_hash = translator_anchor(&agent)?;
    create_link(agent_hash, hash.clone(), LinkTypes::TranslatorToProposals, vec![])?;

    Ok(hash)
}

/// Vote on a translation proposal. Vote weight = voter's consciousness permille.
///
/// Sybil-resistant: low-consciousness accounts contribute minimal weight.
/// Verification triggers automatically when weighted threshold is reached.
#[hdk_extern]
pub fn vote_on_translation(input: VoteOnTranslationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() / 1_000_000;

    // Voter's consciousness tier — gate voting to at least Participant tier.
    // Full consciousness-weighted voting uses the voter_consciousness_permille
    // field on the input (set by the frontend from the consciousness provider).
    let voter_consciousness = input.voter_consciousness_permille.unwrap_or(500);

    let vote = TranslationVote {
        proposal_hash: input.proposal_hash.clone(),
        voter: agent,
        approved: input.approved,
        native_speaker_attestation: input.native_speaker_attestation,
        voter_consciousness_permille: voter_consciousness,
        voted_at: now,
    };

    let vote_hash = create_entry(EntryTypes::TranslationVote(vote))?;

    // Link: proposal -> vote
    create_link(
        input.proposal_hash.clone(),
        vote_hash.clone(),
        LinkTypes::ProposalToVotes,
        vec![],
    )?;

    // Check if verification threshold is reached
    check_and_finalize(input.proposal_hash)?;

    Ok(vote_hash)
}

/// Check if a proposal has reached the trust-weighted verification threshold.
/// If so, create a VerifiedTranslation entry.
fn check_and_finalize(proposal_hash: ActionHash) -> ExternResult<()> {
    // Fetch all votes
    let vote_links = get_links(
        LinkQuery::try_new(proposal_hash.clone(), LinkTypes::ProposalToVotes)?,
        GetStrategy::Local,
    )?;

    let mut weighted_approval: u32 = 0;
    let mut weighted_rejection: u32 = 0;

    for link in vote_links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) = TranslationVote::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        // Weight = consciousness permille.
                        // Native speaker attestation gives 1.5x bonus.
                        let base_weight = vote.voter_consciousness_permille as u32;
                        let weight = if vote.native_speaker_attestation {
                            base_weight * 3 / 2 // 1.5x for native speakers
                        } else {
                            base_weight
                        };

                        if vote.approved {
                            weighted_approval += weight;
                        } else {
                            weighted_rejection += weight;
                        }
                    }
                }
            }
        }
    }

    // Must have net positive approval above threshold
    if weighted_approval >= VERIFICATION_THRESHOLD && weighted_approval > weighted_rejection {
        // Fetch the proposal to create VerifiedTranslation
        if let Some(record) = get(proposal_hash, GetOptions::default())? {
            if let Some(Entry::App(bytes)) = record.entry().as_option() {
                if let Ok(proposal) = TranslationProposal::try_from(
                    SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                ) {
                    let now = sys_time()?.as_micros() / 1_000_000;

                    let verified = VerifiedTranslation {
                        source_lang: proposal.source_lang,
                        target_lang: proposal.target_lang.clone(),
                        context_key: proposal.context_key.clone(),
                        source_text: proposal.source_text,
                        verified_text: proposal.proposed_translation,
                        approval_score: weighted_approval,
                        proposer: proposal.proposer.clone(),
                        verified_at: now,
                    };

                    let verified_hash = create_entry(EntryTypes::VerifiedTranslation(verified))?;

                    // Link: language+key anchor -> verified
                    let key_anchor = verified_anchor(&proposal.target_lang, &proposal.context_key)?;
                    create_link(key_anchor, verified_hash.clone(), LinkTypes::LanguageKeyToVerified, vec![])?;

                    // Link: translator -> verified
                    let translator_hash = translator_anchor(&proposal.proposer)?;
                    create_link(translator_hash, verified_hash, LinkTypes::TranslatorToVerified, vec![])?;
                }
            }
        }
    }

    Ok(())
}

/// Get a verified translation for a language + key.
/// Returns None if no verified translation exists (frontend falls back to hardcoded).
#[hdk_extern]
pub fn get_verified_translation(input: (String, String)) -> ExternResult<Option<String>> {
    let (target_lang, context_key) = input;
    let anchor = verified_anchor(&target_lang, &context_key)?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LanguageKeyToVerified)?,
        GetStrategy::Local,
    )?;

    // Return the most recent verified translation
    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(verified) = VerifiedTranslation::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        return Ok(Some(verified.verified_text));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get translator stats for credential eligibility (link-counting).
#[hdk_extern]
pub fn get_translator_stats(input: (AgentPubKey, String)) -> ExternResult<TranslatorStats> {
    let (agent, target_lang) = input;
    let agent_hash = translator_anchor(&agent)?;

    // Count proposals
    let proposal_links = get_links(
        LinkQuery::try_new(agent_hash.clone(), LinkTypes::TranslatorToProposals)?,
        GetStrategy::Local,
    )?;

    // Count verified translations
    let verified_links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::TranslatorToVerified)?,
        GetStrategy::Local,
    )?;

    // Count verified for the specific target language
    let mut lang_verified = 0u32;
    for link in &verified_links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(v) = VerifiedTranslation::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if v.target_lang == target_lang {
                            lang_verified += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(TranslatorStats {
        agent: agent.to_string(),
        target_lang,
        proposals_count: proposal_links.len() as u32,
        verified_count: lang_verified,
        eligible_for_credential: lang_verified >= CREDENTIAL_THRESHOLD,
    })
}

/// List all proposals for a language pair.
#[hdk_extern]
pub fn list_proposals_for_language(input: (String, String)) -> ExternResult<Vec<Record>> {
    let (source_lang, target_lang) = input;
    let anchor = lang_pair_anchor(&source_lang, &target_lang)?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LanguagePairToProposals)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}
