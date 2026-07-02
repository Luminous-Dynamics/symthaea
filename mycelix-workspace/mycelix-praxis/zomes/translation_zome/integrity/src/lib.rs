#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Translation Integrity Zome
//!
//! Community-driven multilingual content translation with trust-weighted
//! verification and automatic credential issuance.
//!
//! ## Language-Agnostic Design
//!
//! Uses raw ISO 639-1 codes (e.g., "zu", "hi", "pt"), NOT a hardcoded enum.
//! Any language pair is supported without code changes.
//!
//! ## Sybil Resistance
//!
//! Verification votes are weighted by the voter's consciousness tier
//! (per DHT guardrail). A Guardian-tier vote counts more than an Observer.
//! Weighted threshold prevents low-trust Sybils from gaming translations.

use hdi::prelude::*;

/// A proposed translation of a UI key or curriculum content item.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TranslationProposal {
    /// ISO 639-1 source language code (e.g., "en")
    pub source_lang: String,
    /// ISO 639-1 target language code (e.g., "zu", "hi", "pt")
    pub target_lang: String,
    /// The key being translated (i18n key or curriculum node ID)
    pub context_key: String,
    /// Original text in source language
    pub source_text: String,
    /// Proposed translation in target language
    pub proposed_translation: String,
    /// Agent who proposed
    pub proposer: AgentPubKey,
    /// When proposed
    pub proposed_at: i64,
}

/// A verification vote on a translation proposal.
///
/// Weighted by voter's consciousness tier (NOT a simple count).
/// Prevents Sybil attacks: Guardian-tier votes count more than Observer.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TranslationVote {
    /// Hash of the proposal being voted on
    pub proposal_hash: ActionHash,
    /// Voter agent
    pub voter: AgentPubKey,
    /// Approved or rejected
    pub approved: bool,
    /// Self-attested native speaker status
    pub native_speaker_attestation: bool,
    /// Voter's consciousness tier at time of vote (0-1000 permille)
    /// Recorded at vote time for auditability; NOT recomputed during validation
    pub voter_consciousness_permille: u16,
    /// When voted
    pub voted_at: i64,
}

/// A verified translation — community-approved, immutable.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiedTranslation {
    /// ISO 639-1 source language
    pub source_lang: String,
    /// ISO 639-1 target language
    pub target_lang: String,
    /// Key being translated
    pub context_key: String,
    /// Original text
    pub source_text: String,
    /// Verified translation text
    pub verified_text: String,
    /// Weighted approval score at time of verification
    pub approval_score: u32,
    /// Proposer agent
    pub proposer: AgentPubKey,
    /// When verified
    pub verified_at: i64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    TranslationProposal(TranslationProposal),
    #[entry_type(visibility = "public")]
    TranslationVote(TranslationVote),
    #[entry_type(visibility = "public")]
    VerifiedTranslation(VerifiedTranslation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Language pair anchor -> proposals (e.g., "translations.en.zu")
    LanguagePairToProposals,
    /// Proposal -> votes
    ProposalToVotes,
    /// Language + key anchor -> verified translation (e.g., "verified.zu.greeting_morning")
    LanguageKeyToVerified,
    /// Translator agent -> their proposals
    TranslatorToProposals,
    /// Translator agent -> their verified translations
    TranslatorToVerified,
}

pub fn validate_proposal(proposal: &TranslationProposal) -> ExternResult<ValidateCallbackResult> {
    if proposal.source_lang.len() < 2 || proposal.source_lang.len() > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Source language must be a valid ISO 639 code (2-5 chars)".into(),
        ));
    }
    if proposal.target_lang.len() < 2 || proposal.target_lang.len() > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target language must be a valid ISO 639 code (2-5 chars)".into(),
        ));
    }
    if proposal.source_lang == proposal.target_lang {
        return Ok(ValidateCallbackResult::Invalid(
            "Source and target language must be different".into(),
        ));
    }
    if proposal.source_text.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Source text cannot be empty".into()));
    }
    if proposal.proposed_translation.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Proposed translation cannot be empty".into()));
    }
    if proposal.context_key.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Context key cannot be empty".into()));
    }
    if proposal.source_text.len() > 10_000 || proposal.proposed_translation.len() > 10_000 {
        return Ok(ValidateCallbackResult::Invalid("Text cannot exceed 10000 characters".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::TranslationProposal(p) => validate_proposal(&p),
                    EntryTypes::TranslationVote(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::VerifiedTranslation(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { original_action, action, .. } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. } | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. } | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent(fill: u8) -> AgentPubKey {
        let mut bytes = vec![0x84, 0x20, 0x24];
        bytes.extend(std::iter::repeat(fill).take(36));
        AgentPubKey::from_raw_39(bytes)
    }

    fn valid_proposal() -> TranslationProposal {
        TranslationProposal {
            source_lang: "en".to_string(),
            target_lang: "zu".to_string(),
            context_key: "greeting_morning".to_string(),
            source_text: "Good morning".to_string(),
            proposed_translation: "Sawubona ekuseni".to_string(),
            proposer: test_agent(0),
            proposed_at: 1000000,
        }
    }

    #[test]
    fn test_valid_proposal() {
        assert_eq!(validate_proposal(&valid_proposal()).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_same_language_rejected() {
        let mut p = valid_proposal();
        p.target_lang = "en".to_string();
        assert!(matches!(validate_proposal(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_source_rejected() {
        let mut p = valid_proposal();
        p.source_text = "".to_string();
        assert!(matches!(validate_proposal(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_translation_rejected() {
        let mut p = valid_proposal();
        p.proposed_translation = "".to_string();
        assert!(matches!(validate_proposal(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_lang_code() {
        let mut p = valid_proposal();
        p.source_lang = "x".to_string(); // too short
        assert!(matches!(validate_proposal(&p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_any_language_pair_works() {
        // Hindi to Portuguese — proves language-agnostic design
        let mut p = valid_proposal();
        p.source_lang = "hi".to_string();
        p.target_lang = "pt".to_string();
        p.source_text = "Namaste".to_string();
        p.proposed_translation = "Olá".to_string();
        assert_eq!(validate_proposal(&p).unwrap(), ValidateCallbackResult::Valid);
    }
}
