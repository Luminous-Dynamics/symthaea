// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Craft Graph Integrity Zome
//!
//! Defines entry types and validation rules for the Craft/Workforce graph.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CraftProfile {
    pub agent_did: String,
    pub display_name: String,
    pub headline: String,
    pub bio: String,
    pub location: String,
    pub website: String,
    pub avatar_url: String,
    pub primary_skill: String,
    pub mastery_level: u16,
    pub endorsements_count: u32,
    pub updated_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PublishedCredential {
    pub credential_hash: ActionHash,
    pub issuer_did: String,
    pub issuer: String,
    pub visibility: String,
    pub title: String,
    pub summary: Option<String>,
    pub mastery_level_at_issue: Option<u16>,
    pub last_retention_check: Option<String>,
    pub issued_on: String,
    pub expires_on: Option<String>,
    pub source_dna: String,
    pub entry_hash: String,
    pub action_hash: String,
    pub vitality_permille: Option<u16>,
    pub guild_id: Option<String>,
    pub guild_name: Option<String>,
    pub epistemic_code: Option<String>,
    pub fl_model_version: Option<String>,
    pub mastery_permille: Option<u16>,
    pub verified: Option<bool>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SkillEndorsement {
    pub subject_did: String,
    pub endorsed_agent: AgentPubKey,
    pub skill: String,
    pub weight: u16,
    pub rationale: String,
    pub evidence: String,
    pub timestamp: i64,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RetentionCheck {
    pub agent: AgentPubKey,
    pub skill: String,
    pub credential_id: String,
    pub retention_score_permille: u16,
    pub questions_attempted: u16,
    pub questions_correct: u16,
    pub timestamp: i64,
    pub checked_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompositeProfile {
    pub identity_hash: ActionHash,
    pub workforce_hash: ActionHash,
    pub agent: AgentPubKey,
    pub archetype_name: String,
    pub credential_titles: Vec<String>,
    pub coverage_permille: u16,
    pub career_profile_match: Option<String>,
    pub detected_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RetentionProof {
    pub check_hash: ActionHash,
    pub proof_data: Vec<u8>,
    pub credential_id: String,
    pub threshold_permille: u16,
    pub proof_bytes: Vec<u8>,
    pub score_commitment: Vec<u8>,
    pub domain_tag: String,
    pub proven_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    CraftProfile(CraftProfile),
    #[entry_type(visibility = "public")]
    PublishedCredential(PublishedCredential),
    #[entry_type(visibility = "public")]
    SkillEndorsement(SkillEndorsement),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    RetentionCheck(RetentionCheck),
    #[entry_type(visibility = "public")]
    CompositeProfile(CompositeProfile),
    #[entry_type(visibility = "public")]
    RetentionProof(RetentionProof),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToProfile,
    AgentToCredential,
    SkillToCredential,
    SkillEndorsement,
    ProfileToCredential,
    GuildToCredential,
    AgentToEndorsement,
    EndorsedAgentToEndorsement,
    CredentialToRetentionCheck,
    AgentToCompositeProfile,
    CredentialToRetentionProof,
}

// ============== Validation Functions ==============

pub fn validate_create_profile(profile: &CraftProfile) -> ExternResult<ValidateCallbackResult> {
    if profile.mastery_level > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mastery level must be in 0..1000".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_create_skill_endorsement(
    endorsement: &SkillEndorsement,
) -> ExternResult<ValidateCallbackResult> {
    if endorsement.weight > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Weight must be in 0..100".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(entry) => match entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::CraftProfile(profile) => validate_create_profile(&profile),
                EntryTypes::SkillEndorsement(endorsement) => {
                    validate_create_skill_endorsement(&endorsement)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
