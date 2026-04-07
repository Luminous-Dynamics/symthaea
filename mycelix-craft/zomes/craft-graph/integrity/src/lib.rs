#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CraftProfile {
    pub display_name: String,
    pub headline: String,
    pub bio: String,
    pub location: Option<String>,
    pub website: Option<String>,
    pub avatar_url: Option<String>,
    pub updated_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PublishedCredential {
    pub credential_id: String,
    pub title: String,
    pub issuer: String,
    pub issued_on: String,
    pub expires_on: Option<String>,
    pub source_dna: Option<String>,
    pub entry_hash: Option<String>,
    pub action_hash: Option<String>,
    pub summary: Option<String>,
    /// Living Credential: initial vitality when published (0-1000 permille)
    #[serde(default)]
    pub vitality_permille: Option<u16>,
    /// ISO 8601 timestamp of last retention check (or issue date if never checked)
    #[serde(default)]
    pub last_retention_check: Option<String>,
    /// Mastery level at time of credential issuance (for stability calculation)
    #[serde(default)]
    pub mastery_level_at_issue: Option<u16>,
    /// Guild that issued or endorsed this credential
    #[serde(default)]
    pub guild_id: Option<String>,
    /// Human-readable guild name
    #[serde(default)]
    pub guild_name: Option<String>,
    /// Epistemic classification code from Proof of Learning (e.g., "E3-N1-M2")
    #[serde(default)]
    pub epistemic_code: Option<String>,
    /// Federated Learning model version used for assessment
    #[serde(default)]
    pub fl_model_version: Option<String>,
    /// Canonical mastery level (0-1000 permille, from BKT)
    #[serde(default)]
    pub mastery_permille: Option<u16>,
    /// Cross-DNA verification status (true = verified against source DNA)
    #[serde(default)]
    pub verified: Option<bool>,
}

/// A retention check proving the holder still knows the material.
/// Linked from the credential via `CredentialToRetentionCheck`.
/// Integrity validates existence only — time-decay computation is in coordinator.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RetentionCheck {
    /// Hash of the PublishedCredential being refreshed (as string for serialization)
    pub credential_id: String,
    /// Retention score from the challenge (0-1000 permille)
    pub retention_score_permille: u16,
    /// Number of questions attempted
    pub questions_attempted: u16,
    /// Number correct
    pub questions_correct: u16,
    /// When the check was performed
    pub checked_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SkillEndorsement {
    pub endorsed_agent: AgentPubKey,
    pub skill: String,
    pub rationale: String,
    pub evidence: Option<String>,
    pub created_at: Timestamp,
}

/// A composite credential profile — automatically detected when an agent's
/// credentials collectively cover a career archetype.
///
/// Example: "Rust + Holochain + Teaching" = "Full Stack Consciousness Developer"
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompositeProfile {
    /// Agent this composite belongs to
    pub agent: AgentPubKey,
    /// Archetype name (e.g., "Full Stack Consciousness Developer")
    pub archetype_name: String,
    /// Credential titles that compose this profile
    pub credential_titles: Vec<String>,
    /// Coverage: what percentage of the archetype's required skills are covered
    pub coverage_permille: u16,
    /// Career profile field this maps to (if any)
    pub career_profile_match: Option<String>,
    /// When detected
    pub detected_at: Timestamp,
}

/// A zero-knowledge proof that a retention check score meets a threshold.
///
/// Generated client-side using Winterfell STARK range proof circuit.
/// Proves "my retention_score >= threshold" without revealing the exact score.
/// Stored on DHT for any verifier to check independently.
///
/// Domain tag: ZTML:Craft:RetentionScore:v1
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RetentionProof {
    /// Hash of the credential this proof is for
    pub credential_id: String,
    /// Minimum score threshold proven (0-1000 permille)
    pub threshold_permille: u16,
    /// Winterfell STARK proof bytes (~200KB)
    pub proof_bytes: Vec<u8>,
    /// SHA-256 commitment to the actual score (for binding)
    pub score_commitment: Vec<u8>,
    /// Domain separation tag
    pub domain_tag: String,
    /// When the proof was generated
    pub proven_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
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
    AgentToEndorsement,
    EndorsedAgentToEndorsement,
    SkillToEndorsement,
    ProfileToCredential,
    /// Credential -> retention check results (existence proves ongoing competence)
    CredentialToRetentionCheck,
    /// Agent -> composite profiles detected
    AgentToCompositeProfile,
    /// Guild -> credentials issued/endorsed under that guild
    GuildToCredential,
    /// Guild -> endorsements from guild members
    GuildToEndorsement,
    /// Credential -> zero-knowledge retention proofs
    CredentialToRetentionProof,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        },
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        },
    }
}
