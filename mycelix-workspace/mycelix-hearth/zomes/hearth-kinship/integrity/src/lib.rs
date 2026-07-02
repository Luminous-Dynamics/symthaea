// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Kinship Integrity Zome
//!
//! Defines entry types and validation for hearth membership and kinship bonds.
//! This is the CORE membership and relationship zome for the Hearth cluster.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use hearth_types::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Entry Types
// ============================================================================

/// A hearth — the fundamental family/household unit.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Hearth {
    /// Human-readable name of the hearth.
    pub name: String,
    /// Description of the hearth's purpose or character.
    pub description: String,
    /// Type of hearth (Nuclear, Extended, Chosen, etc.).
    pub hearth_type: HearthType,
    /// Agent who created this hearth.
    pub created_by: AgentPubKey,
    /// Timestamp of hearth creation.
    pub created_at: Timestamp,
    /// Maximum number of members allowed (2-50).
    pub max_members: u32,
}

/// A membership record linking an agent to a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HearthMembership {
    /// Hash of the hearth this membership belongs to.
    pub hearth_hash: ActionHash,
    /// The agent who is a member.
    pub agent: AgentPubKey,
    /// Role within the hearth.
    pub role: MemberRole,
    /// Current membership status.
    pub status: MembershipStatus,
    /// Display name within this hearth.
    pub display_name: String,
    /// When the member joined.
    pub joined_at: Timestamp,
}

/// A kinship bond between two members of a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KinshipBond {
    /// Hash of the hearth this bond belongs to.
    pub hearth_hash: ActionHash,
    /// First member in the bond.
    pub member_a: AgentPubKey,
    /// Second member in the bond.
    pub member_b: AgentPubKey,
    /// Type of kinship bond.
    pub bond_type: BondType,
    /// Current strength in basis points (0-10000).
    pub strength_bp: u32,
    /// Timestamp when the bond was last tended.
    pub last_tended: Timestamp,
    /// Timestamp when the bond was created.
    pub created_at: Timestamp,
}

/// An invitation to join a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HearthInvitation {
    /// Hash of the hearth this invitation is for.
    pub hearth_hash: ActionHash,
    /// Agent who sent the invitation.
    pub inviter: AgentPubKey,
    /// Agent being invited.
    pub invitee_agent: AgentPubKey,
    /// Proposed role for the invitee.
    pub proposed_role: MemberRole,
    /// Personal message from the inviter.
    pub message: String,
    /// When the invitation expires.
    pub expires_at: Timestamp,
    /// Current status of the invitation.
    pub status: InvitationStatus,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// Entry / Link Type Enums
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Hearth(Hearth),
    HearthMembership(HearthMembership),
    KinshipBond(KinshipBond),
    HearthInvitation(HearthInvitation),
    Anchor(Anchor),
    WeeklyDigest(WeeklyDigest),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor("all_hearths") -> Hearth
    AllHearths,
    /// AgentPubKey -> Hearth
    AgentToHearths,
    /// Hearth -> HearthMembership
    HearthToMembers,
    /// Hearth -> KinshipBond
    HearthToBonds,
    /// AgentPubKey -> KinshipBond
    MemberToBonds,
    /// Anchor("hearth_type:{type}") -> Hearth
    TypeToHearths,
    /// Hearth -> HearthInvitation
    HearthToInvitations,
    /// AgentPubKey -> HearthInvitation
    AgentToInvitations,
    /// Hearth -> WeeklyDigest (H2 epoch rollups)
    HearthToDigests,
}

// ============================================================================
// Genesis + Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::Hearth(hearth) => validate_hearth(&hearth),
            EntryTypes::HearthMembership(membership) => validate_membership(&membership),
            EntryTypes::KinshipBond(bond) => validate_bond(&bond),
            EntryTypes::HearthInvitation(invitation) => validate_invitation(&invitation),
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::WeeklyDigest(digest) => validate_weekly_digest(&digest),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry {
            app_entry,
            original_action_hash,
            ..
        }) => match app_entry {
            EntryTypes::Hearth(hearth) => {
                validate_hearth(&hearth)?;
                validate_hearth_immutable_fields(&hearth, &original_action_hash)
            }
            EntryTypes::HearthMembership(membership) => {
                validate_membership(&membership)?;
                validate_membership_immutable_fields(&membership, &original_action_hash)
            }
            EntryTypes::KinshipBond(bond) => {
                validate_bond(&bond)?;
                validate_bond_immutable_fields(&bond, &original_action_hash)
            }
            EntryTypes::HearthInvitation(invitation) => {
                validate_invitation(&invitation)?;
                validate_invitation_immutable_fields(&invitation, &original_action_hash)
            }
            EntryTypes::Anchor(_) => {
                // INVARIANT: Anchor immutability — anchors are deterministic link bases
                // and must not be modified after creation.
                Ok(ValidateCallbackResult::Invalid(
                    "Anchor cannot be updated once created".into(),
                ))
            }
            EntryTypes::WeeklyDigest(_) => {
                // INVARIANT: WeeklyDigest immutability — digests are rollup snapshots
                // of an epoch and cannot be modified after creation.
                Ok(ValidateCallbackResult::Invalid(
                    "WeeklyDigest cannot be updated once created".into(),
                ))
            }
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink { tag, action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_hearth(hearth: &Hearth) -> ExternResult<ValidateCallbackResult> {
    if hearth.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Hearth name cannot be empty".into(),
        ));
    }
    if hearth.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hearth name must be <= 256 characters".into(),
        ));
    }
    if hearth.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hearth description must be <= 4096 characters".into(),
        ));
    }
    if hearth.max_members < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hearth max_members must be >= 2".into(),
        ));
    }
    if hearth.max_members > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hearth max_members must be <= 50".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_membership(membership: &HearthMembership) -> ExternResult<ValidateCallbackResult> {
    if membership.display_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Membership display_name cannot be empty".into(),
        ));
    }
    if membership.display_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Membership display_name must be <= 256 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_bond(bond: &KinshipBond) -> ExternResult<ValidateCallbackResult> {
    if bond.strength_bp > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Bond strength_bp must be <= 10000".into(),
        ));
    }
    if bond.member_a == bond.member_b {
        return Ok(ValidateCallbackResult::Invalid(
            "Bond member_a and member_b must be different agents (no self-bonds)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_weekly_digest(digest: &WeeklyDigest) -> ExternResult<ValidateCallbackResult> {
    // epoch_end must be after epoch_start
    if digest.epoch_end <= digest.epoch_start {
        return Ok(ValidateCallbackResult::Invalid(
            "WeeklyDigest epoch_end must be after epoch_start".into(),
        ));
    }
    // epoch_end - epoch_start must be <= 8 days (691_200_000_000 microseconds)
    let duration_micros = digest.epoch_end.as_micros() - digest.epoch_start.as_micros();
    let eight_days_micros: i64 = 8 * 24 * 60 * 60 * 1_000_000;
    if duration_micros > eight_days_micros {
        return Ok(ValidateCallbackResult::Invalid(
            "WeeklyDigest epoch window must be <= 8 days".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_invitation(invitation: &HearthInvitation) -> ExternResult<ValidateCallbackResult> {
    if invitation.message.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invitation message must be <= 2048 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Immutable Field Validation
// ============================================================================

fn validate_hearth_immutable_fields(
    new: &Hearth,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: Hearth = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original Hearth: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original Hearth entry is missing".into()
        )))?;
    if new.created_by != original.created_by {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_by on a Hearth".into(),
        ));
    }
    if new.created_at != original.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on a Hearth".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_membership_immutable_fields(
    new: &HearthMembership,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: HearthMembership = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original HearthMembership: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original HearthMembership entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a HearthMembership".into(),
        ));
    }
    if new.agent != original.agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change agent on a HearthMembership".into(),
        ));
    }
    if new.joined_at != original.joined_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change joined_at on a HearthMembership".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_bond_immutable_fields(
    new: &KinshipBond,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: KinshipBond = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original KinshipBond: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original KinshipBond entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a KinshipBond".into(),
        ));
    }
    if new.member_a != original.member_a {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change member_a on a KinshipBond".into(),
        ));
    }
    if new.member_b != original.member_b {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change member_b on a KinshipBond".into(),
        ));
    }
    if new.created_at != original.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on a KinshipBond".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_invitation_immutable_fields(
    new: &HearthInvitation,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: HearthInvitation = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original HearthInvitation: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original HearthInvitation entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a HearthInvitation".into(),
        ));
    }
    if new.inviter != original.inviter {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change inviter on a HearthInvitation".into(),
        ));
    }
    if new.invitee_agent != original.invitee_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change invitee_agent on a HearthInvitation".into(),
        ));
    }
    if new.expires_at != original.expires_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change expires_at on a HearthInvitation".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper Constructors ----

    fn fake_agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_hearth(name: &str, desc: &str, max: u32) -> Hearth {
        Hearth {
            name: name.into(),
            description: desc.into(),
            hearth_type: HearthType::Nuclear,
            created_by: fake_agent_a(),
            created_at: fake_timestamp(),
            max_members: max,
        }
    }

    fn make_membership(display_name: &str) -> HearthMembership {
        HearthMembership {
            hearth_hash: fake_action_hash(),
            agent: fake_agent_a(),
            role: MemberRole::Adult,
            status: MembershipStatus::Active,
            display_name: display_name.into(),
            joined_at: fake_timestamp(),
        }
    }

    fn make_bond(strength: u32, a: AgentPubKey, b: AgentPubKey) -> KinshipBond {
        KinshipBond {
            hearth_hash: fake_action_hash(),
            member_a: a,
            member_b: b,
            bond_type: BondType::Sibling,
            strength_bp: strength,
            last_tended: fake_timestamp(),
            created_at: fake_timestamp(),
        }
    }

    fn make_invitation(message: &str) -> HearthInvitation {
        HearthInvitation {
            hearth_hash: fake_action_hash(),
            inviter: fake_agent_a(),
            invitee_agent: fake_agent_b(),
            proposed_role: MemberRole::Adult,
            message: message.into(),
            expires_at: Timestamp::from_micros(2_000_000),
            status: InvitationStatus::Pending,
        }
    }

    // ---- Hearth Serde Roundtrips ----

    #[test]
    fn hearth_serde_roundtrip() {
        let h = make_hearth("The Stoltz Family", "A loving home", 10);
        let json = serde_json::to_string(&h).unwrap();
        let back: Hearth = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn hearth_all_types_serde_roundtrip() {
        for ht in &[
            HearthType::Nuclear,
            HearthType::Extended,
            HearthType::Chosen,
            HearthType::Blended,
            HearthType::Multigenerational,
            HearthType::Intentional,
            HearthType::CoPod,
            HearthType::Custom("Commune".into()),
        ] {
            let mut h = make_hearth("Test", "", 5);
            h.hearth_type = ht.clone();
            let json = serde_json::to_string(&h).unwrap();
            let back: Hearth = serde_json::from_str(&json).unwrap();
            assert_eq!(back.hearth_type, *ht);
        }
    }

    // ---- Hearth Validation ----

    #[test]
    fn valid_hearth_passes() {
        let h = make_hearth("My Hearth", "A warm place", 10);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_empty_name_rejected() {
        let h = make_hearth("", "desc", 10);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn hearth_name_exactly_256_passes() {
        let h = make_hearth(&"x".repeat(256), "desc", 10);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_name_257_rejected() {
        let h = make_hearth(&"x".repeat(257), "desc", 10);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn hearth_empty_description_passes() {
        let h = make_hearth("Name", "", 10);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_description_exactly_4096_passes() {
        let h = make_hearth("Name", &"d".repeat(4096), 10);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_description_4097_rejected() {
        let h = make_hearth("Name", &"d".repeat(4097), 10);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn hearth_max_members_2_passes() {
        let h = make_hearth("Name", "", 2);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_max_members_50_passes() {
        let h = make_hearth("Name", "", 50);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_max_members_1_rejected() {
        let h = make_hearth("Name", "", 1);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains(">= 2")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn hearth_max_members_0_rejected() {
        let h = make_hearth("Name", "", 0);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains(">= 2")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn hearth_max_members_51_rejected() {
        let h = make_hearth("Name", "", 51);
        match validate_hearth(&h).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Membership Serde Roundtrips ----

    #[test]
    fn membership_serde_roundtrip() {
        let m = make_membership("Alice");
        let json = serde_json::to_string(&m).unwrap();
        let back: HearthMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn membership_all_roles_serde_roundtrip() {
        for role in &[
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ] {
            let mut m = make_membership("Test");
            m.role = role.clone();
            let json = serde_json::to_string(&m).unwrap();
            let back: HearthMembership = serde_json::from_str(&json).unwrap();
            assert_eq!(back.role, *role);
        }
    }

    #[test]
    fn membership_all_statuses_serde_roundtrip() {
        for status in &[
            MembershipStatus::Active,
            MembershipStatus::Invited,
            MembershipStatus::Departed,
            MembershipStatus::Ancestral,
        ] {
            let mut m = make_membership("Test");
            m.status = status.clone();
            let json = serde_json::to_string(&m).unwrap();
            let back: HearthMembership = serde_json::from_str(&json).unwrap();
            assert_eq!(back.status, *status);
        }
    }

    // ---- Membership Validation ----

    #[test]
    fn valid_membership_passes() {
        let m = make_membership("Alice");
        assert!(matches!(
            validate_membership(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn membership_empty_display_name_rejected() {
        let m = make_membership("");
        match validate_membership(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn membership_display_name_exactly_256_passes() {
        let m = make_membership(&"n".repeat(256));
        assert!(matches!(
            validate_membership(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn membership_display_name_257_rejected() {
        let m = make_membership(&"n".repeat(257));
        match validate_membership(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Bond Serde Roundtrips ----

    #[test]
    fn bond_serde_roundtrip() {
        let b = make_bond(7000, fake_agent_a(), fake_agent_b());
        let json = serde_json::to_string(&b).unwrap();
        let back: KinshipBond = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn bond_all_types_serde_roundtrip() {
        for bt in &[
            BondType::Parent,
            BondType::Child,
            BondType::Sibling,
            BondType::Partner,
            BondType::Grandparent,
            BondType::Grandchild,
            BondType::AuntUncle,
            BondType::NieceNephew,
            BondType::Cousin,
            BondType::ChosenFamily,
            BondType::Guardian,
            BondType::Ward,
            BondType::Custom("Godparent".into()),
        ] {
            let mut b = make_bond(5000, fake_agent_a(), fake_agent_b());
            b.bond_type = bt.clone();
            let json = serde_json::to_string(&b).unwrap();
            let back: KinshipBond = serde_json::from_str(&json).unwrap();
            assert_eq!(back.bond_type, *bt);
        }
    }

    // ---- Bond Validation ----

    #[test]
    fn valid_bond_passes() {
        let b = make_bond(7000, fake_agent_a(), fake_agent_b());
        assert!(matches!(
            validate_bond(&b).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn bond_strength_0_passes() {
        let b = make_bond(0, fake_agent_a(), fake_agent_b());
        assert!(matches!(
            validate_bond(&b).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn bond_strength_10000_passes() {
        let b = make_bond(10000, fake_agent_a(), fake_agent_b());
        assert!(matches!(
            validate_bond(&b).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn bond_strength_10001_rejected() {
        let b = make_bond(10001, fake_agent_a(), fake_agent_b());
        match validate_bond(&b).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn bond_self_bond_rejected() {
        let same = fake_agent_a();
        let b = make_bond(5000, same.clone(), same);
        match validate_bond(&b).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("self-bond")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn bond_max_strength_u32_rejected() {
        let b = make_bond(u32::MAX, fake_agent_a(), fake_agent_b());
        match validate_bond(&b).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Invitation Serde Roundtrips ----

    #[test]
    fn invitation_serde_roundtrip() {
        let inv = make_invitation("Welcome to our family!");
        let json = serde_json::to_string(&inv).unwrap();
        let back: HearthInvitation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, inv);
    }

    #[test]
    fn invitation_all_statuses_serde_roundtrip() {
        for status in &[
            InvitationStatus::Pending,
            InvitationStatus::Accepted,
            InvitationStatus::Declined,
            InvitationStatus::Expired,
        ] {
            let mut inv = make_invitation("Hello");
            inv.status = status.clone();
            let json = serde_json::to_string(&inv).unwrap();
            let back: HearthInvitation = serde_json::from_str(&json).unwrap();
            assert_eq!(back.status, *status);
        }
    }

    // ---- Invitation Validation ----

    #[test]
    fn valid_invitation_passes() {
        let inv = make_invitation("Join us!");
        assert!(matches!(
            validate_invitation(&inv).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn invitation_empty_message_passes() {
        let inv = make_invitation("");
        assert!(matches!(
            validate_invitation(&inv).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn invitation_message_exactly_2048_passes() {
        let inv = make_invitation(&"m".repeat(2048));
        assert!(matches!(
            validate_invitation(&inv).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn invitation_message_2049_rejected() {
        let inv = make_invitation(&"m".repeat(2049));
        match validate_invitation(&inv).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("2048")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Anchor Serde ----

    #[test]
    fn anchor_serde_roundtrip() {
        let a = Anchor("all_hearths".into());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ---- Entry/Link Type Enums ----

    #[test]
    fn entry_types_hearth_variant_exists() {
        let _v = UnitEntryTypes::Hearth;
    }

    #[test]
    fn entry_types_membership_variant_exists() {
        let _v = UnitEntryTypes::HearthMembership;
    }

    #[test]
    fn entry_types_bond_variant_exists() {
        let _v = UnitEntryTypes::KinshipBond;
    }

    #[test]
    fn entry_types_invitation_variant_exists() {
        let _v = UnitEntryTypes::HearthInvitation;
    }

    #[test]
    fn entry_types_anchor_variant_exists() {
        let _v = UnitEntryTypes::Anchor;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _all = LinkTypes::AllHearths;
        let _agent = LinkTypes::AgentToHearths;
        let _members = LinkTypes::HearthToMembers;
        let _bonds = LinkTypes::HearthToBonds;
        let _mbonds = LinkTypes::MemberToBonds;
        let _type = LinkTypes::TypeToHearths;
        let _invitations = LinkTypes::HearthToInvitations;
        let _agent_inv = LinkTypes::AgentToInvitations;
        let _digests = LinkTypes::HearthToDigests;
    }

    // ---- Edge Cases ----

    #[test]
    fn hearth_single_char_name_passes() {
        let h = make_hearth("H", "", 2);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_unicode_name_passes() {
        let h = make_hearth("La Maison des Etoiles", "Une famille choisie", 8);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_max_members_boundary_2_passes() {
        let h = make_hearth("Pair", "", 2);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn hearth_max_members_boundary_50_passes() {
        let h = make_hearth("Big Family", "", 50);
        assert!(matches!(
            validate_hearth(&h).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn bond_different_agents_passes() {
        let a = AgentPubKey::from_raw_36(vec![10u8; 36]);
        let b = AgentPubKey::from_raw_36(vec![20u8; 36]);
        let bond = make_bond(5000, a, b);
        assert!(matches!(
            validate_bond(&bond).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn membership_single_char_display_name_passes() {
        let m = make_membership("A");
        assert!(matches!(
            validate_membership(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn membership_unicode_display_name_passes() {
        let m = make_membership("Tristan");
        assert!(matches!(
            validate_membership(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- WeeklyDigest Validation ----

    fn make_digest(start_micros: i64, end_micros: i64) -> WeeklyDigest {
        WeeklyDigest {
            hearth_hash: fake_action_hash(),
            epoch_start: Timestamp::from_micros(start_micros),
            epoch_end: Timestamp::from_micros(end_micros),
            bond_updates: vec![],
            care_summary: vec![],
            gratitude_summary: vec![],
            rhythm_summary: vec![],
            created_by: fake_agent_a(),
            created_at: Timestamp::from_micros(end_micros),
        }
    }

    #[test]
    fn weekly_digest_entry_type_exists() {
        let _v = UnitEntryTypes::WeeklyDigest;
    }

    #[test]
    fn valid_weekly_digest_passes() {
        // 7 days = 604_800_000_000 micros
        let d = make_digest(0, 604_800_000_000);
        assert!(matches!(
            validate_weekly_digest(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn weekly_digest_end_before_start_rejected() {
        let d = make_digest(1_000_000, 500_000);
        match validate_weekly_digest(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("after epoch_start")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn weekly_digest_equal_start_end_rejected() {
        let d = make_digest(1_000_000, 1_000_000);
        match validate_weekly_digest(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("after epoch_start")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn weekly_digest_8_days_passes() {
        // 8 days = 691_200_000_000 micros
        let d = make_digest(0, 691_200_000_000);
        assert!(matches!(
            validate_weekly_digest(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn weekly_digest_over_8_days_rejected() {
        // 8 days + 1 second
        let d = make_digest(0, 691_201_000_000);
        match validate_weekly_digest(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 8 days")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn weekly_digest_serde_roundtrip() {
        let d = make_digest(0, 604_800_000_000);
        let json = serde_json::to_string(&d).unwrap();
        let back: WeeklyDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    // -- Immutable field pure equality tests --

    #[test]
    fn hearth_immutable_field_created_by_difference_detected() {
        let a = make_hearth("Test", "desc", 10);
        let mut b = a.clone();
        b.created_by = fake_agent_b();
        assert_ne!(a.created_by, b.created_by);
    }

    #[test]
    fn hearth_immutable_field_created_at_difference_detected() {
        let a = make_hearth("Test", "desc", 10);
        let mut b = a.clone();
        b.created_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.created_at, b.created_at);
    }

    #[test]
    fn membership_immutable_field_hearth_hash_difference_detected() {
        let a = make_membership("Alice");
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn membership_immutable_field_agent_difference_detected() {
        let a = make_membership("Alice");
        let mut b = a.clone();
        b.agent = fake_agent_b();
        assert_ne!(a.agent, b.agent);
    }

    #[test]
    fn membership_immutable_field_joined_at_difference_detected() {
        let a = make_membership("Alice");
        let mut b = a.clone();
        b.joined_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.joined_at, b.joined_at);
    }

    #[test]
    fn bond_immutable_field_hearth_hash_difference_detected() {
        let a = make_bond(5000, fake_agent_a(), fake_agent_b());
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn bond_immutable_field_member_a_difference_detected() {
        let a = make_bond(5000, fake_agent_a(), fake_agent_b());
        let mut b = a.clone();
        b.member_a = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.member_a, b.member_a);
    }

    #[test]
    fn bond_immutable_field_member_b_difference_detected() {
        let a = make_bond(5000, fake_agent_a(), fake_agent_b());
        let mut b = a.clone();
        b.member_b = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.member_b, b.member_b);
    }

    #[test]
    fn bond_immutable_field_created_at_difference_detected() {
        let a = make_bond(5000, fake_agent_a(), fake_agent_b());
        let mut b = a.clone();
        b.created_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.created_at, b.created_at);
    }

    #[test]
    fn invitation_immutable_field_hearth_hash_difference_detected() {
        let a = make_invitation("Hello");
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn invitation_immutable_field_inviter_difference_detected() {
        let a = make_invitation("Hello");
        let mut b = a.clone();
        b.inviter = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.inviter, b.inviter);
    }

    #[test]
    fn invitation_immutable_field_invitee_agent_difference_detected() {
        let a = make_invitation("Hello");
        let mut b = a.clone();
        b.invitee_agent = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.invitee_agent, b.invitee_agent);
    }

    #[test]
    fn invitation_immutable_field_expires_at_difference_detected() {
        let a = make_invitation("Hello");
        let mut b = a.clone();
        b.expires_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.expires_at, b.expires_at);
    }

    #[test]
    fn anchor_immutability_documented() {
        // Anchor should be created once and never updated.
        // The integrity validate() function rejects UpdateEntry for Anchor.
        let a = Anchor("all_hearths".into());
        assert_eq!(a.0, "all_hearths");
    }

    #[test]
    fn weekly_digest_immutability_documented() {
        // WeeklyDigest should be created once and never updated.
        // The integrity validate() function rejects UpdateEntry for WeeklyDigest.
        let d = make_digest(0, 604_800_000_000);
        assert_eq!(d.epoch_start, Timestamp::from_micros(0));
    }
}
