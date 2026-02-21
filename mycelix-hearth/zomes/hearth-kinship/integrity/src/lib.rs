//! Hearth Kinship Integrity Zome
//!
//! Defines entry types and validation for hearth membership and kinship bonds.
//! This is the CORE membership and relationship zome for the Hearth cluster.

use hdi::prelude::*;
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
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Hearth(hearth) => validate_hearth(&hearth),
            EntryTypes::HearthMembership(membership) => validate_membership(&membership),
            EntryTypes::KinshipBond(bond) => validate_bond(&bond),
            EntryTypes::HearthInvitation(invitation) => validate_invitation(&invitation),
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
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

pub fn validate_invitation(invitation: &HearthInvitation) -> ExternResult<ValidateCallbackResult> {
    if invitation.message.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invitation message must be <= 2048 characters".into(),
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
}
