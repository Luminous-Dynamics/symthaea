// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Membership Integrity Zome
//! Entry types and validation for co-op members, applications, waitlists, and rent-to-own.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Unit type preference (local copy to avoid cross-integrity linking)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UnitType {
    Studio,
    OneBedroom,
    TwoBedroom,
    ThreeBedroom,
    FourPlus,
    Accessible,
    Family,
}

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of membership in the cooperative
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MembershipType {
    FullShare,
    LimitedEquity,
    RentToOwn,
    Renter,
    Associate,
}

/// Current status of a member
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberStatus {
    Active,
    OnLeave,
    Suspended,
    Former,
}

/// A cooperative member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Member {
    pub agent: AgentPubKey,
    pub unit_hash: Option<ActionHash>,
    pub membership_type: MembershipType,
    pub share_equity_cents: u64,
    pub joined_at: Timestamp,
    pub monthly_charge_cents: u64,
    pub voting_rights: bool,
    pub status: MemberStatus,
}

/// Application status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ApplicationStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    Waitlisted,
}

/// A membership application
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MemberApplication {
    pub applicant: AgentPubKey,
    pub requested_unit: Option<ActionHash>,
    pub membership_type_requested: MembershipType,
    pub applied_at: Timestamp,
    pub household_size: u8,
    pub income_verified: bool,
    pub references: Vec<String>,
    pub status: ApplicationStatus,
}

/// A waitlist entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaitListEntry {
    pub application_hash: ActionHash,
    pub position: u32,
    pub unit_type_preference: Option<UnitType>,
    pub added_at: Timestamp,
}

/// Status of a rent-to-own agreement
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgreementStatus {
    Active,
    Completed,
    Defaulted,
    Terminated,
}

/// A rent-to-own agreement
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RentToOwnAgreement {
    pub member: AgentPubKey,
    pub unit_hash: ActionHash,
    pub total_purchase_price_cents: u64,
    pub monthly_rent_cents: u64,
    pub equity_portion_percent: u8,
    pub accumulated_equity_cents: u64,
    pub started_at: Timestamp,
    pub target_completion: Timestamp,
    pub status: AgreementStatus,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Member(Member),
    MemberApplication(MemberApplication),
    WaitListEntry(WaitListEntry),
    RentToOwnAgreement(RentToOwnAgreement),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All members anchor
    AllMembers,
    /// Agent to their member record
    AgentToMember,
    /// All applications anchor
    AllApplications,
    /// Applicant to application
    ApplicantToApplication,
    /// Waitlist anchor
    Waitlist,
    /// Member to rent-to-own agreement
    MemberToAgreement,
    /// Unit to rent-to-own agreement
    UnitToAgreement,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Member(member) => validate_create_member(action, member),
                EntryTypes::MemberApplication(app) => validate_create_application(action, app),
                EntryTypes::WaitListEntry(entry) => validate_create_waitlist(action, entry),
                EntryTypes::RentToOwnAgreement(agreement) => {
                    validate_create_agreement(action, agreement)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Member(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MemberApplication(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WaitListEntry(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::RentToOwnAgreement(agreement) => validate_update_agreement(agreement),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllMembers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMembers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToMember => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToMember link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllApplications => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllApplications link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ApplicantToApplication => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ApplicantToApplication link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::Waitlist => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Waitlist link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MemberToAgreement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToAgreement link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::UnitToAgreement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "UnitToAgreement link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
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
    }
}

fn validate_create_member(_action: Create, member: Member) -> ExternResult<ValidateCallbackResult> {
    if member.monthly_charge_cents == 0 && member.membership_type != MembershipType::Associate {
        return Ok(ValidateCallbackResult::Invalid(
            "Non-associate members must have a monthly charge".into(),
        ));
    }
    if member.membership_type == MembershipType::FullShare && !member.voting_rights {
        return Ok(ValidateCallbackResult::Invalid(
            "Full share members must have voting rights".into(),
        ));
    }
    if member.equity_portion_percent_valid() {
        Ok(ValidateCallbackResult::Valid)
    } else {
        Ok(ValidateCallbackResult::Invalid(
            "Invalid equity configuration".into(),
        ))
    }
}

fn validate_create_application(
    _action: Create,
    app: MemberApplication,
) -> ExternResult<ValidateCallbackResult> {
    if app.household_size == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Household size must be at least 1".into(),
        ));
    }
    if app.references.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum 10 references allowed".into(),
        ));
    }
    for r in &app.references {
        if r.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Reference too long (max 256 chars per item)".into(),
            ));
        }
    }
    if app.status != ApplicationStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New applications must have Pending status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_waitlist(
    _action: Create,
    _entry: WaitListEntry,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_agreement(
    _action: Create,
    agreement: RentToOwnAgreement,
) -> ExternResult<ValidateCallbackResult> {
    if agreement.total_purchase_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Purchase price must be greater than 0".into(),
        ));
    }
    if agreement.monthly_rent_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Monthly rent must be greater than 0".into(),
        ));
    }
    if agreement.equity_portion_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Equity portion percent cannot exceed 100".into(),
        ));
    }
    if agreement.accumulated_equity_cents != 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "New agreements must start with 0 accumulated equity".into(),
        ));
    }
    if agreement.status != AgreementStatus::Active {
        return Ok(ValidateCallbackResult::Invalid(
            "New agreements must be Active".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_agreement(
    agreement: RentToOwnAgreement,
) -> ExternResult<ValidateCallbackResult> {
    if agreement.equity_portion_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Equity portion percent cannot exceed 100".into(),
        ));
    }
    if agreement.accumulated_equity_cents > agreement.total_purchase_price_cents {
        return Ok(ValidateCallbackResult::Invalid(
            "Accumulated equity cannot exceed purchase price".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Helper trait for member validation
impl Member {
    fn equity_portion_percent_valid(&self) -> bool {
        match self.membership_type {
            MembershipType::FullShare => self.share_equity_cents > 0,
            MembershipType::LimitedEquity => true,
            MembershipType::RentToOwn => true,
            MembershipType::Renter => self.share_equity_cents == 0,
            MembershipType::Associate => self.share_equity_cents == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Valid builder helpers ────────────────────────────────────────────

    fn valid_member() -> Member {
        Member {
            agent: agent_a(),
            unit_hash: Some(ActionHash::from_raw_36(vec![10u8; 36])),
            membership_type: MembershipType::FullShare,
            share_equity_cents: 50000,
            joined_at: Timestamp::from_micros(1000000),
            monthly_charge_cents: 1200_00,
            voting_rights: true,
            status: MemberStatus::Active,
        }
    }

    fn valid_application() -> MemberApplication {
        MemberApplication {
            applicant: agent_a(),
            requested_unit: Some(ActionHash::from_raw_36(vec![20u8; 36])),
            membership_type_requested: MembershipType::FullShare,
            applied_at: Timestamp::from_micros(1000000),
            household_size: 3,
            income_verified: true,
            references: vec!["Alice Smith".to_string(), "Bob Jones".to_string()],
            status: ApplicationStatus::Pending,
        }
    }

    fn valid_waitlist_entry() -> WaitListEntry {
        WaitListEntry {
            application_hash: ActionHash::from_raw_36(vec![30u8; 36]),
            position: 1,
            unit_type_preference: Some(UnitType::TwoBedroom),
            added_at: Timestamp::from_micros(1000000),
        }
    }

    fn valid_agreement() -> RentToOwnAgreement {
        RentToOwnAgreement {
            member: agent_a(),
            unit_hash: ActionHash::from_raw_36(vec![40u8; 36]),
            total_purchase_price_cents: 250_000_00,
            monthly_rent_cents: 1500_00,
            equity_portion_percent: 25,
            accumulated_equity_cents: 0,
            started_at: Timestamp::from_micros(1000000),
            target_completion: Timestamp::from_micros(100_000_000),
            status: AgreementStatus::Active,
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_unit_type() {
        let types = vec![
            UnitType::Studio,
            UnitType::OneBedroom,
            UnitType::TwoBedroom,
            UnitType::ThreeBedroom,
            UnitType::FourPlus,
            UnitType::Accessible,
            UnitType::Family,
        ];
        for ut in &types {
            let json = serde_json::to_string(ut).unwrap();
            let back: UnitType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, ut);
        }
    }

    #[test]
    fn serde_roundtrip_membership_type() {
        let types = vec![
            MembershipType::FullShare,
            MembershipType::LimitedEquity,
            MembershipType::RentToOwn,
            MembershipType::Renter,
            MembershipType::Associate,
        ];
        for mt in &types {
            let json = serde_json::to_string(mt).unwrap();
            let back: MembershipType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, mt);
        }
    }

    #[test]
    fn serde_roundtrip_member_status() {
        let statuses = vec![
            MemberStatus::Active,
            MemberStatus::OnLeave,
            MemberStatus::Suspended,
            MemberStatus::Former,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: MemberStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_application_status() {
        let statuses = vec![
            ApplicationStatus::Pending,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Approved,
            ApplicationStatus::Rejected,
            ApplicationStatus::Waitlisted,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: ApplicationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_agreement_status() {
        let statuses = vec![
            AgreementStatus::Active,
            AgreementStatus::Completed,
            AgreementStatus::Defaulted,
            AgreementStatus::Terminated,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: AgreementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("all_members".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, anchor);
    }

    #[test]
    fn serde_roundtrip_member() {
        let member = valid_member();
        let json = serde_json::to_string(&member).unwrap();
        let back: Member = serde_json::from_str(&json).unwrap();
        assert_eq!(back, member);
    }

    #[test]
    fn serde_roundtrip_member_application() {
        let app = valid_application();
        let json = serde_json::to_string(&app).unwrap();
        let back: MemberApplication = serde_json::from_str(&json).unwrap();
        assert_eq!(back, app);
    }

    #[test]
    fn serde_roundtrip_waitlist_entry() {
        let entry = valid_waitlist_entry();
        let json = serde_json::to_string(&entry).unwrap();
        let back: WaitListEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn serde_roundtrip_rent_to_own_agreement() {
        let agreement = valid_agreement();
        let json = serde_json::to_string(&agreement).unwrap();
        let back: RentToOwnAgreement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, agreement);
    }

    #[test]
    fn serde_roundtrip_member_no_unit() {
        let mut member = valid_member();
        member.unit_hash = None;
        member.membership_type = MembershipType::Associate;
        member.share_equity_cents = 0;
        member.monthly_charge_cents = 0;
        member.voting_rights = false;
        let json = serde_json::to_string(&member).unwrap();
        let back: Member = serde_json::from_str(&json).unwrap();
        assert_eq!(back, member);
    }

    #[test]
    fn serde_roundtrip_application_no_unit_no_refs() {
        let mut app = valid_application();
        app.requested_unit = None;
        app.references = vec![];
        let json = serde_json::to_string(&app).unwrap();
        let back: MemberApplication = serde_json::from_str(&json).unwrap();
        assert_eq!(back, app);
    }

    #[test]
    fn serde_roundtrip_waitlist_no_preference() {
        let mut entry = valid_waitlist_entry();
        entry.unit_type_preference = None;
        let json = serde_json::to_string(&entry).unwrap();
        let back: WaitListEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, entry);
    }

    // ── validate_create_member tests ────────────────────────────────────

    #[test]
    fn create_member_valid_full_share() {
        assert_valid(validate_create_member(fake_create(), valid_member()));
    }

    #[test]
    fn create_member_valid_limited_equity() {
        let mut m = valid_member();
        m.membership_type = MembershipType::LimitedEquity;
        m.share_equity_cents = 10000;
        m.monthly_charge_cents = 800_00;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_valid_rent_to_own() {
        let mut m = valid_member();
        m.membership_type = MembershipType::RentToOwn;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 1000_00;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_valid_renter() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 900_00;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_valid_associate_zero_charge() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Associate;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 0;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_valid_associate_with_charge() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Associate;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 50_00;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_non_associate_zero_charge_invalid() {
        for mt in [
            MembershipType::FullShare,
            MembershipType::LimitedEquity,
            MembershipType::RentToOwn,
            MembershipType::Renter,
        ] {
            let mut m = valid_member();
            m.membership_type = mt;
            m.monthly_charge_cents = 0;
            m.share_equity_cents = if m.membership_type == MembershipType::FullShare {
                50000
            } else {
                0
            };
            m.voting_rights = m.membership_type == MembershipType::FullShare;
            assert_invalid(
                validate_create_member(fake_create(), m),
                "Non-associate members must have a monthly charge",
            );
        }
    }

    #[test]
    fn create_member_full_share_no_voting_rights_invalid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.voting_rights = false;
        assert_invalid(
            validate_create_member(fake_create(), m),
            "Full share members must have voting rights",
        );
    }

    #[test]
    fn create_member_full_share_with_voting_rights_valid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.voting_rights = true;
        m.share_equity_cents = 1;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_full_share_zero_equity_invalid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.share_equity_cents = 0;
        m.voting_rights = true;
        // zero charge triggers first check before equity check
        m.monthly_charge_cents = 1;
        assert_invalid(
            validate_create_member(fake_create(), m),
            "Invalid equity configuration",
        );
    }

    #[test]
    fn create_member_full_share_min_equity_valid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.share_equity_cents = 1;
        m.voting_rights = true;
        m.monthly_charge_cents = 1;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_renter_with_equity_invalid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 1;
        m.monthly_charge_cents = 500_00;
        m.voting_rights = false;
        assert_invalid(
            validate_create_member(fake_create(), m),
            "Invalid equity configuration",
        );
    }

    #[test]
    fn create_member_associate_with_equity_invalid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Associate;
        m.share_equity_cents = 1;
        m.monthly_charge_cents = 0;
        m.voting_rights = false;
        assert_invalid(
            validate_create_member(fake_create(), m),
            "Invalid equity configuration",
        );
    }

    #[test]
    fn create_member_limited_equity_zero_equity_valid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::LimitedEquity;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 700_00;
        m.voting_rights = true;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_limited_equity_with_equity_valid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::LimitedEquity;
        m.share_equity_cents = 25000;
        m.monthly_charge_cents = 700_00;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_rent_to_own_any_equity_valid() {
        for eq in [0, 1, 10000, u64::MAX] {
            let mut m = valid_member();
            m.membership_type = MembershipType::RentToOwn;
            m.share_equity_cents = eq;
            m.monthly_charge_cents = 1000_00;
            m.voting_rights = false;
            assert_valid(validate_create_member(fake_create(), m));
        }
    }

    #[test]
    fn create_member_all_statuses_valid() {
        for status in [
            MemberStatus::Active,
            MemberStatus::OnLeave,
            MemberStatus::Suspended,
            MemberStatus::Former,
        ] {
            let mut m = valid_member();
            m.status = status;
            assert_valid(validate_create_member(fake_create(), m));
        }
    }

    #[test]
    fn create_member_no_unit_hash_valid() {
        let mut m = valid_member();
        m.unit_hash = None;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_max_monthly_charge_valid() {
        let mut m = valid_member();
        m.monthly_charge_cents = u64::MAX;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_max_equity_valid() {
        let mut m = valid_member();
        m.share_equity_cents = u64::MAX;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_full_share_zero_charge_hits_first_check() {
        // FullShare with zero charge and no voting rights: first check fires
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.monthly_charge_cents = 0;
        m.voting_rights = false;
        m.share_equity_cents = 50000;
        assert_invalid(
            validate_create_member(fake_create(), m),
            "Non-associate members must have a monthly charge",
        );
    }

    // ── validate_create_application tests ────────────────────────────────

    #[test]
    fn create_application_valid() {
        assert_valid(validate_create_application(
            fake_create(),
            valid_application(),
        ));
    }

    #[test]
    fn create_application_household_size_zero_invalid() {
        let mut app = valid_application();
        app.household_size = 0;
        assert_invalid(
            validate_create_application(fake_create(), app),
            "Household size must be at least 1",
        );
    }

    #[test]
    fn create_application_household_size_one_valid() {
        let mut app = valid_application();
        app.household_size = 1;
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_household_size_max_valid() {
        let mut app = valid_application();
        app.household_size = u8::MAX;
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_too_many_references_invalid() {
        let mut app = valid_application();
        app.references = (0..11).map(|i| format!("Ref {i}")).collect();
        assert_invalid(
            validate_create_application(fake_create(), app),
            "Maximum 10 references allowed",
        );
    }

    #[test]
    fn create_application_ten_references_valid() {
        let mut app = valid_application();
        app.references = (0..10).map(|i| format!("Ref {i}")).collect();
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_zero_references_valid() {
        let mut app = valid_application();
        app.references = vec![];
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_status_not_pending_invalid() {
        for status in [
            ApplicationStatus::UnderReview,
            ApplicationStatus::Approved,
            ApplicationStatus::Rejected,
            ApplicationStatus::Waitlisted,
        ] {
            let mut app = valid_application();
            app.status = status;
            assert_invalid(
                validate_create_application(fake_create(), app),
                "New applications must have Pending status",
            );
        }
    }

    #[test]
    fn create_application_no_requested_unit_valid() {
        let mut app = valid_application();
        app.requested_unit = None;
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_income_not_verified_valid() {
        let mut app = valid_application();
        app.income_verified = false;
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_all_membership_types_valid() {
        for mt in [
            MembershipType::FullShare,
            MembershipType::LimitedEquity,
            MembershipType::RentToOwn,
            MembershipType::Renter,
            MembershipType::Associate,
        ] {
            let mut app = valid_application();
            app.membership_type_requested = mt;
            assert_valid(validate_create_application(fake_create(), app));
        }
    }

    #[test]
    fn create_application_references_with_empty_strings_valid() {
        let mut app = valid_application();
        app.references = vec!["".to_string(), "".to_string()];
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_references_unicode_valid() {
        let mut app = valid_application();
        app.references = vec!["Ari Takahashi".to_string(), "Muller".to_string()];
        assert_valid(validate_create_application(fake_create(), app));
    }

    // ── validate_create_waitlist tests ───────────────────────────────────

    #[test]
    fn create_waitlist_valid() {
        assert_valid(validate_create_waitlist(
            fake_create(),
            valid_waitlist_entry(),
        ));
    }

    #[test]
    fn create_waitlist_position_zero_valid() {
        let mut entry = valid_waitlist_entry();
        entry.position = 0;
        assert_valid(validate_create_waitlist(fake_create(), entry));
    }

    #[test]
    fn create_waitlist_position_max_valid() {
        let mut entry = valid_waitlist_entry();
        entry.position = u32::MAX;
        assert_valid(validate_create_waitlist(fake_create(), entry));
    }

    #[test]
    fn create_waitlist_no_unit_preference_valid() {
        let mut entry = valid_waitlist_entry();
        entry.unit_type_preference = None;
        assert_valid(validate_create_waitlist(fake_create(), entry));
    }

    #[test]
    fn create_waitlist_all_unit_types_valid() {
        for ut in [
            UnitType::Studio,
            UnitType::OneBedroom,
            UnitType::TwoBedroom,
            UnitType::ThreeBedroom,
            UnitType::FourPlus,
            UnitType::Accessible,
            UnitType::Family,
        ] {
            let mut entry = valid_waitlist_entry();
            entry.unit_type_preference = Some(ut);
            assert_valid(validate_create_waitlist(fake_create(), entry));
        }
    }

    // ── validate_create_agreement tests ──────────────────────────────────

    #[test]
    fn create_agreement_valid() {
        assert_valid(validate_create_agreement(fake_create(), valid_agreement()));
    }

    #[test]
    fn create_agreement_zero_purchase_price_invalid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = 0;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Purchase price must be greater than 0",
        );
    }

    #[test]
    fn create_agreement_min_purchase_price_valid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = 1;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_max_purchase_price_valid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = u64::MAX;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_zero_monthly_rent_invalid() {
        let mut a = valid_agreement();
        a.monthly_rent_cents = 0;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Monthly rent must be greater than 0",
        );
    }

    #[test]
    fn create_agreement_min_monthly_rent_valid() {
        let mut a = valid_agreement();
        a.monthly_rent_cents = 1;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_equity_percent_101_invalid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 101;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Equity portion percent cannot exceed 100",
        );
    }

    #[test]
    fn create_agreement_equity_percent_255_invalid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = u8::MAX;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Equity portion percent cannot exceed 100",
        );
    }

    #[test]
    fn create_agreement_equity_percent_100_valid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 100;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_equity_percent_0_valid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 0;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_nonzero_accumulated_equity_invalid() {
        let mut a = valid_agreement();
        a.accumulated_equity_cents = 1;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "New agreements must start with 0 accumulated equity",
        );
    }

    #[test]
    fn create_agreement_large_accumulated_equity_invalid() {
        let mut a = valid_agreement();
        a.accumulated_equity_cents = u64::MAX;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "New agreements must start with 0 accumulated equity",
        );
    }

    #[test]
    fn create_agreement_status_not_active_invalid() {
        for status in [
            AgreementStatus::Completed,
            AgreementStatus::Defaulted,
            AgreementStatus::Terminated,
        ] {
            let mut a = valid_agreement();
            a.status = status;
            assert_invalid(
                validate_create_agreement(fake_create(), a),
                "New agreements must be Active",
            );
        }
    }

    #[test]
    fn create_agreement_zero_price_and_zero_rent() {
        // Both zero: purchase price check fires first
        let mut a = valid_agreement();
        a.total_purchase_price_cents = 0;
        a.monthly_rent_cents = 0;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Purchase price must be greater than 0",
        );
    }

    // ── validate_update_agreement tests ──────────────────────────────────

    #[test]
    fn update_agreement_valid() {
        let mut a = valid_agreement();
        a.accumulated_equity_cents = 10000;
        assert_valid(validate_update_agreement(a));
    }

    #[test]
    fn update_agreement_equity_percent_over_100_invalid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 101;
        assert_invalid(
            validate_update_agreement(a),
            "Equity portion percent cannot exceed 100",
        );
    }

    #[test]
    fn update_agreement_equity_percent_100_valid() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 100;
        assert_valid(validate_update_agreement(a));
    }

    #[test]
    fn update_agreement_accumulated_exceeds_price_invalid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = 100_000;
        a.accumulated_equity_cents = 100_001;
        assert_invalid(
            validate_update_agreement(a),
            "Accumulated equity cannot exceed purchase price",
        );
    }

    #[test]
    fn update_agreement_accumulated_equals_price_valid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = 100_000;
        a.accumulated_equity_cents = 100_000;
        assert_valid(validate_update_agreement(a));
    }

    #[test]
    fn update_agreement_accumulated_zero_valid() {
        let mut a = valid_agreement();
        a.accumulated_equity_cents = 0;
        assert_valid(validate_update_agreement(a));
    }

    #[test]
    fn update_agreement_all_statuses_valid() {
        for status in [
            AgreementStatus::Active,
            AgreementStatus::Completed,
            AgreementStatus::Defaulted,
            AgreementStatus::Terminated,
        ] {
            let mut a = valid_agreement();
            a.status = status;
            assert_valid(validate_update_agreement(a));
        }
    }

    #[test]
    fn update_agreement_completed_with_full_equity_valid() {
        let mut a = valid_agreement();
        a.status = AgreementStatus::Completed;
        a.total_purchase_price_cents = 250_000_00;
        a.accumulated_equity_cents = 250_000_00;
        a.equity_portion_percent = 100;
        assert_valid(validate_update_agreement(a));
    }

    #[test]
    fn update_agreement_equity_over_100_trumps_accumulated_check() {
        // Both equity_percent > 100 and accumulated > price: first check fires
        let mut a = valid_agreement();
        a.equity_portion_percent = 101;
        a.total_purchase_price_cents = 100;
        a.accumulated_equity_cents = 200;
        assert_invalid(
            validate_update_agreement(a),
            "Equity portion percent cannot exceed 100",
        );
    }

    #[test]
    fn update_agreement_max_values_valid() {
        let mut a = valid_agreement();
        a.total_purchase_price_cents = u64::MAX;
        a.accumulated_equity_cents = u64::MAX;
        a.monthly_rent_cents = u64::MAX;
        a.equity_portion_percent = 100;
        assert_valid(validate_update_agreement(a));
    }

    // ── equity_portion_percent_valid unit tests ─────────────────────────

    #[test]
    fn equity_valid_full_share_with_equity() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.share_equity_cents = 1;
        assert!(m.equity_portion_percent_valid());
    }

    #[test]
    fn equity_invalid_full_share_zero_equity() {
        let mut m = valid_member();
        m.membership_type = MembershipType::FullShare;
        m.share_equity_cents = 0;
        assert!(!m.equity_portion_percent_valid());
    }

    #[test]
    fn equity_valid_limited_equity_always() {
        for eq in [0, 1, 50000, u64::MAX] {
            let mut m = valid_member();
            m.membership_type = MembershipType::LimitedEquity;
            m.share_equity_cents = eq;
            assert!(m.equity_portion_percent_valid());
        }
    }

    #[test]
    fn equity_valid_rent_to_own_always() {
        for eq in [0, 1, 50000, u64::MAX] {
            let mut m = valid_member();
            m.membership_type = MembershipType::RentToOwn;
            m.share_equity_cents = eq;
            assert!(m.equity_portion_percent_valid());
        }
    }

    #[test]
    fn equity_valid_renter_zero_only() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 0;
        assert!(m.equity_portion_percent_valid());
    }

    #[test]
    fn equity_invalid_renter_nonzero() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 1;
        assert!(!m.equity_portion_percent_valid());
    }

    #[test]
    fn equity_valid_associate_zero_only() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Associate;
        m.share_equity_cents = 0;
        assert!(m.equity_portion_percent_valid());
    }

    #[test]
    fn equity_invalid_associate_nonzero() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Associate;
        m.share_equity_cents = 1;
        assert!(!m.equity_portion_percent_valid());
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn create_application_eleven_references_invalid() {
        let mut app = valid_application();
        app.references = (0..11).map(|i| format!("Reference #{i}")).collect();
        assert_invalid(
            validate_create_application(fake_create(), app),
            "Maximum 10 references allowed",
        );
    }

    #[test]
    fn create_application_unicode_references_valid() {
        let mut app = valid_application();
        app.references = vec!["Sakura Yamamoto".to_string(), "Elena Petrova".to_string()];
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_agreement_boundary_equity_percent_100() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 100;
        assert_valid(validate_create_agreement(fake_create(), a));
    }

    #[test]
    fn create_agreement_boundary_equity_percent_101() {
        let mut a = valid_agreement();
        a.equity_portion_percent = 101;
        assert_invalid(
            validate_create_agreement(fake_create(), a),
            "Equity portion percent cannot exceed 100",
        );
    }

    #[test]
    fn create_member_renter_no_voting_rights_valid() {
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 500_00;
        m.voting_rights = false;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn create_member_renter_with_voting_rights_valid() {
        // Validation does not restrict voting_rights for Renter
        let mut m = valid_member();
        m.membership_type = MembershipType::Renter;
        m.share_equity_cents = 0;
        m.monthly_charge_cents = 500_00;
        m.voting_rights = true;
        assert_valid(validate_create_member(fake_create(), m));
    }

    #[test]
    fn serde_roundtrip_member_max_values() {
        let member = Member {
            agent: agent_a(),
            unit_hash: Some(ActionHash::from_raw_36(vec![255u8; 36])),
            membership_type: MembershipType::FullShare,
            share_equity_cents: u64::MAX,
            joined_at: Timestamp::from_micros(i64::MAX),
            monthly_charge_cents: u64::MAX,
            voting_rights: true,
            status: MemberStatus::Active,
        };
        let json = serde_json::to_string(&member).unwrap();
        let back: Member = serde_json::from_str(&json).unwrap();
        assert_eq!(back, member);
    }

    #[test]
    fn serde_roundtrip_agreement_max_values() {
        let agreement = RentToOwnAgreement {
            member: agent_a(),
            unit_hash: ActionHash::from_raw_36(vec![255u8; 36]),
            total_purchase_price_cents: u64::MAX,
            monthly_rent_cents: u64::MAX,
            equity_portion_percent: u8::MAX,
            accumulated_equity_cents: u64::MAX,
            started_at: Timestamp::from_micros(i64::MAX),
            target_completion: Timestamp::from_micros(i64::MAX),
            status: AgreementStatus::Active,
        };
        let json = serde_json::to_string(&agreement).unwrap();
        let back: RentToOwnAgreement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, agreement);
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllMembers
            | LinkTypes::AgentToMember
            | LinkTypes::AllApplications
            | LinkTypes::ApplicantToApplication
            | LinkTypes::Waitlist
            | LinkTypes::MemberToAgreement
            | LinkTypes::UnitToAgreement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_tag_all_members_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::AllMembers,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_all_members_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllMembers, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_waitlist_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::Waitlist,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_waitlist_over_limit() {
        let result = validate_create_link_tag(LinkTypes::Waitlist, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_unit_to_agreement_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::UnitToAgreement,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn link_tag_unit_to_agreement_over_limit() {
        let result = validate_create_link_tag(LinkTypes::UnitToAgreement, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_empty_tag_valid() {
        assert_valid(validate_create_link_tag(LinkTypes::AllMembers, vec![]));
    }

    // ========================================================================
    // STRING LENGTH BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn create_application_reference_at_limit_accepted() {
        let mut app = valid_application();
        app.references = vec!["x".repeat(256)];
        assert_valid(validate_create_application(fake_create(), app));
    }

    #[test]
    fn create_application_reference_over_limit_rejected() {
        let mut app = valid_application();
        app.references = vec!["x".repeat(257)];
        assert_invalid(
            validate_create_application(fake_create(), app),
            "Reference too long (max 256 chars per item)",
        );
    }

    #[test]
    fn create_application_second_reference_over_limit_rejected() {
        let mut app = valid_application();
        app.references = vec!["valid".to_string(), "x".repeat(257)];
        assert_invalid(
            validate_create_application(fake_create(), app),
            "Reference too long (max 256 chars per item)",
        );
    }
}
