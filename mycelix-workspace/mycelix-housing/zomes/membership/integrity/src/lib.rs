// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Membership Integrity Zome
//! Entry types and validation for co-op members, applications, waitlists, and rent-to-own.

use hdi::prelude::*;

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
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllMembers => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToMember => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllApplications => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ApplicantToApplication => Ok(ValidateCallbackResult::Valid),
            LinkTypes::Waitlist => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToAgreement => Ok(ValidateCallbackResult::Valid),
            LinkTypes::UnitToAgreement => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
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
