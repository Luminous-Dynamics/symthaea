// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Membership Coordinator Zome
//! Business logic for co-op membership, applications, waitlist, and rent-to-own.

use hdk::prelude::*;
use housing_membership_integrity::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal, civic_requirement_voting};
use mycelix_zome_helpers::get_latest_record;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Submit a new membership application

#[hdk_extern]
pub fn submit_application(app: MemberApplication) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "submit_application")?;
    for reference in &app.references {
        if reference.len() > 512 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Each reference must be at most 512 characters".into()
            )));
        }
    }

    let action_hash = create_entry(&EntryTypes::MemberApplication(app.clone()))?;

    // Link to all applications
    create_entry(&EntryTypes::Anchor(Anchor("all_applications".to_string())))?;
    create_link(
        anchor_hash("all_applications")?,
        action_hash.clone(),
        LinkTypes::AllApplications,
        (),
    )?;

    // Link applicant to application
    create_link(
        app.applicant,
        action_hash.clone(),
        LinkTypes::ApplicantToApplication,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created application".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReviewApplicationInput {
    pub application_hash: ActionHash,
    pub new_status: ApplicationStatus,
}

/// Review an application (change its status)
#[hdk_extern]
pub fn review_application(input: ReviewApplicationInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "review_application")?;
    let record = get(input.application_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Application not found".into())
    ))?;

    let mut app: MemberApplication = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid application entry".into()
        )))?;

    app.status = input.new_status;

    let new_hash = update_entry(input.application_hash, &EntryTypes::MemberApplication(app))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated application".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveMemberInput {
    pub application_hash: ActionHash,
    pub unit_hash: Option<ActionHash>,
    pub membership_type: MembershipType,
    pub share_equity_cents: u64,
    pub monthly_charge_cents: u64,
}

/// Approve an application and create a member record
#[hdk_extern]
pub fn approve_member(input: ApproveMemberInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "approve_member")?;
    let record = get(input.application_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Application not found".into())
    ))?;

    let mut app: MemberApplication = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid application entry".into()
        )))?;

    if app.status != ApplicationStatus::Pending && app.status != ApplicationStatus::UnderReview {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Application must be Pending or UnderReview to approve".into()
        )));
    }

    // Update application status
    app.status = ApplicationStatus::Approved;
    update_entry(
        input.application_hash,
        &EntryTypes::MemberApplication(app.clone()),
    )?;

    let now = sys_time()?;

    let voting_rights = matches!(
        input.membership_type,
        MembershipType::FullShare | MembershipType::LimitedEquity
    );

    let member = Member {
        agent: app.applicant.clone(),
        unit_hash: input.unit_hash,
        membership_type: input.membership_type,
        share_equity_cents: input.share_equity_cents,
        joined_at: now,
        monthly_charge_cents: input.monthly_charge_cents,
        voting_rights,
        status: MemberStatus::Active,
    };

    let action_hash = create_entry(&EntryTypes::Member(member.clone()))?;

    // Link to all members
    create_entry(&EntryTypes::Anchor(Anchor("all_members".to_string())))?;
    create_link(
        anchor_hash("all_members")?,
        action_hash.clone(),
        LinkTypes::AllMembers,
        (),
    )?;

    // Link agent to member
    create_link(
        member.agent,
        action_hash.clone(),
        LinkTypes::AgentToMember,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created member".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddToWaitlistInput {
    pub application_hash: ActionHash,
    pub unit_type_preference: Option<housing_membership_integrity::UnitType>,
}

/// Add an applicant to the waitlist
#[hdk_extern]
pub fn add_to_waitlist(input: AddToWaitlistInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "add_to_waitlist")?;
    let now = sys_time()?;

    // Determine position by counting existing waitlist entries
    let links = get_links(
        LinkQuery::try_new(anchor_hash("waitlist")?, LinkTypes::Waitlist)?,
        GetStrategy::default(),
    )?;
    let position = links.len() as u32 + 1;

    let entry = WaitListEntry {
        application_hash: input.application_hash,
        position,
        unit_type_preference: input.unit_type_preference,
        added_at: now,
    };

    let action_hash = create_entry(&EntryTypes::WaitListEntry(entry))?;

    create_entry(&EntryTypes::Anchor(Anchor("waitlist".to_string())))?;
    create_link(
        anchor_hash("waitlist")?,
        action_hash.clone(),
        LinkTypes::Waitlist,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created waitlist entry".into()
    )))
}

/// Get the current waitlist, ordered by position
#[hdk_extern]
pub fn get_waitlist(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("waitlist")?, LinkTypes::Waitlist)?,
        GetStrategy::default(),
    )?;

    let mut entries = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            entries.push(record);
        }
    }

    // Sort by position
    entries.sort_by(|a, b| {
        let pos_a = a
            .entry()
            .to_app_option::<WaitListEntry>()
            .ok()
            .flatten()
            .map(|e| e.position)
            .unwrap_or(u32::MAX);
        let pos_b = b
            .entry()
            .to_app_option::<WaitListEntry>()
            .ok()
            .flatten()
            .map(|e| e.position)
            .unwrap_or(u32::MAX);
        pos_a.cmp(&pos_b)
    });

    Ok(entries)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRentToOwnInput {
    pub member: AgentPubKey,
    pub unit_hash: ActionHash,
    pub total_purchase_price_cents: u64,
    pub monthly_rent_cents: u64,
    pub equity_portion_percent: u8,
    pub target_completion: Timestamp,
}

/// Create a rent-to-own agreement
#[hdk_extern]
pub fn create_rent_to_own(input: CreateRentToOwnInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "create_rent_to_own")?;
    let now = sys_time()?;

    let agreement = RentToOwnAgreement {
        member: input.member.clone(),
        unit_hash: input.unit_hash.clone(),
        total_purchase_price_cents: input.total_purchase_price_cents,
        monthly_rent_cents: input.monthly_rent_cents,
        equity_portion_percent: input.equity_portion_percent,
        accumulated_equity_cents: 0,
        started_at: now,
        target_completion: input.target_completion,
        status: AgreementStatus::Active,
    };

    let action_hash = create_entry(&EntryTypes::RentToOwnAgreement(agreement))?;

    // Link member to agreement
    create_link(
        input.member,
        action_hash.clone(),
        LinkTypes::MemberToAgreement,
        (),
    )?;

    // Link unit to agreement
    create_link(
        input.unit_hash,
        action_hash.clone(),
        LinkTypes::UnitToAgreement,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created agreement".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordRentPaymentInput {
    pub agreement_hash: ActionHash,
    pub payment_amount_cents: u64,
}

/// Record a rent payment and update accumulated equity
#[hdk_extern]
pub fn record_rent_payment(input: RecordRentPaymentInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_rent_payment")?;
    let record = get(input.agreement_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Agreement not found".into())
    ))?;

    let mut agreement: RentToOwnAgreement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid agreement entry".into()
        )))?;

    if agreement.status != AgreementStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agreement is not active".into()
        )));
    }

    // Calculate equity portion of payment
    let equity_addition = (input.payment_amount_cents as u128
        * agreement.equity_portion_percent as u128
        / 100) as u64;
    agreement.accumulated_equity_cents += equity_addition;

    // Check if completed
    if agreement.accumulated_equity_cents >= agreement.total_purchase_price_cents {
        agreement.accumulated_equity_cents = agreement.total_purchase_price_cents;
        agreement.status = AgreementStatus::Completed;
    }

    let new_hash = update_entry(
        input.agreement_hash,
        &EntryTypes::RentToOwnAgreement(agreement),
    )?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated agreement".into()
    )))
}

/// Get accumulated equity for a member's rent-to-own agreement
#[hdk_extern]
pub fn get_member_equity(member: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::MemberToAgreement)?,
        GetStrategy::default(),
    )?;

    let mut agreements = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            agreements.push(record);
        }
    }

    Ok(agreements)
}

/// Get all members
#[hdk_extern]
pub fn get_members(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_members")?, LinkTypes::AllMembers)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            members.push(record);
        }
    }

    Ok(members)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn review_application_input_serde_roundtrip() {
        let input = ReviewApplicationInput {
            application_hash: fake_action_hash(),
            new_status: ApplicationStatus::Approved,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ReviewApplicationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, ApplicationStatus::Approved);
    }

    #[test]
    fn approve_member_input_serde_roundtrip() {
        let input = ApproveMemberInput {
            application_hash: fake_action_hash(),
            unit_hash: Some(fake_action_hash()),
            membership_type: MembershipType::FullShare,
            share_equity_cents: 50000,
            monthly_charge_cents: 120000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ApproveMemberInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.membership_type, MembershipType::FullShare);
        assert_eq!(decoded.share_equity_cents, 50000);
        assert_eq!(decoded.monthly_charge_cents, 120000);
    }

    #[test]
    fn approve_member_input_no_unit_serde_roundtrip() {
        let input = ApproveMemberInput {
            application_hash: fake_action_hash(),
            unit_hash: None,
            membership_type: MembershipType::Associate,
            share_equity_cents: 0,
            monthly_charge_cents: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ApproveMemberInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.unit_hash.is_none());
        assert_eq!(decoded.membership_type, MembershipType::Associate);
    }

    #[test]
    fn add_to_waitlist_input_serde_roundtrip() {
        let input = AddToWaitlistInput {
            application_hash: fake_action_hash(),
            unit_type_preference: Some(housing_membership_integrity::UnitType::TwoBedroom),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddToWaitlistInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.unit_type_preference,
            Some(housing_membership_integrity::UnitType::TwoBedroom)
        );
    }

    #[test]
    fn add_to_waitlist_input_no_preference_serde_roundtrip() {
        let input = AddToWaitlistInput {
            application_hash: fake_action_hash(),
            unit_type_preference: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddToWaitlistInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.unit_type_preference.is_none());
    }

    #[test]
    fn create_rent_to_own_input_serde_roundtrip() {
        let input = CreateRentToOwnInput {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            total_purchase_price_cents: 30000000,
            monthly_rent_cents: 150000,
            equity_portion_percent: 25,
            target_completion: Timestamp::from_micros(1_000_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRentToOwnInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_purchase_price_cents, 30000000);
        assert_eq!(decoded.monthly_rent_cents, 150000);
        assert_eq!(decoded.equity_portion_percent, 25);
    }

    #[test]
    fn record_rent_payment_input_serde_roundtrip() {
        let input = RecordRentPaymentInput {
            agreement_hash: fake_action_hash(),
            payment_amount_cents: 150000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordRentPaymentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payment_amount_cents, 150000);
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn membership_type_all_variants_serde() {
        let variants = vec![
            MembershipType::FullShare,
            MembershipType::LimitedEquity,
            MembershipType::RentToOwn,
            MembershipType::Renter,
            MembershipType::Associate,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MembershipType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn member_status_all_variants_serde() {
        let variants = vec![
            MemberStatus::Active,
            MemberStatus::OnLeave,
            MemberStatus::Suspended,
            MemberStatus::Former,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MemberStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn application_status_all_variants_serde() {
        let variants = vec![
            ApplicationStatus::Pending,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Approved,
            ApplicationStatus::Rejected,
            ApplicationStatus::Waitlisted,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: ApplicationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn agreement_status_all_variants_serde() {
        let variants = vec![
            AgreementStatus::Active,
            AgreementStatus::Completed,
            AgreementStatus::Defaulted,
            AgreementStatus::Terminated,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: AgreementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn unit_type_all_variants_serde() {
        let variants = vec![
            housing_membership_integrity::UnitType::Studio,
            housing_membership_integrity::UnitType::OneBedroom,
            housing_membership_integrity::UnitType::TwoBedroom,
            housing_membership_integrity::UnitType::ThreeBedroom,
            housing_membership_integrity::UnitType::FourPlus,
            housing_membership_integrity::UnitType::Accessible,
            housing_membership_integrity::UnitType::Family,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: housing_membership_integrity::UnitType =
                serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn member_application_serde_roundtrip() {
        let app = MemberApplication {
            applicant: fake_agent(),
            requested_unit: Some(fake_action_hash()),
            membership_type_requested: MembershipType::LimitedEquity,
            applied_at: Timestamp::from_micros(1000),
            household_size: 3,
            income_verified: true,
            references: vec!["Ref 1".to_string(), "Ref 2".to_string()],
            status: ApplicationStatus::Pending,
        };
        let json = serde_json::to_string(&app).unwrap();
        let decoded: MemberApplication = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, app);
    }

    #[test]
    fn member_serde_roundtrip() {
        let member = Member {
            agent: fake_agent(),
            unit_hash: Some(fake_action_hash()),
            membership_type: MembershipType::FullShare,
            share_equity_cents: 50000,
            joined_at: Timestamp::from_micros(2000),
            monthly_charge_cents: 120000,
            voting_rights: true,
            status: MemberStatus::Active,
        };
        let json = serde_json::to_string(&member).unwrap();
        let decoded: Member = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, member);
    }

    #[test]
    fn rent_to_own_agreement_serde_roundtrip() {
        let agreement = RentToOwnAgreement {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            total_purchase_price_cents: 30000000,
            monthly_rent_cents: 150000,
            equity_portion_percent: 25,
            accumulated_equity_cents: 750000,
            started_at: Timestamp::from_micros(1000),
            target_completion: Timestamp::from_micros(1_000_000_000),
            status: AgreementStatus::Active,
        };
        let json = serde_json::to_string(&agreement).unwrap();
        let decoded: RentToOwnAgreement = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, agreement);
    }

    #[test]
    fn waitlist_entry_serde_roundtrip() {
        let entry = WaitListEntry {
            application_hash: fake_action_hash(),
            position: 5,
            unit_type_preference: Some(housing_membership_integrity::UnitType::Family),
            added_at: Timestamp::from_micros(3000),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: WaitListEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, entry);
    }

    // ── Review input with all statuses ────────────────────────────────

    #[test]
    fn review_application_input_all_statuses_serde() {
        for status in [
            ApplicationStatus::Pending,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Approved,
            ApplicationStatus::Rejected,
            ApplicationStatus::Waitlisted,
        ] {
            let input = ReviewApplicationInput {
                application_hash: fake_action_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: ReviewApplicationInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    // ── Boundary condition tests ──────────────────────────────────────

    #[test]
    fn create_rent_to_own_input_zero_equity_percent_serde() {
        let input = CreateRentToOwnInput {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            total_purchase_price_cents: 20000000,
            monthly_rent_cents: 100000,
            equity_portion_percent: 0,
            target_completion: Timestamp::from_micros(2_000_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRentToOwnInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.equity_portion_percent, 0);
    }

    #[test]
    fn create_rent_to_own_input_full_equity_percent_serde() {
        let input = CreateRentToOwnInput {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            total_purchase_price_cents: 20000000,
            monthly_rent_cents: 100000,
            equity_portion_percent: 100,
            target_completion: Timestamp::from_micros(2_000_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRentToOwnInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.equity_portion_percent, 100);
    }

    #[test]
    fn record_rent_payment_input_zero_payment_serde() {
        let input = RecordRentPaymentInput {
            agreement_hash: fake_action_hash(),
            payment_amount_cents: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordRentPaymentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payment_amount_cents, 0);
    }

    #[test]
    fn member_application_max_household_size_serde() {
        let app = MemberApplication {
            applicant: fake_agent(),
            requested_unit: None,
            membership_type_requested: MembershipType::Renter,
            applied_at: Timestamp::from_micros(5000),
            household_size: u8::MAX,
            income_verified: false,
            references: vec![],
            status: ApplicationStatus::Pending,
        };
        let json = serde_json::to_string(&app).unwrap();
        let decoded: MemberApplication = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.household_size, u8::MAX);
        assert_eq!(decoded, app);
    }

    #[test]
    fn waitlist_entry_max_position_serde() {
        let entry = WaitListEntry {
            application_hash: fake_action_hash(),
            position: u32::MAX,
            unit_type_preference: None,
            added_at: Timestamp::from_micros(9000),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: WaitListEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.position, u32::MAX);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn approve_member_input_all_membership_types_serde() {
        for mt in [
            MembershipType::FullShare,
            MembershipType::LimitedEquity,
            MembershipType::RentToOwn,
            MembershipType::Renter,
            MembershipType::Associate,
        ] {
            let input = ApproveMemberInput {
                application_hash: fake_action_hash(),
                unit_hash: None,
                membership_type: mt.clone(),
                share_equity_cents: 0,
                monthly_charge_cents: 0,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: ApproveMemberInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.membership_type, mt);
        }
    }
}
