// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Membership Coordinator Zome
//! Business logic for co-op membership, applications, waitlist, and rent-to-own.

use hdk::prelude::*;
use membership_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Submit a new membership application
#[hdk_extern]
pub fn submit_application(app: MemberApplication) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created member".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddToWaitlistInput {
    pub application_hash: ActionHash,
    pub unit_type_preference: Option<membership_integrity::UnitType>,
}

/// Add an applicant to the waitlist
#[hdk_extern]
pub fn add_to_waitlist(input: AddToWaitlistInput) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get(action_hash, GetOptions::default())? {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get(action_hash, GetOptions::default())? {
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
        if let Some(record) = get(action_hash, GetOptions::default())? {
            members.push(record);
        }
    }

    Ok(members)
}
