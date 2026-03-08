//! Space Coordinator Zome
//!
//! Business logic for private spaces within the public Commons DHT.
//! Implements the "membrane factory" concept: create_space, invite_member,
//! grant_access, revoke_access, list_spaces, list_members.
//!
//! Access control uses Holochain capability grants to restrict
//! read/write within a space to approved members only.

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use space_integrity::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create a new private space
///
/// The caller becomes the first admin member automatically.

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn create_space(input: CreateSpaceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_space")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    let space_id = format!("space:{}:{}", input.name, now.as_micros());

    let space = Space {
        id: space_id.clone(),
        name: input.name.clone(),
        space_type: input.space_type.clone(),
        description: input.description,
        creator: caller.clone(),
        max_members: input.max_members.unwrap_or(0),
        requires_approval: input.requires_approval.unwrap_or(false),
        approval_threshold: input.approval_threshold.unwrap_or(1),
        open: true,
        created_at: now,
    };

    let space_hash = create_entry(&EntryTypes::Space(space))?;

    // Link to all spaces
    create_entry(&EntryTypes::Anchor(Anchor("all_spaces".to_string())))?;
    create_link(
        anchor_hash("all_spaces")?,
        space_hash.clone(),
        LinkTypes::AllSpaces,
        (),
    )?;

    // Link to type-specific anchor
    let type_name = match &input.space_type {
        SpaceType::Family => "family",
        SpaceType::Squad => "squad",
        SpaceType::Cooperative => "cooperative",
        SpaceType::Custom(n) => n.as_str(),
    };
    let type_anchor = format!("type:{}", type_name);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        space_hash.clone(),
        LinkTypes::TypeToSpaces,
        (),
    )?;

    // Auto-create admin membership for creator
    let membership = Membership {
        space_id: space_id.clone(),
        member: caller.clone(),
        role: MemberRole::Admin,
        active: true,
        invited_by: caller.clone(),
        joined_at: now,
    };

    let member_hash = create_entry(&EntryTypes::Membership(membership))?;

    // Link space to member
    let space_anchor = format!("space:{}", space_id);
    create_entry(&EntryTypes::Anchor(Anchor(space_anchor.clone())))?;
    create_link(
        anchor_hash(&space_anchor)?,
        member_hash.clone(),
        LinkTypes::SpaceToMembers,
        (),
    )?;

    // Link agent to space
    create_link(
        AnyLinkableHash::from(caller),
        space_hash.clone(),
        LinkTypes::AgentToSpaces,
        (),
    )?;

    get_latest_record(space_hash)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Space not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSpaceInput {
    pub name: String,
    pub space_type: SpaceType,
    pub description: String,
    pub max_members: Option<u32>,
    pub requires_approval: Option<bool>,
    pub approval_threshold: Option<u32>,
}

/// Invite a member to a space
///
/// Creates an invitation. If the space doesn't require approval, the
/// invitation is automatically approved and membership is created.
#[hdk_extern]
pub fn invite_member(input: InviteMemberInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "invite_member")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is an admin or member of the space
    verify_membership(&input.space_id, &caller, true)?;

    // Check if space is open
    let space = get_space_by_id(&input.space_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Space not found".into())))?;

    if !space.open {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Space is not accepting new members".into()
        )));
    }

    // Check max members
    if space.max_members > 0 {
        let member_count = count_active_members(&input.space_id)?;
        if member_count >= space.max_members as usize {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Space has reached maximum member count".into()
            )));
        }
    }

    if space.requires_approval {
        // Create pending invitation
        let invitation = SpaceInvitation {
            space_id: input.space_id.clone(),
            invitee: input.invitee.clone(),
            inviter: caller,
            message: input.message.unwrap_or_default(),
            approvals: 0,
            approved_by: Vec::new(),
            status: InvitationStatus::Pending,
            created_at: now,
        };

        let inv_hash = create_entry(&EntryTypes::SpaceInvitation(invitation))?;

        let space_anchor = format!("space:{}", input.space_id);
        create_link(
            anchor_hash(&space_anchor)?,
            inv_hash.clone(),
            LinkTypes::SpaceToInvitations,
            (),
        )?;

        get_latest_record(inv_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invitation not found".into()
        )))
    } else {
        // Direct add (no approval needed)
        add_member_directly(&input.space_id, &input.invitee, &caller, now)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InviteMemberInput {
    pub space_id: String,
    pub invitee: AgentPubKey,
    pub message: Option<String>,
}

/// Grant capability access to a space member
///
/// Creates a SpaceCapability entry that authorizes the grantee
/// to call specific functions within the space context.
#[hdk_extern]
pub fn grant_access(input: GrantAccessInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "grant_access")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Only admins can grant capabilities
    verify_admin(&input.space_id, &caller)?;

    let cap = SpaceCapability {
        space_id: input.space_id.clone(),
        grantee: input.grantee,
        allowed_functions: input.allowed_functions,
        expires_at: input.expires_at,
        revoked: false,
        granted_at: now,
    };

    let cap_hash = create_entry(&EntryTypes::SpaceCapability(cap))?;

    let space_anchor = format!("space:{}", input.space_id);
    create_link(
        anchor_hash(&space_anchor)?,
        cap_hash.clone(),
        LinkTypes::SpaceToCapabilities,
        (),
    )?;

    get_latest_record(cap_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Capability not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GrantAccessInput {
    pub space_id: String,
    pub grantee: AgentPubKey,
    pub allowed_functions: Vec<String>,
    pub expires_at: Option<Timestamp>,
}

/// Revoke a member's access to a space
///
/// Deactivates the membership and marks any capabilities as revoked.
#[hdk_extern]
pub fn revoke_access(input: RevokeAccessInput) -> ExternResult<bool> {
    require_consciousness(&requirement_for_proposal(), "revoke_access")?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Only admins can revoke access
    verify_admin(&input.space_id, &caller)?;

    // Find and deactivate membership
    let members = get_space_member_records(&input.space_id)?;
    for (record, mut membership) in members {
        if membership.member == input.member && membership.active {
            // Cannot revoke the last admin
            if membership.role == MemberRole::Admin {
                let admin_count = count_admins(&input.space_id)?;
                if admin_count <= 1 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot revoke the last admin".into()
                    )));
                }
            }

            membership.active = false;
            update_entry(
                record.action_address().clone(),
                &EntryTypes::Membership(membership),
            )?;
        }
    }

    // Mark capabilities as revoked
    let caps = get_space_capability_records(&input.space_id)?;
    for (record, mut cap) in caps {
        if cap.grantee == input.member && !cap.revoked {
            cap.revoked = true;
            update_entry(
                record.action_address().clone(),
                &EntryTypes::SpaceCapability(cap),
            )?;
        }
    }

    Ok(true)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeAccessInput {
    pub space_id: String,
    pub member: AgentPubKey,
}

/// List all spaces the calling agent belongs to
#[hdk_extern]
pub fn get_my_spaces(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        LinkQuery::try_new(AnyLinkableHash::from(caller), LinkTypes::AgentToSpaces)?,
        GetStrategy::default(),
    )?;

    let mut spaces = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            spaces.push(record);
        }
    }

    Ok(spaces)
}

/// List members of a space
///
/// Only accessible to space members.
#[hdk_extern]
pub fn get_space_members(space_id: String) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    verify_membership(&space_id, &caller, false)?;

    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToMembers)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            members.push(record);
        }
    }

    Ok(members)
}

/// List all spaces of a given type
#[hdk_extern]
pub fn get_spaces_by_type(space_type: SpaceType) -> ExternResult<Vec<Record>> {
    let type_name = match &space_type {
        SpaceType::Family => "family",
        SpaceType::Squad => "squad",
        SpaceType::Cooperative => "cooperative",
        SpaceType::Custom(n) => n.as_str(),
    };
    let type_anchor = format!("type:{}", type_name);
    let anchor = match anchor_hash(&type_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::TypeToSpaces)?,
        GetStrategy::default(),
    )?;

    let mut spaces = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            spaces.push(record);
        }
    }

    Ok(spaces)
}

/// Get a space by ID
#[hdk_extern]
pub fn get_space(space_id: String) -> ExternResult<Option<Record>> {
    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::AllSpaces)?,
        GetStrategy::default(),
    )?;

    // Fallback: check all_spaces anchor
    if links.is_empty() {
        let all_links = get_links(
            LinkQuery::try_new(anchor_hash("all_spaces")?, LinkTypes::AllSpaces)?,
            GetStrategy::default(),
        )?;

        for link in all_links {
            let ah = ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            if let Some(record) = get_latest_record(ah)? {
                if let Some(space) = record.entry().to_app_option::<Space>().ok().flatten() {
                    if space.id == space_id {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Approve a pending invitation
///
/// Increments the approval count. If threshold is met, auto-creates membership.
#[hdk_extern]
pub fn approve_invitation(input: ApproveInvitationInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "approve_invitation")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is an active member of the space
    let record = get(input.invitation_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Invitation not found".into())
    ))?;

    let mut invitation: SpaceInvitation = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse invitation".into()
        )))?;

    if invitation.status != InvitationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invitation is not pending".into()
        )));
    }

    verify_membership(&invitation.space_id, &caller, true)?;

    // Prevent double-approval
    if invitation.approved_by.contains(&caller) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already approved this invitation".into()
        )));
    }

    invitation.approvals += 1;
    invitation.approved_by.push(caller.clone());

    // Check if threshold is met
    let space = get_space_by_id(&invitation.space_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Space not found".into())))?;

    if invitation.approvals >= space.approval_threshold {
        invitation.status = InvitationStatus::Approved;
        update_entry(
            input.invitation_hash,
            &EntryTypes::SpaceInvitation(invitation.clone()),
        )?;

        // Auto-create membership
        return add_member_directly(
            &invitation.space_id,
            &invitation.invitee,
            &invitation.inviter,
            now,
        );
    }

    let new_hash = update_entry(
        input.invitation_hash,
        &EntryTypes::SpaceInvitation(invitation),
    )?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated invitation".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveInvitationInput {
    pub invitation_hash: ActionHash,
}

/// Reject a pending invitation
#[hdk_extern]
pub fn reject_invitation(invitation_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "reject_invitation")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get_latest_record(invitation_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Invitation not found".into())
    ))?;

    let mut invitation: SpaceInvitation = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse invitation".into()
        )))?;

    if invitation.status != InvitationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invitation is not pending".into()
        )));
    }

    verify_membership(&invitation.space_id, &caller, true)?;

    invitation.status = InvitationStatus::Rejected;
    let new_hash = update_entry(invitation_hash, &EntryTypes::SpaceInvitation(invitation))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated invitation".into()
    )))
}

/// Book a shared resource within a space
#[hdk_extern]
pub fn book_resource(input: BookResourceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "book_resource")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    verify_membership(&input.space_id, &caller, true)?;

    let booking_id = format!(
        "booking:{}:{}:{}",
        input.space_id,
        input.resource_name,
        now.as_micros()
    );

    let booking = ResourceBooking {
        id: booking_id,
        space_id: input.space_id.clone(),
        resource_name: input.resource_name,
        booked_by: caller.clone(),
        start_time: input.start_time,
        end_time: input.end_time,
        status: BookingStatus::Pending,
        notes: input.notes.unwrap_or_default(),
        created_at: now,
    };

    let booking_hash = create_entry(&EntryTypes::ResourceBooking(booking))?;

    // Link space to booking
    let space_anchor = format!("space:{}", input.space_id);
    create_link(
        anchor_hash(&space_anchor)?,
        booking_hash.clone(),
        LinkTypes::SpaceToBookings,
        (),
    )?;

    // Link agent to booking
    create_link(
        AnyLinkableHash::from(caller),
        booking_hash.clone(),
        LinkTypes::AgentToBookings,
        (),
    )?;

    get_latest_record(booking_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Booking not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BookResourceInput {
    pub space_id: String,
    pub resource_name: String,
    pub start_time: u64,
    pub end_time: u64,
    pub notes: Option<String>,
}

/// Cancel a resource booking
#[hdk_extern]
pub fn cancel_booking(booking_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "cancel_booking")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get_latest_record(booking_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Booking not found".into())
    ))?;

    let mut booking: ResourceBooking = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".into()
        )))?;

    if booking.booked_by != caller {
        // Allow admins to cancel too
        verify_admin(&booking.space_id, &caller)?;
    }

    if booking.status == BookingStatus::Cancelled {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Booking is already cancelled".into()
        )));
    }

    booking.status = BookingStatus::Cancelled;
    let new_hash = update_entry(booking_hash, &EntryTypes::ResourceBooking(booking))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated booking".into()
    )))
}

/// Get all bookings for a space
#[hdk_extern]
pub fn get_space_bookings(space_id: String) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    verify_membership(&space_id, &caller, false)?;

    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToBookings)?,
        GetStrategy::default(),
    )?;

    let mut bookings = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            bookings.push(record);
        }
    }

    Ok(bookings)
}

/// Create a recurring schedule/event for a space
#[hdk_extern]
pub fn create_schedule(input: CreateScheduleInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_schedule")?;
    let now = sys_time()?;
    let caller = agent_info()?.agent_initial_pubkey;

    verify_membership(&input.space_id, &caller, true)?;

    let schedule_id = format!(
        "sched:{}:{}:{}",
        input.space_id,
        input.title,
        now.as_micros()
    );

    let schedule = SpaceSchedule {
        id: schedule_id,
        space_id: input.space_id.clone(),
        title: input.title,
        description: input.description.unwrap_or_default(),
        recurrence: input.recurrence,
        next_occurrence: input.next_occurrence,
        duration_minutes: input.duration_minutes,
        creator: caller,
        created_at: now,
    };

    let sched_hash = create_entry(&EntryTypes::SpaceSchedule(schedule))?;

    // Link space to schedule
    let space_anchor = format!("space:{}", input.space_id);
    create_link(
        anchor_hash(&space_anchor)?,
        sched_hash.clone(),
        LinkTypes::SpaceToSchedules,
        (),
    )?;

    get_latest_record(sched_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Schedule not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateScheduleInput {
    pub space_id: String,
    pub title: String,
    pub description: Option<String>,
    pub recurrence: Recurrence,
    pub next_occurrence: u64,
    pub duration_minutes: u32,
}

/// Get all schedules for a space
#[hdk_extern]
pub fn get_space_schedules(space_id: String) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    verify_membership(&space_id, &caller, false)?;

    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToSchedules)?,
        GetStrategy::default(),
    )?;

    let mut schedules = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            schedules.push(record);
        }
    }

    Ok(schedules)
}

/// Check whether an agent has a valid (non-expired, non-revoked) capability for a function
#[hdk_extern]
pub fn check_capability(input: CheckCapabilityInput) -> ExternResult<bool> {
    let now = sys_time()?;
    let caps = get_space_capability_records(&input.space_id)?;

    for (_record, cap) in caps {
        if cap.grantee != input.agent {
            continue;
        }
        if cap.revoked {
            continue;
        }
        if let Some(expires) = cap.expires_at {
            if expires < now {
                continue;
            }
        }
        if cap.allowed_functions.contains(&input.function_name) {
            return Ok(true);
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckCapabilityInput {
    pub space_id: String,
    pub agent: AgentPubKey,
    pub function_name: String,
}

/// Get pending invitations for a space
#[hdk_extern]
pub fn get_space_invitations(space_id: String) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    verify_membership(&space_id, &caller, true)?;

    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToInvitations)?,
        GetStrategy::default(),
    )?;

    let mut invitations = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            invitations.push(record);
        }
    }

    Ok(invitations)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Verify that an agent is a member of a space
fn verify_membership(
    space_id: &str,
    agent: &AgentPubKey,
    require_active: bool,
) -> ExternResult<()> {
    let members = get_space_member_records(space_id)?;
    for (_record, membership) in &members {
        if membership.member == *agent {
            if require_active && !membership.active {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Membership is not active".into()
                )));
            }
            return Ok(());
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Not a member of this space".into()
    )))
}

/// Verify that an agent is an admin of a space
fn verify_admin(space_id: &str, agent: &AgentPubKey) -> ExternResult<()> {
    let members = get_space_member_records(space_id)?;
    for (_record, membership) in &members {
        if membership.member == *agent && membership.active && membership.role == MemberRole::Admin
        {
            return Ok(());
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Not an admin of this space".into()
    )))
}

/// Get all member records for a space
fn get_space_member_records(space_id: &str) -> ExternResult<Vec<(Record, Membership)>> {
    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToMembers)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            if let Some(membership) = record.entry().to_app_option::<Membership>().ok().flatten() {
                members.push((record, membership));
            }
        }
    }

    Ok(members)
}

/// Get all capability records for a space
fn get_space_capability_records(space_id: &str) -> ExternResult<Vec<(Record, SpaceCapability)>> {
    let space_anchor = format!("space:{}", space_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&space_anchor)?, LinkTypes::SpaceToCapabilities)?,
        GetStrategy::default(),
    )?;

    let mut caps = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            if let Some(cap) = record
                .entry()
                .to_app_option::<SpaceCapability>()
                .ok()
                .flatten()
            {
                caps.push((record, cap));
            }
        }
    }

    Ok(caps)
}

/// Count active members in a space
fn count_active_members(space_id: &str) -> ExternResult<usize> {
    let members = get_space_member_records(space_id)?;
    Ok(members.iter().filter(|(_, m)| m.active).count())
}

/// Count admin members in a space
fn count_admins(space_id: &str) -> ExternResult<usize> {
    let members = get_space_member_records(space_id)?;
    Ok(members
        .iter()
        .filter(|(_, m)| m.active && m.role == MemberRole::Admin)
        .count())
}

/// Get space entry by ID
fn get_space_by_id(space_id: &str) -> ExternResult<Option<Space>> {
    let all_links = get_links(
        LinkQuery::try_new(anchor_hash("all_spaces")?, LinkTypes::AllSpaces)?,
        GetStrategy::default(),
    )?;

    for link in all_links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah)? {
            if let Some(space) = record.entry().to_app_option::<Space>().ok().flatten() {
                if space.id == space_id {
                    return Ok(Some(space));
                }
            }
        }
    }

    Ok(None)
}

/// Add a member directly (bypassing approval flow)
fn add_member_directly(
    space_id: &str,
    invitee: &AgentPubKey,
    inviter: &AgentPubKey,
    now: Timestamp,
) -> ExternResult<Record> {
    let membership = Membership {
        space_id: space_id.to_string(),
        member: invitee.clone(),
        role: MemberRole::Member,
        active: true,
        invited_by: inviter.clone(),
        joined_at: now,
    };

    let member_hash = create_entry(&EntryTypes::Membership(membership))?;

    // Link space to member
    let space_anchor = format!("space:{}", space_id);
    create_link(
        anchor_hash(&space_anchor)?,
        member_hash.clone(),
        LinkTypes::SpaceToMembers,
        (),
    )?;

    // Link agent to space
    // Find space action hash from all_spaces anchor
    let all_links = get_links(
        LinkQuery::try_new(anchor_hash("all_spaces")?, LinkTypes::AllSpaces)?,
        GetStrategy::default(),
    )?;

    for link in all_links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(ah.clone())? {
            if let Some(space) = record.entry().to_app_option::<Space>().ok().flatten() {
                if space.id == space_id {
                    create_link(
                        AnyLinkableHash::from(invitee.clone()),
                        ah,
                        LinkTypes::AgentToSpaces,
                        (),
                    )?;
                    break;
                }
            }
        }
    }

    get_latest_record(member_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Member not found".into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_space_input_serde() {
        let input = CreateSpaceInput {
            name: "Test Family".into(),
            space_type: SpaceType::Family,
            description: "A test family space".into(),
            max_members: Some(10),
            requires_approval: Some(true),
            approval_threshold: Some(2),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateSpaceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Test Family");
        assert_eq!(back.max_members, Some(10));
    }

    #[test]
    fn invite_member_input_serde() {
        let input = InviteMemberInput {
            space_id: "space:test:123".into(),
            invitee: AgentPubKey::from_raw_36(vec![0u8; 36]),
            message: Some("Welcome!".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("space:test:123"));
    }

    #[test]
    fn grant_access_input_serde() {
        let input = GrantAccessInput {
            space_id: "space:test:123".into(),
            grantee: AgentPubKey::from_raw_36(vec![0u8; 36]),
            allowed_functions: vec!["read_data".into(), "write_data".into()],
            expires_at: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("read_data"));
    }

    #[test]
    fn revoke_access_input_serde() {
        let input = RevokeAccessInput {
            space_id: "space:test:123".into(),
            member: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("space:test:123"));
    }

    #[test]
    fn approve_invitation_input_serde() {
        let input = ApproveInvitationInput {
            invitation_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ApproveInvitationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input.invitation_hash, back.invitation_hash);
    }

    #[test]
    fn book_resource_input_serde() {
        let input = BookResourceInput {
            space_id: "space:test:123".into(),
            resource_name: "Meeting Room A".into(),
            start_time: 1000000,
            end_time: 2000000,
            notes: Some("Team standup".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: BookResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.resource_name, "Meeting Room A");
        assert_eq!(back.start_time, 1000000);
        assert_eq!(back.end_time, 2000000);
    }

    #[test]
    fn book_resource_input_no_notes_serde() {
        let input = BookResourceInput {
            space_id: "space:test:123".into(),
            resource_name: "Projector".into(),
            start_time: 500,
            end_time: 1000,
            notes: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: BookResourceInput = serde_json::from_str(&json).unwrap();
        assert!(back.notes.is_none());
    }

    #[test]
    fn create_schedule_input_serde() {
        let input = CreateScheduleInput {
            space_id: "space:test:123".into(),
            title: "Weekly Sync".into(),
            description: Some("Team sync meeting".into()),
            recurrence: Recurrence::Weekly,
            next_occurrence: 1000000,
            duration_minutes: 60,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateScheduleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.title, "Weekly Sync");
        assert_eq!(back.recurrence, Recurrence::Weekly);
        assert_eq!(back.duration_minutes, 60);
    }

    #[test]
    fn create_schedule_input_no_description_serde() {
        let input = CreateScheduleInput {
            space_id: "space:test:123".into(),
            title: "Daily Standup".into(),
            description: None,
            recurrence: Recurrence::Daily,
            next_occurrence: 500000,
            duration_minutes: 15,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateScheduleInput = serde_json::from_str(&json).unwrap();
        assert!(back.description.is_none());
        assert_eq!(back.recurrence, Recurrence::Daily);
    }

    #[test]
    fn check_capability_input_serde() {
        let input = CheckCapabilityInput {
            space_id: "space:test:123".into(),
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            function_name: "read_data".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CheckCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.function_name, "read_data");
    }
}
