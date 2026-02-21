//! Hearth Kinship Coordinator Zome
//!
//! CRUD operations and business logic for hearth membership and kinship bonds.
//! This is the CORE membership and relationship zome for the Hearth cluster.

use hdk::prelude::*;
use hearth_kinship_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateHearthInput {
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub max_members: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InviteMemberInput {
    pub hearth_hash: ActionHash,
    pub invitee_agent: AgentPubKey,
    pub proposed_role: MemberRole,
    pub message: String,
    pub expires_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AcceptInvitationInput {
    pub invitation_hash: ActionHash,
    pub display_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateMemberRoleInput {
    pub membership_hash: ActionHash,
    pub new_role: MemberRole,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateBondInput {
    pub hearth_hash: ActionHash,
    pub member_b: AgentPubKey,
    pub bond_type: BondType,
    pub initial_strength_bp: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TendBondInput {
    pub bond_hash: ActionHash,
    pub description: String,
    pub quality_bp: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetBondHealthInput {
    pub bond_hash: ActionHash,
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute the entry hash for an anchor string.
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Collect records from a list of links (resolving each target ActionHash).
fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Extract a typed entry from a Record.
fn entry_from_record<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    record: &Record,
    type_name: &str,
) -> ExternResult<T> {
    record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid {} entry",
                type_name
            )))
        })
}

/// Verify the caller has a guardian-level role (Founder, Elder, or Adult)
/// within the specified hearth. Returns the caller's membership record.
fn require_guardian_role(hearth_hash: &ActionHash) -> ExternResult<HearthMembership> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(hearth_hash.clone(), LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.agent == agent
                && membership.status == MembershipStatus::Active
                && membership.role.is_guardian()
            {
                return Ok(membership);
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Caller does not have guardian-level role (Founder, Elder, or Adult) in this hearth".into()
    )))
}

// ============================================================================
// Hearth CRUD
// ============================================================================

/// Create a new hearth and automatically add the creator as a Founder member.
///
/// Creates links: AllHearths, AgentToHearths, HearthToMembers, TypeToHearths.
#[hdk_extern]
pub fn create_hearth(input: CreateHearthInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let max_members = input.max_members.unwrap_or(10);

    let hearth = Hearth {
        name: input.name,
        description: input.description,
        hearth_type: input.hearth_type.clone(),
        created_by: agent.clone(),
        created_at: now,
        max_members,
    };

    let hearth_hash = create_entry(&EntryTypes::Hearth(hearth.clone()))?;

    // Auto-create founder membership
    let membership = HearthMembership {
        hearth_hash: hearth_hash.clone(),
        agent: agent.clone(),
        role: MemberRole::Founder,
        status: MembershipStatus::Active,
        display_name: "Founder".into(),
        joined_at: now,
    };
    let membership_hash = create_entry(&EntryTypes::HearthMembership(membership))?;

    // Link: AllHearths anchor -> Hearth
    create_entry(&EntryTypes::Anchor(Anchor("all_hearths".to_string())))?;
    create_link(
        anchor_hash("all_hearths")?,
        hearth_hash.clone(),
        LinkTypes::AllHearths,
        (),
    )?;

    // Link: Agent -> Hearth
    create_link(
        agent.clone(),
        hearth_hash.clone(),
        LinkTypes::AgentToHearths,
        (),
    )?;

    // Link: Hearth -> Membership
    create_link(
        hearth_hash.clone(),
        membership_hash,
        LinkTypes::HearthToMembers,
        (),
    )?;

    // Link: TypeToHearths anchor -> Hearth
    let type_anchor = format!("hearth_type:{:?}", input.hearth_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        hearth_hash.clone(),
        LinkTypes::TypeToHearths,
        (),
    )?;

    // H4: Evaluate auto social recovery (will fire when >= 3 guardians exist)
    let _ = propose_auto_recovery(&hearth_hash);

    get(hearth_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created hearth".into()
    )))
}

// ============================================================================
// Invitation Flow
// ============================================================================

/// Invite a member to a hearth. Caller must be Founder, Elder, or Adult.
#[hdk_extern]
pub fn invite_member(input: InviteMemberInput) -> ExternResult<Record> {
    // Validate caller has guardian role
    require_guardian_role(&input.hearth_hash)?;

    let agent = agent_info()?.agent_initial_pubkey;

    let invitation = HearthInvitation {
        hearth_hash: input.hearth_hash.clone(),
        inviter: agent,
        invitee_agent: input.invitee_agent.clone(),
        proposed_role: input.proposed_role,
        message: input.message,
        expires_at: input.expires_at,
        status: InvitationStatus::Pending,
    };

    let invitation_hash = create_entry(&EntryTypes::HearthInvitation(invitation))?;

    // Link: Hearth -> Invitation
    create_link(
        input.hearth_hash,
        invitation_hash.clone(),
        LinkTypes::HearthToInvitations,
        (),
    )?;

    // Link: Invitee Agent -> Invitation
    create_link(
        input.invitee_agent,
        invitation_hash.clone(),
        LinkTypes::AgentToInvitations,
        (),
    )?;

    get(invitation_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created invitation".into()
    )))
}

/// Accept an invitation. Creates a membership and updates invitation status.
#[hdk_extern]
pub fn accept_invitation(input: AcceptInvitationInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Retrieve the invitation
    let invitation_record = get(input.invitation_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Invitation not found".into())),
    )?;
    let invitation: HearthInvitation = entry_from_record(&invitation_record, "HearthInvitation")?;

    // Validate status
    if invitation.status != InvitationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invitation is not Pending, current status: {:?}",
            invitation.status
        ))));
    }

    // Check invitation hasn't expired
    if invitation.expires_at < now {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invitation has expired".into()
        )));
    }

    // Validate the caller is the invitee
    if invitation.invitee_agent != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the invitee can accept this invitation".into()
        )));
    }

    // Capture values before invitation is moved by struct update syntax
    let hearth_hash = invitation.hearth_hash.clone();
    let proposed_role = invitation.proposed_role.clone();

    // Create membership
    let membership = HearthMembership {
        hearth_hash: hearth_hash.clone(),
        agent: agent.clone(),
        role: proposed_role.clone(),
        status: MembershipStatus::Active,
        display_name: input.display_name,
        joined_at: now,
    };
    let membership_hash = create_entry(&EntryTypes::HearthMembership(membership))?;

    // Link: Agent -> Hearth
    create_link(
        agent.clone(),
        hearth_hash.clone(),
        LinkTypes::AgentToHearths,
        (),
    )?;

    // Link: Hearth -> Membership
    create_link(
        hearth_hash.clone(),
        membership_hash.clone(),
        LinkTypes::HearthToMembers,
        (),
    )?;

    // Update invitation status to Accepted
    let updated_invitation = HearthInvitation {
        status: InvitationStatus::Accepted,
        ..invitation
    };
    update_entry(
        input.invitation_hash,
        &EntryTypes::HearthInvitation(updated_invitation),
    )?;

    // Emit signal
    emit_signal(&HearthSignal::MemberJoined {
        hearth_hash: hearth_hash.clone(),
        agent: agent.clone(),
        role: proposed_role.clone(),
    })?;

    // H4: Re-evaluate auto social recovery if the new member is a guardian
    if proposed_role.is_guardian() {
        let _ = propose_auto_recovery(&hearth_hash);
    }

    get(membership_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created membership".into()
    )))
}

/// Decline an invitation. Updates the invitation status to Declined.
#[hdk_extern]
pub fn decline_invitation(invitation_hash: ActionHash) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;

    let invitation_record = get(invitation_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Invitation not found".into())),
    )?;
    let invitation: HearthInvitation = entry_from_record(&invitation_record, "HearthInvitation")?;

    // Validate the caller is the invitee
    if invitation.invitee_agent != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the invitee can decline this invitation".into()
        )));
    }

    if invitation.status != InvitationStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invitation is not Pending, current status: {:?}",
            invitation.status
        ))));
    }

    let updated_invitation = HearthInvitation {
        status: InvitationStatus::Declined,
        ..invitation
    };
    let new_hash = update_entry(
        invitation_hash,
        &EntryTypes::HearthInvitation(updated_invitation),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated invitation".into()
    )))
}

// ============================================================================
// Membership Management
// ============================================================================

/// Leave a hearth. Updates membership status to Departed.
/// Validates the departing member is not the last Founder.
#[hdk_extern]
pub fn leave_hearth(membership_hash: ActionHash) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;

    let record = get(membership_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Membership not found".into())
    ))?;
    let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;

    // Validate the caller is the member
    if membership.agent != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the member can leave a hearth".into()
        )));
    }

    if membership.status != MembershipStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Membership is not Active".into()
        )));
    }

    // If the departing member is a Founder, check that they are not the last one
    if membership.role == MemberRole::Founder {
        let links = get_links(
            LinkQuery::try_new(membership.hearth_hash.clone(), LinkTypes::HearthToMembers)?,
            GetStrategy::default(),
        )?;
        let mut founder_count = 0u32;
        for link in links {
            let target = ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            if let Some(member_record) = get(target, GetOptions::default())? {
                let m: HearthMembership = entry_from_record(&member_record, "HearthMembership")?;
                if m.role == MemberRole::Founder && m.status == MembershipStatus::Active {
                    founder_count += 1;
                }
            }
        }
        if founder_count <= 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot leave: you are the last Founder. Transfer Founder role first.".into()
            )));
        }
    }

    let updated = HearthMembership {
        status: MembershipStatus::Departed,
        ..membership.clone()
    };
    let new_hash = update_entry(membership_hash, &EntryTypes::HearthMembership(updated))?;

    // Emit signal
    emit_signal(&HearthSignal::MemberDeparted {
        hearth_hash: membership.hearth_hash,
        agent: agent.clone(),
    })?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated membership".into()
    )))
}

/// Update a member's role. Caller must be Founder or Elder.
#[hdk_extern]
pub fn update_member_role(input: UpdateMemberRoleInput) -> ExternResult<Record> {
    let record = get(input.membership_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Membership not found".into())
    ))?;
    let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;

    // Validate caller has Founder or Elder role
    let caller_membership = require_guardian_role(&membership.hearth_hash)?;
    if caller_membership.role != MemberRole::Founder && caller_membership.role != MemberRole::Elder
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Founders and Elders can update member roles".into()
        )));
    }

    let updated = HearthMembership {
        role: input.new_role,
        ..membership
    };
    let new_hash = update_entry(
        input.membership_hash,
        &EntryTypes::HearthMembership(updated),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated membership".into()
    )))
}

// ============================================================================
// Kinship Bonds
// ============================================================================

/// Create a kinship bond between the caller and another member.
/// Validates that both the caller and member_b are active hearth members.
#[hdk_extern]
pub fn create_kinship_bond(input: CreateBondInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let initial_strength = input.initial_strength_bp.unwrap_or(BOND_BASE_FAMILY);

    // Verify both caller and member_b are active members of this hearth
    let member_links = get_links(
        LinkQuery::try_new(input.hearth_hash.clone(), LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;
    let mut caller_is_member = false;
    let mut member_b_is_member = false;
    for link in member_links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.status == MembershipStatus::Active {
                if membership.agent == agent {
                    caller_is_member = true;
                }
                if membership.agent == input.member_b {
                    member_b_is_member = true;
                }
            }
        }
        if caller_is_member && member_b_is_member {
            break;
        }
    }
    if !caller_is_member {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Caller is not an active member of this hearth".into()
        )));
    }
    if !member_b_is_member {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "member_b is not an active member of this hearth".into()
        )));
    }

    let bond = KinshipBond {
        hearth_hash: input.hearth_hash.clone(),
        member_a: agent.clone(),
        member_b: input.member_b.clone(),
        bond_type: input.bond_type,
        strength_bp: initial_strength,
        last_tended: now,
        created_at: now,
    };

    let bond_hash = create_entry(&EntryTypes::KinshipBond(bond))?;

    // Link: Hearth -> Bond
    create_link(
        input.hearth_hash,
        bond_hash.clone(),
        LinkTypes::HearthToBonds,
        (),
    )?;

    // Link: member_a -> Bond
    create_link(
        agent.clone(),
        bond_hash.clone(),
        LinkTypes::MemberToBonds,
        (),
    )?;

    // Link: member_b -> Bond
    create_link(
        input.member_b,
        bond_hash.clone(),
        LinkTypes::MemberToBonds,
        (),
    )?;

    get(bond_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created bond".into()
    )))
}

/// Tend a bond — compute current decayed strength, blend with interaction
/// quality, update strength and last_tended timestamp, emit BondTended signal.
///
/// The new bond strength is a weighted blend of current (decayed) health and
/// the quality of the tending interaction: `70% current + 30% quality`.
/// High-quality interactions (quality_bp near 10000) pull neglected bonds upward;
/// low-quality interactions slow recovery.
#[hdk_extern]
pub fn tend_bond(input: TendBondInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = get(input.bond_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Bond not found".into())))?;
    let bond: KinshipBond = entry_from_record(&record, "KinshipBond")?;

    // Compute days since last tended
    let now_micros: i64 = now.as_micros();
    let last_tended_micros: i64 = bond.last_tended.as_micros();
    let elapsed_micros: u64 = if now_micros > last_tended_micros {
        (now_micros - last_tended_micros) as u64
    } else {
        0u64
    };
    let micros_per_day: u64 = 86_400_000_000;
    let days_inactive = (elapsed_micros / micros_per_day) as u32;

    // Compute current decayed strength
    let current_health = decayed_strength(bond.strength_bp, days_inactive);

    // Blend: 70% current decayed health + 30% interaction quality
    let quality_bp = input.quality_bp.min(BOND_MAX);
    let new_strength = ((current_health as u64 * 7 + quality_bp as u64 * 3) / 10) as u32;
    let new_strength = new_strength.clamp(BOND_MIN, BOND_MAX);

    let updated = KinshipBond {
        strength_bp: new_strength,
        last_tended: now,
        ..bond.clone()
    };
    let new_hash = update_entry(input.bond_hash, &EntryTypes::KinshipBond(updated))?;

    // Emit BondTended signal
    emit_signal(&HearthSignal::BondTended {
        member_a: bond.member_a,
        member_b: bond.member_b,
        quality_bp: input.quality_bp,
    })?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated bond".into()
    )))
}

/// Get the current health (decayed strength) of a bond in basis points.
///
/// Uses the deterministic integer lookup table from hearth-types (H1)
/// to compute decay based on days since last tended.
#[hdk_extern]
pub fn get_bond_health(input: GetBondHealthInput) -> ExternResult<u32> {
    let now = sys_time()?;

    let record = get(input.bond_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Bond not found".into())))?;
    let bond: KinshipBond = entry_from_record(&record, "KinshipBond")?;

    // Calculate days inactive using integer microsecond math.
    // Timestamp::as_micros() returns i64; handle clock skew gracefully.
    let now_micros: i64 = now.as_micros();
    let last_tended_micros: i64 = bond.last_tended.as_micros();
    let elapsed_micros: u64 = if now_micros > last_tended_micros {
        (now_micros - last_tended_micros) as u64
    } else {
        0u64
    };
    let micros_per_day: u64 = 86_400_000_000;
    let days_inactive = (elapsed_micros / micros_per_day) as u32;

    Ok(decayed_strength(bond.strength_bp, days_inactive))
}

// ============================================================================
// Weekly Digest (H2 Epoch Rollups)
// ============================================================================

/// Create a weekly digest entry and link it to the hearth.
#[hdk_extern]
pub fn create_weekly_digest(input: WeeklyDigest) -> ExternResult<Record> {
    let hearth_hash = input.hearth_hash.clone();

    let action_hash = create_entry(&EntryTypes::WeeklyDigest(input))?;

    // Link: Hearth -> Digest
    create_link(
        hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToDigests,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created weekly digest".into()
    )))
}

/// Get all weekly digests for a hearth.
#[hdk_extern]
pub fn get_weekly_digests(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToDigests)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// Authorization Helpers (cross-zome callable)
// ============================================================================

/// Check if the calling agent has a guardian-level role in the specified hearth.
/// Returns true if the caller is an active Founder, Elder, or Adult.
/// Designed for cross-zome calls from other coordinators in the same DNA.
#[hdk_extern]
pub fn is_guardian(hearth_hash: ActionHash) -> ExternResult<bool> {
    match require_guardian_role(&hearth_hash) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Get the caller's vote weight in basis points for a given hearth.
/// Returns the role-based default (e.g. Adult=10000, Youth=5000, Child=0).
/// Returns 0 if the caller is not an active member.
#[hdk_extern]
pub fn get_caller_vote_weight(hearth_hash: ActionHash) -> ExternResult<u32> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.agent == agent && membership.status == MembershipStatus::Active {
                return Ok(membership.role.default_vote_weight_bp());
            }
        }
    }

    Ok(0)
}

/// Get the caller's role in a given hearth.
/// Returns None if the caller is not an active member.
/// Used by decisions zome to check eligible_roles and derive vote weight.
#[hdk_extern]
pub fn get_caller_role(hearth_hash: ActionHash) -> ExternResult<Option<MemberRole>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.agent == agent && membership.status == MembershipStatus::Active {
                return Ok(Some(membership.role));
            }
        }
    }

    Ok(None)
}

/// Get the count of active members in a hearth.
/// Used by decisions zome for participation rate calculation.
#[hdk_extern]
pub fn get_active_member_count(hearth_hash: ActionHash) -> ExternResult<u32> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;

    let mut count: u32 = 0;
    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.status == MembershipStatus::Active {
                count += 1;
            }
        }
    }

    Ok(count)
}

/// Compute the recovery threshold: 60% of adult_count, rounded up.
fn recovery_threshold(adult_count: usize) -> usize {
    (adult_count * 60).div_ceil(100)
}

/// H4: Propose auto social recovery if the hearth has >= 3 adult-level members.
/// Cross-cluster call to identity cluster is best-effort (don't block on failure).
fn propose_auto_recovery(hearth_hash: &ActionHash) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash.clone(), LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;

    let mut adult_agents: Vec<AgentPubKey> = Vec::new();
    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(target, GetOptions::default())? {
            let membership: HearthMembership = entry_from_record(&record, "HearthMembership")?;
            if membership.status == MembershipStatus::Active && membership.role.is_guardian() {
                adult_agents.push(membership.agent);
            }
        }
    }

    // Need at least 3 adults for social recovery quorum
    if adult_agents.len() < 3 {
        return Ok(());
    }

    // Compute threshold: 60% rounded up
    let threshold = recovery_threshold(adult_agents.len());

    // Best-effort cross-cluster call to identity recovery
    #[derive(Serialize, Debug)]
    struct SetupRecoveryInput {
        trustees: Vec<AgentPubKey>,
        threshold: usize,
    }

    let recovery_input = SetupRecoveryInput {
        trustees: adult_agents,
        threshold,
    };

    // Best-effort: don't block hearth operations on recovery setup failure
    if let Err(e) = call(
        CallTargetCell::OtherRole(RoleName::from("identity")),
        ZomeName::new("recovery"),
        FunctionName::new("setup_recovery"),
        None,
        recovery_input,
    ) {
        let _ = emit_signal(&HearthSignal::CrossZomeCallFailed {
            zome: "recovery".into(),
            function: "setup_recovery".into(),
            error: format!("{e:?}"),
        });
    }

    Ok(())
}

// ============================================================================
// Query Functions
// ============================================================================

/// Get all members of a hearth.
#[hdk_extern]
pub fn get_hearth_members(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMembers)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all hearths the calling agent belongs to.
#[hdk_extern]
pub fn get_my_hearths(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToHearths)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get the full kinship graph (all bonds) for a hearth.
#[hdk_extern]
pub fn get_kinship_graph(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToBonds)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all bonds in a hearth where the decayed strength is below 3000 bp (neglected).
#[hdk_extern]
pub fn get_neglected_bonds(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let now = sys_time()?;
    let now_micros: i64 = now.as_micros();
    let micros_per_day: u64 = 86_400_000_000;

    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToBonds)?,
        GetStrategy::default(),
    )?;

    let mut neglected = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let bond: KinshipBond = entry_from_record(&record, "KinshipBond")?;
            let last_tended_micros: i64 = bond.last_tended.as_micros();
            let elapsed_micros: u64 = if now_micros > last_tended_micros {
                (now_micros - last_tended_micros) as u64
            } else {
                0u64
            };
            let days_inactive = (elapsed_micros / micros_per_day) as u32;
            let current_strength = decayed_strength(bond.strength_bp, days_inactive);
            if current_strength < 3000 {
                neglected.push(record);
            }
        }
    }

    Ok(neglected)
}

/// Get bond snapshots for a hearth, computing current decayed strength.
/// Used by the bridge for weekly digest assembly.
#[hdk_extern]
pub fn get_bond_snapshots(hearth_hash: ActionHash) -> ExternResult<Vec<BondUpdate>> {
    let now = sys_time()?;
    let now_micros: i64 = now.as_micros();
    let micros_per_day: u64 = 86_400_000_000;

    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToBonds)?,
        GetStrategy::default(),
    )?;

    let mut snapshots = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let bond: KinshipBond = entry_from_record(&record, "KinshipBond")?;
            let last_tended_micros: i64 = bond.last_tended.as_micros();
            let elapsed_micros: u64 = if now_micros > last_tended_micros {
                (now_micros - last_tended_micros) as u64
            } else {
                0u64
            };
            let days_inactive = (elapsed_micros / micros_per_day) as u32;
            let current_strength = decayed_strength(bond.strength_bp, days_inactive);

            snapshots.push(BondUpdate {
                member_a: bond.member_a,
                member_b: bond.member_b,
                co_creation_count: 0,
                quality_sum_bp: current_strength,
            });
        }
    }

    Ok(snapshots)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Pure helpers (testable without conductor) ----

    /// Find the vote weight for an agent from a list of memberships.
    /// Returns 0 if the agent is not found or not active.
    fn find_member_vote_weight(agent: &AgentPubKey, memberships: &[HearthMembership]) -> u32 {
        memberships
            .iter()
            .find(|m| m.agent == *agent && m.status == MembershipStatus::Active)
            .map(|m| m.role.default_vote_weight_bp())
            .unwrap_or(0)
    }

    /// Count active memberships in a list.
    fn count_active_memberships(memberships: &[HearthMembership]) -> u32 {
        memberships
            .iter()
            .filter(|m| m.status == MembershipStatus::Active)
            .count() as u32
    }

    /// Find the role for an agent from a list of memberships.
    /// Returns None if the agent is not found or not active.
    fn find_member_role(
        agent: &AgentPubKey,
        memberships: &[HearthMembership],
    ) -> Option<MemberRole> {
        memberships
            .iter()
            .find(|m| m.agent == *agent && m.status == MembershipStatus::Active)
            .map(|m| m.role.clone())
    }

    // ---- Entry Type Existence ----

    #[test]
    fn hearth_entry_type_exists() {
        let _v = UnitEntryTypes::Hearth;
    }

    #[test]
    fn membership_entry_type_exists() {
        let _v = UnitEntryTypes::HearthMembership;
    }

    #[test]
    fn bond_entry_type_exists() {
        let _v = UnitEntryTypes::KinshipBond;
    }

    #[test]
    fn invitation_entry_type_exists() {
        let _v = UnitEntryTypes::HearthInvitation;
    }

    #[test]
    fn anchor_entry_type_exists() {
        let _v = UnitEntryTypes::Anchor;
    }

    // ---- Link Type Existence ----

    #[test]
    fn link_types_all_hearths_exists() {
        let _v = LinkTypes::AllHearths;
    }

    #[test]
    fn link_types_agent_to_hearths_exists() {
        let _v = LinkTypes::AgentToHearths;
    }

    #[test]
    fn link_types_hearth_to_members_exists() {
        let _v = LinkTypes::HearthToMembers;
    }

    #[test]
    fn link_types_hearth_to_bonds_exists() {
        let _v = LinkTypes::HearthToBonds;
    }

    #[test]
    fn link_types_member_to_bonds_exists() {
        let _v = LinkTypes::MemberToBonds;
    }

    #[test]
    fn link_types_type_to_hearths_exists() {
        let _v = LinkTypes::TypeToHearths;
    }

    #[test]
    fn link_types_hearth_to_invitations_exists() {
        let _v = LinkTypes::HearthToInvitations;
    }

    #[test]
    fn link_types_agent_to_invitations_exists() {
        let _v = LinkTypes::AgentToInvitations;
    }

    // ---- Input Serde Roundtrips ----

    #[test]
    fn create_hearth_input_serde_roundtrip() {
        let input = CreateHearthInput {
            name: "Test Hearth".into(),
            description: "A test family".into(),
            hearth_type: HearthType::Nuclear,
            max_members: Some(10),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateHearthInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Test Hearth");
        assert_eq!(back.max_members, Some(10));
    }

    #[test]
    fn create_hearth_input_no_max_members_serde() {
        let input = CreateHearthInput {
            name: "Minimal".into(),
            description: String::new(),
            hearth_type: HearthType::Chosen,
            max_members: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateHearthInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_members, None);
    }

    #[test]
    fn invite_member_input_serde_roundtrip() {
        let input = InviteMemberInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            invitee_agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
            proposed_role: MemberRole::Adult,
            message: "Please join us!".into(),
            expires_at: Timestamp::from_micros(1_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: InviteMemberInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message, "Please join us!");
    }

    #[test]
    fn accept_invitation_input_serde_roundtrip() {
        let input = AcceptInvitationInput {
            invitation_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            display_name: "Alice".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AcceptInvitationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.display_name, "Alice");
    }

    #[test]
    fn update_member_role_input_serde_roundtrip() {
        let input = UpdateMemberRoleInput {
            membership_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_role: MemberRole::Elder,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateMemberRoleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.new_role, MemberRole::Elder);
    }

    #[test]
    fn create_bond_input_serde_roundtrip() {
        let input = CreateBondInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            member_b: AgentPubKey::from_raw_36(vec![1u8; 36]),
            bond_type: BondType::Partner,
            initial_strength_bp: Some(8000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBondInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.initial_strength_bp, Some(8000));
    }

    #[test]
    fn create_bond_input_default_strength_serde() {
        let input = CreateBondInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            member_b: AgentPubKey::from_raw_36(vec![1u8; 36]),
            bond_type: BondType::Sibling,
            initial_strength_bp: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBondInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.initial_strength_bp, None);
    }

    #[test]
    fn tend_bond_input_serde_roundtrip() {
        let input = TendBondInput {
            bond_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            description: "Had dinner together".into(),
            quality_bp: 8500,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: TendBondInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.description, "Had dinner together");
        assert_eq!(back.quality_bp, 8500);
    }

    #[test]
    fn get_bond_health_input_serde_roundtrip() {
        let input = GetBondHealthInput {
            bond_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: GetBondHealthInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bond_hash, ActionHash::from_raw_36(vec![0u8; 36]));
    }

    // ---- WeeklyDigest serde ----

    #[test]
    fn weekly_digest_input_serde_roundtrip() {
        let digest = WeeklyDigest {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            epoch_start: Timestamp::from_micros(0),
            epoch_end: Timestamp::from_micros(604_800_000_000),
            bond_updates: vec![BondUpdate {
                member_a: AgentPubKey::from_raw_36(vec![0u8; 36]),
                member_b: AgentPubKey::from_raw_36(vec![1u8; 36]),
                co_creation_count: 3,
                quality_sum_bp: 24000,
            }],
            care_summary: vec![CareSummary {
                assignee: AgentPubKey::from_raw_36(vec![0u8; 36]),
                tasks_completed: 5,
                hours_hundredths: 1200,
            }],
            gratitude_summary: vec![GratitudeSummary {
                from_agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
                to_agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
                count: 7,
            }],
            rhythm_summary: vec![RhythmSummary {
                rhythm_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                occurrences: 3,
                avg_participation_bp: 8000,
            }],
            created_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
            created_at: Timestamp::from_micros(604_800_000_000),
        };
        let json = serde_json::to_string(&digest).unwrap();
        let back: WeeklyDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bond_updates.len(), 1);
        assert_eq!(back.care_summary.len(), 1);
        assert_eq!(back.gratitude_summary.len(), 1);
        assert_eq!(back.rhythm_summary.len(), 1);
    }

    #[test]
    fn link_types_hearth_to_digests_exists() {
        let _v = LinkTypes::HearthToDigests;
    }

    // ---- Recovery threshold math ----

    #[test]
    fn recovery_threshold_3_adults() {
        // 3 * 60 / 100 = 1.8, ceil = 2
        assert_eq!(recovery_threshold(3), 2);
    }

    #[test]
    fn recovery_threshold_4_adults() {
        // 4 * 60 / 100 = 2.4, ceil = 3
        assert_eq!(recovery_threshold(4), 3);
    }

    #[test]
    fn recovery_threshold_5_adults() {
        // 5 * 60 / 100 = 3.0, ceil = 3
        assert_eq!(recovery_threshold(5), 3);
    }

    #[test]
    fn recovery_threshold_6_adults() {
        // 6 * 60 / 100 = 3.6, ceil = 4
        assert_eq!(recovery_threshold(6), 4);
    }

    #[test]
    fn recovery_threshold_7_adults() {
        // 7 * 60 / 100 = 4.2, ceil = 5
        assert_eq!(recovery_threshold(7), 5);
    }

    #[test]
    fn recovery_threshold_10_adults() {
        // 10 * 60 / 100 = 6.0, ceil = 6
        assert_eq!(recovery_threshold(10), 6);
    }

    #[test]
    fn recovery_threshold_always_at_least_60_percent() {
        for n in 3..=20 {
            let t = recovery_threshold(n);
            // t/n >= 0.6 => t * 100 >= n * 60
            assert!(
                t * 100 >= n * 60,
                "threshold {t} for {n} adults is below 60%"
            );
        }
    }

    // ---- Pure helper test fixtures ----

    fn fake_agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn make_membership(
        agent: &AgentPubKey,
        role: MemberRole,
        status: MembershipStatus,
    ) -> HearthMembership {
        HearthMembership {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            agent: agent.clone(),
            role,
            status,
            display_name: "test".into(),
            joined_at: Timestamp::from_micros(0),
        }
    }

    // ---- find_member_vote_weight ----

    #[test]
    fn vote_weight_active_adult() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Active,
        )];
        assert_eq!(
            find_member_vote_weight(&fake_agent_a(), &memberships),
            10000
        );
    }

    #[test]
    fn vote_weight_active_youth() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Youth,
            MembershipStatus::Active,
        )];
        assert_eq!(find_member_vote_weight(&fake_agent_a(), &memberships), 5000);
    }

    #[test]
    fn vote_weight_active_child() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Child,
            MembershipStatus::Active,
        )];
        assert_eq!(find_member_vote_weight(&fake_agent_a(), &memberships), 0);
    }

    #[test]
    fn vote_weight_departed_returns_zero() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Departed,
        )];
        assert_eq!(find_member_vote_weight(&fake_agent_a(), &memberships), 0);
    }

    #[test]
    fn vote_weight_not_found_returns_zero() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Active,
        )];
        assert_eq!(find_member_vote_weight(&fake_agent_b(), &memberships), 0);
    }

    #[test]
    fn vote_weight_empty_list() {
        assert_eq!(find_member_vote_weight(&fake_agent_a(), &[]), 0);
    }

    // ---- count_active_memberships ----

    #[test]
    fn count_active_all_active() {
        let memberships = vec![
            make_membership(&fake_agent_a(), MemberRole::Adult, MembershipStatus::Active),
            make_membership(&fake_agent_b(), MemberRole::Elder, MembershipStatus::Active),
        ];
        assert_eq!(count_active_memberships(&memberships), 2);
    }

    #[test]
    fn count_active_mixed_statuses() {
        let memberships = vec![
            make_membership(&fake_agent_a(), MemberRole::Adult, MembershipStatus::Active),
            make_membership(
                &fake_agent_b(),
                MemberRole::Elder,
                MembershipStatus::Departed,
            ),
        ];
        assert_eq!(count_active_memberships(&memberships), 1);
    }

    #[test]
    fn count_active_all_departed() {
        let memberships = vec![
            make_membership(
                &fake_agent_a(),
                MemberRole::Adult,
                MembershipStatus::Departed,
            ),
            make_membership(
                &fake_agent_b(),
                MemberRole::Elder,
                MembershipStatus::Departed,
            ),
        ];
        assert_eq!(count_active_memberships(&memberships), 0);
    }

    #[test]
    fn count_active_empty_list() {
        assert_eq!(count_active_memberships(&[]), 0);
    }

    // ---- find_member_role ----

    #[test]
    fn find_role_active_adult() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Active,
        )];
        assert_eq!(
            find_member_role(&fake_agent_a(), &memberships),
            Some(MemberRole::Adult)
        );
    }

    #[test]
    fn find_role_departed_returns_none() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Departed,
        )];
        assert_eq!(find_member_role(&fake_agent_a(), &memberships), None);
    }

    #[test]
    fn find_role_not_found_returns_none() {
        let memberships = vec![make_membership(
            &fake_agent_a(),
            MemberRole::Adult,
            MembershipStatus::Active,
        )];
        assert_eq!(find_member_role(&fake_agent_b(), &memberships), None);
    }

    #[test]
    fn find_role_empty_list() {
        assert_eq!(find_member_role(&fake_agent_a(), &[]), None);
    }
}
