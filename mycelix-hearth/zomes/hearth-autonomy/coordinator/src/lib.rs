//! Hearth Autonomy Coordinator Zome
//! Business logic for managing graduated autonomy profiles,
//! capability requests, guardian approvals, and tier transitions
//! following Living Primitives Liminality.

use hdk::prelude::*;
use hearth_autonomy_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateAutonomyProfileInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub guardian_agents: Vec<AgentPubKey>,
    pub initial_tier: AutonomyTier,
    pub capabilities: Vec<String>,
    pub restrictions: Vec<String>,
    pub review_schedule: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestCapabilityInput {
    pub hearth_hash: ActionHash,
    pub capability: String,
    pub justification: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveCapabilityInput {
    pub request_hash: ActionHash,
    pub approved: bool,
    pub conditions: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdvanceTierInput {
    pub profile_hash: ActionHash,
    pub new_tier: AutonomyTier,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckCapabilityInput {
    pub member: AgentPubKey,
    pub capability: String,
}

// ============================================================================
// Helpers
// ============================================================================

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

// ============================================================================
// Extern Functions
// ============================================================================

/// Create an autonomy profile for a hearth member.
/// Only guardians (Founder, Elder, or Adult) can call this.
#[hdk_extern]
pub fn create_autonomy_profile(input: CreateAutonomyProfileInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Verify the caller has a guardian-level role in this hearth.
    // Cross-zome call to kinship coordinator's is_guardian() function.
    let guardian_response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("is_guardian"),
        None,
        input.hearth_hash.clone(),
    )?;
    let is_guardian: bool = match guardian_response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode is_guardian response: {}",
                e
            )))
        })?,
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cross-zome call to is_guardian failed: {:?}",
                other
            ))))
        }
    };
    if !is_guardian {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians (Founder, Elder, or Adult) can create autonomy profiles".into()
        )));
    }

    let profile = AutonomyProfile {
        hearth_hash: input.hearth_hash.clone(),
        member: input.member.clone(),
        guardian_agents: input.guardian_agents,
        current_tier: input.initial_tier,
        capabilities: input.capabilities,
        restrictions: input.restrictions,
        review_schedule: input.review_schedule,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::AutonomyProfile(profile))?;

    // Link hearth -> profile
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToProfiles,
        (),
    )?;

    // Link agent -> profile
    create_link(
        input.member,
        action_hash.clone(),
        LinkTypes::AgentToProfile,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created autonomy profile".into()
    )))
}

/// Request a new capability (typically called by a youth member).
#[hdk_extern]
pub fn request_capability(input: RequestCapabilityInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let request = AutonomyRequest {
        hearth_hash: input.hearth_hash.clone(),
        requester: caller.clone(),
        capability: input.capability,
        justification: input.justification,
        status: AutonomyRequestStatus::Pending,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::AutonomyRequest(request))?;

    // Link hearth -> request
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToRequests,
        (),
    )?;

    // Link agent -> request
    create_link(caller, action_hash.clone(), LinkTypes::AgentToRequests, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created autonomy request".into()
    )))
}

/// Approve a capability request (guardian action).
/// Only guardians (Founder, Elder, or Adult) can approve/deny capabilities.
#[hdk_extern]
pub fn approve_capability(input: ApproveCapabilityInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get the request to find its hearth_hash for guardian verification
    let request_record = get(input.request_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Autonomy request not found".into())),
    )?;
    let mut request: AutonomyRequest = request_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid autonomy request entry".into()
        )))?;

    // Verify the caller has a guardian-level role in this hearth.
    let guardian_response = call(
        CallTargetCell::Local,
        ZomeName::new("hearth_kinship"),
        FunctionName::new("is_guardian"),
        None,
        request.hearth_hash.clone(),
    )?;
    let is_guardian: bool = match guardian_response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode is_guardian response: {}",
                e
            )))
        })?,
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cross-zome call to is_guardian failed: {:?}",
                other
            ))))
        }
    };
    if !is_guardian {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only guardians (Founder, Elder, or Adult) can approve capabilities".into()
        )));
    }

    let approval = GuardianApproval {
        request_hash: input.request_hash.clone(),
        guardian: caller,
        approved: input.approved,
        conditions: input.conditions,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::GuardianApproval(approval))?;

    // Link request -> approval
    create_link(
        input.request_hash.clone(),
        action_hash.clone(),
        LinkTypes::RequestToApprovals,
        (),
    )?;

    // Update the request status
    let new_status = if input.approved {
        AutonomyRequestStatus::Approved
    } else {
        AutonomyRequestStatus::Denied
    };
    request.status = new_status;
    update_entry(input.request_hash, &EntryTypes::AutonomyRequest(request))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created guardian approval".into()
    )))
}

/// Deny a capability request (convenience wrapper — calls approve with approved=false).
#[hdk_extern]
pub fn deny_capability(input: ApproveCapabilityInput) -> ExternResult<Record> {
    let denial_input = ApproveCapabilityInput {
        request_hash: input.request_hash,
        approved: false,
        conditions: input.conditions,
    };
    approve_capability(denial_input)
}

/// Advance a member's autonomy tier. Creates a TierTransition with PreLiminal
/// phase and recategorization_blocked=true. If advancing to Autonomous,
/// triggers severance via cross-zome call to hearth_bridge.
#[hdk_extern]
pub fn advance_tier(input: AdvanceTierInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Get the current profile
    let profile_record = get(input.profile_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Autonomy profile not found".into())),
    )?;

    let profile: AutonomyProfile = profile_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid autonomy profile entry".into()
        )))?;

    // Validate forward-only transition
    if tier_rank(&input.new_tier) <= tier_rank(&profile.current_tier) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tier advancement must be forward-only (new tier must be greater than current)".into()
        )));
    }

    let transition = TierTransition {
        hearth_hash: profile.hearth_hash.clone(),
        member: profile.member.clone(),
        from_tier: profile.current_tier.clone(),
        to_tier: input.new_tier.clone(),
        transition_phase: TransitionPhase::PreLiminal,
        recategorization_blocked: true,
        started_at: now,
        completed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::TierTransition(transition))?;

    // Save hearth_hash for potential H3 severance before ownership transfer
    let hearth_hash = profile.hearth_hash.clone();

    // Link hearth -> transition
    create_link(
        profile.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToTransitions,
        (),
    )?;

    // Link agent -> transition
    create_link(
        profile.member,
        action_hash.clone(),
        LinkTypes::AgentToTransitions,
        (),
    )?;

    // H3: If advancing to Autonomous, trigger severance for coming-of-age data migration.
    if input.new_tier == AutonomyTier::Autonomous {
        let severance_input = SeveranceInput {
            hearth_hash,
            member_hash: input.profile_hash.clone(),
            export_milestones: true,
            export_care_history: true,
            export_bond_snapshot: true,
            new_role: MemberRole::Adult,
        };
        // Best-effort: don't block tier advancement on severance failure
        let _ = call(
            CallTargetCell::Local,
            ZomeName::new("hearth_bridge"),
            FunctionName::new("initiate_severance"),
            None,
            severance_input,
        );
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created tier transition".into()
    )))
}

/// Progress a tier transition forward one phase:
/// PreLiminal -> Liminal -> PostLiminal -> Integrated.
/// Sets recategorization_blocked=false only at Integrated,
/// and updates the profile's current_tier at that point.
#[hdk_extern]
pub fn progress_transition(transition_hash: ActionHash) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = get(transition_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Tier transition not found".into())
    ))?;

    let mut transition: TierTransition = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid tier transition entry".into()
        )))?;

    // Advance one phase
    let next_phase = match transition.transition_phase {
        TransitionPhase::PreLiminal => TransitionPhase::Liminal,
        TransitionPhase::Liminal => TransitionPhase::PostLiminal,
        TransitionPhase::PostLiminal => TransitionPhase::Integrated,
        TransitionPhase::Integrated => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Transition is already at Integrated phase".into()
            )));
        }
    };

    transition.transition_phase = next_phase.clone();

    // Only unblock recategorization and set completed_at at Integrated
    if next_phase == TransitionPhase::Integrated {
        transition.recategorization_blocked = false;
        transition.completed_at = Some(now);

        // Update the member's autonomy profile to reflect the new tier.
        // Find the profile via AgentToProfile links.
        let profile_links = get_links(
            LinkQuery::try_new(transition.member.clone(), LinkTypes::AgentToProfile)?,
            GetStrategy::default(),
        )?;

        if let Some(profile_link) = profile_links.last() {
            let profile_hash = ActionHash::try_from(profile_link.target.clone()).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest("Invalid profile link target".into()))
            })?;

            if let Some(profile_record) = get(profile_hash.clone(), GetOptions::default())? {
                let mut profile: AutonomyProfile = profile_record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid autonomy profile entry".into()
                    )))?;

                profile.current_tier = transition.to_tier.clone();
                update_entry(profile_hash, &EntryTypes::AutonomyProfile(profile))?;
            }
        }
    }

    let updated_hash = update_entry(transition_hash, &EntryTypes::TierTransition(transition))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated tier transition".into()
    )))
}

/// Get a member's autonomy profile via AgentToProfile links.
#[hdk_extern]
pub fn get_autonomy_profile(member: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?;

    // Return the most recent profile link target
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        Ok(get(action_hash, GetOptions::default())?)
    } else {
        Ok(None)
    }
}

/// Runtime permission check: returns true if a member has a given capability
/// and that capability is not in their restrictions list.
#[hdk_extern]
pub fn check_capability(input: CheckCapabilityInput) -> ExternResult<bool> {
    let CheckCapabilityInput { member, capability } = input;
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?;

    let profile_link = match links.last() {
        Some(link) => link,
        None => return Ok(false),
    };

    let action_hash = ActionHash::try_from(profile_link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = match get(action_hash, GetOptions::default())? {
        Some(r) => r,
        None => return Ok(false),
    };

    let profile: AutonomyProfile = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid autonomy profile entry".into()
        )))?;

    // Check: capability is in the capabilities list AND not in restrictions
    let has_capability = profile.capabilities.contains(&capability);
    let is_restricted = profile.restrictions.contains(&capability);

    Ok(has_capability && !is_restricted)
}

/// Get all pending autonomy requests for a hearth.
#[hdk_extern]
pub fn get_pending_requests(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToRequests)?,
        GetStrategy::default(),
    )?;

    let all_records = records_from_links(links)?;

    // Filter to only pending requests
    let pending: Vec<Record> = all_records
        .into_iter()
        .filter(|record| {
            if let Some(request) = record
                .entry()
                .to_app_option::<AutonomyRequest>()
                .ok()
                .flatten()
            {
                request.status == AutonomyRequestStatus::Pending
            } else {
                false
            }
        })
        .collect();

    Ok(pending)
}

/// Get all active (non-Integrated) tier transitions for a hearth.
#[hdk_extern]
pub fn get_active_transitions(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToTransitions)?,
        GetStrategy::default(),
    )?;

    let all_records = records_from_links(links)?;

    // Filter to only active (not yet Integrated) transitions
    let active: Vec<Record> = all_records
        .into_iter()
        .filter(|record| {
            if let Some(transition) = record
                .entry()
                .to_app_option::<TierTransition>()
                .ok()
                .flatten()
            {
                transition.transition_phase != TransitionPhase::Integrated
            } else {
                false
            }
        })
        .collect();

    Ok(active)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xaa; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xbb; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa1; 36])
    }

    fn action_hash_2() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa2; 36])
    }

    // -- Input type serde roundtrips --

    #[test]
    fn test_create_autonomy_profile_input_serde() {
        let input = CreateAutonomyProfileInput {
            hearth_hash: action_hash_1(),
            member: agent_b(),
            guardian_agents: vec![agent_a()],
            initial_tier: AutonomyTier::Dependent,
            capabilities: vec!["use_tablet".to_string()],
            restrictions: vec!["no_internet".to_string()],
            review_schedule: Some("monthly".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateAutonomyProfileInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.capabilities, vec!["use_tablet"]);
        assert_eq!(back.restrictions, vec!["no_internet"]);
    }

    #[test]
    fn test_request_capability_input_serde() {
        let input = RequestCapabilityInput {
            hearth_hash: action_hash_1(),
            capability: "use_stove".to_string(),
            justification: "I completed the safety course".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RequestCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.capability, "use_stove");
    }

    #[test]
    fn test_approve_capability_input_serde() {
        let input = ApproveCapabilityInput {
            request_hash: action_hash_1(),
            approved: true,
            conditions: Some("Only weekdays".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ApproveCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.approved, true);
        assert_eq!(back.conditions, Some("Only weekdays".to_string()));
    }

    #[test]
    fn test_approve_capability_input_denied_serde() {
        let input = ApproveCapabilityInput {
            request_hash: action_hash_1(),
            approved: false,
            conditions: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ApproveCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.approved, false);
        assert_eq!(back.conditions, None);
    }

    #[test]
    fn test_advance_tier_input_serde() {
        let input = AdvanceTierInput {
            profile_hash: action_hash_1(),
            new_tier: AutonomyTier::Guided,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AdvanceTierInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.new_tier, AutonomyTier::Guided);
    }

    #[test]
    fn test_create_profile_input_no_review_schedule() {
        let input = CreateAutonomyProfileInput {
            hearth_hash: action_hash_1(),
            member: agent_b(),
            guardian_agents: vec![agent_a()],
            initial_tier: AutonomyTier::Supervised,
            capabilities: vec![],
            restrictions: vec![],
            review_schedule: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateAutonomyProfileInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.review_schedule, None);
    }

    #[test]
    fn test_create_profile_input_multiple_guardians() {
        let input = CreateAutonomyProfileInput {
            hearth_hash: action_hash_1(),
            member: agent_b(),
            guardian_agents: vec![agent_a(), AgentPubKey::from_raw_36(vec![0xcc; 36])],
            initial_tier: AutonomyTier::Dependent,
            capabilities: vec!["play_outside".to_string(), "use_tablet".to_string()],
            restrictions: vec!["no_fire".to_string()],
            review_schedule: Some("quarterly".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateAutonomyProfileInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.guardian_agents.len(), 2);
        assert_eq!(back.capabilities.len(), 2);
    }

    #[test]
    fn test_advance_tier_input_to_autonomous() {
        let input = AdvanceTierInput {
            profile_hash: action_hash_2(),
            new_tier: AutonomyTier::Autonomous,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AdvanceTierInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.new_tier, AutonomyTier::Autonomous);
    }

    #[test]
    fn test_request_capability_input_long_justification() {
        let input = RequestCapabilityInput {
            hearth_hash: action_hash_1(),
            capability: "manage_finances".to_string(),
            justification: "x".repeat(4000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RequestCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.justification.len(), 4000);
    }

    #[test]
    fn test_approve_with_long_conditions() {
        let input = ApproveCapabilityInput {
            request_hash: action_hash_1(),
            approved: true,
            conditions: Some("c".repeat(2000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ApproveCapabilityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.conditions.unwrap().len(), 2000);
    }
}
