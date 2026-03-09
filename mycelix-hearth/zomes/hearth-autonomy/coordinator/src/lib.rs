//! Hearth Autonomy Coordinator Zome
//! Business logic for managing graduated autonomy profiles,
//! capability requests, guardian approvals, and tier transitions
//! following Living Primitives Liminality.

use hdk::prelude::*;
use hearth_autonomy_integrity::*;
use hearth_coordinator_common::{
    decode_zome_response, get_latest_record, records_from_links, require_guardian,
    require_membership,
};
use hearth_types::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_constitutional,
    requirement_for_voting, GovernanceEligibility, GovernanceRequirement,
};

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

/// Check whether a tier advancement is valid (forward-only).
#[allow(dead_code)]
fn is_valid_tier_advancement(from: &AutonomyTier, to: &AutonomyTier) -> bool {
    tier_rank(to) > tier_rank(from)
}

/// Check whether a request is still pending and can be approved/denied.
#[allow(dead_code)]
fn can_approve_request(status: &AutonomyRequestStatus) -> bool {
    *status == AutonomyRequestStatus::Pending
}

/// Check whether a transition phase can be progressed further.
#[allow(dead_code)]
fn can_progress_phase(phase: &TransitionPhase) -> bool {
    *phase != TransitionPhase::Integrated
}

/// Get the next phase in the transition sequence. Returns None for Integrated.
#[allow(dead_code)]
fn next_transition_phase(phase: &TransitionPhase) -> Option<TransitionPhase> {
    match phase {
        TransitionPhase::PreLiminal => Some(TransitionPhase::Liminal),
        TransitionPhase::Liminal => Some(TransitionPhase::PostLiminal),
        TransitionPhase::PostLiminal => Some(TransitionPhase::Integrated),
        TransitionPhase::Integrated => None,
    }
}

/// Check if a capability is granted: present in capabilities AND not in restrictions.
#[allow(dead_code)]
fn is_capability_granted(cap: &str, capabilities: &[String], restrictions: &[String]) -> bool {
    capabilities.iter().any(|c| c == cap) && !restrictions.iter().any(|r| r == cap)
}

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("hearth_bridge", requirement, action_name)
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create an autonomy profile for a hearth member.
/// Only guardians (Founder, Elder, or Adult) can call this.
#[hdk_extern]
pub fn create_autonomy_profile(input: CreateAutonomyProfileInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "create_autonomy_profile")?;
    let now = sys_time()?;

    // Verify the caller has a guardian-level role in this hearth.
    let is_guardian: bool = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("is_guardian"),
            None,
            input.hearth_hash.clone(),
        )?,
        "is_guardian",
    )?;
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
    let _eligibility = require_consciousness(&requirement_for_basic(), "request_capability")?;
    require_membership(&input.hearth_hash)?;
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
    let _eligibility = require_consciousness(&requirement_for_voting(), "approve_capability")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get the request to find its hearth_hash for guardian verification
    let request_record = get_latest_record(input.request_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Autonomy request not found".into())
    ))?;
    let mut request: AutonomyRequest = request_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid autonomy request entry".into()
        )))?;

    // Status check: only pending requests can be approved/denied
    if !can_approve_request(&request.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot approve/deny a request with status {:?} (must be Pending)",
            request.status
        ))));
    }

    // Auth: require guardian role
    require_guardian(&request.hearth_hash)?;

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

    if input.approved {
        emit_signal(&HearthSignal::CapabilityApproved {
            request_hash: input.request_hash.clone(),
            capability: request.capability.clone(),
        })?;
    } else {
        emit_signal(&HearthSignal::CapabilityDenied {
            request_hash: input.request_hash.clone(),
            capability: request.capability.clone(),
        })?;
    }

    update_entry(input.request_hash, &EntryTypes::AutonomyRequest(request))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created guardian approval".into()
    )))
}

/// Deny a capability request (convenience wrapper — calls approve with approved=false).
#[hdk_extern]
pub fn deny_capability(input: ApproveCapabilityInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "deny_capability")?;
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
    let _eligibility = require_consciousness(&requirement_for_constitutional(), "advance_tier")?;
    let now = sys_time()?;

    // Get the current profile (follow update chain)
    let profile_record = get_latest_record(input.profile_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Autonomy profile not found".into())
    ))?;

    let profile: AutonomyProfile = profile_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid autonomy profile entry".into()
        )))?;

    // Auth: only guardians can advance tiers
    require_guardian(&profile.hearth_hash)?;

    // Validate forward-only transition
    if !is_valid_tier_advancement(&profile.current_tier, &input.new_tier) {
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

    emit_signal(&HearthSignal::TierAdvanced {
        profile_hash: input.profile_hash.clone(),
        from_tier: profile.current_tier.clone(),
        to_tier: input.new_tier.clone(),
    })?;

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
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::new("hearth_bridge"),
            FunctionName::new("initiate_severance"),
            None,
            severance_input,
        ) {
            let _ = emit_signal(&HearthSignal::CrossZomeCallFailed {
                zome: "hearth_bridge".into(),
                function: "initiate_severance".into(),
                error: format!("{e:?}"),
            });
        }
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
    let _eligibility = require_consciousness(&requirement_for_voting(), "progress_transition")?;
    let now = sys_time()?;

    let record = get_latest_record(transition_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Tier transition not found".into())
    ))?;

    let mut transition: TierTransition = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid tier transition entry".into()
        )))?;

    require_membership(&transition.hearth_hash)?;

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

            if let Some(profile_record) = get_latest_record(profile_hash.clone())? {
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

    let updated_hash = update_entry(
        transition_hash.clone(),
        &EntryTypes::TierTransition(transition),
    )?;

    emit_signal(&HearthSignal::TransitionProgressed {
        transition_hash: transition_hash.clone(),
        new_phase: next_phase.clone(),
    })?;

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
        Ok(get_latest_record(action_hash)?)
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

    let record = match get_latest_record(action_hash)? {
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
        assert!(back.approved);
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
        assert!(!back.approved);
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

    // ====================================================================
    // Pure helper: is_valid_tier_advancement
    // ====================================================================

    #[test]
    fn tier_rank_ordering() {
        assert!(tier_rank(&AutonomyTier::Dependent) < tier_rank(&AutonomyTier::Supervised));
        assert!(tier_rank(&AutonomyTier::Supervised) < tier_rank(&AutonomyTier::Guided));
        assert!(tier_rank(&AutonomyTier::Guided) < tier_rank(&AutonomyTier::SemiAutonomous));
        assert!(tier_rank(&AutonomyTier::SemiAutonomous) < tier_rank(&AutonomyTier::Autonomous));
    }

    #[test]
    fn valid_single_step_advancement() {
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Dependent,
            &AutonomyTier::Supervised
        ));
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Supervised,
            &AutonomyTier::Guided
        ));
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Guided,
            &AutonomyTier::SemiAutonomous
        ));
        assert!(is_valid_tier_advancement(
            &AutonomyTier::SemiAutonomous,
            &AutonomyTier::Autonomous
        ));
    }

    #[test]
    fn valid_multi_step_advancement() {
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Dependent,
            &AutonomyTier::Guided
        ));
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Dependent,
            &AutonomyTier::Autonomous
        ));
        assert!(is_valid_tier_advancement(
            &AutonomyTier::Supervised,
            &AutonomyTier::Autonomous
        ));
    }

    #[test]
    fn same_tier_not_valid_advancement() {
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::Dependent,
            &AutonomyTier::Dependent
        ));
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::Guided,
            &AutonomyTier::Guided
        ));
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::Autonomous,
            &AutonomyTier::Autonomous
        ));
    }

    #[test]
    fn backward_advancement_rejected() {
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::Supervised,
            &AutonomyTier::Dependent
        ));
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::Autonomous,
            &AutonomyTier::Guided
        ));
        assert!(!is_valid_tier_advancement(
            &AutonomyTier::SemiAutonomous,
            &AutonomyTier::Supervised
        ));
    }

    // ====================================================================
    // Pure helper: can_approve_request
    // ====================================================================

    #[test]
    fn pending_request_can_be_approved() {
        assert!(can_approve_request(&AutonomyRequestStatus::Pending));
    }

    #[test]
    fn approved_request_cannot_be_approved_again() {
        assert!(!can_approve_request(&AutonomyRequestStatus::Approved));
    }

    #[test]
    fn denied_request_cannot_be_approved() {
        assert!(!can_approve_request(&AutonomyRequestStatus::Denied));
    }

    // ====================================================================
    // Pure helper: can_progress_phase / next_transition_phase
    // ====================================================================

    #[test]
    fn preliminal_can_progress() {
        assert!(can_progress_phase(&TransitionPhase::PreLiminal));
        assert_eq!(
            next_transition_phase(&TransitionPhase::PreLiminal),
            Some(TransitionPhase::Liminal)
        );
    }

    #[test]
    fn liminal_can_progress() {
        assert!(can_progress_phase(&TransitionPhase::Liminal));
        assert_eq!(
            next_transition_phase(&TransitionPhase::Liminal),
            Some(TransitionPhase::PostLiminal)
        );
    }

    #[test]
    fn postliminal_can_progress() {
        assert!(can_progress_phase(&TransitionPhase::PostLiminal));
        assert_eq!(
            next_transition_phase(&TransitionPhase::PostLiminal),
            Some(TransitionPhase::Integrated)
        );
    }

    #[test]
    fn integrated_cannot_progress() {
        assert!(!can_progress_phase(&TransitionPhase::Integrated));
        assert_eq!(next_transition_phase(&TransitionPhase::Integrated), None);
    }

    #[test]
    fn full_transition_phase_walk() {
        let mut phase = TransitionPhase::PreLiminal;
        let mut steps = 0;
        while let Some(next) = next_transition_phase(&phase) {
            phase = next;
            steps += 1;
        }
        assert_eq!(phase, TransitionPhase::Integrated);
        assert_eq!(steps, 3);
    }

    // ====================================================================
    // Pure helper: is_capability_granted
    // ====================================================================

    #[test]
    fn capability_granted_when_present_and_unrestricted() {
        let caps = vec!["use_stove".into(), "play_outside".into()];
        let restrictions: Vec<String> = vec![];
        assert!(is_capability_granted("use_stove", &caps, &restrictions));
    }

    #[test]
    fn capability_denied_when_not_present() {
        let caps = vec!["use_stove".into()];
        let restrictions: Vec<String> = vec![];
        assert!(!is_capability_granted("drive_car", &caps, &restrictions));
    }

    #[test]
    fn capability_denied_when_restricted() {
        let caps = vec!["use_stove".into(), "drive_car".into()];
        let restrictions = vec!["drive_car".into()];
        assert!(!is_capability_granted("drive_car", &caps, &restrictions));
        // use_stove is not restricted
        assert!(is_capability_granted("use_stove", &caps, &restrictions));
    }

    #[test]
    fn capability_denied_when_both_empty() {
        let caps: Vec<String> = vec![];
        let restrictions: Vec<String> = vec![];
        assert!(!is_capability_granted("anything", &caps, &restrictions));
    }

    #[test]
    fn capability_case_sensitive() {
        let caps = vec!["Use_Stove".into()];
        let restrictions: Vec<String> = vec![];
        assert!(!is_capability_granted("use_stove", &caps, &restrictions));
        assert!(is_capability_granted("Use_Stove", &caps, &restrictions));
    }

    // ====================================================================
    // Scenario tests
    // ====================================================================

    #[test]
    fn scenario_full_tier_journey() {
        let tiers = [
            AutonomyTier::Dependent,
            AutonomyTier::Supervised,
            AutonomyTier::Guided,
            AutonomyTier::SemiAutonomous,
            AutonomyTier::Autonomous,
        ];
        for i in 0..tiers.len() - 1 {
            assert!(
                is_valid_tier_advancement(&tiers[i], &tiers[i + 1]),
                "Expected {:?} -> {:?} to be valid",
                tiers[i],
                tiers[i + 1]
            );
        }
    }

    #[test]
    fn scenario_guardian_roles_for_tier_advancement() {
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
    }

    #[test]
    fn scenario_request_lifecycle() {
        // Pending -> can be approved or denied
        assert!(can_approve_request(&AutonomyRequestStatus::Pending));
        // After approval -> terminal
        assert!(!can_approve_request(&AutonomyRequestStatus::Approved));
        // After denial -> terminal
        assert!(!can_approve_request(&AutonomyRequestStatus::Denied));
    }

    #[test]
    fn scenario_capability_with_tier_context() {
        // Dependent: minimal caps
        let dependent_caps: Vec<String> = vec!["play_inside".into()];
        let dependent_restrict: Vec<String> = vec!["use_stove".into(), "drive_car".into()];
        assert!(is_capability_granted(
            "play_inside",
            &dependent_caps,
            &dependent_restrict
        ));
        assert!(!is_capability_granted(
            "use_stove",
            &dependent_caps,
            &dependent_restrict
        ));

        // Autonomous: full caps, no restrictions
        let autonomous_caps: Vec<String> = vec![
            "play_inside".into(),
            "use_stove".into(),
            "drive_car".into(),
            "manage_finances".into(),
        ];
        let autonomous_restrict: Vec<String> = vec![];
        assert!(is_capability_granted(
            "drive_car",
            &autonomous_caps,
            &autonomous_restrict
        ));
        assert!(is_capability_granted(
            "manage_finances",
            &autonomous_caps,
            &autonomous_restrict
        ));
    }

    #[test]
    fn all_tiers_have_distinct_ranks() {
        let tiers = [
            AutonomyTier::Dependent,
            AutonomyTier::Supervised,
            AutonomyTier::Guided,
            AutonomyTier::SemiAutonomous,
            AutonomyTier::Autonomous,
        ];
        for i in 0..tiers.len() {
            for j in (i + 1)..tiers.len() {
                assert_ne!(
                    tier_rank(&tiers[i]),
                    tier_rank(&tiers[j]),
                    "{:?} and {:?} should have different ranks",
                    tiers[i],
                    tiers[j]
                );
            }
        }
    }

    // ====================================================================
    // Complex business logic scenario tests
    // ====================================================================

    /// Walk through all 5 tiers from Dependent to Autonomous, verifying
    /// each single-step advancement is valid and the next step is also valid.
    /// Then verify that Autonomous cannot advance further.
    #[test]
    fn scenario_full_tier_advancement_journey() {
        let tiers = [
            AutonomyTier::Dependent,
            AutonomyTier::Supervised,
            AutonomyTier::Guided,
            AutonomyTier::SemiAutonomous,
            AutonomyTier::Autonomous,
        ];

        // Each consecutive step must be valid
        for i in 0..tiers.len() - 1 {
            assert!(
                is_valid_tier_advancement(&tiers[i], &tiers[i + 1]),
                "Step {:?} -> {:?} should be valid",
                tiers[i],
                tiers[i + 1]
            );

            // Verify the NEXT step from the new tier is also valid
            // (except when we are at SemiAutonomous -> Autonomous, there is no further step)
            if i + 2 < tiers.len() {
                assert!(
                    is_valid_tier_advancement(&tiers[i + 1], &tiers[i + 2]),
                    "Following step {:?} -> {:?} should also be valid",
                    tiers[i + 1],
                    tiers[i + 2]
                );
            }
        }

        // Autonomous cannot advance further: no tier is above it
        for tier in &tiers {
            assert!(
                !is_valid_tier_advancement(&AutonomyTier::Autonomous, tier),
                "Autonomous should not be able to advance to {:?}",
                tier
            );
        }
    }

    /// Test that a Pending request can be approved, and after approval
    /// cannot be approved again. Same for denial.
    #[test]
    fn scenario_guardian_approval_lifecycle() {
        // Step 1: Start with Pending request
        let initial = AutonomyRequestStatus::Pending;
        assert!(
            can_approve_request(&initial),
            "Pending request should be approvable"
        );

        // Step 2a: Approve it -> now Approved, cannot approve again
        let approved = AutonomyRequestStatus::Approved;
        assert!(
            !can_approve_request(&approved),
            "Approved request must not be re-approved"
        );

        // Step 2b: Or deny it -> now Denied, cannot approve/deny again
        let denied = AutonomyRequestStatus::Denied;
        assert!(
            !can_approve_request(&denied),
            "Denied request must not be re-approved"
        );

        // Verify the terminal states are truly terminal:
        // neither Approved nor Denied can transition back to an approvable state
        assert!(!can_approve_request(&AutonomyRequestStatus::Approved));
        assert!(!can_approve_request(&AutonomyRequestStatus::Denied));
    }

    /// As a member progresses through tiers, their capability set should grow.
    /// Dependent: minimal caps, many restrictions.
    /// Supervised: a few more caps, some restrictions lifted.
    /// Guided: broader caps, fewer restrictions.
    /// SemiAutonomous: most caps, very few restrictions.
    /// Autonomous: full caps, no restrictions.
    #[test]
    fn scenario_capability_accumulation() {
        // Dependent tier: 1 capability, 4 restrictions
        let dep_caps: Vec<String> = vec!["play_inside".into()];
        let dep_restrict: Vec<String> = vec![
            "use_stove".into(),
            "drive_car".into(),
            "manage_finances".into(),
            "go_out_alone".into(),
        ];
        assert!(is_capability_granted(
            "play_inside",
            &dep_caps,
            &dep_restrict
        ));
        assert!(!is_capability_granted(
            "use_stove",
            &dep_caps,
            &dep_restrict
        ));
        assert!(!is_capability_granted(
            "drive_car",
            &dep_caps,
            &dep_restrict
        ));

        // Supervised tier: 2 capabilities, 3 restrictions
        let sup_caps: Vec<String> = vec!["play_inside".into(), "use_stove".into()];
        let sup_restrict: Vec<String> = vec![
            "drive_car".into(),
            "manage_finances".into(),
            "go_out_alone".into(),
        ];
        assert!(is_capability_granted(
            "play_inside",
            &sup_caps,
            &sup_restrict
        ));
        assert!(is_capability_granted("use_stove", &sup_caps, &sup_restrict));
        assert!(!is_capability_granted(
            "drive_car",
            &sup_caps,
            &sup_restrict
        ));

        // Guided tier: 3 capabilities, 2 restrictions
        let gui_caps: Vec<String> = vec![
            "play_inside".into(),
            "use_stove".into(),
            "go_out_alone".into(),
        ];
        let gui_restrict: Vec<String> = vec!["drive_car".into(), "manage_finances".into()];
        assert!(is_capability_granted(
            "go_out_alone",
            &gui_caps,
            &gui_restrict
        ));
        assert!(!is_capability_granted(
            "drive_car",
            &gui_caps,
            &gui_restrict
        ));

        // SemiAutonomous tier: 4 capabilities, 1 restriction
        let semi_caps: Vec<String> = vec![
            "play_inside".into(),
            "use_stove".into(),
            "go_out_alone".into(),
            "drive_car".into(),
        ];
        let semi_restrict: Vec<String> = vec!["manage_finances".into()];
        assert!(is_capability_granted(
            "drive_car",
            &semi_caps,
            &semi_restrict
        ));
        assert!(!is_capability_granted(
            "manage_finances",
            &semi_caps,
            &semi_restrict
        ));

        // Autonomous tier: 5 capabilities, 0 restrictions
        let auto_caps: Vec<String> = vec![
            "play_inside".into(),
            "use_stove".into(),
            "go_out_alone".into(),
            "drive_car".into(),
            "manage_finances".into(),
        ];
        let auto_restrict: Vec<String> = vec![];
        assert!(is_capability_granted(
            "manage_finances",
            &auto_caps,
            &auto_restrict
        ));
        assert!(is_capability_granted(
            "drive_car",
            &auto_caps,
            &auto_restrict
        ));

        // Verify capability count grows monotonically
        assert!(dep_caps.len() < sup_caps.len());
        assert!(sup_caps.len() < gui_caps.len());
        assert!(gui_caps.len() < semi_caps.len());
        assert!(semi_caps.len() < auto_caps.len());

        // Verify restriction count shrinks monotonically
        assert!(dep_restrict.len() > sup_restrict.len());
        assert!(sup_restrict.len() > gui_restrict.len());
        assert!(gui_restrict.len() > semi_restrict.len());
        assert!(semi_restrict.len() > auto_restrict.len());
    }

    /// Walk through all 4 transition phases. In PreLiminal/Liminal/PostLiminal,
    /// the phase can still progress (recategorization is blocked during these
    /// active phases). Only at Integrated is recategorization unblocked.
    #[test]
    fn scenario_transition_phase_recategorization_blocking() {
        let mut phase = TransitionPhase::PreLiminal;

        // PreLiminal: can progress, recategorization should be blocked
        assert!(
            can_progress_phase(&phase),
            "PreLiminal should be progressable"
        );

        // Advance to Liminal
        phase = next_transition_phase(&phase).expect("PreLiminal should have a next phase");
        assert_eq!(phase, TransitionPhase::Liminal);
        assert!(can_progress_phase(&phase), "Liminal should be progressable");

        // Advance to PostLiminal
        phase = next_transition_phase(&phase).expect("Liminal should have a next phase");
        assert_eq!(phase, TransitionPhase::PostLiminal);
        assert!(
            can_progress_phase(&phase),
            "PostLiminal should be progressable"
        );

        // Advance to Integrated
        phase = next_transition_phase(&phase).expect("PostLiminal should have a next phase");
        assert_eq!(phase, TransitionPhase::Integrated);
        assert!(
            !can_progress_phase(&phase),
            "Integrated should NOT be progressable"
        );

        // Integrated has no next phase
        assert_eq!(
            next_transition_phase(&phase),
            None,
            "Integrated should have no next phase"
        );

        // Total of 3 progressions from PreLiminal to Integrated
        let mut count = 0;
        let mut p = TransitionPhase::PreLiminal;
        while let Some(next) = next_transition_phase(&p) {
            count += 1;
            // During all intermediate phases, can_progress should be true
            assert!(
                can_progress_phase(&p),
                "Phase {:?} before Integrated should allow progress",
                p
            );
            p = next;
        }
        assert_eq!(count, 3);
        assert_eq!(p, TransitionPhase::Integrated);
    }

    /// Verify that Autonomous -> Autonomous tier advancement is rejected.
    #[test]
    fn scenario_autonomous_to_autonomous_rejected() {
        assert!(
            !is_valid_tier_advancement(&AutonomyTier::Autonomous, &AutonomyTier::Autonomous),
            "Same-tier advancement (Autonomous -> Autonomous) must be rejected"
        );
    }
}
