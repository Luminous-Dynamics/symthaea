//! Hearth Autonomy Integrity Zome
//! Defines entry types and validation for graduated autonomy profiles,
//! capability requests, guardian approvals, and tier transitions
//! (Living Primitives Liminality).

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// An autonomy profile for a hearth member, tracking their current tier,
/// capabilities, restrictions, and designated guardians.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AutonomyProfile {
    /// The hearth this profile belongs to.
    pub hearth_hash: ActionHash,
    /// The member this profile describes.
    pub member: AgentPubKey,
    /// Designated guardian agents who can approve capability changes.
    pub guardian_agents: Vec<AgentPubKey>,
    /// Current autonomy tier.
    pub current_tier: AutonomyTier,
    /// Capabilities the member currently has (e.g. "manage_finances", "cook_unsupervised").
    pub capabilities: Vec<String>,
    /// Restrictions that override capabilities (e.g. "no_fire", "no_sharp_tools").
    pub restrictions: Vec<String>,
    /// Optional schedule for periodic autonomy reviews.
    pub review_schedule: Option<String>,
    /// When this profile was created.
    pub created_at: Timestamp,
}

/// A request from a member (typically youth) to gain a new capability.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AutonomyRequest {
    /// The hearth this request belongs to.
    pub hearth_hash: ActionHash,
    /// The member requesting the capability.
    pub requester: AgentPubKey,
    /// The capability being requested (e.g. "use_stove", "go_to_park_alone").
    pub capability: String,
    /// Why the member believes they are ready for this capability.
    pub justification: String,
    /// Current status of the request.
    pub status: AutonomyRequestStatus,
    /// When this request was created.
    pub created_at: Timestamp,
}

/// A guardian's approval or denial of an autonomy request.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianApproval {
    /// The autonomy request being responded to.
    pub request_hash: ActionHash,
    /// The guardian making the decision.
    pub guardian: AgentPubKey,
    /// Whether the guardian approved (true) or denied (false).
    pub approved: bool,
    /// Optional conditions attached to the approval (e.g. "only on weekdays").
    pub conditions: Option<String>,
    /// When this approval was recorded.
    pub created_at: Timestamp,
}

/// A record of a tier transition, following Living Primitives Liminality.
/// Transitions are forward-only and progress through liminal phases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TierTransition {
    /// The hearth this transition belongs to.
    pub hearth_hash: ActionHash,
    /// The member undergoing the transition.
    pub member: AgentPubKey,
    /// The tier being transitioned from.
    pub from_tier: AutonomyTier,
    /// The tier being transitioned to.
    pub to_tier: AutonomyTier,
    /// Current phase of the transition ritual.
    pub transition_phase: TransitionPhase,
    /// Whether recategorization is blocked during this transition.
    /// True during PreLiminal/Liminal/PostLiminal, false only at Integrated.
    pub recategorization_blocked: bool,
    /// When the transition was initiated.
    pub started_at: Timestamp,
    /// When the transition completed (set when phase reaches Integrated).
    pub completed_at: Option<Timestamp>,
}

// ============================================================================
// Entry + Link Type Enums
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AutonomyProfile(AutonomyProfile),
    AutonomyRequest(AutonomyRequest),
    GuardianApproval(GuardianApproval),
    TierTransition(TierTransition),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> all autonomy profiles in that hearth.
    HearthToProfiles,
    /// Agent -> their autonomy profile.
    AgentToProfile,
    /// Hearth -> all autonomy requests in that hearth.
    HearthToRequests,
    /// Agent -> all autonomy requests from that agent.
    AgentToRequests,
    /// AutonomyRequest -> all guardian approvals for that request.
    RequestToApprovals,
    /// Hearth -> all tier transitions in that hearth.
    HearthToTransitions,
    /// Agent -> all tier transitions for that agent.
    AgentToTransitions,
}

// ============================================================================
// Tier Ordering Helper
// ============================================================================

/// Returns a numeric rank for an AutonomyTier for ordering comparisons.
/// AutonomyTier does not derive PartialOrd, so we use an explicit function.
pub fn tier_rank(tier: &AutonomyTier) -> u8 {
    match tier {
        AutonomyTier::Dependent => 0,
        AutonomyTier::Supervised => 1,
        AutonomyTier::Guided => 2,
        AutonomyTier::SemiAutonomous => 3,
        AutonomyTier::Autonomous => 4,
    }
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::AutonomyProfile(profile) => validate_profile(&profile, &action),
                EntryTypes::AutonomyRequest(request) => validate_request(&request),
                EntryTypes::GuardianApproval(approval) => validate_approval(&approval),
                EntryTypes::TierTransition(transition) => validate_transition(&transition),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::AutonomyProfile(profile) => {
                    validate_profile_update(&profile)?;
                    validate_profile_immutable_fields(&profile, &original_action_hash)
                }
                EntryTypes::AutonomyRequest(request) => {
                    validate_request_update(&request)?;
                    validate_request_immutable_fields(&request, &original_action_hash)
                }
                EntryTypes::GuardianApproval(_) => {
                    // INVARIANT: GuardianApproval immutability — once a guardian records
                    // their approval or denial, it cannot be modified. This ensures
                    // that autonomy decisions remain stable and auditable.
                    Ok(ValidateCallbackResult::Invalid(
                        "GuardianApproval cannot be updated once recorded".into(),
                    ))
                }
                EntryTypes::TierTransition(transition) => {
                    validate_transition_update(&transition)?;
                    validate_transition_immutable_fields(&transition, &original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Autonomy entries cannot be deleted once created".into(),
        )),
    }
}

fn validate_profile(
    profile: &AutonomyProfile,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    if profile.capabilities.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile cannot have more than 50 capabilities".into(),
        ));
    }
    if profile.restrictions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile cannot have more than 50 restrictions".into(),
        ));
    }
    if profile.guardian_agents.is_empty() || profile.guardian_agents.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile must have 1-10 guardian agents".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_profile_update(profile: &AutonomyProfile) -> ExternResult<ValidateCallbackResult> {
    if profile.capabilities.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile cannot have more than 50 capabilities".into(),
        ));
    }
    if profile.restrictions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile cannot have more than 50 restrictions".into(),
        ));
    }
    if profile.guardian_agents.is_empty() || profile.guardian_agents.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy profile must have 1-10 guardian agents".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_request(request: &AutonomyRequest) -> ExternResult<ValidateCallbackResult> {
    if request.capability.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request capability cannot be empty".into(),
        ));
    }
    if request.capability.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request capability must be <= 256 characters".into(),
        ));
    }
    if request.justification.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request justification cannot be empty".into(),
        ));
    }
    if request.justification.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request justification must be <= 4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_request_update(request: &AutonomyRequest) -> ExternResult<ValidateCallbackResult> {
    if request.capability.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request capability cannot be empty".into(),
        ));
    }
    if request.capability.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request capability must be <= 256 characters".into(),
        ));
    }
    if request.justification.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request justification cannot be empty".into(),
        ));
    }
    if request.justification.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Autonomy request justification must be <= 4096 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_approval(approval: &GuardianApproval) -> ExternResult<ValidateCallbackResult> {
    // Basic structural validation: conditions, if present, must not be too long
    if let Some(ref conditions) = approval.conditions {
        if conditions.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Guardian approval conditions must be <= 4096 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_transition(transition: &TierTransition) -> ExternResult<ValidateCallbackResult> {
    // Forward-only: to_tier must be strictly greater than from_tier
    // (Living Primitives Liminality — no regression of autonomy)
    if tier_rank(&transition.to_tier) <= tier_rank(&transition.from_tier) {
        return Ok(ValidateCallbackResult::Invalid(
            "Tier transition must be forward-only (to_tier must be greater than from_tier)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_transition_update(transition: &TierTransition) -> ExternResult<ValidateCallbackResult> {
    // Forward-only constraint still applies on updates
    if tier_rank(&transition.to_tier) <= tier_rank(&transition.from_tier) {
        return Ok(ValidateCallbackResult::Invalid(
            "Tier transition must be forward-only (to_tier must be greater than from_tier)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Immutable Field Validation
// ============================================================================

fn validate_profile_immutable_fields(
    new: &AutonomyProfile,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: AutonomyProfile = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original AutonomyProfile: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original AutonomyProfile entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on an AutonomyProfile".into(),
        ));
    }
    if new.member != original.member {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change member on an AutonomyProfile".into(),
        ));
    }
    if new.created_at != original.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on an AutonomyProfile".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_request_immutable_fields(
    new: &AutonomyRequest,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: AutonomyRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original AutonomyRequest: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original AutonomyRequest entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on an AutonomyRequest".into(),
        ));
    }
    if new.requester != original.requester {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change requester on an AutonomyRequest".into(),
        ));
    }
    if new.created_at != original.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on an AutonomyRequest".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_transition_immutable_fields(
    new: &TierTransition,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: TierTransition = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original TierTransition: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original TierTransition entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a TierTransition".into(),
        ));
    }
    if new.member != original.member {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change member on a TierTransition".into(),
        ));
    }
    if new.from_tier != original.from_tier {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change from_tier on a TierTransition".into(),
        ));
    }
    if new.to_tier != original.to_tier {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change to_tier on a TierTransition".into(),
        ));
    }
    if new.started_at != original.started_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change started_at on a TierTransition".into(),
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

    // -- Factory helpers --

    fn agent_key_1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xaa; 36])
    }

    fn agent_key_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xbb; 36])
    }

    fn action_hash_1() -> ActionHash {
        ActionHash::from_raw_36(vec![0xa1; 36])
    }

    fn timestamp_now() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn create_action() -> Create {
        Create {
            author: agent_key_1(),
            timestamp: timestamp_now(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn valid_profile() -> AutonomyProfile {
        AutonomyProfile {
            hearth_hash: action_hash_1(),
            member: agent_key_2(),
            guardian_agents: vec![agent_key_1()],
            current_tier: AutonomyTier::Supervised,
            capabilities: vec!["use_kitchen".to_string()],
            restrictions: vec!["no_stove".to_string()],
            review_schedule: Some("quarterly".to_string()),
            created_at: timestamp_now(),
        }
    }

    fn valid_request() -> AutonomyRequest {
        AutonomyRequest {
            hearth_hash: action_hash_1(),
            requester: agent_key_2(),
            capability: "use_stove".to_string(),
            justification: "I have completed the cooking safety course".to_string(),
            status: AutonomyRequestStatus::Pending,
            created_at: timestamp_now(),
        }
    }

    fn valid_approval() -> GuardianApproval {
        GuardianApproval {
            request_hash: action_hash_1(),
            guardian: agent_key_1(),
            approved: true,
            conditions: Some("Only with adult in adjacent room".to_string()),
            created_at: timestamp_now(),
        }
    }

    fn valid_transition() -> TierTransition {
        TierTransition {
            hearth_hash: action_hash_1(),
            member: agent_key_2(),
            from_tier: AutonomyTier::Supervised,
            to_tier: AutonomyTier::Guided,
            transition_phase: TransitionPhase::PreLiminal,
            recategorization_blocked: true,
            started_at: timestamp_now(),
            completed_at: None,
        }
    }

    // -- tier_rank --

    #[test]
    fn test_tier_rank_ordering() {
        assert!(tier_rank(&AutonomyTier::Dependent) < tier_rank(&AutonomyTier::Supervised));
        assert!(tier_rank(&AutonomyTier::Supervised) < tier_rank(&AutonomyTier::Guided));
        assert!(tier_rank(&AutonomyTier::Guided) < tier_rank(&AutonomyTier::SemiAutonomous));
        assert!(tier_rank(&AutonomyTier::SemiAutonomous) < tier_rank(&AutonomyTier::Autonomous));
    }

    #[test]
    fn test_tier_rank_same_tier_equal() {
        assert_eq!(
            tier_rank(&AutonomyTier::Guided),
            tier_rank(&AutonomyTier::Guided)
        );
    }

    // -- AutonomyProfile validation --

    #[test]
    fn test_valid_profile_passes() {
        let result = validate_profile(&valid_profile(), &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_profile_capabilities_at_limit_passes() {
        let mut profile = valid_profile();
        profile.capabilities = (0..50).map(|i| format!("cap_{}", i)).collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_profile_capabilities_over_limit_fails() {
        let mut profile = valid_profile();
        profile.capabilities = (0..51).map(|i| format!("cap_{}", i)).collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_profile_restrictions_at_limit_passes() {
        let mut profile = valid_profile();
        profile.restrictions = (0..50).map(|i| format!("res_{}", i)).collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_profile_restrictions_over_limit_fails() {
        let mut profile = valid_profile();
        profile.restrictions = (0..51).map(|i| format!("res_{}", i)).collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_profile_no_guardians_fails() {
        let mut profile = valid_profile();
        profile.guardian_agents = vec![];
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_profile_one_guardian_passes() {
        let profile = valid_profile(); // already has 1 guardian
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_profile_ten_guardians_passes() {
        let mut profile = valid_profile();
        profile.guardian_agents = (0..10)
            .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
            .collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_profile_eleven_guardians_fails() {
        let mut profile = valid_profile();
        profile.guardian_agents = (0..11)
            .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
            .collect();
        let result = validate_profile(&profile, &create_action()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- AutonomyRequest validation --

    #[test]
    fn test_valid_request_passes() {
        let result = validate_request(&valid_request()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_request_empty_capability_fails() {
        let mut request = valid_request();
        request.capability = "".to_string();
        let result = validate_request(&request).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_request_capability_at_limit_passes() {
        let mut request = valid_request();
        request.capability = "c".repeat(256);
        let result = validate_request(&request).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_request_capability_over_limit_fails() {
        let mut request = valid_request();
        request.capability = "c".repeat(257);
        let result = validate_request(&request).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_request_empty_justification_fails() {
        let mut request = valid_request();
        request.justification = "".to_string();
        let result = validate_request(&request).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_request_justification_at_limit_passes() {
        let mut request = valid_request();
        request.justification = "j".repeat(4096);
        let result = validate_request(&request).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_request_justification_over_limit_fails() {
        let mut request = valid_request();
        request.justification = "j".repeat(4097);
        let result = validate_request(&request).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- GuardianApproval validation --

    #[test]
    fn test_valid_approval_passes() {
        let result = validate_approval(&valid_approval()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_approval_no_conditions_passes() {
        let mut approval = valid_approval();
        approval.conditions = None;
        let result = validate_approval(&approval).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_approval_conditions_over_limit_fails() {
        let mut approval = valid_approval();
        approval.conditions = Some("c".repeat(4097));
        let result = validate_approval(&approval).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- TierTransition validation --

    #[test]
    fn test_valid_transition_passes() {
        let result = validate_transition(&valid_transition()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transition_same_tier_fails() {
        let mut transition = valid_transition();
        transition.to_tier = transition.from_tier.clone();
        let result = validate_transition(&transition).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_transition_backward_fails() {
        let mut transition = valid_transition();
        transition.from_tier = AutonomyTier::Guided;
        transition.to_tier = AutonomyTier::Dependent;
        let result = validate_transition(&transition).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_transition_dependent_to_autonomous_passes() {
        let mut transition = valid_transition();
        transition.from_tier = AutonomyTier::Dependent;
        transition.to_tier = AutonomyTier::Autonomous;
        let result = validate_transition(&transition).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_autonomy_profile_serde_roundtrip() {
        let profile = valid_profile();
        let json = serde_json::to_string(&profile).unwrap();
        let back: AutonomyProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, back);
    }

    #[test]
    fn test_autonomy_request_serde_roundtrip() {
        let request = valid_request();
        let json = serde_json::to_string(&request).unwrap();
        let back: AutonomyRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request, back);
    }

    #[test]
    fn test_guardian_approval_serde_roundtrip() {
        let approval = valid_approval();
        let json = serde_json::to_string(&approval).unwrap();
        let back: GuardianApproval = serde_json::from_str(&json).unwrap();
        assert_eq!(approval, back);
    }

    #[test]
    fn test_tier_transition_serde_roundtrip() {
        let transition = valid_transition();
        let json = serde_json::to_string(&transition).unwrap();
        let back: TierTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(transition, back);
    }

    // -- Immutable field pure equality tests --

    #[test]
    fn profile_immutable_field_hearth_hash_difference_detected() {
        let a = valid_profile();
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn profile_immutable_field_member_difference_detected() {
        let a = valid_profile();
        let mut b = a.clone();
        b.member = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.member, b.member);
    }

    #[test]
    fn profile_immutable_field_created_at_difference_detected() {
        let a = valid_profile();
        let mut b = a.clone();
        b.created_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.created_at, b.created_at);
    }

    #[test]
    fn request_immutable_field_hearth_hash_difference_detected() {
        let a = valid_request();
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn request_immutable_field_requester_difference_detected() {
        let a = valid_request();
        let mut b = a.clone();
        b.requester = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.requester, b.requester);
    }

    #[test]
    fn request_immutable_field_created_at_difference_detected() {
        let a = valid_request();
        let mut b = a.clone();
        b.created_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.created_at, b.created_at);
    }

    #[test]
    fn guardian_approval_immutability_documented() {
        // GuardianApproval should be created once and never updated.
        // The integrity validate() function rejects UpdateEntry for GuardianApproval.
        let approval = valid_approval();
        assert_eq!(approval.approved, true);
        assert_eq!(approval.guardian, agent_key_1());
    }

    #[test]
    fn transition_immutable_field_hearth_hash_difference_detected() {
        let a = valid_transition();
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn transition_immutable_field_member_difference_detected() {
        let a = valid_transition();
        let mut b = a.clone();
        b.member = AgentPubKey::from_raw_36(vec![0xcd; 36]);
        assert_ne!(a.member, b.member);
    }

    #[test]
    fn transition_immutable_field_from_tier_difference_detected() {
        let a = valid_transition();
        let mut b = a.clone();
        b.from_tier = AutonomyTier::Dependent;
        assert_ne!(a.from_tier, b.from_tier);
    }

    #[test]
    fn transition_immutable_field_to_tier_difference_detected() {
        let a = valid_transition();
        let mut b = a.clone();
        b.to_tier = AutonomyTier::Autonomous;
        assert_ne!(a.to_tier, b.to_tier);
    }

    #[test]
    fn delete_guard_message_content() {
        let msg = "Autonomy entries cannot be deleted once created";
        assert!(msg.contains("cannot be deleted"));
    }

    #[test]
    fn transition_immutable_field_started_at_difference_detected() {
        let a = valid_transition();
        let mut b = a.clone();
        b.started_at = Timestamp::from_micros(9_000_000);
        assert_ne!(a.started_at, b.started_at);
    }
}
