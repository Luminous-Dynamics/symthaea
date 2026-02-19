//! # Agent Lifecycle Management
//!
//! Creation, suspension, and revocation of Instrumental Actors.
//!
//! ## K-Vector Initialization
//!
//! New agents are created with a `new_participant` K-Vector (10 dimensions):
//! - k_r = 0.5 (neutral reputation)
//! - k_a = 0.0 (no activity yet)
//! - k_i = 1.0 (integrity assumed until proven otherwise)
//! - k_p = 0.5 (unknown performance)
//! - k_m = 0.0 (just joined)
//! - k_s = 0.0 (no stake yet)
//! - k_h = 0.5 (no history)
//! - k_topo = 0.0 (not connected yet)
//! - k_v = 0.5 (verification status)
//! - k_coherence = 0.5 (coherence)
//!
//! ## GIS Integration
//!
//! Actions are gated on both:
//! 1. Coherence (Phi) - Is the agent behaving coherently?
//! 2. Uncertainty (GIS) - Is the agent appropriately certain?
//!
//! High uncertainty triggers escalation to human sponsor.

use super::{
    constraints::AgentConstraints, epistemic_classifier::EpistemicStats,
    kredit::calculate_kredit_cap, sponsor_requirements, uncertainty::UncertaintyCalibration,
    AgentClass, AgentId, AgentStatus, InstrumentalActor,
};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};

/// Result of agent creation
#[derive(Debug)]
pub enum CreateAgentResult {
    /// Agent created successfully
    Success(Box<InstrumentalActor>),
    /// Creation failed
    Error(AgentError),
}

/// Agent-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// Sponsor CIV too low
    InsufficientSponsorCiv,
    /// Too many active agents
    MaxAgentsExceeded,
    /// Not the sponsor of this agent
    NotSponsor,
    /// Agent not found
    AgentNotFound,
    /// Invalid state transition
    InvalidStateTransition,
    /// Collateral insufficient
    InsufficientCollateral,
    /// Coherence too low - action blocked
    CoherenceBlocked(String),
    /// Coherence requires sponsor approval
    CoherenceRequiresApproval(String),
}

/// Sponsor information for agent creation
#[derive(Debug, Clone)]
pub struct SponsorInfo {
    /// Sponsor DID
    pub did: String,
    /// Current CIV score
    pub civ_score: f64,
    /// SAP available for collateral
    pub sap_available: u64,
    /// Current active agent count
    pub active_agent_count: u32,
}

/// Create new agent
pub fn create_agent(
    sponsor: &SponsorInfo,
    agent_class: AgentClass,
    sap_to_lock: u64,
    timestamp: u64,
) -> CreateAgentResult {
    // Verify sponsor eligibility
    if sponsor.civ_score < sponsor_requirements::MIN_CIV_TO_CREATE {
        return CreateAgentResult::Error(AgentError::InsufficientSponsorCiv);
    }

    if sponsor.active_agent_count >= sponsor_requirements::MAX_AGENTS_PER_SPONSOR {
        return CreateAgentResult::Error(AgentError::MaxAgentsExceeded);
    }

    if sap_to_lock > sponsor.sap_available {
        return CreateAgentResult::Error(AgentError::InsufficientCollateral);
    }

    // Calculate KREDIT cap
    let kredit_cap = match calculate_kredit_cap(
        sponsor.civ_score,
        sap_to_lock,
        agent_class,
        sponsor.active_agent_count,
    ) {
        Ok(cap) => cap,
        Err(_) => return CreateAgentResult::Error(AgentError::InsufficientSponsorCiv),
    };

    // Create agent with constitutional constraints enforced
    let mut constraints = AgentConstraints::default();
    constraints.merge_constitutional();

    // Initialize with new participant K-Vector
    // Trust will evolve based on behavioral outcomes
    let k_vector = KVector::new_participant();

    let agent = InstrumentalActor {
        agent_id: AgentId::generate(),
        sponsor_did: sponsor.did.clone(),
        agent_class,
        kredit_balance: kredit_cap as i64,
        kredit_cap,
        constraints,
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: timestamp,
        last_activity: timestamp,
        actions_this_hour: 0,
        k_vector,
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: UncertaintyCalibration::default(),
        pending_escalations: vec![],
    };

    CreateAgentResult::Success(Box::new(agent))
}

/// Suspension reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuspensionReason {
    /// Manual sponsor suspension
    SponsorRequested,
    /// KREDIT depleted
    KreditDepleted,
    /// Sponsor CIV dropped
    SponsorCivDropped,
    /// Anomalous behavior detected
    AnomalyDetected,
    /// Rate limit exceeded
    RateLimitExceeded,
}

/// Suspend agent
pub fn suspend_agent(
    agent: &mut InstrumentalActor,
    sponsor_did: &str,
    _reason: SuspensionReason,
) -> Result<(), AgentError> {
    // Verify sponsorship
    if agent.sponsor_did != sponsor_did {
        return Err(AgentError::NotSponsor);
    }

    // Check valid state transition
    if matches!(agent.status, AgentStatus::Revoked) {
        return Err(AgentError::InvalidStateTransition);
    }

    agent.status = AgentStatus::Suspended;

    Ok(())
}

/// Revocation reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RevocationReason {
    /// Sponsor requested permanent termination
    SponsorRequested,
    /// Liability triggered
    LiabilityTriggered,
    /// Constitutional violation
    ConstitutionalViolation,
    /// Audit Guild enforcement
    AuditGuildAction,
    /// Sponsor CIV permanently below threshold
    SponsorCivPermanentlyLow,
}

/// Who initiated revocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RevocationInitiator {
    /// Sponsor revoked
    Sponsor,
    /// Automatic system revocation
    System,
    /// Audit Guild revocation
    AuditGuild,
    /// Karmic Council revocation
    KarmicCouncil,
}

/// Revocation event for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationEvent {
    /// Agent ID
    pub agent_id: String,
    /// Sponsor DID
    pub sponsor_did: String,
    /// Reason for revocation
    pub reason: RevocationReason,
    /// Initiator
    pub initiated_by: RevocationInitiator,
    /// Final KREDIT balance
    pub final_kredit: i64,
    /// Timestamp
    pub timestamp: u64,
}

/// Revoke agent (permanent termination)
pub fn revoke_agent(
    agent: &mut InstrumentalActor,
    reason: RevocationReason,
    initiated_by: RevocationInitiator,
    timestamp: u64,
) -> RevocationEvent {
    // Mark as revoked regardless of current status
    agent.status = AgentStatus::Revoked;

    RevocationEvent {
        agent_id: agent.agent_id.as_str().to_string(),
        sponsor_did: agent.sponsor_did.clone(),
        reason,
        initiated_by,
        final_kredit: agent.kredit_balance,
        timestamp,
    }
}

/// Reactivate suspended agent
pub fn reactivate_agent(
    agent: &mut InstrumentalActor,
    sponsor_did: &str,
    sponsor_civ: f64,
) -> Result<(), AgentError> {
    // Verify sponsorship
    if agent.sponsor_did != sponsor_did {
        return Err(AgentError::NotSponsor);
    }

    // Check valid state transition
    if !matches!(
        agent.status,
        AgentStatus::Suspended | AgentStatus::Throttled
    ) {
        return Err(AgentError::InvalidStateTransition);
    }

    // Check sponsor still qualifies
    if sponsor_civ < sponsor_requirements::MIN_CIV_TO_MAINTAIN {
        return Err(AgentError::InsufficientSponsorCiv);
    }

    agent.status = AgentStatus::Active;

    Ok(())
}

/// Reset epoch KREDIT allocation
pub fn reset_epoch(agent: &mut InstrumentalActor, new_cap: u64) {
    agent.kredit_balance = new_cap as i64;
    agent.kredit_cap = new_cap;
    agent.actions_this_hour = 0;

    // Unthrottle if was throttled due to KREDIT
    if agent.status == AgentStatus::Throttled {
        agent.status = AgentStatus::Active;
    }
}

/// Gate an action based on agent coherence (Phi measurement)
///
/// High-stakes actions require sufficient coherence. Returns error if
/// coherence is too low.
pub fn gate_action_on_coherence(
    coherence_state: super::coherence_bridge::CoherenceState,
    is_high_stakes: bool,
) -> Result<(), AgentError> {
    use super::coherence_bridge::{check_coherence_for_action, CoherenceCheckResult};

    match check_coherence_for_action(coherence_state, is_high_stakes) {
        CoherenceCheckResult::Allowed => Ok(()),
        CoherenceCheckResult::RequiresApproval { reason } => {
            // Could be Ok if sponsor pre-approved, but for now require explicit approval
            Err(AgentError::CoherenceRequiresApproval(reason))
        }
        CoherenceCheckResult::Blocked { reason } => Err(AgentError::CoherenceBlocked(reason)),
    }
}

// ============================================================================
// GIS INTEGRATION: Uncertainty-Based Action Gating
// ============================================================================

/// Maximum pending escalations per agent (security limit to prevent unbounded memory)
pub const MAX_PENDING_ESCALATIONS: usize = 100;

/// Maximum length for action strings in escalations
pub const MAX_ACTION_STRING_LENGTH: usize = 1024;

/// Maximum length for context strings in escalations
pub const MAX_CONTEXT_STRING_LENGTH: usize = 4096;

/// Error for uncertainty-based gating
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UncertaintyGatingError {
    /// Action blocked due to high uncertainty
    HighUncertainty(String),
    /// Action requires human escalation
    EscalationRequired(String),
    /// Escalation queue is full
    EscalationQueueFull,
    /// Input validation failed
    InvalidInput(String),
}

/// Result of checking uncertainty before action
#[derive(Debug, Clone)]
pub enum UncertaintyCheckResult {
    /// Proceed with action
    Proceed,
    /// Proceed but with monitoring
    ProceedWithMonitoring {
        /// Reason monitoring is required
        reason: String,
    },
    /// Action requires escalation to human sponsor
    EscalationRequired {
        /// The escalation request details
        escalation: super::uncertainty::EscalationRequest,
    },
    /// Action blocked
    Blocked {
        /// Reason the action was blocked
        reason: String,
    },
}

/// Validate and sanitize an action string for escalation
fn validate_action_string(action: &str) -> Result<String, UncertaintyGatingError> {
    if action.is_empty() {
        return Err(UncertaintyGatingError::InvalidInput(
            "Action string cannot be empty".to_string(),
        ));
    }

    // Truncate if too long (don't fail, just truncate for safety)
    let sanitized = if action.len() > MAX_ACTION_STRING_LENGTH {
        action[..MAX_ACTION_STRING_LENGTH].to_string()
    } else {
        action.to_string()
    };

    Ok(sanitized)
}

/// Validate and sanitize a context string for escalation
fn validate_context_string(context: Option<String>) -> Option<String> {
    context.map(|c| {
        if c.len() > MAX_CONTEXT_STRING_LENGTH {
            c[..MAX_CONTEXT_STRING_LENGTH].to_string()
        } else {
            c
        }
    })
}

/// Add an escalation to the agent's pending queue with validation
///
/// Returns error if queue is full or inputs are invalid.
pub fn add_escalation(
    agent: &mut super::InstrumentalActor,
    escalation: super::uncertainty::EscalationRequest,
) -> Result<(), UncertaintyGatingError> {
    // Check queue limit
    if agent.pending_escalations.len() >= MAX_PENDING_ESCALATIONS {
        return Err(UncertaintyGatingError::EscalationQueueFull);
    }

    // Check for duplicate escalations (same action already pending)
    if agent
        .pending_escalations
        .iter()
        .any(|e| e.blocked_action == escalation.blocked_action)
    {
        // Already pending, don't add duplicate
        return Ok(());
    }

    agent.pending_escalations.push(escalation);
    Ok(())
}

/// Gate an action based on moral uncertainty (GIS v4.0)
///
/// High uncertainty automatically escalates to human sponsor.
/// Returns escalation request if human intervention is needed.
///
/// Note: This function validates and sanitizes inputs before creating escalations.
pub fn gate_action_on_uncertainty(
    agent: &super::InstrumentalActor,
    uncertainty: &super::uncertainty::MoralUncertainty,
    action: &str,
    context: Option<String>,
) -> UncertaintyCheckResult {
    use super::uncertainty::{EscalationRequest, MoralActionGuidance};

    // Validate action string
    let sanitized_action = match validate_action_string(action) {
        Ok(a) => a,
        Err(e) => {
            return UncertaintyCheckResult::Blocked {
                reason: format!("Invalid action: {:?}", e),
            }
        }
    };

    // Sanitize context
    let sanitized_context = validate_context_string(context);

    let guidance = MoralActionGuidance::from_uncertainty(uncertainty);

    match guidance {
        MoralActionGuidance::ProceedConfidently => UncertaintyCheckResult::Proceed,
        MoralActionGuidance::ProceedWithMonitoring => {
            UncertaintyCheckResult::ProceedWithMonitoring {
                reason: format!(
                    "Moderate uncertainty (total: {:.2}, max: {:.2})",
                    uncertainty.total(),
                    uncertainty.max_dimension()
                ),
            }
        }
        MoralActionGuidance::PauseForReflection => {
            let escalation = EscalationRequest::new(
                agent.agent_id.as_str(),
                *uncertainty,
                &sanitized_action,
                sanitized_context,
            );
            UncertaintyCheckResult::EscalationRequired { escalation }
        }
        MoralActionGuidance::SeekConsultation | MoralActionGuidance::DeferAction => {
            let escalation = EscalationRequest::new(
                agent.agent_id.as_str(),
                *uncertainty,
                &sanitized_action,
                sanitized_context,
            );
            UncertaintyCheckResult::EscalationRequired { escalation }
        }
    }
}

/// Check both coherence and uncertainty before allowing an action
///
/// This is the main gating function that combines:
/// 1. Phi-based coherence check (is agent coherent?)
/// 2. GIS-based uncertainty check (is agent appropriately certain?)
///
/// Returns Ok if action can proceed, Err with reason if blocked/escalation needed.
pub fn check_action_readiness(
    agent: &super::InstrumentalActor,
    uncertainty: &super::uncertainty::MoralUncertainty,
    action: &str,
    is_high_stakes: bool,
) -> Result<ActionReadiness, ActionGatingError> {
    use super::coherence_bridge::{check_agent_coherence_gating, CoherenceCheckResult};

    // Step 1: Check coherence (Phi)
    let coherence_check = check_agent_coherence_gating(agent, is_high_stakes);

    // Coherence check passes if we don't return early
    match coherence_check {
        CoherenceCheckResult::Allowed => { /* Proceed to uncertainty check */ }
        CoherenceCheckResult::RequiresApproval { reason } => {
            return Err(ActionGatingError::CoherenceRequiresApproval(reason));
        }
        CoherenceCheckResult::Blocked { reason } => {
            return Err(ActionGatingError::CoherenceBlocked(reason));
        }
    }

    // Step 2: Check uncertainty (GIS)
    let uncertainty_check = gate_action_on_uncertainty(agent, uncertainty, action, None);

    match uncertainty_check {
        UncertaintyCheckResult::Proceed => Ok(ActionReadiness::Ready { monitoring: false }),
        UncertaintyCheckResult::ProceedWithMonitoring { reason: _ } => {
            Ok(ActionReadiness::Ready { monitoring: true })
        }
        UncertaintyCheckResult::EscalationRequired { escalation } => {
            Err(ActionGatingError::UncertaintyEscalation(escalation))
        }
        UncertaintyCheckResult::Blocked { reason } => {
            Err(ActionGatingError::UncertaintyBlocked(reason))
        }
    }
}

/// Combined result of action readiness check
#[derive(Debug, Clone)]
pub enum ActionReadiness {
    /// Action can proceed
    Ready {
        /// Whether the action is being actively monitored
        monitoring: bool,
    },
}

/// Combined action gating error
#[derive(Debug, Clone)]
pub enum ActionGatingError {
    /// Coherence too low
    CoherenceBlocked(String),
    /// Coherence requires approval
    CoherenceRequiresApproval(String),
    /// Uncertainty too high, needs escalation
    UncertaintyEscalation(super::uncertainty::EscalationRequest),
    /// Uncertainty too high, action blocked
    UncertaintyBlocked(String),
}

/// Record outcome of an uncertain action for calibration tracking
///
/// Call this after an action completes to update the agent's
/// uncertainty calibration profile.
pub fn record_uncertainty_outcome(
    agent: &mut super::InstrumentalActor,
    was_uncertain: bool,
    was_good_outcome: bool,
) {
    agent
        .uncertainty_calibration
        .record(was_uncertain, was_good_outcome);
}

/// Process pending escalations that have been resolved by sponsor
///
/// Returns list of escalation IDs that were processed.
pub fn process_resolved_escalations(
    agent: &mut super::InstrumentalActor,
    resolutions: &[(String, bool)], // (blocked_action, sponsor_approved)
) -> Vec<String> {
    let mut processed = Vec::new();

    for (action, approved) in resolutions {
        if let Some(idx) = agent
            .pending_escalations
            .iter()
            .position(|e| e.blocked_action == *action)
        {
            let escalation = agent.pending_escalations.remove(idx);

            // Record calibration based on sponsor decision
            // If sponsor approved, was the agent too cautious? (high uncertainty but would've been fine)
            // If sponsor rejected, was the agent right to escalate? (high uncertainty, bad outcome avoided)
            let was_uncertain = true; // Escalation means high uncertainty
            let was_good_outcome = *approved; // Sponsor approval = good outcome
            agent
                .uncertainty_calibration
                .record(was_uncertain, was_good_outcome);

            processed.push(escalation.blocked_action);
        }
    }

    processed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::uncertainty::{EscalationRequest, MoralUncertainty};

    fn valid_sponsor() -> SponsorInfo {
        SponsorInfo {
            did: "did:test:sponsor".to_string(),
            civ_score: 0.7,
            sap_available: 10_000,
            active_agent_count: 0,
        }
    }

    #[test]
    fn test_create_agent_success() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        match result {
            CreateAgentResult::Success(agent) => {
                assert_eq!(agent.sponsor_did, sponsor.did);
                assert_eq!(agent.agent_class, AgentClass::Supervised);
                assert!(agent.kredit_cap > 0);
                assert_eq!(agent.status, AgentStatus::Active);
            }
            CreateAgentResult::Error(_) => panic!("Expected success"),
        }
    }

    #[test]
    fn test_create_agent_low_civ() {
        let mut sponsor = valid_sponsor();
        sponsor.civ_score = 0.3; // Below minimum

        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        match result {
            CreateAgentResult::Error(AgentError::InsufficientSponsorCiv) => {}
            _ => panic!("Expected InsufficientSponsorCiv error"),
        }
    }

    #[test]
    fn test_suspend_and_reactivate() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        // Suspend
        suspend_agent(&mut agent, &sponsor.did, SuspensionReason::SponsorRequested).unwrap();
        assert_eq!(agent.status, AgentStatus::Suspended);

        // Reactivate
        reactivate_agent(&mut agent, &sponsor.did, sponsor.civ_score).unwrap();
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_revoke_agent() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        let event = revoke_agent(
            &mut agent,
            RevocationReason::SponsorRequested,
            RevocationInitiator::Sponsor,
            2000,
        );

        assert_eq!(agent.status, AgentStatus::Revoked);
        assert_eq!(event.sponsor_did, sponsor.did);
    }

    // ========================================================================
    // Security Tests for GIS Integration
    // ========================================================================

    #[test]
    fn test_escalation_queue_limit() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        // Fill queue to limit
        for i in 0..MAX_PENDING_ESCALATIONS {
            let escalation = EscalationRequest::new(
                agent.agent_id.as_str(),
                MoralUncertainty::unsure(),
                &format!("action_{}", i),
                None,
            );
            assert!(add_escalation(&mut agent, escalation).is_ok());
        }

        // Next one should fail
        let one_more = EscalationRequest::new(
            agent.agent_id.as_str(),
            MoralUncertainty::unsure(),
            "one_more",
            None,
        );
        let result = add_escalation(&mut agent, one_more);
        assert!(matches!(
            result,
            Err(UncertaintyGatingError::EscalationQueueFull)
        ));
    }

    #[test]
    fn test_duplicate_escalation_deduplicated() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        // Add first escalation
        let escalation1 = EscalationRequest::new(
            agent.agent_id.as_str(),
            MoralUncertainty::unsure(),
            "same_action",
            None,
        );
        assert!(add_escalation(&mut agent, escalation1).is_ok());
        assert_eq!(agent.pending_escalations.len(), 1);

        // Add duplicate - should succeed but not add
        let escalation2 = EscalationRequest::new(
            agent.agent_id.as_str(),
            MoralUncertainty::unsure(),
            "same_action",
            None,
        );
        assert!(add_escalation(&mut agent, escalation2).is_ok());
        assert_eq!(agent.pending_escalations.len(), 1); // Still 1
    }

    #[test]
    fn test_action_string_validation() {
        // Empty string should fail
        let result = validate_action_string("");
        assert!(matches!(
            result,
            Err(UncertaintyGatingError::InvalidInput(_))
        ));

        // Normal string should pass
        let result = validate_action_string("transfer_funds");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "transfer_funds");

        // Long string should be truncated
        let long_action = "a".repeat(MAX_ACTION_STRING_LENGTH + 100);
        let result = validate_action_string(&long_action);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), MAX_ACTION_STRING_LENGTH);
    }

    #[test]
    fn test_context_string_truncation() {
        // Normal context
        let context = validate_context_string(Some("normal context".to_string()));
        assert_eq!(context, Some("normal context".to_string()));

        // Long context gets truncated
        let long_context = "x".repeat(MAX_CONTEXT_STRING_LENGTH + 500);
        let context = validate_context_string(Some(long_context));
        assert!(context.is_some());
        assert_eq!(context.unwrap().len(), MAX_CONTEXT_STRING_LENGTH);

        // None stays None
        let context = validate_context_string(None);
        assert!(context.is_none());
    }

    #[test]
    fn test_calibration_recording_correct_logic() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        // Record: was uncertain (true) and good outcome (approved by sponsor)
        // This should count as "overcautious" - agent was too careful
        record_uncertainty_outcome(&mut agent, true, true);
        assert_eq!(agent.uncertainty_calibration.overcautious, 1);
        assert_eq!(agent.uncertainty_calibration.appropriate_uncertainty, 0);

        // Record: was uncertain (true) and bad outcome (rejected by sponsor)
        // This should count as "appropriate_uncertainty" - agent was right to escalate
        record_uncertainty_outcome(&mut agent, true, false);
        assert_eq!(agent.uncertainty_calibration.appropriate_uncertainty, 1);

        // Record: was confident (false) and good outcome
        // This should count as "appropriate_confidence"
        record_uncertainty_outcome(&mut agent, false, true);
        assert_eq!(agent.uncertainty_calibration.appropriate_confidence, 1);

        // Record: was confident (false) and bad outcome
        // This should count as "overconfident"
        record_uncertainty_outcome(&mut agent, false, false);
        assert_eq!(agent.uncertainty_calibration.overconfident, 1);
    }

    #[test]
    fn test_process_resolved_escalations_calibration() {
        let sponsor = valid_sponsor();
        let result = create_agent(&sponsor, AgentClass::Supervised, 5000, 1000);

        let mut agent = match result {
            CreateAgentResult::Success(a) => *a,
            _ => panic!("Expected success"),
        };

        // Add two escalations
        let esc1 = EscalationRequest::new(
            agent.agent_id.as_str(),
            MoralUncertainty::unsure(),
            "action_a",
            None,
        );
        let esc2 = EscalationRequest::new(
            agent.agent_id.as_str(),
            MoralUncertainty::unsure(),
            "action_b",
            None,
        );
        add_escalation(&mut agent, esc1).unwrap();
        add_escalation(&mut agent, esc2).unwrap();

        // Resolve: action_a approved (agent was overcautious), action_b rejected (appropriate)
        let resolutions = vec![
            ("action_a".to_string(), true),  // approved -> overcautious
            ("action_b".to_string(), false), // rejected -> appropriate
        ];

        let processed = process_resolved_escalations(&mut agent, &resolutions);
        assert_eq!(processed.len(), 2);
        assert!(processed.contains(&"action_a".to_string()));
        assert!(processed.contains(&"action_b".to_string()));

        // Check calibration was updated correctly (fixed bug: was_good_outcome not inverted)
        // approved=true means was_good_outcome=true -> overcautious
        // rejected=false means was_good_outcome=false -> appropriate_uncertainty
        assert_eq!(agent.uncertainty_calibration.overcautious, 1);
        assert_eq!(agent.uncertainty_calibration.appropriate_uncertainty, 1);
    }
}
