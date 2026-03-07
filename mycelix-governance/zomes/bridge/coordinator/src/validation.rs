use super::*;

// =============================================================================
// PURE VALIDATION FUNCTIONS (testable without HDK)
// =============================================================================

/// Validate consciousness snapshot input — pure function
pub fn check_snapshot_input(input: &RecordSnapshotInput) -> Result<(), String> {
    if input.consciousness_level < 0.0 || input.consciousness_level > 1.0 {
        return Err("Consciousness level must be between 0.0 and 1.0".into());
    }
    if input.meta_awareness < 0.0 || input.meta_awareness > 1.0 {
        return Err("Meta awareness must be between 0.0 and 1.0".into());
    }
    if input.self_model_accuracy < 0.0 || input.self_model_accuracy > 1.0 {
        return Err("Self model accuracy must be between 0.0 and 1.0".into());
    }
    if input.coherence < 0.0 || input.coherence > 1.0 {
        return Err("Coherence must be between 0.0 and 1.0".into());
    }
    if input.affective_valence < -1.0 || input.affective_valence > 1.0 {
        return Err("Affective valence must be between -1.0 and 1.0".into());
    }
    if input.care_activation < 0.0 || input.care_activation > 1.0 {
        return Err("Care activation must be between 0.0 and 1.0".into());
    }
    if let Some(ref source) = input.source {
        if source.is_empty() || source.len() > 256 {
            return Err("Source must be 1-256 characters".into());
        }
    }
    Ok(())
}

/// Validate broadcast event input — pure function
pub fn check_broadcast_event_input(input: &BroadcastGovernanceEventInput) -> Result<(), String> {
    if input.subject.is_empty() || input.subject.len() > 256 {
        return Err("Subject must be 1-256 characters".into());
    }
    if input.payload.len() > 4096 {
        return Err("Payload must be at most 4096 characters".into());
    }
    if let Some(ref proposal_id) = input.proposal_id {
        if proposal_id.is_empty() || proposal_id.len() > 256 {
            return Err("Proposal ID must be 1-256 characters".into());
        }
    }
    Ok(())
}

/// Validate execution request input — pure function
pub fn check_execution_request_input(input: &RequestExecutionInput) -> Result<(), String> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err("Proposal ID must be 1-256 characters".into());
    }
    if input.target_happ.is_empty() || input.target_happ.len() > 256 {
        return Err("Target hApp must be 1-256 characters".into());
    }
    if input.action.is_empty() || input.action.len() > 256 {
        return Err("Action must be 1-256 characters".into());
    }
    Ok(())
}

/// Validate alignment assessment input — pure function
pub fn check_alignment_input(input: &AssessAlignmentInput) -> Result<(), String> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err("Proposal ID must be 1-256 characters".into());
    }
    if input.proposal_content.is_empty() || input.proposal_content.len() > 4096 {
        return Err("Proposal content must be 1-4096 characters".into());
    }
    Ok(())
}

/// Validate weighted vote input — pure function
pub fn check_weighted_vote_input(input: &CastWeightedVoteInput) -> Result<(), String> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err("Proposal ID must be 1-256 characters".into());
    }
    if let Some(ref reason) = input.reason {
        if reason.len() > 4096 {
            return Err("Reason must be at most 4096 characters".into());
        }
    }
    Ok(())
}

/// Validate credit transfer input — pure function
pub fn check_transfer_credits_input(input: &TransferCreditsInput) -> Result<(), String> {
    if input.from.is_empty() || input.from.len() > 256 {
        return Err("From account must be 1-256 characters".into());
    }
    if input.to.is_empty() || input.to.len() > 256 {
        return Err("To account must be 1-256 characters".into());
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err("Amount must be positive and finite".into());
    }
    Ok(())
}

/// Validate federated reputation update — pure function
pub fn check_federated_reputation_input(input: &UpdateFederatedReputationInput) -> Result<(), String> {
    // Validate all f64 fields are in valid range if provided
    if let Some(v) = input.identity_verification {
        if !(0.0..=1.0).contains(&v) { return Err("identity_verification must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.credential_quality {
        if !(0.0..=1.0).contains(&v) { return Err("credential_quality must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.epistemic_contributions {
        if !(0.0..=1.0).contains(&v) { return Err("epistemic_contributions must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.factcheck_accuracy {
        if !(0.0..=1.0).contains(&v) { return Err("factcheck_accuracy must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.stake_weight {
        if !(0.0..=1.0).contains(&v) { return Err("stake_weight must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.payment_reliability {
        if !(0.0..=1.0).contains(&v) { return Err("payment_reliability must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.escrow_completion_rate {
        if !(0.0..=1.0).contains(&v) { return Err("escrow_completion_rate must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.pogq_score {
        if !(0.0..=1.0).contains(&v) { return Err("pogq_score must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.byzantine_clean_rate {
        if !(0.0..=1.0).contains(&v) { return Err("byzantine_clean_rate must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.voting_participation {
        if !(0.0..=1.0).contains(&v) { return Err("voting_participation must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.proposal_success_rate {
        if !(0.0..=1.0).contains(&v) { return Err("proposal_success_rate must be 0.0-1.0".into()); }
    }
    if let Some(v) = input.consensus_alignment {
        if !(0.0..=1.0).contains(&v) { return Err("consensus_alignment must be 0.0-1.0".into()); }
    }
    Ok(())
}
