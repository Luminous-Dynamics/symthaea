//! Fail-closed tranche release gates for staged commitments.
//!
//! This is an authorization and evidence gate, not a valuation of real-option
//! value. It makes the decision to release the next tranche explicit and
//! replayable; it does not predict whether the project will succeed.

use crate::EvidenceClaim;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrancheRequest {
    pub tranche_id: String,
    /// Positive integer minor currency units to release now.
    pub amount_minor: i64,
    /// Capital already committed to this project before the requested tranche.
    pub previously_committed_minor: i64,
    /// Maximum cumulative project commitment under the approved mandate.
    pub total_commitment_cap_minor: i64,
    pub evaluation_day: u32,
    /// Must be established by the authorized stage-review process.
    pub milestone_satisfied: bool,
    pub authority_granted: bool,
    /// At least one qualified, in-scope item is required for every tranche.
    pub milestone_evidence: Vec<EvidenceClaim>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrancheRejectionReason {
    AmountMustBePositive,
    InvalidPriorCommitment,
    InvalidCommitmentCap,
    CommitmentCapExceeded,
    ArithmeticOverflow,
    MilestoneNotSatisfied,
    Unauthorized,
    MilestoneEvidenceMissingOrUnusable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrancheDecision {
    pub tranche_id: String,
    pub approved: bool,
    pub rejection_reasons: Vec<TrancheRejectionReason>,
    pub requested_release_minor: i64,
    /// None only when cumulative commitment arithmetic overflows.
    pub cumulative_commitment_after_minor: Option<i64>,
}

impl TrancheRequest {
    /// Approves a tranche only when every hard gate passes. A positive projected
    /// return or a previous sunk cost cannot compensate for a failed gate.
    pub fn evaluate_release(&self) -> TrancheDecision {
        let mut rejection_reasons = Vec::new();

        if self.amount_minor <= 0 {
            rejection_reasons.push(TrancheRejectionReason::AmountMustBePositive);
        }
        if self.previously_committed_minor < 0 {
            rejection_reasons.push(TrancheRejectionReason::InvalidPriorCommitment);
        }
        if self.total_commitment_cap_minor < 0
            || self.previously_committed_minor > self.total_commitment_cap_minor
        {
            rejection_reasons.push(TrancheRejectionReason::InvalidCommitmentCap);
        }

        let cumulative_commitment_after_minor = self
            .previously_committed_minor
            .checked_add(self.amount_minor);

        match cumulative_commitment_after_minor {
            None => rejection_reasons.push(TrancheRejectionReason::ArithmeticOverflow),
            Some(total) if total > self.total_commitment_cap_minor => {
                rejection_reasons.push(TrancheRejectionReason::CommitmentCapExceeded);
            }
            Some(_) => {}
        }

        if !self.milestone_satisfied {
            rejection_reasons.push(TrancheRejectionReason::MilestoneNotSatisfied);
        }
        if !self.authority_granted {
            rejection_reasons.push(TrancheRejectionReason::Unauthorized);
        }
        if self.milestone_evidence.is_empty()
            || self
                .milestone_evidence
                .iter()
                .any(|claim| !claim.is_usable(self.evaluation_day))
        {
            rejection_reasons.push(TrancheRejectionReason::MilestoneEvidenceMissingOrUnusable);
        }

        TrancheDecision {
            tranche_id: self.tranche_id.clone(),
            approved: rejection_reasons.is_empty(),
            rejection_reasons,
            requested_release_minor: self.amount_minor,
            cumulative_commitment_after_minor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvidenceState;

    fn evidence(state: EvidenceState) -> EvidenceClaim {
        EvidenceClaim {
            id: "stage-report-01".into(),
            state,
            source_id: Some("synthetic-stage-evidence-v1".into()),
            observed_day: Some(4),
            valid_through_day: Some(12),
        }
    }

    fn request() -> TrancheRequest {
        TrancheRequest {
            tranche_id: "pilot-stage-2".into(),
            amount_minor: 2_000,
            previously_committed_minor: 3_000,
            total_commitment_cap_minor: 10_000,
            evaluation_day: 8,
            milestone_satisfied: true,
            authority_granted: true,
            milestone_evidence: vec![evidence(EvidenceState::Qualified)],
        }
    }

    #[test]
    fn qualified_milestone_inside_cap_approves_next_tranche() {
        let decision = request().evaluate_release();
        assert!(decision.approved);
        assert!(decision.rejection_reasons.is_empty());
        assert_eq!(decision.cumulative_commitment_after_minor, Some(5_000));
    }

    #[test]
    fn failed_milestone_blocks_release_even_with_qualified_evidence() {
        let mut request = request();
        request.milestone_satisfied = false;
        let decision = request.evaluate_release();
        assert!(!decision.approved);
        assert!(
            decision
                .rejection_reasons
                .contains(&TrancheRejectionReason::MilestoneNotSatisfied)
        );
    }

    #[test]
    fn stale_milestone_evidence_blocks_release() {
        let mut request = request();
        request.milestone_evidence = vec![evidence(EvidenceState::Stale)];
        let decision = request.evaluate_release();
        assert!(!decision.approved);
        assert!(
            decision
                .rejection_reasons
                .contains(&TrancheRejectionReason::MilestoneEvidenceMissingOrUnusable)
        );
    }

    #[test]
    fn commitment_cap_cannot_be_exceeded_by_a_tranche() {
        let mut request = request();
        request.previously_committed_minor = 9_000;
        request.amount_minor = 1_001;
        let decision = request.evaluate_release();
        assert!(!decision.approved);
        assert!(
            decision
                .rejection_reasons
                .contains(&TrancheRejectionReason::CommitmentCapExceeded)
        );
    }

    #[test]
    fn missing_authority_blocks_release() {
        let mut request = request();
        request.authority_granted = false;
        let decision = request.evaluate_release();
        assert!(!decision.approved);
        assert!(
            decision
                .rejection_reasons
                .contains(&TrancheRejectionReason::Unauthorized)
        );
    }

    #[test]
    fn integer_overflow_is_fail_closed() {
        let mut request = request();
        request.previously_committed_minor = i64::MAX - 2;
        request.total_commitment_cap_minor = i64::MAX;
        request.amount_minor = 10;
        let decision = request.evaluate_release();
        assert!(!decision.approved);
        assert_eq!(decision.cumulative_commitment_after_minor, None);
        assert!(
            decision
                .rejection_reasons
                .contains(&TrancheRejectionReason::ArithmeticOverflow)
        );
    }
}
