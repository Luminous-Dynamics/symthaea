// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multidimensional finding-state transitions.
//!
//! Core invariants:
//!
//! ```text
//! authority != confirmation != replication != evidence disposition
//! promotion on one axis != promotion on another
//! high evidentiary support != causal identification
//! replication of association != replication of causal identification
//! robustness not evaluated != robustness established
//! transition admitted != policy authorization
//! current state != erased transition history
//! ```

/// Opaque 32-byte digest supplied by the receipt/canonicalization layer.
///
/// This crate intentionally does not choose a digest algorithm or provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Analytic authority carried by a systemic finding.
///
/// There is deliberately no policy/decision authorization state here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticAuthority {
    Recorded,
    Normalized,
    DerivedStructural,
    MechanismHypothesis,
    CausalCandidate,
    CausalIdentified,
    CounterfactualModel,
}

impl AnalyticAuthority {
    fn next(self) -> Option<Self> {
        match self {
            Self::Recorded => Some(Self::Normalized),
            Self::Normalized => Some(Self::DerivedStructural),
            Self::DerivedStructural => Some(Self::MechanismHypothesis),
            Self::MechanismHypothesis => Some(Self::CausalCandidate),
            Self::CausalCandidate => Some(Self::CausalIdentified),
            Self::CausalIdentified => Some(Self::CounterfactualModel),
            Self::CounterfactualModel => None,
        }
    }
}

/// Whether the inference was discovered and tested on the same evidence, or
/// evaluated under an explicit confirmatory regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceMode {
    Exploratory,
    ConfirmatoryPreSpecified,
    ConfirmatoryHoldout,
    PostSelectiveAdjusted,
}

/// Replication status. The replicated target must be stated in the transition
/// receipts outside this enum; replication alone never upgrades causal authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplicationState {
    Unreplicated,
    IndependentlyReplicated,
    FailedReplication,
    MixedReplication,
}

/// Evidence disposition for the finding as currently known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceDisposition {
    Unassessed,
    Supported,
    Contested,
    Refuted,
    Indeterminate,
}

/// Robustness dimensions are explicit because absence of an assessment must not
/// be confused with having passed that assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RobustnessDimension {
    Boundary,
    Missingness,
    NullModel,
    NegativeControls,
    Assumptions,
    Selection,
    SourceDependence,
    TemporalFreshness,
    Transportability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RobustnessOutcome {
    Robust,
    Sensitive,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustnessAssessment {
    dimension: RobustnessDimension,
    outcome: RobustnessOutcome,
    receipt_digests: Vec<Digest32>,
}

impl RobustnessAssessment {
    pub fn dimension(&self) -> RobustnessDimension {
        self.dimension
    }

    pub fn outcome(&self) -> RobustnessOutcome {
        self.outcome
    }

    pub fn receipt_digests(&self) -> &[Digest32] {
        &self.receipt_digests
    }
}

/// Snapshot of current finding state.
///
/// Fields are private by design: callers cannot promote analytic authority or
/// overwrite evidence state without an admitted transition. Historical events
/// remain external append-only evidence; this object is only their current fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindingState {
    state_digest: Digest32,
    finding_receipt_digest: Digest32,
    authority: AnalyticAuthority,
    inference_mode: InferenceMode,
    replication: ReplicationState,
    evidence: EvidenceDisposition,
    robustness: Vec<RobustnessAssessment>,
    genesis_receipts: Vec<Digest32>,
    as_known_at: i64,
}

impl FindingState {
    pub fn recorded(
        finding_receipt_digest: Digest32,
        state_digest: Digest32,
        as_known_at: i64,
    ) -> Self {
        Self {
            state_digest,
            finding_receipt_digest,
            authority: AnalyticAuthority::Recorded,
            inference_mode: InferenceMode::Exploratory,
            replication: ReplicationState::Unreplicated,
            evidence: EvidenceDisposition::Unassessed,
            robustness: Vec::new(),
            genesis_receipts: Vec::new(),
            as_known_at,
        }
    }

    /// Construct a finding whose confirmatory status was established before
    /// analysis. This is intentionally a genesis operation: exploratory work
    /// cannot later promote itself to `ConfirmatoryPreSpecified`.
    pub fn recorded_pre_specified(
        finding_receipt_digest: Digest32,
        preregistration_receipt_digest: Digest32,
        state_digest: Digest32,
        as_known_at: i64,
    ) -> Self {
        Self {
            state_digest,
            finding_receipt_digest,
            authority: AnalyticAuthority::Recorded,
            inference_mode: InferenceMode::ConfirmatoryPreSpecified,
            replication: ReplicationState::Unreplicated,
            evidence: EvidenceDisposition::Unassessed,
            robustness: Vec::new(),
            genesis_receipts: vec![preregistration_receipt_digest],
            as_known_at,
        }
    }

    pub fn state_digest(&self) -> Digest32 {
        self.state_digest
    }

    pub fn finding_receipt_digest(&self) -> Digest32 {
        self.finding_receipt_digest
    }

    pub fn authority(&self) -> AnalyticAuthority {
        self.authority
    }

    pub fn inference_mode(&self) -> InferenceMode {
        self.inference_mode
    }

    pub fn replication(&self) -> ReplicationState {
        self.replication
    }

    pub fn evidence(&self) -> EvidenceDisposition {
        self.evidence
    }

    pub fn genesis_receipts(&self) -> &[Digest32] {
        &self.genesis_receipts
    }

    pub fn as_known_at(&self) -> i64 {
        self.as_known_at
    }

    /// Returns `None` if this robustness dimension has not been evaluated.
    pub fn robustness(&self, dimension: RobustnessDimension) -> Option<&RobustnessAssessment> {
        self.robustness
            .iter()
            .find(|assessment| assessment.dimension == dimension)
    }

    pub fn apply_admitted_transition(
        &self,
        transition: &FindingTransitionEvidence,
        next_state_digest: Digest32,
    ) -> Result<Self, TransitionError> {
        if transition.from_state_digest != self.state_digest {
            return Err(TransitionError::StaleBaseState);
        }
        if transition.evaluated_at < self.as_known_at {
            return Err(TransitionError::NonMonotonicKnowledgeTime);
        }
        if next_state_digest == self.state_digest {
            return Err(TransitionError::UnchangedStateDigest);
        }
        if transition.rule_id.is_empty() || transition.evaluator_version.is_empty() {
            return Err(TransitionError::MissingRuleIdentity);
        }
        if transition.outcome != TransitionOutcome::Admitted {
            return Err(TransitionError::TransitionNotAdmitted);
        }
        if !transition.axis.matches(&transition.requested_state) {
            return Err(TransitionError::AxisStateMismatch);
        }

        let mut next = self.clone();
        next.state_digest = next_state_digest;
        next.as_known_at = transition.evaluated_at;

        match transition.requested_state {
            AxisState::Authority(target) => {
                if self.authority.next() != Some(target) {
                    return Err(TransitionError::IllegalAuthorityTransition);
                }
                require_receipts(transition)?;
                next.authority = target;
            }
            AxisState::InferenceMode(target) => {
                if !valid_inference_transition(self.inference_mode, target) {
                    return Err(TransitionError::IllegalInferenceTransition);
                }
                require_receipts(transition)?;
                next.inference_mode = target;
            }
            AxisState::Replication(target) => {
                if !valid_replication_transition(self.replication, target) {
                    return Err(TransitionError::IllegalReplicationTransition);
                }
                require_receipts(transition)?;
                next.replication = target;
            }
            AxisState::Evidence(target) => {
                if target == self.evidence {
                    return Err(TransitionError::NoStateChange);
                }
                if target == EvidenceDisposition::Unassessed {
                    return Err(TransitionError::IllegalEvidenceTransition);
                }
                require_receipts(transition)?;
                next.evidence = target;
            }
            AxisState::Robustness { dimension, outcome } => {
                require_receipts(transition)?;
                let assessment = RobustnessAssessment {
                    dimension,
                    outcome,
                    receipt_digests: transition.required_receipts.clone(),
                };
                if let Some(existing) = next
                    .robustness
                    .iter_mut()
                    .find(|assessment| assessment.dimension == dimension)
                {
                    if *existing == assessment {
                        return Err(TransitionError::NoStateChange);
                    }
                    *existing = assessment;
                } else {
                    next.robustness.push(assessment);
                }
            }
        }

        Ok(next)
    }
}

fn require_receipts(transition: &FindingTransitionEvidence) -> Result<(), TransitionError> {
    if transition.required_receipts.is_empty() {
        Err(TransitionError::MissingRequiredReceipt)
    } else {
        Ok(())
    }
}

fn valid_inference_transition(from: InferenceMode, to: InferenceMode) -> bool {
    match (from, to) {
        // A pre-specified analysis must be declared as such before discovery;
        // it cannot be retroactively promoted from exploratory work.
        (InferenceMode::Exploratory, InferenceMode::ConfirmatoryPreSpecified) => false,
        (InferenceMode::Exploratory, InferenceMode::ConfirmatoryHoldout) => true,
        (InferenceMode::Exploratory, InferenceMode::PostSelectiveAdjusted) => true,
        (InferenceMode::ConfirmatoryPreSpecified, InferenceMode::ConfirmatoryHoldout) => true,
        (InferenceMode::ConfirmatoryPreSpecified, InferenceMode::PostSelectiveAdjusted) => true,
        (InferenceMode::ConfirmatoryHoldout, InferenceMode::PostSelectiveAdjusted) => true,
        _ => false,
    }
}

fn valid_replication_transition(from: ReplicationState, to: ReplicationState) -> bool {
    match (from, to) {
        (ReplicationState::Unreplicated, ReplicationState::IndependentlyReplicated) => true,
        (ReplicationState::Unreplicated, ReplicationState::FailedReplication) => true,
        (ReplicationState::Unreplicated, ReplicationState::MixedReplication) => true,
        (ReplicationState::IndependentlyReplicated, ReplicationState::MixedReplication) => true,
        (ReplicationState::FailedReplication, ReplicationState::MixedReplication) => true,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionAxis {
    Authority,
    InferenceMode,
    Replication,
    Evidence,
    Robustness(RobustnessDimension),
}

impl TransitionAxis {
    fn matches(self, requested: &AxisState) -> bool {
        match (self, requested) {
            (Self::Authority, AxisState::Authority(_)) => true,
            (Self::InferenceMode, AxisState::InferenceMode(_)) => true,
            (Self::Replication, AxisState::Replication(_)) => true,
            (Self::Evidence, AxisState::Evidence(_)) => true,
            (
                Self::Robustness(expected),
                AxisState::Robustness {
                    dimension: actual, ..
                },
            ) => expected == *actual,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisState {
    Authority(AnalyticAuthority),
    InferenceMode(InferenceMode),
    Replication(ReplicationState),
    Evidence(EvidenceDisposition),
    Robustness {
        dimension: RobustnessDimension,
        outcome: RobustnessOutcome,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionOutcome {
    Admitted,
    Rejected,
    Indeterminate,
}

/// Proof-carrying request/result for a single-axis finding-state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindingTransitionEvidence {
    pub from_state_digest: Digest32,
    pub axis: TransitionAxis,
    pub requested_state: AxisState,
    pub rule_id: String,
    pub required_receipts: Vec<Digest32>,
    pub evaluator_version: String,
    pub evaluated_at: i64,
    pub outcome: TransitionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    StaleBaseState,
    NonMonotonicKnowledgeTime,
    UnchangedStateDigest,
    MissingRuleIdentity,
    TransitionNotAdmitted,
    AxisStateMismatch,
    MissingRequiredReceipt,
    NoStateChange,
    IllegalAuthorityTransition,
    IllegalInferenceTransition,
    IllegalReplicationTransition,
    IllegalEvidenceTransition,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn state() -> FindingState {
        FindingState::recorded(d(1), d(2), 100)
    }

    fn event(
        from_state_digest: Digest32,
        axis: TransitionAxis,
        requested_state: AxisState,
        evaluated_at: i64,
    ) -> FindingTransitionEvidence {
        FindingTransitionEvidence {
            from_state_digest,
            axis,
            requested_state,
            rule_id: "test-rule-v1".into(),
            required_receipts: vec![d(9)],
            evaluator_version: "test-evaluator-v1".into(),
            evaluated_at,
            outcome: TransitionOutcome::Admitted,
        }
    }

    #[test]
    fn authority_transition_changes_only_authority_axis() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Authority,
            AxisState::Authority(AnalyticAuthority::Normalized),
            200,
        );
        let next = initial.apply_admitted_transition(&transition, d(3)).unwrap();

        assert_eq!(next.authority(), AnalyticAuthority::Normalized);
        assert_eq!(next.inference_mode(), initial.inference_mode());
        assert_eq!(next.replication(), initial.replication());
        assert_eq!(next.evidence(), initial.evidence());
        assert!(next.robustness(RobustnessDimension::Boundary).is_none());
    }

    #[test]
    fn state_axes_cannot_be_mutated_without_transition_api() {
        let initial = state();
        assert_eq!(initial.authority(), AnalyticAuthority::Recorded);
        assert_eq!(initial.evidence(), EvidenceDisposition::Unassessed);
    }

    #[test]
    fn authority_cannot_skip_intermediate_states() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Authority,
            AxisState::Authority(AnalyticAuthority::CausalIdentified),
            200,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::IllegalAuthorityTransition)
        );
    }

    #[test]
    fn authority_promotion_requires_receipt_evidence() {
        let initial = state();
        let mut transition = event(
            initial.state_digest(),
            TransitionAxis::Authority,
            AxisState::Authority(AnalyticAuthority::Normalized),
            200,
        );
        transition.required_receipts.clear();
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::MissingRequiredReceipt)
        );
    }

    #[test]
    fn pre_specified_mode_requires_genesis_receipt() {
        let initial = FindingState::recorded_pre_specified(d(1), d(7), d(2), 100);
        assert_eq!(
            initial.inference_mode(),
            InferenceMode::ConfirmatoryPreSpecified
        );
        assert_eq!(initial.genesis_receipts(), &[d(7)]);
    }

    #[test]
    fn exploratory_work_cannot_be_retroactively_called_pre_specified() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::InferenceMode,
            AxisState::InferenceMode(InferenceMode::ConfirmatoryPreSpecified),
            200,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::IllegalInferenceTransition)
        );
    }

    #[test]
    fn holdout_confirmation_does_not_upgrade_causal_authority() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::InferenceMode,
            AxisState::InferenceMode(InferenceMode::ConfirmatoryHoldout),
            200,
        );
        let next = initial.apply_admitted_transition(&transition, d(3)).unwrap();

        assert_eq!(next.inference_mode(), InferenceMode::ConfirmatoryHoldout);
        assert_eq!(next.authority(), AnalyticAuthority::Recorded);
    }

    #[test]
    fn independent_replication_does_not_change_evidence_or_authority() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Replication,
            AxisState::Replication(ReplicationState::IndependentlyReplicated),
            200,
        );
        let next = initial.apply_admitted_transition(&transition, d(3)).unwrap();

        assert_eq!(next.replication(), ReplicationState::IndependentlyReplicated);
        assert_eq!(next.authority(), initial.authority());
        assert_eq!(next.evidence(), initial.evidence());
    }

    #[test]
    fn conflicting_replication_becomes_mixed_instead_of_overwrite() {
        let initial = state();
        let first = event(
            initial.state_digest(),
            TransitionAxis::Replication,
            AxisState::Replication(ReplicationState::IndependentlyReplicated),
            200,
        );
        let replicated = initial.apply_admitted_transition(&first, d(3)).unwrap();
        let second = event(
            replicated.state_digest(),
            TransitionAxis::Replication,
            AxisState::Replication(ReplicationState::FailedReplication),
            300,
        );
        assert_eq!(
            replicated.apply_admitted_transition(&second, d(4)),
            Err(TransitionError::IllegalReplicationTransition)
        );

        let mixed = event(
            replicated.state_digest(),
            TransitionAxis::Replication,
            AxisState::Replication(ReplicationState::MixedReplication),
            300,
        );
        assert_eq!(
            replicated
                .apply_admitted_transition(&mixed, d(4))
                .unwrap()
                .replication(),
            ReplicationState::MixedReplication
        );
    }

    #[test]
    fn stale_transition_cannot_apply_to_newer_state() {
        let initial = state();
        let transition = event(
            d(8),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Supported),
            200,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::StaleBaseState)
        );
    }

    #[test]
    fn knowledge_time_cannot_move_backward() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Supported),
            99,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::NonMonotonicKnowledgeTime)
        );
    }

    #[test]
    fn state_change_requires_new_state_digest() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Supported),
            200,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, initial.state_digest()),
            Err(TransitionError::UnchangedStateDigest)
        );
    }

    #[test]
    fn axis_target_mismatch_fails_closed() {
        let initial = state();
        let transition = event(
            initial.state_digest(),
            TransitionAxis::Authority,
            AxisState::Evidence(EvidenceDisposition::Supported),
            200,
        );
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::AxisStateMismatch)
        );
    }

    #[test]
    fn unevaluated_robustness_is_not_robustness_pass() {
        assert!(state().robustness(RobustnessDimension::Boundary).is_none());
    }

    #[test]
    fn robustness_evaluation_requires_receipts_and_changes_only_one_dimension() {
        let initial = state();
        let mut transition = event(
            initial.state_digest(),
            TransitionAxis::Robustness(RobustnessDimension::Boundary),
            AxisState::Robustness {
                dimension: RobustnessDimension::Boundary,
                outcome: RobustnessOutcome::Sensitive,
            },
            200,
        );
        transition.required_receipts.clear();
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::MissingRequiredReceipt)
        );

        transition.required_receipts.push(d(9));
        let next = initial.apply_admitted_transition(&transition, d(3)).unwrap();
        assert_eq!(
            next.robustness(RobustnessDimension::Boundary)
                .unwrap()
                .outcome(),
            RobustnessOutcome::Sensitive
        );
        assert_eq!(next.authority(), initial.authority());
        assert_eq!(next.inference_mode(), initial.inference_mode());
    }

    #[test]
    fn evidence_cannot_return_to_unassessed() {
        let initial = state();
        let supported = event(
            initial.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Supported),
            200,
        );
        let supported_state = initial
            .apply_admitted_transition(&supported, d(3))
            .unwrap();
        let erase = event(
            supported_state.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Unassessed),
            300,
        );
        assert_eq!(
            supported_state.apply_admitted_transition(&erase, d(4)),
            Err(TransitionError::IllegalEvidenceTransition)
        );
    }

    #[test]
    fn evidence_can_become_contested_without_erasing_authority() {
        let initial = state();
        let normalize = event(
            initial.state_digest(),
            TransitionAxis::Authority,
            AxisState::Authority(AnalyticAuthority::Normalized),
            150,
        );
        let normalized = initial
            .apply_admitted_transition(&normalize, d(3))
            .unwrap();
        let structural = event(
            normalized.state_digest(),
            TransitionAxis::Authority,
            AxisState::Authority(AnalyticAuthority::DerivedStructural),
            175,
        );
        let structural_state = normalized
            .apply_admitted_transition(&structural, d(4))
            .unwrap();
        let contest = event(
            structural_state.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Contested),
            200,
        );
        let next = structural_state
            .apply_admitted_transition(&contest, d(5))
            .unwrap();

        assert_eq!(next.evidence(), EvidenceDisposition::Contested);
        assert_eq!(next.authority(), AnalyticAuthority::DerivedStructural);
    }

    #[test]
    fn rejected_transition_never_changes_state() {
        let initial = state();
        let mut transition = event(
            initial.state_digest(),
            TransitionAxis::Evidence,
            AxisState::Evidence(EvidenceDisposition::Supported),
            200,
        );
        transition.outcome = TransitionOutcome::Rejected;
        assert_eq!(
            initial.apply_admitted_transition(&transition, d(3)),
            Err(TransitionError::TransitionNotAdmitted)
        );
    }
}