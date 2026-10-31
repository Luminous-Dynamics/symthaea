#![forbid(unsafe_code)]

//! Standalone research harness for proof-obligation-driven metareasoning.
//! It is deliberately independent of the production COG-META ABI and is not
//! a production scheduler or a formal/legal proof checker.

pub mod graph;
pub mod poar;
pub mod recovery;
pub mod recovery_journal;

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObligationState {
    Open,
    CandidateEvidenceAvailable,
    SupportedUnderProfile,
    Contested,
    Contradicted,
    StaleOrRetracted,
    OutOfScope,
    Unauthorized,
    VerifierRejected,
    UnresolvableWithoutExternalInput,
    BudgetInsufficient,
}

impl ObligationState {
    fn unresolved(self) -> bool {
        self != Self::SupportedUnderProfile
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obligation {
    pub id: String,
    pub state: ObligationState,
    /// Task-profile priority only; it does not change proof validity.
    pub consequence_priority: u8,
    /// Identity of the exact evidence/proof workspace state this obligation belongs to.
    /// Empty means unversioned and cannot accept a promoting verifier receipt.
    pub state_identity: String,
}

impl Obligation {
    pub fn new(
        id: impl Into<String>,
        state: ObligationState,
        consequence_priority: u8,
    ) -> Self {
        Self { id: id.into(), state, consequence_priority, state_identity: String::new() }
    }

    /// Binds this obligation to an exact pre-operation workspace/evidence state.
    ///
    /// The lab treats this as an opaque synthetic identity. A production adapter
    /// must derive it from canonical content and dependency identities.
    pub fn with_state_identity(mut self, state_identity: impl Into<String>) -> Self {
        self.state_identity = state_identity.into();
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationKind {
    Retrieve,
    Verify,
    SeekIndependentEvidence,
    RepairDerivation,
    RefreshEvidence,
    CompareAlternatives,
    RequestClarification,
    RequestExternalEvidence,
}

impl OperationKind {
    fn can_address(self, state: ObligationState) -> bool {
        match state {
            ObligationState::Open =>
                matches!(self, Self::Retrieve | Self::RequestExternalEvidence),
            ObligationState::CandidateEvidenceAvailable =>
                matches!(self, Self::Verify | Self::CompareAlternatives),
            ObligationState::SupportedUnderProfile => false,
            ObligationState::Contested =>
                matches!(self, Self::SeekIndependentEvidence | Self::CompareAlternatives),
            ObligationState::Contradicted | ObligationState::VerifierRejected =>
                matches!(self, Self::RepairDerivation | Self::SeekIndependentEvidence),
            ObligationState::StaleOrRetracted =>
                matches!(self, Self::RefreshEvidence | Self::Retrieve),
            ObligationState::OutOfScope | ObligationState::Unauthorized =>
                matches!(self, Self::Retrieve | Self::RequestClarification),
            ObligationState::UnresolvableWithoutExternalInput =>
                matches!(self, Self::RequestExternalEvidence | Self::RequestClarification),
            ObligationState::BudgetInsufficient =>
                matches!(self, Self::CompareAlternatives | Self::Retrieve),
        }
    }

    fn tie_break_rank(self) -> u8 {
        match self {
            Self::Verify => 0,
            Self::RefreshEvidence => 1,
            Self::SeekIndependentEvidence => 2,
            Self::RepairDerivation => 3,
            Self::Retrieve => 4,
            Self::CompareAlternatives => 5,
            Self::RequestClarification => 6,
            Self::RequestExternalEvidence => 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchedulingClass {
    Mandatory,
    Competitive,
    Opportunistic,
}

impl SchedulingClass {
    fn rank(self) -> u8 {
        match self {
            Self::Mandatory => 0,
            Self::Competitive => 1,
            Self::Opportunistic => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateOperation {
    pub id: String,
    pub kind: OperationKind,
    pub obligation_ids: Vec<String>,
    pub scheduling_class: SchedulingClass,
    pub estimated_work_units: u64,
    /// Relevant pre-operation state identity, not a cryptographic digest.
    pub progress_key: String,
    pub applicable: bool,
}

impl CandidateOperation {
    pub fn new<I, S>(
        id: impl Into<String>,
        kind: OperationKind,
        obligation_ids: I,
        scheduling_class: SchedulingClass,
        estimated_work_units: u64,
        progress_key: impl Into<String>,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: id.into(),
            kind,
            obligation_ids: obligation_ids.into_iter().map(Into::into).collect(),
            scheduling_class,
            estimated_work_units,
            progress_key: progress_key.into(),
            applicable: true,
        }
    }

    pub fn inapplicable(mut self) -> Self {
        self.applicable = false;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectionReason {
    Inapplicable,
    UnknownObligation,
    NoUnresolvedTarget,
    OperationDoesNotAddressState,
    RepeatedWithoutSemanticProgress,
    WorkBudgetInsufficient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRejection {
    pub operation_id: String,
    pub reason: RejectionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    AllObligationsDischarged,
    NoSemanticProgress,
    BudgetExhausted,
    MandatoryWorkCannotFit,
    NeedsExternalEvidence,
    NoApplicableOperation,
    InvalidInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    Run {
        operation_id: String,
        kind: OperationKind,
        obligation_ids: Vec<String>,
        progress_key: String,
        work_units: u64,
    },
    Stop(StopReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionDecision {
    pub selection: Selection,
    pub rejected: Vec<CandidateRejection>,
    pub competitive_work_limit: u64,
}

pub struct SelectionContext<'a> {
    pub obligations: &'a [Obligation],
    pub operations: &'a [CandidateOperation],
    pub attempted_without_progress: &'a BTreeSet<(String, String)>,
    pub remaining_work_units: u64,
    pub mandatory_reserve_units: u64,
}

#[derive(Debug, Clone)]
struct EligibleOperation {
    id: String,
    kind: OperationKind,
    obligation_ids: Vec<String>,
    scheduling_class: SchedulingClass,
    work_units: u64,
    progress_key: String,
    max_priority: u8,
}

pub fn choose_next(context: &SelectionContext<'_>) -> SelectionDecision {
    let competitive_work_limit = context.remaining_work_units
        .saturating_sub(context.mandatory_reserve_units);

    let mut seen_obligations = BTreeSet::new();
    if context.obligations.iter().any(|o| o.id.is_empty() || !seen_obligations.insert(o.id.as_str())) {
        return SelectionDecision {
            selection: Selection::Stop(StopReason::InvalidInput),
            rejected: Vec::new(),
            competitive_work_limit,
        };
    }

    let mut seen_operations = BTreeSet::new();
    for operation in context.operations {
        let mut seen_targets = BTreeSet::new();
        if operation.id.is_empty()
            || operation.progress_key.is_empty()
            || operation.obligation_ids.is_empty()
            || !seen_operations.insert(operation.id.as_str())
            || operation.obligation_ids.iter().any(|id| id.is_empty() || !seen_targets.insert(id.as_str()))
        {
            return SelectionDecision {
                selection: Selection::Stop(StopReason::InvalidInput),
                rejected: Vec::new(),
                competitive_work_limit,
            };
        }
    }

    let by_id: BTreeMap<&str, &Obligation> = context.obligations
        .iter()
        .map(|o| (o.id.as_str(), o))
        .collect();

    if context.obligations.iter().all(|o| !o.state.unresolved()) {
        return SelectionDecision {
            selection: Selection::Stop(StopReason::AllObligationsDischarged),
            rejected: Vec::new(),
            competitive_work_limit,
        };
    }

    let mut eligible = Vec::<EligibleOperation>::new();
    let mut rejected = Vec::<CandidateRejection>::new();
    let mut mandatory_unaffordable = false;
    let mut budget_rejections = 0usize;
    let mut repeat_rejections = 0usize;

    for operation in context.operations {
        let reject = |reason| CandidateRejection {
            operation_id: operation.id.clone(),
            reason,
        };

        if !operation.applicable {
            rejected.push(reject(RejectionReason::Inapplicable));
            continue;
        }
        if operation.obligation_ids.is_empty()
            || operation.obligation_ids.iter().any(|id| !by_id.contains_key(id.as_str()))
        {
            rejected.push(reject(RejectionReason::UnknownObligation));
            continue;
        }
        if context.attempted_without_progress
            .contains(&(operation.id.clone(), operation.progress_key.clone()))
        {
            repeat_rejections += 1;
            rejected.push(reject(RejectionReason::RepeatedWithoutSemanticProgress));
            continue;
        }

        let mut targets = Vec::new();
        let mut max_priority = 0u8;
        for id in &operation.obligation_ids {
            let obligation = by_id.get(id.as_str()).expect("validated above");
            if obligation.state.unresolved() && operation.kind.can_address(obligation.state) {
                targets.push(id.clone());
                max_priority = max_priority.max(obligation.consequence_priority);
            }
        }

        if targets.is_empty() {
            rejected.push(reject(RejectionReason::OperationDoesNotAddressState));
            continue;
        }

        targets.sort();
        targets.dedup();

        let work_limit = match operation.scheduling_class {
            SchedulingClass::Mandatory => context.remaining_work_units,
            SchedulingClass::Competitive | SchedulingClass::Opportunistic =>
                competitive_work_limit,
        };
        if operation.estimated_work_units > work_limit {
            budget_rejections += 1;
            if operation.scheduling_class == SchedulingClass::Mandatory {
                mandatory_unaffordable = true;
            }
            rejected.push(reject(RejectionReason::WorkBudgetInsufficient));
            continue;
        }

        eligible.push(EligibleOperation {
            id: operation.id.clone(),
            kind: operation.kind,
            obligation_ids: targets,
            scheduling_class: operation.scheduling_class,
            work_units: operation.estimated_work_units,
            progress_key: operation.progress_key.clone(),
            max_priority,
        });
    }

    rejected.sort_by(|a, b| {
        a.operation_id.cmp(&b.operation_id).then_with(|| a.reason.cmp(&b.reason))
    });

    if mandatory_unaffordable
        && eligible.iter().all(|op| op.scheduling_class != SchedulingClass::Mandatory)
    {
        return SelectionDecision {
            selection: Selection::Stop(StopReason::MandatoryWorkCannotFit),
            rejected,
            competitive_work_limit,
        };
    }

    if eligible.is_empty() {
        let stop = if repeat_rejections > 0 && budget_rejections == 0 {
            StopReason::NoSemanticProgress
        } else if budget_rejections > 0 {
            StopReason::BudgetExhausted
        } else if context.obligations.iter()
            .filter(|o| o.state.unresolved())
            .all(|o| o.state == ObligationState::UnresolvableWithoutExternalInput)
        {
            StopReason::NeedsExternalEvidence
        } else {
            StopReason::NoApplicableOperation
        };
        return SelectionDecision {
            selection: Selection::Stop(stop),
            rejected,
            competitive_work_limit,
        };
    }

    eligible.sort_by(|a, b| {
        a.scheduling_class.rank().cmp(&b.scheduling_class.rank())
            .then_with(|| b.max_priority.cmp(&a.max_priority))
            .then_with(|| b.obligation_ids.len().cmp(&a.obligation_ids.len()))
            .then_with(|| a.work_units.cmp(&b.work_units))
            .then_with(|| a.kind.tie_break_rank().cmp(&b.kind.tie_break_rank()))
            .then_with(|| a.id.cmp(&b.id))
    });

    let selected = eligible.remove(0);
    SelectionDecision {
        selection: Selection::Run {
            operation_id: selected.id,
            kind: selected.kind,
            obligation_ids: selected.obligation_ids,
            progress_key: selected.progress_key,
            work_units: selected.work_units,
        },
        rejected,
        competitive_work_limit,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationStatus {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofVerdict {
    Valid,
    Invalid,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceVerdict {
    Admissible,
    StaleOrRetracted,
    OutOfScope,
    Unauthorized,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierReceipt {
    pub verifier_id: String,
    pub operation_id: String,
    pub progress_key: String,
    pub obligation_ids: BTreeSet<String>,
    pub proof_profile_id: String,
    /// Opaque identity only; this lab does not verify signatures.
    pub proof_identity: String,
    /// New state identity after applying a state-changing verdict.
    /// This must differ from the input progress key when state changes.
    pub result_state_identity: String,
    pub proof_verdict: ProofVerdict,
    pub evidence_verdict: EvidenceVerdict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationReceipt {
    pub producer_id: String,
    pub operation_id: String,
    pub progress_key: String,
    pub obligation_ids: BTreeSet<String>,
    pub status: OperationStatus,
    pub verifier: Option<VerifierReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    EmptyOperationId,
    EmptyProducerIdentity,
    EmptyTargetSet,
    UnknownObligation(String),
    DuplicateExistingObligation(String),
    EmptyVerifierIdentity,
    EmptyProfileIdentity,
    EmptyProofIdentity,
    EmptyResultStateIdentity,
    MissingCurrentStateIdentity(String),
    ResultStateIdentityUnchanged,
    OperationBindingMismatch,
    ObligationBindingMismatch,
    ProgressBindingMismatch,
    VerifierNotDistinctFromProducer,
}

pub fn apply_operation_receipt(
    obligations: &mut [Obligation],
    receipt: &OperationReceipt,
) -> Result<usize, ReceiptError> {
    if receipt.operation_id.is_empty() {
        return Err(ReceiptError::EmptyOperationId);
    }
    if receipt.producer_id.is_empty() {
        return Err(ReceiptError::EmptyProducerIdentity);
    }
    if receipt.progress_key.is_empty() {
        return Err(ReceiptError::ProgressBindingMismatch);
    }
    if receipt.obligation_ids.is_empty() {
        return Err(ReceiptError::EmptyTargetSet);
    }

    if receipt.status != OperationStatus::Completed {
        return Ok(0);
    }
    let Some(verifier) = receipt.verifier.as_ref() else {
        return Ok(0);
    };

    if verifier.verifier_id.is_empty() {
        return Err(ReceiptError::EmptyVerifierIdentity);
    }
    if verifier.proof_profile_id.is_empty() {
        return Err(ReceiptError::EmptyProfileIdentity);
    }
    if verifier.proof_identity.is_empty() {
        return Err(ReceiptError::EmptyProofIdentity);
    }
    if verifier.result_state_identity.is_empty() {
        return Err(ReceiptError::EmptyResultStateIdentity);
    }
    if verifier.verifier_id == receipt.producer_id {
        return Err(ReceiptError::VerifierNotDistinctFromProducer);
    }
    if verifier.operation_id != receipt.operation_id {
        return Err(ReceiptError::OperationBindingMismatch);
    }
    if verifier.progress_key != receipt.progress_key {
        return Err(ReceiptError::ProgressBindingMismatch);
    }
    if verifier.obligation_ids != receipt.obligation_ids {
        return Err(ReceiptError::ObligationBindingMismatch);
    }
    let mut existing_ids = BTreeSet::new();
    for obligation in obligations.iter() {
        if obligation.id.is_empty() {
            return Err(ReceiptError::UnknownObligation(String::new()));
        }
        if !existing_ids.insert(obligation.id.as_str()) {
            return Err(ReceiptError::DuplicateExistingObligation(obligation.id.clone()));
        }
    }
    for id in &receipt.obligation_ids {
        if !existing_ids.contains(id.as_str()) {
            return Err(ReceiptError::UnknownObligation(id.clone()));
        }
    }

    // Bind the receipt to the current state, not merely to a verifier receipt
    // that repeats the same caller-supplied key. This is the anti-replay gate.
    for obligation in obligations.iter().filter(|o| receipt.obligation_ids.contains(&o.id)) {
        if obligation.state_identity.is_empty() {
            return Err(ReceiptError::MissingCurrentStateIdentity(obligation.id.clone()));
        }
        if obligation.state_identity != receipt.progress_key {
            return Err(ReceiptError::ProgressBindingMismatch);
        }
    }

    // Stage all transitions before mutating anything, preserving transactional
    // behavior if a state-identity check fails.
    let mut transitions = Vec::<(usize, ObligationState)>::new();
    for (index, obligation) in obligations.iter().enumerate() {
        if !receipt.obligation_ids.contains(&obligation.id) {
            continue;
        }
        let next = match (verifier.evidence_verdict, verifier.proof_verdict) {
            (EvidenceVerdict::StaleOrRetracted, _) => ObligationState::StaleOrRetracted,
            (EvidenceVerdict::OutOfScope, _) => ObligationState::OutOfScope,
            (EvidenceVerdict::Unauthorized, _) => ObligationState::Unauthorized,
            (EvidenceVerdict::Admissible, ProofVerdict::Valid) =>
                ObligationState::SupportedUnderProfile,
            (EvidenceVerdict::Admissible, ProofVerdict::Invalid) =>
                ObligationState::VerifierRejected,
            _ => obligation.state,
        };
        if obligation.state != next {
            transitions.push((index, next));
        }
    }

    if !transitions.is_empty() && verifier.result_state_identity == receipt.progress_key {
        return Err(ReceiptError::ResultStateIdentityUnchanged);
    }

    for (index, next) in &transitions {
        obligations[*index].state = *next;
        obligations[*index].state_identity = verifier.result_state_identity.clone();
    }
    Ok(transitions.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(id: &str, state: ObligationState, priority: u8) -> Obligation {
        Obligation::new(id, state, priority).with_state_identity("state-1")
    }

    fn op(
        id: &str, kind: OperationKind, targets: &[&str],
        class: SchedulingClass, work: u64, progress: &str,
    ) -> CandidateOperation {
        CandidateOperation::new(id, kind, targets.iter().copied(), class, work, progress)
    }

    fn choose(
        obligations: &[Obligation], operations: &[CandidateOperation],
        attempts: &BTreeSet<(String, String)>, budget: u64, reserve: u64,
    ) -> SelectionDecision {
        choose_next(&SelectionContext {
            obligations,
            operations,
            attempted_without_progress: attempts,
            remaining_work_units: budget,
            mandatory_reserve_units: reserve,
        })
    }

    fn receipt_verifier(
        operation_id: &str, ids: &[&str],
        proof: ProofVerdict, evidence: EvidenceVerdict,
    ) -> VerifierReceipt {
        VerifierReceipt {
            verifier_id: "checker-v1".into(),
            operation_id: operation_id.into(),
            obligation_ids: ids.iter().map(|s| (*s).to_string()).collect(),
            proof_profile_id: "profile-v1".into(),
            proof_identity: "proof-reference-only".into(),
            result_state_identity: "state-2".into(),
            progress_key: "state-1".into(),
            proof_verdict: proof,
            evidence_verdict: evidence,
        }
    }

    #[test]
    fn missing_premise_routes_to_retrieval_not_verification() {
        let obligations = [o("a", ObligationState::Open, 4)];
        let operations = [
            op("retrieve-a", OperationKind::Retrieve, &["a"], SchedulingClass::Competitive, 2, "s1"),
            op("verify-a", OperationKind::Verify, &["a"], SchedulingClass::Competitive, 1, "s1"),
        ];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 10, 1);
        assert!(matches!(decision.selection,
            Selection::Run { ref operation_id, kind: OperationKind::Retrieve, .. }
                if operation_id == "retrieve-a"));
        assert!(decision.rejected.iter().any(|r|
            r.operation_id == "verify-a"
                && r.reason == RejectionReason::OperationDoesNotAddressState));
    }

    #[test]
    fn verifier_rejection_routes_to_repair_not_repeated_verification() {
        let obligations = [o("claim", ObligationState::VerifierRejected, 8)];
        let operations = [
            op("verify-again", OperationKind::Verify, &["claim"], SchedulingClass::Competitive, 1, "s1"),
            op("repair", OperationKind::RepairDerivation, &["claim"], SchedulingClass::Competitive, 3, "s1"),
        ];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 10, 0);
        assert!(matches!(decision.selection,
            Selection::Run { ref operation_id, kind: OperationKind::RepairDerivation, .. }
                if operation_id == "repair"));
    }

    #[test]
    fn mandatory_work_precedes_competitive_work() {
        let obligations = [
            o("mandatory", ObligationState::Open, 2),
            o("important", ObligationState::Open, 9),
        ];
        let operations = [
            op("competitive", OperationKind::Retrieve, &["important"], SchedulingClass::Competitive, 1, "s1"),
            op("mandatory-op", OperationKind::Retrieve, &["mandatory"], SchedulingClass::Mandatory, 4, "s1"),
        ];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 10, 4);
        assert!(matches!(decision.selection,
            Selection::Run { ref operation_id, work_units: 4, .. }
                if operation_id == "mandatory-op"));
        assert_eq!(decision.competitive_work_limit, 6);
    }

    #[test]
    fn competitive_work_cannot_spend_reserved_capacity() {
        let obligations = [o("claim", ObligationState::Open, 9)];
        let operations = [
            op("too-expensive", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 7, "s1"),
        ];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 10, 4);
        assert_eq!(decision.selection, Selection::Stop(StopReason::BudgetExhausted));
    }

    #[test]
    fn unchanged_repeat_stops_for_no_semantic_progress() {
        let obligations = [o("claim", ObligationState::Open, 4)];
        let operations = [op("retrieve", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "state-7")];
        let attempts = BTreeSet::from([("retrieve".into(), "state-7".into())]);
        let decision = choose(&obligations, &operations, &attempts, 10, 0);
        assert_eq!(decision.selection, Selection::Stop(StopReason::NoSemanticProgress));
    }

    #[test]
    fn changed_evidence_state_allows_a_new_attempt() {
        let obligations = [o("claim", ObligationState::Open, 4)];
        let operations = [op("retrieve", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "state-8")];
        let attempts = BTreeSet::from([("retrieve".into(), "state-7".into())]);
        assert!(matches!(
            choose(&obligations, &operations, &attempts, 10, 0).selection,
            Selection::Run { .. }
        ));
    }

    #[test]
    fn operation_success_without_verifier_does_not_discharge() {
        let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 5)];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: None,
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &receipt), Ok(0));
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
    }

    #[test]
    fn exact_verifier_receipt_can_discharge_targeted_obligations() {
        let mut obligations = vec![
            o("a", ObligationState::CandidateEvidenceAvailable, 5),
            o("b", ObligationState::CandidateEvidenceAvailable, 5),
            o("c", ObligationState::Open, 1),
        ];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify-pair".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["a".into(), "b".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify-pair", &["a", "b"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &receipt), Ok(2));
        assert_eq!(obligations[0].state, ObligationState::SupportedUnderProfile);
        assert_eq!(obligations[1].state, ObligationState::SupportedUnderProfile);
        assert_eq!(obligations[2].state, ObligationState::Open);
    }

    #[test]
    fn mismatched_verifier_binding_fails_without_mutating_obligations() {
        let mut obligations = vec![o("a", ObligationState::CandidateEvidenceAvailable, 5)];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["a".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify", &["b"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::ObligationBindingMismatch)
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
    }

    #[test]
    fn revoked_source_cannot_be_promoted_by_a_valid_derivation() {
        let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 5)];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::StaleOrRetracted,
            )),
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &receipt), Ok(1));
        assert_eq!(obligations[0].state, ObligationState::StaleOrRetracted);
    }

    #[test]
    fn invalid_proof_is_rejected_even_when_source_is_admissible() {
        let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 8)];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify", &["claim"], ProofVerdict::Invalid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &receipt), Ok(1));
        assert_eq!(obligations[0].state, ObligationState::VerifierRejected);
    }

    #[test]
    fn failed_or_cancelled_operation_never_promotes() {
        for status in [OperationStatus::Failed, OperationStatus::Cancelled] {
            let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 8)];
            let receipt = OperationReceipt {
                producer_id: "planner-v1".into(),
                operation_id: "verify".into(),
                progress_key: "state-1".into(),
                obligation_ids: BTreeSet::from(["claim".into()]),
                status,
                verifier: Some(receipt_verifier(
                    "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
                )),
            };
            assert_eq!(apply_operation_receipt(&mut obligations, &receipt), Ok(0));
            assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
        }
    }

    #[test]
    fn no_information_case_stops_for_external_evidence() {
        let obligations = [o("physical-fact", ObligationState::UnresolvableWithoutExternalInput, 8)];
        let decision = choose(&obligations, &[], &BTreeSet::new(), 10, 0);
        assert_eq!(decision.selection, Selection::Stop(StopReason::NeedsExternalEvidence));
    }

    #[test]
    fn complete_query_stops_without_extra_reasoning() {
        let obligations = [o("claim", ObligationState::SupportedUnderProfile, 8)];
        let operations = [op("extra-work", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 99, "s1")];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 100, 0);
        assert_eq!(decision.selection, Selection::Stop(StopReason::AllObligationsDischarged));
    }

    #[test]
    fn selection_is_deterministic_under_candidate_permutation() {
        let obligations = [o("a", ObligationState::Open, 5), o("b", ObligationState::Open, 5)];
        let first = [
            op("retrieve-b", OperationKind::Retrieve, &["b"], SchedulingClass::Competitive, 2, "s1"),
            op("retrieve-a", OperationKind::Retrieve, &["a"], SchedulingClass::Competitive, 2, "s1"),
        ];
        let second = [first[1].clone(), first[0].clone()];
        let left = choose(&obligations, &first, &BTreeSet::new(), 10, 0);
        let right = choose(&obligations, &second, &BTreeSet::new(), 10, 0);
        assert_eq!(left.selection, right.selection);
    }

    #[test]
    fn unaffordable_mandatory_work_blocks_competitive_spend() {
        let obligations = [o("must", ObligationState::Open, 1), o("optional", ObligationState::Open, 9)];
        let operations = [
            op("mandatory-heavy", OperationKind::Retrieve, &["must"], SchedulingClass::Mandatory, 11, "s1"),
            op("optional-cheap", OperationKind::Retrieve, &["optional"], SchedulingClass::Competitive, 1, "s1"),
        ];
        let decision = choose(&obligations, &operations, &BTreeSet::new(), 10, 2);
        assert_eq!(decision.selection, Selection::Stop(StopReason::MandatoryWorkCannotFit));
    }
    #[test]
    fn duplicate_obligation_or_operation_ids_fail_closed() {
        let duplicate_obligations = [
            o("same", ObligationState::Open, 1),
            o("same", ObligationState::Contested, 2),
        ];
        assert_eq!(
            choose(&duplicate_obligations, &[], &BTreeSet::new(), 10, 0).selection,
            Selection::Stop(StopReason::InvalidInput)
        );

        let obligations = [o("claim", ObligationState::Open, 1)];
        let duplicate_operations = [
            op("same-op", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "s1"),
            op("same-op", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "s1"),
        ];
        assert_eq!(
            choose(&obligations, &duplicate_operations, &BTreeSet::new(), 10, 0).selection,
            Selection::Stop(StopReason::InvalidInput)
        );
    }

    #[test]
    fn verifier_must_be_declared_distinct_from_operation_producer() {
        let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 5)];
        let mut verifier = receipt_verifier(
            "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
        );
        verifier.verifier_id = "planner-v1".into();
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(verifier),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::VerifierNotDistinctFromProducer)
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
    }

    #[test]
    fn receipt_must_match_the_current_obligation_state_identity() {
        let mut obligations = vec![
            o("claim", ObligationState::CandidateEvidenceAvailable, 5)
                .with_state_identity("current-state"),
        ];
        let mut verifier = receipt_verifier(
            "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
        );
        verifier.progress_key = "old-state".into();
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "old-state".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(verifier),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::ProgressBindingMismatch)
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);

        let mut unversioned = vec![
            Obligation::new("claim", ObligationState::CandidateEvidenceAvailable, 5),
        ];
        let unversioned_receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(
            apply_operation_receipt(&mut unversioned, &unversioned_receipt),
            Err(ReceiptError::MissingCurrentStateIdentity("claim".into()))
        );
    }

    #[test]
    fn state_changing_receipt_must_advance_state_identity() {
        let mut obligations = vec![o("claim", ObligationState::CandidateEvidenceAvailable, 5)];
        let mut verifier = receipt_verifier(
            "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
        );
        verifier.result_state_identity = "state-1".into();
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(verifier),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::ResultStateIdentityUnchanged)
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
        assert_eq!(obligations[0].state_identity, "state-1");
    }

    #[test]
    fn stale_receipt_replay_cannot_resurrect_a_revoked_obligation() {
        let mut obligations = vec![
            o("claim", ObligationState::CandidateEvidenceAvailable, 5)
                .with_state_identity("evidence-v1"),
        ];
        let mut initial_verifier = receipt_verifier(
            "verify-v1", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
        );
        initial_verifier.progress_key = "evidence-v1".into();
        initial_verifier.result_state_identity = "verified-v1".into();
        let original = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify-v1".into(),
            progress_key: "evidence-v1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(initial_verifier),
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &original), Ok(1));
        assert_eq!(obligations[0].state, ObligationState::SupportedUnderProfile);
        assert_eq!(obligations[0].state_identity, "verified-v1");

        let mut revocation_verifier = receipt_verifier(
            "refresh-v2", &["claim"], ProofVerdict::Valid, EvidenceVerdict::StaleOrRetracted,
        );
        revocation_verifier.progress_key = "verified-v1".into();
        revocation_verifier.result_state_identity = "revoked-v2".into();
        let revocation = OperationReceipt {
            producer_id: "ledger-v2".into(),
            operation_id: "refresh-v2".into(),
            progress_key: "verified-v1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(revocation_verifier),
        };
        assert_eq!(apply_operation_receipt(&mut obligations, &revocation), Ok(1));
        assert_eq!(obligations[0].state, ObligationState::StaleOrRetracted);
        assert_eq!(obligations[0].state_identity, "revoked-v2");

        assert_eq!(
            apply_operation_receipt(&mut obligations, &original),
            Err(ReceiptError::ProgressBindingMismatch)
        );
        assert_eq!(obligations[0].state, ObligationState::StaleOrRetracted);
        assert_eq!(obligations[0].state_identity, "revoked-v2");
    }

    #[test]
    fn mixed_state_identities_reject_multi_obligation_receipt_transactionally() {
        let mut obligations = vec![
            o("fresh", ObligationState::CandidateEvidenceAvailable, 5)
                .with_state_identity("state-1"),
            o("stale", ObligationState::CandidateEvidenceAvailable, 5)
                .with_state_identity("state-2"),
        ];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify-pair".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["fresh".into(), "stale".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify-pair", &["fresh", "stale"],
                ProofVerdict::Valid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::ProgressBindingMismatch)
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
        assert_eq!(obligations[1].state, ObligationState::CandidateEvidenceAvailable);
        assert_eq!(obligations[0].state_identity, "state-1");
        assert_eq!(obligations[1].state_identity, "state-2");
    }

    #[test]
    fn mixed_external_and_internal_obligations_do_not_claim_full_external_stop() {
        let obligations = [
            o("external", ObligationState::UnresolvableWithoutExternalInput, 8),
            o("open", ObligationState::Open, 4),
        ];
        let decision = choose(&obligations, &[], &BTreeSet::new(), 10, 0);
        assert_eq!(decision.selection, Selection::Stop(StopReason::NoApplicableOperation));
    }

    #[test]
    fn duplicate_existing_obligation_ids_reject_receipt_transactionally() {
        let mut obligations = vec![
            o("claim", ObligationState::CandidateEvidenceAvailable, 5),
            o("claim", ObligationState::Open, 3),
        ];
        let receipt = OperationReceipt {
            producer_id: "planner-v1".into(),
            operation_id: "verify".into(),
            progress_key: "state-1".into(),
            obligation_ids: BTreeSet::from(["claim".into()]),
            status: OperationStatus::Completed,
            verifier: Some(receipt_verifier(
                "verify", &["claim"], ProofVerdict::Valid, EvidenceVerdict::Admissible,
            )),
        };
        assert_eq!(
            apply_operation_receipt(&mut obligations, &receipt),
            Err(ReceiptError::DuplicateExistingObligation("claim".into()))
        );
        assert_eq!(obligations[0].state, ObligationState::CandidateEvidenceAvailable);
        assert_eq!(obligations[1].state, ObligationState::Open);
    }

}
