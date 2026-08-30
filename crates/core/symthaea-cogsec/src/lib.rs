// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public CogSec API boundary.
//!
//! The detailed sealed-domain implementation is private. This wrapper keeps the
//! public reference-monitor surface deliberately small, prevents application
//! data from supplying a monitor verdict when evidence is produced, and emits
//! evaluation receipts that bind the security context that actually determined
//! the monitor outcome.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod facade;

use serde::{Deserialize, Serialize};

pub use facade::{
    ArtifactIntegrity, AuthorityError, CapabilityFact, CognitiveSecurityLabel, CommitPermit,
    Confidentiality, Consequence, ControlIntegrity, DecisionOutcome, DelegationError, Digest32,
    MonitorDecision, MutationKind, MutationPermit, MutationRequest, OriginState, PolicyRule,
    PolicySnapshot, PrincipalId, ReasonCode, ResourceId, ResourceScope, TaintLevel,
    TrustedFactAuthority, TrustedFacts,
};

/// Stage represented by a durable CogSec evidence record.
///
/// An evaluation receipt proves only what the monitor evaluated. It is not
/// evidence that authorization, precommit, or the protected state mutation
/// itself subsequently occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptStage {
    /// Deterministic policy evaluation against one trusted-facts snapshot.
    Evaluation,
}

/// Serializable evidence for one monitor evaluation.
///
/// This is evidence, not authority: it is intentionally serializable and does
/// not carry the private monitor-domain seal. Runtime integrations should bind
/// exported receipts to an authenticated evidence-plane identity/signature when
/// cross-process or cross-machine provenance is required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReceipt {
    /// Evidence stage represented by this record.
    pub stage: ReceiptStage,
    /// Request identity.
    pub request_id: Digest32,
    /// Principal whose authority was requested.
    pub subject: PrincipalId,
    /// Mutation class.
    pub kind: MutationKind,
    /// Protected resource.
    pub resource: ResourceId,
    /// Exact proposed effect commitment.
    pub mutation_digest: Digest32,
    /// Consequence class of the requested effect.
    pub consequence: Consequence,
    /// Security label/provenance used by policy evaluation.
    pub input_label: CognitiveSecurityLabel,
    /// Resource-state root prepared by the requester.
    pub expected_resource_state_root: Digest32,
    /// Resource-state root supplied by the trusted state owner at evaluation.
    pub observed_resource_state_root: Digest32,
    /// Policy root prepared by the requester.
    pub expected_policy_root: Digest32,
    /// Canonical policy root evaluated by the monitor.
    pub evaluated_policy_root: Digest32,
    /// Trusted policy root supplied independently through the fact authority.
    pub trusted_policy_root: Digest32,
    /// Policy epoch evaluated by the monitor.
    pub policy_epoch: u64,
    /// Authorization epoch observed by the monitor.
    pub authorization_epoch: u64,
    /// Revocation epoch observed by the monitor.
    pub revocation_epoch: u64,
    /// Logical sequence of the proposed transition.
    pub sequence: u64,
    /// Capability selected by the monitor for an allowed request, if one was required.
    pub capability_id: Option<Digest32>,
    /// Monitor outcome.
    pub outcome: DecisionOutcome,
    /// Stable reason codes supporting the outcome.
    pub reasons: Vec<ReasonCode>,
}

/// Public deterministic reference monitor for one private in-process authority domain.
///
/// The underlying implementation is intentionally private so callers cannot
/// reach lower-level helper APIs that accept caller-supplied security verdicts.
/// This type is neither `Clone`, `Copy`, `Default`, nor serializable.
#[derive(Debug)]
pub struct ReferenceMonitor {
    inner: facade::ReferenceMonitor,
}

impl ReferenceMonitor {
    /// Bootstrap one isolated monitor domain and its trusted fact issuer.
    pub fn bootstrap() -> (Self, TrustedFactAuthority) {
        let (inner, authority) = facade::ReferenceMonitor::bootstrap();
        (Self { inner }, authority)
    }

    /// Evaluate one proposed mutation using same-domain trusted facts.
    pub fn evaluate(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MonitorDecision, AuthorityError> {
        self.inner.evaluate(request, facts, policy)
    }

    /// Evaluate and mint a one-use authorization-time permit.
    pub fn authorize(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationPermit, AuthorityError> {
        self.inner.authorize(request, facts, policy)
    }

    /// Revalidate an authorization-time permit against fresh trusted facts.
    pub fn precommit(
        &self,
        permit: MutationPermit,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<CommitPermit, AuthorityError> {
        self.inner.precommit(permit, facts, policy)
    }

    /// Whether a commit-ready permit belongs to this exact monitor domain.
    ///
    /// Protected owners must additionally check the permit's resource, mutation
    /// kind, mutation digest, consequence, and current state binding before
    /// performing the state transition.
    pub fn accepts_commit_permit(&self, permit: &CommitPermit) -> bool {
        self.inner.accepts_commit_permit(permit)
    }

    /// Evaluate and produce the matching evidence receipt as one operation.
    ///
    /// The caller cannot supply the verdict recorded in the receipt. For an
    /// allowed request, the monitor deterministically re-runs authorization on
    /// the same immutable inputs solely to record the capability identity that
    /// would back the authorization token; that temporary token is immediately
    /// dropped and conveys no commit authority.
    pub fn evaluate_with_receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<(MonitorDecision, MutationReceipt), AuthorityError> {
        let decision = self.inner.evaluate(request, facts, policy)?;
        let capability_id = if decision.outcome == DecisionOutcome::Allow {
            self.inner.authorize(request, facts, policy)?.capability_id()
        } else {
            None
        };

        let receipt = MutationReceipt {
            stage: ReceiptStage::Evaluation,
            request_id: request.request_id,
            subject: request.subject.clone(),
            kind: request.kind,
            resource: request.resource.clone(),
            mutation_digest: request.mutation_digest,
            consequence: request.consequence,
            input_label: request.input_label.clone(),
            expected_resource_state_root: request.expected_resource_state_root,
            observed_resource_state_root: facts.resource_state_root(),
            expected_policy_root: request.expected_policy_root,
            evaluated_policy_root: policy.root,
            trusted_policy_root: facts.policy_root(),
            policy_epoch: policy.epoch,
            authorization_epoch: facts.authorization_epoch(),
            revocation_epoch: facts.revocation_epoch(),
            sequence: request.sequence,
            capability_id,
            outcome: decision.outcome,
            reasons: decision.reasons.clone(),
        };
        Ok((decision, receipt))
    }

    /// Produce evidence for the monitor's own freshly evaluated decision.
    pub fn receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<MutationReceipt, AuthorityError> {
        self.evaluate_with_receipt(request, facts, policy)
            .map(|(_, receipt)| receipt)
    }
}

#[cfg(test)]
mod public_api_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn fixture() -> (
        ReferenceMonitor,
        TrustedFactAuthority,
        CapabilityFact,
        MutationRequest,
        PolicySnapshot,
    ) {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId("mind/goals".into());
        let capability = authority.issue_capability(
            d(4),
            subject.clone(),
            MutationKind::GoalActivation,
            ResourceScope::Exact(resource.clone()),
            Consequence::High,
            11,
            13,
            40,
            Some(50),
            false,
        );
        let mut roots = BTreeSet::new();
        roots.insert(d(3));
        let request = MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject,
            resource,
            mutation_digest: d(2),
            expected_resource_state_root: d(8),
            expected_policy_root: d(9),
            input_label: CognitiveSecurityLabel {
                control_integrity: ControlIntegrity::Authenticated,
                confidentiality: Confidentiality::Local,
                origin: OriginState::Authenticated,
                artifact_integrity: ArtifactIntegrity::Authenticated,
                taint: TaintLevel::Clean,
                provenance_roots: roots,
            },
            consequence: Consequence::High,
            sequence: 42,
        };
        let policy = PolicySnapshot {
            root: d(9),
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Authenticated,
                maximum_taint: TaintLevel::Clean,
                capability_required: true,
            }],
        };
        (monitor, authority, capability, request, policy)
    }

    #[test]
    fn receipt_records_the_monitors_decision_and_full_security_context() {
        let (monitor, authority, capability, request, policy) = fixture();
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&capability])
            .unwrap();
        let (decision, receipt) = monitor
            .evaluate_with_receipt(&request, &facts, &policy)
            .unwrap();

        assert_eq!(decision.outcome, DecisionOutcome::Allow);
        assert_eq!(receipt.stage, ReceiptStage::Evaluation);
        assert_eq!(receipt.subject, request.subject);
        assert_eq!(receipt.consequence, Consequence::High);
        assert_eq!(receipt.input_label, request.input_label);
        assert_eq!(receipt.expected_resource_state_root, d(8));
        assert_eq!(receipt.observed_resource_state_root, d(8));
        assert_eq!(receipt.expected_policy_root, d(9));
        assert_eq!(receipt.evaluated_policy_root, d(9));
        assert_eq!(receipt.trusted_policy_root, d(9));
        assert_eq!(receipt.capability_id, Some(d(4)));
        assert_eq!(receipt.outcome, decision.outcome);
        assert_eq!(receipt.reasons, decision.reasons);
    }

    #[test]
    fn receipt_recomputes_non_allow_outcome_and_preserves_taint_context() {
        let (monitor, authority, capability, mut request, policy) = fixture();
        request.input_label.taint = TaintLevel::Tainted;
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&capability])
            .unwrap();
        let receipt = monitor.receipt(&request, &facts, &policy).unwrap();

        assert_eq!(receipt.outcome, DecisionOutcome::Quarantine);
        assert_eq!(receipt.reasons, vec![ReasonCode::TaintedDependency]);
        assert_eq!(receipt.input_label.taint, TaintLevel::Tainted);
        assert_eq!(receipt.capability_id, None);
    }

    #[test]
    fn mismatch_receipt_keeps_expected_and_observed_roots_distinct() {
        let (monitor, authority, capability, mut request, policy) = fixture();
        request.expected_resource_state_root = d(77);
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&capability])
            .unwrap();
        let receipt = monitor.receipt(&request, &facts, &policy).unwrap();

        assert_eq!(receipt.outcome, DecisionOutcome::RequireRevalidation);
        assert_eq!(receipt.expected_resource_state_root, d(77));
        assert_eq!(receipt.observed_resource_state_root, d(8));
    }
}
