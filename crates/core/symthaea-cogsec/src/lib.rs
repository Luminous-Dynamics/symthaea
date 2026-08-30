// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public CogSec API boundary.
//!
//! The detailed sealed-domain implementation is private. This wrapper keeps the
//! public reference-monitor surface deliberately small and prevents application
//! data from supplying a monitor verdict when an audit receipt is produced.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod facade;

pub use facade::{
    ArtifactIntegrity, AuthorityError, CapabilityFact, CognitiveSecurityLabel, CommitPermit,
    Confidentiality, Consequence, ControlIntegrity, DecisionOutcome, DelegationError, Digest32,
    MonitorDecision, MutationKind, MutationPermit, MutationReceipt, MutationRequest, OriginState,
    PolicyRule, PolicySnapshot, PrincipalId, ReasonCode, ResourceId, ResourceScope, TaintLevel,
    TrustedFactAuthority, TrustedFacts,
};

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

    /// Evaluate and produce the matching receipt as one operation.
    ///
    /// The caller cannot supply the verdict recorded in the receipt: the monitor
    /// recomputes it from the request, trusted facts, and policy and passes that
    /// exact decision to the private receipt implementation.
    pub fn evaluate_with_receipt(
        &self,
        request: &MutationRequest,
        facts: &TrustedFacts,
        policy: &PolicySnapshot,
    ) -> Result<(MonitorDecision, MutationReceipt), AuthorityError> {
        let decision = self.inner.evaluate(request, facts, policy)?;
        let receipt = self.inner.receipt(request, facts, policy, &decision)?;
        Ok((decision, receipt))
    }

    /// Produce a receipt for the monitor's own freshly evaluated decision.
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
    fn receipt_records_the_monitors_decision_not_caller_data() {
        let (monitor, authority, capability, request, policy) = fixture();
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&capability])
            .unwrap();
        let (decision, receipt) = monitor
            .evaluate_with_receipt(&request, &facts, &policy)
            .unwrap();
        assert_eq!(decision.outcome, DecisionOutcome::Allow);
        assert_eq!(receipt.outcome, decision.outcome);
        assert_eq!(receipt.reasons, decision.reasons);
    }

    #[test]
    fn receipt_recomputes_non_allow_outcome() {
        let (monitor, authority, capability, mut request, policy) = fixture();
        request.input_label.taint = TaintLevel::Tainted;
        let facts = authority
            .snapshot(d(8), d(9), 7, 11, 13, &[&capability])
            .unwrap();
        let receipt = monitor.receipt(&request, &facts, &policy).unwrap();
        assert_eq!(receipt.outcome, DecisionOutcome::Quarantine);
        assert_eq!(receipt.reasons, vec![ReasonCode::TaintedDependency]);
    }
}
