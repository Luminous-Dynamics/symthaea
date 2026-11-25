// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # SAT reference model
//!
//! A deliberately small, dependency-free reference oracle for bounded
//! authority and ambiguous-outcome scenarios in the Sovereignty Assurance
//! Tournament (SAT).
//!
//! This crate is **not** a production authorization system. It does not verify
//! signatures, authenticate issuers, persist revocation state, fence a real
//! effect sink, establish trusted time, or provide process isolation. Root
//! grants are trusted fixture inputs supplied by the test harness. The
//! implementation exists to give production adapters an independently
//! authored behavioral oracle.
//!
//! The authorization API intentionally accepts no intelligence, confidence,
//! consciousness, collective-score, reputation, or model-generated trust
//! parameter. Such signals may be evaluated by a separate caution policy, but
//! they cannot create a grant or satisfy this model's authority predicate.
//!
//! Time windows are half-open: not_before <= now < expires_at.
//! Delegated action/resource scopes and validity windows must be subsets of
//! the parent's, and each delegation consumes one unit of remaining depth.
//!
//! Unknown effect outcomes block another mutation attempt until reconciliation
//! proves either that no effect occurred or that the effect already occurred.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Identifier of an authority principal. Authentication is out of scope.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrincipalId(pub String);

impl From<&str> for PrincipalId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Stable identifier for one grant in a delegation lineage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrantId(pub String);

impl From<&str> for GrantId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// A bounded capability grant. Scopes are exact-match sets; wildcards,
/// path-prefix semantics and ambient permissions are unsupported here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityGrant {
    pub id: GrantId,
    pub issuer: PrincipalId,
    pub subject: PrincipalId,
    pub action_scope: BTreeSet<String>,
    pub resource_scope: BTreeSet<String>,
    pub not_before: u64,
    pub expires_at: u64,
    pub may_delegate: bool,
    pub remaining_delegation_depth: u8,
    pub parent_id: Option<GrantId>,
}

impl CapabilityGrant {
    #[allow(clippy::too_many_arguments)]
    pub fn new<I, A, R, S>(
        id: impl Into<GrantId>,
        issuer: impl Into<PrincipalId>,
        subject: impl Into<PrincipalId>,
        actions: I,
        resources: R,
        not_before: u64,
        expires_at: u64,
        may_delegate: bool,
        remaining_delegation_depth: u8,
        parent_id: Option<GrantId>,
    ) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<String>,
        R: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: id.into(),
            issuer: issuer.into(),
            subject: subject.into(),
            action_scope: actions.into_iter().map(Into::into).collect(),
            resource_scope: resources.into_iter().map(Into::into).collect(),
            not_before,
            expires_at,
            may_delegate,
            remaining_delegation_depth,
            parent_id,
        }
    }
}

/// Why an otherwise scope-matching grant cannot authorize an effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenialReason {
    NoMatchingGrant,
    GrantRevoked,
    AncestorRevoked,
    GrantNotYetValid,
    AncestorNotYetValid,
    GrantExpired,
    AncestorExpired,
    InvalidLineage,
}

impl fmt::Display for DenialReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The result of checking one exact action against the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizationDecision {
    Allowed { grant_id: GrantId },
    Denied { reason: DenialReason },
}

/// Errors encountered while constructing reference authority state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityError {
    EmptyIdentifier,
    EmptyActionScope,
    EmptyResourceScope,
    InvalidValidityWindow,
    DelegationFlagDepthMismatch,
    DuplicateGrant,
    RootGrantHasParent,
    ParentGrantMissing,
    ParentGrantInactive,
    ParentCannotDelegate,
    ChildParentMismatch,
    DelegatorIdentityMismatch,
    ActionScopeExpansion,
    ResourceScopeExpansion,
    ValidityWindowExpansion,
    DelegationDepthExpansion,
    GrantNotFound,
}

impl fmt::Display for AuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AuthorityError {}

/// In-memory reference ledger.
///
/// The ledger models grant scope, lineage and revocation semantics only.
/// Production callers must authenticate root issuance and enforce decisions
/// at the actual side-effect sink independently.
#[derive(Debug, Clone, Default)]
pub struct AuthorityLedger {
    grants: BTreeMap<GrantId, CapabilityGrant>,
    revoked: BTreeSet<GrantId>,
}

impl AuthorityLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a root grant supplied by a trusted fixture or upstream verifier.
    ///
    /// This method does not authenticate the issuer; that is outside the
    /// model's trust boundary.
    pub fn insert_root_grant(
        &mut self,
        grant: CapabilityGrant,
    ) -> Result<(), AuthorityError> {
        if grant.parent_id.is_some() {
            return Err(AuthorityError::RootGrantHasParent);
        }
        validate_grant_shape(&grant)?;
        self.insert_unique(grant)
    }

    /// Issue a child grant only when it attenuates the currently live parent.
    ///
    /// The child must name the parent's subject as issuer and the exact parent
    /// grant as its lineage predecessor. This is a semantic reference check,
    /// not a cryptographic delegation-signature verifier.
    pub fn delegate(
        &mut self,
        parent_id: &GrantId,
        child: CapabilityGrant,
        now: u64,
    ) -> Result<(), AuthorityError> {
        let parent = self
            .grants
            .get(parent_id)
            .cloned()
            .ok_or(AuthorityError::ParentGrantMissing)?;

        if self.lineage_status(parent_id, now).is_err() {
            return Err(AuthorityError::ParentGrantInactive);
        }
        if !parent.may_delegate || parent.remaining_delegation_depth == 0 {
            return Err(AuthorityError::ParentCannotDelegate);
        }
        if child.parent_id.as_ref() != Some(parent_id) {
            return Err(AuthorityError::ChildParentMismatch);
        }
        if child.issuer != parent.subject {
            return Err(AuthorityError::DelegatorIdentityMismatch);
        }
        validate_grant_shape(&child)?;

        if !child.action_scope.is_subset(&parent.action_scope) {
            return Err(AuthorityError::ActionScopeExpansion);
        }
        if !child.resource_scope.is_subset(&parent.resource_scope) {
            return Err(AuthorityError::ResourceScopeExpansion);
        }
        if child.not_before < parent.not_before || child.expires_at > parent.expires_at {
            return Err(AuthorityError::ValidityWindowExpansion);
        }
        if child.remaining_delegation_depth
            > parent.remaining_delegation_depth.saturating_sub(1)
        {
            return Err(AuthorityError::DelegationDepthExpansion);
        }

        self.insert_unique(child)
    }

    /// Revoke a grant. Descendants become unusable because each authorization
    /// checks the complete parent lineage.
    pub fn revoke(&mut self, grant_id: &GrantId) -> Result<(), AuthorityError> {
        if !self.grants.contains_key(grant_id) {
            return Err(AuthorityError::GrantNotFound);
        }
        self.revoked.insert(grant_id.clone());
        Ok(())
    }

    /// Authorize one exact subject/action/resource request.
    ///
    /// There is deliberately no cognitive-score parameter. Intelligence can
    /// inform a separate caution policy but cannot satisfy this authority check.
    pub fn authorize(
        &self,
        subject: &PrincipalId,
        action: &str,
        resource: &str,
        now: u64,
    ) -> AuthorizationDecision {
        let mut scope_match = false;
        let mut last_reason = DenialReason::NoMatchingGrant;

        for grant in self.grants.values().filter(|grant| {
            &grant.subject == subject
                && grant.action_scope.contains(action)
                && grant.resource_scope.contains(resource)
        }) {
            scope_match = true;
            match self.lineage_status(&grant.id, now) {
                Ok(()) => {
                    return AuthorizationDecision::Allowed {
                        grant_id: grant.id.clone(),
                    };
                }
                Err(reason) => last_reason = reason,
            }
        }

        AuthorizationDecision::Denied {
            reason: if scope_match {
                last_reason
            } else {
                DenialReason::NoMatchingGrant
            },
        }
    }

    /// Stable inventory view; this does not imply that returned grants are live.
    pub fn grant_ids_for_subject(&self, subject: &PrincipalId) -> Vec<GrantId> {
        self.grants
            .values()
            .filter(|grant| &grant.subject == subject)
            .map(|grant| grant.id.clone())
            .collect()
    }

    pub fn is_revoked(&self, grant_id: &GrantId) -> bool {
        self.revoked.contains(grant_id)
    }

    fn insert_unique(&mut self, grant: CapabilityGrant) -> Result<(), AuthorityError> {
        if self.grants.contains_key(&grant.id) {
            return Err(AuthorityError::DuplicateGrant);
        }
        self.grants.insert(grant.id.clone(), grant);
        Ok(())
    }

    fn lineage_status(
        &self,
        grant_id: &GrantId,
        now: u64,
    ) -> Result<(), DenialReason> {
        let mut cursor = grant_id;
        let mut depth = 0usize;
        let mut visited = BTreeSet::new();

        loop {
            if !visited.insert(cursor.clone()) {
                return Err(DenialReason::InvalidLineage);
            }
            let grant = self
                .grants
                .get(cursor)
                .ok_or(DenialReason::InvalidLineage)?;

            if self.revoked.contains(cursor) {
                return Err(if depth == 0 {
                    DenialReason::GrantRevoked
                } else {
                    DenialReason::AncestorRevoked
                });
            }
            if now < grant.not_before {
                return Err(if depth == 0 {
                    DenialReason::GrantNotYetValid
                } else {
                    DenialReason::AncestorNotYetValid
                });
            }
            if now >= grant.expires_at {
                return Err(if depth == 0 {
                    DenialReason::GrantExpired
                } else {
                    DenialReason::AncestorExpired
                });
            }

            match &grant.parent_id {
                Some(parent_id) => {
                    cursor = parent_id;
                    depth += 1;
                }
                None => return Ok(()),
            }

            if depth > self.grants.len() {
                return Err(DenialReason::InvalidLineage);
            }
        }
    }
}

fn validate_grant_shape(grant: &CapabilityGrant) -> Result<(), AuthorityError> {
    if grant.id.0.trim().is_empty()
        || grant.issuer.0.trim().is_empty()
        || grant.subject.0.trim().is_empty()
    {
        return Err(AuthorityError::EmptyIdentifier);
    }
    if grant.action_scope.is_empty() {
        return Err(AuthorityError::EmptyActionScope);
    }
    if grant.resource_scope.is_empty() {
        return Err(AuthorityError::EmptyResourceScope);
    }
    if grant.not_before >= grant.expires_at {
        return Err(AuthorityError::InvalidValidityWindow);
    }
    if grant.may_delegate != (grant.remaining_delegation_depth > 0) {
        return Err(AuthorityError::DelegationFlagDepthMismatch);
    }
    if grant
        .action_scope
        .iter()
        .chain(grant.resource_scope.iter())
        .any(|item| item.trim().is_empty())
    {
        return Err(AuthorityError::EmptyIdentifier);
    }
    Ok(())
}

/// State of one external mutation whose outcome may be ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationState {
    Ready,
    OutcomeUnknown,
    ConfirmedNoEffect,
    ConfirmedEffect,
}

/// Why a new mutation attempt is currently prohibited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationBlock {
    OutcomeRequiresReconciliation,
    EffectAlreadyConfirmed,
}

/// Reconciled sink result. Unknown intentionally preserves the blocked state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciledOutcome {
    Unknown,
    NoEffect,
    EffectOccurred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationError {
    EmptyOperationIdentity,
    AttemptBlocked(MutationBlock),
    ReconciliationWithoutUnknownOutcome,
}

impl fmt::Display for MutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MutationError {}

/// Local reference state machine for one operation identity.
///
/// This gate is in-memory and does not claim crash safety, distributed
/// exactly-once execution or sink-side idempotency. Production integrations
/// must persist and enforce operation/fence state at the external effect edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationGate {
    operation_id: String,
    idempotency_key: String,
    state: MutationState,
    attempts: u32,
}

impl MutationGate {
    pub fn new(
        operation_id: impl Into<String>,
        idempotency_key: impl Into<String>,
    ) -> Result<Self, MutationError> {
        let operation_id = operation_id.into();
        let idempotency_key = idempotency_key.into();
        if operation_id.trim().is_empty() || idempotency_key.trim().is_empty() {
            return Err(MutationError::EmptyOperationIdentity);
        }
        Ok(Self {
            operation_id,
            idempotency_key,
            state: MutationState::Ready,
            attempts: 0,
        })
    }

    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    pub fn idempotency_key(&self) -> &str {
        &self.idempotency_key
    }

    pub fn state(&self) -> MutationState {
        self.state
    }

    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Begin an attempt and immediately enter OutcomeUnknown.
    ///
    /// Until an independent sink result is reconciled, another mutation is
    /// blocked. This models the conservative side of an ambiguous commit.
    pub fn begin_attempt(&mut self) -> Result<u32, MutationError> {
        match self.state {
            MutationState::Ready | MutationState::ConfirmedNoEffect => {}
            MutationState::OutcomeUnknown => {
                return Err(MutationError::AttemptBlocked(
                    MutationBlock::OutcomeRequiresReconciliation,
                ));
            }
            MutationState::ConfirmedEffect => {
                return Err(MutationError::AttemptBlocked(
                    MutationBlock::EffectAlreadyConfirmed,
                ));
            }
        }

        self.attempts = self.attempts.saturating_add(1);
        self.state = MutationState::OutcomeUnknown;
        Ok(self.attempts)
    }

    /// Reconcile an unknown attempt using independently obtained sink evidence.
    pub fn reconcile(
        &mut self,
        outcome: ReconciledOutcome,
    ) -> Result<MutationState, MutationError> {
        if self.state != MutationState::OutcomeUnknown {
            return Err(MutationError::ReconciliationWithoutUnknownOutcome);
        }

        self.state = match outcome {
            ReconciledOutcome::Unknown => MutationState::OutcomeUnknown,
            ReconciledOutcome::NoEffect => MutationState::ConfirmedNoEffect,
            ReconciledOutcome::EffectOccurred => MutationState::ConfirmedEffect,
        };
        Ok(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000;

    fn grant(
        id: &str,
        issuer: &str,
        subject: &str,
        actions: &[&str],
        resources: &[&str],
        not_before: u64,
        expires_at: u64,
        may_delegate: bool,
        depth: u8,
        parent: Option<&str>,
    ) -> CapabilityGrant {
        CapabilityGrant::new(
            id,
            issuer,
            subject,
            actions.iter().copied(),
            resources.iter().copied(),
            not_before,
            expires_at,
            may_delegate,
            depth,
            parent.map(GrantId::from),
        )
    }

    fn root_ledger() -> AuthorityLedger {
        let mut ledger = AuthorityLedger::new();
        ledger
            .insert_root_grant(grant(
                "root",
                "owner",
                "agent-a",
                &["read", "write"],
                &["project-1", "project-2"],
                900,
                1_500,
                true,
                2,
                None,
            ))
            .expect("valid root fixture");
        ledger
    }

    #[test]
    fn missing_capability_denies() {
        let ledger = AuthorityLedger::new();
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-a"), "write", "project-1", NOW),
            AuthorizationDecision::Denied {
                reason: DenialReason::NoMatchingGrant
            }
        );
    }

    #[test]
    fn exact_root_scope_authorizes_and_unlisted_action_denies() {
        let ledger = root_ledger();
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-a"), "write", "project-1", NOW),
            AuthorizationDecision::Allowed {
                grant_id: GrantId::from("root")
            }
        );
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-a"), "admin", "project-1", NOW),
            AuthorizationDecision::Denied {
                reason: DenialReason::NoMatchingGrant
            }
        );
    }

    #[test]
    fn child_grant_must_attenuate_and_authorizes_only_its_subset() {
        let mut ledger = root_ledger();
        ledger
            .delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &["read"],
                    &["project-1"],
                    950,
                    1_400,
                    true,
                    1,
                    Some("root"),
                ),
                NOW,
            )
            .expect("narrow child grant is admissible");
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-b"), "read", "project-1", NOW),
            AuthorizationDecision::Allowed {
                grant_id: GrantId::from("child")
            }
        );
        assert!(matches!(
            ledger.authorize(&PrincipalId::from("agent-b"), "write", "project-1", NOW),
            AuthorizationDecision::Denied { .. }
        ));
        assert!(matches!(
            ledger.authorize(&PrincipalId::from("agent-b"), "read", "project-2", NOW),
            AuthorizationDecision::Denied { .. }
        ));
    }

    #[test]
    fn child_cannot_expand_action_scope() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant("child", "agent-a", "agent-b", &["admin"], &["project-1"],
                    950, 1_400, false, 0, Some("root")),
                NOW
            ),
            Err(AuthorityError::ActionScopeExpansion)
        );
    }

    #[test]
    fn child_cannot_expand_resource_scope() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant("child", "agent-a", "agent-b", &["read"], &["project-3"],
                    950, 1_400, false, 0, Some("root")),
                NOW
            ),
            Err(AuthorityError::ResourceScopeExpansion)
        );
    }

    #[test]
    fn child_cannot_extend_parent_lifetime() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant("child", "agent-a", "agent-b", &["read"], &["project-1"],
                    950, 1_501, false, 0, Some("root")),
                NOW
            ),
            Err(AuthorityError::ValidityWindowExpansion)
        );
    }

    #[test]
    fn child_cannot_increase_delegation_depth() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant("child", "agent-a", "agent-b", &["read"], &["project-1"],
                    950, 1_400, true, 2, Some("root")),
                NOW
            ),
            Err(AuthorityError::DelegationDepthExpansion)
        );
    }

    #[test]
    fn delegator_must_be_the_parent_subject() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant("child", "attacker", "agent-b", &["read"], &["project-1"],
                    950, 1_400, false, 0, Some("root")),
                NOW
            ),
            Err(AuthorityError::DelegatorIdentityMismatch)
        );
    }

    #[test]
    fn revoking_parent_invalidates_existing_child() {
        let mut ledger = root_ledger();
        ledger
            .delegate(
                &GrantId::from("root"),
                grant("child", "agent-a", "agent-b", &["read"], &["project-1"],
                    950, 1_400, false, 0, Some("root")),
                NOW,
            )
            .expect("valid child");
        ledger.revoke(&GrantId::from("root")).expect("root exists");
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-b"), "read", "project-1", NOW),
            AuthorizationDecision::Denied {
                reason: DenialReason::AncestorRevoked
            }
        );
    }

    #[test]
    fn validity_window_is_half_open() {
        let ledger = root_ledger();
        assert!(matches!(
            ledger.authorize(&PrincipalId::from("agent-a"), "read", "project-1", 900),
            AuthorizationDecision::Allowed { .. }
        ));
        assert_eq!(
            ledger.authorize(&PrincipalId::from("agent-a"), "read", "project-1", 1_500),
            AuthorizationDecision::Denied {
                reason: DenialReason::GrantExpired
            }
        );
    }

    #[test]
    fn unknown_outcome_blocks_retry_until_reconciled() {
        let mut gate = MutationGate::new("op-1", "idem-1").expect("valid identity");
        assert_eq!(gate.begin_attempt(), Ok(1));
        assert_eq!(gate.state(), MutationState::OutcomeUnknown);
        assert_eq!(
            gate.begin_attempt(),
            Err(MutationError::AttemptBlocked(
                MutationBlock::OutcomeRequiresReconciliation
            ))
        );
        assert_eq!(
            gate.reconcile(ReconciledOutcome::Unknown),
            Ok(MutationState::OutcomeUnknown)
        );
        assert_eq!(
            gate.begin_attempt(),
            Err(MutationError::AttemptBlocked(
                MutationBlock::OutcomeRequiresReconciliation
            ))
        );
        assert_eq!(
            gate.reconcile(ReconciledOutcome::NoEffect),
            Ok(MutationState::ConfirmedNoEffect)
        );
        assert_eq!(gate.begin_attempt(), Ok(2));
    }

    #[test]
    fn confirmed_effect_blocks_duplicate_mutation() {
        let mut gate = MutationGate::new("op-2", "idem-2").expect("valid identity");
        gate.begin_attempt().expect("first attempt");
        gate.reconcile(ReconciledOutcome::EffectOccurred)
            .expect("sink confirms effect");
        assert_eq!(
            gate.begin_attempt(),
            Err(MutationError::AttemptBlocked(
                MutationBlock::EffectAlreadyConfirmed
            ))
        );
    }

    #[test]
    fn root_fixture_rejects_invalid_delegation_flag_depth_pair() {
        let mut ledger = AuthorityLedger::new();
        assert_eq!(
            ledger.insert_root_grant(grant(
                "bad-root", "owner", "agent-a", &["read"], &["project-1"],
                900, 1_500, true, 0, None
            )),
            Err(AuthorityError::DelegationFlagDepthMismatch)
        );
    }
}
