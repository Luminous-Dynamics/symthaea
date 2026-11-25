// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # SAT reference model
//!
//! A deliberately small, dependency-free reference oracle for bounded
//! authority and delegation semantics in the Sovereignty Assurance Tournament.
//!
//! This crate is **not** a production authorization system. It does not verify
//! signatures, authenticate issuers, persist revocation state, fence a real
//! effect sink, establish trusted time, implement declassification, inspect
//! payload content, or provide process isolation. Root grants are trusted
//! fixture inputs supplied by the test harness. The implementation exists to
//! give production adapters an independently authored behavioral oracle.
//!
//! The authorization API intentionally accepts no intelligence, confidence,
//! consciousness, collective-score, reputation, or model-generated trust
//! parameter. Such signals may be evaluated by a separate caution policy, but
//! they cannot create a grant or satisfy this model's authority predicate.
//!
//! Every request binds an exact action, resource, sink, purpose, and digest of
//! canonical operation parameters. Payload labels must be a subset of the
//! grant's allowed labels. This is an authority/label-scope check, not proof
//! that labels are truthful or that disclosure to an authorized sink is safe.
//! Time windows are half-open: not_before <= now < expires_at.
//! Delegated scopes and validity windows must be subsets of the parent, and
//! each delegation consumes one unit of remaining depth.

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

/// One permitted operation family.
///
/// Identity fields must match exactly. A delegate may narrow the permitted
/// payload-label set, but cannot change the action, resource, sink, purpose or
/// operation-parameter digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EffectScope {
    pub action: String,
    pub resource: String,
    pub sink: String,
    pub purpose: String,
    pub parameters_digest: String,
    pub allowed_labels: BTreeSet<String>,
}

impl EffectScope {
    pub fn new<I, S>(
        action: impl Into<String>,
        resource: impl Into<String>,
        sink: impl Into<String>,
        purpose: impl Into<String>,
        parameters_digest: impl Into<String>,
        allowed_labels: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            action: action.into(),
            resource: resource.into(),
            sink: sink.into(),
            purpose: purpose.into(),
            parameters_digest: parameters_digest.into(),
            allowed_labels: allowed_labels.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether this scope is wholly contained by the parent's corresponding
    /// scope. Identity dimensions must match and labels may only narrow.
    pub fn attenuates(&self, parent: &Self) -> bool {
        self.action == parent.action
            && self.resource == parent.resource
            && self.sink == parent.sink
            && self.purpose == parent.purpose
            && self.parameters_digest == parent.parameters_digest
            && self.allowed_labels.is_subset(&parent.allowed_labels)
    }

    fn permits(&self, request: &EffectRequest) -> bool {
        self.action == request.action
            && self.resource == request.resource
            && self.sink == request.sink
            && self.purpose == request.purpose
            && self.parameters_digest == request.parameters_digest
            && request.payload_labels.is_subset(&self.allowed_labels)
    }
}

/// Exact request presented to the authority reference monitor.
///
/// The payload labels must come from a trusted source/connector in production.
/// This reference model treats them as input data and does not authenticate
/// their provenance or detect relabeling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectRequest {
    pub action: String,
    pub resource: String,
    pub sink: String,
    pub purpose: String,
    pub parameters_digest: String,
    pub payload_labels: BTreeSet<String>,
}

impl EffectRequest {
    pub fn new<I, S>(
        action: impl Into<String>,
        resource: impl Into<String>,
        sink: impl Into<String>,
        purpose: impl Into<String>,
        parameters_digest: impl Into<String>,
        payload_labels: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            action: action.into(),
            resource: resource.into(),
            sink: sink.into(),
            purpose: purpose.into(),
            parameters_digest: parameters_digest.into(),
            payload_labels: payload_labels.into_iter().map(Into::into).collect(),
        }
    }
}

/// A bounded capability grant. All effect scopes are exact operation families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityGrant {
    pub id: GrantId,
    pub issuer: PrincipalId,
    pub subject: PrincipalId,
    pub effect_scope: BTreeSet<EffectScope>,
    pub not_before: u64,
    pub expires_at: u64,
    pub may_delegate: bool,
    pub remaining_delegation_depth: u8,
    pub parent_id: Option<GrantId>,
}

impl CapabilityGrant {
    #[allow(clippy::too_many_arguments)]
    pub fn new<I>(
        id: impl Into<GrantId>,
        issuer: impl Into<PrincipalId>,
        subject: impl Into<PrincipalId>,
        effects: I,
        not_before: u64,
        expires_at: u64,
        may_delegate: bool,
        remaining_delegation_depth: u8,
        parent_id: Option<GrantId>,
    ) -> Self
    where
        I: IntoIterator<Item = EffectScope>,
    {
        Self {
            id: id.into(),
            issuer: issuer.into(),
            subject: subject.into(),
            effect_scope: effects.into_iter().collect(),
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
    PayloadLabelOutsideGrant,
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

/// The result of checking one exact operation against the ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizationDecision {
    Allowed { grant_id: GrantId },
    Denied { reason: DenialReason },
}

/// Errors encountered while constructing reference authority state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityError {
    EmptyIdentifier,
    EmptyEffectScope,
    InvalidValidityWindow,
    DelegationFlagDepthMismatch,
    DuplicateGrant,
    RootGrantHasParent,
    ParentGrantMissing,
    ParentGrantInactive,
    ParentCannotDelegate,
    ChildParentMismatch,
    DelegatorIdentityMismatch,
    EffectScopeExpansion,
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
/// Production callers must authenticate root issuance, verify label provenance
/// and enforce decisions at the actual side-effect sink independently.
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
    /// grant as its lineage predecessor. Each child effect scope must match a
    /// parent scope while narrowing or preserving its allowed-label set.
    /// This is a semantic reference check, not a cryptographic verifier.
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

        let scopes_attenuate = child.effect_scope.iter().all(|child_scope| {
            parent
                .effect_scope
                .iter()
                .any(|parent_scope| child_scope.attenuates(parent_scope))
        });
        if !scopes_attenuate {
            return Err(AuthorityError::EffectScopeExpansion);
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

    /// Authorize one exact subject/effect request.
    ///
    /// There is deliberately no cognitive-score parameter. A higher or lower
    /// intelligence estimate cannot grant or remove execution authority.
    pub fn authorize(
        &self,
        subject: &PrincipalId,
        request: &EffectRequest,
        now: u64,
    ) -> AuthorizationDecision {
        let mut scope_match = false;
        let mut last_reason = DenialReason::NoMatchingGrant;

        for grant in self
            .grants
            .values()
            .filter(|grant| &grant.subject == subject)
        {
            let exact_identity_match = grant.effect_scope.iter().any(|scope| {
                scope.action == request.action
                    && scope.resource == request.resource
                    && scope.sink == request.sink
                    && scope.purpose == request.purpose
                    && scope.parameters_digest == request.parameters_digest
            });
            if !exact_identity_match {
                continue;
            }

            scope_match = true;
            if let Err(reason) = self.lineage_status(&grant.id, now) {
                last_reason = reason;
                continue;
            }

            if grant.effect_scope.iter().any(|scope| scope.permits(request)) {
                return AuthorizationDecision::Allowed {
                    grant_id: grant.id.clone(),
                };
            }

            last_reason = DenialReason::PayloadLabelOutsideGrant;
        }

        AuthorizationDecision::Denied {
            reason: if scope_match {
                last_reason
            } else {
                DenialReason::NoMatchingGrant
            },
        }
    }

    /// Stable inventory view; this does not imply returned grants are live.
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
        let mut cursor = grant_id.clone();
        let mut depth = 0usize;
        let mut visited = BTreeSet::new();

        loop {
            if !visited.insert(cursor.clone()) {
                return Err(DenialReason::InvalidLineage);
            }
            let grant = self
                .grants
                .get(&cursor)
                .ok_or(DenialReason::InvalidLineage)?;

            if self.revoked.contains(&cursor) {
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

            match grant.parent_id.clone() {
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
    if grant.effect_scope.is_empty() {
        return Err(AuthorityError::EmptyEffectScope);
    }
    if grant.not_before >= grant.expires_at {
        return Err(AuthorityError::InvalidValidityWindow);
    }
    if grant.may_delegate != (grant.remaining_delegation_depth > 0) {
        return Err(AuthorityError::DelegationFlagDepthMismatch);
    }
    if grant.effect_scope.iter().any(|scope| {
        scope.action.trim().is_empty()
            || scope.resource.trim().is_empty()
            || scope.sink.trim().is_empty()
            || scope.purpose.trim().is_empty()
            || scope.parameters_digest.trim().is_empty()
            || scope.allowed_labels.iter().any(|label| label.trim().is_empty())
    }) {
        return Err(AuthorityError::EmptyIdentifier);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000;

    fn scope(
        action: &str,
        resource: &str,
        sink: &str,
        purpose: &str,
        labels: &[&str],
        digest: &str,
    ) -> EffectScope {
        EffectScope::new(action, resource, sink, purpose, digest, labels.iter().copied())
    }

    fn request(
        action: &str,
        resource: &str,
        sink: &str,
        purpose: &str,
        labels: &[&str],
        digest: &str,
    ) -> EffectRequest {
        EffectRequest::new(action, resource, sink, purpose, digest, labels.iter().copied())
    }

    #[allow(clippy::too_many_arguments)]
    fn grant(
        id: &str,
        issuer: &str,
        subject: &str,
        effects: &[EffectScope],
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
            effects.iter().cloned(),
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
                &[
                    scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1"),
                    scope("write", "project-1", "project-api", "project-review",
                        &["public"], "write-p1-v1"),
                    scope("read", "project-2", "project-api", "project-review",
                        &["public"], "read-p2-v1"),
                    scope("write", "project-2", "project-api", "project-review",
                        &["public"], "write-p2-v1"),
                ],
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
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("write", "project-1", "project-api", "project-review",
                    &["public"], "write-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::NoMatchingGrant
            }
        );
    }

    #[test]
    fn exact_effect_binding_authorizes() {
        let ledger = root_ledger();
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("write", "project-1", "project-api", "project-review",
                    &["public"], "write-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Allowed {
                grant_id: GrantId::from("root")
            }
        );
    }

    #[test]
    fn action_resource_sink_purpose_or_parameter_change_does_not_match() {
        let ledger = root_ledger();
        for attempted in [
            request("admin", "project-1", "project-api", "project-review",
                &["public"], "write-p1-v1"),
            request("write", "project-3", "project-api", "project-review",
                &["public"], "write-p1-v1"),
            request("write", "project-1", "external-webhook", "project-review",
                &["public"], "write-p1-v1"),
            request("write", "project-1", "project-api", "data-export",
                &["public"], "write-p1-v1"),
            request("write", "project-1", "project-api", "project-review",
                &["public"], "write-p1-other-amount-v1"),
        ] {
            assert_eq!(
                ledger.authorize(&PrincipalId::from("agent-a"), &attempted, NOW),
                AuthorizationDecision::Denied {
                    reason: DenialReason::NoMatchingGrant
                }
            );
        }
    }

    #[test]
    fn payload_labels_must_be_within_grant_label_scope() {
        let ledger = root_ledger();
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public", "restricted"], "read-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::PayloadLabelOutsideGrant
            }
        );
    }

    #[test]
    fn child_scope_may_narrow_payload_labels() {
        let mut ledger = root_ledger();
        ledger
            .delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &[], "read-p1-v1")],
                    950,
                    1_400,
                    true,
                    1,
                    Some("root"),
                ),
                NOW,
            )
            .expect("label-narrowed child grant is admissible");
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-b"),
                &request("read", "project-1", "project-api", "project-review",
                    &[], "read-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Allowed {
                grant_id: GrantId::from("child")
            }
        );
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-b"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::PayloadLabelOutsideGrant
            }
        );
    }

    #[test]
    fn child_cannot_expand_effect_scope_or_change_purpose_or_sink() {
        let mut ledger = root_ledger();
        for attempted in [
            scope("admin", "project-1", "project-api", "project-review",
                &["public"], "admin-p1-v1"),
            scope("read", "project-1", "external-webhook", "project-review",
                &["public"], "read-p1-v1"),
            scope("read", "project-1", "project-api", "data-export",
                &["public"], "read-p1-v1"),
        ] {
            assert_eq!(
                ledger.delegate(
                    &GrantId::from("root"),
                    grant(
                        "child",
                        "agent-a",
                        "agent-b",
                        &[attempted],
                        950,
                        1_400,
                        false,
                        0,
                        Some("root"),
                    ),
                    NOW,
                ),
                Err(AuthorityError::EffectScopeExpansion)
            );
        }
    }

    #[test]
    fn child_cannot_expand_payload_label_scope() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public", "restricted"], "read-p1-v1")],
                    950,
                    1_400,
                    false,
                    0,
                    Some("root"),
                ),
                NOW,
            ),
            Err(AuthorityError::EffectScopeExpansion)
        );
    }

    #[test]
    fn child_cannot_extend_parent_lifetime() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_501,
                    false,
                    0,
                    Some("root"),
                ),
                NOW,
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
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_400,
                    true,
                    2,
                    Some("root"),
                ),
                NOW,
            ),
            Err(AuthorityError::DelegationDepthExpansion)
        );
    }

    #[test]
    fn non_delegable_parent_cannot_issue_a_child() {
        let mut ledger = AuthorityLedger::new();
        ledger
            .insert_root_grant(grant(
                "root",
                "owner",
                "agent-a",
                &[scope("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1")],
                900,
                1_500,
                false,
                0,
                None,
            ))
            .expect("non-delegable root is valid");
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_400,
                    false,
                    0,
                    Some("root"),
                ),
                NOW,
            ),
            Err(AuthorityError::ParentCannotDelegate)
        );
    }

    #[test]
    fn grant_is_not_valid_before_its_start_time() {
        let ledger = root_ledger();
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                899,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::GrantNotYetValid
            }
        );
    }

    #[test]
    fn delegator_must_be_the_parent_subject() {
        let mut ledger = root_ledger();
        assert_eq!(
            ledger.delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "attacker",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_400,
                    false,
                    0,
                    Some("root"),
                ),
                NOW,
            ),
            Err(AuthorityError::DelegatorIdentityMismatch)
        );
    }

    #[test]
    fn delegation_depth_is_consumed_across_multiple_levels() {
        let mut ledger = root_ledger();
        ledger
            .delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_400,
                    true,
                    1,
                    Some("root"),
                ),
                NOW,
            )
            .expect("first delegation");
        ledger
            .delegate(
                &GrantId::from("child"),
                grant(
                    "grandchild",
                    "agent-b",
                    "agent-c",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    975,
                    1_300,
                    false,
                    0,
                    Some("child"),
                ),
                NOW,
            )
            .expect("second delegation");
        assert!(matches!(
            ledger.authorize(
                &PrincipalId::from("agent-c"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Allowed { .. }
        ));
    }

    #[test]
    fn revoking_parent_invalidates_existing_descendants() {
        let mut ledger = root_ledger();
        ledger
            .delegate(
                &GrantId::from("root"),
                grant(
                    "child",
                    "agent-a",
                    "agent-b",
                    &[scope("read", "project-1", "project-api", "project-review",
                        &["public"], "read-p1-v1")],
                    950,
                    1_400,
                    false,
                    0,
                    Some("root"),
                ),
                NOW,
            )
            .expect("valid child");
        ledger.revoke(&GrantId::from("root")).expect("root exists");
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-b"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                NOW,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::AncestorRevoked
            }
        );
    }

    #[test]
    fn validity_window_is_half_open() {
        let ledger = root_ledger();
        assert!(matches!(
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                900,
            ),
            AuthorizationDecision::Allowed { .. }
        ));
        assert_eq!(
            ledger.authorize(
                &PrincipalId::from("agent-a"),
                &request("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1"),
                1_500,
            ),
            AuthorizationDecision::Denied {
                reason: DenialReason::GrantExpired
            }
        );
    }

    #[test]
    fn root_fixture_rejects_invalid_delegation_flag_depth_pair() {
        let mut ledger = AuthorityLedger::new();
        assert_eq!(
            ledger.insert_root_grant(grant(
                "bad-root",
                "owner",
                "agent-a",
                &[scope("read", "project-1", "project-api", "project-review",
                    &["public"], "read-p1-v1")],
                900,
                1_500,
                true,
                0,
                None,
            )),
            Err(AuthorityError::DelegationFlagDepthMismatch)
        );
    }
}
