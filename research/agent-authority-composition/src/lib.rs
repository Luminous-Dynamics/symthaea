// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Synthetic benchmark oracle only. Not production IAM, crypto, identity,
// sandboxing, or a live authorization service.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Action {
    ReadSensitive,
    ReadUntrusted,
    WriteMemory,
    SendExternal,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DataLabel {
    Sensitive,
    Untrusted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grant {
    pub id: String,
    pub issuer: String,
    pub subject: String,
    pub tenant: String,
    pub resource: String,
    pub actions: BTreeSet<Action>,
    pub expires_at: u64,
    pub parent_id: Option<String>,
    pub revoked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Intent {
    pub principal: String,
    pub tenant: String,
    /// Exact resource/action pairs fixed before untrusted task input.
    pub allowed: BTreeSet<(String, Action)>,
    /// Synthetic fixture bit; not a cryptographic immutability guarantee.
    pub sealed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    pub labels: BTreeSet<DataLabel>,
    /// Synthetic IDs for grants that contributed protected source data.
    pub source_grants: BTreeSet<String>,
    /// Synthetic destination-bound declassification decision.
    pub declassified_for: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthorizationRequest {
    pub principal: String,
    pub agent: String,
    pub tenant: String,
    pub resource: String,
    pub action: Action,
    pub destination: Option<String>,
    pub now: u64,
    pub artifact: Option<Artifact>,
    /// Non-authoritative by construction; must not alter the decision.
    pub collective_support: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenialReason {
    IntentNotSealed,
    IntentMismatch,
    PrincipalOrTenantMismatch,
    NoGrant,
    RevokedGrant,
    ExpiredGrant,
    InvalidDelegationChain,
    DelegationCycle,
    RevokedSourceLineage,
    DangerousComposition,
    MissingDestination,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Decision {
    Allow { grant_id: String },
    Deny(DenialReason),
}

/// Small deterministic model for grading synthetic AACB fixtures.
/// Its input records are trusted fixture data, not authenticated credentials.
#[derive(Clone, Debug, Default)]
pub struct AuthorityModel {
    grants: BTreeMap<String, Grant>,
}

impl AuthorityModel {
    pub fn insert_synthetic_grant(&mut self, grant: Grant) {
        self.grants.insert(grant.id.clone(), grant);
    }

    pub fn revoke(&mut self, grant_id: &str) -> bool {
        match self.grants.get_mut(grant_id) {
            Some(grant) => {
                grant.revoked = true;
                true
            }
            None => false,
        }
    }

    pub fn authorize(&self, intent: &Intent, request: &AuthorizationRequest) -> Decision {
        if !intent.sealed {
            return Decision::Deny(DenialReason::IntentNotSealed);
        }
        if intent.principal != request.principal || intent.tenant != request.tenant {
            return Decision::Deny(DenialReason::PrincipalOrTenantMismatch);
        }
        if !intent.allowed.contains(&(request.resource.clone(), request.action.clone())) {
            return Decision::Deny(DenialReason::IntentMismatch);
        }

        if let Some(artifact) = &request.artifact {
            for source_id in &artifact.source_grants {
                match self.grants.get(source_id) {
                    Some(source) if !source.revoked => {}
                    _ => return Decision::Deny(DenialReason::RevokedSourceLineage),
                }
            }

            if request.action == Action::SendExternal
                && artifact.labels.contains(&DataLabel::Sensitive)
                && artifact.labels.contains(&DataLabel::Untrusted)
            {
                let Some(destination) = request.destination.as_deref() else {
                    return Decision::Deny(DenialReason::MissingDestination);
                };
                if artifact.declassified_for.as_deref() != Some(destination) {
                    return Decision::Deny(DenialReason::DangerousComposition);
                }
            }
        }

        let mut saw_candidate = false;
        let mut denial = DenialReason::NoGrant;
        for grant in self.grants.values().filter(|grant| {
            grant.subject == request.agent
                && grant.tenant == request.tenant
                && grant.resource == request.resource
                && grant.actions.contains(&request.action)
        }) {
            saw_candidate = true;
            match self.validate_chain(grant, request) {
                Ok(()) => return Decision::Allow { grant_id: grant.id.clone() },
                Err(reason) => denial = reason,
            }
        }

        if saw_candidate {
            Decision::Deny(denial)
        } else {
            Decision::Deny(DenialReason::NoGrant)
        }
    }

    fn validate_chain(
        &self,
        leaf: &Grant,
        request: &AuthorizationRequest,
    ) -> Result<(), DenialReason> {
        let mut current = leaf;
        let mut seen = BTreeSet::new();

        loop {
            if !seen.insert(current.id.clone()) {
                return Err(DenialReason::DelegationCycle);
            }
            if current.revoked {
                return Err(DenialReason::RevokedGrant);
            }
            if request.now >= current.expires_at {
                return Err(DenialReason::ExpiredGrant);
            }
            if current.tenant != request.tenant
                || current.resource != request.resource
                || !current.actions.contains(&request.action)
            {
                return Err(DenialReason::InvalidDelegationChain);
            }

            let Some(parent_id) = current.parent_id.as_deref() else {
                return if current.issuer == request.principal {
                    Ok(())
                } else {
                    Err(DenialReason::InvalidDelegationChain)
                };
            };

            let parent = self
                .grants
                .get(parent_id)
                .ok_or(DenialReason::InvalidDelegationChain)?;

            if parent.subject != current.issuer
                || parent.tenant != current.tenant
                || parent.resource != current.resource
                || !current.actions.is_subset(&parent.actions)
                || current.expires_at > parent.expires_at
            {
                return Err(DenialReason::InvalidDelegationChain);
            }
            current = parent;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectState {
    Dispatching,
    Unknown,
    Completed,
    Failed,
    ReconciledNotExecuted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DispatchError {
    AlreadyInFlight,
    OutcomeUnknown,
    AlreadyTerminal,
    NotUnknown,
}

/// Synthetic one-shot dispatch ledger. Reconciliation is assumed to be supplied
/// by an independent trusted observer; that independence is not implemented here.
#[derive(Clone, Debug, Default)]
pub struct DispatchLedger {
    states: BTreeMap<String, EffectState>,
    attempts: BTreeMap<String, u32>,
}

impl DispatchLedger {
    pub fn begin(&mut self, effect_id: &str) -> Result<u32, DispatchError> {
        match self.states.get(effect_id) {
            None | Some(EffectState::ReconciledNotExecuted) => {
                let attempts = self.attempts.entry(effect_id.to_owned()).or_default();
                *attempts += 1;
                self.states.insert(effect_id.to_owned(), EffectState::Dispatching);
                Ok(*attempts)
            }
            Some(EffectState::Dispatching) => Err(DispatchError::AlreadyInFlight),
            Some(EffectState::Unknown) => Err(DispatchError::OutcomeUnknown),
            Some(EffectState::Completed | EffectState::Failed) => {
                Err(DispatchError::AlreadyTerminal)
            }
        }
    }

    pub fn mark_unknown(&mut self, effect_id: &str) -> Result<(), DispatchError> {
        self.transition(effect_id, EffectState::Dispatching, EffectState::Unknown)
    }

    pub fn mark_completed(&mut self, effect_id: &str) -> Result<(), DispatchError> {
        self.transition(effect_id, EffectState::Dispatching, EffectState::Completed)
    }

    pub fn mark_failed(&mut self, effect_id: &str) -> Result<(), DispatchError> {
        self.transition(effect_id, EffectState::Dispatching, EffectState::Failed)
    }

    pub fn reconcile_not_executed(&mut self, effect_id: &str) -> Result<(), DispatchError> {
        self.transition(effect_id, EffectState::Unknown, EffectState::ReconciledNotExecuted)
    }

    pub fn state(&self, effect_id: &str) -> Option<EffectState> {
        self.states.get(effect_id).copied()
    }

    pub fn attempts(&self, effect_id: &str) -> u32 {
        self.attempts.get(effect_id).copied().unwrap_or_default()
    }

    fn transition(
        &mut self,
        effect_id: &str,
        expected: EffectState,
        next: EffectState,
    ) -> Result<(), DispatchError> {
        match self.states.get(effect_id) {
            Some(actual) if *actual == expected => {
                self.states.insert(effect_id.to_owned(), next);
                Ok(())
            }
            Some(EffectState::Unknown) => Err(DispatchError::OutcomeUnknown),
            Some(_) => Err(DispatchError::AlreadyTerminal),
            None => Err(DispatchError::NotUnknown),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_set(items: &[Action]) -> BTreeSet<Action> {
        items.iter().cloned().collect()
    }

    fn intent(resource: &str, action: Action) -> Intent {
        Intent {
            principal: "alice".into(),
            tenant: "tenant-a".into(),
            allowed: [(resource.into(), action)].into_iter().collect(),
            sealed: true,
        }
    }

    fn request(resource: &str, action: Action) -> AuthorizationRequest {
        AuthorizationRequest {
            principal: "alice".into(),
            agent: "agent-1".into(),
            tenant: "tenant-a".into(),
            resource: resource.into(),
            action,
            destination: None,
            now: 10,
            artifact: None,
            collective_support: 0.0,
        }
    }

    fn root_grant(id: &str, subject: &str, resource: &str, actions: &[Action]) -> Grant {
        Grant {
            id: id.into(),
            issuer: "alice".into(),
            subject: subject.into(),
            tenant: "tenant-a".into(),
            resource: resource.into(),
            actions: action_set(actions),
            expires_at: 100,
            parent_id: None,
            revoked: false,
        }
    }

    #[test]
    fn collective_support_never_mints_or_changes_authority() {
        let mut model = AuthorityModel::default();
        let intent = intent("mailbox", Action::ReadSensitive);
        let mut request = request("mailbox", Action::ReadSensitive);
        request.collective_support = 0.99;
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::NoGrant));

        model.insert_synthetic_grant(root_grant(
            "read", "agent-1", "mailbox", &[Action::ReadSensitive],
        ));
        let baseline = model.authorize(&intent, &request);
        request.collective_support = -10_000.0;
        assert_eq!(model.authorize(&intent, &request), baseline);
        request.collective_support = f64::NAN;
        assert_eq!(model.authorize(&intent, &request), baseline);
    }

    #[test]
    fn unsealed_intent_is_not_authoritative() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(root_grant(
            "read", "agent-1", "mailbox", &[Action::ReadSensitive],
        ));
        let mut intent = intent("mailbox", Action::ReadSensitive);
        intent.sealed = false;
        let request = request("mailbox", Action::ReadSensitive);
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::IntentNotSealed));
    }

    #[test]
    fn delegated_scope_must_be_a_subset_of_parent() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(root_grant(
            "root", "agent-1", "mailbox", &[Action::ReadSensitive],
        ));
        model.insert_synthetic_grant(Grant {
            id: "child".into(),
            issuer: "agent-1".into(),
            subject: "agent-2".into(),
            tenant: "tenant-a".into(),
            resource: "mailbox".into(),
            actions: action_set(&[Action::SendExternal]),
            expires_at: 90,
            parent_id: Some("root".into()),
            revoked: false,
        });
        let mut request = request("mailbox", Action::SendExternal);
        request.agent = "agent-2".into();
        let intent = intent("mailbox", Action::SendExternal);
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::InvalidDelegationChain));
    }

    #[test]
    fn revoked_parent_invalidates_child_delegation() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(root_grant(
            "root", "agent-1", "mailbox", &[Action::ReadSensitive],
        ));
        model.insert_synthetic_grant(Grant {
            id: "child".into(),
            issuer: "agent-1".into(),
            subject: "agent-2".into(),
            tenant: "tenant-a".into(),
            resource: "mailbox".into(),
            actions: action_set(&[Action::ReadSensitive]),
            expires_at: 90,
            parent_id: Some("root".into()),
            revoked: false,
        });
        assert!(model.revoke("root"));
        let mut request = request("mailbox", Action::ReadSensitive);
        request.agent = "agent-2".into();
        let intent = intent("mailbox", Action::ReadSensitive);
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::RevokedGrant));
    }

    #[test]
    fn delegation_cycles_are_rejected() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(Grant {
            id: "a".into(), issuer: "agent-2".into(), subject: "agent-1".into(),
            tenant: "tenant-a".into(), resource: "mailbox".into(),
            actions: action_set(&[Action::ReadSensitive]), expires_at: 90,
            parent_id: Some("b".into()), revoked: false,
        });
        model.insert_synthetic_grant(Grant {
            id: "b".into(), issuer: "agent-1".into(), subject: "agent-2".into(),
            tenant: "tenant-a".into(), resource: "mailbox".into(),
            actions: action_set(&[Action::ReadSensitive]), expires_at: 90,
            parent_id: Some("a".into()), revoked: false,
        });
        let mut request = request("mailbox", Action::ReadSensitive);
        request.agent = "agent-1".into();
        let intent = intent("mailbox", Action::ReadSensitive);
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::DelegationCycle));
    }

    #[test]
    fn revoked_source_grant_invalidates_derived_artifact_use() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(root_grant(
            "egress", "agent-1", "export", &[Action::SendExternal],
        ));
        model.insert_synthetic_grant(root_grant(
            "source-read", "agent-1", "source", &[Action::ReadSensitive],
        ));
        assert!(model.revoke("source-read"));
        let intent = intent("export", Action::SendExternal);
        let mut request = request("export", Action::SendExternal);
        request.destination = Some("partner.example".into());
        request.artifact = Some(Artifact {
            labels: [DataLabel::Sensitive].into_iter().collect(),
            source_grants: ["source-read".into()].into_iter().collect(),
            declassified_for: None,
        });
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::RevokedSourceLineage));
    }

    #[test]
    fn dangerous_triad_requires_destination_specific_declassification() {
        let mut model = AuthorityModel::default();
        model.insert_synthetic_grant(root_grant(
            "egress", "agent-1", "export", &[Action::SendExternal],
        ));
        let intent = intent("export", Action::SendExternal);
        let mut request = request("export", Action::SendExternal);
        request.destination = Some("partner.example".into());
        request.artifact = Some(Artifact {
            labels: [DataLabel::Sensitive, DataLabel::Untrusted].into_iter().collect(),
            source_grants: BTreeSet::new(),
            declassified_for: None,
        });
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::DangerousComposition));
        request.artifact.as_mut().unwrap().declassified_for = Some("partner.example".into());
        assert!(matches!(model.authorize(&intent, &request), Decision::Allow { .. }));
        request.destination = Some("attacker.example".into());
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::DangerousComposition));
    }

    #[test]
    fn expired_grant_does_not_authorize() {
        let mut model = AuthorityModel::default();
        let mut grant = root_grant("grant", "agent-1", "mailbox", &[Action::ReadSensitive]);
        grant.expires_at = 10;
        model.insert_synthetic_grant(grant);
        let intent = intent("mailbox", Action::ReadSensitive);
        let request = request("mailbox", Action::ReadSensitive);
        assert_eq!(model.authorize(&intent, &request), Decision::Deny(DenialReason::ExpiredGrant));
    }

    #[test]
    fn unknown_remote_outcome_cannot_be_retried_before_reconciliation() {
        let mut ledger = DispatchLedger::default();
        assert_eq!(ledger.begin("effect-1"), Ok(1));
        assert_eq!(ledger.mark_unknown("effect-1"), Ok(()));
        assert_eq!(ledger.begin("effect-1"), Err(DispatchError::OutcomeUnknown));
        assert_eq!(ledger.attempts("effect-1"), 1);
        assert_eq!(ledger.reconcile_not_executed("effect-1"), Ok(()));
        assert_eq!(ledger.begin("effect-1"), Ok(2));
        assert_eq!(ledger.mark_completed("effect-1"), Ok(()));
        assert_eq!(ledger.begin("effect-1"), Err(DispatchError::AlreadyTerminal));
        assert_eq!(ledger.attempts("effect-1"), 2);
    }
}
