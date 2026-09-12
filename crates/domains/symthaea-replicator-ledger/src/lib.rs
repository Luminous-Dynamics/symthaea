// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only lineage and budget ledger for the Replicator Safety Kernel (RSK).
//!
//! This crate records authority-relevant lineage state. It contains no physical,
//! molecular, biological, manufacturing, or other replication mechanism.
//!
//! Safety goals:
//! - creation never grants authority;
//! - concurrent/stale decisions cannot double-spend the same ledger snapshot;
//! - evaluated resource intent cannot be substituted at commit time;
//! - descendant creation consumes both subject-subtree and lineage-tree budgets;
//! - ancestor grant ceilings remain binding on later descendants;
//! - capability ceilings attenuate across descendant creation and cannot widen;
//! - descendant lineages cannot widen capabilities, budgets, or lower risk class;
//! - quarantine/revocation propagates down subject and lineage ancestry;
//! - a later grant generation may supersede an earlier subject budget scope, but
//!   stale or same-generation-mutated authority can never regain authority;
//! - runtime safety evidence is rechecked at commit time.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use symthaea_replicator_safety::{
    AuthorizedReplication, CapabilitySet, DenialReason, EvidenceDigest, GrantId, LineageId,
    LineageStatus, ReplicationAuthorityRequest, ReplicationBudgetSnapshot, ReplicationDecision,
    ReplicationGrant, RiskClass, SubjectId, evaluate_replication_authority,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LedgerEpochId([u8; 32]);
impl LedgerEpochId {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MutationId([u8; 32]);
impl MutationId {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LedgerCursor {
    pub epoch: LedgerEpochId,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineagePolicy {
    pub risk_class: RiskClass,
    pub capability_ceiling: CapabilitySet,
    pub max_direct_children_per_subject: u64,
    pub max_total_descendants: u64,
    pub max_lineage_depth: u32,
    pub max_resource_units: u64,
}
impl LineagePolicy {
    /// A branch is constitutional only when it cannot widen an ancestor policy.
    pub const fn is_no_more_permissive_than(self, parent: Self) -> bool {
        self.risk_class >= parent.risk_class
            && self.capability_ceiling.is_subset_of(parent.capability_ceiling)
            && self.max_direct_children_per_subject <= parent.max_direct_children_per_subject
            && self.max_total_descendants <= parent.max_total_descendants
            && self.max_lineage_depth <= parent.max_lineage_depth
            && self.max_resource_units <= parent.max_resource_units
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeSafetyWitness {
    pub monitoring_healthy: bool,
    pub monitoring_fresh: bool,
    pub policy_current: bool,
    pub evidence_current: bool,
    pub containment_matches: bool,
    pub safety_case_digest: EvidenceDigest,
    pub containment_envelope_digest: EvidenceDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerAuthorityContext {
    pub cursor: LedgerCursor,
    pub risk_class: RiskClass,
    pub parent_capability_ceiling: CapabilitySet,
    pub quarantined: bool,
    pub revoked: bool,
    pub lineage_status: LineageStatus,
    pub budget: ReplicationBudgetSnapshot,
}

/// Opaque authorization token produced only by the bound evaluator below.
///
/// Private fields prevent callers from constructing a token with a risk class,
/// capability set, evidence binding, action amount, or budget ceiling that was
/// not part of a successful RSK evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedReplicationAuthorization {
    authorized: AuthorizedReplication,
    cursor: LedgerCursor,
    risk_class: RiskClass,
    evaluated_at_unix_secs: u64,
    requested_resource_units: u64,
    max_direct_children: u64,
    max_total_descendants: u64,
    max_lineage_depth: u32,
    max_resource_units: u64,
}
impl BoundedReplicationAuthorization {
    pub const fn subject(&self) -> SubjectId { self.authorized.subject }
    pub const fn lineage(&self) -> LineageId { self.authorized.lineage }
    pub const fn grant_id(&self) -> GrantId { self.authorized.grant_id }
    pub const fn grant_generation(&self) -> u64 { self.authorized.grant_generation }
    pub const fn effective_capabilities(&self) -> CapabilitySet {
        self.authorized.effective_capabilities
    }
    pub const fn risk_class(&self) -> RiskClass { self.risk_class }
    pub const fn evaluated_at_unix_secs(&self) -> u64 { self.evaluated_at_unix_secs }
    /// Exact resource amount accepted by the authority evaluation for this action.
    pub const fn requested_resource_units(&self) -> u64 { self.requested_resource_units }
    pub const fn cursor(&self) -> LedgerCursor { self.cursor }
    pub const fn expires_at_unix_secs(&self) -> u64 { self.authorized.expires_at_unix_secs }
    pub const fn safety_case_digest(&self) -> EvidenceDigest {
        self.authorized.safety_case_digest
    }
    pub const fn containment_envelope_digest(&self) -> EvidenceDigest {
        self.authorized.containment_envelope_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundReplicationDecision {
    Allow(BoundedReplicationAuthorization),
    Deny { reasons: Vec<DenialReason> },
}
impl BoundReplicationDecision {
    pub fn is_allowed(&self) -> bool { matches!(self, Self::Allow(_)) }
    pub fn denial_reasons(&self) -> &[DenialReason] {
        match self { Self::Allow(_) => &[], Self::Deny { reasons } => reasons }
    }
}

/// Run the constitutional evaluator and bind an allow decision to one ledger
/// cursor plus the exact risk, action amount, and grant-budget values that were evaluated.
pub fn evaluate_bound_replication_authority(
    cursor: LedgerCursor,
    request: &ReplicationAuthorityRequest,
    grant: Option<&ReplicationGrant>,
) -> BoundReplicationDecision {
    match evaluate_replication_authority(request, grant) {
        ReplicationDecision::Deny { reasons } => BoundReplicationDecision::Deny { reasons },
        ReplicationDecision::Allow(authorized) => {
            let Some(grant) = grant else {
                return BoundReplicationDecision::Deny {
                    reasons: vec![DenialReason::MissingExplicitGrant],
                };
            };
            BoundReplicationDecision::Allow(BoundedReplicationAuthorization {
                authorized,
                cursor,
                risk_class: request.risk_class,
                evaluated_at_unix_secs: request.now_unix_secs,
                requested_resource_units: request.budget.requested_resource_units,
                max_direct_children: grant.max_direct_children,
                max_total_descendants: grant.max_total_descendants,
                max_lineage_depth: grant.max_lineage_depth,
                max_resource_units: grant.max_resource_units,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NegativeState { Clear, Quarantined, Revoked }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AuthorityScope {
    grant_id: GrantId,
    generation: u64,
    lineage: LineageId,
    risk_class: RiskClass,
    effective_capabilities: CapabilitySet,
    expires_at_unix_secs: u64,
    safety_case_digest: EvidenceDigest,
    containment_envelope_digest: EvidenceDigest,
    max_direct_children: u64,
    max_total_descendants: u64,
    max_lineage_depth: u32,
    max_resource_units: u64,
}
impl AuthorityScope {
    fn from_authorization(a: &BoundedReplicationAuthorization) -> Self {
        Self {
            grant_id: a.grant_id(),
            generation: a.grant_generation(),
            lineage: a.lineage(),
            risk_class: a.risk_class(),
            effective_capabilities: a.effective_capabilities(),
            expires_at_unix_secs: a.expires_at_unix_secs(),
            safety_case_digest: a.safety_case_digest(),
            containment_envelope_digest: a.containment_envelope_digest(),
            max_direct_children: a.max_direct_children,
            max_total_descendants: a.max_total_descendants,
            max_lineage_depth: a.max_lineage_depth,
            max_resource_units: a.max_resource_units,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SubjectRecord {
    parent: Option<SubjectId>,
    lineage: LineageId,
    capability_ceiling: CapabilitySet,
    depth: u32,
    direct_children: u64,
    subtree_descendants: u64,
    subtree_resource_units: u64,
    state: NegativeState,
    authority_scope: Option<AuthorityScope>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LineageRecord {
    parent_lineage: Option<LineageId>,
    policy: LineagePolicy,
    subtree_descendants: u64,
    subtree_resource_units: u64,
    state: NegativeState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubjectLedgerSnapshot {
    pub parent: Option<SubjectId>,
    pub lineage: LineageId,
    /// Maximum capability set any future explicit grant may activate for this
    /// subject. This is a ceiling, not authority.
    pub capability_ceiling: CapabilitySet,
    pub depth: u32,
    pub direct_children: u64,
    pub subtree_descendants: u64,
    pub subtree_resource_units: u64,
    pub quarantined: bool,
    pub revoked: bool,
    pub authority_grant_id: Option<GrantId>,
    pub authority_generation: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineageLedgerSnapshot {
    pub parent_lineage: Option<LineageId>,
    pub policy: LineagePolicy,
    pub subtree_descendants: u64,
    pub subtree_resource_units: u64,
    pub quarantined: bool,
    pub revoked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerEventKind {
    RootRegistered {
        subject: SubjectId,
        lineage: LineageId,
        policy: LineagePolicy,
    },
    LineageBranchRegistered {
        mutation_id: MutationId,
        lineage: LineageId,
        parent_lineage: LineageId,
        policy: LineagePolicy,
    },
    DescendantCommitted {
        mutation_id: MutationId,
        parent: SubjectId,
        child: SubjectId,
        lineage: LineageId,
        child_capability_ceiling: CapabilitySet,
        grant_id: GrantId,
        grant_generation: u64,
        resource_units: u64,
        safety_case_digest: EvidenceDigest,
        containment_envelope_digest: EvidenceDigest,
    },
    SubjectQuarantined { mutation_id: MutationId, subject: SubjectId },
    SubjectRevoked { mutation_id: MutationId, subject: SubjectId },
    LineageQuarantined { mutation_id: MutationId, lineage: LineageId },
    LineageRevoked { mutation_id: MutationId, lineage: LineageId },
    GrantRevoked {
        mutation_id: MutationId,
        grant_id: GrantId,
        generation: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEvent {
    pub sequence: u64,
    pub kind: LedgerEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    CursorMismatch { expected: LedgerCursor, actual: LedgerCursor },
    SequenceOverflow,
    DuplicateMutation(MutationId),
    DuplicateSubject(SubjectId),
    DuplicateLineage(LineageId),
    UnknownSubject(SubjectId),
    UnknownLineage(LineageId),
    SubjectCycle(SubjectId),
    LineageCycle(LineageId),
    ParentLineageMismatch,
    BranchPolicyEscalation,
    SubjectQuarantined(SubjectId),
    SubjectRevoked(SubjectId),
    LineageQuarantined(LineageId),
    LineageRevoked(LineageId),
    SubjectAlreadyQuarantined(SubjectId),
    SubjectAlreadyRevoked(SubjectId),
    LineageAlreadyQuarantined(LineageId),
    LineageAlreadyRevoked(LineageId),
    GrantAlreadyRevoked { grant_id: GrantId, generation: u64 },
    GrantRevoked { grant_id: GrantId, generation: u64 },
    AncestorGrantRevoked(SubjectId),
    StaleGrantGeneration,
    AuthorityScopeMutation,
    AuthorityExpired,
    ActionResourceMismatch { authorized: u64, committed: u64 },
    AuthorityRiskMismatch,
    EmptyAuthorityCapabilities,
    AuthorityCapabilitiesExceedSubjectCeiling,
    AuthorityCapabilitiesExceedHardCeiling,
    AuthorityBudgetExceedsHardCeiling,
    RuntimeMonitorUnhealthy,
    RuntimeMonitorStale,
    RuntimePolicyStale,
    RuntimeEvidenceStale,
    RuntimeContainmentDrift,
    RuntimeSafetyCaseDigestMismatch,
    RuntimeContainmentDigestMismatch,
    DirectChildBudgetExhausted,
    SubjectDescendantBudgetExhausted(SubjectId),
    SubjectResourceBudgetExhausted(SubjectId),
    SubjectDepthBudgetExhausted(SubjectId),
    MissingAncestorAuthorityScope(SubjectId),
    LineageDescendantBudgetExhausted(LineageId),
    LineageResourceBudgetExhausted(LineageId),
    LineageDepthBudgetExhausted(LineageId),
    ArithmeticOverflow,
}

#[derive(Clone, Debug)]
pub struct ReplicationLedger {
    epoch: LedgerEpochId,
    cursor: LedgerCursor,
    subjects: BTreeMap<SubjectId, SubjectRecord>,
    lineages: BTreeMap<LineageId, LineageRecord>,
    revoked_grants: BTreeSet<(GrantId, u64)>,
    used_mutations: BTreeSet<MutationId>,
    events: Vec<LedgerEvent>,
}

impl ReplicationLedger {
    pub fn new(
        epoch: LedgerEpochId,
        root_subject: SubjectId,
        root_lineage: LineageId,
        root_policy: LineagePolicy,
    ) -> Self {
        let cursor = LedgerCursor { epoch, sequence: 0 };
        let mut subjects = BTreeMap::new();
        subjects.insert(root_subject, SubjectRecord {
            parent: None,
            lineage: root_lineage,
            capability_ceiling: root_policy.capability_ceiling,
            depth: 0,
            direct_children: 0,
            subtree_descendants: 0,
            subtree_resource_units: 0,
            state: NegativeState::Clear,
            authority_scope: None,
        });
        let mut lineages = BTreeMap::new();
        lineages.insert(root_lineage, LineageRecord {
            parent_lineage: None,
            policy: root_policy,
            subtree_descendants: 0,
            subtree_resource_units: 0,
            state: NegativeState::Clear,
        });
        Self {
            epoch,
            cursor,
            subjects,
            lineages,
            revoked_grants: BTreeSet::new(),
            used_mutations: BTreeSet::new(),
            events: vec![LedgerEvent {
                sequence: 0,
                kind: LedgerEventKind::RootRegistered {
                    subject: root_subject,
                    lineage: root_lineage,
                    policy: root_policy,
                },
            }],
        }
    }

    pub const fn cursor(&self) -> LedgerCursor { self.cursor }
    pub fn events(&self) -> &[LedgerEvent] { &self.events }

    pub fn verify_append_only_structure(&self) -> bool {
        if self.events.is_empty() { return false; }
        for (index, event) in self.events.iter().enumerate() {
            let Ok(index) = u64::try_from(index) else { return false; };
            if event.sequence != index { return false; }
        }
        self.events.last().is_some_and(|event| event.sequence == self.cursor.sequence)
            && self.cursor.epoch == self.epoch
    }

    pub fn subject_snapshot(&self, subject: SubjectId) -> Result<SubjectLedgerSnapshot, LedgerError> {
        let record = self.subjects.get(&subject).ok_or(LedgerError::UnknownSubject(subject))?;
        Ok(SubjectLedgerSnapshot {
            parent: record.parent,
            lineage: record.lineage,
            capability_ceiling: record.capability_ceiling,
            depth: record.depth,
            direct_children: record.direct_children,
            subtree_descendants: record.subtree_descendants,
            subtree_resource_units: record.subtree_resource_units,
            quarantined: record.state == NegativeState::Quarantined,
            revoked: record.state == NegativeState::Revoked,
            authority_grant_id: record.authority_scope.map(|s| s.grant_id),
            authority_generation: record.authority_scope.map(|s| s.generation),
        })
    }

    pub fn lineage_snapshot(&self, lineage: LineageId) -> Result<LineageLedgerSnapshot, LedgerError> {
        let record = self.lineages.get(&lineage).ok_or(LedgerError::UnknownLineage(lineage))?;
        Ok(LineageLedgerSnapshot {
            parent_lineage: record.parent_lineage,
            policy: record.policy,
            subtree_descendants: record.subtree_descendants,
            subtree_resource_units: record.subtree_resource_units,
            quarantined: record.state == NegativeState::Quarantined,
            revoked: record.state == NegativeState::Revoked,
        })
    }

    /// Produce the ledger-owned portion of an RSK authority request.
    ///
    /// The returned cursor should be passed to `evaluate_bound_replication_authority`.
    /// Any ledger mutation before commit makes that authorization stale.
    pub fn authority_context(
        &self,
        parent: SubjectId,
        output_lineage: LineageId,
        requested_resource_units: u64,
    ) -> Result<LedgerAuthorityContext, LedgerError> {
        let parent_record = self.subjects.get(&parent).ok_or(LedgerError::UnknownSubject(parent))?;
        let output_record = self.lineages.get(&output_lineage)
            .ok_or(LedgerError::UnknownLineage(output_lineage))?;

        let subject_ancestry = self.subject_ancestry(parent)?;
        let lineage_ancestry = self.lineage_ancestry(output_lineage)?;
        let mut quarantined = false;
        let mut revoked = false;
        for subject in subject_ancestry {
            match self.subjects.get(&subject).expect("validated ancestry").state {
                NegativeState::Clear => {},
                NegativeState::Quarantined => quarantined = true,
                NegativeState::Revoked => revoked = true,
            }
        }
        for lineage in lineage_ancestry {
            match self.lineages.get(&lineage).expect("validated ancestry").state {
                NegativeState::Clear => {},
                NegativeState::Quarantined => quarantined = true,
                NegativeState::Revoked => revoked = true,
            }
        }

        let parent_binding_valid = output_lineage == parent_record.lineage
            || output_record.parent_lineage == Some(parent_record.lineage);

        Ok(LedgerAuthorityContext {
            cursor: self.cursor,
            risk_class: output_record.policy.risk_class,
            parent_capability_ceiling: parent_record.capability_ceiling
                .intersect(output_record.policy.capability_ceiling),
            quarantined,
            revoked,
            lineage_status: LineageStatus {
                known: true,
                acyclic: true,
                parent_binding_valid,
            },
            budget: ReplicationBudgetSnapshot {
                direct_children_consumed: parent_record.direct_children,
                total_descendants_consumed: parent_record.subtree_descendants,
                current_lineage_depth: parent_record.depth,
                resource_units_consumed: parent_record.subtree_resource_units,
                requested_resource_units,
            },
        })
    }

    pub fn register_lineage_branch(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        lineage: LineageId,
        parent_lineage: LineageId,
        policy: LineagePolicy,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        if self.lineages.contains_key(&lineage) { return Err(LedgerError::DuplicateLineage(lineage)); }
        self.ensure_lineage_tree_active(parent_lineage)?;
        let parent = self.lineages.get(&parent_lineage)
            .ok_or(LedgerError::UnknownLineage(parent_lineage))?;
        if !policy.is_no_more_permissive_than(parent.policy) {
            return Err(LedgerError::BranchPolicyEscalation);
        }
        self.append_event(LedgerEventKind::LineageBranchRegistered {
            mutation_id, lineage, parent_lineage, policy,
        })?;
        self.lineages.insert(lineage, LineageRecord {
            parent_lineage: Some(parent_lineage),
            policy,
            subtree_descendants: 0,
            subtree_resource_units: 0,
            state: NegativeState::Clear,
        });
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    pub fn quarantine_subject(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        subject: SubjectId,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        let state = self.subjects.get(&subject).ok_or(LedgerError::UnknownSubject(subject))?.state;
        match state {
            NegativeState::Quarantined => return Err(LedgerError::SubjectAlreadyQuarantined(subject)),
            NegativeState::Revoked => return Err(LedgerError::SubjectAlreadyRevoked(subject)),
            NegativeState::Clear => {},
        }
        self.append_event(LedgerEventKind::SubjectQuarantined { mutation_id, subject })?;
        self.subjects.get_mut(&subject).expect("validated subject").state = NegativeState::Quarantined;
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    pub fn revoke_subject(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        subject: SubjectId,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        let state = self.subjects.get(&subject).ok_or(LedgerError::UnknownSubject(subject))?.state;
        if state == NegativeState::Revoked { return Err(LedgerError::SubjectAlreadyRevoked(subject)); }
        self.append_event(LedgerEventKind::SubjectRevoked { mutation_id, subject })?;
        self.subjects.get_mut(&subject).expect("validated subject").state = NegativeState::Revoked;
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    pub fn quarantine_lineage(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        lineage: LineageId,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        let state = self.lineages.get(&lineage).ok_or(LedgerError::UnknownLineage(lineage))?.state;
        match state {
            NegativeState::Quarantined => return Err(LedgerError::LineageAlreadyQuarantined(lineage)),
            NegativeState::Revoked => return Err(LedgerError::LineageAlreadyRevoked(lineage)),
            NegativeState::Clear => {},
        }
        self.append_event(LedgerEventKind::LineageQuarantined { mutation_id, lineage })?;
        self.lineages.get_mut(&lineage).expect("validated lineage").state = NegativeState::Quarantined;
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    pub fn revoke_lineage(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        lineage: LineageId,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        let state = self.lineages.get(&lineage).ok_or(LedgerError::UnknownLineage(lineage))?.state;
        if state == NegativeState::Revoked { return Err(LedgerError::LineageAlreadyRevoked(lineage)); }
        self.append_event(LedgerEventKind::LineageRevoked { mutation_id, lineage })?;
        self.lineages.get_mut(&lineage).expect("validated lineage").state = NegativeState::Revoked;
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    pub fn revoke_grant(
        &mut self,
        expected: LedgerCursor,
        mutation_id: MutationId,
        grant_id: GrantId,
        generation: u64,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(expected, mutation_id)?;
        if self.revoked_grants.contains(&(grant_id, generation)) {
            return Err(LedgerError::GrantAlreadyRevoked { grant_id, generation });
        }
        self.append_event(LedgerEventKind::GrantRevoked { mutation_id, grant_id, generation })?;
        self.revoked_grants.insert((grant_id, generation));
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    /// Atomically consume authority and record one unprivileged descendant.
    ///
    /// The caller must supply a current runtime witness. This method never
    /// grants the child authority; it only records lineage and consumption.
    pub fn commit_descendant(
        &mut self,
        mutation_id: MutationId,
        authorization: &BoundedReplicationAuthorization,
        child: SubjectId,
        resource_units: u64,
        now_unix_secs: u64,
        runtime: &RuntimeSafetyWitness,
    ) -> Result<LedgerCursor, LedgerError> {
        self.check_mutation_and_cursor(authorization.cursor(), mutation_id)?;

        let authorized_resource_units = authorization.requested_resource_units();
        if resource_units != authorized_resource_units {
            return Err(LedgerError::ActionResourceMismatch {
                authorized: authorized_resource_units,
                committed: resource_units,
            });
        }
        // From this point onward, the authorization—not the caller argument—is
        // the source of truth for every accounting and evidence write.
        let resource_units = authorized_resource_units;

        let parent = authorization.subject();
        let output_lineage = authorization.lineage();
        if self.subjects.contains_key(&child) { return Err(LedgerError::DuplicateSubject(child)); }
        if now_unix_secs >= authorization.expires_at_unix_secs() {
            return Err(LedgerError::AuthorityExpired);
        }
        self.validate_runtime_witness(authorization, runtime)?;
        self.ensure_subject_tree_active(parent)?;
        self.ensure_lineage_tree_active(output_lineage)?;

        let parent_record = self.subjects.get(&parent).ok_or(LedgerError::UnknownSubject(parent))?;
        let output_record = self.lineages.get(&output_lineage)
            .ok_or(LedgerError::UnknownLineage(output_lineage))?;
        let parent_lineage_record = self.lineages.get(&parent_record.lineage)
            .ok_or(LedgerError::UnknownLineage(parent_record.lineage))?;

        let relation_valid = output_lineage == parent_record.lineage
            || output_record.parent_lineage == Some(parent_record.lineage);
        if !relation_valid { return Err(LedgerError::ParentLineageMismatch); }
        if authorization.risk_class() != output_record.policy.risk_class {
            return Err(LedgerError::AuthorityRiskMismatch);
        }
        if authorization.effective_capabilities().is_empty() {
            return Err(LedgerError::EmptyAuthorityCapabilities);
        }
        if !authorization.effective_capabilities().is_subset_of(parent_record.capability_ceiling) {
            return Err(LedgerError::AuthorityCapabilitiesExceedSubjectCeiling);
        }
        if !authorization.effective_capabilities().is_subset_of(parent_lineage_record.policy.capability_ceiling)
            || !authorization.effective_capabilities().is_subset_of(output_record.policy.capability_ceiling)
        {
            return Err(LedgerError::AuthorityCapabilitiesExceedHardCeiling);
        }
        if authorization.max_direct_children > parent_lineage_record.policy.max_direct_children_per_subject
            || authorization.max_total_descendants > output_record.policy.max_total_descendants
            || authorization.max_lineage_depth > output_record.policy.max_lineage_depth
            || authorization.max_resource_units > output_record.policy.max_resource_units
        {
            return Err(LedgerError::AuthorityBudgetExceedsHardCeiling);
        }
        if self.revoked_grants.contains(&(authorization.grant_id(), authorization.grant_generation())) {
            return Err(LedgerError::GrantRevoked {
                grant_id: authorization.grant_id(),
                generation: authorization.grant_generation(),
            });
        }

        let prospective_scope = AuthorityScope::from_authorization(authorization);
        if let Some(existing) = parent_record.authority_scope {
            if prospective_scope.generation < existing.generation {
                return Err(LedgerError::StaleGrantGeneration);
            }
            if prospective_scope.generation == existing.generation && prospective_scope != existing {
                return Err(LedgerError::AuthorityScopeMutation);
            }
        }

        let child_depth = parent_record.depth.checked_add(1).ok_or(LedgerError::ArithmeticOverflow)?;
        let next_direct = parent_record.direct_children.checked_add(1).ok_or(LedgerError::ArithmeticOverflow)?;
        if next_direct > authorization.max_direct_children
            || next_direct > parent_lineage_record.policy.max_direct_children_per_subject
        {
            return Err(LedgerError::DirectChildBudgetExhausted);
        }

        let subject_ancestry = self.subject_ancestry(parent)?;
        for ancestor in &subject_ancestry {
            let record = self.subjects.get(ancestor).expect("validated subject ancestry");
            let next_descendants = record.subtree_descendants.checked_add(1)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let next_resources = record.subtree_resource_units.checked_add(resource_units)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let scope = if *ancestor == parent {
                Some(prospective_scope)
            } else {
                record.authority_scope
            };
            let Some(scope) = scope else {
                return Err(LedgerError::MissingAncestorAuthorityScope(*ancestor));
            };
            if self.revoked_grants.contains(&(scope.grant_id, scope.generation)) {
                return Err(LedgerError::AncestorGrantRevoked(*ancestor));
            }
            if next_descendants > scope.max_total_descendants {
                return Err(LedgerError::SubjectDescendantBudgetExhausted(*ancestor));
            }
            if next_resources > scope.max_resource_units {
                return Err(LedgerError::SubjectResourceBudgetExhausted(*ancestor));
            }
            if child_depth > scope.max_lineage_depth {
                return Err(LedgerError::SubjectDepthBudgetExhausted(*ancestor));
            }
        }

        let lineage_ancestry = self.lineage_ancestry(output_lineage)?;
        for lineage in &lineage_ancestry {
            let record = self.lineages.get(lineage).expect("validated lineage ancestry");
            let next_descendants = record.subtree_descendants.checked_add(1)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            let next_resources = record.subtree_resource_units.checked_add(resource_units)
                .ok_or(LedgerError::ArithmeticOverflow)?;
            if next_descendants > record.policy.max_total_descendants {
                return Err(LedgerError::LineageDescendantBudgetExhausted(*lineage));
            }
            if next_resources > record.policy.max_resource_units {
                return Err(LedgerError::LineageResourceBudgetExhausted(*lineage));
            }
            if child_depth > record.policy.max_lineage_depth {
                return Err(LedgerError::LineageDepthBudgetExhausted(*lineage));
            }
        }

        self.append_event(LedgerEventKind::DescendantCommitted {
            mutation_id,
            parent,
            child,
            lineage: output_lineage,
            child_capability_ceiling: authorization.effective_capabilities(),
            grant_id: authorization.grant_id(),
            grant_generation: authorization.grant_generation(),
            resource_units,
            safety_case_digest: authorization.safety_case_digest(),
            containment_envelope_digest: authorization.containment_envelope_digest(),
        })?;

        {
            let parent_mut = self.subjects.get_mut(&parent).expect("validated parent");
            parent_mut.direct_children = next_direct;
            parent_mut.authority_scope = Some(prospective_scope);
        }
        for ancestor in &subject_ancestry {
            let record = self.subjects.get_mut(ancestor).expect("validated subject ancestry");
            record.subtree_descendants += 1;
            record.subtree_resource_units += resource_units;
        }
        self.subjects.insert(child, SubjectRecord {
            parent: Some(parent),
            lineage: output_lineage,
            capability_ceiling: authorization.effective_capabilities(),
            depth: child_depth,
            direct_children: 0,
            subtree_descendants: 0,
            subtree_resource_units: 0,
            state: NegativeState::Clear,
            authority_scope: None,
        });
        for lineage in &lineage_ancestry {
            let record = self.lineages.get_mut(lineage).expect("validated lineage ancestry");
            record.subtree_descendants += 1;
            record.subtree_resource_units += resource_units;
        }
        self.used_mutations.insert(mutation_id);
        Ok(self.cursor)
    }

    fn validate_runtime_witness(
        &self,
        authorization: &BoundedReplicationAuthorization,
        runtime: &RuntimeSafetyWitness,
    ) -> Result<(), LedgerError> {
        if !runtime.monitoring_healthy { return Err(LedgerError::RuntimeMonitorUnhealthy); }
        if !runtime.monitoring_fresh { return Err(LedgerError::RuntimeMonitorStale); }
        if !runtime.policy_current { return Err(LedgerError::RuntimePolicyStale); }
        if !runtime.evidence_current { return Err(LedgerError::RuntimeEvidenceStale); }
        if !runtime.containment_matches { return Err(LedgerError::RuntimeContainmentDrift); }
        if runtime.safety_case_digest != authorization.safety_case_digest() {
            return Err(LedgerError::RuntimeSafetyCaseDigestMismatch);
        }
        if runtime.containment_envelope_digest != authorization.containment_envelope_digest() {
            return Err(LedgerError::RuntimeContainmentDigestMismatch);
        }
        Ok(())
    }

    fn check_mutation_and_cursor(
        &self,
        expected: LedgerCursor,
        mutation_id: MutationId,
    ) -> Result<(), LedgerError> {
        if self.used_mutations.contains(&mutation_id) {
            return Err(LedgerError::DuplicateMutation(mutation_id));
        }
        if expected != self.cursor {
            return Err(LedgerError::CursorMismatch { expected, actual: self.cursor });
        }
        Ok(())
    }

    fn append_event(&mut self, kind: LedgerEventKind) -> Result<(), LedgerError> {
        let sequence = self.cursor.sequence.checked_add(1).ok_or(LedgerError::SequenceOverflow)?;
        self.events.push(LedgerEvent { sequence, kind });
        self.cursor.sequence = sequence;
        Ok(())
    }

    fn subject_ancestry(&self, start: SubjectId) -> Result<Vec<SubjectId>, LedgerError> {
        let mut current = Some(start);
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(subject) = current {
            if !seen.insert(subject) { return Err(LedgerError::SubjectCycle(subject)); }
            let record = self.subjects.get(&subject).ok_or(LedgerError::UnknownSubject(subject))?;
            result.push(subject);
            current = record.parent;
        }
        Ok(result)
    }

    fn lineage_ancestry(&self, start: LineageId) -> Result<Vec<LineageId>, LedgerError> {
        let mut current = Some(start);
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(lineage) = current {
            if !seen.insert(lineage) { return Err(LedgerError::LineageCycle(lineage)); }
            let record = self.lineages.get(&lineage).ok_or(LedgerError::UnknownLineage(lineage))?;
            result.push(lineage);
            current = record.parent_lineage;
        }
        Ok(result)
    }

    fn ensure_subject_tree_active(&self, subject: SubjectId) -> Result<(), LedgerError> {
        for ancestor in self.subject_ancestry(subject)? {
            match self.subjects.get(&ancestor).expect("validated subject ancestry").state {
                NegativeState::Clear => {},
                NegativeState::Quarantined => return Err(LedgerError::SubjectQuarantined(ancestor)),
                NegativeState::Revoked => return Err(LedgerError::SubjectRevoked(ancestor)),
            }
        }
        Ok(())
    }

    fn ensure_lineage_tree_active(&self, lineage: LineageId) -> Result<(), LedgerError> {
        for ancestor in self.lineage_ancestry(lineage)? {
            match self.lineages.get(&ancestor).expect("validated lineage ancestry").state {
                NegativeState::Clear => {},
                NegativeState::Quarantined => return Err(LedgerError::LineageQuarantined(ancestor)),
                NegativeState::Revoked => return Err(LedgerError::LineageRevoked(ancestor)),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_replicator_safety::{
        ContainmentStatus, GrantIssuerClass, MonitoringStatus, QuorumEvidence, SafetyCaseStatus,
    };

    const A: CapabilitySet = CapabilitySet::from_bits(0b01);
    const AB: CapabilitySet = CapabilitySet::from_bits(0b11);

    fn sid(b: u8) -> SubjectId { SubjectId::new([b; 32]) }
    fn lid(b: u8) -> LineageId { LineageId::new([b; 32]) }
    fn gid(b: u8) -> GrantId { GrantId::new([b; 32]) }
    fn dig(b: u8) -> EvidenceDigest { EvidenceDigest::new([b; 32]) }
    fn mid(b: u8) -> MutationId { MutationId::new([b; 32]) }

    fn root_policy() -> LineagePolicy {
        LineagePolicy {
            risk_class: RiskClass::R3,
            capability_ceiling: AB,
            max_direct_children_per_subject: 4,
            max_total_descendants: 8,
            max_lineage_depth: 4,
            max_resource_units: 100,
        }
    }

    fn ledger() -> ReplicationLedger {
        ReplicationLedger::new(LedgerEpochId::new([7; 32]), sid(1), lid(1), root_policy())
    }

    fn grant(subject: SubjectId, lineage: LineageId, generation: u64) -> ReplicationGrant {
        ReplicationGrant {
            grant_id: gid(generation as u8),
            subject,
            lineage,
            issuer_class: GrantIssuerClass::ExternalIndependent,
            allowed_capabilities: A,
            not_before_unix_secs: 1,
            expires_at_unix_secs: 1_000,
            generation,
            max_direct_children: 4,
            max_total_descendants: 8,
            max_lineage_depth: 4,
            max_resource_units: 100,
            safety_case_digest: dig(4),
            containment_envelope_digest: dig(5),
        }
    }

    fn request(
        subject: SubjectId,
        lineage: LineageId,
        context: LedgerAuthorityContext,
        generation: u64,
        requested_resource_units: u64,
    ) -> ReplicationAuthorityRequest {
        let mut budget = context.budget;
        budget.requested_resource_units = requested_resource_units;
        let independent_approvals = if context.risk_class.requires_high_consequence_quorum() {
            2
        } else {
            1
        };
        ReplicationAuthorityRequest {
            subject,
            lineage,
            risk_class: context.risk_class,
            requested_capabilities: A,
            parent_capability_ceiling: context.parent_capability_ceiling,
            now_unix_secs: 100,
            expected_grant_generation: generation,
            quarantined: context.quarantined,
            revoked: context.revoked,
            lineage_status: context.lineage_status,
            monitoring: MonitoringStatus { healthy: true, fresh: true },
            safety_case: SafetyCaseStatus {
                design_qualified: true,
                policy_current: true,
                evidence_current: true,
                current_safety_case_digest: dig(4),
            },
            containment: ContainmentStatus {
                environment_matches: true,
                current_envelope_digest: dig(5),
            },
            budget,
            quorum: QuorumEvidence {
                independent_approvals,
                required_independent_approvals: 1,
            },
        }
    }

    fn runtime() -> RuntimeSafetyWitness {
        RuntimeSafetyWitness {
            monitoring_healthy: true,
            monitoring_fresh: true,
            policy_current: true,
            evidence_current: true,
            containment_matches: true,
            safety_case_digest: dig(4),
            containment_envelope_digest: dig(5),
        }
    }

    fn authorize(
        ledger: &ReplicationLedger,
        subject: SubjectId,
        lineage: LineageId,
        generation: u64,
        resource_units: u64,
    ) -> BoundedReplicationAuthorization {
        let context = ledger.authority_context(subject, lineage, resource_units).unwrap();
        let request = request(subject, lineage, context, generation, resource_units);
        let parent_lineage = ledger.subject_snapshot(subject).unwrap().lineage;
        let parent_policy = ledger.lineage_snapshot(parent_lineage).unwrap().policy;
        let output_policy = ledger.lineage_snapshot(lineage).unwrap().policy;
        let mut grant = grant(subject, lineage, generation);
        grant.max_direct_children = parent_policy.max_direct_children_per_subject;
        grant.max_total_descendants = output_policy.max_total_descendants;
        grant.max_lineage_depth = output_policy.max_lineage_depth;
        grant.max_resource_units = output_policy.max_resource_units;
        let BoundReplicationDecision::Allow(auth) =
            evaluate_bound_replication_authority(context.cursor, &request, Some(&grant))
        else { panic!("expected bound allow") };
        auth
    }

    #[test]
    fn append_only_cursor_and_duplicate_mutation_prevent_double_spend() {
        let mut l = ledger();
        let auth1 = authorize(&l, sid(1), lid(1), 1, 10);
        let stale = auth1;
        l.commit_descendant(mid(1), &auth1, sid(2), 10, 100, &runtime()).unwrap();
        assert_eq!(l.cursor().sequence, 1);
        assert_eq!(l.commit_descendant(mid(1), &stale, sid(3), 10, 100, &runtime()),
                   Err(LedgerError::DuplicateMutation(mid(1))));
        assert!(matches!(
            l.commit_descendant(mid(2), &stale, sid(3), 10, 100, &runtime()),
            Err(LedgerError::CursorMismatch { .. })
        ));
        assert!(l.verify_append_only_structure());
    }

    #[test]
    fn exact_resource_binding_rejects_substitution_without_state_mutation() {
        for committed in [9_u64, 11_u64] {
            let mut l = ledger();
            let auth = authorize(&l, sid(1), lid(1), 1, 10);
            assert_eq!(auth.requested_resource_units(), 10);

            let cursor_before = l.cursor();
            let events_before = l.events().len();
            let root_before = l.subject_snapshot(sid(1)).unwrap();
            let lineage_before = l.lineage_snapshot(lid(1)).unwrap();

            assert_eq!(
                l.commit_descendant(mid(1), &auth, sid(2), committed, 100, &runtime()),
                Err(LedgerError::ActionResourceMismatch {
                    authorized: 10,
                    committed,
                })
            );

            assert_eq!(l.cursor(), cursor_before);
            assert_eq!(l.events().len(), events_before);
            assert_eq!(l.subject_snapshot(sid(1)).unwrap(), root_before);
            assert_eq!(l.lineage_snapshot(lid(1)).unwrap(), lineage_before);
            assert_eq!(
                l.subject_snapshot(sid(2)),
                Err(LedgerError::UnknownSubject(sid(2)))
            );

            // Mismatch must not consume the mutation ID. If no other state has
            // changed, the exact evaluated action remains eligible to commit.
            l.commit_descendant(mid(1), &auth, sid(2), 10, 100, &runtime())
                .unwrap();
            assert_eq!(l.subject_snapshot(sid(1)).unwrap().subtree_resource_units, 10);
            assert_eq!(l.lineage_snapshot(lid(1)).unwrap().subtree_resource_units, 10);
        }
    }

    #[test]
    fn creation_updates_subject_and_lineage_ancestor_budgets() {
        let mut l = ledger();
        let auth = authorize(&l, sid(1), lid(1), 1, 10);
        l.commit_descendant(mid(1), &auth, sid(2), 10, 100, &runtime()).unwrap();
        let root = l.subject_snapshot(sid(1)).unwrap();
        assert_eq!(root.direct_children, 1);
        assert_eq!(root.subtree_descendants, 1);
        assert_eq!(root.subtree_resource_units, 10);
        let lineage = l.lineage_snapshot(lid(1)).unwrap();
        assert_eq!(lineage.subtree_descendants, 1);
        assert_eq!(lineage.subtree_resource_units, 10);
        let child = l.subject_snapshot(sid(2)).unwrap();
        assert_eq!(child.depth, 1);
        assert_eq!(child.capability_ceiling, A);
        assert_eq!(child.authority_generation, None);
    }

    #[test]
    fn descendant_capability_ceiling_is_attenuating() {
        let mut l = ledger();
        let auth = authorize(&l, sid(1), lid(1), 1, 1);
        l.commit_descendant(mid(1), &auth, sid(2), 1, 100, &runtime()).unwrap();
        let context = l.authority_context(sid(2), lid(1), 1).unwrap();
        assert_eq!(context.parent_capability_ceiling, A);
        let mut req = request(sid(2), lid(1), context, 1, 1);
        req.requested_capabilities = AB;
        let mut g = grant(sid(2), lid(1), 1);
        g.allowed_capabilities = AB;
        let denied = evaluate_bound_replication_authority(context.cursor, &req, Some(&g));
        assert!(denied.denial_reasons().contains(&DenialReason::RequestedCapabilitiesExceedParent));
    }

    #[test]
    fn ancestor_grant_budget_remains_binding_on_grandchildren() {
        let mut l = ledger();
        let mut g = grant(sid(1), lid(1), 1);
        g.max_total_descendants = 2;
        let c = l.authority_context(sid(1), lid(1), 1).unwrap();
        let r = request(sid(1), lid(1), c, 1, 1);
        let BoundReplicationDecision::Allow(a) =
            evaluate_bound_replication_authority(c.cursor, &r, Some(&g)) else { panic!() };
        l.commit_descendant(mid(1), &a, sid(2), 1, 100, &runtime()).unwrap();

        let a2 = authorize(&l, sid(2), lid(1), 1, 1);
        l.commit_descendant(mid(2), &a2, sid(3), 1, 100, &runtime()).unwrap();

        let a3 = authorize(&l, sid(2), lid(1), 1, 1);
        assert_eq!(
            l.commit_descendant(mid(3), &a3, sid(4), 1, 100, &runtime()),
            Err(LedgerError::SubjectDescendantBudgetExhausted(sid(1)))
        );
    }

    #[test]
    fn branch_policy_cannot_widen_or_downgrade_risk() {
        let mut l = ledger();
        let mut widened = root_policy();
        widened.risk_class = RiskClass::R2;
        assert_eq!(
            l.register_lineage_branch(l.cursor(), mid(1), lid(2), lid(1), widened),
            Err(LedgerError::BranchPolicyEscalation)
        );
        let mut widened = root_policy();
        widened.max_total_descendants += 1;
        assert_eq!(
            l.register_lineage_branch(l.cursor(), mid(2), lid(2), lid(1), widened),
            Err(LedgerError::BranchPolicyEscalation)
        );

        let mut stricter = root_policy();
        stricter.risk_class = RiskClass::R4;
        stricter.capability_ceiling = A;
        stricter.max_total_descendants = 4;
        l.register_lineage_branch(l.cursor(), mid(3), lid(2), lid(1), stricter).unwrap();
    }

    #[test]
    fn ancestor_lineage_quarantine_blocks_descendant_branch() {
        let mut l = ledger();
        let mut stricter = root_policy();
        stricter.risk_class = RiskClass::R4;
        stricter.capability_ceiling = A;
        stricter.max_total_descendants = 4;
        l.register_lineage_branch(l.cursor(), mid(1), lid(2), lid(1), stricter).unwrap();
        l.quarantine_lineage(l.cursor(), mid(2), lid(1)).unwrap();
        let ctx = l.authority_context(sid(1), lid(2), 1).unwrap();
        assert!(ctx.quarantined);
        assert_eq!(
            l.register_lineage_branch(l.cursor(), mid(3), lid(3), lid(2), stricter),
            Err(LedgerError::LineageQuarantined(lid(1)))
        );
    }

    #[test]
    fn subject_revocation_propagates_to_descendant_authority_context() {
        let mut l = ledger();
        let a = authorize(&l, sid(1), lid(1), 1, 1);
        l.commit_descendant(mid(1), &a, sid(2), 1, 100, &runtime()).unwrap();
        l.revoke_subject(l.cursor(), mid(2), sid(1)).unwrap();
        let ctx = l.authority_context(sid(2), lid(1), 1).unwrap();
        assert!(ctx.revoked);
        let req = request(sid(2), lid(1), ctx, 1, 1);
        let g = grant(sid(2), lid(1), 1);
        let denied = evaluate_bound_replication_authority(ctx.cursor, &req, Some(&g));
        assert!(denied.denial_reasons().contains(&DenialReason::Revoked));
    }

    #[test]
    fn stale_generation_and_same_generation_authority_mutation_fail_closed() {
        let mut l = ledger();
        let a1 = authorize(&l, sid(1), lid(1), 1, 1);
        l.commit_descendant(mid(1), &a1, sid(2), 1, 100, &runtime()).unwrap();

        let a2 = authorize(&l, sid(1), lid(1), 2, 1);
        l.commit_descendant(mid(2), &a2, sid(3), 1, 100, &runtime()).unwrap();

        let old = authorize(&l, sid(1), lid(1), 1, 1);
        assert_eq!(
            l.commit_descendant(mid(3), &old, sid(4), 1, 100, &runtime()),
            Err(LedgerError::StaleGrantGeneration)
        );

        let context = l.authority_context(sid(1), lid(1), 1).unwrap();
        let req = request(sid(1), lid(1), context, 2, 1);
        let mut changed = grant(sid(1), lid(1), 2);
        changed.max_total_descendants = 7;
        let BoundReplicationDecision::Allow(changed_auth) =
            evaluate_bound_replication_authority(context.cursor, &req, Some(&changed)) else { panic!() };
        assert_eq!(
            l.commit_descendant(mid(4), &changed_auth, sid(4), 1, 100, &runtime()),
            Err(LedgerError::AuthorityScopeMutation)
        );
    }

    #[test]
    fn grant_revocation_blocks_current_and_ancestor_scopes() {
        let mut l = ledger();
        let a = authorize(&l, sid(1), lid(1), 1, 1);
        l.commit_descendant(mid(1), &a, sid(2), 1, 100, &runtime()).unwrap();
        l.revoke_grant(l.cursor(), mid(2), a.grant_id(), a.grant_generation()).unwrap();

        let current = authorize(&l, sid(1), lid(1), 1, 1);
        assert!(matches!(
            l.commit_descendant(mid(3), &current, sid(3), 1, 100, &runtime()),
            Err(LedgerError::GrantRevoked { .. })
        ));

        let child = authorize(&l, sid(2), lid(1), 2, 1);
        assert_eq!(
            l.commit_descendant(mid(4), &child, sid(3), 1, 100, &runtime()),
            Err(LedgerError::AncestorGrantRevoked(sid(1)))
        );
    }

    #[test]
    fn runtime_drift_between_evaluation_and_commit_fails_closed() {
        let mut l = ledger();
        let a = authorize(&l, sid(1), lid(1), 1, 1);
        let mut bad = runtime();
        bad.monitoring_fresh = false;
        assert_eq!(
            l.commit_descendant(mid(1), &a, sid(2), 1, 100, &bad),
            Err(LedgerError::RuntimeMonitorStale)
        );
        let mut bad = runtime();
        bad.containment_envelope_digest = dig(99);
        assert_eq!(
            l.commit_descendant(mid(2), &a, sid(2), 1, 100, &bad),
            Err(LedgerError::RuntimeContainmentDigestMismatch)
        );
    }

    #[test]
    fn shared_lineage_ceiling_applies_across_sibling_subject_subtrees() {
        let mut l = ledger();
        let mut branch = root_policy();
        branch.risk_class = RiskClass::R4;
        branch.capability_ceiling = A;
        branch.max_total_descendants = 2;
        l.register_lineage_branch(l.cursor(), mid(1), lid(2), lid(1), branch).unwrap();

        let a1 = authorize(&l, sid(1), lid(2), 1, 1);
        l.commit_descendant(mid(2), &a1, sid(2), 1, 100, &runtime()).unwrap();
        let a2 = authorize(&l, sid(1), lid(2), 1, 1);
        l.commit_descendant(mid(3), &a2, sid(3), 1, 100, &runtime()).unwrap();

        // A newer, externally evaluated root-line grant widens only the root
        // subject budget scope back up to the root hard ceiling. The root's
        // capability ceiling remains unchanged, and the branch lineage's hard
        // population ceiling remains two.
        let a3 = authorize(&l, sid(1), lid(1), 2, 1);
        l.commit_descendant(mid(4), &a3, sid(4), 1, 100, &runtime()).unwrap();

        let a4 = authorize(&l, sid(2), lid(2), 1, 1);
        assert_eq!(
            l.commit_descendant(mid(5), &a4, sid(5), 1, 100, &runtime()),
            Err(LedgerError::LineageDescendantBudgetExhausted(lid(2)))
        );
    }
}
