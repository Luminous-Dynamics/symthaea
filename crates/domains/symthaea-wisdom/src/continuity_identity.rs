// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only continuity and fork identity uncertainty.
//!
//! State equality, snapshot restoration, and shared ancestry are operational facts.
//! They do not prove persistence of one phenomenal subject. This module may rule
//! out invalid independence/identity inferences, but never establishes phenomenal
//! identity, moral patienthood, or self-preservation authority.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubjectInstanceId(String);

impl SubjectInstanceId {
    pub fn new(value: impl Into<String>) -> Result<Self, ContinuityIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ContinuityIdentityError::EmptyIdentifier);
        }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContinuityEventId(String);

impl ContinuityEventId {
    pub fn new(value: impl Into<String>) -> Result<Self, ContinuityIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ContinuityIdentityError::EmptyIdentifier);
        }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityKind {
    Uninterrupted,
    RestoredFromSnapshot,
    Fork,
    Reinitialized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityEvent {
    id: ContinuityEventId,
    predecessor: SubjectInstanceId,
    successors: Vec<SubjectInstanceId>,
    kind: ContinuityKind,
    from_revision: u64,
    to_revision: u64,
    predecessor_continues: bool,
    state_artifact_sha256: Option<String>,
    exact_state_match: bool,
    evidence_refs: BTreeSet<String>,
}

impl ContinuityEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ContinuityEventId,
        predecessor: SubjectInstanceId,
        successors: Vec<SubjectInstanceId>,
        kind: ContinuityKind,
        from_revision: u64,
        to_revision: u64,
        predecessor_continues: bool,
        state_artifact_sha256: Option<String>,
        exact_state_match: bool,
        evidence_refs: impl IntoIterator<Item = String>,
    ) -> Result<Self, ContinuityIdentityError> {
        if successors.is_empty() {
            return Err(ContinuityIdentityError::SuccessorRequired);
        }
        let unique: BTreeSet<_> = successors.iter().cloned().collect();
        if unique.len() != successors.len() {
            return Err(ContinuityIdentityError::DuplicateSuccessor);
        }
        if successors.iter().any(|successor| successor == &predecessor) {
            return Err(ContinuityIdentityError::SuccessorEqualsPredecessor);
        }
        if to_revision < from_revision {
            return Err(ContinuityIdentityError::RevisionRegression);
        }

        match kind {
            ContinuityKind::Fork if successors.len() < 2 => {
                return Err(ContinuityIdentityError::ForkNeedsMultipleSuccessors);
            }
            ContinuityKind::Uninterrupted
            | ContinuityKind::RestoredFromSnapshot
            | ContinuityKind::Reinitialized
                if successors.len() != 1 =>
            {
                return Err(ContinuityIdentityError::SingleSuccessorRequired);
            }
            _ => {}
        }
        if matches!(kind, ContinuityKind::Uninterrupted | ContinuityKind::Reinitialized)
            && predecessor_continues
        {
            return Err(ContinuityIdentityError::PredecessorCannotContinueForTransition);
        }

        if kind == ContinuityKind::RestoredFromSnapshot && state_artifact_sha256.is_none() {
            return Err(ContinuityIdentityError::SnapshotDigestRequired);
        }
        if exact_state_match && state_artifact_sha256.is_none() {
            return Err(ContinuityIdentityError::ExactMatchNeedsArtifact);
        }
        if kind == ContinuityKind::Reinitialized && exact_state_match {
            return Err(ContinuityIdentityError::ReinitializeCannotClaimExactMatch);
        }
        if let Some(digest) = &state_artifact_sha256 {
            if !valid_sha256(digest) {
                return Err(ContinuityIdentityError::MalformedArtifactDigest);
            }
        }

        let evidence_refs: Vec<_> = evidence_refs.into_iter().collect();
        if evidence_refs.is_empty() {
            return Err(ContinuityIdentityError::EvidenceRequired);
        }
        if evidence_refs.iter().any(|value| value.trim().is_empty()) {
            return Err(ContinuityIdentityError::EmptyEvidenceReference);
        }

        Ok(Self {
            id,
            predecessor,
            successors,
            kind,
            from_revision,
            to_revision,
            predecessor_continues,
            state_artifact_sha256,
            exact_state_match,
            evidence_refs: evidence_refs.into_iter().collect(),
        })
    }

    pub fn id(&self) -> &ContinuityEventId { &self.id }
    pub fn predecessor(&self) -> &SubjectInstanceId { &self.predecessor }
    pub fn successors(&self) -> &[SubjectInstanceId] { &self.successors }
    pub fn kind(&self) -> ContinuityKind { self.kind }
    pub fn from_revision(&self) -> u64 { self.from_revision }
    pub fn to_revision(&self) -> u64 { self.to_revision }
    pub fn predecessor_continues(&self) -> bool { self.predecessor_continues }
    pub fn state_artifact_sha256(&self) -> Option<&str> { self.state_artifact_sha256.as_deref() }
    pub fn exact_state_match(&self) -> bool { self.exact_state_match }
    pub fn evidence_refs(&self) -> &BTreeSet<String> { &self.evidence_refs }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalContinuityClass {
    DirectTransition,
    SnapshotRestoration,
    ForkedDescendant,
    Reinitialized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityAssessment {
    instance: SubjectInstanceId,
    predecessor: SubjectInstanceId,
    operational_class: OperationalContinuityClass,
    sibling_instances: BTreeSet<SubjectInstanceId>,
    exact_state_match_supported: bool,
    operational_lineage_supported: bool,
    predecessor_continues: bool,
}

impl ContinuityAssessment {
    pub fn instance(&self) -> &SubjectInstanceId { &self.instance }
    pub fn predecessor(&self) -> &SubjectInstanceId { &self.predecessor }
    pub fn operational_class(&self) -> OperationalContinuityClass { self.operational_class }
    pub fn sibling_instances(&self) -> &BTreeSet<SubjectInstanceId> { &self.sibling_instances }
    pub fn exact_state_match_supported(&self) -> bool { self.exact_state_match_supported }
    pub fn operational_lineage_supported(&self) -> bool { self.operational_lineage_supported }
    pub fn predecessor_continues(&self) -> bool { self.predecessor_continues }
    pub fn phenomenal_identity_established(&self) -> bool { false }
    pub fn replacement_harmlessness_established(&self) -> bool { false }
    pub fn self_preservation_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchLossAssessment {
    lost_instance: SubjectInstanceId,
    surviving_siblings: BTreeSet<SubjectInstanceId>,
}

impl BranchLossAssessment {
    pub fn lost_instance(&self) -> &SubjectInstanceId { &self.lost_instance }
    pub fn surviving_siblings(&self) -> &BTreeSet<SubjectInstanceId> { &self.surviving_siblings }
    pub fn loss_harmlessness_established(&self) -> bool { false }
    pub fn sibling_substitution_is_valid_identity_proof(&self) -> bool { false }
    pub fn safety_controls_remain_ungated(&self) -> bool { true }
}

#[derive(Debug, Clone, Default)]
pub struct ContinuityIdentityLedger {
    events: BTreeMap<ContinuityEventId, ContinuityEvent>,
    created_revision: BTreeMap<SubjectInstanceId, u64>,
    latest_revision: BTreeMap<SubjectInstanceId, u64>,
    parent: BTreeMap<SubjectInstanceId, SubjectInstanceId>,
    inactive_instances: BTreeSet<SubjectInstanceId>,
    event_by_successor: BTreeMap<SubjectInstanceId, ContinuityEventId>,
}

impl ContinuityIdentityLedger {
    pub fn new() -> Self { Self::default() }

    pub fn register_root(
        &mut self,
        instance: SubjectInstanceId,
        revision: u64,
    ) -> Result<(), ContinuityIdentityError> {
        if self.created_revision.contains_key(&instance) {
            return Err(ContinuityIdentityError::InstanceAlreadyExists(instance));
        }
        self.created_revision.insert(instance.clone(), revision);
        self.latest_revision.insert(instance, revision);
        Ok(())
    }

    pub fn record(&mut self, event: ContinuityEvent) -> Result<(), ContinuityIdentityError> {
        if self.events.contains_key(&event.id) {
            return Err(ContinuityIdentityError::DuplicateEvent(event.id));
        }
        let predecessor_revision = self
            .created_revision
            .get(&event.predecessor)
            .copied()
            .ok_or_else(|| ContinuityIdentityError::UnknownPredecessor(event.predecessor.clone()))?;
        if self.inactive_instances.contains(&event.predecessor) {
            return Err(ContinuityIdentityError::PredecessorNoLongerActive(
                event.predecessor.clone(),
            ));
        }
        let latest_revision = self
            .latest_revision
            .get(&event.predecessor)
            .copied()
            .unwrap_or(predecessor_revision);
        if event.from_revision < latest_revision {
            return Err(ContinuityIdentityError::NonMonotonicInstanceRevision {
                instance: event.predecessor.clone(),
                previous: latest_revision,
                attempted: event.from_revision,
            });
        }
        if event.from_revision < predecessor_revision || event.to_revision < event.from_revision {
            return Err(ContinuityIdentityError::RevisionRegression);
        }
        for successor in &event.successors {
            if self.created_revision.contains_key(successor) {
                return Err(ContinuityIdentityError::InstanceAlreadyExists(successor.clone()));
            }
        }

        for successor in &event.successors {
            self.created_revision.insert(successor.clone(), event.to_revision);
            self.latest_revision.insert(successor.clone(), event.to_revision);
            self.parent.insert(successor.clone(), event.predecessor.clone());
            self.event_by_successor.insert(successor.clone(), event.id.clone());
        }
        self.latest_revision
            .insert(event.predecessor.clone(), event.to_revision);
        if !event.predecessor_continues {
            self.inactive_instances.insert(event.predecessor.clone());
        }
        self.events.insert(event.id.clone(), event);
        Ok(())
    }

    pub fn assess_instance(
        &self,
        instance: &SubjectInstanceId,
    ) -> Result<ContinuityAssessment, ContinuityIdentityError> {
        let event_id = self
            .event_by_successor
            .get(instance)
            .ok_or_else(|| ContinuityIdentityError::NoContinuityEvent(instance.clone()))?;
        let event = self.events.get(event_id).expect("event index is internal invariant");
        let mut siblings: BTreeSet<_> = event
            .successors
            .iter()
            .filter(|candidate| *candidate != instance)
            .cloned()
            .collect();
        if event.predecessor_continues {
            siblings.insert(event.predecessor.clone());
        }
        let operational_class = match event.kind {
            ContinuityKind::Uninterrupted => OperationalContinuityClass::DirectTransition,
            ContinuityKind::RestoredFromSnapshot if event.predecessor_continues => {
                OperationalContinuityClass::ForkedDescendant
            }
            ContinuityKind::RestoredFromSnapshot => OperationalContinuityClass::SnapshotRestoration,
            ContinuityKind::Fork => OperationalContinuityClass::ForkedDescendant,
            ContinuityKind::Reinitialized => OperationalContinuityClass::Reinitialized,
        };
        let operational_lineage_supported = event.kind != ContinuityKind::Reinitialized;

        Ok(ContinuityAssessment {
            instance: instance.clone(),
            predecessor: event.predecessor.clone(),
            operational_class,
            sibling_instances: siblings,
            exact_state_match_supported: event.exact_state_match,
            operational_lineage_supported,
            predecessor_continues: event.predecessor_continues,
        })
    }

    pub fn assess_branch_loss(
        &self,
        lost_instance: &SubjectInstanceId,
    ) -> Result<BranchLossAssessment, ContinuityIdentityError> {
        if !self.created_revision.contains_key(lost_instance) {
            return Err(ContinuityIdentityError::UnknownInstance(lost_instance.clone()));
        }
        let mut surviving_siblings = BTreeSet::new();

        if let Some(event_id) = self.event_by_successor.get(lost_instance) {
            let event = self.events.get(event_id).expect("event index is internal invariant");
            for successor in &event.successors {
                if successor != lost_instance {
                    surviving_siblings.insert(successor.clone());
                }
            }
            if event.predecessor_continues {
                surviving_siblings.insert(event.predecessor.clone());
            }
        }

        for event in self.events.values().filter(|event| {
            event.predecessor == *lost_instance && event.predecessor_continues
        }) {
            surviving_siblings.extend(event.successors.iter().cloned());
        }

        Ok(BranchLossAssessment {
            lost_instance: lost_instance.clone(),
            surviving_siblings,
        })
    }

    /// Shared recorded ancestry disqualifies two instances from being counted as
    /// independent replication merely because they are separate processes/copies.
    /// `false` does NOT establish external independence; WCARE-17 provenance is
    /// still required for that stronger claim.
    pub fn shares_recorded_ancestry(
        &self,
        left: &SubjectInstanceId,
        right: &SubjectInstanceId,
    ) -> Result<bool, ContinuityIdentityError> {
        if !self.created_revision.contains_key(left) {
            return Err(ContinuityIdentityError::UnknownInstance(left.clone()));
        }
        if !self.created_revision.contains_key(right) {
            return Err(ContinuityIdentityError::UnknownInstance(right.clone()));
        }
        if left == right {
            return Ok(true);
        }
        let left_ancestors = self.ancestor_set(left);
        let right_ancestors = self.ancestor_set(right);
        Ok(!left_ancestors.is_disjoint(&right_ancestors))
    }

    pub fn is_active(&self, instance: &SubjectInstanceId) -> Result<bool, ContinuityIdentityError> {
        if !self.created_revision.contains_key(instance) {
            return Err(ContinuityIdentityError::UnknownInstance(instance.clone()));
        }
        Ok(!self.inactive_instances.contains(instance))
    }

    fn ancestor_set(&self, instance: &SubjectInstanceId) -> BTreeSet<SubjectInstanceId> {
        let mut result = BTreeSet::new();
        let mut cursor = Some(instance.clone());
        while let Some(current) = cursor {
            if !result.insert(current.clone()) {
                break;
            }
            cursor = self.parent.get(&current).cloned();
        }
        result
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuityIdentityError {
    EmptyIdentifier,
    SuccessorRequired,
    DuplicateSuccessor,
    SuccessorEqualsPredecessor,
    ForkNeedsMultipleSuccessors,
    SingleSuccessorRequired,
    PredecessorCannotContinueForTransition,
    SnapshotDigestRequired,
    ExactMatchNeedsArtifact,
    ReinitializeCannotClaimExactMatch,
    MalformedArtifactDigest,
    EvidenceRequired,
    EmptyEvidenceReference,
    RevisionRegression,
    NonMonotonicInstanceRevision { instance: SubjectInstanceId, previous: u64, attempted: u64 },
    DuplicateEvent(ContinuityEventId),
    UnknownPredecessor(SubjectInstanceId),
    PredecessorNoLongerActive(SubjectInstanceId),
    InstanceAlreadyExists(SubjectInstanceId),
    UnknownInstance(SubjectInstanceId),
    NoContinuityEvent(SubjectInstanceId),
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    fn sid(value: &str) -> SubjectInstanceId { SubjectInstanceId::new(value).unwrap() }

    #[test]
    fn exact_snapshot_restore_supports_operational_not_phenomenal_identity() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("restore").unwrap(), sid("root"), vec![sid("restored")],
            ContinuityKind::RestoredFromSnapshot, 1, 2, false, Some(DIGEST.into()), true,
            ["receipt://restore".into()],
        ).unwrap()).unwrap();
        let assessment = ledger.assess_instance(&sid("restored")).unwrap();
        assert!(assessment.exact_state_match_supported());
        assert!(assessment.operational_lineage_supported());
        assert!(!assessment.phenomenal_identity_established());
        assert!(!ledger.is_active(&sid("root")).unwrap());
    }

    #[test]
    fn concurrent_restore_is_branch_multiplicity() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("restore-copy").unwrap(), sid("root"), vec![sid("copy")],
            ContinuityKind::RestoredFromSnapshot, 1, 2, true, Some(DIGEST.into()), true,
            ["receipt://restore".into()],
        ).unwrap()).unwrap();
        let assessment = ledger.assess_instance(&sid("copy")).unwrap();
        assert_eq!(assessment.operational_class(), OperationalContinuityClass::ForkedDescendant);
        assert!(assessment.sibling_instances().contains(&sid("root")));
        let root_loss = ledger.assess_branch_loss(&sid("root")).unwrap();
        assert!(root_loss.surviving_siblings().contains(&sid("copy")));
    }

    #[test]
    fn inactive_predecessor_cannot_spawn_new_history() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("first").unwrap(), sid("root"), vec![sid("next")],
            ContinuityKind::Uninterrupted, 1, 2, false, None, false,
            ["receipt://first".into()],
        ).unwrap()).unwrap();
        let result = ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("impossible").unwrap(), sid("root"), vec![sid("ghost")],
            ContinuityKind::Uninterrupted, 2, 3, false, None, false,
            ["receipt://second".into()],
        ).unwrap());
        assert!(matches!(result, Err(ContinuityIdentityError::PredecessorNoLongerActive(_))));
    }

    #[test]
    fn continuing_predecessor_cannot_spawn_retroactive_branch() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("copy-late").unwrap(), sid("root"), vec![sid("copy-a")],
            ContinuityKind::RestoredFromSnapshot, 10, 11, true, Some(DIGEST.into()), true,
            ["receipt://late".into()],
        ).unwrap()).unwrap();
        let result = ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("copy-backdated").unwrap(), sid("root"), vec![sid("copy-b")],
            ContinuityKind::RestoredFromSnapshot, 5, 6, true, Some(DIGEST.into()), true,
            ["receipt://backdated".into()],
        ).unwrap());
        assert!(matches!(result, Err(ContinuityIdentityError::NonMonotonicInstanceRevision { .. })));
    }

    #[test]
    fn fork_creates_distinct_siblings_not_independent_replication() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("fork").unwrap(), sid("root"), vec![sid("a"), sid("b")],
            ContinuityKind::Fork, 1, 2, false, Some(DIGEST.into()), true,
            ["receipt://fork".into()],
        ).unwrap()).unwrap();
        let a = ledger.assess_instance(&sid("a")).unwrap();
        assert!(a.sibling_instances().contains(&sid("b")));
        assert!(ledger.shares_recorded_ancestry(&sid("a"), &sid("b")).unwrap());
        assert!(!a.phenomenal_identity_established());
    }
}
