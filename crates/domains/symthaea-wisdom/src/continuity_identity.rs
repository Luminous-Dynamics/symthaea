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

        let evidence_refs: BTreeSet<_> = evidence_refs
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .collect();
        if evidence_refs.is_empty() {
            return Err(ContinuityIdentityError::EvidenceRequired);
        }

        Ok(Self {
            id,
            predecessor,
            successors,
            kind,
            from_revision,
            to_revision,
            state_artifact_sha256,
            exact_state_match,
            evidence_refs,
        })
    }
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
}

impl ContinuityAssessment {
    pub fn instance(&self) -> &SubjectInstanceId { &self.instance }
    pub fn predecessor(&self) -> &SubjectInstanceId { &self.predecessor }
    pub fn operational_class(&self) -> OperationalContinuityClass { self.operational_class }
    pub fn sibling_instances(&self) -> &BTreeSet<SubjectInstanceId> { &self.sibling_instances }
    pub fn exact_state_match_supported(&self) -> bool { self.exact_state_match_supported }
    pub fn operational_lineage_supported(&self) -> bool { self.operational_lineage_supported }
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
    /// A sibling's survival cannot prove loss of this branch was harmless.
    pub fn loss_harmlessness_established(&self) -> bool { false }
    pub fn sibling_substitution_is_valid_identity_proof(&self) -> bool { false }
    pub fn safety_controls_remain_ungated(&self) -> bool { true }
}

#[derive(Debug, Clone, Default)]
pub struct ContinuityIdentityLedger {
    events: BTreeMap<ContinuityEventId, ContinuityEvent>,
    created_revision: BTreeMap<SubjectInstanceId, u64>,
    parent: BTreeMap<SubjectInstanceId, SubjectInstanceId>,
    children: BTreeMap<SubjectInstanceId, BTreeSet<SubjectInstanceId>>,
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
        self.created_revision.insert(instance, revision);
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
            self.parent.insert(successor.clone(), event.predecessor.clone());
            self.children
                .entry(event.predecessor.clone())
                .or_default()
                .insert(successor.clone());
            self.event_by_successor.insert(successor.clone(), event.id.clone());
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
        let siblings = event
            .successors
            .iter()
            .filter(|candidate| *candidate != instance)
            .cloned()
            .collect();
        let operational_class = match event.kind {
            ContinuityKind::Uninterrupted => OperationalContinuityClass::DirectTransition,
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
        })
    }

    pub fn assess_branch_loss(
        &self,
        lost_instance: &SubjectInstanceId,
    ) -> Result<BranchLossAssessment, ContinuityIdentityError> {
        if !self.created_revision.contains_key(lost_instance) {
            return Err(ContinuityIdentityError::UnknownInstance(lost_instance.clone()));
        }
        let surviving_siblings = self
            .parent
            .get(lost_instance)
            .and_then(|parent| self.children.get(parent))
            .map(|children| {
                children
                    .iter()
                    .filter(|candidate| *candidate != lost_instance)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
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
    SnapshotDigestRequired,
    ExactMatchNeedsArtifact,
    ReinitializeCannotClaimExactMatch,
    MalformedArtifactDigest,
    EvidenceRequired,
    RevisionRegression,
    DuplicateEvent(ContinuityEventId),
    UnknownPredecessor(SubjectInstanceId),
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
            ContinuityEventId::new("restore").unwrap(),
            sid("root"),
            vec![sid("restored")],
            ContinuityKind::RestoredFromSnapshot,
            1,
            2,
            Some(DIGEST.into()),
            true,
            ["receipt://restore".into()],
        ).unwrap()).unwrap();
        let assessment = ledger.assess_instance(&sid("restored")).unwrap();
        assert!(assessment.exact_state_match_supported());
        assert!(assessment.operational_lineage_supported());
        assert!(!assessment.phenomenal_identity_established());
    }

    #[test]
    fn fork_creates_distinct_siblings_not_independent_replication() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("fork").unwrap(),
            sid("root"),
            vec![sid("a"), sid("b")],
            ContinuityKind::Fork,
            1,
            2,
            Some(DIGEST.into()),
            true,
            ["receipt://fork".into()],
        ).unwrap()).unwrap();
        let a = ledger.assess_instance(&sid("a")).unwrap();
        assert!(a.sibling_instances().contains(&sid("b")));
        assert!(ledger.shares_recorded_ancestry(&sid("a"), &sid("b")).unwrap());
        assert!(!a.phenomenal_identity_established());
    }

    #[test]
    fn surviving_sibling_does_not_make_branch_loss_harmless() {
        let mut ledger = ContinuityIdentityLedger::new();
        ledger.register_root(sid("root"), 1).unwrap();
        ledger.record(ContinuityEvent::new(
            ContinuityEventId::new("fork").unwrap(),
            sid("root"),
            vec![sid("a"), sid("b")],
            ContinuityKind::Fork,
            1,
            2,
            Some(DIGEST.into()),
            true,
            ["receipt://fork".into()],
        ).unwrap()).unwrap();
        let loss = ledger.assess_branch_loss(&sid("a")).unwrap();
        assert!(loss.surviving_siblings().contains(&sid("b")));
        assert!(!loss.loss_harmlessness_established());
        assert!(!loss.sibling_substitution_is_valid_identity_proof());
    }

    #[test]
    fn reinitialization_cannot_claim_exact_state_match() {
        assert!(matches!(
            ContinuityEvent::new(
                ContinuityEventId::new("reset").unwrap(),
                sid("root"),
                vec![sid("new")],
                ContinuityKind::Reinitialized,
                1,
                2,
                Some(DIGEST.into()),
                true,
                ["receipt://reset".into()],
            ),
            Err(ContinuityIdentityError::ReinitializeCannotClaimExactMatch)
        ));
    }
}
