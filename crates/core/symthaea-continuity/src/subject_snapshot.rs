// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world canonical snapshots of typed continuity subjects.
//!
//! `ContinuitySubjectV1` proves the canonical identity of one typed operational
//! subject. A realization/evidence layer also needs to know that an exact
//! `ContinuitySubjectId` resolves inside one exact subject universe and that any
//! declared containment parent resolves there as well.
//!
//! This module provides that structural boundary only. Parent edges remain
//! containment/context edges, not ownership or authority edges.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ContinuitySubjectError, ContinuitySubjectId, ContinuitySubjectV1};

/// Stable schema for one closed typed-subject snapshot.
pub const CONTINUITY_SUBJECT_SNAPSHOT_SCHEMA_V1: &str = "symthaea-continuity-subject-snapshot-v1";

const SUBJECT_SNAPSHOT_DOMAIN: &[u8] = b"symthaea.continuity.subject-snapshot.v1\0";
const MAX_SUBJECTS: usize = 65_536;
const MAX_PARENT_DEPTH: usize = 256;

/// Exact content identity of one canonical closed subject snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContinuitySubjectSnapshotId([u8; 32]);

impl ContinuitySubjectSnapshotId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable candidate containing one exact closed universe of typed
/// continuity subjects.
///
/// Subjects are canonically ordered by exact `ContinuitySubjectId`. Every
/// `parent_subject_id` must resolve within the same snapshot. The snapshot does
/// not prove that any subject exists in the physical world, is currently
/// observed, is owned by the caller, or is authorized for an action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuitySubjectSnapshotV1 {
    schema_version: String,
    subjects: Vec<ContinuitySubjectV1>,
    snapshot_id: ContinuitySubjectSnapshotId,
}

impl ContinuitySubjectSnapshotV1 {
    /// Construct one canonical closed subject snapshot.
    pub fn new(
        mut subjects: Vec<ContinuitySubjectV1>,
    ) -> Result<Self, ContinuitySubjectSnapshotError> {
        validate_count(subjects.len())?;
        for subject in &subjects {
            subject.validate()?;
        }

        subjects.sort_by_key(ContinuitySubjectV1::id);
        validate_unique(&subjects)?;
        validate_parent_closure(&subjects)?;
        validate_parent_structure(&subjects)?;

        let snapshot_id = ContinuitySubjectSnapshotId(hash_snapshot(&subjects));
        Ok(Self {
            schema_version: CONTINUITY_SUBJECT_SNAPSHOT_SCHEMA_V1.to_owned(),
            subjects,
            snapshot_id,
        })
    }

    pub fn id(&self) -> ContinuitySubjectSnapshotId {
        self.snapshot_id
    }

    pub fn subjects(&self) -> &[ContinuitySubjectV1] {
        &self.subjects
    }

    /// Revalidate an untrusted/transported snapshot and return a non-Serde
    /// downstream-trusted wrapper.
    pub fn validate(
        &self,
    ) -> Result<ValidatedContinuitySubjectSnapshotV1, ContinuitySubjectSnapshotError> {
        if self.schema_version != CONTINUITY_SUBJECT_SNAPSHOT_SCHEMA_V1 {
            return Err(ContinuitySubjectSnapshotError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        validate_count(self.subjects.len())?;
        for subject in &self.subjects {
            subject.validate()?;
        }

        if self
            .subjects
            .windows(2)
            .any(|pair| pair[0].id() >= pair[1].id())
        {
            if self
                .subjects
                .windows(2)
                .any(|pair| pair[0].id() == pair[1].id())
            {
                return Err(ContinuitySubjectSnapshotError::DuplicateSubject);
            }
            return Err(ContinuitySubjectSnapshotError::NonCanonicalSubjectOrder);
        }

        validate_parent_closure(&self.subjects)?;
        validate_parent_structure(&self.subjects)?;

        let expected = ContinuitySubjectSnapshotId(hash_snapshot(&self.subjects));
        if expected != self.snapshot_id {
            return Err(ContinuitySubjectSnapshotError::SnapshotIdentityMismatch);
        }

        let index = self
            .subjects
            .iter()
            .enumerate()
            .map(|(index, subject)| (subject.id(), index))
            .collect();

        Ok(ValidatedContinuitySubjectSnapshotV1 {
            inner: self.clone(),
            index,
        })
    }
}

/// Non-Serde validated closed subject universe.
///
/// This wrapper proves only canonical subject identities, closed parent
/// references, bounded/acyclic containment structure, and exact snapshot
/// identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedContinuitySubjectSnapshotV1 {
    inner: ContinuitySubjectSnapshotV1,
    index: BTreeMap<ContinuitySubjectId, usize>,
}

impl ValidatedContinuitySubjectSnapshotV1 {
    pub fn id(&self) -> ContinuitySubjectSnapshotId {
        self.inner.id()
    }

    pub fn len(&self) -> usize {
        self.inner.subjects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.subjects.is_empty()
    }

    pub fn subjects(&self) -> &[ContinuitySubjectV1] {
        self.inner.subjects()
    }

    /// Resolve one exact typed subject identity in this exact snapshot.
    pub fn subject(&self, subject_id: ContinuitySubjectId) -> Option<&ContinuitySubjectV1> {
        self.index
            .get(&subject_id)
            .map(|index| &self.inner.subjects[*index])
    }

    pub fn contains(&self, subject_id: ContinuitySubjectId) -> bool {
        self.index.contains_key(&subject_id)
    }

    pub fn as_raw(&self) -> &ContinuitySubjectSnapshotV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContinuitySubjectSnapshotError {
    #[error("unsupported continuity subject snapshot schema: {0}")]
    UnsupportedSchema(String),
    #[error(transparent)]
    Subject(#[from] ContinuitySubjectError),
    #[error("continuity subject snapshot exceeds {MAX_SUBJECTS} subjects")]
    TooManySubjects,
    #[error("continuity subject snapshot contains duplicate exact subject identity")]
    DuplicateSubject,
    #[error("continuity subjects must be in strict canonical subject-id order")]
    NonCanonicalSubjectOrder,
    #[error("subject {child:?} references parent {parent:?} absent from the same snapshot")]
    MissingParent {
        child: ContinuitySubjectId,
        parent: ContinuitySubjectId,
    },
    #[error(
        "subject traversal from {origin:?} reached {missing:?}, which is absent from the snapshot index"
    )]
    TraversalInvariantMissingSubject {
        origin: ContinuitySubjectId,
        missing: ContinuitySubjectId,
    },
    #[error("subject containment parent graph contains a cycle reachable from {0:?}")]
    ParentCycle(ContinuitySubjectId),
    #[error("subject containment parent chain exceeds {MAX_PARENT_DEPTH} nodes from {0:?}")]
    ParentDepthExceeded(ContinuitySubjectId),
    #[error("stored continuity subject snapshot identity does not match canonical subjects")]
    SnapshotIdentityMismatch,
}

fn validate_count(count: usize) -> Result<(), ContinuitySubjectSnapshotError> {
    if count > MAX_SUBJECTS {
        return Err(ContinuitySubjectSnapshotError::TooManySubjects);
    }
    Ok(())
}

fn validate_unique(subjects: &[ContinuitySubjectV1]) -> Result<(), ContinuitySubjectSnapshotError> {
    if subjects.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
        return Err(ContinuitySubjectSnapshotError::DuplicateSubject);
    }
    Ok(())
}

fn validate_parent_closure(
    subjects: &[ContinuitySubjectV1],
) -> Result<(), ContinuitySubjectSnapshotError> {
    let known: BTreeSet<_> = subjects.iter().map(ContinuitySubjectV1::id).collect();
    for subject in subjects {
        if let Some(parent) = subject.parent_subject_id() {
            if !known.contains(&parent) {
                return Err(ContinuitySubjectSnapshotError::MissingParent {
                    child: subject.id(),
                    parent,
                });
            }
        }
    }
    Ok(())
}

fn validate_parent_structure(
    subjects: &[ContinuitySubjectV1],
) -> Result<(), ContinuitySubjectSnapshotError> {
    let index: BTreeMap<_, _> = subjects
        .iter()
        .map(|subject| (subject.id(), subject))
        .collect();

    for subject in subjects {
        let origin = subject.id();
        let mut seen = BTreeSet::new();
        let mut current = Some(origin);
        let mut depth = 0usize;

        while let Some(subject_id) = current {
            if !seen.insert(subject_id) {
                return Err(ContinuitySubjectSnapshotError::ParentCycle(origin));
            }
            depth += 1;
            if depth > MAX_PARENT_DEPTH {
                return Err(ContinuitySubjectSnapshotError::ParentDepthExceeded(origin));
            }
            let current_subject = index.get(&subject_id).ok_or(
                ContinuitySubjectSnapshotError::TraversalInvariantMissingSubject {
                    origin,
                    missing: subject_id,
                },
            )?;
            current = current_subject.parent_subject_id();
        }
    }

    Ok(())
}

fn hash_snapshot(subjects: &[ContinuitySubjectV1]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(64 + subjects.len() * 32);
    put_str(&mut bytes, CONTINUITY_SUBJECT_SNAPSHOT_SCHEMA_V1);
    put_len(&mut bytes, subjects.len());
    for subject in subjects {
        bytes.extend_from_slice(subject.id().as_bytes());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(SUBJECT_SNAPSHOT_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContinuityScopeV1;

    fn root(name: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", name, scope, None).unwrap()
    }

    #[test]
    fn constructor_canonicalizes_subject_order() {
        let a = root("a", ContinuityScopeV1::Service);
        let b = root("b", ContinuityScopeV1::Machine);
        let left = ContinuitySubjectSnapshotV1::new(vec![a.clone(), b.clone()]).unwrap();
        let right = ContinuitySubjectSnapshotV1::new(vec![b, a]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.id(), right.id());
    }

    #[test]
    fn missing_parent_is_rejected() {
        let parent = root("fabric-a", ContinuityScopeV1::NetworkFabric);
        let child = ContinuitySubjectV1::new(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(parent.id()),
        )
        .unwrap();

        assert_eq!(
            ContinuitySubjectSnapshotV1::new(vec![child.clone()]),
            Err(ContinuitySubjectSnapshotError::MissingParent {
                child: child.id(),
                parent: parent.id(),
            })
        );
    }

    #[test]
    fn parent_structure_fails_closed_without_closure_precondition() {
        let parent = root("fabric-a", ContinuityScopeV1::NetworkFabric);
        let child = ContinuitySubjectV1::new(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(parent.id()),
        )
        .unwrap();

        assert_eq!(
            validate_parent_structure(std::slice::from_ref(&child)),
            Err(
                ContinuitySubjectSnapshotError::TraversalInvariantMissingSubject {
                    origin: child.id(),
                    missing: parent.id(),
                }
            )
        );
    }

    #[test]
    fn exact_duplicate_subject_is_rejected() {
        let subject = root("payments-api", ContinuityScopeV1::Service);
        assert_eq!(
            ContinuitySubjectSnapshotV1::new(vec![subject.clone(), subject]),
            Err(ContinuitySubjectSnapshotError::DuplicateSubject)
        );
    }

    #[test]
    fn closed_parent_chain_resolves_through_validated_wrapper() {
        let site = root("site-a", ContinuityScopeV1::Site);
        let cluster = ContinuitySubjectV1::new(
            "org.example",
            "cluster-a",
            ContinuityScopeV1::Cluster,
            Some(site.id()),
        )
        .unwrap();
        let machine = ContinuitySubjectV1::new(
            "org.example",
            "machine-a",
            ContinuityScopeV1::Machine,
            Some(cluster.id()),
        )
        .unwrap();
        let machine_id = machine.id();

        let validated = ContinuitySubjectSnapshotV1::new(vec![machine, site, cluster])
            .unwrap()
            .validate()
            .unwrap();
        assert_eq!(validated.len(), 3);
        assert_eq!(
            validated.subject(machine_id).unwrap().scope(),
            &ContinuityScopeV1::Machine
        );
    }

    #[test]
    fn transported_noncanonical_order_is_rejected() {
        let a = root("a", ContinuityScopeV1::Service);
        let b = root("b", ContinuityScopeV1::Service);
        let mut snapshot = ContinuitySubjectSnapshotV1::new(vec![a, b]).unwrap();
        snapshot.subjects.reverse();
        assert_eq!(
            snapshot.validate(),
            Err(ContinuitySubjectSnapshotError::NonCanonicalSubjectOrder)
        );
    }

    #[test]
    fn transported_snapshot_identity_mismatch_is_rejected() {
        let mut snapshot =
            ContinuitySubjectSnapshotV1::new(vec![root("service-a", ContinuityScopeV1::Service)])
                .unwrap();
        snapshot.snapshot_id = ContinuitySubjectSnapshotId([7; 32]);
        assert_eq!(
            snapshot.validate(),
            Err(ContinuitySubjectSnapshotError::SnapshotIdentityMismatch)
        );
    }

    #[test]
    fn parent_depth_is_bounded() {
        let mut subjects = Vec::new();
        let mut parent = None;
        for index in 0..=MAX_PARENT_DEPTH {
            let subject = ContinuitySubjectV1::new(
                "org.example",
                format!("subject-{index:03}"),
                ContinuityScopeV1::Custom {
                    kind_id: "containment-test".to_owned(),
                },
                parent,
            )
            .unwrap();
            parent = Some(subject.id());
            subjects.push(subject);
        }

        assert!(matches!(
            ContinuitySubjectSnapshotV1::new(subjects),
            Err(ContinuitySubjectSnapshotError::ParentDepthExceeded(_))
        ));
    }

    #[test]
    fn snapshot_identity_changes_when_subject_context_changes() {
        let site_a = root("site-a", ContinuityScopeV1::Site);
        let site_b = root("site-b", ContinuityScopeV1::Site);
        let machine_a = ContinuitySubjectV1::new(
            "org.example",
            "machine-01",
            ContinuityScopeV1::Machine,
            Some(site_a.id()),
        )
        .unwrap();
        let machine_b = ContinuitySubjectV1::new(
            "org.example",
            "machine-01",
            ContinuityScopeV1::Machine,
            Some(site_b.id()),
        )
        .unwrap();

        let left = ContinuitySubjectSnapshotV1::new(vec![site_a, machine_a]).unwrap();
        let right = ContinuitySubjectSnapshotV1::new(vec![site_b, machine_b]).unwrap();
        assert_ne!(left.id(), right.id());
    }
}
