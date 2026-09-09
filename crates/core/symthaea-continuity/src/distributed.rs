// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic distributed transition safety budgets.
//!
//! A machine-local transition can be correct while the larger system becomes
//! unavailable: two HA peers can reboot together, too many quorum members can be
//! removed, or the only management path can disappear. This module defines a
//! deliberately protocol-neutral policy vocabulary for those cross-subject change
//! constraints.
//!
//! It does not implement quorum mathematics, health observation, fencing, routing,
//! storage ownership, or execution. Those remain verifier/adapter concerns.
//!
//! Core theorem:
//!
//! `DistributedChangeBudgetV1 != CurrentDistributedState != CommitEligibility != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::scope::{ContinuitySubjectError, ContinuitySubjectId, ContinuitySubjectV1};

pub const DISTRIBUTED_CHANGE_BUDGET_SCHEMA_V1: &str =
    "symthaea-continuity-distributed-change-budget-v1";

const BUDGET_DOMAIN: &[u8] = b"symthaea.continuity.distributed-change-budget.v1\0";
const MAX_PARTICIPANTS: usize = 4096;
const MAX_EXCLUSION_SETS: usize = 4096;
const MAX_TEXT_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DistributedChangeBudgetId([u8; 32]);

impl DistributedChangeBudgetId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A recovery mechanism class that a later verifier may prove is available for
/// the proposed distributed transition.
///
/// The budget's `recovery_path_any_of` list means that at least one listed class
/// must later be independently established. Merely naming a class here is policy,
/// not evidence that the path exists or works.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RecoveryPathClassV1 {
    DeviceLocalAutomaticRollback,
    OutOfBandManagement,
    IndependentNetworkPath,
    LocalPhysicalIntervention,
    Custom { kind_id: String },
}

impl RecoveryPathClassV1 {
    pub fn custom(kind_id: impl Into<String>) -> Result<Self, DistributedChangeBudgetError> {
        Ok(Self::Custom {
            kind_id: checked_text("custom recovery path kind_id", kind_id.into())?,
        })
    }

    fn normalize(self) -> Result<Self, DistributedChangeBudgetError> {
        match self {
            Self::Custom { kind_id } => Self::custom(kind_id),
            other => Ok(other),
        }
    }

    fn validate(&self) -> Result<(), DistributedChangeBudgetError> {
        if let Self::Custom { kind_id } = self {
            let canonical = checked_text("custom recovery path kind_id", kind_id.clone())?;
            if canonical != *kind_id {
                return Err(DistributedChangeBudgetError::NonCanonicalText {
                    field: "custom recovery path kind_id",
                });
            }
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::DeviceLocalAutomaticRollback => out.push(1),
            Self::OutOfBandManagement => out.push(2),
            Self::IndependentNetworkPath => out.push(3),
            Self::LocalPhysicalIntervention => out.push(4),
            Self::Custom { kind_id } => {
                out.push(255);
                put_str(out, kind_id);
            }
        }
    }
}

/// A set of exact participant subjects that policy forbids from transitioning
/// concurrently. V1 interprets a set pairwise: at most one member of the set may
/// be in the protected transition state at a time.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MutualExclusionSetV1 {
    members: Vec<ContinuitySubjectId>,
}

impl MutualExclusionSetV1 {
    pub fn new(
        mut members: Vec<ContinuitySubjectId>,
    ) -> Result<Self, DistributedChangeBudgetError> {
        members.sort();
        members.dedup();
        if members.len() < 2 {
            return Err(DistributedChangeBudgetError::ExclusionSetTooSmall);
        }
        Ok(Self { members })
    }

    pub fn members(&self) -> &[ContinuitySubjectId] {
        &self.members
    }

    fn validate(&self) -> Result<(), DistributedChangeBudgetError> {
        if self.members.len() < 2 {
            return Err(DistributedChangeBudgetError::ExclusionSetTooSmall);
        }
        if self.members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(DistributedChangeBudgetError::NonCanonicalExclusionSet);
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        put_len(out, self.members.len());
        for member in &self.members {
            out.extend_from_slice(member.as_bytes());
        }
    }
}

/// Serializable policy proposal for bounded distributed changes.
///
/// This value states organizational transition constraints. It does not prove the
/// participant set is current, that members are healthy, that a recovery path is
/// available, or that the named aggregate subject is trusted. Downstream code
/// should use `validate_against_subject()` to obtain the non-Serde validated form
/// before composing current-state evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedChangeBudgetV1 {
    schema_version: String,
    aggregate_subject_id: ContinuitySubjectId,
    generation: u64,
    participant_subject_ids: Vec<ContinuitySubjectId>,
    max_concurrent_unavailable: u32,
    minimum_healthy: u32,
    mutual_exclusion_sets: Vec<MutualExclusionSetV1>,
    recovery_path_any_of: Vec<RecoveryPathClassV1>,
    budget_id: DistributedChangeBudgetId,
}

impl DistributedChangeBudgetV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        aggregate_subject: &ContinuitySubjectV1,
        generation: u64,
        mut participant_subject_ids: Vec<ContinuitySubjectId>,
        max_concurrent_unavailable: u32,
        minimum_healthy: u32,
        mut mutual_exclusion_sets: Vec<MutualExclusionSetV1>,
        recovery_path_any_of: Vec<RecoveryPathClassV1>,
    ) -> Result<Self, DistributedChangeBudgetError> {
        aggregate_subject.validate()?;
        if generation == 0 {
            return Err(DistributedChangeBudgetError::ZeroGeneration);
        }
        if participant_subject_ids.is_empty() {
            return Err(DistributedChangeBudgetError::NoParticipants);
        }
        if participant_subject_ids.len() > MAX_PARTICIPANTS {
            return Err(DistributedChangeBudgetError::TooManyParticipants);
        }
        participant_subject_ids.sort();
        participant_subject_ids.dedup();
        if participant_subject_ids
            .binary_search(&aggregate_subject.id())
            .is_ok()
        {
            return Err(DistributedChangeBudgetError::AggregateIncludedAsParticipant);
        }

        let participant_count = participant_subject_ids.len() as u32;
        validate_availability_budget(
            participant_count,
            max_concurrent_unavailable,
            minimum_healthy,
        )?;

        if mutual_exclusion_sets.len() > MAX_EXCLUSION_SETS {
            return Err(DistributedChangeBudgetError::TooManyExclusionSets);
        }
        for set in &mutual_exclusion_sets {
            set.validate()?;
            for member in set.members() {
                if participant_subject_ids.binary_search(member).is_err() {
                    return Err(DistributedChangeBudgetError::ExclusionMemberOutsideParticipants {
                        member: *member,
                    });
                }
            }
        }
        mutual_exclusion_sets.sort();
        if mutual_exclusion_sets.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(DistributedChangeBudgetError::DuplicateExclusionSet);
        }

        let mut recovery_paths = Vec::with_capacity(recovery_path_any_of.len());
        for path in recovery_path_any_of {
            recovery_paths.push(path.normalize()?);
        }
        recovery_paths.sort();
        recovery_paths.dedup();
        if recovery_paths.is_empty() {
            return Err(DistributedChangeBudgetError::NoRecoveryPathClass);
        }

        let aggregate_subject_id = aggregate_subject.id();
        let budget_id = DistributedChangeBudgetId(hash_budget(
            aggregate_subject_id,
            generation,
            &participant_subject_ids,
            max_concurrent_unavailable,
            minimum_healthy,
            &mutual_exclusion_sets,
            &recovery_paths,
        ));

        Ok(Self {
            schema_version: DISTRIBUTED_CHANGE_BUDGET_SCHEMA_V1.to_owned(),
            aggregate_subject_id,
            generation,
            participant_subject_ids,
            max_concurrent_unavailable,
            minimum_healthy,
            mutual_exclusion_sets,
            recovery_path_any_of: recovery_paths,
            budget_id,
        })
    }

    /// Intrinsic validation of the serialized policy and stored content identity.
    /// This does not prove that `aggregate_subject_id` corresponds to a currently
    /// trusted local subject object.
    pub fn validate(&self) -> Result<(), DistributedChangeBudgetError> {
        if self.schema_version != DISTRIBUTED_CHANGE_BUDGET_SCHEMA_V1 {
            return Err(DistributedChangeBudgetError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.generation == 0 {
            return Err(DistributedChangeBudgetError::ZeroGeneration);
        }
        if self.participant_subject_ids.is_empty() {
            return Err(DistributedChangeBudgetError::NoParticipants);
        }
        if self.participant_subject_ids.len() > MAX_PARTICIPANTS {
            return Err(DistributedChangeBudgetError::TooManyParticipants);
        }
        if self
            .participant_subject_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(DistributedChangeBudgetError::NonCanonicalParticipants);
        }
        if self
            .participant_subject_ids
            .binary_search(&self.aggregate_subject_id)
            .is_ok()
        {
            return Err(DistributedChangeBudgetError::AggregateIncludedAsParticipant);
        }

        validate_availability_budget(
            self.participant_subject_ids.len() as u32,
            self.max_concurrent_unavailable,
            self.minimum_healthy,
        )?;

        if self.mutual_exclusion_sets.len() > MAX_EXCLUSION_SETS {
            return Err(DistributedChangeBudgetError::TooManyExclusionSets);
        }
        for set in &self.mutual_exclusion_sets {
            set.validate()?;
            for member in set.members() {
                if self.participant_subject_ids.binary_search(member).is_err() {
                    return Err(DistributedChangeBudgetError::ExclusionMemberOutsideParticipants {
                        member: *member,
                    });
                }
            }
        }
        if self
            .mutual_exclusion_sets
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(DistributedChangeBudgetError::NonCanonicalExclusionSets);
        }

        if self.recovery_path_any_of.is_empty() {
            return Err(DistributedChangeBudgetError::NoRecoveryPathClass);
        }
        for path in &self.recovery_path_any_of {
            path.validate()?;
        }
        if self
            .recovery_path_any_of
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(DistributedChangeBudgetError::NonCanonicalRecoveryPaths);
        }

        let expected = DistributedChangeBudgetId(hash_budget(
            self.aggregate_subject_id,
            self.generation,
            &self.participant_subject_ids,
            self.max_concurrent_unavailable,
            self.minimum_healthy,
            &self.mutual_exclusion_sets,
            &self.recovery_path_any_of,
        ));
        if expected != self.budget_id {
            return Err(DistributedChangeBudgetError::BudgetIdentityMismatch);
        }
        Ok(())
    }

    /// Re-bind this serialized policy to the exact local aggregate subject and
    /// return a non-Serde validated wrapper.
    pub fn validate_against_subject(
        &self,
        aggregate_subject: &ContinuitySubjectV1,
    ) -> Result<ValidatedDistributedChangeBudgetV1, DistributedChangeBudgetError> {
        self.validate()?;
        aggregate_subject.validate()?;
        if aggregate_subject.id() != self.aggregate_subject_id {
            return Err(DistributedChangeBudgetError::AggregateSubjectMismatch);
        }
        Ok(ValidatedDistributedChangeBudgetV1 {
            aggregate_subject: aggregate_subject.clone(),
            inner: self.clone(),
        })
    }

    pub fn id(&self) -> DistributedChangeBudgetId {
        self.budget_id
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.aggregate_subject_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn participant_subject_ids(&self) -> &[ContinuitySubjectId] {
        &self.participant_subject_ids
    }

    pub fn max_concurrent_unavailable(&self) -> u32 {
        self.max_concurrent_unavailable
    }

    pub fn minimum_healthy(&self) -> u32 {
        self.minimum_healthy
    }

    pub fn mutual_exclusion_sets(&self) -> &[MutualExclusionSetV1] {
        &self.mutual_exclusion_sets
    }

    pub fn recovery_path_any_of(&self) -> &[RecoveryPathClassV1] {
        &self.recovery_path_any_of
    }
}

/// Non-Serde structurally validated policy bound to one exact local aggregate
/// continuity subject. This still does not prove participant membership or current
/// health and is not an execution permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedDistributedChangeBudgetV1 {
    aggregate_subject: ContinuitySubjectV1,
    inner: DistributedChangeBudgetV1,
}

impl ValidatedDistributedChangeBudgetV1 {
    pub fn id(&self) -> DistributedChangeBudgetId {
        self.inner.id()
    }

    pub fn aggregate_subject(&self) -> &ContinuitySubjectV1 {
        &self.aggregate_subject
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.inner.aggregate_subject_id()
    }

    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    pub fn participant_subject_ids(&self) -> &[ContinuitySubjectId] {
        self.inner.participant_subject_ids()
    }

    pub fn max_concurrent_unavailable(&self) -> u32 {
        self.inner.max_concurrent_unavailable()
    }

    pub fn minimum_healthy(&self) -> u32 {
        self.inner.minimum_healthy()
    }

    pub fn mutual_exclusion_sets(&self) -> &[MutualExclusionSetV1] {
        self.inner.mutual_exclusion_sets()
    }

    pub fn recovery_path_any_of(&self) -> &[RecoveryPathClassV1] {
        self.inner.recovery_path_any_of()
    }

    pub fn as_raw(&self) -> &DistributedChangeBudgetV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DistributedChangeBudgetError {
    #[error(transparent)]
    Subject(#[from] ContinuitySubjectError),
    #[error("unsupported distributed change budget schema: {0}")]
    UnsupportedSchema(String),
    #[error("distributed change budget generation must be non-zero")]
    ZeroGeneration,
    #[error("distributed change budget must contain at least one participant")]
    NoParticipants,
    #[error("distributed change budget exceeds the participant bound")]
    TooManyParticipants,
    #[error("participant subject identities must be in canonical sorted unique order")]
    NonCanonicalParticipants,
    #[error("aggregate continuity subject cannot also be a participant subject")]
    AggregateIncludedAsParticipant,
    #[error("max_concurrent_unavailable {requested} exceeds participant count {participants}")]
    MaxUnavailableExceedsParticipants { requested: u32, participants: u32 },
    #[error("minimum_healthy {requested} exceeds participant count {participants}")]
    MinimumHealthyExceedsParticipants { requested: u32, participants: u32 },
    #[error("availability budget is contradictory: participant count {participants}, max unavailable {max_unavailable}, minimum healthy {minimum_healthy}")]
    ContradictoryAvailabilityBudget {
        participants: u32,
        max_unavailable: u32,
        minimum_healthy: u32,
    },
    #[error("mutual exclusion set must contain at least two distinct participant subjects")]
    ExclusionSetTooSmall,
    #[error("mutual exclusion set members must be in canonical sorted unique order")]
    NonCanonicalExclusionSet,
    #[error("distributed change budget exceeds the mutual-exclusion-set bound")]
    TooManyExclusionSets,
    #[error("mutual exclusion set contains subject outside participant set: {member:?}")]
    ExclusionMemberOutsideParticipants { member: ContinuitySubjectId },
    #[error("distributed change budget contains duplicate mutual exclusion sets")]
    DuplicateExclusionSet,
    #[error("mutual exclusion sets must be in canonical sorted unique order")]
    NonCanonicalExclusionSets,
    #[error("distributed change budget must require at least one acceptable recovery path class")]
    NoRecoveryPathClass,
    #[error("recovery path classes must be in canonical sorted unique order")]
    NonCanonicalRecoveryPaths,
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds {MAX_TEXT_BYTES} bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("{field} is not canonically trimmed")]
    NonCanonicalText { field: &'static str },
    #[error("stored distributed change budget identity does not match canonical fields")]
    BudgetIdentityMismatch,
    #[error("distributed change budget aggregate subject does not match exact local subject")]
    AggregateSubjectMismatch,
}

fn validate_availability_budget(
    participants: u32,
    max_unavailable: u32,
    minimum_healthy: u32,
) -> Result<(), DistributedChangeBudgetError> {
    if max_unavailable > participants {
        return Err(DistributedChangeBudgetError::MaxUnavailableExceedsParticipants {
            requested: max_unavailable,
            participants,
        });
    }
    if minimum_healthy > participants {
        return Err(DistributedChangeBudgetError::MinimumHealthyExceedsParticipants {
            requested: minimum_healthy,
            participants,
        });
    }
    if participants - max_unavailable < minimum_healthy {
        return Err(DistributedChangeBudgetError::ContradictoryAvailabilityBudget {
            participants,
            max_unavailable,
            minimum_healthy,
        });
    }
    Ok(())
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, DistributedChangeBudgetError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(DistributedChangeBudgetError::BlankText { field });
    }
    if trimmed.len() > MAX_TEXT_BYTES {
        return Err(DistributedChangeBudgetError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(DistributedChangeBudgetError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_budget(
    aggregate_subject_id: ContinuitySubjectId,
    generation: u64,
    participants: &[ContinuitySubjectId],
    max_concurrent_unavailable: u32,
    minimum_healthy: u32,
    exclusion_sets: &[MutualExclusionSetV1],
    recovery_paths: &[RecoveryPathClassV1],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(aggregate_subject_id.as_bytes());
    bytes.extend_from_slice(&generation.to_le_bytes());
    put_len(&mut bytes, participants.len());
    for participant in participants {
        bytes.extend_from_slice(participant.as_bytes());
    }
    bytes.extend_from_slice(&max_concurrent_unavailable.to_le_bytes());
    bytes.extend_from_slice(&minimum_healthy.to_le_bytes());
    put_len(&mut bytes, exclusion_sets.len());
    for set in exclusion_sets {
        set.encode(&mut bytes);
    }
    put_len(&mut bytes, recovery_paths.len());
    for path in recovery_paths {
        path.encode(&mut bytes);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(BUDGET_DOMAIN);
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
    use crate::scope::ContinuityScopeV1;

    fn subject(logical_id: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", logical_id, scope, None).unwrap()
    }

    fn participants(count: usize) -> Vec<ContinuitySubjectV1> {
        (0..count)
            .map(|index| subject(&format!("node-{index}"), ContinuityScopeV1::Machine))
            .collect()
    }

    fn ids(subjects: &[ContinuitySubjectV1]) -> Vec<ContinuitySubjectId> {
        subjects.iter().map(ContinuitySubjectV1::id).collect()
    }

    #[test]
    fn input_order_does_not_change_budget_identity() {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let nodes = participants(3);
        let mut reversed = ids(&nodes);
        reversed.reverse();

        let a = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            ids(&nodes),
            1,
            2,
            vec![MutualExclusionSetV1::new(vec![nodes[0].id(), nodes[1].id()]).unwrap()],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let b = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            reversed,
            1,
            2,
            vec![MutualExclusionSetV1::new(vec![nodes[1].id(), nodes[0].id()]).unwrap()],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();

        assert_eq!(a.id(), b.id());
        a.validate_against_subject(&aggregate).unwrap();
    }

    #[test]
    fn contradictory_availability_budget_fails_closed() {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let nodes = participants(3);
        assert!(matches!(
            DistributedChangeBudgetV1::new(
                &aggregate,
                1,
                ids(&nodes),
                2,
                2,
                vec![],
                vec![RecoveryPathClassV1::OutOfBandManagement],
            ),
            Err(DistributedChangeBudgetError::ContradictoryAvailabilityBudget { .. })
        ));
    }

    #[test]
    fn explicit_full_outage_budget_is_representable() {
        let aggregate = subject("batch-service", ContinuityScopeV1::Service);
        let nodes = participants(2);
        let budget = DistributedChangeBudgetV1::new(
            &aggregate,
            1,
            ids(&nodes),
            2,
            0,
            vec![],
            vec![RecoveryPathClassV1::LocalPhysicalIntervention],
        )
        .unwrap();

        assert_eq!(budget.max_concurrent_unavailable(), 2);
        assert_eq!(budget.minimum_healthy(), 0);
    }

    #[test]
    fn aggregate_cannot_be_smuggled_into_participant_set() {
        let aggregate = subject("fabric-a", ContinuityScopeV1::NetworkFabric);
        let nodes = participants(2);
        let mut member_ids = ids(&nodes);
        member_ids.push(aggregate.id());
        assert_eq!(
            DistributedChangeBudgetV1::new(
                &aggregate,
                1,
                member_ids,
                1,
                1,
                vec![],
                vec![RecoveryPathClassV1::DeviceLocalAutomaticRollback],
            )
            .unwrap_err(),
            DistributedChangeBudgetError::AggregateIncludedAsParticipant,
        );
    }

    #[test]
    fn exclusion_set_cannot_reference_nonparticipant() {
        let aggregate = subject("fabric-a", ContinuityScopeV1::NetworkFabric);
        let nodes = participants(2);
        let outsider = subject("outsider", ContinuityScopeV1::NetworkDevice);
        let set = MutualExclusionSetV1::new(vec![nodes[0].id(), outsider.id()]).unwrap();
        assert!(matches!(
            DistributedChangeBudgetV1::new(
                &aggregate,
                1,
                ids(&nodes),
                1,
                1,
                vec![set],
                vec![RecoveryPathClassV1::OutOfBandManagement],
            ),
            Err(DistributedChangeBudgetError::ExclusionMemberOutsideParticipants { .. })
        ));
    }

    #[test]
    fn at_least_one_recovery_class_is_required() {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let nodes = participants(3);
        assert_eq!(
            DistributedChangeBudgetV1::new(
                &aggregate,
                1,
                ids(&nodes),
                1,
                2,
                vec![],
                vec![],
            )
            .unwrap_err(),
            DistributedChangeBudgetError::NoRecoveryPathClass,
        );
    }

    #[test]
    fn exact_aggregate_subject_rebinding_is_required() {
        let aggregate_a = ContinuitySubjectV1::new(
            "org.a",
            "cluster",
            ContinuityScopeV1::Cluster,
            None,
        )
        .unwrap();
        let aggregate_b = ContinuitySubjectV1::new(
            "org.b",
            "cluster",
            ContinuityScopeV1::Cluster,
            None,
        )
        .unwrap();
        let nodes = participants(3);
        let budget = DistributedChangeBudgetV1::new(
            &aggregate_a,
            1,
            ids(&nodes),
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();

        assert_eq!(
            budget.validate_against_subject(&aggregate_b).unwrap_err(),
            DistributedChangeBudgetError::AggregateSubjectMismatch,
        );
    }

    #[test]
    fn scope_generation_and_budget_are_identity_bearing() {
        let cluster = subject("edge", ContinuityScopeV1::Cluster);
        let fabric = subject("edge", ContinuityScopeV1::NetworkFabric);
        let nodes = participants(3);

        let make = |aggregate: &ContinuitySubjectV1, generation, max_unavailable, minimum_healthy| {
            DistributedChangeBudgetV1::new(
                aggregate,
                generation,
                ids(&nodes),
                max_unavailable,
                minimum_healthy,
                vec![],
                vec![RecoveryPathClassV1::OutOfBandManagement],
            )
            .unwrap()
        };

        let base = make(&cluster, 1, 1, 2);
        assert_ne!(base.id(), make(&fabric, 1, 1, 2).id());
        assert_ne!(base.id(), make(&cluster, 2, 1, 2).id());
        assert_ne!(base.id(), make(&cluster, 1, 0, 3).id());
    }

    #[test]
    fn noncanonical_transport_participant_order_is_rejected() {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let nodes = participants(3);
        let mut budget = DistributedChangeBudgetV1::new(
            &aggregate,
            1,
            ids(&nodes),
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        budget.participant_subject_ids.reverse();

        assert_eq!(
            budget.validate().unwrap_err(),
            DistributedChangeBudgetError::NonCanonicalParticipants,
        );
    }
}
