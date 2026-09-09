// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-neutral failure-domain policy for distributed continuity.
//!
//! A distributed change budget constrains how many exact participants may be
//! unavailable. It does not say whether those participants share a rack, chassis,
//! power feed, availability zone, region, or network plane. This module adds that
//! independent structural policy without claiming the declared topology is current.
//!
//! Core theorem:
//!
//! `FailureDomainPolicyV1 != ObservedFailureDomains != DistributedTransitionWitness != ExecutionAuthority`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, ValidatedDistributedChangeBudgetV1};
use crate::scope::ContinuitySubjectId;

pub const FAILURE_DOMAIN_POLICY_SCHEMA_V1: &str = "symthaea-continuity-failure-domain-policy-v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.failure-domain-policy.v1\0";
const MAX_GROUPS: usize = 4096;
const MAX_TEXT_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FailureDomainPolicyId([u8; 32]);

impl FailureDomainPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// One independently named failure-domain layer.
///
/// Multiple policies may describe the same participant world through different
/// layers, for example rack placement and power-feed placement. The kernel does
/// not invent a universal hierarchy between those layers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FailureDomainKindV1 {
    Rack,
    Chassis,
    AvailabilityZone,
    Region,
    PowerDomain,
    NetworkPlane,
    Custom { kind_id: String },
}

impl FailureDomainKindV1 {
    pub fn custom(kind_id: impl Into<String>) -> Result<Self, FailureDomainPolicyError> {
        Ok(Self::Custom {
            kind_id: checked_text("custom failure-domain kind_id", kind_id.into())?,
        })
    }

    fn normalize(self) -> Result<Self, FailureDomainPolicyError> {
        match self {
            Self::Custom { kind_id } => Self::custom(kind_id),
            other => Ok(other),
        }
    }

    fn validate(&self) -> Result<(), FailureDomainPolicyError> {
        if let Self::Custom { kind_id } = self {
            let canonical = checked_text("custom failure-domain kind_id", kind_id.clone())?;
            if canonical != *kind_id {
                return Err(FailureDomainPolicyError::NonCanonicalText {
                    field: "custom failure-domain kind_id",
                });
            }
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Rack => out.push(1),
            Self::Chassis => out.push(2),
            Self::AvailabilityZone => out.push(3),
            Self::Region => out.push(4),
            Self::PowerDomain => out.push(5),
            Self::NetworkPlane => out.push(6),
            Self::Custom { kind_id } => {
                out.push(255);
                put_str(out, kind_id);
            }
        }
    }
}

/// Canonical partition group within one failure-domain layer.
///
/// `group_id` is stable naming material only. It does not prove that the exact
/// participants currently occupy this domain; later authenticated observation must
/// establish that independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDomainGroupV1 {
    group_id: String,
    members: Vec<ContinuitySubjectId>,
}

impl FailureDomainGroupV1 {
    pub fn new(
        group_id: impl Into<String>,
        mut members: Vec<ContinuitySubjectId>,
    ) -> Result<Self, FailureDomainPolicyError> {
        let group_id = checked_text("failure-domain group_id", group_id.into())?;
        if members.is_empty() {
            return Err(FailureDomainPolicyError::EmptyGroup);
        }
        members.sort();
        members.dedup();
        Ok(Self { group_id, members })
    }

    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    pub fn members(&self) -> &[ContinuitySubjectId] {
        &self.members
    }

    fn validate(&self) -> Result<(), FailureDomainPolicyError> {
        let canonical = checked_text("failure-domain group_id", self.group_id.clone())?;
        if canonical != self.group_id {
            return Err(FailureDomainPolicyError::NonCanonicalText {
                field: "failure-domain group_id",
            });
        }
        if self.members.is_empty() {
            return Err(FailureDomainPolicyError::EmptyGroup);
        }
        if self.members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(FailureDomainPolicyError::NonCanonicalGroupMembers);
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        put_str(out, &self.group_id);
        put_len(out, self.members.len());
        for member in &self.members {
            out.extend_from_slice(member.as_bytes());
        }
    }
}

/// Serializable policy for one exact failure-domain partition of a distributed
/// change budget.
///
/// Every budget participant must occur in exactly one group. Requiring a complete
/// partition prevents an omitted/unknown participant from disappearing from the
/// failure-domain safety calculation. Different physical/logical dimensions are
/// represented as separate policies, not by overlapping groups in one layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDomainPolicyV1 {
    schema_version: String,
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    layer_kind: FailureDomainKindV1,
    layer_id: String,
    groups: Vec<FailureDomainGroupV1>,
    minimum_healthy_domains: u32,
    policy_id: FailureDomainPolicyId,
}

impl FailureDomainPolicyV1 {
    pub fn new(
        budget: &ValidatedDistributedChangeBudgetV1,
        layer_kind: FailureDomainKindV1,
        layer_id: impl Into<String>,
        mut groups: Vec<FailureDomainGroupV1>,
        minimum_healthy_domains: u32,
    ) -> Result<Self, FailureDomainPolicyError> {
        let layer_kind = layer_kind.normalize()?;
        let layer_id = checked_text("failure-domain layer_id", layer_id.into())?;
        canonicalize_groups(&mut groups)?;
        validate_groups_against_budget(&groups, budget)?;
        validate_minimum_healthy_domains(
            minimum_healthy_domains,
            groups.len(),
            budget.minimum_healthy(),
        )?;

        let aggregate_subject_id = budget.aggregate_subject_id();
        let budget_id = budget.id();
        let budget_generation = budget.generation();
        let policy_id = FailureDomainPolicyId(hash_policy(
            aggregate_subject_id,
            budget_id,
            budget_generation,
            &layer_kind,
            &layer_id,
            &groups,
            minimum_healthy_domains,
        ));

        Ok(Self {
            schema_version: FAILURE_DOMAIN_POLICY_SCHEMA_V1.to_owned(),
            aggregate_subject_id,
            budget_id,
            budget_generation,
            layer_kind,
            layer_id,
            groups,
            minimum_healthy_domains,
            policy_id,
        })
    }

    /// Intrinsic validation of canonical transport structure and stored identity.
    /// Budget membership/currentness is re-established by `validate_against_budget`.
    pub fn validate(&self) -> Result<(), FailureDomainPolicyError> {
        if self.schema_version != FAILURE_DOMAIN_POLICY_SCHEMA_V1 {
            return Err(FailureDomainPolicyError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.budget_generation == 0 {
            return Err(FailureDomainPolicyError::ZeroBudgetGeneration);
        }
        self.layer_kind.validate()?;
        let canonical = checked_text("failure-domain layer_id", self.layer_id.clone())?;
        if canonical != self.layer_id {
            return Err(FailureDomainPolicyError::NonCanonicalText {
                field: "failure-domain layer_id",
            });
        }
        validate_canonical_groups(&self.groups)?;
        if self.minimum_healthy_domains > self.groups.len() as u32 {
            return Err(FailureDomainPolicyError::MinimumHealthyDomainsExceedsGroups {
                requested: self.minimum_healthy_domains,
                groups: self.groups.len() as u32,
            });
        }

        let expected = FailureDomainPolicyId(hash_policy(
            self.aggregate_subject_id,
            self.budget_id,
            self.budget_generation,
            &self.layer_kind,
            &self.layer_id,
            &self.groups,
            self.minimum_healthy_domains,
        ));
        if expected != self.policy_id {
            return Err(FailureDomainPolicyError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    /// Re-bind this serialized policy to the exact validated distributed budget.
    ///
    /// The returned non-Serde wrapper proves structural agreement only. It does not
    /// prove that the declared domains match current physical/logical placement.
    pub fn validate_against_budget(
        &self,
        budget: &ValidatedDistributedChangeBudgetV1,
    ) -> Result<ValidatedFailureDomainPolicyV1, FailureDomainPolicyError> {
        self.validate()?;
        if self.aggregate_subject_id != budget.aggregate_subject_id() {
            return Err(FailureDomainPolicyError::AggregateSubjectMismatch);
        }
        if self.budget_id != budget.id() {
            return Err(FailureDomainPolicyError::BudgetIdentityMismatch);
        }
        if self.budget_generation != budget.generation() {
            return Err(FailureDomainPolicyError::BudgetGenerationMismatch);
        }
        validate_groups_against_budget(&self.groups, budget)?;
        validate_minimum_healthy_domains(
            self.minimum_healthy_domains,
            self.groups.len(),
            budget.minimum_healthy(),
        )?;

        Ok(ValidatedFailureDomainPolicyV1 {
            budget: budget.clone(),
            inner: self.clone(),
        })
    }

    pub fn id(&self) -> FailureDomainPolicyId {
        self.policy_id
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.aggregate_subject_id
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.budget_id
    }

    pub fn budget_generation(&self) -> u64 {
        self.budget_generation
    }

    pub fn layer_kind(&self) -> &FailureDomainKindV1 {
        &self.layer_kind
    }

    pub fn layer_id(&self) -> &str {
        &self.layer_id
    }

    pub fn groups(&self) -> &[FailureDomainGroupV1] {
        &self.groups
    }

    pub fn minimum_healthy_domains(&self) -> u32 {
        self.minimum_healthy_domains
    }
}

/// Non-Serde failure-domain policy bound to one exact validated distributed budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedFailureDomainPolicyV1 {
    budget: ValidatedDistributedChangeBudgetV1,
    inner: FailureDomainPolicyV1,
}

impl ValidatedFailureDomainPolicyV1 {
    pub fn id(&self) -> FailureDomainPolicyId {
        self.inner.id()
    }

    pub fn budget(&self) -> &ValidatedDistributedChangeBudgetV1 {
        &self.budget
    }

    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId {
        self.inner.aggregate_subject_id()
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.inner.budget_id()
    }

    pub fn budget_generation(&self) -> u64 {
        self.inner.budget_generation()
    }

    pub fn layer_kind(&self) -> &FailureDomainKindV1 {
        self.inner.layer_kind()
    }

    pub fn layer_id(&self) -> &str {
        self.inner.layer_id()
    }

    pub fn groups(&self) -> &[FailureDomainGroupV1] {
        self.inner.groups()
    }

    pub fn minimum_healthy_domains(&self) -> u32 {
        self.inner.minimum_healthy_domains()
    }

    pub fn as_raw(&self) -> &FailureDomainPolicyV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FailureDomainPolicyError {
    #[error("unsupported failure-domain policy schema: {0}")]
    UnsupportedSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds {MAX_TEXT_BYTES} bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("{field} is not canonically trimmed")]
    NonCanonicalText { field: &'static str },
    #[error("failure-domain policy budget generation must be non-zero")]
    ZeroBudgetGeneration,
    #[error("failure-domain policy must contain at least one group")]
    NoGroups,
    #[error("failure-domain policy exceeds the group bound")]
    TooManyGroups,
    #[error("failure-domain group must contain at least one participant")]
    EmptyGroup,
    #[error("failure-domain group members must be in canonical sorted unique order")]
    NonCanonicalGroupMembers,
    #[error("failure-domain groups must be in canonical group_id order")]
    NonCanonicalGroups,
    #[error("duplicate failure-domain group_id: {group_id}")]
    DuplicateGroupId { group_id: String },
    #[error("participant appears in more than one group: {member:?}")]
    DuplicateParticipantMembership { member: ContinuitySubjectId },
    #[error("failure-domain member is outside the distributed budget: {member:?}")]
    MemberOutsideBudget { member: ContinuitySubjectId },
    #[error("failure-domain policy does not cover every distributed-budget participant")]
    IncompleteParticipantCoverage,
    #[error("minimum_healthy_domains {requested} exceeds failure-domain group count {groups}")]
    MinimumHealthyDomainsExceedsGroups { requested: u32, groups: u32 },
    #[error("non-zero distributed minimum_healthy requires at least one healthy failure domain")]
    ZeroHealthyDomainsForNonzeroHealthyBudget,
    #[error("failure-domain policy aggregate subject does not match validated budget")]
    AggregateSubjectMismatch,
    #[error("failure-domain policy references a different distributed budget")]
    BudgetIdentityMismatch,
    #[error("failure-domain policy references a different distributed budget generation")]
    BudgetGenerationMismatch,
    #[error("stored failure-domain policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
}

fn canonicalize_groups(groups: &mut Vec<FailureDomainGroupV1>) -> Result<(), FailureDomainPolicyError> {
    if groups.is_empty() {
        return Err(FailureDomainPolicyError::NoGroups);
    }
    if groups.len() > MAX_GROUPS {
        return Err(FailureDomainPolicyError::TooManyGroups);
    }
    for group in groups.iter() {
        group.validate()?;
    }
    groups.sort_by(|a, b| a.group_id.cmp(&b.group_id));
    if let Some(pair) = groups.windows(2).find(|pair| pair[0].group_id == pair[1].group_id) {
        return Err(FailureDomainPolicyError::DuplicateGroupId {
            group_id: pair[0].group_id.clone(),
        });
    }
    Ok(())
}

fn validate_canonical_groups(groups: &[FailureDomainGroupV1]) -> Result<(), FailureDomainPolicyError> {
    if groups.is_empty() {
        return Err(FailureDomainPolicyError::NoGroups);
    }
    if groups.len() > MAX_GROUPS {
        return Err(FailureDomainPolicyError::TooManyGroups);
    }
    for group in groups {
        group.validate()?;
    }
    for pair in groups.windows(2) {
        if pair[0].group_id == pair[1].group_id {
            return Err(FailureDomainPolicyError::DuplicateGroupId {
                group_id: pair[0].group_id.clone(),
            });
        }
        if pair[0].group_id > pair[1].group_id {
            return Err(FailureDomainPolicyError::NonCanonicalGroups);
        }
    }
    Ok(())
}

fn validate_groups_against_budget(
    groups: &[FailureDomainGroupV1],
    budget: &ValidatedDistributedChangeBudgetV1,
) -> Result<(), FailureDomainPolicyError> {
    let mut observed = BTreeSet::new();
    for group in groups {
        for member in group.members() {
            if budget.participant_subject_ids().binary_search(member).is_err() {
                return Err(FailureDomainPolicyError::MemberOutsideBudget { member: *member });
            }
            if !observed.insert(*member) {
                return Err(FailureDomainPolicyError::DuplicateParticipantMembership {
                    member: *member,
                });
            }
        }
    }
    if observed.len() != budget.participant_subject_ids().len() {
        return Err(FailureDomainPolicyError::IncompleteParticipantCoverage);
    }
    Ok(())
}

fn validate_minimum_healthy_domains(
    minimum_healthy_domains: u32,
    group_count: usize,
    minimum_healthy_participants: u32,
) -> Result<(), FailureDomainPolicyError> {
    if minimum_healthy_domains > group_count as u32 {
        return Err(FailureDomainPolicyError::MinimumHealthyDomainsExceedsGroups {
            requested: minimum_healthy_domains,
            groups: group_count as u32,
        });
    }
    if minimum_healthy_participants > 0 && minimum_healthy_domains == 0 {
        return Err(FailureDomainPolicyError::ZeroHealthyDomainsForNonzeroHealthyBudget);
    }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, FailureDomainPolicyError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(FailureDomainPolicyError::BlankText { field });
    }
    if trimmed.len() > MAX_TEXT_BYTES {
        return Err(FailureDomainPolicyError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(FailureDomainPolicyError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

#[allow(clippy::too_many_arguments)]
fn hash_policy(
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    layer_kind: &FailureDomainKindV1,
    layer_id: &str,
    groups: &[FailureDomainGroupV1],
    minimum_healthy_domains: u32,
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(aggregate_subject_id.as_bytes());
    bytes.extend_from_slice(budget_id.as_bytes());
    bytes.extend_from_slice(&budget_generation.to_le_bytes());
    layer_kind.encode(&mut bytes);
    put_str(&mut bytes, layer_id);
    put_len(&mut bytes, groups.len());
    for group in groups {
        group.encode(&mut bytes);
    }
    bytes.extend_from_slice(&minimum_healthy_domains.to_le_bytes());
    let mut hasher = blake3::Hasher::new();
    hasher.update(POLICY_DOMAIN);
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
    use crate::distributed::{DistributedChangeBudgetV1, RecoveryPathClassV1};
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};

    fn subject(logical_id: &str, scope: ContinuityScopeV1) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new("org.example", logical_id, scope, None).unwrap()
    }

    fn fixture_budget(
    ) -> (
        ContinuitySubjectV1,
        Vec<ContinuitySubjectV1>,
        ValidatedDistributedChangeBudgetV1,
    ) {
        let aggregate = subject("cluster-a", ContinuityScopeV1::Cluster);
        let participants = vec![
            subject("node-a", ContinuityScopeV1::Machine),
            subject("node-b", ContinuityScopeV1::Machine),
            subject("node-c", ContinuityScopeV1::Machine),
        ];
        let raw = DistributedChangeBudgetV1::new(
            &aggregate,
            7,
            participants.iter().map(ContinuitySubjectV1::id).collect(),
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let validated = raw.validate_against_subject(&aggregate).unwrap();
        (aggregate, participants, validated)
    }

    fn groups(participants: &[ContinuitySubjectV1]) -> Vec<FailureDomainGroupV1> {
        vec![
            FailureDomainGroupV1::new("rack-b", vec![participants[2].id()]).unwrap(),
            FailureDomainGroupV1::new(
                "rack-a",
                vec![participants[1].id(), participants[0].id()],
            )
            .unwrap(),
        ]
    }

    #[test]
    fn constructor_canonicalizes_group_and_member_order() {
        let (_, participants, budget) = fixture_budget();
        let policy = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "primary-rack-placement",
            groups(&participants),
            1,
        )
        .unwrap();

        assert_eq!(policy.groups()[0].group_id(), "rack-a");
        assert_eq!(policy.groups()[1].group_id(), "rack-b");
        assert!(policy.groups()[0].members()[0] < policy.groups()[0].members()[1]);
        policy.validate_against_budget(&budget).unwrap();
    }

    #[test]
    fn participant_cannot_appear_in_two_domains_of_same_layer() {
        let (_, participants, budget) = fixture_budget();
        let overlapping = vec![
            FailureDomainGroupV1::new(
                "rack-a",
                vec![participants[0].id(), participants[1].id()],
            )
            .unwrap(),
            FailureDomainGroupV1::new(
                "rack-b",
                vec![participants[1].id(), participants[2].id()],
            )
            .unwrap(),
        ];
        assert!(matches!(
            FailureDomainPolicyV1::new(
                &budget,
                FailureDomainKindV1::Rack,
                "rack-placement",
                overlapping,
                1,
            ),
            Err(FailureDomainPolicyError::DuplicateParticipantMembership { .. })
        ));
    }

    #[test]
    fn incomplete_participant_coverage_fails_closed() {
        let (_, participants, budget) = fixture_budget();
        let incomplete = vec![
            FailureDomainGroupV1::new("rack-a", vec![participants[0].id()]).unwrap(),
            FailureDomainGroupV1::new("rack-b", vec![participants[1].id()]).unwrap(),
        ];
        assert_eq!(
            FailureDomainPolicyV1::new(
                &budget,
                FailureDomainKindV1::Rack,
                "rack-placement",
                incomplete,
                1,
            )
            .unwrap_err(),
            FailureDomainPolicyError::IncompleteParticipantCoverage
        );
    }

    #[test]
    fn participant_outside_budget_fails_closed() {
        let (_, participants, budget) = fixture_budget();
        let outsider = subject("node-x", ContinuityScopeV1::Machine);
        let invalid = vec![
            FailureDomainGroupV1::new(
                "rack-a",
                vec![participants[0].id(), participants[1].id()],
            )
            .unwrap(),
            FailureDomainGroupV1::new(
                "rack-b",
                vec![participants[2].id(), outsider.id()],
            )
            .unwrap(),
        ];
        assert!(matches!(
            FailureDomainPolicyV1::new(
                &budget,
                FailureDomainKindV1::Rack,
                "rack-placement",
                invalid,
                1,
            ),
            Err(FailureDomainPolicyError::MemberOutsideBudget { .. })
        ));
    }

    #[test]
    fn minimum_healthy_domains_cannot_exceed_domain_count() {
        let (_, participants, budget) = fixture_budget();
        assert!(matches!(
            FailureDomainPolicyV1::new(
                &budget,
                FailureDomainKindV1::Rack,
                "rack-placement",
                groups(&participants),
                3,
            ),
            Err(FailureDomainPolicyError::MinimumHealthyDomainsExceedsGroups { .. })
        ));
    }

    #[test]
    fn nonzero_healthy_budget_requires_nonzero_domain_floor() {
        let (_, participants, budget) = fixture_budget();
        assert_eq!(
            FailureDomainPolicyV1::new(
                &budget,
                FailureDomainKindV1::Rack,
                "rack-placement",
                groups(&participants),
                0,
            )
            .unwrap_err(),
            FailureDomainPolicyError::ZeroHealthyDomainsForNonzeroHealthyBudget
        );
    }

    #[test]
    fn different_failure_domain_layers_have_distinct_identity() {
        let (_, participants, budget) = fixture_budget();
        let rack = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "physical-placement",
            groups(&participants),
            1,
        )
        .unwrap();
        let power = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::PowerDomain,
            "physical-placement",
            groups(&participants),
            1,
        )
        .unwrap();
        assert_ne!(rack.id(), power.id());
    }

    #[test]
    fn regrouping_same_participants_changes_policy_identity() {
        let (_, participants, budget) = fixture_budget();
        let a = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            groups(&participants),
            1,
        )
        .unwrap();
        let b = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            vec![
                FailureDomainGroupV1::new(
                    "rack-a",
                    vec![participants[0].id(), participants[2].id()],
                )
                .unwrap(),
                FailureDomainGroupV1::new("rack-b", vec![participants[1].id()]).unwrap(),
            ],
            1,
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn exact_budget_identity_is_required_on_rebind() {
        let (aggregate, participants, budget) = fixture_budget();
        let policy = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            groups(&participants),
            1,
        )
        .unwrap();
        let different_raw = DistributedChangeBudgetV1::new(
            &aggregate,
            8,
            participants.iter().map(ContinuitySubjectV1::id).collect(),
            1,
            2,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        let different = different_raw.validate_against_subject(&aggregate).unwrap();
        assert_eq!(
            policy.validate_against_budget(&different).unwrap_err(),
            FailureDomainPolicyError::BudgetIdentityMismatch
        );
    }

    #[test]
    fn noncanonical_transport_group_order_is_rejected() {
        let (_, participants, budget) = fixture_budget();
        let mut policy = FailureDomainPolicyV1::new(
            &budget,
            FailureDomainKindV1::Rack,
            "rack-placement",
            groups(&participants),
            1,
        )
        .unwrap();
        policy.groups.reverse();
        assert_eq!(
            policy.validate().unwrap_err(),
            FailureDomainPolicyError::NonCanonicalGroups
        );
    }
}
