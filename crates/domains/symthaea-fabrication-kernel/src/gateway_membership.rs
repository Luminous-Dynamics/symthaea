// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical gateway membership and threshold-authorized roster rotation.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const GATEWAY_MEMBERSHIP_SCHEMA: &str = "symthaea.fabrication.gateway-membership.v1";
pub const GATEWAY_MEMBERSHIP_TRANSITION_SCHEMA: &str =
    "symthaea.fabrication.gateway-membership-transition.v1";
pub const MAX_GATEWAY_MEMBERS: usize = 256;
pub const MAX_FAILURE_DOMAIN_BYTES: usize = 128;
pub const MAX_MEMBERSHIP_REASON_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayMember {
    pub gateway_id: String,
    pub voting_weight: u16,
    pub failure_domain: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayMembership {
    pub schema_version: String,
    pub epoch: u64,
    pub activated_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub members: Vec<GatewayMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayMembershipTransition {
    pub schema_version: String,
    pub current_membership_digest: Sha256Digest,
    pub proposed_membership: GatewayMembership,
    pub proposed_membership_digest: Sha256Digest,
    pub activates_at_unix_s: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayMembershipPolicy {
    pub minimum_members: usize,
    pub minimum_total_voting_weight: u32,
    pub minimum_failure_domains: usize,
    pub minimum_retained_voting_weight: u32,
    pub maximum_removed_weight_basis_points: u16,
}

impl Default for GatewayMembershipPolicy {
    fn default() -> Self {
        Self {
            minimum_members: 3,
            minimum_total_voting_weight: 3,
            minimum_failure_domains: 2,
            minimum_retained_voting_weight: 2,
            maximum_removed_weight_basis_points: 5_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayMembershipError {
    UnsupportedSchema,
    EpochZero,
    InvalidWindow,
    EmptyMembership,
    TooManyMembers { actual: usize, maximum: usize },
    InvalidGatewayId(String),
    InvalidFailureDomain(String),
    ZeroVotingWeight(String),
    DuplicateGateway(String),
    NonCanonicalOrder,
    InvalidPolicy,
    EpochNotSuccessor { current: u64, proposed: u64 },
    ActivationBeforeCurrent,
    ActivationOutsideProposedWindow,
    InvalidReason,
    CurrentDigestMismatch,
    ProposedDigestMismatch,
    InsufficientMembers { actual: usize, required: usize },
    InsufficientVotingWeight { actual: u32, required: u32 },
    InsufficientFailureDomains { actual: usize, required: usize },
    InsufficientRetainedWeight { actual: u32, required: u32 },
    ExcessiveRemovedWeight { basis_points: u16, maximum: u16 },
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedGatewayMembership {
    transition: GatewayMembershipTransition,
    transition_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedGatewayMembership {
    pub fn transition(&self) -> &GatewayMembershipTransition {
        &self.transition
    }
    pub fn transition_digest(&self) -> Sha256Digest {
        self.transition_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
    pub fn proposed_membership(&self) -> &GatewayMembership {
        &self.transition.proposed_membership
    }
}

impl GatewayMembership {
    pub fn new(
        epoch: u64,
        activated_at_unix_s: u64,
        expires_at_unix_s: u64,
        mut members: Vec<GatewayMember>,
    ) -> Result<Self, GatewayMembershipError> {
        members.sort_by(|left, right| left.gateway_id.cmp(&right.gateway_id));
        let membership = Self {
            schema_version: GATEWAY_MEMBERSHIP_SCHEMA.into(),
            epoch,
            activated_at_unix_s,
            expires_at_unix_s,
            members,
        };
        membership.validate()?;
        Ok(membership)
    }

    pub fn validate(&self) -> Result<(), GatewayMembershipError> {
        if self.schema_version != GATEWAY_MEMBERSHIP_SCHEMA {
            return Err(GatewayMembershipError::UnsupportedSchema);
        }
        if self.epoch == 0 {
            return Err(GatewayMembershipError::EpochZero);
        }
        if self.activated_at_unix_s >= self.expires_at_unix_s {
            return Err(GatewayMembershipError::InvalidWindow);
        }
        if self.members.is_empty() {
            return Err(GatewayMembershipError::EmptyMembership);
        }
        if self.members.len() > MAX_GATEWAY_MEMBERS {
            return Err(GatewayMembershipError::TooManyMembers {
                actual: self.members.len(),
                maximum: MAX_GATEWAY_MEMBERS,
            });
        }
        let mut previous: Option<&str> = None;
        let mut identities = BTreeSet::new();
        for member in &self.members {
            validate_identifier(&member.gateway_id)
                .map_err(|_| GatewayMembershipError::InvalidGatewayId(member.gateway_id.clone()))?;
            validate_failure_domain(&member.failure_domain)?;
            if member.voting_weight == 0 {
                return Err(GatewayMembershipError::ZeroVotingWeight(
                    member.gateway_id.clone(),
                ));
            }
            if !identities.insert(member.gateway_id.clone()) {
                return Err(GatewayMembershipError::DuplicateGateway(
                    member.gateway_id.clone(),
                ));
            }
            if previous.is_some_and(|value| value >= member.gateway_id.as_str()) {
                return Err(GatewayMembershipError::NonCanonicalOrder);
            }
            previous = Some(&member.gateway_id);
        }
        Ok(())
    }

    pub fn is_active_at(&self, unix_s: u64) -> bool {
        unix_s >= self.activated_at_unix_s && unix_s < self.expires_at_unix_s
    }

    pub fn total_voting_weight(&self) -> u32 {
        self.members
            .iter()
            .map(|member| u32::from(member.voting_weight))
            .sum()
    }

    pub fn member(&self, gateway_id: &str) -> Option<&GatewayMember> {
        self.members
            .iter()
            .find(|member| member.gateway_id == gateway_id)
    }
}

pub fn digest_gateway_membership(
    membership: &GatewayMembership,
) -> Result<Sha256Digest, GatewayMembershipError> {
    membership.validate()?;
    let bytes = serde_json::to_vec(membership)
        .map_err(|error| GatewayMembershipError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-membership-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn build_membership_transition(
    current: &GatewayMembership,
    proposed: GatewayMembership,
    activates_at_unix_s: u64,
    reason: impl Into<String>,
) -> Result<GatewayMembershipTransition, GatewayMembershipError> {
    current.validate()?;
    proposed.validate()?;
    let transition = GatewayMembershipTransition {
        schema_version: GATEWAY_MEMBERSHIP_TRANSITION_SCHEMA.into(),
        current_membership_digest: digest_gateway_membership(current)?,
        proposed_membership_digest: digest_gateway_membership(&proposed)?,
        proposed_membership: proposed,
        activates_at_unix_s,
        reason: reason.into(),
    };
    transition.validate_against(current)?;
    Ok(transition)
}

impl GatewayMembershipTransition {
    pub fn validate_against(
        &self,
        current: &GatewayMembership,
    ) -> Result<(), GatewayMembershipError> {
        if self.schema_version != GATEWAY_MEMBERSHIP_TRANSITION_SCHEMA {
            return Err(GatewayMembershipError::UnsupportedSchema);
        }
        current.validate()?;
        self.proposed_membership.validate()?;
        if self.current_membership_digest != digest_gateway_membership(current)? {
            return Err(GatewayMembershipError::CurrentDigestMismatch);
        }
        if self.proposed_membership_digest != digest_gateway_membership(&self.proposed_membership)?
        {
            return Err(GatewayMembershipError::ProposedDigestMismatch);
        }
        if self.proposed_membership.epoch != current.epoch.saturating_add(1) {
            return Err(GatewayMembershipError::EpochNotSuccessor {
                current: current.epoch,
                proposed: self.proposed_membership.epoch,
            });
        }
        if self.activates_at_unix_s < current.activated_at_unix_s {
            return Err(GatewayMembershipError::ActivationBeforeCurrent);
        }
        if !self
            .proposed_membership
            .is_active_at(self.activates_at_unix_s)
        {
            return Err(GatewayMembershipError::ActivationOutsideProposedWindow);
        }
        if self.reason.trim().is_empty()
            || self.reason != self.reason.trim()
            || self.reason.len() > MAX_MEMBERSHIP_REASON_BYTES
        {
            return Err(GatewayMembershipError::InvalidReason);
        }
        Ok(())
    }
}

pub fn digest_membership_transition(
    transition: &GatewayMembershipTransition,
    current: &GatewayMembership,
) -> Result<Sha256Digest, GatewayMembershipError> {
    transition.validate_against(current)?;
    let bytes = serde_json::to_vec(transition)
        .map_err(|error| GatewayMembershipError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-membership-transition-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_membership_transition(
    current: &GatewayMembership,
    transition: GatewayMembershipTransition,
    policy: &GatewayMembershipPolicy,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedGatewayMembership, GatewayMembershipError> {
    transition.validate_against(current)?;
    validate_membership_policy(policy)?;
    let proposed = &transition.proposed_membership;
    if proposed.members.len() < policy.minimum_members {
        return Err(GatewayMembershipError::InsufficientMembers {
            actual: proposed.members.len(),
            required: policy.minimum_members,
        });
    }
    let total = proposed.total_voting_weight();
    if total < policy.minimum_total_voting_weight {
        return Err(GatewayMembershipError::InsufficientVotingWeight {
            actual: total,
            required: policy.minimum_total_voting_weight,
        });
    }
    let failure_domains = proposed
        .members
        .iter()
        .map(|member| member.failure_domain.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if failure_domains < policy.minimum_failure_domains {
        return Err(GatewayMembershipError::InsufficientFailureDomains {
            actual: failure_domains,
            required: policy.minimum_failure_domains,
        });
    }
    let proposed_by_id: BTreeMap<_, _> = proposed
        .members
        .iter()
        .map(|member| (member.gateway_id.as_str(), u32::from(member.voting_weight)))
        .collect();
    let retained: u32 = current
        .members
        .iter()
        .filter_map(|member| proposed_by_id.get(member.gateway_id.as_str()).copied())
        .sum();
    if retained < policy.minimum_retained_voting_weight {
        return Err(GatewayMembershipError::InsufficientRetainedWeight {
            actual: retained,
            required: policy.minimum_retained_voting_weight,
        });
    }
    let current_weight = current.total_voting_weight();
    let removed_weight = current_weight.saturating_sub(retained);
    let removed_basis_points = if current_weight == 0 {
        10_000
    } else {
        ((u64::from(removed_weight) * 10_000) / u64::from(current_weight)) as u16
    };
    if removed_basis_points > policy.maximum_removed_weight_basis_points {
        return Err(GatewayMembershipError::ExcessiveRemovedWeight {
            basis_points: removed_basis_points,
            maximum: policy.maximum_removed_weight_basis_points,
        });
    }
    let transition_digest = digest_membership_transition(&transition, current)?;
    if ceremony.purpose() != "gateway-membership-rotation" {
        return Err(GatewayMembershipError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != transition_digest {
        return Err(GatewayMembershipError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedGatewayMembership {
        transition,
        transition_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_membership_policy(
    policy: &GatewayMembershipPolicy,
) -> Result<(), GatewayMembershipError> {
    if policy.minimum_members == 0
        || policy.minimum_members > MAX_GATEWAY_MEMBERS
        || policy.minimum_total_voting_weight == 0
        || policy.minimum_failure_domains == 0
        || policy.minimum_failure_domains > policy.minimum_members
        || policy.maximum_removed_weight_basis_points > 10_000
    {
        return Err(GatewayMembershipError::InvalidPolicy);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), ()> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(());
    }
    Ok(())
}

fn validate_failure_domain(value: &str) -> Result<(), GatewayMembershipError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_FAILURE_DOMAIN_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(GatewayMembershipError::InvalidFailureDomain(
            value.to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn membership(epoch: u64, ids: &[(&str, &str)]) -> GatewayMembership {
        GatewayMembership::new(
            epoch,
            100,
            1_000,
            ids.iter()
                .map(|(id, domain)| GatewayMember {
                    gateway_id: (*id).into(),
                    voting_weight: 1,
                    failure_domain: (*domain).into(),
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn digest_is_order_independent_through_constructor() {
        let a = membership(1, &[("b", "two"), ("a", "one"), ("c", "three")]);
        let b = membership(1, &[("c", "three"), ("b", "two"), ("a", "one")]);
        assert_eq!(
            digest_gateway_membership(&a).unwrap(),
            digest_gateway_membership(&b).unwrap()
        );
    }

    #[test]
    fn transition_requires_successor_epoch() {
        let current = membership(1, &[("a", "one"), ("b", "two"), ("c", "three")]);
        let proposed = membership(3, &[("a", "one"), ("b", "two"), ("d", "four")]);
        assert!(matches!(
            build_membership_transition(&current, proposed, 200, "rotate"),
            Err(GatewayMembershipError::EpochNotSuccessor { .. })
        ));
    }
}
