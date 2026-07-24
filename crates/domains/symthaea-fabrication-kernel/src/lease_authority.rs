// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Partition-safe, fenced gateway lease authority.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus::VerifiedGatewayConsensus;
use crate::gateway_membership::{GatewayMembership, digest_gateway_membership};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PARTITION_LEASE_SCHEMA: &str = "symthaea.fabrication.partition-lease.v1";
pub const LEASE_AUTHORITY_TRACKER_SCHEMA: &str = "symthaea.fabrication.lease-authority-tracker.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionLease {
    pub schema_version: String,
    pub membership_digest: Sha256Digest,
    pub membership_epoch: u64,
    pub gateway_consensus_digest: Sha256Digest,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub holder_gateway_id: String,
    pub lease_sequence: u64,
    pub fencing_token: u64,
    pub lease_nonce: Sha256Digest,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionLeasePolicy {
    pub minimum_voting_weight_basis_points: u16,
    pub minimum_failure_domains: usize,
    pub maximum_lease_duration_ms: u64,
}

impl Default for PartitionLeasePolicy {
    fn default() -> Self {
        Self {
            minimum_voting_weight_basis_points: 6_667,
            minimum_failure_domains: 2,
            maximum_lease_duration_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionLeaseError {
    UnsupportedSchema,
    InvalidPolicy,
    MembershipInvalid(String),
    MembershipInactive,
    HolderNotMember,
    HolderNotInConsensus,
    ConsensusStateMismatch,
    InvalidWindow,
    LeaseTooLong {
        actual_ms: u64,
        maximum_ms: u64,
    },
    SequenceZero,
    FencingTokenZero,
    InsufficientVotingWeight {
        actual_basis_points: u16,
        required_basis_points: u16,
    },
    InsufficientFailureDomains {
        actual: usize,
        required: usize,
    },
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedPartitionLease {
    lease: PartitionLease,
    lease_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedPartitionLease {
    pub fn lease(&self) -> &PartitionLease {
        &self.lease
    }
    pub fn lease_digest(&self) -> Sha256Digest {
        self.lease_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseAuthorityTracker {
    pub schema_version: String,
    latest_membership_epoch: Option<u64>,
    latest_lease_sequence: Option<u64>,
    latest_fencing_token: Option<u64>,
    active_lease_digest: Option<Sha256Digest>,
    active_lease_expires_at_unix_ms: Option<u64>,
}

impl Default for LeaseAuthorityTracker {
    fn default() -> Self {
        Self {
            schema_version: LEASE_AUTHORITY_TRACKER_SCHEMA.into(),
            latest_membership_epoch: None,
            latest_lease_sequence: None,
            latest_fencing_token: None,
            active_lease_digest: None,
            active_lease_expires_at_unix_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseTrackingError {
    UnsupportedSchema,
    LeaseExpired,
    MembershipEpochRollback { latest: u64, proposed: u64 },
    LeaseSequenceRollback { latest: u64, proposed: u64 },
    FencingTokenRollback { latest: u64, proposed: u64 },
    ActiveLeaseConflict,
    SameSequenceSubstitution,
    Encoding(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedPartitionLease {
    pub lease_digest: Sha256Digest,
    pub fencing_token: u64,
    pub idempotent_replay: bool,
}

impl PartitionLease {
    pub fn validate(&self) -> Result<(), PartitionLeaseError> {
        if self.schema_version != PARTITION_LEASE_SCHEMA {
            return Err(PartitionLeaseError::UnsupportedSchema);
        }
        if self.lease_sequence == 0 {
            return Err(PartitionLeaseError::SequenceZero);
        }
        if self.fencing_token == 0 {
            return Err(PartitionLeaseError::FencingTokenZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(PartitionLeaseError::InvalidWindow);
        }
        validate_identifier(&self.holder_gateway_id)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn authorize_partition_lease(
    membership: &GatewayMembership,
    consensus: &VerifiedGatewayConsensus,
    holder_gateway_id: impl Into<String>,
    lease_sequence: u64,
    fencing_token: u64,
    lease_nonce: Sha256Digest,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    policy: &PartitionLeasePolicy,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedPartitionLease, PartitionLeaseError> {
    membership
        .validate()
        .map_err(|error| PartitionLeaseError::MembershipInvalid(format!("{error:?}")))?;
    validate_policy(policy)?;
    if !membership.is_active_at(issued_at_unix_ms / 1_000) {
        return Err(PartitionLeaseError::MembershipInactive);
    }
    let holder_gateway_id = holder_gateway_id.into();
    if membership.member(&holder_gateway_id).is_none() {
        return Err(PartitionLeaseError::HolderNotMember);
    }
    if !consensus
        .gateways()
        .iter()
        .any(|gateway| gateway == &holder_gateway_id)
    {
        return Err(PartitionLeaseError::HolderNotInConsensus);
    }
    if issued_at_unix_ms >= expires_at_unix_ms {
        return Err(PartitionLeaseError::InvalidWindow);
    }
    let duration = expires_at_unix_ms - issued_at_unix_ms;
    if duration > policy.maximum_lease_duration_ms {
        return Err(PartitionLeaseError::LeaseTooLong {
            actual_ms: duration,
            maximum_ms: policy.maximum_lease_duration_ms,
        });
    }
    let consensus_gateways: BTreeSet<_> = consensus.gateways().iter().map(String::as_str).collect();
    let consensus_weight: u32 = membership
        .members
        .iter()
        .filter(|member| consensus_gateways.contains(member.gateway_id.as_str()))
        .map(|member| u32::from(member.voting_weight))
        .sum();
    let total_weight = membership.total_voting_weight();
    let weight_basis_points = if total_weight == 0 {
        0
    } else {
        ((u64::from(consensus_weight) * 10_000) / u64::from(total_weight)) as u16
    };
    if weight_basis_points < policy.minimum_voting_weight_basis_points {
        return Err(PartitionLeaseError::InsufficientVotingWeight {
            actual_basis_points: weight_basis_points,
            required_basis_points: policy.minimum_voting_weight_basis_points,
        });
    }
    let failure_domains = membership
        .members
        .iter()
        .filter(|member| consensus_gateways.contains(member.gateway_id.as_str()))
        .map(|member| member.failure_domain.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if failure_domains < policy.minimum_failure_domains {
        return Err(PartitionLeaseError::InsufficientFailureDomains {
            actual: failure_domains,
            required: policy.minimum_failure_domains,
        });
    }
    let lease = PartitionLease {
        schema_version: PARTITION_LEASE_SCHEMA.into(),
        membership_digest: digest_gateway_membership(membership)
            .map_err(|error| PartitionLeaseError::MembershipInvalid(format!("{error:?}")))?,
        membership_epoch: membership.epoch,
        gateway_consensus_digest: consensus.consensus_digest(),
        gateway_state_digest: consensus.state_digest(),
        gateway_generation: consensus.generation(),
        holder_gateway_id,
        lease_sequence,
        fencing_token,
        lease_nonce,
        issued_at_unix_ms,
        expires_at_unix_ms,
    };
    lease.validate()?;
    let lease_digest = digest_partition_lease(&lease)?;
    if ceremony.purpose() != "partition-lease-authority" {
        return Err(PartitionLeaseError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != lease_digest {
        return Err(PartitionLeaseError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedPartitionLease {
        lease,
        lease_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

pub fn digest_partition_lease(lease: &PartitionLease) -> Result<Sha256Digest, PartitionLeaseError> {
    lease.validate()?;
    let bytes = serde_json::to_vec(lease)
        .map_err(|error| PartitionLeaseError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.partition-lease-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

impl LeaseAuthorityTracker {
    pub fn validate(&self) -> Result<(), LeaseTrackingError> {
        if self.schema_version != LEASE_AUTHORITY_TRACKER_SCHEMA {
            return Err(LeaseTrackingError::UnsupportedSchema);
        }
        let fields = [
            self.latest_membership_epoch.is_some(),
            self.latest_lease_sequence.is_some(),
            self.latest_fencing_token.is_some(),
            self.active_lease_digest.is_some(),
            self.active_lease_expires_at_unix_ms.is_some(),
        ];
        if fields.iter().any(|value| *value) && fields.iter().any(|value| !*value) {
            return Err(LeaseTrackingError::SameSequenceSubstitution);
        }
        Ok(())
    }

    pub fn accept(
        &mut self,
        authorized: &AuthorizedPartitionLease,
        now_unix_ms: u64,
    ) -> Result<AcceptedPartitionLease, LeaseTrackingError> {
        self.validate()?;
        let lease = authorized.lease();
        if now_unix_ms < lease.issued_at_unix_ms || now_unix_ms >= lease.expires_at_unix_ms {
            return Err(LeaseTrackingError::LeaseExpired);
        }
        let digest = authorized.lease_digest();
        if let Some(latest_epoch) = self.latest_membership_epoch {
            if lease.membership_epoch < latest_epoch {
                return Err(LeaseTrackingError::MembershipEpochRollback {
                    latest: latest_epoch,
                    proposed: lease.membership_epoch,
                });
            }
        }
        if let Some(latest_sequence) = self.latest_lease_sequence {
            if lease.lease_sequence < latest_sequence {
                return Err(LeaseTrackingError::LeaseSequenceRollback {
                    latest: latest_sequence,
                    proposed: lease.lease_sequence,
                });
            }
            if lease.lease_sequence == latest_sequence {
                if self.active_lease_digest == Some(digest) {
                    return Ok(AcceptedPartitionLease {
                        lease_digest: digest,
                        fencing_token: lease.fencing_token,
                        idempotent_replay: true,
                    });
                }
                return Err(LeaseTrackingError::SameSequenceSubstitution);
            }
        }
        if let Some(latest_fence) = self.latest_fencing_token {
            if lease.fencing_token <= latest_fence {
                return Err(LeaseTrackingError::FencingTokenRollback {
                    latest: latest_fence,
                    proposed: lease.fencing_token,
                });
            }
        }
        if self
            .active_lease_expires_at_unix_ms
            .is_some_and(|expires| now_unix_ms < expires)
            && self.active_lease_digest != Some(digest)
        {
            return Err(LeaseTrackingError::ActiveLeaseConflict);
        }
        self.latest_membership_epoch = Some(lease.membership_epoch);
        self.latest_lease_sequence = Some(lease.lease_sequence);
        self.latest_fencing_token = Some(lease.fencing_token);
        self.active_lease_digest = Some(digest);
        self.active_lease_expires_at_unix_ms = Some(lease.expires_at_unix_ms);
        Ok(AcceptedPartitionLease {
            lease_digest: digest,
            fencing_token: lease.fencing_token,
            idempotent_replay: false,
        })
    }

    pub fn digest(&self) -> Result<Sha256Digest, LeaseTrackingError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| LeaseTrackingError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.lease-authority-tracker-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }
}

fn validate_policy(policy: &PartitionLeasePolicy) -> Result<(), PartitionLeaseError> {
    if policy.minimum_voting_weight_basis_points <= 5_000
        || policy.minimum_voting_weight_basis_points > 10_000
        || policy.minimum_failure_domains == 0
        || policy.maximum_lease_duration_ms == 0
    {
        return Err(PartitionLeaseError::InvalidPolicy);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), PartitionLeaseError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(PartitionLeaseError::HolderNotMember);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_rejects_overlapping_different_lease() {
        let mut tracker = LeaseAuthorityTracker::default();
        tracker.latest_membership_epoch = Some(1);
        tracker.latest_lease_sequence = Some(1);
        tracker.latest_fencing_token = Some(10);
        tracker.active_lease_digest = Some(Sha256Digest([1; 32]));
        tracker.active_lease_expires_at_unix_ms = Some(500);
        let replacement = AuthorizedPartitionLease {
            lease: PartitionLease {
                schema_version: PARTITION_LEASE_SCHEMA.into(),
                membership_digest: Sha256Digest([2; 32]),
                membership_epoch: 1,
                gateway_consensus_digest: Sha256Digest([3; 32]),
                gateway_state_digest: Sha256Digest([4; 32]),
                gateway_generation: 2,
                holder_gateway_id: "gateway-b".into(),
                lease_sequence: 2,
                fencing_token: 11,
                lease_nonce: Sha256Digest([5; 32]),
                issued_at_unix_ms: 200,
                expires_at_unix_ms: 600,
            },
            lease_digest: Sha256Digest([6; 32]),
            ceremony_digest: Sha256Digest([7; 32]),
        };
        assert_eq!(
            tracker.accept(&replacement, 300),
            Err(LeaseTrackingError::ActiveLeaseConflict)
        );
    }
}
