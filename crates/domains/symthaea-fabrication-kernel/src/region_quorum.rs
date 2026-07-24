// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-region quorum evidence for federated gateway authority.
//!
//! Gateway consensus proves that named gateways endorsed one exact state. This
//! module additionally proves that the valid endorsers are distributed across
//! independently named failure domains and that no single region dominates the
//! represented voting weight beyond policy.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus::VerifiedGatewayConsensus;
use crate::gateway_membership::{GatewayMembership, digest_gateway_membership};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REGIONAL_QUORUM_SCHEMA: &str = "symthaea.fabrication.regional-quorum.v1";
pub const MAX_REQUIRED_REGIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionalQuorumPolicy {
    pub minimum_distinct_regions: usize,
    pub minimum_represented_weight_basis_points: u16,
    pub maximum_single_region_weight_basis_points: u16,
    pub required_regions: BTreeSet<String>,
}

impl Default for RegionalQuorumPolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_regions: 2,
            minimum_represented_weight_basis_points: 6_667,
            maximum_single_region_weight_basis_points: 6_667,
            required_regions: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionalQuorumEvidence {
    pub schema_version: String,
    pub membership_digest: Sha256Digest,
    pub membership_epoch: u64,
    pub gateway_consensus_digest: Sha256Digest,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub total_membership_weight: u32,
    pub represented_weight: u32,
    pub represented_regions: Vec<RegionalWeight>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionalWeight {
    pub region: String,
    pub voting_weight: u32,
    pub gateway_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionalQuorumError {
    InvalidPolicy,
    MembershipInvalid(String),
    MembershipInactive,
    UnknownGateway(String),
    InsufficientRegions {
        actual: usize,
        required: usize,
    },
    MissingRequiredRegion(String),
    InsufficientRepresentedWeight {
        basis_points: u16,
        required: u16,
    },
    RegionDominates {
        region: String,
        basis_points: u16,
        maximum: u16,
    },
    Encoding(String),
}

pub fn build_regional_quorum_evidence(
    consensus: &VerifiedGatewayConsensus,
    membership: &GatewayMembership,
    evaluated_at_unix_s: u64,
    policy: &RegionalQuorumPolicy,
) -> Result<RegionalQuorumEvidence, RegionalQuorumError> {
    validate_policy(policy)?;
    membership
        .validate()
        .map_err(|error| RegionalQuorumError::MembershipInvalid(format!("{error:?}")))?;
    if !membership.is_active_at(evaluated_at_unix_s) {
        return Err(RegionalQuorumError::MembershipInactive);
    }

    let members = membership
        .members
        .iter()
        .map(|member| (member.gateway_id.as_str(), member))
        .collect::<BTreeMap<_, _>>();
    let mut by_region: BTreeMap<String, (u32, Vec<String>)> = BTreeMap::new();
    let mut represented_weight = 0u32;
    for gateway_id in consensus.gateways() {
        let Some(member) = members.get(gateway_id.as_str()) else {
            return Err(RegionalQuorumError::UnknownGateway(gateway_id.clone()));
        };
        represented_weight = represented_weight.saturating_add(u32::from(member.voting_weight));
        let entry = by_region
            .entry(member.failure_domain.clone())
            .or_insert_with(|| (0, Vec::new()));
        entry.0 = entry.0.saturating_add(u32::from(member.voting_weight));
        entry.1.push(gateway_id.clone());
    }

    if by_region.len() < policy.minimum_distinct_regions {
        return Err(RegionalQuorumError::InsufficientRegions {
            actual: by_region.len(),
            required: policy.minimum_distinct_regions,
        });
    }
    for region in &policy.required_regions {
        if !by_region.contains_key(region) {
            return Err(RegionalQuorumError::MissingRequiredRegion(region.clone()));
        }
    }

    let total = membership.total_voting_weight();
    let represented_basis_points = basis_points(represented_weight, total);
    if represented_basis_points < policy.minimum_represented_weight_basis_points {
        return Err(RegionalQuorumError::InsufficientRepresentedWeight {
            basis_points: represented_basis_points,
            required: policy.minimum_represented_weight_basis_points,
        });
    }
    for (region, (weight, _)) in &by_region {
        let region_basis_points = basis_points(*weight, represented_weight);
        if region_basis_points > policy.maximum_single_region_weight_basis_points {
            return Err(RegionalQuorumError::RegionDominates {
                region: region.clone(),
                basis_points: region_basis_points,
                maximum: policy.maximum_single_region_weight_basis_points,
            });
        }
    }

    let represented_regions = by_region
        .into_iter()
        .map(|(region, (voting_weight, mut gateway_ids))| {
            gateway_ids.sort();
            RegionalWeight {
                region,
                voting_weight,
                gateway_ids,
            }
        })
        .collect();
    Ok(RegionalQuorumEvidence {
        schema_version: REGIONAL_QUORUM_SCHEMA.into(),
        membership_digest: digest_gateway_membership(membership)
            .map_err(|error| RegionalQuorumError::MembershipInvalid(format!("{error:?}")))?,
        membership_epoch: membership.epoch,
        gateway_consensus_digest: consensus.consensus_digest(),
        gateway_state_digest: consensus.state_digest(),
        gateway_generation: consensus.generation(),
        total_membership_weight: total,
        represented_weight,
        represented_regions,
    })
}

pub fn digest_regional_quorum_evidence(
    evidence: &RegionalQuorumEvidence,
) -> Result<Sha256Digest, RegionalQuorumError> {
    validate_regional_quorum_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| RegionalQuorumError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.regional-quorum-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn validate_regional_quorum_evidence(
    evidence: &RegionalQuorumEvidence,
) -> Result<(), RegionalQuorumError> {
    if evidence.schema_version != REGIONAL_QUORUM_SCHEMA
        || evidence.membership_epoch == 0
        || evidence.total_membership_weight == 0
        || evidence.represented_weight == 0
        || evidence.represented_weight > evidence.total_membership_weight
        || evidence.represented_regions.is_empty()
    {
        return Err(RegionalQuorumError::InvalidPolicy);
    }
    let mut previous_region: Option<&str> = None;
    let mut total = 0u32;
    let mut all_gateways = BTreeSet::new();
    for region in &evidence.represented_regions {
        if region.region.trim().is_empty()
            || region.region != region.region.trim()
            || region.voting_weight == 0
            || region.gateway_ids.is_empty()
            || previous_region.is_some_and(|value| value >= region.region.as_str())
        {
            return Err(RegionalQuorumError::InvalidPolicy);
        }
        let mut previous_gateway: Option<&str> = None;
        for gateway_id in &region.gateway_ids {
            if gateway_id.trim().is_empty()
                || gateway_id != gateway_id.trim()
                || previous_gateway.is_some_and(|value| value >= gateway_id.as_str())
                || !all_gateways.insert(gateway_id.clone())
            {
                return Err(RegionalQuorumError::InvalidPolicy);
            }
            previous_gateway = Some(gateway_id);
        }
        total = total.saturating_add(region.voting_weight);
        previous_region = Some(&region.region);
    }
    if total != evidence.represented_weight {
        return Err(RegionalQuorumError::InvalidPolicy);
    }
    Ok(())
}

fn basis_points(part: u32, whole: u32) -> u16 {
    if whole == 0 {
        return 0;
    }
    ((u64::from(part) * 10_000) / u64::from(whole)).min(10_000) as u16
}

fn validate_policy(policy: &RegionalQuorumPolicy) -> Result<(), RegionalQuorumError> {
    if policy.minimum_distinct_regions == 0
        || policy.minimum_distinct_regions > MAX_REQUIRED_REGIONS
        || policy.minimum_represented_weight_basis_points == 0
        || policy.minimum_represented_weight_basis_points > 10_000
        || policy.maximum_single_region_weight_basis_points == 0
        || policy.maximum_single_region_weight_basis_points > 10_000
        || policy.required_regions.len() > MAX_REQUIRED_REGIONS
        || policy.required_regions.iter().any(|region| {
            region.trim().is_empty()
                || region != region.trim()
                || region.len() > 256
                || region.chars().any(char::is_control)
        })
    {
        return Err(RegionalQuorumError::InvalidPolicy);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_points_is_bounded() {
        assert_eq!(basis_points(2, 3), 6_666);
        assert_eq!(basis_points(20, 10), 10_000);
        assert_eq!(basis_points(1, 0), 0);
    }
}
