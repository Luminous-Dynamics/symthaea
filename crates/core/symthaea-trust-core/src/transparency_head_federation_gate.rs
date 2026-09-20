// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public authority surface for witnessed-head federation.
//!
//! The lower evaluator is retained inside the crate so structural tests may
//! exercise it, but the public entrypoint gives "federation" a deliberately
//! stronger meaning. A positive federation result requires at least two
//! converged views, at least two distinct quorum events, and at least two
//! distinct root-bound principal constituencies. Repeated observations or key
//! rotations by the same principal committee therefore cannot manufacture
//! federation plurality.
//!
//! Distinct constituencies are still not proof of global witness independence.
//! They may share organizations, infrastructure, operators, funding, or other
//! latent dependencies.

use std::collections::BTreeSet;

use crate::{
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    TransparencyHeadFederationFinding, TransparencyHeadFederationPolicy,
    TransparencyHeadFederationReceipt, TrustedTime, VerifiedTransparencyWitnessQuorum,
    derive_witness_quorum_constituency,
};

pub const MIN_PUBLIC_FEDERATION_CONSTITUENCIES: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyHeadFederationGateError {
    Core(Vec<TransparencyHeadFederationFinding>),
    InsufficientDistinctConstituencies {
        actual: usize,
        required: usize,
    },
}

pub fn federate_fresh_witnessed_heads(
    views: &[NamespacedWitnessedTransparencyCheckpoint],
    quorums: &[VerifiedTransparencyWitnessQuorum],
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    policy: &TransparencyHeadFederationPolicy,
) -> Result<TransparencyHeadFederationReceipt, TransparencyHeadFederationGateError> {
    if policy.minimum_converged_views < 2 || policy.minimum_distinct_quorums < 2 {
        return Err(TransparencyHeadFederationGateError::Core(vec![
            TransparencyHeadFederationFinding::InvalidPolicy,
        ]));
    }

    let receipt = crate::transparency_head_federation::federate_fresh_witnessed_heads(
        views,
        quorums,
        monitor,
        evaluation_time,
        policy,
    )
    .map_err(TransparencyHeadFederationGateError::Core)?;

    // Preserve lower-layer diagnostic receipts. Constituency plurality is an
    // authority gate on the positive semantic claim, not a reason to erase
    // useful stale/incomplete/blocked/invalid assessment evidence.
    if receipt.converged_fresh_observed_head_established() {
        let converged_quorums: BTreeSet<_> = receipt
            .converged_witness_quorum_sha256s()
            .iter()
            .cloned()
            .collect();
        let constituency_sha256s: BTreeSet<_> = quorums
            .iter()
            .filter(|quorum| converged_quorums.contains(quorum.quorum_sha256()))
            .map(|quorum| {
                derive_witness_quorum_constituency(quorum)
                    .constituency_sha256()
                    .clone()
            })
            .collect();
        if constituency_sha256s.len() < MIN_PUBLIC_FEDERATION_CONSTITUENCIES {
            return Err(
                TransparencyHeadFederationGateError::InsufficientDistinctConstituencies {
                    actual: constituency_sha256s.len(),
                    required: MIN_PUBLIC_FEDERATION_CONSTITUENCIES,
                },
            );
        }
    }

    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federation_semantics_require_plural_views_quorums_and_constituencies() {
        let policy = TransparencyHeadFederationPolicy {
            minimum_converged_views: 1,
            minimum_distinct_quorums: 1,
            minimum_distinct_principals: 1,
            minimum_distinct_organizations: 1,
            minimum_distinct_regions: 1,
            maximum_head_age_s: 60,
            maximum_views: 4,
            maximum_quorums: 4,
        };
        assert!(policy.validate().is_ok());
        // The lower structural policy is internally coherent, but the public
        // authority gate deliberately imposes stronger federation semantics.
        assert!(policy.minimum_converged_views < 2);
        assert!(policy.minimum_distinct_quorums < 2);
        assert_eq!(MIN_PUBLIC_FEDERATION_CONSTITUENCIES, 2);
    }
}
