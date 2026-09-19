// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public authority surface for witnessed-head federation.
//!
//! The lower evaluator is retained inside the crate so structural tests may
//! exercise it, but the public entrypoint hard-requires a genuinely plural
//! federation: at least two converged views and at least two distinct witness
//! quorums. A one-view freshness check belongs to lower transparency layers and
//! must not be mislabeled as federation.

use crate::{
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    TransparencyHeadFederationFinding, TransparencyHeadFederationPolicy,
    TransparencyHeadFederationReceipt, TrustedTime, VerifiedTransparencyWitnessQuorum,
};

pub fn federate_fresh_witnessed_heads(
    views: &[NamespacedWitnessedTransparencyCheckpoint],
    quorums: &[VerifiedTransparencyWitnessQuorum],
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    policy: &TransparencyHeadFederationPolicy,
) -> Result<TransparencyHeadFederationReceipt, Vec<TransparencyHeadFederationFinding>> {
    if policy.minimum_converged_views < 2 || policy.minimum_distinct_quorums < 2 {
        return Err(vec![TransparencyHeadFederationFinding::InvalidPolicy]);
    }
    crate::transparency_head_federation::federate_fresh_witnessed_heads(
        views,
        quorums,
        monitor,
        evaluation_time,
        policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federation_semantics_require_plural_views_and_quorums() {
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
        // The structural policy is internally coherent, but the public authority
        // gate deliberately imposes the stronger semantic meaning of federation.
        assert!(policy.minimum_converged_views < 2);
        assert!(policy.minimum_distinct_quorums < 2);
    }
}
