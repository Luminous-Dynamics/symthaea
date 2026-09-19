// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Replay-bound policy provenance for qualification-validity readiness.
//!
//! Older readiness/currentness receipts cryptographically bind their policy
//! values, but do not expose a first-class policy identity. Downstream lease
//! logic must not accept a caller-presented stricter policy merely because it
//! looks compatible. This layer solves that without rewriting historical receipt
//! formats: it re-runs head federation, lifecycle freshness, and evidence
//! freshness under the exact supplied policies and retains deterministic policy
//! identities next to the resulting receipt identities.
//!
//! A historical currentness artifact is therefore upgradeable to policy-bound
//! provenance only when its exact readiness assessment can be reproduced by this
//! replay. No status field or caller assertion can supply the missing lineage.

use serde::Serialize;
use symthaea_trust_core::{
    federate_fresh_witnessed_heads, FramedDigest, NamespacedTransparencyMonitorReceipt,
    NamespacedWitnessedTransparencyCheckpoint, Sha256Digest as TrustSha256Digest,
    TransparencyHeadFederationFinding, TransparencyHeadFederationPolicy,
    TransparencyHeadFederationReceipt, TrustedTime, VerifiedTransparencyWitnessQuorum,
};

use crate::{
    assess_qualification_validity_readiness, EvidenceFreshnessPolicy,
    EvidenceValidityReplayInputs, LifecycleHeadFreshnessPolicy, LifecycleValidityReplayInputs,
    QualificationValidityReadinessBundle, QualificationValidityReadinessClosure, Sha256Digest,
};

const HEAD_FEDERATION_POLICY_DOMAIN: &str =
    "symthaea.transparency-head-federation-policy.identity.v1";
const LIFECYCLE_FRESHNESS_POLICY_DOMAIN: &str =
    "symthaea.lifecycle-head-freshness-policy.identity.v1";
const EVIDENCE_FRESHNESS_POLICY_DOMAIN: &str =
    "symthaea.evidence-freshness-policy.identity.v1";
const POLICY_BOUND_READINESS_DOMAIN: &str =
    "symthaea.policy-bound-qualification-validity-readiness.identity.v1";

pub struct HeadFederationReplayInputs<'a> {
    pub views: &'a [NamespacedWitnessedTransparencyCheckpoint],
    pub quorums: &'a [VerifiedTransparencyWitnessQuorum],
    pub monitor: &'a NamespacedTransparencyMonitorReceipt,
    pub policy: &'a TransparencyHeadFederationPolicy,
}

/// Private-field replay receipt proving which exact policy values produced one
/// exact readiness lineage. Serializable for evidence retention, intentionally
/// not deserializable into authority/provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PolicyBoundQualificationValidityReadiness {
    qualification_sha256: Sha256Digest,
    readiness: QualificationValidityReadinessBundle,
    head_federation: TransparencyHeadFederationReceipt,
    head_federation_policy_sha256: TrustSha256Digest,
    lifecycle_freshness_policy_sha256: TrustSha256Digest,
    evidence_freshness_policy_sha256: TrustSha256Digest,
    binding_sha256: TrustSha256Digest,
}

impl PolicyBoundQualificationValidityReadiness {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn readiness(&self) -> &QualificationValidityReadinessBundle { &self.readiness }
    pub fn head_federation(&self) -> &TransparencyHeadFederationReceipt {
        &self.head_federation
    }
    pub fn head_federation_policy_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_policy_sha256
    }
    pub fn lifecycle_freshness_policy_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_freshness_policy_sha256
    }
    pub fn evidence_freshness_policy_sha256(&self) -> &TrustSha256Digest {
        &self.evidence_freshness_policy_sha256
    }
    pub fn binding_sha256(&self) -> &TrustSha256Digest { &self.binding_sha256 }

    pub fn policy_bound_readiness_established(&self) -> bool {
        self.readiness.readiness().closure()
            == QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol
            && self.readiness.readiness().validity_readiness_established()
            && self.head_federation.converged_fresh_observed_head_established()
    }
    pub const fn institutional_currentness_established(&self) -> bool { false }
    pub const fn global_currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn replay_policy_bound_validity_readiness(
    lifecycle_inputs: LifecycleValidityReplayInputs<'_>,
    evidence_inputs: EvidenceValidityReplayInputs<'_>,
    federation_inputs: HeadFederationReplayInputs<'_>,
    evaluation_time: &TrustedTime,
) -> Result<PolicyBoundQualificationValidityReadiness, Vec<TransparencyHeadFederationFinding>> {
    let head_policy_sha256 = head_federation_policy_digest(federation_inputs.policy);
    let lifecycle_policy_sha256 = lifecycle_freshness_policy_digest(lifecycle_inputs.policy);
    let evidence_policy_sha256 = evidence_freshness_policy_digest(evidence_inputs.policy);

    let head_federation = federate_fresh_witnessed_heads(
        federation_inputs.views,
        federation_inputs.quorums,
        federation_inputs.monitor,
        evaluation_time,
        federation_inputs.policy,
    )?;
    let readiness = assess_qualification_validity_readiness(
        lifecycle_inputs,
        evidence_inputs,
        &head_federation,
        evaluation_time,
    );
    let qualification_sha256 = readiness.readiness().qualification_sha256().clone();
    let binding_sha256 = policy_bound_readiness_digest(
        &qualification_sha256,
        &readiness,
        &head_federation,
        &head_policy_sha256,
        &lifecycle_policy_sha256,
        &evidence_policy_sha256,
    );

    Ok(PolicyBoundQualificationValidityReadiness {
        qualification_sha256,
        readiness,
        head_federation,
        head_federation_policy_sha256: head_policy_sha256,
        lifecycle_freshness_policy_sha256: lifecycle_policy_sha256,
        evidence_freshness_policy_sha256: evidence_policy_sha256,
        binding_sha256,
    })
}

pub fn head_federation_policy_digest(
    policy: &TransparencyHeadFederationPolicy,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(HEAD_FEDERATION_POLICY_DOMAIN);
    digest.text(&policy.minimum_converged_views.to_string());
    digest.text(&policy.minimum_distinct_quorums.to_string());
    digest.text(&policy.minimum_distinct_principals.to_string());
    digest.text(&policy.minimum_distinct_organizations.to_string());
    digest.text(&policy.minimum_distinct_regions.to_string());
    digest.text(&policy.maximum_head_age_s.to_string());
    digest.text(&policy.maximum_views.to_string());
    digest.text(&policy.maximum_quorums.to_string());
    digest.digest()
}

pub fn lifecycle_freshness_policy_digest(
    policy: &LifecycleHeadFreshnessPolicy,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(LIFECYCLE_FRESHNESS_POLICY_DOMAIN);
    digest.text(&policy.maximum_checkpoint_age_s.to_string());
    digest.text(&policy.maximum_tail_entries.to_string());
    digest.digest()
}

pub fn evidence_freshness_policy_digest(policy: &EvidenceFreshnessPolicy) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(EVIDENCE_FRESHNESS_POLICY_DOMAIN);
    digest.text(&policy.maximum_search_lag_ms.to_string());
    digest.text(&policy.maximum_sources.to_string());
    digest.digest()
}

fn policy_bound_readiness_digest(
    qualification_sha256: &Sha256Digest,
    readiness: &QualificationValidityReadinessBundle,
    head_federation: &TransparencyHeadFederationReceipt,
    head_policy_sha256: &TrustSha256Digest,
    lifecycle_policy_sha256: &TrustSha256Digest,
    evidence_policy_sha256: &TrustSha256Digest,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(POLICY_BOUND_READINESS_DOMAIN);
    digest.text(qualification_sha256.as_str());
    digest.text(readiness.readiness().assessment_sha256().as_str());
    digest.text(readiness.lifecycle_freshness().assessment_sha256().as_str());
    digest.text(readiness.evidence_freshness().assessment_sha256().as_str());
    digest.text(head_federation.receipt_sha256().as_str());
    digest.text(head_policy_sha256.as_str());
    digest.text(lifecycle_policy_sha256.as_str());
    digest.text(evidence_policy_sha256.as_str());
    digest.text("policy-bound-readiness-replay-established");
    digest.text("institutional-currentness-not-established");
    digest.text("global-currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_digests_change_when_semantics_change() {
        let a = TransparencyHeadFederationPolicy {
            minimum_converged_views: 2,
            minimum_distinct_quorums: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 1,
            maximum_head_age_s: 60,
            maximum_views: 8,
            maximum_quorums: 8,
        };
        let mut b = a.clone();
        b.maximum_head_age_s = 61;
        assert_ne!(head_federation_policy_digest(&a), head_federation_policy_digest(&b));

        let lifecycle_a = LifecycleHeadFreshnessPolicy {
            maximum_checkpoint_age_s: 60,
            maximum_tail_entries: 100,
        };
        let lifecycle_b = LifecycleHeadFreshnessPolicy {
            maximum_checkpoint_age_s: 61,
            maximum_tail_entries: 100,
        };
        assert_ne!(
            lifecycle_freshness_policy_digest(&lifecycle_a),
            lifecycle_freshness_policy_digest(&lifecycle_b),
        );

        let evidence_a = EvidenceFreshnessPolicy {
            maximum_search_lag_ms: 1000,
            maximum_sources: 10,
        };
        let evidence_b = EvidenceFreshnessPolicy {
            maximum_search_lag_ms: 1001,
            maximum_sources: 10,
        };
        assert_ne!(
            evidence_freshness_policy_digest(&evidence_a),
            evidence_freshness_policy_digest(&evidence_b),
        );
    }
}
