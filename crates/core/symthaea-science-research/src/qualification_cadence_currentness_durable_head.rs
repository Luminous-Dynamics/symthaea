// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Durability gate for a published cadence-conformant currentness evaluation.
//!
//! This layer does not extend scientific currentness to a later checkpoint. It
//! proves only that the exact positive publication remains the latest relevant
//! publication through one fresh federated head, while the publication checkpoint
//! and observed head participate in one monitored namespaced history and obey
//! conservative temporal ordering.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorClosure,
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyHeadFederationReceipt,
};

use crate::{
    CadenceConformantInstitutionalCurrentnessAtEvaluation,
    CadenceCurrentnessHeadAssessment, CadenceCurrentnessHeadClosure,
    PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation, Sha256Digest,
};

const DURABLE_CADENCE_CURRENTNESS_HEAD_DOMAIN: &str =
    "symthaea.durable-cadence-currentness-through-observed-head.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableCadenceCurrentnessHeadError {
    QualificationMismatch,
    CapabilityMismatch,
    PublicationMismatch,
    HeadAssessmentNotPositive,
    NamespaceMismatch,
    PublicationViewMismatch,
    HeadViewMismatch,
    MonitorNamespaceMismatch,
    PublicationViewMissingFromMonitor,
    HeadViewMissingFromMonitor,
    MonitorNotConsistent,
    HeadFederationMismatch,
    HeadFederationViewMismatch,
    TreeRollback,
    SameSizeRootMismatch,
    LargerHeadNotDefinitelyAfterPublication,
}

/// Non-deserializable capability establishing durable anti-rollback of one exact
/// cadence-currentness publication only through one exact fresh federated head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DurableCadenceCurrentnessThroughObservedHead {
    qualification_sha256: Sha256Digest,
    capability_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
    head_assessment_sha256: TrustSha256Digest,
    namespace_sha256: TrustSha256Digest,
    monitor_receipt_sha256: TrustSha256Digest,
    head_federation_receipt_sha256: TrustSha256Digest,
    publication_view_sha256: TrustSha256Digest,
    observed_head_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    durable_head_sha256: TrustSha256Digest,
}

impl DurableCadenceCurrentnessThroughObservedHead {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn capability_sha256(&self) -> &TrustSha256Digest { &self.capability_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }
    pub fn head_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.head_assessment_sha256
    }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn monitor_receipt_sha256(&self) -> &TrustSha256Digest { &self.monitor_receipt_sha256 }
    pub fn head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_receipt_sha256
    }
    pub fn publication_view_sha256(&self) -> &TrustSha256Digest {
        &self.publication_view_sha256
    }
    pub fn observed_head_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn durable_head_sha256(&self) -> &TrustSha256Digest { &self.durable_head_sha256 }

    pub const fn durable_anti_rollback_through_observed_head_established(&self) -> bool { true }
    pub const fn currentness_at_observed_head_time_established(&self) -> bool { false }
    pub const fn globally_latest_publication_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn establish_durable_cadence_currentness_through_observed_head(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    publication: &PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation,
    assessment: &CadenceCurrentnessHeadAssessment,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
) -> Result<DurableCadenceCurrentnessThroughObservedHead, DurableCadenceCurrentnessHeadError> {
    if publication.qualification_sha256() != current.qualification_sha256()
        || assessment.qualification_sha256() != current.qualification_sha256()
    {
        return Err(DurableCadenceCurrentnessHeadError::QualificationMismatch);
    }
    if publication.capability_sha256() != current.capability_sha256()
        || assessment.capability_sha256() != current.capability_sha256()
    {
        return Err(DurableCadenceCurrentnessHeadError::CapabilityMismatch);
    }
    if assessment.publication_sha256() != publication.publication_sha256() {
        return Err(DurableCadenceCurrentnessHeadError::PublicationMismatch);
    }
    if assessment.closure()
        != CadenceCurrentnessHeadClosure::LatestRelevantPublicationThroughFederatedHead
        || !assessment.latest_relevant_publication_through_federated_head_established()
    {
        return Err(DurableCadenceCurrentnessHeadError::HeadAssessmentNotPositive);
    }

    let namespace = publication.namespace_sha256();
    if observed_head.namespace_sha256() != namespace
        || assessment.namespace_sha256() != namespace
        || head_federation.namespace_sha256() != namespace
    {
        return Err(DurableCadenceCurrentnessHeadError::NamespaceMismatch);
    }
    if publication_view.namespaced_view_sha256() != publication.namespaced_view_sha256() {
        return Err(DurableCadenceCurrentnessHeadError::PublicationViewMismatch);
    }
    if assessment.observed_head_namespaced_view_sha256() != observed_head.namespaced_view_sha256()
        || assessment.observed_head_tree_size() != observed_head.tree_size()
    {
        return Err(DurableCadenceCurrentnessHeadError::HeadViewMismatch);
    }
    if monitor.namespace_sha256() != Some(namespace) {
        return Err(DurableCadenceCurrentnessHeadError::MonitorNamespaceMismatch);
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(publication_view.namespaced_view_sha256())
    {
        return Err(DurableCadenceCurrentnessHeadError::PublicationViewMissingFromMonitor);
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        return Err(DurableCadenceCurrentnessHeadError::HeadViewMissingFromMonitor);
    }

    match monitor.closure() {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {}
        NamespacedTransparencyMonitorClosure::Incomplete
            if monitor.findings().is_empty()
                && publication_view.tree_size() == observed_head.tree_size()
                && publication_view.root_sha256() == observed_head.root_sha256() => {}
        _ => return Err(DurableCadenceCurrentnessHeadError::MonitorNotConsistent),
    }

    if assessment.head_federation_receipt_sha256() != head_federation.receipt_sha256()
        || head_federation.monitor_receipt_sha256() != monitor.receipt_sha256()
    {
        return Err(DurableCadenceCurrentnessHeadError::HeadFederationMismatch);
    }
    if !head_federation
        .converged_namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
        || observed_head.tree_size() != head_federation.maximal_tree_size()
        || observed_head.root_sha256() != head_federation.maximal_root_sha256()
        || observed_head.witnessed_view().tree_head_sha256()
            != head_federation.maximal_tree_head_sha256()
    {
        return Err(DurableCadenceCurrentnessHeadError::HeadFederationViewMismatch);
    }

    if observed_head.tree_size() < publication_view.tree_size() {
        return Err(DurableCadenceCurrentnessHeadError::TreeRollback);
    }
    if observed_head.tree_size() == publication_view.tree_size() {
        if observed_head.root_sha256() != publication_view.root_sha256() {
            return Err(DurableCadenceCurrentnessHeadError::SameSizeRootMismatch);
        }
    } else {
        let (_, publication_latest) = publication_view.consensus_interval();
        let (head_earliest, _) = observed_head.consensus_interval();
        if head_earliest < publication_latest {
            return Err(
                DurableCadenceCurrentnessHeadError::LargerHeadNotDefinitelyAfterPublication,
            );
        }
    }

    let durable_head_sha256 = durable_head_digest(
        current,
        publication,
        assessment,
        publication_view,
        observed_head,
        monitor,
        head_federation,
    );
    Ok(DurableCadenceCurrentnessThroughObservedHead {
        qualification_sha256: current.qualification_sha256().clone(),
        capability_sha256: current.capability_sha256().clone(),
        publication_sha256: publication.publication_sha256().clone(),
        head_assessment_sha256: assessment.assessment_sha256().clone(),
        namespace_sha256: namespace.clone(),
        monitor_receipt_sha256: monitor.receipt_sha256().clone(),
        head_federation_receipt_sha256: head_federation.receipt_sha256().clone(),
        publication_view_sha256: publication_view.namespaced_view_sha256().clone(),
        observed_head_view_sha256: observed_head.namespaced_view_sha256().clone(),
        observed_head_tree_size: observed_head.tree_size(),
        durable_head_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn durable_head_digest(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    publication: &PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation,
    assessment: &CadenceCurrentnessHeadAssessment,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(DURABLE_CADENCE_CURRENTNESS_HEAD_DOMAIN);
    digest.text(current.qualification_sha256().as_str());
    digest.text(current.capability_sha256().as_str());
    digest.text(publication.publication_sha256().as_str());
    digest.text(assessment.assessment_sha256().as_str());
    digest.text(publication_view.namespaced_view_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(head_federation.receipt_sha256().as_str());
    digest.text("durable-anti-rollback-through-observed-head-established");
    digest.text("currentness-at-observed-head-time-not-established");
    digest.text("globally-latest-publication-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_head_does_not_upgrade_science() {
        fn _assert_api(value: &DurableCadenceCurrentnessThroughObservedHead) {
            if value.durable_anti_rollback_through_observed_head_established() {
                assert!(!value.currentness_at_observed_head_time_established());
                assert!(!value.globally_latest_publication_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
