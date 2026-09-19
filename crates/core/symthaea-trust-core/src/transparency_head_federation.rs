// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fresh witnessed-head federation for one stable transparency namespace.
//!
//! This layer asks a deliberately bounded question: do multiple independently
//! root-bound witness quorums converge on the same maximal witnessed tree head,
//! under one append-only namespaced monitor receipt, and is that head fresh
//! relative to one exact authenticated evaluation-time interval?
//!
//! A positive result does not establish that the converged head is globally
//! latest, that every honest observer has seen it, or that the log is globally
//! consistent. It only establishes convergence among the exact witnessed views
//! and root-bound principals supplied to this assessment.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    FramedDigest, NamespacedTransparencyMonitorClosure,
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest, TrustedTime, VerifiedTransparencyWitnessQuorum,
};

const HEAD_FEDERATION_DOMAIN: &str =
    "symthaea.transparency-head-federation.identity.v1";
pub const MAX_HEAD_FEDERATION_VIEWS: usize = 256;
pub const MAX_HEAD_FEDERATION_QUORUMS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyHeadFederationPolicy {
    pub minimum_converged_views: usize,
    pub minimum_distinct_quorums: usize,
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_organizations: usize,
    pub minimum_distinct_regions: usize,
    pub maximum_head_age_s: u64,
    pub maximum_views: usize,
    pub maximum_quorums: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparencyHeadFederationPolicyIssue {
    ZeroConvergedViewThreshold,
    ZeroQuorumThreshold,
    ZeroPrincipalThreshold,
    ZeroMaximumViews,
    ZeroMaximumQuorums,
    MaximumViewsTooLarge,
    MaximumQuorumsTooLarge,
    ConvergedThresholdExceedsMaximumViews,
    QuorumThresholdExceedsMaximumQuorums,
    OrganizationThresholdExceedsPrincipals,
    RegionThresholdExceedsPrincipals,
}

impl TransparencyHeadFederationPolicy {
    pub fn validate(&self) -> Result<(), Vec<TransparencyHeadFederationPolicyIssue>> {
        let mut issues = Vec::new();
        if self.minimum_converged_views == 0 {
            issues.push(TransparencyHeadFederationPolicyIssue::ZeroConvergedViewThreshold);
        }
        if self.minimum_distinct_quorums == 0 {
            issues.push(TransparencyHeadFederationPolicyIssue::ZeroQuorumThreshold);
        }
        if self.minimum_distinct_principals == 0 {
            issues.push(TransparencyHeadFederationPolicyIssue::ZeroPrincipalThreshold);
        }
        if self.maximum_views == 0 {
            issues.push(TransparencyHeadFederationPolicyIssue::ZeroMaximumViews);
        }
        if self.maximum_quorums == 0 {
            issues.push(TransparencyHeadFederationPolicyIssue::ZeroMaximumQuorums);
        }
        if self.maximum_views > MAX_HEAD_FEDERATION_VIEWS {
            issues.push(TransparencyHeadFederationPolicyIssue::MaximumViewsTooLarge);
        }
        if self.maximum_quorums > MAX_HEAD_FEDERATION_QUORUMS {
            issues.push(TransparencyHeadFederationPolicyIssue::MaximumQuorumsTooLarge);
        }
        if self.minimum_converged_views > self.maximum_views {
            issues.push(
                TransparencyHeadFederationPolicyIssue::ConvergedThresholdExceedsMaximumViews,
            );
        }
        if self.minimum_distinct_quorums > self.maximum_quorums {
            issues.push(TransparencyHeadFederationPolicyIssue::QuorumThresholdExceedsMaximumQuorums);
        }
        if self.minimum_distinct_organizations > self.minimum_distinct_principals {
            issues.push(TransparencyHeadFederationPolicyIssue::OrganizationThresholdExceedsPrincipals);
        }
        if self.minimum_distinct_regions > self.minimum_distinct_principals {
            issues.push(TransparencyHeadFederationPolicyIssue::RegionThresholdExceedsPrincipals);
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TransparencyHeadFederationClosure {
    ConvergedFreshObservedHead,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum TransparencyHeadFederationFinding {
    InvalidPolicy,
    EmptyViews,
    TooManyViews,
    TooManyQuorums,
    DuplicateView,
    DuplicateQuorum,
    NamespaceMissing,
    NamespaceMismatch,
    ViewMissingFromMonitor,
    MonitorIncomplete,
    MonitorEquivocation,
    MonitorTemporalConflict,
    MonitorInvalid,
    MissingViewQuorum,
    UnknownQuorum,
    QuorumCheckpointMismatch,
    QuorumRootAuthorityMismatch,
    EvaluationRootAuthorityMismatch,
    MaxHeadNotDefinitelyBeforeEvaluation,
    MaxHeadTooOld,
    InsufficientConvergedViews { actual: usize, required: usize },
    InsufficientDistinctQuorums { actual: usize, required: usize },
    InsufficientDistinctPrincipals { actual: usize, required: usize },
    InsufficientDistinctOrganizations { actual: usize, required: usize },
    InsufficientDistinctRegions { actual: usize, required: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyHeadFederationReceipt {
    namespace_sha256: Sha256Digest,
    monitor_receipt_sha256: Sha256Digest,
    evaluation_time_authority_sha256: Sha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    maximal_tree_size: u64,
    maximal_root_sha256: Sha256Digest,
    maximal_tree_head_sha256: Sha256Digest,
    converged_namespaced_view_sha256s: Vec<Sha256Digest>,
    converged_witness_quorum_sha256s: Vec<Sha256Digest>,
    distinct_principal_ids: Vec<String>,
    distinct_organization_ids: Vec<String>,
    distinct_region_ids: Vec<String>,
    findings: Vec<TransparencyHeadFederationFinding>,
    closure: TransparencyHeadFederationClosure,
    receipt_sha256: Sha256Digest,
}

impl TransparencyHeadFederationReceipt {
    pub fn namespace_sha256(&self) -> &Sha256Digest { &self.namespace_sha256 }
    pub fn monitor_receipt_sha256(&self) -> &Sha256Digest { &self.monitor_receipt_sha256 }
    pub fn evaluation_time_authority_sha256(&self) -> &Sha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn maximal_tree_size(&self) -> u64 { self.maximal_tree_size }
    pub fn maximal_root_sha256(&self) -> &Sha256Digest { &self.maximal_root_sha256 }
    pub fn maximal_tree_head_sha256(&self) -> &Sha256Digest { &self.maximal_tree_head_sha256 }
    pub fn converged_namespaced_view_sha256s(&self) -> &[Sha256Digest] {
        &self.converged_namespaced_view_sha256s
    }
    pub fn converged_witness_quorum_sha256s(&self) -> &[Sha256Digest] {
        &self.converged_witness_quorum_sha256s
    }
    pub fn distinct_principal_ids(&self) -> &[String] { &self.distinct_principal_ids }
    pub fn distinct_organization_ids(&self) -> &[String] { &self.distinct_organization_ids }
    pub fn distinct_region_ids(&self) -> &[String] { &self.distinct_region_ids }
    pub fn findings(&self) -> &[TransparencyHeadFederationFinding] { &self.findings }
    pub fn closure(&self) -> TransparencyHeadFederationClosure { self.closure }
    pub fn receipt_sha256(&self) -> &Sha256Digest { &self.receipt_sha256 }

    pub fn converged_fresh_observed_head_established(&self) -> bool {
        self.closure == TransparencyHeadFederationClosure::ConvergedFreshObservedHead
    }
    pub const fn globally_latest_head_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn global_witness_independence_established(&self) -> bool { false }
}

pub fn federate_fresh_witnessed_heads(
    views: &[NamespacedWitnessedTransparencyCheckpoint],
    quorums: &[VerifiedTransparencyWitnessQuorum],
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    policy: &TransparencyHeadFederationPolicy,
) -> Result<TransparencyHeadFederationReceipt, Vec<TransparencyHeadFederationFinding>> {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;

    if policy.validate().is_err() {
        findings.push(TransparencyHeadFederationFinding::InvalidPolicy);
        invalid = true;
    }
    if views.is_empty() {
        findings.push(TransparencyHeadFederationFinding::EmptyViews);
        incomplete = true;
    }
    if views.len() > policy.maximum_views || views.len() > MAX_HEAD_FEDERATION_VIEWS {
        findings.push(TransparencyHeadFederationFinding::TooManyViews);
        invalid = true;
    }
    if quorums.len() > policy.maximum_quorums || quorums.len() > MAX_HEAD_FEDERATION_QUORUMS {
        findings.push(TransparencyHeadFederationFinding::TooManyQuorums);
        invalid = true;
    }

    let Some(namespace_sha256) = monitor.namespace_sha256().cloned() else {
        findings.push(TransparencyHeadFederationFinding::NamespaceMissing);
        return Err(findings);
    };

    let mut seen_views = BTreeSet::new();
    for view in views {
        if !seen_views.insert(view.namespaced_view_sha256().clone()) {
            findings.push(TransparencyHeadFederationFinding::DuplicateView);
            invalid = true;
        }
        if view.namespace_sha256() != &namespace_sha256 {
            findings.push(TransparencyHeadFederationFinding::NamespaceMismatch);
            invalid = true;
        }
        if !monitor
            .namespaced_view_sha256s()
            .contains(view.namespaced_view_sha256())
        {
            findings.push(TransparencyHeadFederationFinding::ViewMissingFromMonitor);
            incomplete = true;
        }
    }

    match monitor.closure() {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {}
        NamespacedTransparencyMonitorClosure::Incomplete
            if monitor.findings().is_empty() && !views.is_empty()
                && views.iter().all(|view| {
                    view.tree_size() == views[0].tree_size()
                        && view.root_sha256() == views[0].root_sha256()
                }) => {}
        NamespacedTransparencyMonitorClosure::Incomplete => {
            findings.push(TransparencyHeadFederationFinding::MonitorIncomplete);
            incomplete = true;
        }
        NamespacedTransparencyMonitorClosure::EquivocationObserved => {
            findings.push(TransparencyHeadFederationFinding::MonitorEquivocation);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved => {
            findings.push(TransparencyHeadFederationFinding::MonitorTemporalConflict);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::Invalid => {
            findings.push(TransparencyHeadFederationFinding::MonitorInvalid);
            invalid = true;
        }
    }

    let mut quorum_by_sha = BTreeMap::new();
    for quorum in quorums {
        if quorum_by_sha.insert(quorum.quorum_sha256().clone(), quorum).is_some() {
            findings.push(TransparencyHeadFederationFinding::DuplicateQuorum);
            invalid = true;
        }
    }

    let maximal_tree_size = views.iter().map(|view| view.tree_size()).max().unwrap_or(0);
    let maximal_views: Vec<_> = views
        .iter()
        .filter(|view| view.tree_size() == maximal_tree_size)
        .collect();
    let maximal_root_sha256 = maximal_views
        .first()
        .map(|view| view.root_sha256().clone())
        .unwrap_or_else(|| Sha256Digest::of_bytes(b"empty-head-federation"));
    let maximal_tree_head_sha256 = maximal_views
        .first()
        .map(|view| view.witnessed_view().tree_head_sha256().clone())
        .unwrap_or_else(|| Sha256Digest::of_bytes(b"empty-head-federation-tree-head"));

    let mut converged_view_sha256s = Vec::new();
    let mut converged_quorum_sha256s = BTreeSet::new();
    let mut principal_ids = BTreeSet::new();
    let mut organization_ids = BTreeSet::new();
    let mut region_ids = BTreeSet::new();
    let mut maximal_root_authority = None;
    let mut maximal_earliest = u64::MAX;
    let mut maximal_latest = 0u64;

    for view in &maximal_views {
        if view.root_sha256() != &maximal_root_sha256
            || view.witnessed_view().tree_head_sha256() != &maximal_tree_head_sha256
        {
            findings.push(TransparencyHeadFederationFinding::MonitorEquivocation);
            blocked = true;
            continue;
        }
        converged_view_sha256s.push(view.namespaced_view_sha256().clone());
        let quorum_sha256 = view.witnessed_view().witness_quorum_sha256();
        let Some(quorum) = quorum_by_sha.get(quorum_sha256).copied() else {
            findings.push(TransparencyHeadFederationFinding::MissingViewQuorum);
            incomplete = true;
            continue;
        };
        if quorum.checkpoint_sha256() != view.witnessed_view().checkpoint_sha256() {
            findings.push(TransparencyHeadFederationFinding::QuorumCheckpointMismatch);
            invalid = true;
        }
        if quorum.root_authority_sha256() != view.root_authority_sha256() {
            findings.push(TransparencyHeadFederationFinding::QuorumRootAuthorityMismatch);
            invalid = true;
        }
        converged_quorum_sha256s.insert(quorum.quorum_sha256().clone());
        for principal in quorum.principal_ids() { principal_ids.insert(principal.clone()); }
        for organization in quorum.organization_ids() { organization_ids.insert(organization.clone()); }
        for region in quorum.region_ids() { region_ids.insert(region.clone()); }
        match &maximal_root_authority {
            None => maximal_root_authority = Some(view.root_authority_sha256().clone()),
            Some(expected) if expected == view.root_authority_sha256() => {}
            Some(_) => {
                findings.push(TransparencyHeadFederationFinding::QuorumRootAuthorityMismatch);
                invalid = true;
            }
        }
        let (earliest, latest) = view.consensus_interval();
        maximal_earliest = maximal_earliest.min(earliest);
        maximal_latest = maximal_latest.max(latest);
    }

    for quorum in quorums {
        if !converged_quorum_sha256s.contains(quorum.quorum_sha256()) {
            findings.push(TransparencyHeadFederationFinding::UnknownQuorum);
            invalid = true;
        }
    }

    let (evaluation_earliest, evaluation_latest) = evaluation_time.consensus_interval();
    if maximal_root_authority
        .as_ref()
        .is_some_and(|expected| expected != evaluation_time.root_authority_sha256())
    {
        findings.push(TransparencyHeadFederationFinding::EvaluationRootAuthorityMismatch);
        invalid = true;
    }
    if !maximal_views.is_empty() && maximal_latest > evaluation_earliest {
        findings.push(TransparencyHeadFederationFinding::MaxHeadNotDefinitelyBeforeEvaluation);
        blocked = true;
    }
    if !maximal_views.is_empty()
        && evaluation_latest > maximal_earliest.saturating_add(policy.maximum_head_age_s)
    {
        findings.push(TransparencyHeadFederationFinding::MaxHeadTooOld);
        stale = true;
    }

    if converged_view_sha256s.len() < policy.minimum_converged_views {
        findings.push(TransparencyHeadFederationFinding::InsufficientConvergedViews {
            actual: converged_view_sha256s.len(),
            required: policy.minimum_converged_views,
        });
        incomplete = true;
    }
    if converged_quorum_sha256s.len() < policy.minimum_distinct_quorums {
        findings.push(TransparencyHeadFederationFinding::InsufficientDistinctQuorums {
            actual: converged_quorum_sha256s.len(),
            required: policy.minimum_distinct_quorums,
        });
        incomplete = true;
    }
    if principal_ids.len() < policy.minimum_distinct_principals {
        findings.push(TransparencyHeadFederationFinding::InsufficientDistinctPrincipals {
            actual: principal_ids.len(),
            required: policy.minimum_distinct_principals,
        });
        incomplete = true;
    }
    if organization_ids.len() < policy.minimum_distinct_organizations {
        findings.push(TransparencyHeadFederationFinding::InsufficientDistinctOrganizations {
            actual: organization_ids.len(),
            required: policy.minimum_distinct_organizations,
        });
        incomplete = true;
    }
    if region_ids.len() < policy.minimum_distinct_regions {
        findings.push(TransparencyHeadFederationFinding::InsufficientDistinctRegions {
            actual: region_ids.len(),
            required: policy.minimum_distinct_regions,
        });
        incomplete = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        TransparencyHeadFederationClosure::Invalid
    } else if blocked {
        TransparencyHeadFederationClosure::Blocked
    } else if stale {
        TransparencyHeadFederationClosure::Stale
    } else if incomplete {
        TransparencyHeadFederationClosure::Incomplete
    } else {
        TransparencyHeadFederationClosure::ConvergedFreshObservedHead
    };

    converged_view_sha256s.sort();
    let converged_witness_quorum_sha256s: Vec<_> = converged_quorum_sha256s.into_iter().collect();
    let distinct_principal_ids: Vec<_> = principal_ids.into_iter().collect();
    let distinct_organization_ids: Vec<_> = organization_ids.into_iter().collect();
    let distinct_region_ids: Vec<_> = region_ids.into_iter().collect();
    let receipt_sha256 = head_federation_digest(
        &namespace_sha256,
        monitor,
        evaluation_time,
        maximal_tree_size,
        &maximal_root_sha256,
        &maximal_tree_head_sha256,
        &converged_view_sha256s,
        &converged_witness_quorum_sha256s,
        &distinct_principal_ids,
        &distinct_organization_ids,
        &distinct_region_ids,
        policy,
        &findings,
        closure,
    );

    Ok(TransparencyHeadFederationReceipt {
        namespace_sha256,
        monitor_receipt_sha256: monitor.receipt_sha256().clone(),
        evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
        evaluation_earliest_unix_s: evaluation_earliest,
        evaluation_latest_unix_s: evaluation_latest,
        maximal_tree_size,
        maximal_root_sha256,
        maximal_tree_head_sha256,
        converged_namespaced_view_sha256s: converged_view_sha256s,
        converged_witness_quorum_sha256s,
        distinct_principal_ids,
        distinct_organization_ids,
        distinct_region_ids,
        findings,
        closure,
        receipt_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn head_federation_digest(
    namespace_sha256: &Sha256Digest,
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    maximal_tree_size: u64,
    maximal_root_sha256: &Sha256Digest,
    maximal_tree_head_sha256: &Sha256Digest,
    views: &[Sha256Digest],
    quorums: &[Sha256Digest],
    principals: &[String],
    organizations: &[String],
    regions: &[String],
    policy: &TransparencyHeadFederationPolicy,
    findings: &[TransparencyHeadFederationFinding],
    closure: TransparencyHeadFederationClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(HEAD_FEDERATION_DOMAIN);
    digest.text(namespace_sha256.as_str());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    let (evaluation_earliest, evaluation_latest) = evaluation_time.consensus_interval();
    digest.text(&evaluation_earliest.to_string());
    digest.text(&evaluation_latest.to_string());
    digest.text(&maximal_tree_size.to_string());
    digest.text(maximal_root_sha256.as_str());
    digest.text(maximal_tree_head_sha256.as_str());
    for value in views { digest.text(value.as_str()); }
    for value in quorums { digest.text(value.as_str()); }
    for value in principals { digest.text(value); }
    for value in organizations { digest.text(value); }
    for value in regions { digest.text(value); }
    digest.text(&policy.minimum_converged_views.to_string());
    digest.text(&policy.minimum_distinct_quorums.to_string());
    digest.text(&policy.minimum_distinct_principals.to_string());
    digest.text(&policy.minimum_distinct_organizations.to_string());
    digest.text(&policy.minimum_distinct_regions.to_string());
    digest.text(&policy.maximum_head_age_s.to_string());
    digest.text(&policy.maximum_views.to_string());
    digest.text(&policy.maximum_quorums.to_string());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        TransparencyHeadFederationClosure::ConvergedFreshObservedHead => "converged-fresh-observed-head",
        TransparencyHeadFederationClosure::Stale => "stale",
        TransparencyHeadFederationClosure::Incomplete => "incomplete",
        TransparencyHeadFederationClosure::Blocked => "blocked",
        TransparencyHeadFederationClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-head-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("global-witness-independence-not-established");
    digest.digest()
}

fn digest_finding(digest: &mut FramedDigest, finding: &TransparencyHeadFederationFinding) {
    use TransparencyHeadFederationFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::EmptyViews => digest.text("empty-views"),
        Finding::TooManyViews => digest.text("too-many-views"),
        Finding::TooManyQuorums => digest.text("too-many-quorums"),
        Finding::DuplicateView => digest.text("duplicate-view"),
        Finding::DuplicateQuorum => digest.text("duplicate-quorum"),
        Finding::NamespaceMissing => digest.text("namespace-missing"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::ViewMissingFromMonitor => digest.text("view-missing-from-monitor"),
        Finding::MonitorIncomplete => digest.text("monitor-incomplete"),
        Finding::MonitorEquivocation => digest.text("monitor-equivocation"),
        Finding::MonitorTemporalConflict => digest.text("monitor-temporal-conflict"),
        Finding::MonitorInvalid => digest.text("monitor-invalid"),
        Finding::MissingViewQuorum => digest.text("missing-view-quorum"),
        Finding::UnknownQuorum => digest.text("unknown-quorum"),
        Finding::QuorumCheckpointMismatch => digest.text("quorum-checkpoint-mismatch"),
        Finding::QuorumRootAuthorityMismatch => digest.text("quorum-root-authority-mismatch"),
        Finding::EvaluationRootAuthorityMismatch => digest.text("evaluation-root-authority-mismatch"),
        Finding::MaxHeadNotDefinitelyBeforeEvaluation => digest.text("max-head-not-definitely-before-evaluation"),
        Finding::MaxHeadTooOld => digest.text("max-head-too-old"),
        Finding::InsufficientConvergedViews { actual, required } => {
            digest.text("insufficient-converged-views");
            digest.text(&actual.to_string());
            digest.text(&required.to_string());
        }
        Finding::InsufficientDistinctQuorums { actual, required } => {
            digest.text("insufficient-distinct-quorums");
            digest.text(&actual.to_string());
            digest.text(&required.to_string());
        }
        Finding::InsufficientDistinctPrincipals { actual, required } => {
            digest.text("insufficient-distinct-principals");
            digest.text(&actual.to_string());
            digest.text(&required.to_string());
        }
        Finding::InsufficientDistinctOrganizations { actual, required } => {
            digest.text("insufficient-distinct-organizations");
            digest.text(&actual.to_string());
            digest.text(&required.to_string());
        }
        Finding::InsufficientDistinctRegions { actual, required } => {
            digest.text("insufficient-distinct-regions");
            digest.text(&actual.to_string());
            digest.text(&required.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federation_policy_rejects_impossible_local_thresholds() {
        let policy = TransparencyHeadFederationPolicy {
            minimum_converged_views: 3,
            minimum_distinct_quorums: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 3,
            minimum_distinct_regions: 1,
            maximum_head_age_s: 60,
            maximum_views: 2,
            maximum_quorums: 2,
        };
        let issues = policy.validate().unwrap_err();
        assert!(issues.contains(&TransparencyHeadFederationPolicyIssue::ConvergedThresholdExceedsMaximumViews));
        assert!(issues.contains(&TransparencyHeadFederationPolicyIssue::OrganizationThresholdExceedsPrincipals));
    }

    #[test]
    fn positive_receipt_still_cannot_claim_global_currentness() {
        fn _assert_api(receipt: &TransparencyHeadFederationReceipt) {
            if receipt.converged_fresh_observed_head_established() {
                assert!(!receipt.globally_latest_head_established());
                assert!(!receipt.global_log_consistency_established());
                assert!(!receipt.global_witness_independence_established());
            }
        }
    }
}
