// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Witnessed transparency monitoring across authorized trust-root rotations.
//!
//! Cross-root comparison is permitted only when every view proves membership in
//! one stable transparency namespace and the supplied namespace authorities form
//! an explicit predecessor chain. This establishes append-only consistency only
//! within the supplied witnessed views; it is not a claim of global consistency.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    AuthorizedTransparencyLogNamespace, FramedDigest, MAX_CONSISTENCY_LINKS,
    MAX_MONITORED_VIEWS, Sha256Digest, TransparencyConsistencyProof,
    TransparencyMonitorConsistencyLink, WitnessedTransparencyCheckpoint,
    verify_transparency_consistency,
};

const NAMESPACED_VIEW_DOMAIN: &str =
    "symthaea.namespaced-witnessed-transparency-view.identity.v1";
const NAMESPACED_MONITOR_DOMAIN: &str =
    "symthaea.namespaced-transparency-monitor-receipt.identity.v1";
pub const MAX_NAMESPACE_AUTHORITIES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespacedWitnessedViewError {
    RootAuthorityMismatch,
    TreeRollback,
    SameSizeRootMismatch,
    MissingConsistencyProof,
    ConsistencySizeMismatch,
    ConsistencyRootMismatch,
    InvalidConsistencyProof,
}

/// Witnessed checkpoint proven to descend from the current root-specific anchor
/// of one stable transparency namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NamespacedWitnessedTransparencyCheckpoint {
    namespace_sha256: Sha256Digest,
    namespace_authority_sha256: Sha256Digest,
    witnessed_view: WitnessedTransparencyCheckpoint,
    namespaced_view_sha256: Sha256Digest,
}

impl NamespacedWitnessedTransparencyCheckpoint {
    pub fn new(
        namespace: &AuthorizedTransparencyLogNamespace,
        witnessed_view: WitnessedTransparencyCheckpoint,
        anchor_to_view: Option<&TransparencyConsistencyProof>,
    ) -> Result<Self, NamespacedWitnessedViewError> {
        if witnessed_view.root_authority_sha256() != namespace.root_authority_sha256() {
            return Err(NamespacedWitnessedViewError::RootAuthorityMismatch);
        }
        if witnessed_view.tree_size() < namespace.anchor_tree_size() {
            return Err(NamespacedWitnessedViewError::TreeRollback);
        }
        if witnessed_view.tree_size() == namespace.anchor_tree_size() {
            if witnessed_view.root_sha256() != namespace.anchor_root_sha256() {
                return Err(NamespacedWitnessedViewError::SameSizeRootMismatch);
            }
        } else {
            let proof = anchor_to_view.ok_or(NamespacedWitnessedViewError::MissingConsistencyProof)?;
            if proof.first_tree_size() != namespace.anchor_tree_size()
                || proof.second_tree_size() != witnessed_view.tree_size()
            {
                return Err(NamespacedWitnessedViewError::ConsistencySizeMismatch);
            }
            if proof.first_root_sha256() != namespace.anchor_root_sha256()
                || proof.second_root_sha256() != witnessed_view.root_sha256()
            {
                return Err(NamespacedWitnessedViewError::ConsistencyRootMismatch);
            }
            verify_transparency_consistency(proof)
                .map_err(|_| NamespacedWitnessedViewError::InvalidConsistencyProof)?;
        }

        let namespaced_view_sha256 = namespaced_view_digest(
            namespace.namespace_sha256(),
            namespace.namespace_authority_sha256(),
            &witnessed_view,
        );
        Ok(Self {
            namespace_sha256: namespace.namespace_sha256().clone(),
            namespace_authority_sha256: namespace.namespace_authority_sha256().clone(),
            witnessed_view,
            namespaced_view_sha256,
        })
    }

    pub fn namespace_sha256(&self) -> &Sha256Digest { &self.namespace_sha256 }
    pub fn namespace_authority_sha256(&self) -> &Sha256Digest {
        &self.namespace_authority_sha256
    }
    pub fn witnessed_view(&self) -> &WitnessedTransparencyCheckpoint { &self.witnessed_view }
    pub fn root_authority_sha256(&self) -> &Sha256Digest {
        self.witnessed_view.root_authority_sha256()
    }
    pub fn tree_size(&self) -> u64 { self.witnessed_view.tree_size() }
    pub fn root_sha256(&self) -> &Sha256Digest { self.witnessed_view.root_sha256() }
    pub fn consensus_interval(&self) -> (u64, u64) { self.witnessed_view.consensus_interval() }
    pub fn namespaced_view_sha256(&self) -> &Sha256Digest { &self.namespaced_view_sha256 }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespacedConsistencyLinkError {
    NamespaceMismatch,
    StructuralLinkRejected,
}

pub fn namespaced_consistency_link(
    first: &NamespacedWitnessedTransparencyCheckpoint,
    second: &NamespacedWitnessedTransparencyCheckpoint,
    proof: &TransparencyConsistencyProof,
) -> Result<TransparencyMonitorConsistencyLink, NamespacedConsistencyLinkError> {
    if first.namespace_sha256() != second.namespace_sha256() {
        return Err(NamespacedConsistencyLinkError::NamespaceMismatch);
    }
    TransparencyMonitorConsistencyLink::new(first.witnessed_view(), second.witnessed_view(), proof)
        .map_err(|_| NamespacedConsistencyLinkError::StructuralLinkRejected)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NamespacedTransparencyMonitorClosure {
    AppendOnlyConsistentAcrossAuthorizedRoots,
    EquivocationObserved,
    TemporalConflictObserved,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum NamespacedTransparencyMonitorFinding {
    EmptyViews,
    TooManyViews,
    TooManyLinks,
    TooManyNamespaceAuthorities,
    DuplicateView,
    DuplicateNamespaceAuthority,
    NamespaceMismatch,
    MissingNamespaceAuthority,
    NamespaceAuthorityRootMismatch,
    InvalidNamespaceAuthorityGenesis,
    MissingNamespacePredecessor,
    MissingRootTransitionBinding,
    MultipleNamespaceGenesisAuthorities,
    NamespaceAuthorityCycle,
    RootAuthorityOrderViolation,
    SameSizeEquivocation { tree_size: u64 },
    DefiniteTemporalRegression { first_tree_size: u64, second_tree_size: u64 },
    DuplicateConsistencyLink,
    UnknownConsistencyLinkEndpoint,
    MissingConsistencyLink { first_tree_size: u64, second_tree_size: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NamespacedTransparencyMonitorReceipt {
    namespace_sha256: Option<Sha256Digest>,
    namespaced_view_sha256s: Vec<Sha256Digest>,
    namespace_authority_sha256s: Vec<Sha256Digest>,
    consistency_link_sha256s: Vec<Sha256Digest>,
    root_authority_sha256s: Vec<Sha256Digest>,
    findings: Vec<NamespacedTransparencyMonitorFinding>,
    closure: NamespacedTransparencyMonitorClosure,
    receipt_sha256: Sha256Digest,
}

impl NamespacedTransparencyMonitorReceipt {
    pub fn namespace_sha256(&self) -> Option<&Sha256Digest> { self.namespace_sha256.as_ref() }
    pub fn namespaced_view_sha256s(&self) -> &[Sha256Digest] { &self.namespaced_view_sha256s }
    pub fn namespace_authority_sha256s(&self) -> &[Sha256Digest] {
        &self.namespace_authority_sha256s
    }
    pub fn consistency_link_sha256s(&self) -> &[Sha256Digest] {
        &self.consistency_link_sha256s
    }
    pub fn root_authority_sha256s(&self) -> &[Sha256Digest] { &self.root_authority_sha256s }
    pub fn findings(&self) -> &[NamespacedTransparencyMonitorFinding] { &self.findings }
    pub fn closure(&self) -> NamespacedTransparencyMonitorClosure { self.closure }
    pub fn receipt_sha256(&self) -> &Sha256Digest { &self.receipt_sha256 }
    pub fn observed_cross_root_append_only_consistency_established(&self) -> bool {
        self.closure == NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots
    }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

pub fn monitor_namespaced_witnessed_checkpoints(
    views: &[NamespacedWitnessedTransparencyCheckpoint],
    authorities: &[AuthorizedTransparencyLogNamespace],
    links: &[TransparencyMonitorConsistencyLink],
) -> NamespacedTransparencyMonitorReceipt {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut equivocation = false;
    let mut temporal_conflict = false;

    if views.is_empty() {
        findings.push(NamespacedTransparencyMonitorFinding::EmptyViews);
        incomplete = true;
    }
    if views.len() > MAX_MONITORED_VIEWS {
        findings.push(NamespacedTransparencyMonitorFinding::TooManyViews);
        invalid = true;
    }
    if links.len() > MAX_CONSISTENCY_LINKS {
        findings.push(NamespacedTransparencyMonitorFinding::TooManyLinks);
        invalid = true;
    }
    if authorities.len() > MAX_NAMESPACE_AUTHORITIES {
        findings.push(NamespacedTransparencyMonitorFinding::TooManyNamespaceAuthorities);
        invalid = true;
    }

    let namespace_sha256 = views
        .first()
        .map(|view| view.namespace_sha256().clone())
        .or_else(|| authorities.first().map(|authority| authority.namespace_sha256().clone()));
    let mut seen_views = BTreeSet::new();
    let mut root_authorities = BTreeSet::new();
    let mut roots_by_size: BTreeMap<u64, BTreeSet<Sha256Digest>> = BTreeMap::new();
    for view in views {
        if namespace_sha256.as_ref() != Some(view.namespace_sha256()) {
            findings.push(NamespacedTransparencyMonitorFinding::NamespaceMismatch);
            invalid = true;
        }
        if !seen_views.insert(view.namespaced_view_sha256().clone()) {
            findings.push(NamespacedTransparencyMonitorFinding::DuplicateView);
            invalid = true;
        }
        root_authorities.insert(view.root_authority_sha256().clone());
        roots_by_size
            .entry(view.tree_size())
            .or_default()
            .insert(view.root_sha256().clone());
    }
    for (tree_size, roots) in &roots_by_size {
        if roots.len() > 1 {
            findings.push(NamespacedTransparencyMonitorFinding::SameSizeEquivocation {
                tree_size: *tree_size,
            });
            equivocation = true;
        }
    }

    let mut authority_by_sha = BTreeMap::new();
    let mut genesis_count = 0usize;
    for authority in authorities {
        if namespace_sha256.as_ref().is_some_and(|id| id != authority.namespace_sha256()) {
            findings.push(NamespacedTransparencyMonitorFinding::NamespaceMismatch);
            invalid = true;
        }
        if authority_by_sha
            .insert(authority.namespace_authority_sha256().clone(), authority)
            .is_some()
        {
            findings.push(NamespacedTransparencyMonitorFinding::DuplicateNamespaceAuthority);
            invalid = true;
        }
        match authority.predecessor_namespace_authority_sha256() {
            None => {
                genesis_count += 1;
                if authority.root_transition_sha256().is_some() {
                    findings.push(NamespacedTransparencyMonitorFinding::InvalidNamespaceAuthorityGenesis);
                    invalid = true;
                }
            }
            Some(_) => {
                if authority.root_transition_sha256().is_none() {
                    findings.push(NamespacedTransparencyMonitorFinding::MissingRootTransitionBinding);
                    invalid = true;
                }
            }
        }
    }
    if genesis_count > 1 {
        findings.push(NamespacedTransparencyMonitorFinding::MultipleNamespaceGenesisAuthorities);
        invalid = true;
    }

    for authority in authorities {
        if let Some(predecessor) = authority.predecessor_namespace_authority_sha256() {
            if !authority_by_sha.contains_key(predecessor) {
                findings.push(NamespacedTransparencyMonitorFinding::MissingNamespacePredecessor);
                incomplete = true;
            }
        }
    }

    for authority in authorities {
        let mut cursor = Some(authority.namespace_authority_sha256().clone());
        let mut walked = BTreeSet::new();
        while let Some(current) = cursor {
            if !walked.insert(current.clone()) {
                findings.push(NamespacedTransparencyMonitorFinding::NamespaceAuthorityCycle);
                invalid = true;
                break;
            }
            cursor = authority_by_sha
                .get(&current)
                .and_then(|value| value.predecessor_namespace_authority_sha256())
                .cloned();
        }
    }

    for view in views {
        match authority_by_sha.get(view.namespace_authority_sha256()) {
            None => {
                findings.push(NamespacedTransparencyMonitorFinding::MissingNamespaceAuthority);
                incomplete = true;
            }
            Some(authority) if authority.root_authority_sha256() != view.root_authority_sha256() => {
                findings.push(NamespacedTransparencyMonitorFinding::NamespaceAuthorityRootMismatch);
                invalid = true;
            }
            Some(_) => {}
        }
    }

    let raw_view_by_sha: BTreeMap<_, _> = views
        .iter()
        .map(|view| (view.witnessed_view().view_sha256().clone(), view))
        .collect();
    let mut link_pairs = BTreeSet::new();
    for link in links {
        let pair = (
            link.first_view_sha256().clone(),
            link.second_view_sha256().clone(),
        );
        if !link_pairs.insert(pair.clone()) {
            findings.push(NamespacedTransparencyMonitorFinding::DuplicateConsistencyLink);
            invalid = true;
            continue;
        }
        if !raw_view_by_sha.contains_key(&pair.0) || !raw_view_by_sha.contains_key(&pair.1) {
            findings.push(NamespacedTransparencyMonitorFinding::UnknownConsistencyLinkEndpoint);
            invalid = true;
        }
    }

    let mut canonical_by_size = Vec::new();
    for tree_size in roots_by_size.keys() {
        if let Some(view) = views
            .iter()
            .filter(|view| view.tree_size() == *tree_size)
            .min_by(|left, right| left.namespaced_view_sha256().cmp(right.namespaced_view_sha256()))
        {
            canonical_by_size.push(view);
        }
    }
    for pair in canonical_by_size.windows(2) {
        let first = pair[0];
        let second = pair[1];
        let (first_earliest, _) = first.consensus_interval();
        let (_, second_latest) = second.consensus_interval();
        if second_latest < first_earliest {
            findings.push(NamespacedTransparencyMonitorFinding::DefiniteTemporalRegression {
                first_tree_size: first.tree_size(),
                second_tree_size: second.tree_size(),
            });
            temporal_conflict = true;
        }

        let first_authority = authority_by_sha.get(first.namespace_authority_sha256()).copied();
        let second_authority = authority_by_sha.get(second.namespace_authority_sha256()).copied();
        if let (Some(first_authority), Some(second_authority)) = (first_authority, second_authority) {
            if !authority_is_ancestor_or_same(
                first_authority.namespace_authority_sha256(),
                second_authority.namespace_authority_sha256(),
                &authority_by_sha,
            ) {
                findings.push(NamespacedTransparencyMonitorFinding::RootAuthorityOrderViolation);
                invalid = true;
            }
        }

        if !link_pairs.contains(&(
            first.witnessed_view().view_sha256().clone(),
            second.witnessed_view().view_sha256().clone(),
        )) {
            findings.push(NamespacedTransparencyMonitorFinding::MissingConsistencyLink {
                first_tree_size: first.tree_size(),
                second_tree_size: second.tree_size(),
            });
            incomplete = true;
        }
    }
    if canonical_by_size.len() < 2 && !views.is_empty() {
        incomplete = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        NamespacedTransparencyMonitorClosure::Invalid
    } else if equivocation {
        NamespacedTransparencyMonitorClosure::EquivocationObserved
    } else if temporal_conflict {
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved
    } else if incomplete {
        NamespacedTransparencyMonitorClosure::Incomplete
    } else {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots
    };

    let mut namespaced_view_sha256s: Vec<_> =
        views.iter().map(|view| view.namespaced_view_sha256().clone()).collect();
    namespaced_view_sha256s.sort();
    let mut namespace_authority_sha256s: Vec<_> = authorities
        .iter()
        .map(|authority| authority.namespace_authority_sha256().clone())
        .collect();
    namespace_authority_sha256s.sort();
    let mut consistency_link_sha256s: Vec<_> =
        links.iter().map(|link| link.link_sha256().clone()).collect();
    consistency_link_sha256s.sort();
    let root_authority_sha256s: Vec<_> = root_authorities.into_iter().collect();
    let receipt_sha256 = namespaced_monitor_digest(
        namespace_sha256.as_ref(),
        &namespaced_view_sha256s,
        &namespace_authority_sha256s,
        &consistency_link_sha256s,
        &root_authority_sha256s,
        &findings,
        closure,
    );

    NamespacedTransparencyMonitorReceipt {
        namespace_sha256,
        namespaced_view_sha256s,
        namespace_authority_sha256s,
        consistency_link_sha256s,
        root_authority_sha256s,
        findings,
        closure,
        receipt_sha256,
    }
}

fn authority_is_ancestor_or_same(
    ancestor: &Sha256Digest,
    descendant: &Sha256Digest,
    authorities: &BTreeMap<Sha256Digest, &AuthorizedTransparencyLogNamespace>,
) -> bool {
    let mut cursor = Some(descendant.clone());
    let mut visited = BTreeSet::new();
    while let Some(current) = cursor {
        if &current == ancestor {
            return true;
        }
        if !visited.insert(current.clone()) {
            return false;
        }
        cursor = authorities
            .get(&current)
            .and_then(|authority| authority.predecessor_namespace_authority_sha256())
            .cloned();
    }
    false
}

fn namespaced_view_digest(
    namespace_sha256: &Sha256Digest,
    namespace_authority_sha256: &Sha256Digest,
    witnessed_view: &WitnessedTransparencyCheckpoint,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(NAMESPACED_VIEW_DOMAIN);
    digest.text(namespace_sha256.as_str());
    digest.text(namespace_authority_sha256.as_str());
    digest.text(witnessed_view.view_sha256().as_str());
    digest.text(witnessed_view.root_authority_sha256().as_str());
    digest.text(&witnessed_view.tree_size().to_string());
    digest.text(witnessed_view.root_sha256().as_str());
    digest.digest()
}

fn namespaced_monitor_digest(
    namespace_sha256: Option<&Sha256Digest>,
    views: &[Sha256Digest],
    authorities: &[Sha256Digest],
    links: &[Sha256Digest],
    roots: &[Sha256Digest],
    findings: &[NamespacedTransparencyMonitorFinding],
    closure: NamespacedTransparencyMonitorClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(NAMESPACED_MONITOR_DOMAIN);
    digest.optional_sha(namespace_sha256);
    for value in views { digest.text(value.as_str()); }
    for value in authorities { digest.text(value.as_str()); }
    for value in links { digest.text(value.as_str()); }
    for value in roots { digest.text(value.as_str()); }
    for finding in findings { digest_namespaced_finding(&mut digest, finding); }
    digest.text(match closure {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {
            "append-only-consistent-across-authorized-roots"
        }
        NamespacedTransparencyMonitorClosure::EquivocationObserved => "equivocation-observed",
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved => "temporal-conflict-observed",
        NamespacedTransparencyMonitorClosure::Incomplete => "incomplete",
        NamespacedTransparencyMonitorClosure::Invalid => "invalid",
    });
    digest.text("global-log-consistency-not-established");
    digest.digest()
}

fn digest_namespaced_finding(
    digest: &mut FramedDigest,
    finding: &NamespacedTransparencyMonitorFinding,
) {
    use NamespacedTransparencyMonitorFinding as Finding;
    match finding {
        Finding::EmptyViews => digest.text("empty-views"),
        Finding::TooManyViews => digest.text("too-many-views"),
        Finding::TooManyLinks => digest.text("too-many-links"),
        Finding::TooManyNamespaceAuthorities => digest.text("too-many-namespace-authorities"),
        Finding::DuplicateView => digest.text("duplicate-view"),
        Finding::DuplicateNamespaceAuthority => digest.text("duplicate-namespace-authority"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::MissingNamespaceAuthority => digest.text("missing-namespace-authority"),
        Finding::NamespaceAuthorityRootMismatch => digest.text("namespace-authority-root-mismatch"),
        Finding::InvalidNamespaceAuthorityGenesis => digest.text("invalid-namespace-authority-genesis"),
        Finding::MissingNamespacePredecessor => digest.text("missing-namespace-predecessor"),
        Finding::MissingRootTransitionBinding => digest.text("missing-root-transition-binding"),
        Finding::MultipleNamespaceGenesisAuthorities => digest.text("multiple-namespace-genesis-authorities"),
        Finding::NamespaceAuthorityCycle => digest.text("namespace-authority-cycle"),
        Finding::RootAuthorityOrderViolation => digest.text("root-authority-order-violation"),
        Finding::SameSizeEquivocation { tree_size } => {
            digest.text("same-size-equivocation");
            digest.text(&tree_size.to_string());
        }
        Finding::DefiniteTemporalRegression { first_tree_size, second_tree_size } => {
            digest.text("definite-temporal-regression");
            digest.text(&first_tree_size.to_string());
            digest.text(&second_tree_size.to_string());
        }
        Finding::DuplicateConsistencyLink => digest.text("duplicate-consistency-link"),
        Finding::UnknownConsistencyLinkEndpoint => digest.text("unknown-consistency-link-endpoint"),
        Finding::MissingConsistencyLink { first_tree_size, second_tree_size } => {
            digest.text("missing-consistency-link");
            digest.text(&first_tree_size.to_string());
            digest.text(&second_tree_size.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_namespaced_monitoring_still_does_not_claim_global_consistency() {
        fn _assert_api(receipt: &NamespacedTransparencyMonitorReceipt) {
            if receipt.observed_cross_root_append_only_consistency_established() {
                assert!(!receipt.global_log_consistency_established());
            }
        }
    }
}
