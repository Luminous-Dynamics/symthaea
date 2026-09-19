// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Monitor receipts over witnessed transparency checkpoints.
//!
//! This layer detects equivocation and verifies append-only growth across the
//! exact witnessed views supplied to it. V1 deliberately compares only views
//! under one exact root-authority lineage because the structural log does not yet
//! carry a root-independent stable log namespace. Cross-root views are therefore
//! incomparable rather than silently merged.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    AuthenticatedTransparencyCheckpoint, FramedDigest, Sha256Digest,
    TransparencyConsistencyProof, VerifiedTransparencyWitnessQuorum,
    verify_transparency_consistency,
};

const WITNESSED_CHECKPOINT_VIEW_DOMAIN: &str =
    "symthaea.witnessed-transparency-checkpoint-view.identity.v1";
const CONSISTENCY_LINK_DOMAIN: &str =
    "symthaea.transparency-monitor-consistency-link.identity.v1";
const MONITOR_RECEIPT_DOMAIN: &str = "symthaea.transparency-monitor-receipt.identity.v1";
pub const MAX_MONITORED_VIEWS: usize = 512;
pub const MAX_CONSISTENCY_LINKS: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessedCheckpointViewError {
    QuorumCheckpointMismatch,
    QuorumRootAuthorityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessedTransparencyCheckpoint {
    checkpoint_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    tree_head_sha256: Sha256Digest,
    tree_size: u64,
    root_sha256: Sha256Digest,
    consensus_earliest_unix_s: u64,
    consensus_latest_unix_s: u64,
    witness_quorum_sha256: Sha256Digest,
    view_sha256: Sha256Digest,
}

impl WitnessedTransparencyCheckpoint {
    pub fn new(
        checkpoint: &AuthenticatedTransparencyCheckpoint,
        quorum: &VerifiedTransparencyWitnessQuorum,
    ) -> Result<Self, WitnessedCheckpointViewError> {
        if quorum.checkpoint_sha256() != checkpoint.checkpoint_sha256() {
            return Err(WitnessedCheckpointViewError::QuorumCheckpointMismatch);
        }
        if quorum.root_authority_sha256() != checkpoint.root_authority_sha256() {
            return Err(WitnessedCheckpointViewError::QuorumRootAuthorityMismatch);
        }
        let statement = checkpoint.statement();
        let (earliest, latest) = statement.consensus_interval();
        let view_sha256 = witnessed_view_digest(
            checkpoint.checkpoint_sha256(),
            checkpoint.root_authority_sha256(),
            statement.tree_head_sha256(),
            statement.tree_size(),
            statement.root_sha256(),
            earliest,
            latest,
            quorum.quorum_sha256(),
        );
        Ok(Self {
            checkpoint_sha256: checkpoint.checkpoint_sha256().clone(),
            root_authority_sha256: checkpoint.root_authority_sha256().clone(),
            tree_head_sha256: statement.tree_head_sha256().clone(),
            tree_size: statement.tree_size(),
            root_sha256: statement.root_sha256().clone(),
            consensus_earliest_unix_s: earliest,
            consensus_latest_unix_s: latest,
            witness_quorum_sha256: quorum.quorum_sha256().clone(),
            view_sha256,
        })
    }

    pub fn checkpoint_sha256(&self) -> &Sha256Digest { &self.checkpoint_sha256 }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn tree_head_sha256(&self) -> &Sha256Digest { &self.tree_head_sha256 }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn root_sha256(&self) -> &Sha256Digest { &self.root_sha256 }
    pub fn consensus_interval(&self) -> (u64, u64) {
        (self.consensus_earliest_unix_s, self.consensus_latest_unix_s)
    }
    pub fn witness_quorum_sha256(&self) -> &Sha256Digest { &self.witness_quorum_sha256 }
    pub fn view_sha256(&self) -> &Sha256Digest { &self.view_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyConsistencyLinkError {
    RootAuthorityMismatch,
    NonIncreasingTreeSize,
    ProofSizeMismatch,
    ProofRootMismatch,
    InvalidConsistencyProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyMonitorConsistencyLink {
    first_view_sha256: Sha256Digest,
    second_view_sha256: Sha256Digest,
    first_tree_size: u64,
    second_tree_size: u64,
    first_root_sha256: Sha256Digest,
    second_root_sha256: Sha256Digest,
    proof_path_sha256: Sha256Digest,
    link_sha256: Sha256Digest,
}

impl TransparencyMonitorConsistencyLink {
    pub fn new(
        first: &WitnessedTransparencyCheckpoint,
        second: &WitnessedTransparencyCheckpoint,
        proof: &TransparencyConsistencyProof,
    ) -> Result<Self, TransparencyConsistencyLinkError> {
        if first.root_authority_sha256() != second.root_authority_sha256() {
            return Err(TransparencyConsistencyLinkError::RootAuthorityMismatch);
        }
        if first.tree_size() >= second.tree_size() {
            return Err(TransparencyConsistencyLinkError::NonIncreasingTreeSize);
        }
        if proof.first_tree_size() != first.tree_size()
            || proof.second_tree_size() != second.tree_size()
        {
            return Err(TransparencyConsistencyLinkError::ProofSizeMismatch);
        }
        if proof.first_root_sha256() != first.root_sha256()
            || proof.second_root_sha256() != second.root_sha256()
        {
            return Err(TransparencyConsistencyLinkError::ProofRootMismatch);
        }
        verify_transparency_consistency(proof)
            .map_err(|_| TransparencyConsistencyLinkError::InvalidConsistencyProof)?;

        let proof_path_sha256 = consistency_path_digest(proof);
        let link_sha256 = consistency_link_digest(
            first.view_sha256(),
            second.view_sha256(),
            first.tree_size(),
            second.tree_size(),
            first.root_sha256(),
            second.root_sha256(),
            &proof_path_sha256,
        );
        Ok(Self {
            first_view_sha256: first.view_sha256().clone(),
            second_view_sha256: second.view_sha256().clone(),
            first_tree_size: first.tree_size(),
            second_tree_size: second.tree_size(),
            first_root_sha256: first.root_sha256().clone(),
            second_root_sha256: second.root_sha256().clone(),
            proof_path_sha256,
            link_sha256,
        })
    }

    pub fn first_view_sha256(&self) -> &Sha256Digest { &self.first_view_sha256 }
    pub fn second_view_sha256(&self) -> &Sha256Digest { &self.second_view_sha256 }
    pub fn first_tree_size(&self) -> u64 { self.first_tree_size }
    pub fn second_tree_size(&self) -> u64 { self.second_tree_size }
    pub fn link_sha256(&self) -> &Sha256Digest { &self.link_sha256 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TransparencyMonitorClosure {
    AppendOnlyConsistentWithinObservedViews,
    EquivocationObserved,
    TemporalConflictObserved,
    Incomplete,
    IncomparableAuthority,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum TransparencyMonitorFinding {
    EmptyViews,
    TooManyViews,
    TooManyLinks,
    DuplicateView,
    IncomparableRootAuthority,
    SameSizeEquivocation { tree_size: u64 },
    TemporalRegression { first_tree_size: u64, second_tree_size: u64 },
    MissingConsistencyLink { first_tree_size: u64, second_tree_size: u64 },
    DuplicateConsistencyLink,
    UnknownConsistencyLinkEndpoint,
    ConsistencyLinkSizeMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyMonitorReceipt {
    root_authority_sha256s: Vec<Sha256Digest>,
    view_sha256s: Vec<Sha256Digest>,
    consistency_link_sha256s: Vec<Sha256Digest>,
    findings: Vec<TransparencyMonitorFinding>,
    closure: TransparencyMonitorClosure,
    receipt_sha256: Sha256Digest,
}

impl TransparencyMonitorReceipt {
    pub fn root_authority_sha256s(&self) -> &[Sha256Digest] {
        &self.root_authority_sha256s
    }
    pub fn view_sha256s(&self) -> &[Sha256Digest] { &self.view_sha256s }
    pub fn consistency_link_sha256s(&self) -> &[Sha256Digest] {
        &self.consistency_link_sha256s
    }
    pub fn findings(&self) -> &[TransparencyMonitorFinding] { &self.findings }
    pub fn closure(&self) -> TransparencyMonitorClosure { self.closure }
    pub fn receipt_sha256(&self) -> &Sha256Digest { &self.receipt_sha256 }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub fn observed_append_only_consistency_established(&self) -> bool {
        self.closure == TransparencyMonitorClosure::AppendOnlyConsistentWithinObservedViews
    }
}

pub fn monitor_witnessed_checkpoints(
    views: &[WitnessedTransparencyCheckpoint],
    links: &[TransparencyMonitorConsistencyLink],
) -> TransparencyMonitorReceipt {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut incomparable = false;
    let mut equivocation = false;
    let mut temporal_conflict = false;

    if views.is_empty() {
        findings.push(TransparencyMonitorFinding::EmptyViews);
        incomplete = true;
    }
    if views.len() > MAX_MONITORED_VIEWS {
        findings.push(TransparencyMonitorFinding::TooManyViews);
        invalid = true;
    }
    if links.len() > MAX_CONSISTENCY_LINKS {
        findings.push(TransparencyMonitorFinding::TooManyLinks);
        invalid = true;
    }

    let mut seen_views = BTreeSet::new();
    let mut root_authorities = BTreeSet::new();
    let mut roots_by_size: BTreeMap<u64, BTreeSet<Sha256Digest>> = BTreeMap::new();
    for view in views {
        if !seen_views.insert(view.view_sha256().clone()) {
            findings.push(TransparencyMonitorFinding::DuplicateView);
            invalid = true;
        }
        root_authorities.insert(view.root_authority_sha256().clone());
        roots_by_size
            .entry(view.tree_size())
            .or_default()
            .insert(view.root_sha256().clone());
    }
    if root_authorities.len() > 1 {
        findings.push(TransparencyMonitorFinding::IncomparableRootAuthority);
        incomparable = true;
    }
    for (tree_size, roots) in &roots_by_size {
        if roots.len() > 1 {
            findings.push(TransparencyMonitorFinding::SameSizeEquivocation {
                tree_size: *tree_size,
            });
            equivocation = true;
        }
    }

    let view_by_sha: BTreeMap<_, _> = views
        .iter()
        .map(|view| (view.view_sha256().clone(), view))
        .collect();
    let mut link_pairs = BTreeSet::new();
    for link in links {
        let pair = (
            link.first_view_sha256().clone(),
            link.second_view_sha256().clone(),
        );
        if !link_pairs.insert(pair.clone()) {
            findings.push(TransparencyMonitorFinding::DuplicateConsistencyLink);
            invalid = true;
            continue;
        }
        let (Some(first), Some(second)) =
            (view_by_sha.get(&pair.0), view_by_sha.get(&pair.1))
        else {
            findings.push(TransparencyMonitorFinding::UnknownConsistencyLinkEndpoint);
            invalid = true;
            continue;
        };
        if first.tree_size() != link.first_tree_size()
            || second.tree_size() != link.second_tree_size()
        {
            findings.push(TransparencyMonitorFinding::ConsistencyLinkSizeMismatch);
            invalid = true;
        }
    }

    let mut canonical_by_size = Vec::new();
    for tree_size in roots_by_size.keys() {
        if let Some(view) = views
            .iter()
            .filter(|view| view.tree_size() == *tree_size)
            .min_by(|left, right| left.view_sha256().cmp(right.view_sha256()))
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
            findings.push(TransparencyMonitorFinding::TemporalRegression {
                first_tree_size: first.tree_size(),
                second_tree_size: second.tree_size(),
            });
            temporal_conflict = true;
        }
        if !link_pairs.contains(&(
            first.view_sha256().clone(),
            second.view_sha256().clone(),
        )) {
            findings.push(TransparencyMonitorFinding::MissingConsistencyLink {
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
    let closure = if invalid {
        TransparencyMonitorClosure::Invalid
    } else if equivocation {
        TransparencyMonitorClosure::EquivocationObserved
    } else if incomparable {
        TransparencyMonitorClosure::IncomparableAuthority
    } else if temporal_conflict {
        TransparencyMonitorClosure::TemporalConflictObserved
    } else if incomplete {
        TransparencyMonitorClosure::Incomplete
    } else {
        TransparencyMonitorClosure::AppendOnlyConsistentWithinObservedViews
    };

    let root_authority_sha256s: Vec<_> = root_authorities.into_iter().collect();
    let mut view_sha256s: Vec<_> = views.iter().map(|view| view.view_sha256().clone()).collect();
    view_sha256s.sort();
    let mut consistency_link_sha256s: Vec<_> =
        links.iter().map(|link| link.link_sha256().clone()).collect();
    consistency_link_sha256s.sort();
    let receipt_sha256 = monitor_receipt_digest(
        &root_authority_sha256s,
        &view_sha256s,
        &consistency_link_sha256s,
        &findings,
        closure,
    );
    TransparencyMonitorReceipt {
        root_authority_sha256s,
        view_sha256s,
        consistency_link_sha256s,
        findings,
        closure,
        receipt_sha256,
    }
}

#[allow(clippy::too_many_arguments)]
fn witnessed_view_digest(
    checkpoint_sha256: &Sha256Digest,
    root_authority_sha256: &Sha256Digest,
    tree_head_sha256: &Sha256Digest,
    tree_size: u64,
    root_sha256: &Sha256Digest,
    earliest: u64,
    latest: u64,
    quorum_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(WITNESSED_CHECKPOINT_VIEW_DOMAIN);
    digest.text(checkpoint_sha256.as_str());
    digest.text(root_authority_sha256.as_str());
    digest.text(tree_head_sha256.as_str());
    digest.text(&tree_size.to_string());
    digest.text(root_sha256.as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(quorum_sha256.as_str());
    digest.digest()
}

fn consistency_path_digest(proof: &TransparencyConsistencyProof) -> Sha256Digest {
    let mut digest = FramedDigest::new(CONSISTENCY_LINK_DOMAIN);
    digest.text("proof-path");
    for node in proof.path() {
        digest.text(node.as_str());
    }
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn consistency_link_digest(
    first_view_sha256: &Sha256Digest,
    second_view_sha256: &Sha256Digest,
    first_tree_size: u64,
    second_tree_size: u64,
    first_root_sha256: &Sha256Digest,
    second_root_sha256: &Sha256Digest,
    proof_path_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(CONSISTENCY_LINK_DOMAIN);
    digest.text(first_view_sha256.as_str());
    digest.text(second_view_sha256.as_str());
    digest.text(&first_tree_size.to_string());
    digest.text(&second_tree_size.to_string());
    digest.text(first_root_sha256.as_str());
    digest.text(second_root_sha256.as_str());
    digest.text(proof_path_sha256.as_str());
    digest.digest()
}

fn monitor_receipt_digest(
    root_authority_sha256s: &[Sha256Digest],
    view_sha256s: &[Sha256Digest],
    link_sha256s: &[Sha256Digest],
    findings: &[TransparencyMonitorFinding],
    closure: TransparencyMonitorClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(MONITOR_RECEIPT_DOMAIN);
    for value in root_authority_sha256s {
        digest.text("root-authority");
        digest.text(value.as_str());
    }
    for value in view_sha256s {
        digest.text("view");
        digest.text(value.as_str());
    }
    for value in link_sha256s {
        digest.text("link");
        digest.text(value.as_str());
    }
    for finding in findings {
        digest_monitor_finding(&mut digest, finding);
    }
    digest.text(match closure {
        TransparencyMonitorClosure::AppendOnlyConsistentWithinObservedViews => "consistent",
        TransparencyMonitorClosure::EquivocationObserved => "equivocation",
        TransparencyMonitorClosure::TemporalConflictObserved => "temporal-conflict",
        TransparencyMonitorClosure::Incomplete => "incomplete",
        TransparencyMonitorClosure::IncomparableAuthority => "incomparable-authority",
        TransparencyMonitorClosure::Invalid => "invalid",
    });
    digest.text("global-log-consistency-not-established");
    digest.digest()
}

fn digest_monitor_finding(digest: &mut FramedDigest, finding: &TransparencyMonitorFinding) {
    match finding {
        TransparencyMonitorFinding::EmptyViews => digest.text("empty-views"),
        TransparencyMonitorFinding::TooManyViews => digest.text("too-many-views"),
        TransparencyMonitorFinding::TooManyLinks => digest.text("too-many-links"),
        TransparencyMonitorFinding::DuplicateView => digest.text("duplicate-view"),
        TransparencyMonitorFinding::IncomparableRootAuthority => {
            digest.text("incomparable-root-authority")
        }
        TransparencyMonitorFinding::SameSizeEquivocation { tree_size } => {
            digest.text("same-size-equivocation");
            digest.text(&tree_size.to_string());
        }
        TransparencyMonitorFinding::TemporalRegression {
            first_tree_size,
            second_tree_size,
        } => {
            digest.text("temporal-regression");
            digest.text(&first_tree_size.to_string());
            digest.text(&second_tree_size.to_string());
        }
        TransparencyMonitorFinding::MissingConsistencyLink {
            first_tree_size,
            second_tree_size,
        } => {
            digest.text("missing-consistency-link");
            digest.text(&first_tree_size.to_string());
            digest.text(&second_tree_size.to_string());
        }
        TransparencyMonitorFinding::DuplicateConsistencyLink => {
            digest.text("duplicate-consistency-link")
        }
        TransparencyMonitorFinding::UnknownConsistencyLinkEndpoint => {
            digest.text("unknown-consistency-link-endpoint")
        }
        TransparencyMonitorFinding::ConsistencyLinkSizeMismatch => {
            digest.text("consistency-link-size-mismatch")
        }
    }
}
