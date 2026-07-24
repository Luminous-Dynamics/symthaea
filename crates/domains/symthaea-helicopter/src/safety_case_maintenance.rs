// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safety-case maintenance and staleness control.
//!
//! A previously accepted safety case is not evergreen. This module binds
//! artifact revisions, review age, downstream impact, and closure evidence so
//! changed assumptions cannot silently retain old approvals.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyCaseArtifactKind {
    Hazard,
    Assumption,
    Requirement,
    Mitigation,
    Verification,
    Claim,
    Deployment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyCaseArtifactStatus {
    Accepted,
    Failed,
    Missing,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseArtifact {
    pub artifact_id: String,
    pub kind: SafetyCaseArtifactKind,
    pub revision: u64,
    pub digest: String,
    pub status: SafetyCaseArtifactStatus,
    pub reviewed_at_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub independent_review_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyCaseRelation {
    Supports,
    Verifies,
    Mitigates,
    DependsOn,
    InvalidatesWhenChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseLink {
    pub from_artifact_id: String,
    pub to_artifact_id: String,
    pub relation: SafetyCaseRelation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseChange {
    pub artifact_id: String,
    pub previous_revision: u64,
    pub current_revision: u64,
    pub changed_at_ms: u64,
    pub change_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseMaintenancePolicy {
    pub maximum_review_age_ms: BTreeMap<SafetyCaseArtifactKind, u64>,
    pub require_independent_review_for: BTreeSet<SafetyCaseArtifactKind>,
    pub required_root_kinds: BTreeSet<SafetyCaseArtifactKind>,
    pub propagate_relations: BTreeSet<SafetyCaseRelation>,
}

impl Default for SafetyCaseMaintenancePolicy {
    fn default() -> Self {
        Self {
            maximum_review_age_ms: BTreeMap::from([
                (SafetyCaseArtifactKind::Hazard, 365 * 24 * 60 * 60 * 1000),
                (SafetyCaseArtifactKind::Claim, 180 * 24 * 60 * 60 * 1000),
                (SafetyCaseArtifactKind::Deployment, 30 * 24 * 60 * 60 * 1000),
            ]),
            require_independent_review_for: BTreeSet::from([
                SafetyCaseArtifactKind::Hazard,
                SafetyCaseArtifactKind::Claim,
            ]),
            required_root_kinds: BTreeSet::from([
                SafetyCaseArtifactKind::Hazard,
                SafetyCaseArtifactKind::Claim,
                SafetyCaseArtifactKind::Deployment,
            ]),
            propagate_relations: BTreeSet::from([
                SafetyCaseRelation::DependsOn,
                SafetyCaseRelation::InvalidatesWhenChanged,
                SafetyCaseRelation::Supports,
                SafetyCaseRelation::Verifies,
                SafetyCaseRelation::Mitigates,
            ]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyCaseMaintenanceStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyCaseMaintenanceIssue {
    EmptyIdentity,
    DuplicateArtifact(String),
    DuplicateChange(String),
    MissingRootKind(SafetyCaseArtifactKind),
    BrokenLink {
        from: String,
        to: String,
    },
    InvalidRevisionChange(String),
    MissingChangeEvidence(String),
    ArtifactFailed(String),
    ArtifactMissing(String),
    ArtifactSuperseded(String),
    ArtifactExpired(String),
    ReviewStale {
        artifact_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    MissingIndependentReview(String),
    DownstreamNotRevalidated {
        changed_artifact_id: String,
        affected_artifact_id: String,
    },
    OrphanArtifact(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCaseMaintenanceReport {
    pub status: SafetyCaseMaintenanceStatus,
    pub artifacts: usize,
    pub changes: usize,
    pub affected_artifacts: Vec<String>,
    pub issues: Vec<SafetyCaseMaintenanceIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyCaseMaintenanceError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct SafetyCaseMaintainer {
    policy: SafetyCaseMaintenancePolicy,
}

impl SafetyCaseMaintainer {
    pub fn new(policy: SafetyCaseMaintenancePolicy) -> Result<Self, SafetyCaseMaintenanceError> {
        if policy.required_root_kinds.is_empty()
            || policy.propagate_relations.is_empty()
            || policy.maximum_review_age_ms.values().any(|age| *age == 0)
        {
            return Err(SafetyCaseMaintenanceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        artifacts: &[SafetyCaseArtifact],
        links: &[SafetyCaseLink],
        changes: &[SafetyCaseChange],
        now_ms: u64,
    ) -> SafetyCaseMaintenanceReport {
        let mut issues = Vec::new();
        let mut by_id = BTreeMap::<&str, &SafetyCaseArtifact>::new();
        let mut kinds = BTreeSet::new();
        for artifact in artifacts {
            kinds.insert(artifact.kind);
            if artifact.artifact_id.trim().is_empty() || artifact.digest.trim().is_empty() {
                issues.push(SafetyCaseMaintenanceIssue::EmptyIdentity);
            }
            if by_id
                .insert(artifact.artifact_id.as_str(), artifact)
                .is_some()
            {
                issues.push(SafetyCaseMaintenanceIssue::DuplicateArtifact(
                    artifact.artifact_id.clone(),
                ));
            }
            match artifact.status {
                SafetyCaseArtifactStatus::Accepted => {}
                SafetyCaseArtifactStatus::Failed => issues.push(
                    SafetyCaseMaintenanceIssue::ArtifactFailed(artifact.artifact_id.clone()),
                ),
                SafetyCaseArtifactStatus::Missing => issues.push(
                    SafetyCaseMaintenanceIssue::ArtifactMissing(artifact.artifact_id.clone()),
                ),
                SafetyCaseArtifactStatus::Superseded => issues.push(
                    SafetyCaseMaintenanceIssue::ArtifactSuperseded(artifact.artifact_id.clone()),
                ),
            }
            if artifact
                .valid_until_ms
                .is_some_and(|expiry| now_ms > expiry)
            {
                issues.push(SafetyCaseMaintenanceIssue::ArtifactExpired(
                    artifact.artifact_id.clone(),
                ));
            }
            if let Some(maximum) = self.policy.maximum_review_age_ms.get(&artifact.kind) {
                let age = now_ms.saturating_sub(artifact.reviewed_at_ms);
                if age > *maximum {
                    issues.push(SafetyCaseMaintenanceIssue::ReviewStale {
                        artifact_id: artifact.artifact_id.clone(),
                        age_ms: age,
                        maximum_ms: *maximum,
                    });
                }
            }
            if self
                .policy
                .require_independent_review_for
                .contains(&artifact.kind)
                && artifact
                    .independent_review_id
                    .as_ref()
                    .is_none_or(|id| id.trim().is_empty())
            {
                issues.push(SafetyCaseMaintenanceIssue::MissingIndependentReview(
                    artifact.artifact_id.clone(),
                ));
            }
        }

        for kind in &self.policy.required_root_kinds {
            if !kinds.contains(kind) {
                issues.push(SafetyCaseMaintenanceIssue::MissingRootKind(*kind));
            }
        }

        let mut incoming = BTreeMap::<&str, usize>::new();
        let mut outgoing = BTreeMap::<&str, Vec<&SafetyCaseLink>>::new();
        for link in links {
            if !by_id.contains_key(link.from_artifact_id.as_str())
                || !by_id.contains_key(link.to_artifact_id.as_str())
            {
                issues.push(SafetyCaseMaintenanceIssue::BrokenLink {
                    from: link.from_artifact_id.clone(),
                    to: link.to_artifact_id.clone(),
                });
                continue;
            }
            *incoming.entry(link.to_artifact_id.as_str()).or_default() += 1;
            outgoing
                .entry(link.from_artifact_id.as_str())
                .or_default()
                .push(link);
        }
        for artifact in artifacts {
            if !self.policy.required_root_kinds.contains(&artifact.kind)
                && incoming
                    .get(artifact.artifact_id.as_str())
                    .copied()
                    .unwrap_or(0)
                    == 0
            {
                issues.push(SafetyCaseMaintenanceIssue::OrphanArtifact(
                    artifact.artifact_id.clone(),
                ));
            }
        }

        let mut change_ids = BTreeSet::new();
        let mut affected = BTreeSet::new();
        for change in changes {
            if !change_ids.insert(change.artifact_id.as_str()) {
                issues.push(SafetyCaseMaintenanceIssue::DuplicateChange(
                    change.artifact_id.clone(),
                ));
            }
            if change.current_revision <= change.previous_revision
                || by_id
                    .get(change.artifact_id.as_str())
                    .is_none_or(|artifact| artifact.revision != change.current_revision)
            {
                issues.push(SafetyCaseMaintenanceIssue::InvalidRevisionChange(
                    change.artifact_id.clone(),
                ));
            }
            if change.change_evidence_id.trim().is_empty() {
                issues.push(SafetyCaseMaintenanceIssue::MissingChangeEvidence(
                    change.artifact_id.clone(),
                ));
            }

            let mut queue = VecDeque::from([change.artifact_id.as_str()]);
            let mut visited = BTreeSet::new();
            while let Some(current) = queue.pop_front() {
                if !visited.insert(current) {
                    continue;
                }
                for link in outgoing.get(current).into_iter().flatten() {
                    if !self.policy.propagate_relations.contains(&link.relation) {
                        continue;
                    }
                    let target = link.to_artifact_id.as_str();
                    affected.insert(target.to_string());
                    if let Some(artifact) = by_id.get(target) {
                        if artifact.reviewed_at_ms < change.changed_at_ms {
                            issues.push(SafetyCaseMaintenanceIssue::DownstreamNotRevalidated {
                                changed_artifact_id: change.artifact_id.clone(),
                                affected_artifact_id: target.to_string(),
                            });
                        }
                    }
                    queue.push_back(target);
                }
            }
        }

        let status = if issues.iter().any(is_failure) {
            SafetyCaseMaintenanceStatus::Fail
        } else if issues.is_empty() {
            SafetyCaseMaintenanceStatus::Pass
        } else {
            SafetyCaseMaintenanceStatus::Incomplete
        };
        SafetyCaseMaintenanceReport {
            status,
            artifacts: artifacts.len(),
            changes: changes.len(),
            affected_artifacts: affected.into_iter().collect(),
            issues,
        }
    }
}

fn is_failure(issue: &SafetyCaseMaintenanceIssue) -> bool {
    matches!(
        issue,
        SafetyCaseMaintenanceIssue::InvalidRevisionChange(_)
            | SafetyCaseMaintenanceIssue::ArtifactFailed(_)
            | SafetyCaseMaintenanceIssue::ArtifactExpired(_)
            | SafetyCaseMaintenanceIssue::DownstreamNotRevalidated { .. }
            | SafetyCaseMaintenanceIssue::BrokenLink { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str, kind: SafetyCaseArtifactKind, reviewed_at_ms: u64) -> SafetyCaseArtifact {
        SafetyCaseArtifact {
            artifact_id: id.into(),
            kind,
            revision: 1,
            digest: format!("sha256:{id}"),
            status: SafetyCaseArtifactStatus::Accepted,
            reviewed_at_ms,
            valid_until_ms: Some(10_000),
            independent_review_id: matches!(
                kind,
                SafetyCaseArtifactKind::Hazard | SafetyCaseArtifactKind::Claim
            )
            .then(|| format!("review-{id}")),
        }
    }

    fn policy() -> SafetyCaseMaintenancePolicy {
        SafetyCaseMaintenancePolicy {
            maximum_review_age_ms: BTreeMap::from([
                (SafetyCaseArtifactKind::Hazard, 5_000),
                (SafetyCaseArtifactKind::Claim, 5_000),
                (SafetyCaseArtifactKind::Deployment, 5_000),
            ]),
            ..SafetyCaseMaintenancePolicy::default()
        }
    }

    #[test]
    fn revalidated_change_chain_passes() {
        let mut hazard = artifact("hazard", SafetyCaseArtifactKind::Hazard, 3_000);
        hazard.revision = 2;
        let artifacts = vec![
            hazard,
            artifact("verification", SafetyCaseArtifactKind::Verification, 3_100),
            artifact("claim", SafetyCaseArtifactKind::Claim, 3_200),
            artifact("deployment", SafetyCaseArtifactKind::Deployment, 3_300),
        ];
        let links = vec![
            SafetyCaseLink {
                from_artifact_id: "hazard".into(),
                to_artifact_id: "verification".into(),
                relation: SafetyCaseRelation::InvalidatesWhenChanged,
            },
            SafetyCaseLink {
                from_artifact_id: "verification".into(),
                to_artifact_id: "claim".into(),
                relation: SafetyCaseRelation::Supports,
            },
            SafetyCaseLink {
                from_artifact_id: "claim".into(),
                to_artifact_id: "deployment".into(),
                relation: SafetyCaseRelation::Supports,
            },
        ];
        let report = SafetyCaseMaintainer::new(policy()).unwrap().assess(
            &artifacts,
            &links,
            &[SafetyCaseChange {
                artifact_id: "hazard".into(),
                previous_revision: 1,
                current_revision: 2,
                changed_at_ms: 3_000,
                change_evidence_id: "change-1".into(),
            }],
            4_000,
        );
        assert_eq!(report.status, SafetyCaseMaintenanceStatus::Pass);
    }

    #[test]
    fn stale_downstream_approval_fails() {
        let mut hazard = artifact("hazard", SafetyCaseArtifactKind::Hazard, 3_000);
        hazard.revision = 2;
        let artifacts = vec![
            hazard,
            artifact("claim", SafetyCaseArtifactKind::Claim, 2_000),
            artifact("deployment", SafetyCaseArtifactKind::Deployment, 3_500),
        ];
        let links = vec![SafetyCaseLink {
            from_artifact_id: "hazard".into(),
            to_artifact_id: "claim".into(),
            relation: SafetyCaseRelation::InvalidatesWhenChanged,
        }];
        let report = SafetyCaseMaintainer::new(policy()).unwrap().assess(
            &artifacts,
            &links,
            &[SafetyCaseChange {
                artifact_id: "hazard".into(),
                previous_revision: 1,
                current_revision: 2,
                changed_at_ms: 3_000,
                change_evidence_id: "change-1".into(),
            }],
            4_000,
        );
        assert_eq!(report.status, SafetyCaseMaintenanceStatus::Fail);
    }
}
