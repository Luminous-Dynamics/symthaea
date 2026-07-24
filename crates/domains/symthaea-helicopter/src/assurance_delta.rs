// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safety-assurance change-impact analysis.
//!
//! A new release must not inherit an old safety case by name. This module
//! compares assurance artifacts, applies declared impact rules, and requires
//! candidate evidence for every revalidation obligation created by a change.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AssuranceDeltaArtifactKind {
    Source,
    DependencyLock,
    BuildConfiguration,
    Calibration,
    HardwareContract,
    FlightModel,
    Controller,
    SafetyMonitor,
    OperationalLimit,
    Test,
    Hazard,
    Requirement,
    Claim,
    Deployment,
    QualificationEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceDeltaArtifact {
    pub artifact_id: String,
    pub kind: AssuranceDeltaArtifactKind,
    pub digest: String,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceDeltaSnapshot {
    pub snapshot_id: String,
    pub deployment_id: String,
    pub artifacts: Vec<AssuranceDeltaArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceImpactRule {
    pub changed_kind: AssuranceDeltaArtifactKind,
    pub required_revalidation_kinds: Vec<AssuranceDeltaArtifactKind>,
    pub permit_restriction: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceDeltaPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub critical_kinds: Vec<AssuranceDeltaArtifactKind>,
    pub impact_rules: Vec<AssuranceImpactRule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceDeltaChangeKind {
    Added,
    Removed,
    Modified,
    Unchanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceDeltaChange {
    pub artifact_id: String,
    pub kind: AssuranceDeltaArtifactKind,
    pub change: AssuranceDeltaChangeKind,
    pub baseline_digest: Option<String>,
    pub candidate_digest: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceDeltaStatus {
    Cleared,
    Restricted,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceDeltaIssue {
    InvalidSnapshotIdentity(String),
    DuplicateArtifact {
        snapshot_id: String,
        artifact_id: String,
    },
    InvalidDigest {
        snapshot_id: String,
        artifact_id: String,
    },
    MissingEvidence {
        snapshot_id: String,
        artifact_id: String,
    },
    DeploymentUnchanged,
    CriticalArtifactRemoved {
        artifact_id: String,
        kind: AssuranceDeltaArtifactKind,
    },
    MissingImpactRule(AssuranceDeltaArtifactKind),
    MissingRevalidation {
        changed_artifact_id: String,
        required_kind: AssuranceDeltaArtifactKind,
    },
    RevalidationUnchanged {
        changed_artifact_id: String,
        artifact_id: String,
        required_kind: AssuranceDeltaArtifactKind,
    },
    RestrictedImpact {
        changed_artifact_id: String,
        kind: AssuranceDeltaArtifactKind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceDeltaReport {
    pub schema_version: String,
    pub policy_id: String,
    pub baseline_snapshot_id: String,
    pub candidate_snapshot_id: String,
    pub baseline_deployment_id: String,
    pub candidate_deployment_id: String,
    pub status: AssuranceDeltaStatus,
    pub changes: Vec<AssuranceDeltaChange>,
    pub required_revalidation_kinds: Vec<AssuranceDeltaArtifactKind>,
    pub issues: Vec<AssuranceDeltaIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssuranceDeltaError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct AssuranceDeltaAnalyzer {
    policy: AssuranceDeltaPolicy,
}

impl AssuranceDeltaAnalyzer {
    pub fn new(policy: AssuranceDeltaPolicy) -> Result<Self, AssuranceDeltaError> {
        let critical: BTreeSet<_> = policy.critical_kinds.iter().copied().collect();
        let rule_kinds: BTreeSet<_> = policy
            .impact_rules
            .iter()
            .map(|rule| rule.changed_kind)
            .collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.critical_kinds.is_empty()
            || critical.len() != policy.critical_kinds.len()
            || policy.impact_rules.is_empty()
            || rule_kinds.len() != policy.impact_rules.len()
            || policy.impact_rules.iter().any(|rule| {
                rule.required_revalidation_kinds.is_empty()
                    || BTreeSet::<_>::from_iter(rule.required_revalidation_kinds.iter().copied())
                        .len()
                        != rule.required_revalidation_kinds.len()
            })
        {
            return Err(AssuranceDeltaError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        baseline: &AssuranceDeltaSnapshot,
        candidate: &AssuranceDeltaSnapshot,
    ) -> AssuranceDeltaReport {
        let mut issues = Vec::new();
        let baseline_by_id = collect_snapshot(baseline, &mut issues);
        let candidate_by_id = collect_snapshot(candidate, &mut issues);
        if baseline.deployment_id == candidate.deployment_id {
            issues.push(AssuranceDeltaIssue::DeploymentUnchanged);
        }
        let all_ids: BTreeSet<_> = baseline_by_id
            .keys()
            .chain(candidate_by_id.keys())
            .cloned()
            .collect();
        let mut changes = Vec::new();
        let mut required_revalidation = BTreeSet::new();
        let rules: BTreeMap<_, _> = self
            .policy
            .impact_rules
            .iter()
            .map(|rule| (rule.changed_kind, rule))
            .collect();

        for artifact_id in all_ids {
            let baseline_artifact = baseline_by_id.get(&artifact_id);
            let candidate_artifact = candidate_by_id.get(&artifact_id);
            let (kind, change) = match (baseline_artifact, candidate_artifact) {
                (Some(old), Some(new)) if old.kind != new.kind => {
                    issues.push(AssuranceDeltaIssue::CriticalArtifactRemoved {
                        artifact_id: artifact_id.clone(),
                        kind: old.kind,
                    });
                    (new.kind, AssuranceDeltaChangeKind::Modified)
                }
                (Some(old), Some(new)) if old.digest == new.digest => {
                    (new.kind, AssuranceDeltaChangeKind::Unchanged)
                }
                (Some(_), Some(new)) => (new.kind, AssuranceDeltaChangeKind::Modified),
                (None, Some(new)) => (new.kind, AssuranceDeltaChangeKind::Added),
                (Some(old), None) => {
                    if self.policy.critical_kinds.contains(&old.kind) {
                        issues.push(AssuranceDeltaIssue::CriticalArtifactRemoved {
                            artifact_id: artifact_id.clone(),
                            kind: old.kind,
                        });
                    }
                    (old.kind, AssuranceDeltaChangeKind::Removed)
                }
                (None, None) => unreachable!(),
            };
            changes.push(AssuranceDeltaChange {
                artifact_id: artifact_id.clone(),
                kind,
                change,
                baseline_digest: baseline_artifact.map(|artifact| artifact.digest.clone()),
                candidate_digest: candidate_artifact.map(|artifact| artifact.digest.clone()),
            });
            if change != AssuranceDeltaChangeKind::Unchanged {
                match rules.get(&kind) {
                    Some(rule) => {
                        required_revalidation
                            .extend(rule.required_revalidation_kinds.iter().copied());
                        if rule.permit_restriction {
                            issues.push(AssuranceDeltaIssue::RestrictedImpact {
                                changed_artifact_id: artifact_id.clone(),
                                kind,
                            });
                        }
                        for required_kind in &rule.required_revalidation_kinds {
                            let revalidation = candidate.artifacts.iter().find(|artifact| {
                                artifact.kind == *required_kind && !artifact.evidence_ids.is_empty()
                            });
                            match revalidation {
                                None => issues.push(AssuranceDeltaIssue::MissingRevalidation {
                                    changed_artifact_id: artifact_id.clone(),
                                    required_kind: *required_kind,
                                }),
                                Some(artifact) => {
                                    if baseline_by_id
                                        .get(&artifact.artifact_id)
                                        .is_some_and(|old| old.digest == artifact.digest)
                                    {
                                        issues.push(AssuranceDeltaIssue::RevalidationUnchanged {
                                            changed_artifact_id: artifact_id.clone(),
                                            artifact_id: artifact.artifact_id.clone(),
                                            required_kind: *required_kind,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    None => issues.push(AssuranceDeltaIssue::MissingImpactRule(kind)),
                }
            }
        }

        changes.sort_by_key(|change| (change.kind, change.artifact_id.clone()));
        let rejected = issues.iter().any(|issue| {
            matches!(
                issue,
                AssuranceDeltaIssue::CriticalArtifactRemoved { .. }
                    | AssuranceDeltaIssue::MissingRevalidation { .. }
                    | AssuranceDeltaIssue::RevalidationUnchanged { .. }
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                AssuranceDeltaIssue::InvalidSnapshotIdentity(_)
                    | AssuranceDeltaIssue::DuplicateArtifact { .. }
                    | AssuranceDeltaIssue::InvalidDigest { .. }
                    | AssuranceDeltaIssue::MissingEvidence { .. }
                    | AssuranceDeltaIssue::DeploymentUnchanged
                    | AssuranceDeltaIssue::MissingImpactRule(_)
            )
        });
        let restricted = issues
            .iter()
            .any(|issue| matches!(issue, AssuranceDeltaIssue::RestrictedImpact { .. }));
        let status = if incomplete {
            AssuranceDeltaStatus::Incomplete
        } else if rejected {
            AssuranceDeltaStatus::Rejected
        } else if restricted {
            AssuranceDeltaStatus::Restricted
        } else {
            AssuranceDeltaStatus::Cleared
        };

        AssuranceDeltaReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            baseline_snapshot_id: baseline.snapshot_id.clone(),
            candidate_snapshot_id: candidate.snapshot_id.clone(),
            baseline_deployment_id: baseline.deployment_id.clone(),
            candidate_deployment_id: candidate.deployment_id.clone(),
            status,
            changes,
            required_revalidation_kinds: required_revalidation.into_iter().collect(),
            issues,
        }
    }
}

fn collect_snapshot<'a>(
    snapshot: &'a AssuranceDeltaSnapshot,
    issues: &mut Vec<AssuranceDeltaIssue>,
) -> BTreeMap<String, &'a AssuranceDeltaArtifact> {
    if snapshot.snapshot_id.trim().is_empty() || snapshot.deployment_id.trim().is_empty() {
        issues.push(AssuranceDeltaIssue::InvalidSnapshotIdentity(
            snapshot.snapshot_id.clone(),
        ));
    }
    let mut by_id = BTreeMap::new();
    for artifact in &snapshot.artifacts {
        if artifact.artifact_id.trim().is_empty() {
            issues.push(AssuranceDeltaIssue::InvalidSnapshotIdentity(
                snapshot.snapshot_id.clone(),
            ));
        }
        if by_id
            .insert(artifact.artifact_id.clone(), artifact)
            .is_some()
        {
            issues.push(AssuranceDeltaIssue::DuplicateArtifact {
                snapshot_id: snapshot.snapshot_id.clone(),
                artifact_id: artifact.artifact_id.clone(),
            });
        }
        if !valid_digest(&artifact.digest) {
            issues.push(AssuranceDeltaIssue::InvalidDigest {
                snapshot_id: snapshot.snapshot_id.clone(),
                artifact_id: artifact.artifact_id.clone(),
            });
        }
        if artifact.evidence_ids.is_empty()
            || artifact.evidence_ids.iter().any(|id| id.trim().is_empty())
        {
            issues.push(AssuranceDeltaIssue::MissingEvidence {
                snapshot_id: snapshot.snapshot_id.clone(),
                artifact_id: artifact.artifact_id.clone(),
            });
        }
    }
    by_id
}

fn valid_digest(digest: &str) -> bool {
    let digest = digest.trim();
    digest.starts_with("sha256:") && digest.len() > "sha256:".len()
        || digest.starts_with("fnv1a64:") && digest.len() == "fnv1a64:".len() + 16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> AssuranceDeltaAnalyzer {
        AssuranceDeltaAnalyzer::new(AssuranceDeltaPolicy {
            schema_version: "1".into(),
            policy_id: "delta".into(),
            critical_kinds: vec![AssuranceDeltaArtifactKind::Controller],
            impact_rules: vec![
                AssuranceImpactRule {
                    changed_kind: AssuranceDeltaArtifactKind::Controller,
                    required_revalidation_kinds: vec![
                        AssuranceDeltaArtifactKind::Test,
                        AssuranceDeltaArtifactKind::Claim,
                    ],
                    permit_restriction: false,
                },
                AssuranceImpactRule {
                    changed_kind: AssuranceDeltaArtifactKind::Test,
                    required_revalidation_kinds: vec![AssuranceDeltaArtifactKind::Claim],
                    permit_restriction: false,
                },
                AssuranceImpactRule {
                    changed_kind: AssuranceDeltaArtifactKind::Claim,
                    required_revalidation_kinds: vec![
                        AssuranceDeltaArtifactKind::QualificationEvidence,
                    ],
                    permit_restriction: false,
                },
                AssuranceImpactRule {
                    changed_kind: AssuranceDeltaArtifactKind::QualificationEvidence,
                    required_revalidation_kinds: vec![AssuranceDeltaArtifactKind::Deployment],
                    permit_restriction: true,
                },
                AssuranceImpactRule {
                    changed_kind: AssuranceDeltaArtifactKind::Deployment,
                    required_revalidation_kinds: vec![
                        AssuranceDeltaArtifactKind::QualificationEvidence,
                    ],
                    permit_restriction: true,
                },
            ],
        })
        .unwrap()
    }

    fn artifact(
        id: &str,
        kind: AssuranceDeltaArtifactKind,
        digest: &str,
    ) -> AssuranceDeltaArtifact {
        AssuranceDeltaArtifact {
            artifact_id: id.into(),
            kind,
            digest: digest.into(),
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    fn baseline() -> AssuranceDeltaSnapshot {
        AssuranceDeltaSnapshot {
            snapshot_id: "baseline".into(),
            deployment_id: "deployment-a".into(),
            artifacts: vec![
                artifact(
                    "controller",
                    AssuranceDeltaArtifactKind::Controller,
                    "sha256:controller-a",
                ),
                artifact("test", AssuranceDeltaArtifactKind::Test, "sha256:test-a"),
                artifact("claim", AssuranceDeltaArtifactKind::Claim, "sha256:claim-a"),
                artifact(
                    "qualification",
                    AssuranceDeltaArtifactKind::QualificationEvidence,
                    "sha256:q-a",
                ),
                artifact(
                    "deployment",
                    AssuranceDeltaArtifactKind::Deployment,
                    "sha256:d-a",
                ),
            ],
        }
    }

    #[test]
    fn changed_controller_requires_fresh_test_and_claim() {
        let baseline = baseline();
        let mut candidate = baseline.clone();
        candidate.snapshot_id = "candidate".into();
        candidate.deployment_id = "deployment-b".into();
        candidate.artifacts[0].digest = "sha256:controller-b".into();
        let report = analyzer().assess(&baseline, &candidate);
        assert_eq!(report.status, AssuranceDeltaStatus::Rejected);
    }

    #[test]
    fn fresh_revalidation_clears_change() {
        let baseline = baseline();
        let mut candidate = baseline.clone();
        candidate.snapshot_id = "candidate".into();
        candidate.deployment_id = "deployment-b".into();
        candidate.artifacts[0].digest = "sha256:controller-b".into();
        candidate.artifacts[1].digest = "sha256:test-b".into();
        candidate.artifacts[2].digest = "sha256:claim-b".into();
        candidate.artifacts[3].digest = "sha256:q-b".into();
        candidate.artifacts[4].digest = "sha256:d-b".into();
        let report = analyzer().assess(&baseline, &candidate);
        assert!(matches!(
            report.status,
            AssuranceDeltaStatus::Restricted | AssuranceDeltaStatus::Cleared
        ));
    }

    #[test]
    fn removing_critical_artifact_is_rejected() {
        let baseline = baseline();
        let mut candidate = baseline.clone();
        candidate.snapshot_id = "candidate".into();
        candidate.deployment_id = "deployment-b".into();
        candidate.artifacts.remove(0);
        let report = analyzer().assess(&baseline, &candidate);
        assert_eq!(report.status, AssuranceDeltaStatus::Rejected);
    }

    #[test]
    fn unchanged_deployment_is_incomplete() {
        let baseline = baseline();
        let report = analyzer().assess(&baseline, &baseline);
        assert_eq!(report.status, AssuranceDeltaStatus::Incomplete);
    }
}
