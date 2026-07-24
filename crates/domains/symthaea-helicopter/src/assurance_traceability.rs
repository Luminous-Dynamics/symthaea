// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end assurance traceability graph.
//!
//! Qualification artifacts are only useful when hazards, requirements,
//! mitigations, tests, evidence, claims, and deployment identities remain
//! connected. This module validates those links and refuses a complete status
//! when any safety-critical item is orphaned or any reference is dangling.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TraceArtifactKind {
    Hazard,
    SafetyRequirement,
    Mitigation,
    VerificationTest,
    EvidenceArtifact,
    AssuranceClaim,
    Deployment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceArtifact {
    pub artifact_id: String,
    pub kind: TraceArtifactKind,
    pub revision: String,
    pub digest: Option<String>,
    pub safety_critical: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TraceRelation {
    Addresses,
    Implements,
    VerifiedBy,
    ProducesEvidence,
    SupportsClaim,
    BoundToDeployment,
    DependsOn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLink {
    pub from_id: String,
    pub to_id: String,
    pub relation: TraceRelation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceTraceabilityGraph {
    pub schema_version: String,
    pub graph_id: String,
    pub artifacts: Vec<TraceArtifact>,
    pub links: Vec<TraceLink>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceabilityStatus {
    Complete,
    Incomplete,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceabilityIssue {
    EmptyIdentity,
    DuplicateArtifact(String),
    DuplicateLink(String, String, TraceRelation),
    DanglingLink(String, String),
    InvalidDigest(String),
    HazardWithoutRequirement(String),
    RequirementWithoutMitigation(String),
    MitigationWithoutVerification(String),
    TestWithoutEvidence(String),
    ClaimWithoutEvidence(String),
    DeploymentWithoutClaim(String),
    SafetyCriticalArtifactNotDeploymentReachable(String),
    DependencyCycle(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceabilityAssessment {
    pub graph_id: String,
    pub status: TraceabilityStatus,
    pub issues: Vec<TraceabilityIssue>,
    pub artifact_count: usize,
    pub link_count: usize,
    pub deployment_reachable_artifacts: usize,
    pub canonical_digest_fnv1a64: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceabilityError {
    SerializationFailed,
}

impl AssuranceTraceabilityGraph {
    pub fn assess(&self) -> TraceabilityAssessment {
        let mut issues = Vec::new();
        if self.schema_version.trim().is_empty() || self.graph_id.trim().is_empty() {
            issues.push(TraceabilityIssue::EmptyIdentity);
        }
        let mut artifacts = BTreeMap::<String, &TraceArtifact>::new();
        for artifact in &self.artifacts {
            if artifact.artifact_id.trim().is_empty() || artifact.revision.trim().is_empty() {
                issues.push(TraceabilityIssue::EmptyIdentity);
            }
            if artifacts
                .insert(artifact.artifact_id.clone(), artifact)
                .is_some()
            {
                issues.push(TraceabilityIssue::DuplicateArtifact(
                    artifact.artifact_id.clone(),
                ));
            }
            if artifact
                .digest
                .as_ref()
                .is_some_and(|digest| !valid_digest(digest))
            {
                issues.push(TraceabilityIssue::InvalidDigest(
                    artifact.artifact_id.clone(),
                ));
            }
            if matches!(
                artifact.kind,
                TraceArtifactKind::EvidenceArtifact | TraceArtifactKind::Deployment
            ) && artifact
                .digest
                .as_ref()
                .is_none_or(|digest| !valid_digest(digest))
            {
                issues.push(TraceabilityIssue::InvalidDigest(
                    artifact.artifact_id.clone(),
                ));
            }
        }

        let mut unique_links = BTreeSet::new();
        let mut outgoing = BTreeMap::<String, Vec<&TraceLink>>::new();
        let mut incoming = BTreeMap::<String, Vec<&TraceLink>>::new();
        for link in &self.links {
            let key = (link.from_id.clone(), link.to_id.clone(), link.relation);
            if !unique_links.insert(key.clone()) {
                issues.push(TraceabilityIssue::DuplicateLink(key.0, key.1, key.2));
            }
            if !artifacts.contains_key(&link.from_id) || !artifacts.contains_key(&link.to_id) {
                issues.push(TraceabilityIssue::DanglingLink(
                    link.from_id.clone(),
                    link.to_id.clone(),
                ));
                continue;
            }
            outgoing.entry(link.from_id.clone()).or_default().push(link);
            incoming.entry(link.to_id.clone()).or_default().push(link);
        }

        for artifact in artifacts.values() {
            match artifact.kind {
                TraceArtifactKind::Hazard => {
                    if !has_outgoing_kind(
                        artifact,
                        TraceRelation::Addresses,
                        TraceArtifactKind::SafetyRequirement,
                        &outgoing,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::HazardWithoutRequirement(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::SafetyRequirement => {
                    if !has_outgoing_kind(
                        artifact,
                        TraceRelation::Implements,
                        TraceArtifactKind::Mitigation,
                        &outgoing,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::RequirementWithoutMitigation(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::Mitigation => {
                    if !has_outgoing_kind(
                        artifact,
                        TraceRelation::VerifiedBy,
                        TraceArtifactKind::VerificationTest,
                        &outgoing,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::MitigationWithoutVerification(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::VerificationTest => {
                    if !has_outgoing_kind(
                        artifact,
                        TraceRelation::ProducesEvidence,
                        TraceArtifactKind::EvidenceArtifact,
                        &outgoing,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::TestWithoutEvidence(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::AssuranceClaim => {
                    if !has_incoming_kind(
                        artifact,
                        TraceRelation::SupportsClaim,
                        TraceArtifactKind::EvidenceArtifact,
                        &incoming,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::ClaimWithoutEvidence(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::Deployment => {
                    if !has_incoming_kind(
                        artifact,
                        TraceRelation::BoundToDeployment,
                        TraceArtifactKind::AssuranceClaim,
                        &incoming,
                        &artifacts,
                    ) {
                        issues.push(TraceabilityIssue::DeploymentWithoutClaim(
                            artifact.artifact_id.clone(),
                        ));
                    }
                }
                TraceArtifactKind::EvidenceArtifact => {}
            }
        }

        let deployment_reachable = reverse_reachable_from_deployments(&artifacts, &incoming);
        for artifact in artifacts
            .values()
            .filter(|artifact| artifact.safety_critical)
        {
            if !deployment_reachable.contains(&artifact.artifact_id) {
                issues.push(
                    TraceabilityIssue::SafetyCriticalArtifactNotDeploymentReachable(
                        artifact.artifact_id.clone(),
                    ),
                );
            }
        }
        if let Some(cycle) = dependency_cycle(&artifacts, &outgoing) {
            issues.push(TraceabilityIssue::DependencyCycle(cycle));
        }

        let rejected = issues.iter().any(|issue| {
            matches!(
                issue,
                TraceabilityIssue::EmptyIdentity
                    | TraceabilityIssue::DuplicateArtifact(_)
                    | TraceabilityIssue::DuplicateLink(_, _, _)
                    | TraceabilityIssue::DanglingLink(_, _)
                    | TraceabilityIssue::InvalidDigest(_)
                    | TraceabilityIssue::DependencyCycle(_)
            )
        });
        let status = if rejected {
            TraceabilityStatus::Rejected
        } else if issues.is_empty() {
            TraceabilityStatus::Complete
        } else {
            TraceabilityStatus::Incomplete
        };
        let canonical_digest_fnv1a64 = if rejected {
            None
        } else {
            self.digest_fnv1a64().ok()
        };
        TraceabilityAssessment {
            graph_id: self.graph_id.clone(),
            status,
            issues,
            artifact_count: self.artifacts.len(),
            link_count: self.links.len(),
            deployment_reachable_artifacts: deployment_reachable.len(),
            canonical_digest_fnv1a64,
        }
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, TraceabilityError> {
        let mut canonical = self.clone();
        canonical.artifacts.sort_by(|left, right| {
            left.kind
                .cmp(&right.kind)
                .then_with(|| left.artifact_id.cmp(&right.artifact_id))
        });
        canonical.links.sort_by(|left, right| {
            left.from_id
                .cmp(&right.from_id)
                .then_with(|| left.to_id.cmp(&right.to_id))
                .then_with(|| left.relation.cmp(&right.relation))
        });
        serde_json::to_vec(&canonical).map_err(|_| TraceabilityError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, TraceabilityError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

fn has_outgoing_kind(
    artifact: &TraceArtifact,
    relation: TraceRelation,
    target_kind: TraceArtifactKind,
    outgoing: &BTreeMap<String, Vec<&TraceLink>>,
    artifacts: &BTreeMap<String, &TraceArtifact>,
) -> bool {
    outgoing
        .get(&artifact.artifact_id)
        .into_iter()
        .flatten()
        .any(|link| {
            link.relation == relation
                && artifacts
                    .get(&link.to_id)
                    .is_some_and(|target| target.kind == target_kind)
        })
}

fn has_incoming_kind(
    artifact: &TraceArtifact,
    relation: TraceRelation,
    source_kind: TraceArtifactKind,
    incoming: &BTreeMap<String, Vec<&TraceLink>>,
    artifacts: &BTreeMap<String, &TraceArtifact>,
) -> bool {
    incoming
        .get(&artifact.artifact_id)
        .into_iter()
        .flatten()
        .any(|link| {
            link.relation == relation
                && artifacts
                    .get(&link.from_id)
                    .is_some_and(|source| source.kind == source_kind)
        })
}

fn reverse_reachable_from_deployments(
    artifacts: &BTreeMap<String, &TraceArtifact>,
    incoming: &BTreeMap<String, Vec<&TraceLink>>,
) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::new();
    for artifact in artifacts.values() {
        if artifact.kind == TraceArtifactKind::Deployment {
            reachable.insert(artifact.artifact_id.clone());
            queue.push_back(artifact.artifact_id.clone());
        }
    }
    while let Some(current) = queue.pop_front() {
        for link in incoming.get(&current).into_iter().flatten() {
            if reachable.insert(link.from_id.clone()) {
                queue.push_back(link.from_id.clone());
            }
        }
    }
    reachable
}

fn dependency_cycle(
    artifacts: &BTreeMap<String, &TraceArtifact>,
    outgoing: &BTreeMap<String, Vec<&TraceLink>>,
) -> Option<Vec<String>> {
    fn visit(
        id: &str,
        outgoing: &BTreeMap<String, Vec<&TraceLink>>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        stack: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if visiting.contains(id) {
            let start = stack.iter().position(|entry| entry == id).unwrap_or(0);
            return Some(stack[start..].to_vec());
        }
        if visited.contains(id) {
            return None;
        }
        visiting.insert(id.to_string());
        stack.push(id.to_string());
        for link in outgoing.get(id).into_iter().flatten() {
            if link.relation == TraceRelation::DependsOn {
                if let Some(cycle) = visit(&link.to_id, outgoing, visiting, visited, stack) {
                    return Some(cycle);
                }
            }
        }
        stack.pop();
        visiting.remove(id);
        visited.insert(id.to_string());
        None
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in artifacts.keys() {
        if let Some(cycle) = visit(id, outgoing, &mut visiting, &mut visited, &mut Vec::new()) {
            return Some(cycle);
        }
    }
    None
}

fn valid_digest(value: &str) -> bool {
    let Some((algorithm, digest)) = value.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && digest.len() >= 8
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str, kind: TraceArtifactKind) -> TraceArtifact {
        TraceArtifact {
            artifact_id: id.into(),
            kind,
            revision: "r1".into(),
            digest: matches!(
                kind,
                TraceArtifactKind::EvidenceArtifact | TraceArtifactKind::Deployment
            )
            .then(|| "sha256:0123456789abcdef".into()),
            safety_critical: true,
        }
    }

    fn complete_graph() -> AssuranceTraceabilityGraph {
        let artifacts = vec![
            artifact("hazard", TraceArtifactKind::Hazard),
            artifact("requirement", TraceArtifactKind::SafetyRequirement),
            artifact("mitigation", TraceArtifactKind::Mitigation),
            artifact("test", TraceArtifactKind::VerificationTest),
            artifact("evidence", TraceArtifactKind::EvidenceArtifact),
            artifact("claim", TraceArtifactKind::AssuranceClaim),
            artifact("deployment", TraceArtifactKind::Deployment),
        ];
        let links = vec![
            TraceLink {
                from_id: "hazard".into(),
                to_id: "requirement".into(),
                relation: TraceRelation::Addresses,
            },
            TraceLink {
                from_id: "requirement".into(),
                to_id: "mitigation".into(),
                relation: TraceRelation::Implements,
            },
            TraceLink {
                from_id: "mitigation".into(),
                to_id: "test".into(),
                relation: TraceRelation::VerifiedBy,
            },
            TraceLink {
                from_id: "test".into(),
                to_id: "evidence".into(),
                relation: TraceRelation::ProducesEvidence,
            },
            TraceLink {
                from_id: "evidence".into(),
                to_id: "claim".into(),
                relation: TraceRelation::SupportsClaim,
            },
            TraceLink {
                from_id: "claim".into(),
                to_id: "deployment".into(),
                relation: TraceRelation::BoundToDeployment,
            },
        ];
        AssuranceTraceabilityGraph {
            schema_version: "symthaea-helicopter-trace-v1".into(),
            graph_id: "trace-1".into(),
            artifacts,
            links,
        }
    }

    #[test]
    fn complete_chain_passes() {
        let assessment = complete_graph().assess();
        assert_eq!(assessment.status, TraceabilityStatus::Complete);
        assert_eq!(assessment.deployment_reachable_artifacts, 7);
    }

    #[test]
    fn orphan_hazard_is_incomplete() {
        let mut graph = complete_graph();
        graph.links.remove(0);
        let assessment = graph.assess();
        assert_eq!(assessment.status, TraceabilityStatus::Incomplete);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| matches!(issue, TraceabilityIssue::HazardWithoutRequirement(_)))
        );
    }

    #[test]
    fn dangling_link_is_rejected() {
        let mut graph = complete_graph();
        graph.links.push(TraceLink {
            from_id: "missing".into(),
            to_id: "deployment".into(),
            relation: TraceRelation::BoundToDeployment,
        });
        assert_eq!(graph.assess().status, TraceabilityStatus::Rejected);
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let mut graph = complete_graph();
        graph.links.push(TraceLink {
            from_id: "mitigation".into(),
            to_id: "test".into(),
            relation: TraceRelation::DependsOn,
        });
        graph.links.push(TraceLink {
            from_id: "test".into(),
            to_id: "mitigation".into(),
            relation: TraceRelation::DependsOn,
        });
        let assessment = graph.assess();
        assert_eq!(assessment.status, TraceabilityStatus::Rejected);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| matches!(issue, TraceabilityIssue::DependencyCycle(_)))
        );
    }
}
