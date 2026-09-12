// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Acyclic provenance and anti-circularity assurance for safety evidence.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDependencyNodeKind {
    RawObservation,
    TestArtifact,
    Verification,
    Receipt { contract_digest: String },
    Qualification { contract_digest: String },
    ReadinessDecision { contract_digest: String },
}

impl EvidenceDependencyNodeKind {
    fn validate(&self) -> bool {
        match self {
            Self::RawObservation | Self::TestArtifact | Self::Verification => true,
            Self::Receipt { contract_digest }
            | Self::Qualification { contract_digest }
            | Self::ReadinessDecision { contract_digest } => valid_digest(contract_digest),
        }
    }

    fn contract_digest(&self) -> Option<&str> {
        match self {
            Self::Receipt { contract_digest }
            | Self::Qualification { contract_digest }
            | Self::ReadinessDecision { contract_digest } => Some(contract_digest),
            Self::RawObservation | Self::TestArtifact | Self::Verification => None,
        }
    }

    fn is_evidence_for_contract(&self) -> bool {
        matches!(self, Self::Receipt { .. } | Self::Qualification { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDependencyNode {
    pub node_id: String,
    pub kind: EvidenceDependencyNodeKind,
    pub evidence_ref: String,
}

impl EvidenceDependencyNode {
    pub fn validate(&self) -> bool {
        !self.node_id.trim().is_empty()
            && self.kind.validate()
            && !self.evidence_ref.trim().is_empty()
    }
}

/// Directed dependency edge: `dependent` uses `prerequisite` as evidence/input.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EvidenceDependencyEdge {
    pub dependent: String,
    pub prerequisite: String,
    pub rationale_ref: String,
}

impl EvidenceDependencyEdge {
    pub fn validate(&self) -> bool {
        !self.dependent.trim().is_empty()
            && !self.prerequisite.trim().is_empty()
            && self.dependent != self.prerequisite
            && !self.rationale_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDependencyStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDependencyIssue {
    InvalidNode(String),
    DuplicateNodeId(String),
    InvalidEdge {
        dependent: String,
        prerequisite: String,
    },
    DuplicateEdge {
        dependent: String,
        prerequisite: String,
    },
    UnknownDependent(String),
    UnknownPrerequisite(String),
    CycleDetected {
        node_ids: Vec<String>,
    },
    SameContractReadinessBackDependency {
        evidence_node_id: String,
        readiness_node_id: String,
        contract_digest: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDependencyReport {
    pub status: EvidenceDependencyStatus,
    /// Deterministic prerequisite-before-dependent order when the graph is acyclic.
    pub topological_order: Vec<String>,
    pub issues: Vec<EvidenceDependencyIssue>,
}

impl EvidenceDependencyReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_evidence_dependency_graph(
    nodes: &[EvidenceDependencyNode],
    edges: &[EvidenceDependencyEdge],
) -> EvidenceDependencyReport {
    let mut issues = Vec::new();
    let mut by_id = BTreeMap::<String, &EvidenceDependencyNode>::new();

    for node in nodes {
        if !node.validate() {
            issues.push(EvidenceDependencyIssue::InvalidNode(node.node_id.clone()));
            continue;
        }
        if by_id.insert(node.node_id.clone(), node).is_some() {
            issues.push(EvidenceDependencyIssue::DuplicateNodeId(node.node_id.clone()));
        }
    }

    let mut edge_keys = BTreeSet::<(String, String)>::new();
    let mut dependencies = BTreeMap::<String, BTreeSet<String>>::new();
    let mut reverse = BTreeMap::<String, BTreeSet<String>>::new();
    let mut indegree = BTreeMap::<String, usize>::new();
    for node_id in by_id.keys() {
        dependencies.entry(node_id.clone()).or_default();
        reverse.entry(node_id.clone()).or_default();
        indegree.insert(node_id.clone(), 0);
    }

    for edge in edges {
        if !edge.validate() {
            issues.push(EvidenceDependencyIssue::InvalidEdge {
                dependent: edge.dependent.clone(),
                prerequisite: edge.prerequisite.clone(),
            });
            continue;
        }
        if !by_id.contains_key(&edge.dependent) {
            issues.push(EvidenceDependencyIssue::UnknownDependent(
                edge.dependent.clone(),
            ));
            continue;
        }
        if !by_id.contains_key(&edge.prerequisite) {
            issues.push(EvidenceDependencyIssue::UnknownPrerequisite(
                edge.prerequisite.clone(),
            ));
            continue;
        }
        let key = (edge.dependent.clone(), edge.prerequisite.clone());
        if !edge_keys.insert(key) {
            issues.push(EvidenceDependencyIssue::DuplicateEdge {
                dependent: edge.dependent.clone(),
                prerequisite: edge.prerequisite.clone(),
            });
            continue;
        }
        dependencies
            .entry(edge.dependent.clone())
            .or_default()
            .insert(edge.prerequisite.clone());
        reverse
            .entry(edge.prerequisite.clone())
            .or_default()
            .insert(edge.dependent.clone());
        *indegree.entry(edge.dependent.clone()).or_default() += 1;
    }

    let mut queue = indegree
        .iter()
        .filter_map(|(node_id, degree)| (*degree == 0).then_some(node_id.clone()))
        .collect::<Vec<_>>();
    queue.sort();
    let mut queue = VecDeque::from(queue);
    let mut topological_order = Vec::new();

    while let Some(node_id) = queue.pop_front() {
        topological_order.push(node_id.clone());
        let mut dependents = reverse
            .get(&node_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        dependents.sort();
        for dependent in dependents {
            if let Some(degree) = indegree.get_mut(&dependent) {
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    queue.push_back(dependent);
                }
            }
        }
    }

    if topological_order.len() != by_id.len() {
        let mut cycle_nodes = indegree
            .iter()
            .filter_map(|(node_id, degree)| (*degree > 0).then_some(node_id.clone()))
            .collect::<Vec<_>>();
        cycle_nodes.sort();
        issues.push(EvidenceDependencyIssue::CycleDetected {
            node_ids: cycle_nodes,
        });
    }

    for (node_id, node) in &by_id {
        if !node.kind.is_evidence_for_contract() {
            continue;
        }
        let Some(contract_digest) = node.kind.contract_digest() else {
            continue;
        };
        let mut stack = dependencies
            .get(node_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let mut visited = BTreeSet::new();

        while let Some(prerequisite_id) = stack.pop() {
            if !visited.insert(prerequisite_id.clone()) {
                continue;
            }
            let Some(prerequisite) = by_id.get(&prerequisite_id) else {
                continue;
            };
            if let EvidenceDependencyNodeKind::ReadinessDecision {
                contract_digest: readiness_contract,
            } = &prerequisite.kind
            {
                if readiness_contract == contract_digest {
                    issues.push(
                        EvidenceDependencyIssue::SameContractReadinessBackDependency {
                            evidence_node_id: node_id.clone(),
                            readiness_node_id: prerequisite_id.clone(),
                            contract_digest: contract_digest.to_string(),
                        },
                    );
                }
            }
            if let Some(next) = dependencies.get(&prerequisite_id) {
                stack.extend(next.iter().cloned());
            }
        }
    }

    issues.sort_by_key(issue_sort_key);
    let status = if issues.is_empty() {
        EvidenceDependencyStatus::Valid
    } else {
        EvidenceDependencyStatus::Invalid
    };

    EvidenceDependencyReport {
        status,
        topological_order: if status == EvidenceDependencyStatus::Valid {
            topological_order
        } else {
            Vec::new()
        },
        issues,
    }
}

fn valid_digest(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed
            .split_once(':')
            .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

fn issue_sort_key(issue: &EvidenceDependencyIssue) -> String {
    match issue {
        EvidenceDependencyIssue::InvalidNode(id) => format!("01:{id}"),
        EvidenceDependencyIssue::DuplicateNodeId(id) => format!("02:{id}"),
        EvidenceDependencyIssue::InvalidEdge {
            dependent,
            prerequisite,
        } => format!("03:{dependent}:{prerequisite}"),
        EvidenceDependencyIssue::DuplicateEdge {
            dependent,
            prerequisite,
        } => format!("04:{dependent}:{prerequisite}"),
        EvidenceDependencyIssue::UnknownDependent(id) => format!("05:{id}"),
        EvidenceDependencyIssue::UnknownPrerequisite(id) => format!("06:{id}"),
        EvidenceDependencyIssue::CycleDetected { node_ids } => {
            format!("07:{}", node_ids.join(","))
        }
        EvidenceDependencyIssue::SameContractReadinessBackDependency {
            evidence_node_id,
            readiness_node_id,
            contract_digest,
        } => format!(
            "08:{contract_digest}:{evidence_node_id}:{readiness_node_id}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: EvidenceDependencyNodeKind) -> EvidenceDependencyNode {
        EvidenceDependencyNode {
            node_id: id.into(),
            kind,
            evidence_ref: format!("evidence:{id}"),
        }
    }

    fn edge(dependent: &str, prerequisite: &str) -> EvidenceDependencyEdge {
        EvidenceDependencyEdge {
            dependent: dependent.into(),
            prerequisite: prerequisite.into(),
            rationale_ref: format!("dependency:{dependent}:{prerequisite}"),
        }
    }

    #[test]
    fn ordinary_acyclic_evidence_chain_is_valid() {
        let nodes = vec![
            node("raw", EvidenceDependencyNodeKind::RawObservation),
            node("test", EvidenceDependencyNodeKind::TestArtifact),
            node("verify", EvidenceDependencyNodeKind::Verification),
            node(
                "receipt",
                EvidenceDependencyNodeKind::Receipt {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
        ];
        let edges = vec![
            edge("test", "raw"),
            edge("verify", "test"),
            edge("receipt", "verify"),
        ];
        let report = assess_evidence_dependency_graph(&nodes, &edges);
        assert_eq!(report.status, EvidenceDependencyStatus::Valid);
        assert_eq!(report.topological_order.first().map(String::as_str), Some("raw"));
        assert_eq!(report.topological_order.last().map(String::as_str), Some("receipt"));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn dependency_cycle_is_invalid() {
        let nodes = vec![
            node("a", EvidenceDependencyNodeKind::TestArtifact),
            node("b", EvidenceDependencyNodeKind::Verification),
        ];
        let report = assess_evidence_dependency_graph(
            &nodes,
            &[edge("a", "b"), edge("b", "a")],
        );
        assert_eq!(report.status, EvidenceDependencyStatus::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, EvidenceDependencyIssue::CycleDetected { .. })));
    }

    #[test]
    fn same_contract_readiness_hidden_behind_intermediates_is_rejected() {
        let nodes = vec![
            node(
                "ready-a",
                EvidenceDependencyNodeKind::ReadinessDecision {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
            node("artifact", EvidenceDependencyNodeKind::TestArtifact),
            node("verify", EvidenceDependencyNodeKind::Verification),
            node(
                "receipt-a",
                EvidenceDependencyNodeKind::Receipt {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
        ];
        let edges = vec![
            edge("artifact", "ready-a"),
            edge("verify", "artifact"),
            edge("receipt-a", "verify"),
        ];
        let report = assess_evidence_dependency_graph(&nodes, &edges);
        assert_eq!(report.status, EvidenceDependencyStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            EvidenceDependencyIssue::SameContractReadinessBackDependency {
                evidence_node_id,
                readiness_node_id,
                ..
            } if evidence_node_id == "receipt-a" && readiness_node_id == "ready-a"
        )));
    }

    #[test]
    fn qualification_cannot_depend_on_same_contract_readiness() {
        let nodes = vec![
            node(
                "ready-a",
                EvidenceDependencyNodeKind::ReadinessDecision {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
            node(
                "qualification-a",
                EvidenceDependencyNodeKind::Qualification {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
        ];
        let report = assess_evidence_dependency_graph(
            &nodes,
            &[edge("qualification-a", "ready-a")],
        );
        assert_eq!(report.status, EvidenceDependencyStatus::Invalid);
    }

    #[test]
    fn cross_contract_readiness_dependency_is_explicit_but_not_auto_rejected() {
        let nodes = vec![
            node(
                "ready-b",
                EvidenceDependencyNodeKind::ReadinessDecision {
                    contract_digest: "blake3:contract-b".into(),
                },
            ),
            node(
                "receipt-a",
                EvidenceDependencyNodeKind::Receipt {
                    contract_digest: "blake3:contract-a".into(),
                },
            ),
        ];
        let report = assess_evidence_dependency_graph(
            &nodes,
            &[edge("receipt-a", "ready-b")],
        );
        assert_eq!(report.status, EvidenceDependencyStatus::Valid);
    }

    #[test]
    fn duplicate_edge_is_invalid() {
        let nodes = vec![
            node("raw", EvidenceDependencyNodeKind::RawObservation),
            node("test", EvidenceDependencyNodeKind::TestArtifact),
        ];
        let dependency = edge("test", "raw");
        let report = assess_evidence_dependency_graph(
            &nodes,
            &[dependency.clone(), dependency],
        );
        assert_eq!(report.status, EvidenceDependencyStatus::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, EvidenceDependencyIssue::DuplicateEdge { .. })));
    }
}
