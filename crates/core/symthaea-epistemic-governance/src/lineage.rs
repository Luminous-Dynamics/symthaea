// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-lineage and independence contracts for RCA v1.
//!
//! Multiple evidence objects do not imply multiple independent observations.
//! Independence is derived from a closed ancestry graph; an evidence producer
//! cannot self-assert that its own output is independent corroboration.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};

pub const COGNITIVE_LINEAGE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveDerivationKindV1 {
    RootObservation,
    Retrieval,
    Transformation,
    Inference,
    Simulation,
    Summary,
    Critique,
    FormalDerivation,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceLineageNodeV1 {
    pub schema_version: u16,
    /// Content-addressed identity of this evidence object.
    pub evidence_id: String,
    /// Immediate parent evidence ids. Root observations have no parents.
    #[serde(default)]
    pub parent_ids: Vec<String>,
    pub derivation_kind: CognitiveDerivationKindV1,
}

impl EvidenceLineageNodeV1 {
    pub fn validate(self) -> Result<ValidatedEvidenceLineageNodeV1, CognitiveLineageError> {
        ValidatedEvidenceLineageNodeV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedEvidenceLineageNodeV1(EvidenceLineageNodeV1);

impl ValidatedEvidenceLineageNodeV1 {
    pub fn evidence_id(&self) -> &str {
        &self.0.evidence_id
    }

    pub fn parent_ids(&self) -> &[String] {
        &self.0.parent_ids
    }

    pub const fn derivation_kind(&self) -> CognitiveDerivationKindV1 {
        self.0.derivation_kind
    }
}

impl TryFrom<EvidenceLineageNodeV1> for ValidatedEvidenceLineageNodeV1 {
    type Error = CognitiveLineageError;

    fn try_from(value: EvidenceLineageNodeV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_LINEAGE_SCHEMA_VERSION {
            return Err(CognitiveLineageError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.evidence_id)?;

        let mut parents = HashSet::with_capacity(value.parent_ids.len());
        for parent in &value.parent_ids {
            validate_digest(parent)?;
            if parent == &value.evidence_id {
                return Err(CognitiveLineageError::SelfParent {
                    evidence_id: value.evidence_id.clone(),
                });
            }
            if !parents.insert(parent.as_str()) {
                return Err(CognitiveLineageError::DuplicateParent {
                    evidence_id: value.evidence_id.clone(),
                    parent_id: parent.clone(),
                });
            }
        }

        if value.derivation_kind == CognitiveDerivationKindV1::RootObservation {
            if !value.parent_ids.is_empty() {
                return Err(CognitiveLineageError::RootHasParents {
                    evidence_id: value.evidence_id.clone(),
                });
            }
        } else if value.parent_ids.is_empty() {
            return Err(CognitiveLineageError::DerivedNodeMissingParent {
                evidence_id: value.evidence_id.clone(),
            });
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedEvidenceLineageNodeV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EvidenceLineageNodeV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceLineageGraphV1 {
    pub schema_version: u16,
    pub graph_id: String,
    pub nodes: Vec<ValidatedEvidenceLineageNodeV1>,
}

impl EvidenceLineageGraphV1 {
    pub fn validate(self) -> Result<ValidatedEvidenceLineageGraphV1, CognitiveLineageError> {
        ValidatedEvidenceLineageGraphV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedEvidenceLineageGraphV1(EvidenceLineageGraphV1);

impl ValidatedEvidenceLineageGraphV1 {
    pub fn root_ids(&self, evidence_id: &str) -> Result<HashSet<String>, CognitiveLineageError> {
        validate_digest(evidence_id)?;
        let index: HashMap<&str, &ValidatedEvidenceLineageNodeV1> = self
            .0
            .nodes
            .iter()
            .map(|node| (node.evidence_id(), node))
            .collect();
        if !index.contains_key(evidence_id) {
            return Err(CognitiveLineageError::UnknownEvidenceId {
                evidence_id: evidence_id.to_string(),
            });
        }

        let mut roots = HashSet::new();
        let mut stack = vec![evidence_id];
        let mut visited = HashSet::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let node = index[id];
            if node.parent_ids().is_empty() {
                roots.insert(id.to_string());
            } else {
                for parent in node.parent_ids() {
                    stack.push(parent.as_str());
                }
            }
        }
        Ok(roots)
    }

    /// `Independent` is returned only for complete, disjoint root sets.
    pub fn assess_independence(
        &self,
        left: &str,
        right: &str,
    ) -> Result<EvidenceIndependenceV1, CognitiveLineageError> {
        if left == right {
            return Ok(EvidenceIndependenceV1::SameEvidence);
        }
        let left_roots = self.root_ids(left)?;
        let right_roots = self.root_ids(right)?;

        if is_ancestor(self, left, right)? || is_ancestor(self, right, left)? {
            return Ok(EvidenceIndependenceV1::Derived);
        }

        let shared = left_roots.intersection(&right_roots).count();
        if shared == 0 {
            Ok(EvidenceIndependenceV1::Independent)
        } else if left_roots == right_roots {
            Ok(EvidenceIndependenceV1::SameRoot)
        } else {
            Ok(EvidenceIndependenceV1::PartiallyShared)
        }
    }
}

impl TryFrom<EvidenceLineageGraphV1> for ValidatedEvidenceLineageGraphV1 {
    type Error = CognitiveLineageError;

    fn try_from(value: EvidenceLineageGraphV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_LINEAGE_SCHEMA_VERSION {
            return Err(CognitiveLineageError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.graph_id)?;
        if value.nodes.is_empty() {
            return Err(CognitiveLineageError::EmptyGraph);
        }

        let mut index = HashMap::with_capacity(value.nodes.len());
        for node in &value.nodes {
            if index.insert(node.evidence_id(), node).is_some() {
                return Err(CognitiveLineageError::DuplicateEvidenceId {
                    evidence_id: node.evidence_id().to_string(),
                });
            }
        }

        for node in &value.nodes {
            for parent in node.parent_ids() {
                if !index.contains_key(parent.as_str()) {
                    return Err(CognitiveLineageError::UnknownParent {
                        evidence_id: node.evidence_id().to_string(),
                        parent_id: parent.clone(),
                    });
                }
            }
        }

        let mut state: HashMap<&str, VisitState> = HashMap::new();
        for node in &value.nodes {
            visit(node.evidence_id(), &index, &mut state)?;
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedEvidenceLineageGraphV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EvidenceLineageGraphV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceIndependenceV1 {
    SameEvidence,
    Derived,
    SameRoot,
    PartiallyShared,
    Independent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Complete,
}

fn visit<'a>(
    id: &'a str,
    index: &HashMap<&'a str, &'a ValidatedEvidenceLineageNodeV1>,
    state: &mut HashMap<&'a str, VisitState>,
) -> Result<(), CognitiveLineageError> {
    match state.get(id) {
        Some(VisitState::Complete) => return Ok(()),
        Some(VisitState::Visiting) => {
            return Err(CognitiveLineageError::CycleDetected {
                evidence_id: id.to_string(),
            });
        }
        None => {}
    }
    state.insert(id, VisitState::Visiting);
    for parent in index[id].parent_ids() {
        visit(parent, index, state)?;
    }
    state.insert(id, VisitState::Complete);
    Ok(())
}

fn is_ancestor(
    graph: &ValidatedEvidenceLineageGraphV1,
    possible_ancestor: &str,
    descendant: &str,
) -> Result<bool, CognitiveLineageError> {
    let index: HashMap<&str, &ValidatedEvidenceLineageNodeV1> = graph
        .0
        .nodes
        .iter()
        .map(|node| (node.evidence_id(), node))
        .collect();
    if !index.contains_key(possible_ancestor) {
        return Err(CognitiveLineageError::UnknownEvidenceId {
            evidence_id: possible_ancestor.to_string(),
        });
    }
    if !index.contains_key(descendant) {
        return Err(CognitiveLineageError::UnknownEvidenceId {
            evidence_id: descendant.to_string(),
        });
    }

    let mut stack: Vec<&str> = index[descendant]
        .parent_ids()
        .iter()
        .map(String::as_str)
        .collect();
    let mut visited = HashSet::new();
    while let Some(id) = stack.pop() {
        if id == possible_ancestor {
            return Ok(true);
        }
        if visited.insert(id) {
            for parent in index[id].parent_ids() {
                stack.push(parent.as_str());
            }
        }
    }
    Ok(false)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveLineageError {
    UnsupportedSchemaVersion { found: u16 },
    MalformedDigest,
    SelfParent { evidence_id: String },
    DuplicateParent { evidence_id: String, parent_id: String },
    RootHasParents { evidence_id: String },
    DerivedNodeMissingParent { evidence_id: String },
    EmptyGraph,
    DuplicateEvidenceId { evidence_id: String },
    UnknownParent { evidence_id: String, parent_id: String },
    UnknownEvidenceId { evidence_id: String },
    CycleDetected { evidence_id: String },
}

impl std::fmt::Display for CognitiveLineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported cognitive lineage schema version {found}; expected {COGNITIVE_LINEAGE_SCHEMA_VERSION}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::SelfParent { evidence_id } => {
                write!(f, "evidence {evidence_id} cannot parent itself")
            }
            Self::DuplicateParent {
                evidence_id,
                parent_id,
            } => write!(
                f,
                "evidence {evidence_id} contains duplicate parent {parent_id}"
            ),
            Self::RootHasParents { evidence_id } => write!(
                f,
                "root observation {evidence_id} cannot declare parent evidence"
            ),
            Self::DerivedNodeMissingParent { evidence_id } => write!(
                f,
                "derived evidence {evidence_id} requires at least one parent"
            ),
            Self::EmptyGraph => f.write_str("evidence lineage graph cannot be empty"),
            Self::DuplicateEvidenceId { evidence_id } => write!(
                f,
                "evidence lineage graph contains duplicate evidence id {evidence_id}"
            ),
            Self::UnknownParent {
                evidence_id,
                parent_id,
            } => write!(
                f,
                "evidence {evidence_id} references unknown parent {parent_id}; ancestry must be closed before independence can be assessed"
            ),
            Self::UnknownEvidenceId { evidence_id } => {
                write!(f, "unknown evidence id {evidence_id}")
            }
            Self::CycleDetected { evidence_id } => {
                write!(f, "evidence lineage contains a cycle involving {evidence_id}")
            }
        }
    }
}

impl std::error::Error for CognitiveLineageError {}

fn validate_digest(digest: &str) -> Result<(), CognitiveLineageError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(CognitiveLineageError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CognitiveLineageError::MalformedDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn root(id: &str) -> ValidatedEvidenceLineageNodeV1 {
        EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: id.into(),
            parent_ids: vec![],
            derivation_kind: CognitiveDerivationKindV1::RootObservation,
        }
        .validate()
        .unwrap()
    }

    fn derived(id: &str, parents: &[&str]) -> ValidatedEvidenceLineageNodeV1 {
        EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: id.into(),
            parent_ids: parents.iter().map(|p| (*p).to_string()).collect(),
            derivation_kind: CognitiveDerivationKindV1::Inference,
        }
        .validate()
        .unwrap()
    }

    fn graph(nodes: Vec<ValidatedEvidenceLineageNodeV1>) -> ValidatedEvidenceLineageGraphV1 {
        EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: F.into(),
            nodes,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn same_root_is_not_independent() {
        let g = graph(vec![root(A), derived(B, &[A]), derived(C, &[A])]);
        assert_eq!(
            g.assess_independence(B, C).unwrap(),
            EvidenceIndependenceV1::SameRoot
        );
    }

    #[test]
    fn direct_derivation_is_not_independent() {
        let g = graph(vec![root(A), derived(B, &[A])]);
        assert_eq!(
            g.assess_independence(A, B).unwrap(),
            EvidenceIndependenceV1::Derived
        );
    }

    #[test]
    fn disjoint_roots_can_be_independent() {
        let g = graph(vec![root(A), root(B), derived(C, &[A]), derived(D, &[B])]);
        assert_eq!(
            g.assess_independence(C, D).unwrap(),
            EvidenceIndependenceV1::Independent
        );
    }

    #[test]
    fn partially_shared_ancestry_is_preserved() {
        let g = graph(vec![
            root(A),
            root(B),
            root(C),
            derived(D, &[A, B]),
            derived(E, &[A, C]),
        ]);
        assert_eq!(
            g.assess_independence(D, E).unwrap(),
            EvidenceIndependenceV1::PartiallyShared
        );
    }

    #[test]
    fn unknown_parent_fails_closed() {
        let raw = EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: F.into(),
            nodes: vec![derived(B, &[A])],
        };
        assert!(matches!(
            raw.validate(),
            Err(CognitiveLineageError::UnknownParent { .. })
        ));
    }

    #[test]
    fn cycles_fail_closed() {
        let raw = EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: F.into(),
            nodes: vec![derived(A, &[B]), derived(B, &[A])],
        };
        assert!(matches!(
            raw.validate(),
            Err(CognitiveLineageError::CycleDetected { .. })
        ));
    }

    #[test]
    fn validated_graph_revalidates_after_persistence() {
        let g = graph(vec![root(A), derived(B, &[A])]);
        let json = serde_json::to_string(&g).unwrap();
        let decoded: ValidatedEvidenceLineageGraphV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, g);
    }
}