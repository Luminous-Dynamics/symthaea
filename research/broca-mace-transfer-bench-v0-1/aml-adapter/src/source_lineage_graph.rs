// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic replay of a source-lineage graph manifest.
//!
//! This module verifies graph structure, exact candidate parent bindings,
//! source-root identities, common-cause grouping and closure digests. It does
//! not fetch raw source bytes or independently authenticate ancestor metadata;
//! those remain responsibilities of the source resolver and source-origin
//! receipt verifiers.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::decision_sufficiency::{
    DecisionNeedV1, DecisionSufficiencyUseBindingV1, VerifiedDecisionSufficiencyV1,
};
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationSignatureVerifierV1,
};
use crate::qualified_candidate_evidence::QualifiedCandidateEvidenceBatchV1;
use crate::qualification_dsse::parse_strict_json_bounded;
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;
use crate::source_lineage_receipt::{
    SourceLineageRejectV1, VerifiedSourceLineageSetV1,
};

const MAX_GRAPH_BYTES_V1: usize = 32 * 1024 * 1024;
const MAX_GRAPH_NODES_V1: usize = 100_000;
const MAX_PARENTS_PER_NODE_V1: usize = 1_024;
const MAX_REQUIREMENTS_PER_NODE_V1: usize = 1_024;
const MAX_COMMON_CAUSE_GROUPS_PER_NODE_V1: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceLineageGraphNodeV1 {
    evidence_id: String,
    source_id: String,
    source_sha256: String,
    payload_sha256: String,
    evidence_class_sha256: String,
    satisfies_requirement_ids: Vec<String>,
    parent_evidence_ids: Vec<String>,
    derivation_receipt_sha256: Option<String>,
    common_cause_group_sha256: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceLineageGraphV1 {
    qualified_evidence_set_sha256: String,
    decision_need_sha256: String,
    nodes: Vec<SourceLineageGraphNodeV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayedSourceRootFamilyV1 {
    pub requirement_id: String,
    pub root_family_sha256: String,
    pub lineage_closure_sha256: String,
    pub source_root_count: usize,
    pub closure_node_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceLineageGraphV1 {
    graph_sha256: String,
    qualified_evidence_set_sha256: String,
    decision_need_sha256: String,
    families: Vec<ReplayedSourceRootFamilyV1>,
}

impl VerifiedSourceLineageGraphV1 {
    pub fn graph_sha256(&self) -> &str {
        &self.graph_sha256
    }

    pub fn qualified_evidence_set_sha256(&self) -> &str {
        &self.qualified_evidence_set_sha256
    }

    pub fn decision_need_sha256(&self) -> &str {
        &self.decision_need_sha256
    }

    pub fn families(&self) -> &[ReplayedSourceRootFamilyV1] {
        &self.families
    }
}

/// Replay an exact graph manifest against already qualified evidence and the
/// three-authority lineage receipts. The graph digest is SHA-256 over exact
/// UTF-8 manifest bytes, matching the digest signed by the resolver.
///
/// The caller must separately ensure that every ancestor node's source and
/// derivation metadata is backed by authenticated source-origin receipts.
/// Graph replay proves internal consistency and declared ancestry, not that an
/// operator disclosed every real-world parent or common cause.
pub fn replay_source_lineage_graph_v1(
    graph_json: &str,
    qualified: &QualifiedCandidateEvidenceBatchV1,
    qualification_binding: &EvidenceQualificationBindingV1<'_>,
    current_qualification_verifier: &impl EvidenceQualificationSignatureVerifierV1,
    sufficiency: &VerifiedDecisionSufficiencyV1,
    sufficiency_binding: &DecisionSufficiencyUseBindingV1<'_>,
    current_evaluator_profile: &PinnedEd25519QualificationVerifierV1,
    decision_need: &DecisionNeedV1,
    lineage: &VerifiedSourceLineageSetV1,
    current_lineage_profile: &PinnedEd25519QualificationVerifierV1,
) -> Result<VerifiedSourceLineageGraphV1, SourceLineageRejectV1> {
    lineage.validate_at_use(
        qualified,
        qualification_binding,
        current_qualification_verifier,
        sufficiency,
        sufficiency_binding,
        current_evaluator_profile,
        decision_need,
        current_lineage_profile,
    )?;
    if graph_json.len() > MAX_GRAPH_BYTES_V1 {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    let graph_sha256 = sha256_hex(graph_json.as_bytes());
    let decision_need_sha256 = decision_need.digest()
        .map_err(SourceLineageRejectV1::Sufficiency)?;
    if qualified.decision_need_sha256() != decision_need_sha256.as_str()
        || qualified.qualified_candidate_set_sha256() != lineage.qualified_evidence_set_sha256()
        || qualified.qualified_candidate_set_sha256() != lineage_graph_expected_set(lineage)
        || decision_need_sha256 != lineage_graph_expected_need(lineage)
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }

    let value = parse_strict_json_bounded(graph_json.as_bytes(), MAX_GRAPH_BYTES_V1)
        .map_err(|_| SourceLineageRejectV1::InvalidGraph)?;
    let graph = parse_graph(&value)?;
    if graph.qualified_evidence_set_sha256 != qualified.qualified_candidate_set_sha256()
        || graph.decision_need_sha256 != decision_need_sha256
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }

    let nodes = graph.nodes.into_iter()
        .map(|node| (node.evidence_id.clone(), node))
        .collect::<BTreeMap<_, _>>();
    if nodes.len() != value["nodes"].as_array().map_or(0, Vec::len) {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }

    // Every requirement tag must exist in the consumer-owned manifest and
    // the graph's evidence class must equal that requirement's exact class.
    for node in nodes.values() {
        for requirement_id in &node.satisfies_requirement_ids {
            let requirement = decision_need.requirements.iter()
                .find(|item| item.requirement_id == *requirement_id)
                .ok_or(SourceLineageRejectV1::BindingMismatch)?;
            if node.evidence_class_sha256 != requirement.evidence_class_sha256 {
                return Err(SourceLineageRejectV1::EvidenceClassMismatch);
            }
        }
    }

    // Candidate nodes are not free-form graph annotations: the graph must
    // exactly reproduce the source identities, payload commitments, parents
    // and derivation receipts that were in the qualified candidate records.
    for candidate in qualified.evidence() {
        let node = nodes.get(&candidate.evidence_id)
            .ok_or(SourceLineageRejectV1::CandidateLineageMismatch)?;
        if node.source_id != candidate.source_id
            || node.source_sha256 != candidate.source_sha256
            || node.payload_sha256 != candidate.payload_sha256
            || node.parent_evidence_ids != candidate.parent_evidence_ids
            || node.derivation_receipt_sha256 != candidate.derivation_receipt_sha256
        {
            return Err(SourceLineageRejectV1::CandidateLineageMismatch);
        }
    }

    validate_graph_acyclic(&nodes)?;

    let expected_commitments = lineage.replay_commitments();
    let mut replayed = Vec::new();
    for requirement in &decision_need.requirements {
        let terminals = qualified.evidence().iter()
            .filter(|candidate| {
                nodes.get(&candidate.evidence_id).is_some_and(|node| {
                    node.satisfies_requirement_ids.iter()
                        .any(|id| id == &requirement.requirement_id)
                })
            })
            .map(|candidate| candidate.evidence_id.clone())
            .collect::<Vec<_>>();
        if terminals.is_empty() {
            return Err(SourceLineageRejectV1::MissingRequirementEvidence);
        }
        replayed.extend(replay_requirement_families(
            &requirement.requirement_id,
            &terminals,
            &nodes,
        )?);
    }
    replayed.sort_by(|a, b| {
        (&a.requirement_id, &a.root_family_sha256)
            .cmp(&(&b.requirement_id, &b.root_family_sha256))
    });

    let expected_set = expected_commitments.iter().map(|item| {
        (&item.requirement_id, &item.root_family_sha256, &item.lineage_closure_sha256)
    }).collect::<BTreeSet<_>>();
    let actual_set = replayed.iter().map(|item| {
        (&item.requirement_id, &item.root_family_sha256, &item.lineage_closure_sha256)
    }).collect::<BTreeSet<_>>();
    if expected_commitments.iter().any(|item| item.source_graph_sha256 != graph_sha256) {
        return Err(SourceLineageRejectV1::GraphDigestMismatch);
    }
    if expected_set.len() != expected_commitments.len() || actual_set.len() != replayed.len() {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    if expected_set.iter().any(|(requirement, root, _)| {
        !actual_set.iter().any(|(r, f, _)| r == requirement && f == root)
    }) || actual_set.iter().any(|(requirement, root, _)| {
        !expected_set.iter().any(|(r, f, _)| r == requirement && f == root)
    }) {
        return Err(SourceLineageRejectV1::RootFamilyMismatch);
    }
    if expected_set != actual_set {
        return Err(SourceLineageRejectV1::ClosureDigestMismatch);
    }

    Ok(VerifiedSourceLineageGraphV1 {
        graph_sha256,
        qualified_evidence_set_sha256: graph.qualified_evidence_set_sha256,
        decision_need_sha256: graph.decision_need_sha256,
        families: replayed,
    })
}

fn lineage_graph_expected_set(lineage: &VerifiedSourceLineageSetV1) -> &str {
    lineage.qualified_evidence_set_sha256()
}

fn lineage_graph_expected_need(lineage: &VerifiedSourceLineageSetV1) -> &str {
    lineage.decision_need_sha256()
}

fn validate_graph_acyclic(
    nodes: &BTreeMap<String, SourceLineageGraphNodeV1>,
) -> Result<(), SourceLineageRejectV1> {
    // Edges point child -> parent; Kahn's algorithm runs parent -> child
    // using the reverse adjacency. Missing parents fail before cycle checks.
    let mut indegree = BTreeMap::<String, usize>::new();
    let mut children = BTreeMap::<String, Vec<String>>::new();
    for (id, node) in nodes {
        indegree.insert(id.clone(), node.parent_evidence_ids.len());
        for parent in &node.parent_evidence_ids {
            if !nodes.contains_key(parent) {
                return Err(SourceLineageRejectV1::DanglingParent);
            }
            children.entry(parent.clone()).or_default().push(id.clone());
        }
    }
    let mut queue = VecDeque::new();
    for (id, degree) in &indegree {
        if *degree == 0 {
            queue.push_back(id.clone());
        }
    }
    let mut visited = 0usize;
    while let Some(parent) = queue.pop_front() {
        visited += 1;
        if let Some(descendants) = children.get(&parent) {
            for child in descendants {
                let degree = indegree.get_mut(child)
                    .ok_or(SourceLineageRejectV1::InvalidGraph)?;
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    queue.push_back(child.clone());
                }
            }
        }
    }
    if visited != nodes.len() {
        return Err(SourceLineageRejectV1::GraphCycle);
    }
    Ok(())
}

fn replay_requirement_families(
    requirement_id: &str,
    terminals: &[String],
    nodes: &BTreeMap<String, SourceLineageGraphNodeV1>,
) -> Result<Vec<ReplayedSourceRootFamilyV1>, SourceLineageRejectV1> {
    // root digest -> all evidence nodes on at least one path from a
    // requirement-matching terminal to that root.
    let mut root_paths = BTreeMap::<String, BTreeSet<String>>::new();
    for terminal in terminals {
        let mut stack = vec![terminal.clone()];
        let mut closure_nodes = BTreeSet::new();
        let mut terminal_roots = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !closure_nodes.insert(id.clone()) {
                continue;
            }
            let node = nodes.get(&id).ok_or(SourceLineageRejectV1::DanglingParent)?;
            if node.parent_evidence_ids.is_empty() {
                terminal_roots.insert(source_root_identity_sha256(
                    &node.source_id, &node.source_sha256,
                ));
            } else {
                for parent in &node.parent_evidence_ids {
                    stack.push(parent.clone());
                }
            }
        }
        for root_digest in terminal_roots {
            root_paths.entry(root_digest).or_default()
                .extend(closure_nodes.iter().cloned());
        }
    }
    if root_paths.is_empty() {
        return Err(SourceLineageRejectV1::UnresolvedAncestors);
    }

    let root_ids = root_paths.keys().cloned().collect::<Vec<_>>();
    let mut union = UnionFind::new(root_ids.len());
    let root_index = root_ids.iter().enumerate()
        .map(|(index, root)| (root.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut common_cause_owner = BTreeMap::<String, usize>::new();
    for (root_digest, index) in &root_index {
        for path_node_id in root_paths.get(root_digest).into_iter().flatten() {
            let node = nodes.get(path_node_id).ok_or(SourceLineageRejectV1::InvalidGraph)?;
            for group in &node.common_cause_group_sha256 {
                if let Some(previous) = common_cause_owner.insert(group.clone(), *index) {
                    union.join(previous, *index);
                }
            }
        }
    }

    let mut grouped = BTreeMap::<usize, Vec<String>>::new();
    for (digest, index) in &root_index {
        grouped.entry(union.find(*index)).or_default().push(digest.clone());
    }
    let mut replayed = Vec::new();
    for family_roots in grouped.values_mut() {
        family_roots.sort();
    }
    for family_roots in grouped.values() {
        let mut family_groups = BTreeSet::new();
        let mut closure_nodes = BTreeSet::new();
        for root in family_roots {
            if let Some(path_nodes) = root_paths.get(root) {
                closure_nodes.extend(path_nodes.iter().cloned());
            }
            for path_node_id in root_paths.get(root).into_iter().flatten() {
                let node = nodes.get(path_node_id).ok_or(SourceLineageRejectV1::InvalidGraph)?;
                family_groups.extend(node.common_cause_group_sha256.iter().cloned());
            }
        }
        let family_sha256 = root_family_sha256(family_roots, &family_groups);
        let closure_sha256 = lineage_closure_sha256(
            requirement_id,
            &family_sha256,
            &closure_nodes,
            nodes,
        )?;
        replayed.push(ReplayedSourceRootFamilyV1 {
            requirement_id: requirement_id.to_owned(),
            root_family_sha256: family_sha256,
            lineage_closure_sha256: closure_sha256,
            source_root_count: family_roots.len(),
            closure_node_count: closure_nodes.len(),
        });
    }
    replayed.sort_by(|a, b| a.root_family_sha256.cmp(&b.root_family_sha256));
    Ok(replayed)
}

fn parse_graph(value: &Value) -> Result<SourceLineageGraphV1, SourceLineageRejectV1> {
    let object = value.as_object().ok_or(SourceLineageRejectV1::InvalidGraph)?;
    const TOP_FIELDS: &[&str] = &[
        "schema_version", "qualified_evidence_set_sha256",
        "decision_need_sha256", "nodes",
    ];
    if object.len() != TOP_FIELDS.len()
        || object.keys().any(|key| !TOP_FIELDS.contains(&key.as_str()))
        || object.get("schema_version").and_then(Value::as_str)
            != Some("mace_source_lineage_graph_v1")
    {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    let qualified_evidence_set_sha256 = digest_field(object, "qualified_evidence_set_sha256")?;
    let decision_need_sha256 = digest_field(object, "decision_need_sha256")?;
    let node_values = object.get("nodes").and_then(Value::as_array)
        .ok_or(SourceLineageRejectV1::InvalidGraph)?;
    if node_values.is_empty() || node_values.len() > MAX_GRAPH_NODES_V1 {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    let mut nodes = Vec::with_capacity(node_values.len());
    for value in node_values {
        let item = value.as_object().ok_or(SourceLineageRejectV1::InvalidGraph)?;
        const NODE_FIELDS: &[&str] = &[
            "evidence_id", "source_id", "source_sha256", "payload_sha256",
            "evidence_class_sha256", "satisfies_requirement_ids",
            "parent_evidence_ids", "derivation_receipt_sha256",
            "common_cause_group_sha256",
        ];
        if item.len() != NODE_FIELDS.len()
            || item.keys().any(|key| !NODE_FIELDS.contains(&key.as_str()))
        {
            return Err(SourceLineageRejectV1::InvalidGraph);
        }
        let evidence_id = atom_field(item, "evidence_id", 512)?;
        let source_id = atom_field(item, "source_id", 1024)?;
        let source_sha256 = digest_field(item, "source_sha256")?;
        let payload_sha256 = digest_field(item, "payload_sha256")?;
        let evidence_class_sha256 = digest_field(item, "evidence_class_sha256")?;
        let satisfies_requirement_ids = string_array(item, "satisfies_requirement_ids",
            MAX_REQUIREMENTS_PER_NODE_V1, true)?;
        let parent_evidence_ids = string_array(item, "parent_evidence_ids",
            MAX_PARENTS_PER_NODE_V1, true)?;
        let derivation_receipt_sha256 = match item.get("derivation_receipt_sha256") {
            Some(Value::Null) => None,
            Some(Value::String(value)) if valid_sha256(value) => Some(value.clone()),
            _ => return Err(SourceLineageRejectV1::InvalidGraph),
        };
        let common_cause_group_sha256 = string_array(item, "common_cause_group_sha256",
            MAX_COMMON_CAUSE_GROUPS_PER_NODE_V1, true)?;
        for value in &parent_evidence_ids {
            if value == &evidence_id {
                return Err(SourceLineageRejectV1::GraphCycle);
            }
        }
        if parent_evidence_ids.is_empty() && derivation_receipt_sha256.is_some()
            || !parent_evidence_ids.is_empty() && derivation_receipt_sha256.is_none()
        {
            return Err(SourceLineageRejectV1::InvalidGraph);
        }
        nodes.push(SourceLineageGraphNodeV1 {
            evidence_id,
            source_id,
            source_sha256,
            payload_sha256,
            evidence_class_sha256,
            satisfies_requirement_ids,
            parent_evidence_ids,
            derivation_receipt_sha256,
            common_cause_group_sha256,
        });
    }
    if nodes.windows(2).any(|pair| pair[0].evidence_id >= pair[1].evidence_id) {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    Ok(SourceLineageGraphV1 {
        qualified_evidence_set_sha256,
        decision_need_sha256,
        nodes,
    })
}

fn atom_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
    maximum: usize,
) -> Result<String, SourceLineageRejectV1> {
    let value = object.get(key).and_then(Value::as_str)
        .ok_or(SourceLineageRejectV1::InvalidGraph)?;
    if value.trim().is_empty() || value != value.trim()
        || value.len() > maximum || value.chars().any(char::is_control)
    {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    Ok(value.to_owned())
}

fn digest_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<String, SourceLineageRejectV1> {
    let value = object.get(key).and_then(Value::as_str)
        .ok_or(SourceLineageRejectV1::InvalidGraph)?;
    if !valid_sha256(value) {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    Ok(value.to_owned())
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
    maximum: usize,
    digest: bool,
) -> Result<Vec<String>, SourceLineageRejectV1> {
    let values = object.get(key).and_then(Value::as_array)
        .ok_or(SourceLineageRejectV1::InvalidGraph)?;
    if values.len() > maximum {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    let items = values.iter().map(|value| {
        let value = value.as_str().ok_or(SourceLineageRejectV1::InvalidGraph)?;
        if value.trim().is_empty() || value != value.trim()
            || value.chars().any(char::is_control)
            || digest && !valid_sha256(value)
        {
            return Err(SourceLineageRejectV1::InvalidGraph);
        }
        Ok(value.to_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    if items.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SourceLineageRejectV1::InvalidGraph);
    }
    Ok(items)
}

fn source_root_identity_sha256(source_id: &str, source_sha256: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:source-root-identity:v1\0");
    hash_field(&mut hasher, source_id);
    hash_field(&mut hasher, source_sha256);
    hex_digest(&hasher.finalize())
}

fn root_family_sha256(roots: &[String], common_causes: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:source-root-family:v1\0");
    hasher.update((roots.len() as u64).to_be_bytes());
    for root in roots {
        hash_field(&mut hasher, root);
    }
    hasher.update((common_causes.len() as u64).to_be_bytes());
    for cause in common_causes {
        hash_field(&mut hasher, cause);
    }
    hex_digest(&hasher.finalize())
}

fn lineage_closure_sha256(
    requirement_id: &str,
    root_family_sha256: &str,
    closure_ids: &BTreeSet<String>,
    nodes: &BTreeMap<String, SourceLineageGraphNodeV1>,
) -> Result<String, SourceLineageRejectV1> {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:lineage-closure:v1\0");
    hash_field(&mut hasher, requirement_id);
    hash_field(&mut hasher, root_family_sha256);
    hasher.update((closure_ids.len() as u64).to_be_bytes());
    for id in closure_ids {
        let node = nodes.get(id).ok_or(SourceLineageRejectV1::InvalidGraph)?;
        hash_field(&mut hasher, &node.evidence_id);
        hash_field(&mut hasher, &node.source_id);
        hash_field(&mut hasher, &node.source_sha256);
        hash_field(&mut hasher, &node.payload_sha256);
        hash_field(&mut hasher, &node.evidence_class_sha256);
        match &node.derivation_receipt_sha256 {
            Some(digest) => {
                hasher.update([1]);
                hash_field(&mut hasher, digest);
            }
            None => hasher.update([0]),
        }
        hasher.update((node.parent_evidence_ids.len() as u64).to_be_bytes());
        for parent in &node.parent_evidence_ids {
            hash_field(&mut hasher, parent);
        }
        hasher.update((node.common_cause_group_sha256.len() as u64).to_be_bytes());
        for cause in &node.common_cause_group_sha256 {
            hash_field(&mut hasher, cause);
        }
    }
    Ok(hex_digest(&hasher.finalize()))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(&digest)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect(), rank: vec![0; size] }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent != value {
            self.parent[value] = self.find(parent);
        }
        self.parent[value]
    }

    fn join(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] = self.rank[left_root].saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn root_identity_is_domain_separated_and_source_bound() {
        assert_ne!(
            source_root_identity_sha256("source-a", &"a".repeat(64)),
            source_root_identity_sha256("source-b", &"a".repeat(64))
        );
        assert_ne!(
            source_root_identity_sha256("source-a", &"a".repeat(64)),
            source_root_identity_sha256("source-a", &"b".repeat(64))
        );
    }

    #[test]
    fn common_cause_union_collapses_distinct_root_identities() {
        let mut union = UnionFind::new(3);
        union.join(0, 1);
        assert_eq!(union.find(0), union.find(1));
        assert_ne!(union.find(0), union.find(2));
    }

    #[test]
    fn strict_graph_parser_rejects_unsorted_or_duplicated_parent_edges() {
        let graph = json!({
            "schema_version": "mace_source_lineage_graph_v1",
            "qualified_evidence_set_sha256": "a".repeat(64),
            "decision_need_sha256": "b".repeat(64),
            "nodes": [{
                "evidence_id": "e-1", "source_id": "s-1",
                "source_sha256": "c".repeat(64), "payload_sha256": "d".repeat(64),
                "evidence_class_sha256": "e".repeat(64),
                "satisfies_requirement_ids": [],
                "parent_evidence_ids": ["e-0", "e-0"],
                "derivation_receipt_sha256": "f".repeat(64),
                "common_cause_group_sha256": []
            }]
        });
        assert_eq!(parse_graph(&graph), Err(SourceLineageRejectV1::InvalidGraph));
    }

    #[test]
    fn graph_replay_rejects_dangling_parent_and_cycle() {
        let root = SourceLineageGraphNodeV1 {
            evidence_id: "e-0".into(),
            source_id: "s-0".into(),
            source_sha256: "a".repeat(64),
            payload_sha256: "b".repeat(64),
            evidence_class_sha256: "c".repeat(64),
            satisfies_requirement_ids: Vec::new(),
            parent_evidence_ids: vec!["e-1".into()],
            derivation_receipt_sha256: Some("d".repeat(64)),
            common_cause_group_sha256: Vec::new(),
        };
        let child = SourceLineageGraphNodeV1 {
            evidence_id: "e-1".into(),
            source_id: "s-1".into(),
            source_sha256: "e".repeat(64),
            payload_sha256: "f".repeat(64),
            evidence_class_sha256: "a".repeat(64),
            satisfies_requirement_ids: Vec::new(),
            parent_evidence_ids: vec!["e-0".into()],
            derivation_receipt_sha256: Some("b".repeat(64)),
            common_cause_group_sha256: Vec::new(),
        };
        let cyclic = BTreeMap::from([
            ("e-0".to_owned(), root.clone()),
            ("e-1".to_owned(), child.clone()),
        ]);
        assert_eq!(validate_graph_acyclic(&cyclic), Err(SourceLineageRejectV1::GraphCycle));

        let dangling_node = SourceLineageGraphNodeV1 {
            parent_evidence_ids: vec!["missing-parent".into()],
            ..child
        };
        let dangling = BTreeMap::from([("e-1".to_owned(), dangling_node)]);
        assert_eq!(
            validate_graph_acyclic(&dangling),
            Err(SourceLineageRejectV1::DanglingParent)
        );
    }

    #[test]
    fn strict_graph_parser_rejects_missing_derivation_receipt() {
        let graph = json!({
            "schema_version": "mace_source_lineage_graph_v1",
            "qualified_evidence_set_sha256": "a".repeat(64),
            "decision_need_sha256": "b".repeat(64),
            "nodes": [{
                "evidence_id": "e-1", "source_id": "s-1",
                "source_sha256": "c".repeat(64), "payload_sha256": "d".repeat(64),
                "evidence_class_sha256": "e".repeat(64),
                "satisfies_requirement_ids": [],
                "parent_evidence_ids": ["e-0"],
                "derivation_receipt_sha256": null,
                "common_cause_group_sha256": []
            }]
        });
        assert_eq!(parse_graph(&graph), Err(SourceLineageRejectV1::InvalidGraph));
    }
}
