// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit witnesses for pairwise-independent evidence-item sets.
//!
//! Distinct ancestry roots and independent evidence items are different concepts.
//! A single derived evidence item may inherit several roots; that does not turn it
//! into several independent confirmations. This module issues a witness only for
//! a selected set of evidence **items** whose complete ancestry-root sets are
//! pairwise disjoint and whose items are pairwise `EvidenceIndependenceV1::Independent`.
//!
//! The issued witness is shadow epistemic structure only. It grants no relation,
//! disposition, belief, workspace, action, or self-improvement authority.

use crate::lineage::{
    CognitiveLineageError, EvidenceIndependenceV1, ValidatedEvidenceLineageGraphV1,
};
use serde::Serialize;
use std::collections::HashSet;

pub const INDEPENDENT_EVIDENCE_SET_WITNESS_SCHEMA_VERSION: u16 = 1;
pub const INDEPENDENT_EVIDENCE_SET_WITNESS_PROFILE_V1: &str =
    "rca-independent-evidence-set-witness-v1";

pub const INDEPENDENT_EVIDENCE_SET_WITNESS_CONTRACT_V1: &str = concat!(
    "rca-independent-evidence-set-witness-v1\n",
    "input=validated_evidence_lineage_graph+selected_evidence_item_ids\n",
    "selected_item_identity_is_not_ancestry_root_identity\n",
    "one_multiroot_derived_item_is_one_evidence_item_not_many_confirmations\n",
    "every_selected_item_root_set_is_recomputed_from_closed_lineage\n",
    "every_distinct_selected_item_pair_must_assess_exactly_independent\n",
    "pairwise_independent_items_require_disjoint_complete_ancestry_root_sets\n",
    "ancestor_derived_same_root_and_partial_overlap_pairs_fail_closed\n",
    "selected_item_input_order_does_not_change_witness_identity\n",
    "witness_identity=blake3_explicit_items+root_sets+pair_topology\n",
    "issued_witness=is_private_non_deserializable_shadow_artifact\n",
    "distinct_root_count_is_not_independent_evidence_item_count\n",
    "witness_is_not_relation_disposition_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-independent-evidence-set-witness-contract:v1\0";
const WITNESS_ID_DOMAIN: &[u8] = b"symthaea:rca-independent-evidence-set-witness:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentEvidenceItemV1 {
    evidence_id: String,
    root_ids: Vec<String>,
}

impl IndependentEvidenceItemV1 {
    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    /// Complete canonical ancestry roots for this one selected evidence item.
    /// Multiple roots here do not increase the number of evidence items.
    pub fn root_ids(&self) -> &[String] {
        &self.root_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentEvidenceItemPairV1 {
    left_evidence_id: String,
    right_evidence_id: String,
}

impl IndependentEvidenceItemPairV1 {
    pub fn left_evidence_id(&self) -> &str {
        &self.left_evidence_id
    }

    pub fn right_evidence_id(&self) -> &str {
        &self.right_evidence_id
    }
}

/// Issued witness for one exact selected set of pairwise-independent evidence
/// items. Private fields and absence of `Deserialize` are deliberate: archived
/// bytes are audit material; current trust requires recomputation from a currently
/// validated closed lineage graph.
#[must_use = "independent evidence set witnesses are shadow epistemic artifacts and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentEvidenceSetWitnessV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    witness_id: String,
    items: Vec<IndependentEvidenceItemV1>,
    pairs: Vec<IndependentEvidenceItemPairV1>,
    distinct_root_ids: Vec<String>,
}

impl IndependentEvidenceSetWitnessV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn witness_id(&self) -> &str {
        &self.witness_id
    }

    /// Selected evidence items. Cardinality here is item cardinality, not root
    /// cardinality and not pair-edge cardinality.
    pub fn items(&self) -> &[IndependentEvidenceItemV1] {
        &self.items
    }

    /// Canonical complete pair topology for the selected items. Every pair in an
    /// issued witness has independently re-evaluated to `Independent`.
    pub fn pairs(&self) -> &[IndependentEvidenceItemPairV1] {
        &self.pairs
    }

    /// Union of all ancestry roots, retained for provenance/audit. Its length is
    /// deliberately not exposed as an "independence count" API.
    pub fn distinct_root_ids(&self) -> &[String] {
        &self.distinct_root_ids
    }
}

pub fn independent_evidence_set_witness_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        INDEPENDENT_EVIDENCE_SET_WITNESS_CONTRACT_V1.as_bytes(),
    )
}

/// Issue an exact pairwise-independent evidence-item-set witness.
///
/// The function recomputes complete root sets and pair independence from the
/// validated lineage graph. Callers cannot supply root sets, pair statuses, or an
/// independence count.
pub fn issue_independent_evidence_set_witness_v1(
    graph: &ValidatedEvidenceLineageGraphV1,
    selected_evidence_ids: &[String],
) -> Result<IndependentEvidenceSetWitnessV1, EvidenceSetWitnessError> {
    if selected_evidence_ids.is_empty() {
        return Err(EvidenceSetWitnessError::EmptySelection);
    }

    let mut selected = selected_evidence_ids.to_vec();
    selected.sort();
    for pair in selected.windows(2) {
        if pair[0] == pair[1] {
            return Err(EvidenceSetWitnessError::DuplicateEvidenceId {
                evidence_id: pair[0].clone(),
            });
        }
    }

    let mut items = Vec::with_capacity(selected.len());
    let mut root_union = HashSet::new();
    for evidence_id in &selected {
        let mut root_ids = graph
            .root_ids(evidence_id)
            .map_err(EvidenceSetWitnessError::Lineage)?
            .into_iter()
            .collect::<Vec<_>>();
        root_ids.sort();
        if root_ids.is_empty() {
            return Err(EvidenceSetWitnessError::EmptyRootSet {
                evidence_id: evidence_id.clone(),
            });
        }
        for root_id in &root_ids {
            root_union.insert(root_id.clone());
        }
        items.push(IndependentEvidenceItemV1 {
            evidence_id: evidence_id.clone(),
            root_ids,
        });
    }

    let mut pairs = Vec::with_capacity(pair_count(items.len()));
    for left_index in 0..items.len() {
        for right_index in (left_index + 1)..items.len() {
            let left = &items[left_index];
            let right = &items[right_index];
            let status = graph
                .assess_independence(&left.evidence_id, &right.evidence_id)
                .map_err(EvidenceSetWitnessError::Lineage)?;
            if status != EvidenceIndependenceV1::Independent {
                return Err(EvidenceSetWitnessError::NonIndependentPair {
                    left_evidence_id: left.evidence_id.clone(),
                    right_evidence_id: right.evidence_id.clone(),
                    status,
                });
            }
            if has_root_overlap(&left.root_ids, &right.root_ids) {
                return Err(EvidenceSetWitnessError::IndependentStatusRootOverlap {
                    left_evidence_id: left.evidence_id.clone(),
                    right_evidence_id: right.evidence_id.clone(),
                });
            }
            pairs.push(IndependentEvidenceItemPairV1 {
                left_evidence_id: left.evidence_id.clone(),
                right_evidence_id: right.evidence_id.clone(),
            });
        }
    }

    let mut distinct_root_ids = root_union.into_iter().collect::<Vec<_>>();
    distinct_root_ids.sort();
    let profile_contract_digest = independent_evidence_set_witness_profile_digest_v1();
    let witness_id = independent_evidence_set_witness_id_v1(
        &profile_contract_digest,
        &items,
        &pairs,
        &distinct_root_ids,
    );

    Ok(IndependentEvidenceSetWitnessV1 {
        schema_version: INDEPENDENT_EVIDENCE_SET_WITNESS_SCHEMA_VERSION,
        profile: INDEPENDENT_EVIDENCE_SET_WITNESS_PROFILE_V1.to_string(),
        profile_contract_digest,
        witness_id,
        items,
        pairs,
        distinct_root_ids,
    })
}

fn independent_evidence_set_witness_id_v1(
    profile_contract_digest: &str,
    items: &[IndependentEvidenceItemV1],
    pairs: &[IndependentEvidenceItemPairV1],
    distinct_root_ids: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(WITNESS_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &INDEPENDENT_EVIDENCE_SET_WITNESS_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_count(&mut hasher, b"item_count", items.len());
    for item in items {
        hash_text(&mut hasher, b"evidence_id", &item.evidence_id);
        hash_count(&mut hasher, b"item_root_count", item.root_ids.len());
        for root_id in &item.root_ids {
            hash_text(&mut hasher, b"item_root_id", root_id);
        }
    }
    hash_count(&mut hasher, b"pair_count", pairs.len());
    for pair in pairs {
        hash_text(&mut hasher, b"left_evidence_id", &pair.left_evidence_id);
        hash_text(&mut hasher, b"right_evidence_id", &pair.right_evidence_id);
    }
    hash_count(&mut hasher, b"distinct_root_count", distinct_root_ids.len());
    for root_id in distinct_root_ids {
        hash_text(&mut hasher, b"distinct_root_id", root_id);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn has_root_overlap(left: &[String], right: &[String]) -> bool {
    let left_set: HashSet<&str> = left.iter().map(String::as_str).collect();
    right.iter().any(|root| left_set.contains(root.as_str()))
}

fn pair_count(item_count: usize) -> usize {
    item_count.saturating_mul(item_count.saturating_sub(1)) / 2
}

fn hash_count(hasher: &mut blake3::Hasher, label: &[u8], count: usize) {
    hash_bytes(hasher, label, &(count as u64).to_le_bytes());
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceSetWitnessError {
    EmptySelection,
    DuplicateEvidenceId {
        evidence_id: String,
    },
    EmptyRootSet {
        evidence_id: String,
    },
    NonIndependentPair {
        left_evidence_id: String,
        right_evidence_id: String,
        status: EvidenceIndependenceV1,
    },
    IndependentStatusRootOverlap {
        left_evidence_id: String,
        right_evidence_id: String,
    },
    Lineage(CognitiveLineageError),
}

impl std::fmt::Display for EvidenceSetWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySelection => {
                f.write_str("independent evidence set witness requires at least one evidence item")
            }
            Self::DuplicateEvidenceId { evidence_id } => write!(
                f,
                "independent evidence set selection contains duplicate evidence item {evidence_id}"
            ),
            Self::EmptyRootSet { evidence_id } => {
                write!(f, "evidence item {evidence_id} resolved to an empty ancestry-root set")
            }
            Self::NonIndependentPair {
                left_evidence_id,
                right_evidence_id,
                status,
            } => write!(
                f,
                "evidence items {left_evidence_id} and {right_evidence_id} are not independent: {status:?}"
            ),
            Self::IndependentStatusRootOverlap {
                left_evidence_id,
                right_evidence_id,
            } => write!(
                f,
                "evidence items {left_evidence_id} and {right_evidence_id} reported independent but their complete root sets overlap"
            ),
            Self::Lineage(error) => write!(f, "evidence lineage rejected witness input: {error}"),
        }
    }
}

impl std::error::Error for EvidenceSetWitnessError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::{
        CognitiveDerivationKindV1, EvidenceLineageGraphV1, EvidenceLineageNodeV1,
        COGNITIVE_LINEAGE_SCHEMA_VERSION,
    };

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const GRAPH: &str = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn node(id: &str, parents: &[&str], kind: CognitiveDerivationKindV1) -> crate::lineage::ValidatedEvidenceLineageNodeV1 {
        EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: id.into(),
            parent_ids: parents.iter().map(|value| (*value).to_string()).collect(),
            derivation_kind: kind,
        }
        .validate()
        .unwrap()
    }

    fn graph() -> ValidatedEvidenceLineageGraphV1 {
        EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: GRAPH.into(),
            nodes: vec![
                node(A, &[], CognitiveDerivationKindV1::RootObservation),
                node(B, &[], CognitiveDerivationKindV1::RootObservation),
                node(C, &[A, B], CognitiveDerivationKindV1::Inference),
                node(D, &[], CognitiveDerivationKindV1::RootObservation),
                node(E, &[A], CognitiveDerivationKindV1::Transformation),
            ],
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn multiroot_derived_item_is_one_independent_item_not_multiple_confirmations() {
        let witness = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[C.to_string(), D.to_string()],
        )
        .unwrap();
        assert_eq!(witness.items().len(), 2);
        assert_eq!(witness.items()[0].root_ids().len(), 2);
        assert_eq!(witness.items()[1].root_ids().len(), 1);
        assert_eq!(witness.distinct_root_ids().len(), 3);
        assert_eq!(witness.pairs().len(), 1);
    }

    #[test]
    fn shared_root_siblings_cannot_form_independent_set_witness() {
        let error = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[E.to_string(), C.to_string()],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EvidenceSetWitnessError::NonIndependentPair {
                status: EvidenceIndependenceV1::PartiallyShared,
                ..
            }
        ));
    }

    #[test]
    fn ancestor_and_descendant_cannot_form_independent_set_witness() {
        let error = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[A.to_string(), E.to_string()],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EvidenceSetWitnessError::NonIndependentPair {
                status: EvidenceIndependenceV1::Derived,
                ..
            }
        ));
    }

    #[test]
    fn selection_order_does_not_change_witness_identity() {
        let cd = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[C.to_string(), D.to_string()],
        )
        .unwrap();
        let dc = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[D.to_string(), C.to_string()],
        )
        .unwrap();
        assert_eq!(cd, dc);
        assert_eq!(cd.witness_id(), dc.witness_id());
    }

    #[test]
    fn duplicate_selected_item_fails_closed() {
        assert!(matches!(
            issue_independent_evidence_set_witness_v1(
                &graph(),
                &[D.to_string(), D.to_string()],
            ),
            Err(EvidenceSetWitnessError::DuplicateEvidenceId { .. })
        ));
    }

    #[test]
    fn unknown_item_fails_through_lineage_validation() {
        let unknown = "sha256:9999999999999999999999999999999999999999999999999999999999999999";
        assert!(matches!(
            issue_independent_evidence_set_witness_v1(&graph(), &[unknown.to_string()]),
            Err(EvidenceSetWitnessError::Lineage(
                CognitiveLineageError::UnknownEvidenceId { .. }
            ))
        ));
    }

    #[test]
    fn selected_item_change_changes_witness_identity() {
        let a = issue_independent_evidence_set_witness_v1(&graph(), &[D.to_string()]).unwrap();
        let b = issue_independent_evidence_set_witness_v1(&graph(), &[C.to_string()]).unwrap();
        assert_ne!(a.witness_id(), b.witness_id());
    }

    #[test]
    fn issued_witness_serializes_for_audit_only() {
        let witness = issue_independent_evidence_set_witness_v1(
            &graph(),
            &[C.to_string(), D.to_string()],
        )
        .unwrap();
        let encoded = serde_json::to_string(&witness).unwrap();
        assert!(encoded.contains(witness.witness_id()));
        assert!(encoded.contains(witness.profile_contract_digest()));
    }
}
