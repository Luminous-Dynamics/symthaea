// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit witnesses for pairwise-independent interpretation-root sets.
//!
//! The interpretation-lineage graph preserves exact root topology, but a later
//! disposition engine should not be responsible for discovering an independent
//! root set. This module performs that selection check ahead of disposition and
//! issues a content-addressed witness only when every selected distinct root pair
//! is already `IndependenceQualified` in one exact issued interpretation lineage.
//!
//! The witness is shadow epistemic structure only. It is not evidence truth,
//! proposition support, belief, workspace, action, or self-improvement authority.

use crate::interpretation_lineage::{
    InterpretationIndependenceStatusV1, InterpretationLineageV1,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

pub const INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_SCHEMA_VERSION: u16 = 1;
pub const INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_PROFILE_V1: &str =
    "rca-independent-interpretation-root-set-witness-v1";

pub const INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_CONTRACT_V1: &str = concat!(
    "rca-independent-interpretation-root-set-witness-v1\n",
    "input=issued_interpretation_lineage+selected_interpretation_root_ids\n",
    "selected_root_identity_is_not_declaration_count_or_pair_edge_count\n",
    "every_selected_root_must_exist_in_exact_lineage\n",
    "every_distinct_selected_root_pair_must_be_independence_qualified\n",
    "every_qualified_pair_must_carry_exact_qualification_identity\n",
    "distinct_roots_with_unknown_independence_fail_closed\n",
    "selected_root_input_order_does_not_change_witness_identity\n",
    "witness_identity=blake3_exact_lineage+roots+qualified_pair_ids\n",
    "issued_witness=is_private_non_deserializable_shadow_artifact\n",
    "witness_is_not_truth_support_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] =
    b"symthaea:rca-independent-interpretation-root-set-witness-contract:v1\0";
const WITNESS_ID_DOMAIN: &[u8] =
    b"symthaea:rca-independent-interpretation-root-set-witness:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentInterpretationRootPairV1 {
    left_interpretation_root_id: String,
    right_interpretation_root_id: String,
    qualification_id: String,
}

impl IndependentInterpretationRootPairV1 {
    pub fn left_interpretation_root_id(&self) -> &str {
        &self.left_interpretation_root_id
    }

    pub fn right_interpretation_root_id(&self) -> &str {
        &self.right_interpretation_root_id
    }

    pub fn qualification_id(&self) -> &str {
        &self.qualification_id
    }
}

/// Issued witness for one exact selected set of pairwise-independent
/// interpretation roots under one exact interpretation-lineage generation.
///
/// This type intentionally has private fields and no `Deserialize` path.
/// Archived bytes are audit material only; current trust requires reissuing the
/// witness from a currently issued interpretation lineage.
#[must_use = "interpretation-root-set witnesses are shadow epistemic artifacts and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentInterpretationRootSetWitnessV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    witness_id: String,
    proposition_id: String,
    interpretation_lineage_id: String,
    root_ids: Vec<String>,
    pairs: Vec<IndependentInterpretationRootPairV1>,
}

impl IndependentInterpretationRootSetWitnessV1 {
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

    pub fn proposition_id(&self) -> &str {
        &self.proposition_id
    }

    pub fn interpretation_lineage_id(&self) -> &str {
        &self.interpretation_lineage_id
    }

    /// Canonical selected interpretation roots. Its length is the exact root-set
    /// cardinality a preregistered policy may compare against its threshold.
    pub fn root_ids(&self) -> &[String] {
        &self.root_ids
    }

    /// Complete pair topology for the selected set. Every pair carries the exact
    /// qualification identity that made that pair eligible as independent.
    pub fn pairs(&self) -> &[IndependentInterpretationRootPairV1] {
        &self.pairs
    }
}

pub fn independent_interpretation_root_set_witness_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_CONTRACT_V1.as_bytes(),
    )
}

/// Issue an exact pairwise-independent interpretation-root-set witness.
pub fn issue_independent_interpretation_root_set_witness_v1(
    lineage: &InterpretationLineageV1,
    selected_root_ids: &[String],
) -> Result<IndependentInterpretationRootSetWitnessV1, InterpretationSetWitnessError> {
    let known_roots = lineage
        .roots()
        .iter()
        .map(|root| root.interpretation_root_id().to_string())
        .collect::<HashSet<_>>();

    let pair_facts = lineage
        .root_pair_assessments()
        .iter()
        .map(|pair| {
            let key = canonical_owned_pair(
                pair.left_interpretation_root_id(),
                pair.right_interpretation_root_id(),
            );
            (
                key,
                (
                    pair.status(),
                    pair.qualification_id().map(str::to_string),
                ),
            )
        })
        .collect::<HashMap<_, _>>();

    let (root_ids, pairs) = validate_root_selection(selected_root_ids, &known_roots, &pair_facts)?;
    let profile_contract_digest = independent_interpretation_root_set_witness_profile_digest_v1();
    let witness_id = independent_interpretation_root_set_witness_id_v1(
        &profile_contract_digest,
        lineage.proposition_id(),
        lineage.lineage_id(),
        &root_ids,
        &pairs,
    );

    Ok(IndependentInterpretationRootSetWitnessV1 {
        schema_version: INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_SCHEMA_VERSION,
        profile: INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_PROFILE_V1.to_string(),
        profile_contract_digest,
        witness_id,
        proposition_id: lineage.proposition_id().to_string(),
        interpretation_lineage_id: lineage.lineage_id().to_string(),
        root_ids,
        pairs,
    })
}

type PairFact = (InterpretationIndependenceStatusV1, Option<String>);

fn validate_root_selection(
    selected_root_ids: &[String],
    known_roots: &HashSet<String>,
    pair_facts: &HashMap<(String, String), PairFact>,
) -> Result<(Vec<String>, Vec<IndependentInterpretationRootPairV1>), InterpretationSetWitnessError> {
    if selected_root_ids.is_empty() {
        return Err(InterpretationSetWitnessError::EmptySelection);
    }

    let mut selected = selected_root_ids.to_vec();
    selected.sort();
    for pair in selected.windows(2) {
        if pair[0] == pair[1] {
            return Err(InterpretationSetWitnessError::DuplicateInterpretationRootId {
                interpretation_root_id: pair[0].clone(),
            });
        }
    }

    for root_id in &selected {
        if !known_roots.contains(root_id) {
            return Err(InterpretationSetWitnessError::UnknownInterpretationRootId {
                interpretation_root_id: root_id.clone(),
            });
        }
    }

    let mut pairs = Vec::with_capacity(pair_count(selected.len()));
    for left_index in 0..selected.len() {
        for right_index in (left_index + 1)..selected.len() {
            let left = &selected[left_index];
            let right = &selected[right_index];
            let key = canonical_owned_pair(left, right);
            let Some((status, qualification_id)) = pair_facts.get(&key) else {
                return Err(InterpretationSetWitnessError::MissingRootPairAssessment {
                    left_interpretation_root_id: left.clone(),
                    right_interpretation_root_id: right.clone(),
                });
            };
            if *status != InterpretationIndependenceStatusV1::IndependenceQualified {
                return Err(InterpretationSetWitnessError::NonIndependentRootPair {
                    left_interpretation_root_id: left.clone(),
                    right_interpretation_root_id: right.clone(),
                    status: *status,
                });
            }
            let Some(qualification_id) = qualification_id.as_ref() else {
                return Err(InterpretationSetWitnessError::QualifiedPairMissingQualificationId {
                    left_interpretation_root_id: left.clone(),
                    right_interpretation_root_id: right.clone(),
                });
            };
            pairs.push(IndependentInterpretationRootPairV1 {
                left_interpretation_root_id: left.clone(),
                right_interpretation_root_id: right.clone(),
                qualification_id: qualification_id.clone(),
            });
        }
    }

    Ok((selected, pairs))
}

fn independent_interpretation_root_set_witness_id_v1(
    profile_contract_digest: &str,
    proposition_id: &str,
    interpretation_lineage_id: &str,
    root_ids: &[String],
    pairs: &[IndependentInterpretationRootPairV1],
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
        &INDEPENDENT_INTERPRETATION_ROOT_SET_WITNESS_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", proposition_id);
    hash_text(
        &mut hasher,
        b"interpretation_lineage_id",
        interpretation_lineage_id,
    );
    hash_count(&mut hasher, b"selected_root_count", root_ids.len());
    for root_id in root_ids {
        hash_text(&mut hasher, b"selected_root_id", root_id);
    }
    hash_count(&mut hasher, b"qualified_pair_count", pairs.len());
    for pair in pairs {
        hash_text(
            &mut hasher,
            b"left_interpretation_root_id",
            &pair.left_interpretation_root_id,
        );
        hash_text(
            &mut hasher,
            b"right_interpretation_root_id",
            &pair.right_interpretation_root_id,
        );
        hash_text(&mut hasher, b"qualification_id", &pair.qualification_id);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_owned_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
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
pub enum InterpretationSetWitnessError {
    EmptySelection,
    DuplicateInterpretationRootId {
        interpretation_root_id: String,
    },
    UnknownInterpretationRootId {
        interpretation_root_id: String,
    },
    MissingRootPairAssessment {
        left_interpretation_root_id: String,
        right_interpretation_root_id: String,
    },
    NonIndependentRootPair {
        left_interpretation_root_id: String,
        right_interpretation_root_id: String,
        status: InterpretationIndependenceStatusV1,
    },
    QualifiedPairMissingQualificationId {
        left_interpretation_root_id: String,
        right_interpretation_root_id: String,
    },
}

impl std::fmt::Display for InterpretationSetWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySelection => {
                f.write_str("interpretation-root-set witness requires at least one root")
            }
            Self::DuplicateInterpretationRootId {
                interpretation_root_id,
            } => write!(f, "duplicate interpretation root id: {interpretation_root_id}"),
            Self::UnknownInterpretationRootId {
                interpretation_root_id,
            } => write!(f, "unknown interpretation root id: {interpretation_root_id}"),
            Self::MissingRootPairAssessment {
                left_interpretation_root_id,
                right_interpretation_root_id,
            } => write!(
                f,
                "missing interpretation-root pair assessment for {left_interpretation_root_id} and {right_interpretation_root_id}"
            ),
            Self::NonIndependentRootPair {
                left_interpretation_root_id,
                right_interpretation_root_id,
                status,
            } => write!(
                f,
                "interpretation roots {left_interpretation_root_id} and {right_interpretation_root_id} are not qualified independent: {status:?}"
            ),
            Self::QualifiedPairMissingQualificationId {
                left_interpretation_root_id,
                right_interpretation_root_id,
            } => write!(
                f,
                "qualified interpretation-root pair {left_interpretation_root_id}/{right_interpretation_root_id} is missing its qualification id"
            ),
        }
    }
}

impl std::error::Error for InterpretationSetWitnessError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(id_byte: char) -> String {
        format!("blake3:{}", id_byte.to_string().repeat(64))
    }

    fn qualification(id_byte: char) -> String {
        format!("sha256:{}", id_byte.to_string().repeat(64))
    }

    fn known(ids: &[String]) -> HashSet<String> {
        ids.iter().cloned().collect()
    }

    fn facts(
        entries: &[(&String, &String, InterpretationIndependenceStatusV1, Option<String>)],
    ) -> HashMap<(String, String), PairFact> {
        entries
            .iter()
            .map(|(left, right, status, qualification_id)| {
                (
                    canonical_owned_pair(left, right),
                    (*status, qualification_id.clone()),
                )
            })
            .collect()
    }

    #[test]
    fn selected_root_order_is_canonical() {
        let a = root('a');
        let b = root('b');
        let pair_facts = facts(&[(&a, &b, InterpretationIndependenceStatusV1::IndependenceQualified, Some(qualification('c')))]);
        let (forward, _) = validate_root_selection(&[a.clone(), b.clone()], &known(&[a.clone(), b.clone()]), &pair_facts).unwrap();
        let (reverse, _) = validate_root_selection(&[b, a], &known(&forward), &pair_facts).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn distinct_roots_with_unknown_independence_fail_closed() {
        let a = root('a');
        let b = root('b');
        let pair_facts = facts(&[(&a, &b, InterpretationIndependenceStatusV1::DistinctRootsIndependenceUnknown, None)]);
        assert!(matches!(
            validate_root_selection(&[a.clone(), b.clone()], &known(&[a, b]), &pair_facts),
            Err(InterpretationSetWitnessError::NonIndependentRootPair { .. })
        ));
    }

    #[test]
    fn qualified_pair_requires_qualification_identity() {
        let a = root('a');
        let b = root('b');
        let pair_facts = facts(&[(&a, &b, InterpretationIndependenceStatusV1::IndependenceQualified, None)]);
        assert!(matches!(
            validate_root_selection(&[a.clone(), b.clone()], &known(&[a, b]), &pair_facts),
            Err(InterpretationSetWitnessError::QualifiedPairMissingQualificationId { .. })
        ));
    }

    #[test]
    fn duplicate_selected_root_fails_closed() {
        let a = root('a');
        assert!(matches!(
            validate_root_selection(&[a.clone(), a.clone()], &known(&[a]), &HashMap::new()),
            Err(InterpretationSetWitnessError::DuplicateInterpretationRootId { .. })
        ));
    }

    #[test]
    fn unknown_selected_root_fails_closed() {
        let a = root('a');
        let b = root('b');
        assert!(matches!(
            validate_root_selection(&[a], &known(&[b]), &HashMap::new()),
            Err(InterpretationSetWitnessError::UnknownInterpretationRootId { .. })
        ));
    }

    #[test]
    fn single_root_is_valid_without_pair_edges() {
        let a = root('a');
        let (roots, pairs) =
            validate_root_selection(&[a.clone()], &known(&[a.clone()]), &HashMap::new()).unwrap();
        assert_eq!(roots, vec![a]);
        assert!(pairs.is_empty());
    }
}
