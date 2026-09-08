// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_epistemic_governance::{
    evidence_set_witness::{
        independent_evidence_set_witness_profile_digest_v1,
        issue_independent_evidence_set_witness_v1,
    },
    lineage::{
        CognitiveDerivationKindV1, EvidenceLineageGraphV1, EvidenceLineageNodeV1,
        ValidatedEvidenceLineageNodeV1, COGNITIVE_LINEAGE_SCHEMA_VERSION,
    },
    lineage_identity::{
        canonical_evidence_lineage_graph_id_v1,
        canonical_evidence_lineage_identity_profile_digest_v1,
    },
};

const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const LEGACY_GRAPH_LABEL: &str =
    "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn root(id: &str) -> ValidatedEvidenceLineageNodeV1 {
    EvidenceLineageNodeV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        evidence_id: id.into(),
        parent_ids: vec![],
        derivation_kind: CognitiveDerivationKindV1::RootObservation,
    }
    .validate()
    .expect("known-answer root must validate")
}

fn graph(nodes: Vec<ValidatedEvidenceLineageNodeV1>) -> symthaea_epistemic_governance::lineage::ValidatedEvidenceLineageGraphV1 {
    EvidenceLineageGraphV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        graph_id: LEGACY_GRAPH_LABEL.into(),
        nodes,
    }
    .validate()
    .expect("known-answer graph must validate")
}

#[test]
fn canonical_lineage_profile_digest_v1_known_answer() {
    assert_eq!(
        canonical_evidence_lineage_identity_profile_digest_v1(),
        "blake3:ff23c2ba0796d14eefd10c327542e8f0287bab3d09cb3bb2515c86fd72803d4a"
    );
}

#[test]
fn canonical_two_node_lineage_graph_id_v1_known_answer() {
    let derived = EvidenceLineageNodeV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        evidence_id: B.into(),
        parent_ids: vec![A.into()],
        derivation_kind: CognitiveDerivationKindV1::Inference,
    }
    .validate()
    .expect("known-answer derived node must validate");

    let graph = graph(vec![root(A), derived]);

    assert_eq!(
        canonical_evidence_lineage_graph_id_v1(&graph),
        "blake3:916114cbe9c4ff598097fd2702e965a4b3a763bfcc92e569fb986397371363d6"
    );
}

#[test]
fn independent_evidence_witness_v1_known_answers() {
    let graph = graph(vec![root(A), root(B)]);
    let witness = issue_independent_evidence_set_witness_v1(
        &graph,
        &[A.to_string(), B.to_string()],
    )
    .expect("two disjoint roots must issue an independent witness");

    assert_eq!(
        independent_evidence_set_witness_profile_digest_v1(),
        "blake3:56afd81e7d6db518a69563a185e486a0f3235e9cb5543f456415d3ca01559146"
    );
    assert_eq!(
        witness.lineage_graph_id(),
        "blake3:08b22bdc6eb5775186c3d1f2695fc8b45cd2e7de142178fce0b03f217a31af89"
    );
    assert_eq!(
        witness.witness_id(),
        "blake3:907108bd4dd6283c72085e625e8dfc89ffd9785a795dcee328d651704afe7740"
    );
}
