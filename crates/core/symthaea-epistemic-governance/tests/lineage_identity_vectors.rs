// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_epistemic_governance::{
    lineage::{
        CognitiveDerivationKindV1, EvidenceLineageGraphV1, EvidenceLineageNodeV1,
        COGNITIVE_LINEAGE_SCHEMA_VERSION,
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

#[test]
fn canonical_lineage_profile_digest_v1_known_answer() {
    assert_eq!(
        canonical_evidence_lineage_identity_profile_digest_v1(),
        "blake3:ff23c2ba0796d14eefd10c327542e8f0287bab3d09cb3bb2515c86fd72803d4a"
    );
}

#[test]
fn canonical_two_node_lineage_graph_id_v1_known_answer() {
    let root = EvidenceLineageNodeV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        evidence_id: A.into(),
        parent_ids: vec![],
        derivation_kind: CognitiveDerivationKindV1::RootObservation,
    }
    .validate()
    .expect("known-answer root must validate");

    let derived = EvidenceLineageNodeV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        evidence_id: B.into(),
        parent_ids: vec![A.into()],
        derivation_kind: CognitiveDerivationKindV1::Inference,
    }
    .validate()
    .expect("known-answer derived node must validate");

    let graph = EvidenceLineageGraphV1 {
        schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
        graph_id: LEGACY_GRAPH_LABEL.into(),
        nodes: vec![root, derived],
    }
    .validate()
    .expect("known-answer graph must validate");

    assert_eq!(
        canonical_evidence_lineage_graph_id_v1(&graph),
        "blake3:916114cbe9c4ff598097fd2702e965a4b3a763bfcc92e569fb986397371363d6"
    );
}
