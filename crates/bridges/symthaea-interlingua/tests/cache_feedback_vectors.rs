// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Golden SCIP cache-feedback vectors for independent implementations.

use symthaea_interlingua::{
    SemanticCacheAck, SemanticCacheFeedback, SemanticCacheMiss, SemanticCacheMissKind,
    SemanticCacheRevoke,
};

const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn cache_feedback_ack_golden_json_is_stable() {
    let feedback = SemanticCacheFeedback::Ack(SemanticCacheAck::new(HASH).unwrap());
    assert_eq!(
        feedback.canonical_bytes().unwrap(),
        br#"{"kind":"ack","body":{"semantic_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}"#
    );
}

#[test]
fn cache_feedback_reference_miss_golden_json_is_stable() {
    let feedback = SemanticCacheFeedback::Miss(
        SemanticCacheMiss::new(HASH, SemanticCacheMissKind::SemanticReferenceTarget).unwrap(),
    );
    assert_eq!(
        feedback.canonical_bytes().unwrap(),
        br#"{"kind":"miss","body":{"semantic_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","requirement":"semantic_reference_target"}}"#
    );
}

#[test]
fn cache_feedback_delta_base_miss_golden_json_is_stable() {
    let feedback = SemanticCacheFeedback::Miss(
        SemanticCacheMiss::new(HASH, SemanticCacheMissKind::GraphDeltaBase).unwrap(),
    );
    assert_eq!(
        feedback.canonical_bytes().unwrap(),
        br#"{"kind":"miss","body":{"semantic_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","requirement":"graph_delta_base"}}"#
    );
}

#[test]
fn cache_feedback_revoke_golden_json_is_stable() {
    let feedback = SemanticCacheFeedback::Revoke(SemanticCacheRevoke::new(HASH).unwrap());
    assert_eq!(
        feedback.canonical_bytes().unwrap(),
        br#"{"kind":"revoke","body":{"semantic_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}"#
    );
}
