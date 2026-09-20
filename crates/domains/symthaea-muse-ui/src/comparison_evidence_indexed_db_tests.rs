// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Browser-executed qualification scenarios for private comparison evidence.
//!
//! These tests intentionally exercise the public IndexedDB adapter rather than
//! reaching into its private request/transaction bridges. They establish browser
//! behavior for create/read/list/delete, exact replay, conflicting same-key reuse,
//! and concurrent same-key writers. They do not claim blocked-upgrade timing,
//! task-cancellation behavior, cross-browser parity, listener attention, or any
//! network/export authority.

#![cfg(all(test, target_arch = "wasm32"))]

use futures::join;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use crate::comparison_evidence_envelope::ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION;
use crate::comparison_evidence_indexed_db::{
    IndexedDbComparisonEvidenceCreateOutcome, IndexedDbComparisonEvidenceError,
    IndexedDbComparisonEvidenceStore,
};
use crate::comparison_evidence_record::{
    ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION, ArtifactBoundBlindComparisonEvidenceV1,
    BlindAssignmentProvenanceV1, BlindHumanComparisonChoiceV1, ComparisonEvidenceAuthorityV1,
    ComparisonIdentityRelationV1, DurationEvidenceV1, EvidenceSideV1, ExposurePolicyEvidenceV1,
    HumanComparisonJudgmentV1, ListeningExposureEvidenceV1, MusicalAnchorEvidenceV1,
    ResolvedHumanComparisonChoiceV1, RevealedAssignmentEvidenceV1,
};
use crate::comparison_evidence_store::ComparisonEvidenceStoreContractError;
use crate::comparison_evidence_envelope::ArtifactBoundBlindComparisonEnvelopeV1;
use symthaea_muse_protocol::{
    ArtifactIdentity, CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
};

wasm_bindgen_test_configure!(run_in_browser);

const ROUND_TRIP_TRIAL: &str = "10000000000000000000000000000001";
const REPLAY_TRIAL: &str = "10000000000000000000000000000002";
const CONFLICT_TRIAL: &str = "10000000000000000000000000000003";
const RACE_REPLAY_TRIAL: &str = "10000000000000000000000000000004";
const RACE_CONFLICT_TRIAL: &str = "10000000000000000000000000000005";

fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
    ArtifactIdentity {
        score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
        composition: CompositionArtifactId(composition.to_string().repeat(64)),
        rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
    }
}

fn envelope_for(id: &str, note: &str) -> ArtifactBoundBlindComparisonEnvelopeV1 {
    ArtifactBoundBlindComparisonEnvelopeV1 {
        schema_version: ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION,
        trial_instance_id: id.into(),
        evidence: ArtifactBoundBlindComparisonEvidenceV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION,
            trial_id: 303,
            recorded_at_unix_ms: 1_800_000_100_000,
            side_a_identity: identity('a', 'b', 'c'),
            side_b_identity: identity('d', 'e', 'f'),
            revealed_assignment: RevealedAssignmentEvidenceV1 {
                visible_a_side: EvidenceSideV1::B,
                visible_b_side: EvidenceSideV1::A,
            },
            anchor: MusicalAnchorEvidenceV1 {
                bar_index: 3,
                beat_offset: 0.5,
            },
            exposure_policy: ExposurePolicyEvidenceV1 {
                minimum_seconds_per_side: 0.5,
                max_contiguous_step_seconds: 0.6,
            },
            duration_evidence_at_judgment: DurationEvidenceV1 {
                visible_a_seconds: 0.75,
                visible_b_seconds: 0.8,
            },
            judgment: HumanComparisonJudgmentV1 {
                blind_choice: BlindHumanComparisonChoiceV1::PreferVisibleA,
                resolved_choice: ResolvedHumanComparisonChoiceV1::PreferSideB,
                listening_exposure: ListeningExposureEvidenceV1 {
                    visible_a_auditions: 1,
                    visible_b_auditions: 1,
                },
                self_reported_confidence: Some(0.75),
                note: note.into(),
            },
            identity_relation: ComparisonIdentityRelationV1::DifferentComposition,
            assignment_provenance: BlindAssignmentProvenanceV1::Unspecified,
            authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
        },
    }
}

async fn cleanup(
    store: &IndexedDbComparisonEvidenceStore,
    trial_instance_id: &str,
) -> Result<(), IndexedDbComparisonEvidenceError> {
    if let Some(existing) = store.read(trial_instance_id).await? {
        store.delete(&existing).await?;
    }
    Ok(())
}

#[wasm_bindgen_test]
async fn indexeddb_round_trip_list_and_delete_are_validated() {
    let store = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    cleanup(&store, ROUND_TRIP_TRIAL).await.unwrap();

    assert_eq!(
        store
            .create_once(envelope_for(ROUND_TRIP_TRIAL, "round-trip"))
            .await
            .unwrap(),
        IndexedDbComparisonEvidenceCreateOutcome::Created
    );

    let read = store
        .read(ROUND_TRIP_TRIAL)
        .await
        .unwrap()
        .expect("created record must be readable");
    assert_eq!(read.trial_instance_id(), ROUND_TRIP_TRIAL);

    let listed = store.list().await.unwrap();
    assert!(
        listed
            .iter()
            .any(|entry| entry.trial_instance_id() == ROUND_TRIP_TRIAL),
        "list must contain the validated created trial"
    );

    store.delete(&read).await.unwrap();
    assert!(store.read(ROUND_TRIP_TRIAL).await.unwrap().is_none());
}

#[wasm_bindgen_test]
async fn indexeddb_exact_replay_is_idempotent() {
    let store = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    cleanup(&store, REPLAY_TRIAL).await.unwrap();
    let candidate = envelope_for(REPLAY_TRIAL, "same evidence");

    assert_eq!(
        store.create_once(candidate.clone()).await.unwrap(),
        IndexedDbComparisonEvidenceCreateOutcome::Created
    );
    assert_eq!(
        store.create_once(candidate).await.unwrap(),
        IndexedDbComparisonEvidenceCreateOutcome::ExactReplay
    );

    let existing = store.read(REPLAY_TRIAL).await.unwrap().unwrap();
    store.delete(&existing).await.unwrap();
}

#[wasm_bindgen_test]
async fn indexeddb_same_key_changed_evidence_fails_closed() {
    let store = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    cleanup(&store, CONFLICT_TRIAL).await.unwrap();

    assert_eq!(
        store
            .create_once(envelope_for(CONFLICT_TRIAL, "original"))
            .await
            .unwrap(),
        IndexedDbComparisonEvidenceCreateOutcome::Created
    );

    let error = store
        .create_once(envelope_for(CONFLICT_TRIAL, "changed"))
        .await
        .expect_err("same durable key must not be overwritten");
    assert_eq!(
        error,
        IndexedDbComparisonEvidenceError::Contract(
            ComparisonEvidenceStoreContractError::ConflictingTrialReuse {
                trial_instance_id: CONFLICT_TRIAL.into(),
            }
        )
    );

    let existing = store.read(CONFLICT_TRIAL).await.unwrap().unwrap();
    store.delete(&existing).await.unwrap();
}

#[wasm_bindgen_test]
async fn concurrent_identical_writers_produce_one_create_and_one_exact_replay() {
    let left = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    let right = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    cleanup(&left, RACE_REPLAY_TRIAL).await.unwrap();
    let envelope = envelope_for(RACE_REPLAY_TRIAL, "concurrent identical");

    let (left_result, right_result) = join!(
        left.create_once(envelope.clone()),
        right.create_once(envelope)
    );
    let outcomes = [left_result.unwrap(), right_result.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|&&outcome| outcome == IndexedDbComparisonEvidenceCreateOutcome::Created)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|&&outcome| outcome == IndexedDbComparisonEvidenceCreateOutcome::ExactReplay)
            .count(),
        1
    );

    let existing = left.read(RACE_REPLAY_TRIAL).await.unwrap().unwrap();
    left.delete(&existing).await.unwrap();
}

#[wasm_bindgen_test]
async fn concurrent_conflicting_writers_never_overwrite_each_other() {
    let left = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    let right = IndexedDbComparisonEvidenceStore::open().await.unwrap();
    cleanup(&left, RACE_CONFLICT_TRIAL).await.unwrap();

    let (left_result, right_result) = join!(
        left.create_once(envelope_for(RACE_CONFLICT_TRIAL, "left")),
        right.create_once(envelope_for(RACE_CONFLICT_TRIAL, "right"))
    );

    let results = [left_result, right_result];
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Ok(IndexedDbComparisonEvidenceCreateOutcome::Created)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(
                result,
                Err(IndexedDbComparisonEvidenceError::Contract(
                    ComparisonEvidenceStoreContractError::ConflictingTrialReuse { .. }
                ))
            ))
            .count(),
        1
    );

    let existing = left.read(RACE_CONFLICT_TRIAL).await.unwrap().unwrap();
    assert!(
        existing.evidence().record().judgment.note == "left"
            || existing.evidence().record().judgment.note == "right",
        "winner must be exactly one of the submitted immutable records"
    );
    left.delete(&existing).await.unwrap();
}
