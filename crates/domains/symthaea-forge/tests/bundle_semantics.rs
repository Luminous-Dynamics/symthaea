// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject};
use symthaea_forge::{
    read_completed_bundle, BundleSemanticError, ForgeAbortRecord, ForgeBundleManifest,
    ForgeBundleOutcome, ForgeTraceEvent, SearchFailure, SearchStats, ABORT_FILE, MANIFEST_FILE,
    OBSERVATIONS_FILE, TRACE_FILE,
};

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "symthaea-forge-semantic-bundle-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn observation(schema: &str, payload: serde_json::Value) -> ObservationObject {
    ObservationObject::new(
        schema,
        ObservationEncoding::Json,
        serde_json::to_vec(&payload).unwrap(),
    )
    .unwrap()
}

fn persist_trace_and_objects(root: &Path, trace: &[ForgeTraceEvent], objects: &[ObservationObject]) {
    fs::write(root.join(TRACE_FILE), serde_json::to_vec_pretty(trace).unwrap()).unwrap();
    fs::write(
        root.join(OBSERVATIONS_FILE),
        serde_json::to_vec_pretty(objects).unwrap(),
    )
    .unwrap();
}

fn seal(root: &Path, outcome: ForgeBundleOutcome) {
    let manifest = ForgeBundleManifest::observe(root, outcome).unwrap();
    fs::write(
        root.join(MANIFEST_FILE),
        manifest.to_json_pretty().unwrap(),
    )
    .unwrap();
}

#[test]
fn completed_no_winner_bundle_roundtrips_semantically() {
    let root = temp_dir("no-winner");
    let summary = observation(
        "symthaea.forge.search-summary.v1",
        json!({
            "candidates_attempted": 0,
            "candidates_no_eligible_mutation": 0,
            "candidates_failed_compile": 0,
            "candidates_failed_test": 0,
            "candidates_failed_benchmark": 0,
            "candidates_passed_correctness": 0,
            "candidates_selected_by_search": 0,
            "baseline_score_bits": null,
            "best_artifact_id": null
        }),
    );
    let trace = vec![ForgeTraceEvent::completed(summary.id().clone())];
    persist_trace_and_objects(&root, &trace, &[summary]);
    seal(&root, ForgeBundleOutcome::NoWinner);

    let verified = read_completed_bundle(&root).unwrap();
    assert_eq!(verified.manifest.outcome, ForgeBundleOutcome::NoWinner);
    assert!(verified.survivor.is_none());
    assert!(verified.abort.is_none());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn aborted_bundle_roundtrips_without_inventing_a_survivor() {
    let root = temp_dir("aborted");
    let failure = SearchFailure {
        stats: SearchStats::default(),
        baseline_benchmark_score: None,
        best: None,
        trace: Vec::new(),
        observations: symthaea_algorithms::observation::ObservationStore::new(),
        phase: "sandbox-init".into(),
        detail: "staging lease unavailable".into(),
    };
    let abort_record = ForgeAbortRecord::from_failure(&failure);
    let abort_observation = observation(
        "symthaea.forge.search-aborted.v1",
        json!({
            "phase": failure.phase,
            "detail": failure.detail,
            "candidates_attempted": 0,
            "candidates_no_eligible_mutation": 0,
            "candidates_failed_compile": 0,
            "candidates_failed_test": 0,
            "candidates_failed_benchmark": 0,
            "candidates_passed_correctness": 0,
            "candidates_selected_by_search": 0,
            "baseline_score_bits": null,
            "best_artifact_id": null
        }),
    );
    let trace = vec![ForgeTraceEvent::aborted(abort_observation.id().clone())];
    persist_trace_and_objects(&root, &trace, &[abort_observation]);
    fs::write(root.join(ABORT_FILE), abort_record.to_json_pretty().unwrap()).unwrap();
    seal(&root, ForgeBundleOutcome::Aborted);

    let verified = read_completed_bundle(&root).unwrap();
    assert_eq!(verified.manifest.outcome, ForgeBundleOutcome::Aborted);
    assert!(verified.survivor.is_none());
    assert!(verified.abort.is_some());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn manifest_outcome_cannot_relabel_completed_trace_as_aborted() {
    let root = temp_dir("terminal-mismatch");
    let summary = observation(
        "symthaea.forge.search-summary.v1",
        json!({
            "candidates_attempted": 0,
            "candidates_no_eligible_mutation": 0,
            "candidates_failed_compile": 0,
            "candidates_failed_test": 0,
            "candidates_failed_benchmark": 0,
            "candidates_passed_correctness": 0,
            "candidates_selected_by_search": 0,
            "baseline_score_bits": null,
            "best_artifact_id": null
        }),
    );
    let trace = vec![ForgeTraceEvent::completed(summary.id().clone())];
    persist_trace_and_objects(&root, &trace, &[summary]);

    let failure = SearchFailure {
        stats: SearchStats::default(),
        baseline_benchmark_score: None,
        best: None,
        trace: Vec::new(),
        observations: symthaea_algorithms::observation::ObservationStore::new(),
        phase: "configuration".into(),
        detail: "synthetic mismatch".into(),
    };
    fs::write(
        root.join(ABORT_FILE),
        ForgeAbortRecord::from_failure(&failure)
            .to_json_pretty()
            .unwrap(),
    )
    .unwrap();
    seal(&root, ForgeBundleOutcome::Aborted);

    assert!(matches!(
        read_completed_bundle(&root),
        Err(BundleSemanticError::OutcomeTerminalMismatch)
    ));
    let _ = fs::remove_dir_all(root);
}
