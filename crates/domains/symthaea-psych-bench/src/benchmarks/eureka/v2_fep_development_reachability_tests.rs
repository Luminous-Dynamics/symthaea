// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static reachability checks for the V2 Development-only FEP runner.

#[test]
fn development_runner_has_no_confirmatory_execution_reachability() {
    let whole_source = include_str!("v2_fep_development.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("Development runner has a production section");

    for forbidden in [
        "materialize_canonical_corpora",
        "execute_calibration_selection",
        "FepEvaluationTrial",
        "FepEvaluationSnapshot",
        "score_consequence",
        "V2CalibrationCorpus",
        "V2ComparatorSelectionOutcome",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "Development runner must not reach confirmatory/evaluation authority: {forbidden}"
        );
    }
}

#[test]
fn returned_development_artifact_has_no_trainable_fep_field_type() {
    let whole_source = include_str!("v2_fep_development.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("Development runner has a production section");
    let artifact_start = production_source
        .find("struct V2FepDevelopmentArtifact")
        .expect("Development artifact exists");
    let artifact_tail = &production_source[artifact_start..];
    let artifact_end = artifact_tail
        .find("}\n\nimpl V2FepDevelopmentArtifact")
        .expect("Development artifact body terminates");
    let artifact_body = &artifact_tail[..artifact_end];

    for forbidden in [
        "FepPredictionSession",
        "FepEvaluationSnapshot",
        "ActiveInferenceAgent",
    ] {
        assert!(
            !artifact_body.contains(forbidden),
            "sealed Development artifact must not expose trainable authority: {forbidden}"
        );
    }
    assert!(artifact_body.contains("FepHeldOutSubject"));
    assert!(artifact_body.contains("V2FepTargetContract"));
}
