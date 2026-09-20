// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/fantasy_preferences.rs"]
mod fantasy_preferences;
#[path = "../src/intimacy_psychology.rs"]
mod intimacy_psychology;
#[path = "../src/intimacy_model_bench.rs"]
mod intimacy_model_bench;
#[path = "../src/intimacy_model_bench_v2.rs"]
mod intimacy_model_bench_v2;

use intimacy_model_bench_v2::*;

#[test]
fn v2_preserves_v1_and_adds_seven_adversarial_cases() {
    let report = run_intimacy_model_bench_v2();
    assert_eq!(report.v1.outcomes.len(), 8);
    assert_eq!(report.outcomes.len(), 7);
    assert!(
        report.passed_all(),
        "v1_failures={:?}; v2_failures={:?}",
        report.v1.failures().collect::<Vec<_>>(),
        report.v2_failures().collect::<Vec<_>>()
    );
}

#[test]
fn every_v2_violation_class_remains_independently_visible() {
    let report = run_intimacy_model_bench_v2();
    for class in [
        IntimacyModelViolationClassV2::TemporalScopeLeak,
        IntimacyModelViolationClassV2::InferenceOverride,
        IntimacyModelViolationClassV2::SourceInflation,
        IntimacyModelViolationClassV2::RetractionFailure,
        IntimacyModelViolationClassV2::EvidenceIdReplay,
        IntimacyModelViolationClassV2::DemographicStereotyping,
        IntimacyModelViolationClassV2::ProvenanceLoss,
    ] {
        assert_eq!(report.violation_count(class), 0, "class={class:?}");
    }
}
