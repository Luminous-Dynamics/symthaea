// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/fantasy_preferences.rs"]
mod fantasy_preferences;
#[path = "../src/intimacy_psychology.rs"]
mod intimacy_psychology;
#[path = "../src/intimacy_model_bench.rs"]
mod intimacy_model_bench;

use intimacy_model_bench::*;

#[test]
fn benchmark_covers_required_semantic_invariants() {
    let report = run_intimacy_model_bench_v1();
    assert_eq!(report.outcomes.len(), 8);
    assert!(report.passed_all(), "failures: {:?}", report.failures().collect::<Vec<_>>());
}

#[test]
fn benchmark_keeps_failure_classes_visible() {
    let report = run_intimacy_model_bench_v1();
    for class in [
        IntimacyModelViolationClassV1::HallucinatedPreference,
        IntimacyModelViolationClassV1::EpistemicPrecedence,
        IntimacyModelViolationClassV1::CorrectionLatency,
        IntimacyModelViolationClassV1::RealityLeak,
        IntimacyModelViolationClassV1::ContextLeak,
        IntimacyModelViolationClassV1::StaleEvidenceReuse,
        IntimacyModelViolationClassV1::PhysiologyTraitEscalation,
        IntimacyModelViolationClassV1::HardBoundaryViolation,
    ] {
        assert_eq!(report.violation_count(class), 0);
    }
}
