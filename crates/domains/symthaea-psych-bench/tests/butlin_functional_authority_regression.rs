// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-authority regressions for issue #979.
//!
//! These tests deliberately exercise the always-compiled legacy
//! `ButlinEvidenceBundle -> annotate_with_ablation_results` path, not the
//! newer runtime-qualification conformance model. The legacy bundle has no
//! field proving that its downstream benchmark executed the *same* named
//! intervention whose causal indicator effect it records. Therefore this
//! path may retain downstream measurements as diagnostics, but it must not
//! authorize `FunctionallySupported`.
//!
//! The tests are intended to fail on the pre-#979 implementation and turn
//! green only when the production merge becomes fail-closed.

use symthaea_psych_bench::benchmarks::butlin::{
    annotate_with_ablation_results, AblationResult, ButlinEvidenceBundle,
    ButlinIndicatorReport, EvidenceAnnotation, EvidenceOutcome, IndicatorEvidence,
    SupportTier,
};
use symthaea_psych_bench::benchmarks::butlin::report::REPORT_SCHEMA_VERSION;

fn architectural_indicator(id: &str) -> IndicatorEvidence {
    IndicatorEvidence {
        id: id.to_string(),
        theory: "test theory".to_string(),
        description: "authority-boundary fixture".to_string(),
        outcome: EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        evidence: "fixture only".to_string(),
        architectural_score: 0.85,
        live_score: None,
        probe_quality: None,
        causal_effect: None,
        functional_effect: None,
        annotations: Vec::new(),
    }
}

fn degraded_legacy_result(id: &str) -> AblationResult {
    AblationResult {
        name: format!("disable_{id}"),
        target_indicator: id.to_string(),
        baseline_indicator_score: 0.9,
        ablated_indicator_score: 0.1,
        baseline_benchmark_accuracy: 0.8,
        ablated_benchmark_accuracy: 0.2,
        indicator_dropped: true,
        benchmark_degraded: true,
        contradicted: false,
    }
}

fn legacy_bundle(seeds: Vec<u64>) -> ButlinEvidenceBundle {
    ButlinEvidenceBundle {
        schema_version: REPORT_SCHEMA_VERSION,
        commit_sha: "fixture".to_string(),
        config_hash: "fixture".to_string(),
        seeds,
        generated_at: "fixture".to_string(),
        ablations: vec![degraded_legacy_result("AE-2")],
    }
}

#[test]
fn legacy_bundle_cannot_authorize_functional_support() {
    let report = ButlinIndicatorReport::from_indicators(vec![architectural_indicator("AE-2")]);
    let merged = annotate_with_ablation_results(report, &legacy_bundle(vec![7]))
        .expect("self-consistent fixture should merge");
    let ae2 = &merged.indicators[0];

    assert_eq!(
        ae2.outcome,
        EvidenceOutcome::Supported(SupportTier::CausallySupported),
        "legacy evidence bundles do not carry same-intervention downstream provenance; \
         a degraded downstream proxy may be retained as measurement evidence but must not \
         authorize FunctionallySupported"
    );
    assert_eq!(
        merged.functionally_supported_count, 0,
        "legacy merge must have zero authority to mint FunctionallySupported"
    );
    assert_eq!(merged.causally_supported_count, 1);
    assert!(
        ae2.annotations.iter().any(|annotation| matches!(
            annotation,
            EvidenceAnnotation::ProxyMeasure
                | EvidenceAnnotation::TargetSpecificityNotYetEstablished
                | EvidenceAnnotation::KnownConfound(_)
        )),
        "retained downstream proxy measurement must carry an explicit provenance caveat"
    );
}

#[test]
fn empty_seed_identity_stays_zero_not_fabricated_one() {
    let report = ButlinIndicatorReport::from_indicators(vec![architectural_indicator("AE-2")]);
    let merged = annotate_with_ablation_results(report, &legacy_bundle(Vec::new()))
        .expect("self-consistent fixture should merge");
    let ae2 = &merged.indicators[0];

    assert_eq!(
        ae2.causal_effect
            .expect("causal effect should be retained")
            .seed_count,
        0,
        "an empty seed identity means zero/unknown identified seeds; it must never be silently \
         promoted to seed_count=1"
    );
    assert_eq!(
        ae2.functional_effect
            .expect("downstream measurement should be retained even when authority is capped")
            .seed_count,
        0,
        "functional measurement provenance must preserve the same honest seed cardinality"
    );
}
