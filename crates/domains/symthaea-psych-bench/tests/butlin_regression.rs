// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Always-on structural contract gate for the Butlin consciousness indicator
//! suite (issue #7).
//!
//! **What this test can and can't prove — read before changing floors.**
//!
//! This is the "structural contract gate" per `BUTLIN_EVIDENCE_TIER_DESIGN.md`
//! (crate root): cheap, no `symthaea-backend` needed, fails only when the
//! evidence system itself is malformed (missing/duplicate indicators,
//! non-finite scores, wrong indicator count, a static constant drifting
//! below its floor). It never asserts anything about *live* evidence —
//! with `runtime_consciousness: None`, every indicator is necessarily
//! `SupportTier::ArchitecturalOnly` with no `live_score`, which this file
//! confirms rather than fights.
//!
//! **It cannot detect a runtime/behavioral regression** (a mechanism that
//! stops actually firing, a signal going permanently frozen, etc.) — that's
//! the "evidence-integrity gate" (`butlin_ablation_integration.rs`, gated
//! behind `symthaea-backend`, drives a real cognitive loop, real compute
//! cost) — the authoritative causal/functional check, not run on every
//! commit. A legitimate `NotDemonstrated` or `Contradicted` result from that
//! gate is not a bug and must not fail *this* file — negative scientific
//! findings should not break the measurement framework, only claims that
//! deny them (a separate, explicitly opt-in "claim gate", not implemented
//! here, would be the place to assert e.g. "no canonical indicator remains
//! ArchitecturalOnly").
//!
//! This module — not `examples/butlin_validation.rs` (different indicator
//! taxonomy, still using the old status/blend model, not kept in sync) — is
//! the canonical target for this gate.

use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::butlin::report::{
    EvidenceOutcome, REPORT_SCHEMA_VERSION, SupportTier,
};
use symthaea_psych_bench::harness::config::BenchmarkConfig;

/// (indicator id, architectural-score floor). Values are each indicator's
/// actual hand-assigned constant from `indicators.rs`, verified byte-for-byte
/// against the source when this test was written — not independently
/// guessed. A deliberate, reviewed change to a constant should come with a
/// deliberate change here in the same commit; an *undeliberate* one is
/// exactly what this test exists to catch.
const ARCHITECTURAL_FLOORS: &[(&str, f64)] = &[
    ("RPT-1", 1.00),
    ("RPT-2", 0.85),
    ("GWT-1", 0.90),
    ("GWT-2", 1.00),
    ("GWT-3", 0.80),
    ("GWT-4", 0.80),
    ("HOT-1", 0.90),
    ("HOT-2", 0.85),
    ("HOT-3", 0.84),
    ("HOT-4", 0.80),
    ("PP-1", 0.90),
    ("AST-1", 0.85),
    ("AE-1", 0.85),
    ("AE-2", 0.85),
];

fn static_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        runtime_consciousness: None,
        ..Default::default()
    }
}

#[test]
fn butlin_architectural_scores_meet_floors() {
    let report = ButlinIndicatorSuite::evaluate(&static_config());

    assert_eq!(
        report.indicators.len(),
        14,
        "expected all 14 Butlin indicators to be evaluated"
    );
    assert_eq!(
        ARCHITECTURAL_FLOORS.len(),
        14,
        "ARCHITECTURAL_FLOORS must cover all 14 indicators"
    );

    let mut missing = Vec::new();
    let mut below_floor = Vec::new();

    for &(id, floor) in ARCHITECTURAL_FLOORS {
        match report.indicators.iter().find(|i| i.id == id) {
            Some(ind) => {
                let score = ind.architectural_score;
                assert!(
                    score.is_finite(),
                    "{id}: architectural_score is not finite ({score})"
                );
                if score < floor - 1e-9 {
                    below_floor.push(format!("{id}: {score:.4} < floor {floor:.4}"));
                }
            }
            None => missing.push(id),
        }
    }

    assert!(
        missing.is_empty(),
        "indicators missing from the report: {missing:?}"
    );
    assert!(
        below_floor.is_empty(),
        "{} indicator(s) below their architectural floor:\n{}",
        below_floor.len(),
        below_floor.join("\n")
    );
}

#[test]
fn butlin_static_evaluation_never_exceeds_architectural_only() {
    // With no runtime data at all, nothing can claim more than
    // ArchitecturalOnly -- HOT-4's own probe would embed a real
    // responsiveness test if live data were present, but with
    // runtime_consciousness: None there's no live data for it either, so it
    // too stays ArchitecturalOnly here. This test exists to catch a future
    // change that accidentally lets static evaluation claim stronger
    // evidence than a single snapshot can honestly support.
    let report = ButlinIndicatorSuite::evaluate(&static_config());
    for ind in &report.indicators {
        assert_eq!(
            ind.outcome,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            "{}: static evaluation (no runtime data) must be ArchitecturalOnly, got {:?}",
            ind.id,
            ind.outcome
        );
        assert_eq!(
            ind.live_score, None,
            "{}: no runtime data was given, live_score must be None",
            ind.id
        );
    }
}

#[test]
fn butlin_tier_counts_sum_to_14() {
    let report = ButlinIndicatorSuite::evaluate(&static_config());
    let total = report.architectural_only_count
        + report.observed_count
        + report.causally_supported_count
        + report.functionally_supported_count
        + report.not_demonstrated_count
        + report.contradicted_count;
    assert_eq!(total, 14);
}

#[test]
fn butlin_report_schema_version_is_current() {
    let report = ButlinIndicatorSuite::evaluate(&static_config());
    assert_eq!(report.schema_version, REPORT_SCHEMA_VERSION);
}

#[test]
fn butlin_indicator_ids_are_unique_and_match_paper_taxonomy() {
    let report = ButlinIndicatorSuite::evaluate(&static_config());
    let mut ids: Vec<&str> = report.indicators.iter().map(|i| i.id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "duplicate indicator IDs found: {ids:?}"
    );
    ids.sort_unstable();
    let mut expected: Vec<&str> = ARCHITECTURAL_FLOORS.iter().map(|&(id, _)| id).collect();
    expected.sort_unstable();
    assert_eq!(ids, expected);
}
