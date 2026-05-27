// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Epistemic Auditor Integration Test
// ==================================================================================
//
// End-to-end test: create a CognitiveLoopService with the epistemic auditor enabled,
// run cognitive cycles, flush, and verify the audit trail captures real telemetry.
//
// Feature-gated: requires `epistemic_auditor`.
// ==================================================================================

#![cfg(feature = "epistemic_auditor")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const CYCLE_COUNT: usize = 50;

fn make_audited_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    // In-memory DuckDB (no file path = ":memory:")
    config.epistemic_auditor_db_path = Some(":memory:".to_string());
    CognitiveLoopService::new(config).expect("service with auditor")
}

#[test]
fn test_auditor_captures_cycles() {
    let mut svc = make_audited_service();
    assert!(svc.has_epistemic_auditor(), "auditor should be active");
    assert_eq!(svc.audit_total_records(), 0);

    // Run cycles with varied input
    let inputs = [
        "the sun rises in the east",
        "water flows downhill",
        "consciousness emerges from integration",
        "prediction error drives learning",
        "moral reasoning requires empathy",
    ];

    for i in 0..CYCLE_COUNT {
        let input = inputs[i % inputs.len()];
        let _result = svc.cycle(input);
    }

    // Records should be buffered (not yet flushed — cadence is 1009)
    assert!(
        svc.audit_total_records() > 0,
        "should have buffered records after {} cycles",
        CYCLE_COUNT
    );

    // Force flush
    svc.flush_audit();

    // Verify flushed count
    let flushed = svc.audit_total_flushed();
    assert!(
        flushed >= CYCLE_COUNT as u64,
        "expected >= {CYCLE_COUNT} flushed, got {flushed}"
    );

    // Verify total_cycles_audited matches
    let total = svc.audit_total_cycles();
    assert_eq!(total, flushed, "total_cycles should match flushed count");
}

#[test]
fn test_auditor_phi_statistics() {
    let mut svc = make_audited_service();

    for i in 0..CYCLE_COUNT {
        let input = if i % 2 == 0 {
            "cause leads to effect"
        } else {
            "randomness and noise"
        };
        let _result = svc.cycle(input);
    }
    svc.flush_audit();

    let stats = svc
        .phi_statistics(0, CYCLE_COUNT as u64 + 1)
        .expect("phi_statistics should succeed");

    assert_eq!(stats.count, CYCLE_COUNT as u64);
    assert!(stats.mean.is_finite(), "phi mean should be finite");
    assert!(stats.min <= stats.mean, "min <= mean");
    assert!(stats.mean <= stats.max, "mean <= max");
    assert!(stats.mean_consciousness.is_finite());
}

#[test]
fn test_auditor_summary_all() {
    let mut svc = make_audited_service();

    for _ in 0..CYCLE_COUNT {
        let _result = svc.cycle("testing audit summary");
    }
    svc.flush_audit();

    let summary = svc.audit_summary_all().expect("audit_summary_all");
    assert_eq!(summary.total_cycles, CYCLE_COUNT as u64);
    assert!(summary.phi.mean.is_finite());
    assert!(summary.total_energy >= 0.0);
}

#[test]
fn test_auditor_report_all() {
    let mut svc = make_audited_service();

    for _ in 0..CYCLE_COUNT {
        let _result = svc.cycle("report generation test");
    }
    svc.flush_audit();

    let report = svc.generate_audit_report_all().expect("report");
    assert!(report.contains("EPISTEMIC AUDIT REPORT"));
    assert!(report.contains("CONSCIOUSNESS"));
    assert!(report.contains("MEMORY GRADUATION"));
    assert!(report.contains("MORAL INTEGRITY"));
    assert!(report.contains("NEUROMODULATOR BALANCE"));
    assert!(report.contains("ENERGY & PERFORMANCE"));
}