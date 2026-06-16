// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end integration tests for FabricationManager consciousness coupling.
//!
//! Proves the full pipeline: FabricationEvent → FabricationManager.process() →
//! SubsystemOutput (flags, confidence, LR) + neuromod injections → CycleMetadata telemetry.

#![cfg(feature = "advanced-manufacturing")]

use symthaea::cognitive_loop::{
    CognitiveSubsystem, CycleSnapshot, FabricationEvent, FabricationEventKind, FabricationManager,
    output_flags,
};
use symthaea_fabrication_kernel::manufacturing::ManufacturingSafetyLevel;

fn default_snapshot() -> CycleSnapshot {
    CycleSnapshot::default()
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Cincinnati anomaly → NE injection + ANOMALY_DETECTED flag
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn anomaly_triggers_ne_and_flag() {
    let mut mgr = FabricationManager::default();

    // Inject a severe Cincinnati anomaly
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::CincinnatiAnomaly {
            anomaly_type: "layer_delamination".into(),
            severity: 0.85,
            layer: 150,
        },
        timestamp_secs: 1000,
    });

    // Process — this produces SubsystemOutput + queues neuromod
    let output = mgr.process(&default_snapshot());

    // 1. Flag should be set (severity 0.85 > 0.7 threshold)
    assert!(
        output.has_flag(output_flags::ANOMALY_DETECTED),
        "High-severity anomaly should set ANOMALY_DETECTED flag"
    );

    // 2. NE injection should be queued (severity 0.85 > 0.5 threshold)
    let injections = mgr.drain_injections();
    assert!(
        injections.iter().any(|i| i.target == "norepinephrine"),
        "Anomaly should queue NE phasic burst"
    );

    // 3. Telemetry should reflect the anomaly
    let telem = mgr.telemetry();
    assert_eq!(telem.anomaly_count, 1);
    assert!(telem.anomaly_ema > 0.0, "Anomaly EMA should be positive");
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Anomaly cascade across cycles
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn anomaly_cascade_across_cycles() {
    let mut mgr = FabricationManager::default();

    // Cycle 1: mild anomaly — no flag
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::CincinnatiAnomaly {
            anomaly_type: "porosity".into(),
            severity: 0.4,
            layer: 10,
        },
        timestamp_secs: 100,
    });
    let out1 = mgr.process(&default_snapshot());
    assert!(!out1.has_flag(output_flags::ANOMALY_DETECTED));

    // Cycle 2: severe anomaly — flag
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::CincinnatiAnomaly {
            anomaly_type: "delamination".into(),
            severity: 0.9,
            layer: 50,
        },
        timestamp_secs: 200,
    });
    let out2 = mgr.process(&default_snapshot());
    assert!(out2.has_flag(output_flags::ANOMALY_DETECTED));

    // Anomaly EMA should reflect both readings
    assert!(mgr.anomaly_ema() > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Print success → DA + confidence boost + PoGF oxytocin
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn print_success_full_neuromod_cascade() {
    let mut mgr = FabricationManager::default();

    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "job-alpha".into(),
            quality_score: 0.95,
            pog_score: 0.88,
        },
        timestamp_secs: 500,
    });

    let output = mgr.process(&default_snapshot());

    // Confidence should increase (quality > 0.5)
    assert!(
        output.confidence_delta > 0.0,
        "High-quality print should boost confidence: {}",
        output.confidence_delta
    );

    // LR should not be dampened (quality > 0.5)
    assert!(
        output.lr_modulation >= 1.0 - 1e-10,
        "Good quality should not dampen LR: {}",
        output.lr_modulation
    );

    let injections = mgr.drain_injections();

    // DA should fire (quality > 0.5 → reward prediction confirmed)
    assert!(
        injections.iter().any(|i| i.target == "dopamine"),
        "Successful print should trigger DA phasic burst"
    );

    // Oxytocin should fire (pog_score 0.88 > 0.7 threshold)
    assert!(
        injections.iter().any(|i| i.target == "oxytocin"),
        "High PoGF score should trigger oxytocin"
    );

    // Reward EMA should be positive
    assert!(
        mgr.reward_ema() > 0.0,
        "Reward EMA should be positive after success"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Poor quality → LR dampening + exploration boost
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn poor_quality_dampens_lr_boosts_exploration() {
    let mut mgr = FabricationManager::default();

    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "job-bad".into(),
            quality_score: 0.15, // well below both thresholds
            pog_score: 0.1,
        },
        timestamp_secs: 0,
    });

    let output = mgr.process(&default_snapshot());

    assert!(
        output.lr_modulation < 1.0,
        "Poor quality should dampen LR: {}",
        output.lr_modulation
    );
    assert!(
        output.exploration_delta > 0.0,
        "Very poor quality should boost exploration: {}",
        output.exploration_delta
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Safety Red → VETO + ESCALATE + NE + 5-HT dip
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn safety_red_full_emergency_response() {
    let mut mgr = FabricationManager::default();

    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Red,
        },
        timestamp_secs: 0,
    });

    let output = mgr.process(&default_snapshot());

    // Both flags should be set
    assert!(
        output.has_flag(output_flags::VETO_ACTION),
        "Red should VETO"
    );
    assert!(
        output.has_flag(output_flags::ESCALATE_URGENCY),
        "Red should ESCALATE"
    );
    assert!(output.arousal_delta > 0.0, "Red should raise arousal");

    // NE injection for vigilance
    let injections = mgr.drain_injections();
    assert!(injections.iter().any(|i| i.target == "norepinephrine"));

    // 5-HT baseline dip for stress
    let baselines = mgr.drain_baselines();
    assert!(
        baselines
            .iter()
            .any(|b| b.target == "serotonin" && b.nudge < 0.0),
        "Red should dip serotonin baseline"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Safety escalation path (Yellow → Orange → Red)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn safety_escalation_path() {
    let mut mgr = FabricationManager::default();
    assert_eq!(mgr.safety_level(), ManufacturingSafetyLevel::Green);

    // Yellow: no urgency escalation, no veto
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Yellow,
        },
        timestamp_secs: 0,
    });
    let out = mgr.process(&default_snapshot());
    assert!(!out.has_flag(output_flags::ESCALATE_URGENCY));
    assert!(!out.has_flag(output_flags::VETO_ACTION));

    // Orange: escalate but no veto
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Orange,
        },
        timestamp_secs: 0,
    });
    let out = mgr.process(&default_snapshot());
    assert!(out.has_flag(output_flags::ESCALATE_URGENCY));
    assert!(!out.has_flag(output_flags::VETO_ACTION));

    // Red: both
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Red,
        },
        timestamp_secs: 0,
    });
    let out = mgr.process(&default_snapshot());
    assert!(out.has_flag(output_flags::ESCALATE_URGENCY));
    assert!(out.has_flag(output_flags::VETO_ACTION));
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: All 4 neuromod pathways fire in a single cycle
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn all_neuromod_pathways_fire() {
    let mut mgr = FabricationManager::default();

    // Anomaly → NE
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::CincinnatiAnomaly {
            anomaly_type: "crack".into(),
            severity: 0.8,
            layer: 100,
        },
        timestamp_secs: 0,
    });

    // Print success → DA + oxytocin (high PoGF)
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "good".into(),
            quality_score: 0.9,
            pog_score: 0.9,
        },
        timestamp_secs: 0,
    });

    // Emergency → NE + 5-HT baseline dip
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Red,
        },
        timestamp_secs: 0,
    });

    let _output = mgr.process(&default_snapshot());

    let injections = mgr.drain_injections();
    let baselines = mgr.drain_baselines();

    let inj_targets: Vec<&str> = injections.iter().map(|i| i.target).collect();
    assert!(
        inj_targets.contains(&"norepinephrine"),
        "NE pathway missing"
    );
    assert!(inj_targets.contains(&"dopamine"), "DA pathway missing");
    assert!(
        inj_targets.contains(&"oxytocin"),
        "Oxytocin pathway missing"
    );

    let bl_targets: Vec<&str> = baselines.iter().map(|b| b.target).collect();
    assert!(
        bl_targets.contains(&"serotonin"),
        "5-HT baseline pathway missing"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Reward EMA tracks quality over time
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn reward_ema_tracks_quality_trend() {
    let mut mgr = FabricationManager::default();

    // 10 successful prints
    for i in 0..10 {
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::PrintJobCompleted {
                job_id: format!("s{}", i),
                quality_score: 0.9,
                pog_score: 0.8,
            },
            timestamp_secs: i as u64,
        });
        mgr.process(&default_snapshot());
    }
    let ema_good = mgr.reward_ema();
    assert!(ema_good > 0.0, "Reward should be positive after successes");

    // 5 failures
    for i in 0..5 {
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::PrintJobCompleted {
                job_id: format!("f{}", i),
                quality_score: 0.15,
                pog_score: 0.1,
            },
            timestamp_secs: 100 + i as u64,
        });
        mgr.process(&default_snapshot());
    }
    let ema_after = mgr.reward_ema();
    assert!(ema_after < ema_good, "Reward should drop after failures");
}

// ═══════════════════════════════════════════════════════════════════════════════
// E2E: Checkpoint preserves EMA state
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn checkpoint_preserves_ema() {
    let mut mgr = FabricationManager::default();

    // Build up state
    for _ in 0..5 {
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::CincinnatiAnomaly {
                anomaly_type: "test".into(),
                severity: 0.6,
                layer: 1,
            },
            timestamp_secs: 0,
        });
        mgr.process(&default_snapshot());
    }

    let ema_before = mgr.reward_ema();
    let anomaly_before = mgr.anomaly_ema();

    let data = mgr.checkpoint();
    let mut restored = FabricationManager::default();
    restored.restore(&data).unwrap();

    assert!(
        (restored.reward_ema() - ema_before).abs() < 1e-10,
        "Reward EMA should survive checkpoint"
    );
    assert!(
        (restored.anomaly_ema() - anomaly_before).abs() < 1e-6,
        "Anomaly EMA should survive checkpoint"
    );
}