//! Integration tests for FabricationManager consciousness coupling.
//!
//! Verifies end-to-end behavior: event injection → neuromod output → flag signals.

#![cfg(feature = "advanced-manufacturing")]

use symthaea::cognitive_loop::managers::fabrication_manager::{
    FabricationEvent, FabricationEventKind, FabricationManager,
};
use symthaea::cognitive_loop::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use symthaea_fabrication_kernel::ManufacturingSafetyLevel;

fn default_snapshot() -> CycleSnapshot {
    CycleSnapshot::default()
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY → CONSCIOUSNESS INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_anomaly_cascade_across_cycles() {
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
// SAFETY LEVEL TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_escalation_path() {
    let mut mgr = FabricationManager::default();
    assert_eq!(mgr.safety_level(), ManufacturingSafetyLevel::Green);

    // Yellow → no urgency escalation
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Yellow,
        },
        timestamp_secs: 0,
    });
    let out = mgr.process(&default_snapshot());
    assert!(!out.has_flag(output_flags::ESCALATE_URGENCY));
    assert!(!out.has_flag(output_flags::VETO_ACTION));

    // Orange → escalate, no veto
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Orange,
        },
        timestamp_secs: 0,
    });
    let out = mgr.process(&default_snapshot());
    assert!(out.has_flag(output_flags::ESCALATE_URGENCY));
    assert!(!out.has_flag(output_flags::VETO_ACTION));

    // Red → escalate + veto
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
// QUALITY → LEARNING RATE COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_quality_spectrum_lr_modulation() {
    // High quality → LR unchanged (≥1.0)
    let mut mgr_good = FabricationManager::default();
    mgr_good.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "good".into(),
            quality_score: 0.95,
            pog_score: 0.9,
        },
        timestamp_secs: 0,
    });
    let out_good = mgr_good.process(&default_snapshot());
    assert!(
        out_good.lr_modulation >= 1.0 - 1e-10,
        "High quality should not dampen LR: {}",
        out_good.lr_modulation
    );

    // Low quality → LR dampened
    let mut mgr_bad = FabricationManager::default();
    mgr_bad.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "bad".into(),
            quality_score: 0.2,
            pog_score: 0.1,
        },
        timestamp_secs: 0,
    });
    let out_bad = mgr_bad.process(&default_snapshot());
    assert!(
        out_bad.lr_modulation < 1.0,
        "Poor quality should dampen LR: {}",
        out_bad.lr_modulation
    );
    assert!(
        out_bad.exploration_delta > 0.0,
        "Very poor quality should boost exploration"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// REWARD EMA CONVERGENCE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_reward_ema_tracks_quality() {
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

    // 5 failures
    for i in 0..5 {
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::PrintJobCompleted {
                job_id: format!("f{}", i),
                quality_score: 0.2,
                pog_score: 0.1,
            },
            timestamp_secs: 100 + i as u64,
        });
        mgr.process(&default_snapshot());
    }
    let ema_after_failures = mgr.reward_ema();

    assert!(
        ema_after_failures < ema_good,
        "Reward EMA should drop after failures: before={}, after={}",
        ema_good,
        ema_after_failures
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHECKPOINT / RESTORE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_checkpoint_preserves_ema_state() {
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
    let anomaly_ema_before = mgr.anomaly_ema();

    // Checkpoint & restore
    let data = mgr.checkpoint();
    let mut restored = FabricationManager::default();
    restored.restore(&data).unwrap();

    assert!(
        (restored.reward_ema() - ema_before).abs() < 1e-10,
        "Reward EMA should survive checkpoint"
    );
    assert!(
        (restored.anomaly_ema() - anomaly_ema_before).abs() < 1e-6,
        "Anomaly EMA should survive checkpoint"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMOD PATHWAY VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_neuromod_pathways_complete() {
    let mut mgr = FabricationManager::default();

    // Inject events that should trigger all 4 neuromod pathways
    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::CincinnatiAnomaly {
            anomaly_type: "crack".into(),
            severity: 0.8,
            layer: 100,
        },
        timestamp_secs: 0,
    }); // → NE

    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::PrintJobCompleted {
            job_id: "good".into(),
            quality_score: 0.9,
            pog_score: 0.9,
        },
        timestamp_secs: 0,
    }); // → DA + oxytocin

    mgr.inject_event(FabricationEvent {
        kind: FabricationEventKind::SafetyLevelChanged {
            level: ManufacturingSafetyLevel::Red,
        },
        timestamp_secs: 0,
    }); // → NE + 5-HT baseline

    let _output = mgr.process(&default_snapshot());

    let injections = mgr.drain_injections();
    let baselines = mgr.drain_baselines();

    // Check all neurotransmitter pathways fired
    let targets: Vec<&str> = injections.iter().map(|i| i.target).collect();
    assert!(targets.contains(&"norepinephrine"), "NE pathway missing");
    assert!(targets.contains(&"dopamine"), "DA pathway missing");
    assert!(targets.contains(&"oxytocin"), "Oxytocin pathway missing");

    let baseline_targets: Vec<&str> = baselines.iter().map(|b| b.target).collect();
    assert!(
        baseline_targets.contains(&"serotonin"),
        "5-HT baseline pathway missing"
    );
}
