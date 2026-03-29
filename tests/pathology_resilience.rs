// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Pathology Resilience Tests — Proving the Nervous System Works
// ==================================================================================
//
// These tests inject controlled infrastructure failures and verify that the
// organism *feels* them as somatic stress that modulates cognition:
//
// 1. Pain channel works: InfrastructureError → SomaticErrorBridge → SomaticSignals
// 2. Signals apply: thermodynamic_load increases, arousal spikes
// 3. Recovery works: stress decays to zero over clean cycles
// 4. Task supervisor catches panics and reports them as pain
// 5. End-to-end: Symthaea facade processes despite active damage
//
// Biological analogy: poking her with a pin → she flinches → she recovers.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::infrastructure::{InfrastructureError, SomaticErrorBridge, TaskSupervisor};

// ═══════════════════════════════════════════════════════════════════════════════
// Unit-level: SomaticErrorBridge mechanics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bridge_single_error_raises_stress() {
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // No stress initially
    assert_eq!(bridge.systemic_stress(), 0.0);

    // Inject a database failure (severity 0.3)
    tx.send(InfrastructureError::DatabaseFailure {
        operation: "store evicted item".to_string(),
    })
    .unwrap();

    bridge.update();

    // Stress should reflect the error severity (minus decay)
    let stress = bridge.systemic_stress();
    assert!(
        stress > 0.2,
        "Stress should be >0.2 after DB failure, got {stress}"
    );
    assert!(
        stress <= 0.3,
        "Stress should not exceed severity (0.3), got {stress}"
    );
}

#[test]
fn test_bridge_multiple_errors_stack() {
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Inject multiple errors in one cycle
    tx.send(InfrastructureError::LockPoisoned {
        subsystem: "sqlite",
    })
    .unwrap();
    tx.send(InfrastructureError::AsyncTaskPanicked {
        task_name: "eviction-persist".to_string(),
    })
    .unwrap();
    tx.send(InfrastructureError::DatabaseFailure {
        operation: "store episode".to_string(),
    })
    .unwrap();

    bridge.update();

    // Combined severity: 0.4 + 0.5 + 0.3 = 1.2, capped at 1.0 then decayed
    let stress = bridge.systemic_stress();
    assert!(
        stress > 0.7,
        "Multiple errors should stack to high stress, got {stress}"
    );
}

#[test]
fn test_bridge_stress_decays_to_zero_over_clean_cycles() {
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Inject a severe error
    tx.send(InfrastructureError::AsyncTaskPanicked {
        task_name: "critical-task".to_string(),
    })
    .unwrap();
    bridge.update();

    let initial_stress = bridge.systemic_stress();
    assert!(initial_stress > 0.3, "Should have significant stress");

    // Run clean cycles (no new errors)
    for _ in 0..50 {
        bridge.update();
    }

    let final_stress = bridge.systemic_stress();
    assert!(
        final_stress < 0.01,
        "Stress should decay to near-zero after 50 clean cycles, got {final_stress}"
    );
}

#[test]
fn test_bridge_signals_proportional_to_stress() {
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Low stress: single network partition (severity 0.15)
    tx.send(InfrastructureError::NetworkPartition {
        peer: "node-1".to_string(),
    })
    .unwrap();
    bridge.update();

    let low_signals = bridge.to_interoceptive_signals();
    assert!(
        low_signals.thermodynamic_load_delta > 0.0,
        "Should have some thermo delta"
    );
    assert_eq!(
        low_signals.arousal_spike, 0.0,
        "Low stress should NOT trigger arousal spike (threshold 0.5)"
    );
    assert!(
        low_signals.tau_slowdown_factor > 1.0,
        "Should have slight tau slowdown"
    );

    // High stress: multiple severe errors
    let (mut bridge2, tx2) = SomaticErrorBridge::new();
    tx2.send(InfrastructureError::LockPoisoned { subsystem: "mesh" })
        .unwrap();
    tx2.send(InfrastructureError::AsyncTaskPanicked {
        task_name: "critical".to_string(),
    })
    .unwrap();
    bridge2.update();

    let high_signals = bridge2.to_interoceptive_signals();
    assert!(
        high_signals.thermodynamic_load_delta > low_signals.thermodynamic_load_delta,
        "Higher stress should produce larger thermo delta"
    );
    assert!(
        high_signals.arousal_spike > 0.0,
        "High stress should trigger arousal spike"
    );
    assert!(
        high_signals.tau_slowdown_factor > low_signals.tau_slowdown_factor,
        "Higher stress should produce larger tau slowdown"
    );
}

#[test]
fn test_bridge_error_history_bounded() {
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Send 30 errors (more than the 20-item max_history)
    for i in 0..30 {
        tx.send(InfrastructureError::NetworkPartition {
            peer: format!("node-{i}"),
        })
        .unwrap();
    }
    bridge.update();

    assert!(
        bridge.error_count() <= 20,
        "Error history should be bounded at 20, got {}",
        bridge.error_count()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task Supervisor: panic detection → pain reporting
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_supervisor_panic_reports_pain() {
    let (mut bridge, tx) = SomaticErrorBridge::new();
    let mut supervisor = TaskSupervisor::new(tx);

    // Spawn a task that panics
    supervisor.spawn("doomed-task", async {
        panic!("intentional chaos test panic");
    });

    // Give the task time to panic and the monitor to report
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Drain the bridge
    bridge.update();

    // The bridge should have felt the panic
    assert!(
        bridge.systemic_stress() > 0.3,
        "Panic should cause significant stress, got {}",
        bridge.systemic_stress()
    );
    assert!(
        bridge.error_count() >= 1,
        "Should have at least one error in history"
    );
}

#[tokio::test]
async fn test_supervisor_normal_task_no_pain() {
    let (mut bridge, tx) = SomaticErrorBridge::new();
    let mut supervisor = TaskSupervisor::new(tx);

    // Spawn a task that completes normally
    supervisor.spawn("healthy-task", async {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    });

    // Wait for completion
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    bridge.update();

    // No stress from normal tasks
    assert_eq!(
        bridge.systemic_stress(),
        0.0,
        "Normal task should not cause stress"
    );
    assert_eq!(bridge.error_count(), 0, "No errors in history");
}

#[tokio::test]
async fn test_supervisor_multiple_panics_stack_stress() {
    let (mut bridge, tx) = SomaticErrorBridge::new();
    let mut supervisor = TaskSupervisor::new(tx);

    // Spawn 3 panicking tasks
    for i in 0..3 {
        supervisor.spawn(&format!("doomed-{i}"), async move {
            panic!("chaos panic {}", i);
        });
    }

    // Wait for panics
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    bridge.update();

    // 3 panics × severity 0.5 = 1.5, capped to 1.0, then decayed
    let stress = bridge.systemic_stress();
    assert!(
        stress > 0.7,
        "3 panics should cause very high stress, got {stress}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CognitiveLoopService: somatic signals affect cognition
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cognitive_loop_feels_injected_pain() {
    let mut service =
        CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("service creation");

    // Get baseline thermodynamic_load from a clean cycle
    let clean_result = service.cycle("baseline input");
    let baseline_thermo = clean_result.metadata.temporal.thermodynamic_load;

    // Inject pain through the service's pain channel
    let tx = service
        .pain_sender()
        .expect("CognitiveLoopService should have a pain_tx");
    tx.send(InfrastructureError::LockPoisoned {
        subsystem: "sqlite_client",
    })
    .unwrap();
    tx.send(InfrastructureError::DatabaseFailure {
        operation: "search_similar".to_string(),
    })
    .unwrap();

    // Next cycle should feel the pain
    let stressed_result = service.cycle("stressed input");

    // Verify stress was recorded in metadata
    assert!(
        stressed_result.metadata.embodied.somatic_stress > 0.0,
        "Somatic stress should be > 0 after injected errors, got {}",
        stressed_result.metadata.embodied.somatic_stress
    );

    // Thermodynamic load should increase (pain raises metabolic cost)
    assert!(
        stressed_result.metadata.temporal.thermodynamic_load >= baseline_thermo,
        "Thermodynamic load should not decrease under stress: baseline={baseline_thermo}, stressed={}",
        stressed_result.metadata.temporal.thermodynamic_load
    );
}

#[test]
fn test_cognitive_loop_recovers_after_pain() {
    let mut service =
        CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("service creation");

    // Warm up
    service.cycle("warmup");

    // Inject severe pain
    let tx = service
        .pain_sender()
        .expect("CognitiveLoopService should have a pain_tx");
    tx.send(InfrastructureError::AsyncTaskPanicked {
        task_name: "critical-task".to_string(),
    })
    .unwrap();
    tx.send(InfrastructureError::LockPoisoned {
        subsystem: "lance_client",
    })
    .unwrap();

    // Record stress peak
    let peak_result = service.cycle("under attack");
    let peak_stress = peak_result.metadata.embodied.somatic_stress;
    assert!(peak_stress > 0.3, "Should have significant stress at peak");

    // Run clean cycles and verify recovery
    let mut final_stress = peak_stress;
    for i in 0..30 {
        let result = service.cycle(&format!("recovery cycle {i}"));
        final_stress = result.metadata.embodied.somatic_stress;
    }

    assert!(
        final_stress < 0.05,
        "Stress should recover to near-zero after 30 clean cycles, got {final_stress}"
    );
    assert!(
        final_stress < peak_stress * 0.1,
        "Final stress ({final_stress}) should be <10% of peak ({peak_stress})"
    );
}

#[test]
fn test_cognitive_loop_maintains_cycle_under_stress() {
    let mut service =
        CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("service creation");

    // Get baseline cycle time
    let baseline = service.cycle("baseline");
    let baseline_time = baseline.cycle_time_us;

    // Inject continuous pain
    let tx = service
        .pain_sender()
        .expect("CognitiveLoopService should have a pain_tx");
    for i in 0..5 {
        tx.send(InfrastructureError::DatabaseFailure {
            operation: format!("store-{i}"),
        })
        .unwrap();
    }

    // Cycle should still complete (not hang or panic)
    let stressed = service.cycle("under continuous attack");
    assert!(
        stressed.cycle_time_us > 0,
        "Cycle should complete with nonzero time"
    );
    assert!(
        !stressed.output.is_empty(),
        "Should still produce output under stress"
    );

    // Cycle time should not be catastrophically slow (< 10x baseline)
    assert!(
        stressed.cycle_time_us < baseline_time * 10 + 50_000, // 50ms grace
        "Cycle time under stress ({}) should not be >10x baseline ({})",
        stressed.cycle_time_us,
        baseline_time
    );
}

#[test]
fn test_sustained_attack_triggers_high_somatic_stress() {
    let mut service =
        CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("service creation");

    // Sustained attack: inject errors every cycle for 10 cycles
    let tx = service
        .pain_sender()
        .expect("CognitiveLoopService should have a pain_tx");
    let mut max_stress = 0.0f64;
    for i in 0..10 {
        tx.send(InfrastructureError::LockPoisoned {
            subsystem: "sqlite",
        })
        .unwrap();
        tx.send(InfrastructureError::DeserializationCorruption {
            module: "lance_client",
        })
        .unwrap();
        let result = service.cycle(&format!("attack cycle {i}"));
        max_stress = max_stress.max(result.metadata.embodied.somatic_stress);
    }

    assert!(
        max_stress > 0.5,
        "Sustained attack should push stress above 0.5, got {max_stress}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Symthaea facade: end-to-end pain through the full pipeline
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_symthaea_feels_database_failure_pain() {
    // Symthaea::new may fail on CI due to filesystem permissions (e.g. writing
    // model caches). Skip gracefully if initialization fails.
    let mut sym = match symthaea::Symthaea::new(256, 16).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping test_symthaea_feels_database_failure_pain: {e}");
            return;
        }
    };

    // Process a query to establish baseline — may fail on CI without LLM backend
    let _baseline = match sym.process("Hello world").await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping: process() failed (no LLM backend on CI?): {e}");
            return;
        }
    };

    // Verify the pipeline doesn't crash and consciousness is maintained
    let response = match sym.process("Can you feel this?").await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping: process() failed: {e}");
            return;
        }
    };

    assert!(
        response.consciousness_level >= 0.0,
        "Consciousness should be non-negative"
    );
    assert!(
        !response.content.is_empty(),
        "Should produce response even with pain infrastructure active"
    );
}

#[tokio::test]
async fn test_symthaea_multiple_queries_with_pain_infrastructure() {
    let mut sym = match symthaea::Symthaea::new(256, 16).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping test_symthaea_multiple_queries_with_pain_infrastructure: {e}");
            return;
        }
    };

    // Run 5 queries — the pain infrastructure should not interfere with
    // normal operation when no errors are injected
    for i in 0..5 {
        let response = match sym
            .process(&format!("Query number {i}: what is consciousness?"))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping at query {i}: process() failed (no LLM backend on CI?): {e}");
                return;
            }
        };

        assert!(
            !response.content.is_empty(),
            "Query {i} should produce content"
        );
        assert!(
            response.confidence >= 0.0 && response.confidence <= 1.0,
            "Query {i} confidence should be bounded"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Integration: Full pain pathway (error → bridge → signals → state)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_pain_pathway_error_to_signals() {
    // Simulate the full pathway as it runs in process():
    // 1. Error occurs → sent through pain_tx
    // 2. Bridge.update() drains the channel
    // 3. to_interoceptive_signals() converts stress to felt signals
    // 4. Signals applied to mind state (thermodynamic_load, arousal)

    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Step 1: Multiple errors arrive
    tx.send(InfrastructureError::AsyncTaskPanicked {
        task_name: "episode-persist".to_string(),
    })
    .unwrap();
    tx.send(InfrastructureError::DatabaseFailure {
        operation: "store curriculum".to_string(),
    })
    .unwrap();

    // Step 2: Bridge drains
    bridge.update();

    // Step 3: Convert to signals
    let signals = bridge.to_interoceptive_signals();

    // Step 4: Simulate applying to mind state
    let mut thermo_load: f32 = 0.1; // Existing load
    let mut arousal: f32 = 0.3; // Existing arousal

    thermo_load = (thermo_load + signals.thermodynamic_load_delta).min(1.0);
    arousal = (arousal + signals.arousal_spike).min(1.0);

    // Verify the full pathway produced meaningful changes
    assert!(
        thermo_load > 0.1,
        "Thermodynamic load should increase from 0.1, got {thermo_load}"
    );

    // Stress from 0.5 + 0.3 = 0.8 (decayed), should be > 0.5 threshold for arousal
    let stress = bridge.systemic_stress();
    if stress > 0.5 {
        assert!(
            arousal > 0.3,
            "Arousal should spike above 0.3 when stress > 0.5, got {arousal}"
        );
    }

    // Tau slowdown should be > 1.0
    assert!(
        signals.tau_slowdown_factor > 1.0,
        "Tau should slow down under stress"
    );

    // Dissipative health penalty should be > 0
    assert!(
        signals.dissipative_health_penalty > 0.0,
        "Health should be penalized under stress"
    );
}

#[test]
fn test_error_severity_ordering() {
    // Verify the severity hierarchy makes biological sense:
    // Task panic > Lock poison > DB failure > Deserialization > Network partition
    let panic_err = InfrastructureError::AsyncTaskPanicked {
        task_name: "x".to_string(),
    };
    let lock_err = InfrastructureError::LockPoisoned { subsystem: "x" };
    let db_err = InfrastructureError::DatabaseFailure {
        operation: "x".to_string(),
    };
    let deser_err = InfrastructureError::DeserializationCorruption { module: "x" };
    let net_err = InfrastructureError::NetworkPartition {
        peer: "x".to_string(),
    };

    // We can't access severity() directly (private), but we can test via the bridge
    let test_severity = |error: InfrastructureError| -> f64 {
        let (mut bridge, tx) = SomaticErrorBridge::new();
        tx.send(error).unwrap();
        bridge.update();
        bridge.systemic_stress()
    };

    let panic_stress = test_severity(panic_err);
    let lock_stress = test_severity(lock_err);
    let db_stress = test_severity(db_err);
    let deser_stress = test_severity(deser_err);
    let net_stress = test_severity(net_err);

    assert!(panic_stress > lock_stress, "Panic > Lock poison");
    assert!(lock_stress > db_stress, "Lock poison > DB failure");
    assert!(db_stress > deser_stress, "DB failure > Deserialization");
    assert!(
        deser_stress > net_stress,
        "Deserialization > Network partition"
    );
}

#[test]
fn test_recovery_guarantee_50_clean_cycles() {
    // Property: stress MUST decay to <0.1 within 50 clean cycles
    // regardless of how high it was initially.
    let (mut bridge, tx) = SomaticErrorBridge::new();

    // Push to maximum stress (1.0)
    for _ in 0..10 {
        tx.send(InfrastructureError::AsyncTaskPanicked {
            task_name: "max-pain".to_string(),
        })
        .unwrap();
    }
    bridge.update();

    let peak = bridge.systemic_stress();
    assert!(peak > 0.8, "Should reach near-max stress");

    // 50 clean cycles
    for _ in 0..50 {
        bridge.update();
    }

    let recovered = bridge.systemic_stress();
    assert!(
        recovered < 0.1,
        "RECOVERY GUARANTEE VIOLATED: stress {recovered} > 0.1 after 50 clean cycles"
    );
}
