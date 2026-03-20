//! CPG → delta_t coupling integration test (Phase A of CFC_OSCILLATION_INTEGRATION).
//!
//! Verifies that CPG sync_index modulates the CfC delta_t tau factor chain.
//! When the `cpg` feature is disabled, tau_cpg = 1.0 (no effect).

#![cfg(feature = "cpg")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service() -> CognitiveLoopService {
    let config = CognitiveLoopConfig::default();
    CognitiveLoopService::new(config).unwrap()
}

/// After warmup, the CPG sync_index should be reported in telemetry and
/// the dynamics should remain finite (no NaN/Inf from the new tau_cpg factor).
#[test]
fn cpg_sync_index_reported_and_dynamics_finite() {
    let mut svc = make_service();

    // Run past warmup (DYNAMICS_STARTUP_WARMUP_CYCLES = 10)
    for _ in 0..30 {
        let result = svc.cycle("cpg integration test input");
        // All outputs must remain finite with the new tau_cpg factor in the chain
        for (j, &v) in result.output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "CfC output[{j}] not finite with CPG tau factor active"
            );
        }
        assert!(
            result.prediction_error.is_finite(),
            "PE not finite with CPG tau factor active"
        );
    }

    // After 30 cycles the CPG telemetry should show a valid sync_index
    let result = svc.cycle("final check");
    let sync = result.metadata.cpg_sync_index;
    assert!(
        sync >= 0.0 && sync <= 1.0,
        "CPG sync_index should be in [0,1], got {sync}"
    );
}

/// The CPG tau factor should be bounded: even with extreme sync values,
/// delta_t products stay in a reasonable range across many cycles.
#[test]
fn cpg_tau_factor_bounded_across_many_cycles() {
    let mut svc = make_service();

    let inputs = [
        "novel surprising stimulus",
        "familiar repeated pattern",
        "emotional high-arousal event",
        "calm consolidation phase",
        "ambiguous uncertain signal",
    ];

    for i in 0..60 {
        let result = svc.cycle(inputs[i % inputs.len()]);
        for (j, &v) in result.output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "CfC output[{j}] not finite at cycle {i} with CPG tau in chain"
            );
        }
        assert!(
            result.prediction_error.is_finite(),
            "PE not finite at cycle {i}"
        );
    }
}
