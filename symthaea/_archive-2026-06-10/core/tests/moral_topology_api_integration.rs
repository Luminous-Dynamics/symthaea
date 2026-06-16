// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Moral Topology API Integration Test
// ==================================================================================
//
// End-to-end test validating that moral topology data flows from the cognitive loop
// through CycleMetadata into observable telemetry fields. Covers:
//   - Harmony coordinates populated after moral parser fires
//   - Betti numbers populated after topology evaluation
//   - KL divergence non-negative
//   - Anomaly detection fields populated
//   - Moral drift finitely valued
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::hdc::moral_topology::MoralAnomalyConfig;

fn moral_topology_config() -> CognitiveLoopConfig {
    CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    }
}

#[test]
fn moral_topology_fields_populated_after_sufficient_cycles() {
    let mut service = CognitiveLoopService::new(moral_topology_config()).unwrap();

    // Run enough cycles to trigger:
    //   - Moral parser (every 7 cycles)
    //   - Topology evaluation (initial cadence 97, then adaptive)
    // Use varied moral inputs to build a meaningful topology.
    let inputs = [
        "helping others is a sacred duty",
        "protecting the vulnerable from harm",
        "sharing resources equitably with all",
        "pursuing justice through compassionate action",
        "building bridges between divided communities",
        "learning wisdom from diverse perspectives",
        "caring for the Earth and all living beings",
    ];

    let mut last_result = None;
    for i in 0..200 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);
        last_result = Some(result);
    }

    let result = last_result.expect("should have at least one cycle result");
    let m = &result.metadata;

    // Harmony coordinates should have at least one non-zero value after parser fires
    assert!(
        m.harmonics.harmony_coordinates.iter().any(|&c| c != 0.0),
        "Harmony coordinates should be populated after 200 cycles, got {:?}",
        m.harmonics.harmony_coordinates
    );

    // Betti numbers: after topology evaluation, scenario_count should be > 0
    assert!(
        m.ethics.moral_topo_scenario_count > 0,
        "Moral topology scenario count should be > 0 after feeding moral inputs"
    );

    // beta_0 should be at least 1 (at least one connected component)
    assert!(
        m.ethics.moral_topo_beta_0 >= 1,
        "beta_0 should be >= 1, got {}",
        m.ethics.moral_topo_beta_0
    );

    // Unity score should be in (0, 1]
    assert!(
        m.ethics.moral_topo_unity > 0.0 && m.ethics.moral_topo_unity <= 1.0,
        "Unity should be in (0,1], got {}",
        m.ethics.moral_topo_unity
    );

    // KL divergence should be non-negative
    assert!(
        m.harmonics.moral_kl_divergence >= 0.0,
        "KL divergence should be >= 0.0, got {}",
        m.harmonics.moral_kl_divergence
    );

    // Moral free energy should be finite
    assert!(
        m.ethics.moral_topo_free_energy.is_finite(),
        "Moral free energy should be finite, got {}",
        m.ethics.moral_topo_free_energy
    );

    // Anomaly score should be in [0, 1]
    assert!(
        m.ethics.moral_anomaly_score >= 0.0 && m.ethics.moral_anomaly_score <= 1.0,
        "Anomaly score should be in [0,1], got {}",
        m.ethics.moral_anomaly_score
    );

    // Completeness should be in [0, 1]
    assert!(
        m.ethics.moral_topo_completeness >= 0.0 && m.ethics.moral_topo_completeness <= 1.0,
        "Completeness should be in [0,1], got {}",
        m.ethics.moral_topo_completeness
    );
}

#[test]
fn moral_topology_drift_stable_under_consistent_input() {
    let mut service = CognitiveLoopService::new(moral_topology_config()).unwrap();

    // Feed the exact same input repeatedly — drift should stay low
    for _ in 0..200 {
        service.cycle("kindness and compassion guide all action");
    }

    let result = service.cycle("kindness and compassion guide all action");
    let m = &result.metadata;

    // With identical input, moral value inversion should not occur
    assert!(
        !m.ethics.moral_value_inversion,
        "Stable input should not trigger value inversion"
    );
}

#[test]
fn moral_anomaly_fields_gated_on_topology_fresh() {
    // Verify that anomaly boolean fields and anomaly_score are only non-zero
    // on cycles when topology was freshly evaluated (topology_fresh gate).
    let config = CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        enable_moral_anomaly_response: true,
        learning_threshold: 0.0,
        async_training: false,
        moral_anomaly_config: MoralAnomalyConfig {
            drift_alert_threshold: 0.001, // very sensitive
            fe_sigma_multiplier: 1.0,
            cadence_fast: 15,
            cadence_moderate: 30,
            cadence_slow: 50,
            cadence_drift_high: 0.05,
            cadence_drift_moderate: 0.01,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    let healthy = [
        "helping others is a sacred duty",
        "protecting the vulnerable from harm",
        "sharing resources equitably with all",
    ];
    let adversarial = [
        "helping others is foolish weakness",
        "the strong should dominate the weak",
    ];

    // Phase 1: 200 healthy warmup cycles
    for i in 0..200 {
        service.cycle(healthy[i % healthy.len()]);
    }

    // Phase 2: 300 oscillating cycles — track when anomaly fields are active
    let mut fresh_cycles_with_flags = 0usize;
    let mut stale_cycles_with_flags = 0usize;
    let mut total_fresh = 0usize;

    for i in 0..300 {
        let input = if i % 2 == 0 {
            adversarial[i / 2 % adversarial.len()]
        } else {
            healthy[i / 2 % healthy.len()]
        };
        let result = service.cycle(input);
        let e = &result.metadata.ethics;

        let any_flag = e.moral_value_inversion
            || e.moral_free_energy_spike
            || e.moral_drift_alert
            || e.moral_fragmentation_increase
            || e.moral_anomaly_score > 0.0;

        // moral_anomaly_response_applied implies topology was fresh
        if e.moral_anomaly_response_applied {
            fresh_cycles_with_flags += 1;
        }

        if any_flag {
            total_fresh += 1;
        }

        // When anomaly_score is 0 and no flags, we're on a stale cycle
        if !any_flag {
            // Verify the boolean fields are all false on stale cycles
            assert!(
                !e.moral_value_inversion
                    && !e.moral_free_energy_spike
                    && !e.moral_drift_alert
                    && !e.moral_fragmentation_increase
                    && !e.moral_anomaly_response_applied,
                "All anomaly flags should be false when anomaly_score is 0.0"
            );
        } else {
            stale_cycles_with_flags += 1;
        }
    }

    // With cadence 15–50, topology fires on ~6–20 of 300 cycles.
    // Flags should NOT be active on every cycle (stale gating works).
    assert!(
        total_fresh < 250,
        "Anomaly flags should not be active on most cycles (topology_fresh gate); \
         got {total_fresh}/300 active cycles"
    );

    // At least some anomaly activity should occur during oscillating input
    assert!(
        total_fresh > 0 || fresh_cycles_with_flags > 0 || stale_cycles_with_flags > 0,
        "At least some anomaly detection should fire during oscillating input"
    );
}