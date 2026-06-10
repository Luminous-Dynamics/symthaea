// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extended tests for AUV chemical sensors and hydrodynamics.
//!
//! Tests plume contamination model, WHO anomaly detection, sensor physics,
//! and simulator stability under extreme conditions.

use symthaea_auv::chemical_sensors::*;
use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use symthaea_auv::types::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Chemical sensor anomaly detection — WHO thresholds
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_each_contaminant_detected_individually() {
    let clean = ChemicalReadings::clean_freshwater();

    // pH too low
    let mut acidic = clean;
    acidic.ph = 5.0;
    let anomalies = detect_anomalies(&acidic);
    assert!(!anomalies.is_empty(), "Low pH should be detected");
    assert!(anomalies[0].contains("pH"));

    // pH too high
    let mut alkaline = clean;
    alkaline.ph = 9.5;
    assert!(
        !detect_anomalies(&alkaline).is_empty(),
        "High pH should be detected"
    );

    // Turbidity
    let mut turbid = clean;
    turbid.turbidity_ntu = 10.0;
    assert!(
        !detect_anomalies(&turbid).is_empty(),
        "High turbidity detected"
    );

    // TDS
    let mut tds = clean;
    tds.tds_ppm = 1500.0;
    assert!(!detect_anomalies(&tds).is_empty(), "High TDS detected");

    // Nitrates
    let mut nitrates = clean;
    nitrates.nitrates_mg_l = 60.0;
    assert!(
        !detect_anomalies(&nitrates).is_empty(),
        "High nitrates detected"
    );

    // Arsenic
    let mut arsenic = clean;
    arsenic.arsenic_ug_l = 15.0;
    assert!(
        !detect_anomalies(&arsenic).is_empty(),
        "High arsenic detected"
    );

    // Lead
    let mut lead = clean;
    lead.lead_ug_l = 15.0;
    assert!(!detect_anomalies(&lead).is_empty(), "High lead detected");

    // Chlorine
    let mut chlorine = clean;
    chlorine.chlorine_mg_l = 6.0;
    assert!(
        !detect_anomalies(&chlorine).is_empty(),
        "High chlorine detected"
    );

    // Dissolved oxygen (low)
    let mut low_do = clean;
    low_do.dissolved_oxygen_mg_l = 2.0;
    assert!(!detect_anomalies(&low_do).is_empty(), "Low DO detected");
}

#[test]
fn test_multiple_simultaneous_anomalies() {
    let mut readings = ChemicalReadings::clean_freshwater();
    readings.arsenic_ug_l = 20.0;
    readings.lead_ug_l = 20.0;
    readings.ph = 5.0;

    let anomalies = detect_anomalies(&readings);
    assert!(
        anomalies.len() >= 3,
        "Should detect at least 3 anomalies: {:?}",
        anomalies
    );
}

#[test]
fn test_clean_freshwater_within_who_thresholds() {
    let clean = ChemicalReadings::clean_freshwater();
    assert!(
        detect_anomalies(&clean).is_empty(),
        "Clean freshwater should have no anomalies"
    );
    // Verify the actual clean values are within WHO ranges
    assert!(clean.ph >= 6.5 && clean.ph <= 8.5);
    assert!(clean.turbidity_ntu <= 5.0);
    assert!(clean.tds_ppm <= 1000.0);
    assert!(clean.nitrates_mg_l <= 50.0);
    assert!(clean.arsenic_ug_l <= 10.0);
    assert!(clean.lead_ug_l <= 10.0);
    assert!(clean.chlorine_mg_l <= 5.0);
    assert!(clean.dissolved_oxygen_mg_l >= 4.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Contamination plume model
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_plume_monotonic_falloff() {
    let clean = ChemicalReadings::clean_freshwater();
    let plume = ContaminationPlume {
        center: [0.0, 0.0, 10.0],
        radius: 100.0,
        contaminant: Contaminant::Lead,
        peak_concentration: 50.0,
    };

    // Sample at increasing distances
    let distances = [0.0, 10.0, 25.0, 50.0, 75.0, 99.0];
    let concentrations: Vec<f64> = distances
        .iter()
        .map(|&d| {
            let pos = [d, 0.0, 10.0];
            apply_plume(&clean, &plume, &pos).lead_ug_l
        })
        .collect();

    // Should be monotonically decreasing
    for i in 1..concentrations.len() {
        assert!(
            concentrations[i] <= concentrations[i - 1] + 1e-10,
            "Plume should decrease with distance: d={}→{:.4}, d={}→{:.4}",
            distances[i - 1],
            concentrations[i - 1],
            distances[i],
            concentrations[i]
        );
    }
}

#[test]
fn test_plume_outside_radius_no_effect() {
    let clean = ChemicalReadings::clean_freshwater();
    let plume = ContaminationPlume {
        center: [0.0, 0.0, 0.0],
        radius: 50.0,
        contaminant: Contaminant::Arsenic,
        peak_concentration: 100.0,
    };

    let outside = apply_plume(&clean, &plume, &[100.0, 0.0, 0.0]);
    assert!(
        (outside.arsenic_ug_l - clean.arsenic_ug_l).abs() < 1e-10,
        "Outside radius should have no contamination"
    );
}

#[test]
fn test_plume_at_center_max_concentration() {
    let clean = ChemicalReadings::clean_freshwater();
    let plume = ContaminationPlume {
        center: [0.0, 0.0, 0.0],
        radius: 50.0,
        contaminant: Contaminant::Nitrates,
        peak_concentration: 100.0,
    };

    let at_center = apply_plume(&clean, &plume, &[0.0, 0.0, 0.0]);
    // At center, falloff = exp(0) = 1.0, so full peak added
    assert!(
        (at_center.nitrates_mg_l - (clean.nitrates_mg_l + 100.0)).abs() < 1.0,
        "Center should have peak: {:.1}",
        at_center.nitrates_mg_l
    );
}

#[test]
fn test_all_contaminant_types_apply_correctly() {
    let clean = ChemicalReadings::clean_freshwater();
    let contaminants = [
        Contaminant::Arsenic,
        Contaminant::Lead,
        Contaminant::Nitrates,
        Contaminant::Chlorine,
        Contaminant::HighTurbidity,
        Contaminant::HighTds,
        Contaminant::AcidicPh,
        Contaminant::LowOxygen,
    ];

    for contaminant in &contaminants {
        let plume = ContaminationPlume {
            center: [0.0, 0.0, 0.0],
            radius: 100.0,
            contaminant: *contaminant,
            peak_concentration: 50.0,
        };
        let contaminated = apply_plume(&clean, &plume, &[0.0, 0.0, 0.0]);

        // At least one reading should differ from clean
        let differs = contaminated.ph != clean.ph
            || contaminated.turbidity_ntu != clean.turbidity_ntu
            || contaminated.tds_ppm != clean.tds_ppm
            || contaminated.nitrates_mg_l != clean.nitrates_mg_l
            || contaminated.arsenic_ug_l != clean.arsenic_ug_l
            || contaminated.lead_ug_l != clean.lead_ug_l
            || contaminated.chlorine_mg_l != clean.chlorine_mg_l
            || contaminated.dissolved_oxygen_mg_l != clean.dissolved_oxygen_mg_l;

        assert!(
            differs,
            "Contaminant {:?} should change at least one reading",
            contaminant
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simulator stability
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_auv_simulator_long_horizon() {
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand::zero();

    for i in 0..500 {
        sim.step(&cmd, 0.01);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "AUV state should remain finite at step {i}"
        );
        assert!(
            state.depth >= 0.0,
            "Depth should not go negative at step {i}: {}",
            state.depth
        );
    }
}

#[test]
fn test_auv_simulator_moderate_thrust_stability() {
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand {
        thrusters: [0.5; 8],
    };

    for i in 0..200 {
        sim.step(&cmd, 0.01);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "AUV should survive moderate thrust at step {i}"
        );
    }
}

#[test]
fn test_auv_simulator_full_thrust_stability() {
    // Previously caused NaN at step 35 due to explicit Euler divergence
    // with stiff quadratic drag. Fixed by velocity clamping.
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand {
        thrusters: [1.0; 8], // Full thrust all thrusters
    };

    for i in 0..200 {
        sim.step(&cmd, 0.01);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "AUV should survive full thrust at step {i}"
        );
    }

    // Velocity should be capped at physical limits, not divergent
    let state = sim.state();
    assert!(
        state.speed() <= 6.0,
        "Speed should be bounded: {}",
        state.speed()
    );
}

#[test]
fn test_auv_simulator_alternating_thrust() {
    let mut sim = SimpleAuvSimulator::new();

    // Alternate forward/reverse thrust (vectored propulsion test)
    for i in 0..100 {
        let direction = if i % 20 < 10 { 0.3 } else { -0.3 };
        let cmd = AuvCommand {
            thrusters: [
                direction, -direction, direction, -direction, direction, -direction, direction,
                -direction,
            ],
        };
        sim.step(&cmd, 0.01);
        let state = sim.state();
        assert!(
            state.is_finite(),
            "AUV should handle alternating thrust at step {i}"
        );
    }
}

#[test]
fn test_auv_state_channel_count() {
    assert_eq!(NUM_STATE_CHANNELS, 32, "AUV should have 32 state channels");
    assert_eq!(NUM_KINEMATIC_CHANNELS, 24);
    assert_eq!(NUM_CHEMICAL_CHANNELS, 8);
    assert_eq!(NUM_ACTUATORS, 8);
}
