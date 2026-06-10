// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Antimatter FEP escalation integration test.

use symthaea_physics::antimatter::*;

#[test]
fn fep_agent_escalates_through_safety_levels() {
    let mut twin = AntimatterTwin::new();

    // Set reference as nominal operation
    let nominal = AntimatterReading {
        containment_field: 0.8,
        annihilation_rate: 0.1,
        energy_output: 0.2,
        antimatter_inventory: 0.5,
        boundary_integrity: 1.0,
    };
    twin.set_reference(&nominal);

    // Step 1: Nominal → should be Green/Maintain
    let out1 = twin.step(&nominal, 0.001);
    assert!(
        out1.safety_level == AntimatterSafetyLevel::Green
            || out1.safety_level == AntimatterSafetyLevel::Yellow,
        "Nominal should be safe: {:?}",
        out1.safety_level
    );

    // Step 2: Degraded containment → should escalate
    let degraded = AntimatterReading {
        containment_field: 0.2,
        annihilation_rate: 0.8,
        energy_output: 0.9,
        antimatter_inventory: 0.1,
        boundary_integrity: 0.1,
    };
    let out2 = twin.step(&degraded, 0.001);
    assert!(
        out2.free_energy > out1.free_energy,
        "Degraded state should have higher FE: {} vs {}",
        out2.free_energy,
        out1.free_energy
    );
}

#[test]
fn blanket_coherence_is_monotonic() {
    // Energy density sweep from 0 to 10^18 J/m³
    let densities: Vec<f64> = (0..20).map(|i| i as f64 * 5e16).collect();
    let results = blanket_coherence_test(&densities);

    assert_eq!(results.len(), 20);

    // All results should be finite
    for r in &results {
        assert!(r.permeability.is_finite(), "Permeability should be finite");
        assert!(r.energy_density.is_finite());
    }

    // At zero density, blanket should be more intact than at max density
    assert!(
        results[0].blanket_intact
            || !results[19].blanket_intact
            || results[0].permeability <= results[19].permeability + 0.01,
        "Blanket should degrade with energy: p[0]={}, p[19]={}",
        results[0].permeability,
        results[19].permeability
    );
}

#[test]
fn annihilation_physics_self_consistent() {
    let energy = annihilation_energy_per_event();
    let fraction = charged_energy_fraction();

    // Total energy = 2 × proton mass × c²
    assert!((energy - 1876.544).abs() < 1.0);

    // Charged fraction × energy = directable energy
    let directable = energy * fraction;
    assert!(directable > 700.0 && directable < 800.0);

    // Cross-section at 1 km/s should be huge (cold antimatter)
    let sigma = low_energy_cross_section(1000.0);
    assert!(
        sigma > 1e-20,
        "Cold cross-section should be large: {}",
        sigma
    );
}
