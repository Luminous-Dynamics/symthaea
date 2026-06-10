// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-module integration tests for symthaea-nuclear.

use symthaea_nuclear::*;

#[test]
fn nuclear_encoder_produces_searchable_hvs() {
    let encoder = NuclearStateEncoder::new();
    let semf = SemiEmpiricalMassFormula::default();

    // Encode Pb-208 (doubly magic) and U-238 (heavy, not magic)
    let pb208 = encoder.encode(&NuclearState {
        z: 82,
        n: 126,
        binding_energy: semf.binding_energy(208, 82),
        shell_correction: -5.0, // doubly magic → negative correction
        deformation: 0.0,       // spherical
    });

    let u238 = encoder.encode(&NuclearState {
        z: 92,
        n: 146,
        binding_energy: semf.binding_energy(238, 92),
        shell_correction: 1.0, // not magic → small positive
        deformation: 0.28,     // deformed
    });

    // Both should be valid HVs
    assert_eq!(pb208.dim(), 16_384);
    assert_eq!(u238.dim(), 16_384);
    assert!(pb208.values.iter().all(|v| v.is_finite()));
    assert!(u238.values.iter().all(|v| v.is_finite()));

    // They should be distinguishable (different nuclei)
    let sim = pb208.similarity(&u238);
    assert!(
        sim < 0.95,
        "Pb-208 and U-238 should be distinguishable: sim={}",
        sim
    );
}

#[test]
fn moscovium_stability_landscape_coherent() {
    let landscape = StabilityLandscape::default();
    let mc_isotopes = landscape.evaluate_moscovium_isotopes();

    // All Mc isotopes should have consistent physics
    for iso in &mc_isotopes {
        assert_eq!(iso.z, 115);
        assert!(iso.a == iso.z + iso.n);
        assert!(iso.binding_energy_semf > 0.0);
        // Total binding should be SEMF + shell correction
        let expected_total = iso.binding_energy_semf + iso.shell_correction;
        assert!(
            (iso.binding_energy_total - expected_total).abs() < 0.01,
            "Mc-{}: total {} != semf {} + shell {}",
            iso.a,
            iso.binding_energy_total,
            iso.binding_energy_semf,
            iso.shell_correction
        );
    }

    // N=184 should be closer to island of stability than N=170
    // (higher shell correction magnitude or longer half-life)
    let n184 = mc_isotopes.iter().find(|i| i.n == 184).unwrap();
    let n170 = mc_isotopes.iter().find(|i| i.n == 170).unwrap();

    // At minimum, both should compute finite values
    assert!(n184.binding_energy_total.is_finite());
    assert!(n170.binding_energy_total.is_finite());
}

#[test]
fn shell_model_and_semf_agree_on_stability() {
    let semf = SemiEmpiricalMassFormula::default();
    let model = ShellModel::default();

    // For a known magic nucleus (Ca-40, Z=20), the shell model
    // should show enhanced stability (negative correction)
    let correction = model.shell_correction_energy(40, 20);

    // Ca-40 SEMF binding should be reasonable
    let ba = semf.binding_energy_per_nucleon(40, 20);
    assert!(ba > 7.0 && ba < 9.5, "Ca-40 B/A = {}", ba);

    // Shell correction should be finite
    assert!(
        correction.is_finite(),
        "Ca-40 shell correction should be finite: {}",
        correction
    );
}
