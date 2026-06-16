// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared test helpers for physics validation suites.

/// Assert that `actual` is within `tolerance` relative error of `expected`.
pub(super) fn assert_relative_eq(actual: f64, expected: f64, tolerance: f64, context: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() < tolerance,
            "{context}: expected ~0, got {actual} (abs tol {tolerance})"
        );
        return;
    }
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= tolerance,
        "{context}: expected {expected}, got {actual} (rel err {rel:.2e}, tol {tolerance:.2e})"
    );
}

/// Shared setup for particle physics tests (Sections 56-63).
pub(super) fn particle_physics_setup() -> (
    crate::genesis::GenesisSeed,
    super::standard_model::StandardModel,
    super::hadrons::Hadrons,
    super::periodic_table::PeriodicTable,
    super::nuclear::NuclearPhysics,
    super::antimatter::Antimatter,
) {
    let genesis = crate::genesis::GenesisSeed::from_phrase("r8_particle_physics");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let table = super::periodic_table::PeriodicTable::from_model(&model, &hadrons, &genesis);
    let nuclear = super::nuclear::NuclearPhysics::from_genesis(&genesis, &hadrons, &table);
    let antimatter = super::antimatter::Antimatter::from_model(&model, &hadrons, &genesis);
    (genesis, model, hadrons, table, nuclear, antimatter)
}
