// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Post-Hartree-Fock methods for electron correlation.

pub mod mp2;

pub use mp2::{
    Mp2Result, mp2_correlation_energy, mp2_correlation_energy_frozen_core,
    scs_mp2_correlation_energy, total_frozen_core,
};
