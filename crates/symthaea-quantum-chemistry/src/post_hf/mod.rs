// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Post-Hartree-Fock methods for electron correlation.

pub mod mp2;

pub use mp2::{mp2_correlation_energy, Mp2Result};
