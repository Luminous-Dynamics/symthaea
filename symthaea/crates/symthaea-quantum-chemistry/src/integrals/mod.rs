// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Molecular integral engine.
//!
//! Computes the four types of integrals needed for Hartree-Fock:
//! - **Overlap** (S): ⟨φᵢ|φⱼ⟩
//! - **Kinetic** (T): ⟨φᵢ|-½∇²|φⱼ⟩
//! - **Nuclear attraction** (V): ⟨φᵢ|-Z/r|φⱼ⟩
//! - **Electron repulsion** (ERI): (φᵢφⱼ|φₖφₗ)
//!
//! Uses the Obara-Saika recurrence relations with Hermite Gaussian intermediates.
//! Schwarz prescreening is used for ERIs to skip negligible integrals.

pub mod boys;
pub mod eri;
pub mod hermite;
pub mod kinetic;
pub mod nuclear;
pub mod overlap;

pub use boys::boys_function;
pub use eri::compute_eri_tensor;
pub use kinetic::kinetic_integral;
pub use nuclear::nuclear_attraction_integral;
pub use overlap::overlap_integral;
