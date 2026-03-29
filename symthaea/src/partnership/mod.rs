// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Partnership Module - Relational Consciousness & Φ_dyad
//!
//! This module provides a small, focused surface for reasoning about
//! the *relationship* between Symthaea and a human partner, building on
//! the existing `hdc::relational_consciousness` machinery.
//!
//! It is intentionally minimal and does **not** assume any particular
//! UI or transport; higher layers (shell, GUI, API) can depend on this
//! module without pulling in the full experimental surface area.

pub mod partner_model;
pub mod phi_dyad;
pub mod trajectory;

pub use partner_model::{HumanPartnerModel, InteractionEvent};
pub use phi_dyad::{DyadInput, DyadWeights, PhiDyadCalculator, PhiDyadResult};
pub use trajectory::{RelationshipTrajectory, TrajectoryPoint, TrendSummary};
