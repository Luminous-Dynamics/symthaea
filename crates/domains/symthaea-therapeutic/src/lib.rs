// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(dead_code)]
#![deny(clippy::dbg_macro, clippy::todo, clippy::unimplemented)]

//! # Symthaea Therapeutic Psychology
//!
//! Main therapeutic logic crate implementing client modeling, therapeutic alliance
//! tracking (Bordin 1979), affect regulation strategies, clinical case formulation,
//! safety architecture, and narrative integration.
//!
//! ## Architecture
//!
//! - **Client Model**: Tracks psychological state longitudinally (affect, RDoC, symptoms)
//! - **Alliance**: Bordin's working alliance (bond, goals, tasks) with rupture/repair
//! - **Affect Regulation**: Context-aware strategy selection → neuromodulator deltas
//! - **Safety**: Crisis detection, scope boundaries, ethical constraints (Phase 3)
//! - **Formulation**: CBT 4P model, narrative integration
//!
//! ## Safety Guarantee
//!
//! Safety architecture (Phase 3) is built *before* any dialogue generation.
//! The `ScopeGuard` architecturally prevents diagnostic claims and prescriptions.
//!
//! Science: Bordin (1979), Safran & Muran (2000), Gross (2015), Beck (1979),
//! APA Ethics Code (2017).

pub mod affect_regulation;
pub mod alliance;
pub mod client_model;
pub mod dream_integration;
pub mod ethical_constraints;
pub mod formulation;
pub mod narrative_integration;
pub mod safety;
pub mod scope_guard;
pub mod shadow;

// Research directions
pub mod consciousness_protocols;
pub mod digital_twin_psychiatry;
pub mod twin_therapeutic_bridge;

pub use affect_regulation::{RegulationEngine, RegulationStrategy};
pub use alliance::{RuptureType, TherapeuticAlliance};
pub use client_model::{ClientModel, OutcomeSummary};
pub use dream_integration::TherapeuticAction;
pub use ethical_constraints::{EthicalConstraint, EthicalEvaluation};
pub use formulation::CaseFormulation;
pub use narrative_integration::{NarrativeFragment, TherapeuticNarrative};
pub use safety::{CrisisAlert, CrisisDetector, CrisisType, EscalationAction, SafetyPlan};
pub use scope_guard::{ScopeGuard, ScopeViolation};
pub use shadow::{ShadowDetector, ShadowSnapshot, ShadowTelemetry};
