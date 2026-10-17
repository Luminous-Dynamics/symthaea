// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bounded screening arithmetic for marine freight concepts.
//!
//! This crate does not solve hydrodynamics, naval architecture, structural
//! mechanics, route authorization, or regulatory applicability. It consumes
//! declared inputs and emits separate arithmetic findings. It deliberately
//! has no aggregate pass/fail score and grants no physical-operation authority.

#![deny(unsafe_code)]

mod assessor;
pub mod experiment_selection;
pub mod catamaran_design;
pub mod catamaran_design_space;
pub mod modular_interior;
mod model;

pub use assessor::{assess, AssessmentError, AssessmentReport, Finding, FindingStatus};
pub use model::{
    BatterySystem, DecisionUse, DutyCycle, EvidenceClass, EvidenceValue, MarineFreightCase,
    ProvenanceRecord, RouteEnvelope, SystemBoundary, TerminalInterface, VesselGeometry,
};

/// Frozen maximum claim for this evaluator generation.
pub const CLAIM_CEILING: &str = concat!(
    "Arithmetic screening of declared inputs only; no hydrodynamic optimization, structural ",
    "qualification, seaworthiness, environmental benefit, regulatory compliance, or operating ",
    "authority.",
);
