// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bounded screening arithmetic for marine freight concepts.
//!
//! This crate does not solve hydrodynamics, naval architecture, structural
//! mechanics, route authorization, or regulatory applicability. It consumes
//! declared inputs and emits separate arithmetic findings. It deliberately
//! has no aggregate pass/fail score and grants no physical-operation authority.

mod assessor;
mod model;

pub use assessor::{assess, AssessmentError, AssessmentReport, Finding, FindingStatus};
pub use model::{
    DecisionUse, EvidenceClass, EvidenceValue, MarineFreightCase, ProvenanceRecord,
};

/// Frozen maximum claim for this evaluator generation.
pub const CLAIM_CEILING: &str = "Arithmetic screening of declared inputs only; no hydrodynamic optimization, structural qualification, seaworthiness, environmental benefit, regulatory compliance, or operating authority.";
