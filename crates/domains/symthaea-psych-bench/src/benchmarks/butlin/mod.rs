// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin et al. (2023) consciousness indicators (arXiv:2308.08708, Table 1).
//!
//! Tests architectural properties against the paper's actual 14 indicators:
//! RPT (Recurrent Processing), GWT (Global Workspace), HOT (Higher-Order),
//! PP (Predictive Processing, one indicator), AST (Attention Schema), AE
//! (Agency and Embodiment). The paper explicitly excludes IIT.
//!
//! When `symthaea-backend` is enabled, the ablation module provides
//! mechanistic ablation tests that prove each indicator is load-bearing, and
//! `ButlinIndicatorSuite::run()` merges that evidence into the report via
//! `report::annotate_with_ablation_results`. See
//! `BUTLIN_EVIDENCE_TIER_DESIGN.md` (crate root) for the evidence-tier model.

#[cfg(feature = "symthaea-backend")]
pub mod ablation;
#[cfg(feature = "symthaea-backend")]
pub mod ae2_empirical_runner;
pub mod indicators;
pub mod qualification_design;
pub mod qualification_runtime;
pub mod report;

pub use indicators::ButlinIndicatorSuite;
pub use qualification_design::{
    Comparison, ControlPurpose, ControlReadiness, DesignViolation, EffectDirection,
    EvidenceDependency, ExpectedEffect, MatchedDimension, PositiveControlId, PositiveControlPlan,
    ProbeValidity, QualificationDesign, ShamControlPlan, SharedGroup, planned_designs,
    shared_groups, validate_designs,
};
pub use qualification_runtime::{
    QualificationFailure, QualificationRunError, RuntimeQualification,
    check_identity_against_registry, resolve_outcome,
};
pub use report::{
    AblationResult, ButlinEvidenceBundle, ButlinIndicatorReport, EffectEstimate,
    EvidenceAnnotation, EvidenceMergeError, EvidenceOutcome, IndicatorEvidence, ProbeQuality,
    RuntimeConsciousnessData, SupportTier, annotate_with_ablation_results,
};
