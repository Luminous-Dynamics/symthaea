// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin et al. consciousness indicators from 6 neuroscience theories.
//!
//! Tests architectural properties against 14 indicators:
//! RPT (Recurrent Processing), GWT (Global Workspace),
//! HOT (Higher-Order), PP (Predictive Processing),
//! AST (Attention Schema), IIT (Integrated Information).
//!
//! When `symthaea-backend` is enabled, the ablation module provides
//! mechanistic ablation tests that prove each indicator is load-bearing.

#[cfg(feature = "symthaea-backend")]
pub mod ablation;
pub mod indicators;
pub mod report;

pub use indicators::ButlinIndicatorSuite;
pub use report::{
    ButlinIndicatorReport, IndicatorEvidence, IndicatorStatus, RuntimeConsciousnessData,
};
