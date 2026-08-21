// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root-independent Broca translation-plan bridge into SCIP.
//!
//! Semantic data and trusted renderer control are kept separate. The root
//! `symthaea` crate can map `StructuredThought` into [`BrocaTranslationPlan`]
//! without this bridge depending back on the root crate, avoiding a dependency
//! cycle when `LLMOrgan` integration is added later.

#![forbid(unsafe_code)]

mod bridge;
mod plan;

pub use bridge::*;
pub use plan::*;
