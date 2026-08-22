// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root-independent Broca translation-plan bridge into SCIP.
//!
//! Semantic data and trusted renderer control are kept separate. The root
//! `symthaea` crate can map `StructuredThought` into [`BrocaTranslationPlan`]
//! without this bridge depending back on the root crate, avoiding a dependency
//! cycle when `LLMOrgan` integration is added later.
//!
//! Prefer [`HardenedBrocaScipAdapter`] at the v1 peer-facing boundary. Richer
//! cognitive state is layered additively through [`BrocaFidelityPlan`] and the
//! [`HardenedFidelityBrocaScipAdapter`], preserving the stable v1 contract while
//! rejecting silent semantic loss by default.

#![forbid(unsafe_code)]

mod bridge;
mod fidelity;
// Security-sensitive export policy defaults stay explicit rather than derived:
// adding a future policy field must force a conscious fail-closed default choice.
#[allow(clippy::derivable_impls)]
mod fidelity_hardened;
mod hardened;
mod plan;

pub use bridge::*;
pub use fidelity::*;
pub use fidelity_hardened::*;
pub use hardened::*;
pub use plan::*;

/// Canonical name for the stable v1 plan adapter. The older exported name is
/// retained for source compatibility with the Phase B draft.
pub type BrocaPlanScipAdapter = StructuredThoughtScipAdapter;

/// Canonical name for the stable v1 peer export policy.
pub type BrocaScipExportPolicy = StructuredThoughtScipPolicy;
