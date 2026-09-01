// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for CogSec shadow evidence.
//!
//! The existing v1 event ledger remains unchanged in `lib.rs`. This façade adds
//! an additive portable effect-binding sidecar so independent qualification can
//! re-check exact evaluated-effect ↔ observed-effect equality after export.

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::*;

mod effect_binding;
pub use effect_binding::{
    EFFECT_BINDING_SCHEMA_V1, EffectBindingReport, EffectBindingViolation,
    EffectBoundEvidenceSnapshot, ObservedEffectBinding, validate_effect_bound_snapshot,
};
