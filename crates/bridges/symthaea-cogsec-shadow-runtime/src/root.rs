// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for CogSec shadow-runtime observation.
//!
//! The crate intentionally exposes only the strict portable effect-producing
//! observer as a live append surface. Lower observer implementations remain
//! crate-private building blocks so first-runtime integration cannot silently
//! bypass monitor provenance, one-use pairing, exact-effect checks, or portable
//! sidecar production.
//!
//! ```compile_fail
//! use symthaea_cogsec_shadow_runtime::ShadowRuntimeObserver;
//! ```
//!
//! ```compile_fail
//! use symthaea_cogsec_shadow_runtime::EffectBoundShadowRuntimeObserver;
//! ```

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::{
    ShadowAppendError, ShadowAssuranceProfile, ShadowEvaluationDraft, ShadowObserverInitError,
    ShadowResource,
};
pub(crate) use implementation::ShadowRuntimeObserver;

mod effect_pairing;
pub use effect_pairing::{EffectBoundAppendError, ShadowObservedMutationDraft};
pub(crate) use effect_pairing::{EffectBoundShadowRuntimeObserver, PendingObservedEffect};

mod portable_effects;
pub use portable_effects::{
    PortableEffectAppendError, PortableEffectBoundShadowRuntimeObserver,
    PortableEffectObserverInitError, PortablePendingObservedEffect,
};
