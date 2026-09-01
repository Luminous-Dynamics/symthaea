// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for CogSec shadow-runtime observation.
//!
//! The existing observer implementation remains unchanged in `lib.rs`. This
//! façade adds progressively stricter wrappers for first-runtime integration:
//! exact live effect pairing and automatic portable effect-binding production.

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::*;

mod effect_pairing;
pub use effect_pairing::{
    EffectBoundAppendError, EffectBoundShadowRuntimeObserver, PendingObservedEffect,
    ShadowObservedMutationDraft,
};

mod portable_effects;
pub use portable_effects::{
    PortableEffectAppendError, PortableEffectObserverInitError,
    PortableEffectBoundShadowRuntimeObserver, PortablePendingObservedEffect,
};
