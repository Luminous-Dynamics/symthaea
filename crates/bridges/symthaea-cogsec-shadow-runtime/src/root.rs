// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for CogSec shadow-runtime observation.
//!
//! The existing observer implementation remains unchanged in `lib.rs`. This
//! façade adds a stricter effect-bound wrapper for first-runtime integration so
//! paired observed mutations cannot bypass their exact evaluation token.

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::*;

mod effect_pairing;
pub use effect_pairing::{
    EffectBoundAppendError, EffectBoundShadowRuntimeObserver, PendingObservedEffect,
    ShadowObservedMutationDraft,
};
