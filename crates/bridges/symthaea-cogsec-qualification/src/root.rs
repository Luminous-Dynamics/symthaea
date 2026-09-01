// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for independent CogSec shadow qualification.
//!
//! The existing verifier remains unchanged in `lib.rs`; this façade adds
//! explicit assurance-profile scenario types without letting runtime
//! instrumentation redefine its own expected counts or P0 denominator.

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::*;

mod profiles;
pub use profiles::{
    ObserverOnlyScenario, OwnerAwareScenario, observer_feedback_input_v0,
    observer_goal_no_eviction_v0, observer_goal_with_eviction_v0,
    owner_aware_feedback_input_v0, owner_aware_goal_no_eviction_v0,
    owner_aware_goal_with_eviction_v0,
};
