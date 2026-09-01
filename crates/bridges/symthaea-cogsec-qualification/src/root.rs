// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for independent CogSec shadow qualification.
//!
//! The existing verifier remains unchanged in `lib.rs`; this façade adds
//! explicit assurance-profile scenarios, exact-effect attribution, and
//! checkpoint-integrity composition without collapsing those assurance
//! dimensions into one generic score or authority claim.

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

mod effect_qualification;
pub use effect_qualification::{
    EffectBoundScenarioQualificationReport, qualify_observer_effect_bound_scenario,
    qualify_owner_aware_effect_bound_scenario,
};

mod integrity_qualification;
pub use integrity_qualification::{
    IntegrityBoundScenarioQualificationReport, qualify_checkpointed_observer_scenario,
    qualify_checkpointed_owner_aware_scenario,
};
