// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for canonical CogSec effect and resource-state commitments.
//!
//! Effect identity and resource-state identity are kept in one dependency-neutral
//! bridge so the trusted evaluation adapter and post-legacy observer can share the
//! exact same canonical representations without moving hashing into the logical
//! reference-monitor core.
//!
//! Raw hashing helpers and the untyped inner effect representation remain
//! crate-private. Public callers construct class-bound effect commitments, typed
//! resource-state commitments, and resource-consistent transition commitments,
//! then explicitly unwrap only at the trusted monitor adapter boundary. This
//! prevents application code from bypassing resource or taxonomy binding merely
//! because the logical monitor uses a generic digest type.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[path = "lib.rs"]
mod effects;
pub use effects::WorkingMemoryItemView;
pub(crate) use effects::{
    CognitiveEffectV1, active_state_digest_v1, continuous_hv_digest_v1, effect_digest_v1,
    metadata_digest_v1,
};

mod state_commitments;
pub use state_commitments::{GoalRecordView, StateCommitmentError};
pub(crate) use state_commitments::{
    affect_state_digest_v1, goal_store_state_digest_v1, working_memory_state_digest_v1,
};

mod typed_commitments;
pub use typed_commitments::{
    CanonicalEffectClassV1, CanonicalResourceV1, CanonicalTransitionCommitmentV1,
    EffectCommitmentV1, ResourceCommitmentMismatch, ResourceStateCommitmentV1,
    TransitionCommitmentMismatch,
};

mod eviction_handoff;
pub use eviction_handoff::{
    EVICTION_HANDOFF_RESOURCE_V1, EvictionHandoffItemCommitmentV1,
    EvictionHandoffItemView, EvictionHandoffStateCommitmentV1,
};
