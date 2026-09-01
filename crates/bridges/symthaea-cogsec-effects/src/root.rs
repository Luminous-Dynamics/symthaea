// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for canonical CogSec effect and resource-state commitments.
//!
//! Effect identity and resource-state identity are kept in one dependency-neutral
//! bridge so the trusted evaluation adapter and post-legacy observer can share the
//! exact same canonical representations without moving hashing into the logical
//! reference-monitor core.
//!
//! Raw hashing helpers remain crate-private. Public callers construct typed effects
//! and typed resource-state commitments, then explicitly unwrap only at the trusted
//! monitor adapter boundary. This prevents application code from bypassing resource
//! binding merely because the logical monitor uses a generic 32-byte digest type.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[path = "lib.rs"]
mod effects;
pub use effects::{CognitiveEffectV1, WorkingMemoryItemView};
pub(crate) use effects::{active_state_digest_v1, effect_digest_v1};

mod state_commitments;
pub use state_commitments::{GoalRecordView, StateCommitmentError};
pub(crate) use state_commitments::{
    affect_state_digest_v1, goal_store_state_digest_v1, graduation_queue_state_digest_v1,
    working_memory_state_digest_v1,
};

mod typed_commitments;
pub use typed_commitments::{
    CanonicalResourceV1, EffectCommitmentV1, ResourceCommitmentMismatch,
    ResourceStateCommitmentV1,
};
