// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for CogSec shadow evidence.
//!
//! The existing v1 event ledger remains unchanged in `lib.rs`. This façade adds
//! additive exact-effect bindings and deterministic hash-chain checkpoints so
//! structural qualification, integrity continuity, and later authentication can
//! remain separate assurance layers.

#![forbid(unsafe_code)]

#[path = "lib.rs"]
mod implementation;
pub use implementation::*;

mod effect_binding;
pub use effect_binding::{
    EFFECT_BINDING_SCHEMA_V1, EffectBindingReport, EffectBindingViolation,
    EffectBoundEvidenceSnapshot, ObservedEffectBinding, validate_effect_bound_snapshot,
};

mod checkpoint;
pub use checkpoint::{
    EVIDENCE_CHECKPOINT_SCHEMA_V1, CheckpointBuildError, CheckpointFork,
    CheckpointVerificationReport, CheckpointViolation, CheckpointedEffectBoundEvidence,
    EvidenceCheckpoint, checkpoint_effect_bound_snapshot, effect_bound_snapshot_root,
    verify_checkpoint_chain,
};

mod fork_semantics;
pub use fork_semantics::detect_checkpoint_forks;
