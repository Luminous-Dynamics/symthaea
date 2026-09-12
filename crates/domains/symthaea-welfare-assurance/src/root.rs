// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime welfare-assurance root.
//!
//! The established permit implementation remains isolated in `entry.rs`; this root layers the
//! moral-patient evidence-context commitment on top without making the lower-level interlock
//! reinterpret consciousness evidence itself.

#![deny(unsafe_code)]

#[path = "entry.rs"]
mod permit;

pub use permit::*;

pub mod authority_evidence_binding;
pub mod evidence_context;
pub mod execution_adapter;
pub mod execution_recovery;
pub mod memory_identity;
pub mod memory_intervention;
pub mod memory_quarantine;
#[path = "replay_recovery_v2.rs"]
pub mod replay_recovery;
