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
pub mod episodic_persistence_envelope;
pub mod evidence_context;
pub mod execution_adapter;
pub mod execution_recovery;
pub mod memory_identity;
pub mod memory_intervention;
pub mod memory_quarantine;
pub mod memory_quarantine_anchored;
mod memory_quarantine_anchored_conversions;
pub mod memory_restore;
pub mod quarantine_intent_ledger;
pub mod quarantine_intent_persistence;
pub mod quarantine_ledger_persistence;
pub mod quarantine_state_ledger;
pub mod restart_activation_gate;
#[path = "replay_recovery_v2.rs"]
pub mod replay_recovery;
