// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing scientific research primitives for Symthaea.
//!
//! This crate is intentionally dependency-light and non-executing. It defines
//! identities, bounded authority, evidence classes, independence descriptors,
//! scientific subjects, claims, polarity-aware adjudication, semantic
//! evidence-to-claim binding, typed provenance graphs, frozen study protocols,
//! immutable execution/conformance records, adversarial falsification campaigns,
//! explicit scientific uncertainty budgets, and provenance-checked replication
//! lineage assessments. It does not run experiments, solvers, LLMs, HDC,
//! statistics, formal provers, or network operations.
//!
//! Ordinary records can express only `None / Declared / Bound` authority.
//! Qualification is intentionally reserved for a later non-forgeable wrapper.

pub mod authority;
pub mod claim;
pub mod claim_adjudication;
pub mod claim_binding;
pub mod evidence;
pub mod execution;
pub mod falsification;
pub mod identity;
pub mod independence;
pub mod protocol;
pub mod provenance;
pub mod replication;
pub mod subject;
pub mod uncertainty;

pub use authority::*;
pub use claim::*;
pub use claim_adjudication::*;
pub use claim_binding::*;
pub use evidence::*;
pub use execution::*;
pub use falsification::*;
pub use identity::*;
pub use independence::*;
pub use protocol::*;
pub use provenance::*;
pub use replication::*;
pub use subject::*;
pub use uncertainty::*;
