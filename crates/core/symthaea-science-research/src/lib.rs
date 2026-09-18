// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing scientific research primitives for Symthaea.
//!
//! This crate is intentionally dependency-light and non-executing. It defines
//! identities, bounded authority, evidence classes, independence descriptors,
//! scientific subjects, claims, typed provenance graphs, frozen study protocols,
//! immutable execution/conformance records, adversarial falsification campaigns,
//! and explicit scientific uncertainty budgets. It does not run experiments,
//! solvers, LLMs, HDC, statistics, formal provers, or network operations.
//!
//! Ordinary records can express only `None / Declared / Bound` authority.
//! Qualification is intentionally reserved for a later non-forgeable wrapper.

pub mod authority;
pub mod claim;
pub mod evidence;
pub mod execution;
pub mod falsification;
pub mod identity;
pub mod independence;
pub mod protocol;
pub mod provenance;
pub mod subject;
pub mod uncertainty;

pub use authority::*;
pub use claim::*;
pub use evidence::*;
pub use execution::*;
pub use falsification::*;
pub use identity::*;
pub use independence::*;
pub use protocol::*;
pub use provenance::*;
pub use subject::*;
pub use uncertainty::*;
