// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing scientific research primitives for Symthaea.
//!
//! This crate is intentionally dependency-light and non-executing. It defines
//! identities, authority ceilings, evidence classes, independence descriptors,
//! scientific subjects, and claim aggregation. It does not run experiments,
//! solvers, LLMs, HDC, statistics, formal provers, or network operations.
//!
//! The core invariant is:
//!
//! ```text
//! generation != evidence != execution != qualification != replication
//! ```
//!
//! Domain crates may impose stronger rules. Generic adapters may preserve or
//! lower authority, but must not manufacture stronger authority.

pub mod authority;
pub mod claim;
pub mod evidence;
pub mod identity;
pub mod independence;
pub mod subject;

pub use authority::*;
pub use claim::*;
pub use evidence::*;
pub use identity::*;
pub use independence::*;
pub use subject::*;
