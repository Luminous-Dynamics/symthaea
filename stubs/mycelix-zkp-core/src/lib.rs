// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # mycelix-zkp-core (standalone subset)
//!
//! This is the self-contained subset of `mycelix-zkp-core` needed to build
//! symthaea standalone: `types`, `domain`, `fixed_point`, `pogq`, `error`,
//! a trimmed `consciousness` (just `CivicTier`), and the `dilithium`
//! module (feature-gated, matching the full crate).
//!
//! Everything here is real, verbatim-ported logic (not a mock) -- it just
//! omits the RISC0/Winterfell/Miden STARK backends and their private
//! `proofs-config`/`proofs-commitment` dependencies, which live in the
//! private `mycelix-workspace` monorepo and aren't needed by symthaea's
//! own usage (Dilithium5 signing/verification, PoGQ simulation, and the
//! `CivicTier` consciousness-tier enum).

pub mod consciousness;
#[cfg(feature = "dilithium")]
pub mod dilithium;
pub mod domain;
pub mod error;
pub mod fixed_point;
pub mod pogq;
pub mod types;

pub use consciousness::CivicTier;
#[cfg(feature = "dilithium")]
pub use dilithium::DilithiumKeypair;
pub use domain::DomainTag;
pub use error::{ZkpError, ZkpResult};
pub use fixed_point::{FixedPoint, Q16_16_SCALE};
