// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared wire types for Melothaea/Muse native and wasm32 consumers.
//!
//! The historical monolithic protocol surface is retained byte-for-byte in
//! `legacy.rs` and re-exported here. Keeping this crate root intentionally small
//! lets new platform-neutral wire modules be added without repeatedly rewriting
//! the existing ~108 KiB protocol file.

mod legacy;

pub mod comparison_evidence;

pub use legacy::*;
