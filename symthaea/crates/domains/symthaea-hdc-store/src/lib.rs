// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! symthaea-hdc-store -- Zero-copy mmap'd BinaryHV storage with persistent LSH.
//!
//! Provides efficient on-disk storage for 16,384-bit BinaryHV vectors with
//! O(1) random access via memory-mapped I/O and approximate nearest neighbor
//! search via locality-sensitive hashing.

pub mod compaction;
pub mod error;
pub mod header;
pub mod lsh_persistent;
pub mod store;

pub use error::HdcStoreError;
pub use header::StoreHeader;
pub use lsh_persistent::LshIndex;
pub use store::{HdcStore, StoreConfig};
