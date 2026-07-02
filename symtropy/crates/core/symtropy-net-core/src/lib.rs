// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core networking types and traits for Symtropy.

pub mod authority;
pub mod peer;
pub mod state;

pub use authority::SpatialAuthority;
pub use peer::{PeerId, PeerStateCore};
pub use state::{StateAuthority, SyncableState};
