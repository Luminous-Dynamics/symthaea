// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe types for the Energy cluster frontend.
//!
//! These are serde-only mirror types of the Holochain integrity types.
//! No HDI/HDK dependencies — safe for browser WASM.

use serde::{Deserialize, Serialize};

// Add your view types here, e.g.:
//
// #[derive(Clone, Debug, Serialize, Deserialize)]
// pub struct ItemView {
//     pub id: String,
//     pub title: String,
//     pub status: String,
//     pub created_at: u64,
// }
