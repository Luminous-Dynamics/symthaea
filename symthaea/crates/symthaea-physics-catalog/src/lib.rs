// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-physics-catalog
//!
//! Lightweight, self-contained physics equation catalog with HDC structural
//! search. Compiles to WASM for browser-side discovery — no symthaea-core
//! dependency, just pure f32 vector math.
//!
//! ## Architecture
//!
//! - 148 equations encoded as 1024-dimensional f32 vectors (reduced from 16,384
//!   for WASM binary size — still captures structural similarity)
//! - 4-channel encoding: name hash, description hash, domain code, dimensional signature
//! - Cosine similarity search with domain filtering
//! - LEM epistemic classification for physics claims
//!
//! ## WASM Usage
//!
//! ```javascript
//! import init, { search_catalog, classify_claim } from './symthaea_physics_catalog.js';
//! await init();
//! const results = search_catalog("Yukawa potential with infinite range");
//! const lem = classify_claim("Element 115 produces gravity waves", 0.0, false);
//! ```

pub mod catalog;
pub mod discovery;
pub mod hv;
pub mod search;

#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use catalog::*;
pub use discovery::*;
pub use search::*;
