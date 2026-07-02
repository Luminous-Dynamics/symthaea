// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL Bridge: Multi-Agent Trust Layer
//!
//! This crate provides the bridge between Holochain DHT and external trust systems,
//! implementing the MATL trust formula:
//!
//! **T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy**
//!
//! ## Features
//!
//! - **Trust Synchronization**: Sync trust scores between Holochain and external systems
//! - **PoGQ Integration**: Validate gradient quality for federated learning
//! - **TCDM Modeling**: Track trust evolution over time
//! - **Entropy Calculation**: Measure trust diversity
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │   Holochain     │────▶│   MATL Bridge   │────▶│  External       │
//! │   DHT           │◀────│                 │◀────│  Systems        │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//!                                 │
//!                         ┌───────┴───────┐
//!                         │               │
//!                    ┌────▼────┐    ┌─────▼─────┐
//!                    │  PoGQ   │    │   TCDM    │
//!                    │ Oracle  │    │  Tracker  │
//!                    └─────────┘    └───────────┘
//! ```

pub mod bridge;
pub mod pogq;
pub mod tcdm;
pub mod entropy;
pub mod sync;
pub mod error;
pub mod consciousness_adapter;

pub use bridge::*;
pub use pogq::*;
pub use tcdm::*;
pub use entropy::*;
pub use sync::*;
pub use error::*;
pub use consciousness_adapter::*;

use mycelix_core_types::trust::matl_weights;

/// MATL protocol version
pub const MATL_VERSION: &str = "1.0";

/// Re-export MATL weights
pub const POGQ_WEIGHT: f32 = matl_weights::POGQ;
pub const TCDM_WEIGHT: f32 = matl_weights::TCDM;
pub const ENTROPY_WEIGHT: f32 = matl_weights::ENTROPY;
