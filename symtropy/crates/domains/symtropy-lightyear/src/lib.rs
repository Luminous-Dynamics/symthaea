// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightyear integration for Symtropy.
//!
//! Bridges Symtropy's consciousness-coupled physics engine with Lightyear's
//! production netcode (prediction, rollback, interpolation, interest management).
//!
//! # Architecture
//!
//! ```text
//! Lightyear                          Symtropy
//! ┌─────────────────────┐           ┌─────────────────────────┐
//! │ Prediction + Rollback│ ◄──────► │ PhysicsWorld<D>          │
//! │ Snapshot Interpolation│          │ ConsciousnessField<D>    │
//! │ Delta Compression    │          │ ThermodynamicLedger      │
//! │ Interest Management  │          │ SpatialAuthority         │
//! └──────────┬──────────┘           └─────────────────────────┘
//!            │
//!     ┌──────▼──────┐
//!     │  IrohIo     │  ← ~100 LOC bridge
//!     │  (QUIC P2P) │
//!     └─────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use symtropy_lightyear::SymtropyNetPlugin;
//!
//! App::new()
//!     .add_plugins(SymtropyNetPlugin)
//!     .run();
//! ```

pub mod components;
pub mod iroh_io;
pub mod plugin;
pub mod protocol;

pub use plugin::SymtropyNetPlugin;
