// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Macro/micro world bridge for the Symtropy engine.
//!
//! Connects the multiworld-sim civilization model (monthly ticks) to real-time
//! gameplay (60fps) via threaded simulation with channel-based communication.
//!
//! # Architecture
//!
//! ```text
//! Sim Thread (monthly ticks)          Game Thread (60fps)
//! ┌──────────────────┐                ┌──────────────────┐
//! │ MultiWorld Sim   │  SimSnapshot   │ PhysicsWorld     │
//! │ - governance     │ ──────────────>│ - bodies         │
//! │ - economy        │                │ - consciousness  │
//! │ - factions       │  PlayerAction  │ - rendering      │
//! │ - demographics   │ <──────────────│ - input          │
//! └──────────────────┘                └──────────────────┘
//! ```
//!
//! The sim thread runs ahead, buffering snapshots. The game thread interpolates
//! between snapshots for smooth visualization. Player actions (votes, proposals)
//! are sent back via the reverse channel.

pub mod bridge;
pub mod grid;
pub mod scale;
pub mod snapshot;
pub mod time_control;

pub use bridge::WorldBridge;
pub use grid::{
    BiomeKind, BodyHexGrid, BodyId, EarthH3CellRef, GridSystem, HexCellId, HydrologyState,
    PlanetCell, ProceduralBodyGrid, normalize_lon_deg,
};
pub use scale::WorldScale;
pub use snapshot::{EconomySummary, GovernanceSummary, PlayerAction, SimSnapshot};
pub use time_control::TimeControl;
