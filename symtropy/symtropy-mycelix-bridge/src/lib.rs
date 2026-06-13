// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! # symtropy-mycelix-bridge
//!
//! Bevy plugin wrapping the [`mycelix-conductor-bridge`] subprocess. Lets
//! Bevy systems and NPCs call real Mycelix Holochain zomes via JSON over
//! stdin/stdout.
//!
//! ## Architecture: subprocess IPC, not in-process linking
//!
//! Holochain's Rust client pins `serde = "=1.0.203"` transitively. Bevy
//! 0.18 requires `serde_core >= 1.0.221`. They cannot coexist in one
//! compilation unit. This crate solves that by running the Holochain client
//! as a **separate process** (`mycelix-conductor-bridge` from the monorepo)
//! and exchanging JSON over pipes. The architectural boundary eliminates
//! the dep conflict permanently.
//!
//! ## Milestone 1 scope (2026-04-17)
//!
//! - Single-tenant `HolochainConductor` subprocess.
//! - One zome call: `QueryActiveProposals`.
//! - FIFO request/response correlation (no ids yet).
//! - One-shot startup connect; no reconnect logic.
//! - Tests exercise plumbing only; live-conductor check via example.
//!
//! ## Licensing
//!
//! **AGPL-3.0-or-later** — bundled with the AGPL Mycelix integration toolchain.
//!
//! [`mycelix-conductor-bridge`]: https://github.com/luminous-dynamics/symtropy

pub mod config;
pub mod events;
pub mod plugin;
pub mod resource;
pub mod scenarios;
pub mod systems;

pub use config::MycelixConfig;
pub use events::{MycelixRequest, MycelixResponse};
pub use plugin::BevyMycelixPlugin;
pub use resource::{MycelixClient, MycelixSendError};
pub use scenarios::{proposal_vote_invariant, ScenarioConfig, ScenarioReport};

#[cfg(test)]
mod tests;
