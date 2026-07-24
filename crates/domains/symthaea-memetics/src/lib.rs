// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-memetics
//!
//! Memetics for Symthaea: an idea modeled as a *transmissible, replicating unit
//! with fidelity, mutation, and fitness*, plus the dynamics of how it spreads
//! through a population of minds.
//!
//! This is Phase 0 of `MEMETICS_ANTIMEMETICS_PLAN_2026-07-09.md`. It is
//! deliberately **single-agent / in-process**: no cognitive loop, no mesh, no
//! network. It builds only on the HDC medium (`symthaea-core::BinaryHV`) and the
//! SIR closed form (`symthaea-epidemiology`), so an idea's spread reuses the
//! same math as disease spread — with a **measured R₀**, not a fitted one.
//!
//! ## What's here
//!
//! - [`Meme`] — an idea as a `BinaryHV` payload plus lineage. Replication
//!   (`transmit`) is lossy; `fidelity` measures copy accuracy.
//! - [`BeliefSpread`] — spread parameters. `to_sir` gives the closed-form R₀;
//!   `measure_r0` confirms it agent-by-agent; `outbreak` runs a full mutating
//!   spread and shows that low-fidelity transmission collapses reach.
//! - [`Population`] — minds as belief hypervectors; `aligned_to` builds a
//!   population that already shares an idea.
//!
//! ## The honesty gate (per plan)
//!
//! Every quantity here comes from running the code. The one modeling assumption
//! is [`propagation::resonance_gain`] (confirmation-bias adoption); everything
//! else — R₀, final size, fidelity — is mechanical and cross-checked
//! (`measure_r0` vs `to_sir`).
//!
//! ## Not yet (later phases)
//!
//! - Wiring memes into the live cognitive loop + telemetry (Phase 2).
//! - Cross-agent propagation over the mesh (Phase 3, gated on peer-auth).

pub mod antimeme;
pub mod defense;
pub mod meme;
pub mod propagation;
pub mod ruleset;
pub mod swarm;

pub use antimeme::{AntiMeme, AntiMemeField};
pub use defense::{
    AllowlistMode, FilteredItem, GuardianPosture, MemeticImmuneSystem, MemeticTelemetry,
    ScreenOutcome, WardConfig,
};
pub use meme::Meme;
pub use propagation::{
    BeliefSpread, OutbreakStats, Population, Rng, adoption_probability, resonance_gain,
};
pub use ruleset::{Ruleset, RulesetEntry};
pub use swarm::{MemeSwarm, SwarmOutcome};
