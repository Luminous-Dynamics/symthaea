// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Game systems for Symtropy.

pub mod ai_player;
pub mod audio;
pub mod consciousness;
pub mod consciousness_aura;
pub mod dialogue;
pub mod dimension_transition;
pub mod dimensional_leakage;
pub mod engine_physics;
pub mod fep_behavior;
pub mod four_d_rendering;
pub mod harmonies;
pub mod input;
pub mod leviathan;
pub mod living_dungeon;
pub mod menu;
pub mod muse;
pub mod minimap;
pub mod phi_pcg;
pub mod player;
pub mod postprocess;
pub mod procgen;
pub mod psychology;
pub mod rendering;
pub mod room_memory;
pub mod scavenge;
pub mod thermodynamic;

// Sol Atlas globe view — planetary coordination layer.
// Enabled via `cargo build --features atlas`.
#[cfg(feature = "atlas")]
pub mod atlas;
#[cfg(feature = "atlas")]
pub mod cinematic_director;
#[cfg(feature = "atlas")]
pub mod demo_director;

// Mycelix integration — physicalized cryptography.
// Enabled via `cargo build --features mycelix`.
// DO NOT comment these out — they are feature-gated, not broken.
#[cfg(feature = "mycelix")]
pub mod crypto_visuals;
#[cfg(feature = "mycelix")]
pub mod dkg_ceremony;
#[cfg(feature = "mycelix")]
pub mod economy;
#[cfg(feature = "mycelix")]
pub mod epistemics;
#[cfg(feature = "mycelix")]
pub mod faction;
#[cfg(feature = "mycelix")]
pub mod fl_simulation;
#[cfg(feature = "mycelix")]
pub mod governance;
#[cfg(feature = "mycelix")]
pub mod medical_commons;
