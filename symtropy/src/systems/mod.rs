// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Game systems for Symtropy.

pub mod audio;
pub mod consciousness;
pub mod dialogue;
pub mod room_memory;
pub mod fep_behavior;
pub mod harmonies;
pub mod input;
pub mod leviathan;
pub mod menu;
pub mod minimap;
pub mod player;
pub mod postprocess;
pub mod procgen;
pub mod rendering;
pub mod scavenge;

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
pub mod medical_commons;
#[cfg(feature = "mycelix")]
pub mod governance;
