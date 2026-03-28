// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Game systems for Symtropy.

// Core gameplay systems
pub mod audio;
pub mod consciousness;
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

// Mycelix integration systems — currently disabled due to:
// 1. ConsciousnessProfile → ConsciousnessComp rename (API in flux)
// 2. Bevy Query B0001 conflicts between governance/economy/faction systems
// 3. NpcTrust/KVector API changes from bridge-common integration
// Re-enable when the other session's Mycelix integration stabilizes.
// pub mod dkg_ceremony;
// pub mod economy;
// pub mod epistemics;
// pub mod faction;
// pub mod fl_simulation;
// pub mod governance;
