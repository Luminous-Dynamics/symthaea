// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psychological benchmark implementations.

pub mod affect;
pub mod architecture;
pub mod attention;
pub mod binding;
pub mod butlin;
pub mod causal_reasoning;
pub mod clinical;
pub mod coding;
pub mod cogbench;
pub mod consciousness;
pub mod creativity;
pub mod executive;
pub mod inhibition;
pub mod institutional_reasoning;
pub mod language;
pub mod mathematics;
pub mod memory_agent;
pub mod metacognition;
pub mod motor;
pub mod neuromod;
pub mod normative_integration;
pub mod qualia_confidence;
pub mod reasoning;
pub mod security;
pub mod social;
pub mod spatial;
pub mod speech;
pub mod substrate;
pub mod sustained_attention;
pub mod tombench;
pub mod ual;
pub mod worm;

// Science benchmarks: always-on modules (ode_chaos, chemistry) +
// feature-gated modules (nuclear_physics, materials_design).
pub mod science;

#[cfg(feature = "neural_validation")]
pub mod neural_validation;
pub mod temporal;
