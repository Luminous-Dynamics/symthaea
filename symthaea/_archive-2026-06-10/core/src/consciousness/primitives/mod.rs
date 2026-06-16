// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Consciousness Layer
//!
//! HDC-based consciousness primitives: reasoning, evolution, discovery,
//! lattice structure, composition rules, and belief bridges.

/// Code-specific primitive evolution — discovers novel code patterns via HDC algebra.
#[cfg(feature = "code_generation")]
pub mod code_primitive_evolution;
pub mod code_primitives;
pub mod compositionality;
/// Immune-inspired code evolution — V(D)J recombination, affinity maturation,
/// negative selection, cytokine signaling for self-improving code generation.
#[cfg(feature = "code_generation")]
pub mod immune_code_evolution;
pub mod primitive_belief_bridge;
pub mod primitive_composition_rules;
pub mod primitive_consciousness;
pub mod primitive_discovery;
pub mod primitive_evolution;
pub mod primitive_lattice;
pub mod primitive_reasoning;
pub mod primitive_validation;
pub mod stability_regime;
