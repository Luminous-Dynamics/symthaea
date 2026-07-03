// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Broca Tools: Nix/code-analysis/self-repair toolkit
//!
//! Split out of `symthaea-broca` (2026-07-02) — this crate has no bearing on
//! language generation itself; it houses the code-analysis walkers, IaC
//! harvesting/repair, and self-modification bridges that were previously
//! bundled into the language-pipeline crate.

pub mod architect;
pub mod codebase_bridge;
pub mod cognitive_ledger;
pub mod compiler_feedback_bridge;
pub mod cross_modal_bridge;
pub mod foraging_bridge;
pub mod formal_bridge;
pub mod generic_structural_scorer_integration;
pub mod geodesic_bridge;
pub mod go_walker;
pub mod iac_harvester;
pub mod iac_repair;
pub mod invariant_guard;
pub mod memory_kernel;
pub mod memory_ring;
pub mod mission_tree;
pub mod morphological_bridge;
pub mod python_walker;
pub mod simulation_bridge;
pub mod somatic_bridge;
pub mod sovereign_law;
pub mod sovereignty_bridge;
pub mod substrate_rewriter;
pub mod swarm_bridge;
pub mod wasm_architect;

pub use architect::SimulationArchitect;
pub use generic_structural_scorer_integration::{
    GenericStructuralScorer, StructuralVerdict as GenericStructuralVerdict,
};
pub use go_walker::GoWalker;
pub use python_walker::PythonWalker;
