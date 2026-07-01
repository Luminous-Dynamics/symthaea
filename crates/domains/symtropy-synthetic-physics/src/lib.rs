// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

//! # Synthetic Physics Lab (`symtropy-synthetic-physics`)
//!
//! A **sealed experimental lane** for discovering whether local graph-rewriting rules
//! can produce stable, low-dimensional metric attractors.
//!
//! ## Core Question
//!
//! > Can a local graph-rewriting rule stabilize into something that behaves like a 2D
//! > metric surface?
//!
//! ## Attractor Classes
//!
//! Each run produces one of:
//! - [`AttractorClass::StableManifold`] — calm, useful geometry
//! - [`AttractorClass::OscillatoryAttractor`] — rhythmic, possibly useful
//! - [`AttractorClass::StrangeAttractorRisk`] — chaotic drift, quarantine
//! - [`AttractorClass::HairballExplosion`] — degree explosion, halt
//! - [`AttractorClass::StringCollapse`] — dimension collapse to 1D chain
//! - [`AttractorClass::Fragmentation`] — disconnected components, halt
//! - [`AttractorClass::UsefulEmergentManifold`] — 🎯 estimated dim ≈ 2.0 ± 0.3
//!
//! ## Sealed Lane Rules
//!
//! - NO dependency on production game code
//! - NO dependency on robotics crates
//! - NO dependency on `symthaea-fep` (v0.1)
//! - All runs MUST use deterministic seed replay
//! - All runs MUST pass [`GraphSafetyGuards`] before classification
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                   SYNTHETIC PHYSICS LAB                          │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  UpdateRule ──▶ GraphSafetyGuards ──▶ SyntheticGraph             │
//! │                      │                     │                     │
//! │               (reject/rollback)        GraphMetrics              │
//! │                                             │                    │
//! │                                    RingBuffer<GraphMetrics>      │
//! │                                             │                    │
//! │                                    AttractorClassifier           │
//! │                                             │                    │
//! │                                       AttractorClass             │
//! │                                             │                    │
//! │                                    ProjectionFrame ──▶ (Track B) │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

pub mod attractor;
pub mod circuit_breakers;
pub mod free_energy;
pub mod graph;
pub mod metrics;
pub mod projection_bridge;
pub mod ring_buffer;
pub mod update_rules;

pub use attractor::{AttractorClass, AttractorClassifier};
pub use circuit_breakers::GraphSafetyGuards;
pub use free_energy::StructuralFreeEnergy;
pub use graph::{GraphEdge, GraphNode, SyntheticGraph};
pub use metrics::GraphMetrics;
pub use ring_buffer::RingBuffer;
pub use update_rules::UpdateRule;

/// Run the full synthetic physics lab for `ticks` steps.
///
/// Returns the final [`GraphMetrics`] history and the classified attractor.
///
/// # Panics
///
/// Never panics — circuit breakers halt the run and return the last safe state.
pub fn run_experiment(
    rule: UpdateRule,
    ticks: usize,
    seed: u64,
    guards: GraphSafetyGuards,
) -> ExperimentResult {
    let mut graph = SyntheticGraph::new_connected(seed, 16);
    let mut history = RingBuffer::new(64);
    let classifier = AttractorClassifier::default();
    let free_energy = StructuralFreeEnergy::default();

    for tick in 0..ticks {
        // Compute structural free energy of candidate update
        let delta_f = free_energy.compute_delta(&graph, &rule);

        // Apply update only if ΔF < 0 or safety guards pass
        match graph.apply_update(&rule, &guards, delta_f) {
            UpdateOutcome::Applied => {}
            UpdateOutcome::Rejected { reason } => {
                tracing::debug!(tick, reason, "update rejected by guards");
            }
            UpdateOutcome::Quarantined { reason } => {
                tracing::warn!(tick, reason, "graph quarantined — halting run");
                let class = AttractorClass::StrangeAttractorRisk;
                return ExperimentResult {
                    seed,
                    ticks_completed: tick,
                    history,
                    attractor_class: class,
                    halted_early: true,
                    halt_reason: Some(reason),
                };
            }
        }

        let metrics = GraphMetrics::compute(&graph, tick as u64);
        history.push(metrics);
    }

    let snapshot = history.as_slice();
    let attractor_class = classifier.classify(&snapshot);

    ExperimentResult {
        seed,
        ticks_completed: ticks,
        history,
        attractor_class,
        halted_early: false,
        halt_reason: None,
    }
}

/// The outcome of one graph update attempt.
#[derive(Debug, Clone)]
pub enum UpdateOutcome {
    Applied,
    Rejected { reason: String },
    Quarantined { reason: String },
}

/// The result of a complete synthetic physics experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Deterministic seed used for this run.
    pub seed: u64,
    /// Number of ticks completed before halt (or end).
    pub ticks_completed: usize,
    /// 64-frame ring buffer of [`GraphMetrics`].
    pub history: RingBuffer<GraphMetrics>,
    /// Classified attractor type.
    pub attractor_class: AttractorClass,
    /// Whether the run halted early due to safety guards.
    pub halted_early: bool,
    /// Human-readable halt reason, if any.
    pub halt_reason: Option<String>,
}
