// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Topology Generators using Real-Valued Hypervectors
//!
//! This module generates 33 different network topologies representing different
//! levels of integrated information (Φ) for validating the ContinuousHV-based Φ measurement.
//!
//! Topology counts by tier:
//! - Tier 1 (Original 8): Random, Star, Ring, Line, BinaryTree, DenseNetwork, Modular, Lattice
//! - Tier 2 (Geometric 6): Sphere, Torus, KleinBottle, SmallWorld, MobiusStrip, Hyperbolic
//! - Tier 3 (Fractal 9): ScaleFree, Fractal, SierpinskiGasket, FractalTree, KochSnowflake,
//!   MengerSponge, CantorSet, Hypercube, Quantum
//! - Tier 4 (Neural 10): CorticalColumn, Feedforward, Recurrent, Bipartite, CorePeriphery,
//!   BowTie, Attention, Residual, PetersenGraph, CompleteBipartite
//!
//! Each topology is represented as a set of node vectors, where each node's
//! representation encodes its connections to other nodes via ContinuousHV operations.

mod basic;
mod fractal;
mod geometric;
mod neural;
mod types;

#[cfg(test)]
mod tests;

pub use types::*;
