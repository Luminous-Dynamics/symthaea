// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-operations-research
//!
//! Pure-standard-library operations research primitives for inventory,
//! queueing, graph paths, and auditable maintainability experiments.
//!
//! The maintainability evaluator is a descriptive tool for controlled,
//! non-flight mock-up trials. Its output is not maintenance approval,
//! certification, or an airworthiness determination.

pub mod graph;
pub mod inventory;
pub mod maintainability;
pub mod queue;

pub use graph::dijkstra;
pub use inventory::economic_order_quantity;
pub use queue::MM1;
