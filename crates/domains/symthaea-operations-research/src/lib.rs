// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-operations-research
//!
//! Operations research for Symthaea: inventory optimization, queueing, and
//! shortest paths. A practical decision-science layer the workspace lacked.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form results
//! and an exact Dijkstra, checked against known values.
//!
//! ## Scope
//!
//! - [`inventory`]: economic order quantity (EOQ) + total cost.
//! - [`queue`]: M/M/1 queue metrics (ρ, L, Lq, W, Wq).
//! - [`graph`]: Dijkstra single-source shortest paths.
//! - [`aviation_data_contract`]: fail-closed feature-availability and claim/evidence gates.
//! - [`aviation_calibration_protocol`]: dataset-split integrity, event-cluster leakage,
//!   chronology checks, and a fail-closed guard against declared test-set tuning.
//! - [`aviation_distributional_robustness`]: finite total-variation uncertainty evaluation
//!   over paired candidate/baseline costs, with read-only reports and robust gates.
//! - [`aviation_enterprise`]: tiny exact cross-lifecycle aviation research fixture,
//!   with a separately implemented plan verifier and negative controls. This is
//!   not dispatch, maintenance-release, or airworthiness authority.
//!
//! ## Example
//!
//! ```
//! use symthaea_operations_research::{inventory::economic_order_quantity, queue::MM1};
//! assert!((economic_order_quantity(1000.0, 10.0, 2.0) - 100.0).abs() < 1e-9);
//! assert!((MM1 { arrival_rate: 2.0, service_rate: 3.0 }.avg_in_system() - 2.0).abs() < 1e-12);
//! ```

pub mod aviation_calibration_protocol;
pub mod aviation_data_contract;
pub mod aviation_distributional_robustness;
pub mod aviation_enterprise;
pub mod aviation_robustness;
pub mod aviation_network_externalities;
pub mod aviation_network_robustness;
pub mod graph;
pub mod inventory;
pub mod queue;

pub use graph::dijkstra;
pub use inventory::economic_order_quantity;
pub use queue::MM1;
