// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-operations-research
//!
//! Operations research for Symthaea: inventory optimization, queueing, and
//! shortest paths. A practical decision-science layer the workspace lacked.
//!
//! Core algorithms remain lightweight and avoid a `symthaea-core` link. The
//! aviation evidence-bundle module uses SHA-256 and Ed25519 from established
//! cryptographic crates to verify immutable manifests and trusted receipts.
//!
//! ## Scope
//!
//! - [`inventory`]: economic order quantity (EOQ) + total cost.
//! - [`queue`]: M/M/1 queue metrics (ρ, L, Lq, W, Wq).
//! - [`graph`]: Dijkstra single-source shortest paths.
//! - [`aviation_data_contract`]: fail-closed feature-availability and claim/evidence gates.
//! - [`aviation_calibration_protocol`]: dataset-split integrity, event-cluster leakage,
//!   chronology checks, and a fail-closed guard against declared test-set tuning.
//! - [`aviation_probability_calibration`]: conservative finite-sample TV-radius calibration
//!   from frozen calibration-partition frequencies, with explicit sampling allowance.
//! - [`aviation_online_drift`]: frozen-reference, anytime-valid likelihood-ratio monitoring
//!   for predeclared categorical alternatives, with a separate out-of-support alarm.
//! - [`aviation_evidence_bundle`]: canonical manifest digests and Ed25519-bound verifier
//!   receipts checked against an independently supplied trust store.
//! - [`aviation_decision_readiness`]: composed fail-closed human-review gate over feature,
//!   split, plan verification, active calibration, drift and robust-cost assessments.
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
pub mod aviation_decision_readiness;
pub mod aviation_distributional_robustness;
pub mod aviation_enterprise;
pub mod aviation_evidence_bundle;
pub mod aviation_robustness;
pub mod aviation_network_externalities;
pub mod aviation_network_robustness;
pub mod aviation_online_drift;
pub mod aviation_probability_calibration;
pub mod graph;
pub mod inventory;
pub mod queue;

pub use graph::dijkstra;
pub use inventory::economic_order_quantity;
pub use queue::MM1;
