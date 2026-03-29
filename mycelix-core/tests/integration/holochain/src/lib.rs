// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holochain Conductor Integration Tests
//!
//! This module provides integration tests for Mycelix Holochain zomes using
//! SweetTest (when available) or mock implementations for CI.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    TEST INFRASTRUCTURE                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
//! │  │  SweetTest  │   │   Mock      │   │  Network    │          │
//! │  │  Conductor  │   │   Conductor │   │  Simulator  │          │
//! │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
//! │         │                 │                 │                  │
//! │         └─────────────────┼─────────────────┘                  │
//! │                           │                                     │
//! │                   ┌───────▼───────┐                            │
//! │                   │   Test Runner │                            │
//! │                   └───────┬───────┘                            │
//! │                           │                                     │
//! │         ┌─────────────────┼─────────────────┐                  │
//! │         │                 │                 │                  │
//! │  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐          │
//! │  │ FL Zome     │   │ Bridge Zome │   │ PoGQ Zome   │          │
//! │  │ Tests       │   │ Tests       │   │ Tests       │          │
//! │  └─────────────┘   └─────────────┘   └─────────────┘          │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Running with Real Conductor
//!
//! ```bash
//! # Build zomes to WASM
//! cd zomes/federated_learning && cargo build --release --target wasm32-unknown-unknown
//! cd zomes/bridge && cargo build --release --target wasm32-unknown-unknown
//!
//! # Pack DNA
//! hc dna pack ./dna -o mycelix.dna
//!
//! # Run tests with conductor feature
//! cargo test --features conductor -- --nocapture
//! ```
//!
//! ## Zomes Under Test
//!
//! | Zome | Description | Key Functions |
//! |------|-------------|---------------|
//! | `federated_learning` | Byzantine-resistant FL | submit_gradient, aggregate_round |
//! | `bridge` | Cross-hApp communication | register_happ, share_reputation |
//! | `pogq_validation` | Proof of Gradient Quality | publish_proof, verify_proof |
//! | `agents` | Agent management | register_agent, get_agent_info |

pub mod mock_conductor;
pub mod fl_tests;
pub mod bridge_tests;
pub mod pogq_tests;
pub mod multi_agent;

// Re-exports for test convenience
pub use mock_conductor::{MockConductor, MockCell, MockZome, ZomeCallResult};
