// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Integration Tests
//!
//! Tests the interaction between core libraries:
//! - mycelix-core-types: K-Vector, E/N/M classification
//! - rb-bft-consensus: Reputation-based BFT
//! - matl-bridge: Trust computation
//! - feldman-dkg: Distributed key generation
//! - kvector-zkp: Zero-knowledge proofs
//!
//! ## Ecosystem Stress Test
//!
//! The `ecosystem_stress` module provides comprehensive simulation of the entire
//! Mycelix ecosystem under adversarial conditions:
//!
//! ```rust,ignore
//! use mycelix_integration::ecosystem_stress::{EcosystemStressTest, StressTestConfig};
//!
//! let config = StressTestConfig::default();
//! let mut test = EcosystemStressTest::new(config);
//! let result = test.run();
//! println!("Verdict: {}", result.verdict);
//! ```

pub mod dkg_ceremony;
pub mod trust_lifecycle;
pub mod consensus_integration;
pub mod ecosystem_stress;
pub mod chaos_network;
pub mod e2e_pipeline;
// symthaea_bridge and payment_fl_bridge archived with fl-aggregator (Feb 2026).
// Files moved to: mycelix-core/libs/fl-aggregator.archived-v0.1/integration_tests/
// Production FL integration tests: mycelix-workspace/tests/sweettest/tests/phi_attestation_e2e.rs
pub mod cross_happ_bridge;
pub mod gpu_acceleration;
