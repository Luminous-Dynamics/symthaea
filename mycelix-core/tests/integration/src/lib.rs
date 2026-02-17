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
pub mod symthaea_bridge;
pub mod payment_fl_bridge;
pub mod cross_happ_bridge;
pub mod gpu_acceleration;
