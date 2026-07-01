// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-phi-oracle — Integration Spectrometer
//!
//! Measures the **structural coherence** of external systems — power grids,
//! ecosystems, DHT networks, swarm robotics, or any system expressible as
//! multivariate time series.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use symthaea_phi_oracle::{IntegrationOracle, OracleConfig};
//!
//! // Simple path: just specify number of variables
//! let mut oracle = IntegrationOracle::new_simple(10, OracleConfig::default()).unwrap();
//!
//! // Feed observations (e.g., sensor readings over time)
//! for t in 0..100 {
//!     let observation: Vec<f64> = (0..10).map(|i| {
//!         (t as f64 * 0.1 + i as f64 * 0.3).sin()
//!     }).collect();
//!     oracle.observe(&observation).unwrap();
//! }
//!
//! // Measure integration
//! if let Some(report) = oracle.measure() {
//!     println!("Integration index: {:.4}", report.integration_index);
//!     println!("Normalized:        {:.4}", report.normalized_index);
//!     println!("MIP cut: {:?} | {:?}",
//!         report.minimum_information_partition.0,
//!         report.minimum_information_partition.1);
//! }
//! ```
//!
//! ## Naming Convention
//!
//! Externally, the measured quantity is called **"integration index"** or
//! **"structural coherence proxy"** — not "Phi" or "consciousness." Internally,
//! the algorithms are IIT-inspired (spectral MIP via Fiedler ordering).
//!
//! ## Architecture
//!
//! - **[`SystemEncoder`]** trait — maps domain-specific observations into
//!   hyperdimensional vectors.
//! - **[`TimeSeriesEncoder`]** — built-in encoder using random projection
//!   (Johnson-Lindenstrauss style).
//! - **[`CovarianceEncoder`]** — bypass encoder for pre-computed covariance
//!   matrices.
//! - **[`IntegrationOracle`]** — core orchestrator: observe → build covariance
//!   → spectral MIP → report.
//! - **[`IntegrationReport`]** — full measurement output including MIP,
//!   spectral order, and temporal coherence.
//! - **[`CoherenceTrend`]** — track integration over multiple measurement
//!   windows to detect strengthening/weakening trends.
//!
//! ## Dependency
//!
//! Depends **only** on `symthaea-core`. Zero coupling to the cognitive loop.

mod encoder;
mod error;
mod oracle;
mod result;
mod temporal;
mod trend;
mod window;

pub use encoder::{CovarianceEncoder, SystemEncoder, TimeSeriesEncoder};
pub use error::OracleError;
pub use oracle::{IntegrationOracle, OracleConfig};
pub use result::{HierarchicalReport, IntegrationReport, PersistentCycle, TemporalCoherence};
pub use trend::{CoherenceTrend, TrendPoint};
