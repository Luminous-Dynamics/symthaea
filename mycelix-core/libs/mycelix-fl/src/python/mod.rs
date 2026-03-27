// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Python bindings for the mycelix-fl crate via PyO3.
//!
//! Exposes Byzantine-robust federated learning aggregation algorithms to Python
//! data scientists. Build with `maturin develop --features python`.
//!
//! # Modules
//!
//! - **types** — `PyGradient`, `PyAggregationResult` (gradient I/O)
//! - **defenses** — Stateless aggregation functions (fedavg, krum, trimmed_mean, etc.)
//! - **pogq** — Stateful PoGQ v4.1 Enhanced Byzantine detection
//! - **coordinator** — Full FL round orchestrator

pub mod types;
pub mod defenses;
pub mod pogq;
pub mod coordinator;

use pyo3::prelude::*;

/// Python module: mycelix_fl_native
#[pymodule]
fn mycelix_fl_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    types::register(m)?;
    defenses::register(m)?;
    pogq::register(m)?;
    coordinator::register(m)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
