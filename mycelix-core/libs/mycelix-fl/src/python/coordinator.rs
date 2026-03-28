// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Python bindings for the FL coordinator.
//!
//! Wraps [`crate::coordinator::orchestrator::FLCoordinator`] for Python,
//! providing a synchronous round-based interface for running federated
//! learning experiments from Jupyter notebooks and Python scripts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::coordinator::config::CoordinatorConfig;
use crate::coordinator::node_manager::NodeCredential;
use crate::coordinator::orchestrator::FLCoordinator;
use crate::defenses::{CoordinateMedian, FedAvg, Rfa, TrimmedMean};
use crate::pogq::config::PoGQv41Config;

use super::types::{PyAggregationResult, PyGradient};

// ---------------------------------------------------------------------------
// PyFLCoordinator
// ---------------------------------------------------------------------------

/// FL Coordinator: orchestrates multi-round federated learning with Byzantine detection.
///
/// Manages node registration, round execution, gradient validation, and
/// aggregation with optional PoGQ Byzantine detection.
///
/// Args:
///     min_nodes: Minimum nodes required to start a round (default 3).
///     max_nodes: Maximum registered nodes (default 1000).
///     defense: Name of the aggregation defense: "fedavg", "trimmed_mean",
///              "coordinate_median", or "rfa" (default "fedavg").
///     enable_pogq: Whether to enable PoGQ Byzantine detection (default False).
///     pogq_beta: PoGQ EMA smoothing factor (default 0.85, only used if enable_pogq=True).
///     pogq_warm_up: PoGQ warm-up rounds (default 3, only used if enable_pogq=True).
#[pyclass(name = "FLCoordinator")]
pub struct PyFLCoordinator {
    inner: FLCoordinator,
}

#[pymethods]
impl PyFLCoordinator {
    #[new]
    #[pyo3(signature = (
        min_nodes=3,
        max_nodes=1000,
        defense="fedavg",
        enable_pogq=false,
        pogq_beta=None,
        pogq_warm_up=None,
    ))]
    fn new(
        min_nodes: usize,
        max_nodes: usize,
        defense: &str,
        enable_pogq: bool,
        pogq_beta: Option<f64>,
        pogq_warm_up: Option<usize>,
    ) -> PyResult<Self> {
        let config = CoordinatorConfig {
            min_nodes,
            max_nodes,
            rate_limit_per_minute: 1000,
            rate_limit_per_round: 1,
            ..CoordinatorConfig::default()
        };

        let defense_box: Box<dyn crate::defenses::Defense + Send> = match defense {
            "fedavg" => Box::new(FedAvg),
            "trimmed_mean" => Box::new(TrimmedMean),
            "coordinate_median" => Box::new(CoordinateMedian),
            "rfa" => Box::new(Rfa),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown defense '{}'. Choose from: fedavg, trimmed_mean, coordinate_median, rfa",
                    other
                )));
            }
        };

        let pogq_config = if enable_pogq {
            let mut cfg = PoGQv41Config::default();
            if let Some(beta) = pogq_beta {
                cfg.beta = beta;
            }
            if let Some(wu) = pogq_warm_up {
                cfg.warm_up_rounds = wu;
            }
            Some(cfg)
        } else {
            None
        };

        Ok(Self {
            inner: FLCoordinator::new(config, defense_box, pogq_config),
        })
    }

    /// Register a node with the coordinator.
    ///
    /// Args:
    ///     node_id: Unique identifier for the node.
    fn register_node(&mut self, node_id: String) -> PyResult<()> {
        let credential = NodeCredential {
            node_id,
            public_key: None,
            did: None,
            registered_at: 0,
        };
        self.inner
            .register_node(credential)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Start a new FL round.
    ///
    /// Returns:
    ///     The round number.
    ///
    /// Raises:
    ///     ValueError: If fewer than min_nodes are registered, or a round is in progress.
    fn start_round(&mut self) -> PyResult<u64> {
        self.inner
            .start_round()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Submit a gradient for the current round.
    ///
    /// Args:
    ///     gradient: A Gradient object with node_id, values, and round number.
    ///
    /// Raises:
    ///     ValueError: If the node is unregistered, blacklisted, or the gradient is invalid.
    fn submit_gradient(&mut self, gradient: PyGradient) -> PyResult<()> {
        let g = crate::types::Gradient::from(gradient);
        self.inner
            .submit_gradient(g)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Complete the current round: run PoGQ detection (if enabled), then aggregate.
    ///
    /// Returns:
    ///     AggregationResult with the aggregated gradient and node inclusion info.
    ///
    /// Raises:
    ///     ValueError: If no round is in progress, or insufficient gradients submitted.
    fn complete_round(&mut self) -> PyResult<PyAggregationResult> {
        self.inner
            .complete_round()
            .map(PyAggregationResult::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of registered nodes.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get round history as a list of (round_number, status, node_count) tuples.
    fn round_history(&self) -> Vec<(u64, String, usize)> {
        self.inner
            .round_history()
            .iter()
            .map(|s| (s.round_number, s.status.clone(), s.node_count))
            .collect()
    }

    /// Current round number (None if no round is active).
    fn current_round_number(&self) -> Option<u64> {
        self.inner.current_round_number()
    }

    fn __repr__(&self) -> String {
        format!(
            "FLCoordinator(nodes={}, rounds={})",
            self.inner.node_count(),
            self.inner.round_history().len()
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFLCoordinator>()?;
    Ok(())
}
