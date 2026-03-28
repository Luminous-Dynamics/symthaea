// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Python bindings for the PoGQ v4.1 Enhanced Byzantine detection system.
//!
//! PoGQ is a **stateful** defense that maintains per-node history across rounds.
//! It achieves 45% BFT through reputation-weighted validation and adaptive
//! hybrid scoring.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::pogq::config::PoGQv41Config;
use crate::pogq::v41_enhanced::{NodeRoundScore, PoGQRoundResult, PoGQv41Enhanced};

use super::types::{convert_gradients, PyAggregationResult, PyGradient};

// ---------------------------------------------------------------------------
// PyPoGQRoundResult
// ---------------------------------------------------------------------------

/// Per-node scoring result from a single PoGQ round.
#[pyclass(name = "NodeScore")]
#[derive(Clone)]
pub struct PyNodeScore {
    /// Node identifier.
    #[pyo3(get)]
    pub node_id: String,
    /// Raw cosine-similarity direction score vs. centroid.
    #[pyo3(get)]
    pub direction_score: f64,
    /// Magnitude score (1 - |z_norm|, clamped to [0, 1]).
    #[pyo3(get)]
    pub magnitude_score: f64,
    /// Hybrid score: lambda * direction + (1 - lambda) * magnitude.
    #[pyo3(get)]
    pub hybrid_score: f64,
    /// EMA-smoothed score after this round.
    #[pyo3(get)]
    pub ema_score: f64,
    /// Adaptive lambda used for this gradient.
    #[pyo3(get)]
    pub lambda: f64,
    /// Whether this node is currently quarantined.
    #[pyo3(get)]
    pub quarantined: bool,
    /// Whether this was flagged as a violation this round.
    #[pyo3(get)]
    pub is_violation: bool,
}

#[pymethods]
impl PyNodeScore {
    fn __repr__(&self) -> String {
        format!(
            "NodeScore(node_id='{}', hybrid={:.3}, ema={:.3}, quarantined={})",
            self.node_id, self.hybrid_score, self.ema_score, self.quarantined
        )
    }
}

impl From<&NodeRoundScore> for PyNodeScore {
    fn from(s: &NodeRoundScore) -> Self {
        Self {
            node_id: s.node_id.clone(),
            direction_score: s.direction_score,
            magnitude_score: s.magnitude_score,
            hybrid_score: s.hybrid_score,
            ema_score: s.ema_score,
            lambda: s.lambda,
            quarantined: s.quarantined,
            is_violation: s.is_violation,
        }
    }
}

/// Result of a full PoGQ evaluation round.
#[pyclass(name = "PoGQRoundResult")]
#[derive(Clone)]
pub struct PyPoGQRoundResult {
    /// Per-node scoring details.
    #[pyo3(get)]
    pub node_scores: Vec<PyNodeScore>,
    /// Global threshold used for this round.
    #[pyo3(get)]
    pub threshold: f64,
    /// Number of nodes that passed (not quarantined).
    #[pyo3(get)]
    pub n_honest: usize,
    /// Number of nodes quarantined.
    #[pyo3(get)]
    pub n_quarantined: usize,
    /// Round number.
    #[pyo3(get)]
    pub round: usize,
}

#[pymethods]
impl PyPoGQRoundResult {
    fn __repr__(&self) -> String {
        format!(
            "PoGQRoundResult(round={}, honest={}, quarantined={}, threshold={:.4})",
            self.round, self.n_honest, self.n_quarantined, self.threshold
        )
    }
}

impl From<PoGQRoundResult> for PyPoGQRoundResult {
    fn from(r: PoGQRoundResult) -> Self {
        Self {
            node_scores: r.node_scores.iter().map(PyNodeScore::from).collect(),
            threshold: r.threshold,
            n_honest: r.n_honest,
            n_quarantined: r.n_quarantined,
            round: r.round,
        }
    }
}

// ---------------------------------------------------------------------------
// PyPoGQ
// ---------------------------------------------------------------------------

/// PoGQ v4.1 Enhanced: stateful Byzantine detection system.
///
/// Maintains per-node state across rounds and produces aggregation results
/// that exclude quarantined nodes. Achieves 45% BFT.
///
/// Args:
///     beta: EMA smoothing factor (default 0.85).
///     warm_up_rounds: Rounds before quarantine enforcement (default 3).
///     k_quarantine: Consecutive violations to enter quarantine (default 2).
///     m_release: Consecutive clears to release from quarantine (default 3).
///     direction_weight: Weight for cosine direction score (default 0.6).
///     magnitude_weight: Weight for magnitude score (default 0.4).
#[pyclass(name = "PoGQ")]
pub struct PyPoGQ {
    inner: PoGQv41Enhanced,
}

#[pymethods]
impl PyPoGQ {
    /// Create a new PoGQ instance with optional configuration overrides.
    #[new]
    #[pyo3(signature = (
        beta=None,
        warm_up_rounds=None,
        k_quarantine=None,
        m_release=None,
        direction_weight=None,
        magnitude_weight=None,
    ))]
    fn new(
        beta: Option<f64>,
        warm_up_rounds: Option<usize>,
        k_quarantine: Option<usize>,
        m_release: Option<usize>,
        direction_weight: Option<f64>,
        magnitude_weight: Option<f64>,
    ) -> PyResult<Self> {
        let mut config = PoGQv41Config::default();

        if let Some(v) = beta {
            config.beta = v;
        }
        if let Some(v) = warm_up_rounds {
            config.warm_up_rounds = v;
        }
        if let Some(v) = k_quarantine {
            config.k_quarantine = v;
        }
        if let Some(v) = m_release {
            config.m_release = v;
        }
        if let Some(v) = direction_weight {
            config.direction_weight = v;
        }
        if let Some(v) = magnitude_weight {
            config.magnitude_weight = v;
        }

        config
            .validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(Self {
            inner: PoGQv41Enhanced::new(config),
        })
    }

    /// Run one round of PoGQ evaluation (scoring + state machine update).
    ///
    /// Args:
    ///     gradients: List of Gradient objects for this round.
    ///
    /// Returns:
    ///     PoGQRoundResult with per-node scores and quarantine status.
    fn evaluate_round(&mut self, gradients: Vec<PyGradient>) -> PyResult<PyPoGQRoundResult> {
        let grads = convert_gradients(&gradients);
        self.inner
            .evaluate_round(&grads)
            .map(PyPoGQRoundResult::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Aggregate gradients with PoGQ filtering (evaluate + average honest).
    ///
    /// Runs PoGQ evaluation, excludes quarantined nodes, and averages the
    /// remaining gradients.
    ///
    /// Args:
    ///     gradients: List of Gradient objects for this round.
    ///
    /// Returns:
    ///     AggregationResult with quarantined nodes excluded.
    fn aggregate(&mut self, gradients: Vec<PyGradient>) -> PyResult<PyAggregationResult> {
        let grads = convert_gradients(&gradients);
        self.inner
            .aggregate(&grads)
            .map(PyAggregationResult::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Check if a node is currently quarantined.
    fn is_quarantined(&self, node_id: &str) -> bool {
        self.inner.is_quarantined(node_id)
    }

    /// Get the list of currently quarantined node IDs.
    fn quarantined_nodes(&self) -> Vec<String> {
        self.inner
            .node_states()
            .iter()
            .filter(|(_, state)| state.quarantined)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get current round number.
    #[getter]
    fn current_round(&self) -> usize {
        self.inner.round()
    }

    /// Get the EMA beta parameter.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.config().beta
    }

    fn __repr__(&self) -> String {
        format!(
            "PoGQ(round={}, nodes={}, quarantined={})",
            self.inner.round(),
            self.inner.node_states().len(),
            self.inner
                .node_states()
                .values()
                .filter(|s| s.quarantined)
                .count()
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPoGQ>()?;
    m.add_class::<PyNodeScore>()?;
    m.add_class::<PyPoGQRoundResult>()?;
    Ok(())
}
