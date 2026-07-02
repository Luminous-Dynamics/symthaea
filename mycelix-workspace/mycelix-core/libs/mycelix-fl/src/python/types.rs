// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Python-visible gradient and aggregation types.

use pyo3::prelude::*;

use crate::types::{AggregationResult, Gradient};

// ---------------------------------------------------------------------------
// PyGradient
// ---------------------------------------------------------------------------

/// A gradient submission from a single node in a federated learning round.
///
/// Args:
///     node_id: Unique identifier of the submitting node.
///     values: The gradient vector (flattened model update) as a list of floats.
///     round: The FL round number this gradient belongs to.
#[pyclass(name = "Gradient")]
#[derive(Clone)]
pub struct PyGradient {
    /// Gradient values (flattened model update).
    #[pyo3(get, set)]
    pub values: Vec<f32>,
    /// Unique identifier of the submitting node.
    #[pyo3(get, set)]
    pub node_id: String,
    /// FL round number.
    #[pyo3(get, set)]
    pub round: u64,
}

#[pymethods]
impl PyGradient {
    /// Create a new gradient.
    #[new]
    fn new(node_id: String, values: Vec<f32>, round: u64) -> Self {
        Self {
            values,
            node_id,
            round,
        }
    }

    /// Dimensionality of this gradient.
    fn dim(&self) -> usize {
        self.values.len()
    }

    /// L2 norm of this gradient.
    fn norm(&self) -> f64 {
        self.values
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    fn __repr__(&self) -> String {
        format!(
            "Gradient(node_id='{}', dim={}, round={})",
            self.node_id,
            self.values.len(),
            self.round
        )
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// PyAggregationResult
// ---------------------------------------------------------------------------

/// Result of an aggregation operation.
///
/// Attributes:
///     gradient: The aggregated gradient vector.
///     included_nodes: Node IDs that were included in the aggregation.
///     excluded_nodes: Node IDs that were excluded (filtered out by the defense).
///     scores: Per-node quality scores as a list of (node_id, score) tuples.
#[pyclass(name = "AggregationResult")]
#[derive(Clone)]
pub struct PyAggregationResult {
    /// The aggregated gradient vector.
    #[pyo3(get)]
    pub gradient: Vec<f32>,
    /// Node IDs that were included in the aggregation.
    #[pyo3(get)]
    pub included_nodes: Vec<String>,
    /// Node IDs that were excluded (filtered out by the defense).
    #[pyo3(get)]
    pub excluded_nodes: Vec<String>,
    /// Per-node quality scores as (node_id, score) tuples.
    #[pyo3(get)]
    pub scores: Vec<(String, f64)>,
}

#[pymethods]
impl PyAggregationResult {
    /// Number of dimensions in the aggregated gradient.
    fn dim(&self) -> usize {
        self.gradient.len()
    }

    /// Number of nodes included.
    fn num_included(&self) -> usize {
        self.included_nodes.len()
    }

    /// Number of nodes excluded.
    fn num_excluded(&self) -> usize {
        self.excluded_nodes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "AggregationResult(dim={}, included={}, excluded={})",
            self.gradient.len(),
            self.included_nodes.len(),
            self.excluded_nodes.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<&PyGradient> for Gradient {
    fn from(py: &PyGradient) -> Self {
        Gradient {
            node_id: py.node_id.clone(),
            values: py.values.clone(),
            round: py.round,
        }
    }
}

impl From<PyGradient> for Gradient {
    fn from(py: PyGradient) -> Self {
        Gradient {
            node_id: py.node_id,
            values: py.values,
            round: py.round,
        }
    }
}

impl From<AggregationResult> for PyAggregationResult {
    fn from(r: AggregationResult) -> Self {
        Self {
            gradient: r.gradient,
            included_nodes: r.included_nodes,
            excluded_nodes: r.excluded_nodes,
            scores: r.scores,
        }
    }
}

/// Convert a Python list of PyGradient to a Vec<Gradient>.
pub(crate) fn convert_gradients(gradients: &[PyGradient]) -> Vec<Gradient> {
    gradients.iter().map(Gradient::from).collect()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGradient>()?;
    m.add_class::<PyAggregationResult>()?;
    Ok(())
}
