// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Python-callable stateless defense functions.
//!
//! Each function takes a list of `PyGradient` values and returns a
//! `PyAggregationResult`. These wrap the Rust defense implementations
//! from [`crate::defenses`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::defenses::{CoordinateMedian, Defense, FedAvg, Rfa, TrimmedMean};
use crate::defenses::fltrust::FLTrust;
use crate::defenses::krum::{Bulyan, Krum, MultiKrum};
use crate::types::DefenseConfig;

use super::types::{convert_gradients, PyAggregationResult, PyGradient};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn fl_err_to_py(e: crate::error::FlError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Defense functions
// ---------------------------------------------------------------------------

/// Run FedAvg aggregation (simple unweighted mean, no Byzantine tolerance).
///
/// Args:
///     gradients: List of Gradient objects to aggregate.
///
/// Returns:
///     AggregationResult with the averaged gradient.
#[pyfunction]
fn fedavg(gradients: Vec<PyGradient>) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    let config = DefenseConfig::default();
    FedAvg
        .aggregate(&grads, &config)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run Krum selection (selects the single most central gradient).
///
/// Args:
///     gradients: List of Gradient objects.
///     num_byzantine: Upper bound on the number of Byzantine participants.
///
/// Returns:
///     AggregationResult containing the selected gradient.
///
/// Requires: n >= 2*num_byzantine + 3
#[pyfunction]
fn krum(gradients: Vec<PyGradient>, num_byzantine: usize) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    Krum::aggregate(&grads, num_byzantine)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run Multi-Krum (select top-k most central gradients and average).
///
/// Args:
///     gradients: List of Gradient objects.
///     num_byzantine: Upper bound on the number of Byzantine participants.
///     k: Number of gradients to select. Defaults to n - num_byzantine.
///
/// Returns:
///     AggregationResult with the averaged top-k gradients.
///
/// Requires: n >= 2*num_byzantine + 3
#[pyfunction]
#[pyo3(signature = (gradients, num_byzantine, k=None))]
fn multi_krum(
    gradients: Vec<PyGradient>,
    num_byzantine: usize,
    k: Option<usize>,
) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    MultiKrum::aggregate(&grads, num_byzantine, k)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run Bulyan (Multi-Krum selection + coordinate-wise trimmed mean).
///
/// Args:
///     gradients: List of Gradient objects.
///     num_byzantine: Upper bound on the number of Byzantine participants.
///
/// Returns:
///     AggregationResult with the Bulyan-aggregated gradient.
///
/// Requires: n >= 4*num_byzantine + 3
#[pyfunction]
fn bulyan(gradients: Vec<PyGradient>, num_byzantine: usize) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    Bulyan::aggregate(&grads, num_byzantine)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run Trimmed Mean aggregation (trims extreme values from each coordinate).
///
/// Args:
///     gradients: List of Gradient objects.
///     trim_ratio: Fraction to trim from each end (default 0.1).
///
/// Returns:
///     AggregationResult with the trimmed mean gradient.
#[pyfunction]
#[pyo3(signature = (gradients, trim_ratio=0.1))]
fn trimmed_mean(gradients: Vec<PyGradient>, trim_ratio: f32) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    let config = DefenseConfig {
        trim_ratio,
        ..DefenseConfig::default()
    };
    TrimmedMean
        .aggregate(&grads, &config)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run Coordinate Median aggregation (element-wise median, 50% BFT).
///
/// Args:
///     gradients: List of Gradient objects.
///
/// Returns:
///     AggregationResult with the coordinate-wise median gradient.
#[pyfunction]
fn coordinate_median(gradients: Vec<PyGradient>) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    let config = DefenseConfig::default();
    CoordinateMedian
        .aggregate(&grads, &config)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run RFA (Robust Federated Aggregation) via geometric median (Weiszfeld algorithm).
///
/// Args:
///     gradients: List of Gradient objects.
///     max_iter: Maximum Weiszfeld iterations (default 100).
///     tol: Convergence tolerance (default 1e-6).
///
/// Returns:
///     AggregationResult with the geometric median gradient.
#[pyfunction]
#[pyo3(signature = (gradients, max_iter=None, tol=None))]
fn rfa(
    gradients: Vec<PyGradient>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    let config = DefenseConfig {
        weiszfeld_max_iter: max_iter.unwrap_or(100),
        weiszfeld_tol: tol.unwrap_or(1e-6),
        ..DefenseConfig::default()
    };
    Rfa.aggregate(&grads, &config)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

/// Run FLTrust aggregation (direction-based trust scoring against a server gradient).
///
/// Uses a trusted server gradient to score client gradients by cosine similarity.
/// Clients with negative similarity (opposite direction) are excluded.
///
/// Args:
///     gradients: List of client Gradient objects.
///     server_gradient: The trusted server gradient as a list of floats.
///
/// Returns:
///     AggregationResult weighted by directional trust scores.
#[pyfunction]
fn fltrust(
    gradients: Vec<PyGradient>,
    server_gradient: Vec<f32>,
) -> PyResult<PyAggregationResult> {
    let grads = convert_gradients(&gradients);
    FLTrust::aggregate(&grads, &server_gradient)
        .map(PyAggregationResult::from)
        .map_err(fl_err_to_py)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fedavg, m)?)?;
    m.add_function(wrap_pyfunction!(krum, m)?)?;
    m.add_function(wrap_pyfunction!(multi_krum, m)?)?;
    m.add_function(wrap_pyfunction!(bulyan, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean, m)?)?;
    m.add_function(wrap_pyfunction!(coordinate_median, m)?)?;
    m.add_function(wrap_pyfunction!(rfa, m)?)?;
    m.add_function(wrap_pyfunction!(fltrust, m)?)?;
    Ok(())
}
