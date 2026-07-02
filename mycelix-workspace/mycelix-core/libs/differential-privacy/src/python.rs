// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Python bindings for the Differential Privacy library
//!
//! Provides PyO3-based bindings to use DP primitives from Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::{
    GaussianMechanism, LaplaceMechanism, Mechanism,
    PrivacyBudget, GradientClipper, ClippingMethod,
    MomentsAccountant, ZcdpAccountant,
};

/// Python wrapper for GaussianMechanism
#[pyclass(name = "GaussianMechanism")]
pub struct PyGaussianMechanism {
    inner: GaussianMechanism,
}

#[pymethods]
impl PyGaussianMechanism {
    /// Create a new Gaussian mechanism
    ///
    /// Args:
    ///     sensitivity: L2 sensitivity of the query
    ///     epsilon: Privacy parameter (lower = more privacy)
    ///     delta: Probability of privacy violation
    #[new]
    fn new(sensitivity: f64, epsilon: f64, delta: f64) -> PyResult<Self> {
        GaussianMechanism::new(sensitivity, epsilon, delta)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Add noise to a single value
    fn add_noise(&self, value: f64) -> f64 {
        self.inner.add_noise(value)
    }

    /// Add noise to a list of values
    fn apply(&self, values: Vec<f64>) -> Vec<f64> {
        self.inner.apply(&values)
    }

    /// Get the sigma (noise standard deviation)
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    /// Get epsilon
    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    /// Get delta
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }
}

/// Python wrapper for LaplaceMechanism
#[pyclass(name = "LaplaceMechanism")]
pub struct PyLaplaceMechanism {
    inner: LaplaceMechanism,
}

#[pymethods]
impl PyLaplaceMechanism {
    /// Create a new Laplace mechanism
    ///
    /// Args:
    ///     sensitivity: L1 sensitivity of the query
    ///     epsilon: Privacy parameter
    #[new]
    fn new(sensitivity: f64, epsilon: f64) -> PyResult<Self> {
        LaplaceMechanism::new(sensitivity, epsilon)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Add noise to a single value
    fn add_noise(&self, value: f64) -> f64 {
        self.inner.add_noise(value)
    }

    /// Add noise to a list of values
    fn apply(&self, values: Vec<f64>) -> Vec<f64> {
        self.inner.apply(&values)
    }

    /// Get the scale parameter
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// Get epsilon
    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }
}

/// Python wrapper for PrivacyBudget
#[pyclass(name = "PrivacyBudget")]
pub struct PyPrivacyBudget {
    inner: PrivacyBudget,
}

#[pymethods]
impl PyPrivacyBudget {
    /// Create a new privacy budget
    ///
    /// Args:
    ///     max_epsilon: Maximum total epsilon allowed
    ///     max_delta: Maximum total delta allowed
    #[new]
    fn new(max_epsilon: f64, max_delta: f64) -> Self {
        Self {
            inner: PrivacyBudget::new(max_epsilon, max_delta),
        }
    }

    /// Consume privacy budget for an operation
    fn consume(&mut self, epsilon: f64, delta: f64) -> PyResult<()> {
        self.inner
            .consume(epsilon, delta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Consume with a label
    fn consume_with_label(&mut self, epsilon: f64, delta: f64, operation: &str) -> PyResult<()> {
        self.inner
            .consume_with_label(epsilon, delta, operation)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Check if we can afford an operation
    fn can_afford(&self, epsilon: f64, delta: f64) -> bool {
        self.inner.can_afford(epsilon, delta)
    }

    /// Get remaining epsilon
    #[getter]
    fn remaining_epsilon(&self) -> f64 {
        self.inner.remaining_epsilon()
    }

    /// Get remaining delta
    #[getter]
    fn remaining_delta(&self) -> f64 {
        self.inner.remaining_delta()
    }

    /// Get spent epsilon
    #[getter]
    fn spent_epsilon(&self) -> f64 {
        self.inner.spent_epsilon()
    }

    /// Get spent delta
    #[getter]
    fn spent_delta(&self) -> f64 {
        self.inner.spent_delta()
    }

    /// Get query count
    #[getter]
    fn query_count(&self) -> usize {
        self.inner.query_count()
    }

    /// Reset the budget
    fn reset(&mut self) {
        self.inner.reset()
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Python wrapper for GradientClipper
#[pyclass(name = "GradientClipper")]
pub struct PyGradientClipper {
    inner: GradientClipper,
}

#[pymethods]
impl PyGradientClipper {
    /// Create a new L2 gradient clipper
    ///
    /// Args:
    ///     max_norm: Maximum L2 norm for clipping
    #[staticmethod]
    fn l2(max_norm: f64) -> PyResult<Self> {
        GradientClipper::l2(max_norm)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Create a new L1 gradient clipper
    #[staticmethod]
    fn l1(max_norm: f64) -> PyResult<Self> {
        GradientClipper::l1(max_norm)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Create a new L-infinity gradient clipper
    #[staticmethod]
    fn linf(max_norm: f64) -> PyResult<Self> {
        GradientClipper::new(max_norm, ClippingMethod::LInf)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Clip a gradient
    fn clip(&mut self, gradient: Vec<f64>) -> Vec<f64> {
        self.inner.clip(&gradient)
    }

    /// Clip a batch of gradients
    fn clip_batch(&mut self, gradients: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.inner.clip_batch(&gradients)
    }

    /// Clip and aggregate gradients (for federated learning)
    fn clip_and_aggregate(&mut self, gradients: Vec<Vec<f64>>) -> Vec<f64> {
        self.inner.clip_and_aggregate(&gradients)
    }

    /// Get max norm
    #[getter]
    fn max_norm(&self) -> f64 {
        self.inner.max_norm()
    }

    /// Get clip fraction
    #[getter]
    fn clip_fraction(&self) -> f64 {
        self.inner.clip_fraction()
    }

    /// Get average compression
    #[getter]
    fn average_compression(&self) -> f64 {
        self.inner.average_compression()
    }

    /// Reset statistics
    fn reset_stats(&mut self) {
        self.inner.reset_stats()
    }
}

/// Python wrapper for MomentsAccountant
#[pyclass(name = "MomentsAccountant")]
pub struct PyMomentsAccountant {
    inner: MomentsAccountant,
}

#[pymethods]
impl PyMomentsAccountant {
    /// Create a new moments accountant
    ///
    /// Args:
    ///     sampling_probability: q = batch_size / dataset_size
    ///     noise_multiplier: σ (noise stddev relative to sensitivity)
    ///     target_delta: Target δ for computing ε
    #[new]
    fn new(sampling_probability: f64, noise_multiplier: f64, target_delta: f64) -> PyResult<Self> {
        MomentsAccountant::new(sampling_probability, noise_multiplier, target_delta)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Record a single training step
    fn record_step(&mut self) {
        self.inner.record_step()
    }

    /// Record multiple training steps
    fn record_steps(&mut self, n: usize) {
        self.inner.record_steps(n)
    }

    /// Get current epsilon
    #[getter]
    fn current_epsilon(&self) -> f64 {
        self.inner.current_epsilon()
    }

    /// Compute epsilon for a given delta
    fn compute_epsilon(&self, delta: f64) -> f64 {
        self.inner.compute_epsilon(delta)
    }

    /// Get number of steps
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }

    /// Check if we can afford more steps
    fn can_afford_steps(&self, additional_steps: usize, max_epsilon: f64) -> bool {
        self.inner.can_afford_steps(additional_steps, max_epsilon)
    }

    /// Get remaining steps given a budget
    fn remaining_steps(&self, max_epsilon: f64) -> usize {
        self.inner.remaining_steps(max_epsilon)
    }

    /// Reset the accountant
    fn reset(&mut self) {
        self.inner.reset()
    }
}

/// Python wrapper for ZcdpAccountant
#[pyclass(name = "ZcdpAccountant")]
pub struct PyZcdpAccountant {
    inner: ZcdpAccountant,
}

#[pymethods]
impl PyZcdpAccountant {
    /// Create a new zCDP accountant
    #[new]
    fn new() -> Self {
        Self {
            inner: ZcdpAccountant::new(),
        }
    }

    /// Add a Gaussian mechanism
    fn add_gaussian(&mut self, sensitivity: f64, sigma: f64) {
        self.inner.add_gaussian(sensitivity, sigma)
    }

    /// Convert to (ε, δ)-DP
    fn to_epsilon_delta(&self, delta: f64) -> (f64, f64) {
        self.inner.to_epsilon_delta(delta)
    }

    /// Get current rho
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho()
    }

    /// Get number of compositions
    #[getter]
    fn compositions(&self) -> usize {
        self.inner.compositions()
    }

    /// Reset the accountant
    fn reset(&mut self) {
        self.inner.reset()
    }
}

/// Python module for mycelix_differential_privacy
#[pymodule]
fn mycelix_differential_privacy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianMechanism>()?;
    m.add_class::<PyLaplaceMechanism>()?;
    m.add_class::<PyPrivacyBudget>()?;
    m.add_class::<PyGradientClipper>()?;
    m.add_class::<PyMomentsAccountant>()?;
    m.add_class::<PyZcdpAccountant>()?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
