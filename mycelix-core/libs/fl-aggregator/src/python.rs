//! Python bindings via PyO3.
//!
//! Provides Python classes that wrap the Rust aggregator for use in ML pipelines.
//!
//! ## Usage from Python
//!
//! ```python
//! from fl_aggregator import PyAggregator, Defense
//! import numpy as np
//!
//! # Create aggregator with Krum defense
//! aggregator = PyAggregator(
//!     expected_nodes=5,
//!     defense=Defense.krum(f=1)
//! )
//!
//! # Submit gradients
//! aggregator.submit("node1", np.array([1.0, 2.0, 3.0]))
//! aggregator.submit("node2", np.array([1.1, 2.1, 3.1]))
//! # ...
//!
//! # Get aggregated result
//! if aggregator.is_complete():
//!     result = aggregator.finalize()
//!     print(f"Aggregated gradient: {result}")
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyReadonlyArray1};

use crate::aggregator::{Aggregator, AggregatorConfig};
use crate::byzantine::Defense;
use crate::compression::{CompressionConfig, GradientCompressor, CompressedGradient};
use crate::pogq::{PoGQv41Config, PoGQv41Enhanced, MondrianProfile};
use std::collections::BTreeSet;

/// Python wrapper for Defense enum.
#[pyclass(name = "Defense")]
#[derive(Clone)]
pub struct PyDefense {
    inner: Defense,
}

#[pymethods]
impl PyDefense {
    /// Create FedAvg defense (no Byzantine protection).
    #[staticmethod]
    fn fedavg() -> Self {
        Self {
            inner: Defense::FedAvg,
        }
    }

    /// Create Krum defense.
    #[staticmethod]
    fn krum(f: usize) -> Self {
        Self {
            inner: Defense::Krum { f },
        }
    }

    /// Create Multi-Krum defense.
    #[staticmethod]
    fn multi_krum(f: usize, k: usize) -> Self {
        Self {
            inner: Defense::MultiKrum { f, k },
        }
    }

    /// Create Median defense.
    #[staticmethod]
    fn median() -> Self {
        Self {
            inner: Defense::Median,
        }
    }

    /// Create Trimmed Mean defense.
    #[staticmethod]
    fn trimmed_mean(beta: f32) -> PyResult<Self> {
        if beta < 0.0 || beta >= 0.5 {
            return Err(PyValueError::new_err("beta must be in [0, 0.5)"));
        }
        Ok(Self {
            inner: Defense::TrimmedMean { beta },
        })
    }

    /// Create Geometric Median defense.
    #[staticmethod]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-6))]
    fn geometric_median(max_iterations: usize, tolerance: f32) -> Self {
        Self {
            inner: Defense::GeometricMedian {
                max_iterations,
                tolerance,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("Defense({})", self.inner)
    }
}

/// Python wrapper for the aggregator.
#[pyclass(name = "Aggregator")]
pub struct PyAggregator {
    inner: Aggregator,
}

#[pymethods]
impl PyAggregator {
    /// Create a new aggregator.
    ///
    /// Args:
    ///     expected_nodes: Number of nodes expected per round
    ///     defense: Defense algorithm (default: Krum with f=1)
    ///     max_memory_mb: Maximum memory in MB (default: 2000)
    #[new]
    #[pyo3(signature = (expected_nodes, defense=None, max_memory_mb=2000))]
    fn new(expected_nodes: usize, defense: Option<PyDefense>, max_memory_mb: usize) -> Self {
        let defense = defense.map(|d| d.inner).unwrap_or(Defense::Krum { f: 1 });

        let config = AggregatorConfig::default()
            .with_expected_nodes(expected_nodes)
            .with_defense(defense)
            .with_max_memory(max_memory_mb * 1_000_000);

        Self {
            inner: Aggregator::new(config),
        }
    }

    /// Submit a gradient from a node.
    ///
    /// Args:
    ///     node_id: Unique identifier for the node
    ///     gradient: NumPy array of gradient values
    fn submit(&mut self, node_id: &str, gradient: PyReadonlyArray1<f32>) -> PyResult<()> {
        let array = gradient.as_array().to_owned();
        self.inner
            .submit(node_id, array)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if the round is complete.
    fn is_complete(&self) -> bool {
        self.inner.is_round_complete()
    }

    /// Check if enough nodes have submitted to aggregate.
    fn can_aggregate(&self) -> bool {
        self.inner.can_aggregate()
    }

    /// Get number of submissions in current round.
    fn submission_count(&self) -> usize {
        self.inner.submission_count()
    }

    /// Finalize the round and return aggregated gradient.
    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let result = self
            .inner
            .finalize_round()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyArray1::from_vec(py, result.to_vec()))
    }

    /// Get current round number.
    #[getter]
    fn round(&self) -> u64 {
        self.inner.current_round()
    }

    /// Get current memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Reset the aggregator.
    fn reset(&mut self) {
        self.inner.reset()
    }

    fn __repr__(&self) -> String {
        format!(
            "Aggregator(round={}, submissions={}, memory={})",
            self.inner.current_round(),
            self.inner.submission_count(),
            self.inner.memory_usage()
        )
    }
}

/// Python wrapper for gradient compression.
#[pyclass(name = "GradientCompressor")]
pub struct PyGradientCompressor {
    inner: GradientCompressor,
}

#[pymethods]
impl PyGradientCompressor {
    /// Create a new compressor.
    ///
    /// Args:
    ///     compression_ratio: Target compression (e.g., 10.0 for 10x)
    ///     quantization_bits: Bits for values (8 or 16)
    ///     error_feedback: Accumulate compression errors
    #[new]
    #[pyo3(signature = (compression_ratio=10.0, quantization_bits=8, error_feedback=true))]
    fn new(compression_ratio: f32, quantization_bits: u8, error_feedback: bool) -> PyResult<Self> {
        if compression_ratio < 1.0 {
            return Err(PyValueError::new_err("compression_ratio must be >= 1.0"));
        }
        if quantization_bits != 8 && quantization_bits != 16 {
            return Err(PyValueError::new_err("quantization_bits must be 8 or 16"));
        }

        let config = CompressionConfig {
            compression_ratio,
            quantization_bits,
            error_feedback,
        };

        Ok(Self {
            inner: GradientCompressor::new(config),
        })
    }

    /// Compress a gradient.
    fn compress(&mut self, gradient: PyReadonlyArray1<f32>) -> PyResult<PyCompressedGradient> {
        let array = gradient.as_array().to_owned();
        let compressed = self
            .inner
            .compress(&array)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyCompressedGradient { inner: compressed })
    }

    /// Decompress a gradient.
    fn decompress<'py>(
        &self,
        py: Python<'py>,
        compressed: &PyCompressedGradient,
    ) -> Bound<'py, PyArray1<f32>> {
        let result = self.inner.decompress(&compressed.inner);
        PyArray1::from_vec(py, result.to_vec())
    }

    /// Reset error accumulator.
    fn reset_error(&mut self) {
        self.inner.reset_error()
    }

    /// Get current accumulated error norm.
    fn error_norm(&self) -> Option<f32> {
        self.inner.error_norm()
    }
}

/// Python wrapper for compressed gradient.
#[pyclass(name = "CompressedGradient")]
pub struct PyCompressedGradient {
    inner: CompressedGradient,
}

#[pymethods]
impl PyCompressedGradient {
    /// Original dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Number of non-zero values.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.indices.len()
    }

    /// Compression ratio achieved.
    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio
    }

    /// Size in bytes.
    #[getter]
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    /// Original size in bytes.
    #[getter]
    fn original_size_bytes(&self) -> usize {
        self.inner.original_size_bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressedGradient(dim={}, nnz={}, ratio={:.1}x)",
            self.inner.dimension,
            self.inner.indices.len(),
            self.inner.compression_ratio
        )
    }
}

// =============================================================================
// PoGQ-v4.1 Enhanced: Byzantine Detection
// =============================================================================

/// Python wrapper for PoGQ-v4.1 configuration.
#[pyclass(name = "PoGQConfig")]
#[derive(Clone)]
pub struct PyPoGQConfig {
    inner: PoGQv41Config,
}

#[pymethods]
impl PyPoGQConfig {
    /// Create a new PoGQ-v4.1 configuration with defaults.
    #[new]
    #[pyo3(signature = (
        warmup_rounds=3,
        ema_beta=0.85,
        hysteresis_k=2,
        hysteresis_m=3,
        conformal_alpha=0.10,
        pca_components=32,
        direction_prefilter=true
    ))]
    fn new(
        warmup_rounds: u32,
        ema_beta: f32,
        hysteresis_k: u32,
        hysteresis_m: u32,
        conformal_alpha: f32,
        pca_components: usize,
        direction_prefilter: bool,
    ) -> Self {
        let mut config = PoGQv41Config::default();
        config.warmup_rounds = warmup_rounds;
        config.ema_beta = ema_beta;
        config.hysteresis_k = hysteresis_k;
        config.hysteresis_m = hysteresis_m;
        config.conformal_alpha = conformal_alpha;
        config.pca_components = pca_components;
        config.direction_prefilter = direction_prefilter;
        Self { inner: config }
    }

    /// Create strict configuration (high precision).
    #[staticmethod]
    fn strict() -> Self {
        let mut config = PoGQv41Config::default();
        config.hysteresis_k = 1;
        config.conformal_alpha = 0.05;
        Self { inner: config }
    }

    /// Create permissive configuration (low false positives).
    #[staticmethod]
    fn permissive() -> Self {
        let mut config = PoGQv41Config::default();
        config.warmup_rounds = 5;
        config.hysteresis_k = 3;
        config.conformal_alpha = 0.15;
        Self { inner: config }
    }

    fn __repr__(&self) -> String {
        format!(
            "PoGQConfig(warmup={}, ema_beta={:.2}, hysteresis_k={}, hysteresis_m={})",
            self.inner.warmup_rounds,
            self.inner.ema_beta,
            self.inner.hysteresis_k,
            self.inner.hysteresis_m
        )
    }
}

/// Result of Byzantine detection for a single gradient.
#[pyclass(name = "DetectionResult")]
#[derive(Clone)]
pub struct PyDetectionResult {
    #[pyo3(get)]
    pub is_byzantine: bool,
    #[pyo3(get)]
    pub is_quarantined: bool,
    #[pyo3(get)]
    pub in_warmup: bool,
    #[pyo3(get)]
    pub ema_score: f32,
    #[pyo3(get)]
    pub threshold: f32,
    #[pyo3(get)]
    pub client_id: String,
    #[pyo3(get)]
    pub round_num: u32,
}

#[pymethods]
impl PyDetectionResult {
    fn __repr__(&self) -> String {
        format!(
            "DetectionResult(client='{}', byzantine={}, quarantined={}, ema={:.3})",
            self.client_id,
            self.is_byzantine,
            self.is_quarantined,
            self.ema_score
        )
    }
}

/// Python wrapper for PoGQ-v4.1 Enhanced Byzantine detector.
///
/// Example:
/// ```python
/// from fl_aggregator import PoGQDetector, PoGQConfig
/// import numpy as np
///
/// # Create detector
/// config = PoGQConfig(warmup_rounds=3, ema_beta=0.85)
/// detector = PoGQDetector(config)
///
/// # Calibrate on clean validation data
/// detector.calibrate(validation_scores, validation_classes)
///
/// # Score a gradient
/// result = detector.score(gradient, "client_1", [0, 1], reference, round_num=1)
/// print(f"Byzantine: {result.is_byzantine}")
/// ```
#[pyclass(name = "PoGQDetector")]
pub struct PyPoGQDetector {
    inner: PoGQv41Enhanced,
}

#[pymethods]
impl PyPoGQDetector {
    /// Create a new PoGQ-v4.1 Enhanced detector.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPoGQConfig>) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: PoGQv41Enhanced::new(config),
        }
    }

    /// Fit PCA on clean reference gradients.
    ///
    /// Args:
    ///     gradients: List of NumPy arrays (reference gradients)
    fn fit_pca(&mut self, gradients: Vec<PyReadonlyArray1<f32>>) {
        let arrays: Vec<_> = gradients
            .iter()
            .map(|g| g.as_array().to_owned())
            .collect();
        self.inner.fit_pca(&arrays);
    }

    /// Calibrate Mondrian conformal thresholds.
    ///
    /// Args:
    ///     scores: Validation scores from clean clients
    ///     class_profiles: List of class ID lists for each score
    fn calibrate(&mut self, scores: Vec<f32>, class_profiles: Vec<Vec<u32>>) -> PyResult<()> {
        if scores.len() != class_profiles.len() {
            return Err(PyValueError::new_err("scores and class_profiles must have same length"));
        }
        let profiles: Vec<_> = class_profiles
            .into_iter()
            .map(|classes| MondrianProfile::new(classes))
            .collect();
        self.inner.calibrate_mondrian(&scores, &profiles);
        Ok(())
    }

    /// Score a client gradient for Byzantine behavior.
    ///
    /// Args:
    ///     gradient: NumPy array of gradient values
    ///     client_id: Unique client identifier
    ///     client_classes: List of class IDs this client has data for
    ///     reference: Reference gradient for comparison
    ///     round_num: Current FL round number
    ///
    /// Returns:
    ///     DetectionResult with Byzantine classification
    fn score(
        &mut self,
        gradient: PyReadonlyArray1<f32>,
        client_id: &str,
        client_classes: Vec<u32>,
        reference: PyReadonlyArray1<f32>,
        round_num: u32,
    ) -> PyDetectionResult {
        let gradient_arr = gradient.as_array().to_owned();
        let reference_arr = reference.as_array().to_owned();
        let classes: BTreeSet<u32> = client_classes.into_iter().collect();

        let result = self.inner.score_gradient(
            &gradient_arr,
            client_id,
            &classes,
            &reference_arr,
            round_num,
        );

        PyDetectionResult {
            is_byzantine: result.detection.is_byzantine,
            is_quarantined: result.detection.quarantined,
            in_warmup: result.phase2.in_warmup,
            ema_score: result.scores.ema,
            threshold: result.detection.threshold,
            client_id: result.client_id,
            round_num: result.round,
        }
    }

    /// Check if a client is currently quarantined.
    fn is_quarantined(&self, client_id: &str) -> bool {
        self.inner.is_quarantined(client_id)
    }

    /// Get list of all quarantined clients.
    fn get_quarantined_clients(&self) -> Vec<String> {
        self.inner.get_quarantined_clients()
    }

    /// Reset detector state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.get_statistics();
        format!(
            "PoGQDetector(clients={}, quarantined={})",
            stats.total_clients,
            stats.quarantined_clients
        )
    }
}

/// Register Python module.
#[pymodule]
fn fl_aggregator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Aggregation
    m.add_class::<PyDefense>()?;
    m.add_class::<PyAggregator>()?;

    // Compression
    m.add_class::<PyGradientCompressor>()?;
    m.add_class::<PyCompressedGradient>()?;

    // PoGQ-v4.1 Byzantine Detection
    m.add_class::<PyPoGQConfig>()?;
    m.add_class::<PyPoGQDetector>()?;
    m.add_class::<PyDetectionResult>()?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
