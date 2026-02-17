//! WebAssembly bindings for browser-based federated learning.
//!
//! This module provides WASM bindings that enable running federated learning
//! aggregation directly in the browser, enabling fully decentralized FL
//! without requiring a server.
//!
//! # Features
//!
//! - **WasmAggregator**: Byzantine-resistant gradient aggregation in the browser
//! - **WasmPhiMeasurer**: Phi measurement for coherence tracking
//! - **WasmHyperFeel**: HyperFeel encoding/decoding for gradient compression
//! - **WasmShapley**: Shapley value calculation for fair attribution
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { WasmAggregator, WasmDefense } from 'fl-aggregator';
//!
//! await init();
//!
//! const aggregator = new WasmAggregator(WasmDefense.Median, 5);
//! aggregator.submit("node1", new Float32Array([1.0, 2.0, 3.0]));
//! aggregator.submit("node2", new Float32Array([1.1, 2.1, 3.1]));
//!
//! if (aggregator.isRoundComplete()) {
//!     const result = aggregator.finalizeRound();
//!     console.log("Aggregated gradient:", result);
//! }
//! ```

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::byzantine::{ByzantineAggregator, Defense};
use crate::phi::{PhiConfig, PhiMeasurer};
use crate::hyperfeel::{encode_gradient as hf_encode, decode_gradient as hf_decode, EncodingConfig};
use crate::shapley::{ShapleyCalculator, ShapleyConfig, SamplingMethod};
use crate::payload::{HYPERVECTOR_DIM, Hypervector};

/// Defense types available in WASM.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum WasmDefense {
    FedAvg,
    Median,
    TrimmedMean,
    Krum,
    MultiKrum,
}

impl From<WasmDefense> for Defense {
    fn from(d: WasmDefense) -> Self {
        match d {
            WasmDefense::FedAvg => Defense::FedAvg,
            WasmDefense::Median => Defense::Median,
            WasmDefense::TrimmedMean => Defense::TrimmedMean { beta: 0.1 },
            WasmDefense::Krum => Defense::Krum { f: 1 },
            WasmDefense::MultiKrum => Defense::MultiKrum { f: 1, k: 3 },
        }
    }
}

/// WebAssembly-compatible federated learning aggregator.
///
/// Provides Byzantine-resistant gradient aggregation that runs entirely
/// in the browser.
#[wasm_bindgen]
pub struct WasmAggregator {
    aggregator: ByzantineAggregator,
    expected_nodes: usize,
}

#[wasm_bindgen]
impl WasmAggregator {
    /// Create a new WASM aggregator.
    ///
    /// # Arguments
    ///
    /// * `defense` - The Byzantine defense strategy to use
    /// * `expected_nodes` - Number of nodes expected to participate
    #[wasm_bindgen(constructor)]
    pub fn new(defense: WasmDefense, expected_nodes: usize) -> Self {
        Self {
            aggregator: ByzantineAggregator::new(defense.into()),
            expected_nodes,
        }
    }

    /// Submit a gradient from a node.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Unique identifier for the node
    /// * `gradient` - Float32Array of gradient values
    #[wasm_bindgen]
    pub fn submit(&mut self, node_id: &str, gradient: &[f32]) -> Result<(), JsError> {
        let arr = ndarray::Array1::from(gradient.to_vec());
        self.aggregator
            .submit(node_id.to_string(), arr)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Check if the round has enough submissions.
    #[wasm_bindgen(js_name = isRoundComplete)]
    pub fn is_round_complete(&self) -> bool {
        self.aggregator.submission_count() >= self.expected_nodes
    }

    /// Get the current number of submissions.
    #[wasm_bindgen(js_name = submissionCount)]
    pub fn submission_count(&self) -> usize {
        self.aggregator.submission_count()
    }

    /// Finalize the round and get the aggregated gradient.
    ///
    /// Returns a Float32Array with the aggregated gradient values.
    #[wasm_bindgen(js_name = finalizeRound)]
    pub fn finalize_round(&mut self) -> Result<Vec<f32>, JsError> {
        let result = self.aggregator
            .aggregate()
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(result.to_vec())
    }

    /// Reset for a new round.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.aggregator.clear();
    }

    /// Get list of detected Byzantine nodes (if any).
    #[wasm_bindgen(js_name = getByzantineNodes)]
    pub fn get_byzantine_nodes(&self) -> Vec<String> {
        self.aggregator.get_excluded_nodes()
    }
}

/// WebAssembly-compatible Phi measurer for coherence tracking.
#[wasm_bindgen]
pub struct WasmPhiMeasurer {
    measurer: PhiMeasurer,
}

#[wasm_bindgen]
impl WasmPhiMeasurer {
    /// Create a new Phi measurer.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Hypervector dimension (default 2048)
    /// * `seed` - Random seed for reproducibility
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: Option<usize>, seed: Option<u64>) -> Self {
        let config = PhiConfig::default()
            .with_dimension(dimension.unwrap_or(2048))
            .with_seed(seed.unwrap_or(42));
        Self {
            measurer: PhiMeasurer::new(config),
        }
    }

    /// Measure Phi for a set of layer gradients.
    ///
    /// # Arguments
    ///
    /// * `layers` - Array of Float32Arrays, one per layer
    #[wasm_bindgen(js_name = measurePhi)]
    pub fn measure_phi(&mut self, layers: Vec<js_sys::Float32Array>) -> f32 {
        let layer_vecs: Vec<Vec<f32>> = layers
            .iter()
            .map(|arr| arr.to_vec())
            .collect();
        self.measurer.measure_gradient_phi(&layer_vecs)
    }

    /// Detect Byzantine nodes via Phi degradation.
    ///
    /// Returns a JsValue containing { byzantineNodes: string[], systemPhi: number }
    #[wasm_bindgen(js_name = detectByzantine)]
    pub fn detect_byzantine(
        &mut self,
        gradients_json: &str,
        threshold: f32,
    ) -> Result<JsValue, JsError> {
        let gradients: HashMap<String, Vec<f32>> = serde_json::from_str(gradients_json)
            .map_err(|e| JsError::new(&format!("JSON parse error: {}", e)))?;

        let (byzantine, phi) = self.measurer.detect_byzantine_by_phi(&gradients, threshold);

        let result = PhiByzantineResultJs {
            byzantine_nodes: byzantine,
            system_phi: phi,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(Serialize)]
struct PhiByzantineResultJs {
    byzantine_nodes: Vec<String>,
    system_phi: f32,
}

/// WebAssembly-compatible HyperFeel encoder/decoder.
///
/// Provides ~2000x gradient compression using hyperdimensional computing.
#[wasm_bindgen]
pub struct WasmHyperFeel {
    seed: i64,
}

#[wasm_bindgen]
impl WasmHyperFeel {
    /// Create a new HyperFeel encoder.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for projection matrix (must match on all nodes)
    #[wasm_bindgen(constructor)]
    pub fn new(seed: Option<i64>) -> Self {
        Self {
            seed: seed.unwrap_or(42),
        }
    }

    /// Encode a gradient to a hypervector.
    ///
    /// Returns Int8Array of compressed hypervector (16KB).
    #[wasm_bindgen]
    pub fn encode(&self, gradient: &[f32]) -> Result<Vec<i8>, JsError> {
        let arr = ndarray::Array1::from(gradient.to_vec());
        let config = EncodingConfig::default().with_seed(self.seed);
        let hv = hf_encode(&arr, &config);
        Ok(hv.values.to_vec())
    }

    /// Decode a hypervector back to gradient (approximate).
    ///
    /// Returns Float32Array of reconstructed gradient.
    #[wasm_bindgen]
    pub fn decode(&self, hypervector: &[i8], original_dimension: usize) -> Result<Vec<f32>, JsError> {
        let hv = Hypervector {
            values: ndarray::Array1::from(hypervector.to_vec()),
        };
        let config = EncodingConfig::default().with_seed(self.seed);
        let gradient = hf_decode(&hv, original_dimension, &config);
        Ok(gradient.to_vec())
    }

    /// Get the compression ratio for a given gradient dimension.
    #[wasm_bindgen(js_name = compressionRatio)]
    pub fn compression_ratio(&self, gradient_dimension: usize) -> f32 {
        let original_bytes = gradient_dimension * 4; // f32 = 4 bytes
        let compressed_bytes = HYPERVECTOR_DIM; // i8 = 1 byte
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Get the hypervector dimension.
    #[wasm_bindgen(js_name = hypervectorDimension)]
    pub fn hypervector_dimension() -> usize {
        HYPERVECTOR_DIM
    }
}

/// WebAssembly-compatible Shapley value calculator.
#[wasm_bindgen]
pub struct WasmShapley {
    calculator: ShapleyCalculator,
}

#[wasm_bindgen]
impl WasmShapley {
    /// Create a new Shapley calculator.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for Monte Carlo sampling
    #[wasm_bindgen(constructor)]
    pub fn new(seed: Option<u64>) -> Self {
        let config = ShapleyConfig::default();
        Self {
            calculator: ShapleyCalculator::new(config, seed),
        }
    }

    /// Calculate Shapley values for all nodes.
    ///
    /// # Arguments
    ///
    /// * `gradients_json` - JSON object mapping node_id to gradient array
    /// * `aggregated` - The aggregated gradient (target)
    ///
    /// Returns JSON object with node_id -> shapley_value
    #[wasm_bindgen(js_name = calculateValues)]
    pub fn calculate_values(
        &mut self,
        gradients_json: &str,
        aggregated: &[f32],
    ) -> Result<JsValue, JsError> {
        let gradients: HashMap<String, Vec<f32>> = serde_json::from_str(gradients_json)
            .map_err(|e| JsError::new(&format!("JSON parse error: {}", e)))?;

        let gradient_arrays: HashMap<String, ndarray::Array1<f32>> = gradients
            .into_iter()
            .map(|(k, v)| (k, ndarray::Array1::from(v)))
            .collect();

        let aggregated_arr = ndarray::Array1::from(aggregated.to_vec());

        let result = self.calculator.calculate_values(&gradient_arrays, &aggregated_arr);

        let values: HashMap<String, f32> = result.values;

        serde_wasm_bindgen::to_value(&values)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

/// Initialize console logging for WASM (call once at startup).
#[wasm_bindgen(js_name = initLogging)]
pub fn init_logging() {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Utility: Calculate cosine similarity between two vectors.
#[wasm_bindgen(js_name = cosineSimilarity)]
pub fn cosine_similarity_wasm(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Utility: Calculate L2 norm of a vector.
#[wasm_bindgen(js_name = l2Norm)]
pub fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Batch aggregation for multiple rounds (more efficient than individual calls).
#[wasm_bindgen]
pub struct WasmBatchAggregator {
    defense: Defense,
    results: Vec<Vec<f32>>,
}

#[wasm_bindgen]
impl WasmBatchAggregator {
    /// Create a batch aggregator.
    #[wasm_bindgen(constructor)]
    pub fn new(defense: WasmDefense) -> Self {
        Self {
            defense: defense.into(),
            results: Vec::new(),
        }
    }

    /// Process a batch of rounds.
    ///
    /// # Arguments
    ///
    /// * `rounds_json` - JSON array of rounds, each round is { gradients: { node_id: [...] } }
    #[wasm_bindgen(js_name = processBatch)]
    pub fn process_batch(&mut self, rounds_json: &str) -> Result<JsValue, JsError> {
        let rounds: Vec<HashMap<String, Vec<f32>>> = serde_json::from_str(rounds_json)
            .map_err(|e| JsError::new(&format!("JSON parse error: {}", e)))?;

        self.results.clear();

        for round_gradients in rounds {
            let mut aggregator = ByzantineAggregator::new(self.defense.clone());

            for (node_id, gradient) in round_gradients {
                let arr = ndarray::Array1::from(gradient);
                aggregator.submit(node_id, arr)
                    .map_err(|e| JsError::new(&e.to_string()))?;
            }

            let result = aggregator.aggregate()
                .map_err(|e| JsError::new(&e.to_string()))?;

            self.results.push(result.to_vec());
        }

        serde_wasm_bindgen::to_value(&self.results)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the number of processed rounds.
    #[wasm_bindgen(js_name = roundCount)]
    pub fn round_count(&self) -> usize {
        self.results.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_aggregator_basic() {
        let mut agg = WasmAggregator::new(WasmDefense::FedAvg, 2);

        agg.submit("node1", &[1.0, 2.0, 3.0]).unwrap();
        agg.submit("node2", &[2.0, 3.0, 4.0]).unwrap();

        assert!(agg.is_round_complete());
        assert_eq!(agg.submission_count(), 2);

        let result = agg.finalize_round().unwrap();
        assert_eq!(result.len(), 3);
        // FedAvg should give mean: [1.5, 2.5, 3.5]
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 2.5).abs() < 0.01);
        assert!((result[2] - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_wasm_aggregator_median() {
        let mut agg = WasmAggregator::new(WasmDefense::Median, 3);

        agg.submit("node1", &[1.0, 2.0]).unwrap();
        agg.submit("node2", &[2.0, 3.0]).unwrap();
        agg.submit("node3", &[100.0, 100.0]).unwrap(); // Byzantine

        let result = agg.finalize_round().unwrap();
        // Median should ignore the outlier
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_wasm_phi_measurer() {
        let mut measurer = WasmPhiMeasurer::new(Some(512), Some(42));

        // Can't test measure_phi directly without js_sys in non-WASM env
        // but we can test construction
        assert!(true);
    }

    #[test]
    fn test_wasm_hyperfeel() {
        let hf = WasmHyperFeel::new(Some(42));

        let gradient = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let encoded = hf.encode(&gradient).unwrap();

        assert_eq!(encoded.len(), HYPERVECTOR_DIM);

        let ratio = hf.compression_ratio(1_000_000);
        assert!(ratio > 200.0); // Should be ~244x for 1M params
    }

    #[test]
    fn test_cosine_similarity_wasm() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity_wasm(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity_wasm(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
