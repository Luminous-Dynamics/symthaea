// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge Consciousness Probe
//!
//! This module extends the Neural Bridge with consciousness topology analysis
//! capabilities, enabling probing of LLM internal representations for distinct
//! topological structures between phenomenal and functional concepts.
//!
//! ## Research Hypothesis (H1)
//!
//! LLM internal representations for phenomenal concepts (e.g., "what it is like
//! to see red") exhibit different topological properties (Betti numbers, unity score)
//! than functional concepts (e.g., "recursive function").
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────┐     ┌──────────────┐     ┌───────────────────┐
//! │   Concept     │     │    Neural    │     │   Consciousness   │
//! │  "Qualia of   │────>│    Bridge    │────>│    Topology       │
//! │   seeing red" │     │   (16384D)   │     │    Analysis       │
//! └───────────────┘     └──────────────┘     └───────────────────┘
//!                                                      │
//!                                                      v
//!                                            ┌───────────────────┐
//!                                            │  Betti Numbers    │
//!                                            │  Unity Score      │
//!                                            │  Persistence      │
//!                                            └───────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::{ConsciousnessProbe, ConceptCorpus, ConceptClass};
//!
//! // Load probe with trained weights
//! let probe = ConsciousnessProbe::new("models/neural_bridge/probe_weights.npy")?;
//!
//! // Load concept corpora
//! let phenomenal = ConceptCorpus::load("data/consciousness_probe/phenomenal_concepts.json")?;
//! let functional = ConceptCorpus::load("data/consciousness_probe/functional_concepts.json")?;
//!
//! // Run topological comparison
//! let results = probe.compare_classes(&phenomenal, &functional, activations)?;
//!
//! println!("P-value: {}", results.p_value);
//! println!("Cohen's d: {}", results.cohens_d);
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::consciousness_topology::{
    BettiNumbers, ConsciousnessTopology, TopologicalAssessment, TopologyConfig,
};

use super::NeuralBridge;

#[cfg(feature = "neural-bridge")]
use super::neural_bridge_v2::NeuralBridgeV2;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::PackedBipolar;

/// A single concept for probing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Unique identifier
    pub id: String,
    /// The concept text
    pub text: String,
    /// Main category (e.g., "qualia", "computation")
    pub category: String,
    /// Subcategory (e.g., "visual", "algorithms")
    pub subcategory: String,
}

/// Corpus metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusMetadata {
    /// Description of the corpus
    pub description: String,
    /// Version string
    pub version: String,
    /// Hypothesis being tested
    pub hypothesis: String,
}

/// Collection of concepts for probing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCorpus {
    /// Metadata about the corpus
    pub metadata: CorpusMetadata,
    /// The concepts in this corpus
    pub concepts: Vec<Concept>,
}

impl ConceptCorpus {
    /// Load corpus from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open corpus file: {}", path.display()))?;
        let reader = BufReader::new(file);
        let corpus: ConceptCorpus = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse corpus JSON: {}", path.display()))?;
        Ok(corpus)
    }

    /// Number of concepts
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.concepts.is_empty()
    }

    /// Get concept texts
    pub fn texts(&self) -> Vec<&str> {
        self.concepts.iter().map(|c| c.text.as_str()).collect()
    }
}

/// Result from probing a single concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptProbeResult {
    /// The concept that was probed
    pub concept: Concept,
    /// Betti numbers from topology analysis
    pub betti: BettiNumbers,
    /// Unity score (1/β₀)
    pub unity_score: f64,
    /// Circularity measure
    pub circularity: f64,
    /// Completeness measure
    pub completeness: f64,
    /// Overall topology quality
    pub quality: f64,
}

/// Statistics for a class of concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassStatistics {
    /// Mean unity score
    pub mean_unity: f64,
    /// Standard deviation of unity
    pub std_unity: f64,
    /// Mean β₀ (components)
    pub mean_beta_0: f64,
    /// Mean β₁ (cycles)
    pub mean_beta_1: f64,
    /// Mean β₂ (voids)
    pub mean_beta_2: f64,
    /// Mean quality
    pub mean_quality: f64,
    /// Number of samples
    pub n: usize,
}

impl ClassStatistics {
    /// Compute statistics from probe results
    pub fn from_results(results: &[ConceptProbeResult]) -> Self {
        let n = results.len();
        if n == 0 {
            return Self {
                mean_unity: 0.0,
                std_unity: 0.0,
                mean_beta_0: 0.0,
                mean_beta_1: 0.0,
                mean_beta_2: 0.0,
                mean_quality: 0.0,
                n: 0,
            };
        }

        let sum_unity: f64 = results.iter().map(|r| r.unity_score).sum();
        let mean_unity = sum_unity / n as f64;

        let variance: f64 = results
            .iter()
            .map(|r| (r.unity_score - mean_unity).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_unity = variance.sqrt();

        let mean_beta_0 = results.iter().map(|r| r.betti.beta_0 as f64).sum::<f64>() / n as f64;
        let mean_beta_1 = results.iter().map(|r| r.betti.beta_1 as f64).sum::<f64>() / n as f64;
        let mean_beta_2 = results.iter().map(|r| r.betti.beta_2 as f64).sum::<f64>() / n as f64;
        let mean_quality = results.iter().map(|r| r.quality).sum::<f64>() / n as f64;

        Self {
            mean_unity,
            std_unity,
            mean_beta_0,
            mean_beta_1,
            mean_beta_2,
            mean_quality,
            n,
        }
    }
}

/// Result from comparing two concept classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassComparisonResult {
    /// Statistics for phenomenal concepts
    pub phenomenal_stats: ClassStatistics,
    /// Statistics for functional concepts
    pub functional_stats: ClassStatistics,
    /// P-value from permutation test
    pub p_value: f64,
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Number of permutations used
    pub n_permutations: usize,
    /// Is the difference statistically significant (p < 0.05)?
    pub is_significant: bool,
    /// Observed difference in mean unity
    pub observed_difference: f64,
}

/// Configuration for the consciousness probe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Topology analysis configuration
    pub topology_config: TopologyConfig,
    /// Number of permutations for statistical test
    pub n_permutations: usize,
    /// Scale for topology analysis
    pub analysis_scale: f64,
    /// Minimum number of states for meaningful topology
    pub min_states: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            topology_config: TopologyConfig::default(),
            n_permutations: 10_000,
            analysis_scale: 0.5,
            min_states: 5,
        }
    }
}

/// Consciousness Probe for analyzing topological properties of LLM representations
///
/// This probe extends the Neural Bridge to perform topological data analysis
/// on projected HDC vectors, testing whether phenomenal concepts exhibit
/// distinct geometric structure compared to functional concepts.
pub struct ConsciousnessProbe {
    /// The underlying neural bridge for LLM → HDC projection
    bridge: Option<NeuralBridge>,
    /// Configuration
    config: ProbeConfig,
}

impl ConsciousnessProbe {
    /// Create a new consciousness probe with trained weights
    ///
    /// # Arguments
    /// * `weights_path` - Path to the .npy weights file for NeuralBridge
    pub fn new<P: AsRef<Path>>(weights_path: P) -> Result<Self> {
        let bridge = NeuralBridge::load(weights_path)?;
        Ok(Self {
            bridge: Some(bridge),
            config: ProbeConfig::default(),
        })
    }

    /// Create a probe without pre-loaded weights (for testing or direct HV input)
    pub fn new_without_bridge() -> Self {
        Self {
            bridge: None,
            config: ProbeConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: ProbeConfig) -> Self {
        self.config = config;
        self
    }

    /// Probe a single concept using its LLM activation
    ///
    /// Projects the activation to HDC space and performs topology analysis.
    ///
    /// # Arguments
    /// * `concept` - The concept being probed
    /// * `activation` - LLM hidden state activation vector
    pub fn probe_concept(
        &self,
        concept: &Concept,
        activation: &[f32],
    ) -> Result<ConceptProbeResult> {
        let bridge = self
            .bridge
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Neural bridge not initialized"))?;

        // Project to HDC space
        let hdc_vec = bridge.project_to_packed(activation)?;

        // Convert to BinaryHV for topology analysis
        let hv = packed_bipolar_to_hv16(&hdc_vec);

        self.probe_concept_from_hv(concept, &hv)
    }

    /// Probe a concept from a pre-computed BinaryHV vector
    pub fn probe_concept_from_hv(
        &self,
        concept: &Concept,
        hv: &BinaryHV,
    ) -> Result<ConceptProbeResult> {
        // For single-concept analysis, we create variations via permutation
        // to build a point cloud for topology analysis
        let mut topology = ConsciousnessTopology::new(self.config.topology_config.clone());

        // Add the original state
        topology.add_state(*hv);

        // Add permuted variations to create a point cloud
        for shift in 1..self.config.min_states {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }

        let assessment = topology.analyze(self.config.analysis_scale);

        Ok(ConceptProbeResult {
            concept: concept.clone(),
            betti: assessment.betti,
            unity_score: assessment.unity_score,
            circularity: assessment.circularity,
            completeness: assessment.completeness,
            quality: assessment.quality,
        })
    }

    /// Probe an entire corpus of concepts
    ///
    /// # Arguments
    /// * `corpus` - The concept corpus to probe
    /// * `activations` - Map from concept ID to LLM activation vector
    pub fn probe_corpus(
        &self,
        corpus: &ConceptCorpus,
        activations: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<ConceptProbeResult>> {
        let mut results = Vec::with_capacity(corpus.len());

        for concept in &corpus.concepts {
            if let Some(activation) = activations.get(&concept.id) {
                let result = self.probe_concept(concept, activation)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Probe corpus using pre-computed HDC vectors
    pub fn probe_corpus_from_hvs(
        &self,
        corpus: &ConceptCorpus,
        hvs: &HashMap<String, BinaryHV>,
    ) -> Result<Vec<ConceptProbeResult>> {
        let mut results = Vec::with_capacity(corpus.len());

        for concept in &corpus.concepts {
            if let Some(hv) = hvs.get(&concept.id) {
                let result = self.probe_concept_from_hv(concept, hv)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Compare topological properties between two concept classes
    ///
    /// Performs statistical comparison using permutation test and computes
    /// effect size (Cohen's d).
    ///
    /// # Arguments
    /// * `phenomenal` - Results from probing phenomenal concepts
    /// * `functional` - Results from probing functional concepts
    pub fn compare_classes(
        &self,
        phenomenal: &[ConceptProbeResult],
        functional: &[ConceptProbeResult],
    ) -> ClassComparisonResult {
        let phen_stats = ClassStatistics::from_results(phenomenal);
        let func_stats = ClassStatistics::from_results(functional);

        // Observed difference in mean unity scores
        let observed_diff = phen_stats.mean_unity - func_stats.mean_unity;

        // Compute Cohen's d
        let pooled_std =
            ((phen_stats.std_unity.powi(2) + func_stats.std_unity.powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_std > 0.0 {
            observed_diff / pooled_std
        } else {
            0.0
        };

        // Permutation test
        let all_unity: Vec<f64> = phenomenal
            .iter()
            .chain(functional.iter())
            .map(|r| r.unity_score)
            .collect();

        let n_phen = phenomenal.len();
        let mut count_extreme = 0;

        // Use deterministic pseudo-random for reproducibility
        let mut rng_state: u64 = 42;

        for _ in 0..self.config.n_permutations {
            // Fisher-Yates shuffle with simple LCG
            let mut permuted = all_unity.clone();
            for i in (1..permuted.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                permuted.swap(i, j);
            }

            // Split and compute difference
            let perm_phen_mean: f64 = permuted[..n_phen].iter().sum::<f64>() / n_phen as f64;
            let perm_func_mean: f64 =
                permuted[n_phen..].iter().sum::<f64>() / (all_unity.len() - n_phen) as f64;
            let perm_diff = perm_phen_mean - perm_func_mean;

            if perm_diff.abs() >= observed_diff.abs() {
                count_extreme += 1;
            }
        }

        let p_value = (count_extreme as f64 + 1.0) / (self.config.n_permutations as f64 + 1.0);

        ClassComparisonResult {
            phenomenal_stats: phen_stats,
            functional_stats: func_stats,
            p_value,
            cohens_d,
            n_permutations: self.config.n_permutations,
            is_significant: p_value < 0.05,
            observed_difference: observed_diff,
        }
    }

    /// Perform cross-validation comparison
    ///
    /// Splits each corpus into training/test sets to verify generalization.
    ///
    /// # Arguments
    /// * `phenomenal` - Full phenomenal results
    /// * `functional` - Full functional results
    /// * `test_fraction` - Fraction to use for testing (e.g., 0.2)
    pub fn cross_validate(
        &self,
        phenomenal: &[ConceptProbeResult],
        functional: &[ConceptProbeResult],
        test_fraction: f64,
    ) -> (ClassComparisonResult, ClassComparisonResult) {
        let n_phen_test = (phenomenal.len() as f64 * test_fraction).ceil() as usize;
        let n_func_test = (functional.len() as f64 * test_fraction).ceil() as usize;

        // Split (deterministic for reproducibility)
        let phen_train = &phenomenal[n_phen_test..];
        let phen_test = &phenomenal[..n_phen_test];
        let func_train = &functional[n_func_test..];
        let func_test = &functional[..n_func_test];

        let train_result = self.compare_classes(phen_train, func_train);
        let test_result = self.compare_classes(phen_test, func_test);

        (train_result, test_result)
    }

    /// Compute Wasserstein distance between persistence diagrams
    ///
    /// This provides a more sophisticated comparison of topological structure
    /// than just comparing Betti numbers.
    pub fn wasserstein_distance(
        &self,
        assessment_a: &TopologicalAssessment,
        assessment_b: &TopologicalAssessment,
    ) -> f64 {
        // Simplified Wasserstein distance based on persistence values
        // Full implementation would use optimal transport

        let mut total_distance = 0.0;

        // Compare feature counts at each dimension
        let diff_0 = (assessment_a.betti.beta_0 as f64 - assessment_b.betti.beta_0 as f64).abs();
        let diff_1 = (assessment_a.betti.beta_1 as f64 - assessment_b.betti.beta_1 as f64).abs();
        let diff_2 = (assessment_a.betti.beta_2 as f64 - assessment_b.betti.beta_2 as f64).abs();

        total_distance += diff_0 + diff_1 + diff_2;

        // Add persistence-weighted component
        let avg_persistence_a: f64 = if assessment_a.features.is_empty() {
            0.0
        } else {
            assessment_a
                .features
                .iter()
                .map(|f| f.persistence)
                .sum::<f64>()
                / assessment_a.features.len() as f64
        };

        let avg_persistence_b: f64 = if assessment_b.features.is_empty() {
            0.0
        } else {
            assessment_b
                .features
                .iter()
                .map(|f| f.persistence)
                .sum::<f64>()
                / assessment_b.features.len() as f64
        };

        total_distance += (avg_persistence_a - avg_persistence_b).abs();

        total_distance
    }
}

/// Convert PackedBipolar to BinaryHV
///
/// This is needed because PackedBipolar and BinaryHV have different internal
/// representations but represent the same semantic content.
fn packed_bipolar_to_hv16(packed: &symthaea_core::hdc::PackedBipolar) -> BinaryHV {
    // PackedBipolar stores bipolar values (-1, +1), BinaryHV stores binary (0, 1)
    // Convert via the to_bipolar() method and then to BinaryHV
    let bipolar = packed.to_bipolar();

    // Convert i8 bipolar (-1, +1) to f32 for BinaryHV::from_bipolar
    let bipolar_f32: Vec<f32> = bipolar.iter().map(|&v| v as f32).collect();

    BinaryHV::from_bipolar(&bipolar_f32)
}

// =============================================================================
// ConsciousnessProbeV2 - Direct text-to-topology analysis via BGE-M3
// =============================================================================

/// Consciousness Probe V2 with integrated BGE-M3 embedding model
///
/// This version includes the full text→HDC pipeline, allowing direct
/// probing of concepts from text without pre-extracted activations.
///
/// ## Usage
///
/// ```rust,ignore
/// use symthaea::perception::ConsciousnessProbeV2;
///
/// // Load with default BGE-M3 + probe weights
/// let mut probe = ConsciousnessProbeV2::load_default()?;
///
/// // Probe a concept directly from text
/// let result = probe.probe_text("The subjective experience of seeing red")?;
/// println!("Unity score: {}", result.unity_score);
///
/// // Probe entire corpus
/// let corpus = ConceptCorpus::load("data/consciousness_probe/phenomenal_concepts.json")?;
/// let results = probe.probe_corpus_texts(&corpus)?;
/// ```
#[cfg(feature = "neural-bridge")]
pub struct ConsciousnessProbeV2 {
    /// The underlying neural bridge for text → HDC
    bridge: NeuralBridgeV2,
    /// Configuration
    config: ProbeConfig,
}

#[cfg(feature = "neural-bridge")]
impl ConsciousnessProbeV2 {
    /// Load with default BGE-M3 model and probe weights
    pub fn load_default() -> Result<Self> {
        let bridge = NeuralBridgeV2::load_default()?;
        Ok(Self {
            bridge,
            config: ProbeConfig::default(),
        })
    }

    /// Load with custom probe weights path
    pub fn load_with_probe<P: AsRef<Path>>(probe_path: P) -> Result<Self> {
        use super::neural_bridge_v2::NeuralBridgeV2Builder;

        let path_str = probe_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("probe path contains non-UTF8 characters"))?;

        let bridge = NeuralBridgeV2Builder::new().probe_path(path_str).build()?;

        Ok(Self {
            bridge,
            config: ProbeConfig::default(),
        })
    }

    /// Create with existing NeuralBridgeV2 instance
    pub fn with_bridge(bridge: NeuralBridgeV2) -> Self {
        Self {
            bridge,
            config: ProbeConfig::default(),
        }
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: ProbeConfig) -> Self {
        self.config = config;
        self
    }

    /// Probe a concept directly from its text
    ///
    /// This is the main entry point for text-based probing.
    /// The text is encoded via BGE-M3 → Linear Probe → HDC → Topology.
    pub fn probe_text(&mut self, text: &str) -> Result<ConceptProbeResult> {
        let concept = Concept {
            id: format!("text_{:016x}", hash_text(text)),
            text: text.to_string(),
            category: "unknown".to_string(),
            subcategory: "unknown".to_string(),
        };

        self.probe_concept_text(&concept)
    }

    /// Probe a concept with its metadata
    pub fn probe_concept_text(&mut self, concept: &Concept) -> Result<ConceptProbeResult> {
        // Encode text to HDC via BGE-M3 + probe
        let packed = self.bridge.encode_to_hdc(&concept.text)?;

        // Convert to BinaryHV for topology analysis
        let hv = packed_to_hv16(&packed);

        // Analyze topology
        self.probe_concept_from_hv(concept, &hv)
    }

    /// Probe a concept from a pre-computed BinaryHV vector
    pub fn probe_concept_from_hv(
        &self,
        concept: &Concept,
        hv: &BinaryHV,
    ) -> Result<ConceptProbeResult> {
        let mut topology = ConsciousnessTopology::new(self.config.topology_config.clone());

        // Add the original state
        topology.add_state(*hv);

        // Add permuted variations to create a point cloud
        for shift in 1..self.config.min_states {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }

        let assessment = topology.analyze(self.config.analysis_scale);

        Ok(ConceptProbeResult {
            concept: concept.clone(),
            betti: assessment.betti,
            unity_score: assessment.unity_score,
            circularity: assessment.circularity,
            completeness: assessment.completeness,
            quality: assessment.quality,
        })
    }

    /// Convert a concept text to BinaryHV via BGE-M3 + probe projection
    ///
    /// This is useful for operations that need raw HDC vectors, such as
    /// binding/bundling experiments (H2 hypothesis testing).
    pub fn concept_to_hv(&mut self, text: &str) -> Result<BinaryHV> {
        let packed = self.bridge.encode_to_hdc(text)?;
        Ok(packed_to_hv16(&packed))
    }

    /// Probe an entire corpus of concepts from their text
    ///
    /// This is the main batch processing entry point.
    pub fn probe_corpus_texts(
        &mut self,
        corpus: &ConceptCorpus,
    ) -> Result<Vec<ConceptProbeResult>> {
        let mut results = Vec::with_capacity(corpus.len());

        for concept in &corpus.concepts {
            let result = self.probe_concept_text(concept)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compare two concept classes from their text
    ///
    /// Convenience method that loads, probes, and compares in one call.
    pub fn compare_corpora(
        &mut self,
        phenomenal: &ConceptCorpus,
        functional: &ConceptCorpus,
    ) -> Result<ClassComparisonResult> {
        let phen_results = self.probe_corpus_texts(phenomenal)?;
        let func_results = self.probe_corpus_texts(functional)?;

        Ok(self.compare_classes(&phen_results, &func_results))
    }

    /// Compare topological properties between two concept classes
    pub fn compare_classes(
        &self,
        phenomenal: &[ConceptProbeResult],
        functional: &[ConceptProbeResult],
    ) -> ClassComparisonResult {
        let phen_stats = ClassStatistics::from_results(phenomenal);
        let func_stats = ClassStatistics::from_results(functional);

        let observed_diff = phen_stats.mean_unity - func_stats.mean_unity;

        let pooled_std =
            ((phen_stats.std_unity.powi(2) + func_stats.std_unity.powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_std > 0.0 {
            observed_diff / pooled_std
        } else {
            0.0
        };

        // Permutation test
        let all_unity: Vec<f64> = phenomenal
            .iter()
            .chain(functional.iter())
            .map(|r| r.unity_score)
            .collect();

        let n_phen = phenomenal.len();
        let mut count_extreme = 0;
        let mut rng_state: u64 = 42;

        for _ in 0..self.config.n_permutations {
            let mut permuted = all_unity.clone();
            for i in (1..permuted.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                permuted.swap(i, j);
            }

            let perm_phen_mean: f64 = permuted[..n_phen].iter().sum::<f64>() / n_phen as f64;
            let perm_func_mean: f64 =
                permuted[n_phen..].iter().sum::<f64>() / (all_unity.len() - n_phen) as f64;
            let perm_diff = perm_phen_mean - perm_func_mean;

            if perm_diff.abs() >= observed_diff.abs() {
                count_extreme += 1;
            }
        }

        let p_value = (count_extreme as f64 + 1.0) / (self.config.n_permutations as f64 + 1.0);

        ClassComparisonResult {
            phenomenal_stats: phen_stats,
            functional_stats: func_stats,
            p_value,
            cohens_d,
            n_permutations: self.config.n_permutations,
            is_significant: p_value < 0.05,
            observed_difference: observed_diff,
        }
    }

    /// Get reference to underlying bridge for advanced use
    pub fn bridge(&self) -> &NeuralBridgeV2 {
        &self.bridge
    }

    /// Get mutable reference to underlying bridge
    pub fn bridge_mut(&mut self) -> &mut NeuralBridgeV2 {
        &mut self.bridge
    }

    /// Get bridge statistics (cache hit rates, latency, etc.)
    pub fn stats(&self) -> &super::neural_bridge_v2::BridgeStats {
        self.bridge.stats()
    }

    /// Clear embedding and HDC caches
    pub fn clear_cache(&mut self) {
        self.bridge.clear_cache();
    }
}

/// Convert PackedBipolar to BinaryHV (feature-gated version)
#[cfg(feature = "neural-bridge")]
fn packed_to_hv16(packed: &PackedBipolar) -> BinaryHV {
    let bipolar = packed.to_bipolar();
    let bipolar_f32: Vec<f32> = bipolar.iter().map(|&v| v as f32).collect();
    BinaryHV::from_bipolar(&bipolar_f32)
}

/// Simple text hash for generating concept IDs
#[cfg(feature = "neural-bridge")]
fn hash_text(text: &str) -> u64 {
    text.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_concept(id: &str, text: &str) -> Concept {
        Concept {
            id: id.to_string(),
            text: text.to_string(),
            category: "test".to_string(),
            subcategory: "test".to_string(),
        }
    }

    #[test]
    fn test_class_statistics() {
        let results = vec![
            ConceptProbeResult {
                concept: make_test_concept("1", "test1"),
                betti: BettiNumbers::new(1, 0, 0),
                unity_score: 1.0,
                circularity: 0.0,
                completeness: 1.0,
                quality: 1.0,
            },
            ConceptProbeResult {
                concept: make_test_concept("2", "test2"),
                betti: BettiNumbers::new(2, 1, 0),
                unity_score: 0.5,
                circularity: 0.1,
                completeness: 0.9,
                quality: 0.8,
            },
        ];

        let stats = ClassStatistics::from_results(&results);
        assert_eq!(stats.n, 2);
        assert!((stats.mean_unity - 0.75).abs() < 0.001);
        assert!(stats.mean_beta_0 > 0.0);
    }

    #[test]
    fn test_probe_without_bridge() {
        let probe = ConsciousnessProbe::new_without_bridge();
        let concept = make_test_concept("test", "The subjective experience of seeing red");
        let hv = BinaryHV::random(42);

        let result = probe.probe_concept_from_hv(&concept, &hv).unwrap();
        assert!(result.unity_score > 0.0);
        assert!(result.unity_score <= 1.0);
    }

    #[test]
    fn test_compare_classes() {
        let probe = ConsciousnessProbe::new_without_bridge().with_config(ProbeConfig {
            n_permutations: 100, // Fewer for testing
            ..Default::default()
        });

        // Create mock results with different distributions
        let phenomenal: Vec<ConceptProbeResult> = (0..10)
            .map(|i| {
                let hv = BinaryHV::random(1000 + i);
                let mut result = probe
                    .probe_concept_from_hv(
                        &make_test_concept(&format!("phen_{}", i), "phenomenal"),
                        &hv,
                    )
                    .unwrap();
                // Artificially increase unity for phenomenal
                result.unity_score = 0.8 + (i as f64 * 0.01);
                result
            })
            .collect();

        let functional: Vec<ConceptProbeResult> = (0..10)
            .map(|i| {
                let hv = BinaryHV::random(2000 + i);
                let mut result = probe
                    .probe_concept_from_hv(
                        &make_test_concept(&format!("func_{}", i), "functional"),
                        &hv,
                    )
                    .unwrap();
                // Artificially decrease unity for functional
                result.unity_score = 0.4 + (i as f64 * 0.01);
                result
            })
            .collect();

        let comparison = probe.compare_classes(&phenomenal, &functional);

        assert!(comparison.phenomenal_stats.mean_unity > comparison.functional_stats.mean_unity);
        assert!(comparison.observed_difference > 0.0);
        // With artificial difference, should be significant
        assert!(comparison.cohens_d.abs() > 0.5);
    }

    #[test]
    fn test_corpus_serialization() {
        let corpus = ConceptCorpus {
            metadata: CorpusMetadata {
                description: "Test corpus".to_string(),
                version: "1.0".to_string(),
                hypothesis: "Test hypothesis".to_string(),
            },
            concepts: vec![
                make_test_concept("1", "Concept 1"),
                make_test_concept("2", "Concept 2"),
            ],
        };

        let json = serde_json::to_string(&corpus).unwrap();
        let restored: ConceptCorpus = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.concepts.len(), 2);
        assert_eq!(restored.metadata.version, "1.0");
    }
}
