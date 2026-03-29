// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenal Binding Study via HDC Operations
//!
//! This module implements Research Direction 2: Testing whether HDC binding (XOR)
//! produces representations with higher topological integration than bundling
//! (majority vote), specifically for concept pairs humans report as phenomenally
//! unified.
//!
//! ## Research Hypothesis (H2)
//!
//! HDC binding (⊗) produces representations with higher topological integration
//! (lower component count, higher unity score) than bundling (⊕), specifically
//! for concept pairs humans report as phenomenally unified.
//!
//! ## Theoretical Background
//!
//! The binding problem in consciousness asks: how are separate features unified
//! into single experiences? When we see a "red apple," how do redness and apple-ness
//! become a single unified percept rather than two separate properties?
//!
//! HDC offers two operations that might capture this:
//!
//! 1. **Binding (XOR)**: Creates a NEW vector orthogonal to both inputs
//!    - Properties: Commutative, associative, self-inverse
//!    - Intuition: Creates a "new thing" from the combination
//!    - Mathematical: A ⊗ B is nearly orthogonal to both A and B
//!
//! 2. **Bundling (Majority vote)**: Creates a superposition of inputs
//!    - Properties: Commutative, idempotent
//!    - Intuition: Both inputs are "present" in the result
//!    - Mathematical: bundle({A, B}) is similar to both A and B
//!
//! ## Hypothesis
//!
//! If binding captures something about phenomenal unity, then:
//! - Bound "unified" pairs (red+apple) should have higher Φ/unity than bundled
//! - Bound "separate" pairs (red+mailbox) should show less unity advantage
//! - The interaction effect (Bind vs Bundle) × (Unified vs Separate) should be significant
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea_core::hdc::phenomenal_binding_study::{
//!     PhenomenalBindingStudy, ConceptPair, PairType
//! };
//!
//! let study = PhenomenalBindingStudy::new();
//!
//! // Create concept pair
//! let pair = ConceptPair {
//!     concept_a: "red".to_string(),
//!     concept_b: "apple".to_string(),
//!     pair_type: PairType::Unified,
//!     hv_a: BinaryHV::random(1),
//!     hv_b: BinaryHV::random(2),
//! };
//!
//! // Compare bind vs bundle
//! let result = study.compare_operations(&pair);
//! println!("Bind unity: {}, Bundle unity: {}", result.bind_unity, result.bundle_unity);
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::binary_hv::BinaryHV;
use super::consciousness_topology::{
    BettiNumbers, ConsciousnessTopology, TopologicalAssessment, TopologyConfig,
};

/// Type of concept pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairType {
    /// Concepts that form a unified phenomenal percept (e.g., "red apple")
    Unified,
    /// Concepts that co-occur but remain phenomenally separate (e.g., "red mailbox")
    Separate,
}

impl PairType {
    /// Convert to string for display
    pub fn as_str(&self) -> &'static str {
        match self {
            PairType::Unified => "unified",
            PairType::Separate => "separate",
        }
    }
}

/// A concept pair for the binding study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPair {
    /// Unique identifier
    pub id: String,
    /// First concept (e.g., "red")
    pub concept_a: String,
    /// Second concept (e.g., "apple")
    pub concept_b: String,
    /// Combined description
    pub combined: String,
    /// Type of pair
    pub pair_type: PairType,
    /// Description of the pairing
    pub description: String,
}

/// Concept pair with HDC vectors
#[derive(Debug, Clone)]
pub struct ConceptPairWithVectors {
    /// The concept pair
    pub pair: ConceptPair,
    /// HDC vector for concept A
    pub hv_a: BinaryHV,
    /// HDC vector for concept B
    pub hv_b: BinaryHV,
}

/// Result of comparing bind vs bundle for one pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationComparisonResult {
    /// The concept pair
    pub pair_id: String,
    /// Type of pair
    pub pair_type: PairType,
    /// Unity score for bound representation
    pub bind_unity: f64,
    /// Unity score for bundled representation
    pub bundle_unity: f64,
    /// Unity advantage of binding (bind_unity - bundle_unity)
    pub unity_advantage: f64,
    /// Betti numbers for bound representation
    pub bind_betti: BettiNumbers,
    /// Betti numbers for bundled representation
    pub bundle_betti: BettiNumbers,
    /// Quality score for bound representation
    pub bind_quality: f64,
    /// Quality score for bundled representation
    pub bundle_quality: f64,
}

/// Cell means for 2x2 ANOVA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMeans {
    /// Mean unity for Bind + Unified pairs
    pub bind_unified: f64,
    /// Mean unity for Bind + Separate pairs
    pub bind_separate: f64,
    /// Mean unity for Bundle + Unified pairs
    pub bundle_unified: f64,
    /// Mean unity for Bundle + Separate pairs
    pub bundle_separate: f64,
    /// Number of unified pairs
    pub n_unified: usize,
    /// Number of separate pairs
    pub n_separate: usize,
}

/// Result of 2x2 ANOVA analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResult {
    /// Cell means
    pub cell_means: CellMeans,
    /// Main effect of Operation (Bind vs Bundle)
    pub main_effect_operation: f64,
    /// Main effect of Pair Type (Unified vs Separate)
    pub main_effect_pair_type: f64,
    /// Interaction effect
    pub interaction_effect: f64,
    /// F-statistic for interaction (simplified)
    pub f_interaction: f64,
    /// Is interaction significant (F > critical value)?
    pub interaction_significant: bool,
    /// Cohen's d for binding advantage in unified pairs
    pub cohens_d_unified: f64,
    /// Cohen's d for binding advantage in separate pairs
    pub cohens_d_separate: f64,
    /// Overall Cohen's d for binding advantage
    pub cohens_d_overall: f64,
}

/// Study metadata from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyMetadata {
    pub description: String,
    pub version: String,
    pub hypothesis: String,
}

/// JSON structure for concept pairs file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPairsFile {
    pub metadata: StudyMetadata,
    pub unified_pairs: Vec<ConceptPairJson>,
    pub separate_pairs: Vec<ConceptPairJson>,
}

/// JSON structure for a single concept pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPairJson {
    pub id: String,
    pub concept_a: String,
    pub concept_b: String,
    pub combined: String,
    pub description: String,
}

impl ConceptPairJson {
    /// Convert to ConceptPair with specified type
    pub fn to_concept_pair(&self, pair_type: PairType) -> ConceptPair {
        ConceptPair {
            id: self.id.clone(),
            concept_a: self.concept_a.clone(),
            concept_b: self.concept_b.clone(),
            combined: self.combined.clone(),
            pair_type,
            description: self.description.clone(),
        }
    }
}

/// Configuration for the phenomenal binding study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingStudyConfig {
    /// Topology analysis configuration
    pub topology_config: TopologyConfig,
    /// Scale for topology analysis
    pub analysis_scale: f64,
    /// Number of permutations for statistical significance
    pub n_permutations: usize,
    /// Minimum number of states for topology analysis
    pub min_states: usize,
    /// Alpha level for significance testing
    pub alpha: f64,
}

impl Default for BindingStudyConfig {
    fn default() -> Self {
        Self {
            topology_config: TopologyConfig::default(),
            analysis_scale: 0.5,
            n_permutations: 1000,
            min_states: 5,
            alpha: 0.05,
        }
    }
}

/// Phenomenal Binding Study
///
/// Tests whether HDC binding produces higher topological integration than
/// bundling for concept pairs humans report as phenomenally unified.
pub struct PhenomenalBindingStudy {
    /// Configuration
    config: BindingStudyConfig,
    /// Concept vocabulary (word -> BinaryHV)
    vocabulary: HashMap<String, BinaryHV>,
    /// Seed for deterministic random generation
    seed_counter: u64,
}

impl PhenomenalBindingStudy {
    /// Create a new study with default configuration
    pub fn new() -> Self {
        Self {
            config: BindingStudyConfig::default(),
            vocabulary: HashMap::new(),
            seed_counter: 1000,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BindingStudyConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
            seed_counter: 1000,
        }
    }

    /// Load concept pairs from JSON file
    pub fn load_pairs<P: AsRef<Path>>(&self, path: P) -> Result<Vec<ConceptPair>> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open pairs file: {}", path.display()))?;
        let reader = BufReader::new(file);
        let pairs_file: ConceptPairsFile = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse pairs JSON: {}", path.display()))?;

        let mut pairs = Vec::new();

        for json_pair in pairs_file.unified_pairs {
            pairs.push(json_pair.to_concept_pair(PairType::Unified));
        }

        for json_pair in pairs_file.separate_pairs {
            pairs.push(json_pair.to_concept_pair(PairType::Separate));
        }

        Ok(pairs)
    }

    /// Get or create HDC vector for a concept
    pub fn get_or_create_vector(&mut self, concept: &str) -> BinaryHV {
        if let Some(hv) = self.vocabulary.get(concept) {
            return *hv;
        }

        // Create deterministic random vector based on concept text
        let seed = self.seed_counter;
        self.seed_counter += 1;

        // Use concept hash for additional determinism
        let concept_hash: u64 = concept
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let hv = BinaryHV::random(seed.wrapping_add(concept_hash));
        self.vocabulary.insert(concept.to_string(), hv);
        hv
    }

    /// Add vectors to concept pairs
    pub fn add_vectors(&mut self, pairs: &[ConceptPair]) -> Vec<ConceptPairWithVectors> {
        pairs
            .iter()
            .map(|pair| {
                let hv_a = self.get_or_create_vector(&pair.concept_a);
                let hv_b = self.get_or_create_vector(&pair.concept_b);
                ConceptPairWithVectors {
                    pair: pair.clone(),
                    hv_a,
                    hv_b,
                }
            })
            .collect()
    }

    /// Compare bind vs bundle for a single concept pair
    pub fn compare_operations(&self, pair: &ConceptPairWithVectors) -> OperationComparisonResult {
        // Compute bound representation (XOR)
        let bound = pair.hv_a.bind(&pair.hv_b);

        // Compute bundled representation (majority vote)
        let bundled = BinaryHV::bundle(&[pair.hv_a, pair.hv_b]);

        // Analyze topology for both
        let bind_assessment = self.analyze_topology(&bound);
        let bundle_assessment = self.analyze_topology(&bundled);

        OperationComparisonResult {
            pair_id: pair.pair.id.clone(),
            pair_type: pair.pair.pair_type,
            bind_unity: bind_assessment.unity_score,
            bundle_unity: bundle_assessment.unity_score,
            unity_advantage: bind_assessment.unity_score - bundle_assessment.unity_score,
            bind_betti: bind_assessment.betti,
            bundle_betti: bundle_assessment.betti,
            bind_quality: bind_assessment.quality,
            bundle_quality: bundle_assessment.quality,
        }
    }

    /// Analyze topology for an HDC vector
    fn analyze_topology(&self, hv: &BinaryHV) -> TopologicalAssessment {
        let mut topology = ConsciousnessTopology::new(self.config.topology_config.clone());

        // Add the main vector
        topology.add_state(*hv);

        // Add permuted variations to build a point cloud
        for shift in 1..self.config.min_states {
            let permuted = hv.permute(shift * 100);
            topology.add_state(permuted);
        }

        topology.analyze(self.config.analysis_scale)
    }

    /// Run the full experiment on all pairs
    pub fn run_experiment(
        &self,
        pairs: &[ConceptPairWithVectors],
    ) -> Vec<OperationComparisonResult> {
        pairs
            .iter()
            .map(|pair| self.compare_operations(pair))
            .collect()
    }

    /// Perform 2x2 ANOVA analysis
    ///
    /// Factors:
    /// - Operation: Bind vs Bundle
    /// - Pair Type: Unified vs Separate
    ///
    /// Key test: Interaction effect (binding specifically helps unified pairs)
    pub fn analyze_anova(&self, results: &[OperationComparisonResult]) -> AnovaResult {
        // Separate by pair type
        let unified_results: Vec<_> = results
            .iter()
            .filter(|r| r.pair_type == PairType::Unified)
            .collect();
        let separate_results: Vec<_> = results
            .iter()
            .filter(|r| r.pair_type == PairType::Separate)
            .collect();

        // Compute cell means
        let bind_unified_mean = mean(unified_results.iter().map(|r| r.bind_unity));
        let bundle_unified_mean = mean(unified_results.iter().map(|r| r.bundle_unity));
        let bind_separate_mean = mean(separate_results.iter().map(|r| r.bind_unity));
        let bundle_separate_mean = mean(separate_results.iter().map(|r| r.bundle_unity));

        let cell_means = CellMeans {
            bind_unified: bind_unified_mean,
            bind_separate: bind_separate_mean,
            bundle_unified: bundle_unified_mean,
            bundle_separate: bundle_separate_mean,
            n_unified: unified_results.len(),
            n_separate: separate_results.len(),
        };

        // Compute main effects
        let bind_mean = (bind_unified_mean + bind_separate_mean) / 2.0;
        let bundle_mean = (bundle_unified_mean + bundle_separate_mean) / 2.0;
        let main_effect_operation = bind_mean - bundle_mean;

        let unified_mean = (bind_unified_mean + bundle_unified_mean) / 2.0;
        let separate_mean = (bind_separate_mean + bundle_separate_mean) / 2.0;
        let main_effect_pair_type = unified_mean - separate_mean;

        // Compute interaction effect
        // Interaction = (Bind_Unified - Bundle_Unified) - (Bind_Separate - Bundle_Separate)
        let bind_advantage_unified = bind_unified_mean - bundle_unified_mean;
        let bind_advantage_separate = bind_separate_mean - bundle_separate_mean;
        let interaction_effect = bind_advantage_unified - bind_advantage_separate;

        // Simplified F-statistic (using variance estimate)
        let all_values: Vec<f64> = results
            .iter()
            .flat_map(|r| vec![r.bind_unity, r.bundle_unity])
            .collect();
        let overall_variance = variance(&all_values);
        let mse = overall_variance.max(0.001); // Prevent division by zero

        let f_interaction = (interaction_effect.powi(2) * results.len() as f64) / mse;

        // Critical F value for df1=1, df2=large, alpha=0.05 is approximately 3.84
        let interaction_significant = f_interaction > 3.84;

        // Cohen's d calculations
        let cohens_d_unified = if overall_variance > 0.0 {
            bind_advantage_unified / overall_variance.sqrt()
        } else {
            0.0
        };

        let cohens_d_separate = if overall_variance > 0.0 {
            bind_advantage_separate / overall_variance.sqrt()
        } else {
            0.0
        };

        let overall_advantage = mean(results.iter().map(|r| r.unity_advantage));
        let cohens_d_overall = if overall_variance > 0.0 {
            overall_advantage / overall_variance.sqrt()
        } else {
            0.0
        };

        AnovaResult {
            cell_means,
            main_effect_operation,
            main_effect_pair_type,
            interaction_effect,
            f_interaction,
            interaction_significant,
            cohens_d_unified,
            cohens_d_separate,
            cohens_d_overall,
        }
    }

    /// Compute Wasserstein distance between two topological assessments
    ///
    /// This is a simplified version based on Betti number differences
    /// and persistence values.
    pub fn wasserstein_distance(
        &self,
        a: &TopologicalAssessment,
        b: &TopologicalAssessment,
    ) -> f64 {
        let betti_diff = (a.betti.beta_0 as f64 - b.betti.beta_0 as f64).abs()
            + (a.betti.beta_1 as f64 - b.betti.beta_1 as f64).abs()
            + (a.betti.beta_2 as f64 - b.betti.beta_2 as f64).abs();

        let unity_diff = (a.unity_score - b.unity_score).abs();
        let quality_diff = (a.quality - b.quality).abs();

        betti_diff + unity_diff + quality_diff
    }

    /// Generate summary report of experiment results
    pub fn generate_report(
        &self,
        results: &[OperationComparisonResult],
        anova: &AnovaResult,
    ) -> String {
        let mut report = String::new();

        report.push_str("=== Phenomenal Binding Study Report ===\n\n");

        // Sample sizes
        report.push_str(&format!(
            "Sample Sizes:\n  Unified pairs: {}\n  Separate pairs: {}\n  Total: {}\n\n",
            anova.cell_means.n_unified,
            anova.cell_means.n_separate,
            results.len()
        ));

        // Cell means
        report.push_str("Cell Means (Unity Score):\n");
        report.push_str("                 Unified    Separate\n");
        report.push_str(&format!(
            "  Bind:          {:.4}      {:.4}\n",
            anova.cell_means.bind_unified, anova.cell_means.bind_separate
        ));
        report.push_str(&format!(
            "  Bundle:        {:.4}      {:.4}\n\n",
            anova.cell_means.bundle_unified, anova.cell_means.bundle_separate
        ));

        // Main effects
        report.push_str("Main Effects:\n");
        report.push_str(&format!(
            "  Operation (Bind - Bundle): {:.4}\n",
            anova.main_effect_operation
        ));
        report.push_str(&format!(
            "  Pair Type (Unified - Separate): {:.4}\n\n",
            anova.main_effect_pair_type
        ));

        // Interaction
        report.push_str("Interaction Effect:\n");
        report.push_str(&format!("  Value: {:.4}\n", anova.interaction_effect));
        report.push_str(&format!("  F-statistic: {:.4}\n", anova.f_interaction));
        report.push_str(&format!(
            "  Significant (p < 0.05): {}\n\n",
            if anova.interaction_significant {
                "Yes"
            } else {
                "No"
            }
        ));

        // Effect sizes
        report.push_str("Effect Sizes (Cohen's d):\n");
        report.push_str(&format!(
            "  Binding advantage (unified pairs): {:.4}\n",
            anova.cohens_d_unified
        ));
        report.push_str(&format!(
            "  Binding advantage (separate pairs): {:.4}\n",
            anova.cohens_d_separate
        ));
        report.push_str(&format!(
            "  Overall binding advantage: {:.4}\n\n",
            anova.cohens_d_overall
        ));

        // Interpretation
        report.push_str("Interpretation:\n");
        if anova.interaction_significant && anova.interaction_effect > 0.0 {
            report.push_str("  The interaction effect is SIGNIFICANT and POSITIVE.\n");
            report.push_str("  Binding provides a larger unity advantage for unified pairs\n");
            report.push_str("  than for separate pairs. This supports hypothesis H2.\n");
        } else if anova.interaction_significant && anova.interaction_effect < 0.0 {
            report.push_str("  The interaction effect is SIGNIFICANT but NEGATIVE.\n");
            report.push_str("  Contrary to H2, binding provides more unity for separate pairs.\n");
        } else {
            report.push_str("  The interaction effect is NOT significant.\n");
            report.push_str("  Binding's unity advantage does not depend on pair type.\n");
        }

        report
    }
}

impl Default for PhenomenalBindingStudy {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute mean of an iterator
fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let values: Vec<f64> = iter.collect();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute variance of a slice
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_type() {
        assert_eq!(PairType::Unified.as_str(), "unified");
        assert_eq!(PairType::Separate.as_str(), "separate");
    }

    #[test]
    fn test_study_creation() {
        let study = PhenomenalBindingStudy::new();
        assert!(study.vocabulary.is_empty());
    }

    #[test]
    fn test_vector_creation() {
        let mut study = PhenomenalBindingStudy::new();

        let hv1 = study.get_or_create_vector("red");
        let hv2 = study.get_or_create_vector("apple");
        let hv1_again = study.get_or_create_vector("red");

        // Same word should return same vector
        assert_eq!(hv1, hv1_again);

        // Different words should return different vectors
        assert_ne!(hv1, hv2);
    }

    #[test]
    fn test_compare_operations() {
        let study = PhenomenalBindingStudy::new();

        let pair = ConceptPairWithVectors {
            pair: ConceptPair {
                id: "test_001".to_string(),
                concept_a: "red".to_string(),
                concept_b: "apple".to_string(),
                combined: "red apple".to_string(),
                pair_type: PairType::Unified,
                description: "Test pair".to_string(),
            },
            hv_a: BinaryHV::random(1),
            hv_b: BinaryHV::random(2),
        };

        let result = study.compare_operations(&pair);

        // Unity scores should be in valid range
        assert!(result.bind_unity >= 0.0 && result.bind_unity <= 1.0);
        assert!(result.bundle_unity >= 0.0 && result.bundle_unity <= 1.0);
    }

    #[test]
    fn test_anova_analysis() {
        let study = PhenomenalBindingStudy::new();

        // Create mock results
        let results = vec![
            OperationComparisonResult {
                pair_id: "1".to_string(),
                pair_type: PairType::Unified,
                bind_unity: 0.9,
                bundle_unity: 0.6,
                unity_advantage: 0.3,
                bind_betti: BettiNumbers::new(1, 0, 0),
                bundle_betti: BettiNumbers::new(2, 1, 0),
                bind_quality: 0.9,
                bundle_quality: 0.7,
            },
            OperationComparisonResult {
                pair_id: "2".to_string(),
                pair_type: PairType::Unified,
                bind_unity: 0.85,
                bundle_unity: 0.55,
                unity_advantage: 0.3,
                bind_betti: BettiNumbers::new(1, 0, 0),
                bundle_betti: BettiNumbers::new(2, 0, 0),
                bind_quality: 0.85,
                bundle_quality: 0.65,
            },
            OperationComparisonResult {
                pair_id: "3".to_string(),
                pair_type: PairType::Separate,
                bind_unity: 0.7,
                bundle_unity: 0.6,
                unity_advantage: 0.1,
                bind_betti: BettiNumbers::new(1, 1, 0),
                bundle_betti: BettiNumbers::new(2, 1, 0),
                bind_quality: 0.75,
                bundle_quality: 0.7,
            },
            OperationComparisonResult {
                pair_id: "4".to_string(),
                pair_type: PairType::Separate,
                bind_unity: 0.65,
                bundle_unity: 0.55,
                unity_advantage: 0.1,
                bind_betti: BettiNumbers::new(1, 0, 0),
                bundle_betti: BettiNumbers::new(2, 0, 0),
                bind_quality: 0.7,
                bundle_quality: 0.65,
            },
        ];

        let anova = study.analyze_anova(&results);

        // Check cell means are computed
        assert!(anova.cell_means.bind_unified > 0.0);
        assert!(anova.cell_means.bundle_unified > 0.0);

        // Interaction effect should be positive (binding helps unified more)
        assert!(anova.interaction_effect > 0.0);
    }

    #[test]
    fn test_generate_report() {
        let study = PhenomenalBindingStudy::new();

        let results = vec![OperationComparisonResult {
            pair_id: "1".to_string(),
            pair_type: PairType::Unified,
            bind_unity: 0.8,
            bundle_unity: 0.5,
            unity_advantage: 0.3,
            bind_betti: BettiNumbers::new(1, 0, 0),
            bundle_betti: BettiNumbers::new(2, 0, 0),
            bind_quality: 0.8,
            bundle_quality: 0.6,
        }];

        let anova = study.analyze_anova(&results);
        let report = study.generate_report(&results, &anova);

        assert!(report.contains("Phenomenal Binding Study Report"));
        assert!(report.contains("Cell Means"));
        assert!(report.contains("Interaction Effect"));
    }

    #[test]
    fn test_mean_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(values.iter().copied());
        assert!((m - 3.0).abs() < 0.001);

        let v = variance(&values);
        assert!((v - 2.0).abs() < 0.001);
    }
}
