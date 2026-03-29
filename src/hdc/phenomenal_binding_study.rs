// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenal Binding Study via HDC Operations
//!
//! Research Direction 2: Testing whether HDC binding produces representations
//! with higher topological integration than bundling for phenomenally unified concepts.
//!
//! ## Research Hypothesis (H2)
//!
//! HDC binding (XOR) produces representations with higher topological integration
//! (lower component count, higher unity score) than bundling (majority vote),
//! specifically for concept pairs humans report as phenomenally unified.
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
//!    - Mathematical: A XOR B is nearly orthogonal to both A and B
//!
//! 2. **Bundling (Majority vote)**: Creates a superposition of inputs
//!    - Properties: Commutative, idempotent
//!    - Intuition: Both inputs are "present" in the result
//!    - Mathematical: bundle({A, B}) is similar to both A and B
//!
//! ## Key Insight
//!
//! If binding captures phenomenal unity, then:
//! - Bound "unified" pairs (red+apple) should have higher unity than bundled
//! - Bound "separate" pairs (red+mailbox) should show less unity advantage
//! - The interaction effect should be significant
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::phenomenal_binding_study::{
//!     BindingStudy, BindingStudyConfig, PairCategory
//! };
//!
//! let study = BindingStudy::new(BindingStudyConfig::default());
//! let results = study.run("data/binding_study/concept_pairs.json")?;
//! println!("{}", results.summary_report());
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// Re-export from symthaea-core
pub use symthaea_core::hdc::phenomenal_binding_study::{
    AnovaResult, BindingStudyConfig, CellMeans, ConceptPair, ConceptPairJson,
    ConceptPairWithVectors, ConceptPairsFile, OperationComparisonResult, PairType,
    PhenomenalBindingStudy as CoreBindingStudy, StudyMetadata,
};

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::consciousness_topology::{
    BettiNumbers, ConsciousnessTopology, TopologicalAssessment, TopologyConfig,
};

/// Extended configuration for the binding study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedStudyConfig {
    /// Core configuration
    pub core: BindingStudyConfig,
    /// Enable bootstrap confidence intervals
    pub bootstrap_ci: bool,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Enable permutation tests
    pub permutation_tests: bool,
    /// Number of permutation iterations
    pub n_permutations: usize,
    /// Confidence level for intervals (e.g., 0.95)
    pub confidence_level: f64,
    /// Verbose output during analysis
    pub verbose: bool,
}

impl Default for ExtendedStudyConfig {
    fn default() -> Self {
        Self {
            core: BindingStudyConfig::default(),
            bootstrap_ci: true,
            n_bootstrap: 1000,
            permutation_tests: true,
            n_permutations: 5000,
            confidence_level: 0.95,
            verbose: false,
        }
    }
}

/// Extended result with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedComparisonResult {
    /// Base comparison result
    pub base: OperationComparisonResult,
    /// Bind topology assessment
    pub bind_topology: TopologySummary,
    /// Bundle topology assessment
    pub bundle_topology: TopologySummary,
    /// Similarity between bound and bundled representations
    pub bind_bundle_similarity: f64,
}

/// Summary of topological analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySummary {
    /// Betti numbers
    pub betti: BettiNumbers,
    /// Unity score
    pub unity_score: f64,
    /// Quality score
    pub quality: f64,
    /// Completeness score
    pub completeness: f64,
    /// Circularity score
    pub circularity: f64,
    /// Number of persistent features
    pub num_features: usize,
    /// Total persistence
    pub total_persistence: f64,
}

impl From<TopologicalAssessment> for TopologySummary {
    fn from(assessment: TopologicalAssessment) -> Self {
        Self {
            betti: assessment.betti,
            unity_score: assessment.unity_score,
            quality: assessment.quality,
            completeness: assessment.completeness,
            circularity: assessment.circularity,
            num_features: assessment.features.len(),
            total_persistence: assessment.features.iter().map(|f| f.persistence).sum(),
        }
    }
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name
    pub name: String,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Is significant at alpha=0.05?
    pub significant: bool,
}

/// Complete study results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingStudyResults {
    /// Individual pair results
    pub pair_results: Vec<ExtendedComparisonResult>,
    /// ANOVA results
    pub anova: AnovaResult,
    /// Statistical tests
    pub tests: Vec<StatisticalTest>,
    /// Summary statistics by category
    pub category_stats: HashMap<String, CategoryStatistics>,
    /// Metadata
    pub metadata: StudyResultsMetadata,
}

/// Statistics for a category of pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStatistics {
    /// Category name
    pub name: String,
    /// Number of pairs
    pub count: usize,
    /// Mean bind unity
    pub mean_bind_unity: f64,
    /// Mean bundle unity
    pub mean_bundle_unity: f64,
    /// Mean unity advantage
    pub mean_advantage: f64,
    /// Standard deviation of advantage
    pub std_advantage: f64,
    /// Median advantage
    pub median_advantage: f64,
    /// Min advantage
    pub min_advantage: f64,
    /// Max advantage
    pub max_advantage: f64,
}

/// Metadata for study results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyResultsMetadata {
    /// Timestamp
    pub timestamp: String,
    /// Configuration used
    pub config: ExtendedStudyConfig,
    /// Total pairs analyzed
    pub total_pairs: usize,
    /// Duration in seconds
    pub duration_secs: f64,
}

/// Extended Binding Study with statistical analysis
///
/// Builds on the core phenomenal binding study with additional
/// statistical tests, confidence intervals, and detailed reporting.
pub struct BindingStudy {
    /// Configuration
    config: ExtendedStudyConfig,
    /// Core study implementation
    core: CoreBindingStudy,
    /// Random state for permutation tests
    rng_state: u64,
}

impl BindingStudy {
    /// Create a new binding study
    pub fn new(config: ExtendedStudyConfig) -> Self {
        Self {
            core: CoreBindingStudy::with_config(config.core.clone()),
            config,
            rng_state: 42,
        }
    }

    /// Run the complete study
    pub fn run<P: AsRef<Path>>(&mut self, pairs_path: P) -> Result<BindingStudyResults> {
        let start = std::time::Instant::now();

        // Load pairs
        let pairs = self.core.load_pairs(&pairs_path)?;
        let pairs_with_vectors = self.core.add_vectors(&pairs);

        if self.config.verbose {
            println!("Loaded {} pairs", pairs.len());
        }

        // Run comparisons
        let mut pair_results = Vec::new();
        for (i, pair) in pairs_with_vectors.iter().enumerate() {
            let result = self.compare_extended(pair);
            pair_results.push(result);

            if self.config.verbose && (i + 1) % 10 == 0 {
                println!("  Processed {}/{} pairs", i + 1, pairs_with_vectors.len());
            }
        }

        // Convert to base results for ANOVA
        let base_results: Vec<_> = pair_results.iter().map(|r| r.base.clone()).collect();

        // Compute ANOVA
        let anova = self.core.analyze_anova(&base_results);

        // Compute statistical tests
        let tests = self.compute_statistical_tests(&pair_results);

        // Compute category statistics
        let category_stats = self.compute_category_stats(&pair_results);

        // Create metadata
        let metadata = StudyResultsMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            total_pairs: pairs.len(),
            duration_secs: start.elapsed().as_secs_f64(),
        };

        Ok(BindingStudyResults {
            pair_results,
            anova,
            tests,
            category_stats,
            metadata,
        })
    }

    /// Extended comparison with topology summaries
    fn compare_extended(&self, pair: &ConceptPairWithVectors) -> ExtendedComparisonResult {
        let base = self.core.compare_operations(pair);

        // Compute bound and bundled vectors
        let bound = pair.hv_a.bind(&pair.hv_b);
        let bundled = BinaryHV::bundle(&[pair.hv_a, pair.hv_b]);

        // Get detailed topology
        let bind_topology = self.analyze_topology_detailed(&bound);
        let bundle_topology = self.analyze_topology_detailed(&bundled);

        // Similarity between bound and bundled
        let bind_bundle_similarity = bound.similarity(&bundled) as f64;

        ExtendedComparisonResult {
            base,
            bind_topology,
            bundle_topology,
            bind_bundle_similarity,
        }
    }

    /// Detailed topology analysis
    fn analyze_topology_detailed(&self, hv: &BinaryHV) -> TopologySummary {
        let mut topology = ConsciousnessTopology::new(self.config.core.topology_config.clone());

        // Add main vector and permutations
        topology.add_state(*hv);
        for shift in 1..self.config.core.min_states {
            topology.add_state(hv.permute(shift * 100));
        }

        let assessment = topology.analyze(self.config.core.analysis_scale);
        TopologySummary::from(assessment)
    }

    /// Compute statistical tests
    fn compute_statistical_tests(
        &mut self,
        results: &[ExtendedComparisonResult],
    ) -> Vec<StatisticalTest> {
        let mut tests = Vec::new();

        // Separate by pair type
        let unified: Vec<_> = results
            .iter()
            .filter(|r| r.base.pair_type == PairType::Unified)
            .collect();
        let separate: Vec<_> = results
            .iter()
            .filter(|r| r.base.pair_type == PairType::Separate)
            .collect();

        // Test 1: Binding advantage in unified pairs
        let unified_advantages: Vec<f64> = unified.iter().map(|r| r.base.unity_advantage).collect();
        if !unified_advantages.is_empty() {
            let test = self.one_sample_test("binding_advantage_unified", &unified_advantages, 0.0);
            tests.push(test);
        }

        // Test 2: Binding advantage in separate pairs
        let separate_advantages: Vec<f64> =
            separate.iter().map(|r| r.base.unity_advantage).collect();
        if !separate_advantages.is_empty() {
            let test =
                self.one_sample_test("binding_advantage_separate", &separate_advantages, 0.0);
            tests.push(test);
        }

        // Test 3: Interaction effect
        if !unified_advantages.is_empty() && !separate_advantages.is_empty() {
            let test = self.two_sample_test(
                "interaction_effect",
                &unified_advantages,
                &separate_advantages,
            );
            tests.push(test);
        }

        // Test 4: Betti number differences
        let unified_bind_b0: Vec<f64> = unified
            .iter()
            .map(|r| r.bind_topology.betti.beta_0 as f64)
            .collect();
        let unified_bundle_b0: Vec<f64> = unified
            .iter()
            .map(|r| r.bundle_topology.betti.beta_0 as f64)
            .collect();
        if !unified_bind_b0.is_empty() {
            let diffs: Vec<f64> = unified_bind_b0
                .iter()
                .zip(&unified_bundle_b0)
                .map(|(a, b)| a - b)
                .collect();
            let test = self.one_sample_test("betti0_difference_unified", &diffs, 0.0);
            tests.push(test);
        }

        tests
    }

    /// One-sample test against a null value
    fn one_sample_test(&mut self, name: &str, values: &[f64], null_value: f64) -> StatisticalTest {
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let std = self.std_dev(values);
        let se = std / (n as f64).sqrt();

        let t_stat = if se > 0.0 {
            (mean - null_value) / se
        } else {
            0.0
        };

        // Permutation test p-value
        let p_value = if self.config.permutation_tests {
            self.permutation_test_one_sample(values, null_value)
        } else {
            self.approximate_p_value(t_stat.abs(), n)
        };

        let effect_size = if std > 0.0 {
            (mean - null_value) / std
        } else {
            0.0
        };

        // Confidence interval
        let (ci_lower, ci_upper) = if self.config.bootstrap_ci {
            self.bootstrap_ci(values)
        } else {
            let margin = 1.96 * se;
            (mean - margin, mean + margin)
        };

        StatisticalTest {
            name: name.to_string(),
            statistic: t_stat,
            p_value,
            effect_size,
            ci_lower,
            ci_upper,
            significant: p_value < 0.05,
        }
    }

    /// Two-sample test
    fn two_sample_test(&mut self, name: &str, group_a: &[f64], group_b: &[f64]) -> StatisticalTest {
        let mean_a = group_a.iter().sum::<f64>() / group_a.len() as f64;
        let mean_b = group_b.iter().sum::<f64>() / group_b.len() as f64;
        let diff = mean_a - mean_b;

        let std_a = self.std_dev(group_a);
        let std_b = self.std_dev(group_b);
        let pooled_std = ((std_a.powi(2) * (group_a.len() - 1) as f64
            + std_b.powi(2) * (group_b.len() - 1) as f64)
            / (group_a.len() + group_b.len() - 2) as f64)
            .sqrt();

        let se = pooled_std * (1.0 / group_a.len() as f64 + 1.0 / group_b.len() as f64).sqrt();
        let t_stat = if se > 0.0 { diff / se } else { 0.0 };

        let p_value = if self.config.permutation_tests {
            self.permutation_test_two_sample(group_a, group_b)
        } else {
            self.approximate_p_value(t_stat.abs(), group_a.len() + group_b.len() - 2)
        };

        let effect_size = if pooled_std > 0.0 {
            diff / pooled_std
        } else {
            0.0
        };

        let (ci_lower, ci_upper) = if self.config.bootstrap_ci {
            self.bootstrap_ci_two_sample(group_a, group_b)
        } else {
            let margin = 1.96 * se;
            (diff - margin, diff + margin)
        };

        StatisticalTest {
            name: name.to_string(),
            statistic: t_stat,
            p_value,
            effect_size,
            ci_lower,
            ci_upper,
            significant: p_value < 0.05,
        }
    }

    /// Permutation test for one sample
    fn permutation_test_one_sample(&mut self, values: &[f64], null_value: f64) -> f64 {
        let observed = values.iter().sum::<f64>() / values.len() as f64 - null_value;
        let mut more_extreme = 0;

        for _ in 0..self.config.n_permutations {
            // Randomly flip signs
            let mut sum = 0.0;
            for &v in values {
                let sign = if self.next_random() < 0.5 { 1.0 } else { -1.0 };
                sum += (v - null_value) * sign;
            }
            let perm_mean = sum / values.len() as f64;

            if perm_mean.abs() >= observed.abs() {
                more_extreme += 1;
            }
        }

        more_extreme as f64 / self.config.n_permutations as f64
    }

    /// Permutation test for two samples
    fn permutation_test_two_sample(&mut self, group_a: &[f64], group_b: &[f64]) -> f64 {
        let observed = group_a.iter().sum::<f64>() / group_a.len() as f64
            - group_b.iter().sum::<f64>() / group_b.len() as f64;

        let mut combined: Vec<f64> = group_a.iter().chain(group_b.iter()).copied().collect();
        let n_a = group_a.len();
        let mut more_extreme = 0;

        for _ in 0..self.config.n_permutations {
            // Shuffle combined
            for i in (1..combined.len()).rev() {
                let j = (self.next_random() * (i + 1) as f64) as usize;
                combined.swap(i, j);
            }

            let perm_a: f64 = combined[..n_a].iter().sum::<f64>() / n_a as f64;
            let perm_b: f64 = combined[n_a..].iter().sum::<f64>() / (combined.len() - n_a) as f64;
            let perm_diff = perm_a - perm_b;

            if perm_diff.abs() >= observed.abs() {
                more_extreme += 1;
            }
        }

        more_extreme as f64 / self.config.n_permutations as f64
    }

    /// Bootstrap confidence interval
    fn bootstrap_ci(&mut self, values: &[f64]) -> (f64, f64) {
        let mut means = Vec::with_capacity(self.config.n_bootstrap);

        for _ in 0..self.config.n_bootstrap {
            let mut sum = 0.0;
            for _ in 0..values.len() {
                let idx = (self.next_random() * values.len() as f64) as usize;
                sum += values[idx.min(values.len() - 1)];
            }
            means.push(sum / values.len() as f64);
        }

        means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx =
            ((1.0 - self.config.confidence_level) / 2.0 * self.config.n_bootstrap as f64) as usize;
        let upper_idx =
            ((1.0 + self.config.confidence_level) / 2.0 * self.config.n_bootstrap as f64) as usize;

        (
            means[lower_idx.min(means.len() - 1)],
            means[upper_idx.min(means.len() - 1)],
        )
    }

    /// Bootstrap CI for two-sample difference
    fn bootstrap_ci_two_sample(&mut self, group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
        let mut diffs = Vec::with_capacity(self.config.n_bootstrap);

        for _ in 0..self.config.n_bootstrap {
            let mut sum_a = 0.0;
            for _ in 0..group_a.len() {
                let idx = (self.next_random() * group_a.len() as f64) as usize;
                sum_a += group_a[idx.min(group_a.len() - 1)];
            }

            let mut sum_b = 0.0;
            for _ in 0..group_b.len() {
                let idx = (self.next_random() * group_b.len() as f64) as usize;
                sum_b += group_b[idx.min(group_b.len() - 1)];
            }

            diffs.push(sum_a / group_a.len() as f64 - sum_b / group_b.len() as f64);
        }

        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx =
            ((1.0 - self.config.confidence_level) / 2.0 * self.config.n_bootstrap as f64) as usize;
        let upper_idx =
            ((1.0 + self.config.confidence_level) / 2.0 * self.config.n_bootstrap as f64) as usize;

        (
            diffs[lower_idx.min(diffs.len() - 1)],
            diffs[upper_idx.min(diffs.len() - 1)],
        )
    }

    /// Approximate p-value from t-statistic
    fn approximate_p_value(&self, t: f64, df: usize) -> f64 {
        // Approximation using normal distribution for large df
        let z = t / (1.0 + t.powi(2) / (2.0 * df as f64)).sqrt();
        2.0 * (1.0 - self.standard_normal_cdf(z.abs()))
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Compute category statistics
    fn compute_category_stats(
        &self,
        results: &[ExtendedComparisonResult],
    ) -> HashMap<String, CategoryStatistics> {
        let mut stats = HashMap::new();

        // Unified pairs
        let unified: Vec<_> = results
            .iter()
            .filter(|r| r.base.pair_type == PairType::Unified)
            .collect();
        if !unified.is_empty() {
            stats.insert(
                "unified".to_string(),
                self.compute_single_category_stats("unified", &unified),
            );
        }

        // Separate pairs
        let separate: Vec<_> = results
            .iter()
            .filter(|r| r.base.pair_type == PairType::Separate)
            .collect();
        if !separate.is_empty() {
            stats.insert(
                "separate".to_string(),
                self.compute_single_category_stats("separate", &separate),
            );
        }

        stats
    }

    fn compute_single_category_stats(
        &self,
        name: &str,
        results: &[&ExtendedComparisonResult],
    ) -> CategoryStatistics {
        let advantages: Vec<f64> = results.iter().map(|r| r.base.unity_advantage).collect();
        let bind_unities: Vec<f64> = results.iter().map(|r| r.base.bind_unity).collect();
        let bundle_unities: Vec<f64> = results.iter().map(|r| r.base.bundle_unity).collect();

        let mean_adv = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let std_adv = self.std_dev(&advantages);

        let mut sorted_adv = advantages.clone();
        sorted_adv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_adv = sorted_adv[sorted_adv.len() / 2];

        CategoryStatistics {
            name: name.to_string(),
            count: results.len(),
            mean_bind_unity: bind_unities.iter().sum::<f64>() / bind_unities.len() as f64,
            mean_bundle_unity: bundle_unities.iter().sum::<f64>() / bundle_unities.len() as f64,
            mean_advantage: mean_adv,
            std_advantage: std_adv,
            median_advantage: median_adv,
            min_advantage: sorted_adv.first().copied().unwrap_or(0.0),
            max_advantage: sorted_adv.last().copied().unwrap_or(0.0),
        }
    }

    /// Standard deviation
    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Default for BindingStudy {
    fn default() -> Self {
        Self::new(ExtendedStudyConfig::default())
    }
}

impl BindingStudyResults {
    /// Generate summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("================================================================\n");
        report.push_str("   PHENOMENAL BINDING STUDY RESULTS\n");
        report.push_str("   Research Direction 2: HDC Binding vs Bundling\n");
        report.push_str("================================================================\n\n");

        // Metadata
        report.push_str(&format!("Timestamp: {}\n", self.metadata.timestamp));
        report.push_str(&format!("Total pairs: {}\n", self.metadata.total_pairs));
        report.push_str(&format!(
            "Duration: {:.2}s\n\n",
            self.metadata.duration_secs
        ));

        // ANOVA results
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   2x2 ANOVA: Operation (Bind/Bundle) x Pair Type (Unified/Separate)\n");
        report.push_str("----------------------------------------------------------------\n\n");

        report.push_str("Cell Means (Unity Score):\n");
        report.push_str(&format!(
            "                 Unified ({})    Separate ({})\n",
            self.anova.cell_means.n_unified, self.anova.cell_means.n_separate
        ));
        report.push_str(&format!(
            "  Bind:          {:.4}            {:.4}\n",
            self.anova.cell_means.bind_unified, self.anova.cell_means.bind_separate
        ));
        report.push_str(&format!(
            "  Bundle:        {:.4}            {:.4}\n\n",
            self.anova.cell_means.bundle_unified, self.anova.cell_means.bundle_separate
        ));

        report.push_str("Main Effects:\n");
        report.push_str(&format!(
            "  Operation (Bind - Bundle): {:+.4}\n",
            self.anova.main_effect_operation
        ));
        report.push_str(&format!(
            "  Pair Type (Unified - Separate): {:+.4}\n\n",
            self.anova.main_effect_pair_type
        ));

        report.push_str("Interaction Effect:\n");
        report.push_str(&format!("  Value: {:+.4}\n", self.anova.interaction_effect));
        report.push_str(&format!("  F-statistic: {:.4}\n", self.anova.f_interaction));
        report.push_str(&format!(
            "  Significant: {}\n\n",
            if self.anova.interaction_significant {
                "YES"
            } else {
                "No"
            }
        ));

        // Statistical tests
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   STATISTICAL TESTS\n");
        report.push_str("----------------------------------------------------------------\n\n");

        for test in &self.tests {
            report.push_str(&format!("{}:\n", test.name));
            report.push_str(&format!("  Statistic: {:.4}\n", test.statistic));
            report.push_str(&format!(
                "  p-value: {:.4} {}\n",
                test.p_value,
                if test.significant { "*" } else { "" }
            ));
            report.push_str(&format!("  Effect size (d): {:+.4}\n", test.effect_size));
            report.push_str(&format!(
                "  95% CI: [{:.4}, {:.4}]\n\n",
                test.ci_lower, test.ci_upper
            ));
        }

        // Category statistics
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   CATEGORY STATISTICS\n");
        report.push_str("----------------------------------------------------------------\n\n");

        for (name, stats) in &self.category_stats {
            report.push_str(&format!("{} pairs (n={}):\n", name, stats.count));
            report.push_str(&format!(
                "  Mean bind unity: {:.4}\n",
                stats.mean_bind_unity
            ));
            report.push_str(&format!(
                "  Mean bundle unity: {:.4}\n",
                stats.mean_bundle_unity
            ));
            report.push_str(&format!(
                "  Mean advantage: {:+.4} (SD: {:.4})\n",
                stats.mean_advantage, stats.std_advantage
            ));
            report.push_str(&format!(
                "  Median advantage: {:+.4}\n",
                stats.median_advantage
            ));
            report.push_str(&format!(
                "  Range: [{:.4}, {:.4}]\n\n",
                stats.min_advantage, stats.max_advantage
            ));
        }

        // Interpretation
        report.push_str("================================================================\n");
        report.push_str("   INTERPRETATION\n");
        report.push_str("================================================================\n\n");

        if self.anova.interaction_significant && self.anova.interaction_effect > 0.0 {
            report.push_str("HYPOTHESIS H2 SUPPORTED:\n");
            report.push_str("  Binding produces higher topological integration than bundling,\n");
            report.push_str("  specifically for phenomenally unified concept pairs.\n\n");
            report.push_str(&format!(
                "  Effect size for unified pairs: d = {:+.4}\n",
                self.anova.cohens_d_unified
            ));
        } else if self.anova.main_effect_operation > 0.0 {
            report.push_str("PARTIAL SUPPORT:\n");
            report.push_str("  Binding generally produces higher integration than bundling,\n");
            report.push_str("  but the effect is not specific to unified pairs.\n");
        } else {
            report.push_str("HYPOTHESIS H2 NOT SUPPORTED:\n");
            report.push_str("  No clear advantage of binding over bundling detected.\n");
        }

        report.push_str("\n================================================================\n");

        report
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_config_default() {
        let config = ExtendedStudyConfig::default();
        assert!(config.bootstrap_ci);
        assert_eq!(config.n_bootstrap, 1000);
        assert!(config.permutation_tests);
    }

    #[test]
    fn test_binding_study_creation() {
        let study = BindingStudy::default();
        assert!(!study.config.verbose);
    }

    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 0.001);
        assert!((erf(1.0) - 0.8427).abs() < 0.001);
        assert!((erf(-1.0) + 0.8427).abs() < 0.001);
    }

    #[test]
    fn test_topology_summary() {
        let betti = BettiNumbers::new(1, 0, 0);
        let summary = TopologySummary {
            betti,
            unity_score: 1.0,
            quality: 0.9,
            completeness: 0.95,
            circularity: 0.0,
            num_features: 5,
            total_persistence: 2.5,
        };
        assert_eq!(summary.unity_score, 1.0);
    }

    #[test]
    fn test_statistical_test_structure() {
        let test = StatisticalTest {
            name: "test".to_string(),
            statistic: 2.5,
            p_value: 0.01,
            effect_size: 0.8,
            ci_lower: 0.1,
            ci_upper: 0.5,
            significant: true,
        };
        assert!(test.significant);
        assert!(test.p_value < 0.05);
    }
}
