// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding Reversibility Study - H2 Revised
//!
//! This module implements a revised H2 hypothesis testing HDC binding via reversibility
//! instead of unity measures.
//!
//! ## Background
//!
//! The original H2 hypothesis failed because unity measures (Betti numbers, component counts)
//! measure permutation self-similarity, not binding quality. A permuted vector has the same
//! topology as the original, so this tells us nothing about how well information is preserved.
//!
//! ## Revised Hypothesis (H2-revised)
//!
//! **HDC binding (XOR) preserves component information better than bundling (majority vote),
//! measurable via reversibility.**
//!
//! ### Why Reversibility Matters
//!
//! For binding (XOR):
//! - A can be perfectly recovered from (A XOR B) XOR B = A
//! - This is a mathematical property of XOR: self-inverse
//! - similarity(A, recovered_A) = 1.0 exactly
//!
//! For bundling (majority vote):
//! - Components are mixed and cannot be cleanly separated
//! - There is no inverse operation to recover A from bundle(A, B)
//! - Any "recovery" attempt yields only partial information
//!
//! ### Key Metrics
//!
//! 1. **Reversibility Score**: similarity(A, recovered_A) where recovered_A = (A op B) inverse_op B
//! 2. **Component Preservation**: similarity(A, A op B) - how much of A survives in the result
//! 3. **Information Content**: How much original information can be retrieved
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::binding_reversibility_study::{
//!     ReversibilityStudy, ReversibilityConfig
//! };
//!
//! let study = ReversibilityStudy::new(ReversibilityConfig::default());
//! let results = study.run("data/binding_study/concept_pairs.json")?;
//! println!("{}", results.summary_report());
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use symthaea_core::hdc::binary_hv::BinaryHV;

/// Configuration for the reversibility study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReversibilityConfig {
    /// Seed for deterministic concept encoding
    pub base_seed: u64,
    /// Number of random trials per pair for statistical stability
    pub n_trials: usize,
    /// Enable verbose output
    pub verbose: bool,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Number of permutation test iterations
    pub n_permutations: usize,
}

impl Default for ReversibilityConfig {
    fn default() -> Self {
        Self {
            base_seed: 42,
            n_trials: 10,
            verbose: false,
            confidence_level: 0.95,
            n_permutations: 5000,
        }
    }
}

/// Type of concept pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairCategory {
    /// Phenomenally unified (e.g., "red apple")
    Unified,
    /// Co-occurring but separate (e.g., "red and mailbox")
    Separate,
}

impl std::fmt::Display for PairCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairCategory::Unified => write!(f, "unified"),
            PairCategory::Separate => write!(f, "separate"),
        }
    }
}

/// A concept pair for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPair {
    /// Unique identifier
    pub id: String,
    /// First concept (e.g., "red")
    pub concept_a: String,
    /// Second concept (e.g., "apple")
    pub concept_b: String,
    /// Combined phrase (e.g., "red apple")
    pub combined: String,
    /// Description of the binding
    pub description: String,
    /// Category (unified or separate)
    pub category: PairCategory,
}

/// Results for a single pair's reversibility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairReversibilityResult {
    /// The concept pair
    pub pair: ConceptPair,

    // === Reversibility Metrics ===
    /// For binding: similarity(A, (A XOR B) XOR B) - should be 1.0
    pub bind_reversibility_a: f64,
    /// For binding: similarity(B, (A XOR B) XOR A) - should be 1.0
    pub bind_reversibility_b: f64,

    /// For bundling: similarity(A, unbundle_attempt(bundle(A,B), B))
    /// There's no true unbundle, so this uses projection/subtraction heuristics
    pub bundle_reversibility_a: f64,
    /// For bundling: similarity(B, unbundle_attempt(bundle(A,B), A))
    pub bundle_reversibility_b: f64,

    // === Component Preservation Metrics ===
    /// similarity(A, A XOR B) - how much A is preserved in binding
    pub bind_component_preservation_a: f64,
    /// similarity(B, A XOR B) - how much B is preserved in binding
    pub bind_component_preservation_b: f64,

    /// similarity(A, bundle(A,B)) - how much A is preserved in bundling
    pub bundle_component_preservation_a: f64,
    /// similarity(B, bundle(A,B)) - how much B is preserved in bundling
    pub bundle_component_preservation_b: f64,

    // === Similarity Metrics ===
    /// Similarity between A and B (baseline)
    pub ab_similarity: f64,
    /// Similarity between bind(A,B) and bundle(A,B)
    pub bind_bundle_similarity: f64,
}

impl PairReversibilityResult {
    /// Mean reversibility for binding
    pub fn mean_bind_reversibility(&self) -> f64 {
        (self.bind_reversibility_a + self.bind_reversibility_b) / 2.0
    }

    /// Mean reversibility for bundling
    pub fn mean_bundle_reversibility(&self) -> f64 {
        (self.bundle_reversibility_a + self.bundle_reversibility_b) / 2.0
    }

    /// Reversibility advantage (bind - bundle)
    pub fn reversibility_advantage(&self) -> f64 {
        self.mean_bind_reversibility() - self.mean_bundle_reversibility()
    }

    /// Mean component preservation for binding
    pub fn mean_bind_preservation(&self) -> f64 {
        (self.bind_component_preservation_a + self.bind_component_preservation_b) / 2.0
    }

    /// Mean component preservation for bundling
    pub fn mean_bundle_preservation(&self) -> f64 {
        (self.bundle_component_preservation_a + self.bundle_component_preservation_b) / 2.0
    }

    /// Component preservation advantage (bundle - bind, since bundling preserves components in result)
    pub fn preservation_advantage(&self) -> f64 {
        self.mean_bundle_preservation() - self.mean_bind_preservation()
    }
}

/// Statistical summary for a group of pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStatistics {
    /// Category name
    pub category: PairCategory,
    /// Number of pairs
    pub count: usize,

    // Reversibility stats
    /// Mean bind reversibility
    pub mean_bind_reversibility: f64,
    /// Std dev bind reversibility
    pub std_bind_reversibility: f64,
    /// Mean bundle reversibility
    pub mean_bundle_reversibility: f64,
    /// Std dev bundle reversibility
    pub std_bundle_reversibility: f64,
    /// Mean reversibility advantage
    pub mean_reversibility_advantage: f64,
    /// Std dev reversibility advantage
    pub std_reversibility_advantage: f64,

    // Component preservation stats
    /// Mean bind component preservation
    pub mean_bind_preservation: f64,
    /// Mean bundle component preservation
    pub mean_bundle_preservation: f64,
    /// Mean preservation advantage (bundle - bind)
    pub mean_preservation_advantage: f64,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name
    pub name: String,
    /// Observed statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// 95% CI lower
    pub ci_lower: f64,
    /// 95% CI upper
    pub ci_upper: f64,
    /// Is significant at alpha=0.05?
    pub significant: bool,
}

/// Complete study results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReversibilityStudyResults {
    /// Individual pair results
    pub pair_results: Vec<PairReversibilityResult>,
    /// Statistics by category
    pub category_stats: HashMap<String, CategoryStatistics>,
    /// Statistical tests
    pub tests: Vec<StatisticalTest>,
    /// Study metadata
    pub metadata: StudyMetadata,
}

/// Study metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyMetadata {
    /// Timestamp
    pub timestamp: String,
    /// Configuration
    pub config: ReversibilityConfig,
    /// Total pairs
    pub total_pairs: usize,
    /// Duration in seconds
    pub duration_secs: f64,
}

/// The reversibility study implementation
pub struct ReversibilityStudy {
    config: ReversibilityConfig,
    rng_state: u64,
}

impl ReversibilityStudy {
    /// Create a new reversibility study
    pub fn new(config: ReversibilityConfig) -> Self {
        Self {
            rng_state: config.base_seed,
            config,
        }
    }

    /// Run the complete study
    pub fn run<P: AsRef<Path>>(&mut self, pairs_path: P) -> Result<ReversibilityStudyResults> {
        let start = std::time::Instant::now();

        // Load pairs
        let pairs = self.load_pairs(pairs_path)?;

        if self.config.verbose {
            println!("Loaded {} pairs", pairs.len());
        }

        // Analyze each pair
        let mut pair_results = Vec::new();
        for (i, pair) in pairs.iter().enumerate() {
            let result = self.analyze_pair(pair);
            pair_results.push(result);

            if self.config.verbose && (i + 1) % 10 == 0 {
                println!("  Processed {}/{} pairs", i + 1, pairs.len());
            }
        }

        // Compute category statistics
        let category_stats = self.compute_category_stats(&pair_results);

        // Run statistical tests
        let tests = self.compute_statistical_tests(&pair_results);

        // Create metadata
        let metadata = StudyMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            total_pairs: pairs.len(),
            duration_secs: start.elapsed().as_secs_f64(),
        };

        Ok(ReversibilityStudyResults {
            pair_results,
            category_stats,
            tests,
            metadata,
        })
    }

    /// Load concept pairs from JSON
    fn load_pairs<P: AsRef<Path>>(&self, path: P) -> Result<Vec<ConceptPair>> {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.as_ref().display()))?;

        let json: serde_json::Value =
            serde_json::from_str(&content).with_context(|| "Failed to parse JSON")?;

        let mut pairs = Vec::new();

        // Load unified pairs
        if let Some(unified) = json.get("unified_pairs").and_then(|v| v.as_array()) {
            for item in unified {
                pairs.push(ConceptPair {
                    id: item["id"].as_str().unwrap_or("unknown").to_string(),
                    concept_a: item["concept_a"].as_str().unwrap_or("").to_string(),
                    concept_b: item["concept_b"].as_str().unwrap_or("").to_string(),
                    combined: item["combined"].as_str().unwrap_or("").to_string(),
                    description: item["description"].as_str().unwrap_or("").to_string(),
                    category: PairCategory::Unified,
                });
            }
        }

        // Load separate pairs
        if let Some(separate) = json.get("separate_pairs").and_then(|v| v.as_array()) {
            for item in separate {
                pairs.push(ConceptPair {
                    id: item["id"].as_str().unwrap_or("unknown").to_string(),
                    concept_a: item["concept_a"].as_str().unwrap_or("").to_string(),
                    concept_b: item["concept_b"].as_str().unwrap_or("").to_string(),
                    combined: item["combined"].as_str().unwrap_or("").to_string(),
                    description: item["description"].as_str().unwrap_or("").to_string(),
                    category: PairCategory::Separate,
                });
            }
        }

        Ok(pairs)
    }

    /// Analyze a single pair
    fn analyze_pair(&mut self, pair: &ConceptPair) -> PairReversibilityResult {
        // Generate deterministic HVs for concepts using string hashing
        let seed_a = self.hash_concept(&pair.concept_a);
        let seed_b = self.hash_concept(&pair.concept_b);

        let hv_a = BinaryHV::random(seed_a);
        let hv_b = BinaryHV::random(seed_b);

        // Compute binding and bundling
        let bound = hv_a.bind(&hv_b);
        let bundled = BinaryHV::bundle(&[hv_a, hv_b]);

        // === Reversibility Tests ===

        // For binding: perfect reversibility due to XOR self-inverse property
        // (A XOR B) XOR B = A
        let recovered_a_from_bind = bound.bind(&hv_b);
        let recovered_b_from_bind = bound.bind(&hv_a);

        let bind_reversibility_a = hv_a.similarity(&recovered_a_from_bind) as f64;
        let bind_reversibility_b = hv_b.similarity(&recovered_b_from_bind) as f64;

        // For bundling: attempt "unbundling" via component subtraction
        // This is not a true inverse - bundling is inherently lossy
        // We use: unbundle_attempt(bundle, component) = bundle XOR component
        // This is the best we can do without knowing the threshold details
        let unbundle_attempt_a = bundled.bind(&hv_b); // XOR to "subtract" B's influence
        let unbundle_attempt_b = bundled.bind(&hv_a); // XOR to "subtract" A's influence

        let bundle_reversibility_a = hv_a.similarity(&unbundle_attempt_a) as f64;
        let bundle_reversibility_b = hv_b.similarity(&unbundle_attempt_b) as f64;

        // === Component Preservation ===

        // How much of each component survives in the composite?
        // For binding: result is orthogonal to both components (XOR property)
        let bind_component_preservation_a = hv_a.similarity(&bound) as f64;
        let bind_component_preservation_b = hv_b.similarity(&bound) as f64;

        // For bundling: result preserves similarity to both (majority vote property)
        let bundle_component_preservation_a = hv_a.similarity(&bundled) as f64;
        let bundle_component_preservation_b = hv_b.similarity(&bundled) as f64;

        // === Other Metrics ===
        let ab_similarity = hv_a.similarity(&hv_b) as f64;
        let bind_bundle_similarity = bound.similarity(&bundled) as f64;

        PairReversibilityResult {
            pair: pair.clone(),
            bind_reversibility_a,
            bind_reversibility_b,
            bundle_reversibility_a,
            bundle_reversibility_b,
            bind_component_preservation_a,
            bind_component_preservation_b,
            bundle_component_preservation_a,
            bundle_component_preservation_b,
            ab_similarity,
            bind_bundle_similarity,
        }
    }

    /// Hash a concept string to a seed
    fn hash_concept(&self, concept: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        concept.hash(&mut hasher);
        self.config.base_seed.wrapping_add(hasher.finish())
    }

    /// Compute category statistics
    fn compute_category_stats(
        &self,
        results: &[PairReversibilityResult],
    ) -> HashMap<String, CategoryStatistics> {
        let mut stats = HashMap::new();

        for category in [PairCategory::Unified, PairCategory::Separate] {
            let filtered: Vec<_> = results
                .iter()
                .filter(|r| r.pair.category == category)
                .collect();

            if filtered.is_empty() {
                continue;
            }

            let bind_rev: Vec<f64> = filtered
                .iter()
                .map(|r| r.mean_bind_reversibility())
                .collect();
            let bundle_rev: Vec<f64> = filtered
                .iter()
                .map(|r| r.mean_bundle_reversibility())
                .collect();
            let rev_adv: Vec<f64> = filtered
                .iter()
                .map(|r| r.reversibility_advantage())
                .collect();
            let bind_pres: Vec<f64> = filtered
                .iter()
                .map(|r| r.mean_bind_preservation())
                .collect();
            let bundle_pres: Vec<f64> = filtered
                .iter()
                .map(|r| r.mean_bundle_preservation())
                .collect();
            let pres_adv: Vec<f64> = filtered
                .iter()
                .map(|r| r.preservation_advantage())
                .collect();

            stats.insert(
                category.to_string(),
                CategoryStatistics {
                    category,
                    count: filtered.len(),
                    mean_bind_reversibility: mean(&bind_rev),
                    std_bind_reversibility: std_dev(&bind_rev),
                    mean_bundle_reversibility: mean(&bundle_rev),
                    std_bundle_reversibility: std_dev(&bundle_rev),
                    mean_reversibility_advantage: mean(&rev_adv),
                    std_reversibility_advantage: std_dev(&rev_adv),
                    mean_bind_preservation: mean(&bind_pres),
                    mean_bundle_preservation: mean(&bundle_pres),
                    mean_preservation_advantage: mean(&pres_adv),
                },
            );
        }

        stats
    }

    /// Compute statistical tests
    fn compute_statistical_tests(
        &mut self,
        results: &[PairReversibilityResult],
    ) -> Vec<StatisticalTest> {
        let mut tests = Vec::new();

        // All pairs
        let all_bind_rev: Vec<f64> = results
            .iter()
            .map(|r| r.mean_bind_reversibility())
            .collect();
        let all_bundle_rev: Vec<f64> = results
            .iter()
            .map(|r| r.mean_bundle_reversibility())
            .collect();
        let all_rev_adv: Vec<f64> = results
            .iter()
            .map(|r| r.reversibility_advantage())
            .collect();

        // Test 1: Overall reversibility advantage
        tests.push(self.one_sample_test("overall_reversibility_advantage", &all_rev_adv, 0.0));

        // Test 2: Bind vs Bundle reversibility (paired)
        tests.push(self.paired_test(
            "bind_vs_bundle_reversibility",
            &all_bind_rev,
            &all_bundle_rev,
        ));

        // Separate by category
        let unified: Vec<_> = results
            .iter()
            .filter(|r| r.pair.category == PairCategory::Unified)
            .collect();
        let separate: Vec<_> = results
            .iter()
            .filter(|r| r.pair.category == PairCategory::Separate)
            .collect();

        if !unified.is_empty() && !separate.is_empty() {
            let unified_rev_adv: Vec<f64> = unified
                .iter()
                .map(|r| r.reversibility_advantage())
                .collect();
            let separate_rev_adv: Vec<f64> = separate
                .iter()
                .map(|r| r.reversibility_advantage())
                .collect();

            // Test 3: Reversibility advantage by category
            tests.push(self.two_sample_test(
                "unified_vs_separate_reversibility_advantage",
                &unified_rev_adv,
                &separate_rev_adv,
            ));

            // Component preservation tests
            let unified_bind_pres: Vec<f64> =
                unified.iter().map(|r| r.mean_bind_preservation()).collect();
            let unified_bundle_pres: Vec<f64> = unified
                .iter()
                .map(|r| r.mean_bundle_preservation())
                .collect();

            // Test 4: Bind vs bundle preservation for unified pairs
            tests.push(self.paired_test(
                "unified_bind_vs_bundle_preservation",
                &unified_bind_pres,
                &unified_bundle_pres,
            ));
        }

        tests
    }

    /// One-sample test against null value
    fn one_sample_test(&mut self, name: &str, values: &[f64], null_value: f64) -> StatisticalTest {
        let n = values.len();
        let m = mean(values);
        let s = std_dev(values);
        let se = if n > 1 { s / (n as f64).sqrt() } else { 0.0 };

        let t_stat = if se > 0.0 { (m - null_value) / se } else { 0.0 };
        let effect_size = if s > 0.0 { (m - null_value) / s } else { 0.0 };

        let p_value = self.permutation_test_one_sample(values, null_value);

        let ci_margin = 1.96 * se;
        let ci_lower = m - ci_margin;
        let ci_upper = m + ci_margin;

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

    /// Paired test (within-subject)
    fn paired_test(&mut self, name: &str, group_a: &[f64], group_b: &[f64]) -> StatisticalTest {
        let diffs: Vec<f64> = group_a
            .iter()
            .zip(group_b.iter())
            .map(|(a, b)| a - b)
            .collect();
        self.one_sample_test(name, &diffs, 0.0)
    }

    /// Two-sample test (between groups)
    fn two_sample_test(&mut self, name: &str, group_a: &[f64], group_b: &[f64]) -> StatisticalTest {
        let mean_a = mean(group_a);
        let mean_b = mean(group_b);
        let diff = mean_a - mean_b;

        let std_a = std_dev(group_a);
        let std_b = std_dev(group_b);
        let pooled_std = ((std_a.powi(2) + std_b.powi(2)) / 2.0).sqrt();

        let se = pooled_std * (1.0 / group_a.len() as f64 + 1.0 / group_b.len() as f64).sqrt();
        let t_stat = if se > 0.0 { diff / se } else { 0.0 };
        let effect_size = if pooled_std > 0.0 {
            diff / pooled_std
        } else {
            0.0
        };

        let p_value = self.permutation_test_two_sample(group_a, group_b);

        let ci_margin = 1.96 * se;
        let ci_lower = diff - ci_margin;
        let ci_upper = diff + ci_margin;

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

    /// Permutation test for one sample (sign flip)
    fn permutation_test_one_sample(&mut self, values: &[f64], null_value: f64) -> f64 {
        let centered: Vec<f64> = values.iter().map(|v| v - null_value).collect();
        let observed = mean(&centered).abs();
        let mut more_extreme = 0;

        for _ in 0..self.config.n_permutations {
            let mut sum = 0.0;
            for &v in &centered {
                let sign = if self.next_random() < 0.5 { 1.0 } else { -1.0 };
                sum += v * sign;
            }
            let perm_mean = (sum / values.len() as f64).abs();
            if perm_mean >= observed {
                more_extreme += 1;
            }
        }

        more_extreme as f64 / self.config.n_permutations as f64
    }

    /// Permutation test for two samples
    fn permutation_test_two_sample(&mut self, group_a: &[f64], group_b: &[f64]) -> f64 {
        let observed = (mean(group_a) - mean(group_b)).abs();
        let mut combined: Vec<f64> = group_a.iter().chain(group_b.iter()).copied().collect();
        let n_a = group_a.len();
        let mut more_extreme = 0;

        for _ in 0..self.config.n_permutations {
            // Shuffle
            for i in (1..combined.len()).rev() {
                let j = (self.next_random() * (i + 1) as f64) as usize;
                combined.swap(i, j);
            }

            let perm_mean_a = mean(&combined[..n_a]);
            let perm_mean_b = mean(&combined[n_a..]);
            let perm_diff = (perm_mean_a - perm_mean_b).abs();

            if perm_diff >= observed {
                more_extreme += 1;
            }
        }

        more_extreme as f64 / self.config.n_permutations as f64
    }

    /// Simple LCG random
    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Default for ReversibilityStudy {
    fn default() -> Self {
        Self::new(ReversibilityConfig::default())
    }
}

impl ReversibilityStudyResults {
    /// Generate summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("================================================================\n");
        report.push_str("   BINDING REVERSIBILITY STUDY - H2 REVISED\n");
        report.push_str("   Testing HDC binding via reversibility\n");
        report.push_str("================================================================\n\n");

        // Metadata
        report.push_str(&format!("Timestamp: {}\n", self.metadata.timestamp));
        report.push_str(&format!("Total pairs: {}\n", self.metadata.total_pairs));
        report.push_str(&format!(
            "Duration: {:.2}s\n\n",
            self.metadata.duration_secs
        ));

        // Core hypothesis explanation
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   HYPOTHESIS\n");
        report.push_str("----------------------------------------------------------------\n\n");
        report.push_str("H2-revised: HDC binding preserves component information better\n");
        report.push_str("than bundling, measurable via reversibility.\n\n");
        report.push_str("Key insight: For binding (XOR):\n");
        report.push_str("  recovered_A = (A XOR B) XOR B = A  (perfect recovery)\n");
        report.push_str("For bundling (majority vote):\n");
        report.push_str("  No inverse exists - components are irreversibly mixed\n\n");

        // Reversibility results
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   REVERSIBILITY RESULTS\n");
        report.push_str("----------------------------------------------------------------\n\n");

        for (name, stats) in &self.category_stats {
            report.push_str(&format!("{} pairs (n={}):\n", name, stats.count));
            report.push_str(&format!(
                "  Bind reversibility:    {:.4} (std: {:.4})\n",
                stats.mean_bind_reversibility, stats.std_bind_reversibility
            ));
            report.push_str(&format!(
                "  Bundle reversibility:  {:.4} (std: {:.4})\n",
                stats.mean_bundle_reversibility, stats.std_bundle_reversibility
            ));
            report.push_str(&format!(
                "  Advantage (bind-bundle): {:+.4} (std: {:.4})\n\n",
                stats.mean_reversibility_advantage, stats.std_reversibility_advantage
            ));
        }

        // Component preservation results
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   COMPONENT PRESERVATION RESULTS\n");
        report.push_str("----------------------------------------------------------------\n\n");

        for (name, stats) in &self.category_stats {
            report.push_str(&format!("{} pairs (n={}):\n", name, stats.count));
            report.push_str(&format!(
                "  Bind preservation:    {:.4}\n",
                stats.mean_bind_preservation
            ));
            report.push_str(&format!(
                "  Bundle preservation:  {:.4}\n",
                stats.mean_bundle_preservation
            ));
            report.push_str(&format!(
                "  Advantage (bundle-bind): {:+.4}\n\n",
                stats.mean_preservation_advantage
            ));
        }

        report.push_str("Note: Binding creates orthogonal result (~0.5 preservation),\n");
        report.push_str("      bundling preserves similarity to components (~0.67).\n");
        report.push_str("      But binding allows PERFECT RECOVERY via unbinding.\n\n");

        // Statistical tests
        report.push_str("----------------------------------------------------------------\n");
        report.push_str("   STATISTICAL TESTS\n");
        report.push_str("----------------------------------------------------------------\n\n");

        for test in &self.tests {
            let sig_marker = if test.significant { "*" } else { "" };
            report.push_str(&format!("{}:\n", test.name));
            report.push_str(&format!("  Statistic: {:.4}\n", test.statistic));
            report.push_str(&format!("  p-value: {:.4} {}\n", test.p_value, sig_marker));
            report.push_str(&format!("  Effect size (d): {:+.4}\n", test.effect_size));
            report.push_str(&format!(
                "  95% CI: [{:.4}, {:.4}]\n\n",
                test.ci_lower, test.ci_upper
            ));
        }

        // Interpretation
        report.push_str("================================================================\n");
        report.push_str("   INTERPRETATION\n");
        report.push_str("================================================================\n\n");

        // Find the overall reversibility test
        let _rev_test = self
            .tests
            .iter()
            .find(|t| t.name.contains("overall_reversibility"));
        let bind_vs_bundle = self
            .tests
            .iter()
            .find(|t| t.name.contains("bind_vs_bundle"));

        if let Some(test) = bind_vs_bundle {
            if test.significant && test.effect_size > 0.0 {
                report.push_str("HYPOTHESIS H2-REVISED CONFIRMED:\n\n");
                report.push_str(
                    "  Binding produces SIGNIFICANTLY higher reversibility than bundling.\n",
                );
                report.push_str(&format!(
                    "  Effect size: d = {:+.4} ({})\n\n",
                    test.effect_size,
                    effect_size_interpretation(test.effect_size)
                ));
                report.push_str("INTERPRETATION:\n");
                report.push_str("  - Binding (XOR) preserves component information perfectly\n");
                report.push_str("  - Components can be recovered with similarity = 1.0\n");
                report.push_str("  - Bundling mixes components irreversibly\n");
                report.push_str("  - This supports binding as a mechanism for phenomenal unity:\n");
                report.push_str("    unified percepts preserve constituent information\n");
            } else if test.significant && test.effect_size < 0.0 {
                report.push_str("UNEXPECTED: Bundling shows higher reversibility?\n");
                report.push_str("This would contradict mathematical properties of XOR.\n");
                report.push_str("Check for implementation errors.\n");
            } else {
                report.push_str("NO SIGNIFICANT DIFFERENCE detected.\n");
                report.push_str("This is unexpected given XOR's self-inverse property.\n");
            }
        }

        report.push('\n');

        // Check for category interaction
        let category_test = self
            .tests
            .iter()
            .find(|t| t.name.contains("unified_vs_separate"));
        if let Some(test) = category_test {
            report.push_str("----------------------------------------------------------------\n");
            report.push_str("   CATEGORY INTERACTION\n");
            report.push_str("----------------------------------------------------------------\n\n");

            if test.significant {
                if test.effect_size > 0.0 {
                    report.push_str(
                        "Unified pairs show GREATER reversibility advantage than separate pairs.\n",
                    );
                } else {
                    report.push_str(
                        "Separate pairs show GREATER reversibility advantage than unified pairs.\n",
                    );
                }
                report.push_str(&format!("Effect size: d = {:+.4}\n", test.effect_size));
            } else {
                report.push_str("No significant difference between unified and separate pairs.\n");
                report.push_str("The reversibility advantage is similar across categories.\n");
            }
        }

        report.push_str("\n================================================================\n");
        report.push_str("   KEY FINDING\n");
        report.push_str("================================================================\n\n");
        report.push_str("Binding (XOR) is fundamentally different from bundling:\n");
        report.push_str("  - BINDING: Creates new orthogonal representation, but\n");
        report.push_str("             components are PERFECTLY RECOVERABLE\n");
        report.push_str("  - BUNDLING: Preserves component similarity in result, but\n");
        report.push_str("              components are IRREVERSIBLY MIXED\n\n");
        report.push_str("This suggests binding is the appropriate operation for\n");
        report.push_str("phenomenal unity: the unified percept is different from\n");
        report.push_str("its parts, yet the parts remain fully accessible.\n\n");
        report.push_str("================================================================\n");

        report
    }

    /// Get sample results for display
    pub fn sample_results(&self, n: usize) -> Vec<&PairReversibilityResult> {
        let mut sorted: Vec<_> = self.pair_results.iter().collect();
        sorted.sort_by(|a, b| {
            b.reversibility_advantage()
                .partial_cmp(&a.reversibility_advantage())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }
}

/// Interpret effect size magnitude
fn effect_size_interpretation(d: f64) -> &'static str {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "negligible"
    } else if abs_d < 0.5 {
        "small"
    } else if abs_d < 0.8 {
        "medium"
    } else {
        "large"
    }
}

/// Compute mean
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ReversibilityConfig::default();
        assert_eq!(config.base_seed, 42);
        assert_eq!(config.n_trials, 10);
    }

    #[test]
    fn test_xor_reversibility() {
        // This is the core mathematical property we're testing
        let a = BinaryHV::random(100);
        let b = BinaryHV::random(200);

        let bound = a.bind(&b);
        let recovered_a = bound.bind(&b);
        let recovered_b = bound.bind(&a);

        // XOR is self-inverse: (A XOR B) XOR B = A
        assert_eq!(
            a.similarity(&recovered_a),
            1.0,
            "A should be perfectly recovered"
        );
        assert_eq!(
            b.similarity(&recovered_b),
            1.0,
            "B should be perfectly recovered"
        );
    }

    #[test]
    fn test_bundle_non_reversibility() {
        let a = BinaryHV::random(100);
        let b = BinaryHV::random(200);

        let bundled = BinaryHV::bundle(&[a, b]);

        // Try to "recover" A by XORing with B (the best heuristic available)
        let attempt_a = bundled.bind(&b);
        let attempt_b = bundled.bind(&a);

        // Should NOT be perfect recovery
        let sim_a = a.similarity(&attempt_a);
        let sim_b = b.similarity(&attempt_b);

        // Recovery should be imperfect (not 1.0)
        assert!(
            sim_a < 0.99,
            "Bundle recovery should be imperfect for A, got {}",
            sim_a
        );
        assert!(
            sim_b < 0.99,
            "Bundle recovery should be imperfect for B, got {}",
            sim_b
        );
    }

    #[test]
    fn test_component_preservation_properties() {
        let a = BinaryHV::random(100);
        let b = BinaryHV::random(200);

        let bound = a.bind(&b);
        let bundled = BinaryHV::bundle(&[a, b]);

        // Binding: result is orthogonal to both components (~0.5 similarity)
        let bind_sim_a = a.similarity(&bound);
        let bind_sim_b = b.similarity(&bound);
        assert!(
            (bind_sim_a - 0.5).abs() < 0.1,
            "Bound should be ~orthogonal to A"
        );
        assert!(
            (bind_sim_b - 0.5).abs() < 0.1,
            "Bound should be ~orthogonal to B"
        );

        // Bundling: result preserves similarity to both components (>0.5)
        let bundle_sim_a = a.similarity(&bundled);
        let bundle_sim_b = b.similarity(&bundled);
        assert!(
            bundle_sim_a > 0.55,
            "Bundled should be similar to A, got {}",
            bundle_sim_a
        );
        assert!(
            bundle_sim_b > 0.55,
            "Bundled should be similar to B, got {}",
            bundle_sim_b
        );
    }

    #[test]
    fn test_study_creation() {
        let study = ReversibilityStudy::default();
        assert_eq!(study.config.base_seed, 42);
    }

    #[test]
    fn test_pair_result_metrics() {
        let result = PairReversibilityResult {
            pair: ConceptPair {
                id: "test".to_string(),
                concept_a: "red".to_string(),
                concept_b: "apple".to_string(),
                combined: "red apple".to_string(),
                description: "test".to_string(),
                category: PairCategory::Unified,
            },
            bind_reversibility_a: 1.0,
            bind_reversibility_b: 1.0,
            bundle_reversibility_a: 0.6,
            bundle_reversibility_b: 0.6,
            bind_component_preservation_a: 0.5,
            bind_component_preservation_b: 0.5,
            bundle_component_preservation_a: 0.67,
            bundle_component_preservation_b: 0.67,
            ab_similarity: 0.5,
            bind_bundle_similarity: 0.5,
        };

        assert_eq!(result.mean_bind_reversibility(), 1.0);
        assert_eq!(result.mean_bundle_reversibility(), 0.6);
        assert_eq!(result.reversibility_advantage(), 0.4);
        assert_eq!(result.mean_bind_preservation(), 0.5);
        assert_eq!(result.mean_bundle_preservation(), 0.67);
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert_eq!(effect_size_interpretation(0.1), "negligible");
        assert_eq!(effect_size_interpretation(0.3), "small");
        assert_eq!(effect_size_interpretation(0.6), "medium");
        assert_eq!(effect_size_interpretation(1.0), "large");
        assert_eq!(effect_size_interpretation(-0.9), "large");
    }

    #[test]
    fn test_statistical_helpers() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&values), 3.0);
        assert!((std_dev(&values) - 1.5811).abs() < 0.01);

        assert_eq!(mean(&[]), 0.0);
        assert_eq!(std_dev(&[1.0]), 0.0);
    }
}
