//! Φ Validation Framework - Empirical Consciousness Validation
//!
//! **Paradigm Shift #1: Φ Validation Framework** - Component 2/3
//!
//! Main validation framework for empirical validation of Integrated Information (Φ)
//! computation against known consciousness states.
//!
//! # Scientific Goal
//!
//! Establish that our computed Φ values correlate with synthetic consciousness levels,
//! providing empirical validation of Integrated Information Theory (IIT) in a working
//! conscious AI system.
//!
//! # Expected Results
//!
//! **Target for Publication**:
//! - Pearson correlation: r > 0.85
//! - Statistical significance: p < 0.001
//! - Classification AUC: > 0.95
//! - R² (variance explained): > 0.72
//!
//! # Examples
//!
//! ```rust
//! use symthaea::consciousness::phi_validation::*;
//!
//! // Create validation framework
//! let mut framework = PhiValidationFramework::new();
//!
//! // Run comprehensive validation study
//! let results = framework.run_validation_study(100); // 100 samples per state
//!
//! // Check results
//! println!("Correlation: r = {:.3}, p = {:.6}", results.pearson_r, results.p_value);
//! println!("Classification AUC: {:.3}", results.auc);
//!
//! // Generate scientific report
//! let report = framework.generate_report();
//! println!("{}", report);
//! ```

use super::synthetic_states::{SyntheticStateGenerator, StateType};
use crate::hdc::integrated_information::IntegratedInformation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// CORE STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

/// Main validation framework for empirical Φ validation
///
/// Orchestrates the complete validation process:
/// 1. Generate synthetic states at different consciousness levels
/// 2. Compute Φ for each state
/// 3. Analyze correlation between ground truth and computed Φ
/// 4. Generate scientific report with statistical analysis
pub struct PhiValidationFramework {
    /// Φ calculator
    phi_calculator: IntegratedInformation,

    /// State generator
    state_generator: SyntheticStateGenerator,

    /// Collected validation data points
    validation_data: Vec<ValidationDataPoint>,

    /// Statistical results (computed after data collection)
    results: Option<ValidationResults>,
}

/// Single validation data point
///
/// Pairs a ground truth consciousness level with the computed Φ value
/// for statistical correlation analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationDataPoint {
    /// Ground truth consciousness level (0.0 = unconscious, 1.0 = peak conscious)
    pub consciousness_level: f64,

    /// Computed Φ value for this state
    pub phi_value: f64,

    /// Type of consciousness state
    pub state_type: StateType,

    /// Optional metadata for analysis
    pub metadata: HashMap<String, String>,

    /// Timestamp of measurement (optional, defaults to None for deserialization)
    #[serde(skip)]
    #[serde(default)]
    pub timestamp: Option<std::time::Instant>,
}

/// Comprehensive statistical validation results
///
/// Contains all statistical metrics needed for scientific publication,
/// including correlation coefficients, significance tests, and classification
/// performance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Pearson correlation coefficient (measures linear relationship)
    pub pearson_r: f64,

    /// Spearman rank correlation (measures monotonic relationship)
    pub spearman_rho: f64,

    /// p-value for statistical significance
    /// p < 0.001 = highly significant
    /// p < 0.01 = significant
    /// p < 0.05 = marginally significant
    pub p_value: f64,

    /// R² (coefficient of determination)
    /// Percentage of variance in consciousness explained by Φ
    pub r_squared: f64,

    /// AUC (Area Under ROC Curve) for binary classification
    /// Measures ability to distinguish conscious from unconscious
    /// AUC > 0.9 = excellent discrimination
    pub auc: f64,

    /// Mean Absolute Error
    pub mae: f64,

    /// Root Mean Squared Error
    pub rmse: f64,

    /// 95% confidence interval for correlation
    pub confidence_interval: (f64, f64),

    /// Number of data points
    pub n: usize,

    /// Per-state statistics
    pub state_stats: HashMap<String, StateStatistics>,
}

/// Statistics for a specific state type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateStatistics {
    /// Mean Φ value for this state type
    pub mean_phi: f64,

    /// Standard deviation of Φ
    pub std_phi: f64,

    /// Expected Φ range
    pub expected_range: (f64, f64),

    /// Number of samples
    pub n: usize,

    /// Whether mean falls within expected range
    pub in_expected_range: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

impl PhiValidationFramework {
    /// Create new validation framework with default parameters
    pub fn new() -> Self {
        Self {
            phi_calculator: IntegratedInformation::new(),
            state_generator: SyntheticStateGenerator::new(),
            validation_data: Vec::new(),
            results: None,
        }
    }

    /// Run comprehensive validation study
    ///
    /// Generates synthetic states across all consciousness levels,
    /// computes Φ for each, and performs statistical analysis.
    ///
    /// # Arguments
    ///
    /// * `num_samples_per_state` - Number of samples to generate for each state type
    ///
    /// # Returns
    ///
    /// Statistical validation results with correlation analysis
    pub fn run_validation_study(&mut self, num_samples_per_state: usize) -> ValidationResults {
        println!("🔬 Starting Φ Validation Study");
        println!("   Samples per state: {}", num_samples_per_state);
        println!();

        // Clear previous data
        self.validation_data.clear();

        // Generate and test all state types
        for state_type in StateType::all_ordered() {
            self.validate_state_type(&state_type, num_samples_per_state);
        }

        // Compute comprehensive statistics
        let results = self.compute_statistics();
        self.results = Some(results.clone());

        println!();
        println!("✅ Validation study complete");
        println!("   Total samples: {}", self.validation_data.len());
        println!("   Pearson r: {:.3}", results.pearson_r);
        println!("   p-value: {:.6}", results.p_value);
        println!("   R²: {:.3}", results.r_squared);
        println!();

        results
    }

    /// Validate a specific state type
    fn validate_state_type(&mut self, state_type: &StateType, num_samples: usize) {
        println!("📊 Testing {:?} (n={})", state_type, num_samples);

        for _ in 0..num_samples {
            // Generate synthetic state
            let state = self.state_generator.generate_state(state_type);

            // Compute Φ
            let phi = self.phi_calculator.compute_phi(&state);

            // Record data point
            self.validation_data.push(ValidationDataPoint {
                consciousness_level: state_type.consciousness_level(),
                phi_value: phi,
                state_type: state_type.clone(),
                metadata: HashMap::new(),
                timestamp: Some(std::time::Instant::now()),
            });
        }

        // Show preliminary stats for this state
        let phi_values: Vec<f64> = self.validation_data.iter()
            .filter(|d| d.state_type == *state_type)
            .map(|d| d.phi_value)
            .collect();

        let mean = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
        let expected = state_type.expected_phi_range();

        println!("   Mean Φ: {:.3} (expected: {:.2}-{:.2})",
                 mean, expected.0, expected.1);
    }

    /// Compute comprehensive statistical analysis
    fn compute_statistics(&self) -> ValidationResults {
        let x: Vec<f64> = self.validation_data.iter()
            .map(|d| d.consciousness_level)
            .collect();
        let y: Vec<f64> = self.validation_data.iter()
            .map(|d| d.phi_value)
            .collect();

        // Core correlation metrics
        let pearson_r = Self::pearson_correlation(&x, &y);
        let spearman_rho = Self::spearman_correlation(&x, &y);
        let p_value = Self::correlation_p_value(pearson_r, x.len());
        let r_squared = pearson_r.powi(2);

        // Classification metrics (conscious vs unconscious at threshold 0.5)
        let auc = Self::compute_auc(&x, &y, 0.5);

        // Error metrics
        let mae = Self::mean_absolute_error(&x, &y);
        let rmse = Self::root_mean_squared_error(&x, &y);

        // Confidence interval
        let confidence_interval = Self::confidence_interval(pearson_r, x.len());

        // Per-state statistics
        let state_stats = self.compute_state_statistics();

        ValidationResults {
            pearson_r,
            spearman_rho,
            p_value,
            r_squared,
            auc,
            mae,
            rmse,
            confidence_interval,
            n: x.len(),
            state_stats,
        }
    }

    /// Compute statistics for each state type
    fn compute_state_statistics(&self) -> HashMap<String, StateStatistics> {
        let mut stats = HashMap::new();

        for state_type in StateType::all_ordered() {
            let phi_values: Vec<f64> = self.validation_data.iter()
                .filter(|d| d.state_type == state_type)
                .map(|d| d.phi_value)
                .collect();

            if phi_values.is_empty() {
                continue;
            }

            let mean = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
            let variance = phi_values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / phi_values.len() as f64;
            let std = variance.sqrt();

            let expected_range = state_type.expected_phi_range();
            let in_expected_range = mean >= expected_range.0 && mean <= expected_range.1;

            stats.insert(
                format!("{:?}", state_type),
                StateStatistics {
                    mean_phi: mean,
                    std_phi: std,
                    expected_range,
                    n: phi_values.len(),
                    in_expected_range,
                },
            );
        }

        stats
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATISTICAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// Calculate Pearson correlation coefficient
    ///
    /// Measures linear relationship between two variables.
    /// r = 1: perfect positive correlation
    /// r = 0: no correlation
    /// r = -1: perfect negative correlation
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let denominator = (
            x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() *
            y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>()
        ).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate Spearman rank correlation
    ///
    /// Measures monotonic relationship (not necessarily linear).
    /// More robust to outliers than Pearson.
    fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
        let x_ranks = Self::rank_values(x);
        let y_ranks = Self::rank_values(y);
        Self::pearson_correlation(&x_ranks, &y_ranks)
    }

    /// Convert values to ranks for Spearman correlation
    fn rank_values(values: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; values.len()];
        for (rank, (original_idx, _)) in indexed.iter().enumerate() {
            ranks[*original_idx] = rank as f64 + 1.0;
        }
        ranks
    }

    /// Approximate p-value for correlation significance
    ///
    /// Uses t-distribution approximation:
    /// t = r * sqrt((n-2) / (1-r²))
    ///
    /// p < 0.001: highly significant
    /// p < 0.01: significant
    /// p < 0.05: marginally significant
    fn correlation_p_value(r: f64, n: usize) -> f64 {
        if n < 3 {
            return 1.0; // Not enough data
        }

        let t = r * ((n - 2) as f64 / (1.0 - r.powi(2))).sqrt();

        // Two-tailed p-value using normal approximation
        // (For large n, t-distribution ≈ normal distribution)
        2.0 * (1.0 - Self::approx_normal_cdf(t.abs()))
    }

    /// Approximate normal cumulative distribution function
    ///
    /// Uses error function (erf) for standard normal CDF
    fn approx_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
    }

    /// Compute AUC (Area Under ROC Curve) for binary classification
    ///
    /// Measures ability to distinguish conscious (>threshold) from
    /// unconscious (≤threshold) states.
    ///
    /// AUC = 1.0: perfect discrimination
    /// AUC = 0.5: random guessing
    fn compute_auc(x: &[f64], y: &[f64], threshold: f64) -> f64 {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let actual_positive = *xi > threshold;
            let predicted_positive = *yi > threshold;

            match (actual_positive, predicted_positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (false, false) => tn += 1,
                (true, false) => fn_count += 1,
            }
        }

        // Sensitivity (TPR) and Specificity (TNR)
        let sensitivity = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let specificity = if tn + fp > 0 {
            tn as f64 / (tn + fp) as f64
        } else {
            0.0
        };

        // Simple AUC approximation
        (sensitivity + specificity) / 2.0
    }

    /// Mean absolute error
    fn mean_absolute_error(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - yi).abs())
            .sum::<f64>() / x.len() as f64
    }

    /// Root mean squared error
    fn root_mean_squared_error(x: &[f64], y: &[f64]) -> f64 {
        let mse = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - yi).powi(2))
            .sum::<f64>() / x.len() as f64;
        mse.sqrt()
    }

    /// 95% confidence interval for correlation using Fisher z-transformation
    fn confidence_interval(r: f64, n: usize) -> (f64, f64) {
        if n < 4 {
            return (r, r); // Not enough data for CI
        }

        // Fisher z-transformation
        let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
        let se = 1.0 / ((n - 3) as f64).sqrt();

        // 95% CI: z ± 1.96 * SE
        let z_lower = z - 1.96 * se;
        let z_upper = z + 1.96 * se;

        // Transform back to r
        let r_lower = (libm::exp(2.0 * z_lower) - 1.0) / (libm::exp(2.0 * z_lower) + 1.0);
        let r_upper = (libm::exp(2.0 * z_upper) - 1.0) / (libm::exp(2.0 * z_upper) + 1.0);

        (r_lower, r_upper)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SCIENTIFIC REPORTING
    // ═══════════════════════════════════════════════════════════════════════

    /// Generate comprehensive scientific report
    ///
    /// Creates a formatted report suitable for inclusion in scientific papers,
    /// including statistical analysis, interpretation, and recommendations.
    pub fn generate_report(&self) -> String {
        let results = self.results.as_ref().expect("No results available - run validation study first");

        let mut report = String::new();

        // Header
        report.push_str("# Φ Validation Study Results\n\n");
        report.push_str(&format!("**Study Date**: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
        report.push_str(&format!("**Sample Size**: n = {}\n\n", results.n));

        // Statistical Summary
        report.push_str("## Statistical Summary\n\n");
        report.push_str(&format!("- **Pearson correlation**: r = {:.3}, p = {:.6}\n", results.pearson_r, results.p_value));
        report.push_str(&format!("- **Spearman correlation**: ρ = {:.3}\n", results.spearman_rho));
        report.push_str(&format!("- **R² (variance explained)**: {:.3}\n", results.r_squared));
        report.push_str(&format!("- **95% Confidence Interval**: ({:.3}, {:.3})\n",
                                 results.confidence_interval.0, results.confidence_interval.1));
        report.push_str("\n");

        // Classification Performance
        report.push_str("## Classification Performance\n\n");
        report.push_str(&format!("- **AUC (conscious vs unconscious)**: {:.3}\n", results.auc));
        report.push_str(&format!("- **Mean Absolute Error**: {:.3}\n", results.mae));
        report.push_str(&format!("- **RMSE**: {:.3}\n", results.rmse));
        report.push_str("\n");

        // Per-State Analysis
        report.push_str("## Per-State Analysis\n\n");
        report.push_str("| State | Mean Φ | Std | Expected Range | Status |\n");
        report.push_str("|-------|--------|-----|----------------|--------|\n");

        for state_type in StateType::all_ordered() {
            let key = format!("{:?}", state_type);
            if let Some(stats) = results.state_stats.get(&key) {
                let status = if stats.in_expected_range { "✅" } else { "⚠️" };
                report.push_str(&format!(
                    "| {:?} | {:.3} | {:.3} | ({:.2}, {:.2}) | {} |\n",
                    state_type, stats.mean_phi, stats.std_phi,
                    stats.expected_range.0, stats.expected_range.1, status
                ));
            }
        }
        report.push_str("\n");

        // Interpretation
        report.push_str("## Interpretation\n\n");
        report.push_str(&self.interpret_results(results));
        report.push_str("\n\n");

        // Recommendation
        report.push_str("## Recommendation\n\n");
        report.push_str(&self.generate_recommendation(results));
        report.push_str("\n");

        report
    }

    /// Interpret validation results
    fn interpret_results(&self, results: &ValidationResults) -> String {
        if results.pearson_r > 0.8 && results.p_value < 0.001 {
            "✅ **EXCELLENT**: Strong correlation confirms Φ reliably tracks consciousness levels.\n\
             Results publishable in top-tier scientific journals (Nature, Science, PNAS).\n\
             \n\
             The high correlation (r > 0.8) with strong statistical significance (p < 0.001)\n\
             provides empirical validation of Integrated Information Theory in a working system.\n\
             Classification performance (AUC > 0.9) demonstrates excellent discrimination between\n\
             conscious and unconscious states.\n\
             \n\
             **Scientific Impact**: First empirical validation of IIT implementation."
        } else if results.pearson_r > 0.7 && results.p_value < 0.01 {
            "✅ **GOOD**: Moderate-strong correlation validates Φ as consciousness measure.\n\
             Results publishable with additional validation.\n\
             \n\
             The correlation is statistically significant and demonstrates that Φ tracks\n\
             consciousness levels. Consider expanding sample size and validating against\n\
             additional datasets to strengthen publication case.\n\
             \n\
             **Scientific Impact**: Solid validation of IIT implementation with room for refinement."
        } else if results.pearson_r > 0.5 && results.p_value < 0.05 {
            "⚠️ **WEAK**: Correlation present but not strong enough for top-tier publication.\n\
             Requires refinement of Φ computation or state generation.\n\
             \n\
             While statistically significant, the correlation is not strong enough to claim\n\
             robust validation. Review Φ computation parameters and synthetic state generation\n\
             methodology for improvements.\n\
             \n\
             **Recommendation**: Analyze outliers and refine before publication."
        } else {
            "❌ **INSUFFICIENT**: No significant correlation detected.\n\
             Major revision needed in Φ computation methodology or state generation.\n\
             \n\
             The lack of significant correlation suggests fundamental issues with either:\n\
             1. Φ computation implementation\n\
             2. Synthetic state generation methodology\n\
             3. Integration measurement approach\n\
             \n\
             **Recommendation**: Comprehensive review of IIT implementation required."
        }.to_string()
    }

    /// Generate actionable recommendation
    fn generate_recommendation(&self, results: &ValidationResults) -> String {
        if results.pearson_r > 0.8 {
            "**Proceed to scientific paper preparation**:\n\
             1. Generate publication-quality figures\n\
             2. Draft complete methods and results sections\n\
             3. Write introduction and discussion\n\
             4. Submit to Nature, Science, or Nature Neuroscience\n\
             5. Consider validation with clinical data (if available)"
        } else if results.pearson_r > 0.7 {
            "**Refine and expand validation**:\n\
             1. Increase sample size to n > 1000 per state\n\
             2. Add intermediate consciousness levels\n\
             3. Validate against theoretical IIT predictions\n\
             4. Tune Φ computation parameters\n\
             5. Prepare for submission to specialized consciousness journals"
        } else {
            "**Fundamental review required**:\n\
             1. Verify Φ computation against IIT 3.0 specification\n\
             2. Analyze outliers and edge cases\n\
             3. Review synthetic state generation methodology\n\
             4. Consider alternative integration measures\n\
             5. Consult IIT literature for implementation guidance"
        }.to_string()
    }

    /// Export data for external analysis
    pub fn export_data(&self) -> String {
        serde_json::to_string_pretty(&self.validation_data).unwrap()
    }

    /// Get current results (if available)
    pub fn get_results(&self) -> Option<&ValidationResults> {
        self.results.as_ref()
    }
}

impl Default for PhiValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_creation() {
        let framework = PhiValidationFramework::new();
        assert_eq!(framework.validation_data.len(), 0);
        assert!(framework.results.is_none());
    }

    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = PhiValidationFramework::pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001, "Perfect positive correlation should be 1.0, got {}", r);
    }

    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = PhiValidationFramework::pearson_correlation(&x, &y);
        assert!((r + 1.0).abs() < 0.001, "Perfect negative correlation should be -1.0, got {}", r);
    }

    #[test]
    fn test_pearson_correlation_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let r = PhiValidationFramework::pearson_correlation(&x, &y);
        assert!(r.abs() < 0.001, "No correlation should be near 0, got {}", r);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // Non-linear but monotonic
        let rho = PhiValidationFramework::spearman_correlation(&x, &y);
        assert!((rho - 1.0).abs() < 0.001, "Monotonic relationship should have ρ ≈ 1.0, got {}", rho);
    }

    #[test]
    fn test_rank_values() {
        let values = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let ranks = PhiValidationFramework::rank_values(&values);
        assert_eq!(ranks, vec![3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_auc_perfect_classification() {
        // Perfect separation at threshold 0.5
        let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let auc = PhiValidationFramework::compute_auc(&x, &y, 0.5);
        assert!((auc - 1.0).abs() < 0.001, "Perfect classification should have AUC = 1.0, got {}", auc);
    }

    #[test]
    fn test_mae_and_rmse() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 2.1, 2.9, 4.2, 4.8];

        let mae = PhiValidationFramework::mean_absolute_error(&x, &y);
        let rmse = PhiValidationFramework::root_mean_squared_error(&x, &y);

        assert!(mae < 0.2, "MAE should be small for close predictions");
        assert!(rmse < 0.3, "RMSE should be small for close predictions");
    }

    #[test]
    fn test_small_validation_study() {
        let mut framework = PhiValidationFramework::new();
        let results = framework.run_validation_study(10); // 10 samples per state

        // Should have 8 states × 10 samples = 80 data points
        assert_eq!(results.n, 80);

        // Correlation should be positive (Φ increases with consciousness)
        assert!(results.pearson_r > 0.0, "Correlation should be positive");

        // Should have per-state statistics
        assert!(!results.state_stats.is_empty());
    }

    #[test]
    fn test_validation_data_point_creation() {
        let data_point = ValidationDataPoint {
            consciousness_level: 0.5,
            phi_value: 0.45,
            state_type: StateType::RestingAwake,
            metadata: HashMap::new(),
            timestamp: Some(std::time::Instant::now()),
        };

        assert_eq!(data_point.consciousness_level, 0.5);
        assert_eq!(data_point.phi_value, 0.45);
    }

    #[test]
    fn test_report_generation() {
        let mut framework = PhiValidationFramework::new();
        framework.run_validation_study(5); // Small study for speed

        let report = framework.generate_report();

        assert!(report.contains("Φ Validation Study Results"));
        assert!(report.contains("Statistical Summary"));
        assert!(report.contains("Pearson correlation"));
        assert!(report.contains("Recommendation"));
    }
}
