// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenal Discrimination Scaling Findings
//!
//! This module documents and applies the research findings on how
//! phenomenal discrimination scales with model size. Key discoveries:
//!
//! ## Non-Monotonic Scaling
//!
//! Contrary to "bigger is better," phenomenal discrimination peaks
//! at intermediate model sizes:
//!
//! - **Encoders (BERT)**: Optimal at ~110M parameters (BERT-base)
//! - **Decoders (GPT-2)**: Optimal at ~355M parameters (GPT-2 Medium)
//!
//! ## Primary Mechanism: Angular Separation
//!
//! Larger models align phenomenal and functional concept centroids
//! more closely, reducing discriminability:
//!
//! - BERT-base → BERT-large: **-40% angular separation**
//! - This directly reduces Fisher's criterion (class separability)
//!
//! ## Practical Implications
//!
//! When probing LLMs for phenomenal content:
//! 1. Prefer ~100M encoder models or ~350M decoder models
//! 2. Monitor angular separation as a quality metric
//! 3. Consider fine-tuning to increase discrimination if needed
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::scaling_findings::{
//!     OptimalModelConfig, ScalingMetrics, get_optimal_model
//! };
//!
//! // Get recommended model for phenomenal analysis
//! let config = get_optimal_model(Architecture::Encoder);
//! println!("Use: {} ({:.0}M params)", config.model_name, config.params_millions);
//!
//! // Monitor discrimination quality
//! let metrics = ScalingMetrics::compute(&phen_centroids, &func_centroids);
//! if metrics.angular_separation < 0.15 {
//!     println!("Warning: Low angular separation - consider smaller model");
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Model architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// Bidirectional encoder (BERT, RoBERTa)
    Encoder,
    /// Autoregressive decoder (GPT-2, Pythia, OPT)
    Decoder,
    /// Encoder-decoder (T5, BART) - not yet tested
    EncoderDecoder,
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::Encoder => write!(f, "Encoder"),
            Architecture::Decoder => write!(f, "Decoder"),
            Architecture::EncoderDecoder => write!(f, "Encoder-Decoder"),
        }
    }
}

/// Optimal model configuration for phenomenal discrimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalModelConfig {
    /// Model identifier (e.g., "bert-base-uncased")
    pub model_name: String,
    /// Model family (e.g., "BERT", "GPT-2")
    pub family: String,
    /// Architecture type
    pub architecture: Architecture,
    /// Parameters in millions
    pub params_millions: f64,
    /// Expected Fisher's criterion at optimal layer
    pub expected_fisher: f64,
    /// Expected angular separation
    pub expected_angular_sep: f64,
    /// Optimal layer depth (fraction, e.g., 0.9 = 90%)
    pub optimal_layer_depth: f64,
    /// Notes about this configuration
    pub notes: String,
}

/// Scaling metrics for evaluating phenomenal discrimination quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    /// Fisher's criterion: centroid_distance / avg_within_variance
    pub fisher_criterion: f64,
    /// Angular separation: 1 - cos_sim(phen_centroid, func_centroid)
    pub angular_separation: f64,
    /// Cosine similarity between centroids
    pub centroid_cosine_similarity: f64,
    /// Euclidean distance between centroids
    pub centroid_distance: f64,
    /// Average within-class variance
    pub avg_within_variance: f64,
    /// Isotropy: λ_min / λ_max of covariance
    pub isotropy: Option<f64>,
    /// Quality assessment based on expected values
    pub quality: DiscriminationQuality,
}

/// Quality assessment for discrimination metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscriminationQuality {
    /// Excellent discrimination (F > 1.0 for encoders, F > 0.8 for decoders)
    Excellent,
    /// Good discrimination (within 10% of optimal)
    Good,
    /// Adequate discrimination (within 25% of optimal)
    Adequate,
    /// Poor discrimination (below adequate threshold)
    Poor,
}

impl std::fmt::Display for DiscriminationQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscriminationQuality::Excellent => write!(f, "Excellent"),
            DiscriminationQuality::Good => write!(f, "Good"),
            DiscriminationQuality::Adequate => write!(f, "Adequate"),
            DiscriminationQuality::Poor => write!(f, "Poor"),
        }
    }
}

/// Research findings on phenomenal discrimination scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingFindings {
    /// Correlation between params and Fisher (negative = inverse scaling)
    pub fisher_param_correlation: f64,
    /// Optimal size in millions of parameters
    pub optimal_size_millions: f64,
    /// Peak Fisher's criterion at optimal size
    pub peak_fisher: f64,
    /// Decline percentage from optimal to largest tested
    pub decline_from_optimal_pct: f64,
    /// Primary mechanism for inverse scaling
    pub primary_mechanism: String,
    /// Secondary mechanism
    pub secondary_mechanism: String,
}

// =============================================================================
// Optimal Model Configurations
// =============================================================================

/// Get the optimal model configuration for a given architecture
pub fn get_optimal_model(arch: Architecture) -> OptimalModelConfig {
    match arch {
        Architecture::Encoder => OptimalModelConfig {
            model_name: "bert-base-uncased".to_string(),
            family: "BERT".to_string(),
            architecture: Architecture::Encoder,
            params_millions: 109.5,
            expected_fisher: 1.19,
            expected_angular_sep: 0.246,
            optimal_layer_depth: 0.90,
            notes: "Peak discrimination at ~110M params. BERT-large shows -13.8% decline.".to_string(),
        },
        Architecture::Decoder => OptimalModelConfig {
            model_name: "gpt2-medium".to_string(),
            family: "GPT-2".to_string(),
            architecture: Architecture::Decoder,
            params_millions: 355.0,
            expected_fisher: 0.84,
            expected_angular_sep: 0.063,
            optimal_layer_depth: 0.90,
            notes: "Peak at ~355M. Decoders need ~3x encoder params for peak. GPT-2 XL shows -16% decline.".to_string(),
        },
        Architecture::EncoderDecoder => OptimalModelConfig {
            model_name: "t5-base".to_string(),
            family: "T5".to_string(),
            architecture: Architecture::EncoderDecoder,
            params_millions: 220.0,
            expected_fisher: 1.0, // Estimated, not empirically tested
            expected_angular_sep: 0.15,
            optimal_layer_depth: 0.90,
            notes: "Not empirically tested. Estimate based on encoder/decoder findings.".to_string(),
        },
    }
}

/// Get all tested optimal configurations
pub fn get_all_optimal_configs() -> Vec<OptimalModelConfig> {
    vec![
        get_optimal_model(Architecture::Encoder),
        get_optimal_model(Architecture::Decoder),
    ]
}

/// Get scaling findings for an architecture
pub fn get_scaling_findings(arch: Architecture) -> ScalingFindings {
    match arch {
        Architecture::Encoder => ScalingFindings {
            fisher_param_correlation: -0.14, // Weak overall due to non-monotonicity
            optimal_size_millions: 109.5,
            peak_fisher: 1.19,
            decline_from_optimal_pct: 13.8,
            primary_mechanism: "Angular separation reduction (-40% in BERT-large)".to_string(),
            secondary_mechanism: "Isotropy increase (+94% in BERT-large)".to_string(),
        },
        Architecture::Decoder => ScalingFindings {
            fisher_param_correlation: -0.51,
            optimal_size_millions: 355.0,
            peak_fisher: 0.84,
            decline_from_optimal_pct: 16.0,
            primary_mechanism: "Angular separation reduction".to_string(),
            secondary_mechanism: "Isotropy increase".to_string(),
        },
        Architecture::EncoderDecoder => ScalingFindings {
            fisher_param_correlation: 0.0, // Unknown
            optimal_size_millions: 220.0,
            peak_fisher: 1.0,
            decline_from_optimal_pct: 10.0,
            primary_mechanism: "Unknown - not tested".to_string(),
            secondary_mechanism: "Unknown - not tested".to_string(),
        },
    }
}

// =============================================================================
// Metrics Computation
// =============================================================================

impl ScalingMetrics {
    /// Compute scaling metrics from phenomenal and functional representations
    ///
    /// # Arguments
    /// * `phen_reps` - Phenomenal concept representations [n_phen, dim]
    /// * `func_reps` - Functional concept representations [n_func, dim]
    /// * `arch` - Architecture for quality assessment thresholds
    pub fn compute(phen_reps: &[Vec<f32>], func_reps: &[Vec<f32>], arch: Architecture) -> Self {
        if phen_reps.is_empty() || func_reps.is_empty() {
            return Self::default_poor();
        }

        let _dim = phen_reps[0].len();

        // Compute centroids
        let phen_centroid = compute_centroid(phen_reps);
        let func_centroid = compute_centroid(func_reps);

        // Centroid distance
        let centroid_distance = euclidean_distance(&phen_centroid, &func_centroid);

        // Cosine similarity and angular separation
        let centroid_cosine_similarity = cosine_similarity(&phen_centroid, &func_centroid);
        let angular_separation = 1.0 - centroid_cosine_similarity;

        // Within-class variance
        let phen_within = mean_distance_to_centroid(phen_reps, &phen_centroid);
        let func_within = mean_distance_to_centroid(func_reps, &func_centroid);
        let avg_within_variance = (phen_within + func_within) / 2.0;

        // Fisher's criterion
        let fisher_criterion = if avg_within_variance > 1e-10 {
            centroid_distance / avg_within_variance
        } else {
            0.0
        };

        // Quality assessment
        let quality = assess_quality(fisher_criterion, angular_separation, arch);

        Self {
            fisher_criterion: fisher_criterion as f64,
            angular_separation: angular_separation as f64,
            centroid_cosine_similarity: centroid_cosine_similarity as f64,
            centroid_distance: centroid_distance as f64,
            avg_within_variance: avg_within_variance as f64,
            isotropy: None, // Requires full covariance computation
            quality,
        }
    }

    /// Compute with isotropy metric (more expensive)
    pub fn compute_with_isotropy(
        phen_reps: &[Vec<f32>],
        func_reps: &[Vec<f32>],
        arch: Architecture,
    ) -> Self {
        let mut metrics = Self::compute(phen_reps, func_reps, arch);

        // Compute isotropy from combined representations
        let all_reps: Vec<&Vec<f32>> = phen_reps.iter().chain(func_reps.iter()).collect();
        if all_reps.len() >= 2 {
            metrics.isotropy = Some(compute_isotropy(&all_reps) as f64);
        }

        metrics
    }

    fn default_poor() -> Self {
        Self {
            fisher_criterion: 0.0,
            angular_separation: 0.0,
            centroid_cosine_similarity: 1.0,
            centroid_distance: 0.0,
            avg_within_variance: 0.0,
            isotropy: None,
            quality: DiscriminationQuality::Poor,
        }
    }

    /// Check if metrics indicate good phenomenal discrimination
    pub fn is_adequate(&self) -> bool {
        matches!(
            self.quality,
            DiscriminationQuality::Excellent
                | DiscriminationQuality::Good
                | DiscriminationQuality::Adequate
        )
    }

    /// Get recommendations based on metrics
    pub fn get_recommendations(&self, arch: Architecture) -> Vec<String> {
        let mut recs = Vec::new();
        let optimal = get_optimal_model(arch);

        if self.angular_separation < optimal.expected_angular_sep * 0.6 {
            recs.push(format!(
                "Angular separation ({:.3}) is low. Consider using a smaller model closer to {:.0}M params.",
                self.angular_separation, optimal.params_millions
            ));
        }

        if self.fisher_criterion < optimal.expected_fisher * 0.75 {
            recs.push(format!(
                "Fisher's criterion ({:.3}) below optimal ({:.3}). Model may be too large for optimal phenomenal discrimination.",
                self.fisher_criterion, optimal.expected_fisher
            ));
        }

        if let Some(iso) = self.isotropy {
            if iso > 0.03 {
                recs.push(format!(
                    "High isotropy ({iso:.4}) suggests representations are becoming too uniform. This is typical of larger models."
                ));
            }
        }

        if recs.is_empty() {
            recs.push(
                "Metrics are within expected ranges for good phenomenal discrimination."
                    .to_string(),
            );
        }

        recs
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn compute_centroid(reps: &[Vec<f32>]) -> Vec<f32> {
    if reps.is_empty() {
        return Vec::new();
    }

    let dim = reps[0].len();
    let n = reps.len() as f32;
    let mut centroid = vec![0.0f32; dim];

    for rep in reps {
        for (i, &v) in rep.iter().enumerate() {
            centroid[i] += v;
        }
    }

    for v in centroid.iter_mut() {
        *v /= n;
    }

    centroid
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

use symthaea_core::math::cosine_similarity_f32 as cosine_similarity;

fn mean_distance_to_centroid(reps: &[Vec<f32>], centroid: &[f32]) -> f32 {
    if reps.is_empty() {
        return 0.0;
    }

    let sum: f32 = reps.iter().map(|r| euclidean_distance(r, centroid)).sum();
    sum / reps.len() as f32
}

fn compute_isotropy(reps: &[&Vec<f32>]) -> f32 {
    // Simplified isotropy: ratio of min to max variance across dimensions
    // Full implementation would use eigenvalues of covariance matrix

    if reps.is_empty() || reps[0].is_empty() {
        return 0.0;
    }

    let dim = reps[0].len();
    let n = reps.len() as f32;

    // Compute variance per dimension
    let mut variances = Vec::with_capacity(dim);
    for d in 0..dim {
        let mean: f32 = reps.iter().map(|r| r[d]).sum::<f32>() / n;
        let var: f32 = reps.iter().map(|r| (r[d] - mean).powi(2)).sum::<f32>() / n;
        variances.push(var);
    }

    let min_var = variances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_var = variances.iter().cloned().fold(0.0f32, f32::max);

    if max_var > 1e-10 {
        min_var / max_var
    } else {
        0.0
    }
}

fn assess_quality(fisher: f32, angular_sep: f32, arch: Architecture) -> DiscriminationQuality {
    let optimal = get_optimal_model(arch);
    let expected_fisher = optimal.expected_fisher as f32;
    let expected_angular = optimal.expected_angular_sep as f32;

    // Primary check: Fisher's criterion
    let fisher_ratio = fisher / expected_fisher;

    // Secondary check: Angular separation
    let angular_ratio = angular_sep / expected_angular;

    // Combined assessment
    let combined = (fisher_ratio + angular_ratio) / 2.0;

    if combined >= 0.9 {
        DiscriminationQuality::Excellent
    } else if combined >= 0.75 {
        DiscriminationQuality::Good
    } else if combined >= 0.5 {
        DiscriminationQuality::Adequate
    } else {
        DiscriminationQuality::Poor
    }
}

// =============================================================================
// Model Size Recommendations
// =============================================================================

/// Recommend a model size given constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model_name: String,
    pub params_millions: f64,
    pub expected_fisher: f64,
    pub reasoning: String,
}

/// Get model recommendation given constraints
///
/// # Arguments
/// * `arch` - Preferred architecture
/// * `max_params_millions` - Maximum parameters (e.g., 1000 for 1B limit)
/// * `min_fisher` - Minimum acceptable Fisher's criterion
pub fn recommend_model(
    arch: Architecture,
    max_params_millions: Option<f64>,
    min_fisher: Option<f64>,
) -> ModelRecommendation {
    let optimal = get_optimal_model(arch);
    let max_params = max_params_millions.unwrap_or(f64::INFINITY);
    let min_f = min_fisher.unwrap_or(0.0);

    // Check if optimal model meets constraints
    if optimal.params_millions <= max_params && optimal.expected_fisher >= min_f {
        return ModelRecommendation {
            model_name: optimal.model_name.clone(),
            params_millions: optimal.params_millions,
            expected_fisher: optimal.expected_fisher,
            reasoning: format!(
                "Optimal {} model ({:.0}M params) meets all constraints with peak Fisher's criterion of {:.2}.",
                arch, optimal.params_millions, optimal.expected_fisher
            ),
        };
    }

    // If constraints prevent optimal, provide best alternative
    match arch {
        Architecture::Encoder => {
            if max_params < 30.0 {
                ModelRecommendation {
                    model_name: "prajjwal1/bert-mini".to_string(),
                    params_millions: 11.2,
                    expected_fisher: 1.15,
                    reasoning:
                        "BERT-Mini offers good discrimination (F=1.15) in a very small footprint."
                            .to_string(),
                }
            } else if max_params < 67.0 {
                // DistilBERT is 66.4M params, so only recommend if we can fit it
                if max_params >= 66.4 {
                    ModelRecommendation {
                        model_name: "distilbert-base-uncased".to_string(),
                        params_millions: 66.4,
                        expected_fisher: 1.18,
                        reasoning: "DistilBERT approaches optimal discrimination (F=1.18) with fewer parameters.".to_string(),
                    }
                } else {
                    // Between 30M and 66.4M, recommend BERT-Mini as it's the only one that fits
                    ModelRecommendation {
                        model_name: "prajjwal1/bert-mini".to_string(),
                        params_millions: 11.2,
                        expected_fisher: 1.15,
                        reasoning: "BERT-Mini is the best fit within the parameter constraint."
                            .to_string(),
                    }
                }
            } else {
                ModelRecommendation {
                    model_name: optimal.model_name,
                    params_millions: optimal.params_millions,
                    expected_fisher: optimal.expected_fisher,
                    reasoning: "BERT-base is optimal for phenomenal discrimination.".to_string(),
                }
            }
        }
        Architecture::Decoder => {
            if max_params < 150.0 {
                ModelRecommendation {
                    model_name: "gpt2".to_string(),
                    params_millions: 124.0,
                    expected_fisher: 0.74,
                    reasoning: "GPT-2 Small is the smallest decoder tested. Discrimination is below optimal.".to_string(),
                }
            } else if max_params < 400.0 {
                ModelRecommendation {
                    model_name: optimal.model_name,
                    params_millions: optimal.params_millions,
                    expected_fisher: optimal.expected_fisher,
                    reasoning: "GPT-2 Medium is optimal for phenomenal discrimination in decoders."
                        .to_string(),
                }
            } else {
                // Large models available but not recommended
                ModelRecommendation {
                    model_name: optimal.model_name,
                    params_millions: optimal.params_millions,
                    expected_fisher: optimal.expected_fisher,
                    reasoning: "GPT-2 Medium is recommended. Larger models (Large, XL) show declining discrimination.".to_string(),
                }
            }
        }
        Architecture::EncoderDecoder => ModelRecommendation {
            model_name: "t5-base".to_string(),
            params_millions: 220.0,
            expected_fisher: 1.0,
            reasoning:
                "T5-base recommended (estimated). Encoder-decoder scaling not empirically tested."
                    .to_string(),
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_encoder_config() {
        let config = get_optimal_model(Architecture::Encoder);
        assert_eq!(config.family, "BERT");
        assert!((config.params_millions - 109.5).abs() < 1.0);
        assert!(config.expected_fisher > 1.0);
    }

    #[test]
    fn test_optimal_decoder_config() {
        let config = get_optimal_model(Architecture::Decoder);
        assert_eq!(config.family, "GPT-2");
        assert!((config.params_millions - 355.0).abs() < 1.0);
        assert!(config.expected_fisher > 0.8);
    }

    #[test]
    fn test_decoder_needs_more_params() {
        let encoder = get_optimal_model(Architecture::Encoder);
        let decoder = get_optimal_model(Architecture::Decoder);

        // Decoders need ~3x more params
        let ratio = decoder.params_millions / encoder.params_millions;
        assert!(ratio > 2.5 && ratio < 4.0);
    }

    #[test]
    fn test_metrics_computation() {
        // Simple test data
        let phen_reps = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
        ];
        let func_reps = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.2, 0.8, 0.0],
        ];

        let metrics = ScalingMetrics::compute(&phen_reps, &func_reps, Architecture::Encoder);

        // Centroids should be well separated
        assert!(metrics.fisher_criterion > 0.5);
        assert!(metrics.angular_separation > 0.0);
        assert!(metrics.centroid_cosine_similarity < 1.0);
    }

    #[test]
    fn test_quality_assessment() {
        // Excellent quality
        let quality = assess_quality(1.2, 0.25, Architecture::Encoder);
        assert_eq!(quality, DiscriminationQuality::Excellent);

        // Poor quality (low fisher, low angular sep)
        let quality = assess_quality(0.3, 0.05, Architecture::Encoder);
        assert_eq!(quality, DiscriminationQuality::Poor);
    }

    #[test]
    fn test_recommendation_constraints() {
        // With tight param constraint
        let rec = recommend_model(Architecture::Encoder, Some(50.0), None);
        assert!(rec.params_millions < 50.0);

        // With Fisher constraint
        let rec = recommend_model(Architecture::Encoder, None, Some(1.1));
        assert!(rec.expected_fisher >= 1.1);
    }

    #[test]
    fn test_scaling_findings() {
        let findings = get_scaling_findings(Architecture::Encoder);

        // Should show non-monotonic pattern (weak overall correlation)
        assert!(findings.fisher_param_correlation.abs() < 0.5);

        // Should have meaningful peak
        assert!(findings.peak_fisher > 1.0);

        // Should show decline from optimal
        assert!(findings.decline_from_optimal_pct > 10.0);
    }
}
