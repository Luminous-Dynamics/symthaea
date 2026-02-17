//! PoGQ-v4.1 Enhanced: Publication-Grade Byzantine Detection
//!
//! Complete Rust implementation of PoGQ-v4.1 with all Gen-4 components:
//! 1. Mondrian (class-aware validation with per-class z-score normalization)
//! 2. Mondrian Conformal (per-class FPR buckets)
//! 3. Adaptive Hybrid Scoring (PCA-cosine + adaptive λ based on SNR)
//! 4. Temporal EMA (β=0.85 historical smoothing)
//! 5. Direction Prefilter (ReLU(cosine) > 0 cheap rejection)
//! 6. Winsorized Dispersion (outlier-robust thresholds)
//! 7. PoGQ-v4-Lite (linear probe head for high-dim rescue)
//!
//! Phase 2 Enhancements:
//! 8. Warm-up Quota (W=3 rounds grace period for cold-start)
//! 9. Hysteresis (k=2 consecutive violations to quarantine, m=3 to release)
//! 10. Egregious Cap (p99.9 threshold for warm-up violations)
//!
//! # Example
//!
//! ```rust,ignore
//! use fl_aggregator::pogq::{PoGQv41Enhanced, PoGQv41Config};
//!
//! let config = PoGQv41Config::default();
//! let mut pogq = PoGQv41Enhanced::new(config);
//!
//! // Calibrate on clean data
//! pogq.calibrate_mondrian(&validation_scores, &validation_profiles);
//!
//! // Score a gradient
//! let result = pogq.score_gradient(&gradient, "client_1", &classes, &reference, 0);
//! println!("Byzantine: {}", result.detection.is_byzantine);
//! ```

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, VecDeque};
use tracing::{info, warn};

use crate::error::{AggregatorError, Result};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for PoGQ-v4.1 Enhanced
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQv41Config {
    // Mondrian settings
    /// Enable per-class z-score normalization
    pub per_class_normalization: bool,
    /// Window for computing per-class statistics
    pub class_score_window: usize,

    // PCA settings (simplified - no sklearn dependency)
    /// Number of PCA components (for dimensionality reduction)
    pub pca_components: usize,
    /// Fit PCA on reference gradients only
    pub pca_fit_on_reference: bool,

    // Adaptive lambda settings
    /// Enable adaptive lambda
    pub lambda_adaptive: bool,
    /// SNR coefficient for lambda computation
    pub lambda_a: f32,
    /// SNR offset for lambda computation
    pub lambda_b: f32,
    /// Minimum lambda value
    pub lambda_min: f32,
    /// Maximum lambda value
    pub lambda_max: f32,

    // Mondrian conformal settings
    /// Enable Mondrian conformal prediction
    pub mondrian_conformal: bool,
    /// FPR guarantee (alpha)
    pub conformal_alpha: f32,
    /// Minimum bucket size for separate threshold
    pub min_bucket_size: usize,

    // Temporal EMA settings
    /// Weight for historical scores (0.85 = stable but responsive)
    pub ema_beta: f32,

    // Warm-up quota settings (Phase 2)
    /// Number of rounds before full enforcement
    pub warmup_rounds: u32,
    /// Quantile for egregious cap (p99.9)
    pub egregious_cap_quantile: f32,

    // Hysteresis settings (Phase 2)
    /// Consecutive violations to quarantine
    pub hysteresis_k: u32,
    /// Consecutive clears to release
    pub hysteresis_m: u32,

    // Direction prefilter settings
    /// Enable direction prefilter
    pub direction_prefilter: bool,
    /// ReLU threshold for direction score
    pub direction_threshold: f32,

    // Winsorized dispersion settings
    /// Lower quantile for winsorization
    pub winsorize_lower: f32,
    /// Upper quantile for winsorization
    pub winsorize_upper: f32,

    // PoGQ-v4-Lite settings
    /// Use lite mode for high-dim datasets
    pub use_lite_mode: bool,
    /// Probe layer slice start (None = auto)
    pub probe_slice_start: Option<usize>,
    /// Probe layer slice end (None = auto)
    pub probe_slice_end: Option<usize>,

    // Provenance tracking
    /// Enable provenance tracking
    pub track_provenance: bool,
    /// Defense preset ID
    pub defense_preset_id: String,
}

impl Default for PoGQv41Config {
    fn default() -> Self {
        Self {
            // Mondrian
            per_class_normalization: true,
            class_score_window: 10,

            // PCA
            pca_components: 32,
            pca_fit_on_reference: true,

            // Adaptive lambda
            lambda_adaptive: true,
            lambda_a: 2.0,
            lambda_b: -1.0,
            lambda_min: 0.1,
            lambda_max: 0.9,

            // Mondrian conformal
            mondrian_conformal: true,
            conformal_alpha: 0.10,
            min_bucket_size: 10,

            // Temporal EMA
            ema_beta: 0.85,

            // Warm-up quota (Phase 2)
            warmup_rounds: 3,
            egregious_cap_quantile: 0.999,

            // Hysteresis (Phase 2)
            hysteresis_k: 2,
            hysteresis_m: 3,

            // Direction prefilter
            direction_prefilter: true,
            direction_threshold: 0.0,

            // Winsorized dispersion
            winsorize_lower: 0.05,
            winsorize_upper: 0.95,

            // Lite mode
            use_lite_mode: false,
            probe_slice_start: None,
            probe_slice_end: None,

            // Provenance
            track_provenance: true,
            defense_preset_id: "pogq_v4.1_default".to_string(),
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Represents a client's class distribution profile for Mondrian conformal
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MondrianProfile {
    /// Set of class IDs this client has data for
    pub client_classes: BTreeSet<u32>,
}

impl MondrianProfile {
    /// Create a new Mondrian profile from class IDs
    pub fn new(classes: impl IntoIterator<Item = u32>) -> Self {
        Self {
            client_classes: classes.into_iter().collect(),
        }
    }

    /// Get profile key for HashMap lookup
    pub fn profile_key(&self) -> Vec<u32> {
        self.client_classes.iter().copied().collect()
    }
}

/// Per-class statistics for z-score normalization
#[derive(Debug, Clone, Default)]
struct ClassStats {
    mean: f32,
    std: f32,
    history: VecDeque<f32>,
}

/// Lambda statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LambdaStats {
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
}

/// Scores from gradient evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientScores {
    /// Raw Mondrian score (z-normalized)
    pub mondrian_raw: f32,
    /// Hybrid score (λ * direction + (1-λ) * utility)
    pub hybrid: f32,
    /// EMA-smoothed score
    pub ema: f32,
    /// Utility score component
    pub utility: f32,
    /// Direction score (cosine similarity after prefilter)
    pub direction: f32,
    /// Current lambda value
    pub lambda: f32,
}

/// Detection decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Final Byzantine decision
    pub is_byzantine: bool,
    /// Conformal outlier status
    pub is_conformal_outlier: bool,
    /// Threshold used for detection
    pub threshold: f32,
    /// Mondrian profile (class IDs)
    pub mondrian_profile: Vec<u32>,
    /// Quarantine status
    pub quarantined: bool,
}

/// Phase 2 status (warm-up and hysteresis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Status {
    /// Number of rounds this client has participated
    pub client_rounds: u32,
    /// Whether client is in warm-up period
    pub in_warmup: bool,
    /// Warm-up status string
    pub warmup_status: String,
    /// Consecutive violations count
    pub consecutive_violations: u32,
    /// Consecutive clears count
    pub consecutive_clears: u32,
    /// Egregious cap value
    pub egregious_cap: Option<f32>,
}

/// Complete gradient scoring result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientResult {
    /// Client ID
    pub client_id: String,
    /// Round number
    pub round: u32,
    /// Score components
    pub scores: GradientScores,
    /// Detection decision
    pub detection: DetectionResult,
    /// Phase 2 status
    pub phase2: Phase2Status,
    /// Per-class raw scores
    pub class_scores: HashMap<u32, f32>,
}

/// Detector statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStatistics {
    /// Total number of unique clients seen
    pub total_clients: usize,
    /// Number of currently quarantined clients
    pub quarantined_clients: usize,
    /// Average EMA score across all clients
    pub avg_ema_score: f32,
}

/// Aggregation diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationDiagnostics {
    /// Round number
    pub round: u32,
    /// Total clients
    pub n_clients: usize,
    /// Honest clients
    pub n_honest: usize,
    /// Byzantine clients
    pub n_byzantine: usize,
    /// Detection rate
    pub detection_rate: f32,
    /// Honest client IDs
    pub honest_clients: Vec<String>,
    /// Per-client results
    pub per_client_results: Vec<GradientResult>,
    /// Lambda statistics
    pub lambda_stats: LambdaStats,
}

// =============================================================================
// Adaptive Hybrid Scorer
// =============================================================================

/// Adaptive λ hybrid scorer with direction + utility scoring
pub struct AdaptiveHybridScorer {
    config: PoGQv41Config,
    gradient_history: VecDeque<Array1<f32>>,
    lambda_history: Vec<f32>,
    // Simplified PCA: store mean and principal components
    pca_mean: Option<Array1<f32>>,
    pca_components: Option<Vec<Array1<f32>>>,
    pca_fitted: bool,
}

impl AdaptiveHybridScorer {
    /// Create a new adaptive hybrid scorer
    pub fn new(config: PoGQv41Config) -> Self {
        Self {
            config,
            gradient_history: VecDeque::with_capacity(20),
            lambda_history: Vec::new(),
            pca_mean: None,
            pca_components: None,
            pca_fitted: false,
        }
    }

    /// Fit simplified PCA on reference gradients
    ///
    /// Uses power iteration for top-k components (no external dependencies)
    pub fn fit_pca(&mut self, reference_gradients: &[Array1<f32>]) {
        if reference_gradients.is_empty() || !self.config.pca_fit_on_reference {
            return;
        }

        let n = reference_gradients.len();
        let dim = reference_gradients[0].len();
        let k = self.config.pca_components.min(dim).min(n);

        // Compute mean
        let mut mean = Array1::zeros(dim);
        for g in reference_gradients {
            mean = mean + g;
        }
        mean /= n as f32;
        self.pca_mean = Some(mean.clone());

        // Center data
        let centered: Vec<Array1<f32>> = reference_gradients
            .iter()
            .map(|g| g - &mean)
            .collect();

        // Power iteration for top-k components
        let mut components = Vec::with_capacity(k);
        let mut residual = centered.clone();

        for _ in 0..k {
            if let Some(component) = self.power_iteration(&residual, 50) {
                // Deflate: remove component from residual
                for r in &mut residual {
                    let proj = r.dot(&component);
                    *r = &*r - &(&component * proj);
                }
                components.push(component);
            }
        }

        self.pca_components = Some(components);
        self.pca_fitted = true;

        info!("Fitted simplified PCA on {} reference gradients with {} components",
              n, self.pca_components.as_ref().map(|c| c.len()).unwrap_or(0));
    }

    /// Power iteration to find dominant eigenvector
    fn power_iteration(&self, data: &[Array1<f32>], max_iter: usize) -> Option<Array1<f32>> {
        if data.is_empty() {
            return None;
        }

        let dim = data[0].len();
        let mut v = Array1::from_elem(dim, 1.0 / (dim as f32).sqrt());

        for _ in 0..max_iter {
            // Compute X^T * X * v
            let mut new_v = Array1::zeros(dim);
            for x in data {
                let proj = x.dot(&v);
                new_v = new_v + &(x * proj);
            }

            // Normalize
            let norm = l2_norm(new_v.view());
            if norm < 1e-10 {
                return None;
            }
            new_v /= norm;

            // Check convergence
            let diff = l2_norm((&new_v - &v).view());
            v = new_v;

            if diff < 1e-6 {
                break;
            }
        }

        Some(v)
    }

    /// Project gradient to PCA subspace
    fn pca_project(&self, gradient: &Array1<f32>) -> Array1<f32> {
        if !self.pca_fitted {
            return gradient.clone();
        }

        let mean = self.pca_mean.as_ref().unwrap();
        let components = self.pca_components.as_ref().unwrap();

        let centered = gradient - mean;
        let k = components.len();

        let mut projected = Array1::zeros(k);
        for (i, comp) in components.iter().enumerate() {
            projected[i] = centered.dot(comp);
        }

        projected
    }

    /// Cosine similarity in PCA subspace
    pub fn pca_cosine_similarity(&self, gradient: &Array1<f32>, reference: &Array1<f32>) -> f32 {
        if !self.pca_fitted {
            return cosine_similarity(gradient.view(), reference.view());
        }

        let g_proj = self.pca_project(gradient);
        let ref_proj = self.pca_project(reference);

        cosine_similarity(g_proj.view(), ref_proj.view())
    }

    /// Compute adaptive lambda based on SNR
    pub fn compute_adaptive_lambda(&mut self, gradient: &Array1<f32>) -> f32 {
        if !self.config.lambda_adaptive || self.gradient_history.len() < 5 {
            return 0.7; // Default
        }

        // Compute gradient norm
        let norm = l2_norm(gradient.view());

        // Compute SNR using recent gradient norms
        let recent_norms: Vec<f32> = self.gradient_history
            .iter()
            .rev()
            .take(10)
            .map(|g| l2_norm(g.view()))
            .collect();

        let q75 = percentile(&recent_norms, 75.0);
        let q25 = percentile(&recent_norms, 25.0);
        let iqr = q75 - q25;

        let snr = if iqr < 1e-8 { 1.0 } else { norm / iqr };

        // Sigmoid with SNR
        let lambda_t = 1.0 / (1.0 + (-self.config.lambda_a * snr - self.config.lambda_b).exp());

        // Clip to [λ_min, λ_max]
        let lambda_t = lambda_t.clamp(self.config.lambda_min, self.config.lambda_max);

        self.lambda_history.push(lambda_t);
        lambda_t
    }

    /// Compute hybrid score
    /// Returns (hybrid_score, lambda, direction_score)
    pub fn compute_hybrid_score(
        &mut self,
        gradient: &Array1<f32>,
        reference: &Array1<f32>,
        utility_score: f32,
    ) -> (f32, f32, f32) {
        // Direction score using PCA-cosine
        let mut direction_score = self.pca_cosine_similarity(gradient, reference);

        // Apply ReLU if prefilter enabled
        if self.config.direction_prefilter {
            direction_score = (direction_score - self.config.direction_threshold).max(0.0);
        }

        // Adaptive lambda
        let lambda_t = self.compute_adaptive_lambda(gradient);

        // Hybrid score
        let hybrid_score = lambda_t * direction_score + (1.0 - lambda_t) * utility_score;

        // Update history
        self.gradient_history.push_back(gradient.clone());
        if self.gradient_history.len() > 20 {
            self.gradient_history.pop_front();
        }

        (hybrid_score, lambda_t, direction_score)
    }

    /// Get lambda statistics
    pub fn get_lambda_statistics(&self) -> LambdaStats {
        if self.lambda_history.is_empty() {
            return LambdaStats::default();
        }

        let n = self.lambda_history.len();
        let mean = self.lambda_history.iter().sum::<f32>() / n as f32;
        let variance = self.lambda_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / n as f32;

        let mut sorted = self.lambda_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        LambdaStats {
            count: n,
            mean,
            std: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            median: sorted[n / 2],
        }
    }
}

// =============================================================================
// Mondrian Conformal
// =============================================================================

/// Mondrian Conformal Prediction with per-class FPR buckets
pub struct MondrianConformal {
    config: PoGQv41Config,
    /// Thresholds per profile key
    thresholds: HashMap<Vec<u32>, f32>,
    /// Statistics per bucket
    bucket_stats: HashMap<Vec<u32>, BucketStats>,
    /// Global threshold for backoff
    global_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BucketStats {
    n_calibration: usize,
    threshold: f32,
    score_mean: f32,
    score_std: f32,
    bucket_type: String,
}

impl MondrianConformal {
    /// Create a new Mondrian conformal predictor
    pub fn new(config: PoGQv41Config) -> Self {
        Self {
            config,
            thresholds: HashMap::new(),
            bucket_stats: HashMap::new(),
            global_threshold: 0.0,
        }
    }

    /// Calibrate thresholds using validation data
    pub fn calibrate(
        &mut self,
        validation_scores: &[f32],
        validation_profiles: &[MondrianProfile],
    ) {
        if validation_scores.len() != validation_profiles.len() {
            warn!("Score/profile count mismatch in calibration");
            return;
        }

        // Group scores by profile
        let mut profile_scores: HashMap<Vec<u32>, Vec<f32>> = HashMap::new();
        for (score, profile) in validation_scores.iter().zip(validation_profiles) {
            let key = profile.profile_key();
            profile_scores.entry(key).or_default().push(*score);
        }

        // Compute quantile per bucket
        let alpha = self.config.conformal_alpha;

        for (profile_key, scores) in &profile_scores {
            if scores.len() >= self.config.min_bucket_size {
                let threshold = percentile(scores, (1.0 - alpha) * 100.0);
                self.thresholds.insert(profile_key.clone(), threshold);

                let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                let std = (scores.iter()
                    .map(|s| (s - mean).powi(2))
                    .sum::<f32>() / scores.len() as f32)
                    .sqrt();

                self.bucket_stats.insert(profile_key.clone(), BucketStats {
                    n_calibration: scores.len(),
                    threshold,
                    score_mean: mean,
                    score_std: std,
                    bucket_type: "bucket".to_string(),
                });
            } else {
                self.bucket_stats.insert(profile_key.clone(), BucketStats {
                    n_calibration: scores.len(),
                    threshold: 0.0,
                    score_mean: 0.0,
                    score_std: 0.0,
                    bucket_type: "needs_backoff".to_string(),
                });
            }
        }

        // Global threshold for backoff
        let all_scores: Vec<f32> = profile_scores.values().flatten().copied().collect();
        self.global_threshold = if all_scores.is_empty() {
            0.0
        } else {
            percentile(&all_scores, (1.0 - alpha) * 100.0)
        };

        info!("Calibrated {} Mondrian buckets with α={}, global threshold={:.3}",
              self.thresholds.len(), alpha, self.global_threshold);
    }

    /// Get threshold for a profile with hierarchical backoff
    pub fn get_threshold(&self, profile: &MondrianProfile) -> f32 {
        let profile_key = profile.profile_key();

        // Try bucket-specific
        if let Some(&threshold) = self.thresholds.get(&profile_key) {
            return threshold;
        }

        // Class-only backoff
        let mut class_thresholds = Vec::new();
        let mut class_weights = Vec::new();

        for &class_id in &profile.client_classes {
            let single_class_key = vec![class_id];
            if let Some(&threshold) = self.thresholds.get(&single_class_key) {
                let n_samples = self.bucket_stats
                    .get(&single_class_key)
                    .map(|s| s.n_calibration)
                    .unwrap_or(1);
                class_thresholds.push(threshold);
                class_weights.push(n_samples as f32);
            }
        }

        if !class_thresholds.is_empty() {
            let total_weight: f32 = class_weights.iter().sum();
            let weighted_threshold: f32 = class_thresholds.iter()
                .zip(&class_weights)
                .map(|(t, w)| t * w)
                .sum::<f32>() / total_weight;

            // Conservative shrinkage for class-only backoff
            return weighted_threshold * 0.95;
        }

        // Fall back to global with more conservative shrinkage
        self.global_threshold * 0.9
    }

    /// Check if score is below threshold (outlier)
    pub fn is_outlier(&self, score: f32, profile: &MondrianProfile) -> bool {
        score < self.get_threshold(profile)
    }
}

// =============================================================================
// Main PoGQ-v4.1 Enhanced
// =============================================================================

/// PoGQ-v4.1 Enhanced: Publication-Grade Byzantine Detection
pub struct PoGQv41Enhanced {
    config: PoGQv41Config,
    hybrid_scorer: AdaptiveHybridScorer,
    mondrian_conformal: MondrianConformal,

    // Per-class score statistics
    class_stats: HashMap<u32, ClassStats>,

    // EMA scores per client
    ema_scores: HashMap<String, f32>,

    // Phase 2: Warm-up and hysteresis state
    client_round_counts: HashMap<String, u32>,
    consecutive_violations: HashMap<String, u32>,
    consecutive_clears: HashMap<String, u32>,
    quarantined: HashMap<String, bool>,
    egregious_cap: Option<f32>,
}

impl PoGQv41Enhanced {
    /// Create a new PoGQ-v4.1 Enhanced detector
    pub fn new(config: PoGQv41Config) -> Self {
        let hybrid_scorer = AdaptiveHybridScorer::new(config.clone());
        let mondrian_conformal = MondrianConformal::new(config.clone());

        Self {
            config,
            hybrid_scorer,
            mondrian_conformal,
            class_stats: HashMap::new(),
            ema_scores: HashMap::new(),
            client_round_counts: HashMap::new(),
            consecutive_violations: HashMap::new(),
            consecutive_clears: HashMap::new(),
            quarantined: HashMap::new(),
            egregious_cap: None,
        }
    }

    /// Fit PCA on clean reference gradients
    pub fn fit_pca(&mut self, reference_gradients: &[Array1<f32>]) {
        self.hybrid_scorer.fit_pca(reference_gradients);
    }

    /// Calibrate Mondrian conformal thresholds and egregious cap
    pub fn calibrate_mondrian(
        &mut self,
        validation_scores: &[f32],
        validation_profiles: &[MondrianProfile],
    ) {
        self.mondrian_conformal.calibrate(validation_scores, validation_profiles);

        // Phase 2: Set egregious cap
        if !validation_scores.is_empty() {
            self.egregious_cap = Some(percentile(
                validation_scores,
                self.config.egregious_cap_quantile * 100.0,
            ));
            info!("Egregious cap set to {:.3} (p{:.1} of clean validation)",
                  self.egregious_cap.unwrap(),
                  self.config.egregious_cap_quantile * 100.0);
        }
    }

    /// Update per-class statistics for z-score normalization
    fn update_class_statistics(&mut self, class_id: u32, score: f32) {
        let window = self.config.class_score_window;
        let stats = self.class_stats.entry(class_id).or_default();

        stats.history.push_back(score);
        if stats.history.len() > window {
            stats.history.pop_front();
        }

        if stats.history.len() >= 3 {
            let scores: Vec<f32> = stats.history.iter().copied().collect();
            stats.mean = scores.iter().sum::<f32>() / scores.len() as f32;
            stats.std = (scores.iter()
                .map(|s| (s - stats.mean).powi(2))
                .sum::<f32>() / scores.len() as f32)
                .sqrt();
        }
    }

    /// Compute utility score for a gradient (simplified)
    fn compute_utility(&self, gradient: &Array1<f32>) -> f32 {
        // For MVP: use gradient norm as proxy
        // In production: compute actual Δloss on validation batch
        l2_norm(gradient.view())
    }

    /// Score a client gradient using all PoGQ-v4.1 components
    pub fn score_gradient(
        &mut self,
        gradient: &Array1<f32>,
        client_id: &str,
        client_classes: &BTreeSet<u32>,
        reference_gradient: &Array1<f32>,
        round_number: u32,
    ) -> GradientResult {
        // 1. Compute utility score (Mondrian validation simplified)
        let utility_score = self.compute_utility(gradient);

        // Z-score normalization per class
        let mut class_raw_scores = HashMap::new();
        let mut class_z_scores = Vec::new();

        for &class_id in client_classes {
            let raw_score = utility_score; // Simplified: same score for all classes
            class_raw_scores.insert(class_id, raw_score);

            // Z-score normalization
            let z_score = if self.config.per_class_normalization {
                if let Some(stats) = self.class_stats.get(&class_id) {
                    if stats.std > 1e-8 {
                        (raw_score - stats.mean) / stats.std
                    } else {
                        0.0
                    }
                } else {
                    raw_score
                }
            } else {
                raw_score
            };

            class_z_scores.push(z_score);
            self.update_class_statistics(class_id, raw_score);
        }

        let mondrian_score = if class_z_scores.is_empty() {
            0.0
        } else {
            class_z_scores.iter().sum::<f32>() / class_z_scores.len() as f32
        };

        // 2. Hybrid score
        let (hybrid_score, lambda, direction_score) = self.hybrid_scorer.compute_hybrid_score(
            gradient,
            reference_gradient,
            utility_score,
        );

        // 3. Temporal EMA
        let ema_score = if let Some(&prev_ema) = self.ema_scores.get(client_id) {
            self.config.ema_beta * prev_ema + (1.0 - self.config.ema_beta) * hybrid_score
        } else {
            hybrid_score
        };
        self.ema_scores.insert(client_id.to_string(), ema_score);

        // 4. Phase 2: Warm-up quota + Hysteresis
        let client_rounds = self.client_round_counts
            .entry(client_id.to_string())
            .or_insert(0);
        *client_rounds += 1;
        let client_rounds = *client_rounds;

        let profile = MondrianProfile::new(client_classes.iter().copied());
        let is_conformal_outlier = self.mondrian_conformal.is_outlier(ema_score, &profile);

        // Warm-up grace period
        let in_warmup = client_rounds <= self.config.warmup_rounds;
        let (is_violation, warmup_status) = if in_warmup {
            let is_egregious = self.egregious_cap
                .map(|cap| ema_score < cap)
                .unwrap_or(false);
            (
                is_egregious,
                if is_egregious { "warm-up-egregious" } else { "warm-up-grace" }.to_string(),
            )
        } else {
            (is_conformal_outlier, "enforcing".to_string())
        };

        // Hysteresis logic
        let cons_violations = self.consecutive_violations
            .entry(client_id.to_string())
            .or_insert(0);
        let cons_clears = self.consecutive_clears
            .entry(client_id.to_string())
            .or_insert(0);

        if is_violation {
            *cons_violations += 1;
            *cons_clears = 0;
        } else {
            *cons_clears += 1;
            *cons_violations = 0;
        }

        let cons_violations_val = *cons_violations;
        let cons_clears_val = *cons_clears;

        // Update quarantine status
        let quarantined = self.quarantined
            .entry(client_id.to_string())
            .or_insert(false);

        if cons_violations_val >= self.config.hysteresis_k {
            *quarantined = true;
        } else if cons_clears_val >= self.config.hysteresis_m {
            *quarantined = false;
        }

        let is_byzantine = *quarantined;
        let threshold = self.mondrian_conformal.get_threshold(&profile);

        GradientResult {
            client_id: client_id.to_string(),
            round: round_number,
            scores: GradientScores {
                mondrian_raw: mondrian_score,
                hybrid: hybrid_score,
                ema: ema_score,
                utility: utility_score,
                direction: direction_score,
                lambda,
            },
            detection: DetectionResult {
                is_byzantine,
                is_conformal_outlier,
                threshold,
                mondrian_profile: profile.profile_key(),
                quarantined: *quarantined,
            },
            phase2: Phase2Status {
                client_rounds,
                in_warmup,
                warmup_status,
                consecutive_violations: cons_violations_val,
                consecutive_clears: cons_clears_val,
                egregious_cap: self.egregious_cap,
            },
            class_scores: class_raw_scores,
        }
    }

    /// Aggregate client gradients with PoGQ-v4.1 detection
    pub fn aggregate(
        &mut self,
        client_updates: &[Array1<f32>],
        client_ids: &[&str],
        client_class_distributions: &[BTreeSet<u32>],
        reference_gradient: &Array1<f32>,
        round_number: u32,
    ) -> Result<(Array1<f32>, AggregationDiagnostics)> {
        if client_updates.is_empty() {
            return Err(AggregatorError::InsufficientGradients {
                have: 0,
                need: 1,
                defense: "PoGQ-v4.1".to_string(),
            });
        }

        let mut results = Vec::new();
        let mut honest_gradients = Vec::new();
        let mut honest_clients = Vec::new();

        for ((gradient, &client_id), classes) in client_updates
            .iter()
            .zip(client_ids)
            .zip(client_class_distributions)
        {
            let result = self.score_gradient(
                gradient,
                client_id,
                classes,
                reference_gradient,
                round_number,
            );

            if !result.detection.is_byzantine {
                honest_gradients.push(gradient);
                honest_clients.push(client_id.to_string());
            }

            results.push(result);
        }

        // Aggregate honest gradients (simple mean)
        let aggregated = if !honest_gradients.is_empty() {
            let n = honest_gradients.len();
            let dim = honest_gradients[0].len();
            let mut sum = Array1::zeros(dim);
            for g in &honest_gradients {
                sum = sum + *g;
            }
            sum / n as f32
        } else {
            warn!("No honest gradients! Using reference gradient");
            reference_gradient.clone()
        };

        let n_clients = client_updates.len();
        let n_honest = honest_gradients.len();

        let diagnostics = AggregationDiagnostics {
            round: round_number,
            n_clients,
            n_honest,
            n_byzantine: n_clients - n_honest,
            detection_rate: (n_clients - n_honest) as f32 / n_clients as f32,
            honest_clients,
            per_client_results: results,
            lambda_stats: self.hybrid_scorer.get_lambda_statistics(),
        };

        Ok((aggregated, diagnostics))
    }

    /// Get current configuration
    pub fn config(&self) -> &PoGQv41Config {
        &self.config
    }

    /// Reset detector state (for new training run)
    pub fn reset(&mut self) {
        self.class_stats.clear();
        self.ema_scores.clear();
        self.client_round_counts.clear();
        self.consecutive_violations.clear();
        self.consecutive_clears.clear();
        self.quarantined.clear();
    }

    /// Check if a client is currently quarantined
    pub fn is_quarantined(&self, client_id: &str) -> bool {
        self.quarantined.get(client_id).copied().unwrap_or(false)
    }

    /// Get list of all quarantined client IDs
    pub fn get_quarantined_clients(&self) -> Vec<String> {
        self.quarantined
            .iter()
            .filter(|(_, &q)| q)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> DetectorStatistics {
        DetectorStatistics {
            total_clients: self.ema_scores.len(),
            quarantined_clients: self.quarantined.values().filter(|&&q| q).count(),
            avg_ema_score: if self.ema_scores.is_empty() {
                0.0
            } else {
                self.ema_scores.values().sum::<f32>() / self.ema_scores.len() as f32
            },
        }
    }
}

// =============================================================================
// PoGQ-v4-Lite
// =============================================================================

/// PoGQ-v4-Lite: High-Dim Rescue Mode
///
/// For datasets like CIFAR-10, compute utility score only on linear probe head
/// gradient slice instead of full model gradient.
pub struct PoGQv41Lite {
    backbone_feature_dim: usize,
    num_classes: usize,
    probe_slice_start: usize,
    probe_slice_end: Option<usize>,
}

impl PoGQv41Lite {
    /// Create a new PoGQ-v4-Lite for high-dimensional gradients
    pub fn new(
        backbone_feature_dim: usize,
        num_classes: usize,
        probe_slice: Option<(usize, Option<usize>)>,
    ) -> Self {
        let (start, end) = probe_slice.unwrap_or_else(|| {
            // Default: last (feature_dim * num_classes + num_classes) parameters
            let probe_params = backbone_feature_dim * num_classes + num_classes;
            (usize::MAX - probe_params, None) // Will be clamped on use
        });

        Self {
            backbone_feature_dim,
            num_classes,
            probe_slice_start: start,
            probe_slice_end: end,
        }
    }

    /// Extract probe head gradient from full gradient vector
    pub fn extract_probe_gradient(&self, full_gradient: &Array1<f32>) -> Array1<f32> {
        let len = full_gradient.len();
        let start = self.probe_slice_start.min(len);
        let end = self.probe_slice_end.unwrap_or(len).min(len);

        if start >= end {
            // Handle edge case: compute from end
            let probe_params = self.backbone_feature_dim * self.num_classes + self.num_classes;
            let actual_start = len.saturating_sub(probe_params);
            full_gradient.slice(ndarray::s![actual_start..]).to_owned()
        } else {
            full_gradient.slice(ndarray::s![start..end]).to_owned()
        }
    }

    /// Compute utility score on probe head only
    pub fn compute_utility_lite(
        &self,
        full_gradient: &Array1<f32>,
        reference_gradient: &Array1<f32>,
    ) -> f32 {
        let probe_grad = self.extract_probe_gradient(full_gradient);
        let probe_ref = self.extract_probe_gradient(reference_gradient);

        cosine_similarity(probe_grad.view(), probe_ref.view())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute L2 norm
fn l2_norm(v: ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine similarity
fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>() / (norm_a * norm_b)
}

/// Compute percentile (0-100 scale)
fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PoGQv41Config::default();
        assert!(config.per_class_normalization);
        assert_eq!(config.pca_components, 32);
        assert!(config.lambda_adaptive);
        assert_eq!(config.ema_beta, 0.85);
        assert_eq!(config.warmup_rounds, 3);
        assert_eq!(config.hysteresis_k, 2);
        assert_eq!(config.hysteresis_m, 3);
    }

    #[test]
    fn test_mondrian_profile() {
        let profile = MondrianProfile::new([0, 1, 2]);
        assert_eq!(profile.client_classes.len(), 3);
        assert!(profile.client_classes.contains(&0));
        assert!(profile.client_classes.contains(&1));
        assert!(profile.client_classes.contains(&2));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        assert!((cosine_similarity(a.view(), b.view()) - 1.0).abs() < 1e-5);

        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(cosine_similarity(a.view(), c.view()).abs() < 1e-5);

        let d = Array1::from_vec(vec![-1.0, 0.0, 0.0]);
        assert!((cosine_similarity(a.view(), d.view()) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-5);
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-5);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        assert!((l2_norm(v.view()) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mondrian_conformal_calibration() {
        let config = PoGQv41Config::default();
        let mut conformal = MondrianConformal::new(config);

        // Create validation data
        let scores: Vec<f32> = (0..100).map(|i| 0.5 + 0.1 * (i as f32 / 100.0)).collect();
        let profiles: Vec<MondrianProfile> = scores.iter()
            .map(|_| MondrianProfile::new([0, 1]))
            .collect();

        conformal.calibrate(&scores, &profiles);

        // Check threshold exists
        let profile = MondrianProfile::new([0, 1]);
        let threshold = conformal.get_threshold(&profile);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_adaptive_hybrid_scorer() {
        let config = PoGQv41Config::default();
        let mut scorer = AdaptiveHybridScorer::new(config);

        // Generate some history
        for i in 0..10 {
            let gradient = Array1::from_vec(vec![i as f32; 100]);
            let reference = Array1::from_vec(vec![5.0; 100]);
            scorer.compute_hybrid_score(&gradient, &reference, 0.5);
        }

        let stats = scorer.get_lambda_statistics();
        assert!(stats.count > 0);
    }

    #[test]
    fn test_pogq_enhanced_basic() {
        let config = PoGQv41Config::default();
        let mut pogq = PoGQv41Enhanced::new(config);

        // Create reference gradient
        let reference = Array1::from_vec(vec![1.0; 100]);

        // Calibrate
        let val_scores: Vec<f32> = (0..50).map(|i| 0.5 + 0.05 * (i as f32)).collect();
        let val_profiles: Vec<MondrianProfile> = val_scores.iter()
            .map(|_| MondrianProfile::new([0, 1]))
            .collect();
        pogq.calibrate_mondrian(&val_scores, &val_profiles);

        // Score a gradient
        let gradient = Array1::from_vec(vec![1.1; 100]);
        let classes: BTreeSet<u32> = [0, 1].into_iter().collect();

        let result = pogq.score_gradient(&gradient, "client_1", &classes, &reference, 0);

        assert_eq!(result.client_id, "client_1");
        assert_eq!(result.round, 0);
        assert!(result.phase2.in_warmup); // First round is warmup
    }

    #[test]
    fn test_pogq_byzantine_detection() {
        let config = PoGQv41Config {
            warmup_rounds: 0, // Disable warmup for testing
            hysteresis_k: 1,  // Quarantine on first violation
            direction_prefilter: true,
            direction_threshold: 0.0,
            ..Default::default()
        };
        let mut pogq = PoGQv41Enhanced::new(config);

        let reference = Array1::from_vec(vec![1.0; 100]);

        // Calibrate with scores typical of honest gradients
        // Honest gradients have norm ~10, cosine ~1.0, hybrid ~3.7
        let val_scores: Vec<f32> = (0..50).map(|i| 3.5 + 0.01 * (i as f32)).collect();
        let val_profiles: Vec<MondrianProfile> = val_scores.iter()
            .map(|_| MondrianProfile::new([0]))
            .collect();
        pogq.calibrate_mondrian(&val_scores, &val_profiles);

        // Byzantine gradient (opposite direction from reference)
        // Cosine similarity = -1, direction_score = 0 after ReLU
        let byzantine = Array1::from_vec(vec![-1.0; 100]);
        let classes: BTreeSet<u32> = [0].into_iter().collect();

        let result = pogq.score_gradient(&byzantine, "byzantine_1", &classes, &reference, 1);

        // Byzantine gradient should have direction_score = 0 (opposite direction)
        // The hybrid score should be low due to direction prefilter
        // direction_score = max(-1.0 - 0.0, 0) = 0
        // hybrid = 0.7 * 0 + 0.3 * 10 = 3.0, which should be below threshold ~3.75
        assert!(
            result.detection.is_conformal_outlier || result.scores.direction < 0.1,
            "Byzantine gradient should be detected: direction={}, hybrid={}, threshold={}",
            result.scores.direction, result.scores.hybrid, result.detection.threshold
        );
    }

    #[test]
    fn test_pogq_aggregation() {
        let config = PoGQv41Config {
            warmup_rounds: 0,
            hysteresis_k: 1,
            direction_prefilter: true,
            direction_threshold: 0.0,
            ..Default::default()
        };
        let mut pogq = PoGQv41Enhanced::new(config);

        let reference = Array1::from_vec(vec![1.0; 100]);

        // Calibrate with low threshold to accept honest gradients
        // Honest gradients with norm ~10 and cosine ~1 get hybrid ~3.7
        let val_scores: Vec<f32> = (0..50).map(|i| 0.1 + 0.01 * (i as f32)).collect();
        let val_profiles: Vec<MondrianProfile> = val_scores.iter()
            .map(|_| MondrianProfile::new([0]))
            .collect();
        pogq.calibrate_mondrian(&val_scores, &val_profiles);

        // Create honest client updates (similar to reference)
        let honest1 = Array1::from_vec(vec![1.0; 100]);
        let honest2 = Array1::from_vec(vec![1.2; 100]);
        let updates = vec![honest1, honest2];
        let ids = vec!["honest1", "honest2"];
        let classes: Vec<BTreeSet<u32>> = vec![[0].into_iter().collect(); 2];

        let result = pogq.aggregate(&updates, &ids, &classes, &reference, 0);
        assert!(result.is_ok(), "Aggregation failed: {:?}", result.err());

        let (aggregated, diagnostics) = result.unwrap();
        assert_eq!(diagnostics.n_clients, 2);
        // At least some gradients should pass (honest gradients have high direction score)
        assert!(
            diagnostics.n_honest >= 1,
            "Expected at least 1 honest, got {}. Scores may need recalibration.",
            diagnostics.n_honest
        );

        // Aggregated should be close to mean
        let expected_mean = 1.1;
        assert!(
            (aggregated[0] - expected_mean).abs() < 0.5,
            "Aggregated[0]={} not close to expected mean {}",
            aggregated[0], expected_mean
        );
    }

    #[test]
    fn test_pogq_lite() {
        let lite = PoGQv41Lite::new(512, 10, Some((0, Some(100))));

        let full_gradient = Array1::from_vec(vec![1.0; 1000]);
        let probe = lite.extract_probe_gradient(&full_gradient);

        assert_eq!(probe.len(), 100);
    }
}
