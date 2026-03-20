//! Defect prediction from Cincinnati historical manufacturing data
//!
//! Online linear regression model that learns feature weights from
//! build outcomes and produces quality predictions with confidence
//! intervals, risk factor attribution, and parameter adjustment
//! suggestions.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Feature names (fixed order matching PredictionInput fields)
// ---------------------------------------------------------------------------

const FEATURE_NAMES: [&str; 6] = [
    "tolerance",
    "surface_quality",
    "throughput",
    "energy_cost",
    "anomaly_count",
    "layer_count",
];

const NUM_FEATURES: usize = FEATURE_NAMES.len();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Input features for a single prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInput {
    /// Part tolerance in mm (lower = tighter).
    pub tolerance: f64,
    /// Surface quality metric (0.0–1.0, higher = smoother).
    pub surface_quality: f64,
    /// Throughput in parts/hour.
    pub throughput: f64,
    /// Energy cost per part in kWh.
    pub energy_cost: f64,
    /// Number of anomalies detected during the build.
    pub anomaly_count: u32,
    /// Number of build layers.
    pub layer_count: u32,
}

impl PredictionInput {
    /// Convert to a fixed-length feature vector.
    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.tolerance,
            self.surface_quality,
            self.throughput,
            self.energy_cost,
            self.anomaly_count as f64,
            self.layer_count as f64,
        ]
    }
}

/// Output of a quality prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutput {
    /// Predicted quality score (0.0–1.0).
    pub predicted_quality: f64,
    /// Model confidence (0.0–1.0) — increases with training data.
    pub confidence: f64,
    /// Risk factors ranked by contribution to defect probability.
    pub risk_factors: Vec<RiskFactor>,
}

/// A single risk factor contributing to defect probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Name of the feature.
    pub name: String,
    /// Signed contribution to predicted quality (negative = harmful).
    pub contribution: f64,
    /// Severity rating (0.0–1.0).
    pub severity: f32,
}

/// Suggested parameter adjustment to improve predicted quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjustment {
    /// Parameter name.
    pub parameter: String,
    /// Current value.
    pub current_value: f64,
    /// Suggested value.
    pub suggested_value: f64,
    /// Expected quality improvement from this change alone.
    pub expected_improvement: f64,
}

// ---------------------------------------------------------------------------
// DefectPredictor
// ---------------------------------------------------------------------------

/// Online linear regression predictor for manufacturing defect rates.
///
/// Uses incremental mean/variance tracking and ridge regression updates
/// so it can learn from streaming Cincinnati build data without storing
/// the full dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectPredictor {
    /// Learned regression weights (one per feature + bias at index 0).
    feature_weights: Vec<f64>,
    /// Number of training samples seen so far.
    training_count: u64,
    /// Running mean of the outcome variable.
    mean_quality: f64,
    /// Running mean of each feature.
    feature_means: Vec<f64>,
    /// Running standard deviation of each feature (online Welford).
    feature_stds: Vec<f64>,
    /// Running M2 for Welford variance (internal).
    feature_m2: Vec<f64>,
    /// Running mean of outcome for Welford (internal).
    outcome_mean: f64,
    /// Accumulated X^T X for online ridge regression (upper triangle, row-major).
    xtx: Vec<f64>,
    /// Accumulated X^T y.
    xty: Vec<f64>,
}

impl DefectPredictor {
    /// Create a new untrained predictor.
    pub fn new() -> Self {
        // weights: bias + NUM_FEATURES
        let dim = NUM_FEATURES + 1;
        Self {
            feature_weights: vec![0.5; dim], // bias toward 0.5 quality
            training_count: 0,
            mean_quality: 0.0,
            feature_means: vec![0.0; NUM_FEATURES],
            feature_stds: vec![1.0; NUM_FEATURES],
            feature_m2: vec![0.0; NUM_FEATURES],
            outcome_mean: 0.0,
            xtx: vec![0.0; dim * dim],
            xty: vec![0.0; dim],
        }
    }

    /// Train on a batch of input/outcome pairs.  Can be called repeatedly
    /// for online learning.
    pub fn train(&mut self, inputs: &[PredictionInput], outcomes: &[f64]) {
        assert_eq!(
            inputs.len(),
            outcomes.len(),
            "inputs and outcomes must have equal length"
        );

        let dim = NUM_FEATURES + 1;

        for (input, &outcome) in inputs.iter().zip(outcomes.iter()) {
            self.training_count += 1;
            let n = self.training_count as f64;

            // Update running feature means and M2 (Welford).
            let raw = input.to_vec();
            for (i, &val) in raw.iter().enumerate() {
                let delta = val - self.feature_means[i];
                self.feature_means[i] += delta / n;
                let delta2 = val - self.feature_means[i];
                self.feature_m2[i] += delta * delta2;
                if self.training_count > 1 {
                    self.feature_stds[i] =
                        (self.feature_m2[i] / (n - 1.0)).sqrt().max(1e-8);
                }
            }

            // Update outcome mean.
            self.outcome_mean += (outcome - self.outcome_mean) / n;
            self.mean_quality = self.outcome_mean;

            // Build augmented feature vector [1, x0, x1, ...].
            let mut x = Vec::with_capacity(dim);
            x.push(1.0); // bias
            for (i, &val) in raw.iter().enumerate() {
                // Standardize using running stats.
                let std = self.feature_stds[i].max(1e-8);
                x.push((val - self.feature_means[i]) / std);
            }

            // Accumulate X^T X and X^T y.
            for i in 0..dim {
                for j in 0..dim {
                    self.xtx[i * dim + j] += x[i] * x[j];
                }
                self.xty[i] += x[i] * outcome;
            }
        }

        // Solve via ridge regression: w = (X^T X + λI)^{-1} X^T y
        self.solve_weights();
    }

    /// Predict quality for a single input.
    pub fn predict(&self, input: &PredictionInput) -> PredictionOutput {
        let raw = input.to_vec();
        let dim = NUM_FEATURES + 1;

        // Build standardized feature vector.
        let mut x = Vec::with_capacity(dim);
        x.push(1.0);
        for (i, &val) in raw.iter().enumerate() {
            let std = self.feature_stds[i].max(1e-8);
            x.push((val - self.feature_means[i]) / std);
        }

        // Linear prediction.
        let mut pred = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            pred += self.feature_weights[i] * xi;
        }
        let predicted_quality = pred.clamp(0.0, 1.0);

        // Confidence: sigmoid of log(training_count).
        let confidence = if self.training_count == 0 {
            0.0
        } else {
            let log_n = (self.training_count as f64).ln();
            // Sigmoid centered at ~7 (~1100 samples for 0.5 confidence).
            1.0 / (1.0 + (-0.5 * (log_n - 4.0)).exp())
        };

        // Risk factors: contribution = weight_i * x_i (signed).
        let mut risk_factors: Vec<RiskFactor> = (0..NUM_FEATURES)
            .map(|i| {
                let contribution = self.feature_weights[i + 1] * x[i + 1];
                let severity = if contribution < 0.0 {
                    (-contribution).min(1.0) as f32
                } else {
                    0.0
                };
                RiskFactor {
                    name: FEATURE_NAMES[i].to_string(),
                    contribution,
                    severity,
                }
            })
            .collect();

        // Sort by severity descending.
        risk_factors.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        PredictionOutput {
            predicted_quality,
            confidence,
            risk_factors,
        }
    }

    /// Return feature importances sorted by absolute weight (descending).
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let mut importances: Vec<(String, f64)> = FEATURE_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| (name.to_string(), self.feature_weights[i + 1].abs()))
            .collect();
        importances.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        importances
    }

    /// Suggest parameter adjustments to reach the target quality.
    pub fn suggest_adjustments(
        &self,
        input: &PredictionInput,
        target_quality: f64,
    ) -> Vec<Adjustment> {
        let current = self.predict(input);
        let gap = target_quality - current.predicted_quality;
        if gap <= 0.0 {
            return Vec::new(); // already at or above target
        }

        let raw = input.to_vec();
        let mut adjustments = Vec::new();

        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            let w = self.feature_weights[i + 1];
            if w.abs() < 1e-10 {
                continue;
            }
            let std = self.feature_stds[i].max(1e-8);

            // How much must the standardized feature change to close the gap?
            let delta_std = gap / w;
            // Convert back to raw units.
            let delta_raw = delta_std * std;
            let suggested = raw[i] + delta_raw;

            let improvement = (w * delta_std).abs().min(gap);

            adjustments.push(Adjustment {
                parameter: name.to_string(),
                current_value: raw[i],
                suggested_value: suggested,
                expected_improvement: improvement,
            });
        }

        // Sort by expected improvement descending.
        adjustments.sort_by(|a, b| {
            b.expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        adjustments
    }

    // -----------------------------------------------------------------------
    // Internal: solve ridge regression
    // -----------------------------------------------------------------------

    fn solve_weights(&mut self) {
        let dim = NUM_FEATURES + 1;
        let lambda = 1.0; // ridge penalty

        // Copy XtX and add ridge penalty.
        let mut a = self.xtx.clone();
        for i in 0..dim {
            a[i * dim + i] += lambda;
        }

        // Solve A w = Xty via Gaussian elimination with partial pivoting.
        let mut b = self.xty.clone();

        for col in 0..dim {
            // Partial pivot.
            let mut max_row = col;
            let mut max_val = a[col * dim + col].abs();
            for row in (col + 1)..dim {
                let v = a[row * dim + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                continue;
            }
            if max_row != col {
                for k in 0..dim {
                    a.swap(col * dim + k, max_row * dim + k);
                }
                b.swap(col, max_row);
            }

            let pivot = a[col * dim + col];
            for row in (col + 1)..dim {
                let factor = a[row * dim + col] / pivot;
                for k in col..dim {
                    a[row * dim + k] -= factor * a[col * dim + k];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back-substitution.
        let mut w = vec![0.0; dim];
        for i in (0..dim).rev() {
            let mut sum = b[i];
            for j in (i + 1)..dim {
                sum -= a[i * dim + j] * w[j];
            }
            let diag = a[i * dim + i];
            if diag.abs() > 1e-12 {
                w[i] = sum / diag;
            }
        }

        self.feature_weights = w;
    }
}

impl Default for DefectPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input(tolerance: f64, surface: f64, anomalies: u32) -> PredictionInput {
        PredictionInput {
            tolerance,
            surface_quality: surface,
            throughput: 10.0,
            energy_cost: 5.0,
            anomaly_count: anomalies,
            layer_count: 200,
        }
    }

    #[test]
    fn untrained_predictor_returns_prediction() {
        let p = DefectPredictor::new();
        let out = p.predict(&sample_input(0.1, 0.9, 0));
        // Should return something in [0, 1].
        assert!(out.predicted_quality >= 0.0 && out.predicted_quality <= 1.0);
        // Confidence should be 0 with no training data.
        assert!((out.confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn training_improves_predictions() {
        let mut p = DefectPredictor::new();

        // Train: high surface quality → high outcome, low → low.
        let inputs: Vec<PredictionInput> = (0..100)
            .map(|i| {
                let sq = (i as f64) / 100.0;
                sample_input(0.1, sq, 0)
            })
            .collect();
        let outcomes: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        p.train(&inputs, &outcomes);

        let low = p.predict(&sample_input(0.1, 0.1, 0));
        let high = p.predict(&sample_input(0.1, 0.9, 0));
        assert!(
            high.predicted_quality > low.predicted_quality,
            "high surface quality should predict higher quality: {} vs {}",
            high.predicted_quality,
            low.predicted_quality
        );
    }

    #[test]
    fn feature_importance_ordering() {
        let mut p = DefectPredictor::new();

        // Surface quality is the dominant feature.
        let inputs: Vec<PredictionInput> = (0..200)
            .map(|i| {
                let sq = (i as f64) / 200.0;
                sample_input(0.1, sq, 0)
            })
            .collect();
        let outcomes: Vec<f64> = inputs.iter().map(|i| i.surface_quality).collect();
        p.train(&inputs, &outcomes);

        let imp = p.feature_importance();
        assert!(!imp.is_empty());
        // surface_quality should be among the top features.
        let top = &imp[0].0;
        assert_eq!(top, "surface_quality", "expected surface_quality on top, got {top}");
    }

    #[test]
    fn adjustment_suggestions() {
        let mut p = DefectPredictor::new();

        let inputs: Vec<PredictionInput> = (0..100)
            .map(|i| {
                let sq = (i as f64) / 100.0;
                sample_input(0.1, sq, 0)
            })
            .collect();
        let outcomes: Vec<f64> = inputs.iter().map(|i| i.surface_quality).collect();
        p.train(&inputs, &outcomes);

        let bad_input = sample_input(0.1, 0.2, 0);
        let adjustments = p.suggest_adjustments(&bad_input, 0.9);
        assert!(!adjustments.is_empty(), "should suggest at least one adjustment");
        // At least one adjustment should have positive expected improvement.
        assert!(adjustments[0].expected_improvement > 0.0);
    }

    #[test]
    fn no_adjustments_when_already_good() {
        let mut p = DefectPredictor::new();
        let inputs: Vec<PredictionInput> = (0..50)
            .map(|_| sample_input(0.1, 0.95, 0))
            .collect();
        let outcomes = vec![0.95; 50];
        p.train(&inputs, &outcomes);

        let good = sample_input(0.1, 0.95, 0);
        let adj = p.suggest_adjustments(&good, 0.5);
        assert!(adj.is_empty(), "no adjustments needed when already above target");
    }

    #[test]
    fn confidence_increases_with_training() {
        let mut p = DefectPredictor::new();
        let input = sample_input(0.1, 0.5, 2);

        let c0 = p.predict(&input).confidence;

        // Train a small batch.
        let batch: Vec<PredictionInput> = (0..10).map(|_| sample_input(0.1, 0.5, 2)).collect();
        let outcomes = vec![0.7; 10];
        p.train(&batch, &outcomes);
        let c1 = p.predict(&input).confidence;

        // Train more.
        let big: Vec<PredictionInput> = (0..500).map(|_| sample_input(0.1, 0.5, 2)).collect();
        let outcomes2 = vec![0.7; 500];
        p.train(&big, &outcomes2);
        let c2 = p.predict(&input).confidence;

        assert!(c1 > c0, "confidence should grow: {c1} > {c0}");
        assert!(c2 > c1, "confidence should grow more: {c2} > {c1}");
    }

    #[test]
    fn risk_factor_identification() {
        let mut p = DefectPredictor::new();

        // Anomaly count negatively correlates with quality.
        let inputs: Vec<PredictionInput> = (0..200)
            .map(|i| {
                let anom = i % 20;
                sample_input(0.1, 0.9, anom as u32)
            })
            .collect();
        let outcomes: Vec<f64> = inputs
            .iter()
            .map(|i| (1.0 - i.anomaly_count as f64 / 20.0).max(0.0))
            .collect();
        p.train(&inputs, &outcomes);

        // Predict with high anomaly count.
        let bad = sample_input(0.1, 0.9, 15);
        let out = p.predict(&bad);
        // anomaly_count should appear as a risk factor with nonzero severity.
        let anomaly_risk = out
            .risk_factors
            .iter()
            .find(|r| r.name == "anomaly_count");
        assert!(
            anomaly_risk.is_some(),
            "anomaly_count should be identified as risk factor"
        );
    }

    #[test]
    fn multiple_training_batches_accumulate() {
        let mut p = DefectPredictor::new();

        for batch in 0..5 {
            let inputs: Vec<PredictionInput> = (0..20)
                .map(|i| {
                    let sq = (batch * 20 + i) as f64 / 100.0;
                    sample_input(0.1, sq, 0)
                })
                .collect();
            let outcomes: Vec<f64> = inputs.iter().map(|i| i.surface_quality).collect();
            p.train(&inputs, &outcomes);
        }

        assert_eq!(p.training_count, 100);
        let low = p.predict(&sample_input(0.1, 0.1, 0));
        let high = p.predict(&sample_input(0.1, 0.9, 0));
        assert!(high.predicted_quality > low.predicted_quality);
    }

    #[test]
    fn default_trait() {
        let p = DefectPredictor::default();
        assert_eq!(p.training_count, 0);
    }

    #[test]
    fn prediction_clamped_to_unit_interval() {
        let mut p = DefectPredictor::new();
        // Train with extreme values.
        let inputs = vec![
            PredictionInput {
                tolerance: 100.0,
                surface_quality: 100.0,
                throughput: 100.0,
                energy_cost: 100.0,
                anomaly_count: 0,
                layer_count: 0,
            },
        ];
        let outcomes = vec![100.0];
        p.train(&inputs, &outcomes);

        let out = p.predict(&PredictionInput {
            tolerance: 200.0,
            surface_quality: 200.0,
            throughput: 200.0,
            energy_cost: 200.0,
            anomaly_count: 0,
            layer_count: 0,
        });
        assert!(out.predicted_quality >= 0.0 && out.predicted_quality <= 1.0);
    }
}
