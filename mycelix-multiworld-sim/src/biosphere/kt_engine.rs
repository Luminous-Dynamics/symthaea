// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! K(t) civilizational coherence computation — ported from Python.
//!
//! Original: kosmic-lab/historical_k/{etl.py, compute_k.py, aggregation_methods.py}
//! This port covers the core ETL pipeline: feature loading, normalization,
//! harmony frame construction, and K(t) geometric mean computation.
//! Plotting and K-Codex metadata remain in Python.

use std::collections::HashMap;

/// Normalization strategy for feature series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormStrategy {
    /// No normalization — raw values.
    None,
    /// Global z-score: (x - μ) / σ.
    ZscoreGlobal,
    /// Global min-max: (x - min) / (max - min).
    MinmaxGlobal,
    /// Z-score within each century bucket.
    ZscoreByCentury,
    /// Min-max within each century bucket.
    MinmaxByCentury,
}

impl NormStrategy {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "zscore_global" => Self::ZscoreGlobal,
            "minmax_global" => Self::MinmaxGlobal,
            "zscore_by_century" => Self::ZscoreByCentury,
            "minmax_by_century" => Self::MinmaxByCentury,
            _ => Self::None,
        }
    }
}

/// A feature time series (year → value).
#[derive(Debug, Clone)]
pub struct FeatureSeries {
    pub years: Vec<i32>,
    pub values: Vec<f64>,
}

impl FeatureSeries {
    /// Load from CSV string (columns: year, value).
    /// Missing values → linear interpolation → forward-fill → backward-fill → 0.0.
    pub fn from_csv(csv_data: &str, target_years: &[i32]) -> Self {
        let mut raw: Vec<(i32, f64)> = csv_data
            .lines()
            .skip(1)
            .filter_map(|line| {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let year: i32 = parts[0].trim().parse().ok()?;
                    let value: f64 = parts[1].trim().parse().ok()?;
                    Some((year, value))
                } else {
                    None
                }
            })
            .collect();
        raw.sort_by_key(|(y, _)| *y);
        raw.dedup_by_key(|(y, _)| *y);

        let mut values = Vec::with_capacity(target_years.len());
        for &year in target_years {
            let val = interpolate_raw(&raw, year);
            values.push(val);
        }

        Self {
            years: target_years.to_vec(),
            values,
        }
    }

    /// Create a zero-filled fallback series (deterministic, for missing data).
    pub fn zeros(target_years: &[i32]) -> Self {
        Self {
            years: target_years.to_vec(),
            values: vec![0.0; target_years.len()],
        }
    }

    /// Normalize in-place using the given strategy.
    pub fn normalize(&mut self, strategy: NormStrategy) {
        match strategy {
            NormStrategy::None => {}
            NormStrategy::ZscoreGlobal => {
                let (mu, sigma) = mean_std(&self.values);
                let sigma = if sigma == 0.0 { 1.0 } else { sigma };
                for v in &mut self.values {
                    *v = (*v - mu) / sigma;
                }
            }
            NormStrategy::MinmaxGlobal => {
                let (lo, hi) = min_max(&self.values);
                let range = if (hi - lo).abs() < 1e-15 {
                    1.0
                } else {
                    hi - lo
                };
                for v in &mut self.values {
                    *v = (*v - lo) / range;
                }
            }
            NormStrategy::ZscoreByCentury => {
                self.grouped_transform(|vals| {
                    let (mu, sigma) = mean_std(vals);
                    let sigma = if sigma == 0.0 { 1.0 } else { sigma };
                    for v in vals {
                        *v = (*v - mu) / sigma;
                    }
                });
            }
            NormStrategy::MinmaxByCentury => {
                self.grouped_transform(|vals| {
                    let (lo, hi) = min_max(vals);
                    let range = if (hi - lo).abs() < 1e-15 {
                        1.0
                    } else {
                        hi - lo
                    };
                    for v in vals {
                        *v = (*v - lo) / range;
                    }
                });
            }
        }
    }

    /// Apply a transformation function to each century group.
    fn grouped_transform<F: FnMut(&mut [f64])>(&mut self, mut f: F) {
        // Group indices by century
        let mut groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &year) in self.years.iter().enumerate() {
            let century = (year as f64 / 100.0).floor() as i32 * 100;
            groups.entry(century).or_default().push(i);
        }

        for indices in groups.values() {
            let mut group_vals: Vec<f64> = indices.iter().map(|&i| self.values[i]).collect();
            f(&mut group_vals);
            for (j, &i) in indices.iter().enumerate() {
                self.values[i] = group_vals[j];
            }
        }
    }
}

/// Linear interpolation + forward-fill + backward-fill + zero fallback.
fn interpolate_raw(sorted: &[(i32, f64)], target_year: i32) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    // Exact match
    if let Some(&(_, v)) = sorted.iter().find(|(y, _)| *y == target_year) {
        return v;
    }

    // Interpolation between bracketing points
    for pair in sorted.windows(2) {
        let (y0, v0) = pair[0];
        let (y1, v1) = pair[1];
        if y0 < target_year && target_year < y1 {
            let t = (target_year - y0) as f64 / (y1 - y0) as f64;
            return v0 + t * (v1 - v0);
        }
    }

    // Extrapolation: clamp to nearest
    if target_year <= sorted[0].0 {
        sorted[0].1
    } else {
        sorted.last().unwrap().1
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let n = values.len() as f64;
    let mu = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
    (mu, var.sqrt())
}

fn min_max(values: &[f64]) -> (f64, f64) {
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

/// K(t) aggregation method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationMethod {
    /// Geometric mean (Path C default): K = exp(mean(log(H_i))).
    /// Enforces non-substitutability (weakest link coordination).
    Geometric,
    /// Arithmetic mean (original): K = mean(H_i).
    /// Allows substitutability across harmonies.
    Arithmetic,
}

/// The K(t) computation engine.
///
/// Manages harmony frames and computes the K-index time series.
pub struct KtEngine {
    /// Year labels.
    pub years: Vec<i32>,
    /// Harmony columns (one Vec<f64> per harmony, aligned with years).
    pub harmony_columns: Vec<Vec<f64>>,
    /// Harmony names.
    pub harmony_names: Vec<String>,
}

impl KtEngine {
    /// Build from a map of (harmony_name → Vec<FeatureSeries>).
    /// Each harmony's value = mean of its features at each year.
    pub fn from_harmony_features(
        years: &[i32],
        harmonies: &HashMap<String, Vec<FeatureSeries>>,
    ) -> Self {
        let mut harmony_names = Vec::new();
        let mut harmony_columns = Vec::new();

        let mut sorted_names: Vec<&String> = harmonies.keys().collect();
        sorted_names.sort();

        for name in sorted_names {
            let features = &harmonies[name];
            let mut column = vec![0.0; years.len()];
            if !features.is_empty() {
                for i in 0..years.len() {
                    let sum: f64 = features.iter().map(|f| f.values[i]).sum();
                    column[i] = sum / features.len() as f64;
                }
            }
            harmony_names.push(name.clone());
            harmony_columns.push(column);
        }

        Self {
            years: years.to_vec(),
            harmony_names,
            harmony_columns,
        }
    }

    /// Compute K(t) series using the specified aggregation method.
    pub fn compute_k(
        &self,
        method: AggregationMethod,
        weights: Option<&HashMap<String, f64>>,
    ) -> Vec<f64> {
        let n = self.years.len();
        let h = self.harmony_columns.len();
        if h == 0 || n == 0 {
            return vec![];
        }

        // Normalize weights
        let w: Vec<f64> = if let Some(wt) = weights {
            let mut wvec: Vec<f64> = self
                .harmony_names
                .iter()
                .map(|name| *wt.get(name).unwrap_or(&(1.0 / h as f64)))
                .collect();
            let wsum: f64 = wvec.iter().sum();
            if wsum > 0.0 {
                for v in &mut wvec {
                    *v /= wsum;
                }
            }
            wvec
        } else {
            vec![1.0 / h as f64; h]
        };

        let mut k_values = Vec::with_capacity(n);

        for i in 0..n {
            let val = match method {
                AggregationMethod::Geometric => {
                    let epsilon = 1e-10;
                    let log_sum: f64 = (0..h)
                        .map(|j| w[j] * (self.harmony_columns[j][i] + epsilon).ln())
                        .sum();
                    log_sum.exp()
                }
                AggregationMethod::Arithmetic => {
                    (0..h).map(|j| w[j] * self.harmony_columns[j][i]).sum()
                }
            };
            k_values.push(val);
        }

        k_values
    }

    /// Bootstrap mean CI (scalar bounds on overall mean K).
    pub fn bootstrap_mean_ci(
        &self,
        k_values: &[f64],
        n_samples: usize,
        ci: f64,
        seed: u64,
    ) -> (f64, f64) {
        if n_samples == 0 || k_values.is_empty() {
            let mean = if k_values.is_empty() {
                0.0
            } else {
                k_values.iter().sum::<f64>() / k_values.len() as f64
            };
            return (mean, mean);
        }

        let n = k_values.len();
        let mut rng_state = seed;
        let mut means = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut sum = 0.0;
            for _ in 0..n {
                rng_state = lcg_next(rng_state);
                let idx = (rng_state as usize) % n;
                sum += k_values[idx];
            }
            means.push(sum / n as f64);
        }

        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 1.0 - ci;
        let lo_idx = ((alpha / 2.0) * means.len() as f64) as usize;
        let hi_idx = (((1.0 - alpha / 2.0) * means.len() as f64) as usize).min(means.len() - 1);

        (means[lo_idx], means[hi_idx])
    }
}

/// Simple LCG for deterministic bootstrap (no external RNG dependency).
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_minmax_global() {
        let years = vec![2000, 2010, 2020];
        let mut series = FeatureSeries {
            years: years.clone(),
            values: vec![10.0, 20.0, 30.0],
        };
        series.normalize(NormStrategy::MinmaxGlobal);
        assert!((series.values[0] - 0.0).abs() < 1e-10);
        assert!((series.values[1] - 0.5).abs() < 1e-10);
        assert!((series.values[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_zscore_global() {
        let years = vec![2000, 2010, 2020];
        let mut series = FeatureSeries {
            years: years.clone(),
            values: vec![10.0, 20.0, 30.0],
        };
        series.normalize(NormStrategy::ZscoreGlobal);
        let sum: f64 = series.values.iter().sum();
        assert!(
            (sum).abs() < 1e-10,
            "Z-score mean should be ~0, got {}",
            sum
        );
    }

    #[test]
    fn geometric_less_than_arithmetic() {
        let years = vec![2000, 2010, 2020];
        let mut harmonies = HashMap::new();
        harmonies.insert(
            "h1".to_string(),
            vec![FeatureSeries {
                years: years.clone(),
                values: vec![0.5, 0.6, 0.7],
            }],
        );
        harmonies.insert(
            "h2".to_string(),
            vec![FeatureSeries {
                years: years.clone(),
                values: vec![0.3, 0.8, 0.9],
            }],
        );

        let engine = KtEngine::from_harmony_features(&years, &harmonies);
        let geo = engine.compute_k(AggregationMethod::Geometric, None);
        let arith = engine.compute_k(AggregationMethod::Arithmetic, None);

        for i in 0..years.len() {
            assert!(
                geo[i] <= arith[i] + 1e-8,
                "Geometric ({}) should be <= arithmetic ({}) at year {}",
                geo[i],
                arith[i],
                years[i]
            );
        }
    }

    #[test]
    fn bootstrap_ci_contains_mean() {
        let k_values = vec![0.3, 0.5, 0.7, 0.4, 0.6];
        let years = vec![2000, 2010, 2020, 2030, 2040];
        let engine = KtEngine {
            years,
            harmony_columns: vec![],
            harmony_names: vec![],
        };
        let (lo, hi) = engine.bootstrap_mean_ci(&k_values, 1000, 0.95, 42);
        let mean: f64 = k_values.iter().sum::<f64>() / k_values.len() as f64;
        assert!(lo <= mean, "CI lower ({}) should be <= mean ({})", lo, mean);
        assert!(hi >= mean, "CI upper ({}) should be >= mean ({})", hi, mean);
    }

    #[test]
    fn interpolation_works() {
        let data = vec![(2000, 10.0), (2010, 20.0), (2020, 30.0)];
        assert!((interpolate_raw(&data, 2005) - 15.0).abs() < 0.01);
        assert!((interpolate_raw(&data, 2000) - 10.0).abs() < 0.01);
        assert!((interpolate_raw(&data, 2020) - 30.0).abs() < 0.01);
        // Extrapolation clamps
        assert!((interpolate_raw(&data, 1990) - 10.0).abs() < 0.01);
        assert!((interpolate_raw(&data, 2030) - 30.0).abs() < 0.01);
    }
}
