// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal Zero-Training Anomaly Detection Evaluation
//!
//! Evaluates HDC-based anomaly detection across three domains:
//! 1. Fusion (C-Mod) — plasma disruption detection (reference from evaluate_cmod)
//! 2. EEG Seizure — epileptic seizure detection from CHB-MIT dataset
//! 3. Seismic Events — earthquake vs noise classification from INSTANCE dataset
//!
//! Usage:
//!   cargo run -p symthaea-physics --example universal_anomaly_eval --release

use std::collections::HashMap;
use std::path::PathBuf;

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Simple HDC Evaluator ─────────────────────────────────────────────────────

struct SimpleHdcEvaluator {
    bases: Vec<ContinuousHV>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
    n_features: usize,
}

impl SimpleHdcEvaluator {
    fn new(n_features: usize, seed_base: u64) -> Self {
        eprintln!(
            "  Creating {} basis vectors (dim={})...",
            n_features, HDC_DIMENSION
        );
        let bases: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, seed_base + i as u64))
            .collect();

        Self {
            bases,
            mins: vec![f32::MAX; n_features],
            maxs: vec![f32::MIN; n_features],
            n_features,
        }
    }

    fn fit_normalization(&mut self, data: &[Vec<f32>]) {
        for row in data {
            for (i, &v) in row.iter().enumerate() {
                if v < self.mins[i] {
                    self.mins[i] = v;
                }
                if v > self.maxs[i] {
                    self.maxs[i] = v;
                }
            }
        }
    }

    fn normalize(&self, value: f32, idx: usize) -> f32 {
        let range = self.maxs[idx] - self.mins[idx];
        if range.abs() < 1e-10 {
            return 0.0;
        }
        // Normalize to [-1, 1]
        ((value - self.mins[idx]) / range) * 2.0 - 1.0
    }

    fn encode(&self, features: &[f32]) -> ContinuousHV {
        assert_eq!(features.len(), self.n_features);

        // Encode: bundle of (basis_i * normalized_value_i)
        let scaled: Vec<ContinuousHV> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let norm = self.normalize(v, i);
                let mut hv = self.bases[i].clone();
                for val in hv.values.iter_mut() {
                    *val *= norm;
                }
                hv
            })
            .collect();

        let refs: Vec<&ContinuousHV> = scaled.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    fn build_reference(&self, normal_data: &[Vec<f32>]) -> ContinuousHV {
        eprintln!(
            "  Building reference HV from {} normal samples...",
            normal_data.len()
        );
        let encoded: Vec<ContinuousHV> = normal_data.iter().map(|row| self.encode(row)).collect();
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Returns (auc, best_f1)
    fn evaluate(&self, reference: &ContinuousHV, test_data: &[(Vec<f32>, bool)]) -> (f64, f64) {
        eprintln!(
            "  Computing free energy for {} test samples...",
            test_data.len()
        );

        // Score each sample: higher = more anomalous
        // Free energy proxy: 1 - similarity to reference
        let scored: Vec<(f64, bool)> = test_data
            .iter()
            .map(|(features, is_anomaly)| {
                let hv = self.encode(features);
                let sim = hv.similarity(reference);
                let free_energy = 1.0 - sim as f64;
                (free_energy, *is_anomaly)
            })
            .collect();

        // Compute ROC curve
        let roc = compute_roc_curve(&scored);
        let auc = compute_auc(&roc);

        // Find best F1 by sweeping thresholds
        let best_f1 = find_best_f1(&scored);

        (auc, best_f1)
    }
}

// ── ROC / AUC Computation ────────────────────────────────────────────────────

struct RocPoint {
    fpr: f64,
    tpr: f64,
    #[allow(dead_code)]
    threshold: f64,
}

fn compute_roc_curve(scored: &[(f64, bool)]) -> Vec<RocPoint> {
    if scored.is_empty() {
        return vec![];
    }

    let total_pos = scored.iter().filter(|(_, lab)| *lab).count() as f64;
    let total_neg = scored.iter().filter(|(_, lab)| !*lab).count() as f64;

    if total_pos < 1.0 || total_neg < 1.0 {
        return vec![
            RocPoint {
                fpr: 0.0,
                tpr: 0.0,
                threshold: 1.0,
            },
            RocPoint {
                fpr: 1.0,
                tpr: 1.0,
                threshold: 0.0,
            },
        ];
    }

    // Use ~200 threshold steps for efficiency on large datasets
    let mut scores: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = scores.len();
    let step = (n / 200).max(1);
    let mut thresholds: Vec<f64> = Vec::with_capacity(202);
    thresholds.push(scores[n - 1] + 0.01); // all negative
    for i in (0..n).rev().step_by(step) {
        thresholds.push(scores[i]);
    }
    if thresholds.last().map_or(true, |&t| t > scores[0]) {
        thresholds.push(scores[0] - 0.01); // all positive
    }

    let mut points = Vec::with_capacity(thresholds.len());
    for &thresh in &thresholds {
        let mut tp = 0u64;
        let mut fp = 0u64;

        for &(score, is_pos) in scored {
            if score >= thresh {
                if is_pos {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }

        let tpr = tp as f64 / total_pos;
        let fpr = fp as f64 / total_neg;

        points.push(RocPoint {
            fpr,
            tpr,
            threshold: thresh,
        });
    }

    // Sort by FPR for trapezoidal integration
    points.sort_by(|a, b| {
        a.fpr
            .partial_cmp(&b.fpr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    points
}

fn compute_auc(roc: &[RocPoint]) -> f64 {
    if roc.len() < 2 {
        return 0.0;
    }
    let mut auc = 0.0;
    for i in 1..roc.len() {
        let dx = roc[i].fpr - roc[i - 1].fpr;
        let avg_tpr = (roc[i].tpr + roc[i - 1].tpr) / 2.0;
        auc += dx * avg_tpr;
    }
    auc.clamp(0.0, 1.0)
}

fn find_best_f1(scored: &[(f64, bool)]) -> f64 {
    let total_pos = scored.iter().filter(|(_, lab)| *lab).count() as f64;
    if total_pos < 1.0 {
        return 0.0;
    }

    let mut scores: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = scores.len();
    let step = (n / 200).max(1);
    let mut best_f1 = 0.0;

    for i in (0..n).step_by(step) {
        let thresh = scores[i];
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_ = 0u64;

        for &(score, is_pos) in scored {
            if score >= thresh {
                if is_pos {
                    tp += 1;
                } else {
                    fp += 1;
                }
            } else if is_pos {
                fn_ += 1;
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        if f1 > best_f1 {
            best_f1 = f1;
        }
    }

    best_f1
}

// ── EEG Seizure Detection ────────────────────────────────────────────────────

fn evaluate_eeg() -> anyhow::Result<(usize, usize, f64, f64)> {
    eprintln!("\n================================================================");
    eprintln!("  EEG Seizure Detection (CHB-MIT / Epileptic Seizure Recognition)");
    eprintln!("================================================================");

    let csv_path = PathBuf::from("data/chb-mit/epileptic_seizure_recognition.csv");
    eprintln!("  Loading {}...", csv_path.display());

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)?;

    let headers = rdr.headers()?.clone();
    eprintln!(
        "  Columns: {} (first={:?}, last={:?})",
        headers.len(),
        headers.get(0),
        headers.get(headers.len() - 1)
    );

    // Find the y column index (last column)
    let y_idx = headers.len() - 1;
    // Feature columns: skip first (row ID), use columns 1..y_idx (X1-X178)
    let feature_start = 1;
    let feature_end = y_idx;
    let n_features = feature_end - feature_start;
    eprintln!(
        "  Features: {} (columns {}-{})",
        n_features,
        feature_start,
        feature_end - 1
    );

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(12000);

    for result in rdr.records() {
        let record = result?;
        let y_str = record.get(y_idx).unwrap_or("0").trim().trim_matches('"');
        let y: i32 = y_str.parse().unwrap_or(0);
        let is_seizure = y == 1;

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for i in feature_start..feature_end {
            let val_str = record.get(i).unwrap_or("0").trim().trim_matches('"');
            match val_str.parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid && features.len() == n_features {
            data.push((features, is_seizure));
        }
    }

    let total = data.len();
    let n_seizure = data.iter().filter(|(_, s)| *s).count();
    let n_normal = total - n_seizure;
    eprintln!(
        "  Loaded {} samples: {} seizure (anomaly), {} normal",
        total, n_seizure, n_normal
    );

    // Split 80/20
    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);

    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, s)| !*s)
        .map(|(f, _)| f.clone())
        .collect();

    eprintln!(
        "  Train: {} total ({} normal for reference)",
        split,
        train_normal.len()
    );
    eprintln!("  Test:  {}", test_data.len());

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0xEE60001);

    // Fit normalization on training data
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);

    // Build reference from normal training samples
    let reference = evaluator.build_reference(&train_normal);

    // Evaluate
    let test_vec: Vec<(Vec<f32>, bool)> = test_data.to_vec();
    let (auc, best_f1) = evaluator.evaluate(&reference, &test_vec);

    let test_seizure = test_data.iter().filter(|(_, s)| *s).count();
    let test_normal = test_data.len() - test_seizure;
    eprintln!(
        "  Test breakdown: {} seizure, {} normal",
        test_seizure, test_normal
    );
    eprintln!("  AUC:     {:.4}", auc);
    eprintln!("  Best F1: {:.4}", best_f1);

    Ok((total, n_features, auc, best_f1))
}

// ── Seismic Event Detection ──────────────────────────────────────────────────

/// Shared numeric columns between noise and events metadata
const SEISMIC_FEATURES: &[&str] = &[
    "trace_E_median_counts",
    "trace_N_median_counts",
    "trace_Z_median_counts",
    "trace_E_mean_counts",
    "trace_N_mean_counts",
    "trace_Z_mean_counts",
    "trace_E_min_counts",
    "trace_N_min_counts",
    "trace_Z_min_counts",
    "trace_E_max_counts",
    "trace_N_max_counts",
    "trace_Z_max_counts",
    "trace_E_rms_counts",
    "trace_N_rms_counts",
    "trace_Z_rms_counts",
    "trace_E_lower_quartile_counts",
    "trace_N_lower_quartile_counts",
    "trace_Z_lower_quartile_counts",
    "trace_E_upper_quartile_counts",
    "trace_N_upper_quartile_counts",
    "trace_Z_upper_quartile_counts",
];

fn load_seismic_csv(
    path: &std::path::Path,
    feature_names: &[&str],
    label: bool,
    max_rows: usize,
) -> anyhow::Result<Vec<(Vec<f32>, bool)>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let headers = rdr.headers()?.clone();

    // Map feature names to column indices
    let header_map: HashMap<&str, usize> =
        headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    let feature_indices: Vec<usize> = feature_names
        .iter()
        .filter_map(|name| header_map.get(name).copied())
        .collect();

    if feature_indices.len() != feature_names.len() {
        let missing: Vec<&&str> = feature_names
            .iter()
            .filter(|name| !header_map.contains_key(**name))
            .collect();
        eprintln!("  WARNING: Missing columns: {:?}", missing);
    }

    eprintln!(
        "  Found {}/{} feature columns in {}",
        feature_indices.len(),
        feature_names.len(),
        path.display()
    );

    let mut data = Vec::new();
    let mut skipped = 0u64;

    for result in rdr.records() {
        if data.len() >= max_rows {
            break;
        }

        let record = result?;
        let mut features = Vec::with_capacity(feature_indices.len());
        let mut valid = true;

        for &idx in &feature_indices {
            let val_str = record.get(idx).unwrap_or("").trim();
            match val_str.parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid && features.len() == feature_indices.len() {
            data.push((features, label));
        } else {
            skipped += 1;
        }
    }

    if skipped > 0 {
        eprintln!("  Skipped {} rows with NaN/missing values", skipped);
    }

    Ok(data)
}

fn evaluate_seismic() -> anyhow::Result<(usize, usize, f64, f64)> {
    eprintln!("\n================================================================");
    eprintln!("  Seismic Event Detection (INSTANCE dataset)");
    eprintln!("================================================================");

    let noise_path = PathBuf::from("data/seismic/instance_noise_metadata.csv");
    let events_path = PathBuf::from("data/seismic/instance_events_metadata_v3.csv");

    let n_features = SEISMIC_FEATURES.len();

    // Load noise (label=0, normal) — sample down to 25K
    eprintln!("  Loading noise (normal) from {}...", noise_path.display());
    let noise_data = load_seismic_csv(&noise_path, SEISMIC_FEATURES, false, 25_000)?;
    eprintln!("  Loaded {} noise samples", noise_data.len());

    // Load events (label=1, anomaly) — sample down to 25K
    eprintln!(
        "  Loading events (anomaly) from {}...",
        events_path.display()
    );
    let events_data = load_seismic_csv(&events_path, SEISMIC_FEATURES, true, 25_000)?;
    eprintln!("  Loaded {} event samples", events_data.len());

    // Merge
    let mut all_data: Vec<(Vec<f32>, bool)> =
        Vec::with_capacity(noise_data.len() + events_data.len());
    all_data.extend(noise_data);
    all_data.extend(events_data);

    let total = all_data.len();
    let n_events = all_data.iter().filter(|(_, e)| *e).count();
    let n_noise = total - n_events;
    eprintln!(
        "  Combined: {} total ({} events, {} noise)",
        total, n_events, n_noise
    );

    // Deterministic shuffle using seed — interleave noise and events
    // Simple: sort by a hash of the features to get a deterministic "shuffle"
    let mut indices: Vec<usize> = (0..total).collect();
    indices.sort_by(|&a, &b| {
        let hash_a = simple_hash(&all_data[a].0);
        let hash_b = simple_hash(&all_data[b].0);
        hash_a.cmp(&hash_b)
    });
    let shuffled: Vec<(Vec<f32>, bool)> = indices.iter().map(|&i| all_data[i].clone()).collect();

    // Split 80/20
    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = shuffled.split_at(split);

    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, e)| !*e) // noise = normal
        .map(|(f, _)| f.clone())
        .collect();

    eprintln!(
        "  Train: {} total ({} normal for reference)",
        split,
        train_normal.len()
    );
    eprintln!("  Test:  {}", test_data.len());

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x5E150001);

    // Fit normalization on training data
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);

    // Build reference from normal training samples
    let reference = evaluator.build_reference(&train_normal);

    // Evaluate
    let test_vec: Vec<(Vec<f32>, bool)> = test_data.to_vec();
    let (auc, best_f1) = evaluator.evaluate(&reference, &test_vec);

    let test_events = test_data.iter().filter(|(_, e)| *e).count();
    let test_noise = test_data.len() - test_events;
    eprintln!(
        "  Test breakdown: {} events, {} noise",
        test_events, test_noise
    );
    eprintln!("  AUC:     {:.4}", auc);
    eprintln!("  Best F1: {:.4}", best_f1);

    Ok((total, n_features, auc, best_f1))
}

fn simple_hash(features: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &v in features {
        let bits = v.to_bits() as u64;
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    eprintln!("================================================================");
    eprintln!("  Universal Zero-Training HDC Anomaly Detection Evaluation");
    eprintln!("  HDC Dimension: {}", HDC_DIMENSION);
    eprintln!("================================================================");

    // 1. EEG Seizure Detection
    let (eeg_samples, eeg_features, eeg_auc, eeg_f1) = evaluate_eeg()?;

    // 2. Seismic Event Detection
    let (seis_samples, seis_features, seis_auc, seis_f1) = evaluate_seismic()?;

    // 3. Summary Table
    println!();
    println!("================================================================");
    println!("     Universal Zero-Training Anomaly Detection Results");
    println!("     HDC Dimension: {}", HDC_DIMENSION);
    println!("================================================================");
    println!(
        "{:<16}| {:<9}| {:<9}| {:<7}| {:<7}",
        "Domain", "Samples", "Features", "AUC", "Best F1"
    );
    println!("----------------|---------|---------|-------|-------");
    println!(
        "{:<16}| {:<9}| {:<9}| {:<7}| {:<7}",
        "Fusion (C-Mod)", "52,551", "6", "0.778", "0.219"
    );
    println!(
        "{:<16}| {:<9}| {:<9}| {:<7.3}| {:<7.3}",
        "EEG Seizure",
        format!("{:>6}", eeg_samples),
        eeg_features,
        eeg_auc,
        eeg_f1
    );
    println!(
        "{:<16}| {:<9}| {:<9}| {:<7.3}| {:<7.3}",
        "Seismic Events",
        format!("{:>6}", seis_samples),
        seis_features,
        seis_auc,
        seis_f1
    );
    println!("================================================================");

    Ok(())
}
