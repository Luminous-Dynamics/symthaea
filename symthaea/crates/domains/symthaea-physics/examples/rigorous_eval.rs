// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rigorous Evaluation — Statistical Significance, Dimensionality Ablation, Baseline Comparison
//!
//! Extends mega_eval.rs with:
//! 1. Multi-seed evaluation (10 random splits) for confidence intervals
//! 2. Dimensionality ablation (256..16384) to measure HDC encoding value
//! 3. Mahalanobis (diagonal) baseline comparison
//!
//! Domains: EEG, Turbofan, Intrusion, Power Grid, Battery
//!
//! Usage:
//!   cargo run -p symthaea-physics --example rigorous_eval --release

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ── Constants ────────────────────────────────────────────────────────────────

const SEEDS: [u64; 10] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51];
const DIMS: [usize; 7] = [256, 512, 1024, 2048, 4096, 8192, 16384];
const EPSILON: f64 = 1e-8;

// ── Domain Data ──────────────────────────────────────────────────────────────

struct DomainData {
    name: &'static str,
    samples: Vec<(Vec<f32>, bool)>, // (features, is_anomaly)
    n_features: usize,
}

// ── Variable-Dimension HDC Evaluator ─────────────────────────────────────────

struct SimpleHdcEvaluatorVar {
    bases: Vec<ContinuousHV>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
    n_features: usize,
    #[allow(dead_code)]
    dim: usize,
}

impl SimpleHdcEvaluatorVar {
    fn new(n_features: usize, dim: usize, seed_base: u64) -> Self {
        let bases: Vec<ContinuousHV> = (0..n_features)
            .map(|i| ContinuousHV::random(dim, seed_base + i as u64))
            .collect();

        Self {
            bases,
            mins: vec![f32::MAX; n_features],
            maxs: vec![f32::MIN; n_features],
            n_features,
            dim,
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
        ((value - self.mins[idx]) / range) * 2.0 - 1.0
    }

    fn encode(&self, features: &[f32]) -> ContinuousHV {
        assert_eq!(features.len(), self.n_features);
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
        let encoded: Vec<ContinuousHV> = normal_data.iter().map(|row| self.encode(row)).collect();
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    fn evaluate(&self, reference: &ContinuousHV, test_data: &[(Vec<f32>, bool)]) -> (f64, f64) {
        let scored: Vec<(f64, bool)> = test_data
            .iter()
            .map(|(features, is_anomaly)| {
                let hv = self.encode(features);
                let sim = hv.similarity(reference);
                let free_energy = 1.0 - sim as f64;
                (free_energy, *is_anomaly)
            })
            .collect();

        let roc = compute_roc_curve(&scored);
        let auc = compute_auc(&roc);
        let best_f1 = find_best_f1(&scored);

        (auc, best_f1)
    }
}

// ── Mahalanobis (Diagonal) Baseline ──────────────────────────────────────────

struct MahalanobisBaseline {
    mean: Vec<f64>,
    var: Vec<f64>,
}

impl MahalanobisBaseline {
    fn fit(normal_data: &[Vec<f32>]) -> Self {
        let n = normal_data.len() as f64;
        let d = normal_data[0].len();

        let mut mean = vec![0.0f64; d];
        for row in normal_data {
            for (i, &v) in row.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }

        let mut var = vec![0.0f64; d];
        for row in normal_data {
            for (i, &v) in row.iter().enumerate() {
                let diff = v as f64 - mean[i];
                var[i] += diff * diff;
            }
        }
        for v in var.iter_mut() {
            *v = *v / n + EPSILON; // add epsilon to avoid div-by-zero
        }

        Self { mean, var }
    }

    fn distance(&self, sample: &[f32]) -> f64 {
        let mut sum = 0.0f64;
        for (i, &v) in sample.iter().enumerate() {
            let diff = v as f64 - self.mean[i];
            sum += diff * diff / self.var[i];
        }
        sum.sqrt()
    }

    fn evaluate(&self, test_data: &[(Vec<f32>, bool)]) -> (f64, f64) {
        let scored: Vec<(f64, bool)> = test_data
            .iter()
            .map(|(features, is_anomaly)| {
                let dist = self.distance(features);
                (dist, *is_anomaly)
            })
            .collect();

        let roc = compute_roc_curve(&scored);
        let auc = compute_auc(&roc);
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

    let mut scores: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = scores.len();
    let step = (n / 200).max(1);
    let mut thresholds: Vec<f64> = Vec::with_capacity(202);
    thresholds.push(scores[n - 1] + 0.01);
    for i in (0..n).rev().step_by(step) {
        thresholds.push(scores[i]);
    }
    if thresholds.last().map_or(true, |&t| t > scores[0]) {
        thresholds.push(scores[0] - 0.01);
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

// ── Shuffling ────────────────────────────────────────────────────────────────

fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
}

fn apply_shuffle(data: &[(Vec<f32>, bool)], indices: &[usize]) -> Vec<(Vec<f32>, bool)> {
    indices.iter().map(|&i| data[i].clone()).collect()
}

// ── Data Loaders ─────────────────────────────────────────────────────────────

fn load_eeg() -> anyhow::Result<Option<DomainData>> {
    let csv_path = "data/chb-mit/epileptic_seizure_recognition.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP EEG: {} not found", csv_path);
        return Ok(None);
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    let headers = rdr.headers()?.clone();
    let y_idx = headers.len() - 1;
    let feature_start = 1;
    let feature_end = y_idx;
    let n_features = feature_end - feature_start;

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(12000);
    for result in rdr.records() {
        let record = result?;
        let y: i32 = record
            .get(y_idx)
            .unwrap_or("0")
            .trim()
            .trim_matches('"')
            .parse()
            .unwrap_or(0);
        let is_seizure = y == 1;

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for i in feature_start..feature_end {
            match record
                .get(i)
                .unwrap_or("0")
                .trim()
                .trim_matches('"')
                .parse::<f32>()
            {
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

    let n_anom = data.iter().filter(|(_, s)| *s).count();
    eprintln!(
        "  EEG: {} samples ({} seizure, {} normal), {} features",
        data.len(),
        n_anom,
        data.len() - n_anom,
        n_features
    );

    Ok(Some(DomainData {
        name: "EEG Seizure",
        samples: data,
        n_features,
    }))
}

fn load_turbofan() -> anyhow::Result<Option<DomainData>> {
    let path = "data/turbofan/train_FD001.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP Turbofan: {} not found", path);
        return Ok(None);
    }

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    struct TurbofanRow {
        unit_id: u32,
        cycle: u32,
        sensors: Vec<f32>,
    }

    let mut rows: Vec<TurbofanRow> = Vec::with_capacity(21000);
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 26 {
            continue;
        }

        let unit_id: u32 = parts[0].parse().unwrap_or(0);
        let cycle: u32 = parts[1].parse().unwrap_or(0);

        let mut sensors = Vec::with_capacity(21);
        let mut valid = true;
        for i in 5..26 {
            match parts[i].parse::<f32>() {
                Ok(v) if v.is_finite() => sensors.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && sensors.len() == 21 {
            rows.push(TurbofanRow {
                unit_id,
                cycle,
                sensors,
            });
        }
    }

    let mut max_cycles: HashMap<u32, u32> = HashMap::new();
    for r in &rows {
        let entry = max_cycles.entry(r.unit_id).or_insert(0);
        if r.cycle > *entry {
            *entry = r.cycle;
        }
    }

    let n_features = 21;
    let data: Vec<(Vec<f32>, bool)> = rows
        .iter()
        .map(|r| {
            let max_c = max_cycles[&r.unit_id];
            let is_degraded = r.cycle > max_c.saturating_sub(30);
            (r.sensors.clone(), is_degraded)
        })
        .collect();

    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Turbofan: {} samples ({} degraded, {} normal), {} features",
        data.len(),
        n_anom,
        data.len() - n_anom,
        n_features
    );

    Ok(Some(DomainData {
        name: "Turbofan RUL",
        samples: data,
        n_features,
    }))
}

fn load_intrusion() -> anyhow::Result<Option<DomainData>> {
    let path = "data/intrusion/KDDTrain+.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP Intrusion: {} not found", path);
        return Ok(None);
    }

    let numeric_cols: Vec<usize> = std::iter::once(0).chain(4..=40).collect();
    let n_features = numeric_cols.len(); // 38

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(126000);
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 42 {
            continue;
        }

        let label = parts[41].trim();
        let is_attack = label != "normal";

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for &col in &numeric_cols {
            match parts[col].trim().parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && features.len() == n_features {
            data.push((features, is_attack));
        }
    }

    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Intrusion: {} samples ({} attacks, {} normal), {} features",
        data.len(),
        n_anom,
        data.len() - n_anom,
        n_features
    );

    Ok(Some(DomainData {
        name: "Network Intrusion",
        samples: data,
        n_features,
    }))
}

fn load_power_grid() -> anyhow::Result<Option<DomainData>> {
    let path = "data/power-grid/household_power_consumption.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP Power Grid: {} not found", path);
        return Ok(None);
    }

    let n_features = 7;
    let voltage_feature_offset = 2;

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let max_rows = 50_000usize;
    let mut feature_rows: Vec<Vec<f32>> = Vec::with_capacity(max_rows);
    let mut line_count = 0u64;

    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        if line_count == 1 {
            continue;
        }

        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() < 9 {
            continue;
        }

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for i in 2..9 {
            match parts[i].trim().parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || features.len() != n_features {
            continue;
        }

        feature_rows.push(features);
        if feature_rows.len() >= max_rows {
            break;
        }
    }

    // Compute voltage stats for anomaly labeling
    let voltage_values: Vec<f32> = feature_rows
        .iter()
        .map(|r| r[voltage_feature_offset])
        .collect();
    let mean_v: f64 =
        voltage_values.iter().map(|&v| v as f64).sum::<f64>() / voltage_values.len() as f64;
    let std_v: f64 = (voltage_values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_v;
            d * d
        })
        .sum::<f64>()
        / voltage_values.len() as f64)
        .sqrt();
    let anomaly_threshold = 3.0;

    let data: Vec<(Vec<f32>, bool)> = feature_rows
        .into_iter()
        .map(|r| {
            let voltage = r[voltage_feature_offset] as f64;
            let is_anomaly = (voltage - mean_v).abs() > std_v * anomaly_threshold;
            (r, is_anomaly)
        })
        .collect();

    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Power Grid: {} samples ({} anomalous, {} normal), {} features (V: mean={:.1}, std={:.1})",
        data.len(),
        n_anom,
        data.len() - n_anom,
        n_features,
        mean_v,
        std_v
    );

    Ok(Some(DomainData {
        name: "Power Grid",
        samples: data,
        n_features,
    }))
}

fn load_battery() -> anyhow::Result<Option<DomainData>> {
    let csv_path = "data/battery/discharge.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP Battery: {} not found", csv_path);
        return Ok(None);
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    let headers = rdr.headers()?.clone();
    let col_map: HashMap<&str, usize> = headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    let feature_names = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
    ];
    let feature_indices: Vec<usize> = feature_names
        .iter()
        .filter_map(|n| col_map.get(n).copied())
        .collect();
    let n_features = feature_indices.len();
    if n_features == 0 {
        eprintln!("  SKIP Battery: no matching feature columns");
        return Ok(None);
    }

    let cycle_idx = col_map.get("id_cycle").copied();
    let battery_idx = col_map.get("Battery").copied();

    struct BatteryRow {
        features: Vec<f32>,
        battery: String,
        cycle: u32,
    }

    let mut rows: Vec<BatteryRow> = Vec::with_capacity(170000);
    for result in rdr.records() {
        let record = result?;
        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for &idx in &feature_indices {
            match record.get(idx).unwrap_or("").trim().parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || features.len() != n_features {
            continue;
        }

        let battery = battery_idx
            .and_then(|i| record.get(i))
            .unwrap_or("")
            .trim()
            .to_string();
        let cycle: u32 = cycle_idx
            .and_then(|i| record.get(i).and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0);

        rows.push(BatteryRow {
            features,
            battery,
            cycle,
        });
    }

    let mut max_cycles: HashMap<String, u32> = HashMap::new();
    for r in &rows {
        let entry = max_cycles.entry(r.battery.clone()).or_insert(0);
        if r.cycle > *entry {
            *entry = r.cycle;
        }
    }

    let mut data: Vec<(Vec<f32>, bool)> = rows
        .iter()
        .map(|r| {
            let max_c = *max_cycles.get(&r.battery).unwrap_or(&1);
            let threshold = (max_c as f64 * 0.75) as u32;
            let is_degraded = r.cycle > threshold;
            (r.features.clone(), is_degraded)
        })
        .collect();

    // Sample down to 50K
    if data.len() > 50_000 {
        let indices = shuffle_indices(data.len(), 999);
        data = indices
            .iter()
            .take(50_000)
            .map(|&i| data[i].clone())
            .collect();
    }

    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Battery: {} samples ({} degraded, {} normal), {} features",
        data.len(),
        n_anom,
        data.len() - n_anom,
        n_features
    );

    Ok(Some(DomainData {
        name: "Battery",
        samples: data,
        n_features,
    }))
}

// ── Evaluation Helpers ───────────────────────────────────────────────────────

/// Run one HDC evaluation on a pre-split dataset
fn run_hdc_eval(
    train_normal: &[Vec<f32>],
    all_train: &[Vec<f32>],
    test_data: &[(Vec<f32>, bool)],
    n_features: usize,
    dim: usize,
    seed_base: u64,
) -> (f64, f64) {
    let mut evaluator = SimpleHdcEvaluatorVar::new(n_features, dim, seed_base);
    evaluator.fit_normalization(all_train);
    let reference = evaluator.build_reference(train_normal);
    evaluator.evaluate(&reference, test_data)
}

/// Split data by seed: shuffle, then 80/20
fn split_by_seed(
    data: &[(Vec<f32>, bool)],
    seed: u64,
) -> (Vec<(Vec<f32>, bool)>, Vec<(Vec<f32>, bool)>) {
    let indices = shuffle_indices(data.len(), seed);
    let shuffled = apply_shuffle(data, &indices);
    let split = (shuffled.len() as f64 * 0.8) as usize;
    let train = shuffled[..split].to_vec();
    let test = shuffled[split..].to_vec();
    (train, test)
}

fn extract_normal(data: &[(Vec<f32>, bool)]) -> Vec<Vec<f32>> {
    data.iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect()
}

fn extract_features(data: &[(Vec<f32>, bool)]) -> Vec<Vec<f32>> {
    data.iter().map(|(f, _)| f.clone()).collect()
}

#[allow(dead_code)]
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        format!("{}", n)
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    eprintln!("================================================================");
    eprintln!("  Rigorous Evaluation — Loading Datasets");
    eprintln!("================================================================");

    // Load all domains
    let mut domains: Vec<DomainData> = Vec::new();
    if let Some(d) = load_eeg()? {
        domains.push(d);
    }
    if let Some(d) = load_turbofan()? {
        domains.push(d);
    }
    if let Some(d) = load_intrusion()? {
        domains.push(d);
    }
    if let Some(d) = load_power_grid()? {
        domains.push(d);
    }
    if let Some(d) = load_battery()? {
        domains.push(d);
    }

    if domains.is_empty() {
        eprintln!("No datasets found. Exiting.");
        return Ok(());
    }

    eprintln!(
        "\nLoaded {} domains. Starting evaluations...\n",
        domains.len()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Analysis 1: Multi-Seed Evaluation
    // ═══════════════════════════════════════════════════════════════════════

    eprintln!("================================================================");
    eprintln!("  ANALYSIS 1: Multi-Seed Evaluation (10 random splits, D=16384)");
    eprintln!("================================================================");

    struct MultiSeedResult {
        name: &'static str,
        aucs: Vec<f64>,
        f1s: Vec<f64>,
    }

    let mut multi_seed_results: Vec<MultiSeedResult> = Vec::new();

    for domain in &domains {
        eprintln!(
            "\n  [{}] {} samples, {} features",
            domain.name,
            domain.samples.len(),
            domain.n_features
        );
        let mut aucs = Vec::with_capacity(SEEDS.len());
        let mut f1s = Vec::with_capacity(SEEDS.len());

        for &seed in SEEDS.iter() {
            let (train, test) = split_by_seed(&domain.samples, seed);
            let train_normal = extract_normal(&train);
            let all_train = extract_features(&train);

            let (auc, f1) = run_hdc_eval(
                &train_normal,
                &all_train,
                &test,
                domain.n_features,
                16384,
                0xEE60001,
            );
            eprintln!("    Seed {}: AUC={:.4}, F1={:.4}", seed, auc, f1);
            aucs.push(auc);
            f1s.push(f1);
        }

        multi_seed_results.push(MultiSeedResult {
            name: domain.name,
            aucs,
            f1s,
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Analysis 2: Dimensionality Ablation
    // ═══════════════════════════════════════════════════════════════════════

    eprintln!("\n================================================================");
    eprintln!("  ANALYSIS 2: Dimensionality Ablation (seed=42)");
    eprintln!("================================================================");

    struct DimAblationResult {
        name: &'static str,
        dim_aucs: Vec<(usize, f64)>,
    }

    let mut dim_results: Vec<DimAblationResult> = Vec::new();

    for domain in &domains {
        let (train, test) = split_by_seed(&domain.samples, 42);
        let train_normal = extract_normal(&train);
        let all_train = extract_features(&train);

        eprintln!("\n  [{}]", domain.name);
        let mut dim_aucs = Vec::with_capacity(DIMS.len());

        for &dim in &DIMS {
            let (auc, _f1) = run_hdc_eval(
                &train_normal,
                &all_train,
                &test,
                domain.n_features,
                dim,
                0xEE60001,
            );
            eprintln!("    D={:>5}: AUC={:.4}", dim, auc);
            dim_aucs.push((dim, auc));
        }

        dim_results.push(DimAblationResult {
            name: domain.name,
            dim_aucs,
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Analysis 3: HDC vs Mahalanobis Baseline
    // ═══════════════════════════════════════════════════════════════════════

    eprintln!("\n================================================================");
    eprintln!("  ANALYSIS 3: HDC vs Mahalanobis Baseline (seed=42, D=16384)");
    eprintln!("================================================================");

    struct BaselineResult {
        name: &'static str,
        hdc_auc: f64,
        mahal_auc: f64,
    }

    let mut baseline_results: Vec<BaselineResult> = Vec::new();

    for domain in &domains {
        let (train, test) = split_by_seed(&domain.samples, 42);
        let train_normal = extract_normal(&train);
        let all_train = extract_features(&train);

        // HDC
        let (hdc_auc, _) = run_hdc_eval(
            &train_normal,
            &all_train,
            &test,
            domain.n_features,
            16384,
            0xEE60001,
        );

        // Mahalanobis
        let mahal = MahalanobisBaseline::fit(&train_normal);
        let (mahal_auc, _) = mahal.evaluate(&test);

        eprintln!(
            "  [{}] HDC AUC={:.4}, Mahalanobis AUC={:.4}",
            domain.name, hdc_auc, mahal_auc
        );

        baseline_results.push(BaselineResult {
            name: domain.name,
            hdc_auc,
            mahal_auc,
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Print Final Report
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("================================================================");
    println!("     RIGOROUS EVALUATION -- Statistical Significance");
    println!("================================================================");

    // Table 1: Multi-Seed
    println!();
    println!("=== Multi-Seed (10 splits, D=16384) ===");
    println!(
        "{:<22}| {:<17}| {:<17}| {:<9}| {}",
        "Domain", "Mean AUC +/- Std", "Mean F1 +/- Std", "Min AUC", "Max AUC"
    );
    println!("----------------------|-----------------|-----------------|---------|--------");

    for r in &multi_seed_results {
        let mean_auc = r.aucs.iter().sum::<f64>() / r.aucs.len() as f64;
        let std_auc = (r.aucs.iter().map(|&a| (a - mean_auc).powi(2)).sum::<f64>()
            / r.aucs.len() as f64)
            .sqrt();
        let mean_f1 = r.f1s.iter().sum::<f64>() / r.f1s.len() as f64;
        let std_f1 =
            (r.f1s.iter().map(|&f| (f - mean_f1).powi(2)).sum::<f64>() / r.f1s.len() as f64).sqrt();
        let min_auc = r.aucs.iter().cloned().fold(f64::MAX, f64::min);
        let max_auc = r.aucs.iter().cloned().fold(f64::MIN, f64::max);

        println!(
            "{:<22}| {:.3} +/- {:.3}     | {:.3} +/- {:.3}     | {:.3}   | {:.3}",
            r.name, mean_auc, std_auc, mean_f1, std_f1, min_auc, max_auc
        );
    }

    // Table 2: Dimensionality Ablation
    println!();
    println!("=== Dimensionality Ablation (seed=42) ===");
    println!(
        "{:<22}| {:<7}| {:<7}| {:<7}| {:<7}| {:<7}| {:<7}| {}",
        "Domain", "D=256", "D=512", "D=1024", "D=2048", "D=4096", "D=8192", "D=16384"
    );
    println!("----------------------|-------|-------|-------|-------|-------|-------|-------");

    for r in &dim_results {
        let vals: Vec<String> = r
            .dim_aucs
            .iter()
            .map(|(_, auc)| format!("{:.3}", auc))
            .collect();
        println!(
            "{:<22}| {:<7}| {:<7}| {:<7}| {:<7}| {:<7}| {:<7}| {}",
            r.name,
            vals.get(0).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(1).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(2).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(3).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(4).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(5).map(|s| s.as_str()).unwrap_or("---"),
            vals.get(6).map(|s| s.as_str()).unwrap_or("---"),
        );
    }

    // Table 3: Baseline Comparison
    println!();
    println!("=== HDC vs Mahalanobis Baseline (seed=42, D=16384) ===");
    println!(
        "{:<22}| {:<9}| {:<11}| {}",
        "Domain", "HDC AUC", "Mahal AUC", "HDC Wins?"
    );
    println!("----------------------|---------|-----------|----------");

    for r in &baseline_results {
        let wins = if r.hdc_auc > r.mahal_auc { "YES" } else { "no" };
        println!(
            "{:<22}| {:<9.3}| {:<11.3}| {}",
            r.name, r.hdc_auc, r.mahal_auc, wins
        );
    }

    println!("================================================================");

    // Summary
    println!();
    let n = multi_seed_results.len();
    let grand_mean_auc: f64 = multi_seed_results
        .iter()
        .map(|r| r.aucs.iter().sum::<f64>() / r.aucs.len() as f64)
        .sum::<f64>()
        / n as f64;
    let hdc_wins = baseline_results
        .iter()
        .filter(|r| r.hdc_auc > r.mahal_auc)
        .count();
    println!("Grand mean AUC (10 seeds, D=16384): {:.3}", grand_mean_auc);
    println!(
        "HDC wins vs Mahalanobis baseline: {}/{}",
        hdc_wins,
        baseline_results.len()
    );

    Ok(())
}
