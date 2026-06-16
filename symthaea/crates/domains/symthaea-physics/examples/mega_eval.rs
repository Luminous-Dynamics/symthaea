// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal Zero-Training Anomaly Detection — Mega Evaluation
//!
//! Evaluates the same HDC 16,384D architecture across ALL available datasets:
//! A. EEG Seizure (CHB-MIT)
//! B. LIGO Glitches (Gravity Spy)
//! C. Turbofan RUL (NASA C-MAPSS FD001)
//! D. Network Intrusion (KDD Cup 1999)
//! E. Battery Degradation (NASA)
//! F. Power Grid (UCI household power)
//! G. Solar Flare (GOES X-ray flux)
//!
//! Usage:
//!   cargo run -p symthaea-physics --example mega_eval --release

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
        eprintln!(
            "  Building reference HV from {} normal samples...",
            normal_data.len()
        );
        let encoded: Vec<ContinuousHV> = normal_data.iter().map(|row| self.encode(row)).collect();
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    fn evaluate(&self, reference: &ContinuousHV, test_data: &[(Vec<f32>, bool)]) -> (f64, f64) {
        eprintln!(
            "  Computing free energy for {} test samples...",
            test_data.len()
        );

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

fn simple_hash(features: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in features {
        let bits = v.to_bits() as u64;
        h ^= bits;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn deterministic_shuffle(data: &mut [(Vec<f32>, bool)]) {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let ha = simple_hash(&data[a].0);
        let hb = simple_hash(&data[b].0);
        ha.cmp(&hb)
    });
    // Apply permutation via clone (safe for our dataset sizes)
    let cloned: Vec<(Vec<f32>, bool)> = indices.iter().map(|&i| data[i].clone()).collect();
    for (i, item) in cloned.into_iter().enumerate() {
        data[i] = item;
    }
}

/// Result for one domain evaluation
struct DomainResult {
    name: &'static str,
    samples: usize,
    features: usize,
    auc: f64,
    f1: f64,
    status: &'static str,
}

// ── A. EEG Seizure Detection ─────────────────────────────────────────────────

fn evaluate_eeg() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [A] EEG Seizure Detection (CHB-MIT)");
    eprintln!("================================================================");

    let csv_path = "data/chb-mit/epileptic_seizure_recognition.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP: {} not found", csv_path);
        return Ok(DomainResult {
            name: "EEG Seizure",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
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

    let total = data.len();
    let n_anom = data.iter().filter(|(_, s)| *s).count();
    eprintln!(
        "  Loaded {} samples: {} seizure, {} normal",
        total,
        n_anom,
        total - n_anom
    );

    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, s)| !*s)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0xEE60001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "EEG Seizure",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── B. LIGO Glitches ─────────────────────────────────────────────────────────

fn evaluate_ligo() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [B] LIGO Glitch Detection (Gravity Spy)");
    eprintln!("================================================================");

    let csv_path = "data/ligo/trainingset_v1d1_metadata.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP: {} not found", csv_path);
        return Ok(DomainResult {
            name: "LIGO Glitches",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    let headers = rdr.headers()?.clone();

    // Build column index map
    let col_map: HashMap<&str, usize> = headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    // Use 8 numeric features for richer encoding
    let feature_names = [
        "peak_frequency",
        "central_freq",
        "snr",
        "bandwidth",
        "amplitude",
        "duration",
        "confidence",
        "chisq",
    ];
    let feature_indices: Vec<usize> = feature_names
        .iter()
        .filter_map(|name| col_map.get(name).copied())
        .collect();

    let label_idx = col_map.get("label").copied();
    let n_features = feature_indices.len();
    eprintln!(
        "  Using {} features: {:?}",
        n_features,
        &feature_names[..n_features.min(feature_names.len())]
    );

    // Common glitch types (>300 samples each) = "normal" baseline
    // Rare types (<100 samples) + "None_of_the_Above" = anomaly
    let common_types = [
        "Blip",
        "Koi_Fish",
        "Low_Frequency_Burst",
        "Light_Modulation",
        "Power_Line",
        "Low_Frequency_Lines",
        "Extremely_Loud",
        "Scattered_Light",
        "Violin_Mode",
        "Scratchy",
        "1080Lines",
    ];

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(8000);
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

        // Label: rare glitch morphologies = anomaly, common types = normal
        let label = label_idx.and_then(|i| record.get(i)).unwrap_or("").trim();
        let is_common = common_types.iter().any(|&t| t == label);
        let is_anomaly = !is_common;
        data.push((features, is_anomaly));
    }

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Loaded {} samples: {} high-SNR (anomaly), {} normal",
        total,
        n_anom,
        total - n_anom
    );

    deterministic_shuffle(&mut data);

    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x11600001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "LIGO Glitches",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── C. Turbofan RUL ──────────────────────────────────────────────────────────

fn evaluate_turbofan() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [C] Turbofan Degradation (NASA C-MAPSS FD001)");
    eprintln!("================================================================");

    let path = "data/turbofan/train_FD001.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP: {} not found", path);
        return Ok(DomainResult {
            name: "Turbofan RUL",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    // Parse space-separated file: unit_id(0), cycle(1), op1-3(2-4), sensor1-21(5-25)
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    struct TurbofanRow {
        unit_id: u32,
        cycle: u32,
        sensors: Vec<f32>, // 21 sensor values
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

    // Find max cycle per unit
    let mut max_cycles: HashMap<u32, u32> = HashMap::new();
    for r in &rows {
        let entry = max_cycles.entry(r.unit_id).or_insert(0);
        if r.cycle > *entry {
            *entry = r.cycle;
        }
    }

    let n_features = 21;
    // Label: last 30 cycles = degraded (anomaly)
    let mut data: Vec<(Vec<f32>, bool)> = rows
        .iter()
        .map(|r| {
            let max_c = max_cycles[&r.unit_id];
            let is_degraded = r.cycle > max_c.saturating_sub(30);
            (r.sensors.clone(), is_degraded)
        })
        .collect();

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Loaded {} samples: {} degraded, {} normal",
        total,
        n_anom,
        total - n_anom
    );

    // Split by unit: first 70 units train, rest test
    let train_units: std::collections::HashSet<u32> = (1..=70).collect();
    let mut train_data: Vec<(Vec<f32>, bool)> = Vec::new();
    let mut test_data: Vec<(Vec<f32>, bool)> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        if train_units.contains(&r.unit_id) {
            train_data.push(data[i].clone());
        } else {
            test_data.push(data[i].clone());
        }
    }

    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        train_data.len(),
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x7F0B0001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data);

    // Use full dataset size for reporting
    let _ = data;
    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Turbofan RUL",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── D. Network Intrusion ─────────────────────────────────────────────────────

fn evaluate_intrusion() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [D] Network Intrusion Detection (KDD Cup)");
    eprintln!("================================================================");

    let path = "data/intrusion/KDDTrain+.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP: {} not found", path);
        return Ok(DomainResult {
            name: "Network Intrusion",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    // 43 columns. Col 0=duration, 1-3=categorical (skip), 4-40=numeric, 41=label, 42=difficulty
    // Numeric feature indices: 0, 4..=40 → 38 features
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

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Loaded {} samples: {} attacks, {} normal",
        total,
        n_anom,
        total - n_anom
    );

    // Split: first 80% train, last 20% test
    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x1D500001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Network Intrusion",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── E. Battery Degradation ───────────────────────────────────────────────────

fn evaluate_battery() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [E] Battery Degradation (NASA)");
    eprintln!("================================================================");

    let csv_path = "data/battery/discharge.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP: {} not found", csv_path);
        return Ok(DomainResult {
            name: "Battery Degradation",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    let headers = rdr.headers()?.clone();
    let col_map: HashMap<&str, usize> = headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    eprintln!("  Columns: {:?}", headers.iter().collect::<Vec<_>>());

    // Features: Voltage_measured, Current_measured, Temperature_measured
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
        eprintln!("  SKIP: no matching feature columns");
        return Ok(DomainResult {
            name: "Battery Degradation",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "no features",
        });
    }

    // We also need id_cycle and Capacity to derive labels
    let cycle_idx = col_map.get("id_cycle").copied();
    let capacity_idx = col_map.get("Capacity").copied();
    let battery_idx = col_map.get("Battery").copied();

    eprintln!(
        "  Using {} features, cycle_idx={:?}, capacity_idx={:?}",
        n_features, cycle_idx, capacity_idx
    );

    struct BatteryRow {
        features: Vec<f32>,
        battery: String,
        cycle: u32,
        capacity: f32,
    }

    let mut rows: Vec<BatteryRow> = Vec::with_capacity(170000);
    let mut skipped = 0u64;
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
            skipped += 1;
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
        let capacity: f32 = capacity_idx
            .and_then(|i| record.get(i).and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0.0);

        rows.push(BatteryRow {
            features,
            battery,
            cycle,
            capacity,
        });
    }
    if skipped > 0 {
        eprintln!("  Skipped {} invalid rows", skipped);
    }

    // Find max cycle per battery to define "degraded"
    let mut max_cycles: HashMap<String, u32> = HashMap::new();
    for r in &rows {
        let entry = max_cycles.entry(r.battery.clone()).or_insert(0);
        if r.cycle > *entry {
            *entry = r.cycle;
        }
    }

    // Label: last 25% of cycles for each battery = degraded
    let mut data: Vec<(Vec<f32>, bool)> = rows
        .iter()
        .map(|r| {
            let max_c = *max_cycles.get(&r.battery).unwrap_or(&1);
            let threshold = (max_c as f64 * 0.75) as u32;
            let is_degraded = r.cycle > threshold;
            (r.features.clone(), is_degraded)
        })
        .collect();

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Loaded {} samples: {} degraded, {} normal",
        total,
        n_anom,
        total - n_anom
    );

    // Sample down to 50K for speed
    if data.len() > 50_000 {
        deterministic_shuffle(&mut data);
        data.truncate(50_000);
        eprintln!("  Sampled down to {}", data.len());
    }

    let split = (data.len() as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0xBA770001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Battery Degradation",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── F. Power Grid ────────────────────────────────────────────────────────────

fn evaluate_power_grid() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [F] Power Grid Anomaly Detection (UCI Household Power)");
    eprintln!("================================================================");

    let path = "data/power-grid/household_power_consumption.txt";
    if !Path::new(path).exists() {
        eprintln!("  SKIP: {} not found", path);
        return Ok(DomainResult {
            name: "Power Grid",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    // Semicolon-separated. Header: Date;Time;Global_active_power;Global_reactive_power;Voltage;
    //   Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    // Features: cols 2-8 (7 numeric features)
    // Label: voltage anomaly — |Voltage - 240| > 15 → anomaly (typical EU voltage ~230-240V)
    let n_features = 7;
    let voltage_feature_offset = 2; // col 4 = Voltage, but within our 7-feature vector it's index 2

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(60000);
    let mut line_count = 0u64;
    let mut skipped = 0u64;

    // Compute mean/std of voltage in first pass for labeling
    // Actually, let's do single pass: collect, then label based on stats
    struct PowerRow {
        features: Vec<f32>,
    }

    let mut rows: Vec<PowerRow> = Vec::with_capacity(60000);
    let max_rows = 50_000usize;

    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        if line_count == 1 {
            continue;
        } // skip header

        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() < 9 {
            skipped += 1;
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
            skipped += 1;
            continue;
        }

        rows.push(PowerRow { features });
        if rows.len() >= max_rows {
            break;
        }
    }

    eprintln!(
        "  Read {} valid rows (skipped {} with missing/? values)",
        rows.len(),
        skipped
    );

    // Compute voltage statistics for anomaly labeling
    // Voltage is feature index 2 (Global_active_power=0, Global_reactive_power=1, Voltage=2, ...)
    let voltage_values: Vec<f32> = rows
        .iter()
        .map(|r| r.features[voltage_feature_offset])
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
    let anomaly_threshold = 3.0; // 3 sigma
    eprintln!(
        "  Voltage: mean={:.2}, std={:.2}, anomaly threshold: |V - {:.2}| > {:.2}",
        mean_v,
        std_v,
        mean_v,
        std_v * anomaly_threshold
    );

    data = rows
        .into_iter()
        .map(|r| {
            let voltage = r.features[voltage_feature_offset] as f64;
            let is_anomaly = (voltage - mean_v).abs() > std_v * anomaly_threshold;
            (r.features, is_anomaly)
        })
        .collect();

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  {} samples: {} anomalous voltage, {} normal ({:.1}% anomaly rate)",
        total,
        n_anom,
        total - n_anom,
        n_anom as f64 / total as f64 * 100.0
    );

    deterministic_shuffle(&mut data);

    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0xAC0D0001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Power Grid",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── G. Solar Flare ───────────────────────────────────────────────────────────

fn evaluate_solar() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [G] Solar X-Ray Flux Anomaly Detection (GOES)");
    eprintln!("================================================================");

    let csv_path = "data/solar/goes_xray_flux_7day.csv";
    if !Path::new(csv_path).exists() {
        eprintln!("  SKIP: {} not found", csv_path);
        return Ok(DomainResult {
            name: "Solar Flare",
            samples: 0,
            features: 0,
            auc: 0.0,
            f1: 0.0,
            status: "file missing",
        });
    }

    // Two rows per timestamp (0.05-0.4nm and 0.1-0.8nm bands)
    // Pivot: group by time_tag, create features from both bands
    // Features per timestamp: flux_short, observed_flux_short, electron_correction_short,
    //                         flux_long, observed_flux_long, electron_correction_long (6 features)
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    let headers = rdr.headers()?.clone();
    let col_map: HashMap<&str, usize> = headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    eprintln!("  Columns: {:?}", headers.iter().collect::<Vec<_>>());

    let time_idx = col_map.get("time_tag").copied().unwrap_or(0);
    let flux_idx = col_map.get("flux").copied().unwrap_or(2);
    let obs_flux_idx = col_map.get("observed_flux").copied().unwrap_or(3);
    let elec_corr_idx = col_map.get("electron_correction").copied().unwrap_or(4);
    let energy_idx = col_map.get("energy").copied().unwrap_or(6);

    // Group rows by timestamp
    struct SolarRow {
        time: String,
        flux: f32,
        observed_flux: f32,
        electron_correction: f32,
        is_short: bool,
    }

    let mut all_rows: Vec<SolarRow> = Vec::with_capacity(21000);
    for result in rdr.records() {
        let record = result?;
        let time = record.get(time_idx).unwrap_or("").trim().to_string();
        let energy = record.get(energy_idx).unwrap_or("").trim().to_string();
        let is_short = energy.contains("0.05") || energy.contains("0.4nm");

        let flux: f32 = record
            .get(flux_idx)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let obs_flux: f32 = record
            .get(obs_flux_idx)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let elec_corr: f32 = record
            .get(elec_corr_idx)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        if flux.is_finite() && obs_flux.is_finite() && elec_corr.is_finite() {
            all_rows.push(SolarRow {
                time,
                flux,
                observed_flux: obs_flux,
                electron_correction: elec_corr,
                is_short,
            });
        }
    }

    eprintln!("  Read {} raw rows", all_rows.len());

    // Pivot: pair short and long band rows by timestamp
    let mut by_time: HashMap<String, (Option<[f32; 3]>, Option<[f32; 3]>)> = HashMap::new();
    for r in &all_rows {
        let entry = by_time.entry(r.time.clone()).or_insert((None, None));
        let vals = [r.flux, r.observed_flux, r.electron_correction];
        if r.is_short {
            entry.0 = Some(vals);
        } else {
            entry.1 = Some(vals);
        }
    }

    let n_features = 6;
    let mut feature_rows: Vec<Vec<f32>> = Vec::with_capacity(by_time.len());
    for (_, (short, long)) in &by_time {
        if let (Some(s), Some(l)) = (short, long) {
            // Use log scale for flux values (they span many orders of magnitude)
            let features: Vec<f32> = vec![
                (s[0].abs() + 1e-15).ln(),
                (s[1].abs() + 1e-15).ln(),
                (s[2].abs() + 1e-15).ln(),
                (l[0].abs() + 1e-15).ln(),
                (l[1].abs() + 1e-15).ln(),
                (l[2].abs() + 1e-15).ln(),
            ];
            if features.iter().all(|v| v.is_finite()) {
                feature_rows.push(features);
            }
        }
    }

    // Sort by time for temporal ordering (important for label derivation)
    // Since we lost ordering in HashMap, sort by feature hash for determinism
    feature_rows.sort_by(|a, b| simple_hash(a).cmp(&simple_hash(b)));

    eprintln!(
        "  Pivoted to {} timestamp rows, {} features",
        feature_rows.len(),
        n_features
    );

    // Label: flux spike > mean + 3*std on short-band flux (feature 0)
    let mean_flux: f64 =
        feature_rows.iter().map(|r| r[0] as f64).sum::<f64>() / feature_rows.len() as f64;
    let std_flux: f64 = (feature_rows
        .iter()
        .map(|r| {
            let d = r[0] as f64 - mean_flux;
            d * d
        })
        .sum::<f64>()
        / feature_rows.len() as f64)
        .sqrt();
    // Use 2-sigma for solar data — 3-sigma yields zero anomalies in quiet-sun periods
    let thresh = mean_flux + 2.0 * std_flux;
    eprintln!(
        "  Log-flux (short): mean={:.3}, std={:.3}, anomaly > {:.3} (2-sigma)",
        mean_flux, std_flux, thresh
    );

    let mut data: Vec<(Vec<f32>, bool)> = feature_rows
        .into_iter()
        .map(|f| {
            let is_anomaly = (f[0] as f64) > thresh;
            (f, is_anomaly)
        })
        .collect();

    let total = data.len();
    let n_anom = data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  {} samples: {} flare anomalies, {} normal ({:.1}%)",
        total,
        n_anom,
        total - n_anom,
        n_anom as f64 / total as f64 * 100.0
    );

    if n_anom < 5 || total < 100 {
        eprintln!("  SKIP: insufficient anomalies or samples for meaningful evaluation");
        return Ok(DomainResult {
            name: "Solar Flare",
            samples: total,
            features: n_features,
            auc: 0.0,
            f1: 0.0,
            status: "insufficient anomalies",
        });
    }

    let split = (total as f64 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split);
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!(
        "  Train: {} ({} normal) | Test: {}",
        split,
        train_normal.len(),
        test_data.len()
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x50140001);
    let all_train: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train);
    let reference = evaluator.build_reference(&train_normal);
    let (auc, f1) = evaluator.evaluate(&reference, &test_data.to_vec());

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Solar Flare",
        samples: total,
        features: n_features,
        auc,
        f1,
        status: "evaluated",
    })
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    eprintln!("================================================================");
    eprintln!("  Universal Zero-Training HDC Anomaly Detection — Mega Eval");
    eprintln!("  HDC Dimension: {}", HDC_DIMENSION);
    eprintln!("================================================================");

    let mut results: Vec<DomainResult> = Vec::new();

    // A. EEG Seizure
    results.push(evaluate_eeg()?);

    // B. LIGO Glitches
    results.push(evaluate_ligo()?);

    // C. Turbofan RUL
    results.push(evaluate_turbofan()?);

    // D. Network Intrusion
    results.push(evaluate_intrusion()?);

    // E. Battery Degradation
    results.push(evaluate_battery()?);

    // F. Power Grid
    results.push(evaluate_power_grid()?);

    // G. Solar Flare
    results.push(evaluate_solar()?);

    // ── Final Summary Table ──────────────────────────────────────────────────
    println!();
    println!("================================================================");
    println!("     Universal Zero-Training Anomaly Detection — All Domains");
    println!(
        "     Architecture: HDC {}D | Encoding: ContinuousHV | Training: ZERO",
        HDC_DIMENSION
    );
    println!("================================================================");
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Domain", "Samples", "Features", "AUC", "Best F1", "Status"
    );
    println!("----------------------|---------|---------|-------|--------|------------------");

    // Prior results (hardcoded from previous evaluations)
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Fusion (C-Mod)", "52,551", "6", "0.778", "0.219", "V1 baseline"
    );

    for r in &results {
        if r.samples > 0 {
            println!(
                "{:<22}| {:<9}| {:<9}| {:<7.3}| {:<8.3}| {}",
                r.name,
                format_number(r.samples),
                r.features,
                r.auc,
                r.f1,
                r.status
            );
        } else {
            println!(
                "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
                r.name, "---", "---", "---", "---", r.status
            );
        }
    }

    // Datasets needing preprocessing (not evaluated)
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Spacecraft (SMAP)", "---", "---", "---", "---", "needs .npy loader"
    );
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Spacecraft (MSL)", "---", "---", "---", "---", "needs .npy loader"
    );
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Sepsis", "---", "---", "---", "---", "needs .psv merger"
    );
    println!(
        "{:<22}| {:<9}| {:<9}| {:<7}| {:<8}| {}",
        "Seismic (INSTANCE)", "50,000", "21", "0.442", "0.663", "prior result"
    );

    println!("================================================================");

    // Summary statistics
    let evaluated: Vec<&DomainResult> =
        results.iter().filter(|r| r.status == "evaluated").collect();
    if !evaluated.is_empty() {
        let avg_auc: f64 = evaluated.iter().map(|r| r.auc).sum::<f64>() / evaluated.len() as f64;
        let avg_f1: f64 = evaluated.iter().map(|r| r.f1).sum::<f64>() / evaluated.len() as f64;
        println!(
            "\nEvaluated {} domains. Mean AUC: {:.3}, Mean F1: {:.3}",
            evaluated.len(),
            avg_auc,
            avg_f1
        );
        println!("(Including prior: Fusion AUC 0.778, Seismic AUC 0.442)");
    }

    Ok(())
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        format!("{}", n)
    }
}
