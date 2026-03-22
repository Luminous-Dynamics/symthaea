// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spacecraft (SMAP + MSL) and Sepsis — Zero-Training HDC Anomaly Detection
//!
//! Evaluates HDC 16,384D anomaly detection on:
//!   A. SMAP satellite telemetry (25 features, 135K train / 427K test)
//!   B. MSL Mars rover telemetry (55 features, 58K train / 73K test)
//!   C. Sepsis ICU vitals (7 features, ~78K rows, patient-grouped split)
//!
//! Usage:
//!   cargo run -p symthaea-physics --example spacecraft_sepsis_eval --release

use std::path::Path;

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Simple HDC Evaluator (same pattern as mega_eval.rs) ──────────────────────

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
    let step = (n / 500).max(1); // finer granularity for large datasets
    let mut thresholds: Vec<f64> = Vec::with_capacity(502);
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
    let step = (n / 500).max(1);
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

fn deterministic_subsample(data: &[Vec<f32>], max_n: usize) -> Vec<Vec<f32>> {
    if data.len() <= max_n {
        return data.to_vec();
    }
    // Deterministic sampling: pick evenly-spaced indices + hash-based tiebreak
    let step = data.len() as f64 / max_n as f64;
    (0..max_n)
        .map(|i| {
            let idx = (i as f64 * step) as usize;
            data[idx.min(data.len() - 1)].clone()
        })
        .collect()
}

// ── Domain result ────────────────────────────────────────────────────────────

struct DomainResult {
    name: &'static str,
    train: usize,
    test: usize,
    features: usize,
    auc: f64,
    f1: f64,
}

// ── A. SMAP Satellite ────────────────────────────────────────────────────────

fn load_spacecraft_csv(path: &str, n_features: usize) -> anyhow::Result<Vec<(Vec<f32>, bool)>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();
    let label_idx = headers
        .iter()
        .position(|h| h == "label")
        .unwrap_or(headers.len() - 1);

    let mut data: Vec<(Vec<f32>, bool)> = Vec::with_capacity(150_000);
    for result in rdr.records() {
        let record = result?;
        let label: i32 = record
            .get(label_idx)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0);
        let is_anomaly = label == 1;

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for i in 0..n_features {
            match record.get(i).unwrap_or("0").trim().parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && features.len() == n_features {
            data.push((features, is_anomaly));
        }
    }
    Ok(data)
}

fn evaluate_smap() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [A] SMAP Satellite Telemetry (25 features)");
    eprintln!("================================================================");

    let train_path = "data/spacecraft/smap_train.csv";
    let test_path = "data/spacecraft/smap_test.csv";
    if !Path::new(train_path).exists() || !Path::new(test_path).exists() {
        anyhow::bail!("SMAP data files not found");
    }

    let n_features = 25;

    eprintln!("  Loading training data...");
    let train_data = load_spacecraft_csv(train_path, n_features)?;
    let n_train = train_data.len();
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!("  Train: {} rows ({} normal)", n_train, train_normal.len());

    eprintln!("  Loading test data...");
    let test_data = load_spacecraft_csv(test_path, n_features)?;
    let n_test = test_data.len();
    let n_anom = test_data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Test: {} rows ({} anomalous = {:.1}%)",
        n_test,
        n_anom,
        n_anom as f64 / n_test as f64 * 100.0
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x5AAA0001);

    // Fit normalization on training data
    let all_train_feats: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train_feats);

    // Subsample for reference building (encoding 135K rows is very slow)
    let reference_sample = deterministic_subsample(&train_normal, 10_000);
    let reference = evaluator.build_reference(&reference_sample);

    let (auc, f1) = evaluator.evaluate(&reference, &test_data);

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "SMAP (satellite)",
        train: n_train,
        test: n_test,
        features: n_features,
        auc,
        f1,
    })
}

// ── B. MSL Mars Rover ────────────────────────────────────────────────────────

fn evaluate_msl() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [B] MSL Mars Rover Telemetry (55 features)");
    eprintln!("================================================================");

    let train_path = "data/spacecraft/msl_train.csv";
    let test_path = "data/spacecraft/msl_test.csv";
    if !Path::new(train_path).exists() || !Path::new(test_path).exists() {
        anyhow::bail!("MSL data files not found");
    }

    let n_features = 55;

    eprintln!("  Loading training data...");
    let train_data = load_spacecraft_csv(train_path, n_features)?;
    let n_train = train_data.len();
    let train_normal: Vec<Vec<f32>> = train_data
        .iter()
        .filter(|(_, a)| !*a)
        .map(|(f, _)| f.clone())
        .collect();
    eprintln!("  Train: {} rows ({} normal)", n_train, train_normal.len());

    eprintln!("  Loading test data...");
    let test_data = load_spacecraft_csv(test_path, n_features)?;
    let n_test = test_data.len();
    let n_anom = test_data.iter().filter(|(_, a)| *a).count();
    eprintln!(
        "  Test: {} rows ({} anomalous = {:.1}%)",
        n_test,
        n_anom,
        n_anom as f64 / n_test as f64 * 100.0
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x4510001);

    let all_train_feats: Vec<Vec<f32>> = train_data.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train_feats);

    let reference_sample = deterministic_subsample(&train_normal, 10_000);
    let reference = evaluator.build_reference(&reference_sample);

    let (auc, f1) = evaluator.evaluate(&reference, &test_data);

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "MSL (Mars rover)",
        train: n_train,
        test: n_test,
        features: n_features,
        auc,
        f1,
    })
}

// ── C. Sepsis ICU ────────────────────────────────────────────────────────────

fn evaluate_sepsis() -> anyhow::Result<DomainResult> {
    eprintln!("\n================================================================");
    eprintln!("  [C] Sepsis ICU Detection (7 vital sign features)");
    eprintln!("================================================================");

    let csv_path = "data/sepsis/sepsis_combined.csv";
    if !Path::new(csv_path).exists() {
        anyhow::bail!("Sepsis data file not found");
    }

    // Columns: patient_id,hour,HR,O2Sat,Temp,SBP,MAP,DBP,Resp,SepsisLabel
    // Features: HR(2), O2Sat(3), Temp(4), SBP(5), MAP(6), DBP(7), Resp(8)
    // Label: SepsisLabel(9)
    let feature_cols: Vec<usize> = vec![2, 3, 4, 5, 6, 7, 8];
    let n_features = feature_cols.len(); // 7

    eprintln!("  Loading sepsis data...");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;

    struct SepsisRow {
        patient_id: String,
        features: Vec<f32>,
        label: bool,
    }

    let mut rows: Vec<SepsisRow> = Vec::with_capacity(80_000);
    for result in rdr.records() {
        let record = result?;
        let patient_id = record.get(0).unwrap_or("").trim().to_string();
        let label: i32 = record.get(9).unwrap_or("0").trim().parse().unwrap_or(0);

        let mut features = Vec::with_capacity(n_features);
        let mut valid = true;
        for &col in &feature_cols {
            match record.get(col).unwrap_or("0").trim().parse::<f32>() {
                Ok(v) if v.is_finite() => features.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && features.len() == n_features {
            rows.push(SepsisRow {
                patient_id,
                features,
                label: label == 1,
            });
        }
    }

    let total = rows.len();
    let n_sepsis = rows.iter().filter(|r| r.label).count();
    eprintln!(
        "  Loaded {} rows: {} sepsis-positive, {} normal ({:.1}%)",
        total,
        n_sepsis,
        total - n_sepsis,
        n_sepsis as f64 / total as f64 * 100.0
    );

    // Group by patient_id, preserve order (consecutive rows share patient_id)
    let mut patient_ids_ordered: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for r in &rows {
        if seen.insert(r.patient_id.clone()) {
            patient_ids_ordered.push(r.patient_id.clone());
        }
    }

    let n_patients = patient_ids_ordered.len();
    let train_patient_count = (n_patients as f64 * 0.8) as usize;
    let train_patients: std::collections::HashSet<&str> = patient_ids_ordered
        [..train_patient_count]
        .iter()
        .map(|s| s.as_str())
        .collect();

    eprintln!(
        "  {} unique patients, {} train / {} test",
        n_patients,
        train_patient_count,
        n_patients - train_patient_count
    );

    let mut train_rows: Vec<(Vec<f32>, bool)> = Vec::new();
    let mut test_rows: Vec<(Vec<f32>, bool)> = Vec::new();
    for r in &rows {
        let item = (r.features.clone(), r.label);
        if train_patients.contains(r.patient_id.as_str()) {
            train_rows.push(item);
        } else {
            test_rows.push(item);
        }
    }

    let train_normal: Vec<Vec<f32>> = train_rows
        .iter()
        .filter(|(_, lab)| !*lab)
        .map(|(f, _)| f.clone())
        .collect();

    let n_train = train_rows.len();
    let n_test = test_rows.len();
    let test_pos = test_rows.iter().filter(|(_, lab)| *lab).count();
    eprintln!(
        "  Train: {} ({} normal) | Test: {} ({} sepsis)",
        n_train,
        train_normal.len(),
        n_test,
        test_pos
    );

    let mut evaluator = SimpleHdcEvaluator::new(n_features, 0x5E9510001);

    let all_train_feats: Vec<Vec<f32>> = train_rows.iter().map(|(f, _)| f.clone()).collect();
    evaluator.fit_normalization(&all_train_feats);

    // Subsample reference if too large
    let reference_sample = deterministic_subsample(&train_normal, 10_000);
    let reference = evaluator.build_reference(&reference_sample);

    let (auc, f1) = evaluator.evaluate(&reference, &test_rows);

    eprintln!("  AUC: {:.4} | Best F1: {:.4}", auc, f1);
    Ok(DomainResult {
        name: "Sepsis (ICU)",
        train: n_train,
        test: n_test,
        features: n_features,
        auc,
        f1,
    })
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    eprintln!("================================================================");
    eprintln!("  Spacecraft + Sepsis — Zero-Training HDC Anomaly Detection");
    eprintln!("  HDC Dimension: {}", HDC_DIMENSION);
    eprintln!("================================================================");

    let smap = evaluate_smap()?;
    let msl = evaluate_msl()?;
    let sepsis = evaluate_sepsis()?;

    let results = [&smap, &msl, &sepsis];

    println!();
    println!("================================================================");
    println!("     Spacecraft + Sepsis — Zero-Training HDC Evaluation");
    println!(
        "     Architecture: HDC {}D | Encoding: ContinuousHV",
        HDC_DIMENSION
    );
    println!("================================================================");
    println!(
        "{:<22}| {:<9}| {:<9}| {:<9}| {:<7}| {}",
        "Domain", "Train", "Test", "Features", "AUC", "F1"
    );
    println!("--------------------|---------|---------|---------|-------|------");

    for r in &results {
        println!(
            "{:<22}| {:<9}| {:<9}| {:<9}| {:<7.4}| {:.4}",
            r.name,
            format_number(r.train),
            format_number(r.test),
            r.features,
            r.auc,
            r.f1,
        );
    }

    println!("================================================================");

    let avg_auc: f64 = results.iter().map(|r| r.auc).sum::<f64>() / results.len() as f64;
    let avg_f1: f64 = results.iter().map(|r| r.f1).sum::<f64>() / results.len() as f64;
    println!("\nMean AUC: {:.4}, Mean F1: {:.4}", avg_auc, avg_f1);

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
