// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC vs HDC+CfC — Temporal Anomaly Detection Evaluation
//!
//! Compares pure HDC snapshot encoding against HDC+CfC temporal trajectory encoding
//! on battery degradation and spacecraft anomaly detection.
//!
//! The CfC path adds a seeded, untrained continuous-time state transform to HDC snapshots.
//! This example currently evaluates a fixed dynamical feature map, not a trained anomaly
//! detector: it does not learn a healthy-trajectory model or calibrate thresholds on a
//! separate validation split. Treat results as exploratory feature comparisons.
//!
//! Usage:
//!   cargo run -p symthaea-physics --example hdc_cfc_eval --release

use std::collections::HashMap;
use std::path::PathBuf;

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ── Constants ────────────────────────────────────────────────────────────────

const DIM: usize = 4096; // Smaller dim for speed; still high-D enough for HDC
const HEALTHY_FRAC: f32 = 0.30;
const DEGRADED_FRAC: f32 = 0.30;
const SPACECRAFT_WINDOW: usize = 100;

// CfC tuning — moderate memory with meaningful per-step evolution.
// The gating sigma = 1 - exp(-dt/tau) * (1 - sigma_base).
// With dt=0.5, tau=2.0: exp(-0.25) ~ 0.78, so each step integrates ~22% new info.
// This means early samples still have ~0.78^200 ~ 0 influence at the end,
// but the nonlinear gating creates a richer trajectory-dependent representation
// compared to the linear bundle.
const CFC_TAU_BASE: f32 = 2.0;
const CFC_BACKBONE_TAU: f32 = 1.0; // State-dependent: complex states get more memory
const CFC_DT: f32 = 0.5;
const CFC_SEED: u64 = 0xCAFE;

// Subsampling: take every Nth sample from long sequences for CfC (speed)
const CFC_MAX_SAMPLES: usize = 200;

// ── Data Types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BatterySample {
    voltage: f32,
    current: f32,
    temperature: f32,
    #[allow(dead_code)]
    capacity: f32,
}

#[derive(Debug)]
struct BatteryCycle {
    battery_id: String,
    cycle_num: u32,
    samples: Vec<BatterySample>,
    #[allow(dead_code)]
    capacity: f32,
}

// ── HDC Encoder ──────────────────────────────────────────────────────────────

struct HdcEncoder {
    bases: Vec<ContinuousHV>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
    n_features: usize,
}

impl HdcEncoder {
    fn new(n_features: usize, seed_base: u64) -> Self {
        let bases = (0..n_features)
            .map(|i| ContinuousHV::random(DIM, seed_base + i as u64))
            .collect();

        Self {
            bases,
            mins: vec![f32::MAX; n_features],
            maxs: vec![f32::MIN; n_features],
            n_features,
        }
    }

    fn fit(&mut self, data: &[&[f32]]) {
        for row in data {
            for (i, &v) in row.iter().enumerate().take(self.n_features) {
                if v.is_finite() {
                    if v < self.mins[i] {
                        self.mins[i] = v;
                    }
                    if v > self.maxs[i] {
                        self.maxs[i] = v;
                    }
                }
            }
        }
        for i in 0..self.n_features {
            if (self.maxs[i] - self.mins[i]).abs() < 1e-10 {
                self.mins[i] = -1.0;
                self.maxs[i] = 1.0;
            }
        }
    }

    fn normalize(&self, v: f32, idx: usize) -> f32 {
        let range = self.maxs[idx] - self.mins[idx];
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((v - self.mins[idx]) / range) * 2.0 - 1.0
    }

    fn encode_sample(&self, features: &[f32]) -> ContinuousHV {
        let weights: Vec<f32> = features
            .iter()
            .enumerate()
            .take(self.n_features)
            .map(|(i, &v)| self.normalize(v, i))
            .collect();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }

    /// HDC-only: encode a cycle by bundling all sample encodings (order-invariant)
    fn encode_cycle_hdc_only(&self, samples: &[&[f32]]) -> ContinuousHV {
        if samples.is_empty() {
            return ContinuousHV::zero(DIM);
        }
        let encoded: Vec<ContinuousHV> = samples.iter().map(|s| self.encode_sample(s)).collect();
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// HDC+CfC: seeded, untrained continuous-time temporal feature transform.
    ///
    /// The neuron resets to the same deterministic initialization for each cycle and is not
    /// trained in this evaluation. State/input dissimilarity is a candidate temporal feature,
    /// not a learned or calibrated one-step prediction error. We also capture multi-point
    /// snapshots of the CfC state.
    ///
    /// Returns (cfc_surprise_profile, cfc_hybrid):
    /// - cfc_surprise_profile: accumulated surprise HVs at 4 checkpoints
    /// - cfc_hybrid: CfC surprise + HDC bundle combined
    fn encode_cycle_cfc(&self, samples: &[&[f32]]) -> (ContinuousHV, ContinuousHV) {
        if samples.len() < 2 {
            return (ContinuousHV::zero(DIM), ContinuousHV::zero(DIM));
        }

        let config = UnifiedConfig {
            tau_base: CFC_TAU_BASE,
            backbone_tau: CFC_BACKBONE_TAU,
            dimension: DIM,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, CFC_SEED);

        let step = if samples.len() > CFC_MAX_SAMPLES {
            samples.len() / CFC_MAX_SAMPLES
        } else {
            1
        };

        let subsampled: Vec<&[f32]> = samples.iter().step_by(step).copied().collect();
        let n = subsampled.len();

        // Checkpoints for multi-scale snapshots
        let checkpoints = [n / 4, n / 2, 3 * n / 4, n - 1];
        let mut snapshots: Vec<ContinuousHV> = Vec::with_capacity(4);
        let mut bundle_accum = ContinuousHV::zero(DIM);

        // Accumulate state/input dissimilarity scores (untrained "surprise")
        let mut total_surprise = 0.0f32;
        let mut surprise_accum = ContinuousHV::zero(DIM);

        for (i, &sample) in subsampled.iter().enumerate() {
            let feat_hv = self.encode_sample(sample);

            // State/input dissimilarity is an exploratory temporal feature. The neuron is
            // untrained here, so this is not a calibrated prediction error.
            let previous_state = neuron.state();
            let surprise = 1.0 - previous_state.similarity(&feat_hv);
            total_surprise += surprise;

            // Weight the feature HV by surprise: high-surprise samples get
            // MORE representation in the accumulated profile
            let weighted = feat_hv.scale(1.0 + surprise * 2.0);
            surprise_accum = surprise_accum.add(&weighted);

            // Evolve CfC neuron
            neuron.evolve_closed_form(CFC_DT, &feat_hv);

            // Snapshot at checkpoints
            if checkpoints.contains(&i) {
                snapshots.push(neuron.state().clone());
            }

            bundle_accum = bundle_accum.add(&feat_hv);
        }

        // Normalize
        let inv_n = 1.0 / n.max(1) as f32;
        bundle_accum = bundle_accum.scale(inv_n);
        surprise_accum = surprise_accum.scale(inv_n);

        // Bundle CfC snapshots
        let snap_refs: Vec<&ContinuousHV> = snapshots.iter().collect();
        let cfc_multiscale = if snap_refs.is_empty() {
            neuron.state().clone()
        } else {
            ContinuousHV::bundle(&snap_refs)
        };

        // CfC surprise profile: combine snapshots with surprise-weighted features
        let surprise_profile = ContinuousHV::bundle(&[&cfc_multiscale, &surprise_accum]);

        // Hybrid: combine surprise profile with HDC bundle
        let cfc_hybrid = ContinuousHV::bundle(&[&surprise_profile, &bundle_accum]);

        (surprise_profile, cfc_hybrid)
    }

    /// Return a candidate untrained-CfC state/input dissimilarity score for a cycle.
    /// Higher scores are a hypothesis for anomaly ranking, not a calibrated probability.
    fn cfc_surprise_score(&self, samples: &[&[f32]]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let config = UnifiedConfig {
            tau_base: CFC_TAU_BASE,
            backbone_tau: CFC_BACKBONE_TAU,
            dimension: DIM,
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, CFC_SEED);

        let step = if samples.len() > CFC_MAX_SAMPLES {
            samples.len() / CFC_MAX_SAMPLES
        } else {
            1
        };

        let mut total_surprise = 0.0f64;
        let mut count = 0u32;

        for sample in samples.iter().step_by(step) {
            let feat_hv = self.encode_sample(sample);
            let surprise = 1.0 - neuron.state().similarity(&feat_hv) as f64;
            total_surprise += surprise;
            count += 1;
            neuron.evolve_closed_form(CFC_DT, &feat_hv);
        }

        total_surprise / count.max(1) as f64
    }
}

// ── ROC / AUC Computation ────────────────────────────────────────────────────

struct RocPoint {
    fpr: f64,
    tpr: f64,
}

fn compute_roc_curve(scored: &[(f64, bool)]) -> Vec<RocPoint> {
    if scored.is_empty() {
        return vec![];
    }

    let total_pos = scored.iter().filter(|(_, l)| *l).count() as f64;
    let total_neg = scored.iter().filter(|(_, l)| !*l).count() as f64;

    if total_pos < 1.0 || total_neg < 1.0 {
        return vec![
            RocPoint { fpr: 0.0, tpr: 0.0 },
            RocPoint { fpr: 1.0, tpr: 1.0 },
        ];
    }

    let mut scores: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = scores.len();
    let step = (n / 200).max(1);
    let mut thresholds: Vec<f64> = Vec::with_capacity(204);
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
        points.push(RocPoint {
            fpr: fp as f64 / total_neg,
            tpr: tp as f64 / total_pos,
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
        return 0.5;
    }
    let mut auc = 0.0;
    for i in 1..roc.len() {
        let dx = roc[i].fpr - roc[i - 1].fpr;
        let avg_tpr = (roc[i].tpr + roc[i - 1].tpr) / 2.0;
        auc += dx * avg_tpr;
    }
    auc.clamp(0.0, 1.0)
}

/// Oracle F1 for exploratory inspection: threshold selection uses the supplied labels.
/// Do not report this as held-out performance; use a separate validation split to select
/// a deployment threshold and then evaluate that frozen threshold on test data.
fn find_best_f1(scored: &[(f64, bool)]) -> f64 {
    let total_pos = scored.iter().filter(|(_, l)| *l).count() as f64;
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

// ── Battery Data Loading ─────────────────────────────────────────────────────

fn load_battery_data(path: &std::path::Path) -> Vec<BatteryCycle> {
    eprintln!("Loading battery data from {:?}...", path);

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("Failed to open battery CSV");

    // Columns: Voltage_measured, Current_measured, Temperature_measured,
    //          Current_charge, Voltage_charge, Time, Capacity,
    //          id_cycle, type, ambient_temperature, time, Battery
    let mut cycles_map: HashMap<(String, u32), Vec<BatterySample>> = HashMap::new();
    let mut capacity_map: HashMap<(String, u32), f32> = HashMap::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        if record.len() < 12 {
            continue;
        }

        let voltage: f32 = record[0].parse().unwrap_or(0.0);
        let current: f32 = record[1].parse().unwrap_or(0.0);
        let temperature: f32 = record[2].parse().unwrap_or(0.0);
        let capacity: f32 = record[6].parse().unwrap_or(0.0);
        let cycle_num: u32 = record[7].parse().unwrap_or(0);
        let rec_type: &str = &record[8];
        let battery_id: String = record[11].to_string();

        if rec_type != "discharge" {
            continue;
        }
        if cycle_num == 0 {
            continue;
        }

        let key = (battery_id.clone(), cycle_num);
        cycles_map
            .entry(key.clone())
            .or_default()
            .push(BatterySample {
                voltage,
                current,
                temperature,
                capacity,
            });
        capacity_map.entry(key).or_insert(capacity);
    }

    let mut cycles: Vec<BatteryCycle> = cycles_map
        .into_iter()
        .map(|((battery_id, cycle_num), samples)| {
            let capacity = capacity_map
                .get(&(battery_id.clone(), cycle_num))
                .copied()
                .unwrap_or(0.0);
            BatteryCycle {
                battery_id,
                cycle_num,
                samples,
                capacity,
            }
        })
        .collect();

    cycles.sort_by(|a, b| {
        a.battery_id
            .cmp(&b.battery_id)
            .then(a.cycle_num.cmp(&b.cycle_num))
    });

    let total_samples: usize = cycles.iter().map(|c| c.samples.len()).sum();
    eprintln!(
        "  Loaded {} cycles, {} total samples across {} batteries",
        cycles.len(),
        total_samples,
        cycles
            .iter()
            .map(|c| &c.battery_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    cycles
}

// ── Battery Evaluation ───────────────────────────────────────────────────────

struct EvalResult {
    auc_hdc: f64,
    auc_cfc: f64,
    auc_hybrid: f64,
    auc_fused: f64,
    f1_hdc: f64,
    f1_cfc: f64,
    f1_hybrid: f64,
    f1_fused: f64,
}

/// HDC distance calibration estimated from training-only healthy representations.
///
/// The tail block is held out from the HDC reference prototype, preserving temporal order
/// for the battery/spacecraft datasets. Evaluation/test scores are never used to choose this
/// mean or scale. With fewer than two examples, use a neutral unit scale rather than
/// estimating calibration from the evaluation distribution.
struct HdcScoreCalibration {
    reference: ContinuousHV,
    mean: f64,
    std: f64,
    calibration_count: usize,
}

fn fit_hdc_score_calibration(healthy_vectors: &[ContinuousHV]) -> HdcScoreCalibration {
    match healthy_vectors.len() {
        0 => {
            return HdcScoreCalibration {
                reference: ContinuousHV::zero(DIM),
                mean: 0.0,
                std: 1.0,
                calibration_count: 0,
            };
        }
        1 => {
            return HdcScoreCalibration {
                reference: healthy_vectors[0].clone(),
                mean: 0.0,
                std: 1.0,
                calibration_count: 0,
            };
        }
        _ => {}
    }

    // Reserve the final 20% (rounded up) as a chronological calibration block.
    let calibration_count = ((healthy_vectors.len() + 4) / 5).min(healthy_vectors.len() - 1);
    let fit_count = healthy_vectors.len() - calibration_count;
    let fit_refs: Vec<&ContinuousHV> = healthy_vectors[..fit_count].iter().collect();
    let reference = ContinuousHV::bundle(&fit_refs);

    let calibration_scores: Vec<f64> = healthy_vectors[fit_count..]
        .iter()
        .map(|vector| 1.0 - vector.similarity(&reference) as f64)
        .collect();
    let mean = calibration_scores.iter().sum::<f64>() / calibration_scores.len() as f64;
    let std = if calibration_scores.len() >= 2 {
        let variance = calibration_scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / calibration_scores.len() as f64;
        variance.sqrt().max(0.001)
    } else {
        // A single calibration example does not identify a reliable scale.
        1.0
    };

    HdcScoreCalibration {
        reference,
        mean,
        std,
        calibration_count,
    }
}

fn evaluate_battery(cycles: &[BatteryCycle]) -> EvalResult {
    let mut by_battery: HashMap<&str, Vec<&BatteryCycle>> = HashMap::new();
    for c in cycles {
        by_battery.entry(c.battery_id.as_str()).or_default().push(c);
    }
    for v in by_battery.values_mut() {
        v.sort_by_key(|c| c.cycle_num);
    }

    let mut batteries: Vec<&str> = by_battery.keys().copied().collect();
    batteries.sort();
    eprintln!("  Batteries: {:?}", batteries);

    // Train on first 3, test on last 1
    let train_batteries = &batteries[..batteries.len().saturating_sub(1).max(1)];
    let test_batteries = &batteries[batteries.len().saturating_sub(1)..];
    eprintln!("  Train: {:?}, Test: {:?}", train_batteries, test_batteries);

    // Fit feature scaling on training batteries only. Test-battery extrema must not
    // influence the transform used to evaluate generalization to an unseen battery.
    let mut training_features: Vec<[f32; 3]> = Vec::new();
    for &bat in train_batteries {
        for cycle in &by_battery[bat] {
            for sample in &cycle.samples {
                training_features.push([sample.voltage, sample.current, sample.temperature]);
            }
        }
    }

    let train_feature_refs: Vec<&[f32]> =
        training_features.iter().map(|feature| feature.as_slice()).collect();
    let mut encoder = HdcEncoder::new(3, 0xBA7_0001);
    encoder.fit(&train_feature_refs);

    // Build healthy references
    let mut healthy_hdc: Vec<ContinuousHV> = Vec::new();
    let mut healthy_cfc_pure: Vec<ContinuousHV> = Vec::new();
    let mut healthy_cfc_hybrid: Vec<ContinuousHV> = Vec::new();

    // Also compute CfC surprise baseline from training healthy cycles
    let mut healthy_surprise_scores: Vec<f64> = Vec::new();

    for &bat in train_batteries {
        let bat_cycles = &by_battery[bat];
        let n_healthy = (bat_cycles.len() as f32 * HEALTHY_FRAC).ceil() as usize;

        for cycle in bat_cycles.iter().take(n_healthy) {
            let samples: Vec<[f32; 3]> = cycle
                .samples
                .iter()
                .map(|s| [s.voltage, s.current, s.temperature])
                .collect();
            let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();

            healthy_hdc.push(encoder.encode_cycle_hdc_only(&sample_refs));
            let (cfc_p, cfc_h) = encoder.encode_cycle_cfc(&sample_refs);
            healthy_cfc_pure.push(cfc_p);
            healthy_cfc_hybrid.push(cfc_h);
            healthy_surprise_scores.push(encoder.cfc_surprise_score(&sample_refs));
        }
    }

    let HdcScoreCalibration {
        reference: ref_hdc,
        mean: hdc_calibration_mean,
        std: hdc_calibration_std,
        calibration_count: hdc_calibration_count,
    } = fit_hdc_score_calibration(&healthy_hdc);
    let cfc_p_refs: Vec<&ContinuousHV> = healthy_cfc_pure.iter().collect();
    let cfc_h_refs: Vec<&ContinuousHV> = healthy_cfc_hybrid.iter().collect();
    let ref_cfc_pure = ContinuousHV::bundle(&cfc_p_refs);
    let ref_cfc_hybrid = ContinuousHV::bundle(&cfc_h_refs);

    eprintln!(
        "  HDC score calibration: {} held-out healthy cycles (mean={:.4}, std={:.4})",
        hdc_calibration_count, hdc_calibration_mean, hdc_calibration_std
    );

    // Reference surprise: mean and std of healthy cycles
    let mean_fn = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let std_fn = |v: &[f64]| {
        let m = mean_fn(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    let ref_surprise_mean = mean_fn(&healthy_surprise_scores);
    let ref_surprise_std = std_fn(&healthy_surprise_scores).max(0.001);

    eprintln!(
        "  Built reference from {} healthy cycles (surprise: {:.4}+/-{:.4})",
        healthy_hdc.len(),
        ref_surprise_mean,
        ref_surprise_std
    );

    // Evaluate on test battery
    let mut scored_hdc: Vec<(f64, bool)> = Vec::new();
    let mut scored_cfc: Vec<(f64, bool)> = Vec::new();
    let mut scored_hybrid: Vec<(f64, bool)> = Vec::new();
    let mut scored_surprise: Vec<(f64, bool)> = Vec::new();

    for &bat in test_batteries {
        let bat_cycles = &by_battery[bat];
        let n = bat_cycles.len();
        let n_healthy = (n as f32 * HEALTHY_FRAC).ceil() as usize;
        let n_degraded_start = n - (n as f32 * DEGRADED_FRAC).floor() as usize;

        for (i, cycle) in bat_cycles.iter().enumerate() {
            let is_degraded = if i < n_healthy {
                false
            } else if i >= n_degraded_start {
                true
            } else {
                continue;
            };

            let samples: Vec<[f32; 3]> = cycle
                .samples
                .iter()
                .map(|s| [s.voltage, s.current, s.temperature])
                .collect();
            let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();

            // HDC-only score
            let hdc_hv = encoder.encode_cycle_hdc_only(&sample_refs);
            scored_hdc.push((1.0 - hdc_hv.similarity(&ref_hdc) as f64, is_degraded));

            // CfC multi-snapshot + surprise profile scores
            let (cfc_p, cfc_h) = encoder.encode_cycle_cfc(&sample_refs);
            scored_cfc.push((1.0 - cfc_p.similarity(&ref_cfc_pure) as f64, is_degraded));
            scored_hybrid.push((1.0 - cfc_h.similarity(&ref_cfc_hybrid) as f64, is_degraded));

            // CfC surprise score: z-score of surprise vs healthy baseline
            let surprise = encoder.cfc_surprise_score(&sample_refs);
            let z_surprise = (surprise - ref_surprise_mean) / ref_surprise_std;
            scored_surprise.push((z_surprise, is_degraded));
        }
    }

    eprintln!(
        "  Test samples: {} (healthy: {}, degraded: {})",
        scored_hdc.len(),
        scored_hdc.iter().filter(|(_, l)| !l).count(),
        scored_hdc.iter().filter(|(_, l)| *l).count(),
    );

    for (label, scored) in &[
        ("HDC", &scored_hdc),
        ("CfC-multi", &scored_cfc),
        ("CfC-hybrid", &scored_hybrid),
        ("CfC-surprise", &scored_surprise),
    ] {
        let h: Vec<f64> = scored.iter().filter(|(_, l)| !l).map(|(s, _)| *s).collect();
        let d: Vec<f64> = scored.iter().filter(|(_, l)| *l).map(|(s, _)| *s).collect();
        eprintln!(
            "  {} scores: healthy={:.4}+/-{:.4}  degraded={:.4}+/-{:.4}  sep={:.4}",
            label,
            mean_fn(&h),
            std_fn(&h),
            mean_fn(&d),
            std_fn(&d),
            mean_fn(&d) - mean_fn(&h)
        );
    }

    // Score-level fusion uses training-only healthy calibration, never evaluation statistics.
    let hdc_mean = hdc_calibration_mean;
    let hdc_std = hdc_calibration_std;

    let scored_fused: Vec<(f64, bool)> = scored_hdc
        .iter()
        .zip(scored_surprise.iter())
        .map(|((h, label), (s, _))| {
            let h_z = (h - hdc_mean) / hdc_std;
            // Weight: HDC provides strong signal, CfC surprise adds temporal
            (0.7 * h_z + 0.3 * s, *label)
        })
        .collect();

    EvalResult {
        auc_hdc: compute_auc(&compute_roc_curve(&scored_hdc)),
        auc_cfc: compute_auc(&compute_roc_curve(&scored_cfc)),
        auc_hybrid: compute_auc(&compute_roc_curve(&scored_hybrid)),
        auc_fused: compute_auc(&compute_roc_curve(&scored_fused)),
        f1_hdc: find_best_f1(&scored_hdc),
        f1_cfc: find_best_f1(&scored_cfc),
        f1_hybrid: find_best_f1(&scored_hybrid),
        f1_fused: find_best_f1(&scored_fused),
    }
}

// ── Spacecraft Data Loading & Evaluation ─────────────────────────────────────

fn parse_labeled_row(record: &csv::StringRecord, max_cols: usize) -> Option<(Vec<f32>, bool)> {
    if record.len() < max_cols + 1 {
        return None;
    }

    let mut features = Vec::with_capacity(max_cols);
    for index in 0..max_cols {
        let value = record.get(index)?.parse::<f32>().ok()?;
        if !value.is_finite() {
            return None;
        }
        features.push(value);
    }

    let label = match record.get(max_cols)?.parse::<u8>().ok()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    Some((features, label))
}

fn load_csv_rows(path: &std::path::Path, max_cols: usize) -> Vec<(Vec<f32>, bool)> {
    eprintln!("Loading {:?}...", path);
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("Failed to open CSV");

    let mut rows: Vec<(Vec<f32>, bool)> = Vec::new();
    let mut skipped_rows = 0usize;
    for result in rdr.records() {
        match result {
            Ok(record) => match parse_labeled_row(&record, max_cols) {
                Some(row) => rows.push(row),
                None => skipped_rows += 1,
            },
            Err(_) => skipped_rows += 1,
        }
    }
    eprintln!(
        "  Loaded {} valid rows; skipped {} malformed, non-finite, short, or invalid-label rows",
        rows.len(), skipped_rows
    );
    rows
}

fn evaluate_spacecraft(
    train_path: &std::path::Path,
    test_path: &std::path::Path,
) -> Option<EvalResult> {
    let n_features = 25;

    let train_rows = load_csv_rows(train_path, n_features);
    let test_rows = load_csv_rows(test_path, n_features);
    if train_rows.is_empty() || test_rows.is_empty() {
        eprintln!(
            "[Spacecraft] Skipping evaluation: no valid labelled rows remain in train or test data."
        );
        return None;
    }

    // Fit normalization on the training partition only to keep test values held out.
    let train_refs: Vec<&[f32]> = train_rows
        .iter()
        .map(|(features, _)| features.as_slice())
        .collect();
    let mut encoder = HdcEncoder::new(n_features, 0x5C_0001);
    encoder.fit(&train_refs);

    // Group training data into windows, retaining their labels so labelled anomalies do
    // not contaminate the healthy reference or the HDC score calibration.
    eprintln!("  Grouping into windows of {}...", SPACECRAFT_WINDOW);
    let total_train_windows = train_rows.chunks(SPACECRAFT_WINDOW).len();
    let normal_train_window_count = train_rows
        .chunks(SPACECRAFT_WINDOW)
        .filter(|chunk| !chunk.iter().any(|(_, is_anomaly)| *is_anomaly))
        .count();
    let ref_windows: Vec<Vec<&[f32]>> = train_rows
        .chunks(SPACECRAFT_WINDOW)
        .filter(|chunk| !chunk.iter().any(|(_, is_anomaly)| *is_anomaly))
        .take(200)
        .map(|chunk| chunk.iter().map(|(features, _)| features.as_slice()).collect())
        .collect();
    eprintln!(
        "  Using {} reference windows ({} normal-labelled of {} total training windows)",
        ref_windows.len(),
        normal_train_window_count,
        total_train_windows
    );
    if ref_windows.is_empty() {
        eprintln!(
            "[Spacecraft] Skipping evaluation: no normal-labelled training window is available."
        );
        return None;
    }

    // Build healthy reference (subset for speed)
    let mut healthy_hdc: Vec<ContinuousHV> = Vec::new();
    let mut healthy_cfc_pure: Vec<ContinuousHV> = Vec::new();
    let mut healthy_cfc_hybrid: Vec<ContinuousHV> = Vec::new();
    let mut healthy_surprise: Vec<f64> = Vec::new();

    for window in &ref_windows {
        healthy_hdc.push(encoder.encode_cycle_hdc_only(window));
        let (cp, ch) = encoder.encode_cycle_cfc(window);
        healthy_cfc_pure.push(cp);
        healthy_cfc_hybrid.push(ch);
        healthy_surprise.push(encoder.cfc_surprise_score(window));
    }

    let HdcScoreCalibration {
        reference: ref_hdc,
        mean: hdc_calibration_mean,
        std: hdc_calibration_std,
        calibration_count: hdc_calibration_count,
    } = fit_hdc_score_calibration(&healthy_hdc);
    let cfc_p_refs: Vec<&ContinuousHV> = healthy_cfc_pure.iter().collect();
    let cfc_h_refs: Vec<&ContinuousHV> = healthy_cfc_hybrid.iter().collect();
    let ref_cfc_pure = ContinuousHV::bundle(&cfc_p_refs);
    let ref_cfc_hybrid = ContinuousHV::bundle(&cfc_h_refs);

    eprintln!(
        "  HDC score calibration: {} held-out healthy windows (mean={:.4}, std={:.4})",
        hdc_calibration_count, hdc_calibration_mean, hdc_calibration_std
    );

    let mean_fn = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let std_fn = |v: &[f64]| {
        let m = mean_fn(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    let ref_surp_mean = mean_fn(&healthy_surprise);
    let ref_surp_std = std_fn(&healthy_surprise).max(0.001);

    eprintln!(
        "  Built reference from {} training windows (surprise: {:.4}+/-{:.4})",
        ref_windows.len(),
        ref_surp_mean,
        ref_surp_std
    );

    // Evaluate test windows
    let test_windows: Vec<(Vec<&[f32]>, bool)> = test_rows
        .chunks(SPACECRAFT_WINDOW)
        .map(|chunk| {
            let feats: Vec<&[f32]> = chunk.iter().map(|(f, _)| f.as_slice()).collect();
            let is_anomaly = chunk.iter().any(|(_, l)| *l);
            (feats, is_anomaly)
        })
        .collect();

    eprintln!(
        "  Test windows: {} (normal: {}, anomalous: {})",
        test_windows.len(),
        test_windows.iter().filter(|(_, l)| !l).count(),
        test_windows.iter().filter(|(_, l)| *l).count(),
    );

    let mut scored_hdc: Vec<(f64, bool)> = Vec::new();
    let mut scored_cfc: Vec<(f64, bool)> = Vec::new();
    let mut scored_hybrid: Vec<(f64, bool)> = Vec::new();
    let mut scored_surprise: Vec<(f64, bool)> = Vec::new();

    for (idx, (window, is_anomaly)) in test_windows.iter().enumerate() {
        let hdc_hv = encoder.encode_cycle_hdc_only(window);
        scored_hdc.push((1.0 - hdc_hv.similarity(&ref_hdc) as f64, *is_anomaly));

        let (cp, ch) = encoder.encode_cycle_cfc(window);
        scored_cfc.push((1.0 - cp.similarity(&ref_cfc_pure) as f64, *is_anomaly));
        scored_hybrid.push((1.0 - ch.similarity(&ref_cfc_hybrid) as f64, *is_anomaly));

        let surprise = encoder.cfc_surprise_score(window);
        scored_surprise.push(((surprise - ref_surp_mean) / ref_surp_std, *is_anomaly));

        if (idx + 1) % 1000 == 0 {
            eprintln!(
                "  Processed {}/{} test windows...",
                idx + 1,
                test_windows.len()
            );
        }
    }

    // Score-level fusion uses training-only healthy calibration, never test-window statistics.
    let hdc_mean = hdc_calibration_mean;
    let hdc_std = hdc_calibration_std;

    let scored_fused: Vec<(f64, bool)> = scored_hdc
        .iter()
        .zip(scored_surprise.iter())
        .map(|((h, label), (s, _))| {
            let h_z = (h - hdc_mean) / hdc_std;
            (0.7 * h_z + 0.3 * s, *label)
        })
        .collect();

    Some(EvalResult {
        auc_hdc: compute_auc(&compute_roc_curve(&scored_hdc)),
        auc_cfc: compute_auc(&compute_roc_curve(&scored_cfc)),
        auc_hybrid: compute_auc(&compute_roc_curve(&scored_hybrid)),
        auc_fused: compute_auc(&compute_roc_curve(&scored_fused)),
        f1_hdc: find_best_f1(&scored_hdc),
        f1_cfc: find_best_f1(&scored_cfc),
        f1_hybrid: find_best_f1(&scored_hybrid),
        f1_fused: find_best_f1(&scored_fused),
    })
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");

    println!();
    println!("================================================================");
    println!("     HDC vs HDC+CfC — Temporal Anomaly Detection");
    println!("================================================================");
    println!();
    println!("Architecture:");
    println!("  HDC-Only:       encode(V,I,T) -> bundle per window -> cosine distance");
    println!("  CfC-Multiscale: encode -> CfC evolve -> snapshots at 25/50/75/100% -> bundle");
    println!("  CfC-HV-Hybrid:  bundle(CfC-multiscale, HDC-bundle) at HV level");
    println!("  HDC+CfC Fused:  average(HDC_score, CfC_score) at score level");
    println!(
        "  CfC neuron: HdcLtcUnifiedNeuron (tau={}, backbone={}, dt={}, dim={})",
        CFC_TAU_BASE, CFC_BACKBONE_TAU, CFC_DT, DIM
    );
    println!();

    // ── Battery evaluation ──────────────────────────────────────────────────
    let battery_path = base.join("data/battery/discharge.csv");
    if battery_path.exists() {
        eprintln!("[Battery] Starting evaluation...");
        let cycles = load_battery_data(&battery_path);
        let r = evaluate_battery(&cycles);
        eprintln!("[Battery] Done.");

        println!("---- Battery Degradation -----------------------------------------------");
        println!("  Method                AUC       Best-F1*");
        println!(
            "  HDC-Only              {:.3}     {:.3}",
            r.auc_hdc, r.f1_hdc
        );
        println!(
            "  CfC Multiscale        {:.3}     {:.3}",
            r.auc_cfc, r.f1_cfc
        );
        println!(
            "  CfC Surprise+HDC HV   {:.3}     {:.3}",
            r.auc_hybrid, r.f1_hybrid
        );
        println!(
            "  HDC+CfC Score Fused   {:.3}     {:.3}",
            r.auc_fused, r.f1_fused
        );
        let best_cfc = r.auc_cfc.max(r.auc_hybrid).max(r.auc_fused);
        let delta = best_cfc - r.auc_hdc;
        println!(
            "  Best vs HDC           {:+.3}     {:+.3}",
            delta,
            [r.f1_cfc, r.f1_hybrid, r.f1_fused]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                - r.f1_hdc
        );
        println!();
    } else {
        eprintln!("Battery data not found at {:?}, skipping.", battery_path);
    }

    // ── Spacecraft evaluation ───────────────────────────────────────────────
    let smap_train = base.join("data/spacecraft/smap_train.csv");
    let smap_test = base.join("data/spacecraft/smap_test.csv");
    if smap_train.exists() && smap_test.exists() {
        eprintln!("[Spacecraft] Starting evaluation...");
        match evaluate_spacecraft(&smap_train, &smap_test) {
            Some(r) => {
                eprintln!("[Spacecraft] Done.");
                println!("---- Spacecraft (SMAP) ----");
                println!("  Method                AUC       Best-F1*");
                println!(
                    "  HDC-Only              {:.3}     {:.3}",
                    r.auc_hdc, r.f1_hdc
                );
                println!(
                    "  CfC Multiscale        {:.3}     {:.3}",
                    r.auc_cfc, r.f1_cfc
                );
                println!(
                    "  CfC Surprise+HDC HV   {:.3}     {:.3}",
                    r.auc_hybrid, r.f1_hybrid
                );
                println!(
                    "  HDC+CfC Score Fused   {:.3}     {:.3}",
                    r.auc_fused, r.f1_fused
                );
                let best_cfc = r.auc_cfc.max(r.auc_hybrid).max(r.auc_fused);
                let delta = best_cfc - r.auc_hdc;
                println!(
                    "  Best vs HDC           {:+.3}     {:+.3}",
                    delta,
                    [r.f1_cfc, r.f1_hybrid, r.f1_fused]
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                        - r.f1_hdc
                );
                println!();
            }
            None => {
                eprintln!(
                    "[Spacecraft] Skipped: insufficient valid train/test calibration data."
                );
            }
        }
    } else {
        eprintln!("Spacecraft data not found, skipping.");
    }

    // ── Interpretation limits ───────────────────────────────────────────────
    println!("================================================================");
    println!("INTERPRETATION LIMITS:");
    println!("  CfC is a seeded, untrained temporal transform, not a learned detector.");
    println!("  Normalization is fit on training partitions only.");
    println!("  Best-F1 thresholds use evaluation labels and are oracle diagnostics,");
    println!("  not deployable metrics. Select thresholds on a separate validation split.");
    println!("  Pre-register split, seeds and tuning; repeat on independent data.");
    println!("================================================================");
}


#[cfg(test)]
mod hdc_score_calibration_tests {
    use super::*;

    #[test]
    fn spacecraft_row_parser_rejects_malformed_nonfinite_and_invalid_labels() {
        let mut valid = csv::StringRecord::new();
        valid.push_field("1.25");
        valid.push_field("-2.5");
        valid.push_field("1");
        assert_eq!(
            parse_labeled_row(&valid, 2),
            Some((vec![1.25, -2.5], true))
        );

        let mut nonfinite = csv::StringRecord::new();
        nonfinite.push_field("NaN");
        nonfinite.push_field("0.5");
        nonfinite.push_field("0");
        assert!(parse_labeled_row(&nonfinite, 2).is_none());

        let mut malformed = csv::StringRecord::new();
        malformed.push_field("not-a-number");
        malformed.push_field("0.5");
        malformed.push_field("0");
        assert!(parse_labeled_row(&malformed, 2).is_none());

        let mut invalid_label = csv::StringRecord::new();
        invalid_label.push_field("1.0");
        invalid_label.push_field("0.5");
        invalid_label.push_field("2");
        assert!(parse_labeled_row(&invalid_label, 2).is_none());
    }

    #[test]
    fn hdc_reference_and_scale_use_disjoint_training_calibration_block() {
        let healthy: Vec<ContinuousHV> = (0..6)
            .map(|index| ContinuousHV::random(DIM, 0xC411_BA7E + index))
            .collect();

        let calibration = fit_hdc_score_calibration(&healthy);
        assert_eq!(calibration.calibration_count, 2);

        let fit_refs: Vec<&ContinuousHV> = healthy[..4].iter().collect();
        let expected_reference = ContinuousHV::bundle(&fit_refs);
        assert!(
            (calibration.reference.similarity(&expected_reference) - 1.0).abs() < 1.0e-6,
            "reference must be built only from the first four training representations"
        );

        let expected_scores = [
            1.0 - healthy[4].similarity(&expected_reference) as f64,
            1.0 - healthy[5].similarity(&expected_reference) as f64,
        ];
        let expected_mean = expected_scores.iter().sum::<f64>() / expected_scores.len() as f64;
        let expected_std = (expected_scores
            .iter()
            .map(|score| (score - expected_mean).powi(2))
            .sum::<f64>()
            / expected_scores.len() as f64)
            .sqrt()
            .max(0.001);

        assert!((calibration.mean - expected_mean).abs() < 1.0e-12);
        assert!((calibration.std - expected_std).abs() < 1.0e-12);
    }

    #[test]
    fn hdc_calibration_uses_neutral_scale_when_data_is_insufficient() {
        let one = ContinuousHV::random(DIM, 7);
        let single = fit_hdc_score_calibration(std::slice::from_ref(&one));
        assert_eq!(single.calibration_count, 0);
        assert_eq!(single.mean, 0.0);
        assert_eq!(single.std, 1.0);
        assert!((single.reference.similarity(&one) - 1.0).abs() < 1.0e-6);

        let two = vec![one, ContinuousHV::random(DIM, 8)];
        let pair = fit_hdc_score_calibration(&two);
        assert_eq!(pair.calibration_count, 1);
        assert_eq!(pair.std, 1.0);
        assert!(pair.mean.is_finite());
    }
}
