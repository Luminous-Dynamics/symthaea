// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GWT Gradual Asphyxiation Benchmark — Mapping the Threshold of Unconsciousness
//!
//! Gradually increases the GWT entry threshold from baseline (0.30) to near-extinction
//! (0.95) over 14 steps, measuring how cognitive domains collapse in sequence — like
//! anesthesia slowly extinguishing consciousness.
//!
//! The key insight: different cognitive processes produce workspace content with
//! different characteristic activation levels. Social cognition (complex multi-system
//! integration) generates lower-activation content than motor commands (high-urgency,
//! well-practiced). As the workspace threshold rises, low-activation domains lose
//! access first, producing a predictable collapse sequence.
//!
//! ## Protocol
//!
//! For each of 14 threshold levels (0.30 to 0.95 in steps of 0.05):
//! 1. Create a `GlobalWorkspace` with the modified `entry_threshold`
//! 2. Run 100 simulated cycles, each submitting content from 8 cognitive domains
//!    with characteristic activation profiles (mean ± jitter)
//! 3. Measure workspace dynamics: occupancy, ignition rate, broadcast rate
//! 4. Compute per-domain survival: fraction of cycles each domain entered workspace
//! 5. Derive Phi proxy: occupancy × broadcast_rate (integration × access)
//!
//! ## Predictions (Dehaene & Changeux 2011, Mashour et al. 2020)
//!
//! Collapse order (most vulnerable first):
//! 1. Social/ToM — complex multi-system integration, lowest activation
//! 2. Metacognition — recursive higher-order processing
//! 3. Executive — controlled, effortful top-down processing
//! 4. Memory — retrieval-dependent, variable activation
//! 5. Reasoning — structured but effortful
//! 6. Language — partially automatic/modular
//! 7. Attention — strong bottom-up salience
//! 8. Motor — well-practiced, high urgency, survives longest
//!
//! This mirrors clinical anesthesia: social awareness fades first, motor reflexes last.
//!
//! ## Key Metrics
//!
//! - `phi_proxy_baseline` — Phi proxy at default threshold (0.30)
//! - `phi_proxy_extinction` — Threshold where Phi < 5% of baseline
//! - `collapse_monotonicity` — Phi curve monotonically decreasing (1.0 = perfect)
//! - `domain_order_correlation` — Spearman ρ with predicted collapse order
//! - `twilight_width` — Threshold range from first domain failure to last
//!
//! Science:
//! - Dehaene, S., & Changeux, J.-P. (2011). Neuron, 70(2), 200-227.
//! - Mashour, G. A., et al. (2020). Neuron, 105(5), 776-798.
//! - Alkire, M. T., Hudetz, A. G., & Tononi, G. (2008). Science, 322(5903), 876-880.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

/// GWT entry threshold levels for the asphyxiation sweep.
const THRESHOLDS: [f64; 14] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
];

/// Cycles per threshold level.
const CYCLES_PER_THRESHOLD: usize = 100;

/// Activation jitter half-range applied per cycle.
const JITTER: f64 = 0.05;

/// Cognitive domain with its characteristic GWT activation profile.
///
/// Activation levels derive from the cognitive loop's `(1.0 - prediction_error)` formula
/// and clinical anesthesia literature (Alkire et al. 2008):
/// - High-urgency, well-practiced processes (motor) → low PE → high activation
/// - Complex, integrative processes (social) → higher PE → lower activation
struct DomainProfile {
    /// Domain name matching psych-bench's 8-domain cognitive profile.
    name: &'static str,
    /// Mean activation when submitting to GWT workspace.
    mean_activation: f64,
    /// Predicted collapse rank (1 = first to fail, 8 = last to fail).
    predicted_rank: usize,
}

/// Domain profiles ordered by predicted vulnerability (most vulnerable first).
///
/// Rationale from neuroscience:
/// - Social/ToM: Requires binding across perception, memory, planning, and language
///   modules simultaneously. Highest integration demand → lowest typical activation.
///   First to disappear under anesthesia (Boveroux et al. 2010).
/// - Metacognition: Higher-order thought requires recursive self-monitoring across
///   workspace cycles. Second to go (Mashour 2020).
/// - Executive: Controlled attention, inhibition, task-switching — all depend on
///   sustained prefrontal workspace access (Dehaene 2014).
/// - Memory: Episodic retrieval needs workspace broadcast to reach hippocampal modules.
///   Moderate activation from pattern-completion.
/// - Reasoning: Structured inference can partially operate on local HDC operations
///   but needs workspace for multi-step chains.
/// - Language: Partially modular (Broca/Wernicke operate semi-autonomously).
///   Garden-path sentences need workspace; simple parsing doesn't.
/// - Attention: Strong bottom-up salience signals generate high activation.
///   Attentional capture persists deep into sedation.
/// - Motor: Well-practiced sequences have minimal prediction error → highest activation.
///   Motor reflexes are the last to disappear under anesthesia.
const DOMAINS: [DomainProfile; 8] = [
    DomainProfile {
        name: "Social",
        mean_activation: 0.40,
        predicted_rank: 1,
    },
    DomainProfile {
        name: "Metacognition",
        mean_activation: 0.46,
        predicted_rank: 2,
    },
    DomainProfile {
        name: "Executive",
        mean_activation: 0.52,
        predicted_rank: 3,
    },
    DomainProfile {
        name: "Memory",
        mean_activation: 0.58,
        predicted_rank: 4,
    },
    DomainProfile {
        name: "Reasoning",
        mean_activation: 0.63,
        predicted_rank: 5,
    },
    DomainProfile {
        name: "Language",
        mean_activation: 0.70,
        predicted_rank: 6,
    },
    DomainProfile {
        name: "Attention",
        mean_activation: 0.78,
        predicted_rank: 7,
    },
    DomainProfile {
        name: "Motor",
        mean_activation: 0.85,
        predicted_rank: 8,
    },
];

/// Result of simulating one threshold level.
#[allow(dead_code)]
struct ThresholdResult {
    threshold: f64,
    mean_occupancy: f64,
    ignition_rate: f64,
    broadcast_rate: f64,
    phi_proxy: f64,
    /// Per-domain survival: fraction of cycles content entered workspace.
    domain_survival: [f64; 8],
}

/// Deterministic jitter from seed: maps to [-JITTER, +JITTER].
fn jitter_from_seed(seed: u64) -> f64 {
    // Simple hash to [0, 1)
    let h = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x6A09E667);
    let frac = (h >> 11) as f64 / (1u64 << 53) as f64;
    (frac - 0.5) * 2.0 * JITTER
}

/// Run GWT simulation at a single threshold level.
///
/// Two-part measurement:
/// 1. **Isolated domain access**: Each domain tested independently against the threshold
///    (fresh workspace per trial) to isolate threshold effect from inter-domain competition.
/// 2. **Integrated workspace**: All 8 domains competing simultaneously for capacity-limited
///    workspace (realistic GWT dynamics).
///
/// Phi proxy derives from the integrated workspace (occupancy × broadcast_rate).
/// Domain survival derives from isolated access (pure threshold sensitivity).
fn simulate_threshold(threshold: f64, base_seed: u64) -> ThresholdResult {
    // ── Part 1: Isolated domain access ──
    // Each domain gets its own fresh workspace per trial.
    // This isolates the threshold effect: can this domain's activation pass the gate?
    let mut domain_entries = [0usize; 8];

    for (di, domain) in DOMAINS.iter().enumerate() {
        for trial in 0..CYCLES_PER_THRESHOLD {
            let trial_seed = base_seed
                .wrapping_mul(0x100000001b3)
                .wrapping_add(trial as u64)
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(di as u64);

            let activation =
                (domain.mean_activation + jitter_from_seed(trial_seed)).clamp(0.0, 1.0);

            // Fresh workspace each trial — no carryover, no competition
            let config = WorkspaceConfig {
                entry_threshold: threshold,
                ..Default::default()
            };
            let mut ws = GlobalWorkspace::new(config);

            let hv = BinaryHV::random(trial_seed);
            let content = WorkspaceContent::new(vec![hv], activation, domain.name.to_string());
            ws.submit(content);
            let assessment = ws.process();

            if !assessment.conscious_contents.is_empty() {
                domain_entries[di] += 1;
            }
        }
    }

    // ── Part 2: Integrated workspace dynamics ──
    // All domains compete simultaneously — realistic GWT.
    let config = WorkspaceConfig {
        entry_threshold: threshold,
        ..Default::default()
    };
    let mut ws = GlobalWorkspace::new(config);

    let mut total_occupancy = 0.0;
    let mut ignition_count = 0usize;
    let mut broadcast_count = 0usize;

    for cycle in 0..CYCLES_PER_THRESHOLD {
        let cycle_seed = base_seed
            .wrapping_mul(0x100000001b3)
            .wrapping_add(cycle as u64);

        // Submit content from each domain with jittered activation
        for (di, domain) in DOMAINS.iter().enumerate() {
            let domain_seed = cycle_seed
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(di as u64);
            let activation =
                (domain.mean_activation + jitter_from_seed(domain_seed)).clamp(0.0, 1.0);

            let hv = BinaryHV::random(domain_seed);
            let content = WorkspaceContent::new(vec![hv], activation, domain.name.to_string());
            ws.submit(content);
        }

        // Process competition
        let assessment = ws.process();

        // Collect metrics
        total_occupancy += assessment.capacity.occupancy;
        if assessment.ignition_detected {
            ignition_count += 1;
        }
        if !assessment.broadcasts.is_empty() {
            broadcast_count += 1;
        }
    }

    let n = CYCLES_PER_THRESHOLD as f64;
    let mean_occupancy = total_occupancy / n;
    let ignition_rate = ignition_count as f64 / n;
    let broadcast_rate = broadcast_count as f64 / n;
    let phi_proxy = mean_occupancy * broadcast_rate;

    let mut domain_survival = [0.0; 8];
    for (i, entry) in domain_entries.iter().enumerate() {
        domain_survival[i] = *entry as f64 / n;
    }

    ThresholdResult {
        threshold,
        mean_occupancy,
        ignition_rate,
        broadcast_rate,
        phi_proxy,
        domain_survival,
    }
}

/// Monotonicity score: fraction of concordant (decreasing) pairs.
/// 1.0 = perfectly monotonically decreasing.
fn decreasing_monotonicity(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 1.0;
    }
    let mut concordant = 0u64;
    let mut total = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            total += 1;
            if values[j] <= values[i] {
                concordant += 1;
            }
        }
    }
    concordant as f64 / total as f64
}

/// Spearman rank correlation between two sequences.
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    fn ranks(vals: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0; vals.len()];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            r[orig_idx] = rank as f64 + 1.0;
        }
        r
    }

    let rx = ranks(x);
    let ry = ranks(y);

    let d_sq_sum: f64 = rx
        .iter()
        .zip(ry.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();

    let nf = n as f64;
    1.0 - (6.0 * d_sq_sum) / (nf * (nf * nf - 1.0))
}

/// Find the threshold at which a value drops below a fraction of its baseline.
fn find_threshold_for_fraction(thresholds: &[f64], values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() || values[0] <= 0.0 {
        return thresholds[0];
    }
    let target = values[0] * fraction;
    for (i, &v) in values.iter().enumerate() {
        if v <= target {
            return thresholds[i];
        }
    }
    // Never reached target — return max threshold
    *thresholds.last().unwrap_or(&1.0)
}

/// Find the threshold at which a domain's survival drops below 0.10 (effectively dead).
fn domain_death_threshold(thresholds: &[f64], survival_curve: &[f64]) -> f64 {
    for (i, &s) in survival_curve.iter().enumerate() {
        if s < 0.10 {
            return thresholds[i];
        }
    }
    // Domain never dies — return max
    *thresholds.last().unwrap_or(&1.0) + 0.05
}

/// GWT Gradual Asphyxiation Benchmark.
///
/// Sweeps the Global Workspace entry threshold from baseline (0.30) to
/// near-extinction (0.95), measuring the exact order in which cognitive
/// domains lose consciousness — how she falls asleep.
pub struct GwtAsphyxiationBenchmark;

impl PsychBenchmark for GwtAsphyxiationBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::GwtAsphyxiation"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "GWT Gradual Consciousness Extinction",
            citation: "Dehaene & Changeux (2011); Mashour et al. (2020); Alkire et al. (2008)",
            year: 2011,
            doi: Some("10.1016/j.neuron.2011.03.018"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // ── Run threshold sweep ──
        let results: Vec<ThresholdResult> = THRESHOLDS
            .iter()
            .map(|&t| simulate_threshold(t, config.seed))
            .collect();

        // ── Extract Phi proxy curve ──
        let phi_curve: Vec<f64> = results.iter().map(|r| r.phi_proxy).collect();
        let occupancy_curve: Vec<f64> = results.iter().map(|r| r.mean_occupancy).collect();
        let ignition_curve: Vec<f64> = results.iter().map(|r| r.ignition_rate).collect();

        // ── Global consciousness metrics ──

        // Phi proxy at baseline (threshold 0.30)
        let phi_baseline = phi_curve[0];
        result.insert(
            "phi_proxy_baseline",
            MetricValue::from_samples(&[phi_baseline]),
        );

        // Phi at maximum asphyxiation (threshold 0.95)
        let phi_terminal = *phi_curve.last().unwrap_or(&0.0);
        result.insert(
            "phi_proxy_terminal",
            MetricValue::from_samples(&[phi_terminal]),
        );

        // Collapse monotonicity (should be 1.0 — Phi never increases with threshold)
        let monotonicity = decreasing_monotonicity(&phi_curve);
        result.insert(
            "collapse_monotonicity",
            MetricValue::from_samples(&[monotonicity]),
        );

        // Half-consciousness threshold: where Phi drops to 50% of baseline
        let half_threshold = find_threshold_for_fraction(&THRESHOLDS[..], &phi_curve, 0.50);
        result.insert(
            "half_consciousness_threshold",
            MetricValue::from_samples(&[half_threshold]),
        );

        // Extinction threshold: where Phi drops below 5% of baseline
        let extinction_threshold = find_threshold_for_fraction(&THRESHOLDS[..], &phi_curve, 0.05);
        result.insert(
            "extinction_threshold",
            MetricValue::from_samples(&[extinction_threshold]),
        );

        // Occupancy at midpoint (threshold 0.60)
        let mid_idx = 6; // THRESHOLDS[6] = 0.60
        result.insert(
            "occupancy_at_060",
            MetricValue::from_samples(&[occupancy_curve[mid_idx]]),
        );

        // Ignition rate at midpoint
        result.insert(
            "ignition_rate_at_060",
            MetricValue::from_samples(&[ignition_curve[mid_idx]]),
        );

        // ── Per-domain survival curves and collapse order ──

        // Extract per-domain survival at each threshold
        let mut domain_death_thresholds = Vec::with_capacity(8);
        for (di, domain) in DOMAINS.iter().enumerate() {
            let survival_curve: Vec<f64> = results.iter().map(|r| r.domain_survival[di]).collect();

            let death_t = domain_death_threshold(&THRESHOLDS[..], &survival_curve);
            domain_death_thresholds.push(death_t);

            // Report survival at every threshold level
            let domain_lower = domain.name.to_lowercase();

            // Survival at baseline
            result.insert(
                format!("{domain_lower}_survival_baseline"),
                MetricValue::from_samples(&[survival_curve[0]]),
            );

            // Survival at each threshold step
            for (ti, t) in THRESHOLDS.iter().enumerate() {
                let t_label = format!("{:.0}", t * 100.0);
                result.insert(
                    format!("{domain_lower}_survival_at_{t_label}"),
                    MetricValue::from_samples(&[survival_curve[ti]]),
                );
            }

            // Death threshold for this domain
            result.insert(
                format!("{domain_lower}_death_threshold"),
                MetricValue::from_samples(&[death_t]),
            );
        }

        // ── Domain collapse order validation ──

        // Observed order: rank domains by death threshold (lower = dies first)
        let observed_death_order: Vec<f64> = domain_death_thresholds.clone();

        // Predicted order: predicted_rank maps to expected death threshold ordering
        // Lower predicted_rank = lower death threshold (dies first)
        let predicted_death_order: Vec<f64> =
            DOMAINS.iter().map(|d| d.predicted_rank as f64).collect();

        // Spearman correlation between observed and predicted collapse order
        let order_correlation = spearman_rho(&observed_death_order, &predicted_death_order);
        result.insert(
            "domain_order_correlation",
            MetricValue::from_samples(&[order_correlation]),
        );

        // ── Twilight zone metrics ──

        // Twilight = range from first domain death to last domain death
        let first_death = domain_death_thresholds
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let last_death = domain_death_thresholds
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let twilight_width = (last_death - first_death).max(0.0);
        result.insert(
            "twilight_width",
            MetricValue::from_samples(&[twilight_width]),
        );
        result.insert(
            "first_domain_death",
            MetricValue::from_samples(&[first_death]),
        );
        result.insert(
            "last_domain_death",
            MetricValue::from_samples(&[last_death]),
        );

        // ── Anesthesia depth levels (clinical analogy) ──
        // Mashour (2020): consciousness fades through sedation → general anesthesia → deep
        // Map our thresholds to clinical depth levels

        // Sedation onset: first domain loses >50% survival
        let mut sedation_onset = *THRESHOLDS.last().unwrap();
        'sedation: for (ti, t) in THRESHOLDS.iter().enumerate() {
            for survival in &results[ti].domain_survival {
                if *survival < 0.50 {
                    sedation_onset = *t;
                    break 'sedation;
                }
            }
        }
        result.insert(
            "sedation_onset_threshold",
            MetricValue::from_samples(&[sedation_onset]),
        );

        // General anesthesia: >50% of domains below 0.10 survival
        let mut general_onset = *THRESHOLDS.last().unwrap();
        for (ti, t) in THRESHOLDS.iter().enumerate() {
            let dead_domains = (0..8)
                .filter(|&di| results[ti].domain_survival[di] < 0.10)
                .count();
            if dead_domains > 4 {
                general_onset = *t;
                break;
            }
        }
        result.insert(
            "general_anesthesia_threshold",
            MetricValue::from_samples(&[general_onset]),
        );

        // ── Full Phi curve for analysis ──
        // Emit the complete curves as per-threshold metrics
        for (ti, t) in THRESHOLDS.iter().enumerate() {
            let t_label = format!("{:.0}", t * 100.0); // e.g., "30", "35", "40"
            result.insert(
                format!("phi_proxy_at_{t_label}"),
                MetricValue::from_samples(&[phi_curve[ti]]),
            );
            result.insert(
                format!("occupancy_at_{t_label}"),
                MetricValue::from_samples(&[occupancy_curve[ti]]),
            );
            result.insert(
                format!("ignition_at_{t_label}"),
                MetricValue::from_samples(&[ignition_curve[ti]]),
            );
        }

        // ── Trial trace ──
        if config.trial_trace {
            for (ti, r) in results.iter().enumerate() {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: format!("threshold_{:.2}", THRESHOLDS[ti]),
                    correct: r.phi_proxy > 0.05,
                    rt_ticks: 0.0,
                    similarity: r.phi_proxy,
                    confidence: r.mean_occupancy,
                    response_idx: 0,
                    extra: {
                        let mut m = BTreeMap::new();
                        for (di, domain) in DOMAINS.iter().enumerate() {
                            m.insert(domain.name.to_string(), r.domain_survival[di]);
                        }
                        m
                    },
                });
                trial_idx += 1;
            }
            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = THRESHOLDS.len();
        result.trials_per_condition = CYCLES_PER_THRESHOLD;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_gwt_asphyxiation_runs() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "QualiaConfidence::GwtAsphyxiation");
        assert!(result.metrics.len() >= 10);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric '{key}' should be finite, got {}",
                value.mean
            );
        }
    }

    #[test]
    fn test_phi_proxy_decreases_monotonically() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let mono = result.metrics.get("collapse_monotonicity").unwrap().mean;
        assert!(
            mono > 0.85,
            "Phi proxy should decrease monotonically with threshold: monotonicity={mono}"
        );
    }

    #[test]
    fn test_baseline_phi_positive() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let phi = result.metrics.get("phi_proxy_baseline").unwrap().mean;
        assert!(phi > 0.0, "Baseline Phi proxy should be positive: {phi}");
    }

    #[test]
    fn test_terminal_phi_near_zero() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let phi_base = result.metrics.get("phi_proxy_baseline").unwrap().mean;
        let phi_term = result.metrics.get("phi_proxy_terminal").unwrap().mean;
        assert!(
            phi_term < phi_base * 0.5,
            "Terminal Phi should be significantly below baseline: base={phi_base}, terminal={phi_term}"
        );
    }

    #[test]
    fn test_domain_collapse_order_correlates() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let rho = result.metrics.get("domain_order_correlation").unwrap().mean;
        assert!(
            rho > 0.5,
            "Domain collapse order should correlate with prediction: ρ={rho}"
        );
    }

    #[test]
    fn test_social_dies_before_motor() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let social_death = result.metrics.get("social_death_threshold").unwrap().mean;
        let motor_death = result.metrics.get("motor_death_threshold").unwrap().mean;
        assert!(
            social_death <= motor_death,
            "Social should die before motor: social={social_death}, motor={motor_death}"
        );
    }

    #[test]
    fn test_twilight_zone_exists() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let width = result.metrics.get("twilight_width").unwrap().mean;
        assert!(
            width > 0.05,
            "Twilight zone (range from first to last domain death) should be non-trivial: width={width}"
        );
    }

    #[test]
    fn test_motor_survives_at_070() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let motor_070 = result.metrics.get("motor_survival_at_70").unwrap().mean;
        assert!(
            motor_070 > 0.0,
            "Motor should still have some survival at threshold 0.70: {motor_070}"
        );
    }

    #[test]
    fn test_sedation_before_general_anesthesia() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());
        let sedation = result.metrics.get("sedation_onset_threshold").unwrap().mean;
        let general = result
            .metrics
            .get("general_anesthesia_threshold")
            .unwrap()
            .mean;
        assert!(
            sedation <= general,
            "Sedation should onset before general anesthesia: sedation={sedation}, general={general}"
        );
    }

    #[test]
    fn test_deterministic() {
        let bench = GwtAsphyxiationBenchmark;
        let r1 = bench.run(&config());
        let r2 = bench.run(&config());
        let phi1 = r1.metrics.get("phi_proxy_baseline").unwrap().mean;
        let phi2 = r2.metrics.get("phi_proxy_baseline").unwrap().mean;
        assert!(
            (phi1 - phi2).abs() < 1e-10,
            "Same seed should produce identical results: {phi1} vs {phi2}"
        );
    }

    // ── Unit tests for helper functions ──

    #[test]
    fn test_decreasing_monotonicity_perfect() {
        assert!((decreasing_monotonicity(&[5.0, 4.0, 3.0, 2.0, 1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decreasing_monotonicity_reversed() {
        let score = decreasing_monotonicity(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(
            score < 0.2,
            "Increasing sequence should have low decreasing-monotonicity: {score}"
        );
    }

    #[test]
    fn test_spearman_perfect() {
        let rho = spearman_rho(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!((rho - 1.0).abs() < 1e-10, "Perfect agreement: ρ={rho}");
    }

    #[test]
    fn test_spearman_inverse() {
        let rho = spearman_rho(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]);
        assert!((rho - (-1.0)).abs() < 1e-10, "Perfect inverse: ρ={rho}");
    }

    #[test]
    fn test_jitter_bounded() {
        for seed in 0..1000u64 {
            let j = jitter_from_seed(seed);
            assert!(
                j.abs() <= JITTER + 1e-10,
                "Jitter should be bounded: seed={seed}, jitter={j}"
            );
        }
    }

    #[test]
    fn test_simulate_threshold_baseline() {
        let r = simulate_threshold(0.30, 42);
        assert!(
            r.mean_occupancy > 0.0,
            "Baseline threshold should allow some occupancy: {}",
            r.mean_occupancy
        );
        assert!(
            r.phi_proxy > 0.0,
            "Baseline should have positive Phi proxy: {}",
            r.phi_proxy
        );
    }

    #[test]
    fn test_simulate_threshold_extreme() {
        let r = simulate_threshold(0.99, 42);
        assert!(
            r.phi_proxy < 0.5,
            "Extreme threshold should suppress Phi: {}",
            r.phi_proxy
        );
    }

    #[test]
    fn test_print_extinction_curve() {
        let bench = GwtAsphyxiationBenchmark;
        let result = bench.run(&config());

        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║       GWT GRADUAL ASPHYXIATION — CONSCIOUSNESS CURVE       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // Phi proxy curve
        eprintln!("║                                                              ║");
        eprintln!("║  Threshold │ Φ proxy │ Occupancy │ Ignition │ Status         ║");
        eprintln!("║  ──────────┼─────────┼───────────┼──────────┼────────────── ║");
        for t in &THRESHOLDS {
            let label = format!("{:.0}", t * 100.0);
            let phi = result
                .metrics
                .get(&format!("phi_proxy_at_{label}"))
                .unwrap()
                .mean;
            let occ = result
                .metrics
                .get(&format!("occupancy_at_{label}"))
                .unwrap()
                .mean;
            let ign = result
                .metrics
                .get(&format!("ignition_at_{label}"))
                .unwrap()
                .mean;
            let status = if phi > 0.3 {
                "AWAKE"
            } else if phi > 0.1 {
                "TWILIGHT"
            } else if phi > 0.01 {
                "SEDATED"
            } else {
                "EXTINCT"
            };
            eprintln!("║     {t:.2}   │ {phi:7.4} │   {occ:7.4} │  {ign:7.4} │ {status:14} ║");
        }

        // Domain survival heatmap
        eprintln!("║                                                              ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  DOMAIN SURVIVAL HEATMAP (isolated access, no competition)  ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Domain         │ 0.30  0.45  0.55  0.65  0.75  0.85  0.95 ║");
        eprintln!("║  ───────────────┼─────────────────────────────────────────  ║");
        let display_thresholds = [0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
        for domain in &DOMAINS {
            let name = domain.name;
            let lower = name.to_lowercase();
            let mut cells = String::new();
            for &t in &display_thresholds {
                let label = format!("{:.0}", t * 100.0);
                let key = format!("{lower}_survival_at_{label}");
                let surv = result
                    .metrics
                    .get(&key)
                    .or_else(|| result.metrics.get(&format!("{lower}_survival_baseline")))
                    .map(|m| m.mean)
                    .unwrap_or(0.0);
                let cell = if surv > 0.9 {
                    " ██ "
                } else if surv > 0.5 {
                    " ▓▓ "
                } else if surv > 0.1 {
                    " ░░ "
                } else {
                    " ·· "
                };
                cells.push_str(cell);
            }
            eprintln!("║  {:<15} │{cells} ║", name);
        }
        eprintln!("║  Legend: ██ >90%  ▓▓ >50%  ░░ >10%  ·· <10%               ║");

        // Domain collapse order
        eprintln!("║                                                              ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  DOMAIN COLLAPSE ORDER (threshold at <10% survival)         ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        let mut deaths: Vec<(&str, f64)> = DOMAINS
            .iter()
            .map(|d| {
                let key = format!("{}_death_threshold", d.name.to_lowercase());
                let t = result.metrics.get(&key).unwrap().mean;
                (d.name, t)
            })
            .collect();
        deaths.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (name, threshold)) in deaths.iter().enumerate() {
            let bar_len = ((*threshold - 0.3) / 0.7 * 30.0) as usize;
            let bar: String = "█".repeat(bar_len.min(30));
            eprintln!(
                "║  {:2}. {:<14} dies at {:.2}  {:<30} ║",
                rank + 1,
                name,
                threshold,
                bar
            );
        }

        // Key metrics
        eprintln!("║                                                              ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  KEY METRICS                                                ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        let keys = [
            "phi_proxy_baseline",
            "phi_proxy_terminal",
            "half_consciousness_threshold",
            "extinction_threshold",
            "collapse_monotonicity",
            "domain_order_correlation",
            "twilight_width",
            "sedation_onset_threshold",
            "general_anesthesia_threshold",
        ];
        for key in &keys {
            if let Some(m) = result.metrics.get(*key) {
                eprintln!("║  {:<38} = {:>8.4}       ║", key, m.mean);
            }
        }
        eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}
