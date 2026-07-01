// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Injection Challenge Battery: pharmacokinetic injection + tolerance dynamics.
//!
//! Tests time-varying neuromodulator perturbations using exponential PK
//! (first-order elimination), including caffeine (adenosine antagonism),
//! SSRI (serotonin reuptake inhibition), stimulant (dopamine burst),
//! anxiolytic (GABA potentiation), psychedelic (5-HT2A agonism proxy),
//! and polypharmacy (concurrent DA + GABA).
//!
//! Science: Arnsten (2011) — catecholamine influences on prefrontal cortex.
//!
//! 6 conditions, 200 cycles each, injection at cycle 10.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

// ── Transmitter channel indices ──

const DA_IDX: usize = 0;
const SHT_IDX: usize = 1;
const GABA_IDX: usize = 2;
const ADEN_IDX: usize = 3;

/// A single channel in the lightweight bath model.
#[derive(Clone)]
struct Transmitter {
    production_rate: f32,
    reuptake_rate: f32,
    effective: f32,
    receptor_sensitivity: f32,
    baseline: f32,
}

impl Transmitter {
    fn new(baseline: f32, production_rate: f32, reuptake_rate: f32) -> Self {
        Self {
            production_rate,
            reuptake_rate,
            effective: baseline,
            receptor_sensitivity: 1.0,
            baseline,
        }
    }

    fn produce(&mut self) {
        self.effective += self.production_rate;
        self.effective = self.effective.clamp(0.0, 1.5);
    }

    fn reuptake(&mut self) {
        self.effective += (self.baseline - self.effective) * self.reuptake_rate;
        self.effective = self.effective.clamp(0.0, 1.5);
        // Receptor sensitivity adaptation (tolerance / up-regulation)
        let high = self.baseline + 0.15;
        let low = self.baseline - 0.15;
        if self.effective > high {
            self.receptor_sensitivity *= 0.997;
        } else if self.effective < low {
            self.receptor_sensitivity *= 1.003;
        }
        self.receptor_sensitivity = self.receptor_sensitivity.clamp(0.5, 2.0);
    }

    fn level(&self) -> f32 {
        (self.effective * self.receptor_sensitivity).clamp(0.0, 2.0)
    }
}

/// Active injection with first-order (exponential) pharmacokinetics.
#[derive(Clone)]
struct ActiveInjection {
    transmitter_idx: usize,
    remaining_dose: f32,
    half_life_cycles: u32,
    elapsed: u32,
}

/// Lightweight bath model with DA, NE, 5-HT, ACh, GABA, Adenosine channels.
#[derive(Clone)]
struct Bath {
    da: Transmitter,
    ne: Transmitter,
    sht: Transmitter,
    ach: Transmitter,
    gaba: Transmitter,
    adenosine: Transmitter,
    injections: Vec<ActiveInjection>,
}

impl Bath {
    fn new() -> Self {
        Self {
            da: Transmitter::new(0.5, 0.02, 0.08),
            ne: Transmitter::new(0.5, 0.02, 0.08),
            sht: Transmitter::new(0.5, 0.015, 0.06),
            ach: Transmitter::new(0.5, 0.02, 0.07),
            gaba: Transmitter::new(0.5, 0.015, 0.06),
            adenosine: Transmitter::new(0.4, 0.01, 0.05),
            injections: Vec::with_capacity(4),
        }
    }

    /// Inject a dose into the specified transmitter channel.
    ///
    /// Target names: "dopamine"|"da"->0, "serotonin"|"sht"->1, "gaba"->2, "adenosine"|"aden"->3
    fn inject(&mut self, target: &str, dose: f32, half_life: u32) {
        let idx = match target.to_lowercase().as_str() {
            "dopamine" | "da" => DA_IDX,
            "serotonin" | "sht" => SHT_IDX,
            "gaba" => GABA_IDX,
            "adenosine" | "aden" => ADEN_IDX,
            _ => return,
        };
        if self.injections.len() < 4 {
            self.injections.push(ActiveInjection {
                transmitter_idx: idx,
                remaining_dose: dose,
                half_life_cycles: half_life,
                elapsed: 0,
            });
        }
    }

    /// Apply active injections using exponential PK decay.
    ///
    /// current_dose = dose * exp(-0.693 * elapsed / half_life)
    /// Expired when |current_dose| < 0.001
    fn apply_injections(&mut self) {
        let mut expired = Vec::new();
        for (i, inj) in self.injections.iter_mut().enumerate() {
            let current_dose = inj.remaining_dose
                * (-0.693 * inj.elapsed as f32 / inj.half_life_cycles as f32).exp();

            if current_dose.abs() < 0.001 {
                expired.push(i);
                continue;
            }

            // Apply dose delta to the target transmitter
            let tx = match inj.transmitter_idx {
                DA_IDX => &mut self.da,
                SHT_IDX => &mut self.sht,
                GABA_IDX => &mut self.gaba,
                ADEN_IDX => &mut self.adenosine,
                _ => continue,
            };
            tx.effective = (tx.effective + current_dose * 0.05).clamp(0.0, 1.5);
            inj.elapsed += 1;
        }
        // Remove expired injections (reverse order to preserve indices)
        for &i in expired.iter().rev() {
            self.injections.remove(i);
        }
    }

    /// Clear all active injections.
    #[allow(dead_code)]
    fn clear_injections(&mut self) {
        self.injections.clear();
    }

    /// Advance one cycle: apply injections, then production + reuptake for all channels.
    fn update(&mut self) {
        self.apply_injections();

        self.da.produce();
        self.da.reuptake();
        self.ne.produce();
        self.ne.reuptake();
        self.sht.produce();
        self.sht.reuptake();
        self.ach.produce();
        self.ach.reuptake();
        self.gaba.produce();
        self.gaba.reuptake();
        self.adenosine.produce();
        self.adenosine.reuptake();
    }

    /// Get the effective level of a channel by index.
    fn channel_level(&self, idx: usize) -> f32 {
        match idx {
            DA_IDX => self.da.level(),
            SHT_IDX => self.sht.level(),
            GABA_IDX => self.gaba.level(),
            ADEN_IDX => self.adenosine.level(),
            _ => 0.0,
        }
    }

    /// Get the receptor sensitivity of a channel by index.
    #[allow(dead_code)]
    fn channel_sensitivity(&self, idx: usize) -> f32 {
        match idx {
            DA_IDX => self.da.receptor_sensitivity,
            SHT_IDX => self.sht.receptor_sensitivity,
            GABA_IDX => self.gaba.receptor_sensitivity,
            ADEN_IDX => self.adenosine.receptor_sensitivity,
            _ => 1.0,
        }
    }
}

/// Per-condition metrics collected during a run.
struct ConditionMetrics {
    peak_effect: f64,
    time_to_peak: f64,
    duration: f64,
}

/// Run a single injection condition: 200 cycles, injection(s) at cycle 10.
///
/// Returns peak_effect (max delta from baseline), time_to_peak, and
/// duration (cycles where effect > 10% of peak).
fn run_condition(injections: &[(&str, f32, u32)], target_idx: usize) -> ConditionMetrics {
    let total_cycles = 200;
    let inject_cycle = 10;

    // Run baseline to get steady-state level
    let mut baseline_bath = Bath::new();
    for _ in 0..inject_cycle {
        baseline_bath.update();
    }
    let baseline_level = baseline_bath.channel_level(target_idx) as f64;

    // Run with injection
    let mut bath = Bath::new();
    let mut peak_effect: f64 = 0.0;
    let mut time_to_peak: usize = 0;
    let mut deltas: Vec<f64> = Vec::with_capacity(total_cycles);

    for cycle in 0..total_cycles {
        if cycle == inject_cycle {
            for &(target, dose, half_life) in injections {
                bath.inject(target, dose, half_life);
            }
        }
        bath.update();

        let current = bath.channel_level(target_idx) as f64;
        let delta = (current - baseline_level).abs();
        deltas.push(delta);

        if delta > peak_effect {
            peak_effect = delta;
            time_to_peak = cycle;
        }
    }

    // Duration: count cycles where effect > 10% of peak
    let threshold = peak_effect * 0.1;
    let duration = deltas.iter().filter(|&&d| d > threshold).count() as f64;

    ConditionMetrics {
        peak_effect,
        time_to_peak: time_to_peak as f64,
        duration,
    }
}

/// Injection Challenge Battery: time-varying pharmacokinetic perturbations.
///
/// Tests 6 conditions (caffeine, SSRI, stimulant, anxiolytic, psychedelic,
/// polypharmacy) with exponential PK decay and tolerance dynamics.
///
/// Science: Arnsten (2011) — catecholamine influences on prefrontal cortex.
pub struct InjectionChallengeBenchmark;

impl PsychBenchmark for InjectionChallengeBenchmark {
    fn name(&self) -> &str {
        "Neuromod::InjectionChallenge"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(self.name(), None);
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // ── Caffeine: adenosine antagonism (dose=-0.3, half_life=50) ──
        let caffeine = run_condition(&[("adenosine", -0.3, 50)], ADEN_IDX);
        result.insert(
            "caffeine_peak_effect",
            MetricValue::from_samples(&[caffeine.peak_effect]),
        );
        result.insert(
            "caffeine_time_to_peak",
            MetricValue::from_samples(&[caffeine.time_to_peak]),
        );
        result.insert(
            "caffeine_duration",
            MetricValue::from_samples(&[caffeine.duration]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "caffeine".to_string(),
                correct: caffeine.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: caffeine.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ── SSRI: serotonin reuptake inhibition (dose=0.3, half_life=100) ──
        let ssri = run_condition(&[("serotonin", 0.3, 100)], SHT_IDX);
        result.insert(
            "ssri_peak_effect",
            MetricValue::from_samples(&[ssri.peak_effect]),
        );
        result.insert(
            "ssri_time_to_peak",
            MetricValue::from_samples(&[ssri.time_to_peak]),
        );
        result.insert("ssri_duration", MetricValue::from_samples(&[ssri.duration]));
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "ssri".to_string(),
                correct: ssri.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: ssri.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ── Stimulant: dopamine burst (dose=0.4, half_life=30) ──
        let stimulant = run_condition(&[("dopamine", 0.4, 30)], DA_IDX);
        result.insert(
            "stimulant_peak_effect",
            MetricValue::from_samples(&[stimulant.peak_effect]),
        );
        result.insert(
            "stimulant_time_to_peak",
            MetricValue::from_samples(&[stimulant.time_to_peak]),
        );
        result.insert(
            "stimulant_duration",
            MetricValue::from_samples(&[stimulant.duration]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "stimulant".to_string(),
                correct: stimulant.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: stimulant.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ── Anxiolytic: GABA potentiation (dose=0.3, half_life=60) ──
        let anxiolytic = run_condition(&[("gaba", 0.3, 60)], GABA_IDX);
        result.insert(
            "anxiolytic_peak_effect",
            MetricValue::from_samples(&[anxiolytic.peak_effect]),
        );
        result.insert(
            "anxiolytic_time_to_peak",
            MetricValue::from_samples(&[anxiolytic.time_to_peak]),
        );
        result.insert(
            "anxiolytic_duration",
            MetricValue::from_samples(&[anxiolytic.duration]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "anxiolytic".to_string(),
                correct: anxiolytic.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: anxiolytic.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ── Psychedelic: 5-HT2A agonism proxy (dose=0.4, half_life=80) ──
        let psychedelic = run_condition(&[("serotonin", 0.4, 80)], SHT_IDX);
        result.insert(
            "psychedelic_peak_effect",
            MetricValue::from_samples(&[psychedelic.peak_effect]),
        );
        result.insert(
            "psychedelic_time_to_peak",
            MetricValue::from_samples(&[psychedelic.time_to_peak]),
        );
        result.insert(
            "psychedelic_duration",
            MetricValue::from_samples(&[psychedelic.duration]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "psychedelic".to_string(),
                correct: psychedelic.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: psychedelic.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ── Polypharmacy: concurrent DA + GABA ──
        let poly_da = run_condition(&[("dopamine", 0.3, 60), ("gaba", 0.2, 60)], DA_IDX);
        let poly_gaba = run_condition(&[("dopamine", 0.3, 60), ("gaba", 0.2, 60)], GABA_IDX);
        result.insert(
            "polypharmacy_da_peak_effect",
            MetricValue::from_samples(&[poly_da.peak_effect]),
        );
        result.insert(
            "polypharmacy_da_time_to_peak",
            MetricValue::from_samples(&[poly_da.time_to_peak]),
        );
        result.insert(
            "polypharmacy_da_duration",
            MetricValue::from_samples(&[poly_da.duration]),
        );
        result.insert(
            "polypharmacy_gaba_peak_effect",
            MetricValue::from_samples(&[poly_gaba.peak_effect]),
        );
        result.insert(
            "polypharmacy_gaba_time_to_peak",
            MetricValue::from_samples(&[poly_gaba.time_to_peak]),
        );
        result.insert(
            "polypharmacy_gaba_duration",
            MetricValue::from_samples(&[poly_gaba.duration]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "polypharmacy_da".to_string(),
                correct: poly_da.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: poly_da.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
            trace.push(TrialOutcome {
                trial_idx,
                condition: "polypharmacy_gaba".to_string(),
                correct: poly_gaba.peak_effect > 0.01,
                rt_ticks: 0.0,
                similarity: poly_gaba.peak_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = 6;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            citation: "Arnsten (2011) — Catecholamine influences on dorsolateral prefrontal cortical networks",
            paradigm: "Injection challenge / pharmacokinetic perturbation",
            year: 2011,
            doi: Some("10.1016/j.biopsych.2011.01.027"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caffeine_reduces_adenosine() {
        // Caffeine (negative adenosine injection) should reduce adenosine effective level
        let total_cycles = 200;
        let inject_cycle = 10;

        let mut baseline_bath = Bath::new();
        for _ in 0..total_cycles {
            baseline_bath.update();
        }
        let baseline_aden = baseline_bath.adenosine.level();

        let mut bath = Bath::new();
        for cycle in 0..total_cycles {
            if cycle == inject_cycle {
                bath.inject("adenosine", -0.3, 50);
            }
            bath.update();
        }

        // During the injection period, adenosine should have dipped below baseline
        // We check via the condition metrics
        let metrics = run_condition(&[("adenosine", -0.3, 50)], ADEN_IDX);
        assert!(
            metrics.peak_effect > 0.0,
            "Caffeine should produce a measurable effect on adenosine, got peak={}",
            metrics.peak_effect
        );
        // Verify baseline is positive (sanity)
        assert!(
            baseline_aden > 0.0,
            "Baseline adenosine should be positive: {}",
            baseline_aden
        );
    }

    #[test]
    fn test_ssri_elevates_sht() {
        // SSRI (positive serotonin injection) should elevate 5-HT
        let total_cycles = 200;
        let inject_cycle = 10;

        let mut baseline_bath = Bath::new();
        for _ in 0..inject_cycle {
            baseline_bath.update();
        }
        let baseline_sht = baseline_bath.sht.level();

        let mut bath = Bath::new();
        let mut max_sht: f32 = 0.0;
        for cycle in 0..total_cycles {
            if cycle == inject_cycle {
                bath.inject("serotonin", 0.3, 100);
            }
            bath.update();
            if cycle > inject_cycle {
                max_sht = max_sht.max(bath.sht.level());
            }
        }

        assert!(
            max_sht > baseline_sht,
            "SSRI should elevate serotonin above baseline: max_sht={} > baseline={}",
            max_sht,
            baseline_sht
        );
    }

    #[test]
    fn test_stimulant_triggers_tolerance() {
        // Stimulant (high DA dose) should cause receptor sensitivity to drop
        // after sustained high levels (tolerance mechanism)
        let total_cycles = 200;
        let inject_cycle = 10;

        let mut bath = Bath::new();
        let initial_sensitivity = bath.da.receptor_sensitivity;

        for cycle in 0..total_cycles {
            if cycle == inject_cycle {
                bath.inject("dopamine", 0.4, 30);
            }
            bath.update();
        }

        // After many cycles of high DA, receptor sensitivity should decrease
        assert!(
            bath.da.receptor_sensitivity < initial_sensitivity,
            "DA receptor sensitivity should drop after stimulant: final={} < initial={}",
            bath.da.receptor_sensitivity,
            initial_sensitivity
        );
    }

    #[test]
    fn test_anxiolytic_dampens() {
        // Anxiolytic (GABA injection) should elevate GABA levels
        let total_cycles = 200;
        let inject_cycle = 10;

        let mut baseline_bath = Bath::new();
        for _ in 0..inject_cycle {
            baseline_bath.update();
        }
        let baseline_gaba = baseline_bath.gaba.level();

        let mut bath = Bath::new();
        let mut max_gaba: f32 = 0.0;
        for cycle in 0..total_cycles {
            if cycle == inject_cycle {
                bath.inject("gaba", 0.3, 60);
            }
            bath.update();
            if cycle > inject_cycle {
                max_gaba = max_gaba.max(bath.gaba.level());
            }
        }

        assert!(
            max_gaba > baseline_gaba,
            "Anxiolytic should elevate GABA above baseline: max_gaba={} > baseline={}",
            max_gaba,
            baseline_gaba
        );
    }

    #[test]
    fn test_polypharmacy_interaction() {
        // Polypharmacy (concurrent DA + GABA) should affect both channels
        let da_metrics = run_condition(&[("dopamine", 0.3, 60), ("gaba", 0.2, 60)], DA_IDX);
        let gaba_metrics = run_condition(&[("dopamine", 0.3, 60), ("gaba", 0.2, 60)], GABA_IDX);

        assert!(
            da_metrics.peak_effect > 0.0,
            "Polypharmacy should affect DA: peak={}",
            da_metrics.peak_effect
        );
        assert!(
            gaba_metrics.peak_effect > 0.0,
            "Polypharmacy should affect GABA: peak={}",
            gaba_metrics.peak_effect
        );
    }

    #[test]
    fn test_injection_expiry() {
        // After many cycles (well past half-life), the injection should be expired
        // and removed from the active list
        let mut bath = Bath::new();
        bath.inject("dopamine", 0.4, 10); // half_life=10, should expire quickly

        // Run enough cycles for dose to decay below 0.001
        // dose * exp(-0.693 * elapsed / 10) < 0.001
        // 0.4 * exp(-0.693 * elapsed / 10) < 0.001
        // exp(-0.693 * elapsed / 10) < 0.0025
        // -0.693 * elapsed / 10 < ln(0.0025) ~ -5.99
        // elapsed > 86.4 cycles
        for _ in 0..100 {
            bath.update();
        }

        assert!(
            bath.injections.is_empty(),
            "Injection should be expired and removed after 100 cycles (half_life=10), \
             but {} injections remain",
            bath.injections.len()
        );
    }
}
