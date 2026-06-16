// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autonomous self-assessment monitor.
//!
//! Metacognitive performance monitor that tracks cognitive loop metrics
//! and triggers self-calibration when performance drifts beyond thresholds.

use super::normative::NeuromodCalibration;

/// Metacognitive performance monitor that tracks cognitive loop metrics
/// and triggers self-calibration when performance drifts.
///
/// Uses internal cognitive loop signals as proxy z-scores:
/// - **DA proxy**: prediction error trend (rising PE → interference-like → attenuate DA)
/// - **ACh proxy**: coherence stability (dropping coherence → WM-like deficit → boost ACh)
/// - **NE proxy**: error rate on safety checks (high → inhibition deficit → boost NE)
/// - **5-HT proxy**: attention budget utilization (consistently maxed → sustained attn deficit)
/// - **Confidence**: prediction confidence calibration vs actual error
///
/// Science: Schmidhuber (2010) — "Formal Theory of Creativity, Fun, and Intrinsic
/// Motivation" — metacognitive monitoring drives self-improvement cycles.
#[derive(Debug, Clone)]
pub struct SelfAssessmentMonitor {
    /// Exponential moving average of prediction error (proxy for DA/interference).
    pub(crate) pe_ema: f64,
    /// EMA of coherence (proxy for ACh/working memory).
    pub(crate) coherence_ema: f64,
    /// EMA of confidence calibration error (|predicted - actual|).
    pub(crate) confidence_error_ema: f64,
    /// EMA of attention budget utilization.
    pub(crate) attention_utilization_ema: f64,
    /// EMA of inhibition error rate (proxy for NE/prefrontal gating).
    pub(crate) inhibition_error_ema: f64,
    /// EMA of social coherence (proxy for Oxytocin).
    /// Kosfeld et al. (2005) — oxytocin mediates social trust.
    pub(crate) social_coherence_ema: f64,
    /// EMA of E/I ratio (proxy for GABA).
    /// Bhatt et al. (2009) — E/I balance homeostasis.
    pub(crate) ei_ratio_ema: f64,
    /// EMA of excitotoxicity risk (proxy for Glutamate).
    /// Olney (1969) — glutamate excitotoxicity.
    pub(crate) excitotoxicity_ema: f64,
    /// EMA of sleep pressure (proxy for Adenosine).
    /// Porkka-Heiskanen et al. (1997) — adenosine and sleep pressure.
    pub(crate) sleep_pressure_ema: f64,
    /// EMA of allostatic load (proxy for Endocannabinoid).
    /// Piomelli (2003) — endocannabinoid stress buffering.
    pub(crate) allostatic_load_ema: f64,
    /// Number of observations since last calibration.
    pub(crate) observations_since_calibration: u32,
    /// Minimum observations before triggering (avoids premature calibration).
    pub(crate) warmup_threshold: u32,
    /// Cooldown cycles remaining (prevents rapid re-calibration).
    pub(crate) cooldown: u32,
    /// Cooldown duration after calibration.
    pub(crate) cooldown_duration: u32,
    /// Whether calibration was triggered on the most recent `check_trigger()` call.
    /// Reset to false on each `update()`.
    pub(crate) last_triggered: bool,
}

impl Default for SelfAssessmentMonitor {
    fn default() -> Self {
        Self {
            pe_ema: 0.0,
            coherence_ema: 0.7,
            confidence_error_ema: 0.0,
            attention_utilization_ema: 0.5,
            inhibition_error_ema: 0.0,
            social_coherence_ema: 1.0,
            ei_ratio_ema: 1.0,
            excitotoxicity_ema: 0.0,
            sleep_pressure_ema: 0.3,
            allostatic_load_ema: 0.2,
            observations_since_calibration: 0,
            warmup_threshold: 200, // ~4 seconds at 50Hz
            cooldown: 0,
            cooldown_duration: 500, // ~10 seconds between calibrations
            last_triggered: false,
        }
    }
}

/// Signals from the cognitive loop that feed the self-assessment monitor.
pub struct SelfAssessmentInput {
    /// Current prediction error (0.0 = perfect, higher = worse).
    pub prediction_error: f32,
    /// Current coherence score (0.0–1.0, higher = better).
    pub coherence: f32,
    /// Whether prediction confidence was well-calibrated this cycle.
    /// Computed as |confidence - (1.0 - normalized_error)|.
    pub confidence_calibration_error: f32,
    /// Current attention budget utilization (0.0–1.0).
    pub attention_utilization: f32,
    /// Inhibition error rate this cycle (0.0–1.0).
    /// Fraction of active veto signals (prefrontal_veto, reasoning_gate_blocked,
    /// safety_blocked) that fired, indicating insufficient prefrontal gating.
    /// Science: Arnsten (2007) — NE modulates prefrontal cortex inhibitory control.
    pub inhibition_error_rate: f32,
    /// Whether personality drift is flagged as anomalous.
    pub drift_anomalous: bool,
    /// Social coherence factor (~1.0 = normal, <0.8 = low social bonding).
    /// Proxy for Oxytocin. Science: Kosfeld et al. (2005).
    pub social_coherence: f32,
    /// Excitatory/inhibitory ratio (~1.0 = balanced, >1.3 = excitatory dominant).
    /// Proxy for GABA. Science: Bhatt et al. (2009).
    pub ei_ratio: f32,
    /// Excitotoxicity risk (0.0 = safe, >0.3 = elevated).
    /// Proxy for Glutamate. Science: Olney (1969).
    pub excitotoxicity_risk: f32,
    /// Sleep pressure / adenosine accumulation (0.0–1.0).
    /// Proxy for Adenosine. Science: Porkka-Heiskanen et al. (1997).
    pub sleep_pressure: f32,
    /// Allostatic load (0.0–1.0, cumulative stress).
    /// Proxy for Endocannabinoid. Science: McEwen (1998), Piomelli (2003).
    pub allostatic_load: f32,
}

impl SelfAssessmentMonitor {
    /// Update the monitor with this cycle's signals.
    ///
    /// Call once per cycle during the neuromod phase.
    pub fn update(&mut self, input: &SelfAssessmentInput) {
        const ALPHA: f64 = 0.02; // ~50 cycle half-life
        self.last_triggered = false;

        self.pe_ema = self.pe_ema * (1.0 - ALPHA) + input.prediction_error as f64 * ALPHA;
        self.coherence_ema = self.coherence_ema * (1.0 - ALPHA) + input.coherence as f64 * ALPHA;
        self.confidence_error_ema = self.confidence_error_ema * (1.0 - ALPHA)
            + input.confidence_calibration_error as f64 * ALPHA;
        self.attention_utilization_ema = self.attention_utilization_ema * (1.0 - ALPHA)
            + input.attention_utilization as f64 * ALPHA;
        self.inhibition_error_ema =
            self.inhibition_error_ema * (1.0 - ALPHA) + input.inhibition_error_rate as f64 * ALPHA;
        self.social_coherence_ema =
            self.social_coherence_ema * (1.0 - ALPHA) + input.social_coherence as f64 * ALPHA;
        self.ei_ratio_ema = self.ei_ratio_ema * (1.0 - ALPHA) + input.ei_ratio as f64 * ALPHA;
        self.excitotoxicity_ema =
            self.excitotoxicity_ema * (1.0 - ALPHA) + input.excitotoxicity_risk as f64 * ALPHA;
        self.sleep_pressure_ema =
            self.sleep_pressure_ema * (1.0 - ALPHA) + input.sleep_pressure as f64 * ALPHA;
        self.allostatic_load_ema =
            self.allostatic_load_ema * (1.0 - ALPHA) + input.allostatic_load as f64 * ALPHA;

        self.observations_since_calibration = self.observations_since_calibration.saturating_add(1);

        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
    }

    /// Adjust trigger sensitivity based on calibration effectiveness.
    ///
    /// Called externally with the CalibrationValidator's improvement ratio.
    /// If calibrations consistently help, lower thresholds (calibrate more often);
    /// if regressions dominate, raise thresholds (calibrate less often).
    ///
    /// Science: Rescorla-Wagner (1972) — learning rate adapts to prediction accuracy.
    pub fn adapt_sensitivity(&mut self, improvement_ratio: f32) {
        // improvement_ratio: improvements / total, range [0, 1]
        // > 0.5 → calibrations are helping → lower cooldown
        // < 0.3 → calibrations are hurting → increase cooldown
        if improvement_ratio > 0.5 {
            // Effective calibrations → reduce cooldown (calibrate more often)
            self.cooldown_duration = (self.cooldown_duration as f32 * 0.9) as u32;
            self.cooldown_duration = self.cooldown_duration.max(200); // floor: 200 cycles
        } else if improvement_ratio < 0.3 {
            // Regressions dominate → increase cooldown (calibrate less often)
            self.cooldown_duration = (self.cooldown_duration as f32 * 1.15) as u32;
            self.cooldown_duration = self.cooldown_duration.min(2000); // cap: 2000 cycles
        }
        // 0.3..0.5 → neutral, no change
    }

    /// Check if calibration should be triggered.
    ///
    /// Returns `Some(calibration)` when performance drift exceeds thresholds
    /// and cooldown has elapsed. The calibration is built from internal
    /// performance proxies, not from psych-bench z-scores.
    pub fn check_trigger(&mut self, drift_anomalous: bool) -> Option<NeuromodCalibration> {
        // Gate: warmup and minimum observation count
        if self.observations_since_calibration < self.warmup_threshold {
            return None;
        }

        // Cooldown gate: override when drift is anomalous OR PE is severely elevated.
        // - Anomalous drift (Turrigiano 2008) = receptor sensitivity wander → urgent
        // - Severe PE (>0.35) = acute prediction failure beyond normal correction range
        // Lower threshold (0.15) was too aggressive — effectively disabled cooldown
        // whenever calibration was warranted. 0.35 preserves cooldown for moderate
        // drift while allowing emergency override for extreme performance failures.
        const COOLDOWN_OVERRIDE_PE: f64 = 0.35;
        if self.cooldown > 0 && !(drift_anomalous || self.pe_ema > COOLDOWN_OVERRIDE_PE) {
            return None;
        }

        // Compute proxy z-scores from EMA deviations.
        // These are unitless deviations from "expected good performance".
        let mut z_scores: Vec<(&str, f64)> = Vec::new();
        let mut needs_calibration = false;

        // DA proxy: prediction error trend
        // Baseline PE ~0.1; if EMA > 0.2, interference-like → positive z
        let pe_z = (self.pe_ema - 0.1) / 0.05; // ~2σ at PE=0.2
        if pe_z.abs() > 1.0 {
            z_scores.push(("Executive::Stroop", pe_z));
            needs_calibration = true;
        }

        // ACh proxy: coherence
        // Baseline ~0.7; if EMA < 0.5, WM-like deficit → negative z
        let coherence_z = (self.coherence_ema - 0.7) / 0.1; // ~2σ at coherence=0.5
        if coherence_z < -1.0 {
            z_scores.push(("WorM::N-back", coherence_z));
            needs_calibration = true;
        }

        // NE proxy: inhibition error rate (Arnsten 2007)
        // Baseline ~0.02; if EMA > 0.1, inhibition deficit → negative z (boost NE)
        let inhib_z = -(self.inhibition_error_ema - 0.02) / 0.04; // negative z when high
        if inhib_z < -1.0 {
            z_scores.push(("Inhibition::StopSignal", inhib_z));
            needs_calibration = true;
        }

        // 5-HT proxy: attention utilization
        // Baseline ~0.6; if consistently >0.9, sustained attention deficit
        let attn_z = -(self.attention_utilization_ema - 0.6) / 0.15; // negative z when high
        if attn_z < -1.0 {
            z_scores.push(("SustainedAttention::CPT", attn_z));
            needs_calibration = true;
        }

        // Confidence proxy: calibration error (bidirectional)
        // Baseline ~0.05; both overconfidence (>0.1) and underconfidence trigger
        let fok_z = (self.confidence_error_ema - 0.05) / 0.05;
        if fok_z.abs() > 1.0 {
            z_scores.push(("Metacognition::FeelingOfKnowing", fok_z));
            needs_calibration = true;
        }

        // GABA proxy: E/I ratio deviation (Bhatt 2009)
        // Baseline ~1.0; if EMA > 1.25, excitatory dominance → insufficient GABA
        let gaba_z = -(self.ei_ratio_ema - 1.0) / 0.25;
        if gaba_z < -1.0 {
            z_scores.push(("Executive::DualTask", gaba_z));
            needs_calibration = true;
        }

        // Oxytocin proxy: social coherence (Kosfeld 2005)
        // Baseline ~0.80 (matches input transform: oxy.effective() * 0.5 + 0.5 ≈ 0.75)
        // If EMA < 0.70, low social bonding → insufficient oxytocin
        let oxy_z = (self.social_coherence_ema - 0.80) / 0.1;
        if oxy_z < -1.0 {
            z_scores.push(("Social::UltimatumGame", oxy_z));
            needs_calibration = true;
        }

        // Glutamate proxy: excitotoxicity risk (Olney 1969)
        // Baseline ~0.1; if EMA > 0.25, excitotoxicity → glutamate fatigue
        let glut_z = (self.excitotoxicity_ema - 0.1) / 0.15;
        if glut_z > 1.0 {
            z_scores.push(("PVT::lapse_rate", glut_z));
            needs_calibration = true;
        }

        // Adenosine proxy: sleep pressure (Porkka-Heiskanen 1997)
        // Baseline ~0.3; if EMA > 0.55, high sleep debt
        let aden_z = (self.sleep_pressure_ema - 0.3) / 0.25;
        if aden_z > 1.0 {
            z_scores.push(("SustainedAttention::PVT", aden_z));
            needs_calibration = true;
        }

        // Endocannabinoid proxy: allostatic load (Piomelli 2003)
        // Baseline ~0.2; if EMA > 0.35, stress buffering depleted → positive z
        // (No negation: high load = positive z, which from_z_scores inverts
        //  to produce factor > 1.0 = boost ECB sensitivity.)
        let ecb_z = (self.allostatic_load_ema - 0.2) / 0.15;
        if ecb_z > 1.0 {
            z_scores.push(("Allostatic::Endocannabinoid", ecb_z));
            needs_calibration = true;
        }

        if !needs_calibration {
            return None;
        }

        // Build and return calibration
        self.observations_since_calibration = 0;
        self.cooldown = self.cooldown_duration;
        self.last_triggered = true;

        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        tracing::info!(
            pe_ema = %format!("{:.3}", self.pe_ema),
            coherence_ema = %format!("{:.3}", self.coherence_ema),
            confidence_error = %format!("{:.3}", self.confidence_error_ema),
            attention_util = %format!("{:.3}", self.attention_utilization_ema),
            inhibition_error = %format!("{:.3}", self.inhibition_error_ema),
            adjustments = cal.adjustments.len(),
            "Self-assessment triggered calibration"
        );

        Some(cal)
    }

    // ── Telemetry accessors ──────────────────────────────────────────────

    /// Current PE exponential moving average.
    pub fn pe_ema(&self) -> f64 {
        self.pe_ema
    }

    /// Current coherence exponential moving average.
    pub fn coherence_ema(&self) -> f64 {
        self.coherence_ema
    }

    /// Current confidence calibration error EMA.
    pub fn confidence_error_ema(&self) -> f64 {
        self.confidence_error_ema
    }

    /// Current attention utilization EMA.
    pub fn attention_utilization_ema(&self) -> f64 {
        self.attention_utilization_ema
    }

    /// Current inhibition error EMA (proxy for NE).
    pub fn inhibition_error_ema(&self) -> f64 {
        self.inhibition_error_ema
    }

    /// Current adaptive cooldown duration (cycles between calibrations).
    pub fn cooldown_duration(&self) -> u32 {
        self.cooldown_duration
    }

    /// Current social coherence EMA (proxy for Oxytocin).
    pub fn social_coherence_ema(&self) -> f64 {
        self.social_coherence_ema
    }

    /// Current E/I ratio EMA (proxy for GABA).
    pub fn ei_ratio_ema(&self) -> f64 {
        self.ei_ratio_ema
    }

    /// Current excitotoxicity risk EMA (proxy for Glutamate).
    pub fn excitotoxicity_ema(&self) -> f64 {
        self.excitotoxicity_ema
    }

    /// Current sleep pressure EMA (proxy for Adenosine).
    pub fn sleep_pressure_ema(&self) -> f64 {
        self.sleep_pressure_ema
    }

    /// Current allostatic load EMA (proxy for Endocannabinoid).
    pub fn allostatic_load_ema(&self) -> f64 {
        self.allostatic_load_ema
    }

    /// Whether the last `check_trigger()` call fired a calibration.
    pub fn last_triggered(&self) -> bool {
        self.last_triggered
    }

    /// Number of observations since last calibration reset.
    pub fn observations_count(&self) -> u32 {
        self.observations_since_calibration
    }

    /// Remaining cooldown cycles before trigger is eligible again.
    pub fn cooldown_remaining(&self) -> u32 {
        self.cooldown
    }

    /// Reset after external calibration (e.g., from psych-bench).
    pub fn reset_after_calibration(&mut self) {
        self.observations_since_calibration = 0;
        self.cooldown = self.cooldown_duration;
    }
}
