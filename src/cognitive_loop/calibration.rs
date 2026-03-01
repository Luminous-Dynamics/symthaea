//! Psych-bench → cognitive loop calibration bridge.
//!
//! Translates normative z-scores from the psych-bench suite into neuromodulator
//! sensitivity adjustments and feedback variable proposals. This closes the loop
//! between *measuring* cognitive performance and *tuning* the system.
//!
//! ## Architecture
//!
//! ```text
//! NormativeReport (z-scores)
//!   ├── Stroop/Flanker z → DA receptor_sensitivity
//!   ├── N-back z         → ACh receptor_sensitivity
//!   ├── Stop-signal z    → NE receptor_sensitivity
//!   ├── CPT d' z         → 5-HT receptor_sensitivity
//!   └── FoK calibration  → confidence proposal
//! ```
//!
//! ## Usage
//!
//! Run the calibration after a psych-bench battery (e.g., during idle/sleep):
//!
//! ```ignore
//! let report = psych_bench::run_battery(&config);
//! let normative = NormativeReport::from_report(&report);
//! let calibration = NeuromodCalibration::from_normative(&normative);
//! calibration.apply(&mut bath);
//! ```

/// Calibration adjustment for a single neuromodulator transmitter.
#[derive(Debug, Clone)]
pub struct TransmitterCalibration {
    /// Target transmitter (DA, NE, 5-HT, ACh).
    pub transmitter: &'static str,
    /// Current z-score from psych-bench normative comparison.
    pub z_score: f64,
    /// Benchmark(s) driving this calibration.
    pub source_benchmarks: Vec<String>,
    /// Proposed receptor_sensitivity adjustment (multiplicative).
    pub sensitivity_factor: f32,
    /// Reasoning for the adjustment.
    pub rationale: String,
}

/// Complete neuromodulator calibration derived from psych-bench results.
#[derive(Debug, Clone)]
pub struct NeuromodCalibration {
    /// Per-transmitter calibration adjustments.
    pub adjustments: Vec<TransmitterCalibration>,
    /// Proposed confidence calibration (additive delta).
    pub confidence_delta: f32,
    /// Overall calibration quality (fraction of benchmarks with valid z-scores).
    pub coverage: f64,
}

impl NeuromodCalibration {
    /// Derive calibration from normative z-scores.
    ///
    /// Maps benchmark z-scores to transmitter sensitivity adjustments:
    /// - **DA**: Stroop/Flanker interference → if z > 1 (too much interference),
    ///   reduce DA sensitivity (less reward-driven impulsivity); if z < -1
    ///   (too little), increase DA sensitivity.
    /// - **ACh**: N-back/DigitSpan → working memory below human baseline
    ///   suggests insufficient attentional gating → boost ACh.
    /// - **NE**: Stop-signal/GoNoGo → poor inhibition suggests insufficient
    ///   arousal modulation → boost NE sensitivity.
    /// - **5-HT**: CPT/SART → poor sustained attention suggests insufficient
    ///   tonic serotonergic regulation → boost 5-HT.
    /// - **Confidence**: FoK calibration error → if overconfident (z > 0),
    ///   reduce prediction_confidence; if underconfident, boost.
    pub fn from_z_scores(z_scores: &[(&str, f64)]) -> Self {
        let mut adjustments = Vec::new();
        let mut confidence_delta: f32 = 0.0;
        let mut valid_count = 0;

        // DA calibration: interference benchmarks
        let da_benchmarks = ["Executive::Stroop", "Executive::Flanker"];
        let da_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| da_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !da_z.is_empty() {
            let mean_z = da_z.iter().sum::<f64>() / da_z.len() as f64;
            valid_count += 1;
            // Lower-is-better metrics (interference effects) → positive z = too much interference
            // Positive z → DA too high → reduce sensitivity (factor < 1.0)
            // z-score direction already correct: positive z = worse = attenuate
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "DA",
                z_score: mean_z,
                source_benchmarks: da_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Interference z={mean_z:.2}: {}",
                    if mean_z > 0.5 {
                        "above human baseline → attenuate DA"
                    } else if mean_z < -0.5 {
                        "below human baseline → boost DA"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // ACh calibration: working memory benchmarks
        let ach_benchmarks = ["WorM::N-back", "WorM::DigitSpan"];
        let ach_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| ach_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !ach_z.is_empty() {
            let mean_z = ach_z.iter().sum::<f64>() / ach_z.len() as f64;
            valid_count += 1;
            // Higher-is-better → positive z = good → keep; negative z = bad → boost
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "ACh",
                z_score: mean_z,
                source_benchmarks: ach_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Working memory z={mean_z:.2}: {}",
                    if mean_z < -0.5 {
                        "below human baseline → boost ACh"
                    } else if mean_z > 1.5 {
                        "well above baseline → slight ACh attenuation"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // NE calibration: inhibition benchmarks
        let ne_benchmarks = ["Inhibition::StopSignal", "Inhibition::GoNoGo"];
        let ne_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| ne_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !ne_z.is_empty() {
            let mean_z = ne_z.iter().sum::<f64>() / ne_z.len() as f64;
            valid_count += 1;
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "NE",
                z_score: mean_z,
                source_benchmarks: ne_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Inhibition z={mean_z:.2}: {}",
                    if mean_z < -0.5 {
                        "below human baseline → boost NE"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // 5-HT calibration: sustained attention benchmarks
        let sht_benchmarks = ["SustainedAttention::CPT", "SustainedAttention::SART"];
        let sht_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| sht_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !sht_z.is_empty() {
            let mean_z = sht_z.iter().sum::<f64>() / sht_z.len() as f64;
            valid_count += 1;
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "5-HT",
                z_score: mean_z,
                source_benchmarks: sht_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Sustained attention z={mean_z:.2}: {}",
                    if mean_z < -0.5 {
                        "below human baseline → boost 5-HT"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Confidence calibration: metacognitive calibration error
        if let Some((_, z)) = z_scores
            .iter()
            .find(|(name, _)| *name == "Metacognition::FeelingOfKnowing")
        {
            valid_count += 1;
            // calibration_error_ece is lower-is-better → positive z = worse calibration
            // If Symthaea is overconfident, reduce prediction_confidence
            confidence_delta = if *z > 0.5 {
                -0.02 // overconfident → reduce
            } else if *z < -0.5 {
                0.01 // underconfident → boost slightly
            } else {
                0.0
            };
        }

        let total_benchmarks = z_scores.len().max(1);
        NeuromodCalibration {
            adjustments,
            confidence_delta,
            coverage: valid_count as f64 / total_benchmarks as f64,
        }
    }

    /// Apply calibration adjustments to a neuromodulator bath.
    ///
    /// Multiplies each transmitter's `receptor_sensitivity` by the proposed
    /// factor, clamping to `[0.5, 2.0]` (the physiological range).
    pub fn apply(&self, bath: &mut symthaea_neuromodulators::NeuromodulatorBath) {
        for adj in &self.adjustments {
            let transmitter = match adj.transmitter {
                "DA" => &mut bath.dopamine,
                "NE" => &mut bath.noradrenaline,
                "5-HT" => &mut bath.serotonin,
                "ACh" => &mut bath.acetylcholine,
                _ => continue,
            };
            transmitter.receptor_sensitivity =
                (transmitter.receptor_sensitivity * adj.sensitivity_factor).clamp(0.5, 2.0);
        }
    }

    /// Format calibration as a human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Neuromod Calibration (coverage: {:.0}%)",
            self.coverage * 100.0
        ));
        for adj in &self.adjustments {
            lines.push(format!(
                "  {} → sensitivity ×{:.3} (z={:.2}, {})",
                adj.transmitter, adj.sensitivity_factor, adj.z_score, adj.rationale
            ));
        }
        if self.confidence_delta.abs() > 0.001 {
            lines.push(format!(
                "  confidence → {:+.3}",
                self.confidence_delta
            ));
        }
        lines.join("\n")
    }

    /// Construct from NormativeReport-style z-scores (sign-corrected: positive = better).
    ///
    /// NormativeReport negates lower-is-better metrics so positive always means
    /// "better than human baseline". This constructor un-corrects the sign for
    /// interference metrics (DA pathway) before delegating to `from_z_scores`.
    ///
    /// The `is_lower_better` predicate is applied to each benchmark's key metric
    /// to determine whether to negate.
    pub fn from_normative_z_scores(scores: &[(&str, &str, f64)]) -> Self {
        let raw: Vec<(&str, f64)> = scores
            .iter()
            .map(|(bench, metric, z)| {
                // NormativeReport already negated lower-is-better metrics.
                // Un-negate so calibration sees raw direction:
                //   interference: raw positive = worse = attenuate DA
                let raw_z = if is_lower_better_metric(metric) { -z } else { *z };
                (*bench, raw_z)
            })
            .collect();
        Self::from_z_scores(&raw)
    }
}

/// Check if a metric key represents a lower-is-better measure.
///
/// Mirrors the canonical `report::is_lower_better()` from psych-bench
/// without pulling in the full psych-bench dependency.
fn is_lower_better_metric(metric: &str) -> bool {
    matches!(
        metric,
        "stroop_effect"
            | "flanker_effect"
            | "dual_task_cost"
            | "calibration_error_ece"
            | "commission_errors"
            | "ssrt_ticks"
            | "coordination_cost"
            | "vigilance_decrement"
            | "disambiguation_cost"
            | "blink_magnitude"
    )
}

/// Convert a z-score to a receptor sensitivity multiplier.
///
/// Base mapping: `factor = 1.0 - z * 0.05` (5% per sigma)
/// - Positive z → factor < 1.0 (attenuate sensitivity)
/// - Negative z → factor > 1.0 (boost sensitivity)
/// - z = 0 → factor = 1.0 (no change)
///
/// `invert`: flips the z direction. Use when the z-score convention
/// is opposite to the desired adjustment direction (e.g., a metric where
/// positive z means good performance but the transmitter should be boosted).
///
/// For most psych-bench metrics, z already encodes the right direction:
/// - Interference (positive z = worse) → attenuate → `invert=false`
/// - WM accuracy (negative z = worse) → boost → `invert=false`
///
/// Clamped to `[0.85, 1.15]` — conservative adjustments to prevent
/// catastrophic parameter drift.
fn z_to_sensitivity_factor(z: f64, invert: bool) -> f32 {
    let z = if invert { -z } else { z };
    let raw = 1.0 - z * 0.05;
    raw.clamp(0.85, 1.15) as f32
}

// ═══════════════════════════════════════════════════════════════════════════════
// Autonomous Self-Assessment
// ═══════════════════════════════════════════════════════════════════════════════

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
    pe_ema: f64,
    /// EMA of coherence (proxy for ACh/working memory).
    coherence_ema: f64,
    /// EMA of confidence calibration error (|predicted - actual|).
    confidence_error_ema: f64,
    /// EMA of attention budget utilization.
    attention_utilization_ema: f64,
    /// Number of observations since last calibration.
    observations_since_calibration: u32,
    /// Minimum observations before triggering (avoids premature calibration).
    warmup_threshold: u32,
    /// Cooldown cycles remaining (prevents rapid re-calibration).
    cooldown: u32,
    /// Cooldown duration after calibration.
    cooldown_duration: u32,
    /// Whether calibration was triggered on the most recent `check_trigger()` call.
    /// Reset to false on each `update()`.
    last_triggered: bool,
}

impl Default for SelfAssessmentMonitor {
    fn default() -> Self {
        Self {
            pe_ema: 0.0,
            coherence_ema: 0.7,
            confidence_error_ema: 0.0,
            attention_utilization_ema: 0.5,
            observations_since_calibration: 0,
            warmup_threshold: 200,    // ~4 seconds at 50Hz
            cooldown: 0,
            cooldown_duration: 500,   // ~10 seconds between calibrations
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
    /// Whether personality drift is flagged as anomalous.
    pub drift_anomalous: bool,
}

impl SelfAssessmentMonitor {
    /// Update the monitor with this cycle's signals.
    ///
    /// Call once per cycle during the neuromod phase.
    pub fn update(&mut self, input: &SelfAssessmentInput) {
        const ALPHA: f64 = 0.02; // ~50 cycle half-life
        self.last_triggered = false;

        self.pe_ema = self.pe_ema * (1.0 - ALPHA) + input.prediction_error as f64 * ALPHA;
        self.coherence_ema =
            self.coherence_ema * (1.0 - ALPHA) + input.coherence as f64 * ALPHA;
        self.confidence_error_ema =
            self.confidence_error_ema * (1.0 - ALPHA) + input.confidence_calibration_error as f64 * ALPHA;
        self.attention_utilization_ema =
            self.attention_utilization_ema * (1.0 - ALPHA) + input.attention_utilization as f64 * ALPHA;

        self.observations_since_calibration += 1;

        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
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

        // Cooldown gate: skip when drift is anomalous AND prediction error is elevated.
        // Anomalous personality drift (Turrigiano 2008) signals receptor sensitivity
        // has wandered — urgent recalibration overrides the normal cooldown.
        if self.cooldown > 0 && !(drift_anomalous && self.pe_ema > 0.15) {
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

        // 5-HT proxy: attention utilization
        // Baseline ~0.6; if consistently >0.9, sustained attention deficit
        let attn_z = -(self.attention_utilization_ema - 0.6) / 0.15; // negative z when high
        if attn_z < -1.0 {
            z_scores.push(("SustainedAttention::CPT", attn_z));
            needs_calibration = true;
        }

        // Confidence proxy: calibration error
        // Baseline ~0.05; if > 0.15, metacognitive miscalibration
        if self.confidence_error_ema > 0.1 {
            // Positive z = overconfident (high error despite high confidence)
            let fok_z = (self.confidence_error_ema - 0.05) / 0.05;
            z_scores.push(("Metacognition::FeelingOfKnowing", fok_z));
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

    /// Whether the last `check_trigger()` call fired a calibration.
    pub fn last_triggered(&self) -> bool {
        self.last_triggered
    }

    /// Reset after external calibration (e.g., from psych-bench).
    pub fn reset_after_calibration(&mut self) {
        self.observations_since_calibration = 0;
        self.cooldown = self.cooldown_duration;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_from_z_scores() {
        let z_scores = vec![
            ("Executive::Stroop", 1.5),       // interference too high
            ("Executive::Flanker", 1.2),       // interference too high
            ("WorM::N-back", -0.8),            // WM below baseline
            ("Inhibition::StopSignal", 0.0),   // normal
            ("SustainedAttention::CPT", -1.0), // attention below baseline
        ];

        let cal = NeuromodCalibration::from_z_scores(&z_scores);

        // Should have 4 transmitter adjustments (DA, ACh, NE, 5-HT)
        assert_eq!(cal.adjustments.len(), 4);

        // DA should be attenuated (interference too high → reduce DA)
        let da = cal.adjustments.iter().find(|a| a.transmitter == "DA").unwrap();
        assert!(
            da.sensitivity_factor < 1.0,
            "High interference z should reduce DA sensitivity, got {}",
            da.sensitivity_factor
        );

        // ACh should be boosted (WM below baseline)
        let ach = cal.adjustments.iter().find(|a| a.transmitter == "ACh").unwrap();
        assert!(
            ach.sensitivity_factor > 1.0,
            "Low WM z should boost ACh sensitivity, got {}",
            ach.sensitivity_factor
        );

        // 5-HT should be boosted (attention below baseline)
        let sht = cal.adjustments.iter().find(|a| a.transmitter == "5-HT").unwrap();
        assert!(
            sht.sensitivity_factor > 1.0,
            "Low attention z should boost 5-HT sensitivity, got {}",
            sht.sensitivity_factor
        );
    }

    #[test]
    fn test_calibration_normal_range() {
        let z_scores = vec![
            ("Executive::Stroop", 0.0),
            ("WorM::N-back", 0.0),
            ("Inhibition::StopSignal", 0.0),
            ("SustainedAttention::CPT", 0.0),
        ];

        let cal = NeuromodCalibration::from_z_scores(&z_scores);

        // At z=0, all factors should be ~1.0
        for adj in &cal.adjustments {
            assert!(
                (adj.sensitivity_factor - 1.0).abs() < 0.01,
                "{}: factor should be ~1.0 at z=0, got {}",
                adj.transmitter,
                adj.sensitivity_factor
            );
        }
    }

    #[test]
    fn test_calibration_clamping() {
        let z_scores = vec![
            ("Executive::Stroop", 10.0), // extreme z
        ];

        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        let da = cal.adjustments.iter().find(|a| a.transmitter == "DA").unwrap();

        // Should be clamped, not run away
        assert!(da.sensitivity_factor >= 0.85);
        assert!(da.sensitivity_factor <= 1.15);
    }

    #[test]
    fn test_confidence_calibration() {
        let z_scores = vec![
            ("Metacognition::FeelingOfKnowing", 1.5), // overconfident
        ];

        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        assert!(
            cal.confidence_delta < 0.0,
            "Overconfident FoK should reduce confidence, got {}",
            cal.confidence_delta
        );
    }

    #[test]
    fn test_sensitivity_factor_symmetry() {
        let boost = z_to_sensitivity_factor(-1.0, false);
        let attenuate = z_to_sensitivity_factor(1.0, false);

        // Symmetric around 1.0
        assert!((boost - 1.0).abs() - (attenuate - 1.0).abs() < 0.01);
        assert!(boost > 1.0);
        assert!(attenuate < 1.0);
    }

    #[test]
    fn test_invert_flag() {
        // For lower-is-better metrics: positive z = bad = should reduce
        let normal = z_to_sensitivity_factor(1.0, false); // positive z, normal → attenuate
        let inverted = z_to_sensitivity_factor(1.0, true); // positive z, inverted → boost

        assert!(normal < 1.0);
        assert!(inverted > 1.0);
    }

    #[test]
    fn test_summary_format() {
        let z_scores = vec![
            ("Executive::Stroop", 1.0),
            ("WorM::N-back", -0.5),
        ];
        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        let summary = cal.summary();
        assert!(summary.contains("DA"));
        assert!(summary.contains("ACh"));
        assert!(summary.contains("Neuromod Calibration"));
    }

    #[test]
    fn test_from_normative_z_scores() {
        // NormativeReport z-scores are sign-corrected: positive = better.
        // For lower-is-better metrics (stroop_effect), positive z means
        // "less interference than baseline" = good.
        // from_normative_z_scores un-corrects the sign for calibration.
        let scores = vec![
            // Stroop: positive normative z = less interference = good
            // Un-corrected: raw z = -1.0 → don't attenuate DA
            ("Executive::Stroop", "stroop_effect", 1.0),
            // N-back: positive normative z = better WM = good
            // Higher-is-better → no sign change
            ("WorM::N-back", "nback_2::accuracy", -1.0),
        ];

        let cal = NeuromodCalibration::from_normative_z_scores(&scores);

        // DA: normative z=+1.0 on stroop_effect (lower-is-better, sign-corrected)
        // Un-corrected: raw z = -1.0 → DA should be BOOSTED (factor > 1.0)
        let da = cal.adjustments.iter().find(|a| a.transmitter == "DA").unwrap();
        assert!(
            da.sensitivity_factor > 1.0,
            "Good Stroop → boost DA, got {}",
            da.sensitivity_factor
        );

        // ACh: normative z=-1.0 on nback accuracy (higher-is-better)
        // No sign change → raw z = -1.0 → ACh should be BOOSTED
        let ach = cal.adjustments.iter().find(|a| a.transmitter == "ACh").unwrap();
        assert!(
            ach.sensitivity_factor > 1.0,
            "Poor WM → boost ACh, got {}",
            ach.sensitivity_factor
        );
    }

    #[test]
    fn test_apply_modifies_bath() {
        let z_scores = vec![
            ("Executive::Stroop", 2.0),       // high interference → attenuate DA
            ("SustainedAttention::CPT", -2.0), // poor attention → boost 5-HT
        ];
        let cal = NeuromodCalibration::from_z_scores(&z_scores);

        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        let da_before = bath.dopamine.receptor_sensitivity;
        let sht_before = bath.serotonin.receptor_sensitivity;

        cal.apply(&mut bath);

        assert!(
            bath.dopamine.receptor_sensitivity < da_before,
            "DA sensitivity should decrease: {} → {}",
            da_before, bath.dopamine.receptor_sensitivity
        );
        assert!(
            bath.serotonin.receptor_sensitivity > sht_before,
            "5-HT sensitivity should increase: {} → {}",
            sht_before, bath.serotonin.receptor_sensitivity
        );
    }

    #[test]
    fn test_is_lower_better_metric() {
        assert!(is_lower_better_metric("stroop_effect"));
        assert!(is_lower_better_metric("flanker_effect"));
        assert!(is_lower_better_metric("ssrt_ticks"));
        assert!(!is_lower_better_metric("nback_2::accuracy"));
        assert!(!is_lower_better_metric("overall_accuracy"));
        assert!(!is_lower_better_metric("fok_gamma"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Self-Assessment Monitor tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_self_assessment_no_trigger_during_warmup() {
        let mut monitor = SelfAssessmentMonitor::default();
        // Feed bad signals but within warmup period (200 cycles)
        let bad_input = SelfAssessmentInput {
            prediction_error: 0.5,
            coherence: 0.2,
            confidence_calibration_error: 0.3,
            attention_utilization: 0.95,
            drift_anomalous: false,
        };
        for _ in 0..100 {
            monitor.update(&bad_input);
        }
        // Should NOT trigger — only 100 observations, warmup is 200
        assert!(
            monitor.check_trigger(false).is_none(),
            "Should not trigger during warmup"
        );
    }

    #[test]
    fn test_self_assessment_triggers_on_high_pe() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 10, // low warmup for testing
            cooldown_duration: 5,
            ..Default::default()
        };

        let bad_input = SelfAssessmentInput {
            prediction_error: 0.5, // well above 0.1 baseline
            coherence: 0.7,
            confidence_calibration_error: 0.0,
            attention_utilization: 0.5,
            drift_anomalous: false,
        };

        // Feed enough cycles past warmup
        for _ in 0..50 {
            monitor.update(&bad_input);
        }

        let cal = monitor.check_trigger(false);
        assert!(cal.is_some(), "High PE should trigger calibration");
        let cal = cal.unwrap();
        // Should have DA adjustment (PE → interference proxy)
        assert!(
            cal.adjustments.iter().any(|a| a.transmitter == "DA"),
            "PE proxy should map to DA adjustment"
        );
    }

    #[test]
    fn test_self_assessment_cooldown() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 5,
            cooldown_duration: 100,
            ..Default::default()
        };

        let bad_input = SelfAssessmentInput {
            prediction_error: 0.5,
            coherence: 0.7,
            confidence_calibration_error: 0.0,
            attention_utilization: 0.5,
            drift_anomalous: false,
        };

        for _ in 0..30 {
            monitor.update(&bad_input);
        }

        // First trigger should succeed
        assert!(monitor.check_trigger(false).is_some());

        // Feed more bad data
        for _ in 0..20 {
            monitor.update(&bad_input);
        }

        // Second trigger should fail — cooldown active (100 cycles)
        assert!(
            monitor.check_trigger(false).is_none(),
            "Should not trigger during cooldown"
        );
    }

    #[test]
    fn test_self_assessment_drift_anomalous_bypasses_cooldown() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 5,
            cooldown_duration: 100,
            ..Default::default()
        };

        let bad_input = SelfAssessmentInput {
            prediction_error: 0.5, // high PE → pe_ema will converge above 0.15
            coherence: 0.7,
            confidence_calibration_error: 0.0,
            attention_utilization: 0.5,
            drift_anomalous: true,
        };

        for _ in 0..30 {
            monitor.update(&bad_input);
        }

        // First trigger should succeed
        assert!(monitor.check_trigger(true).is_some());

        // Feed more bad data (still in cooldown, only 20 cycles < 100)
        for _ in 0..20 {
            monitor.update(&bad_input);
        }

        // Normal check should fail — cooldown active
        assert!(
            monitor.check_trigger(false).is_none(),
            "Normal check should block during cooldown"
        );

        // But drift_anomalous=true should bypass cooldown because pe_ema > 0.15
        // (Turrigiano 2008: anomalous drift warrants urgent recalibration)
        assert!(
            monitor.check_trigger(true).is_some(),
            "Drift anomalous + elevated PE should bypass cooldown"
        );
    }

    #[test]
    fn test_self_assessment_no_trigger_normal_performance() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 5,
            cooldown_duration: 5,
            ..Default::default()
        };

        // Normal, healthy signals
        let good_input = SelfAssessmentInput {
            prediction_error: 0.08,
            coherence: 0.75,
            confidence_calibration_error: 0.03,
            attention_utilization: 0.5,
            drift_anomalous: false,
        };

        for _ in 0..100 {
            monitor.update(&good_input);
        }

        assert!(
            monitor.check_trigger(false).is_none(),
            "Normal performance should not trigger calibration"
        );
    }

    #[test]
    fn test_self_assessment_reset_after_calibration() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 5,
            cooldown_duration: 50,
            ..Default::default()
        };

        let bad_input = SelfAssessmentInput {
            prediction_error: 0.5,
            coherence: 0.7,
            confidence_calibration_error: 0.0,
            attention_utilization: 0.5,
            drift_anomalous: false,
        };

        for _ in 0..20 {
            monitor.update(&bad_input);
        }
        assert!(monitor.check_trigger(false).is_some());

        // External calibration resets cooldown
        monitor.reset_after_calibration();
        assert_eq!(monitor.observations_since_calibration, 0);
        assert_eq!(monitor.cooldown, 50);
    }
}
