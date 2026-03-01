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
}
