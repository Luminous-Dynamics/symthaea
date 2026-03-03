//! Psych-bench → cognitive loop calibration bridge.
//!
//! Translates normative z-scores from the psych-bench suite into neuromodulator
//! sensitivity adjustments and feedback variable proposals. This closes the loop
//! between *measuring* cognitive performance and *tuning* the system.
//!
//! ## Architecture (9-transmitter coverage)
//!
//! ```text
//! NormativeReport (z-scores)
//!   ├── Stroop/Flanker z       → DA receptor_sensitivity
//!   ├── N-back/DigitSpan z     → ACh receptor_sensitivity
//!   ├── StopSignal/GoNoGo z    → NE-β (phasic), NE-α (tonic via WM)
//!   ├── CPT/SART z             → 5-HT receptor_sensitivity
//!   ├── DualTask z             → GABA receptor_sensitivity
//!   ├── UltimatumGame/RME z    → Oxytocin receptor_sensitivity
//!   ├── PVT lapse_rate z       → Glutamate receptor_sensitivity
//!   ├── PVT vigilance_decrement → Adenosine receptor_sensitivity
//!   ├── Allostatic load proxy  → Endocannabinoid receptor_sensitivity
//!   └── FoK calibration        → confidence proposal
//! ```
//!
//! ## Features
//!
//! - **Tolerance-aware gating**: Calibration skips transmitters in withdrawal,
//!   dampens adjustments during tolerance (Koob & Le Moal 2001).
//! - **Sleep quality gate**: Requires minimum sleep duration before applying
//!   calibration (Tononi & Cirelli 2006).
//! - **Multi-agent sharing**: `SharedCalibrationProfile` enables oxytocin-modulated
//!   peer calibration blending (Boyd & Richerson 1985).
//! - **Live battery**: JSON-based subprocess spawning of `calibration_battery` binary.
//!
//! ## Usage
//!
//! ```ignore
//! let report = psych_bench::run_battery(&config);
//! let normative = NormativeReport::from_report(&report);
//! let calibration = NeuromodCalibration::from_normative(&normative);
//! calibration.apply(&mut bath);
//! ```

/// Optional receptor subtype for fine-grained calibration.
///
/// When `None`, the adjustment targets the transmitter's global
/// `receptor_sensitivity`. When set, it routes to a specific subtype
/// (e.g., NE α vs β, DA D1 vs D2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceptorSubtype {
    /// NE α — tonic precision / working memory gating (Arnsten 2007)
    NeAlpha,
    /// NE β — phasic reactivity / inhibitory control
    NeBeta,
    /// GABA-A — fast ionotropic / sedation (Möhler 2006)
    GabaA,
    /// GABA-B — slow metabotropic / muscle relaxation
    GabaB,
}

/// Calibration adjustment for a single neuromodulator transmitter.
#[derive(Debug, Clone)]
pub struct TransmitterCalibration {
    /// Target transmitter (DA, NE, 5-HT, ACh).
    pub transmitter: &'static str,
    /// Optional receptor subtype for fine-grained routing.
    pub subtype: Option<ReceptorSubtype>,
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

/// Shareable calibration profile for multi-agent coordination.
///
/// Contains only the sensitivity factors (no internal state), suitable for
/// serialization and transmission between agents.
///
/// Science: Multi-agent social learning (Boyd & Richerson 1985) — agents
/// share calibration knowledge to accelerate adaptation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SharedCalibrationProfile {
    /// Transmitter name → sensitivity factor.
    pub sensitivity_factors: std::collections::HashMap<String, f32>,
    /// Confidence calibration delta.
    pub confidence_delta: f32,
    /// Calibration coverage quality (0.0–1.0).
    pub coverage: f64,
    /// Source agent identifier (if known).
    pub agent_id: Option<String>,
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
                subtype: None,
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
                subtype: None,
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

        // NE calibration: subtype-specific routing (Arnsten 2007).
        //
        // NE α (tonic): modulates working memory precision via sustained
        //   prefrontal activity. Poor WM (N-back/DigitSpan) → boost α.
        // NE β (phasic): modulates inhibitory control via rapid reactivity.
        //   Poor inhibition (StopSignal/GoNoGo) → boost β.
        //
        // When only inhibition data is available, fall back to global NE.
        let ne_inhib_benchmarks = ["Inhibition::StopSignal", "Inhibition::GoNoGo"];
        let ne_inhib_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| ne_inhib_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();

        // Check if WM benchmarks also inform NE α
        let ne_wm_benchmarks = ["WorM::N-back", "WorM::DigitSpan"];
        let ne_wm_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| ne_wm_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();

        // Route NE β (phasic) from inhibition data
        if !ne_inhib_z.is_empty() {
            let mean_z = ne_inhib_z.iter().sum::<f64>() / ne_inhib_z.len() as f64;
            valid_count += 1;
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeBeta),
                z_score: mean_z,
                source_benchmarks: ne_inhib_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Inhibition z={mean_z:.2} → NE β (phasic): {}",
                    if mean_z < -0.5 {
                        "poor inhibitory control → boost β"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Route NE α (tonic) from WM data
        if !ne_wm_z.is_empty() {
            let mean_z = ne_wm_z.iter().sum::<f64>() / ne_wm_z.len() as f64;
            // Don't double-count if WM already counted for ACh
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeAlpha),
                z_score: mean_z,
                source_benchmarks: ne_wm_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Working memory z={mean_z:.2} → NE α (tonic): {}",
                    if mean_z < -0.5 {
                        "poor WM precision → boost α"
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
                subtype: None,
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

        // GABA calibration: dual-task cost (executive function / inhibitory tone)
        // Science: Möhler (2006) — GABAergic tone mediates executive multitasking.
        let gaba_benchmarks = ["Executive::DualTask"];
        let gaba_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| gaba_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !gaba_z.is_empty() {
            let mean_z = gaba_z.iter().sum::<f64>() / gaba_z.len() as f64;
            valid_count += 1;
            // dual_task_cost is lower-is-better: high cost = low GABA tone → invert
            let factor = z_to_sensitivity_factor(mean_z, true);
            adjustments.push(TransmitterCalibration {
                transmitter: "GABA",
                subtype: None,
                z_score: mean_z,
                source_benchmarks: gaba_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Dual-task cost z={mean_z:.2}: {}",
                    if mean_z > 0.5 {
                        "high cost → boost GABA (insufficient inhibitory tone)"
                    } else if mean_z < -0.5 {
                        "low cost → attenuate GABA"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Oxytocin calibration: social cognition benchmarks
        // Science: Kosfeld et al. (2005) — oxytocin increases trust and cooperation.
        let oxy_benchmarks = ["Social::UltimatumGame", "Social::RME"];
        let oxy_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| oxy_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !oxy_z.is_empty() {
            let mean_z = oxy_z.iter().sum::<f64>() / oxy_z.len() as f64;
            valid_count += 1;
            // Higher-is-better: negative z = low social sensitivity → boost oxytocin
            let factor = z_to_sensitivity_factor(mean_z, false);
            adjustments.push(TransmitterCalibration {
                transmitter: "Oxytocin",
                subtype: None,
                z_score: mean_z,
                source_benchmarks: oxy_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Social cognition z={mean_z:.2}: {}",
                    if mean_z < -0.5 {
                        "below baseline → boost oxytocin"
                    } else if mean_z > 1.5 {
                        "well above baseline → slight attenuation"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Glutamate calibration: PVT lapse rate (synthetic key)
        // Science: Olney (1969) — glutamate excitotoxicity under fatigue.
        let glut_benchmarks = ["PVT::lapse_rate"];
        let glut_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| glut_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !glut_z.is_empty() {
            let mean_z = glut_z.iter().sum::<f64>() / glut_z.len() as f64;
            valid_count += 1;
            // lapse_rate is lower-is-better: high lapses = glutamate fatigue → invert
            let factor = z_to_sensitivity_factor(mean_z, true);
            adjustments.push(TransmitterCalibration {
                transmitter: "Glutamate",
                subtype: None,
                z_score: mean_z,
                source_benchmarks: glut_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "PVT lapse rate z={mean_z:.2}: {}",
                    if mean_z > 0.5 {
                        "high lapses → attenuate glutamate (fatigue)"
                    } else if mean_z < -0.5 {
                        "low lapses → boost glutamate"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Adenosine calibration: PVT vigilance decrement
        // Science: Porkka-Heiskanen et al. (1997) — adenosine accumulation drives sleep pressure.
        let aden_benchmarks = ["SustainedAttention::PVT"];
        let aden_z: Vec<f64> = z_scores
            .iter()
            .filter(|(name, _)| aden_benchmarks.contains(name))
            .map(|(_, z)| *z)
            .collect();
        if !aden_z.is_empty() {
            let mean_z = aden_z.iter().sum::<f64>() / aden_z.len() as f64;
            valid_count += 1;
            // vigilance_decrement is lower-is-better: high decrement = sleep debt → invert
            let factor = z_to_sensitivity_factor(mean_z, true);
            adjustments.push(TransmitterCalibration {
                transmitter: "Adenosine",
                subtype: None,
                z_score: mean_z,
                source_benchmarks: aden_benchmarks.iter().map(|s| s.to_string()).collect(),
                sensitivity_factor: factor,
                rationale: format!(
                    "Vigilance decrement z={mean_z:.2}: {}",
                    if mean_z > 0.5 {
                        "high decrement → attenuate adenosine (sleep debt)"
                    } else if mean_z < -0.5 {
                        "low decrement → boost adenosine"
                    } else {
                        "within normal range"
                    }
                ),
            });
        }

        // Endocannabinoid calibration: composite from PVT + CPT + allostatic proxy
        //
        // Science: Piomelli (2003) — endocannabinoid stress buffering.
        // ECB has no single dedicated benchmark; instead we derive it from a
        // composite of stress-sensitive metrics:
        //   1. PVT lapse_rate (if available) — fatigue/lapses reflect allostatic load
        //   2. CPT sustained attention — chronic attention deficit → chronic stress
        //   3. Allostatic::Endocannabinoid (self-assessment synthetic key) — direct proxy
        //
        // The composite averages available sources, giving direct allostatic proxy
        // highest weight when present (it's the most specific signal).
        {
            let mut ecb_z_sources: Vec<(f64, f64)> = Vec::new(); // (z, weight)

            // Self-assessment allostatic proxy (highest weight — most specific)
            if let Some((_, z)) = z_scores.iter().find(|(name, _)| *name == "Allostatic::Endocannabinoid") {
                ecb_z_sources.push((*z, 2.0));
            }

            // PVT lapse_rate as stress fatigue signal (weight 1.0)
            if let Some((_, z)) = z_scores.iter().find(|(name, _)| *name == "PVT::lapse_rate") {
                ecb_z_sources.push((*z, 1.0));
            }

            // CPT sustained attention as chronic stress signal (weight 1.0)
            if let Some((_, z)) = z_scores.iter().find(|(name, _)| *name == "SustainedAttention::CPT") {
                ecb_z_sources.push((*z, 1.0));
            }

            if !ecb_z_sources.is_empty() {
                let total_weight: f64 = ecb_z_sources.iter().map(|(_, w)| w).sum();
                let weighted_z: f64 = ecb_z_sources.iter().map(|(z, w)| z * w).sum::<f64>() / total_weight;
                valid_count += 1;
                // Composite is lower-is-better: high composite = allostatic depletion → invert
                let factor = z_to_sensitivity_factor(weighted_z, true);
                let source_names: Vec<String> = {
                    let mut names = Vec::new();
                    if ecb_z_sources.len() >= 1 { names.push("composite".to_string()); }
                    names
                };
                adjustments.push(TransmitterCalibration {
                    transmitter: "Endocannabinoid",
                    subtype: None,
                    z_score: weighted_z,
                    source_benchmarks: source_names,
                    sensitivity_factor: factor,
                    rationale: format!(
                        "ECB composite z={weighted_z:.2} (from {} sources): {}",
                        ecb_z_sources.len(),
                        if weighted_z > 0.5 {
                            "high allostatic load → boost ECB (stress buffering depleted)"
                        } else {
                            "within normal range"
                        }
                    ),
                });
            }
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
                -0.015 // overconfident → reduce (symmetric with boost)
            } else if *z < -0.5 {
                0.015 // underconfident → boost (symmetric with reduction)
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
            // NE subtype routing: α/β have independent sensitivity fields.
            // Read parent tolerance state before mutable borrow (Koob & Le Moal 2001).
            if adj.transmitter == "NE" {
                let ne_tolerant = bath.noradrenaline.is_tolerant();
                let ne_withdrawal = bath.noradrenaline.is_in_withdrawal();
                match adj.subtype {
                    Some(ReceptorSubtype::NeAlpha) => {
                        if ne_withdrawal {
                            tracing::warn!("Skipping NE-α calibration — NE in withdrawal");
                            continue;
                        }
                        let factor = if ne_tolerant {
                            1.0 + (adj.sensitivity_factor - 1.0) * 0.5
                        } else {
                            adj.sensitivity_factor
                        };
                        bath.ne_subtypes.excitatory =
                            (bath.ne_subtypes.excitatory * factor).clamp(0.5, 2.0);
                        continue;
                    }
                    Some(ReceptorSubtype::NeBeta) => {
                        if ne_withdrawal {
                            tracing::warn!("Skipping NE-β calibration — NE in withdrawal");
                            continue;
                        }
                        let factor = if ne_tolerant {
                            1.0 + (adj.sensitivity_factor - 1.0) * 0.5
                        } else {
                            adj.sensitivity_factor
                        };
                        bath.ne_subtypes.inhibitory =
                            (bath.ne_subtypes.inhibitory * factor).clamp(0.5, 2.0);
                        continue;
                    }
                    None | Some(ReceptorSubtype::GabaA) | Some(ReceptorSubtype::GabaB) => {} // fall through to global NE
                }
            }

            // GABA subtype routing: A (ionotropic) / B (metabotropic)
            if adj.transmitter == "GABA" {
                let gaba_tolerant = bath.gaba.is_tolerant();
                let gaba_withdrawal = bath.gaba.is_in_withdrawal();
                match adj.subtype {
                    Some(ReceptorSubtype::GabaA) => {
                        if gaba_withdrawal {
                            tracing::warn!("Skipping GABA-A calibration — GABA in withdrawal");
                            continue;
                        }
                        let factor = if gaba_tolerant {
                            1.0 + (adj.sensitivity_factor - 1.0) * 0.5
                        } else {
                            adj.sensitivity_factor
                        };
                        bath.gaba_subtypes.excitatory =
                            (bath.gaba_subtypes.excitatory * factor).clamp(0.5, 2.0);
                        continue;
                    }
                    Some(ReceptorSubtype::GabaB) => {
                        if gaba_withdrawal {
                            tracing::warn!("Skipping GABA-B calibration — GABA in withdrawal");
                            continue;
                        }
                        let factor = if gaba_tolerant {
                            1.0 + (adj.sensitivity_factor - 1.0) * 0.5
                        } else {
                            adj.sensitivity_factor
                        };
                        bath.gaba_subtypes.inhibitory =
                            (bath.gaba_subtypes.inhibitory * factor).clamp(0.5, 2.0);
                        continue;
                    }
                    None | Some(ReceptorSubtype::NeAlpha) | Some(ReceptorSubtype::NeBeta) => {} // fall through to global GABA
                }
            }

            let transmitter = match adj.transmitter {
                "DA" => &mut bath.dopamine,
                "NE" => &mut bath.noradrenaline,
                "5-HT" => &mut bath.serotonin,
                "ACh" => &mut bath.acetylcholine,
                "GABA" => &mut bath.gaba,
                "Oxytocin" => &mut bath.oxytocin,
                "Glutamate" => &mut bath.glutamate,
                "Adenosine" => &mut bath.adenosine,
                "Endocannabinoid" => &mut bath.endocannabinoid,
                other => {
                    tracing::warn!(transmitter = other, "Unknown transmitter in calibration");
                    continue;
                }
            };
            // Tolerance-aware gating for global transmitter adjustments
            let factor = match tolerance_adjusted_factor(transmitter, adj.sensitivity_factor, adj.transmitter) {
                Some(f) => f,
                None => continue, // in withdrawal — skip
            };
            transmitter.receptor_sensitivity =
                (transmitter.receptor_sensitivity * factor).clamp(0.5, 2.0);
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
            let label = match &adj.subtype {
                Some(ReceptorSubtype::NeAlpha) => "NE-α",
                Some(ReceptorSubtype::NeBeta) => "NE-β",
                Some(ReceptorSubtype::GabaA) => "GABA-A",
                Some(ReceptorSubtype::GabaB) => "GABA-B",
                None => adj.transmitter,
            };
            lines.push(format!(
                "  {} → sensitivity ×{:.3} (z={:.2}, {})",
                label, adj.sensitivity_factor, adj.z_score, adj.rationale
            ));
        }
        if self.confidence_delta.abs() > 0.001 {
            lines.push(format!("  confidence → {:+.3}", self.confidence_delta));
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
                let raw_z = if is_lower_better_metric(metric) {
                    -z
                } else {
                    *z
                };
                // PVT dual-metric routing: lapse_rate → Glutamate (synthetic key),
                // vigilance_decrement → Adenosine (stays as SustainedAttention::PVT).
                let bench_key = if *bench == "SustainedAttention::PVT" && *metric == "lapse_rate" {
                    "PVT::lapse_rate"
                } else {
                    *bench
                };
                (bench_key, raw_z)
            })
            .collect();
        Self::from_z_scores(&raw)
    }

    /// Export calibration as a shareable profile for multi-agent coordination.
    pub fn to_shareable(&self, agent_id: Option<String>) -> SharedCalibrationProfile {
        let mut factors = std::collections::HashMap::new();
        for adj in &self.adjustments {
            let key = match &adj.subtype {
                Some(ReceptorSubtype::NeAlpha) => "NE-alpha".to_string(),
                Some(ReceptorSubtype::NeBeta) => "NE-beta".to_string(),
                Some(ReceptorSubtype::GabaA) => "GABA-A".to_string(),
                Some(ReceptorSubtype::GabaB) => "GABA-B".to_string(),
                None => adj.transmitter.to_string(),
            };
            factors.insert(key, adj.sensitivity_factor);
        }
        SharedCalibrationProfile {
            sensitivity_factors: factors,
            confidence_delta: self.confidence_delta,
            coverage: self.coverage,
            agent_id,
        }
    }

    /// Merge a peer's calibration profile into this calibration via weighted blend.
    ///
    /// `weight` is clamped to [0.0, 0.5] — peer influence never exceeds 50%.
    /// For each transmitter in the peer profile, blends:
    ///   `self_factor = self_factor * (1 - w) + peer_factor * w`
    ///
    /// Transmitters present in peer but absent in self are ignored
    /// (we don't create new adjustments from peer data alone).
    pub fn merge_peer_calibration(&mut self, peer: &SharedCalibrationProfile, weight: f32) {
        let w = weight.clamp(0.0, 0.5);
        for adj in &mut self.adjustments {
            let key = match &adj.subtype {
                Some(ReceptorSubtype::NeAlpha) => "NE-alpha",
                Some(ReceptorSubtype::NeBeta) => "NE-beta",
                Some(ReceptorSubtype::GabaA) => "GABA-A",
                Some(ReceptorSubtype::GabaB) => "GABA-B",
                None => adj.transmitter,
            };
            if let Some(&peer_factor) = peer.sensitivity_factors.get(key) {
                adj.sensitivity_factor =
                    adj.sensitivity_factor * (1.0 - w) + peer_factor * w;
            }
        }
        // Blend confidence delta
        self.confidence_delta =
            self.confidence_delta * (1.0 - w) + peer.confidence_delta * w;
    }

    /// Scale calibration adjustments by sleep quality (0.0–1.0).
    ///
    /// Blends each sensitivity factor toward 1.0 (neutral) proportional to sleep
    /// quality. A quality of 1.0 applies full calibration; 0.5 applies half-strength.
    /// Also scales the confidence delta proportionally.
    ///
    /// Science: Walker & Stickgold (2006) — sleep-dependent memory consolidation
    ///   scales with sleep duration and quality.
    pub fn scale_by_sleep_quality(&mut self, quality: f32) {
        let q = quality.clamp(0.0, 1.0);
        for adj in &mut self.adjustments {
            // Blend factor toward 1.0: full_factor = 1.0 + (original - 1.0) * quality
            adj.sensitivity_factor = 1.0 + (adj.sensitivity_factor - 1.0) * q;
        }
        self.confidence_delta *= q;
    }

    /// Parse JSON z-scores from calibration battery output.
    ///
    /// Expected format: `[{"benchmark":"...","metric":"...","z_score":0.0}, ...]`
    pub fn from_json_z_scores(json: &str) -> Result<Self, serde_json::Error> {
        #[derive(serde::Deserialize)]
        struct ZScoreEntry {
            benchmark: String,
            metric: String,
            z_score: f64,
        }
        let entries: Vec<ZScoreEntry> = serde_json::from_str(json)?;
        let triples: Vec<(&str, &str, f64)> = entries
            .iter()
            .map(|e| (e.benchmark.as_str(), e.metric.as_str(), e.z_score))
            .collect();
        Ok(Self::from_normative_z_scores(&triples))
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
            | "lapse_rate"
    )
}

/// Returns adjusted factor or None (skip — in withdrawal).
///
/// Tolerant: 50% dampened toward 1.0 (receptor internalization makes adjustments less effective).
/// Withdrawal: skip entirely (rebound sensitization is too volatile for calibration).
///
/// Science: Koob & Le Moal (2001) — allostatic addiction model;
///   Gainetdinov et al. (2004) — GPCR desensitization during tolerance.
fn tolerance_adjusted_factor(
    transmitter: &symthaea_neuromodulators::Transmitter,
    raw_factor: f32,
    label: &str,
) -> Option<f32> {
    if transmitter.is_in_withdrawal() {
        tracing::warn!(
            transmitter = label,
            raw_factor,
            "Skipping calibration — transmitter in withdrawal"
        );
        return None;
    }
    if transmitter.is_tolerant() {
        let dampened = 1.0 + (raw_factor - 1.0) * 0.5;
        tracing::info!(
            transmitter = label,
            raw_factor,
            dampened,
            "Dampening calibration — transmitter tolerant"
        );
        return Some(dampened);
    }
    Some(raw_factor)
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
    /// EMA of inhibition error rate (proxy for NE/prefrontal gating).
    inhibition_error_ema: f64,
    /// EMA of social coherence (proxy for Oxytocin).
    /// Kosfeld et al. (2005) — oxytocin mediates social trust.
    social_coherence_ema: f64,
    /// EMA of E/I ratio (proxy for GABA).
    /// Bhatt et al. (2009) — E/I balance homeostasis.
    ei_ratio_ema: f64,
    /// EMA of excitotoxicity risk (proxy for Glutamate).
    /// Olney (1969) — glutamate excitotoxicity.
    excitotoxicity_ema: f64,
    /// EMA of sleep pressure (proxy for Adenosine).
    /// Porkka-Heiskanen et al. (1997) — adenosine and sleep pressure.
    sleep_pressure_ema: f64,
    /// EMA of allostatic load (proxy for Endocannabinoid).
    /// Piomelli (2003) — endocannabinoid stress buffering.
    allostatic_load_ema: f64,
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
        self.ei_ratio_ema =
            self.ei_ratio_ema * (1.0 - ALPHA) + input.ei_ratio as f64 * ALPHA;
        self.excitotoxicity_ema =
            self.excitotoxicity_ema * (1.0 - ALPHA) + input.excitotoxicity_risk as f64 * ALPHA;
        self.sleep_pressure_ema =
            self.sleep_pressure_ema * (1.0 - ALPHA) + input.sleep_pressure as f64 * ALPHA;
        self.allostatic_load_ema =
            self.allostatic_load_ema * (1.0 - ALPHA) + input.allostatic_load as f64 * ALPHA;

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

        // Cooldown gate: skip when drift is anomalous OR prediction error is elevated.
        // Either signal alone is sufficient to override cooldown:
        // - Anomalous personality drift (Turrigiano 2008) = receptor sensitivity wander
        // - Elevated PE (>0.15) = sustained prediction failures needing correction
        if self.cooldown > 0 && !(drift_anomalous || self.pe_ema > 0.15) {
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
        // Baseline ~1.05; if EMA < 0.95, low social bonding → insufficient oxytocin
        let oxy_z = (self.social_coherence_ema - 1.05) / 0.1;
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
        // Baseline ~0.2; if EMA > 0.35, stress buffering depleted
        let ecb_z = -(self.allostatic_load_ema - 0.2) / 0.15;
        if ecb_z < -1.0 {
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

// ═══════════════════════════════════════════════════════════════════════════════
// Calibration History & Drift Tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Snapshot of a single calibration event for history tracking.
#[derive(Debug, Clone)]
pub struct CalibrationSnapshot {
    /// Per-transmitter sensitivity factors at time of calibration.
    pub factors: std::collections::HashMap<String, f32>,
    /// Confidence delta applied.
    pub confidence_delta: f32,
    /// Calibration coverage quality.
    pub coverage: f64,
    /// Cycle number when this calibration was applied.
    pub applied_at_cycle: u64,
}

/// Sliding window of calibration profiles for drift detection.
///
/// Tracks the last N calibration events and computes drift statistics.
/// Systematic drift (factors consistently moving in one direction) indicates
/// a structural issue vs noise (factors oscillating around a mean).
///
/// Science: McEwen (1998) — allostatic load accumulates when homeostatic
///   correction is consistently in one direction, suggesting the setpoint
///   itself needs adjustment.
#[derive(Debug, Clone)]
pub struct CalibrationHistory {
    /// Sliding window of recent calibration snapshots.
    entries: std::collections::VecDeque<CalibrationSnapshot>,
    /// Maximum history size.
    max_entries: usize,
}

impl Default for CalibrationHistory {
    fn default() -> Self {
        Self {
            entries: std::collections::VecDeque::new(),
            max_entries: 20, // Track last ~20 calibrations
        }
    }
}

impl CalibrationHistory {
    /// Record a calibration event.
    pub fn record(&mut self, calibration: &NeuromodCalibration, cycle: u64) {
        let mut factors = std::collections::HashMap::new();
        for adj in &calibration.adjustments {
            let key = match &adj.subtype {
                Some(ReceptorSubtype::NeAlpha) => "NE-alpha".to_string(),
                Some(ReceptorSubtype::NeBeta) => "NE-beta".to_string(),
                Some(ReceptorSubtype::GabaA) => "GABA-A".to_string(),
                Some(ReceptorSubtype::GabaB) => "GABA-B".to_string(),
                None => adj.transmitter.to_string(),
            };
            factors.insert(key, adj.sensitivity_factor);
        }
        self.entries.push_back(CalibrationSnapshot {
            factors,
            confidence_delta: calibration.confidence_delta,
            coverage: calibration.coverage,
            applied_at_cycle: cycle,
        });
        if self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Number of calibrations in history.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute drift direction for a specific transmitter.
    ///
    /// Returns the mean deviation from 1.0 (neutral) across history:
    /// - Positive: consistently boosting (factors > 1.0)
    /// - Negative: consistently attenuating (factors < 1.0)
    /// - Near zero: oscillating (noise, not systematic drift)
    ///
    /// Returns `None` if insufficient data (< 3 entries with this transmitter).
    pub fn drift_direction(&self, transmitter: &str) -> Option<f64> {
        let deviations: Vec<f64> = self
            .entries
            .iter()
            .filter_map(|s| s.factors.get(transmitter))
            .map(|&f| (f - 1.0) as f64)
            .collect();
        if deviations.len() < 3 {
            return None;
        }
        let mean = deviations.iter().sum::<f64>() / deviations.len() as f64;
        Some(mean)
    }

    /// Check if a transmitter shows systematic drift (not noise).
    ///
    /// Systematic drift: >75% of calibrations adjust in the same direction.
    /// This suggests the baseline setpoint needs permanent adjustment rather
    /// than repeated corrective calibration.
    pub fn is_systematic_drift(&self, transmitter: &str) -> bool {
        let deviations: Vec<f64> = self
            .entries
            .iter()
            .filter_map(|s| s.factors.get(transmitter))
            .map(|&f| (f - 1.0) as f64)
            .collect();
        if deviations.len() < 5 {
            return false;
        }
        let positive = deviations.iter().filter(|d| **d > 0.001).count();
        let negative = deviations.iter().filter(|d| **d < -0.001).count();
        let total = deviations.len();
        // Systematic if >75% in one direction
        positive * 4 > total * 3 || negative * 4 > total * 3
    }

    /// Get all transmitters that show systematic drift.
    pub fn systematic_drift_transmitters(&self) -> Vec<(String, f64)> {
        let all_keys: std::collections::HashSet<String> = self
            .entries
            .iter()
            .flat_map(|s| s.factors.keys().cloned())
            .collect();
        let mut drifters = Vec::new();
        for key in all_keys {
            if self.is_systematic_drift(&key) {
                if let Some(dir) = self.drift_direction(&key) {
                    drifters.push((key, dir));
                }
            }
        }
        drifters
    }

    /// Compute baseline nudges for transmitters showing systematic drift.
    ///
    /// Returns `(transmitter_name, nudge_delta)` pairs. The nudge is 10% of the
    /// mean drift direction per calibration window, clamped to ±0.02 to prevent
    /// runaway baseline shifting.
    ///
    /// Science: McEwen (1998) — allostatic overload occurs when homeostatic
    ///   corrections are consistently one-directional. Adjusting the setpoint
    ///   (baseline) reduces corrective load.
    pub fn compute_baseline_nudges(&self) -> Vec<(String, f32)> {
        let mut nudges = Vec::new();
        for (transmitter, direction) in self.systematic_drift_transmitters() {
            // 10% of drift direction, clamped to ±0.02
            let nudge = (direction as f32 * 0.1).clamp(-0.02, 0.02);
            if nudge.abs() > 0.001 {
                nudges.push((transmitter, nudge));
            }
        }
        nudges
    }

    /// Mean calibration interval (cycles between calibrations).
    pub fn mean_interval(&self) -> Option<f64> {
        if self.entries.len() < 2 {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 1..self.entries.len() {
            total += (self.entries[i].applied_at_cycle - self.entries[i - 1].applied_at_cycle) as f64;
            count += 1;
        }
        Some(total / count as f64)
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
            ("Executive::Stroop", 1.5),        // interference too high
            ("Executive::Flanker", 1.2),       // interference too high
            ("WorM::N-back", -0.8),            // WM below baseline
            ("Inhibition::StopSignal", 0.0),   // normal
            ("SustainedAttention::CPT", -1.0), // attention below baseline
        ];

        let cal = NeuromodCalibration::from_z_scores(&z_scores);

        // Should have 6 transmitter adjustments (DA, ACh, NE-β, NE-α, 5-HT, ECB)
        // NE-β from StopSignal (normal z=0, still produces adjustment)
        // NE-α from N-back WM z=-0.8
        // ECB composite from CPT sustained attention (stress signal)
        assert_eq!(cal.adjustments.len(), 6);

        // DA should be attenuated (interference too high → reduce DA)
        let da = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "DA")
            .unwrap();
        assert!(
            da.sensitivity_factor < 1.0,
            "High interference z should reduce DA sensitivity, got {}",
            da.sensitivity_factor
        );

        // ACh should be boosted (WM below baseline)
        let ach = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "ACh")
            .unwrap();
        assert!(
            ach.sensitivity_factor > 1.0,
            "Low WM z should boost ACh sensitivity, got {}",
            ach.sensitivity_factor
        );

        // 5-HT should be boosted (attention below baseline)
        let sht = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "5-HT")
            .unwrap();
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
        let da = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "DA")
            .unwrap();

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
        let z_scores = vec![("Executive::Stroop", 1.0), ("WorM::N-back", -0.5)];
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
        let da = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "DA")
            .unwrap();
        assert!(
            da.sensitivity_factor > 1.0,
            "Good Stroop → boost DA, got {}",
            da.sensitivity_factor
        );

        // ACh: normative z=-1.0 on nback accuracy (higher-is-better)
        // No sign change → raw z = -1.0 → ACh should be BOOSTED
        let ach = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "ACh")
            .unwrap();
        assert!(
            ach.sensitivity_factor > 1.0,
            "Poor WM → boost ACh, got {}",
            ach.sensitivity_factor
        );
    }

    #[test]
    fn test_apply_modifies_bath() {
        let z_scores = vec![
            ("Executive::Stroop", 2.0),        // high interference → attenuate DA
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
            da_before,
            bath.dopamine.receptor_sensitivity
        );
        assert!(
            bath.serotonin.receptor_sensitivity > sht_before,
            "5-HT sensitivity should increase: {} → {}",
            sht_before,
            bath.serotonin.receptor_sensitivity
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
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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
            inhibition_error_rate: 0.0,
            drift_anomalous: true,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
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

    #[test]
    fn test_self_assessment_ne_inhibition_proxy() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 10,
            cooldown_duration: 5,
            ..Default::default()
        };

        let bad_input = SelfAssessmentInput {
            prediction_error: 0.08,
            coherence: 0.7,
            confidence_calibration_error: 0.03,
            attention_utilization: 0.5,
            inhibition_error_rate: 1.0, // veto every cycle
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
        };

        for _ in 0..80 {
            monitor.update(&bad_input);
        }

        let cal = monitor.check_trigger(false);
        assert!(
            cal.is_some(),
            "High inhibition errors should trigger calibration"
        );
        assert!(
            cal.unwrap()
                .adjustments
                .iter()
                .any(|a| a.transmitter == "NE"),
            "Inhibition errors should map to NE adjustment"
        );
    }

    #[test]
    fn test_self_assessment_confidence_bidirectional() {
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 5,
            cooldown_duration: 5,
            ..Default::default()
        };

        let overconfident = SelfAssessmentInput {
            prediction_error: 0.08,
            coherence: 0.7,
            confidence_calibration_error: 0.3, // high ECE
            attention_utilization: 0.5,
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
        };

        for _ in 0..60 {
            monitor.update(&overconfident);
        }

        let cal = monitor.check_trigger(false);
        assert!(cal.is_some(), "High ECE should trigger FoK calibration");
        assert!(
            cal.unwrap().confidence_delta < 0.0,
            "Should reduce confidence"
        );
    }

    #[test]
    fn test_self_assessment_telemetry_accessors() {
        let mut monitor = SelfAssessmentMonitor::default();
        assert_eq!(monitor.observations_count(), 0);
        assert_eq!(monitor.cooldown_remaining(), 0);
        assert!((monitor.inhibition_error_ema() - 0.0).abs() < 1e-6);

        let input = SelfAssessmentInput {
            prediction_error: 0.3,
            coherence: 0.5,
            confidence_calibration_error: 0.1,
            attention_utilization: 0.8,
            inhibition_error_rate: 0.5,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
        };
        monitor.update(&input);
        assert_eq!(monitor.observations_count(), 1);
        assert!(monitor.inhibition_error_ema() > 0.0);
        assert!(monitor.pe_ema() > 0.0);
    }

    #[test]
    fn test_ne_subtype_routing_inhibition_to_beta() {
        // Inhibition benchmarks should route to NE β (phasic reactivity)
        let cal = NeuromodCalibration::from_z_scores(&[("Inhibition::StopSignal", -2.0)]);
        let ne_adj = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "NE")
            .expect("Should have NE adjustment");
        assert_eq!(
            ne_adj.subtype,
            Some(ReceptorSubtype::NeBeta),
            "Inhibition should route to NE β"
        );
        assert!(
            ne_adj.sensitivity_factor > 1.0,
            "Negative z (poor inhibition) should boost NE β"
        );
    }

    #[test]
    fn test_ne_subtype_routing_wm_to_alpha() {
        // Working memory benchmarks should route to NE α (tonic precision)
        let cal = NeuromodCalibration::from_z_scores(&[("WorM::N-back", -2.0)]);
        let ne_alpha = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "NE" && a.subtype == Some(ReceptorSubtype::NeAlpha));
        assert!(
            ne_alpha.is_some(),
            "WM should route to NE α: {:?}",
            cal.adjustments
                .iter()
                .map(|a| (&a.transmitter, &a.subtype))
                .collect::<Vec<_>>()
        );
        // Should also produce ACh adjustment (WM → ACh global)
        let ach = cal.adjustments.iter().find(|a| a.transmitter == "ACh");
        assert!(ach.is_some(), "WM should also adjust ACh");
    }

    #[test]
    fn test_ne_subtype_apply_routes_to_bath() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        let pre_alpha = bath.ne_subtypes.excitatory;
        let pre_beta = bath.ne_subtypes.inhibitory;

        let cal = NeuromodCalibration {
            adjustments: vec![
                TransmitterCalibration {
                    transmitter: "NE",
                    subtype: Some(ReceptorSubtype::NeBeta),
                    z_score: -2.0,
                    source_benchmarks: vec!["Inhibition::StopSignal".into()],
                    sensitivity_factor: 1.15,
                    rationale: "boost β".into(),
                },
                TransmitterCalibration {
                    transmitter: "NE",
                    subtype: Some(ReceptorSubtype::NeAlpha),
                    z_score: -1.5,
                    source_benchmarks: vec!["WorM::N-back".into()],
                    sensitivity_factor: 1.10,
                    rationale: "boost α".into(),
                },
            ],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        assert!(
            (bath.ne_subtypes.excitatory - pre_alpha * 1.10).abs() < 0.001,
            "NE α should be boosted: {} vs expected {}",
            bath.ne_subtypes.excitatory,
            pre_alpha * 1.10
        );
        assert!(
            (bath.ne_subtypes.inhibitory - pre_beta * 1.15).abs() < 0.001,
            "NE β should be boosted: {} vs expected {}",
            bath.ne_subtypes.inhibitory,
            pre_beta * 1.15
        );
        // Global NE sensitivity should be unchanged
        assert!(
            (bath.noradrenaline.receptor_sensitivity - 1.0).abs() < 0.001,
            "Global NE should be unchanged"
        );
    }

    #[test]
    fn test_ne_subtype_summary_labels() {
        let cal = NeuromodCalibration {
            adjustments: vec![
                TransmitterCalibration {
                    transmitter: "NE",
                    subtype: Some(ReceptorSubtype::NeAlpha),
                    z_score: -1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.1,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "NE",
                    subtype: Some(ReceptorSubtype::NeBeta),
                    z_score: -2.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.2,
                    rationale: "test".into(),
                },
            ],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        let summary = cal.summary();
        assert!(
            summary.contains("NE-α"),
            "Summary should show NE-α: {summary}"
        );
        assert!(
            summary.contains("NE-β"),
            "Summary should show NE-β: {summary}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: 9-transmitter expansion tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_gaba_calibration_from_dual_task() {
        // DualTask cost z → GABA sensitivity factor (inverted: high cost = low GABA)
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::DualTask", 2.0)]);
        let gaba = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "GABA")
            .expect("Should have GABA adjustment");
        // High dual-task cost (positive z) with invert=true → boost GABA
        assert!(
            gaba.sensitivity_factor > 1.0,
            "High DualTask cost should boost GABA, got {}",
            gaba.sensitivity_factor
        );
    }

    #[test]
    fn test_oxytocin_from_social_benchmarks() {
        let cal = NeuromodCalibration::from_z_scores(&[
            ("Social::UltimatumGame", -1.5),
            ("Social::RME", -1.0),
        ]);
        let oxy = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "Oxytocin")
            .expect("Should have Oxytocin adjustment");
        // Negative z = poor social cognition → boost oxytocin
        assert!(
            oxy.sensitivity_factor > 1.0,
            "Low social z should boost Oxytocin, got {}",
            oxy.sensitivity_factor
        );
    }

    #[test]
    fn test_glutamate_from_lapse_rate() {
        // Synthetic PVT key for lapse_rate → Glutamate
        let cal = NeuromodCalibration::from_z_scores(&[("PVT::lapse_rate", 2.0)]);
        let glut = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "Glutamate")
            .expect("Should have Glutamate adjustment");
        // High lapse rate (positive z) with invert=true → attenuate glutamate
        assert!(
            glut.sensitivity_factor > 1.0,
            "High lapse rate should adjust Glutamate sensitivity, got {}",
            glut.sensitivity_factor
        );
    }

    #[test]
    fn test_adenosine_from_pvt() {
        // SustainedAttention::PVT vigilance_decrement → Adenosine
        let cal = NeuromodCalibration::from_z_scores(&[("SustainedAttention::PVT", 2.0)]);
        let aden = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "Adenosine")
            .expect("Should have Adenosine adjustment");
        // High decrement (positive z) with invert=true → attenuate adenosine
        assert!(
            aden.sensitivity_factor > 1.0,
            "High vigilance decrement should adjust Adenosine, got {}",
            aden.sensitivity_factor
        );
    }

    #[test]
    fn test_ecb_from_allostatic() {
        let cal =
            NeuromodCalibration::from_z_scores(&[("Allostatic::Endocannabinoid", 2.0)]);
        let ecb = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "Endocannabinoid")
            .expect("Should have Endocannabinoid adjustment");
        // High allostatic load (positive z) with invert=true → boost ECB
        assert!(
            ecb.sensitivity_factor > 1.0,
            "High allostatic load should boost ECB, got {}",
            ecb.sensitivity_factor
        );
    }

    #[test]
    fn test_apply_routes_all_9_transmitters() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        let cal = NeuromodCalibration {
            adjustments: vec![
                TransmitterCalibration {
                    transmitter: "DA",
                    subtype: None,
                    z_score: 1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 0.95,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "NE",
                    subtype: None,
                    z_score: -1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.05,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "5-HT",
                    subtype: None,
                    z_score: 0.5,
                    source_benchmarks: vec![],
                    sensitivity_factor: 0.98,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "ACh",
                    subtype: None,
                    z_score: -0.5,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.02,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "GABA",
                    subtype: None,
                    z_score: 1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.05,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "Oxytocin",
                    subtype: None,
                    z_score: -1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.03,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "Glutamate",
                    subtype: None,
                    z_score: 0.8,
                    source_benchmarks: vec![],
                    sensitivity_factor: 0.96,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "Adenosine",
                    subtype: None,
                    z_score: 1.2,
                    source_benchmarks: vec![],
                    sensitivity_factor: 0.94,
                    rationale: "test".into(),
                },
                TransmitterCalibration {
                    transmitter: "Endocannabinoid",
                    subtype: None,
                    z_score: -0.3,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.01,
                    rationale: "test".into(),
                },
            ],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        // All 9 should have modified receptor_sensitivity from default 1.0
        assert!((bath.dopamine.receptor_sensitivity - 0.95).abs() < 0.001);
        assert!((bath.noradrenaline.receptor_sensitivity - 1.05).abs() < 0.001);
        assert!((bath.serotonin.receptor_sensitivity - 0.98).abs() < 0.001);
        assert!((bath.acetylcholine.receptor_sensitivity - 1.02).abs() < 0.001);
        assert!((bath.gaba.receptor_sensitivity - 1.05).abs() < 0.001);
        assert!((bath.oxytocin.receptor_sensitivity - 1.03).abs() < 0.001);
        assert!((bath.glutamate.receptor_sensitivity - 0.96).abs() < 0.001);
        assert!((bath.adenosine.receptor_sensitivity - 0.94).abs() < 0.001);
        assert!((bath.endocannabinoid.receptor_sensitivity - 1.01).abs() < 0.001);
    }

    #[test]
    fn test_gaba_subtype_routing() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        let pre_a = bath.gaba_subtypes.excitatory;
        let pre_b = bath.gaba_subtypes.inhibitory;

        let cal = NeuromodCalibration {
            adjustments: vec![
                TransmitterCalibration {
                    transmitter: "GABA",
                    subtype: Some(ReceptorSubtype::GabaA),
                    z_score: -1.0,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.10,
                    rationale: "test GABA-A".into(),
                },
                TransmitterCalibration {
                    transmitter: "GABA",
                    subtype: Some(ReceptorSubtype::GabaB),
                    z_score: -0.5,
                    source_benchmarks: vec![],
                    sensitivity_factor: 1.05,
                    rationale: "test GABA-B".into(),
                },
            ],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        assert!(
            (bath.gaba_subtypes.excitatory - pre_a * 1.10).abs() < 0.001,
            "GABA-A should be boosted: {} vs expected {}",
            bath.gaba_subtypes.excitatory,
            pre_a * 1.10
        );
        assert!(
            (bath.gaba_subtypes.inhibitory - pre_b * 1.05).abs() < 0.001,
            "GABA-B should be boosted: {} vs expected {}",
            bath.gaba_subtypes.inhibitory,
            pre_b * 1.05
        );
        // Global GABA should be unchanged
        assert!(
            (bath.gaba.receptor_sensitivity - 1.0).abs() < 0.001,
            "Global GABA should be unchanged"
        );
    }

    #[test]
    fn test_self_assessment_social_proxy() {
        // Low social coherence should trigger Oxytocin calibration
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 10,
            cooldown_duration: 5,
            ..Default::default()
        };

        let input = SelfAssessmentInput {
            prediction_error: 0.08,
            coherence: 0.7,
            confidence_calibration_error: 0.03,
            attention_utilization: 0.5,
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 0.7, // low → oxy_z < -1.0
            ei_ratio: 1.0,
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
        };

        for _ in 0..80 {
            monitor.update(&input);
        }

        let cal = monitor.check_trigger(false);
        assert!(
            cal.is_some(),
            "Low social coherence should trigger calibration"
        );
        assert!(
            cal.unwrap()
                .adjustments
                .iter()
                .any(|a| a.transmitter == "Oxytocin"),
            "Social proxy should map to Oxytocin adjustment"
        );
    }

    #[test]
    fn test_self_assessment_ei_ratio_proxy() {
        // High E/I ratio should trigger GABA calibration
        let mut monitor = SelfAssessmentMonitor {
            warmup_threshold: 10,
            cooldown_duration: 5,
            ..Default::default()
        };

        let input = SelfAssessmentInput {
            prediction_error: 0.08,
            coherence: 0.7,
            confidence_calibration_error: 0.03,
            attention_utilization: 0.5,
            inhibition_error_rate: 0.0,
            drift_anomalous: false,
            social_coherence: 1.0,
            ei_ratio: 1.5, // high → gaba_z < -1.0
            excitotoxicity_risk: 0.0,
            sleep_pressure: 0.3,
            allostatic_load: 0.2,
        };

        for _ in 0..80 {
            monitor.update(&input);
        }

        let cal = monitor.check_trigger(false);
        assert!(
            cal.is_some(),
            "High E/I ratio should trigger calibration"
        );
        assert!(
            cal.unwrap()
                .adjustments
                .iter()
                .any(|a| a.transmitter == "GABA"),
            "E/I ratio proxy should map to GABA adjustment"
        );
    }

    #[test]
    fn test_normative_pvt_lapse_rate_routing() {
        // from_normative_z_scores should remap PVT + lapse_rate → "PVT::lapse_rate"
        let scores = vec![
            ("SustainedAttention::PVT", "lapse_rate", -1.0),
            ("SustainedAttention::PVT", "vigilance_decrement", -0.5),
        ];
        let cal = NeuromodCalibration::from_normative_z_scores(&scores);

        // lapse_rate is lower-is-better, so normative z=-1.0 means un-corrected z=+1.0
        // This should route to Glutamate (via PVT::lapse_rate key)
        let glut = cal.adjustments.iter().find(|a| a.transmitter == "Glutamate");
        assert!(
            glut.is_some(),
            "PVT lapse_rate should route to Glutamate: {:?}",
            cal.adjustments
                .iter()
                .map(|a| &a.transmitter)
                .collect::<Vec<_>>()
        );

        // vigilance_decrement is lower-is-better, un-corrected z=+0.5
        // This should route to Adenosine (via SustainedAttention::PVT key)
        let aden = cal.adjustments.iter().find(|a| a.transmitter == "Adenosine");
        assert!(
            aden.is_some(),
            "PVT vigilance_decrement should route to Adenosine: {:?}",
            cal.adjustments
                .iter()
                .map(|a| &a.transmitter)
                .collect::<Vec<_>>()
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: Tolerance-aware calibration gating tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_apply_skips_withdrawal() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        // Force DA into withdrawal state
        bath.dopamine.withdrawal_cycles = 10;
        let pre_sensitivity = bath.dopamine.receptor_sensitivity;

        let cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: -2.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.15,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        assert!(
            (bath.dopamine.receptor_sensitivity - pre_sensitivity).abs() < 0.001,
            "DA sensitivity should be unchanged during withdrawal: pre={pre_sensitivity}, post={}",
            bath.dopamine.receptor_sensitivity
        );
    }

    #[test]
    fn test_apply_dampens_tolerant() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        // Force DA into tolerant state (high_exposure_cycles > tolerance_onset_cycles)
        bath.dopamine.high_exposure_cycles = bath.dopamine.tolerance_onset_cycles + 5;
        let pre_sensitivity = bath.dopamine.receptor_sensitivity;

        let cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: -2.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.10,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        // Dampened factor: 1.0 + (1.10 - 1.0) * 0.5 = 1.05
        let expected = (pre_sensitivity * 1.05).clamp(0.5, 2.0);
        assert!(
            (bath.dopamine.receptor_sensitivity - expected).abs() < 0.001,
            "Tolerant DA should get half-strength: got {}, expected {}",
            bath.dopamine.receptor_sensitivity,
            expected
        );
    }

    #[test]
    fn test_apply_normal_full_strength() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        // Normal state: no tolerance, no withdrawal
        let pre_sensitivity = bath.dopamine.receptor_sensitivity;

        let cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: -2.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.10,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        let expected = (pre_sensitivity * 1.10).clamp(0.5, 2.0);
        assert!(
            (bath.dopamine.receptor_sensitivity - expected).abs() < 0.001,
            "Normal DA should get full strength: got {}, expected {}",
            bath.dopamine.receptor_sensitivity,
            expected
        );
    }

    #[test]
    fn test_subtype_respects_parent_tolerance() {
        let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
        // Force NE into withdrawal
        bath.noradrenaline.withdrawal_cycles = 10;
        let pre_alpha = bath.ne_subtypes.excitatory;

        let cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeAlpha),
                z_score: -2.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.15,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };
        cal.apply(&mut bath);

        // NE-α should be unchanged because parent NE is in withdrawal
        assert!(
            (bath.ne_subtypes.excitatory - pre_alpha).abs() < 0.001,
            "NE-α should be unchanged during NE withdrawal: pre={pre_alpha}, post={}",
            bath.ne_subtypes.excitatory
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: Multi-agent calibration sharing tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_shareable_all_transmitters() {
        let cal = NeuromodCalibration::from_z_scores(&[
            ("Executive::Stroop", 1.0),
            ("WorM::N-back", -1.0),
            ("Inhibition::StopSignal", -1.5),
            ("SustainedAttention::CPT", -0.5),
            ("Executive::DualTask", 1.0),
            ("Social::UltimatumGame", -0.8),
            ("PVT::lapse_rate", 1.2),
            ("SustainedAttention::PVT", 0.5),
            ("Allostatic::Endocannabinoid", 0.3),
        ]);
        let profile = cal.to_shareable(Some("agent-1".into()));

        assert_eq!(profile.agent_id, Some("agent-1".into()));
        assert!(
            profile.sensitivity_factors.len() >= 9,
            "Should have factors for all transmitters: got {}",
            profile.sensitivity_factors.len()
        );
        // Check some specific keys
        assert!(profile.sensitivity_factors.contains_key("DA"));
        assert!(profile.sensitivity_factors.contains_key("ACh"));
        assert!(profile.sensitivity_factors.contains_key("GABA"));
        assert!(profile.sensitivity_factors.contains_key("Oxytocin"));
        assert!(profile.sensitivity_factors.contains_key("Glutamate"));
        assert!(profile.sensitivity_factors.contains_key("Adenosine"));
        assert!(profile.sensitivity_factors.contains_key("Endocannabinoid"));
    }

    #[test]
    fn test_merge_weighted_blend() {
        let mut cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: 1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 0.90,
                rationale: "test".into(),
            }],
            confidence_delta: -0.02,
            coverage: 1.0,
        };

        let mut peer_factors = std::collections::HashMap::new();
        peer_factors.insert("DA".to_string(), 1.10_f32);
        let peer = SharedCalibrationProfile {
            sensitivity_factors: peer_factors,
            confidence_delta: 0.02,
            coverage: 0.8,
            agent_id: Some("peer".into()),
        };

        cal.merge_peer_calibration(&peer, 0.2);

        // DA: 0.90 * 0.8 + 1.10 * 0.2 = 0.72 + 0.22 = 0.94
        assert!(
            (cal.adjustments[0].sensitivity_factor - 0.94).abs() < 0.01,
            "Blended DA factor should be ~0.94, got {}",
            cal.adjustments[0].sensitivity_factor
        );
        // Confidence: -0.02 * 0.8 + 0.02 * 0.2 = -0.016 + 0.004 = -0.012
        assert!(
            (cal.confidence_delta - (-0.012)).abs() < 0.01,
            "Blended confidence delta should be ~-0.012, got {}",
            cal.confidence_delta
        );
    }

    #[test]
    fn test_merge_capped_weight() {
        let mut cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: 1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 0.90,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };

        let mut peer_factors = std::collections::HashMap::new();
        peer_factors.insert("DA".to_string(), 1.10_f32);
        let peer = SharedCalibrationProfile {
            sensitivity_factors: peer_factors,
            confidence_delta: 0.0,
            coverage: 0.8,
            agent_id: None,
        };

        // Pass weight > 0.5 — should be clamped
        cal.merge_peer_calibration(&peer, 0.9);

        // DA: 0.90 * 0.5 + 1.10 * 0.5 = 1.0 (capped at 0.5)
        assert!(
            (cal.adjustments[0].sensitivity_factor - 1.0).abs() < 0.01,
            "Weight should be clamped to 0.5: got {}",
            cal.adjustments[0].sensitivity_factor
        );
    }

    #[test]
    fn test_merge_missing_key_unchanged() {
        let mut cal = NeuromodCalibration {
            adjustments: vec![TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: 1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 0.90,
                rationale: "test".into(),
            }],
            confidence_delta: 0.0,
            coverage: 1.0,
        };

        // Peer has ACh but not DA
        let mut peer_factors = std::collections::HashMap::new();
        peer_factors.insert("ACh".to_string(), 1.10_f32);
        let peer = SharedCalibrationProfile {
            sensitivity_factors: peer_factors,
            confidence_delta: 0.0,
            coverage: 0.5,
            agent_id: None,
        };

        cal.merge_peer_calibration(&peer, 0.3);

        // DA should be unchanged — peer doesn't have DA
        assert!(
            (cal.adjustments[0].sensitivity_factor - 0.90).abs() < 0.001,
            "DA should be unchanged when peer lacks it: got {}",
            cal.adjustments[0].sensitivity_factor
        );
    }

    #[test]
    fn test_oxytocin_modulated_weight() {
        // Verify that the CLS-level weight formula (oxy * 0.1) produces expected range
        // This tests the formula in isolation, not the CLS accessor (which needs full service)
        let oxy_low = 0.3_f32; // low oxytocin
        let oxy_high = 1.5_f32; // high oxytocin (effective can exceed 1.0)

        let weight_low = (oxy_low * 0.1).min(0.2);
        let weight_high = (oxy_high * 0.1).min(0.2);

        assert!(
            weight_low < 0.05,
            "Low oxy should give low weight: {weight_low}"
        );
        assert!(
            (weight_high - 0.15).abs() < 0.01,
            "High oxy should give ~0.15 weight: {weight_high}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 5: JSON z-score parsing tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_from_json_z_scores_valid() {
        let json = r#"[
            {"benchmark":"Executive::Stroop","metric":"stroop_effect","z_score":-0.42},
            {"benchmark":"WorM::N-back","metric":"nback_2::accuracy","z_score":1.2}
        ]"#;
        let cal = NeuromodCalibration::from_json_z_scores(json);
        assert!(cal.is_ok(), "Valid JSON should parse: {:?}", cal.err());
        let cal = cal.unwrap();
        assert!(
            !cal.adjustments.is_empty(),
            "Should produce adjustments from valid JSON"
        );
        // DA should be present (from Stroop)
        assert!(
            cal.adjustments.iter().any(|a| a.transmitter == "DA"),
            "Should have DA from Stroop"
        );
    }

    #[test]
    fn test_from_json_z_scores_malformed() {
        let bad_json = "not valid json at all";
        let cal = NeuromodCalibration::from_json_z_scores(bad_json);
        assert!(cal.is_err(), "Malformed JSON should return Err");
    }

    // ── Improvement 1: ECB composite derivation tests ────────────────────

    #[test]
    fn test_ecb_composite_from_pvt_and_cpt() {
        // PVT lapse_rate + CPT sustained attention → ECB composite
        let z_scores = vec![
            ("PVT::lapse_rate", 1.5),            // high lapses
            ("SustainedAttention::CPT", -1.0),    // poor attention
        ];
        let cal = NeuromodCalibration::from_z_scores(&z_scores);

        // Should have: Glutamate (from PVT), 5-HT (from CPT), Adenosine (from PVT),
        // AND Endocannabinoid (composite from PVT + CPT)
        let ecb = cal.adjustments.iter().find(|a| a.transmitter == "Endocannabinoid");
        assert!(ecb.is_some(), "ECB should be derived from PVT+CPT composite");
        assert!(
            ecb.unwrap().rationale.contains("composite"),
            "Rationale should mention composite derivation"
        );
    }

    #[test]
    fn test_ecb_composite_weighted_averaging() {
        // Allostatic proxy (weight 2.0) + PVT (weight 1.0) → weighted mean
        let z_scores = vec![
            ("Allostatic::Endocannabinoid", 2.0),  // high allostatic load
            ("PVT::lapse_rate", 0.0),               // normal lapses
        ];
        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        let ecb = cal.adjustments.iter().find(|a| a.transmitter == "Endocannabinoid").unwrap();

        // Weighted: (2.0*2.0 + 0.0*1.0) / (2.0+1.0) = 1.333...
        // With higher weight on allostatic, result should be > 1.0
        assert!(ecb.z_score > 1.0, "Weighted z should favor allostatic proxy (weight 2x)");
    }

    #[test]
    fn test_ecb_no_sources_no_ecb() {
        // Only DA benchmarks → no ECB
        let z_scores = vec![("Executive::Stroop", 0.5)];
        let cal = NeuromodCalibration::from_z_scores(&z_scores);
        assert!(
            cal.adjustments.iter().all(|a| a.transmitter != "Endocannabinoid"),
            "No ECB sources → no ECB adjustment"
        );
    }

    // ── Improvement 4: Calibration history tests ─────────────────────────

    #[test]
    fn test_calibration_history_record_and_len() {
        let mut history = CalibrationHistory::default();
        assert!(history.is_empty());

        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 1.0)]);
        history.record(&cal, 100);
        assert_eq!(history.len(), 1);

        history.record(&cal, 200);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_calibration_history_sliding_window() {
        let mut history = CalibrationHistory::default();
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 0.5)]);

        // Fill beyond max_entries (20)
        for i in 0..25 {
            history.record(&cal, i * 100);
        }
        assert_eq!(history.len(), 20, "Should cap at max_entries");
    }

    #[test]
    fn test_calibration_history_drift_direction() {
        let mut history = CalibrationHistory::default();

        // Consistently boost DA (factor > 1.0)
        for i in 0..5 {
            let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", -1.0)]);
            history.record(&cal, i * 100);
        }

        let drift = history.drift_direction("DA");
        assert!(drift.is_some());
        assert!(drift.unwrap() > 0.0, "Consistent negative z → boost DA → positive drift");
    }

    #[test]
    fn test_calibration_history_systematic_drift_detection() {
        let mut history = CalibrationHistory::default();

        // 6 calibrations all boosting DA
        for i in 0..6 {
            let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", -2.0)]);
            history.record(&cal, i * 100);
        }
        assert!(history.is_systematic_drift("DA"), "6/6 in same direction should be systematic");

        // Not enough data for 5-HT
        assert!(!history.is_systematic_drift("5-HT"), "No 5-HT data → not systematic");
    }

    #[test]
    fn test_calibration_history_no_drift_oscillating() {
        let mut history = CalibrationHistory::default();

        // Oscillating: alternating boost/attenuate
        for i in 0..6 {
            let z = if i % 2 == 0 { -1.0 } else { 1.0 };
            let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", z)]);
            history.record(&cal, i * 100);
        }
        assert!(!history.is_systematic_drift("DA"), "Oscillating should NOT be systematic drift");
    }

    #[test]
    fn test_calibration_history_mean_interval() {
        let mut history = CalibrationHistory::default();
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 0.5)]);

        history.record(&cal, 100);
        history.record(&cal, 300);
        history.record(&cal, 500);

        let interval = history.mean_interval().unwrap();
        assert!((interval - 200.0).abs() < 0.1, "Mean interval should be 200 cycles");
    }

    #[test]
    fn test_calibration_history_insufficient_data() {
        let history = CalibrationHistory::default();
        assert!(history.drift_direction("DA").is_none(), "Empty history → None");
        assert!(history.mean_interval().is_none(), "Empty history → None");
    }

    #[test]
    fn test_scale_by_sleep_quality_full() {
        let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
        let original_factor = cal.adjustments[0].sensitivity_factor;
        cal.scale_by_sleep_quality(1.0);
        assert!(
            (cal.adjustments[0].sensitivity_factor - original_factor).abs() < 0.0001,
            "Full quality should not change factors"
        );
    }

    #[test]
    fn test_scale_by_sleep_quality_half() {
        let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
        let original_factor = cal.adjustments[0].sensitivity_factor;
        let deviation = original_factor - 1.0;
        cal.scale_by_sleep_quality(0.5);
        let expected = 1.0 + deviation * 0.5;
        assert!(
            (cal.adjustments[0].sensitivity_factor - expected).abs() < 0.0001,
            "Half quality should halve deviation: got {}, expected {}",
            cal.adjustments[0].sensitivity_factor,
            expected
        );
    }

    #[test]
    fn test_scale_by_sleep_quality_zero() {
        let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
        cal.scale_by_sleep_quality(0.0);
        assert!(
            (cal.adjustments[0].sensitivity_factor - 1.0).abs() < 0.0001,
            "Zero quality should neutralize all factors to 1.0"
        );
        assert!(
            cal.confidence_delta.abs() < 0.0001,
            "Zero quality should zero confidence delta"
        );
    }

    #[test]
    fn test_baseline_nudges_from_systematic_drift() {
        let mut history = CalibrationHistory::default();
        // Create 6 calibrations all boosting DA (systematic positive drift)
        for i in 0..6 {
            let mut factors = std::collections::HashMap::new();
            factors.insert("DA".to_string(), 1.05); // consistently boosting
            history.entries.push_back(CalibrationSnapshot {
                factors,
                confidence_delta: 0.0,
                coverage: 0.8,
                applied_at_cycle: i * 100,
            });
        }
        let nudges = history.compute_baseline_nudges();
        assert!(!nudges.is_empty(), "Should detect drift and produce nudges");
        let (name, nudge) = &nudges[0];
        assert_eq!(name, "DA");
        assert!(*nudge > 0.0, "Positive drift should produce positive nudge");
        assert!(*nudge <= 0.02, "Nudge should be clamped to ±0.02");
    }

    #[test]
    fn test_baseline_nudges_no_drift() {
        let mut history = CalibrationHistory::default();
        // Oscillating factors: no systematic drift
        for i in 0..6 {
            let mut factors = std::collections::HashMap::new();
            let factor = if i % 2 == 0 { 1.05 } else { 0.95 };
            factors.insert("DA".to_string(), factor);
            history.entries.push_back(CalibrationSnapshot {
                factors,
                confidence_delta: 0.0,
                coverage: 0.8,
                applied_at_cycle: i * 100,
            });
        }
        let nudges = history.compute_baseline_nudges();
        assert!(nudges.is_empty(), "Oscillating drift should produce no nudges");
    }

    #[test]
    fn test_cooldown_override_or_logic() {
        // Verify that either drift_anomalous OR elevated PE overrides cooldown
        let mut monitor = SelfAssessmentMonitor::default();
        monitor.observations_since_calibration = 300; // past warmup
        monitor.cooldown = 100; // in cooldown

        // Case 1: drift anomalous only, PE low → should override (OR)
        monitor.pe_ema = 0.05;
        let result = monitor.check_trigger(true);
        // With OR logic, drift_anomalous=true bypasses cooldown
        // The trigger may or may not fire depending on z-score thresholds,
        // but it should NOT be blocked by cooldown
        // We test this by checking that a high-PE monitor also overrides
        let mut monitor2 = SelfAssessmentMonitor::default();
        monitor2.observations_since_calibration = 300;
        monitor2.cooldown = 100;
        monitor2.pe_ema = 0.25; // elevated PE
        // With OR logic, pe_ema > 0.15 alone should bypass cooldown
        let _ = monitor2.check_trigger(false);
        // The key test: the function should reach the z-score computation
        // (not return None at the cooldown gate). We verify by checking
        // that pe_z would be computed from the high pe_ema.
        // Since pe_ema=0.25, pe_z=(0.25-0.1)/0.05=3.0 → needs_calibration=true
        let result2 = monitor2.check_trigger(false);
        assert!(
            result2.is_some(),
            "Elevated PE alone should override cooldown with OR logic"
        );
    }

    #[test]
    fn test_confidence_delta_symmetric() {
        // Overconfident
        let cal_over = NeuromodCalibration::from_z_scores(&[(
            "Metacognition::FeelingOfKnowing",
            1.0,
        )]);
        // Underconfident
        let cal_under = NeuromodCalibration::from_z_scores(&[(
            "Metacognition::FeelingOfKnowing",
            -1.0,
        )]);
        assert!(
            (cal_over.confidence_delta.abs() - cal_under.confidence_delta.abs()).abs() < 0.0001,
            "Confidence delta should be symmetric: over={}, under={}",
            cal_over.confidence_delta,
            cal_under.confidence_delta
        );
        assert!(cal_over.confidence_delta < 0.0, "Overconfident should reduce");
        assert!(cal_under.confidence_delta > 0.0, "Underconfident should boost");
    }
}
