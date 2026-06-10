// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psych-bench z-score → neuromodulator sensitivity adjustment pipeline.
//!
//! Translates normative z-scores into `NeuromodCalibration` structs
//! that can be applied to a `NeuromodulatorBath`.

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
    /// Sacred Stillness quality: system's capacity for generative rest (0.0–1.0).
    ///
    /// Derived from PVT vigilance decrement and GABA/adenosine calibration:
    /// low vigilance decrement + healthy inhibitory tone = high stillness quality.
    /// Science: Tononi & Cirelli (2006) — synaptic homeostasis; rest quality
    /// correlates with GABA-mediated inhibitory balance.
    pub stillness_quality: f32,
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
            if let Some((_, z)) = z_scores
                .iter()
                .find(|(name, _)| *name == "Allostatic::Endocannabinoid")
            {
                ecb_z_sources.push((*z, 2.0));
            }

            // PVT lapse_rate as stress fatigue signal (weight 1.0)
            if let Some((_, z)) = z_scores.iter().find(|(name, _)| *name == "PVT::lapse_rate") {
                ecb_z_sources.push((*z, 1.0));
            }

            // CPT sustained attention as chronic stress signal (weight 1.0)
            if let Some((_, z)) = z_scores
                .iter()
                .find(|(name, _)| *name == "SustainedAttention::CPT")
            {
                ecb_z_sources.push((*z, 1.0));
            }

            if !ecb_z_sources.is_empty() {
                let total_weight: f64 = ecb_z_sources.iter().map(|(_, w)| w).sum();
                let weighted_z: f64 =
                    ecb_z_sources.iter().map(|(z, w)| z * w).sum::<f64>() / total_weight;
                valid_count += 1;
                // Composite is lower-is-better: high composite = allostatic depletion → invert
                let factor = z_to_sensitivity_factor(weighted_z, true);
                let source_names: Vec<String> = {
                    let mut names = Vec::new();
                    if !ecb_z_sources.is_empty() {
                        names.push("composite".to_string());
                    }
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

        // Sacred Stillness quality: composite from GABA and adenosine calibration z-scores.
        // Low vigilance decrement (negative adenosine z) + good inhibitory tone (negative GABA z)
        // = high stillness quality. Both inverted: negative z = healthier rest capacity.
        let stillness_quality = {
            let gaba_z_opt = adjustments
                .iter()
                .find(|a| a.transmitter == "GABA")
                .map(|a| a.z_score);
            let aden_z_opt = adjustments
                .iter()
                .find(|a| a.transmitter == "Adenosine")
                .map(|a| a.z_score);
            match (gaba_z_opt, aden_z_opt) {
                (Some(g), Some(a)) => {
                    // Both negative z = healthy → quality approaches 1.0
                    let composite = -(g * 0.5 + a * 0.5); // negate: negative z = good
                    (0.5 + composite * 0.25).clamp(0.0, 1.0) as f32
                }
                (Some(z), None) | (None, Some(z)) => (0.5 - z * 0.25).clamp(0.0, 1.0) as f32,
                (None, None) => 0.5, // no data → neutral
            }
        };

        let total_benchmarks = z_scores.len().max(1);
        NeuromodCalibration {
            adjustments,
            confidence_delta,
            coverage: valid_count as f64 / total_benchmarks as f64,
            stillness_quality,
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
                    None => {} // fall through to global NE handler
                    _ => {
                        tracing::warn!(
                            subtype = ?adj.subtype,
                            "Invalid subtype for NE transmitter — skipping"
                        );
                        continue;
                    }
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
                    None => {} // fall through to global GABA handler
                    _ => {
                        tracing::warn!(
                            subtype = ?adj.subtype,
                            "Invalid subtype for GABA transmitter — skipping"
                        );
                        continue;
                    }
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
            let factor = match tolerance_adjusted_factor(
                transmitter,
                adj.sensitivity_factor,
                adj.transmitter,
            ) {
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
                adj.sensitivity_factor = adj.sensitivity_factor * (1.0 - w) + peer_factor * w;
            }
        }
        // Blend confidence delta
        self.confidence_delta = self.confidence_delta * (1.0 - w) + peer.confidence_delta * w;
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
pub(crate) fn is_lower_better_metric(metric: &str) -> bool {
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
pub(crate) fn z_to_sensitivity_factor(z: f64, invert: bool) -> f32 {
    // NaN/infinity guard: non-finite z-scores produce neutral factor (1.0)
    // to prevent permanent bath corruption from upstream normalization errors.
    if !z.is_finite() {
        return 1.0;
    }
    let z = if invert { -z } else { z };
    let raw = 1.0 - z * 0.05;
    raw.clamp(0.85, 1.15) as f32
}
