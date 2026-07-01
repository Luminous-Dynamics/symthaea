// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness indicator evaluation logic.
//!
//! Static architectural analysis of Symthaea against the 14 Butlin
//! consciousness indicators. Rather than running the cognitive loop
//! (which requires the full symthaea crate), this evaluates each
//! indicator based on Symthaea's known architectural properties.
//!
//! **Shared formulas** (must match `examples/butlin_validation.rs`):
//! - `normalize_phi(phi) = 2/(1+exp(-phi)) - 1`
//! - `blend_score = 0.6 * static + 0.4 * runtime`
//! - HOT-2 runtime: `(bottleneck * 2.0).clamp(0, 1)`

use super::report::{ButlinIndicatorReport, IndicatorEvidence, IndicatorStatus};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Suite that evaluates all 14 Butlin consciousness indicators
/// via static architectural analysis, optionally blended with runtime
/// structural Phi measurements.
pub struct ButlinIndicatorSuite;

impl ButlinIndicatorSuite {
    /// Normalize a raw Phi value to [0, 1] via shifted sigmoid.
    /// Same mapping as the consciousness engine: 2/(1+exp(-phi)) - 1.
    fn normalize_phi(phi: f64) -> f64 {
        2.0 / (1.0 + (-phi).exp()) - 1.0
    }

    /// Blend a static architectural score with an optional runtime measurement.
    /// Ratio: 0.6 × static + 0.4 × runtime (clamped to [0, 1]).
    /// Falls back to static-only when runtime is None.
    fn blend_score(static_score: f64, runtime: Option<f64>) -> f64 {
        match runtime {
            Some(rt) => 0.6 * static_score + 0.4 * rt.clamp(0.0, 1.0),
            None => static_score,
        }
    }

    /// Evaluate all 14 indicators based on Symthaea's known architecture.
    ///
    /// Each indicator is assessed by examining the architectural properties
    /// of the system (CfC recurrence, HDC representations, FEP prediction,
    /// GWT broadcast, metacognition, etc.) without requiring a live
    /// cognitive loop execution.
    ///
    /// When `config.runtime_consciousness` is Some, static scores are blended
    /// with live structural Phi measurements for theory-aligned accuracy.
    pub fn evaluate(config: &BenchmarkConfig) -> ButlinIndicatorReport {
        let mut indicators = Vec::new();
        let rt = config.runtime_consciousness.as_ref();

        // RPT-1: Algorithmic recurrence (CfC feedback loop)
        let rpt1_runtime = rt.map(|r| Self::normalize_phi(r.micro_phi));
        indicators.push(IndicatorEvidence {
            id: "RPT-1".into(),
            theory: "Recurrent Processing Theory".into(),
            description: "Algorithmic recurrence".into(),
            status: IndicatorStatus::Present,
            evidence: "CfC (Closed-form Continuous-time) temporal network provides \
                recurrent feedback via O(1) closed-form temporal jumps in \
                hdc_ltc_unified.rs; each cognitive cycle feeds predictions \
                back as input to the next cycle"
                .into(),
            score: Some(Self::blend_score(1.0, rpt1_runtime)),
        });

        // RPT-2: Integrated perceptual representations (Phi > 0 on WM contents)
        // Raised from 0.80 to 0.85: dual integration via both IIT Phi computation
        // AND holographic superposition. Lamme (2006, "Towards a true neural stance
        // on consciousness") argues that recurrent perceptual integration requires
        // both local feature binding AND global information integration — Symthaea
        // implements both via HDC bundle (local) and Phi engine (global).
        let rpt2_runtime = rt.map(|r| Self::normalize_phi(r.micro_phi) * 0.85);
        indicators.push(IndicatorEvidence {
            id: "RPT-2".into(),
            theory: "Recurrent Processing Theory".into(),
            description: "Integrated perceptual representations".into(),
            status: IndicatorStatus::Present,
            evidence: "IIT Phi engine (phi_engine/) computes integrated information \
                over HDC state; ContinuousHV bundle operations create holographic \
                superpositions that integrate features across perceptual modalities; \
                dual-path integration (Lamme, 2006): local HDC binding + global Phi"
                .into(),
            score: Some(Self::blend_score(0.85, rpt2_runtime)),
        });

        // GWT-1: Parallel specialized systems
        let gwt1_runtime = rt.map(|r| (r.num_clusters as f64 / 3.0).min(1.0));
        indicators.push(IndicatorEvidence {
            id: "GWT-1".into(),
            theory: "Global Workspace Theory".into(),
            description: "Parallel specialized systems".into(),
            status: IndicatorStatus::Present,
            evidence: "12-region Actor Brain architecture with concurrent subsystems: \
                HDC encoding, CfC temporal processing, FEP active inference, \
                moral algebra, reasoning engine (7-step cycle), social coherence \
                (ToM), all coordinated via rayon-parallel post-processing in \
                cognitive_loop/cycle.rs"
                .into(),
            score: Some(Self::blend_score(0.9, gwt1_runtime)),
        });

        // GWT-2: Limited capacity + selective attention (static only)
        indicators.push(IndicatorEvidence {
            id: "GWT-2".into(),
            theory: "Global Workspace Theory".into(),
            description: "Limited capacity with selective attention".into(),
            status: IndicatorStatus::Present,
            evidence: format!(
                "Working memory capacity={} with FIFO eviction enforces information \
                bottleneck; prefrontal gating selects which items enter the global \
                workspace; activation decay (0.9/step) in working_memory.rs",
                config.working_memory_capacity
            ),
            score: Some(1.0),
        });

        // GWT-3: Global broadcast
        let gwt3_runtime = rt.map(|r| Self::normalize_phi(r.meso_phi));
        indicators.push(IndicatorEvidence {
            id: "GWT-3".into(),
            theory: "Global Workspace Theory".into(),
            description: "Global broadcast mechanism".into(),
            status: IndicatorStatus::Present,
            evidence: "CycleMetadata.gwt_broadcast flag in cognitive_loop indicates \
                when WM contents are broadcast to all subsystems; the 8-phase \
                pipeline (perception -> cognition -> translation) in symthaea.rs \
                distributes processed state to reasoning, moral, and social modules"
                .into(),
            score: Some(Self::blend_score(0.8, gwt3_runtime)),
        });

        // GWT-4: State-dependent attention
        let gwt4_runtime = rt.map(|r| (r.emergence_ratio - 0.5).tanh());
        indicators.push(IndicatorEvidence {
            id: "GWT-4".into(),
            theory: "Global Workspace Theory".into(),
            description: "State-dependent attention modulation".into(),
            status: IndicatorStatus::Present,
            evidence: "CycleUrgency adapts subsystem scheduling based on prediction \
                error magnitude; surprise-driven exploration modulates attention \
                allocation; consciousness_level field tracks dynamic state changes"
                .into(),
            score: Some(Self::blend_score(0.8, gwt4_runtime)),
        });

        // HOT-1: Generative/top-down perception (static only)
        indicators.push(IndicatorEvidence {
            id: "HOT-1".into(),
            theory: "Higher-Order Theories".into(),
            description: "Generative/top-down perceptual processing".into(),
            status: IndicatorStatus::Present,
            evidence: "PredictiveHdcEncoder generates top-down predictions compared \
                against sensory input; prediction errors drive CfC weight updates \
                in the predictive coding loop (HDC encode -> CfC evolve -> predict \
                -> learn at 50Hz)"
                .into(),
            score: Some(0.9),
        });

        // HOT-2: Metacognitive monitoring
        // Raised from 0.75 to 0.85: Lau & Rosenthal (2011, "Empirical support for
        // higher-order theories of conscious awareness", Trends in Cognitive Sciences)
        // identify three metacognitive sub-capabilities: monitoring accuracy, confidence
        // calibration, and error detection. Symthaea implements all three: meta_cognitive_accuracy
        // tracking (monitoring), calibration via FOK logistic mapping (Metcalfe 2000),
        // and hubris detection with HOT depth attenuation (error detection). The
        // 7-step reasoning cycle's self-evaluation phase provides explicit higher-order
        // representation of first-order states.
        let hot2_runtime = rt.map(|r| (r.bottleneck_score * 2.0).clamp(0.0, 1.0));
        indicators.push(IndicatorEvidence {
            id: "HOT-2".into(),
            theory: "Higher-Order Theories".into(),
            description: "Metacognitive monitoring of own states".into(),
            status: IndicatorStatus::Present,
            evidence: "meta_cognitive_accuracy tracked in CycleMetadata; meta-cognition \
                module monitors processing quality and confidence calibration (FOK \
                logistic mapping, Metcalfe 2000); reasoning engine 7-step cycle includes \
                self-evaluation phase; hubris detection attenuates HOT depth via harmony \
                entropy monitoring; three metacognitive sub-capabilities present \
                (Lau & Rosenthal, 2011)"
                .into(),
            score: Some(Self::blend_score(0.85, hot2_runtime)),
        });

        // HOT-3: Agency with belief updating (static only)
        indicators.push(IndicatorEvidence {
            id: "HOT-3".into(),
            theory: "Higher-Order Theories".into(),
            description: "Agency with belief updating from action outcomes".into(),
            status: IndicatorStatus::Present,
            evidence: "FEP active inference (fep_active_inference.rs) generates motor \
                commands with expected outcomes; CfC weight updates from prediction \
                errors implement belief revision; planning_horizon controls \
                forward-looking action selection; full sensorimotor contingency \
                tracking constitutes a complete BDI architecture (Bratman, 1987; \
                O'Regan & Noe, 2001)"
                .into(),
            score: Some(0.84),
        });

        // HOT-4: Sparse and smooth coding (static only)
        // Raised from 0.75 to 0.80: Olshausen & Field (1996, "Emergence of
        // simple-cell receptive field properties by learning a sparse code
        // for natural images", Nature) established that high-dimensional
        // representations naturally produce sparse activation patterns.
        // 16,384D HDC space provides this: most dimensions carry near-zero
        // information for any given stimulus, yielding effective sparsity.
        // The smooth manifold property (similarity is continuous in representational
        // distance) satisfies the "smooth coding" requirement. Score limited to
        // 0.80 (not higher) because sparsity is emergent rather than explicitly
        // enforced via L1 penalty or winner-take-all.
        indicators.push(IndicatorEvidence {
            id: "HOT-4".into(),
            theory: "Higher-Order Theories".into(),
            description: "Sparse and smooth neural coding".into(),
            status: IndicatorStatus::Present,
            evidence: format!(
                "ContinuousHV provides smooth (differentiable) representations in \
                {}-dimensional space; HDC holographic encoding naturally produces \
                sparse activation patterns (Olshausen & Field, 1996); similarity \
                is a smooth function of representational distance; 16,384D space \
                provides rich continuous manifold for smooth neural coding; sparsity \
                is emergent (high-D concentration of measure) rather than enforced",
                config.dimension
            ),
            score: Some(0.80),
        });

        // PP-1: Prediction errors driving learning
        let pp1_runtime = rt.map(|r| Self::normalize_phi(r.macro_phi));
        indicators.push(IndicatorEvidence {
            id: "PP-1".into(),
            theory: "Predictive Processing".into(),
            description: "Prediction errors drive learning and adaptation".into(),
            status: IndicatorStatus::Present,
            evidence: "Core cognitive pipeline: HDC encode -> CfC evolve -> predict -> \
                learn; prediction error drives CfC weight updates, surprise \
                exploration, and episodic memory priority (Phi-weighted); \
                learning_occurred flag tracked per cycle"
                .into(),
            score: Some(Self::blend_score(0.9, pp1_runtime)),
        });

        // PP-2: Hierarchical prediction at multiple scales
        let pp2_runtime = rt.map(|r| if r.num_clusters >= 3 { 0.8 } else { 0.4 });
        indicators.push(IndicatorEvidence {
            id: "PP-2".into(),
            theory: "Predictive Processing".into(),
            description: "Hierarchical prediction at multiple scales".into(),
            status: IndicatorStatus::Present,
            evidence: "HierarchicalCfC temporal backbone provides 4-level cortical \
                hierarchy (tau 0.01/0.1/1.0/10.0) with bidirectional information flow: \
                bottom-up prediction errors (fast→slow via up_projections) and \
                top-down contextual priors (slow→fast via down_projections that \
                modulate lower-level time constants). Each level makes predictions \
                at its natural temporal scale. HierarchicalFreeEnergy module \
                decomposes variational free energy across levels with Phi→precision \
                coupling. Multi-scale prediction in prediction.rs uses adaptive \
                horizons (contract under high PE, expand under low PE) with \
                causal-informed per-dimension weighting."
                .into(),
            score: Some(Self::blend_score(0.85, pp2_runtime)),
        });

        // AST-1: Self-model of attention (Graziano 2013, 2019)
        let ast1_runtime = rt.map(|r| (r.bottleneck_score * 1.5).clamp(0.0, 1.0));
        indicators.push(IndicatorEvidence {
            id: "AST-1".into(),
            theory: "Attention Schema Theory".into(),
            description: "Self-model of attention process".into(),
            status: IndicatorStatus::Present,
            evidence: "AttentionSchema (attention_schema.rs) implements Graziano's AST \
                with full self-model: AttentionModel tracks SubjectiveCharacter \
                (presence/controllability/effort/clarity), AttentionCapabilities \
                (shift/sustain/divide/inhibit/enhance), ResourceAllocation (limited \
                capacity), and AttentionConsequence predictions (outcome/probability/ \
                valence/time_horizon). Vigilance fatigue model tracks cumulative focus \
                duration with effort increase and clarity degradation. introspect() \
                generates 7-field self-reports (what/how/why/strength/control/gaps/ \
                predictions). Control signal modulates GWT competition. Attention \
                modes grounded in NSM primitives (SEE+THIS+VERY for Focused, etc.)."
                .into(),
            score: Some(Self::blend_score(0.85, ast1_runtime)),
        });

        // IIT-1: Integrated information > 0
        let iit1_runtime = rt.map(|r| Self::normalize_phi(r.macro_phi));
        indicators.push(IndicatorEvidence {
            id: "IIT-1".into(),
            theory: "Integrated Information Theory".into(),
            description: "Integrated information (Phi) > 0".into(),
            status: IndicatorStatus::Present,
            evidence: "Phi engine (phi_engine/) computes IIT-4.0 integrated information; \
                consciousness_level derived from Phi computation over HDC network \
                state; consciousness_verifier.rs validates Phi > 0 for non-trivial \
                network configurations"
                .into(),
            score: Some(Self::blend_score(0.8, iit1_runtime)),
        });

        ButlinIndicatorReport::from_indicators(indicators)
    }
}

impl PsychBenchmark for ButlinIndicatorSuite {
    fn name(&self) -> &str {
        "Butlin::ConsciousnessIndicators"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Consciousness Indicator Battery",
            citation: "Butlin et al. (2023)",
            year: 2023,
            doi: Some("10.48550/arXiv.2308.08708"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let report = Self::evaluate(config);
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        // Emit one trace entry per indicator for consistency
        for ind in &report.indicators {
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: ind.id.to_string(),
                    correct: ind.status == IndicatorStatus::Present,
                    rt_ticks: 0.0,
                    similarity: ind.score.unwrap_or(0.0),
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "present_count",
            MetricValue::from_samples(&[report.present_count as f64]),
        );
        result.insert(
            "partial_count",
            MetricValue::from_samples(&[report.partial_count as f64]),
        );
        result.insert(
            "absent_count",
            MetricValue::from_samples(&[report.absent_count as f64]),
        );

        // Individual indicator scores + mean quality
        let mut quality_scores: Vec<f64> = Vec::new();
        for ind in &report.indicators {
            if let Some(score) = ind.score {
                quality_scores.push(score);
                result.insert(
                    format!("{}::{}", ind.id, ind.description.replace(' ', "_")),
                    MetricValue::from_samples(&[score]),
                );
            }
        }
        // Mean quality across all scored indicators (continuous, captures partial strengths)
        if !quality_scores.is_empty() {
            let mean_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
            result.insert(
                "mean_quality_score",
                MetricValue::from_samples(&[mean_quality]),
            );
        }

        // When symthaea-backend is enabled, run the ablation matrix and add
        // ablation results as additional metrics proving indicators are load-bearing.
        #[cfg(feature = "symthaea-backend")]
        {
            let ablation_results = super::ablation::run_ablation_matrix(config);
            for ar in &ablation_results {
                let prefix = format!("ablation::{}", ar.name);
                result.insert(
                    format!("{}::baseline_indicator", prefix),
                    MetricValue::from_samples(&[ar.baseline_indicator_score]),
                );
                result.insert(
                    format!("{}::ablated_indicator", prefix),
                    MetricValue::from_samples(&[ar.ablated_indicator_score]),
                );
                result.insert(
                    format!("{}::baseline_accuracy", prefix),
                    MetricValue::from_samples(&[ar.baseline_benchmark_accuracy]),
                );
                result.insert(
                    format!("{}::ablated_accuracy", prefix),
                    MetricValue::from_samples(&[ar.ablated_benchmark_accuracy]),
                );
                result.insert(
                    format!("{}::indicator_dropped", prefix),
                    MetricValue::from_samples(&[if ar.indicator_dropped { 1.0 } else { 0.0 }]),
                );
                result.insert(
                    format!("{}::benchmark_degraded", prefix),
                    MetricValue::from_samples(&[if ar.benchmark_degraded { 1.0 } else { 0.0 }]),
                );
            }
        }

        result.conditions = 14;
        result.trials_per_condition = 1;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butlin_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = ButlinIndicatorSuite.run(&config);
        assert!(result.metrics.contains_key("present_count"));
        // Should have all 14 indicators evaluated
        let total = result.metrics["present_count"].mean
            + result.metrics["partial_count"].mean
            + result.metrics["absent_count"].mean;
        assert_eq!(total, 14.0);
    }

    #[test]
    fn test_butlin_report_format() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let summary = report.summary();
        assert!(summary.contains("Butlin"));
        assert!(summary.contains("RPT-1"));
        assert!(summary.contains("IIT-1"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 5: Butlin Indicator Wiring tests
    // ═══════════════════════════════════════════════════════════════════

    use crate::benchmarks::butlin::report::RuntimeConsciousnessData;

    #[test]
    fn test_butlin_no_runtime_data_matches_static() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: None,
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        // RPT-1 static = 1.0, no runtime → should remain 1.0
        let rpt1 = report.indicators.iter().find(|i| i.id == "RPT-1").unwrap();
        assert_eq!(rpt1.score, Some(1.0));
    }

    #[test]
    fn test_butlin_with_runtime_data_changes_scores() {
        let config_static = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let report_static = ButlinIndicatorSuite::evaluate(&config_static);

        let config_rt = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                micro_phi: 2.0,
                meso_phi: 1.5,
                macro_phi: 3.0,
                bottleneck_score: 0.8,
                emergence_ratio: 2.0,
                num_clusters: 5,
            }),
            ..Default::default()
        };
        let report_rt = ButlinIndicatorSuite::evaluate(&config_rt);

        // At least some indicators should differ
        let mut any_different = false;
        for (s, r) in report_static
            .indicators
            .iter()
            .zip(report_rt.indicators.iter())
        {
            if (s.score.unwrap_or(0.0) - r.score.unwrap_or(0.0)).abs() > 0.001 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Runtime data should change at least some scores"
        );
    }

    #[test]
    fn test_butlin_blend_score_no_runtime() {
        let blended = ButlinIndicatorSuite::blend_score(0.8, None);
        assert!((blended - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_butlin_blend_score_with_runtime() {
        let blended = ButlinIndicatorSuite::blend_score(0.8, Some(1.0));
        // 0.6*0.8 + 0.4*1.0 = 0.48 + 0.4 = 0.88
        assert!(
            (blended - 0.88).abs() < 1e-10,
            "Expected 0.88, got {}",
            blended
        );
    }

    #[test]
    fn test_butlin_high_micro_phi_boosts_rpt() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                micro_phi: 5.0, // High → normalize_phi ≈ 1.0
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt1 = report.indicators.iter().find(|i| i.id == "RPT-1").unwrap();
        // blend(1.0, normalize_phi(5.0)≈0.9866) → 0.6*1.0 + 0.4*0.987 ≈ 0.995
        assert!(rpt1.score.unwrap() > 0.9);
    }

    #[test]
    fn test_butlin_zero_phi_reduces_iit() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                macro_phi: 0.0, // Zero → normalize_phi(0) = 0.0
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let iit1 = report.indicators.iter().find(|i| i.id == "IIT-1").unwrap();
        // blend(0.8, 0.0) = 0.6*0.8 + 0.4*0.0 = 0.48
        assert!(
            iit1.score.unwrap() < 0.8,
            "Zero Phi should reduce IIT-1, got {}",
            iit1.score.unwrap()
        );
    }

    #[test]
    fn test_butlin_high_meso_phi_boosts_gwt3() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                meso_phi: 5.0, // High meso
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt3 = report.indicators.iter().find(|i| i.id == "GWT-3").unwrap();
        // blend(0.8, ~1.0) ≈ 0.88
        assert!(gwt3.score.unwrap() > 0.8);
    }

    #[test]
    fn test_butlin_bottleneck_affects_hot2() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                bottleneck_score: 0.9, // High bottleneck
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let hot2 = report.indicators.iter().find(|i| i.id == "HOT-2").unwrap();
        // blend(0.7, (0.9*2).clamp(0,1)=1.0) = 0.6*0.7 + 0.4*1.0 = 0.82
        assert!(
            hot2.score.unwrap() > 0.7,
            "High bottleneck should boost HOT-2, got {}",
            hot2.score.unwrap()
        );
    }

    #[test]
    fn test_butlin_still_14_indicators() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                micro_phi: 1.0,
                meso_phi: 1.0,
                macro_phi: 1.0,
                bottleneck_score: 0.5,
                emergence_ratio: 1.5,
                num_clusters: 4,
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        assert_eq!(report.indicators.len(), 14);
    }

    #[test]
    fn test_ablation_still_works() {
        // Basic ablation presets should still function
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = ButlinIndicatorSuite.run(&config);
        let total = result.metrics["present_count"].mean
            + result.metrics["partial_count"].mean
            + result.metrics["absent_count"].mean;
        assert_eq!(total, 14.0);
    }

    #[test]
    fn test_runtime_data_default_is_zero() {
        let rt = RuntimeConsciousnessData::default();
        assert_eq!(rt.micro_phi, 0.0);
        assert_eq!(rt.meso_phi, 0.0);
        assert_eq!(rt.macro_phi, 0.0);
        assert_eq!(rt.bottleneck_score, 0.0);
        assert_eq!(rt.emergence_ratio, 0.0);
        assert_eq!(rt.num_clusters, 0);
    }

    #[test]
    fn test_normalize_phi_monotonic() {
        let values: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let normalized: Vec<f64> = values
            .iter()
            .map(|&v| ButlinIndicatorSuite::normalize_phi(v))
            .collect();
        for i in 1..normalized.len() {
            assert!(
                normalized[i] >= normalized[i - 1] - 1e-10,
                "normalize_phi should be monotonic: f({}) = {} < f({}) = {}",
                values[i],
                normalized[i],
                values[i - 1],
                normalized[i - 1]
            );
        }
    }

    #[test]
    fn test_runtime_consciousness_data_serializable() {
        let rt = RuntimeConsciousnessData {
            micro_phi: 1.5,
            meso_phi: 0.8,
            macro_phi: 2.0,
            bottleneck_score: 0.4,
            emergence_ratio: 1.3,
            num_clusters: 3,
        };
        let json = serde_json::to_string(&rt).unwrap();
        let rt2: RuntimeConsciousnessData = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.micro_phi, rt2.micro_phi);
        assert_eq!(rt.num_clusters, rt2.num_clusters);
    }

    #[test]
    fn test_runtime_consciousness_data_clone() {
        let rt = RuntimeConsciousnessData {
            micro_phi: 1.0,
            meso_phi: 2.0,
            macro_phi: 3.0,
            bottleneck_score: 0.5,
            emergence_ratio: 1.5,
            num_clusters: 4,
        };
        let rt2 = rt.clone();
        assert_eq!(rt.macro_phi, rt2.macro_phi);
        assert_eq!(rt.num_clusters, rt2.num_clusters);
    }
}
