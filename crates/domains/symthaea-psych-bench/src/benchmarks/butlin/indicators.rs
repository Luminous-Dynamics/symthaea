// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness indicator evaluation logic.
//!
//! Evaluates Symthaea against the actual 14 Butlin et al. (2023) consciousness
//! indicators (arXiv:2308.08708, Table 1) — RPT-1/2, GWT-1/2/3/4, HOT-1/2/3/4,
//! AST-1, PP-1 (one, not two), AE-1/2 (Agency and Embodiment). The paper
//! explicitly excludes IIT ("not compatible with computational
//! functionalism").
//!
//! **Evidence model (2026-07-26 redesign — see `BUTLIN_EVIDENCE_TIER_DESIGN.md`
//! at the crate root for full rationale):** `evaluate()` alone, given a
//! single snapshot of live data, can only ever produce
//! `SupportTier::ArchitecturalOnly` — one scalar value can't rule out being a
//! frozen constant or fallback default, so it isn't honest to call it
//! `Observed` on that basis alone. `architectural_score` (the hand-assigned
//! constant) and `live_score` (the raw, unblended probe value) are reported
//! *separately*, never averaged — a blended number let a fully-dead live
//! signal still read as "present" (found via issue #7's regression-gate
//! review). Two exceptions genuinely earn `Observed` directly from a single
//! `evaluate()` call because their own measurement already embeds a real
//! responsiveness test: HOT-4's smoothness probe checks dissimilarity growth
//! across several perturbation magnitudes, and GWT-1 is a derived aggregate
//! (explicitly annotated as such, not independent evidence).
//! `CausallySupported`/`FunctionallySupported`/`NotDemonstrated`/
//! `Contradicted` only come from `report::annotate_with_ablation_results`,
//! which has the comparative baseline-vs-ablated evidence `evaluate()` alone
//! lacks.
//!
//! **This module — not `examples/butlin_validation.rs` — is the canonical
//! target for any Butlin CI regression gate.** The two files have diverged;
//! that drift is a known, disclosed, not-yet-triaged issue, out of scope here.

use super::report::{
    ButlinIndicatorReport, EvidenceAnnotation, EvidenceOutcome, IndicatorEvidence, ProbeQuality,
    Responsiveness, SupportTier,
};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Suite that evaluates all 14 Butlin consciousness indicators.
#[derive(Default)]
pub struct ButlinIndicatorSuite;

impl ButlinIndicatorSuite {
    /// Normalize a raw Phi value to [0, 1] via shifted sigmoid.
    /// Same mapping as the consciousness engine: 2/(1+exp(-phi)) - 1.
    /// Retained as a shared formula (also duplicated in
    /// `examples/butlin_validation.rs`); no longer called by `evaluate()`
    /// itself now that blending is gone, but kept for that formula's
    /// documented shared use and its own direct unit test.
    fn normalize_phi(phi: f64) -> f64 {
        2.0 / (1.0 + (-phi).exp()) - 1.0
    }

    /// GWT-1's real signal: fraction of the other measured mechanisms that
    /// are meaningfully engaged (activity fractions above 0.1; the two raw
    /// learning-rate signals use the same 0.0005 presence epsilon the
    /// ablation matrix uses elsewhere). A genuine architecture of "many
    /// independent specialized systems" should show most of these active;
    /// an architecture where most mechanisms are causally inert (per the
    /// separate E1 subsystem-ablation audit) should honestly show a low
    /// fraction here instead.
    fn specialization_fraction(b: &super::report::BehavioralIndicatorSignals) -> f64 {
        let unit_signals = [
            b.rpt1_temporal_coherence,
            b.rpt2_binding_activity,
            b.gwt2_bounded_coalition,
            b.gwt3_broadcast_activity,
            b.gwt4_state_dependent_attention,
            b.hot1_prediction_differentiation,
            b.hot2_meta_cognitive_accuracy,
            b.ae1_action_diversity,
            b.ae2_embodied_agency,
            b.ast1_attention_focus,
            b.hot4_sparsity,
            b.hot4_smoothness,
        ];
        let raw_signals = [b.pp1_effective_lr, b.hot3_effective_lr];

        let unit_active = unit_signals.iter().filter(|&&v| v > 0.1).count();
        let raw_active = raw_signals.iter().filter(|&&v| v > 0.0005).count();
        let total_active = unit_active + raw_active;
        let total = unit_signals.len() + raw_signals.len();
        total_active as f64 / total as f64
    }

    /// Build an `ArchitecturalOnly` indicator, with a raw live value attached
    /// when one was computed (either a real behavioral probe or, absent one,
    /// the coarser structural-Phi-proxy — annotated `ProxyMeasure` in that
    /// case since it isn't a targeted measurement of this indicator's own
    /// mechanism).
    fn architectural_indicator(
        id: &str,
        theory: &str,
        description: &str,
        evidence: String,
        architectural_score: f64,
        live_value: Option<f64>,
        is_proxy: bool,
    ) -> IndicatorEvidence {
        let mut annotations = Vec::new();
        if is_proxy && live_value.is_some() {
            annotations.push(EvidenceAnnotation::ProxyMeasure);
        }
        IndicatorEvidence {
            id: id.to_string(),
            theory: theory.to_string(),
            description: description.to_string(),
            outcome: EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            evidence,
            architectural_score,
            live_score: live_value,
            probe_quality: live_value.map(|_| ProbeQuality {
                sample_count: 1,
                finite_fraction: 1.0,
                variance: None,
                responsiveness: Responsiveness::NotAssessed,
                fallback_used: false,
                degeneracy: None,
            }),
            causal_effect: None,
            functional_effect: None,
            annotations,
        }
    }

    /// Evaluate all 14 indicators. Static-architecture reasoning always
    /// applies; when `config.runtime_consciousness` is `Some`, a live value
    /// is attached as `live_score` alongside (never blended into) the
    /// hand-assigned `architectural_score`.
    pub fn evaluate(config: &BenchmarkConfig) -> ButlinIndicatorReport {
        let mut indicators = Vec::new();
        let rt = config.runtime_consciousness.as_ref();
        let behavioral = rt.and_then(|r| r.behavioral.as_ref());

        // RPT-1: Algorithmic recurrence (CfC feedback loop)
        let rpt1_live = behavioral
            .map(|b| b.rpt1_temporal_coherence.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| Self::normalize_phi(r.micro_phi)));
        indicators.push(Self::architectural_indicator(
            "RPT-1",
            "Recurrent Processing Theory",
            "Algorithmic recurrence",
            "CfC (Closed-form Continuous-time) temporal network provides \
                recurrent feedback via O(1) closed-form temporal jumps in \
                hdc_ltc_unified.rs; each cognitive cycle feeds predictions \
                back as input to the next cycle"
                .into(),
            1.0,
            rpt1_live,
            behavioral.is_none(),
        ));

        // RPT-2: Integrated perceptual representations
        let rpt2_live = behavioral
            .map(|b| b.rpt2_binding_activity.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| Self::normalize_phi(r.micro_phi) * 0.85));
        indicators.push(Self::architectural_indicator(
            "RPT-2",
            "Recurrent Processing Theory",
            "Integrated perceptual representations",
            "IIT Phi engine (phi_engine/) computes integrated information \
                over HDC state; ContinuousHV bundle operations create holographic \
                superpositions that integrate features across perceptual modalities; \
                dual-path integration (Lamme, 2006): local HDC binding + global Phi"
                .into(),
            0.85,
            rpt2_live,
            behavioral.is_none(),
        ));

        // GWT-1: Parallel specialized systems — a derived aggregate, not an
        // independent probe (see module doc + BUTLIN_EVIDENCE_TIER_DESIGN.md).
        let gwt1_live = behavioral
            .map(Self::specialization_fraction)
            .or_else(|| rt.map(|r| (r.num_clusters as f64 / 3.0).min(1.0)));
        let mut gwt1 = Self::architectural_indicator(
            "GWT-1",
            "Global Workspace Theory",
            "Parallel specialized systems",
            "12-region Actor Brain architecture with concurrent subsystems: \
                HDC encoding, CfC temporal processing, FEP active inference, \
                moral algebra, reasoning engine (7-step cycle), social coherence \
                (ToM), all coordinated via rayon-parallel post-processing in \
                cognitive_loop/cycle.rs"
                .into(),
            0.9,
            gwt1_live,
            behavioral.is_none(),
        );
        gwt1.annotations.push(EvidenceAnnotation::DerivedAggregate {
            sources: [
                "RPT-1", "RPT-2", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2", "AE-1", "AE-2",
                "AST-1", "HOT-4",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        });
        indicators.push(gwt1);

        // GWT-2: Limited capacity with selective attention
        let gwt2_live = behavioral.map(|b| b.gwt2_bounded_coalition.clamp(0.0, 1.0));
        indicators.push(Self::architectural_indicator(
            "GWT-2",
            "Global Workspace Theory",
            "Limited capacity with selective attention",
            format!(
                "Working memory capacity={} with FIFO eviction enforces information \
                bottleneck; prefrontal gating selects which items enter the global \
                workspace; activation decay (0.9/step) in working_memory.rs",
                config.working_memory_capacity
            ),
            1.0,
            gwt2_live,
            false,
        ));

        // GWT-3: Global broadcast mechanism
        let gwt3_live = behavioral
            .map(|b| b.gwt3_broadcast_activity.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| Self::normalize_phi(r.meso_phi)));
        indicators.push(Self::architectural_indicator(
            "GWT-3",
            "Global Workspace Theory",
            "Global broadcast mechanism",
            "CycleMetadata.gwt_broadcast flag in cognitive_loop indicates \
                when WM contents are broadcast to all subsystems; the 8-phase \
                pipeline (perception -> cognition -> translation) in symthaea.rs \
                distributes processed state to reasoning, moral, and social modules"
                .into(),
            0.8,
            gwt3_live,
            behavioral.is_none(),
        ));

        // GWT-4: State-dependent attention modulation
        let gwt4_live = behavioral
            .map(|b| b.gwt4_state_dependent_attention.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| (r.emergence_ratio - 0.5).tanh()));
        indicators.push(Self::architectural_indicator(
            "GWT-4",
            "Global Workspace Theory",
            "State-dependent attention modulation",
            "CycleUrgency adapts subsystem scheduling based on prediction \
                error magnitude; surprise-driven exploration modulates attention \
                allocation; consciousness_level field tracks dynamic state changes"
                .into(),
            0.8,
            gwt4_live,
            behavioral.is_none(),
        ));

        // HOT-1: Generative/top-down perceptual processing
        let hot1_live = behavioral.map(|b| b.hot1_prediction_differentiation.clamp(0.0, 1.0));
        indicators.push(Self::architectural_indicator(
            "HOT-1",
            "Higher-Order Theories",
            "Generative/top-down perceptual processing",
            "PredictiveHdcEncoder generates top-down predictions compared \
                against sensory input; prediction errors drive CfC weight updates \
                in the predictive coding loop (HDC encode -> CfC evolve -> predict \
                -> learn at 50Hz)"
                .into(),
            0.9,
            hot1_live,
            false,
        ));

        // HOT-2: Metacognitive monitoring of own states
        let hot2_live = behavioral
            .map(|b| b.hot2_meta_cognitive_accuracy.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| (r.bottleneck_score * 2.0).clamp(0.0, 1.0)));
        indicators.push(Self::architectural_indicator(
            "HOT-2",
            "Higher-Order Theories",
            "Metacognitive monitoring of own states",
            "meta_cognitive_accuracy tracked in CycleMetadata; meta-cognition \
                module monitors processing quality and confidence calibration (FOK \
                logistic mapping, Metcalfe 2000); reasoning engine 7-step cycle includes \
                self-evaluation phase; hubris detection attenuates HOT depth via harmony \
                entropy monitoring; three metacognitive sub-capabilities present \
                (Lau & Rosenthal, 2011)"
                .into(),
            0.85,
            hot2_live,
            behavioral.is_none(),
        ));

        // HOT-3: Agency with belief updating from action outcomes
        let hot3_live = behavioral.map(|b| {
            if b.hot3_effective_lr > 0.0005 {
                1.0
            } else {
                0.0
            }
        });
        indicators.push(Self::architectural_indicator(
            "HOT-3",
            "Higher-Order Theories",
            "Agency with belief updating from action outcomes",
            "FEP active inference (fep_active_inference.rs) generates motor \
                commands with expected outcomes; CfC weight updates from prediction \
                errors implement belief revision; planning_horizon controls \
                forward-looking action selection; full sensorimotor contingency \
                tracking constitutes a complete BDI architecture (Bratman, 1987; \
                O'Regan & Noe, 2001)"
                .into(),
            0.84,
            hot3_live,
            false,
        ));

        // HOT-4: Sparse and smooth neural coding — earns Observed directly:
        // the smoothness half of this measurement (`measure_hot4_sparse_smooth_coding`)
        // already checks dissimilarity growth across several distinct
        // perturbation magnitudes, a genuine within-probe responsiveness
        // test, not a single frozen scalar.
        let hot4_live =
            behavioral.map(|b| ((b.hot4_sparsity + b.hot4_smoothness) / 2.0).clamp(0.0, 1.0));
        let hot4_evidence = format!(
            "ContinuousHV provides smooth (differentiable) representations in \
                {}-dimensional space; HDC holographic encoding naturally produces \
                sparse activation patterns (Olshausen & Field, 1996); similarity \
                is a smooth function of representational distance; 16,384D space \
                provides rich continuous manifold for smooth neural coding; sparsity \
                is emergent (high-D concentration of measure) rather than enforced",
            config.dimension
        );
        let hot4 = if let Some(live) = hot4_live {
            IndicatorEvidence {
                id: "HOT-4".into(),
                theory: "Higher-Order Theories".into(),
                description: "Sparse and smooth neural coding".into(),
                outcome: EvidenceOutcome::Supported(SupportTier::Observed),
                evidence: hot4_evidence,
                architectural_score: 0.80,
                live_score: Some(live),
                probe_quality: Some(ProbeQuality {
                    sample_count: 2,
                    finite_fraction: 1.0,
                    variance: None,
                    responsiveness: Responsiveness::SensitiveToManipulation,
                    fallback_used: false,
                    degeneracy: None,
                }),
                causal_effect: None,
                functional_effect: None,
                annotations: vec![EvidenceAnnotation::InternalTelemetry],
            }
        } else {
            Self::architectural_indicator(
                "HOT-4",
                "Higher-Order Theories",
                "Sparse and smooth neural coding",
                hot4_evidence,
                0.80,
                None,
                false,
            )
        };
        indicators.push(hot4);

        // PP-1: Prediction errors drive learning and adaptation
        let pp1_live = behavioral
            .map(|b| {
                if b.pp1_effective_lr > 0.0005 {
                    1.0
                } else {
                    0.0
                }
            })
            .or_else(|| rt.map(|r| Self::normalize_phi(r.macro_phi)));
        indicators.push(Self::architectural_indicator(
            "PP-1",
            "Predictive Processing",
            "Prediction errors drive learning and adaptation",
            "Core cognitive pipeline: HDC encode -> CfC evolve -> predict -> \
                learn; prediction error drives CfC weight updates, surprise \
                exploration, and episodic memory priority (Phi-weighted); \
                learning_occurred flag tracked per cycle"
                .into(),
            0.9,
            pp1_live,
            behavioral.is_none(),
        ));

        // AE-1: Agency (Butlin et al. 2023 Table 1)
        let ae1_live = behavioral.map(|b| b.ae1_action_diversity.clamp(0.0, 1.0));
        indicators.push(Self::architectural_indicator(
            "AE-1",
            "Agency and Embodiment",
            "Agency: learning from feedback, flexible goal pursuit",
            "FEP active inference (fep_module.rs) selects among 4 actions \
                (exploit/consolidate/explore/tighten) via expected free energy; \
                trajectory planning (enable_trajectory_planning) simulates future \
                horizons to choose among competing action policies (Friston 2010); \
                CfC weight updates from prediction errors implement belief revision \
                feeding back into action selection."
                .into(),
            0.85,
            ae1_live,
            false,
        ));

        // AST-1: Self-model of attention (Graziano 2013, 2019)
        let ast1_live = behavioral
            .map(|b| b.ast1_attention_focus.clamp(0.0, 1.0))
            .or_else(|| rt.map(|r| (r.bottleneck_score * 1.5).clamp(0.0, 1.0)));
        indicators.push(Self::architectural_indicator(
            "AST-1",
            "Attention Schema Theory",
            "Self-model of attention process",
            "AttentionSchema (attention_schema.rs) implements Graziano's AST \
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
            0.85,
            ast1_live,
            behavioral.is_none(),
        ));

        // AE-2: Embodiment (Butlin et al. 2023 Table 1)
        let ae2_live = behavioral.map(|b| b.ae2_embodied_agency.clamp(0.0, 1.0));
        indicators.push(Self::architectural_indicator(
            "AE-2",
            "Agency and Embodiment",
            "Embodiment: modeling output-input contingencies",
            "EmbodimentBridge trait (embodiment.rs) implemented by 10 robot \
                platforms (manipulator, humanoid, multirotor, vehicle, AUV, etc.); \
                embodied_cognition subsystem computes embodied_agency (0-1) from \
                body-state feedback into perception, independently found causally \
                load-bearing by the 2026-07-15 E1 subsystem-ablation audit; \
                proprioceptive loop blends body state into perception each cycle."
                .into(),
            0.85,
            ae2_live,
            false,
        ));

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

        // When symthaea-backend is enabled, run the real ablation matrix and
        // merge its evidence into the report so tier counts/outcomes reflect
        // actual causal/functional support, not just the static layer. Merge
        // errors (unknown/duplicate indicator IDs) would indicate a real bug
        // in this crate's own indicator-ID bookkeeping, not user input, so
        // panicking here (rather than silently reporting stale data) is the
        // right failure mode.
        #[cfg(feature = "symthaea-backend")]
        let (report, ablation_results) = {
            let ablation_results = super::ablation::run_ablation_matrix(config);
            let bundle = super::ablation::build_evidence_bundle(config, ablation_results.clone());
            let report = super::report::annotate_with_ablation_results(report, &bundle).expect(
                "ablation evidence bundle should always match this suite's own indicator IDs",
            );
            (report, ablation_results)
        };

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        // Emit one trace entry per indicator for consistency
        for ind in &report.indicators {
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: ind.id.to_string(),
                    correct: matches!(
                        ind.outcome,
                        EvidenceOutcome::Supported(
                            SupportTier::CausallySupported | SupportTier::FunctionallySupported
                        )
                    ),
                    rt_ticks: 0.0,
                    similarity: ind.live_score.unwrap_or(ind.architectural_score),
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "architectural_only_count",
            MetricValue::from_samples(&[report.architectural_only_count as f64]),
        );
        result.insert(
            "observed_count",
            MetricValue::from_samples(&[report.observed_count as f64]),
        );
        result.insert(
            "causally_supported_count",
            MetricValue::from_samples(&[report.causally_supported_count as f64]),
        );
        result.insert(
            "functionally_supported_count",
            MetricValue::from_samples(&[report.functionally_supported_count as f64]),
        );
        result.insert(
            "not_demonstrated_count",
            MetricValue::from_samples(&[report.not_demonstrated_count as f64]),
        );
        result.insert(
            "contradicted_count",
            MetricValue::from_samples(&[report.contradicted_count as f64]),
        );

        // Individual indicator architectural/live scores (reported
        // separately, never blended — see module doc).
        for ind in &report.indicators {
            result.insert(
                format!(
                    "{}::{}::architectural",
                    ind.id,
                    ind.description.replace(' ', "_")
                ),
                MetricValue::from_samples(&[ind.architectural_score]),
            );
            if let Some(live) = ind.live_score {
                result.insert(
                    format!("{}::{}::live", ind.id, ind.description.replace(' ', "_")),
                    MetricValue::from_samples(&[live]),
                );
            }
        }

        // Ablation results (already merged into `report` above) as additional
        // metrics proving indicators are load-bearing.
        #[cfg(feature = "symthaea-backend")]
        {
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
                result.insert(
                    format!("{}::contradicted", prefix),
                    MetricValue::from_samples(&[if ar.contradicted { 1.0 } else { 0.0 }]),
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
    use crate::benchmarks::butlin::report::{BehavioralIndicatorSignals, RuntimeConsciousnessData};

    fn find<'a>(report: &'a ButlinIndicatorReport, id: &str) -> &'a IndicatorEvidence {
        report.indicators.iter().find(|i| i.id == id).unwrap()
    }

    #[test]
    fn test_butlin_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = ButlinIndicatorSuite.run(&config);
        let total = result.metrics["architectural_only_count"].mean
            + result.metrics["observed_count"].mean
            + result.metrics["causally_supported_count"].mean
            + result.metrics["functionally_supported_count"].mean
            + result.metrics["not_demonstrated_count"].mean
            + result.metrics["contradicted_count"].mean;
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
        assert!(summary.contains("AE-2"));
    }

    #[test]
    fn test_static_evaluation_never_earns_more_than_architectural_only_or_hot4_observed() {
        // Required test: without ablation evidence, no indicator can claim
        // Causal/Functional support, and only HOT-4 (whose own probe embeds
        // a real responsiveness test) may reach Observed.
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    rpt1_temporal_coherence: 1.0,
                    rpt2_binding_activity: 1.0,
                    gwt2_bounded_coalition: 1.0,
                    gwt3_broadcast_activity: 1.0,
                    gwt4_state_dependent_attention: 1.0,
                    hot1_prediction_differentiation: 1.0,
                    hot2_meta_cognitive_accuracy: 1.0,
                    hot3_effective_lr: 0.01,
                    pp1_effective_lr: 0.01,
                    ae1_action_diversity: 1.0,
                    ae2_embodied_agency: 1.0,
                    ast1_attention_focus: 1.0,
                    hot4_sparsity: 1.0,
                    hot4_smoothness: 1.0,
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        for ind in &report.indicators {
            if ind.id == "HOT-4" {
                assert_eq!(
                    ind.outcome,
                    EvidenceOutcome::Supported(SupportTier::Observed)
                );
            } else {
                assert_eq!(
                    ind.outcome,
                    EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
                    "{} should not exceed ArchitecturalOnly from a static evaluate() call",
                    ind.id
                );
            }
        }
    }

    #[test]
    fn test_live_score_reported_separately_from_architectural_score() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData {
                    micro_phi: 5.0,
                    ..Default::default()
                }
                .with_behavioral(BehavioralIndicatorSignals {
                    rpt1_temporal_coherence: 0.0,
                    ..Default::default()
                }),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt1 = find(&report, "RPT-1");
        // architectural_score is the unchanged hand-assigned constant...
        assert_eq!(rpt1.architectural_score, 1.0);
        // ...and live_score is the raw, unblended probe value -- a real 0.0
        // measurement is reported as exactly 0.0, not diluted by the high
        // micro_phi that would have inflated an old Phi-proxy blend.
        assert_eq!(rpt1.live_score, Some(0.0));
    }

    #[test]
    fn test_no_runtime_data_gives_architectural_only_no_live_score() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: None,
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt1 = find(&report, "RPT-1");
        assert_eq!(
            rpt1.outcome,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly)
        );
        assert_eq!(rpt1.architectural_score, 1.0);
        assert_eq!(rpt1.live_score, None);
    }

    #[test]
    fn test_proxy_fallback_annotated_when_no_behavioral_field() {
        // Runtime data present but no behavioral signals -- falls back to
        // the coarser structural-Phi proxy, which must be annotated as such.
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData {
                micro_phi: 2.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt1 = find(&report, "RPT-1");
        assert!(rpt1.live_score.is_some());
        assert!(rpt1.annotations.contains(&EvidenceAnnotation::ProxyMeasure));
    }

    #[test]
    fn test_real_behavioral_probe_not_annotated_as_proxy() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    rpt1_temporal_coherence: 0.7,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt1 = find(&report, "RPT-1");
        assert!(!rpt1.annotations.contains(&EvidenceAnnotation::ProxyMeasure));
    }

    #[test]
    fn test_gwt1_is_derived_aggregate_and_stays_architectural_only() {
        let all_active = BehavioralIndicatorSignals {
            rpt1_temporal_coherence: 1.0,
            rpt2_binding_activity: 1.0,
            gwt2_bounded_coalition: 1.0,
            gwt3_broadcast_activity: 1.0,
            gwt4_state_dependent_attention: 1.0,
            hot1_prediction_differentiation: 1.0,
            hot2_meta_cognitive_accuracy: 1.0,
            hot3_effective_lr: 0.01,
            pp1_effective_lr: 0.01,
            ae1_action_diversity: 1.0,
            ae2_embodied_agency: 1.0,
            ast1_attention_focus: 1.0,
            hot4_sparsity: 1.0,
            hot4_smoothness: 1.0,
        };
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData::default().with_behavioral(all_active),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt1 = find(&report, "GWT-1");
        assert_eq!(gwt1.live_score, Some(1.0));
        assert_eq!(
            gwt1.outcome,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly)
        );
        assert!(
            gwt1.annotations
                .iter()
                .any(|a| matches!(a, EvidenceAnnotation::DerivedAggregate { .. }))
        );
    }

    #[test]
    fn test_gwt1_specialization_fraction_all_inert() {
        let all_inert = BehavioralIndicatorSignals::default();
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData::default().with_behavioral(all_inert),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt1 = find(&report, "GWT-1");
        assert_eq!(gwt1.live_score, Some(0.0));
    }

    #[test]
    fn test_hot4_earns_observed_with_probe_quality() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    hot4_sparsity: 0.8,
                    hot4_smoothness: 0.6,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let hot4 = find(&report, "HOT-4");
        assert_eq!(
            hot4.outcome,
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
        assert_eq!(hot4.live_score, Some(0.7));
        let pq = hot4.probe_quality.expect("HOT-4 should carry ProbeQuality");
        assert_eq!(pq.responsiveness, Responsiveness::SensitiveToManipulation);
        assert!(pq.qualifies_as_observed());
    }

    #[test]
    fn test_hot4_without_behavioral_data_stays_architectural_only() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: None,
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let hot4 = find(&report, "HOT-4");
        assert_eq!(
            hot4.outcome,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly)
        );
        assert_eq!(hot4.live_score, None);
    }

    #[test]
    fn test_pp1_effective_lr_below_epsilon_scores_zero_live() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData {
                    macro_phi: 5.0,
                    ..Default::default()
                }
                .with_behavioral(BehavioralIndicatorSignals {
                    pp1_effective_lr: 0.0002,
                    ..Default::default()
                }),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let pp1 = find(&report, "PP-1");
        assert_eq!(pp1.live_score, Some(0.0));
    }

    #[test]
    fn test_pp1_effective_lr_above_epsilon_scores_active() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    pp1_effective_lr: 0.01,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let pp1 = find(&report, "PP-1");
        assert_eq!(pp1.live_score, Some(1.0));
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
                ..Default::default()
            }),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        assert_eq!(report.indicators.len(), 14);
    }

    #[test]
    fn test_ablation_still_works() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = ButlinIndicatorSuite.run(&config);
        let total = result.metrics["architectural_only_count"].mean
            + result.metrics["observed_count"].mean
            + result.metrics["causally_supported_count"].mean
            + result.metrics["functionally_supported_count"].mean
            + result.metrics["not_demonstrated_count"].mean
            + result.metrics["contradicted_count"].mean;
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
            ..Default::default()
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
            ..Default::default()
        };
        let rt2 = rt.clone();
        assert_eq!(rt.macro_phi, rt2.macro_phi);
        assert_eq!(rt.num_clusters, rt2.num_clusters);
    }

    #[test]
    fn test_no_serialization_path_emits_old_present_status() {
        // Required test: the old IndicatorStatus::Present/Partial/Absent
        // vocabulary must not survive anywhere in serialized report output.
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("\"Present\""));
        assert!(!json.contains("\"Partial\""));
        assert!(!json.contains("\"Absent\""));
    }
}
