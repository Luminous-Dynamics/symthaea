// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness indicator evaluation logic.
//!
//! Evaluates Symthaea against 14 Butlin et al. (2023) consciousness
//! indicators. Works with no runtime data at all (a static architectural
//! score per indicator), but as of 2026-07-24, 13 of the 14 also accept a
//! real, ablation-validated behavioral measurement
//! (`report::BehavioralIndicatorSignals`, populated by
//! `harness::live_runner::CognitiveLoopBenchmarkRunner::snapshot_behavioral_indicators`)
//! that takes priority over the plain structural-Phi-sigmoid proxy when
//! present. Only IIT-1 stays on the Phi proxy for live scoring — its claim
//! can only be tested via the ablation matrix's baseline-vs-ablated
//! comparison, not a single snapshot; see its evaluate() comment.
//!
//! **This module — not `examples/butlin_validation.rs` — is the canonical
//! target for any Butlin CI regression gate.** The two files have diverged:
//! `butlin_validation.rs` uses a different 14-indicator taxonomy (2 IIT/AST
//! split differently, no HOT-3/HOT-4/GWT-4) and has not been updated to use
//! real signals. That drift is a known, disclosed, not-yet-triaged issue —
//! reconciling the two is a separate task, not done as part of this one.
//!
//! **Shared formulas** (still used by `examples/butlin_validation.rs`, which
//! remains static-only):
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
#[derive(Default)]
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
            b.pp2_hierarchical_activity,
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
        // Prefers the real ablation-validated temporal-coherence probe
        // (`ablation::measure_indicator`) over the structural-Phi-sigmoid
        // proxy when a live measurement is available.
        let rpt1_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.rpt1_temporal_coherence.clamp(0.0, 1.0))
                .unwrap_or_else(|| Self::normalize_phi(r.micro_phi))
        });
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
        // Prefers the real cross-modal-binding-activity probe over the
        // structural-Phi-sigmoid proxy — the ablation matrix validates this
        // specific probe drops to 0 when `enable_cross_modal_binding = false`.
        let rpt2_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.rpt2_binding_activity.clamp(0.0, 1.0))
                .unwrap_or_else(|| Self::normalize_phi(r.micro_phi) * 0.85)
        });
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
        // Prefers a real aggregate over the num_clusters proxy: "are there
        // genuinely many independent specialized systems" is directly
        // testable as "of the mechanisms we've measured causal/behavioral
        // load for, what fraction are actually engaged" — reusing the other
        // 11 behavioral probes rather than a separate ablation row.
        let gwt1_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| Self::specialization_fraction(b))
                .unwrap_or_else(|| (r.num_clusters as f64 / 3.0).min(1.0))
        });
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

        // GWT-2: Limited capacity + selective attention
        // Was static-only (constant 1.0, never measured). Now prefers a real
        // probe: fraction of cycles with a non-empty but bounded GWT
        // coalition — the ablation matrix validates this drops to 0 (fails
        // the non-empty half) when `enable_gwt = false`.
        let gwt2_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.gwt2_bounded_coalition.clamp(0.0, 1.0))
        });
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
            score: Some(Self::blend_score(1.0, gwt2_runtime.flatten())),
        });

        // GWT-3: Global broadcast
        // Prefers the real broadcast-activity probe (fraction of cycles the
        // GWT module actually ran) over the structural-Phi-sigmoid proxy —
        // the ablation matrix validates this specific probe drops to 0 when
        // `enable_gwt = false`.
        let gwt3_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.gwt3_broadcast_activity.clamp(0.0, 1.0))
                .unwrap_or_else(|| Self::normalize_phi(r.meso_phi))
        });
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
        // Prefers the real phi_attention_weight-deviation probe over the
        // emergence-ratio proxy — the ablation matrix validates this drops
        // to 0 (stays neutral) when `enable_phi_attention = false`.
        let gwt4_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.gwt4_state_dependent_attention.clamp(0.0, 1.0))
                .unwrap_or_else(|| (r.emergence_ratio - 0.5).tanh())
        });
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

        // HOT-1: Generative/top-down perception
        // Was static-only (constant 0.9, never measured). Now prefers a real
        // probe: does prediction_error actually differentiate across
        // distinct inputs? Honestly near-zero while PE is frozen at a
        // degenerate-case sentinel (see
        // memory/symthaea_prediction_error_frozen_investigation.md) — this
        // is not a bug in the measurement, it's the measurement correctly
        // reporting a real, separately-tracked problem.
        let hot1_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.hot1_prediction_differentiation.clamp(0.0, 1.0))
        });
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
            score: Some(Self::blend_score(0.9, hot1_runtime.flatten())),
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
        // Prefers the real metacognitive-accuracy probe over the
        // structural-bottleneck proxy — the ablation matrix validates this
        // specific probe drops to 0 when `enable_meta_cognition = false`.
        let hot2_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.hot2_meta_cognitive_accuracy.clamp(0.0, 1.0))
                .unwrap_or_else(|| (r.bottleneck_score * 2.0).clamp(0.0, 1.0))
        });
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

        // HOT-3: Agency with belief updating
        // Was static-only (constant 0.84, never measured). Now prefers a
        // real presence signal: does online learning actually apply an
        // effective learning rate this cycle? Same underlying mechanism as
        // PP-1 (`actual_effective_lr`), gated by `enable_online_learning`
        // rather than `enable_prediction_learning` — a different Butlin
        // theoretical claim about the same real signal, not a duplicate.
        let hot3_runtime = rt.map(|r| {
            r.behavioral.as_ref().map(|b| {
                if b.hot3_effective_lr > 0.0005 {
                    1.0
                } else {
                    0.0
                }
            })
        });
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
            score: Some(Self::blend_score(0.84, hot3_runtime.flatten())),
        });

        // HOT-4: Sparse and smooth coding
        // Was static-only (constant 0.80, explicitly documented as "emergent,
        // not measured"). Now prefers a real, direct measurement — no
        // cognitive-loop ablation needed, since sparsity/smoothness are
        // properties of the encoded representations themselves: see
        // `live_runner::measure_hot4_sparse_smooth_coding`.
        //
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
        let hot4_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| ((b.hot4_sparsity + b.hot4_smoothness) / 2.0).clamp(0.0, 1.0))
        });
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
            score: Some(Self::blend_score(0.80, hot4_runtime.flatten())),
        });

        // PP-1: Prediction errors driving learning
        // Prefers the real effective-learning-rate probe over the
        // structural-Phi-sigmoid proxy. `actual_effective_lr` isn't a 0-1
        // quantity, so this is treated as a presence signal — the same
        // epsilon `ablation::run_ablation_matrix` uses to decide "baseline
        // already near zero, can't prove a drop" (0.0005) — rather than a
        // graded magnitude we have no principled scale for.
        let pp1_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| {
                    if b.pp1_effective_lr > 0.0005 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .unwrap_or_else(|| Self::normalize_phi(r.macro_phi))
        });
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
        // Prefers the real hierarchical-free-energy-activity probe over the
        // num_clusters proxy — the ablation matrix validates this drops to 0
        // when `enable_hierarchical_free_energy = false`. Coarser than a
        // true per-tau-level error trace (see report.rs's field doc), but a
        // real, mechanism-specific measurement rather than a cluster-count
        // guess.
        let pp2_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.pp2_hierarchical_activity.clamp(0.0, 1.0))
                .unwrap_or_else(|| if r.num_clusters >= 3 { 0.8 } else { 0.4 })
        });
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
        // Prefers the real attention-schema-focus probe over the
        // structural-bottleneck proxy — the ablation matrix validates this
        // specific probe drops when `enable_attention_schema = false`.
        let ast1_runtime = rt.map(|r| {
            r.behavioral
                .as_ref()
                .map(|b| b.ast1_attention_focus.clamp(0.0, 1.0))
                .unwrap_or_else(|| (r.bottleneck_score * 1.5).clamp(0.0, 1.0))
        });
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
        // Deliberately still on the structural-Phi-sigmoid proxy for live
        // scoring: IIT-1's real claim ("Phi is genuinely integrated
        // information, not a constant of the architecture") can only be
        // tested by a baseline-vs-ablated comparison, which lives in
        // `ablation_specs`'s `disable_gwt_for_iit1` row and
        // `butlin_ablation_integration.rs`'s tightened assertions — not in
        // a single live snapshot. Known reality as of the 2026-07-15 E1
        // subsystem-ablation audit: structural Phi was frozen at the same
        // value across nearly every ablation arm, so that row is expected
        // to honestly report NOT dropped until/unless that's independently
        // fixed.
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

    use crate::benchmarks::butlin::report::{BehavioralIndicatorSignals, RuntimeConsciousnessData};

    // ═══════════════════════════════════════════════════════════════════
    // Real behavioral signals (2026-07-22): these 5 indicators now prefer
    // ablation::measure_indicator's live probes over the structural-Phi
    // proxy. The point of these tests is that the REAL signal must win even
    // when it disagrees with what the Phi proxy would say.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_behavioral_signal_overrides_phi_proxy_for_rpt1() {
        // High micro_phi would normally push RPT-1's Phi-proxy toward ~1.0,
        // but a real behavioral measurement of 0.0 must win instead.
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
        let rpt1 = report.indicators.iter().find(|i| i.id == "RPT-1").unwrap();
        // blend(1.0, 0.0) = 0.6*1.0 + 0.4*0.0 = 0.6 — the Phi proxy alone
        // would have given blend(1.0, ~0.9866) ≈ 0.995.
        assert!(
            (rpt1.score.unwrap() - 0.6).abs() < 1e-9,
            "expected the real 0.0 behavioral signal to override the high-Phi proxy, got {}",
            rpt1.score.unwrap()
        );
    }

    #[test]
    fn test_behavioral_signal_used_for_gwt3_hot2_ast1() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    gwt3_broadcast_activity: 1.0,
                    hot2_meta_cognitive_accuracy: 1.0,
                    ast1_attention_focus: 1.0,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt3 = report.indicators.iter().find(|i| i.id == "GWT-3").unwrap();
        let hot2 = report.indicators.iter().find(|i| i.id == "HOT-2").unwrap();
        let ast1 = report.indicators.iter().find(|i| i.id == "AST-1").unwrap();
        // All Phi-derived fields default to 0.0, so the proxy fallback would
        // give a low/zero runtime component; the real signal of 1.0 must
        // instead push these to their maximum blend.
        assert!((gwt3.score.unwrap() - (0.6 * 0.8 + 0.4)).abs() < 1e-9);
        assert!((hot2.score.unwrap() - (0.6 * 0.85 + 0.4)).abs() < 1e-9);
        assert!((ast1.score.unwrap() - (0.6 * 0.85 + 0.4)).abs() < 1e-9);
    }

    #[test]
    fn test_pp1_effective_lr_below_epsilon_scores_zero_runtime() {
        // Today's measured reality (2026-07-22): the ablation matrix's own
        // baseline for `disable_prediction_learning` reads ~0.0002 — below
        // the 0.0005 epsilon `run_ablation_matrix` uses to decide "can't
        // prove a drop". PP-1's real-signal scoring must treat that
        // honestly as "not meaningfully active", not silently pass it
        // through a Phi proxy that would otherwise inflate the score.
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData {
                    macro_phi: 5.0, // would normalize_phi to ~0.9866 under the old proxy
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
        let pp1 = report.indicators.iter().find(|i| i.id == "PP-1").unwrap();
        // blend(0.9, 0.0) = 0.54 — the Phi proxy alone would have given
        // blend(0.9, ~0.9866) ≈ 0.9346.
        assert!(
            (pp1.score.unwrap() - 0.54).abs() < 1e-9,
            "expected a below-epsilon effective LR to score as inactive, got {}",
            pp1.score.unwrap()
        );
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
        let pp1 = report.indicators.iter().find(|i| i.id == "PP-1").unwrap();
        assert!((pp1.score.unwrap() - (0.6 * 0.9 + 0.4)).abs() < 1e-9);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Extended real behavioral signals (2026-07-24): the 6 indicators that
    // were previously either Phi-proxy-only or fully static-only.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_behavioral_signal_overrides_phi_proxy_for_rpt2() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData {
                    micro_phi: 5.0, // would push the old Phi-proxy toward ~1.0
                    ..Default::default()
                }
                .with_behavioral(BehavioralIndicatorSignals {
                    rpt2_binding_activity: 0.0,
                    ..Default::default()
                }),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let rpt2 = report.indicators.iter().find(|i| i.id == "RPT-2").unwrap();
        // blend(0.85, 0.0) = 0.51 — the Phi proxy alone would have given
        // blend(0.85, normalize_phi(5.0)*0.85≈0.838) ≈ 0.845.
        assert!(
            (rpt2.score.unwrap() - 0.51).abs() < 1e-9,
            "expected the real 0.0 binding-activity signal to override the high-Phi proxy, got {}",
            rpt2.score.unwrap()
        );
    }

    #[test]
    fn test_gwt2_static_only_becomes_measured() {
        // GWT-2 used to be a hardcoded 1.0 regardless of any runtime data.
        let config_no_signal = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default()),
            ..Default::default()
        };
        let report_no_signal = ButlinIndicatorSuite::evaluate(&config_no_signal);
        let gwt2_no_signal = report_no_signal
            .indicators
            .iter()
            .find(|i| i.id == "GWT-2")
            .unwrap();
        assert_eq!(gwt2_no_signal.score, Some(1.0));

        let config_low = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    gwt2_bounded_coalition: 0.0,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report_low = ButlinIndicatorSuite::evaluate(&config_low);
        let gwt2_low = report_low
            .indicators
            .iter()
            .find(|i| i.id == "GWT-2")
            .unwrap();
        // blend(1.0, 0.0) = 0.6 — proves a real 0.0 measurement can now pull
        // GWT-2 below its old unconditional 1.0.
        assert!((gwt2_low.score.unwrap() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_gwt4_behavioral_signal_used() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    gwt4_state_dependent_attention: 1.0,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt4 = report.indicators.iter().find(|i| i.id == "GWT-4").unwrap();
        assert!((gwt4.score.unwrap() - (0.6 * 0.8 + 0.4)).abs() < 1e-9);
    }

    #[test]
    fn test_hot1_static_only_becomes_measured() {
        let config_no_signal = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default()),
            ..Default::default()
        };
        let report_no_signal = ButlinIndicatorSuite::evaluate(&config_no_signal);
        let hot1_no_signal = report_no_signal
            .indicators
            .iter()
            .find(|i| i.id == "HOT-1")
            .unwrap();
        assert_eq!(hot1_no_signal.score, Some(0.9));

        let config_low = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    hot1_prediction_differentiation: 0.0,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report_low = ButlinIndicatorSuite::evaluate(&config_low);
        let hot1_low = report_low
            .indicators
            .iter()
            .find(|i| i.id == "HOT-1")
            .unwrap();
        // blend(0.9, 0.0) = 0.54 — honestly reflects a frozen-PE reality
        // instead of the old unconditional 0.9.
        assert!((hot1_low.score.unwrap() - 0.54).abs() < 1e-9);
    }

    #[test]
    fn test_hot3_effective_lr_presence_threshold() {
        let config_below = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    hot3_effective_lr: 0.0002,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report_below = ButlinIndicatorSuite::evaluate(&config_below);
        let hot3_below = report_below
            .indicators
            .iter()
            .find(|i| i.id == "HOT-3")
            .unwrap();
        assert!((hot3_below.score.unwrap() - (0.6 * 0.84)).abs() < 1e-9);

        let config_above = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    hot3_effective_lr: 0.01,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report_above = ButlinIndicatorSuite::evaluate(&config_above);
        let hot3_above = report_above
            .indicators
            .iter()
            .find(|i| i.id == "HOT-3")
            .unwrap();
        assert!((hot3_above.score.unwrap() - (0.6 * 0.84 + 0.4)).abs() < 1e-9);
    }

    #[test]
    fn test_hot4_sparsity_and_smoothness_averaged() {
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
        let hot4 = report.indicators.iter().find(|i| i.id == "HOT-4").unwrap();
        // mean(0.8, 0.6) = 0.7; blend(0.80, 0.7) = 0.48 + 0.28 = 0.76
        assert!((hot4.score.unwrap() - 0.76).abs() < 1e-9);
    }

    #[test]
    fn test_pp2_behavioral_signal_used() {
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(RuntimeConsciousnessData::default().with_behavioral(
                BehavioralIndicatorSignals {
                    pp2_hierarchical_activity: 0.0,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let pp2 = report.indicators.iter().find(|i| i.id == "PP-2").unwrap();
        // blend(0.85, 0.0) = 0.51 — with num_clusters defaulting to 0 here,
        // the old fallback formula would have given blend(0.85, 0.4) = 0.67;
        // 0.51 proves the real signal, not the fallback, drove this score.
        assert!((pp2.score.unwrap() - 0.51).abs() < 1e-9);
    }

    #[test]
    fn test_gwt1_specialization_fraction_all_active() {
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
            pp2_hierarchical_activity: 1.0,
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
        let gwt1 = report.indicators.iter().find(|i| i.id == "GWT-1").unwrap();
        // All 13 signals active → specialization_fraction = 1.0 →
        // blend(0.9, 1.0) = 0.94.
        assert!((gwt1.score.unwrap() - 0.94).abs() < 1e-9);
    }

    #[test]
    fn test_gwt1_specialization_fraction_all_inert() {
        let all_inert = BehavioralIndicatorSignals::default(); // all zero
        let config = BenchmarkConfig {
            dimension: 256,
            runtime_consciousness: Some(
                RuntimeConsciousnessData::default().with_behavioral(all_inert),
            ),
            ..Default::default()
        };
        let report = ButlinIndicatorSuite::evaluate(&config);
        let gwt1 = report.indicators.iter().find(|i| i.id == "GWT-1").unwrap();
        // All 13 signals inert → specialization_fraction = 0.0 →
        // blend(0.9, 0.0) = 0.54. Honestly reflects a scenario where most
        // measured mechanisms carry no causal load, matching the
        // independently-documented E1 finding rather than papering over it
        // with the old num_clusters-only proxy.
        assert!((gwt1.score.unwrap() - 0.54).abs() < 1e-9);
    }

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
                ..Default::default()
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
                ..Default::default()
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
}
