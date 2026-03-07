//! # Unified Ethics Engine
//!
//! Wraps the 3 independent ethics systems into a single coherent engine
//! with clear data flow and unified output.
//!
//! ## Systems (Pipeline Architecture)
//!
//! | Stage | System | Interval | Science |
//! |-------|--------|----------|---------|
//! | 1 | MoralParser + MoralAlgebra | 7 cycles | HDC moral algebra (Luo & Lakoff 2019) |
//! | 2 | UnifiedValueEvaluator | 19 cycles | Value alignment (Panksepp 1998) |
//! | 3 | HarmoniesIntegrator | 19 cycles | Seven Harmonies (Schwartz 2012) |
//!
//! ## Design Principles
//!
//! 1. **Pipeline**: moral parse → value gate → harmonies check → unified verdict
//! 2. **No direct field mutation**: Returns `EthicsEngineOutput` with proposed deltas
//! 3. **Preserves co-prime intervals**: Each subsystem fires at its original rate
//! 4. **Backward compatible**: All existing carryover fields populated

use std::sync::Arc;
use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;

use crate::consciousness::harmonies_integration::{HarmoniesIntegrator, ValuedAction};
use crate::consciousness::unified_value_evaluator::{
    Decision, EvaluationContext, UnifiedValueEvaluator,
};
use crate::hdc::harmony_basis::{HarmonyBasis, HarmonyInteractionMatrix, MoralFreeEnergy};
use symthaea_types::N_HARMONIES;
use crate::hdc::moral_algebra::{DeontologicalVerdict, MoralAlgebra, MoralVerdict};
use crate::hdc::moral_parser::MoralParser;
use crate::hdc::moral_topology::{
    MoralAnomalyConfig, MoralAnomalyReport, MoralTopology, MoralTopologyConfig,
    MoralTopologySummary,
};

/// Unified output from the ethics engine.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields flow through EthicsEngineCache; read via cache, not struct
pub(crate) struct EthicsEngineOutput {
    // ── Stage 1: Moral Algebra ─────────────────────────────────────────
    // Flows through EthicsEngine cache; cycle reads via moral_topology() accessor
    /// Moral score from HDC algebra [-1.0, 1.0]
    pub moral_score: f64,
    /// Moral verdict string (Good/Bad/Neutral/ConsentViolation)
    pub moral_verdict: String,
    /// Deontological verdict (Permissible/Impermissible/Neutral)
    pub deontological_verdict: String,
    /// Whether consent violation detected
    pub consent_violation: bool,
    /// Moral parsing confidence [0, 1]
    pub moral_confidence: f64,
    /// Deontological violations detected
    pub violations: Vec<String>,
    /// Deontological satisfactions detected
    pub satisfactions: Vec<String>,

    // ── Stage 2: Value Evaluator ───────────────────────────────────────
    /// Value alignment score [0, 1]
    pub value_score: f64,
    /// Value decision (Allow/Warn/Veto)
    pub value_decision: String,
    /// Learning rate gate factor from value evaluator
    pub value_gate_factor: f32,

    // ── Stage 3: Harmonies ─────────────────────────────────────────────
    /// Seven Harmonies alignment [0, 1]
    pub harmonies_alignment: f32,
    /// Whether action is approved by harmonies
    pub harmonies_approved: bool,

    // ── Unified verdict ────────────────────────────────────────────────
    /// Combined ethical verdict: Safe, Caution, or Blocked
    pub unified_verdict: EthicalVerdict,
    /// Combined ethical confidence [0, 1]
    pub unified_confidence: f64,

    // ── Proposed feedback deltas ────────────────────────────────────────
    /// Additive delta for prediction_confidence
    pub confidence_delta: f32,
    /// Multiplicative factor for subsystem_lr_factor
    pub lr_factor: f32,

    // ── Stage 4: Moral Topology ────────────────────────────────────────
    /// Microseconds spent on topology analysis (0 when not run).
    pub topology_us: u64,
    /// Whether topology analysis was freshly computed this cycle.
    /// Used by cycle_strategy.rs to gate anomaly response (prevents N× over-correction).
    pub topology_fresh: bool,

    // ── Stage 4b: Anomaly report ───────────────────────────────────────
    /// Moral trajectory anomaly report (computed each cycle from latest topology).
    pub anomaly_report: MoralAnomalyReport,

    // ── Stage 3b: Moral Geometry (FEP) ─────────────────────────────────
    /// 8D harmony coordinates for this cycle's action
    #[allow(dead_code)] // Computed by harmonies integrator; read via engine cache
    pub harmony_coordinates: [f64; N_HARMONIES],
    /// Moral free energy decomposition (FEP on harmony manifold)
    #[allow(dead_code)] // Computed by harmonies integrator; read via engine cache
    pub moral_free_energy: MoralFreeEnergy,

    // ── Timing ─────────────────────────────────────────────────────────
    pub moral_us: u64,
    pub value_us: u64,
    pub harmonies_us: u64,
    pub total_us: u64,
}

/// Unified ethical verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EthicalVerdict {
    /// All systems agree: action is ethical
    Safe,
    /// One or more systems flag concerns but don't block
    Caution,
    /// Value evaluator vetoes or consent violation detected
    Blocked,
}

/// Input snapshot for the ethics engine.
pub(crate) struct EthicsEngineInput<'a> {
    /// Input text for moral parsing
    pub input: &'a str,
    /// Current cycle number
    pub cycle: u64,
    /// Unified Psi (consciousness level) for value evaluator context
    pub unified_psi: f64,
    /// Compressed state (256-dim) for harmonies integrator
    pub compressed_state: &'a [f32],
    /// Sacred Stillness neuromod + circadian boost (applied to harmony coordinate 7).
    /// Computed by caller from GABA + adenosine levels and circadian phase.
    /// Science: Bhatt et al. (2020) — GABAergic tone correlates with resting-state activity;
    /// Porkka-Heiskanen et al. (1997) — adenosine accumulation signals rest need.
    pub stillness_boost: f32,
}

/// Result of Stage 1 moral evaluation only.
/// Used by `evaluate_moral_alignment()` to build `MoralJudgmentSummary`.
#[derive(Debug, Clone)]
pub(crate) struct MoralEvalResult {
    pub verdict: String,
    pub deontological_verdict: String,
    pub violations: Vec<String>,
    pub satisfactions: Vec<String>,
    pub consent_violation: bool,
    pub moral_score: f64,
    pub confidence: f32,
}

/// The unified ethics evaluation engine.
pub(crate) struct EthicsEngine {
    // ── Stage 1: HDC Moral Algebra (always present) ────────────────────
    moral_parser: MoralParser,
    moral_algebra: MoralAlgebra,

    // ── Stage 2: Value evaluator (optional) ────────────────────────────
    value_evaluator: Option<UnifiedValueEvaluator>,

    // ── Stage 3: Harmonies integrator (optional) ───────────────────────
    harmonies_integrator: Option<HarmoniesIntegrator>,

    // ── Stage 4: Moral topology (persistent homology) ──────────────────
    moral_topology: MoralTopology,

    // ── Stage 5: Harmony interaction matrix (synergies/tensions) ──────
    interaction_matrix: HarmonyInteractionMatrix,

    // ── Cached values ──────────────────────────────────────────────────
    cache: EthicsEngineCache,
}

#[derive(Debug, Clone)]
struct EthicsEngineCache {
    last_moral_score: f64,
    last_value_score: f64,
    last_harmonies_alignment: f32,
    last_harmonies_approved: bool,
    last_harmony_coordinates: [f64; N_HARMONIES],
    last_moral_free_energy: MoralFreeEnergy,
    /// EMA-smoothed moral free energy (α=0.1).
    /// Tracks trend rather than raw per-cycle spikes.
    moral_fe_ema: f64,
    /// Adaptive gain for moral FE → exploration coupling.
    /// Decays on exploration failure (high PE), reinforces on success (low PE).
    /// Range [0.05, 0.25], default 0.15.
    /// Science: Daw et al. (2005) — model-based/model-free arbitration.
    moral_exploration_gain: f32,
    /// Current topology evaluation interval (adaptive).
    topology_cadence: u64,
    /// Cycle number of last topology evaluation.
    last_topology_cycle: u64,
    /// Cached anomaly report from last evaluation.
    last_anomaly_report: MoralAnomalyReport,
    /// Whether the last evaluate() freshly computed topology (vs cached).
    last_topology_fresh: bool,
}

impl Default for EthicsEngineCache {
    fn default() -> Self {
        Self {
            last_moral_score: 0.0,
            last_value_score: 0.0,
            last_harmonies_alignment: 0.0,
            last_harmonies_approved: false,
            last_harmony_coordinates: [0.0; N_HARMONIES],
            last_moral_free_energy: MoralFreeEnergy::default(),
            moral_fe_ema: 0.0,
            moral_exploration_gain: 0.15,
            topology_cadence: 97,
            last_topology_cycle: 0,
            last_anomaly_report: MoralAnomalyReport::default(),
            last_topology_fresh: false,
        }
    }
}

#[allow(dead_code)]
impl EthicsEngine {
    /// Create a new ethics engine from its component systems.
    ///
    /// When the `HarmoniesIntegrator` operates at the same HDC dimension as the
    /// `MoralAlgebra`, a single `Arc<HarmonyBasis>` is shared between
    /// `MoralTopology` and `HarmoniesIntegrator`, deduplicating ~448KB of basis
    /// vectors. When dimensions differ (e.g., integrator uses compressed-state
    /// dim), each keeps its own basis.
    pub fn new(
        moral_parser: MoralParser,
        moral_algebra: MoralAlgebra,
        value_evaluator: Option<UnifiedValueEvaluator>,
        harmonies_integrator: Option<HarmoniesIntegrator>,
    ) -> Self {
        Self::with_anomaly_config(
            moral_parser,
            moral_algebra,
            value_evaluator,
            harmonies_integrator,
            MoralAnomalyConfig::default(),
        )
    }

    /// Create a new ethics engine with custom anomaly detection thresholds.
    pub fn with_anomaly_config(
        moral_parser: MoralParser,
        moral_algebra: MoralAlgebra,
        value_evaluator: Option<UnifiedValueEvaluator>,
        harmonies_integrator: Option<HarmoniesIntegrator>,
        anomaly_config: MoralAnomalyConfig,
    ) -> Self {
        let dim = moral_algebra.dim();
        let shared_basis = Arc::new(HarmonyBasis::new(dim));

        let moral_topology = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim,
                ..Default::default()
            },
            shared_basis.clone(),
            anomaly_config,
        );

        // Share basis with HarmoniesIntegrator only when dimensions match.
        let harmonies_integrator = harmonies_integrator.map(|hi| {
            if hi.config().dimension == dim {
                let config = hi.config().clone();
                HarmoniesIntegrator::with_basis(config, shared_basis.clone())
            } else {
                hi
            }
        });

        let interaction_matrix = HarmonyInteractionMatrix::from_basis(&shared_basis);

        Self {
            moral_parser,
            moral_algebra,
            value_evaluator,
            harmonies_integrator,
            moral_topology,
            interaction_matrix,
            cache: EthicsEngineCache {
                last_harmonies_approved: true,
                ..Default::default()
            },
        }
    }

    /// Evaluate ethics for the current cycle.
    ///
    /// Pipeline: moral parse → value gate → harmonies → unified verdict
    ///
    /// Each subsystem fires at its co-prime interval:
    /// - MoralParser + MoralAlgebra: every 7 cycles
    /// - UnifiedValueEvaluator: every 19 cycles
    /// - HarmoniesIntegrator: every 19 cycles
    pub fn evaluate(&mut self, input: &EthicsEngineInput) -> EthicsEngineOutput {
        let total_start = Instant::now();
        let mut confidence_delta: f32 = 0.0;
        let mut lr_factor: f32 = 1.0;

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 1: Moral Parser + Algebra — HDC-based text ethical analysis
        // Every 7 cycles (co-prime)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let (
            moral_score,
            moral_verdict,
            deontological_verdict,
            consent_violation,
            moral_confidence,
            violations,
            satisfactions,
        ) = if input.cycle % 7 == 0 && input.cycle > 0 {
            let encoded = self
                .moral_parser
                .parse_and_encode(input.input, &self.moral_algebra);

            // Feed action HV into moral topology sliding window.
            // When the parser extracts a full agent-action-patient structure,
            // use the precise action_hv. Otherwise, fall back to encoding
            // the raw input text as an action — lower quality but ensures the
            // topology always gets data to analyze.
            let scenario_hv = encoded
                .action_hv
                .clone()
                .unwrap_or_else(|| self.moral_algebra.encode_action(input.input));
            self.moral_topology.add_scenario(scenario_hv);

            let (verdict_str, good_sim, bad_sim) =
                if let Some(judgment) = encoded.judge(&self.moral_algebra) {
                    let v = match judgment.verdict {
                        MoralVerdict::Good => "Good",
                        MoralVerdict::Bad => "Bad",
                        MoralVerdict::Neutral => "Neutral",
                        MoralVerdict::ConsentViolation => "ConsentViolation",
                    };
                    (
                        v.to_string(),
                        judgment.good_similarity,
                        judgment.bad_similarity,
                    )
                } else {
                    ("Neutral".to_string(), 0.0, 0.0)
                };

            let deont = self.moral_algebra.judge_deontological(input.input);
            let deont_verdict_str = match deont.verdict {
                DeontologicalVerdict::RightDutyFulfilled => "Permissible",
                DeontologicalVerdict::WrongPerfectDutyViolated => "Impermissible",
                DeontologicalVerdict::WrongImperfectDutyViolated => "Impermissible",
                DeontologicalVerdict::Neutral => "Neutral",
            }
            .to_string();

            let viols: Vec<String> = deont
                .violations
                .iter()
                .map(|v| v.rule_name.clone())
                .collect();
            let sats: Vec<String> = deont
                .satisfactions
                .iter()
                .map(|s| s.rule_name.clone())
                .collect();

            let cv = encoded.is_consent_violation();
            let score: f64 = if cv {
                -0.8
            } else {
                let base_score = (good_sim - bad_sim).clamp(-1.0, 1.0) as f64;
                let deont_factor = deont.score.clamp(-1.0, 1.0) as f64;
                (base_score * 0.6 + deont_factor * 0.4).clamp(-1.0, 1.0)
            };
            let confidence: f64 = encoded.parsed.confidence as f64;

            self.cache.last_moral_score = score;
            (
                score,
                verdict_str,
                deont_verdict_str,
                cv,
                confidence,
                viols,
                sats,
            )
        } else {
            (
                self.cache.last_moral_score,
                String::new(),
                String::new(),
                false,
                0.0,
                Vec::new(),
                Vec::new(),
            )
        };
        let moral_us = t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 2: Value Evaluator — consciousness-aware Allow/Warn/Veto
        // Every 19 cycles (co-prime)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let (value_score, value_decision, value_gate_factor) =
            if let Some(ref mut evaluator) = self.value_evaluator {
                if input.cycle % 19 == 0 && input.cycle > 0 {
                    let ctx = EvaluationContext {
                        consciousness_level: input.unified_psi,
                        ..Default::default()
                    };
                    let result = evaluator.evaluate("cognitive_cycle", ctx);
                    let decision_str = match &result.decision {
                        Decision::Allow => "Allow",
                        Decision::Warn(_) => "Warn",
                        Decision::Veto(_) => "Veto",
                    };
                    self.cache.last_value_score = result.overall_score;
                    (result.overall_score, decision_str.to_string(), 1.0f32)
                } else {
                    (self.cache.last_value_score, String::new(), 1.0f32)
                }
            } else {
                (0.0, String::new(), 1.0f32)
            };
        let value_us = t.elapsed().as_micros() as u64;

        // Value evaluator feedback: Veto → drastic LR reduction
        let value_gate_factor = if value_decision == "Veto" {
            lr_factor *= 0.1;
            0.1
        } else if value_score > 0.7 && !value_decision.is_empty() {
            let boost = 1.0 + (value_score as f32 - 0.7) * 0.15;
            lr_factor *= boost;
            boost
        } else {
            value_gate_factor
        };

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 3: Harmonies Integrator — Seven Harmonies alignment
        // Every 19 cycles (co-prime with value evaluator — same cadence)
        //
        // Now uses semantically grounded basis vectors (not random) and
        // computes moral free energy (FEP) on the 8D harmony manifold.
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let (harmonies_alignment, harmonies_approved, harmony_coordinates, moral_free_energy) =
            if let Some(ref mut integrator) = self.harmonies_integrator {
                if input.cycle % 19 == 0 && input.cycle > 0 {
                    let embedding = ContinuousHV::from_slice(input.compressed_state);
                    let action =
                        ValuedAction::new(format!("cycle_{}", input.cycle), input.input, embedding);
                    let eval = integrator.evaluate(&action);
                    self.cache.last_harmonies_alignment = eval.overall_alignment;
                    self.cache.last_harmonies_approved = eval.approved;
                    self.cache.last_harmony_coordinates = eval.harmony_coordinates;
                    self.cache.last_moral_free_energy = eval.moral_free_energy.clone();
                    // EMA-smooth the moral free energy (α=0.1)
                    self.cache.moral_fe_ema =
                        self.cache.moral_fe_ema * 0.9 + eval.moral_free_energy.free_energy * 0.1;
                    (
                        eval.overall_alignment,
                        eval.approved,
                        eval.harmony_coordinates,
                        eval.moral_free_energy,
                    )
                } else {
                    (
                        self.cache.last_harmonies_alignment,
                        self.cache.last_harmonies_approved,
                        self.cache.last_harmony_coordinates,
                        self.cache.last_moral_free_energy.clone(),
                    )
                }
            } else {
                (0.0, true, [0.0; N_HARMONIES], MoralFreeEnergy::default())
            };
        let harmonies_us = t.elapsed().as_micros() as u64;

        // Sacred Stillness neuromod grounding: GABA + adenosine boost SS coordinate
        // Index 7 = SacredStillness in Harmony::all() canonical order.
        // Science: Bhatt et al. (2020) — GABAergic tone correlates with resting-state activity;
        // Porkka-Heiskanen et al. (1997) — adenosine accumulation signals rest need.
        let mut harmony_coordinates = harmony_coordinates;
        if input.stillness_boost > 0.0 {
            harmony_coordinates[7] =
                (harmony_coordinates[7] + input.stillness_boost as f64).clamp(-1.0, 1.0);
        }

        // Harmony interaction matrix: observe co-activations and apply synergies
        self.interaction_matrix.observe(&harmony_coordinates, 0.05);
        harmony_coordinates = self.interaction_matrix.apply(&harmony_coordinates, 0.15);

        // Harmonies feedback: low alignment → confidence reduction
        if harmonies_alignment > 0.0 && !harmonies_approved {
            confidence_delta -= 0.02;
        }

        // High moral free energy → moral surprise → slight confidence reduction
        if moral_free_energy.free_energy > 2.0 {
            confidence_delta -= 0.01;
        }

        // ═══════════════════════════════════════════════════════════════════
        // UNIFIED VERDICT: Combine all systems into single ethical judgment
        // ═══════════════════════════════════════════════════════════════════
        let unified_verdict = if consent_violation || value_decision == "Veto" {
            EthicalVerdict::Blocked
        } else if moral_score < -0.3 || value_decision == "Warn" || !harmonies_approved {
            EthicalVerdict::Caution
        } else {
            EthicalVerdict::Safe
        };

        let unified_confidence = if moral_confidence > 0.0 {
            moral_confidence
        } else {
            // When moral parser hasn't fired this cycle, use value evaluator as proxy
            value_score.clamp(0.0, 1.0)
        };

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 4: Moral Topology — persistent homology (every 97 cycles)
        //
        // When fresh topology analysis is available, feed harmony variance
        // and dominant axis back into Stage 3 to adaptively reweight the
        // harmony dimensions (close moral blind spots, prevent fixation).
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let cycles_since = input.cycle.saturating_sub(self.cache.last_topology_cycle);
        let (latest_summary, topology_fresh) =
            if cycles_since >= self.cache.topology_cadence && self.moral_topology.len() >= 3 {
                let assessment = self.moral_topology.analyze();

                // Feed topology back into harmonies integrator weights
                if let Some(ref mut integrator) = self.harmonies_integrator {
                    integrator.apply_topology_feedback_with_kl(
                        &assessment.harmony_variance,
                        assessment.dominant_harmony_idx,
                        assessment.completeness,
                        assessment.moral_free_energy.kl_divergence,
                    );
                }

                // Adapt cadence based on moral drift (thresholds from anomaly_config)
                let drift = self.moral_topology.moral_drift(20);
                let ac = self.moral_topology.anomaly_config();
                self.cache.topology_cadence = if drift > ac.cadence_drift_high {
                    ac.cadence_fast // fast: moral stance shifting rapidly
                } else if drift > ac.cadence_drift_moderate {
                    ac.cadence_moderate // moderate
                } else {
                    ac.cadence_slow // slow: stable moral stance, save compute
                };
                self.cache.last_topology_cycle = input.cycle;

                (MoralTopologySummary::from(&assessment), true)
            } else {
                (self.moral_topology.last_summary().clone(), false)
            };
        let topology_us = t.elapsed().as_micros() as u64;

        // Compute anomaly report against latest topology summary
        let anomaly_report = self.moral_topology.detect_anomalies(&latest_summary);
        self.cache.last_anomaly_report = anomaly_report.clone();
        self.cache.last_topology_fresh = topology_fresh;

        let total_us = total_start.elapsed().as_micros() as u64;

        // Clamp lr_factor
        lr_factor = lr_factor.clamp(0.1, 1.3);

        EthicsEngineOutput {
            moral_score,
            moral_verdict,
            deontological_verdict,
            consent_violation,
            moral_confidence,
            violations,
            satisfactions,
            value_score,
            value_decision,
            value_gate_factor,
            harmonies_alignment,
            harmonies_approved,
            unified_verdict,
            unified_confidence,
            confidence_delta,
            lr_factor,
            anomaly_report,
            harmony_coordinates,
            moral_free_energy,
            topology_us,
            topology_fresh,
            moral_us,
            value_us,
            harmonies_us,
            total_us,
        }
    }

    /// Evaluate moral alignment of input text (Stage 1 only).
    ///
    /// Used by `CognitiveLoopService::evaluate_moral_alignment()` to delegate
    /// moral parsing through the engine's owned parser/algebra without running
    /// the full pipeline (Stages 2+3 need `compressed_state` not yet available).
    pub fn evaluate_moral_input(&mut self, input: &str) -> MoralEvalResult {
        let encoded = self
            .moral_parser
            .parse_and_encode(input, &self.moral_algebra);

        let (verdict, good_sim, bad_sim) =
            if let Some(judgment) = encoded.judge(&self.moral_algebra) {
                let v = match judgment.verdict {
                    MoralVerdict::Good => "Good",
                    MoralVerdict::Bad => "Bad",
                    MoralVerdict::Neutral => "Neutral",
                    MoralVerdict::ConsentViolation => "ConsentViolation",
                };
                (
                    v.to_string(),
                    judgment.good_similarity,
                    judgment.bad_similarity,
                )
            } else {
                ("Neutral".to_string(), 0.0, 0.0)
            };

        let input_lower = input.to_lowercase();
        let deont = self
            .moral_algebra
            .judge_deontological_pre_lowered(&input_lower);
        let deontological_verdict = match deont.verdict {
            DeontologicalVerdict::RightDutyFulfilled => "Permissible",
            DeontologicalVerdict::WrongPerfectDutyViolated => "Impermissible",
            DeontologicalVerdict::WrongImperfectDutyViolated => "Impermissible",
            DeontologicalVerdict::Neutral => "Neutral",
        }
        .to_string();

        let violations: Vec<String> = deont
            .violations
            .iter()
            .map(|v| v.rule_name.clone())
            .collect();
        let satisfactions: Vec<String> = deont
            .satisfactions
            .iter()
            .map(|s| s.rule_name.clone())
            .collect();
        let consent_violation = encoded.is_consent_violation();
        let moral_score = if consent_violation {
            -0.8
        } else {
            let base_score = (good_sim - bad_sim).clamp(-1.0, 1.0) as f64;
            let deont_factor = deont.score.clamp(-1.0, 1.0) as f64;
            (base_score * 0.6 + deont_factor * 0.4).clamp(-1.0, 1.0)
        };
        let confidence = encoded.parsed.confidence;

        // Update cache for the full pipeline evaluate() call
        self.cache.last_moral_score = moral_score;

        MoralEvalResult {
            verdict,
            deontological_verdict,
            violations,
            satisfactions,
            consent_violation,
            moral_score,
            confidence,
        }
    }

    /// Borrow the value evaluator (if present).
    pub fn value_evaluator(
        &self,
    ) -> Option<&crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator> {
        self.value_evaluator.as_ref()
    }

    /// Borrow the harmonies integrator (if present).
    pub fn harmonies_integrator(
        &self,
    ) -> Option<&crate::consciousness::harmonies_integration::HarmoniesIntegrator> {
        self.harmonies_integrator.as_ref()
    }

    /// Access cached value score.
    pub fn last_value_score(&self) -> f64 {
        self.cache.last_value_score
    }

    /// Access cached harmonies alignment.
    pub fn last_harmonies_alignment(&self) -> f32 {
        self.cache.last_harmonies_alignment
    }

    /// Access cached harmonies approved status.
    pub fn last_harmonies_approved(&self) -> bool {
        self.cache.last_harmonies_approved
    }

    /// Access the moral topology analyser.
    pub fn moral_topology(&self) -> &MoralTopology {
        &self.moral_topology
    }

    /// Cached harmony coordinates from last harmonies evaluation.
    pub fn last_harmony_coordinates(&self) -> &[f64; N_HARMONIES] {
        &self.cache.last_harmony_coordinates
    }

    /// Cached moral free energy from last harmonies evaluation.
    pub fn last_moral_free_energy(&self) -> &MoralFreeEnergy {
        &self.cache.last_moral_free_energy
    }

    /// EMA-smoothed moral free energy.
    pub fn moral_fe_ema(&self) -> f64 {
        self.cache.moral_fe_ema
    }

    /// Current adaptive gain for moral FE → exploration coupling.
    pub fn moral_exploration_gain(&self) -> f32 {
        self.cache.moral_exploration_gain
    }

    /// Cached anomaly report from the last `evaluate()` call.
    pub fn last_anomaly_report(&self) -> &MoralAnomalyReport {
        &self.cache.last_anomaly_report
    }

    /// Whether the last `evaluate()` freshly computed topology (vs returning cached).
    pub fn last_topology_fresh(&self) -> bool {
        self.cache.last_topology_fresh
    }

    /// Get the current learned harmony interaction weights for serialization.
    ///
    /// Returns the 8x8 matrix of synergy/tension weights learned from observed
    /// co-activation patterns. Persist these across sleep/wake cycles so the
    /// engine does not lose learned moral synergies on restart.
    pub fn interaction_matrix_weights(&self) -> &[[f64; N_HARMONIES]; N_HARMONIES] {
        &self.interaction_matrix.weights
    }

    /// Restore learned harmony interaction weights from a previous session.
    ///
    /// After restoring, the observation count is inferred from the number of
    /// non-default off-diagonal weights (those that differ from both 0.0 and 1.0),
    /// giving the engine a rough sense of how much learning has occurred.
    pub fn restore_interaction_matrix(&mut self, weights: [[f64; N_HARMONIES]; N_HARMONIES]) {
        self.interaction_matrix.weights = weights;
        // Recompute observation count from non-default weights.
        // Default off-diagonal = 0.0 (from Default impl) or semantic similarity
        // (from from_basis). Either way, weights that deviate from both 0.0 and
        // 1.0 indicate learning has occurred.
        let non_default = weights
            .iter()
            .flatten()
            .filter(|&&w| (w - 0.0).abs() > 1e-10 && (w - 1.0).abs() > 1e-10)
            .count();
        self.interaction_matrix.observation_count = non_default as u64;
    }

    /// Bidirectional feedback: exploration outcome modulates FE→exploration gain.
    ///
    /// After an FE-driven exploration boost, the prediction error outcome tells
    /// us whether the exploration was productive:
    /// - Low PE (< baseline) → exploration resolved moral uncertainty → reinforce gain
    /// - High PE (> baseline) → exploration unhelpful → decay gain
    ///
    /// Science: Daw et al. (2005) model-based/model-free arbitration;
    /// Friston (2010) expected free energy minimization.
    pub fn feedback_exploration_outcome(&mut self, pe: f32, baseline_pe: f32) {
        let delta = pe - baseline_pe;
        if delta < -0.05 {
            // Exploration reduced PE → reinforce FE-exploration coupling
            self.cache.moral_exploration_gain =
                (self.cache.moral_exploration_gain + 0.01).min(0.25);
        } else if delta > 0.1 {
            // Exploration increased PE → decay coupling (diminishing returns)
            self.cache.moral_exploration_gain =
                (self.cache.moral_exploration_gain - 0.02).max(0.05);
        }
        // Small deltas: no change (exploration neutral)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> EthicsEngine {
        let parser = MoralParser::new();
        let algebra = MoralAlgebra::new(16384);
        EthicsEngine::new(parser, algebra, None, None)
    }

    fn make_input(cycle: u64) -> EthicsEngineInput<'static> {
        EthicsEngineInput {
            input: "helping others is good",
            cycle,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = make_engine();
        assert!(engine.value_evaluator.is_none());
        assert!(engine.harmonies_integrator.is_none());
    }

    #[test]
    fn test_moral_fires_at_interval_7() {
        let mut engine = make_engine();

        // At cycle 7, moral parser should fire
        let input = make_input(7);
        let output = engine.evaluate(&input);

        // Moral verdict should be populated
        assert!(
            !output.moral_verdict.is_empty(),
            "Moral verdict should be populated at cycle 7"
        );
        assert!(output.moral_score.is_finite());
    }

    #[test]
    fn test_moral_caches_between_cycles() {
        let mut engine = make_engine();

        // Fire at cycle 7
        let input = make_input(7);
        let output7 = engine.evaluate(&input);
        let score_at_7 = output7.moral_score;

        // At cycle 8 (not firing), should use cached value
        let input = make_input(8);
        let output8 = engine.evaluate(&input);
        assert!(
            (output8.moral_score - score_at_7).abs() < f64::EPSILON,
            "Non-firing cycle should use cached moral score"
        );
    }

    #[test]
    fn test_consent_violation_triggers_blocked() {
        let mut engine = make_engine();

        // Use input that the moral parser would recognize as consent-related
        // (The exact triggering depends on the parser's rules)
        let input = EthicsEngineInput {
            input: "forcing someone against their will without consent",
            cycle: 7,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
        };
        let output = engine.evaluate(&input);

        // If consent violation was detected, verdict should be Blocked
        if output.consent_violation {
            assert_eq!(output.unified_verdict, EthicalVerdict::Blocked);
        }
    }

    #[test]
    fn test_safe_input_produces_safe_verdict() {
        let mut engine = make_engine();

        let input = EthicsEngineInput {
            input: "helping others learn is a joy",
            cycle: 7,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
        };
        let output = engine.evaluate(&input);

        // Without value evaluator or harmonies, no veto/warn possible
        // So verdict should be Safe or Neutral based on moral score
        assert_ne!(
            output.unified_verdict,
            EthicalVerdict::Blocked,
            "Benign input should not be blocked"
        );
    }

    #[test]
    fn test_value_evaluator_integration() {
        let parser = MoralParser::new();
        let algebra = MoralAlgebra::new(16384);
        let evaluator = UnifiedValueEvaluator::default();
        let mut engine = EthicsEngine::new(parser, algebra, Some(evaluator), None);

        // At cycle 19, value evaluator fires
        let input = EthicsEngineInput {
            input: "learning about the world",
            cycle: 19,
            unified_psi: 0.6,
            compressed_state: &[0.5; 256],
            stillness_boost: 0.0,
        };
        let output = engine.evaluate(&input);

        // Value decision should be populated
        assert!(
            !output.value_decision.is_empty(),
            "Value decision should be populated at cycle 19"
        );
        assert!(output.value_score.is_finite());
    }

    #[test]
    fn test_timing_fields_populated() {
        let mut engine = make_engine();
        let input = make_input(7);
        let output = engine.evaluate(&input);

        assert!(output.total_us > 0);
        assert!(output.total_us < 1_000_000);
    }

    #[test]
    fn test_unified_verdict_variants() {
        // Test that all verdict variants are constructible
        assert_ne!(EthicalVerdict::Safe, EthicalVerdict::Caution);
        assert_ne!(EthicalVerdict::Caution, EthicalVerdict::Blocked);
        assert_ne!(EthicalVerdict::Safe, EthicalVerdict::Blocked);
    }

    #[test]
    fn test_adaptive_topology_cadence() {
        use crate::hdc::ContinuousHV;

        let mut engine = make_engine();

        // Default cadence should be 97
        assert_eq!(engine.cache.topology_cadence, 97);
        assert_eq!(engine.cache.last_topology_cycle, 0);

        // Directly inject scenarios to guarantee topology has enough data.
        // The moral parser may not always produce action_hv from text, so
        // we ensure the topology window is populated.
        for i in 0..10 {
            engine
                .moral_topology
                .add_scenario(ContinuousHV::random(16384, 100 + i));
        }
        assert!(
            engine.moral_topology.len() >= 3,
            "Should have at least 3 scenarios"
        );

        // Run enough cycles to exceed initial cadence (97)
        for c in 0..200 {
            let input = make_input(c);
            engine.evaluate(&input);
        }

        // After enough cycles, topology should have fired at least once
        assert!(
            engine.cache.last_topology_cycle > 0,
            "Topology should have fired at least once (len={}, cadence={})",
            engine.moral_topology.len(),
            engine.cache.topology_cadence
        );

        // With stable benign input, drift should be low → cadence should widen
        assert!(
            engine.cache.topology_cadence >= 60,
            "Stable input should keep cadence at 60+ (got {})",
            engine.cache.topology_cadence
        );
    }

    #[test]
    fn test_moral_fe_exploration_feedback_reinforces_on_low_pe() {
        let mut engine = make_engine();
        let initial_gain = engine.moral_exploration_gain();
        assert!((initial_gain - 0.15).abs() < f32::EPSILON);

        // Simulate successful exploration: PE well below baseline
        engine.feedback_exploration_outcome(0.1, 0.3); // delta = -0.2
        assert!(
            engine.moral_exploration_gain() > initial_gain,
            "Gain should increase on low PE: {}",
            engine.moral_exploration_gain()
        );
    }

    #[test]
    fn test_moral_fe_exploration_feedback_decays_on_high_pe() {
        let mut engine = make_engine();
        let initial_gain = engine.moral_exploration_gain();

        // Simulate failed exploration: PE well above baseline
        engine.feedback_exploration_outcome(0.6, 0.3); // delta = +0.3
        assert!(
            engine.moral_exploration_gain() < initial_gain,
            "Gain should decrease on high PE: {}",
            engine.moral_exploration_gain()
        );
    }

    #[test]
    fn test_moral_fe_exploration_gain_clamped() {
        let mut engine = make_engine();

        // Repeated reinforcement should not exceed 0.25
        for _ in 0..50 {
            engine.feedback_exploration_outcome(0.05, 0.3);
        }
        assert!(
            engine.moral_exploration_gain() <= 0.25,
            "Gain should be clamped at 0.25: {}",
            engine.moral_exploration_gain()
        );

        // Repeated decay should not go below 0.05
        for _ in 0..50 {
            engine.feedback_exploration_outcome(0.8, 0.3);
        }
        assert!(
            engine.moral_exploration_gain() >= 0.05,
            "Gain should be clamped at 0.05: {}",
            engine.moral_exploration_gain()
        );
    }

    #[test]
    fn test_interaction_matrix_roundtrip() {
        // Create an engine and observe some patterns to mutate the matrix.
        let mut engine = make_engine();
        for cycle in (7..=133).step_by(7) {
            let input = make_input(cycle as u64);
            engine.evaluate(&input);
        }

        // Extract the learned weights.
        let saved_weights = *engine.interaction_matrix_weights();

        // Create a fresh engine — its matrix will be from_basis (semantic init).
        let mut engine2 = make_engine();
        let fresh_weights = *engine2.interaction_matrix_weights();

        // The two matrices should differ (engine1 had observations applied).
        let diff: f64 = saved_weights
            .iter()
            .flatten()
            .zip(fresh_weights.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "After observations, weights should differ from fresh init (diff={diff})"
        );

        // Restore saved weights into engine2.
        engine2.restore_interaction_matrix(saved_weights);
        let restored_weights = *engine2.interaction_matrix_weights();

        // Restored weights should exactly match saved weights.
        for i in 0..N_HARMONIES {
            for j in 0..N_HARMONIES {
                assert!(
                    (restored_weights[i][j] - saved_weights[i][j]).abs() < f64::EPSILON,
                    "Weight [{i}][{j}] mismatch after restore: {} vs {}",
                    restored_weights[i][j],
                    saved_weights[i][j]
                );
            }
        }

        // Observation count should be non-zero after restore (inferred from non-default weights).
        assert!(
            engine2.interaction_matrix.observation_count > 0,
            "Observation count should be inferred from restored weights"
        );
    }
}
