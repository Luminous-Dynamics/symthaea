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
use crate::hdc::harmony_basis::{HarmonyBasis, MoralFreeEnergy};
use crate::hdc::moral_algebra::{DeontologicalVerdict, MoralAlgebra, MoralVerdict};
use crate::hdc::moral_parser::MoralParser;
use crate::hdc::moral_topology::{MoralTopology, MoralTopologyConfig, MoralTopologySummary};

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
    /// Compact topology summary (default when analysis not run this cycle).
    #[allow(dead_code)] // Constructed by engine; read via ethics_engine.moral_topology() accessor
    pub topology_summary: MoralTopologySummary,
    /// Microseconds spent on topology analysis (0 when not run).
    pub topology_us: u64,
    /// Whether topology analysis was freshly computed this cycle.
    #[allow(dead_code)] // Constructed by engine; read via ethics_engine.moral_topology() accessor
    pub topology_fresh: bool,

    // ── Stage 3b: Moral Geometry (FEP) ─────────────────────────────────
    /// 7D harmony coordinates for this cycle's action
    #[allow(dead_code)] // Computed by harmonies integrator; read via engine cache
    pub harmony_coordinates: [f64; 7],
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

    // ── Cached values ──────────────────────────────────────────────────
    cache: EthicsEngineCache,
}

#[derive(Debug, Clone, Default)]
struct EthicsEngineCache {
    last_moral_score: f64,
    last_value_score: f64,
    last_harmonies_alignment: f32,
    last_harmonies_approved: bool,
    last_harmony_coordinates: [f64; 7],
    last_moral_free_energy: MoralFreeEnergy,
}

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
        let dim = moral_algebra.dim();
        let shared_basis = Arc::new(HarmonyBasis::new(dim));

        let moral_topology = MoralTopology::with_basis(
            MoralTopologyConfig {
                dim,
                ..Default::default()
            },
            shared_basis.clone(),
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

        Self {
            moral_parser,
            moral_algebra,
            value_evaluator,
            harmonies_integrator,
            moral_topology,
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

            // Feed action HV into moral topology sliding window
            if let Some(ref hv) = encoded.action_hv {
                self.moral_topology.add_scenario(hv.clone());
            }

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
        // computes moral free energy (FEP) on the 7D harmony manifold.
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
                (0.0, true, [0.0; 7], MoralFreeEnergy::default())
            };
        let harmonies_us = t.elapsed().as_micros() as u64;

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
        let (topology_summary, topology_fresh) =
            if input.cycle % 97 == 0 && self.moral_topology.len() >= 3 {
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

                (MoralTopologySummary::from(&assessment), true)
            } else {
                (self.moral_topology.last_summary().clone(), false)
            };
        let topology_us = t.elapsed().as_micros() as u64;

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
            harmony_coordinates,
            moral_free_energy,
            topology_summary,
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
        let deont = self.moral_algebra.judge_deontological_pre_lowered(&input_lower);
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
    pub fn last_harmony_coordinates(&self) -> &[f64; 7] {
        &self.cache.last_harmony_coordinates
    }

    /// Cached moral free energy from last harmonies evaluation.
    pub fn last_moral_free_energy(&self) -> &MoralFreeEnergy {
        &self.cache.last_moral_free_energy
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
}
