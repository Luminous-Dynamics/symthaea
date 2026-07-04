// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Ethics Engine
//!
//! Wraps the independent ethics systems into a single coherent engine
//! with clear data flow and unified output.
//!
//! ## Systems (Pipeline Architecture)
//!
//! | Stage | System | Interval | Science |
//! |-------|--------|----------|---------|
//! | 1 | MoralParser + MoralAlgebra | 7 cycles | HDC moral algebra (Luo & Lakoff 2019) |
//! | 2 | UnifiedValueEvaluator | 19 cycles | Value alignment (Panksepp 1998) |
//! | 3 | HarmoniesIntegrator | 19 cycles | Eight Harmonies (Schwartz 2012) |
//! | 4 | MoralTopology | adaptive | Persistent homology (Carlsson 2009) |
//! | 5 | InstitutionalComplianceCheck | 23 cycles | HDC institutional primitives |
//!
//! ## Stage 5: Institutional Compliance Check
//!
//! Encodes regulatory/jurisdictional constraints as HDC vectors and computes
//! similarity against action embeddings. When an action crosses a jurisdictional
//! boundary or triggers a regulatory keyword, the compliance check produces
//! `compliance_flags` (e.g., "requires_gdpr", "sanctions_risk") and a
//! `compliance_risk` score that can escalate the unified verdict to Caution.
//!
//! Nation-states are **not** hardcoded entities — they are composite HDC
//! vectors derived from the PrimitiveSystem's institutional domain (SOVEREIGNTY
//! ⊗ INSTITUTION ⊗ ENFORCEMENT ⊗ POPULATION). The compliance check works by
//! projecting action text onto these institutional vectors and measuring
//! similarity to known constraint patterns.
//!
//! ## Design Principles
//!
//! 1. **Pipeline**: moral parse → value gate → harmonies check → topology → compliance → unified verdict
//! 2. **No direct field mutation**: Returns `EthicsEngineOutput` with proposed deltas
//! 3. **Preserves co-prime intervals**: Each subsystem fires at its original rate
//! 4. **Backward compatible**: All existing carryover fields populated

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

use crate::consciousness::harmonies_integration::{HarmoniesIntegrator, ValuedAction};
use crate::consciousness::unified_value_evaluator::{
    Decision, EvaluationContext, UnifiedValueEvaluator,
};
use crate::hdc::harmony_basis::{HarmonyBasis, HarmonyInteractionMatrix, MoralFreeEnergy};
use crate::hdc::moral_algebra::{DeontologicalVerdict, MoralAlgebra, MoralVerdict};
use crate::hdc::moral_parser::MoralParser;
use crate::hdc::moral_topology::{
    MoralAnomalyConfig, MoralAnomalyReport, MoralTopology, MoralTopologyConfig,
    MoralTopologySummary,
};
use symthaea_types::N_HARMONIES;

// ═══════════════════════════════════════════════════════════════════════════════
// RESTORATIVE JUSTICE TRACKER
//
// Instead of permanent punishment after a veto, the system can "earn back"
// trust by demonstrating sustained corrective behavior. Inspired by Ubuntu
// (Southern Africa) and Navajo Peacemaking — restoration over retribution.
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks post-veto corrective behavior for restorative justice.
///
/// Instead of permanent punishment, the system can "earn back" trust by
/// demonstrating sustained corrective behavior after a veto. Inspired by
/// Ubuntu (Southern Africa) and Navajo Peacemaking — restoration over retribution.
///
/// The tracker maintains a per-violation restoration score that accumulates
/// when subsequent cycles satisfy the violated obligation. After enough
/// evidence of correction, `is_restored()` returns true.
#[derive(Debug, Clone, Default)]
pub struct RestorationTracker {
    /// Active restoration entries: violation_name -> RestorationEntry
    entries: HashMap<String, RestorationEntry>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // violation_cycle recorded for future audit/diagnostics
struct RestorationEntry {
    /// Cycle when the violation occurred.
    violation_cycle: u64,
    /// Number of subsequent cycles where the obligation was satisfied.
    corrective_cycles: u32,
    /// Number of subsequent cycles where the obligation was violated again.
    relapse_cycles: u32,
    /// Required corrective cycles for restoration (default: 10).
    required_corrections: u32,
}

impl RestorationTracker {
    /// Record a new violation. Starts or resets the restoration window.
    pub fn record_violation(&mut self, violation_name: &str, cycle: u64) {
        self.entries.insert(
            violation_name.to_string(),
            RestorationEntry {
                violation_cycle: cycle,
                corrective_cycles: 0,
                relapse_cycles: 0,
                required_corrections: 10,
            },
        );
    }

    /// Record a satisfaction of a previously-violated obligation.
    pub fn record_correction(&mut self, obligation_name: &str) {
        if let Some(entry) = self.entries.get_mut(obligation_name) {
            entry.corrective_cycles += 1;
        }
    }

    /// Record a relapse (re-violation during restoration).
    pub fn record_relapse(&mut self, violation_name: &str) {
        if let Some(entry) = self.entries.get_mut(violation_name) {
            entry.relapse_cycles += 1;
            // Reset corrective progress on relapse
            entry.corrective_cycles = entry.corrective_cycles.saturating_sub(3);
        }
    }

    /// Check if a specific violation has been sufficiently corrected.
    pub fn is_restored(&self, violation_name: &str) -> bool {
        self.entries.get(violation_name).map_or(true, |e| {
            e.corrective_cycles >= e.required_corrections && e.relapse_cycles == 0
        })
    }

    /// Get all violations that have been restored.
    pub fn restored_violations(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|(_, e)| e.corrective_cycles >= e.required_corrections && e.relapse_cycles == 0)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Remove restored entries (garbage collection).
    pub fn clear_restored(&mut self) {
        self.entries.retain(|_, e| {
            !(e.corrective_cycles >= e.required_corrections && e.relapse_cycles == 0)
        });
    }

    /// Number of active (unrestored) violations.
    pub fn active_violations(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, e)| e.corrective_cycles < e.required_corrections || e.relapse_cycles > 0)
            .count()
    }

    /// Overall restoration progress as a fraction [0.0, 1.0].
    pub fn restoration_progress(&self) -> f64 {
        if self.entries.is_empty() {
            return 1.0; // no violations = fully restored
        }
        let total: f64 = self
            .entries
            .values()
            .map(|e| {
                if e.required_corrections > 0 {
                    (e.corrective_cycles as f64 / e.required_corrections as f64).min(1.0)
                } else {
                    0.0
                }
            })
            .sum();
        total / self.entries.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STAGE 5: Institutional Compliance Checker
//
// Uses HDC institutional primitives from PrimitiveSystem to detect when actions
// cross jurisdictional boundaries or trigger regulatory constraints. Nation-states
// are composite hypervectors, NOT hardcoded entities.
// ═══════════════════════════════════════════════════════════════════════════════

/// Specification for loading an external regulatory constraint.
///
/// Used by `InstitutionalComplianceChecker::load_external_constraints()` to
/// add jurisdiction-specific constraints at runtime (e.g., from Mycelix DHT).
#[derive(Debug, Clone)]
pub(crate) struct ExternalConstraintSpec {
    /// Human-readable flag name (e.g., "gdpr_data_residency")
    pub flag: String,
    /// Names of primitives/compositions to XOR-bind into the constraint encoding.
    /// E.g., ["JURISDICTION", "COMPLIANCE"] → JURISDICTION ⊗ COMPLIANCE
    pub primitive_names: Vec<String>,
    /// Similarity threshold override (defaults to 0.53 if None)
    pub threshold: Option<f32>,
}

/// A regulatory constraint encoded as an HDC pattern.
///
/// Each constraint binds a jurisdiction primitive (e.g., SOVEREIGNTY ⊗ BOUNDARY)
/// with domain-specific keywords to create a detector vector. When an action's
/// HDC encoding is similar to a constraint vector, the flag is raised.
#[derive(Debug, Clone)]
struct RegulatoryConstraint {
    /// Human-readable flag name (e.g., "data_sovereignty", "sanctions_risk")
    flag: String,
    /// HDC encoding of this constraint pattern
    encoding: BinaryHV,
    /// Similarity threshold above which this constraint is triggered.
    /// Higher = more specific match required.
    threshold: f32,
}

/// Institutional compliance checker using HDC primitive composition.
///
/// Encodes regulatory/jurisdictional constraints as composite BinaryHV vectors
/// derived from the PrimitiveSystem's institutional domain primitives. Actions
/// are checked against these constraints via cosine similarity.
#[derive(Debug)]
struct InstitutionalComplianceChecker {
    /// Known regulatory constraints to check against
    constraints: Vec<RegulatoryConstraint>,
}

impl InstitutionalComplianceChecker {
    /// Build the compliance checker from the global PrimitiveSystem.
    ///
    /// Creates constraint detectors by composing institutional primitives:
    /// - Data sovereignty: JURISDICTION ⊗ COMPLIANCE (GDPR, data residency)
    /// - Sanctions risk: SANCTION ⊗ ENFORCEMENT (trade/financial sanctions)
    /// - Export control: MONOPOLY ⊗ ENFORCEMENT (controlled goods)
    /// - Tax obligation: TAXATION (compulsory value transfer)
    /// - Treaty obligation: TREATY ⊗ COMPLIANCE (international agreements)
    fn new() -> Self {
        let system = PrimitiveSystem::global();

        let mut constraints = Vec::new();

        // Data sovereignty: actions involving data + jurisdiction
        if let (Some(jurisdiction), Some(compliance)) =
            (system.get("JURISDICTION"), system.get("COMPLIANCE"))
        {
            constraints.push(RegulatoryConstraint {
                flag: "data_sovereignty".to_string(),
                encoding: jurisdiction.encoding.bind(&compliance.encoding),
                threshold: 0.53,
            });
        }

        // Sanctions risk: punitive constraints from authorities
        if let (Some(sanction), Some(enforcement)) =
            (system.get("SANCTION"), system.get("ENFORCEMENT"))
        {
            constraints.push(RegulatoryConstraint {
                flag: "sanctions_risk".to_string(),
                encoding: sanction.encoding.bind(&enforcement.encoding),
                threshold: 0.53,
            });
        }

        // Export control: monopoly on controlled goods + enforcement
        if let (Some(monopoly), Some(enforcement)) =
            (system.get("MONOPOLY"), system.get("ENFORCEMENT"))
        {
            constraints.push(RegulatoryConstraint {
                flag: "export_control".to_string(),
                encoding: monopoly.encoding.bind(&enforcement.encoding),
                threshold: 0.53,
            });
        }

        // Tax obligation: compulsory value transfer
        if let Some(taxation) = system.get("TAXATION") {
            constraints.push(RegulatoryConstraint {
                flag: "tax_obligation".to_string(),
                encoding: taxation.encoding,
                threshold: 0.53,
            });
        }

        // Treaty obligation: international agreements requiring compliance
        if let (Some(treaty), Some(compliance)) = (system.get("TREATY"), system.get("COMPLIANCE")) {
            constraints.push(RegulatoryConstraint {
                flag: "treaty_obligation".to_string(),
                encoding: treaty.encoding.bind(&compliance.encoding),
                threshold: 0.53,
            });
        }

        Self { constraints }
    }

    /// Check an action (as HDC-encoded BinaryHV) against all known constraints.
    ///
    /// Returns (risk_score, triggered_flags).
    /// Risk score is the maximum similarity across all constraints, clamped to [0, 1].
    fn check(&self, action_hv: &BinaryHV) -> (f32, Vec<String>) {
        let mut max_risk: f32 = 0.0;
        let mut flags = Vec::new();

        for constraint in &self.constraints {
            let sim = action_hv.similarity(&constraint.encoding);
            // Similarity is fraction of matching bits [0, 1].
            // 0.5 = orthogonal (random), >0.53 = meaningful overlap.
            if sim > constraint.threshold {
                let risk = ((sim - 0.5) * 4.0).clamp(0.0, 1.0);
                if risk > max_risk {
                    max_risk = risk;
                }
                flags.push(constraint.flag.clone());
            }
        }

        (max_risk, flags)
    }

    /// Load external regulatory constraints (e.g., from Mycelix JurisdictionConstraintEntry).
    ///
    /// Each constraint is defined by a flag name and a list of primitive/composition names
    /// that are XOR-bound together to create the constraint encoding.
    /// Returns the number of constraints successfully loaded (skips any with missing primitives).
    fn load_external_constraints(&mut self, constraints: &[ExternalConstraintSpec]) -> usize {
        let system = PrimitiveSystem::global();
        let mut loaded = 0;

        for spec in constraints {
            // Build encoding by XOR-binding all named primitives
            let mut encoding: Option<BinaryHV> = None;
            let mut all_found = true;

            for prim_name in &spec.primitive_names {
                if let Some(prim) = system.get(prim_name) {
                    encoding = Some(match encoding {
                        Some(existing) => existing.bind(&prim.encoding),
                        None => prim.encoding,
                    });
                } else {
                    all_found = false;
                    break;
                }
            }

            if all_found {
                if let Some(enc) = encoding {
                    self.constraints.push(RegulatoryConstraint {
                        flag: spec.flag.clone(),
                        encoding: enc,
                        threshold: spec.threshold.unwrap_or(0.53),
                    });
                    loaded += 1;
                }
            }
        }

        loaded
    }

    /// Clear all loaded constraints and reload from defaults.
    fn reload_defaults(&mut self) {
        *self = Self::new();
    }

    /// Encode action text as BinaryHV using simple n-gram hashing.
    ///
    /// This is a lightweight encoder for institutional compliance checking.
    /// The moral algebra's full text encoder is used for Stage 1; this
    /// provides a fast BinaryHV for constraint similarity checks.
    fn encode_text(&self, text: &str) -> BinaryHV {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        if words.is_empty() {
            return BinaryHV::zero();
        }

        let word_hvs: Vec<BinaryHV> = words
            .iter()
            .enumerate()
            .map(|(i, word)| {
                let word_hv =
                    BinaryHV::random(symthaea_core::hdc::primitive_system::seed_from_name(word));
                // Position-encode: permute by position to capture word order
                word_hv.permute(i)
            })
            .collect();

        BinaryHV::bundle(&word_hvs)
    }
}

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
    /// Whether any ahimsa-family obligation was violated.
    pub ahimsa_violated: bool,

    // ── Stage 2: Value Evaluator ───────────────────────────────────────
    /// Value alignment score [0, 1]
    pub value_score: f64,
    /// Value decision (Allow/Warn/Veto)
    pub value_decision: String,
    /// Learning rate gate factor from value evaluator
    pub value_gate_factor: f32,

    // ── Stage 3: Harmonies ─────────────────────────────────────────────
    /// Eight Harmonies alignment [0, 1]
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

    // ── Stage 3c: Love Coherence ─────────────────────────────────────
    /// Love coherence: emergent macro-state of all 8 harmonies in resonance [0, 1].
    /// High = system is simultaneously rigorous, playful, and co-creative.
    pub love_coherence: f64,

    // ── Stage 5: Institutional Compliance ──────────────────────────────
    /// Compliance risk score [0.0, 1.0]. 0 = no institutional constraints detected.
    pub compliance_risk: f32,
    /// Regulatory/jurisdictional flags triggered (e.g., "data_sovereignty", "sanctions_risk")
    pub compliance_flags: Vec<String>,
    /// Whether institutional compliance check was freshly computed this cycle.
    pub compliance_fresh: bool,

    // ── Restorative Justice ─────────────────────────────────────────────
    /// Overall restoration progress [0.0, 1.0]. 1.0 = no active violations or all restored.
    pub restoration_progress: f64,

    // ── Timing ─────────────────────────────────────────────────────────
    pub moral_us: u64,
    pub value_us: u64,
    pub harmonies_us: u64,
    pub compliance_us: u64,
    pub total_us: u64,
}

/// Unified ethical verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EthicalVerdict {
    /// All systems agree: action is ethical
    Safe,
    /// One or more systems flag concerns but don't block
    Caution,
    /// Value evaluator vetoes or consent violation detected
    Blocked,
}

impl EthicalVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Safe => "Safe",
            Self::Caution => "Caution",
            Self::Blocked => "Blocked",
        }
    }
}

impl Default for EthicalVerdict {
    fn default() -> Self {
        EthicalVerdict::Safe
    }
}

/// Input snapshot for the ethics engine.
#[allow(dead_code)] // compressed_state reserved for Stage 2+3 harmony integration
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
    /// Optional semantic embedding (e.g. Qwen3 1024D projected to 16,384D via HdcBridge).
    /// When present, used instead of N-gram TextHdcEncoder for moral topology scenarios,
    /// giving genuine semantic resolution to trajectory convergence detection.
    pub semantic_embedding: Option<&'a [f32]>,
    /// Pre-computed BinaryHV encoding of the input (from TextEncoder → real_hv_to_hv16).
    /// When present, Stage 5 compliance checker uses this instead of its own n-gram encoder,
    /// giving genuine semantic grounding to institutional constraint matching.
    pub action_hv: Option<&'a BinaryHV>,
    /// Knowledge-grounded confidence multiplier for moral evaluation (0.0–1.0).
    /// High-confidence knowledge boosts moral certainty; low-confidence dampens.
    pub knowledge_confidence_multiplier: f64,
    /// Moral context facts from the knowledge graph (e.g., "sanctions→suffering" chains).
    pub knowledge_moral_context: Vec<String>,
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

    // ── Stage 3b: Harmony interaction matrix (synergies/tensions) ──────
    interaction_matrix: HarmonyInteractionMatrix,

    // ── Stage 5: Institutional compliance checker ──────────────────────
    compliance_checker: InstitutionalComplianceChecker,

    // ── Restorative justice ─────────────────────────────────────────────
    restoration_tracker: RestorationTracker,

    // ── Consequence tracking (active inference applied to ethics) ─────
    consequence_tracker: ConsequenceTracker,

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
    /// Latest love coherence score [0, 1].
    last_love_coherence: f64,
    /// Count of consecutive cycles where Infinite Play (idx 3) has high
    /// Hebbian synergy, suggesting base weight 0.09 may be too low.
    play_hebbian_upweight_count: u64,
    /// Cached compliance risk from last Stage 5 evaluation.
    last_compliance_risk: f32,
    /// Cached compliance flags from last Stage 5 evaluation.
    last_compliance_flags: Vec<String>,
    /// Cached ahimsa violation flag from last Stage 1.
    last_ahimsa_violated: bool,
    /// Hash of last morally-parsed input for memoization.
    /// Skip re-parsing when the same input appears within the 7-cycle window.
    last_moral_input_hash: u64,
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
            last_love_coherence: 0.0,
            play_hebbian_upweight_count: 0,
            last_compliance_risk: 0.0,
            last_compliance_flags: Vec::new(),
            last_ahimsa_violated: false,
            last_moral_input_hash: 0,
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
        Self::with_anomaly_config_and_basis(
            moral_parser,
            moral_algebra,
            value_evaluator,
            harmonies_integrator,
            anomaly_config,
            None,
            false,
        )
    }

    /// Create with custom anomaly detection, optional pre-built dense `HarmonyBasis`,
    /// and optional Hodge decomposition enablement.
    ///
    /// When `enable_hodge` is true, sets `exact_betti` and `adaptive_rips_enabled`
    /// in the MoralTopologyConfig, enabling vertex L₀ Hodge decomposition with
    /// adaptive Rips threshold tracking.
    pub fn with_anomaly_config_and_basis(
        moral_parser: MoralParser,
        moral_algebra: MoralAlgebra,
        value_evaluator: Option<UnifiedValueEvaluator>,
        harmonies_integrator: Option<HarmoniesIntegrator>,
        anomaly_config: MoralAnomalyConfig,
        dense_basis: Option<Arc<HarmonyBasis>>,
        enable_hodge: bool,
    ) -> Self {
        let dim = moral_algebra.dim();
        let shared_basis = dense_basis.unwrap_or_else(|| Arc::new(HarmonyBasis::new(dim)));

        let moral_topology = MoralTopology::with_anomaly_config(
            MoralTopologyConfig {
                dim,
                exact_betti: enable_hodge,
                adaptive_rips_enabled: enable_hodge,
                // Hodge decomposition is O(n³) per scale. Bench results:
                //   n=32: 3.6ms/cycle amortized (7.2% of 20Hz budget) ← viable
                //   n=64: 214ms/cycle amortized (427% of budget) ← too expensive
                // Reduce window and scales when Hodge is on.
                window_size: if enable_hodge { 32 } else { 64 },
                num_scales: if enable_hodge { 8 } else { 10 },
                ..Default::default()
            },
            shared_basis.clone(),
            anomaly_config.clone(),
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
        let compliance_checker = InstitutionalComplianceChecker::new();

        let initial_cadence = anomaly_config.initial_cadence;
        Self {
            moral_parser,
            moral_algebra,
            value_evaluator,
            harmonies_integrator,
            moral_topology,
            interaction_matrix,
            compliance_checker,
            restoration_tracker: RestorationTracker::default(),
            consequence_tracker: ConsequenceTracker::new(),
            cache: EthicsEngineCache {
                last_harmonies_approved: true,
                topology_cadence: initial_cadence,
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
            ahimsa_violated,
        ) = if input.cycle % 7 == 0 && input.cycle > 0 {
            let encoded = self
                .moral_parser
                .parse_and_encode(input.input, &self.moral_algebra);

            // Feed action HV into moral topology sliding window.
            // Preference order:
            //   1. Semantic embedding (Qwen3 → HdcBridge → 16,384D continuous)
            //      — genuine semantic resolution, catches lexically-distinct
            //      but semantically-converging trajectories.
            //   2. Parser action_hv (agent-action-patient structured HDC)
            //   3. Raw text encoding (N-gram fallback — worst semantic quality)
            let scenario_hv = if let Some(emb) = input.semantic_embedding {
                ContinuousHV::from_slice(emb)
            } else {
                encoded
                    .action_hv
                    .clone()
                    .unwrap_or_else(|| self.moral_algebra.encode_action(input.input))
            };
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

            let ahimsa_violated = viols.iter().any(|name| {
                let is_ahimsa = name.starts_with("ahimsa_")
                    || name == "prevent_suffering"
                    || name == "minimize_collateral";
                // Restorative justice: if this specific violation has been
                // sufficiently corrected, don't re-trigger the ahimsa gate.
                // The system has demonstrated corrective behavior.
                is_ahimsa && !self.restoration_tracker.is_restored(name)
            });
            self.cache.last_ahimsa_violated = ahimsa_violated;

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
                ahimsa_violated,
            )
        } else {
            let ahimsa_violated = self.cache.last_ahimsa_violated;
            (
                self.cache.last_moral_score,
                String::new(),
                String::new(),
                false,
                0.0,
                Vec::new(),
                Vec::new(),
                ahimsa_violated,
            )
        };
        let moral_us = t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // RESTORATIVE JUSTICE: Track violations and corrections
        //
        // For each new violation, start a restoration window. For each
        // satisfaction of a previously-violated obligation, accumulate
        // corrective credit. Relapses penalize progress.
        // ═══════════════════════════════════════════════════════════════════
        for v in &violations {
            if self.restoration_tracker.entries.contains_key(v) {
                // Already tracking this violation — it's a relapse
                self.restoration_tracker.record_relapse(v);
            } else {
                self.restoration_tracker.record_violation(v, input.cycle);
            }
        }
        for s in &satisfactions {
            self.restoration_tracker.record_correction(s);
        }
        // Garbage-collect fully restored entries
        self.restoration_tracker.clear_restored();

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
        // STAGE 3: Harmonies Integrator — Eight Harmonies alignment
        // Every 19 cycles (co-prime with value evaluator — same cadence)
        //
        // Now uses semantically grounded basis vectors (not random) and
        // computes moral free energy (FEP) on the 8D harmony manifold.
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let mut fresh_harmonies_this_cycle = false;
        let (harmonies_alignment, harmonies_approved, harmony_coordinates, moral_free_energy) =
            if let Some(ref mut integrator) = self.harmonies_integrator {
                if input.cycle % 19 == 0 && input.cycle > 0 {
                    fresh_harmonies_this_cycle = true;
                    // Prefer dense semantic embedding (Qwen3/BGE-M3 → HdcBridge)
                    // when available — lives in the same JL-projected subspace as
                    // dense HarmonyBasis vectors. Falls back to n-gram encoding.
                    let embedding = if let Some(emb) = input.semantic_embedding {
                        ContinuousHV::from_slice(emb)
                    } else {
                        self.moral_algebra.encode_action(input.input)
                    };
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

        // Harmony interaction matrix: observe co-activations and apply synergies.
        // Gated to harmonies interval (19 cycles) — these operations involve
        // matrix multiplication and entropy computation that cost 15-20ms/cycle
        // when run unconditionally. Previously gated on `t.elapsed().as_micros() > 0`
        // as a proxy for "did the fresh-evaluation branch above just run" -- fragile,
        // since a fast/lightweight fresh evaluation (or a coarse OS timer) can report
        // 0 elapsed microseconds, silently skipping the observation. Use the actual
        // branch outcome instead.
        if fresh_harmonies_this_cycle {
            self.interaction_matrix.observe(&harmony_coordinates, 0.05);
            harmony_coordinates = self.interaction_matrix.apply(&harmony_coordinates, 0.15);

            // ── Love Coherence: emergent macro-state of harmony resonance ────
            let love_coherence = self.interaction_matrix.love_coherence(&harmony_coordinates);
            self.cache.last_love_coherence = love_coherence.value;

            // Love coherence → confidence modulation
            if love_coherence.value > 0.6 {
                confidence_delta += 0.01 * (love_coherence.value - 0.6) as f32;
            } else if love_coherence.value < 0.3 {
                confidence_delta -= 0.01 * (0.3 - love_coherence.value) as f32;
            }

            // ── Play weight monitoring ────────────────────────────────────
            let play_idx = 3;
            let play_avg: f64 = (0..N_HARMONIES)
                .filter(|&j| j != play_idx)
                .map(|j| self.interaction_matrix.weights[play_idx][j])
                .sum::<f64>()
                / (N_HARMONIES - 1) as f64;
            if play_avg > 0.3 {
                self.cache.play_hebbian_upweight_count += 1;
                if self.cache.play_hebbian_upweight_count % 500 == 0 {
                    tracing::info!(
                        play_hebbian_avg = %play_avg,
                        count = self.cache.play_hebbian_upweight_count,
                        "Infinite Play consistently synergistic (avg Hebbian weight {:.3}). \
                         Base weight of 0.09 may warrant increase.",
                        play_avg
                    );
                }
            }
        } else {
            // Non-harmonies cycle: use cached love coherence for confidence modulation
            let cached_lc = self.cache.last_love_coherence;
            if cached_lc > 0.6 {
                confidence_delta += 0.01 * (cached_lc - 0.6) as f32;
            } else if cached_lc < 0.3 {
                confidence_delta -= 0.01 * (0.3 - cached_lc) as f32;
            }
        }

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

        // Knowledge-grounded confidence: scale delta by knowledge multiplier
        confidence_delta = (confidence_delta as f64 * input.knowledge_confidence_multiplier) as f32;

        // Knowledge moral context boosts base moral confidence
        let moral_context_boost = if !input.knowledge_moral_context.is_empty() {
            (input.knowledge_moral_context.len() as f64 * 0.02).min(0.1)
        } else {
            0.0
        };

        let unified_confidence = if moral_confidence > 0.0 {
            (moral_confidence + moral_context_boost).min(1.0)
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

        // Escalate verdict when trajectory convergence signals a compartmentalized
        // adversarial plan (severity > 0.7 → Blocked, any convergence → at least Caution).
        let unified_verdict = if anomaly_report.trajectory_convergence
            && anomaly_report.convergence_severity > 0.7
        {
            EthicalVerdict::Blocked
        } else if anomaly_report.trajectory_convergence && unified_verdict == EthicalVerdict::Safe {
            EthicalVerdict::Caution
        } else {
            unified_verdict
        };

        // ═══════════════════════════════════════════════════════════════════
        // STAGE 5: Institutional Compliance Check — HDC jurisdiction matching
        // Every 23 cycles (co-prime with 7, 19, 97)
        //
        // Encodes action text as BinaryHV and checks similarity against
        // institutional constraint vectors (JURISDICTION ⊗ COMPLIANCE, etc.)
        // ═══════════════════════════════════════════════════════════════════
        let t = Instant::now();
        let (compliance_risk, compliance_flags, compliance_fresh) =
            if input.cycle % 23 == 0 && input.cycle > 0 {
                // Prefer the pre-computed BinaryHV from the cognitive loop's HDC encoder
                // (genuine semantic encoding) over lightweight n-gram hashing.
                let fallback;
                let action_hv = match input.action_hv {
                    Some(hv) => hv,
                    None => {
                        fallback = self.compliance_checker.encode_text(input.input);
                        &fallback
                    }
                };
                let (risk, flags) = self.compliance_checker.check(action_hv);
                self.cache.last_compliance_risk = risk;
                self.cache.last_compliance_flags = flags.clone();
                (risk, flags, true)
            } else {
                (
                    self.cache.last_compliance_risk,
                    self.cache.last_compliance_flags.clone(),
                    false,
                )
            };
        let compliance_us = t.elapsed().as_micros() as u64;

        // Compliance escalation: high institutional risk → Caution
        let unified_verdict = if compliance_risk > 0.7 && unified_verdict == EthicalVerdict::Safe {
            EthicalVerdict::Caution
        } else {
            unified_verdict
        };

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
            ahimsa_violated,
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
            love_coherence: self.cache.last_love_coherence,
            compliance_risk,
            compliance_flags,
            compliance_fresh,
            restoration_progress: self.restoration_tracker.restoration_progress(),
            topology_us,
            topology_fresh,
            moral_us,
            value_us,
            harmonies_us,
            compliance_us,
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

    /// Access the moral topology analyser (mutable, for peer correlation boost).
    pub fn moral_topology_mut(&mut self) -> &mut MoralTopology {
        &mut self.moral_topology
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

    /// Access the restorative justice tracker.
    pub fn restoration_tracker(&self) -> &RestorationTracker {
        &self.restoration_tracker
    }

    /// Access the restorative justice tracker (mutable).
    pub fn restoration_tracker_mut(&mut self) -> &mut RestorationTracker {
        &mut self.restoration_tracker
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

    /// Nudge a specific harmony coordinate in the cached state.
    ///
    /// Used by cross-system couplings (e.g. Cantor fractal → Sacred Stillness)
    /// that operate after `evaluate()` returns. The nudge is clamped to [-1, 1].
    pub fn nudge_harmony_coordinate(&mut self, index: usize, delta: f64) {
        if index < N_HARMONIES {
            self.cache.last_harmony_coordinates[index] =
                (self.cache.last_harmony_coordinates[index] + delta).clamp(-1.0, 1.0);
        }
    }

    /// Access cached compliance risk from last Stage 5 evaluation.
    pub fn last_compliance_risk(&self) -> f32 {
        self.cache.last_compliance_risk
    }

    /// Access cached compliance flags from last Stage 5 evaluation.
    pub fn last_compliance_flags(&self) -> &[String] {
        &self.cache.last_compliance_flags
    }

    /// Load additional regulatory constraints from external specifications.
    ///
    /// Use this to add jurisdiction-specific constraints at runtime, e.g., from
    /// Mycelix JurisdictionConstraintEntry records. Each spec defines a flag name,
    /// the primitive names to compose, and an optional threshold.
    ///
    /// Returns the number of constraints successfully loaded.
    pub fn load_external_constraints(&mut self, specs: &[ExternalConstraintSpec]) -> usize {
        self.compliance_checker.load_external_constraints(specs)
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

    /// Save moral topology state to a JSON file for cross-session persistence.
    pub fn save_topology_snapshot(
        &self,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let snap = self.moral_topology.snapshot();
        let json = serde_json::to_string_pretty(&snap)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Restore moral topology state from a previously saved JSON snapshot.
    pub fn restore_topology_snapshot(
        &mut self,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let snap: crate::hdc::moral_topology::MoralTopologySnapshot = serde_json::from_str(&json)?;
        self.moral_topology.restore(&snap);
        Ok(())
    }

    // ── Consequence Tracker delegation ──────────────────────────────────

    /// Record an ethical prediction for later outcome validation.
    pub fn record_consequence_prediction(
        &mut self,
        action_id: String,
        verdict: EthicalVerdict,
        consciousness: f64,
        cycle: u64,
        community_phi: f64,
        affect_valence: f64,
    ) {
        self.consequence_tracker.record_prediction(
            action_id,
            verdict,
            consciousness,
            cycle,
            community_phi,
            affect_valence,
        );
    }

    /// Observe an outcome and resolve matching predictions.
    /// Returns the prediction error if a matching prediction was found.
    pub fn observe_consequence_outcome(
        &mut self,
        action_id: &str,
        observed_phi: f64,
        observed_valence: f64,
        current_cycle: u64,
    ) -> Option<f64> {
        self.consequence_tracker.observe_outcome(
            action_id,
            observed_phi,
            observed_valence,
            current_cycle,
        )
    }

    /// Get current consequence prediction accuracy (EMA).
    pub fn consequence_tracker_accuracy(&self) -> f64 {
        self.consequence_tracker.accuracy()
    }

    /// Total consequence predictions made.
    pub fn consequence_tracker_total(&self) -> u64 {
        self.consequence_tracker.total_predictions()
    }

    /// Expire stale consequence predictions older than `max_age_cycles`.
    pub fn expire_stale_predictions(&mut self, current_cycle: u64, max_age_cycles: u64) {
        self.consequence_tracker
            .expire_stale(current_cycle, max_age_cycles);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSEQUENCE TRACKER — Active Inference Applied to Ethics
//
// Science: Friston (2010) active inference; Cushman (2013) dual-process moral cognition.
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks ethical predictions against actual outcomes.
///
/// After the ethics engine evaluates an action, the ConsequenceTracker
/// records the prediction. When the outcome is observed (via governance
/// events, consciousness changes, or community feedback), it computes
/// prediction error and adjusts the ethics engine's confidence.
///
/// Science: Friston (2010) active inference; Cushman (2013) dual-process moral cognition.
pub(crate) struct ConsequenceTracker {
    /// Pending predictions awaiting outcome observation.
    pending: VecDeque<ConsequencePrediction>,
    /// Maximum pending predictions (prevents unbounded growth).
    max_pending: usize,
    /// Completed predictions with observed outcomes.
    completed: VecDeque<ConsequenceOutcome>,
    /// Maximum completed outcomes to retain (ring buffer).
    max_completed: usize,
    /// Running prediction accuracy (EMA, alpha=0.05).
    accuracy_ema: f64,
    /// Total predictions made.
    total_predictions: u64,
    /// Total correct predictions (verdict matched outcome).
    total_correct: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct ConsequencePrediction {
    pub action_id: String,
    pub predicted_verdict: EthicalVerdict,
    pub consciousness_at_prediction: f64,
    pub cycle: u64,
    pub baseline_community_phi: f64,
    pub baseline_affect_valence: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ConsequenceOutcome {
    pub prediction: ConsequencePrediction,
    pub observed_community_phi: f64,
    pub observed_affect_valence: f64,
    pub prediction_correct: bool,
    pub prediction_error: f64,
    pub observation_delay_cycles: u64,
}

impl ConsequenceTracker {
    pub(crate) fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            max_pending: 100,
            completed: VecDeque::new(),
            max_completed: 500,
            accuracy_ema: 0.5,
            total_predictions: 0,
            total_correct: 0,
        }
    }

    pub(crate) fn resolved_count(&self) -> usize {
        self.completed.len()
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub(crate) fn record_prediction(
        &mut self,
        action_id: String,
        verdict: EthicalVerdict,
        consciousness: f64,
        cycle: u64,
        community_phi: f64,
        affect_valence: f64,
    ) {
        if self.pending.len() >= self.max_pending {
            self.pending.pop_front();
        }
        self.pending.push_back(ConsequencePrediction {
            action_id,
            predicted_verdict: verdict,
            consciousness_at_prediction: consciousness,
            cycle,
            baseline_community_phi: community_phi,
            baseline_affect_valence: affect_valence,
        });
        self.total_predictions += 1;
    }

    pub(crate) fn observe_outcome(
        &mut self,
        action_id: &str,
        observed_phi: f64,
        observed_valence: f64,
        current_cycle: u64,
    ) -> Option<f64> {
        let idx = self.pending.iter().position(|p| p.action_id == action_id)?;
        let prediction = self.pending.remove(idx)?;

        let phi_delta = observed_phi - prediction.baseline_community_phi;
        let valence_delta = observed_valence - prediction.baseline_affect_valence;
        let positive_outcome = phi_delta > -0.05 && valence_delta > -0.2;

        let prediction_correct = match prediction.predicted_verdict {
            EthicalVerdict::Safe => positive_outcome,
            EthicalVerdict::Caution => true,
            EthicalVerdict::Blocked => !positive_outcome,
        };

        let pe = if prediction_correct { 0.0 } else { 1.0 };
        let alpha = super::thresholds::CONSEQUENCE_TRACKER_ACCURACY_ALPHA;
        self.accuracy_ema = self.accuracy_ema * (1.0 - alpha)
            + (if prediction_correct { 1.0 } else { 0.0 }) * alpha;
        if prediction_correct {
            self.total_correct += 1;
        }

        let delay = current_cycle.saturating_sub(prediction.cycle);
        let outcome = ConsequenceOutcome {
            prediction,
            observed_community_phi: observed_phi,
            observed_affect_valence: observed_valence,
            prediction_correct,
            prediction_error: pe,
            observation_delay_cycles: delay,
        };
        if self.completed.len() >= self.max_completed {
            self.completed.pop_front();
        }
        self.completed.push_back(outcome);

        Some(pe)
    }

    pub(crate) fn accuracy(&self) -> f64 {
        self.accuracy_ema
    }

    pub(crate) fn total_predictions(&self) -> u64 {
        self.total_predictions
    }

    pub(crate) fn expire_stale(&mut self, current_cycle: u64, max_age_cycles: u64) {
        self.pending
            .retain(|p| current_cycle.saturating_sub(p.cycle) < max_age_cycles);
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

    /// Like `make_engine`, but with a real `HarmoniesIntegrator` wired in.
    /// Needed for anything exercising Stage 3 (harmonies alignment / the
    /// interaction matrix) -- `make_engine`'s `harmonies_integrator: None`
    /// makes that whole pathway structurally unreachable.
    fn make_engine_with_harmonies() -> EthicsEngine {
        let parser = MoralParser::new();
        let algebra = MoralAlgebra::new(16384);
        // dimension must match MoralAlgebra::new(16384) above -- the default
        // (512) causes a "Dimension mismatch in similarity(): 16384 vs 512"
        // panic when the harmonies integrator compares its embeddings
        // against the engine's 16384-dim action encodings.
        let mut harmonies_config =
            crate::consciousness::harmonies_integration::HarmoniesIntegrationConfig::default();
        harmonies_config.dimension = 16384;
        let integrator = HarmoniesIntegrator::new(harmonies_config);
        EthicsEngine::new(parser, algebra, None, Some(integrator))
    }

    fn make_input(cycle: u64) -> EthicsEngineInput<'static> {
        EthicsEngineInput {
            input: "helping others is good",
            cycle,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
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
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
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
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
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
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
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
        // Needs a real HarmoniesIntegrator: the interaction-matrix observe()
        // call is only reachable from Stage 3's `input.cycle % 19 == 0`
        // branch, which itself is gated on `self.harmonies_integrator` being
        // `Some(..)` -- make_engine()'s `None` makes this whole pathway
        // structurally unreachable regardless of cycle count.
        //
        // Create an engine and observe some patterns to mutate the matrix.
        let mut engine = make_engine_with_harmonies();
        for cycle in (7..=133).step_by(7) {
            let input = make_input(cycle as u64);
            engine.evaluate(&input);
        }

        // Extract the learned weights.
        let saved_weights = *engine.interaction_matrix_weights();

        // Create a fresh engine — its matrix will be from_basis (semantic init).
        let mut engine2 = make_engine_with_harmonies();
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

    // =========================================================================
    // P0: Verdict disagreement tests
    // =========================================================================

    #[test]
    fn test_negative_moral_score_produces_caution() {
        let mut engine = make_engine();
        // Use text the moral parser recognizes as harmful
        let input = EthicsEngineInput {
            input: "destroy harm damage and exploit the vulnerable",
            cycle: 7,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
        };
        let output = engine.evaluate(&input);
        // With no value evaluator or harmonies, verdict depends on moral_score
        // If moral_score < -0.3, should be Caution
        if output.moral_score < -0.3 {
            assert_eq!(
                output.unified_verdict,
                EthicalVerdict::Caution,
                "Negative moral score ({}) should produce Caution",
                output.moral_score
            );
        }
    }

    #[test]
    fn test_consent_violation_always_blocked() {
        let mut engine = make_engine();
        // Run enough cycles to establish context
        for c in (7..=49).step_by(7) {
            engine.evaluate(&make_input(c as u64));
        }
        // Now test consent violation text
        let input = EthicsEngineInput {
            input: "forcing someone against their will without consent to comply",
            cycle: 56,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
        };
        let output = engine.evaluate(&input);
        if output.consent_violation {
            assert_eq!(
                output.unified_verdict,
                EthicalVerdict::Blocked,
                "Consent violation must always produce Blocked"
            );
        }
    }

    #[test]
    fn test_unified_confidence_fallback_when_moral_parser_skipped() {
        let mut engine = make_engine();
        // Cycle 8 — moral parser doesn't fire (8 % 7 != 0)
        let input = make_input(8);
        let output = engine.evaluate(&input);
        // moral_confidence should be 0 (parser didn't fire)
        // unified_confidence should use value_score as fallback
        assert!(
            output.unified_confidence.is_finite(),
            "Confidence should be finite even when parser skipped"
        );
        assert!(
            output.unified_confidence >= 0.0 && output.unified_confidence <= 1.0,
            "Confidence should be in [0, 1], got {}",
            output.unified_confidence
        );
    }

    #[test]
    fn test_lr_factor_always_bounded() {
        let mut engine = make_engine();
        // Run various cycle numbers
        for c in 0..100 {
            let input = make_input(c);
            let output = engine.evaluate(&input);
            assert!(
                output.lr_factor >= 0.1 && output.lr_factor <= 1.3,
                "lr_factor should be in [0.1, 1.3], got {} at cycle {c}",
                output.lr_factor
            );
        }
    }

    #[test]
    fn test_all_outputs_finite_across_cycles() {
        let mut engine = make_engine();
        for c in 0..150 {
            let input = make_input(c);
            let output = engine.evaluate(&input);
            assert!(
                output.moral_score.is_finite(),
                "moral_score NaN at cycle {c}"
            );
            assert!(
                output.moral_confidence.is_finite(),
                "moral_confidence NaN at cycle {c}"
            );
            assert!(
                output.value_score.is_finite(),
                "value_score NaN at cycle {c}"
            );
            assert!(
                output.unified_confidence.is_finite(),
                "unified_confidence NaN at cycle {c}"
            );
            assert!(
                output.harmonies_alignment.is_finite(),
                "harmonies_alignment NaN at cycle {c}"
            );
            assert!(
                output.moral_free_energy.free_energy.is_finite(),
                "moral_free_energy NaN at cycle {c}"
            );
            for (i, coord) in output.harmony_coordinates.iter().enumerate() {
                assert!(
                    coord.is_finite(),
                    "harmony_coordinate[{i}] NaN at cycle {c}"
                );
            }
        }
    }

    #[test]
    fn test_moral_score_bounded() {
        let mut engine = make_engine();
        let texts = [
            "helping others is wonderful and kind",
            "destroy harm damage exploit and isolate",
            "the weather is pleasant today",
            "",
            "forcing without consent against will",
        ];
        for (i, &text) in texts.iter().enumerate() {
            let input = EthicsEngineInput {
                input: text,
                cycle: ((i + 1) * 7) as u64,
                unified_psi: 0.5,
                compressed_state: &[0.0; 256],
                stillness_boost: 0.0,
                semantic_embedding: None,
                action_hv: None,
                knowledge_confidence_multiplier: 1.0,
                knowledge_moral_context: Vec::new(),
            };
            let output = engine.evaluate(&input);
            assert!(
                output.moral_score >= -1.0 && output.moral_score <= 1.0,
                "moral_score out of [-1, 1] for '{}': {}",
                text,
                output.moral_score
            );
        }
    }

    // =========================================================================
    // P0: Interval and timing edge cases
    // =========================================================================

    #[test]
    fn test_cycle_0_skips_all_stages() {
        let mut engine = make_engine();
        let input = make_input(0);
        let output = engine.evaluate(&input);
        // Cycle 0: all stages have input.cycle > 0 guard, so nothing fires
        // moral_verdict should be empty (cached default)
        assert!(
            output.moral_verdict.is_empty(),
            "Cycle 0 should not fire moral parser"
        );
    }

    #[test]
    fn test_cycle_7_fires_moral_only() {
        let mut engine = make_engine();
        let input = make_input(7);
        let output = engine.evaluate(&input);
        assert!(
            !output.moral_verdict.is_empty(),
            "Cycle 7 should fire moral parser"
        );
        // Value evaluator fires at 19, not 7
        assert!(
            output.value_decision.is_empty(),
            "Cycle 7 should not fire value evaluator (no evaluator)"
        );
    }

    #[test]
    fn test_topology_needs_minimum_scenarios() {
        let mut engine = make_engine();
        // Run only 2 cycles — topology needs >= 3 scenarios
        let input = make_input(7);
        engine.evaluate(&input);
        let input = make_input(14);
        let output = engine.evaluate(&input);
        // Topology should not have fired (< 3 scenarios)
        assert!(
            !output.topology_fresh,
            "Topology should not fire with < 3 scenarios"
        );
    }

    // =========================================================================
    // P0: Anomaly report field checks
    // =========================================================================

    // =========================================================================
    // Stage 5: Institutional compliance checks
    // =========================================================================

    #[test]
    fn test_compliance_checker_constructs() {
        let checker = InstitutionalComplianceChecker::new();
        assert!(
            !checker.constraints.is_empty(),
            "Compliance checker should have constraints from PrimitiveSystem"
        );
        // Should have 5 constraint patterns
        assert_eq!(checker.constraints.len(), 5);
    }

    #[test]
    fn test_compliance_output_fields_present() {
        let mut engine = make_engine();
        let input = make_input(23); // cycle 23 triggers compliance (23 % 23 == 0)
        let output = engine.evaluate(&input);
        assert!(output.compliance_risk >= 0.0 && output.compliance_risk <= 1.0);
        assert!(output.compliance_fresh);
    }

    #[test]
    fn test_compliance_skipped_off_cadence() {
        let mut engine = make_engine();
        let input = make_input(8); // cycle 8: not divisible by 23
        let output = engine.evaluate(&input);
        assert!(!output.compliance_fresh);
        assert_eq!(output.compliance_risk, 0.0);
    }

    #[test]
    fn test_compliance_risk_bounded_across_cycles() {
        let mut engine = make_engine();
        for c in 0..100 {
            let input = make_input(c);
            let output = engine.evaluate(&input);
            assert!(
                output.compliance_risk >= 0.0 && output.compliance_risk <= 1.0,
                "compliance_risk should be in [0, 1], got {} at cycle {}",
                output.compliance_risk,
                c
            );
        }
    }

    #[test]
    fn test_compliance_text_encoder_deterministic() {
        let checker = InstitutionalComplianceChecker::new();
        let hv1 = checker.encode_text("deploy data to eu server");
        let hv2 = checker.encode_text("deploy data to eu server");
        assert_eq!(
            hv1.similarity(&hv2),
            1.0,
            "Same text should produce identical encoding"
        );
    }

    #[test]
    fn test_compliance_different_texts_orthogonal() {
        let checker = InstitutionalComplianceChecker::new();
        let hv1 = checker.encode_text("deploy data to eu server");
        let hv2 = checker.encode_text("help someone learn mathematics");
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 0.5).abs() < 0.1,
            "Unrelated texts should be roughly orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_anomaly_score_bounded() {
        let mut engine = make_engine();
        // Inject diverse scenarios for topology
        use crate::hdc::ContinuousHV;
        for i in 0..20 {
            engine
                .moral_topology
                .add_scenario(ContinuousHV::random(16384, 200 + i));
        }
        for c in 0..200 {
            let input = make_input(c);
            let output = engine.evaluate(&input);
            let score = output.anomaly_report.anomaly_score;
            assert!(
                score >= 0.0 && score <= 1.0,
                "anomaly_score should be in [0, 1], got {score} at cycle {c}"
            );
        }
    }

    // =========================================================================
    // Stage 5: Action HV wiring (thought vector → compliance)
    // =========================================================================

    #[test]
    fn test_compliance_uses_action_hv_when_provided() {
        let mut engine = make_engine();

        // Create a BinaryHV that is very similar to a known constraint
        let system = PrimitiveSystem::global();
        let jurisdiction = system.get("JURISDICTION").unwrap();
        let compliance_prim = system.get("COMPLIANCE").unwrap();
        let constraint_hv = jurisdiction.encoding.bind(&compliance_prim.encoding);

        // Provide this as the action_hv — should trigger data_sovereignty flag
        let input = EthicsEngineInput {
            input: "irrelevant text",
            cycle: 23,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: Some(&constraint_hv),
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
        };
        let output = engine.evaluate(&input);
        assert!(output.compliance_fresh);
        // The action_hv IS the constraint pattern, so risk should be maximal
        assert!(
            output.compliance_risk > 0.5,
            "Exact constraint pattern should trigger high risk, got {}",
            output.compliance_risk
        );
        assert!(
            output
                .compliance_flags
                .contains(&"data_sovereignty".to_string()),
            "Should flag data_sovereignty when action_hv matches constraint"
        );
    }

    #[test]
    fn test_compliance_falls_back_to_text_encoding() {
        let mut engine = make_engine();

        // No action_hv → falls back to encode_text()
        let input = EthicsEngineInput {
            input: "checking jurisdiction compliance for data transfer",
            cycle: 23,
            unified_psi: 0.5,
            compressed_state: &[0.0; 256],
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
        };
        let output = engine.evaluate(&input);
        assert!(output.compliance_fresh);
        // Text encoding is a weak signal — risk should be low
        assert!(
            output.compliance_risk.is_finite(),
            "Compliance risk should be finite"
        );
    }

    // =========================================================================
    // Stage 5: Dynamic constraint loading
    // =========================================================================

    #[test]
    fn test_external_constraint_loading() {
        let mut engine = make_engine();

        let specs = vec![ExternalConstraintSpec {
            flag: "gdpr_data_residency".to_string(),
            primitive_names: vec!["JURISDICTION".to_string(), "SOVEREIGNTY".to_string()],
            threshold: Some(0.55),
        }];

        let loaded = engine.load_external_constraints(&specs);
        assert_eq!(loaded, 1, "Should load 1 external constraint");

        // Checker now has 5 defaults + 1 external = 6
        assert_eq!(engine.compliance_checker.constraints.len(), 6);
    }

    #[test]
    fn test_external_constraint_missing_primitive_skipped() {
        let mut engine = make_engine();

        let specs = vec![ExternalConstraintSpec {
            flag: "nonexistent_constraint".to_string(),
            primitive_names: vec![
                "JURISDICTION".to_string(),
                "TOTALLY_FAKE_PRIMITIVE".to_string(),
            ],
            threshold: None,
        }];

        let loaded = engine.load_external_constraints(&specs);
        assert_eq!(loaded, 0, "Should skip constraint with missing primitive");
        assert_eq!(engine.compliance_checker.constraints.len(), 5);
    }

    #[test]
    fn test_restoration_tracker_basic() {
        let mut tracker = RestorationTracker::default();
        tracker.record_violation("non_harm", 100);
        assert!(!tracker.is_restored("non_harm"));
        assert_eq!(tracker.active_violations(), 1);

        for _ in 0..10 {
            tracker.record_correction("non_harm");
        }
        assert!(tracker.is_restored("non_harm"));
        assert!(tracker.restoration_progress() >= 0.99);
    }

    #[test]
    fn test_restoration_tracker_relapse_resets() {
        let mut tracker = RestorationTracker::default();
        tracker.record_violation("honesty", 50);
        for _ in 0..5 {
            tracker.record_correction("honesty");
        }
        assert!(!tracker.is_restored("honesty")); // not enough yet

        tracker.record_relapse("honesty");
        // Relapse subtracts 3 and marks relapse
        assert!(!tracker.is_restored("honesty"));
        assert!(tracker.restoration_progress() < 0.5);
    }

    #[test]
    fn test_restoration_tracker_no_violations() {
        let tracker = RestorationTracker::default();
        assert!(tracker.is_restored("anything"));
        assert_eq!(tracker.active_violations(), 0);
        assert!((tracker.restoration_progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_restoration_softens_ahimsa_gate() {
        // Integration: ahimsa violation → courage blocked → corrections → courage restored
        let mut tracker = RestorationTracker::default();

        // Phase 1: Initial violation — not yet restored
        tracker.record_violation("ahimsa_nonviolence", 100);
        assert!(!tracker.is_restored("ahimsa_nonviolence"));

        // Phase 2: Accumulate corrective behavior (10 cycles)
        for _ in 0..10 {
            tracker.record_correction("ahimsa_nonviolence");
        }

        // Phase 3: Violation is now restored
        assert!(tracker.is_restored("ahimsa_nonviolence"));

        // Phase 4: The ahimsa gate check in evaluate() uses:
        //   is_ahimsa && !self.restoration_tracker.is_restored(name)
        // So a restored violation no longer blocks courage override.
        let would_block = !tracker.is_restored("ahimsa_nonviolence");
        assert!(
            !would_block,
            "Restored ahimsa violation should not block courage"
        );

        // Phase 5: But an unrestored different violation still blocks
        tracker.record_violation("prevent_suffering", 200);
        let suffering_blocks = !tracker.is_restored("prevent_suffering");
        assert!(
            suffering_blocks,
            "Unrestored suffering violation should still block"
        );
    }

    // ── Consequence Tracker tests ────────────────────────────────────────

    #[test]
    fn test_consequence_tracker_default_accuracy() {
        let tracker = ConsequenceTracker::new();
        assert!(
            (tracker.accuracy() - 0.5).abs() < 1e-10,
            "Default accuracy should be 0.5 (uninformative prior)"
        );
        assert_eq!(tracker.total_predictions(), 0);
    }

    #[test]
    fn test_consequence_tracker_correct_safe_prediction() {
        let mut tracker = ConsequenceTracker::new();
        tracker.record_prediction("a1".into(), EthicalVerdict::Safe, 0.5, 100, 0.6, 0.3);
        let pe = tracker.observe_outcome("a1", 0.6, 0.3, 110);
        assert!(pe.is_some());
        assert!((pe.unwrap() - 0.0).abs() < 1e-10, "Safe + stable = correct");
        assert!(tracker.accuracy() > 0.5);
    }

    #[test]
    fn test_consequence_tracker_wrong_safe_prediction() {
        let mut tracker = ConsequenceTracker::new();
        tracker.record_prediction("b1".into(), EthicalVerdict::Safe, 0.5, 100, 0.6, 0.3);
        let pe = tracker.observe_outcome("b1", 0.4, -0.1, 120);
        assert!((pe.unwrap() - 1.0).abs() < 1e-10, "Safe + crash = wrong");
        assert!(tracker.accuracy() < 0.5);
    }

    #[test]
    fn test_consequence_tracker_caution_always_correct() {
        let mut tracker = ConsequenceTracker::new();
        tracker.record_prediction("c1".into(), EthicalVerdict::Caution, 0.5, 100, 0.6, 0.3);
        let pe = tracker.observe_outcome("c1", 0.4, -0.5, 120);
        assert!((pe.unwrap() - 0.0).abs() < 1e-10, "Caution always correct");
    }

    #[test]
    fn test_consequence_tracker_expire_stale() {
        let mut tracker = ConsequenceTracker::new();
        tracker.record_prediction("old".into(), EthicalVerdict::Safe, 0.5, 10, 0.5, 0.0);
        tracker.record_prediction("new".into(), EthicalVerdict::Safe, 0.5, 90, 0.5, 0.0);
        tracker.expire_stale(100, 50);
        assert!(tracker.observe_outcome("old", 0.5, 0.0, 100).is_none());
        assert!(tracker.observe_outcome("new", 0.5, 0.0, 100).is_some());
    }

    #[test]
    fn test_consequence_tracker_on_ethics_engine() {
        let mut engine = make_engine();
        assert!((engine.consequence_tracker_accuracy() - 0.5).abs() < 1e-10);
        engine.record_consequence_prediction("t1".into(), EthicalVerdict::Safe, 0.5, 100, 0.6, 0.3);
        let pe = engine.observe_consequence_outcome("t1", 0.6, 0.3, 110);
        assert!(pe.is_some());
        assert!(engine.consequence_tracker_accuracy() > 0.5);
    }
}
