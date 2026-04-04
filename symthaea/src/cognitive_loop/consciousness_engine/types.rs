// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Types for the consciousness engine — output, input, weights, coupling, cache.

use crate::consciousness::consciousness_equation_v2::CoreComponent;
use symthaea_core::consciousness_metrics::StructuralPhiResult;
use symthaea_core::hdc::{BinaryHV, ContinuousHV};

/// Classification of weight convergence dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightConvergenceState {
    /// < 20 samples in weight history.
    Initializing,
    /// Variance is decreasing (recent half < older half).
    Converging,
    /// Variance < 0.001 for 50+ consecutive cycles.
    Converged,
    /// Variance is increasing or > 0.005.
    Oscillating,
}

impl WeightConvergenceState {
    /// Static string matching Debug output — avoids `format!("{:?}")` on hot path.
    #[inline]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Initializing => "Initializing",
            Self::Converging => "Converging",
            Self::Converged => "Converged",
            Self::Oscillating => "Oscillating",
        }
    }
}

/// Unified output from the consciousness engine.
///
/// Contains all measurement results plus proposed feedback deltas.
/// The caller applies deltas to prediction_confidence, fep_lr_boost, etc.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields flow through ConsciousnessCache; read via cache, not struct
pub(crate) struct ConsciousnessEngineOutput {
    // ── Raw measurements ───────────────────────────────────────────────
    /// Spectral MIP Phi — integrated information via Fiedler ordering [0, ∞)
    pub spectral_mip_phi: Option<f64>,
    /// Hierarchical MIP Phi — multi-scale (32→64→128) [0, ∞)
    pub hierarchical_mip_phi: Option<f64>,
    /// Structural Phi decomposition — cluster-level micro/meso/macro
    pub structural_phi: Option<StructuralPhiResult>,
    /// Multi-modal integrated Phi — cross-modal binding [0, 1]
    pub multimodal_phi: f64,
    /// Consciousness Equation V2 — 7-theory unified C(t) [0, 1]
    pub equation_v2_consciousness: f64,
    /// Pipeline consciousness — end-to-end sensory→consciousness [0, 1]
    pub pipeline_consciousness: f64,
    /// Limiting component from equation v2
    pub limiting_component: Option<CoreComponent>,

    // ── Unified consciousness level ────────────────────────────────────
    /// Weighted consensus consciousness level [0, 1].
    ///
    /// Computed as: 0.35 × spectral_phi_norm + 0.25 × equation_v2 + 0.25 × pipeline + 0.15 × multimodal
    /// (spectral_phi_norm = sigmoid(phi) to map [0,∞) → [0,1])
    pub unified_consciousness: f64,

    // ── Sigma (backward compatibility for memory coordinator) ──────────
    pub sigma: Option<f64>,

    // ── Proposed feedback deltas ────────────────────────────────────────
    /// Additive delta for prediction_confidence (positive = boost, negative = dampen)
    pub confidence_delta: f32,
    /// Multiplicative factor for fep_lr_boost (1.0 = no change)
    pub lr_factor: f32,
    /// Additive delta for exploration_urge
    pub exploration_delta: f32,
    /// Multiplicative factor for subsystem_lr_factor
    pub subsystem_lr_factor: f32,
    /// Whether to boost episodic consolidation (high consciousness moment)
    pub episodic_consolidation_boost: Option<f64>,

    // ── Dynamic weights telemetry ──────────────────────────────────────
    /// Current consciousness weights [spectral, equation, pipeline, multimodal].
    pub current_weights: [f64; 4],
    /// Weight stability variance (0.0 = perfectly stable, >0.01 = oscillating).
    pub weight_variance: f64,
    /// Current convergence classification.
    pub convergence_state: WeightConvergenceState,

    // ── Timing ─────────────────────────────────────────────────────────
    pub spectral_mip_us: u64,
    pub equation_v2_us: u64,
    pub pipeline_us: u64,
    pub multimodal_us: u64,
    pub total_us: u64,
}

/// Input snapshot for the consciousness engine.
///
/// Collected once per cycle by the caller, passed immutably.
pub(crate) struct ConsciousnessEngineInput<'a> {
    /// Current HDC encoding (ContinuousHV, 16384-dim) — for SpectralMIP push
    pub hdv: &'a ContinuousHV,
    /// Current HDC encoding (BinaryHV, 16384-bit) — for multimodal + pipeline
    pub hv16: &'a BinaryHV,
    /// Current cycle number
    pub cycle: u64,
    /// Unified Psi from primitive consciousness
    pub unified_psi: f64,
    /// Smoothed coherence from HDC pipeline
    pub coherence: f32,
    /// Current prediction error
    pub prediction_error: f32,
    /// Phi attention weight
    pub phi_attention_weight: f32,
    /// Last epistemic quality (knowledge component for equation v2)
    pub epistemic_quality: f64,
    /// Phi validation correlation (for adaptive weighting)
    pub phi_validation_correlation: f64,

    // ── Phase 6: Bath → consciousness coupling (Seth 2013) ──────────
    /// Bath phase space entropy (from BathPhaseTracker).
    /// Reserved for bath→consciousness coupling (Seth 2013 protocol).
    #[allow(dead_code)]
    pub bath_entropy: f32,
    /// Whether an attractor has been detected.
    pub attractor_detected: bool,
    /// 5-HT2A signal (psychedelic consciousness amplifier).
    pub sht_2a_signal: f32,
    /// GABA-A signal (global gain reduction).
    pub gaba_a_signal: f32,
    /// Substrate feasibility [0,1] from SubstrateRequirements.
    /// Scales Equation V2 consciousness to reflect substrate limitations.
    pub substrate_feasibility: f64,

    // ── Substrate requirement dimensions → consciousness coupling ────
    /// Binding capability [0,1] — modulates CoreComponent::Binding.
    pub binding_capability: f64,
    /// Workspace capability [0,1] — modulates CoreComponent::Workspace.
    pub workspace_capability: f64,
    /// Attention capability [0,1] — modulates phi_attention_weight in CoreComponent::Attention.
    pub attention_capability: f64,

    // ── Moral topology → consciousness coupling ─────────────────────
    /// Moral drift magnitude from moral_drift(20). Higher = greater shift.
    /// Used for epistemic quality attenuation in EquationV2 Layer 3.
    pub moral_drift: f64,
    /// Composite moral anomaly score [0,1] from MoralAnomalyReport.
    /// Used for unified consciousness dampening alongside bath coupling.
    pub moral_anomaly_score: f64,

    // ── HOT (Higher-Order Thought) → Recursion component ──────────
    /// Normalized HOT recursion depth [0.0, 1.0].
    /// Computed as: (meta_cognition.depth / 3.0) × substrate.hot_capability.
    /// Replaces the hardcoded Recursion=0.5 in ConsciousnessEquationV2.
    /// Defaults to 0.5 when meta_cognition is disabled (backward compat).
    pub hot_depth: f64,

    // ── CPG sync → consciousness coupling (Varela et al. 2001) ────────
    /// CPG oscillator synchronization index [0.0, 1.0].
    /// Modulates unified consciousness ±5%: full sync → +5%, no sync → −5%.
    /// Science: Varela et al. (2001) — large-scale neural synchrony correlates
    /// with conscious awareness; Engel & Singer (2001) — binding-by-synchrony.
    pub cpg_sync_index: f64,

    // ── Cantor metacognitive depth → consciousness coupling ─────────
    /// Self-similarity of the most recent GWT-promoted CRHV [0.0, 1.0].
    /// Higher values indicate richer fractal self-reference structure (strange loops).
    /// Modulates unified consciousness ±3%: deep self-similarity → richer experience.
    /// Science: Hofstadter (1979) — strange loops as substrate of consciousness;
    ///          Metzinger (2003) — self-model depth correlates with phenomenal richness.
    pub cantor_metacognitive_depth: f64,

    // ── Governance collective Phi → consciousness coupling ──────────
    /// Collective Phi from the most recent governance tally [0.0, 1.0].
    /// Measures inter-agent consciousness integration during democratic deliberation.
    /// Modulates unified consciousness ±2%: high collective_phi → richer social consciousness.
    /// Neutral at 0.0 (no governance data). Only active with feature `mycelix`.
    /// Science: Haidt (2012) — shared moral reasoning amplifies collective intelligence;
    ///          Woolley et al. (2010) — collective intelligence emerges from social sensitivity.
    pub governance_collective_phi: f64,

    // ── GWT broadcast state → Workspace component ───────────────────
    /// Whether a GWT broadcast (workspace ignition) occurred this cycle.
    /// Replaces coherence-proxy for Workspace component in EquationV2.
    /// Science: Dehaene & Changeux (2011) — workspace ignition is the signature
    /// of conscious access; mere coherence without ignition is subliminal.
    pub gwt_broadcast_occurred: bool,
    /// Size of the winning GWT coalition (number of participating modules).
    /// Larger coalitions indicate broader workspace access.
    pub gwt_coalition_size: u32,

    // ── Prediction precision → Efficacy component ─────────────────────
    /// Precision (inverse variance) of recent prediction errors [0.1, 10.0].
    /// Distinguishes genuine predictive success from inactivity: high precision
    /// + low PE = real efficacy; low precision + low PE = uncertain/sleeping.
    /// Science: Feldman & Friston (2010) — precision weighting is the core
    /// mechanism of predictive processing.
    pub prediction_precision: f32,

    // ── Knowledge grounding → epistemic quality coupling ─────────────
    /// Knowledge grounding score [0.0, 1.0] from the knowledge engine.
    /// Combines fact relevance and certainty: high grounding → more confident reasoning.
    /// Neutral at 0.5 when knowledge engine is disabled.
    /// Science: Baddeley (2000) — working memory benefits from grounded semantic content.
    pub knowledge_grounding: f64,

    /// Knowledge coherence score [0.0, 1.0] from the knowledge engine.
    /// Composite of graph size (log-scaled), calibration quality (1 - ECE), and
    /// contradiction pressure (penalised by contradiction_count). A large, well-calibrated,
    /// contradiction-free graph scores near 1.0; empty or inconsistent graphs near 0.0.
    /// Neutral at 0.0 when knowledge engine is disabled (weight keeps impact minimal).
    /// Formula: (log2(graph_size+1)/10) × (1-ece) × (1/(1 + contradictions×0.1))
    /// Science: Stanovich (2009) — epistemic rationality; Guo et al. (2017) — calibration.
    pub knowledge_coherence: f64,

    // ── Glyph coherence → symbolic consciousness coupling ─────────────
    /// Glyph field coherence [0.0, 0.95] from the GlyphManager.
    /// Measures integration across all 11 Field Modalities (symbolic consciousness depth).
    /// Modulates unified consciousness ±2%: high coherence → deeper symbolic integration.
    /// Neutral at 0.0 when glyph_codex feature is disabled.
    /// Science: Jung (1959) — archetypal integration; Grof (1985) — consciousness cartography.
    pub glyph_coherence: f64,

    // ── CfC temporal coherence → consciousness coupling ──────────────
    /// CfC temporal coherence phi contribution [0.0, 1.0] from the VoiceCoherenceBridge.
    /// Measures how well the CfC temporal dynamics contribute to integrated information.
    /// Additive nudge to Knowledge component in EquationV2: max +5% at perfect coherence.
    /// Neutral at 0.0 when voice coherence bridge is inactive.
    /// Science: Clark (2013) — temporal integration supports unified conscious experience.
    pub temporal_coherence_phi: f32,
}

/// Dynamic weights for the unified consciousness computation.
///
/// Self-calibrates based on structural Phi decomposition: high emergence
/// (whole > sum of parts) boosts spectral weight, low emergence boosts
/// equation/pipeline weights.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConsciousnessWeights {
    /// Weight for spectral MIP Phi (IIT). Default: 0.35
    pub spectral: f64,
    /// Weight for consciousness equation V2 (7-theory). Default: 0.25
    pub equation: f64,
    /// Weight for unified pipeline. Default: 0.25
    pub pipeline: f64,
    /// Weight for multimodal Phi (cross-modal binding). Default: 0.15
    pub multimodal: f64,
}

impl Default for ConsciousnessWeights {
    fn default() -> Self {
        Self {
            spectral: 0.35,
            equation: 0.25,
            pipeline: 0.25,
            multimodal: 0.15,
        }
    }
}

#[allow(dead_code)] // is_normalized used in tests
impl ConsciousnessWeights {
    /// Normalize weights so they sum to 1.0, preserving ratios.
    /// If all weights are zero, returns default weights.
    pub fn normalize(&mut self) {
        let sum = self.spectral + self.equation + self.pipeline + self.multimodal;
        if sum < 1e-12 {
            *self = Self::default();
            return;
        }
        let inv = 1.0 / sum;
        self.spectral *= inv;
        self.equation *= inv;
        self.pipeline *= inv;
        self.multimodal *= inv;
    }

    /// Check if weights sum to approximately 1.0.
    pub fn is_normalized(&self) -> bool {
        let sum = self.spectral + self.equation + self.pipeline + self.multimodal;
        (sum - 1.0).abs() < 1e-6
    }

    /// Return weights as an array [spectral, equation, pipeline, multimodal].
    pub fn as_array(&self) -> [f64; 4] {
        [self.spectral, self.equation, self.pipeline, self.multimodal]
    }
}

/// Configuration for moral topology → consciousness coupling.
///
/// Two mechanisms:
/// - **Drift-driven epistemic attenuation**: High moral drift reduces the Knowledge
///   component in EquationV2, reflecting epistemic humility during value shifts.
/// - **Anomaly dampening**: High moral anomaly score dampens unified consciousness
///   alongside the existing bath (Seth 2013) coupling terms.
#[derive(Debug, Clone)]
pub(crate) struct MoralConsciousnessCoupling {
    /// Whether moral-consciousness coupling is active.
    pub enabled: bool,
    /// Maximum attenuation of epistemic quality from moral drift (default: 0.30).
    /// At `drift_saturation`, epistemic quality is reduced by this fraction.
    pub drift_epistemic_attenuation: f64,
    /// Drift value at which attenuation saturates (default: 0.5).
    pub drift_saturation: f64,
    /// Strength of anomaly dampening on unified consciousness (default: 0.15).
    pub anomaly_dampening_strength: f64,
}

impl Default for MoralConsciousnessCoupling {
    fn default() -> Self {
        Self {
            enabled: true,
            drift_epistemic_attenuation: 0.30,
            drift_saturation: 0.5,
            anomaly_dampening_strength: 0.15,
        }
    }
}

/// Internal cache for inter-cycle persistence.
#[derive(Debug, Clone)]
pub struct ConsciousnessEngineCache {
    pub(crate) last_spectral_mip_phi: Option<f64>,
    pub(crate) last_hierarchical_mip_phi: Option<f64>,
    pub(crate) last_structural_phi: Option<StructuralPhiResult>,
    pub(crate) last_sigma: Option<f64>,
    pub(crate) last_multimodal_phi: f64,
    pub(crate) last_equation_v2_consciousness: f64,
    pub(crate) last_pipeline_consciousness: f64,
    /// Limiting component from last equation v2 computation
    pub(crate) last_limiting_component: Option<CoreComponent>,
    /// Dynamic consciousness weights (self-calibrating).
    pub(crate) weights: ConsciousnessWeights,
    /// EMA-smoothed emergence ratio from structural Phi.
    pub(crate) smoothed_emergence_ratio: Option<f64>,
    /// Rolling window of recent weight snapshots for variance computation.
    pub(crate) weight_history: std::collections::VecDeque<[f64; 4]>,
    /// Consecutive cycles with weight variance < 0.001 (for Converged detection).
    pub(crate) converged_streak: usize,
    /// Last PAC modulation index from ConsciousnessEquationV2 (for CTC wiring).
    #[cfg(feature = "ctc_wiring")]
    pub(crate) last_pac_modulation: f64,
    /// Last multimodal binding coherence (for CTC wiring).
    #[cfg(feature = "ctc_wiring")]
    pub(crate) last_binding_coherence: f64,
}

impl Default for ConsciousnessEngineCache {
    fn default() -> Self {
        Self {
            last_spectral_mip_phi: None,
            last_hierarchical_mip_phi: None,
            last_structural_phi: None,
            last_sigma: None,
            last_multimodal_phi: 0.0,
            last_equation_v2_consciousness: 0.0,
            last_pipeline_consciousness: 0.0,
            last_limiting_component: None,
            weights: ConsciousnessWeights::default(),
            smoothed_emergence_ratio: None,
            weight_history: std::collections::VecDeque::new(),
            converged_streak: 0,
            #[cfg(feature = "ctc_wiring")]
            last_pac_modulation: 0.0,
            #[cfg(feature = "ctc_wiring")]
            last_binding_coherence: 0.0,
        }
    }
}
