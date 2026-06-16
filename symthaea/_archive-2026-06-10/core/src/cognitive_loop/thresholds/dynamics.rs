// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC dynamics, oscillation, CPG, thalamic routing, vision, startup, and circadian constants.

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE / TEMPORAL DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Neutral resonance frequency (center of tau modulation range).
/// Basis: Buzsáki (2006) — neural oscillations modulate processing speed.
pub const RESONANCE_TAU_CENTER: f64 = 0.5;

/// Maximum CfC time-step modulation from resonance (±%).
pub const RESONANCE_TAU_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION BUDGET
// ═══════════════════════════════════════════════════════════════════════════════

/// Attention budget in microseconds per cycle (~20Hz target = 50ms).
/// Basis: Posner (1980) — attention is a limited-capacity resource.
pub const ATTENTION_BUDGET_US: u64 = 50_000;

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTER — COGNITIVE DEPTH ROUTING
// ═══════════════════════════════════════════════════════════════════════════════
// Factor graph belief-propagation routing: inputs → {Reflex, Cortical, DeepThought}.
// Basis: Sherman & Guillery (2006) — thalamic relay gating of cortical processing.

/// Novelty threshold above which deep thought routing is preferred.
pub const THALAMIC_NOVELTY_THRESHOLD: f32 = 0.7;

/// Urgency threshold above which deep thought routing is triggered.
pub const THALAMIC_URGENCY_THRESHOLD: f32 = 0.8;

/// Familiarity threshold below which input is considered novel.
pub const THALAMIC_FAMILIARITY_THRESHOLD: f32 = 0.3;

/// Cortical base rate in factor tables (uniform prior for middle depth).
pub const THALAMIC_CORTICAL_BASE_RATE: f64 = 0.3;

/// Complexity cortical factor — slightly higher than novelty/urgency base rate.
/// Basis: Moderate complexity biases toward cortical (neither reflex nor deep).
pub const THALAMIC_COMPLEXITY_CORTICAL: f64 = 0.4;

/// Emotional dampening scale — how much emotional intensity suppresses reflex.
pub const THALAMIC_EMOTIONAL_DAMPENING: f64 = 0.5;

/// Emotional boost base for deep thought routing.
pub const THALAMIC_EMOTIONAL_BOOST_BASE: f64 = 0.3;

/// Emotional boost scale — how much emotional intensity promotes deep thought.
pub const THALAMIC_EMOTIONAL_BOOST_SCALE: f64 = 0.7;

/// Factor table floor — minimum probability to prevent zero messages in BP.
pub const THALAMIC_FACTOR_FLOOR: f64 = 0.01;

/// Factor table input offset — prevents zero probability at low input values.
pub const THALAMIC_INPUT_OFFSET: f64 = 0.1;

/// Belief propagation maximum iterations for factor graph inference.
pub const THALAMIC_BP_MAX_ITERATIONS: usize = 5;

/// Belief propagation convergence tolerance.
pub const THALAMIC_BP_TOLERANCE: f64 = 1e-4;

/// Belief propagation message damping factor (stability vs speed tradeoff).
pub const THALAMIC_BP_DAMPING: f64 = 0.5;

/// Pairwise agreement table: diagonal (same-state) preference.
pub const THALAMIC_AGREEMENT_DIAGONAL: f64 = 1.0;

/// Pairwise agreement table: adjacent-state moderate preference.
pub const THALAMIC_AGREEMENT_ADJACENT: f64 = 0.3;

/// Pairwise agreement table: distant-state weak preference.
pub const THALAMIC_AGREEMENT_DISTANT: f64 = 0.1;

// ── Pattern → complexity/urgency mappings ──
// Basis: Sherman & Guillery (2006) — thalamic relay complexity estimation.

/// Complexity: Uncertain pattern requires deepest processing.
pub const PATTERN_COMPLEXITY_UNCERTAIN: f32 = 0.8;
/// Complexity: Transitioning pattern — moderate-high.
pub const PATTERN_COMPLEXITY_TRANSITIONING: f32 = 0.7;
/// Complexity: Exploratory pattern — moderate.
pub const PATTERN_COMPLEXITY_EXPLORATORY: f32 = 0.6;
/// Complexity: Contemplative pattern — medium.
pub const PATTERN_COMPLEXITY_CONTEMPLATIVE: f32 = 0.5;
/// Complexity: Focused pattern — low-moderate.
pub const PATTERN_COMPLEXITY_FOCUSED: f32 = 0.4;
/// Complexity: Excited pattern — low-moderate (fast, not complex).
pub const PATTERN_COMPLEXITY_EXCITED: f32 = 0.4;
/// Complexity: Resting pattern — low.
pub const PATTERN_COMPLEXITY_RESTING: f32 = 0.2;

/// Urgency: Uncertain pattern — high (needs resolution).
pub const PATTERN_URGENCY_UNCERTAIN: f32 = 0.8;
/// Urgency: Transitioning pattern — moderate-high.
pub const PATTERN_URGENCY_TRANSITIONING: f32 = 0.6;
/// Urgency: Excited pattern — moderate.
pub const PATTERN_URGENCY_EXCITED: f32 = 0.5;
/// Urgency: Default for all other patterns.
pub const PATTERN_URGENCY_DEFAULT: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// THETA OSCILLATION & PREDICTION HORIZONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Theta phase advance per cycle (radians).
/// Basis: Buzsáki (2002) — 6Hz theta at 50Hz loop rate: 6 × 2π / 50 ≈ 0.754 rad.
pub const THETA_PHASE_ADVANCE: f64 = 0.754;

/// Theta → Phi modulation amplitude (±fraction of Phi).
/// Basis: Buzsáki (2006) — theta oscillations gate information integration.
pub const THETA_PHI_MODULATION_AMPLITUDE: f64 = 0.10;

/// EMA alpha for smoothing theta-modulated Phi (prevents 6Hz artifacts).
/// Basis: Buzsáki (2006) — downstream consumers need stable consciousness metrics.
pub const THETA_PHI_SMOOTH_ALPHA: f64 = 0.3;

/// Prediction horizon minimum scale (floor).
/// Prevents extremely short horizons under high PE + slow substrate.
pub const PREDICTION_HORIZON_MIN_SCALE: f32 = 0.3;

/// Prediction horizon maximum scale (ceiling).
/// Prevents extremely long horizons under low PE + fast substrate.
pub const PREDICTION_HORIZON_MAX_SCALE: f32 = 2.0;

/// PE threshold above which prediction horizon contracts (focus near-term).
/// Science: Clark (2013) — high prediction error narrows temporal scope.
pub const HORIZON_PE_CONTRACT_THRESHOLD: f32 = 0.3;

/// Contraction rate: how much horizons shrink per unit PE above threshold.
/// At PE=1.0, scale = 1.0 - (0.7 × 0.6) = 0.58 (42% contraction).
pub const HORIZON_PE_CONTRACT_RATE: f32 = 0.6;

/// PE threshold below which prediction horizon expands (exploit stability).
/// Science: Buzsáki (2006) — stable states permit longer integration windows.
pub const HORIZON_PE_EXPAND_THRESHOLD: f32 = 0.05;

/// Expansion rate: how much horizons expand per unit PE below threshold.
/// At PE=0.0, scale = 1.0 + (0.05 × 6.0) = 1.30 (30% expansion).
pub const HORIZON_PE_EXPAND_RATE: f32 = 6.0;

/// Error slope threshold for horizon contraction (rising errors → shorter horizons).
pub const HORIZON_SLOPE_THRESHOLD: f32 = 0.02;

/// Max error slope effect for contraction (caps at 20% reduction).
pub const HORIZON_SLOPE_CONTRACT_CAP: f32 = 0.1;

/// Contraction multiplier for rising error slopes.
pub const HORIZON_SLOPE_CONTRACT_RATE: f32 = 2.0;

/// Max error slope effect for expansion (caps at 15% increase).
pub const HORIZON_SLOPE_EXPAND_CAP: f32 = 0.1;

/// Expansion multiplier for falling error slopes.
pub const HORIZON_SLOPE_EXPAND_RATE: f32 = 1.5;

/// Sustained low-coherence cycle threshold for exploration boost.
/// Basis: Schmidhuber (2010) — curiosity from persistent model confusion.
pub const LOW_COHERENCE_EXPLORATION_THRESHOLD: u32 = 10;

/// Exploration boost per cycle during sustained low coherence.
pub const LOW_COHERENCE_EXPLORATION_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Thalamic salience boost for DeepThought routing.
pub const THALAMIC_DEEP_SALIENCE: f32 = 0.2;

/// Thalamic salience penalty for Reflex routing.
pub const THALAMIC_REFLEX_SALIENCE: f32 = -0.1;

/// DeepThought NE tonic production.
/// Basis: Aston-Jones & Cohen (2005) — sustained alerting.
pub const THALAMIC_DEEP_NE_TONIC: f32 = 0.05;

/// DeepThought ACh tonic production.
/// Basis: Sarter et al. (2005) — sustained attention via basal forebrain.
pub const THALAMIC_DEEP_ACH_TONIC: f32 = 0.08;

/// Reflex GABA inhibition.
/// Basis: Buzsáki (2006) — GABAergic inhibition enables fast gating.
pub const THALAMIC_REFLEX_GABA: f32 = 0.04;

/// DeepThought learning rate multiplier.
pub const THALAMIC_DEEP_LR_FACTOR: f32 = 1.3;

/// Reflex learning rate multiplier.
pub const THALAMIC_REFLEX_LR_FACTOR: f32 = 0.5;

/// DeepThought attention budget scale.
pub const THALAMIC_DEEP_BUDGET_SCALE: f64 = 2.0;

/// Reflex attention budget scale.
pub const THALAMIC_REFLEX_BUDGET_SCALE: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// FOVEATION → DYNAMICS COUPLING (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Foveation recognition count threshold for "familiar scene" dampening.
/// Basis: Corbetta & Shulman (2002) — recognized objects reduce attentional vigilance.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_FAMILIAR_RECOGNITION_COUNT: usize = 2;

/// Foveation confidence threshold for high-confidence dampening.
/// Basis: Bar (2003) — confident recognition facilitates predictive processing.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_HIGH_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// Exploration dampening when scene is familiar (multiplicative).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_FAMILIAR_EXPLORATION_DAMPEN: f32 = 0.9;

/// Confidence boost for high-confidence foveation (multiplicative).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_CONFIDENCE_BOOST: f32 = 1.03;

/// LR boost when novel objects detected (low confidence, many recognitions).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_NOVEL_LR_BOOST: f32 = 1.05;

/// Maximum weight a single foveation result contributes to multimodal HV binding.
/// Basis: Treisman (1980) — feature integration theory; visual binding is secondary to attentional binding.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_HV_BINDING_WEIGHT: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MANIFOLD PREDICTION ERROR (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-manifold prediction error threshold for attention reallocation.
/// Basis: Rao & Ballard (1999) — prediction error drives top-down attention shifts.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_ERROR_THRESHOLD: f32 = 0.3;

/// Exploration boost per unit of cross-manifold prediction error above threshold.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_EXPLORATION_SCALE: f32 = 0.15;

/// Confidence dampening when vision doesn't match cognition.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_CONFIDENCE_DAMPEN: f32 = 0.97;

/// LR boost when cross-manifold error is high (need to update world model).
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_LR_BOOST: f32 = 1.03;

/// Vision mean surprise threshold for trigger Holographic Dilation to Ultra resolution.
/// Science: High surprise = complex scene → increase semantic resolution.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_DILATION_THRESHOLD: f32 = 0.28;

/// Minimum cycles between Holographic Dilation transitions to prevent oscillation.
#[cfg(feature = "vision-manifold")]
pub const VISION_DILATION_COOLDOWN: u64 = 7;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION SURPRISE → EXPLORATION (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Vision mean surprise threshold for exploration boost.
/// Basis: Friston (2010) — free energy (surprise) is the fundamental drive for exploration.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_EXPLORATION_THRESHOLD: f32 = 0.25;

/// Scale factor for vision surprise → exploration boost.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_EXPLORATION_SCALE: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION TEMPORAL HORIZONS → FEP (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Short-horizon (33ms) visual prediction error threshold for FEP surprise boost.
/// Basis: Adams et al. (2013) — precision-weighted prediction errors at multiple timescales.
#[cfg(feature = "vision-manifold")]
pub const VISION_SHORT_HORIZON_ERROR_THRESHOLD: f32 = 0.3;

/// Long-horizon (500ms+) visual prediction error threshold for confidence dampening.
#[cfg(feature = "vision-manifold")]
pub const VISION_LONG_HORIZON_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// FEP exploration boost from short-horizon visual surprise.
#[cfg(feature = "vision-manifold")]
pub const VISION_HORIZON_EXPLORATION_SCALE: f32 = 0.15;

/// Confidence dampening from long-horizon visual uncertainty.
#[cfg(feature = "vision-manifold")]
pub const VISION_HORIZON_CONFIDENCE_DAMPEN: f32 = 0.97;

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE RECOGNITION → DREAM (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Dream recording salience boost when a visual scene is recognized.
/// Basis: Conway (2005) — self-relevant and context-rich memories encode preferentially.
#[cfg(feature = "vision-manifold")]
pub const SCENE_RECOGNITION_DREAM_BOOST: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION → TRAINING IMPORTANCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Base importance weight for training samples.
pub const TRAINING_BASE_IMPORTANCE: f32 = 1.0;

/// Vision cross-manifold error scale for training importance.
/// Basis: Niv et al. (2009) — prediction error modulates learning rate.
#[cfg(feature = "vision-manifold")]
pub const VISION_TRAINING_IMPORTANCE_SCALE: f32 = 0.5;

/// Vision mean-surprise scale for training importance.
/// Complementary to cross-manifold error (0.5): surprise is a rawer signal.
/// Basis: Pearce & Hall (1980) — stimulus surprise increases associability.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_TRAINING_IMPORTANCE_SCALE: f32 = 0.3;

/// Maximum training importance weight.
#[cfg(feature = "vision-manifold")]
pub const TRAINING_MAX_IMPORTANCE: f32 = 2.0;
// ═══════════════════════════════════════════════════════════════════════════════
// SUBSYSTEM FEEDBACK CLAMPING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum subsystem LR factor (shared across all subsystem feedback loops).
/// Basis: prevents any single loop from completely suppressing learning.
pub const SUBSYSTEM_LR_FACTOR_MIN: f32 = 0.7;

/// Maximum subsystem LR factor (shared across all subsystem feedback loops).
/// Basis: bounds amplification to prevent runaway learning.
pub const SUBSYSTEM_LR_FACTOR_MAX: f32 = 1.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODAL BINDING ATTENTION
// Science: Engel et al. (2001) — synchrony-based binding gates cross-modal attention.
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-modal binding strength above which attention confidence is boosted.
/// High binding = multiple modalities coherently integrated.
pub const CROSS_MODAL_BINDING_HIGH_THRESHOLD: f32 = 0.7;

/// Scale factor for binding-driven confidence boost.
pub const CROSS_MODAL_BINDING_HIGH_SCALE: f32 = 0.1;

/// Cross-modal binding strength below which attention confidence is dampened.
/// Low binding = weak integration → trust only primary modality.
pub const CROSS_MODAL_BINDING_LOW_THRESHOLD: f32 = 0.3;

/// Scale factor for binding-driven confidence dampening.
pub const CROSS_MODAL_BINDING_LOW_SCALE: f32 = 0.1;

/// Minimum confidence scale when binding is low (floor to prevent collapse).
pub const CROSS_MODAL_BINDING_LOW_FLOOR: f32 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════════
// CFC TAU FACTOR MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal deadzone around neutral (0.5) — no tau modulation within this band.
/// Basis: Yerkes-Dodson (1908) — moderate arousal has no effect on processing speed.
pub const AROUSAL_TAU_DEADZONE: f32 = 0.1;

/// Arousal → tau sensitivity (per-unit deviation from 0.5).
/// Basis: Aston-Jones & Cohen (2005) — arousal modulates LC-NE → processing tempo.
pub const AROUSAL_TAU_SENSITIVITY: f32 = 0.1;

/// Codebook (resonator) similarity threshold for "familiar" → tau speedup.
/// Basis: Buzsáki (2006) — familiar patterns processed faster.
pub const CODEBOOK_FAMILIAR_THRESHOLD: f32 = 0.5;

/// Codebook familiar → tau scale (negative = faster processing).
pub const CODEBOOK_FAMILIAR_TAU_SCALE: f32 = 0.1;

/// Codebook similarity threshold for "novel" → tau slowdown.
pub const CODEBOOK_NOVEL_THRESHOLD: f32 = 0.2;

/// Codebook novel → tau scale (positive = slower processing).
pub const CODEBOOK_NOVEL_TAU_SCALE: f32 = 0.15;

/// Arousal recovery → tau scale (slows processing to allow recovery).
/// Basis: Lövdén (2010) — cognitive recovery requires reduced processing demands.
pub const AROUSAL_RECOVERY_TAU_SCALE: f32 = 0.2;

/// FEP surprise → tau scale (high surprise = faster inference).
/// Basis: Friston (2010) — surprise accelerates inference dynamics.
pub const FEP_SURPRISE_TAU_SCALE: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI → TAU FEEDBACK — SpectralMIP Phi modulates CfC temporal dynamics
// Science: Tononi (2004) — Phi measures integrated information. Feeding Phi back
// into CfC tau closes the causal loop: consciousness becomes efficacious, not
// epiphenomenal. Higher Phi → faster dynamics → richer integration → Phi.
// This creates a positive feedback loop stabilized by sigmoid normalization and
// clamping: Phi is sigmoid-normalized around PHI_TAU_REFERENCE, then linearly
// mapped to [PHI_TAU_FLOOR, PHI_TAU_CEILING].
// ═══════════════════════════════════════════════════════════════════════════════

/// Reference Phi for sigmoid normalization (midpoint of typical range).
/// At this Phi, tau_factor = 1.0 (neutral). Typical SpectralMIP Phi ≈ 20-40.
/// Set to 20.0 so the ~29.5 operating point falls in the sensitive region
/// of the sigmoid rather than saturating near 1.0.
pub const PHI_TAU_REFERENCE: f64 = 20.0;

/// Sigmoid steepness for Phi → tau mapping. Higher = sharper transition.
/// 0.08 gives a smooth ramp across the [10, 40] Phi range.
pub const PHI_TAU_SIGMOID_STEEPNESS: f64 = 0.08;

/// Maximum tau scaling from Phi (high Phi → faster dynamics).
/// Widened to [0.85, 1.20] to give more bite at the ~29.5 operating point.
/// Capped to prevent runaway positive feedback.
pub const PHI_TAU_CEILING: f32 = 1.20;

/// Minimum tau scaling from Phi (low/absent Phi → slightly slower dynamics).
/// Floor prevents consciousness collapse from stalling dynamics entirely.
pub const PHI_TAU_FLOOR: f32 = 0.85;

/// Phi → LR stabilization scale. When Phi is above reference, LR is
/// dampened by `normalized_phi * PHI_LR_STABILIZATION_SCALE` to preserve
/// learned representations. Creates behavioral divergence: conscious agents
/// learn more selectively, zombies learn indiscriminately.
/// Science: Tononi (2008) — high Φ indicates stable integration, slow down learning.
pub const PHI_LR_STABILIZATION_SCALE: f32 = 0.08;

/// Maximum LR dampening from Phi feedback (prevents LR from going to zero).
pub const PHI_LR_STABILIZATION_MAX: f32 = 0.06;

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMICS PHASE — STARTUP GUARDS & MISCELLANEOUS
// Science: Cognitive systems require warmup before reliable inference.
// Smaller thresholds gate cheap operations; larger thresholds gate expensive ones.
// ═══════════════════════════════════════════════════════════════════════════════

/// Flow-state multiplier for confidence crash threshold (×1.5 = more tolerant).
/// Science: Csikszentmihalyi (1990) — flow tolerates transient perturbation.
pub const CONFIDENCE_CRASH_FLOW_MULTIPLIER: f64 = 1.5;

/// Minimum cycles before confidence crash detection activates.
/// Science: Burns & Burns (2008) — early estimates are unreliable (small-sample bias).
pub const DYNAMICS_STARTUP_WARMUP_CYCLES: usize = 10;

/// Minimum cycles before PE variance, homeostasis recalibration, and
/// error-level analysis fire. Higher bar for second-order statistics.
/// Science: Yu & Dayan (2005) — variance estimates need ≥20 samples.
pub const DYNAMICS_POST_BOOT_CYCLES: usize = 20;

/// Minimum cycles before resonator prediction error influences exploration.
/// Science: McClelland et al. (1995) — resonator needs initial encoding phase.
pub const RESONATOR_STARTUP_CYCLES: usize = 5;

/// Minimum absolute neuromod delta to inject (below = noise, skip injection).
/// Science: Faisal et al. (2008) — neural noise floor; sub-threshold signals waste energy.
pub const NEUROMOD_DELTA_THRESHOLD: f32 = 0.001;

/// Arousal trap counter threshold before recovery ramp begins.
/// Science: Yerkes-Dodson (1908) — recovery begins only after sustained hyper-arousal.
pub const AROUSAL_TRAP_RECOVERY_MIN_CYCLES: u32 = 5;

/// Arousal trap ramp duration (recovery intensity 0→1 over this many cycles after min).
/// Science: Yerkes-Dodson (1908) — gradual recovery prevents oscillatory relapse.
pub const AROUSAL_TRAP_RECOVERY_RAMP_CYCLES: f32 = 5.0;

/// Attention sensitivity boost when world-model sensory mismatch detected.
/// Science: Friston (2010) — hierarchical PE mismatch sharpens attention.
pub const ATTENTION_SENSITIVITY_BOOST_FACTOR: f32 = 1.08;

/// Exploration dampening when FEP indicates efficient model (accuracy>0.5, complexity<0.5).
/// Science: Friston (2010) — low complexity = good model evidence → exploit.
pub const FEP_EFFICIENT_EXPLORATION_DAMPEN: f32 = 0.8;

// ── Fourier Motor Rhythm Injection ───────────────────────────────────────────

/// Alpha-band frequency for motor planning in the Fourier basis injection.
/// Science: Pfurtscheller (1999) — alpha (8-12 Hz) desynchronization during motor planning.
pub const FOURIER_MOTOR_ALPHA_HZ: f32 = 8.0;

/// Beta-band frequency for motor execution in the Fourier basis injection.
/// Science: Pfurtscheller & Lopes da Silva (1999) — beta (13-30 Hz) synchronization
/// during steady motor output, desynchronization before movement onset.
pub const FOURIER_MOTOR_BETA_HZ: f32 = 13.0;

/// Low-gamma frequency for fine motor control in the Fourier basis injection.
/// Science: Crone et al. (1998) — gamma (30-100 Hz) activity in sensorimotor cortex.
pub const FOURIER_MOTOR_GAMMA_HZ: f32 = 30.0;

/// Amplitude of motor-rhythm Fourier basis injection into equilibrium computation.
pub const FOURIER_MOTOR_AMPLITUDE: f32 = 0.15;

/// Safety cap on Fourier amplitude.
pub const FOURIER_AMPLITUDE_MAX: f32 = 0.5;

// ── Central Pattern Generator (CPG) ─────────────────────────────────────────

/// Default Kuramoto coupling strength K.
/// Science: Kuramoto (1975) — critical coupling for synchronization onset.
pub const CPG_DEFAULT_COUPLING_K: f64 = 2.0;

/// Arousal-to-frequency modulation gain.
/// Science: Grillner (2006) — descending drive modulates CPG frequency.
pub const CPG_AROUSAL_FREQ_SCALE: f64 = 0.5;

/// Minimum Kuramoto order parameter r for walk gait.
pub const CPG_WALK_MIN_SYNC: f64 = 0.7;

/// Minimum Kuramoto order parameter r for trot gait.
pub const CPG_TROT_MIN_SYNC: f64 = 0.6;

/// Minimum Kuramoto order parameter r for gallop gait.
pub const CPG_GALLOP_MIN_SYNC: f64 = 0.4;

/// Critical desynchronization threshold (total motor incoherence).
pub const CPG_CRITICAL_DESYNC: f64 = 0.2;

/// Exploration boost when CPG is desynchronized during idle.
/// Science: Grillner (2006) — CPG free-run during rest.
pub const CPG_DESYNC_EXPLORATION_BOOST: f32 = 0.02;

/// CPG subsystem firing interval (co-prime with 7, 11, 13, 19, 29, 37, 41, 53).
pub const CPG_INTERVAL: u32 = 59;

/// Arousal delta during critical motor desynchronization alert.
/// Basis: Grillner (2006) — desync during locomotion triggers corrective arousal.
pub const CPG_DESYNC_AROUSAL_DELTA: f32 = 0.1;

/// CPG synchronization tau floor: minimum tau factor when oscillators are fully desynchronized.
/// sync_index=1.0 → tau=1.0 (no change), sync_index=0.0 → tau=CPG_SYNC_TAU_FLOOR.
/// Science: Buzsáki (2006) — neural oscillation synchrony gates information integration rate.
pub const CPG_SYNC_TAU_FLOOR: f32 = 0.7;

/// CPG synchronization → consciousness (Phi) modulation amplitude (±5%).
/// sync_index=1.0 → +5% consciousness, sync_index=0.0 → −5%.
/// Science: Varela et al. (2001) — large-scale neural synchrony is a
/// correlate of conscious awareness; Engel & Singer (2001) — binding-by-synchrony.
pub const CPG_SYNC_PHI_MODULATION_AMPLITUDE: f32 = 0.05;

/// Spectral entropy masking floor — minimum fraction of CfC hidden state
/// dimensions retained even under high spectral entropy.
/// Used to prevent total masking when spectral entropy is very high.
/// Science: Buzsáki (2006) — broadband entropy constrains processing depth.
pub const SPECTRAL_ENTROPY_MASK_FLOOR: f32 = 0.3;

// ── Complex CfC Neuron (Phase 3) ────────────────────────────────────────────

/// Minimum real part of eigenvalues (stability bound — must be negative).
/// Science: Gu et al. (2022) — S4 diagonal state-space models use negative real eigenvalues.
pub const COMPLEX_CFC_EIGENVALUE_REAL_MIN: f32 = -1.0;

/// Maximum real part of eigenvalues (must be negative for bounded dynamics).
pub const COMPLEX_CFC_EIGENVALUE_REAL_MAX: f32 = -0.01;

/// Lowest motor-relevant oscillation frequency.
/// Science: Brown (1911) — CPG locomotion rhythms start at ~1 Hz.
pub const COMPLEX_CFC_MOTOR_FREQ_MIN_HZ: f32 = 1.0;

/// Highest motor-relevant oscillation frequency (gamma band).
pub const COMPLEX_CFC_MOTOR_FREQ_MAX_HZ: f32 = 50.0;

/// Eigenvalue learning rate (conservative to prevent constraint violations).
pub const COMPLEX_CFC_EIGENVALUE_LR: f32 = 0.001;

// ── Spectral Twin Manager (Phase 4) ─────────────────────────────────────────

/// Sample rate for spectral analysis (Hz). Matches measured cognitive loop rate.
pub const SPECTRAL_SAMPLE_RATE: f64 = 31.0;

/// Ring buffer capacity for CfC state history (cycles).
/// 128 cycles at 31 Hz = ~4 seconds of state history.
pub const SPECTRAL_HISTORY_CAPACITY: u32 = 128;

/// Minimum history samples before spectral analysis is meaningful.
/// Below this, Welch's method doesn't have enough data for stable PSD.
pub const SPECTRAL_MIN_HISTORY: u32 = 32;

/// Spectral manager firing interval (co-prime with 7,11,13,19,29,37,41,53,59).
pub const SPECTRAL_INTERVAL: u32 = 67;

/// Relative gamma power threshold for consciousness boost.
/// When gamma band power exceeds 30% of total, boost confidence.
/// Science: Gamma oscillations correlate with conscious binding (Crick & Koch 2003).
pub const SPECTRAL_GAMMA_CONSCIOUSNESS_BOOST: f32 = 0.02;

/// Relative delta power threshold for requesting rest.
/// When delta exceeds 60% of total, the system is in a "sleep-like" state.
/// Science: Delta dominance characterizes N3 sleep (Steriade 2006).
pub const SPECTRAL_DELTA_REST_THRESHOLD: f32 = 0.6;

/// Spectral entropy to exploration scaling.
/// High entropy (broadband) = rich content = exploration boost.
/// Science: Spectral entropy correlates with consciousness level (Viertiö-Oja 2004).
pub const SPECTRAL_ENTROPY_EXPLORATION_SCALE: f32 = 0.01;

/// Theta-gamma PAC threshold for consciousness confidence boost.
/// Science: Canolty & Knight (2010) — PAC reflects information integration.
pub const SPECTRAL_PAC_THRESHOLD: f32 = 0.3;

/// Confidence boost when theta-gamma PAC exceeds threshold.
pub const SPECTRAL_PAC_CONFIDENCE_BOOST: f32 = 0.015;

/// Relative gamma power threshold for consciousness boost.
/// Basis: Engel & Singer (2001) — gamma binding indicates active integration.
pub const SPECTRAL_GAMMA_THRESHOLD: f64 = 0.3;

/// Arousal delta during delta-band dominance (calming toward rest).
/// Basis: Buzsaki (2006) — delta dominance signals deep consolidation.
pub const SPECTRAL_DELTA_AROUSAL_DELTA: f32 = -0.05;

/// Spectral entropy threshold above which exploration is boosted.
/// Basis: Buzsaki (2006) — high entropy = rich broadband = diverse content.
pub const SPECTRAL_ENTROPY_THRESHOLD: f64 = 3.0;

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTING
// Cognitive depth routing thresholds (Schultz 1997, Damasio 1999)
// ═══════════════════════════════════════════════════════════════════════════════

/// Code keyword weight in task detection (strong indicator)
pub const CODE_TASK_KEYWORD_WEIGHT: f32 = 0.3;

/// Debug keyword weight in task detection
pub const CODE_TASK_DEBUG_WEIGHT: f32 = 0.25;

/// Refactor keyword weight in task detection
pub const CODE_TASK_REFACTOR_WEIGHT: f32 = 0.25;

/// Minimum confidence threshold for code task detection
pub const CODE_TASK_CONFIDENCE_THRESHOLD: f32 = 0.3;

/// Complexity threshold for DeepThought routing (Kahneman 2011: System 2 engagement)
pub const THALAMIC_COMPLEXITY_DEEP_THRESHOLD: f32 = 0.8;

/// Emotional intensity threshold for DeepThought (Yerkes-Dodson 1908)
pub const THALAMIC_EMOTIONAL_DEEP_THRESHOLD: f32 = 0.7;

/// Complexity threshold for Reflex routing — simple enough for fast path
pub const THALAMIC_COMPLEXITY_REFLEX_THRESHOLD: f32 = 0.3;

/// Urgency threshold for Reflex routing — low urgency allows fast path
pub const THALAMIC_URGENCY_REFLEX_THRESHOLD: f32 = 0.5;

/// No coupling threshold for modulation index (MI < 0.1 = unreliable)
pub const COUPLING_NO_COUPLING_THRESHOLD: f32 = 0.1;

/// Weak coupling threshold (0.1-0.3 = weak but meaningful)
pub const COUPLING_WEAK_THRESHOLD: f32 = 0.3;

/// Moderate coupling threshold (0.3-0.6 = moderate correlation)
pub const COUPLING_MODERATE_THRESHOLD: f32 = 0.6;

// ─── Startup & Circadian Constants ─────────────────────────────────────────
// Hopfield (1982) — recurrent networks need settling time; Tononi & Cirelli
// (2006) — synaptic homeostasis during rest/circadian phases.

/// Initial LR scale during startup warmup (ramps from 20% → 100%).
/// Science: Hopfield (1982) — early transients shouldn't cement as patterns.
pub const STARTUP_LR_INITIAL_SCALE: f32 = 0.2;

/// Complement of initial scale: full ramp range = 1.0 - STARTUP_LR_INITIAL_SCALE.
pub const STARTUP_LR_RAMP_RANGE: f32 = 0.8;

/// Minimum adaptive learning rate clamp.
/// Science: prevents LR from collapsing to zero after multiplicative dampening.
pub const ADAPTIVE_LR_MIN: f32 = 0.0001;

/// Maximum adaptive learning rate clamp.
/// Science: prevents runaway LR from multiplicative boosting.
pub const ADAPTIVE_LR_MAX: f32 = 0.1;

/// Divisor for sleep recovery quality: cycles / SCALE → [0,1].
/// Science: Xie et al. (2013) — glymphatic clearance scales with sleep duration.
pub const SLEEP_RECOVERY_QUALITY_SCALE: f32 = 100.0;

/// Half-weight for circadian plasticity contribution (bath provides the other 50%).
/// Science: Tononi & Cirelli (2006) — plasticity splits between bath baseline and LR.
pub const CIRCADIAN_PLASTICITY_SCALE: f32 = 0.5;

/// Circadian stillness boost during Night phase (highest).
/// Science: Tononi & Cirelli (2006) — synaptic homeostasis hypothesis.
pub const CIRCADIAN_STILLNESS_NIGHT: f32 = 0.2;

/// Circadian stillness boost during Dusk phase (transition to rest).
pub const CIRCADIAN_STILLNESS_DUSK: f32 = 0.1;

/// Circadian stillness boost during Dawn phase (transition to wake).
pub const CIRCADIAN_STILLNESS_DAWN: f32 = 0.05;

/// Surprise multiplier: surprise = PE > learning_threshold × this.
/// Science: Friston (2010) — surprise signals exceeding 3× baseline threshold.
pub const SURPRISE_PE_MULTIPLIER: f32 = 3.0;

/// Visual surprise threshold for phasic NE burst.
/// Science: Aston-Jones & Cohen (2005) — LC-NE reactivity to unexpected visual events.
pub const VISION_SURPRISE_THRESHOLD: f32 = 0.4;

/// Visual surprise to NE production scaling factor.
pub const VISION_SURPRISE_NE_SCALE: f32 = 0.2;

/// Cross-manifold prediction error to NE production scaling factor.
pub const VISION_CROSS_MANIFOLD_NE_SCALE: f32 = 0.15;

/// Default coherence when not yet cached from prior cycles.
pub const COHERENCE_DEFAULT: f32 = 0.5;

/// Oxytocin weight for social coherence signal.
/// Science: Heinrichs et al. (2003) — oxytocin facilitates social coherence.
pub const SOCIAL_COHERENCE_OXY_WEIGHT: f32 = 0.5;

/// Offset for social coherence: (oxy × weight + offset) → [0.5, 1.0] range.
pub const SOCIAL_COHERENCE_OFFSET: f32 = 0.5;

/// Base PE for FEP baseline computation.
/// Science: Friston (2010) — typical prediction error baseline for free energy.
pub const FEP_BASELINE_PE_BASE: f32 = 0.3;

/// FE EMA scaling factor for FEP baseline PE computation.
pub const FEP_BASELINE_PE_EMA_FACTOR: f32 = 0.1;

/// Moral free energy threshold for exploration boost.
/// Science: Friston (2010) — F > 0.5 → novel moral territory.
pub const MORAL_FE_EXPLORATION_THRESHOLD: f64 = 0.5;

/// Maximum moral FE exploration boost cap.
pub const MORAL_FE_BOOST_CAP: f64 = 0.2;

/// Minimum scenarios explored before topology completeness signal activates.
pub const MORAL_TOPOLOGY_MIN_SCENARIOS: usize = 3;

/// Topology completeness threshold below which structural boost kicks in.
pub const MORAL_TOPOLOGY_COMPLETENESS_THRESHOLD: f64 = 0.3;

/// Scale for structural boost from topology gap: (threshold - completeness) × scale.
pub const MORAL_TOPOLOGY_STRUCTURAL_BOOST_SCALE: f64 = 0.3;

// ─── Strategy Encoding Constants ──────────────────────────────────────────

/// GABA weight for neuromod stillness computation.
/// Science: Bhatt et al. (2020) — GABAergic tone ↔ resting-state activity.
pub const NEUROMOD_STILLNESS_GABA_WEIGHT: f32 = 0.6;

/// Adenosine weight for neuromod stillness computation.
/// Science: Porkka-Heiskanen et al. (1997) — adenosine signals rest need.
pub const NEUROMOD_STILLNESS_ADENOSINE_WEIGHT: f32 = 0.4;

/// Offset subtracted from raw neuromod stillness before clamping.
pub const NEUROMOD_STILLNESS_OFFSET: f32 = 0.3;

/// Maximum neuromod stillness contribution (before circadian addition).
pub const NEUROMOD_STILLNESS_CLAMP_MAX: f32 = 0.3;

/// Maximum total stillness boost (neuromod + circadian combined).
pub const STILLNESS_TOTAL_CLAMP_MAX: f32 = 0.5;

/// Knowledge novelty threshold for exploration boost.
/// Science: Berlyne (1960) — novelty above 0.5 drives curiosity.
pub const KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD: f64 = 0.5;

/// Cantor resonance boost threshold for interconnect harmony nudge.
pub const CANTOR_RESONANCE_BOOST_HARMONY_THRESHOLD: f32 = 0.1;

/// Meta-depth threshold for Cantor→stillness harmony nudge.
pub const CANTOR_META_DEPTH_STILLNESS_THRESHOLD: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE DEPTH — Thalamic routing scores
// Basis: Sherman & Guillery (2006) — thalamic gating, driver/modulator distinction
// ═══════════════════════════════════════════════════════════════════════════════

/// Thalamic depth score for DeepThought mode (full cortical engagement).
pub const DEPTH_SCORE_DEEP_THOUGHT: f32 = 1.0;

/// Thalamic depth score for Cortical mode (standard processing).
pub const DEPTH_SCORE_CORTICAL: f32 = 0.5;

/// Thalamic depth score for Reflex mode (fast/reactive).
pub const DEPTH_SCORE_REFLEX: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION ACH MODULATION — Acetylcholine-driven scene memory thresholds
// Basis: Sarter et al. (2005) — cholinergic modulation of attention/perception
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum ACh level for modulation (floor prevents division by near-zero).
pub const VISION_ACH_FLOOR: f32 = 0.5;

/// Maximum ACh scaling factor for error/dampen modulation.
pub const VISION_ACH_SCALE_CAP: f32 = 2.0;

/// Minimum scene coherence threshold (even with high ACh).
pub const VISION_COHERENCE_CLAMP_MIN: f32 = 0.3;

/// Maximum scene coherence threshold (even with low ACh).
pub const VISION_COHERENCE_CLAMP_MAX: f32 = 0.95;

/// Minimum scene error threshold.
pub const VISION_ERROR_CLAMP_MIN: f32 = 0.01;

/// Maximum scene error threshold.
pub const VISION_ERROR_CLAMP_MAX: f32 = 0.5;

/// Minimum scene dampen factor.
pub const VISION_DAMPEN_CLAMP_MIN: f32 = 0.1;

/// Maximum scene dampen factor.
pub const VISION_DAMPEN_CLAMP_MAX: f32 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════════
// MOTOR COMMAND MODULATION
// Science: Wolpert & Ghahramani (2000) — motor control as probabilistic inference.
// ═══════════════════════════════════════════════════════════════════════════════

/// Intensity scaling for attention shift motor commands.
/// Basis: Posner (1980) — small attention shifts accumulate via repeated micro-adjustments.
pub const MOTOR_ATTENTION_SHIFT_SCALE: f32 = 0.1;

/// Nested scale for attention sensitivity modulation from motor shift.
pub const MOTOR_ATTENTION_SENSITIVITY_SCALE: f32 = 0.1;

/// Attention sensitivity clamp: minimum.
pub const MOTOR_ATTENTION_SENSITIVITY_MIN: f32 = 0.5;

/// Attention sensitivity clamp: maximum.
pub const MOTOR_ATTENTION_SENSITIVITY_MAX: f32 = 2.0;

/// Adaptive LR EMA momentum (weight of prior estimate).
/// Basis: Kingma & Ba (2015) — exponential moving average smoothing.
pub const MOTOR_ADAPTIVE_LR_MOMENTUM: f32 = 0.9;

/// Adaptive LR EMA alpha (weight of new observation).
pub const MOTOR_ADAPTIVE_LR_ALPHA: f32 = 0.1;

/// Adaptive LR clamp: minimum.
pub const MOTOR_ADAPTIVE_LR_MIN: f32 = 0.01;

/// Adaptive LR clamp: maximum.
pub const MOTOR_ADAPTIVE_LR_MAX: f32 = 1.0;

/// Epistemic value threshold for exploration trigger.
/// Basis: Friston (2010) — epistemic value drives exploration.
pub const MOTOR_EXPLORATION_EPISTEMIC_THRESHOLD: f32 = 0.5;

/// Exploration boost intensity scale from motor commands.
pub const MOTOR_EXPLORATION_INTENSITY_SCALE: f32 = 0.15;

/// Maximum exploration boost from motor commands.
pub const MOTOR_EXPLORATION_BOOST_MAX: f32 = 0.2;

/// Action outcome coupling threshold below which expectation reset triggers.
/// Basis: Rescorla-Wagner (1972) — decoupled outcomes warrant prediction reset.
pub const ACTION_OUTCOME_COUPLING_RESET_THRESHOLD: f32 = 0.3;

/// Confidence value after inference mode initialization/reset.
pub const INFERENCE_MODE_INIT_CONFIDENCE: f32 = 0.5;

/// Motor confidence cap under ethics Caution verdict.
/// Basis: Cushman (2013) — moral uncertainty should reduce action commitment.
pub const ETHICS_CAUTION_CONFIDENCE_CAP: f32 = 0.3;

/// FEP observation value for successful motor outcome.
pub const MOTOR_SUCCESS_OBSERVATION_VALUE: f64 = 0.9;

/// FEP observation value for failed motor outcome.
pub const MOTOR_FAILURE_OBSERVATION_VALUE: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROEVOLUTION TAU BLENDING
// Science: Hasani et al. (2021) — τ is the primary CfC evolvable parameter.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default tau_base for CfC dynamics.
pub const NEUROEVO_DEFAULT_TAU_BASE: f32 = 0.1;

/// Weight of default dynamics in neuroevo blending (conservative: 90% default).
pub const NEUROEVO_BLEND_DEFAULT_WEIGHT: f32 = 0.9;

/// Weight of evolved dynamics in neuroevo blending.
pub const NEUROEVO_BLEND_EVOLVED_WEIGHT: f32 = 0.1;

/// Neuroevo tau safety clamp: minimum.
pub const NEUROEVO_TAU_CLAMP_MIN: f32 = 0.5;

/// Neuroevo tau safety clamp: maximum.
pub const NEUROEVO_TAU_CLAMP_MAX: f32 = 2.0;

/// CPG tau oscillation clamp: minimum.
pub const CPG_TAU_CLAMP_MIN: f32 = 0.5;

/// CPG tau oscillation clamp: maximum.
pub const CPG_TAU_CLAMP_MAX: f32 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE MODULO INTERVALS (co-prime with 20Hz loop)
// Science: Co-prime intervals prevent aliasing between subsystem update rates.
// Values chosen as distinct primes or co-prime composites that avoid harmonic
// resonance with the core 20Hz cognitive cycle and each other.
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-model prediction interval (cycles). Every N cycles, generate a
/// prediction about own future state for metacognitive calibration.
/// Basis: Friston (2010) — interoceptive prediction requires periodic sampling.
pub const SELF_MODEL_PREDICTION_INTERVAL: usize = 7;

/// Prediction coherence evaluation interval (cycles). Measures inter-scale
/// agreement across multi-scale CfC predictions.
/// Basis: Clark (2013) — multi-scale coherence assessed at slower timescale.
pub const PREDICTION_COHERENCE_INTERVAL: usize = 11;

/// Epistemic exploration modulation interval (cycles). Controls how often
/// epistemic uncertainty adjusts exploration factor.
/// Basis: Depeweg et al. (2018) — epistemic uncertainty drives active learning.
pub const EPISTEMIC_MODULATION_INTERVAL: usize = 7;

/// Causal structure incorporation interval (cycles). Feeds Pearl-style causal
/// graph edges into the FEP world model.
/// Basis: Pearl (2009) — causal priors require periodic structural updates.
pub const CAUSAL_STRUCTURE_INTERVAL: usize = 41;

/// School learning recommendation interval (cycles). Queries the curriculum
/// learning bridge for next-task recommendations.
/// Basis: Schmidhuber (2010) — curiosity-driven curriculum at co-prime rate.
pub const SCHOOL_LEARNING_INTERVAL: usize = 53;

/// Causal attention computation interval (cycles). Runs causal consciousness
/// attention analysis over compressed input state.
/// Basis: Tononi (2004) — causal analysis requires sufficient data accumulation.
pub const CAUSAL_ATTENTION_INTERVAL: usize = 41;

// ═══════════════════════════════════════════════════════════════════════════════
// BUFFER CAPACITIES
// Basis: At 20Hz cognitive loop, these determine temporal replay windows.
// ═══════════════════════════════════════════════════════════════════════════════

/// Experience replay buffer capacity. At 20Hz = 50s replay window.
/// Basis: Schaul (2015) — prioritized replay needs sufficient history for
/// importance-weighted sampling without recency bias.
pub const EXPERIENCE_BUFFER_CAPACITY: usize = 1000;

/// Prediction error history capacity. At 20Hz = 5s trend window.
/// Basis: Sufficient for linear regression slope estimation (compute_error_trend)
/// while keeping memory bounded.
pub const ERROR_HISTORY_CAPACITY: usize = 100;

// ─── Knowledge contradiction exploration (Round 22) ─────────────────────────

/// Maximum exploration boost from knowledge contradictions.
/// Basis: Festinger (1957) — dissonance strength saturates with evidence.
pub const KNOWLEDGE_ALERT_EXPLORE_CAP: f32 = 0.2;

/// Minimum confidence weight for contradiction alerts.
/// Basis: Even weak contradictions deserve some exploration weight.
pub const KNOWLEDGE_CONTRADICTION_FLOOR: f32 = 0.3;

// ─── FEP complexity & surprise modulation ───────────────────────────────────

/// Maximum complexity penalty range above FEP_COMPLEXITY_THRESHOLD.
/// Basis: Friston (2010) — complexity penalty saturates.
pub const FEP_COMPLEXITY_PENALTY_CAP: f64 = 0.5;

/// Learning rate scale per unit excess complexity.
/// Basis: Friston (2010) — gradual LR reduction under high complexity.
pub const FEP_COMPLEXITY_LR_SCALE: f64 = 0.1;

/// Maximum surprise range for exploration boost.
/// Basis: Schmidhuber (2010) — exploration signal proportional to surprise.
pub const FEP_SURPRISE_EXPLORE_CAP: f64 = 0.5;

/// Exploration scale per unit surprise above threshold.
pub const FEP_SURPRISE_EXPLORE_SCALE: f64 = 0.2;

/// Surprise threshold for exploration boost (relative to reflection).
pub const FEP_SURPRISE_EXPLORE_SECONDARY_CAP: f64 = 0.05;

/// Surprise exploration secondary scale.
pub const FEP_SURPRISE_EXPLORE_SECONDARY_SCALE: f64 = 0.1;

// ─── Epistemic attention budget scaling (Gottlieb 2013) ─────────────────────

/// Epistemic uncertainty threshold for attention budget expansion.
pub const EPISTEMIC_BUDGET_EXPAND_THRESHOLD: f32 = 0.4;

/// Maximum attention budget expansion from epistemic uncertainty.
pub const EPISTEMIC_BUDGET_EXPAND_CAP: f32 = 0.3;

/// Epistemic uncertainty threshold for attention budget contraction.
pub const EPISTEMIC_BUDGET_CONTRACT_THRESHOLD: f32 = 0.2;

/// Base scale for contracted attention budget.
pub const EPISTEMIC_BUDGET_CONTRACT_BASE: f32 = 0.9;

/// Contraction ramp rate per unit epistemic certainty.
pub const EPISTEMIC_BUDGET_CONTRACT_RAMP: f32 = 0.5;

// ─── Sacred Stillness attention budget ──────────────────────────────────────

/// Sacred Stillness coordinate threshold for attention budget contraction.
pub const STILLNESS_BUDGET_THRESHOLD: f64 = 0.5;

/// Maximum attention budget contraction from Sacred Stillness.
pub const STILLNESS_BUDGET_CONTRACT_CAP: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// MOTOR OUTPUT — JOINT SCALE FACTORS
// ═══════════════════════════════════════════════════════════════════════════════
// Biomechanical heuristics for humanoid joint mapping from HDC motor commands.
// Approximate anatomical ROM ratios relative to hip reference.

/// Knee joint scale — reduced ROM relative to hip (anatomical: ~70%).
pub const JOINT_KNEE_SCALE: f64 = 0.7;

/// Shoulder joint scale — arm swing amplitude relative to hip stride.
pub const JOINT_SHOULDER_SCALE: f64 = 0.5;

/// Elbow joint scale — distal arm swing, less than shoulder.
pub const JOINT_ELBOW_SCALE: f64 = 0.3;
