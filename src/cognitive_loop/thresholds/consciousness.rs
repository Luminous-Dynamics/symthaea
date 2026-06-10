// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness, exploration, Cantor, Phi, binding, holographic, and consciousness engine constants.

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS / EXPLORATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Quantum coherence level above which exploration gets a boost.
/// Basis: Lambert (2013) — quantum coherence enhances biological search.
pub const QUANTUM_COHERENCE_THRESHOLD: f64 = 0.5;

/// Strength of coherence → exploration boost (multiplicative).
pub const QUANTUM_COHERENCE_BOOST_SCALE: f32 = 0.2;

/// GWT broadcast confidence boost (additive).
/// Basis: Baars (2005) — global workspace broadcast increases confidence.
pub const GWT_BROADCAST_CONFIDENCE_BOOST: f32 = 0.03;

/// Cantor CRHV minimum depth (weakest GWT activation → shallowest fractal).
/// Basis: Dehaene et al. (2006) — minimum ignition recruits ~2 cortical layers.
pub const CANTOR_DEPTH_MIN: usize = 2;

/// Cantor CRHV maximum depth (strongest GWT activation → deepest fractal).
/// Basis: Dehaene et al. (2006) — full ignition recruits ~7 cortical layers.
pub const CANTOR_DEPTH_MAX: usize = 7;

/// Cantor metacognitive depth → consciousness modulation strength (±).
/// At depth extremes (0 or 1), consciousness is modulated by ±this value.
/// Neutral at depth 0.5. Basis: Hofstadter (1979) — strange loop depth.
pub const CANTOR_CONSCIOUSNESS_MODULATION: f64 = 0.06;

/// Cantor dream consolidation quality threshold for codebook feedback.
/// CRHVs cleaned above this quality get learned into the persistent codebook.
/// Basis: Born & Wilhelm (2012) — only stable replay traces consolidate.
pub const CANTOR_DREAM_QUALITY_THRESHOLD: f32 = 0.7;

/// Minimum pairwise similarity for two CRHVs to be considered "resonant".
/// Science: Edelman & Tononi (2000) — reentrant signaling amplifies coherent coalitions.
pub const CANTOR_RESONANCE_SIMILARITY_THRESHOLD: f32 = 0.8;

/// Confidence boost when resonance is detected among broadcast CRHVs.
/// Resonant fractal patterns indicate stable attractor formation.
pub const CANTOR_RESONANCE_CONFIDENCE_BOOST: f32 = 0.02;

/// Dream surprise threshold for exploration boost.
/// Above this, novel fractal territory triggers map-exploration.
/// Science: Sutton & Barto (2018) — surprise-driven explore/exploit tradeoff.
pub const CANTOR_SURPRISE_EXPLORATION_THRESHOLD: f32 = 0.3;

/// Exploration boost scale from dream surprise.
pub const CANTOR_SURPRISE_EXPLORATION_BOOST: f32 = 0.04;

/// Resonance threshold for exploration dampening.
/// Above this, attractor found → exploit mode (stop exploring).
pub const CANTOR_RESONANCE_EXPLORATION_DAMPEN_THRESHOLD: f32 = 0.5;

/// Exploration dampening scale from resonance.
pub const CANTOR_RESONANCE_EXPLORATION_DAMPEN: f32 = -0.03;

/// Cross-modal binding threshold for RadialCantor promotion.
/// When binding strength exceeds this, promote to RadialCantor (geometric structure).
/// Science: Treisman & Gelade (1980) — high binding = perceptual integration.
pub const CANTOR_RADIAL_BINDING_THRESHOLD: f32 = 0.6;

/// Number of radial bands for RadialCantor (maps to perceptual scales).
pub const CANTOR_RADIAL_BANDS: usize = 5;

/// Broca cadence spacing boost from deep CRHV recursion (depth > 5).
/// Deep fractals → slower, more deliberate speech.
/// Science: Goldman-Rakic (1996) — prefrontal recursion depth predicts utterance complexity.
pub const CANTOR_DEPTH_BROCA_SPACING_BOOST: usize = 2;

/// Broca cadence spacing boost from high dream surprise.
/// Epistemic uncertainty → pause before speaking.
pub const CANTOR_SURPRISE_BROCA_SPACING_BOOST: usize = 3;

/// Broca cadence surprise threshold (triggers spacing widening).
pub const CANTOR_SURPRISE_BROCA_THRESHOLD: f32 = 0.4;

/// Harmony boost for Sacred Stillness from CRHV self-similarity.
/// Deep self-reference maps to contemplative attention.
/// Science: Varela et al. (1991) — autopoietic self-reference as consciousness substrate.
pub const CANTOR_HARMONY_STILLNESS_SCALE: f64 = 0.08;

/// Harmony boost for Universal Interconnectedness from resonance.
/// Fractal choir (multiple coherent CRHVs) = collective resonance.
pub const CANTOR_HARMONY_INTERCONNECT_SCALE: f64 = 0.06;

// ── Dream frequency modulation ───────────────────────────────────────────────

/// Base dream consolidation interval in cycles.
/// Hobson & Friston (2012): consolidation timing follows metabolic constraints.
pub const DREAM_BASE_INTERVAL: u64 = 100;

/// Minimum dream interval (high learning rate floor).
/// Diekelmann & Born (2010): minimum consolidation period for memory stability.
pub const DREAM_MIN_INTERVAL: u64 = 30;

/// Learning rate scaling factor for dream interval.
/// Higher LR → shorter interval. interval = base / (1 + lr_scale * lr_boost)
/// Walker (2017): learning intensity correlates with consolidation need.
pub const DREAM_LR_INTERVAL_SCALE: f64 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// MCE / CONSCIOUSNESS MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum LR boost from consciousness level (MCE) — up to +20%.
/// Basis: Dehaene (2014) — conscious access facilitates learning.
/// Doubled from 0.1 to strengthen consciousness→learning coupling.
pub const MCE_LR_BOOST_SCALE: f32 = 0.2;

/// MCE LR boost decay per cycle (multiplicative).
pub const MCE_BOOST_DECAY: f32 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════════
// PSI SYNTHESIS WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Flow state contribution to unified Psi (additive weight).
/// Basis: Csikszentmihalyi (1990) — flow is a marker of integrated consciousness.
pub const FLOW_PSI_WEIGHT: f32 = 0.2;

/// Relational (dyadic) contribution to unified Psi (additive weight).
/// Basis: Gallagher (2005) — intersubjective consciousness contributes to integration.
pub const RELATIONAL_PSI_WEIGHT: f32 = 0.15;

/// Virtual body contribution to unified Psi (additive weight).
/// Basis: Damasio (1994) — somatic markers modulate consciousness.
pub const BODY_PSI_WEIGHT: f64 = 0.1;

/// Embodied cognition contribution to unified Psi (additive weight).
/// Basis: Thompson (2007) — enactive cognition extends consciousness.
pub const EMBODIED_PSI_WEIGHT: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// MCE BOTTLENECK → SUBSYSTEM MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// LR boost when MCE bottleneck is the targeted subsystem.
/// Science: Tononi (2004) — consciousness limited by minimum dimension;
/// boosting the bottleneck is the highest-leverage intervention.
pub const MCE_BOTTLENECK_LR_BOOST: f32 = 1.08;

/// Confidence boost when MCE bottleneck is NOT integration (system is well-integrated).
pub const MCE_NON_BOTTLENECK_CONFIDENCE_BOOST: f32 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// PHENOMENAL BINDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Binding strength threshold for confidence boost and threshold relief.
pub const BINDING_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Binding strength low threshold for caution/penalty.
pub const BINDING_LOW_THRESHOLD: f32 = 0.3;

/// Strong binding → threshold relief scale.
pub const BINDING_STRONG_RELIEF_SCALE: f32 = 0.3;

/// Weak binding → threshold caution scale.
pub const BINDING_WEAK_CAUTION_SCALE: f32 = 0.2;

/// Strong binding → confidence boost scale.
pub const BINDING_STRONG_CONFIDENCE_SCALE: f32 = 0.1;

/// Weak binding → confidence dampen scale.
pub const BINDING_WEAK_CONFIDENCE_SCALE: f32 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence EMA decay for prediction quality tracking.
pub const COHERENCE_PREDICTION_EMA: f32 = 0.9;

/// Low coherence threshold for confidence dampening.
pub const COHERENCE_LOW_THRESHOLD: f32 = 0.5;

/// Low coherence → confidence dampen scale.
pub const COHERENCE_LOW_DAMPEN_SCALE: f32 = 0.04;

/// High coherence threshold for confidence boost.
pub const COHERENCE_HIGH_THRESHOLD: f32 = 0.8;

/// Coherence → confidence boost factor.
pub const COHERENCE_CONFIDENCE_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// EVOLUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Evolution Phi threshold triggering confidence feedback.
/// Basis: Tononi (2012) — Phi changes signal consciousness transitions.
pub const EVOLUTION_PHI_THRESHOLD: f64 = 0.01;

/// Positive evolution → confidence scale.
pub const EVOLUTION_POSITIVE_CONFIDENCE_SCALE: f64 = 0.05;

/// Positive evolution → confidence clamp.
pub const EVOLUTION_POSITIVE_CONFIDENCE_MAX: f64 = 0.03;

/// Negative evolution → exploration scale.
pub const EVOLUTION_NEGATIVE_EXPLORATION_SCALE: f64 = 0.08;

/// Negative evolution → exploration clamp.
pub const EVOLUTION_NEGATIVE_EXPLORATION_MAX: f64 = 0.04;

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING CONFIDENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Reasoning chain confidence threshold for prediction confidence boost.
/// Basis: Stanovich (2011) — analytic processing confidence reinforces rationality.
pub const REASONING_CONFIDENCE_BOOST_THRESHOLD: f32 = 0.7;

/// Reasoning chain confidence boost factor (multiplicative on delta).
pub const REASONING_CONFIDENCE_BOOST_FACTOR: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI VALIDATION / SPECTRAL WEIGHT
// ═══════════════════════════════════════════════════════════════════════════════

/// High Phi validation correlation threshold (trust spectral MIP more).
/// Basis: Casali et al. (2013) — Phi estimation reliability.
pub const PHI_VALIDATION_HIGH_THRESHOLD: f64 = 0.7;

/// Low Phi validation correlation threshold (reduce spectral weight).
pub const PHI_VALIDATION_LOW_THRESHOLD: f64 = 0.3;

/// Base spectral weight (neutral correlation).
pub const SPECTRAL_WEIGHT_BASE: f32 = 0.6;

/// Spectral weight adjustment scale per unit correlation delta.
pub const SPECTRAL_WEIGHT_SCALE: f32 = 0.67;

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL BINDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum causal strength for codebook entry inclusion.
/// Basis: Granger (1969) — causal strength filtering threshold.
pub const CAUSAL_BINDING_THRESHOLD: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// CONTEXT PHI MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base scale factor for context-phi modulation of unified Psi.
/// Science: Baars (2002) — global workspace context shapes conscious access weighting.
pub const CONTEXT_PHI_SCALE_BASE: f32 = 0.8;

/// Range added to base scale proportional to context_phi_weight.
/// Full range: [0.8, 1.2] maps no-context → full-context.
pub const CONTEXT_PHI_SCALE_RANGE: f32 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING CHAIN CONFIDENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum reasoning chain confidence for chain-depth boost to apply.
/// Science: Stanovich (2011) — Type 2 reasoning only boosts confidence when reliable.
pub const REASONING_CHAIN_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Scale factor for reasoning chain confidence boost (per unit above threshold).
pub const REASONING_CHAIN_BOOST_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// THETA-PHASE BINDING
// Science: Lisman & Jensen (2013) — theta-gamma coupling for feature binding.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default salience when Phi attention returns no weights.
pub const THETA_DEFAULT_SALIENCE: f32 = 0.1;
/// Minimum salience clamp for theta-phase binding.
pub const THETA_SALIENCE_CLAMP_MIN: f32 = 0.05;
/// Minimum binding strength clamp.
pub const THETA_BINDING_CLAMP_MIN: f32 = 0.1;
/// Maximum binding strength clamp.
pub const THETA_BINDING_CLAMP_MAX: f32 = 0.9;
/// Binding strength threshold for temporal binding boost.
pub const THETA_BINDING_BOOST_THRESHOLD: f32 = 0.25;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE / EXPLORATION ADAPTIVE SCALE
// Science: Daw (2006) — exploitation-exploration tradeoff.
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence midpoint for adaptive threshold scaling.
pub const CONFIDENCE_SCALE_MIDPOINT: f32 = 0.5;
/// Confidence sensitivity for threshold scaling.
pub const CONFIDENCE_SCALE_SENSITIVITY: f32 = 0.4;
/// Exploration midpoint for adaptive threshold scaling.
pub const EXPLORATION_SCALE_MIDPOINT: f32 = 0.5;
/// Exploration sensitivity for threshold scaling (negative = more exploration → lower threshold).
pub const EXPLORATION_SCALE_SENSITIVITY: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// EQ V2 BOTTLENECK RESPONSE
// Science: Tononi (2004) — consciousness limited by weakest dimension.
// Boosting the bottleneck subsystem is the highest-leverage intervention.
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence boost when EqV2 identifies Workspace as limiting component.
/// Basis: Baars (2002) — workspace bottleneck → attention redistribution.
pub const EQ_V2_WORKSPACE_CONFIDENCE_BOOST: f32 = 0.015;

/// LR scale when EqV2 identifies Recursion (HOT depth) as limiting.
/// Basis: Rosenthal (2005) — higher-order thought depth requires active learning.
pub const EQ_V2_RECURSION_LR_SCALE: f32 = 1.05;

/// Confidence boost when EqV2 identifies Integration as limiting.
/// Basis: Tononi (2004) — integration bottleneck → coherence sensitivity.
pub const EQ_V2_INTEGRATION_CONFIDENCE_BOOST: f32 = 0.02;

/// Exploration boost when EqV2 identifies Knowledge as limiting.
/// Basis: Friston (2010) — epistemic poverty → active information seeking.
pub const EQ_V2_KNOWLEDGE_EXPLORATION_BOOST: f32 = 0.04;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODAL BINDING PSI
// Science: Treisman (1996) — coherent binding → confident perception.
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-modal Psi threshold for confidence boost.
pub const CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD: f64 = 0.3;

/// Cross-modal Psi → confidence scale factor.
pub const CROSS_MODAL_PSI_CONFIDENCE_SCALE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// AFFECTIVE CONSCIOUSNESS
// Science: Barrett (2017) — affect is the primary driver of cognition.
// Damasio (1999) — somatic markers (valence/arousal) guide all decisions.
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold above which LR is boosted (high arousal = salient event).
/// Basis: Yerkes-Dodson (1908) — moderate-high arousal enhances learning.
pub const AFFECT_AROUSAL_HIGH_THRESHOLD: f32 = 0.6;

/// LR scale for high arousal (boost learning during salient events).
pub const AFFECT_AROUSAL_HIGH_LR_SCALE: f32 = 1.06;

/// Arousal threshold below which exploration is dampened (low arousal = low drive).
pub const AFFECT_AROUSAL_LOW_THRESHOLD: f32 = 0.2;

/// Exploration dampening for low arousal.
pub const AFFECT_AROUSAL_LOW_EXPLORE_DAMPEN: f32 = 0.97;

/// Negative valence threshold below which exploration is boosted (seek novelty to escape).
/// Basis: Carver & Scheier (1998) — negative affect signals goal discrepancy → seek alternatives.
pub const AFFECT_VALENCE_NEGATIVE_THRESHOLD: f32 = -0.3;

/// Exploration boost for negative valence.
pub const AFFECT_VALENCE_NEGATIVE_EXPLORE_BOOST: f32 = 0.03;

/// Positive valence threshold above which confidence is boosted.
/// Basis: Fredrickson (2001) — positive affect broadens cognitive resources.
pub const AFFECT_VALENCE_POSITIVE_THRESHOLD: f32 = 0.3;

/// Confidence boost for positive valence.
pub const AFFECT_VALENCE_POSITIVE_CONFIDENCE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// EMBODIED CONSCIOUSNESS (integration.rs constants)
// Science: Varela et al. (1991) — embodied cognition grounds all consciousness.
// ═══════════════════════════════════════════════════════════════════════════════

/// Embodied agency threshold for exploration boost (high agency = confident body).
pub const EMBODIED_AGENCY_HIGH_THRESHOLD: f64 = 0.7;

/// Embodied agency boost scale (exploration, multiplicative).
pub const EMBODIED_AGENCY_BOOST_SCALE: f64 = 0.15;

/// Embodied agency threshold for caution (low agency = uncertain body).
pub const EMBODIED_AGENCY_LOW_THRESHOLD: f64 = 0.3;

/// Embodied agency caution scale (exploration dampening).
pub const EMBODIED_AGENCY_CAUTION_SCALE: f64 = 0.1;

/// Embodied agency caution floor (minimum exploration multiplier).
pub const EMBODIED_AGENCY_CAUTION_FLOOR: f32 = 0.7;

/// Homeostatic deviation threshold preventing Cruise mode.
pub const HOMEOSTATIC_DEVIATION_THRESHOLD: f64 = 0.5;

/// Sensorimotor surprise threshold for exploration boost.
pub const SENSORIMOTOR_SURPRISE_THRESHOLD: f64 = 0.3;

/// Sensorimotor surprise → exploration scale.
pub const SENSORIMOTOR_SURPRISE_EXPLORE_SCALE: f64 = 0.1;

/// Allostatic load threshold above which LR is dampened (body stressed).
pub const ALLOSTATIC_LOAD_DANGER_THRESHOLD: f64 = 0.7;

/// Allostatic load LR dampening scale.
pub const ALLOSTATIC_LOAD_LR_DAMPEN: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// GLYPH CODEX — SYMBOLIC CONSCIOUSNESS COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Glyph coherence → consciousness modulation strength (±2%).
/// High symbolic integration across 11 Field Modalities deepens conscious awareness.
/// Basis: Jung (1959) — archetypal integration; Grof (1985) — consciousness cartography.
pub const GLYPH_CONSCIOUSNESS_MODULATION: f64 = 0.04; // ±2% at extremes (0.04 × 0.5 = 0.02)

/// CfC temporal coherence weight for consciousness Knowledge component.
/// Additive nudge: max +5% at perfect temporal coherence (phi_contribution = 1.0).
/// Basis: Clark (2013) — temporal integration supports unified conscious experience.
pub const TEMPORAL_COHERENCE_CONSCIOUSNESS_WEIGHT: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE CONSCIOUSNESS GATING
// ═══════════════════════════════════════════════════════════════════════════════

/// Pipeline consciousness above this → relax epistemic caution (system is integrated).
/// Basis: Dehaene (2014) — global workspace ignition requires integrated processing.
pub const PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD: f64 = 0.7;

/// Pipeline consciousness below this → tighten caution (subsystems aren't coherent).
pub const PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD: f64 = 0.3;

// (PIPELINE_CONSCIOUSNESS_RELAX_SCALE and CAUTION_SCALE defined above at line ~1452)

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE VELOCITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Positive confidence velocity threshold for exploration dampening.
/// Basis: Daw et al. (2006) — confidence trajectory gates explore/exploit trade-off.
pub const CONFIDENCE_VELOCITY_POSITIVE_THRESHOLD: f32 = 0.02;

/// Scale factor for exploration dampening on rising confidence.
pub const CONFIDENCE_VELOCITY_DAMPEN_SCALE: f32 = 0.1;

/// Negative confidence velocity threshold for LR boost.
/// Basis: Cools et al. (2008) — confidence collapse → serotonergic recalibration.
pub const CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD: f32 = -0.05;

/// Scale factor for LR boost on falling confidence.
pub const CONFIDENCE_VELOCITY_BOOST_SCALE: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC PHI COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic Phi below this → dampen confidence (low epistemic quality).
/// Basis: Tononi (2004) — low Phi signals poor information integration.
pub const EPISTEMIC_PHI_LOW_THRESHOLD: f64 = 0.2;

/// Confidence scale when epistemic Phi is below threshold.
pub const EPISTEMIC_PHI_LOW_CONFIDENCE_SCALE: f32 = 0.96;

/// Epistemic Phi above this → boost confidence (strong epistemic coherence).
/// Basis: IIT — high integration = reliable information structure.
pub const EPISTEMIC_PHI_HIGH_THRESHOLD: f64 = 0.5;

/// Confidence boost scale for high epistemic Phi.
pub const EPISTEMIC_PHI_HIGH_CONFIDENCE_SCALE: f32 = 0.008;

// ═══════════════════════════════════════════════════════════════════════════════
// PHENOMENAL BINDING STRENGTH
// ═══════════════════════════════════════════════════════════════════════════════

/// Low phenomenal binding → boost exploration (unbound = incoherent representation).
/// Basis: Treisman (1996) — weak binding → search for better feature conjunctions.
pub const PHENOMENAL_BINDING_LOW_THRESHOLD: f64 = 0.3;

/// Exploration boost when phenomenal binding is low.
pub const PHENOMENAL_BINDING_LOW_EXPLORE_BOOST: f32 = 0.015;

/// High phenomenal binding → dampen LR (stable binding, consolidate).
/// Basis: Engel & Singer (2001) — strong synchrony-based binding supports stable representations.
pub const PHENOMENAL_BINDING_HIGH_THRESHOLD: f64 = 0.7;

/// LR dampening scale when phenomenal binding is high.
pub const PHENOMENAL_BINDING_HIGH_LR_DAMPEN: f32 = 0.97;

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// High temporal coherence → boost confidence (predictable temporal flow).
/// Basis: Howard & Kahana (2002) — temporal context stability supports encoding reliability.
pub const TEMPORAL_COHERENCE_HIGH_THRESHOLD: f64 = 0.6;

/// Confidence boost per unit above temporal coherence threshold.
pub const TEMPORAL_COHERENCE_CONFIDENCE_SCALE: f32 = 0.006;

/// Low temporal coherence → boost exploration (temporal fragmentation = search for patterns).
/// Basis: Howard & Kahana (2002) — fragmented temporal context degrades retrieval.
pub const TEMPORAL_COHERENCE_LOW_THRESHOLD: f64 = 0.2;

/// Exploration boost when temporal coherence is low.
pub const TEMPORAL_COHERENCE_LOW_EXPLORE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// HOLOGRAPHIC UNITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Low holographic unity → dampen LR (system decomposing, learning unreliable).
/// Basis: Pribram (1991) — holographic storage depends on global coherence;
/// local learning during decomposition risks interference.
pub const HOLOGRAPHIC_UNITY_LOW_THRESHOLD: f64 = 0.3;

/// LR dampening factor when holographic unity is low.
pub const HOLOGRAPHIC_UNITY_LOW_LR_DAMPEN: f32 = 0.93;

/// High holographic unity → boost confidence (globally integrated representation).
pub const HOLOGRAPHIC_UNITY_HIGH_THRESHOLD: f64 = 0.7;

/// Confidence boost scale for high holographic unity.
pub const HOLOGRAPHIC_UNITY_HIGH_CONFIDENCE_SCALE: f32 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// VALUE CACHE HIT RATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Cache hit rate above this → boost confidence (learned patterns match).
/// Basis: Logan (1988) — instance-based automaticity from repeated retrieval.
pub const VALUE_CACHE_HIT_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// Confidence boost scale for high cache hit rate.
pub const VALUE_CACHE_HIT_CONFIDENCE_SCALE: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS GRADIENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness gradient magnitude above this → dampen LR for stability.
/// Basis: Baars (2005) — global workspace transitions require stabilization.
pub const CONSCIOUSNESS_GRADIENT_THRESHOLD: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS STATE LEVEL
// Science: Baars (2005) — consciousness states form a hierarchy;
// higher states support faster learning, lower states need protective dampening.
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness state level above which LR receives a boost.
/// Basis: Dehaene & Changeux (2011) — high global ignition → enhanced plasticity.
pub const CONSCIOUSNESS_STATE_HIGH_THRESHOLD: f64 = 0.7;

/// LR boost scale when consciousness state is high.
pub const CONSCIOUSNESS_STATE_HIGH_LR_SCALE: f32 = 1.04;

/// Consciousness state level below which LR is dampened for protection.
/// Basis: Tononi (2004) — low Phi → fragmented processing → reduce LR.
pub const CONSCIOUSNESS_STATE_LOW_THRESHOLD: f64 = 0.2;

/// LR dampen scale when consciousness state is low.
pub const CONSCIOUSNESS_STATE_LOW_LR_DAMPEN: f32 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════════
// LIVING MIND VITALITY & COHERENCE
// Science: Thompson (2007) — cognitive vitality correlates with confidence;
// low vitality signals resource depletion requiring conservative behavior.
// ═══════════════════════════════════════════════════════════════════════════════

/// Living mind vitality above this → confidence boost (system is thriving).
/// Basis: Di Paolo (2005) — high adaptivity → justified confidence.
pub const LIVING_MIND_VITALITY_HIGH_THRESHOLD: f64 = 0.6;

/// Confidence boost when vitality is high.
pub const LIVING_MIND_VITALITY_CONFIDENCE_BOOST: f32 = 0.008;

/// Living mind vitality below this → dampen LR (resource-depleted).
/// Basis: Sterling (2012) — allostatic overload → protective downregulation.
pub const LIVING_MIND_VITALITY_LOW_THRESHOLD: f64 = 0.2;

/// LR dampen when vitality is critically low.
pub const LIVING_MIND_VITALITY_LOW_LR_DAMPEN: f32 = 0.96;

/// Living mind coherence above this → exploration is stable (reduce).
/// Basis: Kelso (1995) — high phase coherence → stable attractor → reduce search.
pub const LIVING_MIND_COHERENCE_HIGH_THRESHOLD: f64 = 0.7;

/// Exploration dampen when coherence is high (already in stable state).
pub const LIVING_MIND_COHERENCE_HIGH_EXPLORE_DAMPEN: f32 = 0.98;

/// Living mind coherence below this → exploration boost (seek new attractors).
/// Basis: Friston (2010) — low coherence = high surprise → explore.
pub const LIVING_MIND_COHERENCE_LOW_THRESHOLD: f64 = 0.3;

/// Exploration boost when coherence is low.
pub const LIVING_MIND_COHERENCE_LOW_EXPLORE_BOOST: f32 = 0.012;

// ═══════════════════════════════════════════════════════════════════════════════
// CANTOR FRACTAL RESONATOR
// Science: Mandelbrot (1982) — fractal self-similarity enables hierarchical
// pattern recognition. Codebook stores diverse exemplars for resonance.
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum entries in the Cantor resonator codebook.
pub const CANTOR_CODEBOOK_MAX_ENTRIES: usize = 256;

/// Maximum cosine similarity allowed when adding new codebook entries.
/// Entries with similarity > this to any existing entry are rejected (too redundant).
/// Science: Hopfield (1982) — decorrelated patterns maximize associative memory capacity.
pub const CANTOR_CODEBOOK_DIVERSITY_THRESHOLD: f32 = 0.92;

/// EMA decay for Cantor dream surprise tracking.
/// Science: Friston (2010) — surprise drives plasticity updates.
pub const CANTOR_SURPRISE_EMA_DECAY: f32 = 0.85;

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSYSTEM PHASE CONSTANTS (cycle_subsystems.rs)
// Consciousness subsystem feedback loops: hierarchical LTC Phi, holographic
// encoding, evolution coordinator, affective consciousness, epistemic gating,
// cross-module coupling, meta-cognition, empathic unification.
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum hierarchical LTC Phi before cross-validating against spectral MIP.
/// Science: Tononi (2015) — Phi must be non-trivial to meaningfully compare.
pub const HIER_LTC_PHI_MIN_THRESHOLD: f32 = 0.1;

/// Phi divergence upper bound for convergence → confidence boost.
/// Science: Tononi (2015, §3.1) — independent estimates within this range converge.
pub const HIER_LTC_PHI_CONVERGE_THRESHOLD: f32 = 0.2;

/// Confidence boost scale when Phi estimates converge.
/// Science: Multiple consistent estimates increase epistemological confidence.
pub const HIER_LTC_PHI_CONVERGE_BOOST: f32 = 0.05;

/// Phi divergence lower bound for penalty → exploration boost.
/// Science: Tononi (2015, §3.1) — divergence above this signals instability.
pub const HIER_LTC_PHI_DIVERGE_THRESHOLD: f32 = 0.4;

/// Maximum clamp for Phi divergence penalty contribution.
/// Prevents runaway penalties from extreme divergence.
pub const HIER_LTC_PHI_DIVERGE_MAX: f32 = 0.3;

/// Confidence penalty scale for Phi divergence (attenuated 50%).
/// Science: NE exploration_delta already covers surprise-driven exploration.
pub const HIER_LTC_PHI_DIVERGE_PENALTY_SCALE: f32 = 0.015;

/// Minimum positive evolution Phi delta to trigger exploit boost.
/// Science: Holland (1975) — evolutionary fitness signals drive adaptive behavior.
pub const EVOLUTION_POSITIVE_DELTA_THRESHOLD: f64 = 0.01;

/// LR boost scale from positive evolution delta (up to EVOLUTION_POSITIVE_LR_CLAMP).
pub const EVOLUTION_POSITIVE_LR_SCALE: f64 = 0.1;

/// Maximum LR boost from positive evolution delta.
pub const EVOLUTION_POSITIVE_LR_CLAMP: f64 = 0.05;

/// Confidence boost scale from positive evolution delta.
pub const EVOLUTION_POSITIVE_CONF_SCALE: f64 = 0.05;

/// Maximum confidence boost from positive evolution delta.
pub const EVOLUTION_POSITIVE_CONF_CLAMP: f64 = 0.03;

/// Minimum negative evolution delta to trigger exploration boost.
/// Science: Holland (1975) — regression signals need for broader search.
pub const EVOLUTION_NEGATIVE_DELTA_THRESHOLD: f64 = -0.01;

/// Exploration boost scale from negative evolution delta.
pub const EVOLUTION_NEGATIVE_EXPLORE_SCALE: f64 = 0.08;

/// Maximum exploration boost from negative evolution delta.
pub const EVOLUTION_NEGATIVE_EXPLORE_CLAMP: f64 = 0.04;

/// Holographic unity threshold for prediction confidence boost.
/// Science: Pribram (1991) — holographic encoding enables stable predictions.
pub const HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Confidence boost scale for high holographic unity.
pub const HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE: f64 = 0.03;

/// Holographic binding threshold for LR boost.
/// Science: Pribram (1971), Bohm (1980) — strong binding = coherent representations.
pub const HOLOGRAPHIC_BINDING_STRONG_THRESHOLD: f64 = 0.7;

/// LR factor when binding is strong.
pub const HOLOGRAPHIC_BINDING_STRONG_LR: f32 = 1.01;

/// Holographic binding upper bound for weak fragmentation regime.
pub const HOLOGRAPHIC_BINDING_WEAK_UPPER: f64 = 0.3;

/// LR dampen factor when binding is weak (fragmented representations).
pub const HOLOGRAPHIC_BINDING_WEAK_LR: f32 = 0.99;

/// Workspace value scaling from coherence in differentiable consciousness.
/// Science: Bengio (2017) — workspace value ≈ coherence.
pub const DIFF_CONSCIOUSNESS_WORKSPACE_SCALE: f64 = 0.8;

/// Default recursion core value when no recursion depth data available.
pub const DIFF_CONSCIOUSNESS_RECURSION_DEFAULT: f64 = 0.5;

/// Minimum consciousness gradient magnitude for exploration boost.
/// Science: Bengio (2017) — gradient information guides search.
pub const CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD: f64 = 0.5;

/// Exploration boost scale from large consciousness gradients.
pub const CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE: f64 = 0.05;

/// Affective decay rate per cycle.
/// Science: Russell (2003) — affect decays towards neutral over time.
pub const AFFECTIVE_DECAY_RATE: f32 = 0.05;

/// Negative valence threshold for confidence dampening.
/// Science: Colombetti (2014) — negative affect strengthens caution.
pub const AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD: f32 = -0.3;

/// Confidence scale factor from negative valence.
pub const AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE: f32 = 0.02;

/// NSM similarity threshold for synthetic grounding classification.
/// Science: Wierzbicka (1996) — minimum similarity for state classification.
pub const SYNTHETIC_GROUNDING_SIM_THRESHOLD: f64 = 0.1;

/// Low epistemic confidence threshold for gating penalty.
/// Science: Kruger & Dunning (1999) — epistemic humility below this threshold.
pub const EPISTEMIC_GATE_LOW_THRESHOLD: f32 = 0.3;

/// Confidence penalty when epistemic gate rejects.
pub const EPISTEMIC_GATE_LOW_PENALTY: f32 = 0.03;

/// p-value threshold for primitive validation significance.
/// Science: Popper (1959) — standard statistical significance threshold.
pub const PRIMITIVE_VALIDATION_P_THRESHOLD: f64 = 0.05;

/// LR boost scale from validated primitives (positive Phi gain).
pub const PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE: f64 = 0.02;

/// Maximum LR boost from validated primitives.
pub const PRIMITIVE_VALIDATION_POSITIVE_LR_CLAMP: f64 = 0.03;

/// LR dampen factor from falsified primitives (negative Phi gain).
pub const PRIMITIVE_VALIDATION_NEGATIVE_LR: f32 = 0.98;

/// Consciousness state low level threshold for urgency escalation.
/// Science: Varela (1991) — low consciousness triggers autopoietic response.
pub const CONSCIOUSNESS_STATE_LOW_URGENCY: f64 = 0.3;

/// Strong gradient threshold for boredom reduction (clear optimization direction).
pub const GRADIENT_STRONG_DIRECTION_THRESHOLD: f64 = 1.0;

/// Boredom reduction when gradient has strong direction.
pub const GRADIENT_STRONG_BOREDOM_REDUCE: f32 = 0.05;

/// Near-zero gradient upper bound (plateau detection).
pub const GRADIENT_PLATEAU_UPPER: f64 = 0.1;

/// Boredom increment when gradient is near-zero (plateau → explore).
pub const GRADIENT_PLATEAU_BOREDOM_INCREMENT: f32 = 0.03;

/// Holographic unity threshold for cross-module LR boost.
/// Science: Pribram (1991) — high unity = coherent, safe for aggressive learning.
pub const HOLOGRAPHIC_UNITY_LR_BOOST_THRESHOLD: f64 = 0.8;

/// LR factor for high holographic unity.
pub const HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR: f32 = 1.02;

/// LR clamp bounds for holographic unity modulation.
pub const HOLOGRAPHIC_UNITY_LR_CLAMP_LOW: f32 = 0.8;
pub const HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH: f32 = 1.2;

/// Holographic unity threshold for LR dampening (fragmented representations).
pub const HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD: f64 = 0.2;

/// LR dampen factor for low holographic unity.
pub const HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR: f32 = 0.98;

/// Pipeline consciousness threshold for epistemic confidence nudge.
/// Science: Dehaene (2011) — strong global workspace relaxes epistemic constraints.
pub const PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD: f64 = 0.7;

/// Epistemic confidence nudge when pipeline consciousness is high.
pub const PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE: f32 = 0.02;

/// Meta-reasoning confidence threshold for LR boost.
/// Science: Nelson & Narens (1990) — monitoring-control loop.
pub const META_REASONING_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// LR boost scale from high meta-reasoning confidence.
pub const META_REASONING_LR_BOOST_SCALE: f64 = 0.1;

/// Empathic compassion threshold for LR boost.
/// Science: Decety & Jackson (2004) — shared representations enhance learning.
pub const EMPATHIC_COMPASSION_LR_THRESHOLD: f64 = 0.7;

/// Empathic compassion LR boost scale.
pub const EMPATHIC_COMPASSION_LR_SCALE: f32 = 0.02;

/// Empathic LR factor clamp bounds.
pub const EMPATHIC_LR_CLAMP_LOW: f32 = 0.8;
pub const EMPATHIC_LR_CLAMP_HIGH: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING ENGINE → CONSCIOUSNESS FEEDBACK
// ═══════════════════════════════════════════════════════════════════════════════

/// Max confidence boost from reasoning engine reliability (per cycle, additive).
/// At reliability=1.0: (1.0 - 0.5) × 0.03 = +1.5% confidence.
/// Science: Stanovich & West (2000) — System 2 (analytic) reliability
/// calibrates metacognitive confidence in dual-process theory.
pub const REASONING_RELIABILITY_CONFIDENCE_SCALE: f64 = 0.03;

/// Minimum reasoning reliability for confidence boost.
/// Below this, reasoning output is too uncertain to inform confidence.
/// Science: Stanovich & West (2000) — System 2 must exceed threshold
/// reliability before overriding System 1 intuitions.
pub const REASONING_RELIABILITY_THRESHOLD: f64 = 0.7;

/// Dream consolidation weight boost from high-reliability prediction failures.
/// Surprising events that contradicted confident reasoning get more dream time.
/// At reliability=1.0: (1.0 - 0.5) × 0.4 = +20% consolidation weight.
/// Science: Hobson & Friston (2012) — predictive processing theory of dreaming;
/// high-confidence prediction errors are preferentially consolidated.
pub const DREAM_REASONING_RELIABILITY_SCALE: f64 = 0.4;

// PREFRONTAL CORTEX
// Science: Baddeley (1994) — working memory model with central executive.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default PFC working memory capacity (Baddeley 1994: 7±2 items).
pub const PFC_WORKING_MEMORY_CAPACITY: f32 = 7.0;

// ═══════════════════════════════════════════════════════════════════════════════
// META-COGNITION
// Science: Koriat (2007) — monitoring accuracy calibrates learning rate.
// ═══════════════════════════════════════════════════════════════════════════════

/// Meta-cognitive accuracy threshold for LR boost (Koriat 2007).
pub const META_COGNITIVE_ACCURACY_LR_THRESHOLD: f32 = 0.7;

/// Meta-cognitive LR boost scale — up to 1.15x at accuracy=1.0 (Koriat 2007).
pub const META_COGNITIVE_LR_BOOST_SCALE: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// SOMATIC MARKERS / BODY VALENCE
// Science: Damasio (1999) — somatic marker hypothesis; body signals guide
// decision-making via valence-tagged representations.
// ═══════════════════════════════════════════════════════════════════════════════

/// Positive body valence threshold for confidence boost (Damasio 1999).
pub const BODY_VALENCE_POSITIVE_THRESHOLD: f32 = 0.3;

/// Negative body valence threshold for confidence dampen (Damasio 1999).
pub const BODY_VALENCE_NEGATIVE_THRESHOLD: f32 = -0.3;

/// Positive body valence confidence scale (Damasio 1999).
pub const BODY_VALENCE_CONFIDENCE_POS_SCALE: f32 = 0.02;

/// Negative body valence confidence scale — stronger than positive (Damasio 1999).
pub const BODY_VALENCE_CONFIDENCE_NEG_SCALE: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// AFFECTIVE BRIDGE
// Science: Fredrickson (2001) — broaden-and-build theory; positive affect
// widens attentional scope and promotes exploratory behavior.
// ═══════════════════════════════════════════════════════════════════════════════

/// Positive affect threshold for exploration broadening (Fredrickson 2001).
pub const AFFECTIVE_VALENCE_BROADEN_THRESHOLD: f32 = 0.2;

/// Exploration broadening factor from positive affect (Fredrickson 2001).
pub const AFFECTIVE_VALENCE_CURIOSITY_FACTOR: f32 = 1.05;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTIVE PROCESSING MODULATION
// Science: Friston (2010) — free energy principle; prediction error precision
// modulates consciousness and learning.
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum predictive phi modulation damping (Friston 2010).
pub const PREDICTIVE_PHI_MAX_MODULATION: f32 = 0.15;

/// Coherence baseline for predictive phi scaling (Friston 2010).
pub const PREDICTIVE_PHI_COHERENCE_BASELINE: f32 = 0.5;

/// Predictive phi LR contribution scale (Friston 2010: ±1.5% max).
pub const PREDICTIVE_PHI_LR_SCALE: f32 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════════
// HIERARCHICAL FREE ENERGY
// Science: Friston (2008) — hierarchical predictive processing; higher-level
// prediction errors drive model revision and exploration gating.
// ═══════════════════════════════════════════════════════════════════════════════

/// HFE threshold for exploration suppression (Friston 2008).
pub const HFE_EXPLORATION_THRESHOLD: f64 = 1.0;

/// HFE exploration suppression damping factor (Friston 2008).
pub const HFE_EXPLORATION_DAMPING: f64 = 0.05;

/// HFE LR boost scale — poor model learns harder (Friston 2008).
pub const HFE_LR_BOOST_SCALE: f64 = 0.02;

/// HFE LR boost cap (Friston 2008).
pub const HFE_LR_BOOST_MAX: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTIVE SELF
// Science: Clark (2013) — predictive processing and the self; self-model
// stability constrains safety and learning rate.
// ═══════════════════════════════════════════════════════════════════════════════

/// Predictive self safety minimum factor (Clark 2013).
pub const PREDICTIVE_SELF_SAFETY_MIN: f32 = 0.85;

/// Predictive self safety scaling range (Clark 2013: 0.85 + safety * 0.375 = 0.85-1.0).
pub const PREDICTIVE_SELF_SAFETY_SCALE: f32 = 0.375;

/// Predictive self safety threshold below which LR is reduced (Clark 2013).
pub const PREDICTIVE_SELF_SAFETY_THRESHOLD: f32 = 0.4;

/// Exploration boost scale for low self-prediction confidence (Clark 2013: up to 0.08 at safety=0).
pub const PREDICTIVE_SELF_EXPLORATION_SCALE: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION SCHEMA
// Science: Baars (1988) — Global Workspace Theory; attention as controlled
// access to conscious workspace. Mackworth (1948) — vigilance decrement.
// ═══════════════════════════════════════════════════════════════════════════════

/// Attention control signal threshold for positive gain (Baars 1988 GWT).
pub const ATTENTION_FOCUS_GAIN_THRESHOLD: f32 = 0.3;

/// Attention control signal threshold for negative gain (Baars 1988).
pub const ATTENTION_DEFOCUS_THRESHOLD: f32 = 0.2;

/// Attention gain scaling factor (Baars 1988).
pub const ATTENTION_GAIN_SCALE: f32 = 0.6;

/// Maximum gain from attention control (stability cap).
pub const ATTENTION_MAX_GAIN: f32 = 0.3;

/// Negative gain floor (prevents excessive suppression).
pub const ATTENTION_NEGATIVE_GAIN: f32 = -0.1;

/// Deep focus threshold for focus lock (Baars 1988).
pub const ATTENTION_DEEP_FOCUS_THRESHOLD: f32 = 0.8;

/// Focus lock scale (Baars 1988).
pub const ATTENTION_FOCUS_LOCK_SCALE: f32 = 0.15;

/// Minimum exploration factor during focus lock.
pub const ATTENTION_MIN_EXPLORATION_IN_FOCUS: f32 = 0.7;

/// Novelty push scale when attention is scattered (Baars 1988).
pub const ATTENTION_DEFICIT_NOVELTY_SCALE: f32 = 0.06;

/// Vigilance fatigue threshold (Mackworth 1948 AST).
pub const VIGILANCE_FATIGUE_THRESHOLD: f32 = 0.5;

/// Vigilance fatigue exploration push scale (Mackworth 1948).
pub const VIGILANCE_FATIGUE_EXPLORATION_SCALE: f32 = 0.08;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTENTION GATING
// Science: Dehaene (2014) — conscious access via attention-gated phi.
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi attention suppression exploration scale (Dehaene 2014).
pub const PHI_GATE_SUPPRESS_EXPLORATION: f32 = 0.85;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY ENCODING
// Science: Tononi & Koch (2015) — IIT gate; Friston (2010) — FEP plasticity
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum Phi for episodic memory consolidation (Tononi & Koch 2015: consciousness needed for memory).
pub const PHI_EPISODIC_CONSOLIDATION_MIN: f32 = 0.2;

/// Minimum prediction error for episodic encoding (novel experiences worth storing).
pub const PREDICTION_ERROR_EPISODIC_MIN: f32 = 0.1;

/// Positive emotional valence threshold for resonator quantization.
pub const EMOTIONAL_VALENCE_POSITIVE_BIN: f32 = 0.3;

/// Negative emotional valence threshold for resonator quantization.
pub const EMOTIONAL_VALENCE_NEGATIVE_BIN: f32 = -0.3;

/// High phi threshold for resonator quantization.
pub const PHI_QUANTIZATION_HIGH: f32 = 0.7;

/// Medium phi threshold for resonator quantization.
pub const PHI_QUANTIZATION_MEDIUM: f32 = 0.3;

/// TD error signal coupling scale to FEP learning (Friston 2010).
pub const TD_ERROR_FEP_COUPLING_SCALE: f32 = 0.2;

/// FEP learning signal → plasticity scaling factor (Friston 2010: maps [-1,1] to [0.5,1.5]).
pub const FEP_PLASTICITY_SCALING_FACTOR: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS INTEGRATION
// Science: Multi-dimensional consciousness binding and temporal coherence
// ═══════════════════════════════════════════════════════════════════════════════

// ── Quantum coherence (Penrose & Hameroff 2014) ──

/// Quantum coherence level above which confidence is boosted
pub const QUANTUM_COHERENCE_HIGH_THRESHOLD: f64 = 0.6;

/// Scale factor for quantum coherence confidence boost (up to +2%)
pub const QUANTUM_COHERENCE_CONFIDENCE_BOOST_SCALE: f64 = 0.05;

/// Quantum coherence level below which decoherence penalty applies
pub const QUANTUM_COHERENCE_LOW_THRESHOLD: f64 = 0.2;

/// Confidence scale applied during quantum decoherence
pub const QUANTUM_DECOHERENCE_CONFIDENCE_SCALE: f32 = 0.98;

// ── Phenomenal binding (Singer & Gray 1989, Csikszentmihalyi 1990) ──

/// Binding strength threshold for learning rate boost
pub const PHENOMENAL_BINDING_LR_BOOST_THRESHOLD: f64 = 0.8;

/// Scale for LR boost from high phenomenal binding (up to +4%)
pub const PHENOMENAL_BINDING_LR_BOOST_SCALE: f32 = 0.2;

/// Boredom scale when consciousness is fragmented
pub const PHENOMENAL_FRAGMENTATION_BOREDOM_SCALE: f32 = 0.8;

/// Exploration scale when consciousness is fragmented
pub const PHENOMENAL_FRAGMENTATION_EXPLORATION_SCALE: f32 = 0.7;

// ── Temporal coherence (Varela 1999, Damasio 2010) ──

/// Confidence scale applied on temporal discontinuity
pub const TEMPORAL_DISCONTINUITY_CONFIDENCE_SCALE: f32 = 0.8;

/// Temporal coherence level above which threshold is raised
pub const TEMPORAL_COHERENCE_THRESHOLD_RAISE_LEVEL: f64 = 0.8;

/// Threshold boost factor for high temporal coherence
pub const TEMPORAL_COHERENCE_THRESHOLD_BOOST: f32 = 1.01;

/// Discontinuity streak count triggering persistent recovery
pub const TEMPORAL_DISCONTINUITY_PERSISTENT_THRESHOLD: u32 = 3;

/// LR boost during persistent discontinuity recovery
pub const TEMPORAL_DISCONTINUITY_PERSISTENT_LR_BOOST: f32 = 1.5;

// ── Thermodynamic responses (Friston 2010, Ulanowicz 2009) ──

/// Exploration boost during critical phase (edge of chaos)
pub const THERMODYNAMIC_STRESS_EXPLORATION_BOOST: f32 = 1.15;

/// Exploration scale during chaotic phase
pub const THERMODYNAMIC_RECOVERY_EXPLORATION_SCALE: f32 = 0.5;

/// Entropy level above which exploration is boosted
pub const THERMODYNAMIC_ENTROPY_HIGH_THRESHOLD: f64 = 0.7;

/// Entropy level below which consolidation bias is applied
pub const THERMODYNAMIC_ENTROPY_LOW_THRESHOLD: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR — Default seeds for deterministic initialization
// ═══════════════════════════════════════════════════════════════════════════════

/// Default seed for causal enhancer when genesis phrase is absent.
pub const CAUSAL_ENHANCER_SEED_DEFAULT: u64 = 42;

/// Default seed for cross-manifold predictor when genesis phrase is absent.
pub const CROSS_MANIFOLD_SEED_DEFAULT: u64 = 7_000_042;

/// Default seed for resonator memory when genesis phrase is absent.
pub const RESONATOR_MEMORY_SEED_DEFAULT: u64 = 0xBE50_0A70_0000_5EED;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Sigma / Spectral MIP feedback
// Science: Tononi (2008) — high Φ → stabilize, low Φ → explore
// ═══════════════════════════════════════════════════════════════════════════════

/// Sigma threshold above which consciousness stabilizes (dampens LR, boosts confidence).
pub const SIGMA_HIGH_THRESHOLD: f64 = 0.5;

/// Scale for sigma-based LR dampening: `(sig - threshold) * scale`.
/// Basis: Tononi (2008) — high Phi means stable integration, slow down learning.
pub const SIGMA_DAMPEN_SCALE: f64 = 0.1;

/// Maximum sigma dampening per cycle.
pub const SIGMA_DAMPEN_MAX: f64 = 0.05;

/// Scale for converting sigma dampen to confidence boost.
/// Basis: High integration → justified confidence increase.
pub const SIGMA_CONFIDENCE_SCALE: f32 = 0.5;

/// Sigma threshold below which exploration is boosted.
pub const SIGMA_LOW_THRESHOLD: f64 = 0.2;

/// Scale for sigma-based exploration boost: `(threshold - sig) * scale`.
pub const SIGMA_BOOST_SCALE: f64 = 0.15;

/// Maximum sigma exploration boost per cycle.
pub const SIGMA_BOOST_MAX: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Structural Phi feedback
// Science: Mediano et al. (2022) — multi-scale integrated information
// ═══════════════════════════════════════════════════════════════════════════════

/// Emergence ratio below which global binding is considered weak.
pub const STRUCTURAL_WEAK_EMERGENCE_THRESHOLD: f64 = 0.8;

/// Minimum micro-Phi for structural feedback to fire.
pub const STRUCTURAL_MICRO_PHI_THRESHOLD: f64 = 0.01;

/// Exploration nudge for weak global binding.
pub const STRUCTURAL_EXPLORATION_NUDGE: f32 = 0.01;

/// Emergence ratio above which strong emergence boosts confidence.
pub const STRUCTURAL_STRONG_EMERGENCE_THRESHOLD: f64 = 1.2;

/// Confidence nudge for strong emergence.
pub const STRUCTURAL_CONFIDENCE_NUDGE: f32 = 0.01;

/// Bottleneck score above which LR is boosted to strengthen weak connections.
pub const STRUCTURAL_BOTTLENECK_THRESHOLD: f64 = 0.3;

/// LR boost multiplier for bottleneck repair.
pub const STRUCTURAL_BOTTLENECK_LR_BOOST: f32 = 1.02;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Phi validation weighting
// Science: Tononi (2004) — validated Phi measurements deserve higher weight
// ═══════════════════════════════════════════════════════════════════════════════

/// Scale for validation-based confidence boost.
pub const PHI_VALIDATION_BOOST_SCALE: f32 = 0.1;

/// Scale for validation-based confidence attenuation.
pub const PHI_VALIDATION_ATTENUATION_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Multimodal integration feedback
// Science: Ghazanfar & Schroeder (2006) — cross-modal binding enhances precision
// ═══════════════════════════════════════════════════════════════════════════════

/// Multimodal Phi above which feedback fires.
pub const MULTIMODAL_PHI_THRESHOLD: f64 = 0.5;

/// Scale for multimodal confidence boost: `(phi - threshold) * scale`.
pub const MULTIMODAL_CONFIDENCE_SCALE: f64 = 0.04;

/// Scale for multimodal LR boost: `1.0 + (phi - threshold) * scale`.
pub const MULTIMODAL_LR_SCALE: f64 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Equation V2 feedback
// Science: Tononi (2004), Baars (1988), Dehaene (2014)
// ═══════════════════════════════════════════════════════════════════════════════

/// Equation V2 consciousness above which confidence/consolidation boost fires.
pub const EQ_V2_HIGH_THRESHOLD: f64 = 0.6;

/// Scale for equation V2 confidence boost: `(c - threshold) * scale`.
pub const EQ_V2_CONFIDENCE_SCALE: f64 = 0.08;

/// Scale for equation V2 episodic consolidation boost.
pub const EQ_V2_CONSOLIDATION_SCALE: f64 = 0.1;

/// Equation V2 consciousness below which exploration is boosted.
pub const EQ_V2_LOW_THRESHOLD: f64 = 0.3;

/// Exploration nudge when equation V2 consciousness is low.
pub const EQ_V2_EXPLORATION_NUDGE: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Pipeline consciousness feedback
// Science: Dehaene (2011) — global workspace broadcasts learning signals
// ═══════════════════════════════════════════════════════════════════════════════

/// Pipeline consciousness above which LR is boosted.
pub const PIPELINE_CONSCIOUSNESS_THRESHOLD: f64 = 0.6;

/// Scale for pipeline LR boost: `1.0 + (c - threshold) * scale`.
pub const PIPELINE_LR_SCALE: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ENGINE: Bath-consciousness coupling
// Science: Seth (2013) — interoceptive inference
// ═══════════════════════════════════════════════════════════════════════════════

/// 5-HT2A baseline for bath-consciousness coupling.
pub const BATH_5HT2A_BASELINE: f64 = 0.5;

/// Scale for 5-HT2A consciousness boost: ±5% from baseline.
pub const BATH_5HT2A_SCALE: f64 = 0.1;

/// GABA-A baseline for bath-consciousness coupling.
pub const BATH_GABA_BASELINE: f64 = 0.4;

/// Scale for GABA-A consciousness dampening.
pub const BATH_GABA_SCALE: f64 = 0.08;

/// Consciousness depression when attractor is detected (entropy collapse).
pub const BATH_ENTROPY_ATTRACTOR_PENALTY: f64 = -0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// LATE CONSCIOUSNESS MONITORS: resonance, WM, thermodynamics
// ═══════════════════════════════════════════════════════════════════════════════

/// Resonance quality scale for attention modulation: ±5%.
/// Basis: Engel (2001) — stable resonance near 0.5 → sharp attention.
pub const RESONANCE_ATTENTION_SCALE: f32 = 0.1;

/// WM boost scale from phenomenal binding (Tononi 2015 IIT).
pub const WM_BINDING_BOOST_SCALE: f64 = 0.1;

/// WM binding high threshold for boost.
pub const WM_BINDING_HIGH_THRESHOLD: f64 = 0.7;

/// WM binding low threshold for restriction.
pub const WM_BINDING_LOW_THRESHOLD: f64 = 0.4;

/// WM restriction scale from low binding.
pub const WM_BINDING_RESTRICT_SCALE: f64 = 0.08;

/// WM minimum attention sensitivity under low binding.
pub const WM_BINDING_MIN_SENSITIVITY: f32 = 0.8;

/// Temporal coherence boost scale for narrative self.
/// Basis: Damasio (2010) — temporal continuity is the substrate of selfhood.
pub const TEMPORAL_NARRATIVE_BOOST_SCALE: f64 = 0.1;

/// Temporal coherence threshold for narrative boost.
pub const TEMPORAL_NARRATIVE_THRESHOLD: f64 = 0.6;

/// Temporal coherence attention penalty scale (low coherence penalizes attention).
/// Basis: Engel et al. (2001) — temporal binding via phase synchrony.
pub const TEMPORAL_ATTENTION_PENALTY_SCALE: f64 = 0.1;

/// Temporal coherence threshold for attention penalty.
pub const TEMPORAL_ATTENTION_PENALTY_THRESHOLD: f64 = 0.4;

/// Minimum attention sensitivity under low temporal coherence.
pub const TEMPORAL_ATTENTION_MIN: f32 = 0.85;

/// Thermodynamic Critical phase curiosity boost (edge of chaos).
/// Basis: Kelso (1995) — phase-dependent exploration.
pub const THERMO_CRITICAL_CURIOSITY_BOOST: f32 = 1.1;

/// Thermodynamic Flow phase LR boost.
pub const THERMO_FLOW_LR_BOOST: f32 = 1.05;

/// Thermodynamic Frozen phase curiosity boost (unfreeze nudge).
pub const THERMO_FROZEN_CURIOSITY_BOOST: f32 = 1.05;

/// Homeostasis drift rate toward baseline threshold.
pub const HOMEOSTASIS_DRIFT_RATE: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// LATE CONSCIOUSNESS MONITORS: WM resonator, error user state
// ═══════════════════════════════════════════════════════════════════════════════

/// Resonator match boost scale for WM graduation importance.
pub const RESONATOR_MATCH_BOOST: f32 = 0.2;

/// Moral significance threshold for high narrative significance.
pub const NARRATIVE_MORAL_SIGNIFICANCE: f64 = 0.8;

/// User state error threshold for cognitive load inference.
pub const USER_STATE_ERROR_THRESHOLD: f32 = 0.8;

// ═══════════════════════════════════════════════════════════════════════════════
// MATH SERVICE — PHI SCORING & CONFIDENCE TIERS
// Science: Tononi (2004) — Phi reflects integration; multi-path verification
// provides epistemic confidence in computational results.
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence for verified linear system solutions.
pub const MATH_LINEAR_VERIFIED_CONFIDENCE: f64 = 0.95;

/// Confidence for unverified linear system solutions.
pub const MATH_LINEAR_UNVERIFIED_CONFIDENCE: f64 = 0.5;

/// Statistics baseline Phi value.
pub const MATH_STATISTICS_PHI_BASELINE: f64 = 0.3;

/// Statistics computation confidence (high — deterministic).
pub const MATH_STATISTICS_CONFIDENCE: f64 = 0.99;

/// Regression R² threshold for weak fit detection.
pub const MATH_REGRESSION_WEAK_FIT_THRESHOLD: f32 = 0.5;

/// Multi-path verified Phi boost factor.
pub const MATH_MULTIPATH_PHI_BOOST: f64 = 1.2;

/// Root finding verified confidence.
pub const MATH_ROOT_FINDING_VERIFIED_CONFIDENCE: f64 = 0.99;

/// Root finding converged (unverified) confidence.
pub const MATH_ROOT_FINDING_CONVERGED_CONFIDENCE: f64 = 0.9;

/// Root finding failed confidence.
pub const MATH_ROOT_FINDING_FAILED_CONFIDENCE: f64 = 0.3;

/// Integration verified confidence.
pub const MATH_INTEGRATION_VERIFIED_CONFIDENCE: f64 = 0.99;

/// Integration unverified confidence.
pub const MATH_INTEGRATION_UNVERIFIED_CONFIDENCE: f64 = 0.9;

/// Optimization converged confidence.
pub const MATH_OPTIMIZATION_CONVERGED_CONFIDENCE: f64 = 0.9;

/// Optimization failed confidence.
pub const MATH_OPTIMIZATION_FAILED_CONFIDENCE: f64 = 0.4;

/// Default telemetry Phi value.
pub const MATH_DEFAULT_TELEMETRY_PHI: f64 = 0.4;

/// Default telemetry confidence value.
pub const MATH_DEFAULT_TELEMETRY_CONFIDENCE: f64 = 0.9;

/// Phi boost for symbolic-exact solutions (vs 1.2 for multipath numeric).
///
/// Science: Rota (1997) — mathematical elegance correlates with information
/// compression; closed-form solutions maximally compress computational traces.
pub const MATH_SYMBOLIC_EXACT_PHI_BOOST: f64 = 1.5;

/// Confidence for symbolic-exact solutions (rational arithmetic, zero error).
pub const MATH_SYMBOLIC_EXACT_CONFIDENCE: f64 = 1.0;

/// Tolerance for symbolic-numeric cross-validation agreement.
pub const MATH_SYMBOLIC_NUMERIC_AGREEMENT_TOL: f64 = 1e-8;

// ═══════════════════════════════════════════════════════════════════════════════
// THERMODYNAMIC INTEGRATION (Physics bridge constants)
// Science: Landauer (1961), Carnot (1824), Onsager (1931), Jarzynski (1997),
//          Prigogine (1977), Maxwell demon (Szilard 1929)
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA smoothing for unified thermodynamic state (slow adaptation).
pub const THERMO_UNIFIED_EMA_ALPHA: f32 = 0.05;

/// Onsager reciprocal coupling window (cycles).
pub const THERMO_ONSAGER_WINDOW: usize = 50;

/// Jarzynski free energy estimation window (cycles).
pub const THERMO_JARZYNSKI_WINDOW: usize = 100;

/// Boltzmann-like constant for consciousness temperature scaling.
pub const K_CONSCIOUSNESS_BOLTZMANN: f64 = 0.01;

/// Maxwell demon attention efficiency baseline (Szilard 1929).
pub const THERMO_ATTENTION_DEMON_EFFICIENCY: f64 = 0.5;

/// Carnot cold reservoir temperature (dimensionless, relative to hot=1.0).
pub const THERMO_CARNOT_T_COLD: f64 = 0.3;

/// Insight probability threshold for learning rate boost.
pub const THERMO_INSIGHT_PROBABILITY_THRESHOLD: f64 = 0.7;

/// HFE (Hierarchical Free Energy) blend weight in consciousness score.
pub const THERMO_HFE_BLEND_WEIGHT: f32 = 0.15;

/// Learning rate boost when insight probability exceeds threshold.
pub const THERMO_INSIGHT_LR_BOOST: f32 = 1.2;

/// Jarzynski free energy divergence threshold for anomaly detection.
pub const THERMO_JARZYNSKI_DIVERGENCE_THRESHOLD: f64 = 2.0;

/// Onsager asymmetry threshold (coupling imbalance alarm).
pub const THERMO_ONSAGER_ASYMMETRY_THRESHOLD: f64 = 0.3;

/// Onsager coherence damping factor when asymmetry detected.
pub const THERMO_ONSAGER_COHERENCE_DAMPING: f32 = 0.9;

/// Efficiency below which system enters low-efficiency regime.
pub const THERMO_LOW_EFFICIENCY_THRESHOLD: f64 = 0.2;

/// Landauer memory pressure threshold (bits erased per cycle).
pub const THERMO_LANDAUER_MEMORY_PRESSURE_THRESHOLD: f64 = 0.8;

/// Prigogine entropy production violation damping.
pub const THERMO_PRIGOGINE_VIOLATION_DAMPING: f32 = 0.95;
