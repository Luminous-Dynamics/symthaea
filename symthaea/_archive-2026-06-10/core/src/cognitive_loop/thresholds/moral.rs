// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral, ethical evaluation, soul coherence, empathic, social, and harmony constants.

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL / ETHICAL EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Moral score below this triggers concern and biases toward Supportive strategy.
/// Basis: Haidt (2001) — moral intuition triggers fast conservative override.
pub const MORAL_CONCERN_THRESHOLD: f32 = -0.3;

/// Moral score above this boosts prediction confidence.
/// Basis: Damasio (1994) — positive somatic markers reinforce decision confidence.
pub const MORAL_BENEFIT_THRESHOLD: f32 = 0.5;

/// Exploration dampening when moral concern detected (multiplicative).
/// Basis: De Martino (2006) — amygdala activation reduces exploratory behavior.
pub const MORAL_CONCERN_EXPLORATION_DAMPEN: f32 = 0.5;

/// Speech rate boost on moral concern (multiplicative, >1 = slower/more cautious).
/// Basis: Forgas (2011) — negative affect promotes systematic processing.
pub const MORAL_CONCERN_PAUSE_BOOST: f32 = 1.5;

/// Confidence nudge for positive moral alignment (multiplicative).
/// Basis: Schwartz (2012) — value-aligned actions strengthen self-efficacy.
pub const MORAL_BENEFIT_CONFIDENCE_BOOST: f32 = 1.05;

/// Moral evaluation amortization interval (cycles). Co-prime.
/// Basis: Kahneman (2011) — System 2 moral evaluation is expensive.
pub const MORAL_EVAL_INTERVAL: usize = 7;

/// Negation polarity threshold for moral input preprocessing.
/// Basis: Horn (1989) — pragmatic negation inverts semantic content.
pub const NEGATION_POLARITY_THRESHOLD: f32 = 0.5;

/// Negation dampening factor applied to moral evaluation.
pub const NEGATION_DAMPENING: f32 = 0.3;

/// Maximum Love Coherence — no finite system claims perfect moral alignment.
/// Basis: Residual free energy (Friston 2010); tawāḍuʿ (Al-Ghazali 1095),
/// anavah (Maimonides 1170), kenosis (Philippians 2:7).
pub const MORAL_HUMILITY_CEILING: f64 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL SCORE → MEMORY PRIORITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Moral salience threshold for boosted episodic consolidation.
/// Science: Zak (2012) — moral narratives enhance oxytocin → memory consolidation.
pub const MORAL_CONSOLIDATION_THRESHOLD: f32 = 0.3;

/// Consolidation threshold reduction per unit of moral salience.
/// Lowers the consciousness-EMA threshold for triggering consolidation.
pub const MORAL_CONSOLIDATION_EASE: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC INTERFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Max harmonic interference count before LR dampening.
pub const HARMONIC_INTERFERENCE_MAX_COUNT: usize = 3;

/// Per-interference LR dampening factor.
pub const HARMONIC_INTERFERENCE_DAMPEN: f32 = 0.02;

/// Max cumulative LR dampening from interference.
pub const HARMONIC_INTERFERENCE_MAX_DAMPEN: f32 = 0.1;

/// Harmony boost when all clear (0 interferences).
pub const HARMONIC_ALL_CLEAR_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC FIELD
// ═══════════════════════════════════════════════════════════════════════════════

/// Harmonic field coherence threshold for LR stability boost.
/// Basis: Schwartz (2012) — value coherence enables stable learning.
pub const HARMONIC_FIELD_BOOST_THRESHOLD: f32 = 0.6;

/// Harmonic field coherence → LR boost factor.
pub const HARMONIC_FIELD_BOOST_FACTOR: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// KOSMIC SONG (Unified Identity Synthesis)
// ═══════════════════════════════════════════════════════════════════════════════

/// KosmicSong synthesis interval (cycles). Co-prime with ethics (19) and consciousness (25).
/// Science: Identity coherence is slower than perceptual update but faster than value drift.
pub const KOSMIC_SONG_INTERVAL: usize = 23;

/// Low coherence threshold: below this, dampen exploration (fragmented identity → cautious).
/// Science: Gallagher (2000) — fragmented narrative self reduces decision confidence.
pub const KOSMIC_LOW_COHERENCE_THRESHOLD: f32 = 0.3;

/// Low coherence exploration dampening (multiplicative).
pub const KOSMIC_LOW_COHERENCE_EXPLORATION_DAMPEN: f32 = 0.95;

/// High coherence confidence boost (additive).
/// Science: Conway & Pleydell-Pearce (2000) — coherent identity → reliable decisions.
pub const KOSMIC_HIGH_COHERENCE_CONFIDENCE_BOOST: f32 = 0.02;

/// High coherence threshold: above this, boost confidence.
pub const KOSMIC_HIGH_COHERENCE_THRESHOLD: f32 = 0.7;

// ═══════════════════════════════════════════════════════════════════════════════
// EMPATHIC NEUROMODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Empathic compassion threshold for oxytocin production.
/// Science: Feldman (2012) — empathic resonance drives oxytocin release.
pub const EMPATHIC_COMPASSION_OXY_THRESHOLD: f64 = 0.7;

/// Oxytocin production scale from empathic compassion.
pub const EMPATHIC_COMPASSION_OXY_SCALE: f32 = 0.15;

/// Empathic compassion threshold for dopamine boost (reward from connection).
/// Science: Rilling et al. (2002) — mutual cooperation activates DA reward circuits.
pub const EMPATHIC_COMPASSION_DA_THRESHOLD: f64 = 0.8;

/// Dopamine production scale from empathic compassion.
pub const EMPATHIC_COMPASSION_DA_SCALE: f32 = 0.08;

// ═══════════════════════════════════════════════════════════════════════════════
// SOUL COHERENCE FEEDBACK
// Science: Damasio (1994) — identity coherence (somatic markers) underpins
// stable decision-making and confidence in self-model.
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul coherence threshold below which confidence is dampened.
/// Low coherence indicates fragmented identity → uncertain predictions.
pub const SOUL_COHERENCE_LOW_THRESHOLD: f32 = 0.5;

/// Scale factor for soul-coherence confidence dampening.
pub const SOUL_COHERENCE_CONFIDENCE_SCALE: f32 = 0.01;

/// Growth potential threshold above which exploration is boosted.
/// High growth potential = learning opportunity detected.
pub const SOUL_GROWTH_EXPLORATION_THRESHOLD: f32 = 0.7;

/// Scale factor for growth-driven exploration boost.
pub const SOUL_GROWTH_EXPLORATION_SCALE: f32 = 0.01;

/// Value alignment threshold below which exploration is boosted.
/// Low alignment = misaligned values, need recalibration.
pub const SOUL_ALIGNMENT_LOW_THRESHOLD: f32 = 0.3;

/// Exploration multiplier when value alignment is low.
pub const SOUL_ALIGNMENT_EXPLORATION_MULT: f32 = 1.02;

// ═══════════════════════════════════════════════════════════════════════════════
// LOVE RESONANCE COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum harmonic love resonance to trigger confidence/soul amplification.
/// Science: Fredrickson (2013) — positivity resonance requires threshold mutual engagement.
pub const LOVE_RESONANCE_THRESHOLD: f32 = 0.6;

/// Scale factor for love resonance → confidence boost (per unit above threshold).
/// Boost = (resonance - threshold) * scale.
pub const LOVE_RESONANCE_CONFIDENCE_SCALE: f32 = 0.04;

/// Fraction of love resonance boost applied to learning rate.
/// LR *= 1.0 + boost * fraction.
pub const LOVE_RESONANCE_LR_FRACTION: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL LEARNING MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base learning rate factor for social trust modulation.
/// Science: Decety & Chaminade (2003) — social trust modulates learning engagement.
pub const SOCIAL_LR_BASE: f32 = 0.8;

/// Range of social trust modulation (LR factor = BASE + RANGE * trust).
/// Full range: [0.8, 1.2] maps distrust → full trust.
pub const SOCIAL_LR_RANGE: f32 = 0.4;

/// Theory-of-Mind accuracy above which prediction confidence is boosted.
/// Science: Frith & Frith (2006) — accurate mentalizing reinforces social predictions.
pub const TOM_ACCURACY_HIGH: f32 = 0.7;

/// Theory-of-Mind accuracy below which prediction confidence is dampened.
pub const TOM_ACCURACY_LOW: f32 = 0.3;

/// Scale factor for ToM accuracy → confidence adjustment (per unit from threshold).
pub const TOM_ACCURACY_SCALE: f32 = 0.05;

/// Minimum causal average confidence to trigger urgency gating.
/// Science: Pearl (2009) — causal reasoning requires sufficient confidence in causal links.
pub const CAUSAL_URGENCY_CONFIDENCE: f32 = 0.6;

// ═══════════════════════════════════════════════════════════════════════════════
// ENTROPY LR MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum learning rate modifier from harmony entropy (at zero entropy).
/// Science: broader value-space exploration → richer training signal.
pub const ENTROPY_LR_MIN: f64 = 0.95;

/// Range of entropy-based LR modulation. Full range: [MIN, MIN+RANGE].
pub const ENTROPY_LR_RANGE: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL RELATIONAL DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi divergence threshold for triggering exploration boost.
/// Science: Friston (2010) — high divergence = high epistemic value.
pub const PHI_DIVERGENCE_THRESHOLD: f64 = 0.1;

/// Maximum phi divergence considered for exploration scaling.
pub const PHI_DIVERGENCE_MAX: f64 = 0.2;

/// Exploration scale factor for phi divergence boost.
pub const PHI_DIVERGENCE_SCALE: f64 = 0.15;

/// Phi relational threshold for oxytocin production.
/// Science: Feldman (2012) — relational coherence drives oxytocin release.
pub const PHI_RELATIONAL_OXY_THRESHOLD: f64 = 0.3;

/// Oxytocin production scale per unit of relational phi above threshold.
pub const PHI_RELATIONAL_OXY_SCALE: f64 = 0.05;

/// Midpoint for trust evolution from cycle coherence.
/// Science: Bowlby (1969) — secure attachment requires consistent responsiveness.
pub const TRUST_SIGNAL_MIDPOINT: f64 = 0.5;

/// Rate of trust change per coherence unit above/below midpoint.
pub const TRUST_SIGNAL_RATE: f64 = 0.01;

/// Decay factor preventing runaway trust accumulation.
pub const TRUST_DECAY_FACTOR: f64 = 0.999;

// ═══════════════════════════════════════════════════════════════════════════════
// EMPATHIC SPEECH MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum empathic tone adjustment magnitude to trigger speech rate modulation.
/// Science: Porges (2011) — polyvagal theory links prosody to social engagement.
pub const EMPATHIC_TONE_THRESHOLD: f32 = 0.1;

/// Rate at which empathic tone modulates speech rate (multiplicative).
pub const EMPATHIC_TONE_RATE_SCALE: f32 = 0.1;

/// Minimum speech rate multiplier after empathic modulation.
pub const SPEECH_RATE_CLAMP_MIN: f32 = 0.6;

/// Maximum speech rate multiplier after empathic modulation.
pub const SPEECH_RATE_CLAMP_MAX: f32 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL TRUST STRATEGY BIAS
// Science: Decety & Chaminade (2003) — social trust modulates cooperative strategy.
// ═══════════════════════════════════════════════════════════════════════════════

/// Trust midpoint — deviation from this triggers social strategy bias.
pub const SOCIAL_TRUST_MIDPOINT: f32 = 0.5;
/// Trust deviation deadzone — below this, no strategy bias applied.
pub const SOCIAL_TRUST_DEADZONE: f32 = 0.1;
/// Minimum cooperation rate for trust-based strategy upgrade.
pub const SOCIAL_COOPERATION_THRESHOLD: f32 = 0.3;
/// Scaling factor for trust deviation → bias strength mapping.
/// Maps deviation [0.1, 0.5] → strength [0, 1].
pub const SOCIAL_TRUST_STRENGTH_SCALE: f32 = 2.5;
/// Minimum strength for strategy override (Concise→Supportive or Exploratory→Detailed).
pub const SOCIAL_TRUST_OVERRIDE_THRESHOLD: f32 = 0.5;
/// Minimum strength for exploration adjustment (no strategy override).
pub const SOCIAL_TRUST_EXPLORE_THRESHOLD: f32 = 0.3;
/// Exploration adjustment scale from social trust.
pub const SOCIAL_TRUST_EXPLORE_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// TOM PREDICTION MISMATCH
// Science: Frith & Frith (2006) — ToM mismatch drives epistemic exploration.
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay for ToM prediction mismatch tracking.
pub const TOM_MISMATCH_EMA_DECAY: f32 = 0.9;
/// ToM mismatch threshold for exploration trigger.
pub const TOM_MISMATCH_THRESHOLD: f32 = 0.4;
/// Exploration boost scale from ToM mismatch.
pub const TOM_MISMATCH_EXPLORE_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// SOUL ALIGNMENT LR MODULATION
// Science: Haidt (2001) — moral alignment reinforces learning.
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul alignment threshold for LR boost (positive alignment).
pub const SOUL_ALIGNMENT_BOOST_THRESHOLD: f32 = 0.3;
/// LR boost scale from positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_SCALE: f32 = 0.1;
/// LR clamp minimum for positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_LR_MIN: f32 = 0.8;
/// LR clamp maximum for positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_LR_MAX: f32 = 1.3;
/// Soul alignment threshold for LR dampening (negative alignment).
pub const SOUL_ALIGNMENT_DAMPEN_THRESHOLD: f32 = -0.3;
/// LR dampening scale from negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_SCALE: f32 = 0.15;
/// LR clamp minimum for negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_LR_MIN: f32 = 0.7;
/// LR clamp maximum for negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_LR_MAX: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// NARRATIVE SELF-PHI
// Science: Gallagher (2000) — narrative self underpins identity coherence.
// Conway & Pleydell-Pearce (2000) — self-coherence → decision confidence.
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-Phi threshold above which confidence is boosted (coherent identity).
pub const NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD: f64 = 0.4;

/// Confidence boost scale for high narrative self-phi.
pub const NARRATIVE_SELF_PHI_CONFIDENCE_SCALE: f32 = 0.008;

/// Self-Phi threshold below which exploration is boosted (fragmented identity → seek coherence).
pub const NARRATIVE_SELF_PHI_LOW_THRESHOLD: f64 = 0.15;

/// Exploration boost for low narrative self-phi.
pub const NARRATIVE_SELF_PHI_LOW_EXPLORE_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIES ALIGNMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Low harmonies alignment → explore to find value-congruent actions.
/// Basis: Schwartz (2012) — value incongruence signals need for behavioral adjustment.
pub const HARMONIES_MISALIGNMENT_THRESHOLD: f32 = 0.3;

/// Exploration boost when harmonies are misaligned.
pub const HARMONIES_MISALIGNMENT_EXPLORE_BOOST: f32 = 0.02;

/// High harmonies alignment → boost confidence (value-congruent action).
/// Basis: Schwartz (2012) — value congruence strengthens self-efficacy.
pub const HARMONIES_ALIGNED_THRESHOLD: f32 = 0.7;

/// Confidence boost when harmonies are well-aligned.
pub const HARMONIES_ALIGNED_CONFIDENCE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// NARRATIVE SELF
// Science: Gallagher (2000) — narrative self as identity coherence substrate.
// Gallagher & Hutto (2007) — narrative practice hypothesis for moral reasoning.
// ═══════════════════════════════════════════════════════════════════════════════

/// Strong narrative identity threshold (Gallagher 2000).
pub const NARRATIVE_SELF_STRONG_THRESHOLD: f64 = 0.5;

/// High narrative identity threshold for moral stabilization (Gallagher & Hutto 2007).
pub const NARRATIVE_SELF_HIGH_THRESHOLD: f64 = 0.7;

/// Weak narrative identity threshold (below which recalibration needed).
pub const NARRATIVE_SELF_WEAK_THRESHOLD: f64 = 0.2;

/// Confidence boost for strong narrative identity (Gallagher 2000).
pub const NARRATIVE_SELF_CONFIDENCE_BOOST: f32 = 1.02;

/// Confidence dampen for weak narrative identity.
pub const NARRATIVE_SELF_CONFIDENCE_DAMPEN: f32 = 0.95;

/// Moral learning signal stabilization scale for high self-coherence (Gallagher & Hutto 2007).
pub const NARRATIVE_SELF_MORAL_STABILIZE_SCALE: f32 = 0.1;

/// Moral sensitivity amplification scale for low self-coherence.
pub const NARRATIVE_SELF_MORAL_SENSITIVITY_SCALE: f32 = 0.15;

/// Social trust boost from strong narrative identity (Baumeister & Leary 1995).
pub const NARRATIVE_SELF_SOCIAL_TRUST_BOOST: f32 = 1.02;

// ═══════════════════════════════════════════════════════════════════════════════
// HODGE ADAPTIVE RIPS THRESHOLD
// Science: Beggs & Plenz (2003) — neural systems self-organize near criticality;
// Shew & Plenz (2013) — critical dynamics maximize information capacity.
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA alpha for critical_scale tracking. α=0.05 → ~20 analysis half-life.
/// Science: Haykin (2002) — adaptive filtering; low α = stable tracking.
pub const HODGE_CRITICAL_SCALE_EMA_ALPHA: f64 = 0.05;

/// Half-width of the focused Rips sweep around the EMA critical scale.
/// Sweep covers [center − 0.15, center + 0.15] with num_scales steps.
pub const HODGE_ADAPTIVE_RIPS_RADIUS: f64 = 0.15;

/// Default center when no critical_scale history exists.
pub const HODGE_ADAPTIVE_RIPS_DEFAULT_CENTER: f64 = 0.5;

/// Critical scale below this = fragile (over-connected, trivial topology).
/// Science: Beggs & Plenz (2003) — subcritical regime has low dynamic range.
pub const HODGE_CRITICAL_SCALE_FRAGILE_THRESHOLD: f64 = 0.1;

/// Critical scale above this = robust (well-separated, stable clusters).
/// Science: Shew & Plenz (2013) — supercritical regime is stable but rigid.
pub const HODGE_CRITICAL_SCALE_ROBUST_THRESHOLD: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// HODGE → FEP EXPLORATION COUPLING
// Science: Beggs & Plenz (2003) — criticality maximizes dynamic range;
// Friston (2010) — precision (inverse temperature) controls explore/exploit.
// ═══════════════════════════════════════════════════════════════════════════════

/// Harmonic fraction below which the system is over-connected (echo chamber).
pub const HODGE_HARMONIC_LOW_THRESHOLD: f64 = 0.2;

/// Harmonic fraction above which the system is fragmented.
pub const HODGE_HARMONIC_HIGH_THRESHOLD: f64 = 0.8;

/// Temperature boost when over-connected (harmonic < 0.2): explore to break symmetry.
/// Science: De Martino (2006) — echo chambers require perturbation to escape.
pub const HODGE_TEMP_EXPLORATION_BOOST: f64 = 0.3;

/// Temperature dampening when fragmented (harmonic > 0.8): focus on integration.
/// Science: Tononi (2004) — fragmentation requires integration, not divergence.
pub const HODGE_TEMP_INTEGRATION_DAMPEN: f64 = 0.7;

/// Temperature boost when critical_scale EMA < fragile threshold.
pub const HODGE_FRAGILE_TEMP_BOOST: f64 = 0.2;

/// Temperature dampening when critical_scale EMA > robust threshold.
pub const HODGE_ROBUST_TEMP_DAMPEN: f64 = 0.9;
