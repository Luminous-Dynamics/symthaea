// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca language generation and voice heartbeat constants.

/// Broca coherence threshold for language generation quality gating.
pub const BROCA_COHERENT_THRESHOLD: f32 = 0.7;

/// Confidence boost scale when Broca coherence exceeds threshold.
/// Applied as: (coherence - BROCA_COHERENT_THRESHOLD) * SCALE → adjust_confidence.
/// Science: Pickering & Garrod (2013) — coherent language production reinforces cognitive confidence.
pub const BROCA_COHERENT_CONFIDENCE_SCALE: f32 = 0.1;

/// Exploration damping when Broca semantic veto fires.
/// A veto means the model rejected incoherent output → exploitation is safer than exploration.
/// Science: Pickering & Garrod (2013) — veto = self-correction → reduce drift.
pub const BROCA_VETO_EXPLORATION_SCALE: f32 = -0.02;
/// Voice heartbeat base speech rate multiplier.
/// Science: Liberman & Mattingly (1985) — synthetic proxy for vocal tract feedback.
pub const VOICE_HEARTBEAT_BASE_RATE: f32 = 4.0;

/// Voice heartbeat coarticulation smoothness weight.
pub const VOICE_HEARTBEAT_COARTICULATION_WEIGHT: f32 = 0.8;

/// Voice heartbeat listener prediction: success value.
pub const VOICE_HEARTBEAT_LISTENER_SUCCESS: f32 = 0.8;

/// Voice heartbeat listener prediction: failure value.
pub const VOICE_HEARTBEAT_LISTENER_FAIL: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// BROCA QUALITY TUNING
// Science: Pickering & Garrod (2013) — self-monitoring in language production.
// ═══════════════════════════════════════════════════════════════════════════════

/// Weight of final coherence in Broca quality composite.
pub const BROCA_QUALITY_COHERENCE_WEIGHT: f32 = 0.4;

/// Weight of semantic PE (inverted) in Broca quality composite.
pub const BROCA_QUALITY_PE_WEIGHT: f32 = 0.4;

/// Weight of long-range coherence in Broca quality composite.
pub const BROCA_QUALITY_LONG_COHERENCE_WEIGHT: f32 = 0.2;

/// Broca quality EMA momentum (weight of prior estimate).
pub const BROCA_QUALITY_EMA_MOMENTUM: f32 = 0.85;

/// Broca quality EMA alpha (weight of new sample).
pub const BROCA_QUALITY_EMA_ALPHA: f32 = 0.15;

/// Broca quality threshold below which low-quality streak increments.
pub const BROCA_LOW_QUALITY_THRESHOLD: f32 = 0.3;

/// Consciousness threshold increase when low-quality streak reaches 3.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_INCREASE: f32 = 0.05;

/// Maximum consciousness threshold for Broca generation.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_MAX: f32 = 0.5;

/// Consciousness threshold decrease when quality EMA is high.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_DECREASE: f32 = 0.02;

/// Minimum consciousness threshold for Broca generation.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_MIN: f32 = 0.1;

/// Broca quality EMA threshold below which cadence widens (generation slows).
pub const BROCA_QUALITY_CADENCE_THRESHOLD: f32 = 0.4;

/// Broca quality EMA threshold above which threshold can decrease.
pub const BROCA_QUALITY_HIGH_THRESHOLD: f32 = 0.7;

/// Broca incoherent output threshold (final_coherence below this).
pub const BROCA_INCOHERENT_THRESHOLD: f32 = 0.3;

/// Confidence dampening rate for incoherent Broca output.
pub const BROCA_INCOHERENT_DAMPEN_RATE: f32 = 0.05;

/// Broca quality threshold above which LR is boosted.
pub const BROCA_QUALITY_LR_THRESHOLD: f32 = 0.6;

/// Broca quality LR boost scale.
pub const BROCA_QUALITY_LR_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS VELOCITY / VALENCE
// Science: Cannon (1929) — homeostatic regulation; Sokolov (1963) — habituation.
// ═══════════════════════════════════════════════════════════════════════════════

/// Homeostasis pull scale for velocity toward neutral.
pub const HOMEOSTASIS_PULL_VELOCITY_SCALE: f32 = 0.05;

/// Homeostasis pull scale for arousal toward target.
pub const HOMEOSTASIS_PULL_AROUSAL_SCALE: f32 = 0.05;

/// Valence homeostasis EMA momentum.
pub const VALENCE_HOMEOSTASIS_MOMENTUM: f32 = 0.95;

/// Valence homeostasis EMA alpha.
pub const VALENCE_HOMEOSTASIS_ALPHA: f32 = 0.05;

/// Consciousness resize center (consciousness level center for resize factor).
pub const CONSCIOUSNESS_RESIZE_CENTER: f32 = 0.5;

/// Consciousness resize scale (deviation from center × this = factor offset).
pub const CONSCIOUSNESS_RESIZE_SCALE: f32 = 0.3;

/// Goal progress base step per cycle.
pub const GOAL_DELTA_BASE_STEP: f32 = 0.01;

/// Goal progress confidence scaling factor.
pub const GOAL_DELTA_CONFIDENCE_SCALE: f32 = 0.5;

/// World model error importance scale.
pub const WORLD_MODEL_ERROR_IMPORTANCE_SCALE: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// NSM SEMANTIC PRIMITIVE INTEGRATION
// Science: Wierzbicka (1996) — universal semantic primes ground language production.
// ═══════════════════════════════════════════════════════════════════════════════

/// Blend weight for NSM primitive grounding into domain_familiarity channel.
/// When detected primitives cover input well, domain familiarity increases,
/// enabling more confident generation.
/// Science: Wierzbicka (1996) — semantic decomposition depth correlates with domain knowledge.
pub const NSM_GROUNDING_DOMAIN_BLEND: f32 = 0.5;

/// Scaling factor for detected primitive count → concept_count channel (index 19).
/// Raw count is clamped to [0, 10] after scaling.
pub const NSM_CONCEPT_COUNT_SCALE: f32 = 1.0;

/// Minimum GroundedUnderstanding confidence to inject NSM semantic HV into Broca.
/// Below this, too few words mapped to recognized primes to trust the decomposition.
/// Science: Wierzbicka (1996) — partial decomposition may mislead more than help.
pub const NSM_MIN_CONFIDENCE: f32 = 0.2;

/// Base blend weight for NSM semantic HV into thought HV (0.0–1.0).
/// Higher values give more weight to the NSM semantic content.
/// Science: Barsalou (1999) — ~30% semantic modulation for grounded cognition.
pub const NSM_SEMANTIC_BLEND_ALPHA: f32 = 0.3;

/// Logit boost for tokens expressing active NSM primes (Phase 3).
/// Applied per-prime: tokens expressing multiple active primes get multiple boosts.
/// Science: Collins & Loftus (1975) — spreading activation in semantic networks.
pub const NSM_PRIME_LOGIT_BOOST: f32 = 0.5;

/// How much NSM prime coverage relaxes semantic veto threshold (Phase 3).
/// High coverage suggests generation is on-track semantically.
/// Science: Grice (1975) — cooperative principle; semantic coverage = communicative success.
pub const NSM_COVERAGE_VETO_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// NSM CONSCIOUSNESS FEEDBACK — Phase 4
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence modulation from NSM expressive coverage (Phase 4).
/// coverage > 0.5 → positive boost; < 0.5 → dampens confidence.
/// Science: Levelt (1989) — successful articulation feeds back into speaker confidence.
pub const NSM_COVERAGE_CONFIDENCE_SCALE: f32 = 0.05;

/// Learning rate modulation from NSM expressive coverage (Phase 4).
/// High coverage → consolidate (less LR boost); low → learn more.
pub const NSM_COVERAGE_LR_SCALE: f32 = 0.02;

/// Exploration dampening when NSM coverage is high (Phase 4).
/// Successful expression reduces need to explore alternative strategies.
pub const NSM_COVERAGE_EXPLORATION_SCALE: f32 = -0.01;

/// HOT depth modulation from expressive coverage (Phase 4).
/// Linguistic expression of mental states is a form of meta-cognitive access.
/// Science: Rosenthal (2005) — higher-order thought theory.
pub const NSM_HOT_MODULATION: f32 = 0.1;

/// Broca → Phi bidirectional feedback scale.
/// Articulating a thought with high coherence and NSM coverage is itself
/// information integration. This small boost (±2% max) feeds generation
/// quality back into the consciousness level (psi).
/// Science: Dehaene (2014) — global workspace broadcasting of linguistic
/// content is a signature of conscious access.
pub const NSM_BROCA_PHI_SCALE: f32 = 0.02;
