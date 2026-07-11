// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manager intervals, governance, swarm, knowledge, trust, and sentinel constants.

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE NEUROMODULATORY CONTAGION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum dose magnitude to queue a neuromod effect (prevents spurious micro-nudges).
pub const GOV_NEUROMOD_FLOOR: f32 = 0.01;

/// NE baseline nudge on EmergencyDeclared.
/// Basis: Arnsten (2009) — acute stress NE surge for vigilance.
pub const GOV_EMERGENCY_NE_NUDGE: f32 = 0.05;

/// Oxytocin injection dose per ReciprocityPledge.
/// Basis: Zak (2012) — reciprocity → oxytocin for social bonding.
pub const GOV_RECIPROCITY_OXY_DOSE: f32 = 0.02;

/// Maximum cumulative oxytocin from reciprocity per cycle.
pub const GOV_RECIPROCITY_OXY_CAP: f32 = 0.10;

/// Half-life (in cycles) for reciprocity oxytocin injection.
pub const GOV_RECIPROCITY_OXY_HALFLIFE: u32 = 40;

/// NE baseline nudge for self-involved JusticeDispute.
/// Basis: Sapolsky (2004) — personal conflict → cortisol proxy.
pub const GOV_DISPUTE_NE_NUDGE: f32 = 0.03;

/// 5-HT baseline nudge for self-involved JusticeDispute (negative = dip).
/// Basis: Sapolsky (2004) — stress → serotonin suppression.
pub const GOV_DISPUTE_SHT_NUDGE: f32 = -0.02;

/// DA phasic injection dose on aligned pass.
/// Basis: Schultz (1997) — reward prediction confirmation → phasic dopamine.
pub const GOV_ALIGNED_PASS_DA_DOSE: f32 = 0.10;

/// Half-life (in cycles) for aligned-pass DA injection.
pub const GOV_ALIGNED_PASS_DA_HALFLIFE: u32 = 20;

/// DA baseline nudge on aligned fail (negative = dip).
/// Basis: Schultz (1997) — reward prediction error → dopamine suppression.
pub const GOV_ALIGNED_FAIL_DA_NUDGE: f32 = -0.02;

/// 5-HT baseline nudge on negative reputation change.
/// Basis: Crockett (2009) — social rejection → serotonin dip.
pub const GOV_REPUTATION_DECLINE_SHT: f32 = -0.02;

/// 5-HT baseline nudge on positive reputation change.
/// Basis: Crockett (2009) — social approval → serotonin boost (symmetric with decline).
pub const GOV_REPUTATION_GAIN_SHT: f32 = 0.02;

/// ECB baseline nudge on high collective Phi (>0.5).
/// Group coherence → endocannabinoid system activation.
pub const GOV_COLLECTIVE_PHI_ECB: f32 = 0.01;

/// Collective Phi → consciousness modulation strength (±2%).
/// High collective Phi boosts unified consciousness via social integration.
/// Basis: Woolley et al. (2010) — collective intelligence factor.
pub const GOV_CONSCIOUSNESS_MODULATION: f64 = 0.04; // ±2% at extremes (0.04 × 0.5 = 0.02)

/// Number of lagged consciousness values for governance decorrelation.
/// At 20Hz, 50 cycles ≈ 2.5s — enough temporal separation to break the
/// consciousness → governance → neuromod → consciousness feedback loop.
/// Basis: Granger (1969) — temporal decorrelation breaks circular causation.
pub const GOVERNANCE_CONSCIOUSNESS_LAG_SIZE: usize = 50;

/// Maximum age (in cycles) before stale ethics consequence predictions are expired.
/// Basis: Cushman (2013) — dual-process moral cognition outcome observation windows.
pub const CONSEQUENCE_TRACKER_MAX_AGE_CYCLES: u64 = 2000;

/// EMA alpha for consequence tracker prediction accuracy.
/// Basis: Friston (2010) — precision-weighted prediction error learning.
pub const CONSEQUENCE_TRACKER_ACCURACY_ALPHA: f64 = 0.05;

/// Cycles between recording an ethical verdict as a consequence prediction
/// and resolving it against the Ψ/valence that actually materialized.
/// At ~31Hz, 20 cycles ≈ 0.65s — long enough for an input's effects to
/// propagate through several ticks, short enough that predictions resolve
/// within the same behavioral episode. Must stay well below
/// CONSEQUENCE_TRACKER_MAX_AGE_CYCLES (2000) and the tracker's pending cap
/// (100), since one prediction is recorded per cycle.
/// Basis: Cushman (2013) — outcome observation follows action at short lag.
pub const CONSEQUENCE_OBSERVATION_HORIZON_CYCLES: u64 = 20;

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE LEARNING — OUTCOME-BASED FEEDBACK
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence boost when agent's vote aligned with outcome.
/// Basis: Schultz (1997) — reward prediction confirmation → confidence.
pub const GOV_ALIGNED_VOTE_CONFIDENCE: f64 = 0.02;

/// Confidence penalty when agent's vote misaligned with outcome.
/// Basis: Schultz (1997) — prediction error → confidence reduction.
pub const GOV_MISALIGNED_VOTE_CONFIDENCE: f64 = -0.03;

/// Prediction error → LR boost scaling factor.
/// Higher PE → faster learning about governance dynamics.
/// Basis: Friston (2010) — precision-weighted free-energy minimization.
pub const GOV_PE_LR_SCALE: f64 = 0.3;

/// Maximum LR boost from governance prediction error.
pub const GOV_PE_LR_MAX_BOOST: f64 = 0.5;

/// Outcome sign for failed proposals (asymmetric: losses loom larger).
/// Basis: Kahneman & Tversky (1979) — loss aversion in prospect theory.
pub const GOV_FAILED_OUTCOME_SIGN: f64 = -0.5;

/// EMA decay for governance reward tracking (alpha = 1 - decay).
/// Basis: Sutton & Barto (2018) — exponential recency-weighted average.
pub const GOV_REWARD_EMA_DECAY: f64 = 0.9;

/// Collective Phi threshold for high-coherence governance actions.
/// Above this, group is coherent enough to trigger ECB/confidence signals.
/// Basis: Woolley et al. (2010) — collective intelligence factor.
pub const GOV_COLLECTIVE_PHI_HIGH: f64 = 0.5;

/// Collective Phi threshold below which consensus is fragile → explore.
/// Basis: Sunstein (2002) — deliberative polling under low agreement.
pub const GOV_FRAGILE_CONSENSUS_PHI: f64 = 0.3;

/// Exploration boost per fragile consensus detection.
pub const GOV_FRAGILE_CONSENSUS_EXPLORE: f64 = 0.05;

/// Confidence boost per high-Phi voter observed.
pub const GOV_HIGH_PHI_VOTER_CONFIDENCE: f64 = 0.005;

/// Arousal delta on emergency declaration.
pub const GOV_EMERGENCY_AROUSAL: f32 = 0.1;

/// Exploration suppression during emergency.
pub const GOV_EMERGENCY_EXPLORE_SUPPRESS: f64 = 0.1;

/// Per-tally LR boost scaling (each tally adds this × tally_count, capped).
/// Basis: active governance engagement → heightened learning plasticity.
pub const GOV_TALLY_LR_SCALE: f64 = 0.02;

/// Maximum per-cycle LR boost from tally count.
pub const GOV_TALLY_LR_MAX_BOOST: f64 = 0.1;

/// Blind spot severity → exploration scaling.
/// Basis: Friston (2010) — epistemic affordance drives exploration.
pub const GOV_BLIND_SPOT_EXPLORE_SCALE: f64 = 0.05;

/// Community mode harmonic bias per cycle.
/// Basis: gentle influence, not override — social context nudges moral weight.
pub const GOV_COMMUNITY_HARMONIC_BIAS: f64 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// DRIVE MANAGER — BOREDOM, FLOW & CURIOSITY DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Boredom increment per low-error cycle (linear ramp to MAX_BOREDOM).
/// Basis: Berlyne (1960) — arousal potential theory; monotony → exploration drive.
pub const DRIVE_BOREDOM_INCREMENT: f32 = 0.03;

/// Boredom level that triggers exploration boost.
/// Basis: Berlyne (1960) — above this, boredom generates exploration urge.
pub const DRIVE_BOREDOM_EXPLORATION_THRESHOLD: f32 = 0.3;

/// Boredom→exploration coupling scale.
pub const DRIVE_BOREDOM_EXPLORATION_SCALE: f64 = 0.1;

/// Minimum arousal for flow state (below this, system is disengaged).
/// Basis: Csikszentmihalyi (1990) — flow requires optimal arousal.
pub const DRIVE_FLOW_AROUSAL_MIN: f32 = 0.3;

/// Maximum arousal for flow state (above this, system is over-stimulated).
pub const DRIVE_FLOW_AROUSAL_MAX: f32 = 0.8;

/// Flow→learning rate boost scale.
/// Basis: Csikszentmihalyi (1990) — peak plasticity during flow.
pub const DRIVE_FLOW_LR_BOOST: f64 = 0.1;

/// Flow→exploration dampening scale.
pub const DRIVE_FLOW_EXPLORATION_DAMPEN: f64 = 0.05;

/// Flow→confidence boost scale.
pub const DRIVE_FLOW_CONFIDENCE_BOOST: f64 = 0.01;

/// Boredom reset factor when entering flow.
pub const DRIVE_FLOW_BOREDOM_RESET: f32 = 0.5;

/// Surprise→exploration coupling scale (positive PE excess drives exploration).
/// Basis: Yerkes-Dodson (1908) — moderate arousal optimal for exploration.
pub const DRIVE_SURPRISE_EXPLORATION_SCALE: f64 = 0.15;

/// Surprise→arousal coupling scale.
pub const DRIVE_SURPRISE_AROUSAL_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// PERCEPTION MANAGER — ATTENTION BUDGET & COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay for attention budget utilization tracking.
pub const PERCEPTION_BUDGET_EMA_DECAY: f32 = 0.8;

/// EMA weight for new budget utilization data (1 - decay).
pub const PERCEPTION_BUDGET_EMA_NEW: f32 = 0.2;

/// Exploration reduction when attention budget consistently exceeded.
pub const PERCEPTION_BUDGET_EXPLORATION_DAMPEN: f64 = 0.02;

/// LR dampen factor when budget utilization exceeds warning threshold.
pub const PERCEPTION_BUDGET_LR_DAMPEN: f64 = 0.95;

/// Coherence→confidence boost coupling scale.
/// Basis: Friston (2010) — precision-weighted prediction error.
pub const PERCEPTION_COHERENCE_CONFIDENCE_SCALE: f64 = 0.015;

/// Low coherence→exploration boost coupling scale.
pub const PERCEPTION_COHERENCE_EXPLORATION_SCALE: f64 = 0.02;

/// Arousal correction gain (Yerkes-Dodson homeostasis).
/// Basis: Yerkes & Dodson (1908) — inverted-U performance curve.
pub const PERCEPTION_AROUSAL_CORRECTION_GAIN: f32 = 0.03;

/// Vigilance mode attention amplification factor.
pub const PERCEPTION_VIGILANCE_AMPLIFY: f32 = 1.2;

/// Vigilance exit attention recovery factor.
pub const PERCEPTION_VIGILANCE_RECOVERY: f32 = 0.9;

/// Phenomenal binding→confidence modulation scale.
/// Basis: Treisman & Gelade (1980) — feature integration theory.
pub const PERCEPTION_BINDING_CONFIDENCE_SCALE: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING MANAGER — PLASTICITY & CONSOLIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Surprise threshold above which plasticity increases.
/// Basis: BCM metaplasticity (Bienenstock, Cooper & Munro 1982).
pub const LEARNING_PLASTICITY_HIGH_SURPRISE: f32 = 0.3;

/// Surprise threshold below which plasticity decreases.
pub const LEARNING_PLASTICITY_LOW_SURPRISE: f32 = 0.1;

/// LR modulation floor (minimum learning rate multiplier from plasticity).
pub const LEARNING_LR_FLOOR: f64 = 0.8;

/// Plasticity→LR scaling range (plasticity [0.1,0.95] maps to [floor, floor+scale]).
pub const LEARNING_LR_PLASTICITY_SCALE: f64 = 0.4;

/// Exploration suppression during dream consolidation.
/// Basis: Walker & Stickgold (2006) — sleep consolidation suppresses external seeking.
pub const LEARNING_DREAM_EXPLORATION_DAMPEN: f64 = 0.03;

/// LR boost during dream consolidation phase.
/// Basis: Walker & Stickgold (2006) — enhanced synaptic consolidation.
pub const LEARNING_DREAM_LR_BOOST: f64 = 1.1;

/// Error trend→exploration coupling scale.
/// Basis: Friston (2010) — increasing prediction error drives active inference.
pub const LEARNING_ERROR_TREND_EXPLORATION: f64 = 0.1;

/// Error trend→LR boost (increasing errors need faster adaptation).
pub const LEARNING_ERROR_TREND_LR_BOOST: f64 = 1.05;

/// Error trend→confidence coupling on decreasing errors.
pub const LEARNING_ERROR_TREND_CONFIDENCE: f64 = 0.01;

/// Dissipative health threshold for learning dampening.
pub const LEARNING_DISSIPATIVE_HEALTH_THRESHOLD: f64 = 0.5;

/// Dissipative health→LR dampening sensitivity.
pub const LEARNING_DISSIPATIVE_HEALTH_SENSITIVITY: f64 = 0.4;

/// Somatic stress threshold for plasticity dampening.
pub const LEARNING_SOMATIC_STRESS_THRESHOLD: f64 = 0.5;

/// Somatic stress→LR dampening sensitivity.
pub const LEARNING_SOMATIC_STRESS_SENSITIVITY: f64 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM NEUROMODULATORY COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Oxytocin dose per √(connected_peers). Diminishing returns via sqrt.
/// Basis: Zak (2012) — social bonding → oxytocin release.
pub const SWARM_OXY_PER_SQRT_PEER: f32 = 0.02;

/// Maximum oxytocin dose from peer bonding (caps sqrt scaling).
pub const SWARM_OXY_CAP: f32 = 0.08;

/// Oxytocin injection half-life (cycles) for peer bonding signal.
pub const SWARM_OXY_HALFLIFE: u32 = 60;

/// NE baseline nudge multiplier per anomaly (capped at 3 anomalies).
/// Basis: Arnsten (2009) — network disruption → noradrenergic alarm.
pub const SWARM_ANOMALY_NE_MULT: f32 = 0.03;

/// Maximum NE nudge from network anomalies.
pub const SWARM_ANOMALY_NE_CAP: f32 = 0.09;

/// 5-HT gain from peer Phi delta (mean_peer_phi - 0.5).
/// Basis: Crockett (2009) — collective flourishing → serotonin (social satisfaction).
pub const SWARM_PHI_SHT_GAIN: f32 = 0.04;

/// Maximum 5-HT nudge from high collective Phi.
pub const SWARM_PHI_SHT_CAP: f32 = 0.03;

/// DA gain from affective contagion intensity.
/// Basis: Schultz (1997) — shared positive affect → dopaminergic reward.
pub const SWARM_CONTAGION_DA_GAIN: f32 = 0.03;

/// Maximum DA nudge from affective contagion.
pub const SWARM_CONTAGION_DA_CAP: f32 = 0.04;

/// Minimum affective contagion to trigger DA modulation.
pub const SWARM_CONTAGION_DA_THRESHOLD: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROEVOLUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Bits encoding tau_base in the neural genome.
/// Hasani et al. (2021) — LTC time constant is the primary evolutionary target.
pub const NEUROEVO_TAU_BASE_BITS: usize = 16;

/// Maximum CfC network layers in the evolved topology.
pub const NEUROEVO_MAX_LAYERS: usize = 5;

/// Maximum neurons per layer in evolved topology.
pub const NEUROEVO_MAX_NEURONS_PER_LAYER: usize = 32;

/// FEP hidden state dimension (projected from 16,384D via strided sampling).
/// Friston (2010) — 32D sufficient for belief state dynamics.
pub const NEUROEVO_FEP_STATE_DIM: usize = 32;

/// Maximum organism age before death eligibility.
pub const NEUROEVO_MAX_AGE_CYCLES: u32 = 500;

/// Floor for fitness to prevent -inf domination.
/// Stanley & Miikkulainen (2002) — capped negative fitness.
pub const NEUROEVO_FITNESS_FLOOR: f64 = -10.0;

/// Evaluation steps per organism (after warmup).
/// Hasani et al. (2021) — 100 steps sufficient for LTC dynamics characterization.
pub const NEUROEVO_EVAL_STEPS: usize = 100;

/// Warmup steps excluded from fitness computation.
/// LTC dynamics need ~20τ to settle.
pub const NEUROEVO_WARMUP_STEPS: usize = 20;

/// Default free energy fitness weight.
/// Friston (2010) — free energy as primary fitness signal.
pub const NEUROEVO_FE_FITNESS_WEIGHT: f64 = 0.3;

/// Default Phi fitness weight.
/// Tononi (2004) — information integration as consciousness measure.
pub const NEUROEVO_PHI_FITNESS_WEIGHT: f64 = 0.3;

/// Default population size.
/// Stanley & Miikkulainen (2002) — 50 balances diversity vs compute.
pub const NEUROEVO_POPULATION_SIZE: usize = 50;

/// Default tournament selection size.
/// Goldberg & Deb (1991) — 3 gives moderate selection pressure.
pub const NEUROEVO_TOURNAMENT_SIZE: usize = 3;

/// Default elitism fraction.
/// De Jong (1975) — 10% elite preservation.
pub const NEUROEVO_ELITISM_FRACTION: f32 = 0.1;

/// Default per-bit mutation rate.
/// Back (1993) — ~0.02 balances exploration and stability.
pub const NEUROEVO_MUTATION_RATE: f32 = 0.02;

/// Default crossover probability.
/// Holland (1975) — 0.7 crossover rate for GA.
pub const NEUROEVO_CROSSOVER_RATE: f32 = 0.7;

/// Generations without improvement before convergence declared.
pub const NEUROEVO_CONVERGENCE_PATIENCE: usize = 10;

/// Hamming distance threshold for speciation.
pub const NEUROEVO_SPECIATION_THRESHOLD: f32 = 0.15;

/// Manager trigger interval (co-prime with other managers).
pub const NEUROEVO_MANAGER_INTERVAL: usize = 71;

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY MANAGER — Consolidation, retrieval, Phi-weighted encoding
// Basis: Tulving (2002) — episodic consolidation; Cowan (2001) — capacity limits
// ═══════════════════════════════════════════════════════════════════════════════

/// Exploration dampening during active consolidation.
/// Basis: Wixted (2004) — consolidation benefits from reduced interference.
pub const MEMORY_CONSOLIDATION_EXPLORATION_DAMPEN: f64 = 0.02;

/// Learning rate boost during consolidation (slight integration bonus).
/// Basis: McClelland et al. (1995) — complementary learning systems.
pub const MEMORY_CONSOLIDATION_LR_BOOST: f64 = 1.05;

/// Weight of prediction confidence in retrieval signal blend.
/// Basis: Cowan (2001) — confidence + coherence jointly index memory quality.
pub const MEMORY_RETRIEVAL_CONFIDENCE_WEIGHT: f64 = 0.5;

/// Weight of coherence in retrieval signal blend.
pub const MEMORY_RETRIEVAL_COHERENCE_WEIGHT: f64 = 0.5;

/// Retrieval quality threshold for confidence boost.
/// Basis: Koriat (2000) — high-confidence retrieval signals reliable encoding.
pub const MEMORY_RETRIEVAL_HIGH_QUALITY: f32 = 0.7;

/// Confidence gain per unit of above-threshold retrieval quality.
pub const MEMORY_RETRIEVAL_CONFIDENCE_GAIN: f64 = 0.02;

/// Retrieval quality threshold below which exploration is boosted.
pub const MEMORY_RETRIEVAL_LOW_QUALITY: f32 = 0.3;

/// Exploration gain per unit of below-threshold retrieval quality.
pub const MEMORY_RETRIEVAL_EXPLORATION_GAIN: f64 = 0.03;

/// Unified Ψ threshold above which episodic consolidation is prioritized.
/// Basis: Tononi & Cirelli (2014) — high-consciousness moments deserve priority encoding.
pub const MEMORY_PSI_CONSOLIDATION_THRESHOLD: f64 = 0.6;

/// LR scale for Ψ-driven consolidation bonus.
pub const MEMORY_PSI_LR_SCALE: f64 = 0.1;

/// Consolidation pressure threshold for exploration gating.
/// Basis: Wixted (2004) — "digest before exploring" principle.
pub const MEMORY_PRESSURE_EXPLORATION_THRESHOLD: f64 = 0.5;

/// Exploration dampening scale per unit of excess consolidation pressure.
pub const MEMORY_PRESSURE_EXPLORATION_DAMPEN: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// SWARM MANAGER — Peer consciousness, social buffering, collective Φ
// Basis: Heinrichs (2003) — social support; Hatfield (1993) — emotional contagion
// ═══════════════════════════════════════════════════════════════════════════════

/// Trust-scaled initial Φ for newly-joined peers.
/// Basis: Dunbar (1998) — conservative initial assessment.
pub const SWARM_PEER_PHI_TRUST_SCALE: f64 = 0.5;

/// Arousal center for affective sync (neutral arousal baseline).
/// Basis: Yerkes-Dodson (1908) — 0.5 as optimal midpoint.
pub const SWARM_AFFECTIVE_AROUSAL_CENTER: f64 = 0.5;

/// Per-corroboration confidence boost for shared knowledge.
/// Basis: Surowiecki (2004) — wisdom of crowds via independent confirmation.
pub const SWARM_CORROBORATION_BOOST: f32 = 0.05;

/// Maximum corroboration confidence bonus.
pub const SWARM_CORROBORATION_CAP: f32 = 0.3;

/// Maximum confidence delta from social buffering.
/// Basis: Heinrichs (2003) — social support has diminishing returns.
pub const SWARM_SOCIAL_BUFFERING_CAP: f64 = 0.05;

/// Collective Φ threshold above which learning rate is boosted.
/// Basis: Woolley et al. (2010) — collective intelligence emerges above threshold.
pub const SWARM_COLLECTIVE_PHI_THRESHOLD: f64 = 0.3;

/// Scale for collective Φ → LR boost.
pub const SWARM_COLLECTIVE_PHI_LR_SCALE: f64 = 0.2;

/// Maximum LR boost from collective Φ.
pub const SWARM_COLLECTIVE_PHI_LR_CAP: f64 = 0.1;

/// Multiplier for federated boost calculation.
/// Basis: McMahan et al. (2017) — federated rounds amplify gradient quality.
pub const SWARM_FEDERATED_BOOST_MULTIPLIER: f64 = 2.0;

/// Per-streak exploration boost from network anomaly.
/// Basis: Aston-Jones & Cohen (2005) — unexpected events trigger exploration.
pub const SWARM_ANOMALY_EXPLORATION: f64 = 0.03;

/// Per-streak confidence penalty from sustained anomaly.
pub const SWARM_ANOMALY_CONFIDENCE: f64 = 0.02;

/// Connectivity EMA threshold for isolation detection.
pub const SWARM_ISOLATION_THRESHOLD: f64 = 0.2;

/// Exploration boost when isolated (no connected peers).
pub const SWARM_ISOLATION_EXPLORATION_BOOST: f64 = 0.05;

// ── Space alerts (feature: space-alerts) ─────────────────────────────────

/// Arousal delta from conjunction warning.
/// Basis: Kahneman (2011) — threat salience drives attentional capture.
pub const SPACE_CONJUNCTION_AROUSAL: f32 = 0.10;

/// Arousal delta from debris proximity (fight-or-flight analogue).
/// Basis: Cannon (1929) — acute threat triggers sympathetic arousal.
pub const SPACE_DEBRIS_AROUSAL: f32 = 0.15;

/// Valence delta from debris threat (negative affect).
/// Basis: LeDoux (2003) — amygdala-mediated aversive valence.
pub const SPACE_DEBRIS_VALENCE: f32 = -0.08;

/// Confidence reduction from debris uncertainty.
/// Basis: Kahneman & Tversky (1979) — uncertainty reduces decision confidence.
pub const SPACE_DEBRIS_CONFIDENCE: f32 = -0.10;

/// Confidence boost from communication window opportunity.
/// Basis: Heinrichs (2003) — social/resource access buffers stress.
pub const SPACE_COMM_CONFIDENCE: f32 = 0.05;

/// Learning rate boost during communication windows.
/// Basis: Schultz (1997) — reward prediction boosts dopaminergic learning.
pub const SPACE_COMM_LR_BOOST: f32 = 0.10;

/// Exploration boost from conjunction (search for alternatives).
/// Basis: Aston-Jones & Cohen (2005) — threat uncertainty promotes exploration.
pub const SPACE_CONJUNCTION_EXPLORATION: f32 = 0.05;

/// Arousal delta from orbital anomaly detection.
/// Basis: Sokolov (1963) — orienting response to unexpected stimuli.
pub const SPACE_ANOMALY_AROUSAL: f32 = 0.08;

/// Confidence boost from maneuver announcement (neutral information).
pub const SPACE_MANEUVER_CONFIDENCE: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// CIRCULAR ECONOMY — Waste circularity → neuromodulator coupling
// Basis: Ellen MacArthur Foundation (2015) — circular economy as feedback loop
// McEwen (2007) — serotonin role in system stability/homeostatic confidence
// Arnsten (2009) — noradrenaline vigilance to resource constraints
// ═══════════════════════════════════════════════════════════════════════════════

/// Serotonin gain from high circularity potential (system health signal).
/// Basis: McEwen (2007) — serotonin supports homeostatic confidence.
pub const CIRCULAR_ECONOMY_SHT_GAIN: f32 = 0.15;

/// Serotonin half-life (cycles) for circularity boost.
pub const CIRCULAR_ECONOMY_SHT_HALFLIFE: f32 = 8.0;

/// Noradrenaline gain from low material entropy (concentration risk).
/// Basis: Arnsten (2009) — NE drives vigilance under resource constraint.
pub const CIRCULAR_ECONOMY_NE_GAIN: f32 = 0.20;

/// NE half-life (cycles) for waste vigilance — fast transient.
pub const CIRCULAR_ECONOMY_NE_HALFLIFE: f32 = 4.0;

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST MANAGER — Violation response, anomaly detection
// Basis: Zak (2012) — trust/oxytocin; Dunbar (1998) — social brain
// ═══════════════════════════════════════════════════════════════════════════════

/// Trust slashing factor on violation (multiplied against current trust).
/// Basis: Zak (2012) — betrayal triggers steep trust loss.
pub const TRUST_VIOLATION_SLASH_FACTOR: f64 = 0.5;

/// Maximum arousal delta from trust violations (NE cap).
pub const TRUST_VIOLATION_AROUSAL_CAP: f32 = 0.1;

/// Negative valence per violation event (betrayal penalty).
/// Basis: Zak (2012) — cortisol-mediated aversive response.
pub const TRUST_BETRAYAL_VALENCE_PENALTY: f32 = 0.02;

/// Arousal spike from trust anomaly detection.
pub const TRUST_ANOMALY_AROUSAL: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL FABRIC MANAGER — Resonance, diversity, echo-chamber detection
// Basis: Woolley et al. (2010) — collective intelligence factor
// ═══════════════════════════════════════════════════════════════════════════════

/// Resonance threshold for oxytocin-mediated valence boost.
/// Basis: Woolley (2010) — high resonance signals productive collaboration.
pub const SOCIAL_RESONANCE_HIGH_THRESHOLD: f64 = 0.6;

/// Range denominator for normalizing resonance excess.
pub const SOCIAL_RESONANCE_RANGE: f64 = 0.4;

/// Maximum arousal from resonance drop (NE cap).
pub const SOCIAL_RESONANCE_DROP_AROUSAL_CAP: f64 = 0.05;

/// Diversity threshold for dopamine-mediated curiosity boost.
/// Basis: Page (2007) — cognitive diversity drives innovation.
pub const SOCIAL_DIVERSITY_THRESHOLD: f64 = 0.3;

/// Echo chamber risk threshold triggering anomaly flag.
/// Basis: Sunstein (2001) — group polarization above critical homogeneity.
pub const SOCIAL_ECHO_CHAMBER_THRESHOLD: f64 = 0.85;

// ═══════════════════════════════════════════════════════════════════════════════
// TIME MANAGER — Drift surprise, consensus stability
// Basis: Mills (1985) — NTP; Aston-Jones & Cohen (2005) — temporal alertness
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum arousal delta from drift surprise (NE cap).
pub const TIME_DRIFT_AROUSAL_CAP: f64 = 0.05;

/// Normalization divisor for drift surprise → arousal conversion.
pub const TIME_DRIFT_SURPRISE_DIVISOR: f64 = 100.0;

// ═══════════════════════════════════════════════════════════════════════════════
// SENTINEL MANAGER — Threat detection, vigilance, immune response
// Basis: Aston-Jones & Cohen (2005) — LC-NE vigilance; immune system analogy
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal scaling for low-level threat (proportional vigilance).
/// Basis: Aston-Jones & Cohen (2005) — LC-NE governs arousal/vigilance.
pub const SENTINEL_AROUSAL_SCALE_NORMAL: f32 = 0.05;

/// Threat level threshold for moderate response (exploration dampening).
pub const SENTINEL_THREAT_MODERATE: f32 = 0.3;

/// Exploration dampening scale per unit of threat level.
pub const SENTINEL_EXPLORATION_DAMPEN_SCALE: f64 = 0.1;

/// Arousal scaling for moderate threat (heightened vigilance).
pub const SENTINEL_AROUSAL_SCALE_HEIGHTENED: f32 = 0.08;

/// Threat level threshold for critical response (confidence reduction).
pub const SENTINEL_THREAT_CRITICAL: f32 = 0.6;

/// Confidence dampening scale per unit of critical threat level.
/// Basis: epistemic caution under significant threat.
pub const SENTINEL_CONFIDENCE_DAMPEN_SCALE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// GUIDING QUESTION — Top-down attention modulation
// Basis: Desimone & Duncan (1995) — biased competition for task-relevant features
// ═══════════════════════════════════════════════════════════════════════════════

/// Exploration boost for epistemic guiding questions ("know", "learn", "understand").
/// Basis: Gottlieb et al. (2013) — curiosity-driven exploration.
pub const GUIDING_EPISTEMIC_EXPLORATION_BOOST: f64 = 0.03;

/// Confidence boost for affective guiding questions ("feel", "emotion", "care").
/// Basis: Damasio (1994) — somatic marker hypothesis, emotional salience.
pub const GUIDING_AFFECTIVE_CONFIDENCE_BOOST: f64 = 0.01;

/// Learning rate factor for pragmatic guiding questions ("do", "act", "make").
/// Basis: Dolan & Dayan (2013) — action-oriented processing boosts learning.
pub const GUIDING_PRAGMATIC_LR_FACTOR: f64 = 1.02;

/// Confidence boost for social guiding questions ("connect", "relate", "together").
/// Basis: Woolley et al. (2010) — social coherence sensitivity.
pub const GUIDING_SOCIAL_CONFIDENCE_BOOST: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// CIVIC CRISIS DETECTOR — Classification and severity
// Basis: Friston (2010) — prediction error; Tononi (2004) — Phi collapse
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum threshold denominator to prevent division by zero in confidence computation.
pub const CRISIS_CONFIDENCE_MIN_DENOMINATOR: f64 = 0.001;

/// Maximum signal strength multiplier in confidence computation.
pub const CRISIS_CONFIDENCE_MAX_SIGNAL: f64 = 2.0;

/// Safety ordinal threshold for Red level (CyberAttack classification).
pub const CRISIS_SAFETY_RED_ORDINAL: u8 = 3;

/// Safety ordinal threshold for Orange level.
pub const CRISIS_SAFETY_ORANGE_ORDINAL: u8 = 2;

/// Severity boost when safety level is Red.
pub const CRISIS_SEVERITY_BOOST_RED: u8 = 2;

/// Severity boost when safety level is Orange.
pub const CRISIS_SEVERITY_BOOST_ORANGE: u8 = 1;

/// Minimum crisis severity.
pub const CRISIS_SEVERITY_MIN: u8 = 1;

/// Maximum crisis severity.
pub const CRISIS_SEVERITY_MAX: u8 = 5;
