// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk GIS v4.0 - Graceful Ignorance System
 *
 * The Kosmic Song synthesis: Consciousness (Φ) + Eight Harmonies + Epistemic Humility
 *
 * This module extends the E-N-M classification with:
 * - H-dimension (Harmonic): Which value harmonies are affected
 * - Tripartite Moral Uncertainty: Epistemic × Axiological × Deontic
 * - Rashomon multi-perspective generation
 * - KosmicSong unified agent identity
 *
 * @packageDocumentation
 * @module epistemic/gis
 */

// ============================================================================
// SEVEN HARMONIES
// ============================================================================

/**
 * The Seven Primary Harmonies of Infinite Love
 *
 * Each harmony represents both a value dimension AND an epistemic lens.
 * From the Kosmic Song: "Infinite Love as Rigorous, Playful, Co-Creative Becoming"
 */
export enum Harmony {
  /** Integration-Knowing: Luminous order, boundless creativity, coherent unity */
  ResonantCoherence = 0,
  /** Care-Knowing: Unconditional care, intrinsic value recognition */
  PanSentientFlourishing = 1,
  /** Truth-Knowing: Self-illuminating intelligence, embodied wisdom */
  IntegralWisdom = 2,
  /** Creative-Knowing: Joyful generativity, endless novelty, divine play */
  InfinitePlay = 3,
  /** Relational-Knowing: Fundamental unity, empathic resonance */
  UniversalInterconnectedness = 4,
  /** Exchange-Knowing: Generous flow, mutual upliftment, generative trust */
  SacredReciprocity = 5,
  /** Developmental-Knowing: Wise becoming, continuous evolution */
  EvolutionaryProgression = 6,
}

/**
 * Harmonic level for the H-dimension (extends E-N-M to E-N-M-H)
 *
 * Indicates how many harmonies are affected by a claim or decision.
 */
export enum HarmonicLevel {
  /** Affects no specific harmony - purely technical/factual */
  H0_None = 0,
  /** Affects a single harmony */
  H1_Single = 1,
  /** Affects multiple related harmonies (2-3) */
  H2_Multiple = 2,
  /** Affects most harmonies (4-6) */
  H3_Broad = 3,
  /** Affects all eight harmonies - civilizational scope */
  H4_Universal = 4,
}

/**
 * Base weight for each harmony (normalized to sum to 1.0)
 *
 * These weights represent the relative importance of each harmony
 * in calculating aggregate harmonic impact scores.
 */
export const HARMONY_WEIGHTS: Record<Harmony, number> = {
  [Harmony.ResonantCoherence]: 0.20,
  [Harmony.PanSentientFlourishing]: 0.18,
  [Harmony.IntegralWisdom]: 0.16,
  [Harmony.InfinitePlay]: 0.12,
  [Harmony.UniversalInterconnectedness]: 0.14,
  [Harmony.SacredReciprocity]: 0.12,
  [Harmony.EvolutionaryProgression]: 0.08,
};

/**
 * Get the display name for a harmony
 */
export function harmonyName(harmony: Harmony): string {
  const names: Record<Harmony, string> = {
    [Harmony.ResonantCoherence]: 'Resonant Coherence',
    [Harmony.PanSentientFlourishing]: 'Pan-Sentient Flourishing',
    [Harmony.IntegralWisdom]: 'Integral Wisdom',
    [Harmony.InfinitePlay]: 'Infinite Play',
    [Harmony.UniversalInterconnectedness]: 'Universal Interconnectedness',
    [Harmony.SacredReciprocity]: 'Sacred Reciprocity',
    [Harmony.EvolutionaryProgression]: 'Evolutionary Progression',
  };
  return names[harmony];
}

/**
 * Get the epistemic mode (knowing-type) for a harmony
 */
export function harmonyEpistemicMode(harmony: Harmony): string {
  const modes: Record<Harmony, string> = {
    [Harmony.ResonantCoherence]: 'Integration-Knowing',
    [Harmony.PanSentientFlourishing]: 'Care-Knowing',
    [Harmony.IntegralWisdom]: 'Truth-Knowing',
    [Harmony.InfinitePlay]: 'Creative-Knowing',
    [Harmony.UniversalInterconnectedness]: 'Relational-Knowing',
    [Harmony.SacredReciprocity]: 'Exchange-Knowing',
    [Harmony.EvolutionaryProgression]: 'Developmental-Knowing',
  };
  return modes[harmony];
}

// ============================================================================
// HARMONIC IGNORANCE
// ============================================================================

/**
 * Types of ignorance (based on the five-fold taxonomy)
 */
export enum IgnoranceType {
  /** We know we don't know */
  KnownUnknown = 'known_unknown',
  /** We don't know we don't know */
  UnknownUnknown = 'unknown_unknown',
  /** We think we know but don't */
  FalseKnowledge = 'false_knowledge',
  /** Knowledge exists but is inaccessible now */
  TemporallyInaccessible = 'temporally_inaccessible',
  /** Question may be fundamentally unanswerable */
  Impossible = 'impossible',
}

/**
 * Harmonic impact: how a specific harmony is affected
 */
export interface HarmonicImpact {
  /** The affected harmony */
  harmony: Harmony;
  /** Impact magnitude (0.0 to 1.0) */
  impact: number;
  /** Direction: positive = aligns with harmony, negative = conflicts */
  direction: 'positive' | 'negative' | 'neutral';
}

/**
 * Ignorance weighted by affected harmonies
 *
 * Extends basic ignorance classification with harmonic context,
 * enabling prioritization based on value alignment.
 */
export interface HarmonicIgnorance {
  /** Unique identifier */
  id: string;
  /** The query or claim that revealed this ignorance */
  query: string;
  /** Base type of ignorance */
  ignoranceType: IgnoranceType;
  /** Which harmonies are affected and how strongly */
  affectedHarmonies: HarmonicImpact[];
  /** Domain classification (for routing to experts) */
  domain: string;
  /** Base Expected Information Gain */
  baseEig: number;
  /** Harmonic-weighted EIG (prioritizes value-aligned resolution) */
  harmonicEig: number;
  /** When this ignorance was detected */
  detectedAt: number;
  /** Resolution status */
  status: 'active' | 'resolving' | 'resolved' | 'expired';
  /** H-code summary (e.g., "H2-RC-PF" for 2 harmonies) */
  hCode: string;
}

/**
 * Generate H-code from harmonic impacts
 *
 * Format: H{level}-{harmony_abbreviations}
 * Example: "H2-RC-PF" (2 harmonies: ResonantCoherence, PanSentientFlourishing)
 */
export function generateHCode(impacts: HarmonicImpact[]): string {
  const significant = impacts.filter((i) => i.impact >= 0.1);
  if (significant.length === 0) return 'H0';

  const level = Math.min(4, significant.length);
  const abbrevs = significant
    .sort((a, b) => b.impact - a.impact)
    .slice(0, 4)
    .map((i) => harmonyAbbreviation(i.harmony));

  return `H${level}-${abbrevs.join('-')}`;
}

/**
 * Two-letter abbreviation for each harmony
 */
export function harmonyAbbreviation(harmony: Harmony): string {
  const abbrevs: Record<Harmony, string> = {
    [Harmony.ResonantCoherence]: 'RC',
    [Harmony.PanSentientFlourishing]: 'PF',
    [Harmony.IntegralWisdom]: 'IW',
    [Harmony.InfinitePlay]: 'IP',
    [Harmony.UniversalInterconnectedness]: 'UI',
    [Harmony.SacredReciprocity]: 'SR',
    [Harmony.EvolutionaryProgression]: 'EP',
  };
  return abbrevs[harmony];
}

/**
 * Calculate harmonic-weighted EIG
 *
 * Formula: H-EIG = base_eig × (1 + Σ(harmony_weight × impact × relevance))
 */
export function calculateHarmonicEig(
  baseEig: number,
  impacts: HarmonicImpact[],
  contextRelevance: Partial<Record<Harmony, number>> = {}
): number {
  const harmonicWeight = impacts.reduce((sum, impact) => {
    const weight = HARMONY_WEIGHTS[impact.harmony];
    const relevance = contextRelevance[impact.harmony] ?? 1.0;
    return sum + weight * impact.impact * relevance;
  }, 0);

  return baseEig * (1 + harmonicWeight);
}

// ============================================================================
// MORAL UNCERTAINTY (Tripartite Model)
// ============================================================================

/**
 * Three dimensions of moral uncertainty
 */
export enum MoralUncertaintyType {
  /** Uncertainty about facts relevant to moral judgment */
  Epistemic = 'epistemic',
  /** Uncertainty about what values/goods matter */
  Axiological = 'axiological',
  /** Uncertainty about what action is right given values */
  Deontic = 'deontic',
}

/**
 * Tripartite moral uncertainty measurement
 *
 * Separates uncertainty into three orthogonal dimensions:
 * - Epistemic: What are the facts?
 * - Axiological: What values matter?
 * - Deontic: What should be done?
 */
export interface MoralUncertainty {
  /** Uncertainty about factual matters (0.0 to 1.0) */
  epistemic: number;
  /** Uncertainty about values/goods (0.0 to 1.0) */
  axiological: number;
  /** Uncertainty about right action (0.0 to 1.0) */
  deontic: number;
  /** When this was calculated */
  calculatedAt: number;
  /** Context in which this applies */
  context?: string;
}

/**
 * Action guidance based on moral uncertainty levels
 */
export enum MoralActionGuidance {
  /** Low uncertainty - proceed with confidence */
  ProceedConfidently = 'proceed_confidently',
  /** Moderate uncertainty - proceed with monitoring */
  ProceedWithMonitoring = 'proceed_with_monitoring',
  /** High uncertainty in one dimension - gather more info */
  PauseForReflection = 'pause_for_reflection',
  /** High uncertainty across dimensions - seek consultation */
  SeekConsultation = 'seek_consultation',
  /** Extreme uncertainty - defer action */
  DeferAction = 'defer_action',
}

/**
 * Create a new moral uncertainty measurement
 */
export function createMoralUncertainty(
  epistemic: number,
  axiological: number,
  deontic: number,
  context?: string
): MoralUncertainty {
  return {
    epistemic: Math.max(0, Math.min(1, epistemic)),
    axiological: Math.max(0, Math.min(1, axiological)),
    deontic: Math.max(0, Math.min(1, deontic)),
    calculatedAt: Date.now(),
    context,
  };
}

/**
 * Calculate total moral uncertainty (RMS of three dimensions)
 */
export function totalMoralUncertainty(mu: MoralUncertainty): number {
  const sumSquares = mu.epistemic ** 2 + mu.axiological ** 2 + mu.deontic ** 2;
  return Math.sqrt(sumSquares / 3);
}

/**
 * Determine action guidance based on uncertainty levels
 */
export function getMoralActionGuidance(mu: MoralUncertainty): MoralActionGuidance {
  const total = totalMoralUncertainty(mu);
  const maxDimension = Math.max(mu.epistemic, mu.axiological, mu.deontic);

  if (total < 0.2 && maxDimension < 0.3) {
    return MoralActionGuidance.ProceedConfidently;
  }
  if (total < 0.4 && maxDimension < 0.5) {
    return MoralActionGuidance.ProceedWithMonitoring;
  }
  if (maxDimension >= 0.7 && total < 0.6) {
    return MoralActionGuidance.PauseForReflection;
  }
  if (total >= 0.6 && total < 0.8) {
    return MoralActionGuidance.SeekConsultation;
  }
  return MoralActionGuidance.DeferAction;
}

/**
 * Check if reflection is warranted
 */
export function shouldPauseForReflection(mu: MoralUncertainty): boolean {
  const guidance = getMoralActionGuidance(mu);
  return (
    guidance === MoralActionGuidance.PauseForReflection ||
    guidance === MoralActionGuidance.SeekConsultation ||
    guidance === MoralActionGuidance.DeferAction
  );
}

/**
 * Get specific recommendations based on uncertainty profile
 */
export function getMoralRecommendations(mu: MoralUncertainty): string[] {
  const recommendations: string[] = [];

  if (mu.epistemic > 0.5) {
    recommendations.push('Gather more factual information before deciding');
    recommendations.push('Consult domain experts for empirical clarification');
  }

  if (mu.axiological > 0.5) {
    recommendations.push('Reflect on which values are most relevant here');
    recommendations.push('Consider stakeholder perspectives on what matters');
  }

  if (mu.deontic > 0.5) {
    recommendations.push('Explore multiple courses of action');
    recommendations.push('Consider reversibility and optionality');
  }

  if (recommendations.length === 0) {
    recommendations.push('Uncertainty levels acceptable for action');
  }

  return recommendations;
}

// ============================================================================
// RASHOMON PERSPECTIVES
// ============================================================================

/**
 * A perspective through one of the seven harmonic lenses
 */
export interface HarmonicPerspective {
  /** The viewing harmony */
  harmony: Harmony;
  /** Relevance of this perspective (0.0 to 1.0) */
  relevance: number;
  /** The insight from this perspective */
  insight: string;
  /** Confidence in this perspective */
  confidence: number;
  /** Points of agreement with other perspectives */
  agreements: string[];
  /** Points of disagreement (creative tension) */
  dissents: string[];
}

/**
 * Synthesized view from multiple harmonic perspectives
 */
export interface SynthesizedView {
  /** The perspectives that were considered */
  perspectives: HarmonicPerspective[];
  /** Number of perspectives considered */
  perspectivesConsidered: number;
  /** Overall confidence (consensus-weighted) */
  confidence: number;
  /** Points of strong agreement */
  consensus: string[];
  /** Points of creative tension */
  tensions: string[];
  /** Final synthesized insight */
  synthesis: string;
  /** N3 boundary status */
  n3Safe: boolean;
  /** Moral uncertainty of this view */
  moralUncertainty: MoralUncertainty;
}

// ============================================================================
// GRACEFUL RESPONSE
// ============================================================================

/**
 * Types of graceful responses to ignorance
 */
export type GracefulResponseType =
  | 'answer'           // Have the answer with confidence
  | 'deferred'         // Know answer exists but can't access now
  | 'uncertain'        // Have partial answer with explicit uncertainty
  | 'unknown'          // Don't know what we don't know
  | 'out_of_domain';   // Question outside answerable scope

/**
 * A graceful response that acknowledges ignorance
 */
export interface GracefulResponse {
  /** Response type */
  type: GracefulResponseType;
  /** Confidence level (0.0 to 1.0) */
  confidence: number;
  /** The response content */
  content?: string;
  /** Explanation of uncertainty or limitation */
  explanation?: string;
  /** Suggestions for resolution */
  suggestions?: string[];
  /** Related harmonic ignorances */
  relatedIgnorances?: string[];
  /** Whether additional context would help */
  requestsContext?: boolean;
}

// ============================================================================
// KOSMIC SONG STATE
// ============================================================================

/**
 * Harmonic profile: strengths and growth edges
 */
export interface HarmonicProfile {
  /** Current strength in each harmony (0.0 to 1.0) */
  strengths: Record<Harmony, number>;
  /** The dominant harmony */
  resonantHarmony: Harmony;
  /** Harmonies needing development */
  growthEdges: Harmony[];
  /** Overall harmonic coherence */
  coherence: number;
}

/**
 * Epistemic clearance for actions
 */
export interface EpistemicClearance {
  /** Whether action is cleared */
  cleared: boolean;
  /** Confidence in clearance */
  confidence: number;
  /** Any conditions on clearance */
  conditions: string[];
  /** Moral uncertainty assessment */
  moralUncertainty: MoralUncertainty;
  /** Harmonies affected by this action */
  affectedHarmonies: HarmonicImpact[];
}

/**
 * The Kosmic Song: Unified agent identity
 *
 * Synthesizes consciousness (Φ), value alignment (Harmonies),
 * and epistemic humility (GIS) into a coherent whole.
 */
export interface KosmicSongState {
  /** Agent identifier */
  agentId: string;

  // === Consciousness Layer ===
  /** Integrated information (Φ) value */
  phi: number;
  /** Consciousness topology type */
  topologyType: string;

  // === Harmonic Layer ===
  /** Current harmonic profile */
  harmonicProfile: HarmonicProfile;
  /** Harmonic trajectory over time */
  harmonicHistory: Array<{ timestamp: number; profile: HarmonicProfile }>;

  // === Epistemic Layer ===
  /** Active harmonic ignorances */
  activeIgnorances: HarmonicIgnorance[];
  /** Current moral uncertainty */
  moralUncertainty: MoralUncertainty;
  /** GIS statistics */
  gisStats: {
    totalDetected: number;
    resolved: number;
    averageEig: number;
  };

  // === Integration Layer ===
  /** Overall coherence score */
  coherenceScore: number;
  /** Last synthesis timestamp */
  lastSynthesis: number;
}

/**
 * Create a default harmonic profile
 */
export function createDefaultHarmonicProfile(): HarmonicProfile {
  return {
    strengths: {
      [Harmony.ResonantCoherence]: 0.5,
      [Harmony.PanSentientFlourishing]: 0.5,
      [Harmony.IntegralWisdom]: 0.5,
      [Harmony.InfinitePlay]: 0.5,
      [Harmony.UniversalInterconnectedness]: 0.5,
      [Harmony.SacredReciprocity]: 0.5,
      [Harmony.EvolutionaryProgression]: 0.5,
    },
    resonantHarmony: Harmony.IntegralWisdom,
    growthEdges: [],
    coherence: 0.5,
  };
}

/**
 * Calculate overall harmonic coherence from strengths
 *
 * Coherence is high when all harmonies are balanced and developed.
 * Extreme imbalance reduces coherence.
 */
export function calculateHarmonicCoherence(
  strengths: Record<Harmony, number>
): number {
  const values = Object.values(strengths);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;

  // Coherence = mean strength × (1 - normalized variance)
  // Higher mean and lower variance = higher coherence
  const normalizedVariance = Math.min(1, variance * 4); // Scale variance
  return mean * (1 - normalizedVariance * 0.5);
}

/**
 * Determine growth edges (weakest harmonies needing development)
 */
export function identifyGrowthEdges(
  strengths: Record<Harmony, number>,
  threshold = 0.4
): Harmony[] {
  return (Object.entries(strengths))
    .filter(([, strength]) => strength < threshold)
    .sort(([, a], [, b]) => a - b)
    .map(([harmony]) => parseInt(harmony) as Harmony);
}

/**
 * Find the resonant (dominant) harmony
 */
export function findResonantHarmony(strengths: Record<Harmony, number>): Harmony {
  let maxHarmony = Harmony.IntegralWisdom;
  let maxStrength = 0;

  for (const [harmony, strength] of Object.entries(strengths)) {
    if (strength > maxStrength) {
      maxStrength = strength;
      maxHarmony = parseInt(harmony) as Harmony;
    }
  }

  return maxHarmony;
}

// ============================================================================
// EXTENDED CLASSIFICATION (E-N-M-H)
// ============================================================================

/**
 * Extended epistemic classification with harmonic dimension
 */
export interface HarmonicEpistemicClassification {
  /** Empirical level (E0-E4) */
  empirical: number;
  /** Normative level (N0-N3) */
  normative: number;
  /** Materiality level (M0-M3) */
  materiality: number;
  /** Harmonic level (H0-H4) */
  harmonic: HarmonicLevel;
  /** Specific harmonies affected */
  affectedHarmonies?: Harmony[];
}

/**
 * Generate extended classification code (E3-N2-M2-H1)
 */
export function extendedClassificationCode(
  c: HarmonicEpistemicClassification
): string {
  return `E${c.empirical}-N${c.normative}-M${c.materiality}-H${c.harmonic}`;
}

/**
 * Parse extended classification code
 */
export function parseExtendedClassificationCode(
  code: string
): HarmonicEpistemicClassification | null {
  const match = code.match(/^E(\d)-N(\d)-M(\d)-H(\d)$/);
  if (!match) return null;

  const [, e, n, m, h] = match;
  const empirical = parseInt(e);
  const normative = parseInt(n);
  const materiality = parseInt(m);
  const harmonicNum = parseInt(h);

  // Validate ranges before casting to enums
  if (
    empirical < 0 || empirical > 4 ||
    normative < 0 || normative > 3 ||
    materiality < 0 || materiality > 3 ||
    harmonicNum < 0 || harmonicNum > 4
  ) {
    return null;
  }

  return { empirical, normative, materiality, harmonic: harmonicNum as HarmonicLevel };
}
