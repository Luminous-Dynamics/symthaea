/**
 * Temporal Consciousness System
 *
 * Enables dialogue with your past and future selves through:
 * - Belief trajectory prediction (where your thinking is headed)
 * - Past-self dialogue (converse with who you were)
 * - Worldview archaeology (excavate the layers of your evolving mind)
 */

import type { Thought } from '@mycelix/lucid-client';
import { ThoughtType } from '@mycelix/lucid-client';

// ============================================================================
// TYPES
// ============================================================================

export interface TemporalSnapshot {
  timestamp: number;
  thoughts: Thought[];
  worldviewProfile: WorldviewProfile;
  dominantThemes: string[];
  epistemicBalance: EpistemicBalance;
}

export interface WorldviewProfile {
  coreBeliefs: CoreBelief[];
  valueHierarchy: ValueRanking[];
  epistemicStyle: EpistemicStyle;
  uncertaintyTolerance: number;
  changeVelocity: number;
}

export interface CoreBelief {
  content: string;
  category: string;
  confidence: number;
  stability: number;
  firstAppeared: number;
  lastReinforced: number;
  evolutionHistory: BeliefEvolution[];
}

export interface BeliefEvolution {
  timestamp: number;
  previousContent: string;
  newContent: string;
  changeType: 'strengthened' | 'weakened' | 'modified' | 'reversed' | 'nuanced';
  trigger?: string;
}

export interface ValueRanking {
  value: string;
  rank: number;
  weight: number;
  derivedFrom: string[];
}

export interface EpistemicStyle {
  empiricalBias: number;
  normativeBias: number;
  materialBias: number;
  harmonicBias: number;
  preferredThoughtTypes: string[];
}

export interface EpistemicBalance {
  empirical: number;
  theoretical: number;
  normative: number;
  descriptive: number;
  material: number;
  conceptual: number;
}

export interface BeliefTrajectory {
  belief: CoreBelief;
  currentPosition: number;
  velocity: number;
  acceleration: number;
  predictedFuture: TrajectoryPrediction[];
  confidenceInPrediction: number;
}

export interface TrajectoryPrediction {
  timestamp: number;
  predictedConfidence: number;
  predictedContent?: string;
  probabilityOfChange: number;
  likelyDirection: 'strengthen' | 'weaken' | 'transform' | 'stable';
}

export interface PastSelfDialogue {
  pastTimestamp: number;
  pastWorldview: WorldviewProfile;
  messages: DialogueMessage[];
  insightsGained: string[];
}

export interface DialogueMessage {
  speaker: 'present' | 'past';
  content: string;
  timestamp: number;
  emotionalTone?: string;
}

export interface WorldviewLayer {
  periodStart: number;
  periodEnd: number;
  label: string;
  dominantBeliefs: CoreBelief[];
  significantShifts: BeliefShift[];
  influentialSources: string[];
  overallTone: string;
}

export interface BeliefShift {
  from: string;
  to: string;
  domain: string;
  magnitude: number;
  timestamp: number;
}

export interface ArchaeologyReport {
  layers: WorldviewLayer[];
  majorTransitions: Transition[];
  persistentBeliefs: CoreBelief[];
  abandonedBeliefs: CoreBelief[];
  emergentPatterns: string[];
}

export interface Transition {
  timestamp: number;
  description: string;
  beforeSnapshot: Partial<WorldviewProfile>;
  afterSnapshot: Partial<WorldviewProfile>;
  catalysts: string[];
}

// ============================================================================
// TEMPORAL SNAPSHOT CREATION
// ============================================================================

/**
 * Create a temporal snapshot from a collection of thoughts
 */
export function createTemporalSnapshot(thoughts: Thought[], timestamp?: number): TemporalSnapshot {
  const profile = extractWorldviewProfile(thoughts);
  const themes = extractDominantThemes(thoughts);
  const balance = calculateEpistemicBalance(thoughts);

  return {
    timestamp: timestamp || Date.now(),
    thoughts,
    worldviewProfile: profile,
    dominantThemes: themes,
    epistemicBalance: balance,
  };
}

function extractWorldviewProfile(thoughts: Thought[]): WorldviewProfile {
  const coreBeliefs = extractCoreBeliefs(thoughts);
  const valueHierarchy = extractValueHierarchy(thoughts);
  const epistemicStyle = calculateEpistemicStyle(thoughts);
  const uncertaintyTolerance = calculateUncertaintyTolerance(thoughts);
  const changeVelocity = calculateChangeVelocity(thoughts);

  return {
    coreBeliefs,
    valueHierarchy,
    epistemicStyle,
    uncertaintyTolerance,
    changeVelocity,
  };
}

function extractCoreBeliefs(thoughts: Thought[]): CoreBelief[] {
  const beliefClusters = new Map<string, Thought[]>();

  // Cluster thoughts by semantic similarity (simplified: by tags)
  for (const thought of thoughts) {
    if (thought.thought_type !== ThoughtType.Claim && thought.thought_type !== ThoughtType.Insight) continue;

    for (const tag of thought.tags || []) {
      if (!beliefClusters.has(tag)) beliefClusters.set(tag, []);
      beliefClusters.get(tag)!.push(thought);
    }
  }

  const coreBeliefs: CoreBelief[] = [];

  for (const [category, cluster] of beliefClusters) {
    if (cluster.length < 2) continue;

    // Sort by creation time
    cluster.sort((a, b) => (a.created_at || 0) - (b.created_at || 0));

    const avgConfidence = cluster.reduce((sum, t) => sum + t.confidence, 0) / cluster.length;
    const stability = calculateBeliefStability(cluster);

    coreBeliefs.push({
      content: cluster[cluster.length - 1].content.slice(0, 200),
      category,
      confidence: avgConfidence,
      stability,
      firstAppeared: cluster[0].created_at || Date.now(),
      lastReinforced: cluster[cluster.length - 1].created_at || Date.now(),
      evolutionHistory: trackBeliefEvolution(cluster),
    });
  }

  return coreBeliefs.sort((a, b) => b.confidence * b.stability - a.confidence * a.stability);
}

function calculateBeliefStability(thoughts: Thought[]): number {
  if (thoughts.length < 2) return 1;

  // Measure consistency of confidence over time
  let totalVariance = 0;
  for (let i = 1; i < thoughts.length; i++) {
    totalVariance += Math.abs(thoughts[i].confidence - thoughts[i - 1].confidence);
  }

  const avgVariance = totalVariance / (thoughts.length - 1);
  return Math.max(0, 1 - avgVariance);
}

function trackBeliefEvolution(thoughts: Thought[]): BeliefEvolution[] {
  const evolution: BeliefEvolution[] = [];

  for (let i = 1; i < thoughts.length; i++) {
    const prev = thoughts[i - 1];
    const curr = thoughts[i];
    const confDiff = curr.confidence - prev.confidence;

    let changeType: BeliefEvolution['changeType'];
    if (Math.abs(confDiff) < 0.1) {
      changeType = 'nuanced';
    } else if (confDiff > 0.2) {
      changeType = 'strengthened';
    } else if (confDiff < -0.2) {
      changeType = 'weakened';
    } else {
      changeType = 'modified';
    }

    evolution.push({
      timestamp: curr.created_at || Date.now(),
      previousContent: prev.content.slice(0, 100),
      newContent: curr.content.slice(0, 100),
      changeType,
    });
  }

  return evolution;
}

function extractValueHierarchy(thoughts: Thought[]): ValueRanking[] {
  const valueWords: Record<string, string[]> = {
    truth: ['truth', 'honest', 'accurate', 'factual', 'real'],
    freedom: ['freedom', 'liberty', 'autonomy', 'choice', 'independent'],
    justice: ['justice', 'fair', 'equal', 'rights', 'moral'],
    beauty: ['beauty', 'aesthetic', 'elegant', 'harmonious', 'art'],
    knowledge: ['knowledge', 'understanding', 'wisdom', 'learn', 'insight'],
    love: ['love', 'compassion', 'care', 'empathy', 'connection'],
    power: ['power', 'influence', 'control', 'strength', 'ability'],
    security: ['security', 'safety', 'stability', 'protect', 'secure'],
    growth: ['growth', 'progress', 'improve', 'develop', 'evolve'],
    creativity: ['creativity', 'create', 'innovate', 'original', 'novel'],
  };

  const valueCounts: Record<string, number> = {};
  const valueDerivations: Record<string, string[]> = {};

  for (const thought of thoughts) {
    const content = thought.content.toLowerCase();
    for (const [value, keywords] of Object.entries(valueWords)) {
      const matches = keywords.filter((kw) => content.includes(kw));
      if (matches.length > 0) {
        valueCounts[value] = (valueCounts[value] || 0) + matches.length * thought.confidence;
        if (!valueDerivations[value]) valueDerivations[value] = [];
        valueDerivations[value].push(thought.content.slice(0, 50));
      }
    }
  }

  const totalWeight = Object.values(valueCounts).reduce((a, b) => a + b, 0) || 1;

  return Object.entries(valueCounts)
    .map(([value, count], i) => ({
      value,
      rank: i + 1,
      weight: count / totalWeight,
      derivedFrom: (valueDerivations[value] || []).slice(0, 3),
    }))
    .sort((a, b) => b.weight - a.weight)
    .map((v, i) => ({ ...v, rank: i + 1 }));
}

function calculateEpistemicStyle(thoughts: Thought[]): EpistemicStyle {
  // Track scores for each dimension (E0=0, E1=1, E2=2, etc.)
  let empiricalSum = 0;
  let normativeSum = 0;
  let materialitySum = 0;
  let harmonicSum = 0;
  const typeCounts: Record<string, number> = {};

  for (const thought of thoughts) {
    // Extract numeric level from enum values (e.g., 'E2' -> 2)
    const empiricalLevel = parseInt(thought.epistemic.empirical.slice(1)) || 0;
    const normativeLevel = parseInt(thought.epistemic.normative.slice(1)) || 0;
    const materialityLevel = parseInt(thought.epistemic.materiality.slice(1)) || 0;
    const harmonicLevel = parseInt(thought.epistemic.harmonic.slice(1)) || 0;

    empiricalSum += empiricalLevel;
    normativeSum += normativeLevel;
    materialitySum += materialityLevel;
    harmonicSum += harmonicLevel;
    typeCounts[thought.thought_type] = (typeCounts[thought.thought_type] || 0) + 1;
  }

  const total = thoughts.length || 1;

  return {
    // Normalize biases (E has max 4, N has max 3, M has max 3, H has max 4)
    empiricalBias: empiricalSum / (total * 4),
    normativeBias: normativeSum / (total * 3),
    materialBias: materialitySum / (total * 3),
    harmonicBias: harmonicSum / (total * 4),
    preferredThoughtTypes: Object.entries(typeCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([type]) => type),
  };
}

function calculateUncertaintyTolerance(thoughts: Thought[]): number {
  if (thoughts.length === 0) return 0.5;

  // Higher tolerance = more thoughts with moderate confidence
  const moderateConfidence = thoughts.filter(
    (t) => t.confidence >= 0.3 && t.confidence <= 0.7
  ).length;

  return moderateConfidence / thoughts.length;
}

function calculateChangeVelocity(thoughts: Thought[]): number {
  if (thoughts.length < 5) return 0;

  // Measure how quickly beliefs evolve
  const sorted = [...thoughts].sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
  const recentHalf = sorted.slice(Math.floor(sorted.length / 2));
  const olderHalf = sorted.slice(0, Math.floor(sorted.length / 2));

  const recentAvgConf = recentHalf.reduce((s, t) => s + t.confidence, 0) / recentHalf.length;
  const olderAvgConf = olderHalf.reduce((s, t) => s + t.confidence, 0) / olderHalf.length;

  return Math.abs(recentAvgConf - olderAvgConf);
}

function extractDominantThemes(thoughts: Thought[]): string[] {
  const tagCounts = new Map<string, number>();

  for (const thought of thoughts) {
    for (const tag of thought.tags || []) {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    }
  }

  return [...tagCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([tag]) => tag);
}

function calculateEpistemicBalance(thoughts: Thought[]): EpistemicBalance {
  const balance = {
    empirical: 0,
    theoretical: 0,
    normative: 0,
    descriptive: 0,
    material: 0,
    conceptual: 0,
  };

  const total = thoughts.length || 1;

  for (const thought of thoughts) {
    // High empirical (E2+) = empirical, low = theoretical
    const empiricalLevel = parseInt(thought.epistemic.empirical.slice(1)) || 0;
    if (empiricalLevel >= 2) balance.empirical++;
    else balance.theoretical++;

    // High normative (N2+) = normative, low = descriptive
    const normativeLevel = parseInt(thought.epistemic.normative.slice(1)) || 0;
    if (normativeLevel >= 2) balance.normative++;
    else balance.descriptive++;

    // High materiality (M2+) = material, low = conceptual
    const materialityLevel = parseInt(thought.epistemic.materiality.slice(1)) || 0;
    if (materialityLevel >= 2) balance.material++;
    else balance.conceptual++;
  }

  return {
    empirical: balance.empirical / total,
    theoretical: balance.theoretical / total,
    normative: balance.normative / total,
    descriptive: balance.descriptive / total,
    material: balance.material / total,
    conceptual: balance.conceptual / total,
  };
}

// ============================================================================
// BELIEF TRAJECTORY PREDICTION
// ============================================================================

/**
 * Predict where a belief is headed based on its evolution
 */
export function predictBeliefTrajectory(
  belief: CoreBelief,
  horizon: number = 30 * 24 * 60 * 60 * 1000 // 30 days default
): BeliefTrajectory {
  const history = belief.evolutionHistory;

  // Calculate velocity (rate of confidence change)
  let velocity = 0;
  let acceleration = 0;

  if (history.length >= 2) {
    const recentChanges = history.slice(-3);
    const confChanges = recentChanges.map((e) => {
      switch (e.changeType) {
        case 'strengthened': return 0.2;
        case 'weakened': return -0.2;
        case 'modified': return 0.05;
        case 'nuanced': return 0.02;
        case 'reversed': return -0.5;
        default: return 0;
      }
    });

    velocity = confChanges.reduce<number>((a, b) => a + b, 0) / confChanges.length;

    if (history.length >= 3) {
      const oldVelocity = confChanges.slice(0, -1).reduce<number>((a, b) => a + b, 0) / (confChanges.length - 1);
      acceleration = velocity - oldVelocity;
    }
  }

  // Generate predictions
  const predictions: TrajectoryPrediction[] = [];
  const intervals = 5;
  const intervalMs = horizon / intervals;

  for (let i = 1; i <= intervals; i++) {
    const futureTime = Date.now() + intervalMs * i;
    const projectedConfidence = Math.max(0, Math.min(1,
      belief.confidence + velocity * i + acceleration * i * 0.5
    ));

    const probabilityOfChange = Math.min(1, Math.abs(velocity) * 3 + Math.abs(acceleration) * 5);

    let direction: TrajectoryPrediction['likelyDirection'] = 'stable';
    if (velocity > 0.1) direction = 'strengthen';
    else if (velocity < -0.1) direction = 'weaken';
    else if (Math.abs(acceleration) > 0.05) direction = 'transform';

    predictions.push({
      timestamp: futureTime,
      predictedConfidence: projectedConfidence,
      probabilityOfChange,
      likelyDirection: direction,
    });
  }

  // Confidence in prediction decreases with instability
  const confidenceInPrediction = Math.max(0.1, belief.stability * (1 - Math.abs(acceleration) * 2));

  return {
    belief,
    currentPosition: belief.confidence,
    velocity,
    acceleration,
    predictedFuture: predictions,
    confidenceInPrediction,
  };
}

/**
 * Predict trajectories for all core beliefs
 */
export function predictAllTrajectories(
  profile: WorldviewProfile,
  horizon?: number
): BeliefTrajectory[] {
  return profile.coreBeliefs.map((belief) => predictBeliefTrajectory(belief, horizon));
}

// ============================================================================
// PAST-SELF DIALOGUE
// ============================================================================

/**
 * Initiate a dialogue with your past self
 */
export function initiatePastSelfDialogue(
  currentSnapshot: TemporalSnapshot,
  pastSnapshot: TemporalSnapshot
): PastSelfDialogue {
  return {
    pastTimestamp: pastSnapshot.timestamp,
    pastWorldview: pastSnapshot.worldviewProfile,
    messages: [],
    insightsGained: [],
  };
}

/**
 * Generate a past-self response based on their worldview
 */
export function generatePastSelfResponse(
  dialogue: PastSelfDialogue,
  presentMessage: string
): string {
  const past = dialogue.pastWorldview;

  // Add the present message
  dialogue.messages.push({
    speaker: 'present',
    content: presentMessage,
    timestamp: Date.now(),
  });

  // Generate response based on past worldview
  const response = synthesizePastResponse(presentMessage, past);

  dialogue.messages.push({
    speaker: 'past',
    content: response,
    timestamp: Date.now(),
    emotionalTone: inferEmotionalTone(past),
  });

  return response;
}

function synthesizePastResponse(message: string, pastWorldview: WorldviewProfile): string {
  const msgLower = message.toLowerCase();
  const topValues = pastWorldview.valueHierarchy.slice(0, 3).map((v) => v.value);
  const coreBeliefs = pastWorldview.coreBeliefs.slice(0, 3);
  const style = pastWorldview.epistemicStyle;

  let response = '';

  // React based on epistemic style
  if (style.empiricalBias > 0.6) {
    response += 'I would need to see evidence for that. ';
  } else if (style.normativeBias > 0.6) {
    response += 'But is that the right thing to do? ';
  }

  // Reference core beliefs if relevant
  for (const belief of coreBeliefs) {
    if (msgLower.includes(belief.category.toLowerCase())) {
      response += `Back then, I believed: "${belief.content.slice(0, 80)}..." `;
      break;
    }
  }

  // React based on values
  if (topValues.includes('truth') && msgLower.includes('believe')) {
    response += 'I was always searching for what was really true. ';
  }
  if (topValues.includes('growth') && msgLower.includes('change')) {
    response += 'I was open to growth, but only if it felt authentic. ';
  }

  // Default responses based on uncertainty tolerance
  if (pastWorldview.uncertaintyTolerance > 0.6) {
    response += 'I was comfortable not knowing everything. What mattered was the journey of understanding.';
  } else {
    response += 'I wanted clear answers. Uncertainty was uncomfortable.';
  }

  return response || 'I wonder what I would have thought about that...';
}

function inferEmotionalTone(worldview: WorldviewProfile): string {
  if (worldview.uncertaintyTolerance > 0.7) return 'curious';
  if (worldview.changeVelocity > 0.3) return 'searching';
  if (worldview.epistemicStyle.normativeBias > 0.6) return 'principled';
  if (worldview.epistemicStyle.empiricalBias > 0.6) return 'analytical';
  return 'reflective';
}

/**
 * Extract insights from the dialogue
 */
export function extractDialogueInsights(dialogue: PastSelfDialogue): string[] {
  const insights: string[] = [];
  const past = dialogue.pastWorldview;

  // Compare values
  const pastTopValues = past.valueHierarchy.slice(0, 3).map((v) => v.value);
  insights.push(`Your past self prioritized: ${pastTopValues.join(', ')}`);

  // Note epistemic style
  if (past.epistemicStyle.empiricalBias > 0.6) {
    insights.push('You were highly empirically-minded, requiring evidence for beliefs');
  }
  if (past.epistemicStyle.normativeBias > 0.6) {
    insights.push('You were strongly values-driven, always asking "should" questions');
  }

  // Note uncertainty tolerance
  if (past.uncertaintyTolerance > 0.6) {
    insights.push('You were comfortable with ambiguity and uncertainty');
  } else if (past.uncertaintyTolerance < 0.3) {
    insights.push('You sought certainty and clear answers');
  }

  // Note core beliefs
  for (const belief of past.coreBeliefs.slice(0, 2)) {
    insights.push(`Core belief in ${belief.category}: "${belief.content.slice(0, 60)}..."`);
  }

  dialogue.insightsGained = insights;
  return insights;
}

// ============================================================================
// WORLDVIEW ARCHAEOLOGY
// ============================================================================

/**
 * Excavate the layers of worldview evolution over time
 */
export function excavateWorldview(
  thoughtsByPeriod: Map<string, Thought[]>
): ArchaeologyReport {
  const layers: WorldviewLayer[] = [];
  const allSnapshots: TemporalSnapshot[] = [];

  // Create layers for each period
  for (const [period, thoughts] of thoughtsByPeriod) {
    if (thoughts.length === 0) continue;

    const timestamps = thoughts.map((t) => t.created_at || Date.now());
    const periodStart = Math.min(...timestamps);
    const periodEnd = Math.max(...timestamps);

    const snapshot = createTemporalSnapshot(thoughts, periodEnd);
    allSnapshots.push(snapshot);

    layers.push({
      periodStart,
      periodEnd,
      label: period,
      dominantBeliefs: snapshot.worldviewProfile.coreBeliefs.slice(0, 5),
      significantShifts: [], // Will be filled when comparing layers
      influentialSources: extractSources(thoughts),
      overallTone: determineOverallTone(snapshot),
    });
  }

  // Sort layers by time
  layers.sort((a, b) => a.periodStart - b.periodStart);

  // Find shifts between layers
  for (let i = 1; i < layers.length; i++) {
    const prevLayer = layers[i - 1];
    const currLayer = layers[i];
    currLayer.significantShifts = findShifts(prevLayer, currLayer);
  }

  // Identify transitions
  const transitions = identifyTransitions(layers);

  // Find persistent and abandoned beliefs
  const { persistent, abandoned } = categorizeBeliefs(layers);

  // Identify emergent patterns
  const patterns = identifyEmergentPatterns(layers);

  return {
    layers,
    majorTransitions: transitions,
    persistentBeliefs: persistent,
    abandonedBeliefs: abandoned,
    emergentPatterns: patterns,
  };
}

function extractSources(thoughts: Thought[]): string[] {
  const sources: string[] = [];

  for (const thought of thoughts) {
    if (thought.source_hashes && thought.source_hashes.length > 0) {
      sources.push(`${thought.source_hashes.length} linked source(s)`);
    }
  }

  return [...new Set(sources)].slice(0, 5);
}

function determineOverallTone(snapshot: TemporalSnapshot): string {
  const balance = snapshot.epistemicBalance;

  if (balance.empirical > 0.6) return 'analytical';
  if (balance.normative > 0.6) return 'evaluative';
  if (snapshot.worldviewProfile.uncertaintyTolerance > 0.6) return 'exploratory';
  if (snapshot.worldviewProfile.changeVelocity > 0.3) return 'transformative';

  return 'contemplative';
}

function findShifts(prev: WorldviewLayer, curr: WorldviewLayer): BeliefShift[] {
  const shifts: BeliefShift[] = [];

  // Compare dominant beliefs
  const prevCategories = new Set(prev.dominantBeliefs.map((b) => b.category));
  const currCategories = new Set(curr.dominantBeliefs.map((b) => b.category));

  // Find disappeared categories
  for (const cat of prevCategories) {
    if (!currCategories.has(cat)) {
      const prevBelief = prev.dominantBeliefs.find((b) => b.category === cat);
      if (prevBelief) {
        shifts.push({
          from: prevBelief.content.slice(0, 50),
          to: '(belief diminished)',
          domain: cat,
          magnitude: prevBelief.confidence,
          timestamp: curr.periodStart,
        });
      }
    }
  }

  // Find new categories
  for (const cat of currCategories) {
    if (!prevCategories.has(cat)) {
      const currBelief = curr.dominantBeliefs.find((b) => b.category === cat);
      if (currBelief) {
        shifts.push({
          from: '(new belief)',
          to: currBelief.content.slice(0, 50),
          domain: cat,
          magnitude: currBelief.confidence,
          timestamp: curr.periodStart,
        });
      }
    }
  }

  // Find changed beliefs in same category
  for (const cat of prevCategories) {
    if (currCategories.has(cat)) {
      const prevBelief = prev.dominantBeliefs.find((b) => b.category === cat);
      const currBelief = curr.dominantBeliefs.find((b) => b.category === cat);

      if (prevBelief && currBelief) {
        const confDiff = Math.abs(currBelief.confidence - prevBelief.confidence);
        if (confDiff > 0.2) {
          shifts.push({
            from: `${prevBelief.content.slice(0, 30)} (${Math.round(prevBelief.confidence * 100)}%)`,
            to: `${currBelief.content.slice(0, 30)} (${Math.round(currBelief.confidence * 100)}%)`,
            domain: cat,
            magnitude: confDiff,
            timestamp: curr.periodStart,
          });
        }
      }
    }
  }

  return shifts.sort((a, b) => b.magnitude - a.magnitude);
}

function identifyTransitions(layers: WorldviewLayer[]): Transition[] {
  const transitions: Transition[] = [];

  for (let i = 1; i < layers.length; i++) {
    const prev = layers[i - 1];
    const curr = layers[i];

    if (curr.significantShifts.length >= 2) {
      transitions.push({
        timestamp: curr.periodStart,
        description: `Transition from ${prev.label} to ${curr.label}`,
        beforeSnapshot: { coreBeliefs: prev.dominantBeliefs },
        afterSnapshot: { coreBeliefs: curr.dominantBeliefs },
        catalysts: curr.significantShifts.map((s) => `${s.domain}: ${s.from} → ${s.to}`),
      });
    }
  }

  return transitions;
}

function categorizeBeliefs(layers: WorldviewLayer[]): {
  persistent: CoreBelief[];
  abandoned: CoreBelief[];
} {
  if (layers.length < 2) {
    return { persistent: [], abandoned: [] };
  }

  const firstLayer = layers[0];
  const lastLayer = layers[layers.length - 1];

  const firstCategories = new Set(firstLayer.dominantBeliefs.map((b) => b.category));
  const lastCategories = new Set(lastLayer.dominantBeliefs.map((b) => b.category));

  const persistent = lastLayer.dominantBeliefs.filter((b) => firstCategories.has(b.category));
  const abandoned = firstLayer.dominantBeliefs.filter((b) => !lastCategories.has(b.category));

  return { persistent, abandoned };
}

function identifyEmergentPatterns(layers: WorldviewLayer[]): string[] {
  const patterns: string[] = [];

  // Track tone evolution
  const tones = layers.map((l) => l.overallTone);
  if (new Set(tones).size === 1) {
    patterns.push(`Consistent ${tones[0]} tone across all periods`);
  } else {
    patterns.push(`Tone evolution: ${tones.join(' → ')}`);
  }

  // Track belief stability
  const avgShifts = layers.reduce((sum, l) => sum + l.significantShifts.length, 0) / layers.length;
  if (avgShifts < 1) {
    patterns.push('Highly stable worldview with few major shifts');
  } else if (avgShifts > 3) {
    patterns.push('Dynamic worldview with frequent re-evaluation');
  }

  // Track domain focus
  const allDomains = layers.flatMap((l) => l.dominantBeliefs.map((b) => b.category));
  const domainCounts = new Map<string, number>();
  for (const d of allDomains) {
    domainCounts.set(d, (domainCounts.get(d) || 0) + 1);
  }

  const topDomains = [...domainCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([d]) => d);

  if (topDomains.length > 0) {
    patterns.push(`Persistent focus on: ${topDomains.join(', ')}`);
  }

  return patterns;
}

// ============================================================================
// TIME TRAVEL UTILITIES
// ============================================================================

/**
 * Group thoughts by time period
 */
export function groupThoughtsByPeriod(
  thoughts: Thought[],
  periodType: 'week' | 'month' | 'quarter' | 'year'
): Map<string, Thought[]> {
  const periods = new Map<string, Thought[]>();

  for (const thought of thoughts) {
    const timestamp = thought.created_at || Date.now();
    const periodKey = getPeriodKey(timestamp, periodType);

    if (!periods.has(periodKey)) {
      periods.set(periodKey, []);
    }
    periods.get(periodKey)!.push(thought);
  }

  return periods;
}

function getPeriodKey(timestamp: number, periodType: string): string {
  const date = new Date(timestamp);
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const week = Math.ceil(date.getDate() / 7);

  switch (periodType) {
    case 'week':
      return `${year}-W${month.toString().padStart(2, '0')}-${week}`;
    case 'month':
      return `${year}-${month.toString().padStart(2, '0')}`;
    case 'quarter':
      return `${year}-Q${Math.ceil(month / 3)}`;
    case 'year':
      return `${year}`;
    default:
      return `${year}-${month.toString().padStart(2, '0')}`;
  }
}

/**
 * Compare two snapshots and return key differences
 */
export function compareSnapshots(
  snapshot1: TemporalSnapshot,
  snapshot2: TemporalSnapshot
): string[] {
  const differences: string[] = [];

  // Compare themes
  const themes1 = new Set(snapshot1.dominantThemes);
  const themes2 = new Set(snapshot2.dominantThemes);

  const newThemes = [...themes2].filter((t) => !themes1.has(t));
  const lostThemes = [...themes1].filter((t) => !themes2.has(t));

  if (newThemes.length > 0) {
    differences.push(`New themes: ${newThemes.join(', ')}`);
  }
  if (lostThemes.length > 0) {
    differences.push(`Diminished themes: ${lostThemes.join(', ')}`);
  }

  // Compare epistemic balance
  const b1 = snapshot1.epistemicBalance;
  const b2 = snapshot2.epistemicBalance;

  if (Math.abs(b1.empirical - b2.empirical) > 0.2) {
    differences.push(
      b2.empirical > b1.empirical
        ? 'Became more empirically-focused'
        : 'Became more theoretically-focused'
    );
  }

  if (Math.abs(b1.normative - b2.normative) > 0.2) {
    differences.push(
      b2.normative > b1.normative
        ? 'Became more evaluative'
        : 'Became more descriptive'
    );
  }

  // Compare uncertainty tolerance
  const tol1 = snapshot1.worldviewProfile.uncertaintyTolerance;
  const tol2 = snapshot2.worldviewProfile.uncertaintyTolerance;

  if (Math.abs(tol1 - tol2) > 0.2) {
    differences.push(
      tol2 > tol1
        ? 'Increased comfort with uncertainty'
        : 'Increased desire for certainty'
    );
  }

  return differences.length > 0 ? differences : ['No significant differences detected'];
}
