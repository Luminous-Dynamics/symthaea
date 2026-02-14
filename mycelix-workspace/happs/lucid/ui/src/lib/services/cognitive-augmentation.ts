/**
 * Cognitive Augmentation Engine
 *
 * Enhances human reasoning by:
 * - Generating steelman arguments (strongest counter-positions)
 * - Detecting blind spots in thinking
 * - Alerting to cognitive biases in real-time
 * - Simulating alternative perspectives
 */

import type { Thought } from '@mycelix/lucid-client';
import { EmpiricalLevel } from '@mycelix/lucid-client';

// ============================================================================
// TYPES
// ============================================================================

export interface SteelmanArgument {
  originalPosition: string;
  steelmanVersion: string;
  counterArguments: CounterArgument[];
  strengthScore: number;
  generatedAt: number;
}

export interface CounterArgument {
  argument: string;
  type: CounterArgumentType;
  strength: number;
  sourceFramework?: string;
}

export type CounterArgumentType =
  | 'logical'
  | 'empirical'
  | 'moral'
  | 'practical'
  | 'definitional'
  | 'precedent';

export interface BlindSpot {
  domain: string;
  description: string;
  suggestedExplorations: string[];
  detectedAt: number;
  severity: 'low' | 'medium' | 'high';
}

export interface CognitiveBias {
  biasType: BiasType;
  description: string;
  triggeredBy: string;
  debiasingSuggestion: string;
  confidence: number;
  detectedAt: number;
}

export type BiasType =
  | 'confirmation'
  | 'anchoring'
  | 'availability'
  | 'bandwagon'
  | 'dunning_kruger'
  | 'sunk_cost'
  | 'hindsight'
  | 'optimism'
  | 'pessimism'
  | 'framing'
  | 'attribution'
  | 'halo_effect'
  | 'in_group'
  | 'authority'
  | 'recency';

export interface PerspectiveSimulation {
  perspectiveName: string;
  perspectiveType: PerspectiveType;
  worldview: WorldviewSnapshot;
  response: string;
  keyDifferences: string[];
  generatedAt: number;
}

export type PerspectiveType =
  | 'philosopher'
  | 'scientist'
  | 'artist'
  | 'pragmatist'
  | 'skeptic'
  | 'optimist'
  | 'devil_advocate'
  | 'custom';

export interface WorldviewSnapshot {
  coreValues: string[];
  assumptions: string[];
  methodology: string;
  priors: Record<string, number>;
}

export interface AugmentationReport {
  originalThought: Thought;
  steelman?: SteelmanArgument;
  blindSpots: BlindSpot[];
  biases: CognitiveBias[];
  perspectives: PerspectiveSimulation[];
  overallScore: number;
  recommendations: string[];
}

// ============================================================================
// STEELMAN GENERATOR
// ============================================================================

/**
 * Generate the strongest possible version of an opposing argument
 */
export async function generateSteelman(
  thought: Thought,
  useLLM: boolean = false
): Promise<SteelmanArgument> {
  const content = thought.content.toLowerCase();

  if (useLLM) {
    return generateSteelmanWithLLM(thought);
  }

  // Heuristic-based steelman generation
  const counterArguments: CounterArgument[] = [];

  // Identify the position type and generate appropriate counters
  const positionAnalysis = analyzePosition(content);

  for (const counterType of positionAnalysis.vulnerabilities) {
    const counter = generateCounterForType(content, counterType);
    if (counter) {
      counterArguments.push(counter);
    }
  }

  // Generate steelman version
  const steelmanVersion = constructSteelmanVersion(content, counterArguments);

  return {
    originalPosition: thought.content,
    steelmanVersion,
    counterArguments,
    strengthScore: calculateStrengthScore(counterArguments),
    generatedAt: Date.now(),
  };
}

interface PositionAnalysis {
  type: 'claim' | 'value' | 'prediction' | 'recommendation';
  vulnerabilities: CounterArgumentType[];
  certaintyLevel: number;
}

function analyzePosition(content: string): PositionAnalysis {
  const vulnerabilities: CounterArgumentType[] = [];

  // Check for absolute claims
  if (/\b(always|never|all|none|every|no one)\b/i.test(content)) {
    vulnerabilities.push('logical', 'empirical');
  }

  // Check for value judgments
  if (/\b(should|must|ought|better|worse|right|wrong)\b/i.test(content)) {
    vulnerabilities.push('moral', 'definitional');
  }

  // Check for predictions
  if (/\b(will|going to|inevitably|certainly)\b/i.test(content)) {
    vulnerabilities.push('empirical', 'practical');
  }

  // Check for causal claims
  if (/\b(because|causes|leads to|results in)\b/i.test(content)) {
    vulnerabilities.push('logical', 'empirical');
  }

  // Default vulnerabilities
  if (vulnerabilities.length === 0) {
    vulnerabilities.push('logical', 'practical');
  }

  return {
    type: determinePositionType(content),
    vulnerabilities,
    certaintyLevel: estimateCertainty(content),
  };
}

function determinePositionType(content: string): PositionAnalysis['type'] {
  if (/\b(should|must|recommend|suggest)\b/i.test(content)) return 'recommendation';
  if (/\b(will|going to|predict|expect)\b/i.test(content)) return 'prediction';
  if (/\b(good|bad|better|worse|right|wrong|moral)\b/i.test(content)) return 'value';
  return 'claim';
}

function estimateCertainty(content: string): number {
  let certainty = 0.5;

  // High certainty markers
  if (/\b(definitely|certainly|absolutely|clearly|obviously)\b/i.test(content)) {
    certainty += 0.3;
  }

  // Low certainty markers
  if (/\b(maybe|perhaps|might|possibly|probably)\b/i.test(content)) {
    certainty -= 0.2;
  }

  return Math.max(0, Math.min(1, certainty));
}

function generateCounterForType(content: string, type: CounterArgumentType): CounterArgument | null {
  const counters: Record<CounterArgumentType, () => CounterArgument> = {
    logical: () => ({
      argument: 'The conclusion doesn\'t necessarily follow from the premises. Consider alternative explanations or missing logical steps.',
      type: 'logical',
      strength: 0.7,
      sourceFramework: 'Classical Logic',
    }),
    empirical: () => ({
      argument: 'This claim would benefit from empirical evidence. What data would confirm or refute this position?',
      type: 'empirical',
      strength: 0.8,
      sourceFramework: 'Scientific Method',
    }),
    moral: () => ({
      argument: 'Different ethical frameworks might evaluate this differently. Consider consequentialist, deontological, and virtue ethics perspectives.',
      type: 'moral',
      strength: 0.6,
      sourceFramework: 'Moral Philosophy',
    }),
    practical: () => ({
      argument: 'Even if theoretically sound, implementation challenges may arise. Consider second-order effects and unintended consequences.',
      type: 'practical',
      strength: 0.7,
      sourceFramework: 'Systems Thinking',
    }),
    definitional: () => ({
      argument: 'Key terms may be interpreted differently. Clarifying definitions might reveal the disagreement is partly semantic.',
      type: 'definitional',
      strength: 0.5,
      sourceFramework: 'Analytic Philosophy',
    }),
    precedent: () => ({
      argument: 'Historical precedents may support or challenge this position. What can we learn from similar situations?',
      type: 'precedent',
      strength: 0.6,
      sourceFramework: 'Historical Analysis',
    }),
  };

  return counters[type]?.() ?? null;
}

function constructSteelmanVersion(content: string, counters: CounterArgument[]): string {
  const mainCounters = counters.slice(0, 3).map((c) => c.argument);

  return `The strongest version of the opposing view would acknowledge: ${mainCounters.join(' Additionally, ')} A thoughtful opponent would argue these points while recognizing the valid concerns in the original position.`;
}

function calculateStrengthScore(counters: CounterArgument[]): number {
  if (counters.length === 0) return 0;
  return counters.reduce((sum, c) => sum + c.strength, 0) / counters.length;
}

async function generateSteelmanWithLLM(thought: Thought): Promise<SteelmanArgument> {
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.2',
        prompt: `You are a philosopher skilled at steelmanning arguments. Given this position:

"${thought.content}"

Generate:
1. The strongest possible version of an opposing argument
2. 3 specific counter-arguments with their types (logical, empirical, moral, practical)
3. A strength score from 0-1

Respond in JSON format:
{
  "steelmanVersion": "...",
  "counterArguments": [{"argument": "...", "type": "...", "strength": 0.X}],
  "strengthScore": 0.X
}`,
        stream: false,
      }),
    });

    const data = await response.json();
    const parsed = JSON.parse(data.response);

    return {
      originalPosition: thought.content,
      steelmanVersion: parsed.steelmanVersion,
      counterArguments: parsed.counterArguments,
      strengthScore: parsed.strengthScore,
      generatedAt: Date.now(),
    };
  } catch {
    // Fallback to heuristic
    return generateSteelman(thought, false);
  }
}

// ============================================================================
// BLIND SPOT DETECTOR
// ============================================================================

/**
 * Detect areas the user hasn't explored or considered
 */
export function detectBlindSpots(thoughts: Thought[]): BlindSpot[] {
  const blindSpots: BlindSpot[] = [];

  // Analyze domain coverage
  const domainCoverage = analyzeDomainCoverage(thoughts);

  // Check for missing counter-considerations
  const missingConsiderations = findMissingConsiderations(thoughts);

  // Check for temporal blind spots
  const temporalGaps = findTemporalGaps(thoughts);

  // Check for perspective blind spots
  const perspectiveGaps = findPerspectiveGaps(thoughts);

  // Combine all blind spots
  blindSpots.push(...domainCoverage, ...missingConsiderations, ...temporalGaps, ...perspectiveGaps);

  return blindSpots.sort((a, b) => {
    const severityOrder = { high: 0, medium: 1, low: 2 };
    return severityOrder[a.severity] - severityOrder[b.severity];
  });
}

function analyzeDomainCoverage(thoughts: Thought[]): BlindSpot[] {
  const domains = [
    'ethics',
    'economics',
    'psychology',
    'technology',
    'environment',
    'politics',
    'science',
    'culture',
    'history',
    'philosophy',
  ];

  const coveredDomains = new Set<string>();
  const domainKeywords: Record<string, string[]> = {
    ethics: ['moral', 'ethical', 'right', 'wrong', 'should', 'justice'],
    economics: ['cost', 'price', 'market', 'money', 'trade', 'value', 'economy'],
    psychology: ['mind', 'behavior', 'emotion', 'cognitive', 'mental', 'think'],
    technology: ['tech', 'digital', 'software', 'ai', 'computer', 'algorithm'],
    environment: ['climate', 'nature', 'environment', 'sustainable', 'ecology'],
    politics: ['government', 'policy', 'political', 'vote', 'democracy', 'law'],
    science: ['research', 'study', 'data', 'experiment', 'evidence', 'hypothesis'],
    culture: ['culture', 'social', 'society', 'tradition', 'community'],
    history: ['history', 'historical', 'past', 'ancient', 'era', 'period'],
    philosophy: ['philosophy', 'metaphysics', 'epistemology', 'existence', 'truth'],
  };

  for (const thought of thoughts) {
    const content = thought.content.toLowerCase();
    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      if (keywords.some((kw) => content.includes(kw))) {
        coveredDomains.add(domain);
      }
    }
  }

  const blindSpots: BlindSpot[] = [];
  const uncoveredDomains = domains.filter((d) => !coveredDomains.has(d));

  // Only flag if there's significant content but missing domains
  if (thoughts.length >= 10 && uncoveredDomains.length > 0) {
    for (const domain of uncoveredDomains.slice(0, 3)) {
      blindSpots.push({
        domain,
        description: `Your thoughts haven't explored ${domain}-related considerations`,
        suggestedExplorations: getSuggestionsForDomain(domain),
        detectedAt: Date.now(),
        severity: thoughts.length > 30 ? 'medium' : 'low',
      });
    }
  }

  return blindSpots;
}

function getSuggestionsForDomain(domain: string): string[] {
  const suggestions: Record<string, string[]> = {
    ethics: [
      'What are the moral implications of your positions?',
      'Who might be harmed or benefited?',
      'What ethical principles are at stake?',
    ],
    economics: [
      'What are the economic costs and benefits?',
      'Who bears the costs? Who gains?',
      'What are the incentive structures involved?',
    ],
    psychology: [
      'What psychological factors might influence this?',
      'How do emotions affect this situation?',
      'What cognitive processes are involved?',
    ],
    technology: [
      'How might technology change this?',
      'What are the technological constraints?',
      'What technical solutions might exist?',
    ],
    environment: [
      'What are the environmental impacts?',
      'Is this sustainable long-term?',
      'How does this affect natural systems?',
    ],
    politics: [
      'What are the political dynamics?',
      'How might policy affect this?',
      'What are the power structures involved?',
    ],
    science: [
      'What does the empirical evidence say?',
      'Are there relevant studies or data?',
      'What would falsify this belief?',
    ],
    culture: [
      'How do cultural factors influence this?',
      'Would this differ across cultures?',
      'What social norms are relevant?',
    ],
    history: [
      'What historical precedents exist?',
      'How has this evolved over time?',
      'What can we learn from past examples?',
    ],
    philosophy: [
      'What are the underlying assumptions?',
      'What philosophical frameworks apply?',
      'How would different philosophies view this?',
    ],
  };

  return suggestions[domain] || ['Explore this domain further'];
}

function findMissingConsiderations(thoughts: Thought[]): BlindSpot[] {
  const blindSpots: BlindSpot[] = [];

  // Check if user considers opposing views
  const hasCounterConsiderations = thoughts.some(
    (t) =>
      /\b(however|but|although|on the other hand|alternatively|critics|opponents)\b/i.test(
        t.content
      )
  );

  if (thoughts.length >= 5 && !hasCounterConsiderations) {
    blindSpots.push({
      domain: 'dialectics',
      description: 'Your thoughts show limited engagement with opposing viewpoints',
      suggestedExplorations: [
        'What would critics say about your positions?',
        'What evidence would change your mind?',
        'Who disagrees and why?',
      ],
      detectedAt: Date.now(),
      severity: 'high',
    });
  }

  // Check for uncertainty acknowledgment
  const hasUncertainty = thoughts.some(
    (t) => /\b(uncertain|unsure|might be wrong|don\'t know|unclear)\b/i.test(t.content)
  );

  if (thoughts.length >= 10 && !hasUncertainty) {
    blindSpots.push({
      domain: 'epistemic humility',
      description: 'Your thoughts rarely acknowledge uncertainty or limitations',
      suggestedExplorations: [
        'Where might your knowledge be incomplete?',
        'What are you most uncertain about?',
        'What assumptions might be wrong?',
      ],
      detectedAt: Date.now(),
      severity: 'medium',
    });
  }

  return blindSpots;
}

function findTemporalGaps(thoughts: Thought[]): BlindSpot[] {
  const blindSpots: BlindSpot[] = [];

  // Check for future consideration
  const considersFuture = thoughts.some(
    (t) => /\b(future|long-term|eventually|years from now|generations)\b/i.test(t.content)
  );

  // Check for past consideration
  const considersPast = thoughts.some(
    (t) => /\b(historically|in the past|previously|tradition|origins)\b/i.test(t.content)
  );

  if (thoughts.length >= 10) {
    if (!considersFuture) {
      blindSpots.push({
        domain: 'temporal-future',
        description: 'Your thoughts focus on the present without considering long-term implications',
        suggestedExplorations: [
          'What might this look like in 10 years?',
          'What are the long-term consequences?',
          'How might future generations view this?',
        ],
        detectedAt: Date.now(),
        severity: 'low',
      });
    }

    if (!considersPast) {
      blindSpots.push({
        domain: 'temporal-past',
        description: 'Your thoughts don\'t draw on historical context',
        suggestedExplorations: [
          'What historical precedents exist?',
          'How has thinking on this evolved?',
          'What can we learn from past mistakes?',
        ],
        detectedAt: Date.now(),
        severity: 'low',
      });
    }
  }

  return blindSpots;
}

function findPerspectiveGaps(thoughts: Thought[]): BlindSpot[] {
  const blindSpots: BlindSpot[] = [];

  // Check for diverse perspective consideration
  const considersOthers = thoughts.some(
    (t) =>
      /\b(from (their|his|her|another) perspective|they might|others think|different viewpoint)\b/i.test(
        t.content
      )
  );

  if (thoughts.length >= 10 && !considersOthers) {
    blindSpots.push({
      domain: 'perspective-taking',
      description: 'Your thoughts primarily reflect a single perspective',
      suggestedExplorations: [
        'How would someone with opposite views see this?',
        'What would an expert in another field say?',
        'How might this look to someone from a different culture?',
      ],
      detectedAt: Date.now(),
      severity: 'medium',
    });
  }

  return blindSpots;
}

// ============================================================================
// COGNITIVE BIAS DETECTOR
// ============================================================================

/**
 * Detect cognitive biases in a thought
 */
export function detectBiases(thought: Thought, recentThoughts: Thought[] = []): CognitiveBias[] {
  const biases: CognitiveBias[] = [];
  const content = thought.content.toLowerCase();

  // Confirmation bias
  const confirmationBias = detectConfirmationBias(thought, recentThoughts);
  if (confirmationBias) biases.push(confirmationBias);

  // Anchoring bias
  const anchoringBias = detectAnchoringBias(content);
  if (anchoringBias) biases.push(anchoringBias);

  // Availability bias
  const availabilityBias = detectAvailabilityBias(content);
  if (availabilityBias) biases.push(availabilityBias);

  // Bandwagon effect
  const bandwagonBias = detectBandwagonBias(content);
  if (bandwagonBias) biases.push(bandwagonBias);

  // Authority bias
  const authorityBias = detectAuthorityBias(content);
  if (authorityBias) biases.push(authorityBias);

  // Framing effect
  const framingBias = detectFramingBias(content);
  if (framingBias) biases.push(framingBias);

  // Sunk cost fallacy
  const sunkCostBias = detectSunkCostBias(content);
  if (sunkCostBias) biases.push(sunkCostBias);

  // Hindsight bias
  const hindsightBias = detectHindsightBias(content);
  if (hindsightBias) biases.push(hindsightBias);

  return biases.filter((b) => b.confidence > 0.4);
}

function detectConfirmationBias(
  thought: Thought,
  recentThoughts: Thought[]
): CognitiveBias | null {
  // Check if user is only seeking confirming evidence
  const content = thought.content.toLowerCase();

  const confirmingPatterns = [
    /proves (that|my point)/,
    /this confirms/,
    /as I (thought|expected|knew)/,
    /obviously/,
    /clearly shows/,
  ];

  const isConfirming = confirmingPatterns.some((p) => p.test(content));

  // Check if recent thoughts all support the same position
  if (recentThoughts.length >= 3) {
    const allSameDirection = recentThoughts
      .slice(-5)
      .every(
        (t) =>
          t.thought_type === thought.thought_type &&
          t.epistemic.normative === thought.epistemic.normative
      );

    if (allSameDirection && isConfirming) {
      return {
        biasType: 'confirmation',
        description:
          'You may be selectively attending to information that confirms existing beliefs',
        triggeredBy: thought.content.slice(0, 50),
        debiasingSuggestion:
          'Actively seek out disconfirming evidence. Ask: "What would convince me I\'m wrong?"',
        confidence: 0.7,
        detectedAt: Date.now(),
      };
    }
  }

  if (isConfirming) {
    return {
      biasType: 'confirmation',
      description: 'Language suggests confirming a pre-existing belief rather than testing it',
      triggeredBy: thought.content.slice(0, 50),
      debiasingSuggestion:
        'Consider: What evidence would change your mind? Have you looked for it?',
      confidence: 0.5,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectAnchoringBias(content: string): CognitiveBias | null {
  const anchoringPatterns = [
    /first (impression|thought|instinct)/,
    /originally (said|thought|believed)/,
    /started (at|with|from)/,
    /initial (estimate|guess|assumption)/,
  ];

  if (anchoringPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'anchoring',
      description: 'Reference to initial information may be anchoring subsequent judgments',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Try generating your estimate independently before comparing to the anchor. Consider: Would you think differently without that initial reference point?',
      confidence: 0.5,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectAvailabilityBias(content: string): CognitiveBias | null {
  const availabilityPatterns = [
    /recently (read|saw|heard)/,
    /just (learned|discovered|found out)/,
    /in the news/,
    /everyone is talking/,
    /viral|trending/,
  ];

  if (availabilityPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'availability',
      description: 'Recent or memorable information may be given disproportionate weight',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Consider base rates and statistical frequencies. Memorable examples aren\'t necessarily representative.',
      confidence: 0.55,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectBandwagonBias(content: string): CognitiveBias | null {
  const bandwagonPatterns = [
    /everyone (knows|thinks|agrees)/,
    /most people/,
    /the consensus is/,
    /common (knowledge|sense)/,
    /obviously true/,
    /no one disagrees/,
  ];

  if (bandwagonPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'bandwagon',
      description: 'Appeal to popularity doesn\'t establish truth',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Popular beliefs can be wrong. Evaluate the evidence independently. History is full of popular beliefs that were later overturned.',
      confidence: 0.6,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectAuthorityBias(content: string): CognitiveBias | null {
  const authorityPatterns = [
    /expert(s)? (say|claim|believe)/,
    /according to (scientists|studies|research)/,
    /(he|she) (is|was) (a|an) (expert|professor|doctor)/,
    /the (study|research) (shows|proves)/,
  ];

  if (authorityPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'authority',
      description: 'Authority claims should be evaluated, not automatically accepted',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Consider: Is this expert speaking within their domain? Is there consensus? What\'s the quality of the evidence?',
      confidence: 0.45,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectFramingBias(content: string): CognitiveBias | null {
  const framingPatterns = [
    /\b(only|just|merely)\s+\d+%/,
    /\b(as much as|up to)\s+\d+/,
    /\bsaving\b.*\brather than\b.*\blosing\b/i,
    /\brisk\b.*\b(vs|versus|compared to)\b.*\bsafe\b/i,
  ];

  if (framingPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'framing',
      description: 'The way information is presented may be influencing the conclusion',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Reframe the same information differently. "90% survival rate" vs "10% mortality rate" are the same fact.',
      confidence: 0.55,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectSunkCostBias(content: string): CognitiveBias | null {
  const sunkCostPatterns = [
    /already (invested|spent|put in)/,
    /come (this|so) far/,
    /too late to/,
    /can't (quit|stop|give up) now/,
    /wasted if/,
  ];

  if (sunkCostPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'sunk_cost',
      description: 'Past investments shouldn\'t influence future decisions',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Ask: "If I hadn\'t already invested, would I make this choice now?" Focus on future costs and benefits.',
      confidence: 0.65,
      detectedAt: Date.now(),
    };
  }

  return null;
}

function detectHindsightBias(content: string): CognitiveBias | null {
  const hindsightPatterns = [
    /knew (it|this) (all along|would happen)/,
    /should have (seen|known)/,
    /obvious(ly)? in (hindsight|retrospect)/,
    /predictable/,
    /was inevitable/,
  ];

  if (hindsightPatterns.some((p) => p.test(content))) {
    return {
      biasType: 'hindsight',
      description: 'Outcomes often seem more predictable after the fact than they were',
      triggeredBy: content.slice(0, 50),
      debiasingSuggestion:
        'Remember what you actually knew at the time. Could other outcomes have occurred? Be humble about predictability.',
      confidence: 0.6,
      detectedAt: Date.now(),
    };
  }

  return null;
}

// ============================================================================
// PERSPECTIVE SIMULATOR
// ============================================================================

const PERSPECTIVE_PROFILES: Record<string, WorldviewSnapshot> = {
  stoic: {
    coreValues: ['virtue', 'reason', 'self-control', 'acceptance'],
    assumptions: ['We control our judgments, not external events', 'Emotions follow from beliefs'],
    methodology: 'Focus on what is within your control; accept what is not',
    priors: { external_control: 0.1, internal_control: 0.9, emotion_follows_reason: 0.8 },
  },
  utilitarian: {
    coreValues: ['well-being', 'impartiality', 'consequences'],
    assumptions: ['Happiness is measurable', 'All beings\' interests count equally'],
    methodology: 'Maximize total well-being across all affected parties',
    priors: { individual_rights: 0.4, collective_good: 0.8, measurable_welfare: 0.7 },
  },
  pragmatist: {
    coreValues: ['practical results', 'experimentation', 'adaptability'],
    assumptions: ['Truth is what works', 'Ideas should be tested by consequences'],
    methodology: 'Judge beliefs by their practical effects; experiment and iterate',
    priors: { abstract_truth: 0.3, practical_success: 0.9, flexibility: 0.8 },
  },
  skeptic: {
    coreValues: ['evidence', 'doubt', 'intellectual humility'],
    assumptions: ['Claims require evidence', 'Certainty is rarely justified'],
    methodology: 'Suspend judgment until sufficient evidence; question assumptions',
    priors: { default_belief: 0.3, evidence_requirement: 0.9, certainty_possible: 0.2 },
  },
  deontologist: {
    coreValues: ['duty', 'rights', 'universal principles'],
    assumptions: ['Some actions are inherently right/wrong', 'Persons have inviolable rights'],
    methodology: 'Act according to rules that could be universal laws',
    priors: { ends_justify_means: 0.1, universal_rights: 0.9, rule_following: 0.8 },
  },
  existentialist: {
    coreValues: ['authenticity', 'freedom', 'responsibility'],
    assumptions: ['Existence precedes essence', 'We create our own meaning'],
    methodology: 'Embrace radical freedom and take responsibility for choices',
    priors: { inherent_meaning: 0.1, personal_freedom: 0.9, self_creation: 0.9 },
  },
};

/**
 * Simulate how a thought would be viewed from different perspectives
 */
export function simulatePerspective(
  thought: Thought,
  perspectiveName: string
): PerspectiveSimulation {
  const profile = PERSPECTIVE_PROFILES[perspectiveName.toLowerCase()];

  if (!profile) {
    return generateCustomPerspective(thought, perspectiveName);
  }

  const response = generatePerspectiveResponse(thought, perspectiveName, profile);
  const differences = identifyKeyDifferences(thought, profile);

  return {
    perspectiveName,
    perspectiveType: 'philosopher',
    worldview: profile,
    response,
    keyDifferences: differences,
    generatedAt: Date.now(),
  };
}

function generatePerspectiveResponse(
  thought: Thought,
  perspectiveName: string,
  profile: WorldviewSnapshot
): string {
  const content = thought.content;

  const responses: Record<string, string> = {
    stoic: `A Stoic would ask: "Is this within your control?" ${
      content.includes('should')
        ? 'Focus on your own judgments and actions, not what others should do.'
        : ''
    } Remember: it's not events that disturb us, but our judgments about them. What virtue can you practice here?`,

    utilitarian: `From a utilitarian view: Who is affected by this? What outcomes maximize well-being? ${
      content.includes('right') || content.includes('wrong')
        ? 'Rather than inherent rightness, consider: what consequences follow?'
        : ''
    } Have you considered all stakeholders impartially?`,

    pragmatist: `A pragmatist would ask: What practical difference does this make? ${
      thought.confidence > 0.7
        ? 'What experiments could test this belief?'
        : 'Good that you hold this tentatively.'
    } Ideas should be judged by their fruits in experience.`,

    skeptic: `A skeptic would ask: What's the evidence? ${
      (thought.epistemic.empirical === EmpiricalLevel.E3 || thought.epistemic.empirical === EmpiricalLevel.E4)
        ? 'You claim empirical grounding—can this be independently verified?'
        : 'This seems more theoretical than empirical.'
    } What would falsify this belief? How confident should you really be?`,

    deontologist: `A deontologist would ask: What principle are you following? Could this be a universal law? ${
      content.includes('everyone')
        ? 'Good, you\'re thinking about universalizability.'
        : 'Would you want everyone to act this way?'
    } Are any rights or duties at stake?`,

    existentialist: `An existentialist would ask: Is this an authentic choice? ${
      content.includes('should')
        ? 'Beware of "should"—are these your values or inherited expectations?'
        : ''
    } You are radically free—and responsible. What meaning are you creating?`,
  };

  return responses[perspectiveName.toLowerCase()] || `Consider this from a ${perspectiveName} perspective...`;
}

function identifyKeyDifferences(thought: Thought, profile: WorldviewSnapshot): string[] {
  const differences: string[] = [];
  const content = thought.content.toLowerCase();

  // Check for methodology conflicts
  if (content.includes('feel') && profile.methodology.includes('reason')) {
    differences.push('This perspective emphasizes reason over feeling');
  }

  if (content.includes('should') && profile.coreValues.includes('acceptance')) {
    differences.push('This perspective focuses on acceptance rather than prescription');
  }

  if (thought.confidence > 0.8 && profile.priors.certainty_possible && profile.priors.certainty_possible < 0.5) {
    differences.push('This perspective would hold this belief more tentatively');
  }

  // Add core value differences
  if (profile.coreValues.includes('consequences') && !content.includes('result')) {
    differences.push('This perspective emphasizes consequences you haven\'t mentioned');
  }

  if (profile.coreValues.includes('duty') && !content.includes('obligat')) {
    differences.push('This perspective would frame this in terms of duties');
  }

  return differences.length > 0 ? differences : ['This perspective would approach this differently'];
}

function generateCustomPerspective(thought: Thought, name: string): PerspectiveSimulation {
  return {
    perspectiveName: name,
    perspectiveType: 'custom',
    worldview: {
      coreValues: [],
      assumptions: [],
      methodology: 'Unknown',
      priors: {},
    },
    response: `Imagining ${name}'s perspective on: "${thought.content.slice(0, 50)}..." - Consider what values and assumptions ${name} would bring to this question.`,
    keyDifferences: ['Custom perspective - define the worldview for more specific analysis'],
    generatedAt: Date.now(),
  };
}

/**
 * Get all available perspective names
 */
export function getAvailablePerspectives(): string[] {
  return Object.keys(PERSPECTIVE_PROFILES);
}

// ============================================================================
// COMPREHENSIVE AUGMENTATION
// ============================================================================

/**
 * Generate a full cognitive augmentation report for a thought
 */
export async function generateAugmentationReport(
  thought: Thought,
  recentThoughts: Thought[] = [],
  useLLM: boolean = false
): Promise<AugmentationReport> {
  // Generate all augmentations
  const steelman = await generateSteelman(thought, useLLM);
  const blindSpots = detectBlindSpots([thought, ...recentThoughts]);
  const biases = detectBiases(thought, recentThoughts);

  // Generate multiple perspectives
  const perspectives = getAvailablePerspectives()
    .slice(0, 3)
    .map((p) => simulatePerspective(thought, p));

  // Calculate overall cognitive quality score
  const biasScore = 1 - (biases.length * 0.15); // Lose points for biases
  const blindSpotScore = 1 - (blindSpots.filter((b) => b.severity === 'high').length * 0.2);
  const overallScore = Math.max(0, Math.min(1, (biasScore + blindSpotScore + thought.confidence) / 3));

  // Generate recommendations
  const recommendations = generateRecommendations(steelman, blindSpots, biases);

  return {
    originalThought: thought,
    steelman,
    blindSpots,
    biases,
    perspectives,
    overallScore,
    recommendations,
  };
}

function generateRecommendations(
  steelman: SteelmanArgument,
  blindSpots: BlindSpot[],
  biases: CognitiveBias[]
): string[] {
  const recommendations: string[] = [];

  // Steelman-based recommendations
  if (steelman.strengthScore > 0.6) {
    recommendations.push(
      'Consider engaging seriously with the strongest counter-arguments identified'
    );
  }

  // Bias-based recommendations
  if (biases.some((b) => b.biasType === 'confirmation')) {
    recommendations.push('Actively seek disconfirming evidence for your position');
  }

  if (biases.some((b) => b.biasType === 'anchoring')) {
    recommendations.push('Try generating estimates independently before comparing');
  }

  // Blind spot-based recommendations
  const highSeverity = blindSpots.filter((b) => b.severity === 'high');
  if (highSeverity.length > 0) {
    recommendations.push(`Explore: ${highSeverity[0].suggestedExplorations[0]}`);
  }

  // Default recommendations
  if (recommendations.length === 0) {
    recommendations.push('Consider exploring your reasoning from multiple philosophical perspectives');
  }

  return recommendations.slice(0, 5);
}
