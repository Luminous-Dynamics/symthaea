// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Demo Mode Data for LUCID
 *
 * Provides pre-populated thoughts with E/N/M/H classification, phi scores,
 * and deliberate contradictions for demonstrating coherence detection.
 *
 * Activate with: VITE_DEMO_MODE=true (in .env) or ?demo=true (URL param)
 */

import type { Thought } from '@mycelix/lucid-client';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  HarmonicLevel,
  ThoughtType,
} from '@mycelix/lucid-client';
import type { ThoughtCoherenceStatus } from '../stores/thoughts';

let nextId = 1;

function makeId(): string {
  return `demo-thought-${nextId++}`;
}

const NOW = Date.now() * 1000; // Holochain uses microseconds
const HOUR = 3600 * 1_000_000;

export const DEMO_THOUGHTS: Thought[] = [
  {
    id: makeId(),
    content:
      'Consciousness may be substrate-independent — if Integrated Information Theory is correct, then any system with sufficient phi could be conscious, whether biological or silicon-based.',
    thought_type: ThoughtType.Hypothesis,
    epistemic: {
      empirical: EmpiricalLevel.E2,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M0,
      harmonic: HarmonicLevel.H3,
    },
    confidence: 0.72,
    tags: ['consciousness', 'philosophy', 'IIT'],
    domain: 'philosophy',
    created_at: NOW - 8 * HOUR,
    updated_at: NOW - 8 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'Peer-reviewed studies show that meditation practitioners exhibit increased gamma-band coherence and higher phi values in EEG analysis, suggesting meditation genuinely increases neural integration.',
    thought_type: ThoughtType.Claim,
    epistemic: {
      empirical: EmpiricalLevel.E3,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M2,
      harmonic: HarmonicLevel.H3,
    },
    confidence: 0.85,
    tags: ['neuroscience', 'meditation', 'EEG'],
    domain: 'science',
    created_at: NOW - 6 * HOUR,
    updated_at: NOW - 6 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'Decentralized systems should prioritize sovereignty over efficiency. The ability for individuals to control their own data is a foundational right, even if centralized alternatives are faster.',
    thought_type: ThoughtType.Claim,
    epistemic: {
      empirical: EmpiricalLevel.E0,
      normative: NormativeLevel.N3,
      materiality: MaterialityLevel.M3,
      harmonic: HarmonicLevel.H4,
    },
    confidence: 0.91,
    tags: ['decentralization', 'ethics', 'sovereignty'],
    domain: 'technology',
    created_at: NOW - 5 * HOUR,
    updated_at: NOW - 5 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'How does Byzantine fault tolerance relate to trust in human communities? Is the 34% threshold analogous to social trust dynamics?',
    thought_type: ThoughtType.Question,
    epistemic: {
      empirical: EmpiricalLevel.E1,
      normative: NormativeLevel.N1,
      materiality: MaterialityLevel.M1,
      harmonic: HarmonicLevel.H2,
    },
    confidence: 0.55,
    tags: ['BFT', 'trust', 'social-systems'],
    domain: 'technology',
    created_at: NOW - 4 * HOUR,
    updated_at: NOW - 4 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'I observed that my CfC neural network tends to collapse to attractor states when trained on periodic signals. The amplitude error grows with period length, suggesting a fundamental limitation.',
    thought_type: ThoughtType.Note,
    epistemic: {
      empirical: EmpiricalLevel.E3,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M2,
      harmonic: HarmonicLevel.H2,
    },
    confidence: 0.88,
    tags: ['CfC', 'neural-networks', 'limitations'],
    domain: 'science',
    created_at: NOW - 3 * HOUR,
    updated_at: NOW - 3 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'Efficiency should be the primary design goal for distributed systems. Centralized coordination is almost always preferable when latency matters.',
    thought_type: ThoughtType.Claim,
    epistemic: {
      empirical: EmpiricalLevel.E1,
      normative: NormativeLevel.N2,
      materiality: MaterialityLevel.M2,
      harmonic: HarmonicLevel.H1,
    },
    confidence: 0.62,
    tags: ['decentralization', 'efficiency', 'systems'],
    domain: 'technology',
    created_at: NOW - 2 * HOUR,
    updated_at: NOW - 2 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'The Eight Harmonies framework provides a non-reductive way to evaluate technology: sacred reciprocity, integral wisdom, evolutionary progression, and the rest form a coherent ethical compass.',
    thought_type: ThoughtType.Reflection,
    epistemic: {
      empirical: EmpiricalLevel.E0,
      normative: NormativeLevel.N3,
      materiality: MaterialityLevel.M1,
      harmonic: HarmonicLevel.H4,
    },
    confidence: 0.78,
    tags: ['seven-harmonies', 'ethics', 'philosophy'],
    domain: 'philosophy',
    created_at: NOW - 1 * HOUR,
    updated_at: NOW - 1 * HOUR,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
  {
    id: makeId(),
    content:
      'HDC with 16,384 dimensions achieves 92.9% on the ETHICS benchmark using only compositional vector operations — no language model needed. This suggests ethical reasoning can be grounded in algebraic structure.',
    thought_type: ThoughtType.Insight,
    epistemic: {
      empirical: EmpiricalLevel.E4,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M2,
      harmonic: HarmonicLevel.H3,
    },
    confidence: 0.93,
    tags: ['HDC', 'ethics', 'benchmark'],
    domain: 'science',
    created_at: NOW - 30 * 60 * 1_000_000,
    updated_at: NOW - 30 * 60 * 1_000_000,
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    version: 1,
  },
];

/**
 * Generate demo coherence statuses, including a deliberate contradiction
 * between thoughts 3 (sovereignty > efficiency) and 6 (efficiency > sovereignty)
 */
export function generateDemoCoherence(): Map<string, ThoughtCoherenceStatus> {
  const map = new Map<string, ThoughtCoherenceStatus>();

  // Most thoughts are coherent
  for (const thought of DEMO_THOUGHTS) {
    if (thought.id === DEMO_THOUGHTS[5].id) continue; // Handle contradiction separately

    map.set(thought.id, {
      thoughtId: thought.id,
      coherenceScore: 0.7 + Math.random() * 0.25,
      phiScore: 0.5 + Math.random() * 0.4,
      isCoherent: true,
      contradictions: [],
      suggestions: [],
      checkedAt: Date.now(),
    });
  }

  // Thought 3 (sovereignty) and Thought 6 (efficiency) contradict each other
  const sovereigntyId = DEMO_THOUGHTS[2].id;
  const efficiencyId = DEMO_THOUGHTS[5].id;

  // Update sovereignty thought to note the contradiction
  map.set(sovereigntyId, {
    thoughtId: sovereigntyId,
    coherenceScore: 0.42,
    phiScore: 0.71,
    isCoherent: false,
    contradictions: [
      {
        conflictingThoughtId: efficiencyId,
        conflictingContentPreview:
          'Efficiency should be the primary design goal for distributed systems...',
        severity: 0.7,
        contradictionType: 'negation',
        description:
          "Opposing stance on sovereignty vs efficiency in distributed systems. This may represent a belief evolution.",
      },
    ],
    suggestions: [
      'Found 1 potential contradiction. Review and resolve for better cognitive clarity.',
    ],
    checkedAt: Date.now(),
  });

  // Efficiency thought also flagged
  map.set(efficiencyId, {
    thoughtId: efficiencyId,
    coherenceScore: 0.38,
    phiScore: 0.45,
    isCoherent: false,
    contradictions: [
      {
        conflictingThoughtId: sovereigntyId,
        conflictingContentPreview:
          'Decentralized systems should prioritize sovereignty over efficiency...',
        severity: 0.7,
        contradictionType: 'negation',
        description:
          "Conflicts with your earlier stance that sovereignty should take priority over efficiency.",
      },
    ],
    suggestions: [
      'This thought contradicts an earlier belief. Consider which position better reflects your current understanding.',
    ],
    checkedAt: Date.now(),
  });

  return map;
}

/**
 * Check if demo mode is active
 */
export function isDemoMode(): boolean {
  // Check URL parameter
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    if (params.get('demo') === 'true') return true;
  }

  // Check env variable (Vite)
  try {
    if (import.meta.env?.VITE_DEMO_MODE === 'true') return true;
  } catch {
    // Not in Vite context
  }

  return false;
}
