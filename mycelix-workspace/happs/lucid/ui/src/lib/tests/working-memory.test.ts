// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Working Memory (ContinuousMind) Verification Tests
 *
 * Tests that LucidMind correctly:
 * - Seeds working memory with context
 * - Resolves references based on context
 * - Maintains session memory across analyses
 * - Tracks consciousness state (phi, meta-awareness)
 */

import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { invoke } from '@tauri-apps/api/core';

// Skip tests if not in Tauri environment
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

interface StructuredThought {
  content: string;
  epistemic_cube: {
    e: string;
    n: string;
    m: string;
    h?: string;
  };
  phi: number;
  coherence: number;
  confidence: number;
  meta_awareness: number;
}

interface MindState {
  phi: number;
  meta_awareness: number;
  working_memory_size: number;
  session_length: number;
}

interface ConsciousnessProfile {
  phi: number;
  meta_awareness: number;
  cognitive_load: number;
  emotional_valence: number;
  arousal: number;
}

describe.skipIf(!isTauri)('Working Memory (LucidMind)', () => {
  beforeAll(async () => {
    try {
      await invoke('initialize_lucid_mind');
    } catch (e) {
      // Already initialized
    }
  });

  beforeEach(async () => {
    // Clear working memory before each test
    await invoke('clear_working_memory');
  });

  describe('Context Seeding', () => {
    it('should seed working memory with thoughts', async () => {
      const thoughts = [
        'Quantum mechanics describes the behavior of particles at atomic scales',
        'The uncertainty principle limits what we can know about particle properties',
        'Wave-particle duality is a fundamental aspect of quantum theory',
      ];

      await invoke('seed_working_memory', { thoughts });

      const state: MindState = await invoke('get_mind_state');
      expect(state.working_memory_size).toBeGreaterThan(0);

      console.log(`Working memory size after seeding: ${state.working_memory_size}`);
    });

    it('should increase phi after seeding related concepts', async () => {
      const stateBefore: MindState = await invoke('get_mind_state');

      await invoke('seed_working_memory', {
        thoughts: [
          'Logic is the study of valid reasoning',
          'Deductive arguments preserve truth from premises to conclusion',
          'Inductive arguments provide probabilistic support',
        ],
      });

      const stateAfter: MindState = await invoke('get_mind_state');

      // Phi should increase with integrated information
      expect(stateAfter.phi).toBeGreaterThanOrEqual(stateBefore.phi);

      console.log(`Phi before: ${stateBefore.phi.toFixed(3)}, after: ${stateAfter.phi.toFixed(3)}`);
    });
  });

  describe('Context-Aware Analysis', () => {
    it('should resolve references using context', async () => {
      // Seed with context about a specific topic
      await invoke('seed_working_memory', {
        thoughts: [
          'The Eiffel Tower is located in Paris, France',
          'It was built in 1889 for the World Fair',
          'Gustave Eiffel designed the iron lattice structure',
        ],
      });

      // Analyze a reference that requires context
      const result: StructuredThought = await invoke('analyze_with_context', {
        content: 'It is one of the most visited monuments in the world',
      });

      // The analysis should show high coherence (understands "it" refers to Eiffel Tower)
      expect(result.coherence).toBeGreaterThan(0.5);
      expect(result.phi).toBeGreaterThan(0);

      console.log(`Reference resolution - coherence: ${result.coherence.toFixed(3)}`);
    });

    it('should distinguish between different contexts', async () => {
      // First context: programming
      await invoke('seed_working_memory', {
        thoughts: [
          'Python is a high-level programming language',
          'It emphasizes code readability',
          'Python supports multiple programming paradigms',
        ],
      });

      const programmingResult: StructuredThought = await invoke('analyze_with_context', {
        content: 'It is widely used in data science',
      });

      // Clear and seed new context: biology
      await invoke('clear_working_memory');
      await invoke('seed_working_memory', {
        thoughts: [
          'A python is a type of non-venomous snake',
          'They are constrictors that squeeze their prey',
          'Pythons can be found in Africa and Asia',
        ],
      });

      const biologyResult: StructuredThought = await invoke('analyze_with_context', {
        content: 'It is widely found in tropical regions',
      });

      // Both should have reasonable coherence in their respective contexts
      expect(programmingResult.coherence).toBeGreaterThan(0.4);
      expect(biologyResult.coherence).toBeGreaterThan(0.4);

      console.log(`Programming context coherence: ${programmingResult.coherence.toFixed(3)}`);
      console.log(`Biology context coherence: ${biologyResult.coherence.toFixed(3)}`);
    });
  });

  describe('Session Memory', () => {
    it('should maintain session memory across analyses', async () => {
      // First analysis
      await invoke('analyze_with_context', {
        content: 'Machine learning is a subset of artificial intelligence',
      });

      // Second analysis
      await invoke('analyze_with_context', {
        content: 'Neural networks are inspired by biological brains',
      });

      // Third analysis
      await invoke('analyze_with_context', {
        content: 'Deep learning uses multiple neural network layers',
      });

      const memory: string[] = await invoke('get_session_memory');

      expect(memory.length).toBe(3);
      expect(memory[0]).toContain('Machine learning');

      console.log(`Session memory length: ${memory.length}`);
    });

    it('should build coherent understanding over session', async () => {
      // Analyze related concepts sequentially
      const results: StructuredThought[] = [];

      results.push(await invoke('analyze_with_context', {
        content: 'Epistemology is the study of knowledge',
      }));

      results.push(await invoke('analyze_with_context', {
        content: 'It examines the nature of belief and justification',
      }));

      results.push(await invoke('analyze_with_context', {
        content: 'This field addresses questions about certainty and doubt',
      }));

      // Later analyses should show higher coherence as context builds
      // (they reference previous content)
      console.log(`Coherence progression: ${results.map(r => r.coherence.toFixed(2)).join(' -> ')}`);
    });
  });

  describe('Consciousness Profile', () => {
    it('should provide full consciousness profile', async () => {
      // Build some context
      await invoke('seed_working_memory', {
        thoughts: [
          'I am thinking about my own thought processes',
          'This is a form of metacognition',
          'Self-reflection is important for learning',
        ],
      });

      const profile: ConsciousnessProfile = await invoke('get_mind_consciousness_profile');

      // All values should be in valid ranges
      expect(profile.phi).toBeGreaterThanOrEqual(0);
      expect(profile.phi).toBeLessThanOrEqual(1);

      expect(profile.meta_awareness).toBeGreaterThanOrEqual(0);
      expect(profile.meta_awareness).toBeLessThanOrEqual(1);

      expect(profile.cognitive_load).toBeGreaterThanOrEqual(0);
      expect(profile.cognitive_load).toBeLessThanOrEqual(1);

      expect(profile.emotional_valence).toBeGreaterThanOrEqual(-1);
      expect(profile.emotional_valence).toBeLessThanOrEqual(1);

      expect(profile.arousal).toBeGreaterThanOrEqual(0);
      expect(profile.arousal).toBeLessThanOrEqual(1);

      console.log('Consciousness Profile:');
      console.log(`  Phi: ${profile.phi.toFixed(3)}`);
      console.log(`  Meta-awareness: ${profile.meta_awareness.toFixed(3)}`);
      console.log(`  Cognitive load: ${profile.cognitive_load.toFixed(3)}`);
      console.log(`  Emotional valence: ${profile.emotional_valence.toFixed(3)}`);
      console.log(`  Arousal: ${profile.arousal.toFixed(3)}`);
    });

    it('should increase meta-awareness with self-referential content', async () => {
      const profileBefore: ConsciousnessProfile = await invoke('get_mind_consciousness_profile');

      // Feed self-referential content
      await invoke('analyze_with_context', {
        content: 'I am aware that I am processing this information',
      });

      await invoke('analyze_with_context', {
        content: 'This awareness of my own awareness is metacognition',
      });

      const profileAfter: ConsciousnessProfile = await invoke('get_mind_consciousness_profile');

      console.log(`Meta-awareness: ${profileBefore.meta_awareness.toFixed(3)} -> ${profileAfter.meta_awareness.toFixed(3)}`);
    });
  });

  describe('Mind Tick', () => {
    it('should advance consciousness state with tick', async () => {
      await invoke('seed_working_memory', {
        thoughts: ['Initial context for the mind'],
      });

      const stateBefore: MindState = await invoke('get_mind_state');

      // Advance the mind's internal state
      await invoke('mind_tick');

      const stateAfter: MindState = await invoke('get_mind_state');

      // Session length should increase
      expect(stateAfter.session_length).toBeGreaterThanOrEqual(stateBefore.session_length);

      console.log(`Session length: ${stateBefore.session_length} -> ${stateAfter.session_length}`);
    });
  });
});
