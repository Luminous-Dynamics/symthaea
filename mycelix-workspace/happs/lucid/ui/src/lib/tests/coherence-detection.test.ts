/**
 * Coherence Detection Verification Tests
 *
 * Tests that the coherence analysis correctly identifies:
 * - Contradictory beliefs (low coherence, low phi)
 * - Consistent beliefs (high coherence, high phi)
 * - Logical inconsistencies
 * - Semantic divergence
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { invoke } from '@tauri-apps/api/core';

// Skip tests if not in Tauri environment
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

interface CoherenceResult {
  is_coherent: boolean;
  overall_coherence: number;
  phi_estimate: number;
  contradictions: DetectedContradiction[];
  analysis_details: string;
}

interface DetectedContradiction {
  thought_a_idx: number;
  thought_b_idx: number;
  contradiction_type: string;
  severity: number;
  explanation: string;
}

describe.skipIf(!isTauri)('Coherence Detection', () => {
  beforeAll(async () => {
    try {
      await invoke('initialize_symthaea');
    } catch (e) {
      // Already initialized
    }
  });

  /**
   * Check coherence of multiple thoughts via Tauri
   */
  async function checkCoherence(thoughts: string[]): Promise<CoherenceResult> {
    return await invoke('check_coherence', { thoughtContents: thoughts });
  }

  describe('Direct Contradictions', () => {
    it('should detect explicit contradictions', async () => {
      const result = await checkCoherence([
        'I believe the Earth is flat',
        'I believe the Earth is round',
      ]);

      expect(result.is_coherent).toBe(false);
      expect(result.contradictions.length).toBeGreaterThan(0);
      expect(result.overall_coherence).toBeLessThan(0.5);

      console.log(`Contradiction detected: ${result.contradictions[0]?.explanation}`);
      console.log(`Overall coherence: ${result.overall_coherence.toFixed(3)}`);
    });

    it('should detect love/hate contradictions', async () => {
      const result = await checkCoherence([
        'I love coffee',
        'I hate coffee',
      ]);

      expect(result.is_coherent).toBe(false);
      expect(result.contradictions.length).toBeGreaterThan(0);

      console.log(`Phi estimate: ${result.phi_estimate.toFixed(3)}`);
    });

    it('should detect always/never contradictions', async () => {
      const result = await checkCoherence([
        'Exercise is always beneficial for health',
        'Exercise is never beneficial for health',
      ]);

      expect(result.is_coherent).toBe(false);
      expect(result.overall_coherence).toBeLessThan(0.4);
    });
  });

  describe('Consistent Beliefs', () => {
    it('should recognize coherent related beliefs', async () => {
      const result = await checkCoherence([
        'Regular exercise improves cardiovascular health',
        'A healthy diet supports physical fitness',
        'Sleep is important for recovery after workouts',
      ]);

      expect(result.is_coherent).toBe(true);
      expect(result.overall_coherence).toBeGreaterThan(0.6);
      expect(result.contradictions.length).toBe(0);

      console.log(`Coherent beliefs - coherence: ${result.overall_coherence.toFixed(3)}`);
    });

    it('should handle epistemically consistent statements', async () => {
      const result = await checkCoherence([
        'I think this might be true based on limited evidence',
        'More research is needed to confirm this hypothesis',
        'Current data suggests a possible correlation',
      ]);

      expect(result.is_coherent).toBe(true);
      expect(result.phi_estimate).toBeGreaterThan(0.5);
    });
  });

  describe('Subtle Contradictions', () => {
    it('should detect implicit contradictions', async () => {
      const result = await checkCoherence([
        'All mammals are warm-blooded',
        'Whales are mammals',
        'Whales are cold-blooded',
      ]);

      // This requires logical inference
      expect(result.overall_coherence).toBeLessThan(0.6);
      console.log(`Implicit contradiction - coherence: ${result.overall_coherence.toFixed(3)}`);
    });

    it('should detect temporal contradictions', async () => {
      const result = await checkCoherence([
        'The meeting starts at 9am',
        'The meeting starts at 10am',
      ]);

      expect(result.is_coherent).toBe(false);
      console.log(`Temporal contradiction detected`);
    });
  });

  describe('Epistemic Level Conflicts', () => {
    it('should flag certainty mismatches', async () => {
      const result = await checkCoherence([
        'I am absolutely certain that climate change is real',
        'I am not sure if climate change is happening',
      ]);

      expect(result.is_coherent).toBe(false);
      // Should detect as epistemic status conflict
      const hasEpistemicConflict = result.contradictions.some(
        c => c.contradiction_type.includes('epistemic') || c.contradiction_type.includes('status')
      );
      console.log(`Epistemic conflict detected: ${hasEpistemicConflict}`);
    });
  });

  describe('Multi-Thought Coherence', () => {
    it('should analyze coherence across many thoughts', async () => {
      const thoughts = [
        'Democracy requires informed citizens',
        'Education is essential for democratic participation',
        'Free press helps keep citizens informed',
        'Censorship threatens democratic values',
        'Open debate strengthens democratic institutions',
      ];

      const result = await checkCoherence(thoughts);

      expect(result.is_coherent).toBe(true);
      expect(result.phi_estimate).toBeGreaterThan(0.6);

      console.log(`5-thought coherence: ${result.overall_coherence.toFixed(3)}`);
      console.log(`Phi estimate: ${result.phi_estimate.toFixed(3)}`);
    });

    it('should detect the odd one out', async () => {
      const thoughts = [
        'Honesty is the best policy',
        'Trust is built through truthfulness',
        'Integrity matters in relationships',
        'Deception is an effective strategy', // Contradicts the theme
      ];

      const result = await checkCoherence(thoughts);

      expect(result.contradictions.length).toBeGreaterThan(0);
      // The last thought should be flagged
      const involvesLast = result.contradictions.some(
        c => c.thought_a_idx === 3 || c.thought_b_idx === 3
      );
      expect(involvesLast).toBe(true);
    });
  });

  describe('Phi (Integrated Information) Estimates', () => {
    it('should return higher phi for tightly integrated beliefs', async () => {
      // Tightly integrated set
      const integratedResult = await checkCoherence([
        'Water freezes at 0°C',
        'Ice is frozen water',
        'Below freezing, water becomes solid',
      ]);

      // Loosely related set
      const looseResult = await checkCoherence([
        'The sky is blue',
        'Cats have four legs',
        'Pizza is delicious',
      ]);

      expect(integratedResult.phi_estimate).toBeGreaterThan(looseResult.phi_estimate);

      console.log(`Integrated beliefs phi: ${integratedResult.phi_estimate.toFixed(3)}`);
      console.log(`Loose beliefs phi: ${looseResult.phi_estimate.toFixed(3)}`);
    });
  });
});
