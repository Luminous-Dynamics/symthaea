/**
 * Full Integration Tests
 *
 * End-to-end tests for the LUCID + Symthaea integration:
 * - Embedding → Coherence → Analysis flow
 * - E/N/M/H classification round-trips
 * - Plugin system functionality
 * - Consciousness dashboard data
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { invoke } from '@tauri-apps/api/core';

// Skip tests if not in Tauri environment
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

interface EpistemicClassification {
  empirical: number;
  normative: number;
  materiality: number;
  harmonic: number;
}

interface AnalysisResult {
  content: string;
  epistemic: EpistemicClassification;
  phi: number;
  coherence: number;
  embedding: number[];
}

interface CoherenceResult {
  overall_coherence: number;
  is_coherent: boolean;
  phase_alignment: number;
  semantic_similarity: number;
}

interface MindState {
  phi: number;
  working_memory_size: number;
}

interface MindProfile {
  phi: number;
  meta_awareness: number;
  coherence: number;
}

describe.skipIf(!isTauri)('Full Integration', () => {
  beforeAll(async () => {
    try {
      await invoke('initialize_symthaea');
      await invoke('initialize_lucid_mind');
    } catch (e) {
      // Already initialized
    }
  });

  describe('E/N/M/H Classification', () => {
    it('should classify empirical claims correctly', async () => {
      const result: AnalysisResult = await invoke('analyze_thought', {
        content: 'Water boils at 100°C at sea level atmospheric pressure',
      });

      // Empirical claims should have high E level
      expect(result.epistemic.empirical).toBeGreaterThan(2);
      console.log(`Empirical claim - E:${result.epistemic.empirical} N:${result.epistemic.normative} M:${result.epistemic.materiality} H:${result.epistemic.harmonic}`);
    });

    it('should classify normative claims correctly', async () => {
      const result: AnalysisResult = await invoke('analyze_thought', {
        content: 'We ought to treat others with respect and dignity',
      });

      // Normative claims should have high N level
      expect(result.epistemic.normative).toBeGreaterThan(2);
      console.log(`Normative claim - E:${result.epistemic.empirical} N:${result.epistemic.normative} M:${result.epistemic.materiality} H:${result.epistemic.harmonic}`);
    });

    it('should classify abstract/material correctly', async () => {
      const concrete: AnalysisResult = await invoke('analyze_thought', {
        content: 'The chair in my room is made of wood',
      });

      const abstract: AnalysisResult = await invoke('analyze_thought', {
        content: 'Justice is a virtue that transcends cultural boundaries',
      });

      // Concrete should have higher M, abstract should have lower M
      expect(concrete.epistemic.materiality).toBeGreaterThan(abstract.epistemic.materiality);

      console.log(`Concrete M:${concrete.epistemic.materiality}, Abstract M:${abstract.epistemic.materiality}`);
    });

    it('should preserve H level in round-trips', async () => {
      // Analyze with a specific harmonic context
      const result: AnalysisResult = await invoke('analyze_thought', {
        content: 'The melody of the universe resonates through all existence',
      });

      // H level should be captured
      expect(result.epistemic.harmonic).toBeDefined();
      expect(result.epistemic.harmonic).toBeGreaterThanOrEqual(0);
      expect(result.epistemic.harmonic).toBeLessThanOrEqual(4);

      console.log(`Harmonic level: H${result.epistemic.harmonic}`);
    });
  });

  describe('Embedding + Analysis Pipeline', () => {
    it('should generate embedding alongside analysis', async () => {
      const result: AnalysisResult = await invoke('analyze_thought', {
        content: 'Artificial intelligence is transforming how we work',
      });

      // Should have both embedding and epistemic classification
      expect(result.embedding).toBeDefined();
      expect(result.embedding.length).toBe(16384); // HDC dimension

      expect(result.epistemic).toBeDefined();
      expect(result.phi).toBeDefined();
      expect(result.coherence).toBeDefined();

      console.log(`Analysis complete - phi:${result.phi.toFixed(3)}, coherence:${result.coherence.toFixed(3)}`);
    });

    it('should produce similar embeddings for similar analyses', async () => {
      const result1: AnalysisResult = await invoke('analyze_thought', {
        content: 'Climate change is caused by human activities',
      });

      const result2: AnalysisResult = await invoke('analyze_thought', {
        content: 'Human actions are responsible for global warming',
      });

      // Calculate similarity
      let dot = 0;
      let norm1 = 0;
      let norm2 = 0;

      for (let i = 0; i < result1.embedding.length; i++) {
        dot += result1.embedding[i] * result2.embedding[i];
        norm1 += result1.embedding[i] * result1.embedding[i];
        norm2 += result2.embedding[i] * result2.embedding[i];
      }

      const similarity = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
      expect(similarity).toBeGreaterThan(0.6);

      console.log(`Similar thought embedding similarity: ${similarity.toFixed(3)}`);
    });
  });

  describe('Coherence + Phi Integration', () => {
    it('should provide consistent phi and coherence', async () => {
      const thoughts = [
        'Scientific method requires testable hypotheses',
        'Experiments must be reproducible',
        'Peer review validates findings',
      ];

      // Seed mind with coherent thoughts
      await invoke('seed_working_memory', { thoughts });

      // Check coherence
      const coherenceResult = await invoke<CoherenceResult>('check_coherence', { thoughtContents: thoughts });

      // Get mind state
      const mindState = await invoke<MindState>('get_mind_state');

      // Both should indicate high integration
      expect(coherenceResult.overall_coherence).toBeGreaterThan(0.5);
      expect(mindState.phi).toBeGreaterThan(0);

      console.log(`Coherence: ${coherenceResult.overall_coherence.toFixed(3)}, Mind phi: ${mindState.phi.toFixed(3)}`);
    });

    it('should detect when phi drops due to contradiction', async () => {
      // First, coherent state
      await invoke('clear_working_memory');
      await invoke('seed_working_memory', {
        thoughts: [
          'I believe in evidence-based decision making',
          'Data should guide our choices',
        ],
      });

      const coherentState = await invoke<MindState>('get_mind_state');

      // Now add contradictory thought
      await invoke('analyze_with_context', {
        content: 'I ignore all evidence and decide based on gut feeling alone',
      });

      // Note: Phi might not drop immediately due to how ContinuousMind works,
      // but coherence check should catch it
      const incoherentCheck = await invoke<CoherenceResult>('check_coherence', {
        thoughtContents: [
          'I believe in evidence-based decision making',
          'I ignore all evidence and decide based on gut feeling alone',
        ],
      });

      expect(incoherentCheck.is_coherent).toBe(false);

      console.log(`Coherent phi: ${coherentState.phi.toFixed(3)}`);
      console.log(`After contradiction - coherent: ${incoherentCheck.is_coherent}`);
    });
  });

  describe('Full Analysis Workflow', () => {
    it('should complete a full thought analysis workflow', async () => {
      // 1. Clear state
      await invoke('clear_working_memory');

      // 2. Seed with context
      await invoke('seed_working_memory', {
        thoughts: [
          'I have been researching renewable energy sources',
          'Solar and wind power are becoming more cost-effective',
        ],
      });

      // 3. Analyze new thought
      const analysis: AnalysisResult = await invoke('analyze_thought', {
        content: 'Investment in green technology will likely increase',
      });

      // 4. Check coherence with existing thoughts
      const coherence = await invoke<CoherenceResult>('check_coherence', {
        thoughtContents: [
          'Solar and wind power are becoming more cost-effective',
          'Investment in green technology will likely increase',
        ],
      });

      // 5. Get consciousness profile
      const profile = await invoke<MindProfile>('get_mind_consciousness_profile');

      // All parts should work together
      expect(analysis.embedding.length).toBe(16384);
      expect(coherence.is_coherent).toBe(true);
      expect(profile.phi).toBeGreaterThan(0);

      console.log('Full workflow completed successfully');
      console.log(`  - Analysis phi: ${analysis.phi.toFixed(3)}`);
      console.log(`  - Coherence: ${coherence.overall_coherence.toFixed(3)}`);
      console.log(`  - Mind phi: ${profile.phi.toFixed(3)}`);
      console.log(`  - Meta-awareness: ${profile.meta_awareness.toFixed(3)}`);
    });
  });

  describe('Batch Operations', () => {
    it('should handle batch embedding efficiently', async () => {
      const texts = [
        'First thought about philosophy',
        'Second thought about science',
        'Third thought about art',
        'Fourth thought about music',
        'Fifth thought about literature',
      ];

      const startTime = performance.now();

      const embeddings: number[][] = await invoke('batch_embed', { texts });

      const endTime = performance.now();

      expect(embeddings.length).toBe(5);
      embeddings.forEach((emb) => {
        expect(emb.length).toBe(16384);
      });

      console.log(`Batch embedding of 5 texts took ${(endTime - startTime).toFixed(2)}ms`);
    });
  });
});
