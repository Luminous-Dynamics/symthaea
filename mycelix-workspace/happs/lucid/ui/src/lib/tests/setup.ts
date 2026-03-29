// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vitest Test Setup
 *
 * Global setup for LUCID verification tests.
 * Handles Tauri environment detection and mocking when needed.
 */

import { beforeAll, afterAll, vi } from 'vitest';

// Check if we're in a Tauri environment
export const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

// Create mock embedding for tests
const mockEmbedding = new Array(16384).fill(0).map(() => Math.random() - 0.5);

// Use vi.hoisted to ensure mockInvoke is available when vi.mock runs
const { mockInvoke } = vi.hoisted(() => {
  return {
    mockInvoke: vi.fn()
  };
});

// Mock Tauri API at module level (vi.mock is hoisted)
vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
}));

// Configure mock behavior
function setupMockBehavior() {
  mockInvoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
    switch (cmd) {
      case 'initialize_symthaea':
      case 'initialize_lucid_mind':
      case 'clear_working_memory':
      case 'mind_tick':
        return undefined;

      case 'embed_text':
        // Return slightly varied embedding based on input
        const text = args?.text as string || '';
        return mockEmbedding.map((v, i) => v + (text.charCodeAt(i % text.length) || 0) * 0.001);

      case 'batch_embed':
        const texts = args?.texts as string[] || [];
        return texts.map((t, idx) =>
          mockEmbedding.map((v, i) => v + (t.charCodeAt(i % t.length) || 0) * 0.001 + idx * 0.0001)
        );

      case 'check_coherence':
        const thoughts = args?.thoughtContents as string[] || [];
        // Simple mock: check for contradiction keywords
        const hasContradiction = thoughts.some((t, i) =>
          thoughts.some((t2, j) =>
            i !== j && (
              (t.includes('love') && t2.includes('hate')) ||
              (t.includes('always') && t2.includes('never')) ||
              (t.includes('flat') && t2.includes('round'))
            )
          )
        );

        return {
          is_coherent: !hasContradiction,
          overall_coherence: hasContradiction ? 0.3 : 0.8,
          phi_estimate: hasContradiction ? 0.4 : 0.7,
          contradictions: hasContradiction ? [{
            thought_a_idx: 0,
            thought_b_idx: 1,
            contradiction_type: 'semantic',
            severity: 0.8,
            explanation: 'Detected contradictory statements',
          }] : [],
          analysis_details: 'Mock analysis',
        };

      case 'seed_working_memory':
        return undefined;

      case 'get_mind_state':
        return {
          phi: 0.6,
          meta_awareness: 0.5,
          working_memory_size: 3,
          session_length: 5,
        };

      case 'get_session_memory':
        return ['Memory item 1', 'Memory item 2', 'Memory item 3'];

      case 'get_mind_consciousness_profile':
        return {
          phi: 0.65,
          meta_awareness: 0.55,
          cognitive_load: 0.4,
          emotional_valence: 0.2,
          arousal: 0.5,
        };

      case 'analyze_with_context':
      case 'analyze_thought':
        return {
          content: args?.content as string || '',
          epistemic_cube: { e: 'E2', n: 'N1', m: 'M2', h: 'H1' },
          epistemic: {
            empirical: 2,
            normative: 1,
            materiality: 2,
            harmonic: 1,
          },
          phi: 0.65,
          coherence: 0.75,
          confidence: 0.7,
          meta_awareness: 0.5,
          embedding: mockEmbedding,
        };

      // ZKP commands
      case 'zkp_ready':
        return true;

      case 'generate_anonymous_belief_proof':
        return {
          proof_json: JSON.stringify({
            type: 'anonymous_belief',
            commitment: 'mock_commitment',
            belief_hash: args?.belief_hash,
          }),
          author_commitment: 'mock_author_commitment_' + Date.now(),
          belief_hash: args?.belief_hash as string || 'mock_hash',
        };

      case 'generate_reputation_range_proof':
        return {
          proof_json: JSON.stringify({
            type: 'reputation_range',
            threshold: args?.min_threshold,
          }),
          reputation_commitment: 'mock_rep_commitment_' + Date.now(),
          min_threshold: args?.min_threshold as number || 0,
        };

      case 'verify_proof':
        return {
          valid: true,
          error: null,
        };

      case 'create_value_commitment':
        return {
          commitment: 'mock_commitment_' + Date.now(),
          nonce: 'mock_nonce_' + Date.now(),
        };

      case 'hash_belief_content':
        const content = args?.content as string || '';
        // Simple hash simulation
        let hash = 0;
        for (let i = 0; i < content.length; i++) {
          hash = ((hash << 5) - hash) + content.charCodeAt(i);
          hash |= 0;
        }
        return Math.abs(hash).toString(16).padStart(64, '0');

      default:
        console.warn(`Mock invoke: unhandled command "${cmd}"`);
        return undefined;
    }
  });
}

// Initialize mock behavior for non-Tauri environments
if (!isTauri) {
  setupMockBehavior();
}

beforeAll(() => {
  console.log(`Running tests in ${isTauri ? 'Tauri' : 'mock'} environment`);
});

afterAll(() => {
  console.log('Tests completed');
});
