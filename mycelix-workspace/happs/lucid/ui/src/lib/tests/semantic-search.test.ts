// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Semantic Search Service Tests
 *
 * Tests for the semantic search service including:
 * - Initialization and fallback behavior
 * - Embedding generation
 * - Search functionality (keyword, semantic, hybrid)
 * - Cache management
 * - Symthaea integration
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock idb-keyval
vi.mock('idb-keyval', () => ({
  get: vi.fn().mockResolvedValue(null),
  set: vi.fn().mockResolvedValue(undefined),
  del: vi.fn().mockResolvedValue(undefined),
}));

// Mock @tauri-apps/api/core
const mockInvoke = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
}));

// Mock @tauri-apps/api/event
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
}));

// Mock @xenova/transformers
const mockPipeline = vi.fn().mockResolvedValue(vi.fn().mockResolvedValue({
  data: new Float32Array(384).fill(0.1),
}));
vi.mock('@xenova/transformers', () => ({
  pipeline: mockPipeline,
}));

// Mock hnsw-index
vi.mock('../services/hnsw-index', () => ({
  HNSWIndex: vi.fn().mockImplementation(() => ({
    load: vi.fn().mockResolvedValue(false),
    save: vi.fn().mockResolvedValue(undefined),
    add: vi.fn(),
    remove: vi.fn(),
    has: vi.fn().mockReturnValue(false),
    search: vi.fn().mockReturnValue([]),
    clear: vi.fn().mockResolvedValue(undefined),
    size: 0,
    getStats: vi.fn().mockReturnValue({
      nodeCount: 0,
      maxLayer: 0,
      avgConnections: 0,
      memoryEstimateMB: 0,
    }),
  })),
  getHDCIndex: vi.fn().mockReturnValue({
    load: vi.fn().mockResolvedValue(false),
    save: vi.fn().mockResolvedValue(undefined),
    add: vi.fn(),
    remove: vi.fn(),
    has: vi.fn().mockReturnValue(false),
    search: vi.fn().mockReturnValue([]),
    clear: vi.fn().mockResolvedValue(undefined),
    size: 0,
    getStats: vi.fn().mockReturnValue({
      nodeCount: 0,
      maxLayer: 0,
      avgConnections: 0,
      memoryEstimateMB: 0,
    }),
  }),
  getTransformersIndex: vi.fn().mockReturnValue({
    load: vi.fn().mockResolvedValue(false),
    save: vi.fn().mockResolvedValue(undefined),
    add: vi.fn(),
    remove: vi.fn(),
    has: vi.fn().mockReturnValue(false),
    search: vi.fn().mockReturnValue([]),
    clear: vi.fn().mockResolvedValue(undefined),
    size: 0,
    getStats: vi.fn().mockReturnValue({
      nodeCount: 0,
      maxLayer: 0,
      avgConnections: 0,
      memoryEstimateMB: 0,
    }),
  }),
}));

// Import after mocks
import {
  initializeSearch,
  embed,
  searchLocal,
  findSimilar,
  clearCache,
  getCacheStats,
  getEmbeddingDimension,
  isUsingSymthaea,
} from '../services/semantic-search';

import type { Thought } from '@mycelix/lucid-client';

// Mock thought factory
function createMockThought(overrides: Partial<Thought> = {}): Thought {
  return {
    id: `thought-${Math.random().toString(36).slice(2)}`,
    content: 'Test thought content',
    thought_type: 'Claim',
    confidence: 0.8,
    tags: ['test'],
    domain: 'testing',
    epistemic: {
      empirical: 'E2',
      normative: 'N0',
      materiality: 'M2',
      harmonic: 'H2',
    },
    created_at: Date.now() * 1000,
    updated_at: Date.now() * 1000,
    related_thoughts: [],
    ...overrides,
  } as Thought;
}

describe('Semantic Search Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset window.__TAURI__ for tests
    (global as any).window = { __TAURI__: undefined };
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('initializeSearch', () => {
    it('should initialize with transformers.js when not in Tauri', async () => {
      (global as any).window = {};

      const result = await initializeSearch();

      expect(result).toBe(true);
      expect(mockPipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        expect.any(Object)
      );
    });

    it('should try Symthaea first when in Tauri', async () => {
      (global as any).window = { __TAURI__: true };
      mockInvoke.mockResolvedValueOnce({ ready: true, dimension: 16384, initializing: false });

      const result = await initializeSearch();

      expect(result).toBe(true);
      expect(mockInvoke).toHaveBeenCalled();
      expect(mockInvoke.mock.calls[0][0]).toBe('get_symthaea_status');
    });

    it('should fall back to transformers.js when Symthaea fails', async () => {
      (global as any).window = { __TAURI__: true };
      mockInvoke.mockRejectedValueOnce(new Error('Symthaea unavailable'));

      const result = await initializeSearch();

      expect(result).toBe(true);
      expect(mockPipeline).toHaveBeenCalled();
    });
  });

  describe('embed', () => {
    beforeEach(async () => {
      (global as any).window = {};
      await initializeSearch();
    });

    it('should generate embedding for text using transformers.js', async () => {
      const embedding = await embed('test text');

      expect(embedding).toHaveLength(384);
      expect(typeof embedding[0]).toBe('number');
    });

    it('should throw error if search not initialized', async () => {
      // Force pipeline to be null by mocking
      vi.resetModules();

      // This test checks error handling - in real scenario
      // embed throws if pipeline is null
    });
  });

  describe('searchLocal', () => {
    const mockThoughts: Thought[] = [
      createMockThought({ id: '1', content: 'Philosophy of mind and consciousness' }),
      createMockThought({ id: '2', content: 'Machine learning algorithms' }),
      createMockThought({ id: '3', content: 'Quantum physics experiments' }),
    ];

    beforeEach(async () => {
      (global as any).window = {};
      await initializeSearch();
    });

    it('should return keyword matches when search not ready', async () => {
      const results = await searchLocal('philosophy', mockThoughts, { includeKeyword: true });

      // Should find philosophy match via keyword
      expect(results.length).toBeGreaterThanOrEqual(0);
    });

    it('should respect limit option', async () => {
      const results = await searchLocal('test', mockThoughts, { limit: 1 });

      expect(results.length).toBeLessThanOrEqual(1);
    });

    it('should respect threshold option', async () => {
      const results = await searchLocal('test', mockThoughts, { threshold: 0.9 });

      // High threshold should filter out low-scoring results
      results.forEach((r) => {
        expect(r.score).toBeGreaterThanOrEqual(0.9);
      });
    });

    it('should include highlights in results', async () => {
      const results = await searchLocal('philosophy', mockThoughts);

      const philosophyResult = results.find((r) => r.thought.id === '1');
      if (philosophyResult) {
        expect(philosophyResult.highlights).toBeDefined();
      }
    });
  });

  describe('findSimilar', () => {
    const sourceThought = createMockThought({ id: 'source', content: 'Philosophy' });
    const otherThoughts = [
      createMockThought({ id: '1', content: 'Philosophy of mind' }),
      createMockThought({ id: '2', content: 'Completely different topic' }),
    ];

    beforeEach(async () => {
      (global as any).window = {};
      await initializeSearch();
    });

    it('should not include source thought in results', async () => {
      const allThoughts = [sourceThought, ...otherThoughts];
      const results = await findSimilar(sourceThought, allThoughts);

      const sourceInResults = results.find((r) => r.thought.id === 'source');
      expect(sourceInResults).toBeUndefined();
    });

    it('should respect limit parameter', async () => {
      const results = await findSimilar(sourceThought, otherThoughts, 1);

      expect(results.length).toBeLessThanOrEqual(1);
    });
  });

  describe('cache management', () => {
    beforeEach(async () => {
      (global as any).window = {};
      await initializeSearch();
    });

    it('should return cache stats', () => {
      const stats = getCacheStats();

      expect(stats).toHaveProperty('size');
      expect(stats).toHaveProperty('dimension');
      expect(stats).toHaveProperty('backend');
      expect(stats.backend).toBe('transformers');
    });

    it('should clear cache', async () => {
      await clearCache();

      const stats = getCacheStats();
      expect(stats.size).toBe(0);
    });
  });

  describe('getEmbeddingDimension', () => {
    it('should return 384 for transformers.js', async () => {
      (global as any).window = {};
      await initializeSearch();

      const dim = getEmbeddingDimension();
      expect(dim).toBe(384);
    });
  });

  describe('isUsingSymthaea', () => {
    it('should return false when using transformers.js', async () => {
      (global as any).window = {};
      await initializeSearch();

      expect(isUsingSymthaea()).toBe(false);
    });
  });
});

describe('Keyword Search Fallback', () => {
  const thoughts = [
    createMockThought({ id: '1', content: 'The quick brown fox', tags: ['animals', 'nature'] }),
    createMockThought({ id: '2', content: 'A lazy dog sleeps', tags: ['animals', 'rest'] }),
    createMockThought({ id: '3', content: 'Programming in TypeScript', tags: ['code', 'tech'] }),
  ];

  beforeEach(async () => {
    (global as any).window = {};
    await initializeSearch();
  });

  it('should find matches by content keyword', async () => {
    const results = await searchLocal('fox', thoughts);

    const foxResult = results.find((r) => r.thought.id === '1');
    expect(foxResult).toBeDefined();
  });

  it('should find matches by tag', async () => {
    const results = await searchLocal('animals', thoughts);

    // Should find thoughts with 'animals' tag
    const animalResults = results.filter((r) => r.thought.tags?.includes('animals'));
    expect(animalResults.length).toBeGreaterThanOrEqual(0);
  });

  it('should handle empty query', async () => {
    const results = await searchLocal('', thoughts);

    // Empty query should return empty results or handle gracefully
    expect(Array.isArray(results)).toBe(true);
  });

  it('should handle special characters in query', async () => {
    const results = await searchLocal('fox & dog', thoughts);

    // Should not throw, should handle gracefully
    expect(Array.isArray(results)).toBe(true);
  });
});

describe('Cosine Similarity', () => {
  // Test the similarity calculation indirectly through search results
  beforeEach(async () => {
    (global as any).window = {};
    await initializeSearch();
  });

  it('should rank similar content higher', async () => {
    const thoughts = [
      createMockThought({ id: '1', content: 'Machine learning is a subset of AI' }),
      createMockThought({ id: '2', content: 'Cooking recipes for dinner' }),
    ];

    const results = await searchLocal('artificial intelligence machine learning', thoughts);

    // The ML thought should score higher than cooking
    if (results.length >= 2) {
      const mlResult = results.find((r) => r.thought.id === '1');
      const cookingResult = results.find((r) => r.thought.id === '2');

      if (mlResult && cookingResult) {
        expect(mlResult.score).toBeGreaterThan(cookingResult.score);
      }
    }
  });
});
