// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Embedding Plugin System for LUCID
 *
 * Allows switching between different embedding models:
 * - Symthaea HDC: 16,384-dimensional hyperdimensional computing vectors
 * - Neural Bridge v2: BGE-M3 based 1024-dimensional dense vectors (when available)
 *
 * The plugin system provides a unified interface for:
 * - Text embedding generation
 * - Similarity computation
 * - Batch embedding
 */

import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// TYPES
// ============================================================================

export interface EmbeddingPlugin {
  /** Human-readable plugin name */
  name: string;

  /** Short identifier for the plugin */
  id: string;

  /** Embedding dimension */
  dimension: number;

  /** Description of the embedding model */
  description: string;

  /** Initialize the plugin (load models, etc.) */
  initialize(): Promise<void>;

  /** Check if the plugin is available and ready */
  isAvailable(): Promise<boolean>;

  /** Embed a single text */
  embed(text: string): Promise<number[]>;

  /** Embed multiple texts */
  batchEmbed(texts: string[]): Promise<number[][]>;

  /** Compute similarity between two embeddings */
  similarity(a: number[], b: number[]): number;
}

export interface PluginCapabilities {
  /** Supports semantic similarity */
  semanticSimilarity: boolean;

  /** Supports cross-lingual embeddings */
  crossLingual: boolean;

  /** GPU acceleration available */
  gpuAcceleration: boolean;

  /** Supports streaming/incremental embedding */
  streaming: boolean;
}

export interface PluginStatus {
  id: string;
  name: string;
  available: boolean;
  initialized: boolean;
  dimension: number;
  capabilities: PluginCapabilities;
}

// ============================================================================
// SYMTHAEA HDC PLUGIN
// ============================================================================

/**
 * Symthaea HDC Plugin
 *
 * Uses 16,384-dimensional hyperdimensional computing vectors.
 * This is the primary embedding method in LUCID, providing:
 * - High-dimensional sparse representations
 * - Algebraic compositionality
 * - Fast CPU inference
 */
export const HDC_PLUGIN: EmbeddingPlugin = {
  name: 'Symthaea HDC',
  id: 'hdc',
  dimension: 16384,
  description: 'Hyperdimensional computing vectors (16,384D). Default for LUCID.',

  async initialize() {
    await invoke('initialize_symthaea');
  },

  async isAvailable(): Promise<boolean> {
    try {
      const ready = await invoke<boolean>('symthaea_ready');
      return ready;
    } catch {
      return false;
    }
  },

  async embed(text: string): Promise<number[]> {
    return invoke<number[]>('embed_text', { text });
  },

  async batchEmbed(texts: string[]): Promise<number[][]> {
    return invoke<number[][]>('batch_embed', { texts });
  },

  similarity(a: number[], b: number[]): number {
    return cosineSimilarity(a, b);
  },
};

// ============================================================================
// NEURAL BRIDGE V2 PLUGIN
// ============================================================================

/**
 * Neural Bridge v2 Plugin
 *
 * Uses BGE-M3 encoder for high-quality semantic embeddings.
 * Only available when compiled with the neural-bridge feature.
 *
 * Provides:
 * - Dense 1024-dimensional vectors
 * - State-of-the-art semantic similarity
 * - Cross-lingual support
 */
export const NEURAL_BRIDGE_PLUGIN: EmbeddingPlugin = {
  name: 'Neural Bridge v2 (BGE-M3)',
  id: 'neural-bridge',
  dimension: 1024,
  description: 'BGE-M3 dense embeddings (1024D). Requires neural-bridge feature.',

  async initialize() {
    // Neural Bridge initializes with Symthaea
    await invoke('initialize_symthaea');
  },

  async isAvailable(): Promise<boolean> {
    try {
      // Check if neural bridge is enabled via status
      const status = await invoke<{ hasNeuralBridge?: boolean }>('get_symthaea_status');
      // Neural bridge availability is determined at compile time
      // We check via env variable set during build
      return import.meta.env.VITE_ENABLE_NEURAL_BRIDGE === 'true';
    } catch {
      return false;
    }
  },

  async embed(text: string): Promise<number[]> {
    // When neural bridge is available, Symthaea uses it internally
    // The embed_text command returns the appropriate embedding
    return invoke<number[]>('embed_text', { text });
  },

  async batchEmbed(texts: string[]): Promise<number[][]> {
    return invoke<number[][]>('batch_embed', { texts });
  },

  similarity(a: number[], b: number[]): number {
    return cosineSimilarity(a, b);
  },
};

// ============================================================================
// FALLBACK PLUGIN (No Tauri)
// ============================================================================

/**
 * Fallback Plugin for when Tauri is not available
 *
 * Uses hash-based embeddings for basic functionality.
 * This is not semantically meaningful but allows the UI to function.
 */
export const FALLBACK_PLUGIN: EmbeddingPlugin = {
  name: 'Fallback (Hash-based)',
  id: 'fallback',
  dimension: 1024,
  description: 'Hash-based embeddings for offline/web mode. Not semantic.',

  async initialize() {
    // No-op
  },

  async isAvailable(): Promise<boolean> {
    return true; // Always available as fallback
  },

  async embed(text: string): Promise<number[]> {
    return hashBasedEmbedding(text, 1024);
  },

  async batchEmbed(texts: string[]): Promise<number[][]> {
    return texts.map(t => hashBasedEmbedding(t, 1024));
  },

  similarity(a: number[], b: number[]): number {
    return cosineSimilarity(a, b);
  },
};

// ============================================================================
// PLUGIN REGISTRY
// ============================================================================

class EmbeddingPluginRegistry {
  private plugins: Map<string, EmbeddingPlugin> = new Map();
  private activePluginId: string = 'hdc';
  private initialized: boolean = false;

  constructor() {
    // Register built-in plugins
    this.register(HDC_PLUGIN);
    this.register(NEURAL_BRIDGE_PLUGIN);
    this.register(FALLBACK_PLUGIN);
  }

  /**
   * Register a new plugin
   */
  register(plugin: EmbeddingPlugin): void {
    this.plugins.set(plugin.id, plugin);
  }

  /**
   * Get a plugin by ID
   */
  get(id: string): EmbeddingPlugin | undefined {
    return this.plugins.get(id);
  }

  /**
   * Get the currently active plugin
   */
  getActive(): EmbeddingPlugin {
    return this.plugins.get(this.activePluginId) || FALLBACK_PLUGIN;
  }

  /**
   * Set the active plugin
   */
  async setActive(id: string): Promise<boolean> {
    const plugin = this.plugins.get(id);
    if (!plugin) {
      console.warn(`Plugin ${id} not found`);
      return false;
    }

    const available = await plugin.isAvailable();
    if (!available) {
      console.warn(`Plugin ${id} is not available`);
      return false;
    }

    this.activePluginId = id;
    console.log(`Activated embedding plugin: ${plugin.name}`);
    return true;
  }

  /**
   * List all registered plugins
   */
  list(): string[] {
    return Array.from(this.plugins.keys());
  }

  /**
   * Get status of all plugins
   */
  async getStatus(): Promise<PluginStatus[]> {
    const statuses: PluginStatus[] = [];

    for (const [id, plugin] of this.plugins) {
      const available = await plugin.isAvailable();
      statuses.push({
        id,
        name: plugin.name,
        available,
        initialized: this.initialized && available,
        dimension: plugin.dimension,
        capabilities: getPluginCapabilities(id),
      });
    }

    return statuses;
  }

  /**
   * Initialize the registry and select the best available plugin
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Try to initialize HDC first (primary)
    const hdcAvailable = await HDC_PLUGIN.isAvailable();
    if (hdcAvailable) {
      this.activePluginId = 'hdc';
      this.initialized = true;
      console.log('Embedding system initialized with Symthaea HDC');
      return;
    }

    // Fall back to fallback plugin
    this.activePluginId = 'fallback';
    this.initialized = true;
    console.warn('Using fallback embedding plugin (hash-based)');
  }

  /**
   * Embed text using the active plugin
   */
  async embed(text: string): Promise<number[]> {
    const plugin = this.getActive();
    return plugin.embed(text);
  }

  /**
   * Batch embed texts using the active plugin
   */
  async batchEmbed(texts: string[]): Promise<number[][]> {
    const plugin = this.getActive();
    return plugin.batchEmbed(texts);
  }

  /**
   * Compute similarity using the active plugin
   */
  similarity(a: number[], b: number[]): number {
    const plugin = this.getActive();
    return plugin.similarity(a, b);
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom < 1e-10) return 0;

  return dot / denom;
}

/**
 * Generate a hash-based embedding (fallback, not semantic)
 */
function hashBasedEmbedding(text: string, dimension: number): number[] {
  const embedding = new Array(dimension).fill(0);

  const words = text.toLowerCase().split(/\s+/);
  for (let wordIdx = 0; wordIdx < words.length; wordIdx++) {
    const word = words[wordIdx];
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = ((hash << 5) - hash) + word.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Spread across dimensions
    for (let i = 0; i < 8; i++) {
      const bit = (hash >> i) & 1;
      const dim = Math.abs((wordIdx * 64 + i * 13 + hash) % dimension);
      embedding[dim] += bit ? 1 : -1;
    }
  }

  // Normalize
  const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  if (norm > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }

  return embedding;
}

/**
 * Get capabilities for a plugin
 */
function getPluginCapabilities(id: string): PluginCapabilities {
  switch (id) {
    case 'hdc':
      return {
        semanticSimilarity: true,
        crossLingual: false,
        gpuAcceleration: false,
        streaming: false,
      };
    case 'neural-bridge':
      return {
        semanticSimilarity: true,
        crossLingual: true,
        gpuAcceleration: true,
        streaming: false,
      };
    case 'fallback':
    default:
      return {
        semanticSimilarity: false,
        crossLingual: false,
        gpuAcceleration: false,
        streaming: false,
      };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

/** Global embedding plugin registry */
export const EMBEDDING_PLUGINS = new EmbeddingPluginRegistry();

/** Map of all available plugins for direct access */
export const PLUGIN_MAP: Map<string, EmbeddingPlugin> = new Map([
  ['hdc', HDC_PLUGIN],
  ['neural-bridge', NEURAL_BRIDGE_PLUGIN],
  ['fallback', FALLBACK_PLUGIN],
]);

// Auto-initialize when module is imported
if (typeof window !== 'undefined') {
  EMBEDDING_PLUGINS.initialize().catch(console.error);
}
