// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * HNSW (Hierarchical Navigable Small World) Vector Index
 *
 * Provides O(log n) approximate nearest neighbor search for LUCID's
 * semantic search functionality. This implementation is optimized
 * for the 16,384-dimensional HDC embeddings from Symthaea.
 *
 * Algorithm based on: https://arxiv.org/abs/1603.09320
 */

import { get as idbGet, set as idbSet, del as idbDel } from 'idb-keyval';

// ============================================================================
// TYPES
// ============================================================================

export interface HNSWConfig {
  /** Maximum number of connections per node at each layer (default: 16) */
  M: number;

  /** Maximum connections at layer 0 (default: 2 * M) */
  M0: number;

  /** Size of dynamic candidate list during construction (default: 200) */
  efConstruction: number;

  /** Size of dynamic candidate list during search (default: 50) */
  efSearch: number;

  /** Normalization factor for level generation (default: 1 / ln(M)) */
  ml: number;
}

export interface HNSWNode {
  id: string;
  vector: number[];
  connections: Map<number, Set<string>>; // layer -> connected node IDs
  layer: number; // highest layer this node appears in
}

export interface SearchCandidate {
  id: string;
  distance: number;
}

export interface HNSWStats {
  nodeCount: number;
  maxLayer: number;
  avgConnections: number;
  memoryEstimateMB: number;
}

// ============================================================================
// HNSW INDEX IMPLEMENTATION
// ============================================================================

export class HNSWIndex {
  private config: HNSWConfig;
  private nodes: Map<string, HNSWNode> = new Map();
  private entryPoint: string | null = null;
  private maxLayer: number = 0;
  private dimension: number;
  private cacheKey: string;

  constructor(dimension: number, config: Partial<HNSWConfig> = {}) {
    this.dimension = dimension;
    this.cacheKey = `lucid-hnsw-${dimension}d-v1`;

    // Default HNSW parameters optimized for ~10K-100K vectors
    const M = config.M ?? 16;
    this.config = {
      M,
      M0: config.M0 ?? 2 * M,
      efConstruction: config.efConstruction ?? 200,
      efSearch: config.efSearch ?? 50,
      ml: config.ml ?? 1 / Math.log(M),
    };
  }

  // ==========================================================================
  // INDEX OPERATIONS
  // ==========================================================================

  /**
   * Add a vector to the index
   */
  add(id: string, vector: number[]): void {
    if (vector.length !== this.dimension) {
      throw new Error(`Dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
    }

    // Determine the layer for this node
    const nodeLayer = this.randomLayer();

    // Create the node
    const node: HNSWNode = {
      id,
      vector,
      connections: new Map(),
      layer: nodeLayer,
    };

    // Initialize connection sets for each layer
    for (let l = 0; l <= nodeLayer; l++) {
      node.connections.set(l, new Set());
    }

    this.nodes.set(id, node);

    // Update max layer if necessary
    if (nodeLayer > this.maxLayer) {
      this.maxLayer = nodeLayer;
    }

    // If this is the first node, set as entry point
    if (!this.entryPoint) {
      this.entryPoint = id;
      return;
    }

    // Find entry point for insertion
    let currentNode = this.entryPoint;

    // Traverse from top layer down to node's layer + 1
    for (let layer = this.maxLayer; layer > nodeLayer; layer--) {
      const nearest = this.searchLayer(vector, currentNode, 1, layer);
      if (nearest.length > 0) {
        currentNode = nearest[0].id;
      }
    }

    // Insert at each layer from nodeLayer down to 0
    for (let layer = Math.min(nodeLayer, this.maxLayer); layer >= 0; layer--) {
      const candidates = this.searchLayer(vector, currentNode, this.config.efConstruction, layer);

      // Select neighbors using heuristic
      const maxConnections = layer === 0 ? this.config.M0 : this.config.M;
      const neighbors = this.selectNeighbors(vector, candidates, maxConnections);

      // Add bidirectional connections
      for (const neighbor of neighbors) {
        node.connections.get(layer)!.add(neighbor.id);

        const neighborNode = this.nodes.get(neighbor.id)!;
        if (!neighborNode.connections.has(layer)) {
          neighborNode.connections.set(layer, new Set());
        }
        neighborNode.connections.get(layer)!.add(id);

        // Prune if neighbor has too many connections
        this.pruneConnections(neighborNode, layer, maxConnections);
      }

      // Update entry point for next layer
      if (candidates.length > 0) {
        currentNode = candidates[0].id;
      }
    }

    // Update entry point if new node is at higher layer
    if (nodeLayer > this.maxLayer) {
      this.entryPoint = id;
    }
  }

  /**
   * Remove a vector from the index
   */
  remove(id: string): boolean {
    const node = this.nodes.get(id);
    if (!node) return false;

    // Remove all connections to this node
    for (const [layer, connections] of node.connections) {
      for (const neighborId of connections) {
        const neighbor = this.nodes.get(neighborId);
        if (neighbor && neighbor.connections.has(layer)) {
          neighbor.connections.get(layer)!.delete(id);
        }
      }
    }

    this.nodes.delete(id);

    // Update entry point if necessary
    if (this.entryPoint === id) {
      this.entryPoint = this.nodes.size > 0 ? (this.nodes.keys().next().value ?? null) : null;
      this.maxLayer = 0;
      for (const n of this.nodes.values()) {
        if (n.layer > this.maxLayer) this.maxLayer = n.layer;
      }
    }

    return true;
  }

  /**
   * Update a vector in the index
   */
  update(id: string, vector: number[]): void {
    if (this.nodes.has(id)) {
      this.remove(id);
    }
    this.add(id, vector);
  }

  /**
   * Check if ID exists in index
   */
  has(id: string): boolean {
    return this.nodes.has(id);
  }

  /**
   * Get vector by ID
   */
  get(id: string): number[] | undefined {
    return this.nodes.get(id)?.vector;
  }

  // ==========================================================================
  // SEARCH OPERATIONS
  // ==========================================================================

  /**
   * Find k nearest neighbors to query vector
   */
  search(query: number[], k: number = 10): SearchCandidate[] {
    if (query.length !== this.dimension) {
      throw new Error(`Dimension mismatch: expected ${this.dimension}, got ${query.length}`);
    }

    if (!this.entryPoint || this.nodes.size === 0) {
      return [];
    }

    // Start from entry point
    let currentNode = this.entryPoint;

    // Traverse from top layer down to layer 1
    for (let layer = this.maxLayer; layer > 0; layer--) {
      const nearest = this.searchLayer(query, currentNode, 1, layer);
      if (nearest.length > 0) {
        currentNode = nearest[0].id;
      }
    }

    // Search at layer 0 with efSearch candidates
    const candidates = this.searchLayer(query, currentNode, this.config.efSearch, 0);

    // Return top k
    return candidates.slice(0, k);
  }

  /**
   * Search layer with beam search
   */
  private searchLayer(
    query: number[],
    entryId: string,
    ef: number,
    layer: number
  ): SearchCandidate[] {
    const visited = new Set<string>([entryId]);
    const entryNode = this.nodes.get(entryId);
    if (!entryNode) return [];

    const entryDist = this.distance(query, entryNode.vector);

    // Min-heap for candidates (closest first)
    const candidates: SearchCandidate[] = [{ id: entryId, distance: entryDist }];

    // Max-heap for results (furthest first for pruning)
    const results: SearchCandidate[] = [{ id: entryId, distance: entryDist }];

    while (candidates.length > 0) {
      // Get closest candidate
      candidates.sort((a, b) => a.distance - b.distance);
      const current = candidates.shift()!;

      // Get furthest result
      results.sort((a, b) => b.distance - a.distance);
      const furthest = results[0];

      // Stop if closest candidate is further than furthest result
      if (current.distance > furthest.distance) {
        break;
      }

      // Explore neighbors
      const currentNode = this.nodes.get(current.id);
      if (!currentNode || !currentNode.connections.has(layer)) continue;

      for (const neighborId of currentNode.connections.get(layer)!) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);

        const neighbor = this.nodes.get(neighborId);
        if (!neighbor) continue;

        const dist = this.distance(query, neighbor.vector);

        // Add to results if better than worst result
        if (results.length < ef || dist < results[0].distance) {
          candidates.push({ id: neighborId, distance: dist });
          results.push({ id: neighborId, distance: dist });

          // Keep only ef best results
          if (results.length > ef) {
            results.sort((a, b) => b.distance - a.distance);
            results.shift();
          }
        }
      }
    }

    // Return sorted by distance
    results.sort((a, b) => a.distance - b.distance);
    return results;
  }

  // ==========================================================================
  // HELPER METHODS
  // ==========================================================================

  /**
   * Generate random layer for new node
   */
  private randomLayer(): number {
    let layer = 0;
    while (Math.random() < Math.exp(-layer * this.config.ml)) {
      layer++;
    }
    return layer;
  }

  /**
   * Select neighbors using simple heuristic
   */
  private selectNeighbors(
    target: number[],
    candidates: SearchCandidate[],
    maxCount: number
  ): SearchCandidate[] {
    // Sort by distance
    candidates.sort((a, b) => a.distance - b.distance);

    // Use simple heuristic: take closest maxCount
    return candidates.slice(0, maxCount);
  }

  /**
   * Prune connections if node has too many
   */
  private pruneConnections(node: HNSWNode, layer: number, maxConnections: number): void {
    const connections = node.connections.get(layer);
    if (!connections || connections.size <= maxConnections) return;

    // Calculate distances and keep closest
    const withDistances: SearchCandidate[] = [];
    for (const neighborId of connections) {
      const neighbor = this.nodes.get(neighborId);
      if (neighbor) {
        withDistances.push({
          id: neighborId,
          distance: this.distance(node.vector, neighbor.vector),
        });
      }
    }

    withDistances.sort((a, b) => a.distance - b.distance);

    // Keep only maxConnections closest
    connections.clear();
    for (let i = 0; i < Math.min(maxConnections, withDistances.length); i++) {
      connections.add(withDistances[i].id);
    }
  }

  /**
   * Calculate distance between vectors (1 - cosine similarity)
   */
  private distance(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom < 1e-10) return 1;

    // Convert similarity to distance
    return 1 - dot / denom;
  }

  // ==========================================================================
  // PERSISTENCE
  // ==========================================================================

  /**
   * Serialize index to JSON-compatible format
   */
  serialize(): object {
    const nodes: Array<{
      id: string;
      vector: number[];
      layer: number;
      connections: Record<number, string[]>;
    }> = [];

    for (const [id, node] of this.nodes) {
      const connections: Record<number, string[]> = {};
      for (const [layer, conn] of node.connections) {
        connections[layer] = Array.from(conn);
      }
      nodes.push({
        id,
        vector: node.vector,
        layer: node.layer,
        connections,
      });
    }

    return {
      config: this.config,
      dimension: this.dimension,
      entryPoint: this.entryPoint,
      maxLayer: this.maxLayer,
      nodes,
    };
  }

  /**
   * Deserialize index from JSON
   */
  static deserialize(data: any): HNSWIndex {
    const index = new HNSWIndex(data.dimension, data.config);
    index.entryPoint = data.entryPoint;
    index.maxLayer = data.maxLayer;

    for (const nodeData of data.nodes) {
      const connections = new Map<number, Set<string>>();
      for (const [layer, conn] of Object.entries(nodeData.connections)) {
        connections.set(Number(layer), new Set(conn as string[]));
      }

      index.nodes.set(nodeData.id, {
        id: nodeData.id,
        vector: nodeData.vector,
        layer: nodeData.layer,
        connections,
      });
    }

    return index;
  }

  /**
   * Save index to IndexedDB
   */
  async save(): Promise<void> {
    const serialized = this.serialize();
    await idbSet(this.cacheKey, serialized);
    console.log(`HNSW index saved (${this.nodes.size} nodes)`);
  }

  /**
   * Load index from IndexedDB
   */
  async load(): Promise<boolean> {
    try {
      const data = await idbGet(this.cacheKey);
      if (!data) return false;

      const loaded = HNSWIndex.deserialize(data);
      this.nodes = loaded.nodes;
      this.entryPoint = loaded.entryPoint;
      this.maxLayer = loaded.maxLayer;
      this.config = loaded.config;

      console.log(`HNSW index loaded (${this.nodes.size} nodes)`);
      return true;
    } catch (error) {
      console.error('Failed to load HNSW index:', error);
      return false;
    }
  }

  /**
   * Clear persisted index
   */
  async clear(): Promise<void> {
    this.nodes.clear();
    this.entryPoint = null;
    this.maxLayer = 0;
    await idbDel(this.cacheKey);
  }

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  /**
   * Get index statistics
   */
  getStats(): HNSWStats {
    let totalConnections = 0;

    for (const node of this.nodes.values()) {
      for (const connections of node.connections.values()) {
        totalConnections += connections.size;
      }
    }

    // Estimate memory usage
    const vectorBytes = this.nodes.size * this.dimension * 4; // float32
    const connectionBytes = totalConnections * 8; // string ID pointers
    const overheadBytes = this.nodes.size * 100; // Map overhead etc.
    const memoryMB = (vectorBytes + connectionBytes + overheadBytes) / (1024 * 1024);

    return {
      nodeCount: this.nodes.size,
      maxLayer: this.maxLayer,
      avgConnections: this.nodes.size > 0 ? totalConnections / this.nodes.size : 0,
      memoryEstimateMB: Math.round(memoryMB * 100) / 100,
    };
  }

  /**
   * Get node count
   */
  get size(): number {
    return this.nodes.size;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export function createHNSWIndex(dimension: number, config?: Partial<HNSWConfig>): HNSWIndex {
  return new HNSWIndex(dimension, config);
}

// Singleton for Symthaea HDC dimension
let globalHDCIndex: HNSWIndex | null = null;

export function getHDCIndex(): HNSWIndex {
  if (!globalHDCIndex) {
    globalHDCIndex = new HNSWIndex(16384, {
      M: 32, // Higher M for high-dimensional vectors
      efConstruction: 400,
      efSearch: 100,
    });
  }
  return globalHDCIndex;
}

// Singleton for transformers.js dimension
let globalTransformersIndex: HNSWIndex | null = null;

export function getTransformersIndex(): HNSWIndex {
  if (!globalTransformersIndex) {
    globalTransformersIndex = new HNSWIndex(384);
  }
  return globalTransformersIndex;
}
