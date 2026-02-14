/**
 * Semantic Clustering Service
 *
 * Enhanced clustering using Symthaea embeddings for:
 * - Belief similarity detection
 * - Pattern emergence discovery
 * - Semantic neighborhood exploration
 * - Topic modeling
 */

import { invoke } from '@tauri-apps/api/core';
import type { BeliefShare } from './collective-sensemaking';

// ============================================================================
// TYPES
// ============================================================================

export interface SemanticCluster {
  id: string;
  centroid: number[];
  members: ClusterMember[];
  label: string;
  coherence: number;
  density: number;
  topTerms: string[];
}

export interface ClusterMember {
  beliefHash: string;
  content: string;
  embedding: number[];
  distance: number;
  contribution: number;
}

export interface SimilarityResult {
  beliefHash1: string;
  beliefHash2: string;
  similarity: number;
  sharedDimensions: number[];
}

export interface ClusteringConfig {
  minClusterSize: number;
  maxClusters: number;
  similarityThreshold: number;
  useSymthaea: boolean;
}

export interface SemanticNeighborhood {
  center: BeliefShare;
  neighbors: Array<{
    belief: BeliefShare;
    similarity: number;
    relationship: 'similar' | 'contrasting' | 'complementary';
  }>;
  radius: number;
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

const DEFAULT_CONFIG: ClusteringConfig = {
  minClusterSize: 3,
  maxClusters: 10,
  similarityThreshold: 0.7,
  useSymthaea: true,
};

// ============================================================================
// EMBEDDING SERVICE
// ============================================================================

let symthaeaAvailable = false;

async function checkSymthaeaAvailability(): Promise<boolean> {
  try {
    await invoke('embed_text', { text: 'test' });
    symthaeaAvailable = true;
    return true;
  } catch {
    symthaeaAvailable = false;
    return false;
  }
}

async function getEmbedding(text: string): Promise<number[]> {
  if (symthaeaAvailable) {
    try {
      return await invoke('embed_text', { text });
    } catch (e) {
      console.warn('Symthaea embedding failed, using local fallback');
    }
  }
  return generateLocalEmbedding(text);
}

async function getBatchEmbeddings(texts: string[]): Promise<number[][]> {
  if (symthaeaAvailable) {
    try {
      return await invoke('batch_embed', { texts });
    } catch (e) {
      console.warn('Symthaea batch embedding failed, using local fallback');
    }
  }
  return Promise.all(texts.map(generateLocalEmbedding));
}

/**
 * Generate a simple local embedding (fallback when Symthaea unavailable)
 * Uses a basic bag-of-words + character n-gram approach
 */
function generateLocalEmbedding(text: string): number[] {
  const embedding = new Array(256).fill(0);
  const words = text.toLowerCase().split(/\s+/);

  // Word-based features
  for (const word of words) {
    const hash = simpleHash(word);
    embedding[hash % 128] += 1;

    // Character n-grams
    for (let i = 0; i < word.length - 2; i++) {
      const trigram = word.slice(i, i + 3);
      const trigramHash = simpleHash(trigram);
      embedding[128 + (trigramHash % 128)] += 0.5;
    }
  }

  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0)) || 1;
  return embedding.map(v => v / magnitude);
}

function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

// ============================================================================
// SIMILARITY FUNCTIONS
// ============================================================================

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
  return magnitude > 0 ? dotProduct / magnitude : 0;
}

export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) return Infinity;

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

// ============================================================================
// CLUSTERING ALGORITHMS
// ============================================================================

/**
 * K-means++ clustering with smart initialization
 */
export async function kMeansClustering(
  beliefs: BeliefShare[],
  k: number,
  maxIterations: number = 100
): Promise<SemanticCluster[]> {
  if (beliefs.length < k) {
    return [];
  }

  // Get embeddings
  const texts = beliefs.map(b => b.content);
  const embeddings = await getBatchEmbeddings(texts);

  // Initialize centroids using k-means++
  const centroids = initializeCentroids(embeddings, k);

  let assignments = new Array(beliefs.length).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign points to nearest centroid
    const newAssignments = embeddings.map((emb) => {
      let minDist = Infinity;
      let nearest = 0;
      for (let j = 0; j < centroids.length; j++) {
        const dist = euclideanDistance(emb, centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          nearest = j;
        }
      }
      return nearest;
    });

    // Check for convergence
    if (assignments.every((a, i) => a === newAssignments[i])) {
      break;
    }
    assignments = newAssignments;

    // Update centroids
    for (let j = 0; j < centroids.length; j++) {
      const members = embeddings.filter((_, i) => assignments[i] === j);
      if (members.length > 0) {
        centroids[j] = computeCentroid(members);
      }
    }
  }

  // Build cluster objects
  return centroids.map((centroid, i) => {
    const memberIndices = assignments
      .map((a, idx) => a === i ? idx : -1)
      .filter(idx => idx >= 0);

    const members: ClusterMember[] = memberIndices.map(idx => ({
      beliefHash: beliefs[idx].content_hash,
      content: beliefs[idx].content,
      embedding: embeddings[idx],
      distance: euclideanDistance(embeddings[idx], centroid),
      contribution: 1 / memberIndices.length,
    }));

    const coherence = computeClusterCoherence(members.map(m => m.embedding));
    const density = computeClusterDensity(members.map(m => m.embedding), centroid);
    const topTerms = extractTopTerms(members.map(m => m.content));

    return {
      id: `cluster-${i}`,
      centroid,
      members,
      label: topTerms.slice(0, 3).join(', '),
      coherence,
      density,
      topTerms,
    };
  }).filter(c => c.members.length > 0);
}

function initializeCentroids(embeddings: number[][], k: number): number[][] {
  const centroids: number[][] = [];

  // First centroid: random
  centroids.push([...embeddings[Math.floor(Math.random() * embeddings.length)]]);

  // k-means++ initialization
  while (centroids.length < k) {
    const distances = embeddings.map(emb => {
      const minDist = Math.min(...centroids.map(c => euclideanDistance(emb, c)));
      return minDist * minDist;
    });

    const total = distances.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;

    for (let i = 0; i < distances.length; i++) {
      r -= distances[i];
      if (r <= 0) {
        centroids.push([...embeddings[i]]);
        break;
      }
    }
  }

  return centroids;
}

function computeCentroid(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return [];

  const dim = embeddings[0].length;
  const centroid = new Array(dim).fill(0);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      centroid[i] += emb[i];
    }
  }

  return centroid.map(v => v / embeddings.length);
}

function computeClusterCoherence(embeddings: number[][]): number {
  if (embeddings.length < 2) return 1;

  let totalSim = 0;
  let count = 0;

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      totalSim += cosineSimilarity(embeddings[i], embeddings[j]);
      count++;
    }
  }

  return count > 0 ? totalSim / count : 0;
}

function computeClusterDensity(embeddings: number[][], centroid: number[]): number {
  if (embeddings.length === 0) return 0;

  const avgDist = embeddings.reduce(
    (sum, emb) => sum + euclideanDistance(emb, centroid),
    0
  ) / embeddings.length;

  return 1 / (1 + avgDist);
}

function extractTopTerms(contents: string[], n: number = 5): string[] {
  const termCounts = new Map<string, number>();

  for (const content of contents) {
    const words = content.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 3);

    for (const word of words) {
      termCounts.set(word, (termCounts.get(word) || 0) + 1);
    }
  }

  return Array.from(termCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([term]) => term);
}

// ============================================================================
// SEMANTIC NEIGHBORHOOD
// ============================================================================

export async function findSemanticNeighbors(
  centerBelief: BeliefShare,
  allBeliefs: BeliefShare[],
  maxNeighbors: number = 10,
  minSimilarity: number = 0.5
): Promise<SemanticNeighborhood> {
  const centerText = centerBelief.content;
  const centerEmbedding = await getEmbedding(centerText);

  const neighbors: SemanticNeighborhood['neighbors'] = [];

  for (const belief of allBeliefs) {
    if (belief.content_hash === centerBelief.content_hash) continue;

    const embedding = await getEmbedding(belief.content);
    const similarity = cosineSimilarity(centerEmbedding, embedding);

    if (similarity >= minSimilarity) {
      neighbors.push({
        belief,
        similarity,
        relationship: categorizeRelationship(similarity, centerBelief, belief),
      });
    }
  }

  // Sort by similarity and take top N
  neighbors.sort((a, b) => b.similarity - a.similarity);
  const topNeighbors = neighbors.slice(0, maxNeighbors);

  const radius = topNeighbors.length > 0
    ? topNeighbors[topNeighbors.length - 1].similarity
    : 0;

  return {
    center: centerBelief,
    neighbors: topNeighbors,
    radius,
  };
}

function categorizeRelationship(
  similarity: number,
  a: BeliefShare,
  b: BeliefShare
): 'similar' | 'contrasting' | 'complementary' {
  // Use epistemic codes to detect contrast/complementarity
  if (a.epistemic_code && b.epistemic_code) {
    const aCode = a.epistemic_code;
    const bCode = b.epistemic_code;

    // Check for epistemic dimension differences
    const differences = countEpistemicDifferences(aCode, bCode);

    if (differences >= 3) return 'contrasting';
    if (differences >= 1 && similarity > 0.7) return 'complementary';
  }

  return 'similar';
}

function countEpistemicDifferences(codeA: string, codeB: string): number {
  let differences = 0;
  const levelA = parseEpistemicCode(codeA);
  const levelB = parseEpistemicCode(codeB);

  if (levelA.e !== levelB.e) differences++;
  if (levelA.n !== levelB.n) differences++;
  if (levelA.m !== levelB.m) differences++;
  if (levelA.h !== levelB.h) differences++;

  return differences;
}

function parseEpistemicCode(code: string): { e: number; n: number; m: number; h: number } {
  const match = code.match(/E(\d)N(\d)M(\d)H(\d)/);
  if (!match) return { e: 0, n: 0, m: 0, h: 0 };
  return {
    e: parseInt(match[1]),
    n: parseInt(match[2]),
    m: parseInt(match[3]),
    h: parseInt(match[4]),
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  checkSymthaeaAvailability,
  getEmbedding,
  getBatchEmbeddings,
};
