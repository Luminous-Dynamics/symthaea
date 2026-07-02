// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useDiscovery Hook
 *
 * Smart music discovery using ML embeddings and similarity search.
 * Provides "find similar tracks", mood-based playlists, and genre exploration.
 */

import { useCallback, useState, useRef } from 'react';

// === Types ===

export interface TrackFeatures {
  trackId: string;
  embedding?: number[];
  genreScores?: number[];
  moodScores?: number[];
  bpm?: number;
  key?: string;
  energy: number;
  valence: number;
  danceability: number;
  acousticness: number;
  instrumentalness: number;
}

export interface SimilarityResult {
  trackId: string;
  score: number;
  distance: number;
  // Enriched track data (populated by caller)
  track?: {
    title: string;
    artist: string;
    coverUrl?: string;
    duration?: number;
  };
}

export interface SearchQuery {
  seedTracks: string[];
  targetMood?: number[];
  targetGenre?: number[];
  bpmRange?: [number, number];
  energyRange?: [number, number];
  valenceRange?: [number, number];
  limit: number;
  exclude: string[];
  embeddingWeight: number;
  moodWeight: number;
  genreWeight: number;
  featureWeight: number;
}

export interface DiscoveryState {
  isIndexed: boolean;
  indexSize: number;
  isSearching: boolean;
  results: SimilarityResult[];
  error: string | null;
}

// === Mood and Genre Labels ===

export const MOOD_LABELS = [
  'happy', 'sad', 'energetic', 'calm', 'aggressive',
  'romantic', 'melancholic', 'uplifting', 'dark', 'dreamy',
] as const;

export const GENRE_LABELS = [
  'electronic', 'rock', 'pop', 'hip-hop', 'jazz',
  'classical', 'folk', 'r&b', 'metal', 'ambient',
  'indie', 'punk', 'blues', 'country', 'reggae',
] as const;

export type MoodLabel = typeof MOOD_LABELS[number];
export type GenreLabel = typeof GENRE_LABELS[number];

// === Helper Functions ===

function normalizeVector(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return norm > 0 ? v.map(x => x / norm) : v;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  const dot = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
  const normB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));

  return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
}

function averageVectors(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const avg = new Array(dim).fill(0);

  for (const v of vectors) {
    for (let i = 0; i < dim; i++) {
      avg[i] += v[i] / vectors.length;
    }
  }

  return avg;
}

function audioFeatureSimilarity(a: TrackFeatures, b: TrackFeatures): number {
  let similarity = 0;
  let count = 0;

  similarity += 1 - Math.abs(a.energy - b.energy);
  count++;

  similarity += 1 - Math.abs(a.valence - b.valence);
  count++;

  similarity += 1 - Math.abs(a.danceability - b.danceability);
  count++;

  similarity += 1 - Math.abs(a.acousticness - b.acousticness);
  count++;

  similarity += 1 - Math.abs(a.instrumentalness - b.instrumentalness);
  count++;

  if (a.bpm && b.bpm) {
    const bpmDiff = Math.abs(a.bpm - b.bpm) / Math.max(a.bpm, b.bpm);
    similarity += 1 - Math.min(bpmDiff, 1);
    count++;
  }

  return similarity / count;
}

function moodFromLabel(label: MoodLabel): number[] {
  const vec = new Array(MOOD_LABELS.length).fill(0);
  const idx = MOOD_LABELS.indexOf(label);
  if (idx >= 0) vec[idx] = 1;
  return vec;
}

function genreFromLabel(label: GenreLabel): number[] {
  const vec = new Array(GENRE_LABELS.length).fill(0);
  const idx = GENRE_LABELS.indexOf(label);
  if (idx >= 0) vec[idx] = 1;
  return vec;
}

// === Discovery Engine (client-side) ===

class DiscoveryEngine {
  private tracks: Map<string, TrackFeatures> = new Map();
  private normalizedEmbeddings: Map<string, number[]> = new Map();
  private isBuilt = false;

  addTrack(features: TrackFeatures): void {
    this.tracks.set(features.trackId, features);
    this.isBuilt = false;
  }

  removeTrack(trackId: string): boolean {
    const removed = this.tracks.delete(trackId);
    if (removed) {
      this.normalizedEmbeddings.delete(trackId);
    }
    return removed;
  }

  getTrack(trackId: string): TrackFeatures | undefined {
    return this.tracks.get(trackId);
  }

  build(): void {
    this.normalizedEmbeddings.clear();

    for (const [id, features] of this.tracks) {
      if (features.embedding && features.embedding.length > 0) {
        this.normalizedEmbeddings.set(id, normalizeVector(features.embedding));
      }
    }

    this.isBuilt = true;
  }

  get size(): number {
    return this.tracks.size;
  }

  search(query: SearchQuery): SimilarityResult[] {
    if (!this.isBuilt && this.tracks.size > 0) {
      this.build();
    }

    // Get seed track features
    const seedFeatures = query.seedTracks
      .map(id => this.tracks.get(id))
      .filter((f): f is TrackFeatures => f !== undefined);

    const seedEmbeddings = seedFeatures
      .filter(f => f.embedding && f.embedding.length > 0)
      .map(f => normalizeVector(f.embedding!));

    const avgSeedEmbedding = seedEmbeddings.length > 0
      ? averageVectors(seedEmbeddings)
      : [];

    // Calculate scores for all tracks
    const results: SimilarityResult[] = [];
    const excludeSet = new Set([...query.exclude, ...query.seedTracks]);

    for (const [trackId, features] of this.tracks) {
      if (excludeSet.has(trackId)) continue;

      // Apply filters
      if (query.bpmRange && features.bpm) {
        if (features.bpm < query.bpmRange[0] || features.bpm > query.bpmRange[1]) {
          continue;
        }
      }

      if (query.energyRange) {
        if (features.energy < query.energyRange[0] || features.energy > query.energyRange[1]) {
          continue;
        }
      }

      if (query.valenceRange) {
        if (features.valence < query.valenceRange[0] || features.valence > query.valenceRange[1]) {
          continue;
        }
      }

      // Calculate similarity score
      let score = 0;
      let weightSum = 0;

      // Embedding similarity
      if (query.embeddingWeight > 0 && avgSeedEmbedding.length > 0) {
        const normalized = this.normalizedEmbeddings.get(trackId);
        if (normalized) {
          const embeddingSim = cosineSimilarity(avgSeedEmbedding, normalized);
          score += embeddingSim * query.embeddingWeight;
          weightSum += query.embeddingWeight;
        }
      }

      // Mood similarity
      if (query.moodWeight > 0) {
        const moodSim = query.targetMood
          ? (features.moodScores
              ? cosineSimilarity(query.targetMood, features.moodScores)
              : 0)
          : (seedFeatures.length > 0
              ? seedFeatures.reduce((sum, sf) =>
                  sum + (sf.moodScores && features.moodScores
                    ? cosineSimilarity(sf.moodScores, features.moodScores)
                    : 0), 0) / seedFeatures.length
              : 0);

        score += moodSim * query.moodWeight;
        weightSum += query.moodWeight;
      }

      // Genre similarity
      if (query.genreWeight > 0) {
        const genreSim = query.targetGenre
          ? (features.genreScores
              ? cosineSimilarity(query.targetGenre, features.genreScores)
              : 0)
          : (seedFeatures.length > 0
              ? seedFeatures.reduce((sum, sf) =>
                  sum + (sf.genreScores && features.genreScores
                    ? cosineSimilarity(sf.genreScores, features.genreScores)
                    : 0), 0) / seedFeatures.length
              : 0);

        score += genreSim * query.genreWeight;
        weightSum += query.genreWeight;
      }

      // Audio feature similarity
      if (query.featureWeight > 0 && seedFeatures.length > 0) {
        const featureSim = seedFeatures.reduce(
          (sum, sf) => sum + audioFeatureSimilarity(sf, features),
          0
        ) / seedFeatures.length;

        score += featureSim * query.featureWeight;
        weightSum += query.featureWeight;
      }

      const finalScore = weightSum > 0 ? score / weightSum : 0;

      results.push({
        trackId,
        score: finalScore,
        distance: 1 - finalScore,
      });
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    // Limit results
    return results.slice(0, query.limit);
  }

  findByMood(mood: MoodLabel | number[], limit: number = 20): SimilarityResult[] {
    const moodVector = Array.isArray(mood) ? mood : moodFromLabel(mood);

    const results: SimilarityResult[] = [];

    for (const [trackId, features] of this.tracks) {
      if (features.moodScores && features.moodScores.length > 0) {
        const sim = cosineSimilarity(moodVector, features.moodScores);
        results.push({
          trackId,
          score: sim,
          distance: 1 - sim,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }

  findByGenre(genre: GenreLabel | number[], limit: number = 20): SimilarityResult[] {
    const genreVector = Array.isArray(genre) ? genre : genreFromLabel(genre);

    const results: SimilarityResult[] = [];

    for (const [trackId, features] of this.tracks) {
      if (features.genreScores && features.genreScores.length > 0) {
        const sim = cosineSimilarity(genreVector, features.genreScores);
        results.push({
          trackId,
          score: sim,
          distance: 1 - sim,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }

  clear(): void {
    this.tracks.clear();
    this.normalizedEmbeddings.clear();
    this.isBuilt = false;
  }
}

// === Hook ===

export interface UseDiscoveryReturn extends DiscoveryState {
  // Index management
  addTrack: (features: TrackFeatures) => void;
  addTracks: (features: TrackFeatures[]) => void;
  removeTrack: (trackId: string) => void;
  buildIndex: () => void;
  clearIndex: () => void;

  // Search
  findSimilar: (trackIds: string[], options?: Partial<SearchQuery>) => Promise<SimilarityResult[]>;
  findByMood: (mood: MoodLabel, limit?: number) => Promise<SimilarityResult[]>;
  findByGenre: (genre: GenreLabel, limit?: number) => Promise<SimilarityResult[]>;
  search: (query: SearchQuery) => Promise<SimilarityResult[]>;

  // Smart playlists
  createMoodPlaylist: (moods: MoodLabel[], options?: { limit?: number; diversity?: number }) => Promise<SimilarityResult[]>;
  createGenrePlaylist: (genres: GenreLabel[], options?: { limit?: number; diversity?: number }) => Promise<SimilarityResult[]>;
  createRadioStation: (seedTracks: string[], options?: { limit?: number }) => Promise<SimilarityResult[]>;

  // Utilities
  moodFromLabel: typeof moodFromLabel;
  genreFromLabel: typeof genreFromLabel;
}

export function useDiscovery(): UseDiscoveryReturn {
  const engineRef = useRef<DiscoveryEngine>(new DiscoveryEngine());

  const [state, setState] = useState<DiscoveryState>({
    isIndexed: false,
    indexSize: 0,
    isSearching: false,
    results: [],
    error: null,
  });

  // Add single track
  const addTrack = useCallback((features: TrackFeatures) => {
    engineRef.current.addTrack(features);
    setState(s => ({
      ...s,
      indexSize: engineRef.current.size,
      isIndexed: false,
    }));
  }, []);

  // Add multiple tracks
  const addTracks = useCallback((features: TrackFeatures[]) => {
    for (const f of features) {
      engineRef.current.addTrack(f);
    }
    setState(s => ({
      ...s,
      indexSize: engineRef.current.size,
      isIndexed: false,
    }));
  }, []);

  // Remove track
  const removeTrack = useCallback((trackId: string) => {
    engineRef.current.removeTrack(trackId);
    setState(s => ({
      ...s,
      indexSize: engineRef.current.size,
    }));
  }, []);

  // Build index
  const buildIndex = useCallback(() => {
    engineRef.current.build();
    setState(s => ({ ...s, isIndexed: true }));
  }, []);

  // Clear index
  const clearIndex = useCallback(() => {
    engineRef.current.clear();
    setState({
      isIndexed: false,
      indexSize: 0,
      isSearching: false,
      results: [],
      error: null,
    });
  }, []);

  // Find similar tracks
  const findSimilar = useCallback(async (
    trackIds: string[],
    options?: Partial<SearchQuery>
  ): Promise<SimilarityResult[]> => {
    setState(s => ({ ...s, isSearching: true, error: null }));

    try {
      const query: SearchQuery = {
        seedTracks: trackIds,
        limit: 20,
        exclude: [],
        embeddingWeight: 0.6,
        moodWeight: 0.2,
        genreWeight: 0.1,
        featureWeight: 0.1,
        ...options,
      };

      const results = engineRef.current.search(query);
      setState(s => ({ ...s, isSearching: false, results }));
      return results;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed';
      setState(s => ({ ...s, isSearching: false, error: message }));
      return [];
    }
  }, []);

  // Find by mood
  const findByMood = useCallback(async (
    mood: MoodLabel,
    limit: number = 20
  ): Promise<SimilarityResult[]> => {
    setState(s => ({ ...s, isSearching: true, error: null }));

    try {
      const results = engineRef.current.findByMood(mood, limit);
      setState(s => ({ ...s, isSearching: false, results }));
      return results;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed';
      setState(s => ({ ...s, isSearching: false, error: message }));
      return [];
    }
  }, []);

  // Find by genre
  const findByGenre = useCallback(async (
    genre: GenreLabel,
    limit: number = 20
  ): Promise<SimilarityResult[]> => {
    setState(s => ({ ...s, isSearching: true, error: null }));

    try {
      const results = engineRef.current.findByGenre(genre, limit);
      setState(s => ({ ...s, isSearching: false, results }));
      return results;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed';
      setState(s => ({ ...s, isSearching: false, error: message }));
      return [];
    }
  }, []);

  // Generic search
  const search = useCallback(async (query: SearchQuery): Promise<SimilarityResult[]> => {
    setState(s => ({ ...s, isSearching: true, error: null }));

    try {
      const results = engineRef.current.search(query);
      setState(s => ({ ...s, isSearching: false, results }));
      return results;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed';
      setState(s => ({ ...s, isSearching: false, error: message }));
      return [];
    }
  }, []);

  // Create mood-based playlist
  const createMoodPlaylist = useCallback(async (
    moods: MoodLabel[],
    options?: { limit?: number; diversity?: number }
  ): Promise<SimilarityResult[]> => {
    const limit = options?.limit ?? 30;
    const diversity = options?.diversity ?? 0.3;

    // Combine mood vectors
    const combinedMood = new Array(MOOD_LABELS.length).fill(0);
    for (const mood of moods) {
      const idx = MOOD_LABELS.indexOf(mood);
      if (idx >= 0) combinedMood[idx] += 1 / moods.length;
    }

    // Get initial results
    const allResults = engineRef.current.findByMood(combinedMood, limit * 3);

    // Apply diversity (select varied tracks)
    const selected: SimilarityResult[] = [];
    const usedGenres = new Set<number>();

    for (const result of allResults) {
      if (selected.length >= limit) break;

      const features = engineRef.current.getTrack(result.trackId);
      if (!features?.genreScores) {
        selected.push(result);
        continue;
      }

      // Check genre diversity
      const topGenre = features.genreScores.indexOf(Math.max(...features.genreScores));
      const genreCount = [...usedGenres].filter(g => g === topGenre).length;
      const maxPerGenre = Math.ceil(limit * (1 - diversity) / GENRE_LABELS.length);

      if (genreCount < maxPerGenre) {
        selected.push(result);
        usedGenres.add(topGenre);
      }
    }

    setState(s => ({ ...s, results: selected }));
    return selected;
  }, []);

  // Create genre-based playlist
  const createGenrePlaylist = useCallback(async (
    genres: GenreLabel[],
    options?: { limit?: number; diversity?: number }
  ): Promise<SimilarityResult[]> => {
    const limit = options?.limit ?? 30;
    const diversity = options?.diversity ?? 0.3;

    // Combine genre vectors
    const combinedGenre = new Array(GENRE_LABELS.length).fill(0);
    for (const genre of genres) {
      const idx = GENRE_LABELS.indexOf(genre);
      if (idx >= 0) combinedGenre[idx] += 1 / genres.length;
    }

    const allResults = engineRef.current.findByGenre(combinedGenre, limit * 3);

    // Apply diversity (vary energy/valence)
    const selected: SimilarityResult[] = [];
    let lastEnergy = 0.5;

    for (const result of allResults) {
      if (selected.length >= limit) break;

      const features = engineRef.current.getTrack(result.trackId);
      if (!features) {
        selected.push(result);
        continue;
      }

      // Ensure energy variety
      const energyDiff = Math.abs(features.energy - lastEnergy);
      if (selected.length === 0 || energyDiff > diversity * 0.5) {
        selected.push(result);
        lastEnergy = features.energy;
      }
    }

    setState(s => ({ ...s, results: selected }));
    return selected;
  }, []);

  // Create radio station from seeds
  const createRadioStation = useCallback(async (
    seedTracks: string[],
    options?: { limit?: number }
  ): Promise<SimilarityResult[]> => {
    const limit = options?.limit ?? 50;

    // Mix different similarity weights for variety
    const results = await findSimilar(seedTracks, {
      limit: limit * 2,
      embeddingWeight: 0.5,
      moodWeight: 0.25,
      genreWeight: 0.15,
      featureWeight: 0.1,
    });

    // Shuffle slightly for variety (keep top results but add randomness)
    const shuffled = results.map((r, i) => ({
      ...r,
      sortScore: r.score * (1 + (Math.random() - 0.5) * 0.2 * (i / results.length)),
    }));

    shuffled.sort((a, b) => b.sortScore - a.sortScore);

    const finalResults = shuffled.slice(0, limit).map(({ sortScore, ...r }) => r);
    setState(s => ({ ...s, results: finalResults }));

    return finalResults;
  }, [findSimilar]);

  return {
    ...state,
    addTrack,
    addTracks,
    removeTrack,
    buildIndex,
    clearIndex,
    findSimilar,
    findByMood,
    findByGenre,
    search,
    createMoodPlaylist,
    createGenrePlaylist,
    createRadioStation,
    moodFromLabel,
    genreFromLabel,
  };
}

export default useDiscovery;
