// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useAudioAnalysis Hook
 *
 * Provides audio analysis features powered by the Rust ML backend.
 * Integrates with the Mycelix audio analysis engine for:
 * - Genre classification
 * - Mood detection
 * - BPM estimation
 * - Key detection
 * - Beat tracking
 * - Similar track discovery
 */

import { useCallback, useState } from 'react';
import { rustApi, type AnalysisResult, type SimilarTrack, type SearchQuery } from '@/lib/rustApi';
import { Song } from '@/lib/api';

export interface AudioAnalysisState {
  isAnalyzing: boolean;
  analysis: AnalysisResult | null;
  similarTracks: SimilarTrack[];
  isLoadingSimilar: boolean;
  error: string | null;
}

export interface UseAudioAnalysisReturn extends AudioAnalysisState {
  // Analyze audio file directly
  analyzeFile: (file: File) => Promise<AnalysisResult | null>;

  // Analyze from URL
  analyzeFromUrl: (url: string, filename?: string) => Promise<AnalysisResult | null>;

  // Find similar tracks
  findSimilar: (trackId: string, limit?: number) => Promise<SimilarTrack[]>;

  // Search by audio features
  searchByFeatures: (query: SearchQuery) => Promise<Song[]>;

  // Clear state
  clear: () => void;
}

export function useAudioAnalysis(): UseAudioAnalysisReturn {
  const [state, setState] = useState<AudioAnalysisState>({
    isAnalyzing: false,
    analysis: null,
    similarTracks: [],
    isLoadingSimilar: false,
    error: null,
  });

  /**
   * Analyze an audio file
   */
  const analyzeFile = useCallback(async (file: File): Promise<AnalysisResult | null> => {
    setState((s) => ({ ...s, isAnalyzing: true, error: null }));

    try {
      const analysis = await rustApi.analyzeAudio(file);
      setState((s) => ({
        ...s,
        isAnalyzing: false,
        analysis,
      }));
      return analysis;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Analysis failed';
      setState((s) => ({
        ...s,
        isAnalyzing: false,
        error: message,
      }));
      return null;
    }
  }, []);

  /**
   * Analyze audio from URL
   */
  const analyzeFromUrl = useCallback(
    async (url: string, filename: string = 'audio.mp3'): Promise<AnalysisResult | null> => {
      setState((s) => ({ ...s, isAnalyzing: true, error: null }));

      try {
        // Fetch the audio file
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch audio: ${response.statusText}`);
        }

        const blob = await response.blob();
        const file = new File([blob], filename, { type: blob.type || 'audio/mpeg' });

        const analysis = await rustApi.analyzeAudio(file);
        setState((s) => ({
          ...s,
          isAnalyzing: false,
          analysis,
        }));
        return analysis;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Analysis failed';
        setState((s) => ({
          ...s,
          isAnalyzing: false,
          error: message,
        }));
        return null;
      }
    },
    []
  );

  /**
   * Find similar tracks
   */
  const findSimilar = useCallback(
    async (trackId: string, limit: number = 10): Promise<SimilarTrack[]> => {
      setState((s) => ({ ...s, isLoadingSimilar: true, error: null }));

      try {
        const similar = await rustApi.findSimilar(trackId, limit);
        setState((s) => ({
          ...s,
          isLoadingSimilar: false,
          similarTracks: similar,
        }));
        return similar;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to find similar tracks';
        setState((s) => ({
          ...s,
          isLoadingSimilar: false,
          error: message,
        }));
        return [];
      }
    },
    []
  );

  /**
   * Search by audio features
   */
  const searchByFeatures = useCallback(async (query: SearchQuery): Promise<Song[]> => {
    try {
      const results = await rustApi.search(query);
      // Convert SearchResult to Song format (minimal data)
      return results.map((r) => ({
        id: r.id,
        title: r.title || 'Unknown',
        artist: '',
        artistAddress: '',
        duration: 0,
        coverArt: '',
        audioUrl: '',
        playCount: 0,
        likeCount: 0,
        createdAt: '',
      }));
    } catch (error) {
      console.error('Feature search failed:', error);
      return [];
    }
  }, []);

  /**
   * Clear analysis state
   */
  const clear = useCallback(() => {
    setState({
      isAnalyzing: false,
      analysis: null,
      similarTracks: [],
      isLoadingSimilar: false,
      error: null,
    });
  }, []);

  return {
    ...state,
    analyzeFile,
    analyzeFromUrl,
    findSimilar,
    searchByFeatures,
    clear,
  };
}

// === Utility Functions ===

/**
 * Get mood color based on analysis
 */
export function getMoodColor(mood: string): string {
  const moodColors: Record<string, string> = {
    energetic: '#ff6b6b',
    happy: '#ffd93d',
    calm: '#6bcb77',
    melancholic: '#4d96ff',
    aggressive: '#e74c3c',
    romantic: '#ff69b4',
    dark: '#2c3e50',
    uplifting: '#f39c12',
  };
  return moodColors[mood.toLowerCase()] || '#888888';
}

/**
 * Get key notation (e.g., "Am" -> "A minor")
 */
export function formatKey(key: string): string {
  const keyMap: Record<string, string> = {
    'Am': 'A minor',
    'A': 'A major',
    'Bm': 'B minor',
    'B': 'B major',
    'Cm': 'C minor',
    'C': 'C major',
    'Dm': 'D minor',
    'D': 'D major',
    'Em': 'E minor',
    'E': 'E major',
    'Fm': 'F minor',
    'F': 'F major',
    'Gm': 'G minor',
    'G': 'G major',
  };
  return keyMap[key] || key;
}

/**
 * Format BPM display
 */
export function formatBpm(bpm: number): string {
  const rounded = Math.round(bpm);
  if (rounded < 90) return `${rounded} (Slow)`;
  if (rounded < 120) return `${rounded} (Medium)`;
  if (rounded < 150) return `${rounded} (Fast)`;
  return `${rounded} (Very Fast)`;
}

/**
 * Get genre icon
 */
export function getGenreEmoji(genre: string): string {
  const genreEmojis: Record<string, string> = {
    rock: '\uD83C\uDFB8',
    pop: '\uD83C\uDFA4',
    'hip-hop': '\uD83C\uDFA7',
    electronic: '\uD83D\uDD0A',
    classical: '\uD83C\uDFBB',
    jazz: '\uD83C\uDFB7',
    'r&b': '\uD83D\uDC9C',
    country: '\uD83E\uDD20',
    metal: '\uD83E\uDD18',
    folk: '\uD83C\uDFB6',
    blues: '\uD83C\uDFB9',
    reggae: '\uD83C\uDFDD',
    latin: '\uD83D\uDC83',
    punk: '\uD83D\uDCA5',
    indie: '\uD83C\uDF1F',
  };
  return genreEmojis[genre.toLowerCase()] || '\uD83C\uDFB5';
}

export default useAudioAnalysis;
