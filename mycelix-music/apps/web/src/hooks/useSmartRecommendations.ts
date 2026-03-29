// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Smart Recommendations Hook
 *
 * AI-powered content recommendations:
 * - Sample/stem suggestions for current project
 * - Mood-based music discovery
 * - Collaborative filtering
 * - Context-aware recommendations
 */

import { useState, useCallback, useEffect, useRef } from 'react';

// Types
export interface Track {
  id: string;
  title: string;
  artist: string;
  album?: string;
  duration: number;
  bpm?: number;
  key?: string;
  genre?: string;
  mood?: string[];
  energy?: number;
  danceability?: number;
  coverUrl?: string;
}

export interface Stem {
  id: string;
  name: string;
  type: 'drums' | 'bass' | 'melody' | 'vocals' | 'fx' | 'other';
  bpm: number;
  key: string;
  genre: string;
  mood: string[];
  duration: number;
  previewUrl?: string;
  price?: number;
  artist: string;
}

export interface Sample {
  id: string;
  name: string;
  category: 'oneshot' | 'loop' | 'fx' | 'vocal';
  bpm?: number;
  key?: string;
  genre: string;
  tags: string[];
  duration: number;
  previewUrl?: string;
}

export interface RecommendationContext {
  currentBpm?: number;
  currentKey?: string;
  currentGenre?: string;
  currentMood?: string[];
  projectStems?: string[];  // types of stems already in project
  timeOfDay?: 'morning' | 'afternoon' | 'evening' | 'night';
  activity?: 'working' | 'relaxing' | 'exercising' | 'focusing' | 'partying';
  recentListens?: string[]; // track IDs
}

export interface RecommendationResult<T> {
  items: T[];
  reason: string;
  confidence: number;
}

export interface MoodDescriptor {
  mood: string;
  intensity: number;
  energy: number;
  valence: number;  // negative to positive
}

export interface SmartRecommendationsState {
  isLoading: boolean;
  trackRecommendations: RecommendationResult<Track> | null;
  stemRecommendations: RecommendationResult<Stem> | null;
  sampleRecommendations: RecommendationResult<Sample> | null;
  moodPlaylist: Track[] | null;
  error: string | null;
}

// Mood presets
export const MOOD_PRESETS: Record<string, MoodDescriptor> = {
  happy: { mood: 'happy', intensity: 0.7, energy: 0.8, valence: 0.9 },
  sad: { mood: 'sad', intensity: 0.6, energy: 0.3, valence: 0.2 },
  energetic: { mood: 'energetic', intensity: 0.9, energy: 1.0, valence: 0.7 },
  calm: { mood: 'calm', intensity: 0.3, energy: 0.2, valence: 0.6 },
  dark: { mood: 'dark', intensity: 0.8, energy: 0.5, valence: 0.1 },
  uplifting: { mood: 'uplifting', intensity: 0.8, energy: 0.7, valence: 1.0 },
  melancholic: { mood: 'melancholic', intensity: 0.5, energy: 0.3, valence: 0.3 },
  aggressive: { mood: 'aggressive', intensity: 1.0, energy: 0.9, valence: 0.4 },
  dreamy: { mood: 'dreamy', intensity: 0.4, energy: 0.3, valence: 0.7 },
  mysterious: { mood: 'mysterious', intensity: 0.6, energy: 0.4, valence: 0.5 },
};

// Genre compatibility matrix
const GENRE_COMPATIBILITY: Record<string, string[]> = {
  house: ['techno', 'disco', 'deep-house', 'electro'],
  techno: ['house', 'industrial', 'electro', 'ambient'],
  hiphop: ['trap', 'rnb', 'soul', 'jazz'],
  trap: ['hiphop', 'dubstep', 'edm'],
  lofi: ['jazz', 'hiphop', 'ambient', 'chillhop'],
  dnb: ['jungle', 'dubstep', 'breakbeat'],
  ambient: ['drone', 'electronic', 'neoclassical'],
};

// Key compatibility (circle of fifths)
const KEY_COMPATIBILITY: Record<string, string[]> = {
  'C': ['G', 'F', 'Am', 'Em', 'Dm'],
  'G': ['D', 'C', 'Em', 'Bm', 'Am'],
  'D': ['A', 'G', 'Bm', 'F#m', 'Em'],
  'A': ['E', 'D', 'F#m', 'C#m', 'Bm'],
  'E': ['B', 'A', 'C#m', 'G#m', 'F#m'],
  'Am': ['Em', 'Dm', 'C', 'G', 'F'],
  'Em': ['Bm', 'Am', 'G', 'D', 'C'],
};

export function useSmartRecommendations() {
  const [state, setState] = useState<SmartRecommendationsState>({
    isLoading: false,
    trackRecommendations: null,
    stemRecommendations: null,
    sampleRecommendations: null,
    moodPlaylist: null,
    error: null,
  });

  const contextRef = useRef<RecommendationContext>({});

  /**
   * Update recommendation context
   */
  const updateContext = useCallback((context: Partial<RecommendationContext>) => {
    contextRef.current = { ...contextRef.current, ...context };
  }, []);

  /**
   * Get track recommendations based on context
   */
  const getTrackRecommendations = useCallback(async (
    seedTracks?: string[],
    limit: number = 20
  ): Promise<RecommendationResult<Track>> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const context = contextRef.current;

      // Simulate API call - would use actual recommendation engine
      await new Promise(resolve => setTimeout(resolve, 500));

      // Generate recommendations based on context
      const recommendations = generateTrackRecommendations(context, seedTracks, limit);

      setState(prev => ({
        ...prev,
        isLoading: false,
        trackRecommendations: recommendations,
      }));

      return recommendations;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get recommendations';
      setState(prev => ({ ...prev, isLoading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  /**
   * Get stem recommendations for current project
   */
  const getStemRecommendations = useCallback(async (
    projectContext: {
      bpm: number;
      key: string;
      genre: string;
      existingStems: string[];
    },
    limit: number = 10
  ): Promise<RecommendationResult<Stem>> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await new Promise(resolve => setTimeout(resolve, 400));

      // Find complementary stems
      const missingTypes = getMissingStems(projectContext.existingStems);
      const compatibleKeys = KEY_COMPATIBILITY[projectContext.key] || [projectContext.key];
      const compatibleGenres = GENRE_COMPATIBILITY[projectContext.genre] || [projectContext.genre];

      const stems: Stem[] = missingTypes.slice(0, limit).map((type, i) => ({
        id: `stem-${i}`,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Loop ${i + 1}`,
        type: type as Stem['type'],
        bpm: projectContext.bpm,
        key: compatibleKeys[i % compatibleKeys.length],
        genre: compatibleGenres[i % compatibleGenres.length],
        mood: ['energetic'],
        duration: 8,
        artist: 'Sample Pack',
        price: 4.99,
      }));

      const result: RecommendationResult<Stem> = {
        items: stems,
        reason: `Stems matching ${projectContext.key} at ${projectContext.bpm} BPM`,
        confidence: 0.85,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        stemRecommendations: result,
      }));

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get stem recommendations';
      setState(prev => ({ ...prev, isLoading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  /**
   * Get sample recommendations
   */
  const getSampleRecommendations = useCallback(async (
    category: Sample['category'],
    context?: { genre?: string; bpm?: number; key?: string },
    limit: number = 20
  ): Promise<RecommendationResult<Sample>> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await new Promise(resolve => setTimeout(resolve, 300));

      const samples: Sample[] = Array.from({ length: limit }, (_, i) => ({
        id: `sample-${i}`,
        name: `${category} Sample ${i + 1}`,
        category,
        bpm: context?.bpm,
        key: context?.key,
        genre: context?.genre || 'electronic',
        tags: ['punchy', 'clean', 'processed'],
        duration: category === 'loop' ? 4 : 0.5,
      }));

      const result: RecommendationResult<Sample> = {
        items: samples,
        reason: `${category} samples for ${context?.genre || 'your project'}`,
        confidence: 0.78,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        sampleRecommendations: result,
      }));

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get sample recommendations';
      setState(prev => ({ ...prev, isLoading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  /**
   * Generate mood-based playlist from text description
   */
  const generateMoodPlaylist = useCallback(async (
    description: string,
    length: number = 20
  ): Promise<Track[]> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await new Promise(resolve => setTimeout(resolve, 600));

      // Analyze mood from description
      const moodAnalysis = analyzeMoodFromText(description);

      // Generate playlist based on mood
      const playlist: Track[] = Array.from({ length }, (_, i) => ({
        id: `mood-track-${i}`,
        title: `${moodAnalysis.mood.charAt(0).toUpperCase() + moodAnalysis.mood.slice(1)} Track ${i + 1}`,
        artist: `Artist ${i + 1}`,
        duration: 180 + Math.random() * 120,
        bpm: Math.round(80 + moodAnalysis.energy * 80),
        energy: moodAnalysis.energy,
        mood: [moodAnalysis.mood],
      }));

      setState(prev => ({
        ...prev,
        isLoading: false,
        moodPlaylist: playlist,
      }));

      return playlist;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate playlist';
      setState(prev => ({ ...prev, isLoading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  /**
   * Get "more like this" recommendations
   */
  const getMoreLikeThis = useCallback(async (
    trackId: string,
    limit: number = 10
  ): Promise<Track[]> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await new Promise(resolve => setTimeout(resolve, 400));

      // Would use audio fingerprinting and ML similarity
      const tracks: Track[] = Array.from({ length: limit }, (_, i) => ({
        id: `similar-${i}`,
        title: `Similar Track ${i + 1}`,
        artist: `Artist ${i + 1}`,
        duration: 200,
        genre: 'electronic',
      }));

      setState(prev => ({ ...prev, isLoading: false }));
      return tracks;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed' }));
      throw error;
    }
  }, []);

  /**
   * Get recommendations based on current time and activity
   */
  const getContextualRecommendations = useCallback(async (): Promise<Track[]> => {
    const hour = new Date().getHours();
    let timeOfDay: RecommendationContext['timeOfDay'];
    let suggestedMood: string;

    if (hour >= 6 && hour < 12) {
      timeOfDay = 'morning';
      suggestedMood = 'uplifting';
    } else if (hour >= 12 && hour < 17) {
      timeOfDay = 'afternoon';
      suggestedMood = 'energetic';
    } else if (hour >= 17 && hour < 21) {
      timeOfDay = 'evening';
      suggestedMood = 'calm';
    } else {
      timeOfDay = 'night';
      suggestedMood = 'dreamy';
    }

    updateContext({ timeOfDay });
    return generateMoodPlaylist(suggestedMood, 15);
  }, [updateContext, generateMoodPlaylist]);

  return {
    ...state,
    updateContext,
    getTrackRecommendations,
    getStemRecommendations,
    getSampleRecommendations,
    generateMoodPlaylist,
    getMoreLikeThis,
    getContextualRecommendations,
    moodPresets: MOOD_PRESETS,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function generateTrackRecommendations(
  context: RecommendationContext,
  seedTracks?: string[],
  limit: number = 20
): RecommendationResult<Track> {
  const tracks: Track[] = Array.from({ length: limit }, (_, i) => ({
    id: `rec-${i}`,
    title: `Recommended Track ${i + 1}`,
    artist: `Artist ${i + 1}`,
    duration: 180 + Math.random() * 120,
    bpm: context.currentBpm || 120,
    key: context.currentKey || 'C',
    genre: context.currentGenre || 'electronic',
    mood: context.currentMood || ['energetic'],
    energy: 0.7,
  }));

  return {
    items: tracks,
    reason: context.currentGenre
      ? `Based on your ${context.currentGenre} preferences`
      : 'Personalized for you',
    confidence: 0.82,
  };
}

function getMissingStems(existing: string[]): string[] {
  const allTypes = ['drums', 'bass', 'melody', 'vocals', 'fx', 'other'];
  return allTypes.filter(t => !existing.includes(t));
}

function analyzeMoodFromText(text: string): MoodDescriptor {
  const lowerText = text.toLowerCase();

  // Simple keyword matching - would use NLP in production
  for (const [mood, descriptor] of Object.entries(MOOD_PRESETS)) {
    if (lowerText.includes(mood)) {
      return descriptor;
    }
  }

  // Check for common mood words
  if (lowerText.includes('happy') || lowerText.includes('joy')) {
    return MOOD_PRESETS.happy;
  }
  if (lowerText.includes('sad') || lowerText.includes('lonely')) {
    return MOOD_PRESETS.sad;
  }
  if (lowerText.includes('chill') || lowerText.includes('relax')) {
    return MOOD_PRESETS.calm;
  }
  if (lowerText.includes('pump') || lowerText.includes('workout')) {
    return MOOD_PRESETS.energetic;
  }

  // Default
  return MOOD_PRESETS.calm;
}

export default useSmartRecommendations;
