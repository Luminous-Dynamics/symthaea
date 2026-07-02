// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type TimeOfDay = 'dawn' | 'morning' | 'afternoon' | 'evening' | 'night' | 'late_night';
export type Weather = 'sunny' | 'cloudy' | 'rainy' | 'stormy' | 'snowy' | 'windy' | 'foggy';
export type Activity = 'waking' | 'working' | 'exercising' | 'commuting' | 'relaxing' | 'sleeping' | 'creating' | 'socializing';
export type Energy = 'low' | 'medium' | 'high' | 'peak';
export type Mood = 'happy' | 'calm' | 'focused' | 'melancholic' | 'energetic' | 'romantic' | 'nostalgic' | 'adventurous';

export interface ContextualProfile {
  id: string;
  name: string;
  icon: string;
  conditions: {
    timeOfDay?: TimeOfDay[];
    weather?: Weather[];
    activity?: Activity[];
    energy?: Energy[];
    mood?: Mood[];
  };
  musicAttributes: {
    tempo: { min: number; max: number };
    energy: number; // 0-1
    valence: number; // 0-1 (negative to positive)
    acousticness?: number;
    instrumentalness?: number;
    genres?: string[];
    tags?: string[];
  };
}

export interface ContextState {
  // Current context
  timeOfDay: TimeOfDay;
  weather: Weather | null;
  activity: Activity;
  energy: Energy;
  mood: Mood;

  // Location (for weather)
  location: { lat: number; lon: number } | null;
  locationName: string | null;

  // Context detection
  autoDetectTime: boolean;
  autoDetectWeather: boolean;
  autoDetectActivity: boolean;

  // Active profile
  activeProfileId: string | null;
  customProfiles: ContextualProfile[];

  // Preferences
  contextualPlayEnabled: boolean;
  smoothTransitions: boolean;
  transitionDuration: number; // seconds

  // Actions
  setTimeOfDay: (time: TimeOfDay) => void;
  setWeather: (weather: Weather) => void;
  setActivity: (activity: Activity) => void;
  setEnergy: (energy: Energy) => void;
  setMood: (mood: Mood) => void;
  setLocation: (lat: number, lon: number, name?: string) => void;
  setAutoDetectTime: (enabled: boolean) => void;
  setAutoDetectWeather: (enabled: boolean) => void;
  setAutoDetectActivity: (enabled: boolean) => void;
  setActiveProfile: (profileId: string | null) => void;
  addCustomProfile: (profile: ContextualProfile) => void;
  removeCustomProfile: (profileId: string) => void;
  toggleContextualPlay: () => void;
  setSmoothTransitions: (enabled: boolean) => void;
  setTransitionDuration: (seconds: number) => void;

  // Computed
  getCurrentMusicAttributes: () => ContextualProfile['musicAttributes'];
}

// Default contextual profiles based on common scenarios
const DEFAULT_PROFILES: ContextualProfile[] = [
  {
    id: 'morning-energy',
    name: 'Morning Energy',
    icon: '🌅',
    conditions: {
      timeOfDay: ['dawn', 'morning'],
      activity: ['waking'],
    },
    musicAttributes: {
      tempo: { min: 100, max: 130 },
      energy: 0.6,
      valence: 0.7,
      acousticness: 0.4,
      genres: ['indie pop', 'electronic', 'upbeat'],
    },
  },
  {
    id: 'deep-focus',
    name: 'Deep Focus',
    icon: '🎯',
    conditions: {
      activity: ['working', 'creating'],
      mood: ['focused'],
    },
    musicAttributes: {
      tempo: { min: 80, max: 120 },
      energy: 0.4,
      valence: 0.5,
      instrumentalness: 0.8,
      genres: ['ambient', 'lo-fi', 'classical', 'electronic'],
    },
  },
  {
    id: 'workout-power',
    name: 'Workout Power',
    icon: '💪',
    conditions: {
      activity: ['exercising'],
      energy: ['high', 'peak'],
    },
    musicAttributes: {
      tempo: { min: 130, max: 180 },
      energy: 0.9,
      valence: 0.8,
      genres: ['electronic', 'hip-hop', 'rock', 'dance'],
    },
  },
  {
    id: 'rainy-day',
    name: 'Rainy Day',
    icon: '🌧️',
    conditions: {
      weather: ['rainy', 'stormy'],
    },
    musicAttributes: {
      tempo: { min: 60, max: 100 },
      energy: 0.3,
      valence: 0.4,
      acousticness: 0.7,
      genres: ['ambient', 'jazz', 'indie folk', 'classical'],
    },
  },
  {
    id: 'evening-wind-down',
    name: 'Evening Wind Down',
    icon: '🌙',
    conditions: {
      timeOfDay: ['evening', 'night'],
      activity: ['relaxing'],
    },
    musicAttributes: {
      tempo: { min: 60, max: 100 },
      energy: 0.3,
      valence: 0.5,
      acousticness: 0.6,
      genres: ['chill', 'ambient', 'jazz', 'neo-soul'],
    },
  },
  {
    id: 'late-night-vibes',
    name: 'Late Night Vibes',
    icon: '🌃',
    conditions: {
      timeOfDay: ['late_night'],
    },
    musicAttributes: {
      tempo: { min: 70, max: 110 },
      energy: 0.4,
      valence: 0.5,
      genres: ['synthwave', 'electronic', 'ambient', 'trip-hop'],
    },
  },
  {
    id: 'sunny-optimism',
    name: 'Sunny Optimism',
    icon: '☀️',
    conditions: {
      weather: ['sunny'],
      mood: ['happy', 'energetic'],
    },
    musicAttributes: {
      tempo: { min: 110, max: 140 },
      energy: 0.7,
      valence: 0.85,
      genres: ['pop', 'indie', 'funk', 'disco'],
    },
  },
  {
    id: 'commute-flow',
    name: 'Commute Flow',
    icon: '🚗',
    conditions: {
      activity: ['commuting'],
    },
    musicAttributes: {
      tempo: { min: 90, max: 130 },
      energy: 0.6,
      valence: 0.6,
      genres: ['indie', 'electronic', 'hip-hop', 'rock'],
    },
  },
  {
    id: 'melancholic-reflection',
    name: 'Melancholic Reflection',
    icon: '💭',
    conditions: {
      mood: ['melancholic', 'nostalgic'],
    },
    musicAttributes: {
      tempo: { min: 60, max: 100 },
      energy: 0.3,
      valence: 0.3,
      acousticness: 0.6,
      genres: ['indie folk', 'ambient', 'classical', 'singer-songwriter'],
    },
  },
  {
    id: 'social-energy',
    name: 'Social Energy',
    icon: '🎉',
    conditions: {
      activity: ['socializing'],
      mood: ['happy', 'energetic'],
    },
    musicAttributes: {
      tempo: { min: 115, max: 140 },
      energy: 0.75,
      valence: 0.8,
      genres: ['pop', 'dance', 'hip-hop', 'funk'],
    },
  },
];

// Helper to determine time of day
function getTimeOfDay(): TimeOfDay {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 7) return 'dawn';
  if (hour >= 7 && hour < 12) return 'morning';
  if (hour >= 12 && hour < 17) return 'afternoon';
  if (hour >= 17 && hour < 21) return 'evening';
  if (hour >= 21 && hour < 24) return 'night';
  return 'late_night'; // 0-5
}

export const useContextStore = create<ContextState>()(
  persist(
    (set, get) => ({
      // Initial state
      timeOfDay: getTimeOfDay(),
      weather: null,
      activity: 'relaxing',
      energy: 'medium',
      mood: 'calm',
      location: null,
      locationName: null,
      autoDetectTime: true,
      autoDetectWeather: true,
      autoDetectActivity: false,
      activeProfileId: null,
      customProfiles: [],
      contextualPlayEnabled: true,
      smoothTransitions: true,
      transitionDuration: 3,

      // Actions
      setTimeOfDay: (time) => set({ timeOfDay: time }),
      setWeather: (weather) => set({ weather }),
      setActivity: (activity) => set({ activity }),
      setEnergy: (energy) => set({ energy }),
      setMood: (mood) => set({ mood }),

      setLocation: (lat, lon, name) => set({
        location: { lat, lon },
        locationName: name || null,
      }),

      setAutoDetectTime: (enabled) => set({ autoDetectTime: enabled }),
      setAutoDetectWeather: (enabled) => set({ autoDetectWeather: enabled }),
      setAutoDetectActivity: (enabled) => set({ autoDetectActivity: enabled }),

      setActiveProfile: (profileId) => set({ activeProfileId: profileId }),

      addCustomProfile: (profile) => set((state) => ({
        customProfiles: [...state.customProfiles, profile],
      })),

      removeCustomProfile: (profileId) => set((state) => ({
        customProfiles: state.customProfiles.filter((p) => p.id !== profileId),
      })),

      toggleContextualPlay: () => set((state) => ({
        contextualPlayEnabled: !state.contextualPlayEnabled,
      })),

      setSmoothTransitions: (enabled) => set({ smoothTransitions: enabled }),
      setTransitionDuration: (seconds) => set({ transitionDuration: seconds }),

      // Get current music attributes based on context
      getCurrentMusicAttributes: () => {
        const state = get();
        const allProfiles = [...DEFAULT_PROFILES, ...state.customProfiles];

        // If a profile is manually selected, use it
        if (state.activeProfileId) {
          const profile = allProfiles.find((p) => p.id === state.activeProfileId);
          if (profile) return profile.musicAttributes;
        }

        // Otherwise, find the best matching profile
        const scoredProfiles = allProfiles.map((profile) => {
          let score = 0;
          const conditions = profile.conditions;

          if (conditions.timeOfDay?.includes(state.timeOfDay)) score += 2;
          if (conditions.weather && state.weather && conditions.weather.includes(state.weather)) score += 3;
          if (conditions.activity?.includes(state.activity)) score += 3;
          if (conditions.energy?.includes(state.energy)) score += 2;
          if (conditions.mood?.includes(state.mood)) score += 2;

          return { profile, score };
        });

        const bestMatch = scoredProfiles.sort((a, b) => b.score - a.score)[0];

        if (bestMatch && bestMatch.score > 0) {
          return bestMatch.profile.musicAttributes;
        }

        // Default neutral attributes
        return {
          tempo: { min: 80, max: 130 },
          energy: 0.5,
          valence: 0.5,
        };
      },
    }),
    {
      name: 'mycelix-context',
      partialize: (state) => ({
        activity: state.activity,
        energy: state.energy,
        mood: state.mood,
        autoDetectTime: state.autoDetectTime,
        autoDetectWeather: state.autoDetectWeather,
        autoDetectActivity: state.autoDetectActivity,
        customProfiles: state.customProfiles,
        contextualPlayEnabled: state.contextualPlayEnabled,
        smoothTransitions: state.smoothTransitions,
        transitionDuration: state.transitionDuration,
      }),
    }
  )
);

// Export default profiles for reference
export { DEFAULT_PROFILES };
