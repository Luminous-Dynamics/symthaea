// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useAdaptive Hooks
 *
 * Custom hooks for learner profiles, skill mastery, and recommendations
 * from the Adaptive Learning zome (BKT, ZPD, VARK).
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';
import type {
  LearnerProfile,
  SkillMastery,
  Recommendation,
  ActionHash,
} from '../types/zomes';

// =============================================================================
// Types
// =============================================================================

export interface UseProfileResult {
  /** Current learner profile */
  profile: LearnerProfile | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh profile from source */
  refresh: () => Promise<void>;
  /** Submit a VARK style assessment */
  submitAssessment: (scores: { visual: number; auditory: number; reading: number; kinesthetic: number }) => Promise<LearnerProfile | null>;
}

export interface UseMasteryResult {
  /** List of skill masteries */
  skills: SkillMastery[];
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh skills from source */
  refresh: () => Promise<void>;
  /** Record a practice attempt */
  recordAttempt: (skillHash: ActionHash, correct: boolean) => Promise<SkillMastery | null>;
}

export interface UseRecommendationsResult {
  /** List of recommendations */
  recommendations: Recommendation[];
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh recommendations from source */
  refresh: () => Promise<void>;
  /** Mark a recommendation as acted on */
  actOn: (recommendationHash: ActionHash) => Promise<Recommendation | null>;
}

// =============================================================================
// Mock Data
// =============================================================================

const MOCK_PROFILE: LearnerProfile = {
  learner: new Uint8Array(32),
  primary_style: 'Visual',
  visual_score_permille: 720,
  auditory_score_permille: 450,
  reading_score_permille: 580,
  kinesthetic_score_permille: 350,
  preferred_content_types: ['Video', 'Interactive'],
  session_duration_preference_minutes: 25,
  daily_goal_minutes: 45,
  difficulty_preference_permille: 600,
  assessment_count: 2,
  last_assessment_at: Math.floor(Date.now() / 1000) - 86400 * 5,
  created_at: Math.floor(Date.now() / 1000) - 86400 * 30,
  updated_at: Math.floor(Date.now() / 1000) - 86400 * 5,
};

const MOCK_SKILLS: SkillMastery[] = [
  {
    learner: new Uint8Array(32),
    skill_hash: new Uint8Array(32),
    skill_name: 'Distributed Consensus',
    mastery_permille: 680,
    level: 'Competent',
    learn_rate_permille: 120,
    guess_permille: 200,
    slip_permille: 80,
    practice_count: 24,
    correct_count: 19,
    last_practiced_at: Math.floor(Date.now() / 1000) - 3600,
    zpd_lower_permille: 550,
    zpd_upper_permille: 800,
    created_at: Math.floor(Date.now() / 1000) - 86400 * 14,
    updated_at: Math.floor(Date.now() / 1000) - 3600,
  },
  {
    learner: new Uint8Array(32),
    skill_hash: new Uint8Array(32),
    skill_name: 'Cryptographic Primitives',
    mastery_permille: 420,
    level: 'Beginner',
    learn_rate_permille: 150,
    guess_permille: 250,
    slip_permille: 100,
    practice_count: 12,
    correct_count: 7,
    last_practiced_at: Math.floor(Date.now() / 1000) - 86400,
    zpd_lower_permille: 300,
    zpd_upper_permille: 600,
    created_at: Math.floor(Date.now() / 1000) - 86400 * 7,
    updated_at: Math.floor(Date.now() / 1000) - 86400,
  },
];

// =============================================================================
// Hooks
// =============================================================================

export function useProfile(): UseProfileResult {
  const { client, status, isReal } = useHolochain();
  const [profile, setProfile] = useState<LearnerProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProfile = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useProfile] Fetching from Holochain...');
        const result = await client.adaptive.get_my_profile();
        setProfile(result);
      } else {
        console.log('[useProfile] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 300));
        setProfile(MOCK_PROFILE);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch profile';
      console.error('[useProfile] Error:', errorMessage);
      setError(errorMessage);
      setProfile(MOCK_PROFILE);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchProfile();
    }
  }, [status, fetchProfile]);

  const submitAssessment = useCallback(async (
    scores: { visual: number; auditory: number; reading: number; kinesthetic: number },
  ): Promise<LearnerProfile | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.adaptive.submit_style_assessment(scores);
        setProfile(updated);
        return updated;
      }
      // Mock: derive primary style from highest score
      const entries = Object.entries(scores) as [string, number][];
      const highest = entries.reduce((a, b) => b[1] > a[1] ? b : a);
      const styleMap: Record<string, LearnerProfile['primary_style']> = {
        visual: 'Visual', auditory: 'Auditory', reading: 'ReadingWriting', kinesthetic: 'Kinesthetic',
      };
      const updated: LearnerProfile = {
        ...MOCK_PROFILE,
        primary_style: styleMap[highest[0]] ?? 'Multimodal',
        visual_score_permille: scores.visual,
        auditory_score_permille: scores.auditory,
        reading_score_permille: scores.reading,
        kinesthetic_score_permille: scores.kinesthetic,
        assessment_count: (profile?.assessment_count ?? 0) + 1,
      };
      setProfile(updated);
      return updated;
    } catch (err) {
      console.error('[useProfile] Failed to submit assessment:', err);
      return null;
    }
  }, [client, status, isReal, profile]);

  return { profile, loading, error, isReal, refresh: fetchProfile, submitAssessment };
}

export function useMastery(): UseMasteryResult {
  const { client, status, isReal } = useHolochain();
  const [skills, setSkills] = useState<SkillMastery[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSkills = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useMastery] Fetching from Holochain...');
        const result = await client.adaptive.get_my_skills();
        setSkills(result);
      } else {
        console.log('[useMastery] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 300));
        setSkills(MOCK_SKILLS);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch skills';
      console.error('[useMastery] Error:', errorMessage);
      setError(errorMessage);
      setSkills(MOCK_SKILLS);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchSkills();
    }
  }, [status, fetchSkills]);

  const recordAttempt = useCallback(async (
    skillHash: ActionHash,
    correct: boolean,
  ): Promise<SkillMastery | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.adaptive.update_skill_mastery({ skill_hash: skillHash, correct });
        await fetchSkills();
        return updated;
      }
      // Mock: update the matching skill
      setSkills(prev => prev.map(s => {
        // Simple mock: increment counts
        return {
          ...s,
          practice_count: s.practice_count + 1,
          correct_count: correct ? s.correct_count + 1 : s.correct_count,
          mastery_permille: Math.min(1000, s.mastery_permille + (correct ? 20 : -10)),
        };
      }));
      return MOCK_SKILLS[0];
    } catch (err) {
      console.error('[useMastery] Failed to record attempt:', err);
      return null;
    }
  }, [client, status, isReal, fetchSkills]);

  return { skills, loading, error, isReal, refresh: fetchSkills, recordAttempt };
}

export function useRecommendations(limit: number = 10): UseRecommendationsResult {
  const { client, status, isReal } = useHolochain();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendations = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useRecommendations] Fetching from Holochain...');
        const result = await client.adaptive.get_recommendations(limit);
        setRecommendations(result);
      } else {
        console.log('[useRecommendations] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 400));
        setRecommendations([]);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch recommendations';
      console.error('[useRecommendations] Error:', errorMessage);
      setError(errorMessage);
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal, limit]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchRecommendations();
    }
  }, [status, fetchRecommendations]);

  const actOn = useCallback(async (recommendationHash: ActionHash): Promise<Recommendation | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.adaptive.act_on_recommendation(recommendationHash);
        await fetchRecommendations();
        return updated;
      }
      // Mock: mark as acted on
      setRecommendations(prev => prev.map(r => ({ ...r, acted_on: true })));
      return null;
    } catch (err) {
      console.error('[useRecommendations] Failed to act on recommendation:', err);
      return null;
    }
  }, [client, status, isReal, fetchRecommendations]);

  return { recommendations, loading, error, isReal, refresh: fetchRecommendations, actOn };
}
