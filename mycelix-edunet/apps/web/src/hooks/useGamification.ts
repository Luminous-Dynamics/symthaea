// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useGamification Hooks
 *
 * Custom hooks for XP, badges, and streaks from the Gamification zome.
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';
import type {
  LearnerXp,
  XpActivityType,
  LearnerStreak,
  BadgeDefinition,
  EarnedBadge,
  ActionHash,
} from '../types/zomes';

// =============================================================================
// Types
// =============================================================================

export interface UseXpResult {
  /** Current XP record */
  xp: LearnerXp | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh XP from source */
  refresh: () => Promise<void>;
  /** Award XP for an activity */
  awardXp: (amount: number, activityType: XpActivityType, sourceHash: ActionHash, description: string) => Promise<ActionHash | null>;
}

export interface UseBadgesResult {
  /** List of badge definitions */
  definitions: BadgeDefinition[];
  /** Badges earned by the current learner */
  earned: EarnedBadge[];
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh badges from source */
  refresh: () => Promise<void>;
}

export interface UseStreakResult {
  /** Current streak data */
  streak: LearnerStreak | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh streak from source */
  refresh: () => Promise<void>;
  /** Record daily activity to maintain streak */
  recordActivity: () => Promise<LearnerStreak | null>;
  /** Use a streak freeze */
  useFreeze: () => Promise<LearnerStreak | null>;
}

// =============================================================================
// Mock Data
// =============================================================================

const MOCK_XP: LearnerXp = {
  learner: new Uint8Array(32),
  total_xp: BigInt(4250),
  level: 7,
  xp_to_next_level: 750,
  lifetime_xp: BigInt(4250),
  multiplier_permille: 1000,
  last_activity_at: Math.floor(Date.now() / 1000),
  created_at: Math.floor(Date.now() / 1000) - 86400 * 30,
};

const MOCK_STREAK: LearnerStreak = {
  learner: new Uint8Array(32),
  current_streak: 12,
  longest_streak: 21,
  last_activity_date: parseInt(new Date().toISOString().slice(0, 10).replace(/-/g, '')),
  streak_start_date: parseInt(new Date(Date.now() - 86400000 * 12).toISOString().slice(0, 10).replace(/-/g, '')),
  freeze_available: true,
  freeze_count_used: 1,
  streak_broken_count: 3,
};

// =============================================================================
// Hooks
// =============================================================================

export function useXp(): UseXpResult {
  const { client, status, isReal } = useHolochain();
  const [xp, setXp] = useState<LearnerXp | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchXp = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useXp] Fetching from Holochain...');
        const result = await client.gamification.get_my_xp();
        setXp(result);
      } else {
        console.log('[useXp] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 300));
        setXp(MOCK_XP);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch XP';
      console.error('[useXp] Error:', errorMessage);
      setError(errorMessage);
      setXp(MOCK_XP);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchXp();
    }
  }, [status, fetchXp]);

  const awardXp = useCallback(async (
    amount: number,
    activityType: XpActivityType,
    sourceHash: ActionHash,
    description: string,
  ): Promise<ActionHash | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const hash = await client.gamification.award_xp({
          amount, activity_type: activityType, source_hash: sourceHash, description,
        });
        await fetchXp();
        return hash;
      }
      // Mock: update local state
      setXp(prev => prev ? { ...prev, total_xp: prev.total_xp + BigInt(amount) } : prev);
      return new Uint8Array(32);
    } catch (err) {
      console.error('[useXp] Failed to award XP:', err);
      return null;
    }
  }, [client, status, isReal, fetchXp]);

  return { xp, loading, error, isReal, refresh: fetchXp, awardXp };
}

export function useBadges(): UseBadgesResult {
  const { client, status, isReal } = useHolochain();
  const [definitions, setDefinitions] = useState<BadgeDefinition[]>([]);
  const [earned, setEarned] = useState<EarnedBadge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBadges = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useBadges] Fetching from Holochain...');
        const [defs, mine] = await Promise.all([
          client.gamification.get_badge_definitions(),
          client.gamification.get_my_badges(),
        ]);
        setDefinitions(defs);
        setEarned(mine);
      } else {
        console.log('[useBadges] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 300));
        setDefinitions([]);
        setEarned([]);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch badges';
      console.error('[useBadges] Error:', errorMessage);
      setError(errorMessage);
      setDefinitions([]);
      setEarned([]);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchBadges();
    }
  }, [status, fetchBadges]);

  return { definitions, earned, loading, error, isReal, refresh: fetchBadges };
}

export function useStreak(): UseStreakResult {
  const { client, status, isReal } = useHolochain();
  const [streak, setStreak] = useState<LearnerStreak | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStreak = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useStreak] Fetching from Holochain...');
        const result = await client.gamification.get_my_streak();
        setStreak(result);
      } else {
        console.log('[useStreak] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 300));
        setStreak(MOCK_STREAK);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch streak';
      console.error('[useStreak] Error:', errorMessage);
      setError(errorMessage);
      setStreak(MOCK_STREAK);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchStreak();
    }
  }, [status, fetchStreak]);

  const recordActivity = useCallback(async (): Promise<LearnerStreak | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.gamification.record_daily_activity();
        setStreak(updated);
        return updated;
      }
      const updated: LearnerStreak = {
        ...MOCK_STREAK,
        current_streak: (streak?.current_streak ?? 0) + 1,
      };
      setStreak(updated);
      return updated;
    } catch (err) {
      console.error('[useStreak] Failed to record activity:', err);
      return null;
    }
  }, [client, status, isReal, streak]);

  const useFreeze = useCallback(async (): Promise<LearnerStreak | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.gamification.use_streak_freeze();
        setStreak(updated);
        return updated;
      }
      const updated: LearnerStreak = {
        ...(streak ?? MOCK_STREAK),
        freeze_available: false,
        freeze_count_used: (streak?.freeze_count_used ?? 0) + 1,
      };
      setStreak(updated);
      return updated;
    } catch (err) {
      console.error('[useStreak] Failed to use streak freeze:', err);
      return null;
    }
  }, [client, status, isReal, streak]);

  return { streak, loading, error, isReal, refresh: fetchStreak, recordActivity, useFreeze };
}
