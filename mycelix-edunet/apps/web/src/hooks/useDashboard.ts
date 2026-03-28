// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useDashboard Hook
 *
 * Custom hook that calls the Integration zome's unified dashboard endpoint
 * to combine data from multiple zomes into a single learner overview.
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';

// =============================================================================
// Types
// =============================================================================

/** Unified dashboard data aggregated from all zomes */
export interface DashboardData {
  // XP & Level (from Gamification)
  totalXp: number;
  level: number;
  xpToNextLevel: number;
  // Streak (from Gamification)
  currentStreak: number;
  longestStreak: number;
  streakFreezeAvailable: boolean;
  // SRS Stats (from SRS)
  cardsDue: number;
  cardsTotal: number;
  reviewsToday: number;
  retentionPermille: number;
  // Skills (from Adaptive)
  totalSkills: number;
  avgMasteryPermille: number;
  skillsInZpd: number;
  // Courses (from Learning)
  enrolledCourses: number;
  completedCourses: number;
  // Governance (from DAO)
  activeProposals: number;
  // Session (from Integration)
  totalStudyMinutesToday: number;
  sessionsToday: number;
}

export interface UseDashboardResult {
  /** Unified dashboard data */
  dashboard: DashboardData | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh dashboard from source */
  refresh: () => Promise<void>;
}

// =============================================================================
// Mock Data
// =============================================================================

const MOCK_DASHBOARD: DashboardData = {
  totalXp: 4250,
  level: 7,
  xpToNextLevel: 750,
  currentStreak: 12,
  longestStreak: 21,
  streakFreezeAvailable: true,
  cardsDue: 18,
  cardsTotal: 142,
  reviewsToday: 23,
  retentionPermille: 870,
  totalSkills: 14,
  avgMasteryPermille: 520,
  skillsInZpd: 6,
  enrolledCourses: 3,
  completedCourses: 1,
  activeProposals: 2,
  totalStudyMinutesToday: 47,
  sessionsToday: 2,
};

// =============================================================================
// Helper
// =============================================================================

/**
 * Map the raw integration zome response to DashboardData.
 * The integration zome returns a loosely-typed aggregate; this normalises it.
 */
function mapToDashboard(raw: Record<string, unknown>): DashboardData {
  return {
    totalXp: Number(raw.total_xp ?? raw.totalXp ?? 0),
    level: Number(raw.level ?? 0),
    xpToNextLevel: Number(raw.xp_to_next_level ?? raw.xpToNextLevel ?? 0),
    currentStreak: Number(raw.current_streak ?? raw.current_streak_days ?? 0),
    longestStreak: Number(raw.longest_streak ?? raw.longest_streak_days ?? 0),
    streakFreezeAvailable: Boolean(raw.streak_freeze_available ?? raw.streakFreezeAvailable ?? false),
    cardsDue: Number(raw.cards_due ?? raw.srs_cards_due ?? 0),
    cardsTotal: Number(raw.cards_total ?? raw.srs_cards_total ?? 0),
    reviewsToday: Number(raw.reviews_today ?? raw.srs_reviews_today ?? 0),
    retentionPermille: Number(raw.retention_permille ?? raw.srs_accuracy_permille ?? 0),
    totalSkills: Number(raw.total_skills ?? 0),
    avgMasteryPermille: Number(raw.avg_mastery_permille ?? 0),
    skillsInZpd: Number(raw.skills_in_zpd ?? 0),
    enrolledCourses: Number(raw.enrolled_courses ?? 0),
    completedCourses: Number(raw.completed_courses ?? 0),
    activeProposals: Number(raw.active_proposals ?? 0),
    totalStudyMinutesToday: Number(raw.total_study_minutes_today ?? raw.total_minutes ?? 0),
    sessionsToday: Number(raw.sessions_today ?? raw.sessions_count ?? 0),
  };
}

// =============================================================================
// Hook
// =============================================================================

export function useDashboard(): UseDashboardResult {
  const { client, status, isReal } = useHolochain();
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useDashboard] Fetching from Holochain...');
        const raw = await client.integration.get_unified_dashboard();
        if (raw && typeof raw === 'object') {
          setDashboard(mapToDashboard(raw as Record<string, unknown>));
        } else {
          console.log('[useDashboard] Empty response, using mock data');
          setDashboard(MOCK_DASHBOARD);
        }
      } else {
        console.log('[useDashboard] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 500));
        setDashboard(MOCK_DASHBOARD);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch dashboard';
      console.error('[useDashboard] Error:', errorMessage);
      setError(errorMessage);
      // Fall back to mock data on error
      setDashboard(MOCK_DASHBOARD);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchDashboard();
    }
  }, [status, fetchDashboard]);

  return { dashboard, loading, error, isReal, refresh: fetchDashboard };
}

// =============================================================================
// Export
// =============================================================================

export default useDashboard;
