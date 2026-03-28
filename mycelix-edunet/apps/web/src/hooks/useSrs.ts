// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useSrs Hooks
 *
 * Custom hooks for spaced repetition cards and review sessions from the SRS zome.
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';
import type {
  ReviewCard,
  RecallQuality,
  ReviewSession,
  ActionHash,
} from '../types/zomes';

// =============================================================================
// Types
// =============================================================================

export interface SrsStats {
  cards_total: number;
  cards_due: number;
  reviews_today: number;
  accuracy_permille: number;
}

export interface UseCardsResult {
  /** All due cards */
  dueCards: ReviewCard[];
  /** SRS statistics */
  stats: SrsStats | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh cards from source */
  refresh: () => Promise<void>;
  /** Submit a review for a card */
  reviewCard: (cardHash: ActionHash, quality: RecallQuality) => Promise<ReviewCard | null>;
  /** Create a new card */
  createCard: (deckHash: ActionHash, front: string, back: string, tags: string[]) => Promise<ActionHash | null>;
}

export interface UseSessionsResult {
  /** Current active session hash */
  activeSession: ActionHash | null;
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Start a new review session */
  startSession: (deckHash?: ActionHash) => Promise<ActionHash | null>;
  /** End the current session */
  endSession: () => Promise<ReviewSession | null>;
}

// =============================================================================
// Mock Data
// =============================================================================

const MOCK_STATS: SrsStats = {
  cards_total: 142,
  cards_due: 18,
  reviews_today: 23,
  accuracy_permille: 870,
};

const MOCK_DUE_CARDS: ReviewCard[] = [
  {
    deck_hash: new Uint8Array(32),
    front: 'What is Byzantine Fault Tolerance?',
    back: 'The ability of a distributed system to reach consensus even when some nodes are malicious or faulty.',
    tags: ['distributed-systems', 'consensus'],
    status: 'Review',
    ease_factor_permille: 2500,
    interval_days: 7,
    step_index: 0,
    next_review_at: Math.floor(Date.now() / 1000) - 3600,
    lapses: 0,
    review_count: 5,
    created_at: Math.floor(Date.now() / 1000) - 86400 * 14,
  },
  {
    deck_hash: new Uint8Array(32),
    front: 'Define the CAP theorem.',
    back: 'A distributed system can provide at most two of: Consistency, Availability, Partition tolerance.',
    tags: ['distributed-systems', 'theory'],
    status: 'Review',
    ease_factor_permille: 2100,
    interval_days: 3,
    step_index: 0,
    next_review_at: Math.floor(Date.now() / 1000) - 7200,
    lapses: 1,
    review_count: 8,
    created_at: Math.floor(Date.now() / 1000) - 86400 * 21,
  },
];

// =============================================================================
// Hooks
// =============================================================================

export function useCards(limit: number = 50): UseCardsResult {
  const { client, status, isReal } = useHolochain();
  const [dueCards, setDueCards] = useState<ReviewCard[]>([]);
  const [stats, setStats] = useState<SrsStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCards = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useCards] Fetching from Holochain...');
        const [cards, srsStats] = await Promise.all([
          client.srs.get_due_cards({ limit }),
          client.srs.get_stats(),
        ]);
        setDueCards(cards);
        setStats(srsStats);
      } else {
        console.log('[useCards] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 400));
        setDueCards(MOCK_DUE_CARDS);
        setStats(MOCK_STATS);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch cards';
      console.error('[useCards] Error:', errorMessage);
      setError(errorMessage);
      setDueCards(MOCK_DUE_CARDS);
      setStats(MOCK_STATS);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal, limit]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchCards();
    }
  }, [status, fetchCards]);

  const reviewCard = useCallback(async (
    cardHash: ActionHash,
    quality: RecallQuality,
  ): Promise<ReviewCard | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const updated = await client.srs.review_card({ card_hash: cardHash, quality });
        // Remove reviewed card from due list
        setDueCards(prev => prev.filter(c => c !== prev.find(p => p.front === updated.front)));
        await fetchCards();
        return updated;
      }
      // Mock: remove first card from due list
      setDueCards(prev => prev.slice(1));
      return MOCK_DUE_CARDS[0];
    } catch (err) {
      console.error('[useCards] Failed to review card:', err);
      return null;
    }
  }, [client, status, isReal, fetchCards]);

  const createCard = useCallback(async (
    deckHash: ActionHash,
    front: string,
    back: string,
    tags: string[],
  ): Promise<ActionHash | null> => {
    if (!client || status !== 'connected') return null;

    try {
      if (isReal) {
        const hash = await client.srs.create_card({ deck_hash: deckHash, front, back, tags });
        await fetchCards();
        return hash;
      }
      return new Uint8Array(32);
    } catch (err) {
      console.error('[useCards] Failed to create card:', err);
      return null;
    }
  }, [client, status, isReal, fetchCards]);

  return { dueCards, stats, loading, error, isReal, refresh: fetchCards, reviewCard, createCard };
}

export function useSessions(): UseSessionsResult {
  const { client, status, isReal } = useHolochain();
  const [activeSession, setActiveSession] = useState<ActionHash | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startSession = useCallback(async (deckHash?: ActionHash): Promise<ActionHash | null> => {
    if (!client || status !== 'connected') return null;
    setLoading(true);
    setError(null);

    try {
      if (isReal) {
        const sessionHash = await client.srs.start_session(deckHash);
        setActiveSession(sessionHash);
        return sessionHash;
      }
      const mockHash = new Uint8Array(32);
      setActiveSession(mockHash);
      return mockHash;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start session';
      console.error('[useSessions] Error:', errorMessage);
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  const endSession = useCallback(async (): Promise<ReviewSession | null> => {
    if (!client || status !== 'connected' || !activeSession) return null;
    setLoading(true);
    setError(null);

    try {
      if (isReal) {
        const session = await client.srs.end_session(activeSession);
        setActiveSession(null);
        return session;
      }
      setActiveSession(null);
      return {
        started_at: Math.floor(Date.now() / 1000) - 600,
        ended_at: Math.floor(Date.now() / 1000),
        cards_reviewed: 15,
        cards_new: 3,
        cards_learning: 5,
        accuracy_permille: 850,
        avg_time_ms: 4200,
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to end session';
      console.error('[useSessions] Error:', errorMessage);
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal, activeSession]);

  return { activeSession, loading, error, isReal, startSession, endSession };
}
