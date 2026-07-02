// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useMycelium - Unified hook for Mycelium network integration
 *
 * This hook connects all platform features through the mycelium network:
 * - Listening creates connections between listener and artist
 * - Circles create connections between participants
 * - Collaborations create connections between creators
 * - Patronage strengthens bonds
 * - Presence tokens record shared moments
 *
 * The mycelium network is the living substrate that connects all souls.
 */

'use client';

import { useCallback, useEffect, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { usePlayerStore } from '@/store/playerStore';

// ============================================================================
// Types
// ============================================================================

export interface MyceliumConnection {
  id: string;
  sourceId: string;
  sourceType: 'soul' | 'artist' | 'song' | 'genre' | 'event';
  targetId: string;
  targetType: 'soul' | 'artist' | 'song' | 'genre' | 'event';
  connectionType: ConnectionType;
  strength: number; // 0-1
  createdAt: Date;
  lastInteraction: Date;
  metadata?: Record<string, unknown>;
}

export type ConnectionType =
  | 'listen'           // Listener -> Artist/Song
  | 'patron'           // Patron -> Artist
  | 'collaborate'      // Artist -> Artist
  | 'influence'        // Artist -> Artist (one-way)
  | 'circle_member'    // Soul -> Soul (via circles)
  | 'similar_taste'    // Soul -> Soul (algorithmic)
  | 'genre_affinity'   // Soul -> Genre
  | 'event_presence';  // Soul -> Event

export interface MyceliumStats {
  totalConnections: number;
  connectionsByType: Record<ConnectionType, number>;
  strongestConnections: MyceliumConnection[];
  recentConnections: MyceliumConnection[];
  networkReach: number; // How many souls are within 2 degrees
  resonanceFlow: number; // Total resonance flowing through connections
}

export interface NetworkNode {
  id: string;
  type: 'soul' | 'artist' | 'song' | 'genre' | 'event';
  name: string;
  avatar?: string;
  size: number;
  connections: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  type: ConnectionType;
  strength: number;
}

// ============================================================================
// Connection Strength Calculation
// ============================================================================

const CONNECTION_WEIGHTS: Record<ConnectionType, number> = {
  listen: 0.1,
  patron: 0.8,
  collaborate: 0.9,
  influence: 0.5,
  circle_member: 0.4,
  similar_taste: 0.3,
  genre_affinity: 0.2,
  event_presence: 0.6,
};

const STRENGTH_DECAY_DAYS = 30; // Connections decay if no interaction

function calculateStrengthDelta(
  type: ConnectionType,
  duration: number, // seconds of interaction
  isRepeat: boolean
): number {
  const baseWeight = CONNECTION_WEIGHTS[type];
  const durationBonus = Math.min(duration / 3600, 1) * 0.1; // Up to 10% for hour+
  const repeatBonus = isRepeat ? 0.05 : 0.15; // New connections grow faster

  return baseWeight * (1 + durationBonus) * repeatBonus;
}

function applyDecay(strength: number, daysSinceInteraction: number): number {
  if (daysSinceInteraction <= 0) return strength;

  const decayFactor = Math.exp(-daysSinceInteraction / STRENGTH_DECAY_DAYS);
  return strength * decayFactor;
}

// ============================================================================
// Main Hook
// ============================================================================

export function useMycelium(soulId?: string) {
  const queryClient = useQueryClient();
  const { currentSong, position, isPlaying } = usePlayerStore();

  // Fetch user's connections
  const {
    data: connections,
    isLoading: isLoadingConnections,
  } = useQuery({
    queryKey: ['mycelium', 'connections', soulId],
    queryFn: async () => {
      // In production, fetch from API
      // const response = await api.get(`/mycelium/${soulId}/connections`);
      // return response.data;
      return [] as MyceliumConnection[];
    },
    enabled: !!soulId,
  });

  // Fetch network stats
  const {
    data: stats,
    isLoading: isLoadingStats,
  } = useQuery({
    queryKey: ['mycelium', 'stats', soulId],
    queryFn: async () => {
      // In production, fetch from API
      return {
        totalConnections: 0,
        connectionsByType: {} as Record<ConnectionType, number>,
        strongestConnections: [],
        recentConnections: [],
        networkReach: 0,
        resonanceFlow: 0,
      } as MyceliumStats;
    },
    enabled: !!soulId,
  });

  // Record a connection mutation
  const recordConnection = useMutation({
    mutationFn: async (params: {
      targetId: string;
      targetType: MyceliumConnection['targetType'];
      connectionType: ConnectionType;
      duration?: number;
      metadata?: Record<string, unknown>;
    }) => {
      // In production, POST to API
      // await api.post('/mycelium/connections', params);
      console.log('Recording mycelium connection:', params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mycelium'] });
    },
  });

  // ========================================================================
  // Automatic Connection Recording
  // ========================================================================

  // Record listening connection when playing
  useEffect(() => {
    if (!soulId || !currentSong || !isPlaying) return;

    const interval = setInterval(() => {
      // Record listen every 30 seconds of playback
      recordConnection.mutate({
        targetId: currentSong.artistId,
        targetType: 'artist',
        connectionType: 'listen',
        duration: 30,
        metadata: {
          songId: currentSong.id,
          position,
        },
      });
    }, 30000);

    return () => clearInterval(interval);
  }, [soulId, currentSong?.id, isPlaying, position, recordConnection]);

  // ========================================================================
  // Manual Connection Recording Functions
  // ========================================================================

  /**
   * Record that user joined a listening circle
   */
  const recordCircleJoin = useCallback(
    (circleId: string, participantIds: string[]) => {
      if (!soulId) return;

      // Connect to all other participants
      participantIds.forEach((participantId) => {
        if (participantId !== soulId) {
          recordConnection.mutate({
            targetId: participantId,
            targetType: 'soul',
            connectionType: 'circle_member',
            metadata: { circleId },
          });
        }
      });
    },
    [soulId, recordConnection]
  );

  /**
   * Record a collaboration session
   */
  const recordCollaboration = useCallback(
    (projectId: string, collaboratorIds: string[], duration: number) => {
      if (!soulId) return;

      collaboratorIds.forEach((collaboratorId) => {
        if (collaboratorId !== soulId) {
          recordConnection.mutate({
            targetId: collaboratorId,
            targetType: 'artist',
            connectionType: 'collaborate',
            duration,
            metadata: { projectId },
          });
        }
      });
    },
    [soulId, recordConnection]
  );

  /**
   * Record patronage of an artist
   */
  const recordPatronage = useCallback(
    (artistId: string, tier: string, amount: number) => {
      if (!soulId) return;

      recordConnection.mutate({
        targetId: artistId,
        targetType: 'artist',
        connectionType: 'patron',
        metadata: { tier, amount },
      });
    },
    [soulId, recordConnection]
  );

  /**
   * Record presence at an event
   */
  const recordEventPresence = useCallback(
    (eventId: string, duration: number, artistIds: string[]) => {
      if (!soulId) return;

      // Connect to event
      recordConnection.mutate({
        targetId: eventId,
        targetType: 'event',
        connectionType: 'event_presence',
        duration,
      });

      // Connect to performing artists
      artistIds.forEach((artistId) => {
        recordConnection.mutate({
          targetId: artistId,
          targetType: 'artist',
          connectionType: 'listen',
          duration,
          metadata: { eventId },
        });
      });
    },
    [soulId, recordConnection]
  );

  /**
   * Record genre affinity based on listening patterns
   */
  const recordGenreAffinity = useCallback(
    (genreId: string, listenCount: number) => {
      if (!soulId) return;

      recordConnection.mutate({
        targetId: genreId,
        targetType: 'genre',
        connectionType: 'genre_affinity',
        metadata: { listenCount },
      });
    },
    [soulId, recordConnection]
  );

  // ========================================================================
  // Network Graph Data
  // ========================================================================

  /**
   * Get network graph data for visualization
   */
  const getNetworkGraph = useCallback(
    async (depth: number = 2): Promise<{ nodes: NetworkNode[]; edges: NetworkEdge[] }> => {
      // In production, fetch from API with depth parameter
      // const response = await api.get(`/mycelium/${soulId}/graph?depth=${depth}`);
      // return response.data;

      return { nodes: [], edges: [] };
    },
    [soulId]
  );

  /**
   * Find paths between two souls
   */
  const findConnectionPath = useCallback(
    async (targetSoulId: string): Promise<MyceliumConnection[]> => {
      // In production, use graph traversal on backend
      // const response = await api.get(`/mycelium/${soulId}/path/${targetSoulId}`);
      // return response.data;

      return [];
    },
    [soulId]
  );

  /**
   * Get recommendations based on network connections
   */
  const getNetworkRecommendations = useCallback(
    async (type: 'artists' | 'songs' | 'souls'): Promise<string[]> => {
      // In production, use collaborative filtering on network
      // const response = await api.get(`/mycelium/${soulId}/recommendations/${type}`);
      // return response.data;

      return [];
    },
    [soulId]
  );

  // ========================================================================
  // Computed Values
  // ========================================================================

  const strongestArtistConnections = useMemo(() => {
    if (!connections) return [];

    return connections
      .filter((c) => c.targetType === 'artist')
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 10);
  }, [connections]);

  const recentSoulConnections = useMemo(() => {
    if (!connections) return [];

    return connections
      .filter((c) => c.targetType === 'soul')
      .sort((a, b) => new Date(b.lastInteraction).getTime() - new Date(a.lastInteraction).getTime())
      .slice(0, 10);
  }, [connections]);

  const connectionHealth = useMemo(() => {
    if (!connections || connections.length === 0) return 100;

    const avgStrength = connections.reduce((sum, c) => sum + c.strength, 0) / connections.length;
    return Math.round(avgStrength * 100);
  }, [connections]);

  return {
    // Data
    connections,
    stats,
    strongestArtistConnections,
    recentSoulConnections,
    connectionHealth,

    // Loading states
    isLoading: isLoadingConnections || isLoadingStats,
    isRecording: recordConnection.isPending,

    // Actions
    recordCircleJoin,
    recordCollaboration,
    recordPatronage,
    recordEventPresence,
    recordGenreAffinity,

    // Graph
    getNetworkGraph,
    findConnectionPath,
    getNetworkRecommendations,
  };
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for circle-specific mycelium features
 */
export function useCircleMycelium(circleId: string, soulId?: string) {
  const { recordCircleJoin } = useMycelium(soulId);

  const recordCircleSession = useCallback(
    (participantIds: string[], duration: number) => {
      if (!soulId) return;

      recordCircleJoin(circleId, participantIds);

      // Additional circle-specific logic
      console.log(`Circle ${circleId} session: ${participantIds.length} participants, ${duration}s`);
    },
    [circleId, soulId, recordCircleJoin]
  );

  return { recordCircleSession };
}

/**
 * Hook for studio-specific mycelium features
 */
export function useStudioMycelium(projectId: string, soulId?: string) {
  const { recordCollaboration } = useMycelium(soulId);

  const recordStudioSession = useCallback(
    (collaboratorIds: string[], duration: number, contributions: Record<string, number>) => {
      if (!soulId) return;

      recordCollaboration(projectId, collaboratorIds, duration);

      // Weight connections by contribution
      console.log(`Studio ${projectId}: contributions`, contributions);
    },
    [projectId, soulId, recordCollaboration]
  );

  return { recordStudioSession };
}

/**
 * Hook for presence-specific mycelium features
 */
export function usePresenceMycelium(soulId?: string) {
  const { recordEventPresence, recordPatronage } = useMycelium(soulId);

  const recordPresenceToken = useCallback(
    (
      eventId: string,
      presenceType: string,
      duration: number,
      artistIds: string[],
      resonance: number
    ) => {
      if (!soulId) return;

      recordEventPresence(eventId, duration, artistIds);

      // Presence tokens also boost patronage connections
      if (presenceType === 'PATRONAGE_RENEWAL') {
        artistIds.forEach((artistId) => {
          recordPatronage(artistId, 'renewal', resonance);
        });
      }
    },
    [soulId, recordEventPresence, recordPatronage]
  );

  return { recordPresenceToken };
}
