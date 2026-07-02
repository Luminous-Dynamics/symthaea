// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useListeningParty Hook
 *
 * Manages listening party sessions for synchronized group playback.
 * Supports shared playhead, reactions, and peer presence.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlayerStore } from '@/store/playerStore';
import { getWebSocket, type MycelixWebSocket } from '@/lib/websocket';

// === Types ===

export interface Participant {
  id: string;
  displayName: string;
  avatarUrl?: string;
  color: string;
  role: 'host' | 'participant' | 'listener';
  position: number;
  lastSync: number;
  isActive: boolean;
}

export interface PartySettings {
  title: string;
  description?: string;
  maxParticipants: number;
  isPublic: boolean;
  allowReactions: boolean;
  syncTolerance: number; // ms drift allowed before re-sync
}

export interface Reaction {
  id: string;
  emoji: string;
  participantId: string;
  displayName: string;
  timestamp: number;
  x: number;
  y: number;
}

export interface ListeningPartyState {
  partyId: string | null;
  isHost: boolean;
  isConnected: boolean;
  participants: Participant[];
  settings: PartySettings | null;
  reactions: Reaction[];
  syncStatus: 'synced' | 'syncing' | 'drifting';
  error: string | null;
}

// Default party settings
const defaultSettings: PartySettings = {
  title: 'Listening Party',
  maxParticipants: 20,
  isPublic: false,
  allowReactions: true,
  syncTolerance: 500,
};

// Participant colors palette
const COLORS = [
  '#f87171', '#fb923c', '#fbbf24', '#a3e635',
  '#34d399', '#22d3d8', '#38bdf8', '#818cf8',
  '#a78bfa', '#e879f9', '#f472b6', '#fb7185',
];

function getRandomColor(): string {
  return COLORS[Math.floor(Math.random() * COLORS.length)];
}

function generateReactionId(): string {
  return `reaction_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

// === Hook ===

export interface UseListeningPartyReturn extends ListeningPartyState {
  // Party lifecycle
  createParty: (settings?: Partial<PartySettings>) => Promise<string | null>;
  joinParty: (partyId: string, displayName: string) => Promise<boolean>;
  leaveParty: () => void;

  // Host controls
  pauseForAll: () => void;
  playForAll: () => void;
  skipTrack: () => void;
  kickParticipant: (participantId: string) => void;
  updateSettings: (settings: Partial<PartySettings>) => void;

  // Reactions
  sendReaction: (emoji: string) => void;

  // Sync
  requestSync: () => void;
  getShareLink: () => string | null;
}

export function useListeningParty(): UseListeningPartyReturn {
  const wsRef = useRef<MycelixWebSocket | null>(null);
  const syncIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const localIdRef = useRef<string>(`local_${Date.now()}`);

  const [state, setState] = useState<ListeningPartyState>({
    partyId: null,
    isHost: false,
    isConnected: false,
    participants: [],
    settings: null,
    reactions: [],
    syncStatus: 'synced',
    error: null,
  });

  // Player store
  const { position, isPlaying, seek, pause, resume } = usePlayerStore();

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (wsRef.current?.isConnected) return;

    wsRef.current = getWebSocket();

    wsRef.current.on({
      onConnected: () => {
        setState((s) => ({ ...s, isConnected: true, error: null }));
      },
      onDisconnected: () => {
        setState((s) => ({ ...s, isConnected: false }));
      },
      onError: (message) => {
        setState((s) => ({ ...s, error: message }));
      },
    });

    await wsRef.current.connect();
  }, []);

  // Create a new party
  const createParty = useCallback(async (settings?: Partial<PartySettings>): Promise<string | null> => {
    try {
      await connect();

      const partySettings: PartySettings = { ...defaultSettings, ...settings };
      const partyId = `party_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

      // Create host participant
      const hostParticipant: Participant = {
        id: localIdRef.current,
        displayName: 'You (Host)',
        color: getRandomColor(),
        role: 'host',
        position: position,
        lastSync: Date.now(),
        isActive: true,
      };

      setState((s) => ({
        ...s,
        partyId,
        isHost: true,
        settings: partySettings,
        participants: [hostParticipant],
        error: null,
      }));

      // Start sync broadcast
      startSyncBroadcast();

      return partyId;
    } catch (error) {
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Failed to create party',
      }));
      return null;
    }
  }, [connect, position]);

  // Join an existing party
  const joinParty = useCallback(async (partyId: string, displayName: string): Promise<boolean> => {
    try {
      await connect();

      const participant: Participant = {
        id: localIdRef.current,
        displayName,
        color: getRandomColor(),
        role: 'participant',
        position: 0,
        lastSync: Date.now(),
        isActive: true,
      };

      setState((s) => ({
        ...s,
        partyId,
        isHost: false,
        settings: defaultSettings,
        participants: [participant],
        error: null,
      }));

      // Start listening for sync
      startSyncListener();

      return true;
    } catch (error) {
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Failed to join party',
      }));
      return false;
    }
  }, [connect]);

  // Leave party
  const leaveParty = useCallback(() => {
    if (syncIntervalRef.current) {
      clearInterval(syncIntervalRef.current);
    }

    setState({
      partyId: null,
      isHost: false,
      isConnected: false,
      participants: [],
      settings: null,
      reactions: [],
      syncStatus: 'synced',
      error: null,
    });
  }, []);

  // Host controls
  const pauseForAll = useCallback(() => {
    if (!state.isHost) return;
    pause();
    broadcastMessage({ type: 'host_control', action: 'pause_all' });
  }, [state.isHost, pause]);

  const playForAll = useCallback(() => {
    if (!state.isHost) return;
    resume();
    broadcastMessage({ type: 'host_control', action: 'play_all' });
  }, [state.isHost, resume]);

  const skipTrack = useCallback(() => {
    if (!state.isHost) return;
    broadcastMessage({ type: 'host_control', action: 'skip' });
  }, [state.isHost]);

  const kickParticipant = useCallback((participantId: string) => {
    if (!state.isHost) return;
    setState((s) => ({
      ...s,
      participants: s.participants.filter((p) => p.id !== participantId),
    }));
    broadcastMessage({ type: 'host_control', action: 'kick', target: participantId });
  }, [state.isHost]);

  const updateSettings = useCallback((newSettings: Partial<PartySettings>) => {
    if (!state.isHost) return;
    setState((s) => ({
      ...s,
      settings: s.settings ? { ...s.settings, ...newSettings } : null,
    }));
  }, [state.isHost]);

  // Reactions
  const sendReaction = useCallback((emoji: string) => {
    if (!state.settings?.allowReactions) return;

    const reaction: Reaction = {
      id: generateReactionId(),
      emoji,
      participantId: localIdRef.current,
      displayName: state.participants.find((p) => p.id === localIdRef.current)?.displayName || 'You',
      timestamp: Date.now(),
      x: 0.2 + Math.random() * 0.6, // Random x position (20-80%)
      y: 0.8, // Start near bottom
    };

    // Add locally
    setState((s) => ({
      ...s,
      reactions: [...s.reactions.slice(-20), reaction], // Keep last 20
    }));

    // Broadcast
    broadcastMessage({ type: 'reaction', ...reaction });

    // Auto-remove after animation
    setTimeout(() => {
      setState((s) => ({
        ...s,
        reactions: s.reactions.filter((r) => r.id !== reaction.id),
      }));
    }, 3000);
  }, [state.settings, state.participants]);

  // Sync
  const requestSync = useCallback(() => {
    setState((s) => ({ ...s, syncStatus: 'syncing' }));
    // Request current position from host
    broadcastMessage({ type: 'sync_request' });
  }, []);

  // Get share link
  const getShareLink = useCallback((): string | null => {
    if (!state.partyId) return null;
    const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
    return `${baseUrl}/party/${state.partyId}`;
  }, [state.partyId]);

  // Broadcast helper
  const broadcastMessage = useCallback((message: Record<string, unknown>) => {
    if (!wsRef.current?.isConnected || !state.partyId) return;
    // In production, this would send via WebSocket
    console.log('[Party] Broadcasting:', message);
  }, [state.partyId]);

  // Start sync broadcast (host)
  const startSyncBroadcast = useCallback(() => {
    if (syncIntervalRef.current) {
      clearInterval(syncIntervalRef.current);
    }

    syncIntervalRef.current = setInterval(() => {
      const currentPosition = usePlayerStore.getState().position;
      const currentIsPlaying = usePlayerStore.getState().isPlaying;

      broadcastMessage({
        type: 'playhead_sync',
        position: currentPosition,
        isPlaying: currentIsPlaying,
        timestamp: Date.now(),
      });

      // Update local participant position
      setState((s) => ({
        ...s,
        participants: s.participants.map((p) =>
          p.id === localIdRef.current
            ? { ...p, position: currentPosition, lastSync: Date.now() }
            : p
        ),
      }));
    }, 200); // Broadcast every 200ms
  }, [broadcastMessage]);

  // Start sync listener (participant)
  const startSyncListener = useCallback(() => {
    // In production, this would listen for WebSocket messages
    // and sync local playback to host
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
      }
    };
  }, []);

  // Simulate participants for demo
  useEffect(() => {
    if (!state.partyId || state.participants.length >= 3) return;

    // Add demo participants after a delay
    const timeout = setTimeout(() => {
      const demoParticipants: Participant[] = [
        {
          id: 'demo_1',
          displayName: 'Alex',
          avatarUrl: undefined,
          color: '#34d399',
          role: 'participant',
          position: position - 0.5,
          lastSync: Date.now(),
          isActive: true,
        },
        {
          id: 'demo_2',
          displayName: 'Jordan',
          avatarUrl: undefined,
          color: '#f472b6',
          role: 'participant',
          position: position + 0.3,
          lastSync: Date.now(),
          isActive: true,
        },
      ];

      setState((s) => ({
        ...s,
        participants: [...s.participants, ...demoParticipants],
      }));
    }, 2000);

    return () => clearTimeout(timeout);
  }, [state.partyId, state.participants.length, position]);

  return {
    ...state,
    createParty,
    joinParty,
    leaveParty,
    pauseForAll,
    playForAll,
    skipTrack,
    kickParticipant,
    updateSettings,
    sendReaction,
    requestSync,
    getShareLink,
  };
}

export default useListeningParty;
