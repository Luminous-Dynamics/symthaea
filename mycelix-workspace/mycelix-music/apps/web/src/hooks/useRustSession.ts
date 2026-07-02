// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useRustSession Hook
 *
 * Integrates the Rust backend session management with the player store.
 * Provides synchronized playback across devices and collaborative listening.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlayerStore } from '@/store/playerStore';
import { rustApi, type AnalysisResult } from '@/lib/rustApi';
import {
  MycelixWebSocket,
  getWebSocket,
  type SessionState,
  type WsEventHandlers,
} from '@/lib/websocket';

export interface RustSessionState {
  isConnected: boolean;
  connectionId: string | null;
  sessionId: string | null;
  peers: string[];
  isCreatingSession: boolean;
  isJoiningSession: boolean;
  error: string | null;
}

export interface UseRustSessionReturn extends RustSessionState {
  // Connection
  connect: () => Promise<void>;
  disconnect: () => void;

  // Session management
  createSession: () => Promise<string | null>;
  joinSession: (sessionId: string) => Promise<void>;
  leaveSession: () => void;

  // Playback (synced with Rust backend)
  syncPlay: () => void;
  syncPause: () => void;
  syncSeek: (position: number) => void;
  syncVolume: (volume: number) => void;

  // Analysis
  analyzeCurrentTrack: () => Promise<AnalysisResult | null>;
}

export function useRustSession(): UseRustSessionReturn {
  const wsRef = useRef<MycelixWebSocket | null>(null);

  const [state, setState] = useState<RustSessionState>({
    isConnected: false,
    connectionId: null,
    sessionId: null,
    peers: [],
    isCreatingSession: false,
    isJoiningSession: false,
    error: null,
  });

  // Player store integration
  const {
    currentSong,
    isPlaying,
    position,
    volume,
    play,
    pause,
    seek,
    setVolume,
    syncPosition,
  } = usePlayerStore();

  // Initialize WebSocket event handlers
  const setupHandlers = useCallback(() => {
    const handlers: WsEventHandlers = {
      onConnected: (connectionId) => {
        setState((s) => ({
          ...s,
          isConnected: true,
          connectionId,
          error: null,
        }));
      },

      onSessionCreated: (sessionId) => {
        setState((s) => ({
          ...s,
          sessionId,
          isCreatingSession: false,
        }));
      },

      onSessionJoined: (sessionId, sessionState) => {
        setState((s) => ({
          ...s,
          sessionId,
          isJoiningSession: false,
        }));

        // Sync local state with session state
        if (sessionState.position !== position) {
          syncPosition(sessionState.position);
        }
      },

      onPlaybackStateChanged: (sessionId, playbackState) => {
        if (sessionId === state.sessionId) {
          if (playbackState === 'Playing' && !isPlaying) {
            // Resume local playback
            usePlayerStore.getState().resume();
          } else if (playbackState === 'Paused' && isPlaying) {
            // Pause local playback
            usePlayerStore.getState().pause();
          }
        }
      },

      onPositionChanged: (sessionId, newPosition) => {
        if (sessionId === state.sessionId) {
          const currentPosition = usePlayerStore.getState().position;
          // Only sync if difference is significant (>1 second)
          if (Math.abs(newPosition - currentPosition) > 1) {
            usePlayerStore.getState().seek(newPosition);
          }
        }
      },

      onVolumeChanged: (sessionId, newVolume) => {
        if (sessionId === state.sessionId) {
          usePlayerStore.getState().setVolume(newVolume);
        }
      },

      onPeerJoined: (sessionId, peerId) => {
        if (sessionId === state.sessionId) {
          setState((s) => ({
            ...s,
            peers: [...s.peers, peerId],
          }));
        }
      },

      onPeerLeft: (sessionId, peerId) => {
        if (sessionId === state.sessionId) {
          setState((s) => ({
            ...s,
            peers: s.peers.filter((p) => p !== peerId),
          }));
        }
      },

      onError: (message) => {
        setState((s) => ({
          ...s,
          error: message,
          isCreatingSession: false,
          isJoiningSession: false,
        }));
      },

      onDisconnected: () => {
        setState((s) => ({
          ...s,
          isConnected: false,
          connectionId: null,
        }));
      },

      onReconnecting: (attempt) => {
        console.log(`Reconnecting to Rust backend (attempt ${attempt})...`);
      },
    };

    return handlers;
  }, [isPlaying, position, state.sessionId, syncPosition]);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    try {
      wsRef.current = getWebSocket();
      wsRef.current.on(setupHandlers());
      await wsRef.current.connect();
    } catch (error) {
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }
  }, [setupHandlers]);

  // Disconnect
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
    setState((s) => ({
      ...s,
      isConnected: false,
      connectionId: null,
      sessionId: null,
      peers: [],
    }));
  }, []);

  // Create a new session
  const createSession = useCallback(async (): Promise<string | null> => {
    setState((s) => ({ ...s, isCreatingSession: true, error: null }));

    try {
      // Use REST API for session creation (more reliable than WS for this)
      const session = await rustApi.createSession();
      setState((s) => ({
        ...s,
        sessionId: session.id,
        isCreatingSession: false,
      }));

      // Subscribe to session events via WebSocket
      if (wsRef.current?.isConnected) {
        wsRef.current.subscribeToSession(session.id);
      }

      return session.id;
    } catch (error) {
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Failed to create session',
        isCreatingSession: false,
      }));
      return null;
    }
  }, []);

  // Join an existing session
  const joinSession = useCallback(async (sessionId: string) => {
    setState((s) => ({ ...s, isJoiningSession: true, error: null }));

    try {
      // Get session state via REST
      const sessionDetail = await rustApi.getSession(sessionId);

      setState((s) => ({
        ...s,
        sessionId,
        peers: sessionDetail.peers,
        isJoiningSession: false,
      }));

      // Subscribe to session events via WebSocket
      if (wsRef.current?.isConnected) {
        wsRef.current.joinSession(sessionId);
        wsRef.current.subscribeToSession(sessionId);
      }

      // Sync local state with session state
      syncPosition(sessionDetail.position);
    } catch (error) {
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Failed to join session',
        isJoiningSession: false,
      }));
    }
  }, [syncPosition]);

  // Leave session
  const leaveSession = useCallback(() => {
    if (state.sessionId) {
      rustApi.closeSession(state.sessionId).catch(console.error);
    }
    setState((s) => ({
      ...s,
      sessionId: null,
      peers: [],
    }));
  }, [state.sessionId]);

  // Synced playback controls (broadcast to session)
  const syncPlay = useCallback(() => {
    if (state.sessionId && wsRef.current?.isConnected) {
      wsRef.current.play(state.sessionId);
    }
    usePlayerStore.getState().resume();
  }, [state.sessionId]);

  const syncPause = useCallback(() => {
    if (state.sessionId && wsRef.current?.isConnected) {
      wsRef.current.pause(state.sessionId);
    }
    usePlayerStore.getState().pause();
  }, [state.sessionId]);

  const syncSeek = useCallback((position: number) => {
    if (state.sessionId && wsRef.current?.isConnected) {
      wsRef.current.seek(state.sessionId, position);
    }
    usePlayerStore.getState().seek(position);
  }, [state.sessionId]);

  const syncVolume = useCallback((volume: number) => {
    if (state.sessionId && wsRef.current?.isConnected) {
      wsRef.current.setVolume(state.sessionId, volume);
    }
    usePlayerStore.getState().setVolume(volume);
  }, [state.sessionId]);

  // Analyze current track
  const analyzeCurrentTrack = useCallback(async (): Promise<AnalysisResult | null> => {
    if (!currentSong?.audioUrl) {
      return null;
    }

    try {
      // Fetch the audio file and send for analysis
      const response = await fetch(currentSong.audioUrl);
      const blob = await response.blob();
      const file = new File([blob], `${currentSong.id}.mp3`, { type: 'audio/mpeg' });

      const analysis = await rustApi.analyzeAudio(file);
      return analysis;
    } catch (error) {
      console.error('Analysis failed:', error);
      setState((s) => ({
        ...s,
        error: error instanceof Error ? error.message : 'Analysis failed',
      }));
      return null;
    }
  }, [currentSong]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.disconnect();
      }
    };
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    createSession,
    joinSession,
    leaveSession,
    syncPlay,
    syncPause,
    syncSeek,
    syncVolume,
    analyzeCurrentTrack,
  };
}

export default useRustSession;
