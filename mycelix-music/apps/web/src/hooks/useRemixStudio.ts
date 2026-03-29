// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collaborative Remix Studio Hook
 *
 * Provides real-time multi-user collaboration for remixing tracks:
 * - Real-time cursor and selection sync
 * - Version control with branching
 * - Stem sharing marketplace integration
 * - Live jam sessions with WebRTC audio
 *
 * Integrates with mycelix-realtime CRDT infrastructure
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// User/Collaborator
export interface Collaborator {
  id: string;
  name: string;
  avatar?: string;
  color: string;
  cursor?: {
    time: number; // Position in timeline
    track: number; // Track index
  };
  selection?: {
    start: number;
    end: number;
    tracks: number[];
  };
  isOwner: boolean;
  permissions: CollaboratorPermissions;
  joinedAt: number;
  lastActive: number;
}

export interface CollaboratorPermissions {
  canEdit: boolean;
  canAddTracks: boolean;
  canDeleteTracks: boolean;
  canExport: boolean;
  canInvite: boolean;
}

// Project/Session
export interface RemixProject {
  id: string;
  name: string;
  description?: string;
  ownerId: string;
  createdAt: number;
  updatedAt: number;
  tempo: number;
  timeSignature: [number, number];
  duration: number;
  tracks: RemixTrack[];
  markers: Marker[];
  version: ProjectVersion;
  history: ProjectVersion[];
  isPublic: boolean;
  allowForks: boolean;
}

export interface RemixTrack {
  id: string;
  name: string;
  type: 'audio' | 'stem' | 'midi' | 'generated';
  color: string;
  muted: boolean;
  solo: boolean;
  volume: number;
  pan: number;
  clips: Clip[];
  effects: Effect[];
  lockedBy?: string; // Collaborator ID if locked
}

export interface Clip {
  id: string;
  name: string;
  start: number; // Start time in beats
  duration: number;
  offset: number; // Offset within source
  sourceId: string; // Reference to audio source
  fadeIn: number;
  fadeOut: number;
  gain: number;
  pitch: number;
}

export interface Effect {
  id: string;
  type: 'eq' | 'compressor' | 'reverb' | 'delay' | 'filter' | 'distortion';
  enabled: boolean;
  params: Record<string, number>;
}

export interface Marker {
  id: string;
  time: number;
  name: string;
  color: string;
}

// Version Control
export interface ProjectVersion {
  id: string;
  name: string;
  description?: string;
  createdBy: string;
  createdAt: number;
  parentId?: string;
  snapshot: string; // Serialized project state
  tags: string[];
}

// Real-time operations (CRDT-compatible)
export interface Operation {
  id: string;
  type: OperationType;
  timestamp: number;
  userId: string;
  data: Record<string, unknown>;
}

export type OperationType =
  | 'track_add'
  | 'track_remove'
  | 'track_update'
  | 'clip_add'
  | 'clip_remove'
  | 'clip_move'
  | 'clip_resize'
  | 'effect_add'
  | 'effect_remove'
  | 'effect_update'
  | 'marker_add'
  | 'marker_remove'
  | 'tempo_change'
  | 'selection_change'
  | 'cursor_move';

// Chat/Comments
export interface Comment {
  id: string;
  userId: string;
  userName: string;
  content: string;
  timestamp: number;
  timePosition?: number; // Position in timeline
  trackId?: string;
  replies: Comment[];
}

// Stem Marketplace
export interface MarketplaceStem {
  id: string;
  name: string;
  artist: string;
  type: 'vocals' | 'drums' | 'bass' | 'melody' | 'other';
  genre: string;
  tempo: number;
  key: string;
  duration: number;
  price: number; // In platform tokens
  previewUrl: string;
  downloads: number;
  rating: number;
  license: 'creative-commons' | 'royalty-free' | 'exclusive';
}

// Hook state
export interface RemixStudioState {
  project: RemixProject | null;
  collaborators: Collaborator[];
  currentUser: Collaborator | null;
  comments: Comment[];
  isConnected: boolean;
  isSyncing: boolean;
  pendingOperations: Operation[];
  playbackPosition: number;
  isPlaying: boolean;
  selectedTracks: string[];
  selectedClips: string[];
  error: string | null;
}

// Collaborator colors
const COLLABORATOR_COLORS = [
  '#8B5CF6', '#EC4899', '#10B981', '#F59E0B',
  '#3B82F6', '#EF4444', '#06B6D4', '#84CC16',
];

export function useRemixStudio(projectId?: string) {
  const [state, setState] = useState<RemixStudioState>({
    project: null,
    collaborators: [],
    currentUser: null,
    comments: [],
    isConnected: false,
    isSyncing: false,
    pendingOperations: [],
    playbackPosition: 0,
    isPlaying: false,
    selectedTracks: [],
    selectedClips: [],
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const operationQueueRef = useRef<Operation[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Initialize audio context
  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // Connect to collaboration server
  const connect = useCallback(async (sessionId: string, userName: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setState(prev => ({ ...prev, isSyncing: true, error: null }));

    try {
      // In production, this would connect to the actual WebRTC signaling server
      const ws = new WebSocket(`wss://api.mycelix.music/remix/${sessionId}`);

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'join',
          userId: `user-${Date.now()}`,
          userName,
        }));

        setState(prev => ({ ...prev, isConnected: true, isSyncing: false }));
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
      };

      ws.onclose = () => {
        setState(prev => ({ ...prev, isConnected: false }));
      };

      ws.onerror = () => {
        setState(prev => ({
          ...prev,
          error: 'Connection failed',
          isConnected: false,
          isSyncing: false,
        }));
      };

      wsRef.current = ws;
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: 'Failed to connect',
        isSyncing: false,
      }));
    }
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'init':
        setState(prev => ({
          ...prev,
          project: message.project,
          collaborators: message.collaborators,
          currentUser: message.currentUser,
          comments: message.comments || [],
        }));
        break;

      case 'user_joined':
        setState(prev => ({
          ...prev,
          collaborators: [...prev.collaborators, message.user],
        }));
        break;

      case 'user_left':
        setState(prev => ({
          ...prev,
          collaborators: prev.collaborators.filter(c => c.id !== message.userId),
        }));
        break;

      case 'cursor_update':
        setState(prev => ({
          ...prev,
          collaborators: prev.collaborators.map(c =>
            c.id === message.userId ? { ...c, cursor: message.cursor } : c
          ),
        }));
        break;

      case 'operation':
        applyOperation(message.operation);
        break;

      case 'comment':
        setState(prev => ({
          ...prev,
          comments: [...prev.comments, message.comment],
        }));
        break;

      case 'sync':
        setState(prev => ({
          ...prev,
          project: message.project,
        }));
        break;
    }
  }, []);

  // Apply operation to project state
  const applyOperation = useCallback((operation: Operation) => {
    setState(prev => {
      if (!prev.project) return prev;

      const project = { ...prev.project };

      switch (operation.type) {
        case 'track_add':
          project.tracks = [...project.tracks, operation.data.track as RemixTrack];
          break;

        case 'track_remove':
          project.tracks = project.tracks.filter(t => t.id !== operation.data.trackId);
          break;

        case 'track_update':
          project.tracks = project.tracks.map(t =>
            t.id === operation.data.trackId ? { ...t, ...operation.data.updates } : t
          );
          break;

        case 'clip_add':
          project.tracks = project.tracks.map(t =>
            t.id === operation.data.trackId
              ? { ...t, clips: [...t.clips, operation.data.clip as Clip] }
              : t
          );
          break;

        case 'clip_remove':
          project.tracks = project.tracks.map(t =>
            t.id === operation.data.trackId
              ? { ...t, clips: t.clips.filter(c => c.id !== operation.data.clipId) }
              : t
          );
          break;

        case 'clip_move':
          project.tracks = project.tracks.map(t =>
            t.id === operation.data.trackId
              ? {
                  ...t,
                  clips: t.clips.map(c =>
                    c.id === operation.data.clipId
                      ? { ...c, start: operation.data.start as number }
                      : c
                  ),
                }
              : t
          );
          break;

        case 'tempo_change':
          project.tempo = operation.data.tempo as number;
          break;
      }

      project.updatedAt = Date.now();

      return { ...prev, project };
    });
  }, []);

  // Send operation to server
  const sendOperation = useCallback((type: OperationType, data: Record<string, unknown>) => {
    const operation: Operation = {
      id: `op-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      timestamp: Date.now(),
      userId: state.currentUser?.id || '',
      data,
    };

    // Apply locally first (optimistic update)
    applyOperation(operation);

    // Queue for sending
    operationQueueRef.current.push(operation);

    // Send to server
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'operation',
        operation,
      }));
    }
  }, [state.currentUser, applyOperation]);

  // Track operations
  const addTrack = useCallback((track: Partial<RemixTrack>) => {
    const newTrack: RemixTrack = {
      id: `track-${Date.now()}`,
      name: track.name || 'New Track',
      type: track.type || 'audio',
      color: track.color || COLLABORATOR_COLORS[state.project?.tracks.length || 0 % COLLABORATOR_COLORS.length],
      muted: false,
      solo: false,
      volume: 1,
      pan: 0,
      clips: [],
      effects: [],
      ...track,
    };

    sendOperation('track_add', { track: newTrack });
    return newTrack.id;
  }, [state.project, sendOperation]);

  const removeTrack = useCallback((trackId: string) => {
    sendOperation('track_remove', { trackId });
  }, [sendOperation]);

  const updateTrack = useCallback((trackId: string, updates: Partial<RemixTrack>) => {
    sendOperation('track_update', { trackId, updates });
  }, [sendOperation]);

  const lockTrack = useCallback((trackId: string) => {
    if (!state.currentUser) return;
    sendOperation('track_update', {
      trackId,
      updates: { lockedBy: state.currentUser.id },
    });
  }, [state.currentUser, sendOperation]);

  const unlockTrack = useCallback((trackId: string) => {
    sendOperation('track_update', {
      trackId,
      updates: { lockedBy: undefined },
    });
  }, [sendOperation]);

  // Clip operations
  const addClip = useCallback((trackId: string, clip: Partial<Clip>) => {
    const newClip: Clip = {
      id: `clip-${Date.now()}`,
      name: clip.name || 'New Clip',
      start: clip.start || 0,
      duration: clip.duration || 4,
      offset: 0,
      sourceId: clip.sourceId || '',
      fadeIn: 0,
      fadeOut: 0,
      gain: 1,
      pitch: 0,
      ...clip,
    };

    sendOperation('clip_add', { trackId, clip: newClip });
    return newClip.id;
  }, [sendOperation]);

  const removeClip = useCallback((trackId: string, clipId: string) => {
    sendOperation('clip_remove', { trackId, clipId });
  }, [sendOperation]);

  const moveClip = useCallback((trackId: string, clipId: string, start: number) => {
    sendOperation('clip_move', { trackId, clipId, start });
  }, [sendOperation]);

  // Cursor/Selection sync
  const updateCursor = useCallback((time: number, track: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({
      type: 'cursor_update',
      cursor: { time, track },
    }));
  }, []);

  const updateSelection = useCallback((start: number, end: number, tracks: number[]) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({
      type: 'selection_update',
      selection: { start, end, tracks },
    }));
  }, []);

  // Version control
  const createVersion = useCallback((name: string, description?: string) => {
    if (!state.project) return;

    const version: ProjectVersion = {
      id: `version-${Date.now()}`,
      name,
      description,
      createdBy: state.currentUser?.id || '',
      createdAt: Date.now(),
      parentId: state.project.version.id,
      snapshot: JSON.stringify(state.project),
      tags: [],
    };

    setState(prev => ({
      ...prev,
      project: prev.project ? {
        ...prev.project,
        version,
        history: [prev.project.version, ...prev.project.history],
      } : null,
    }));

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'version_create',
        version,
      }));
    }

    return version.id;
  }, [state.project, state.currentUser]);

  const restoreVersion = useCallback((versionId: string) => {
    const version = state.project?.history.find(v => v.id === versionId);
    if (!version) return;

    const restoredProject = JSON.parse(version.snapshot) as RemixProject;

    setState(prev => ({
      ...prev,
      project: restoredProject,
    }));

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'version_restore',
        versionId,
      }));
    }
  }, [state.project]);

  // Comments
  const addComment = useCallback((content: string, timePosition?: number, trackId?: string) => {
    if (!state.currentUser) return;

    const comment: Comment = {
      id: `comment-${Date.now()}`,
      userId: state.currentUser.id,
      userName: state.currentUser.name,
      content,
      timestamp: Date.now(),
      timePosition,
      trackId,
      replies: [],
    };

    setState(prev => ({
      ...prev,
      comments: [...prev.comments, comment],
    }));

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'comment',
        comment,
      }));
    }
  }, [state.currentUser]);

  // Playback
  const setPlaybackPosition = useCallback((position: number) => {
    setState(prev => ({ ...prev, playbackPosition: position }));
  }, []);

  const togglePlayback = useCallback(() => {
    setState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  }, []);

  // Selection
  const selectTracks = useCallback((trackIds: string[]) => {
    setState(prev => ({ ...prev, selectedTracks: trackIds }));
  }, []);

  const selectClips = useCallback((clipIds: string[]) => {
    setState(prev => ({ ...prev, selectedClips: clipIds }));
  }, []);

  // Export
  const exportProject = useCallback(async () => {
    if (!state.project) return;

    // In production, this would render the full mix
    const ctx = audioContextRef.current;
    if (!ctx) return;

    // Create a simple bounce of all tracks
    // This is simplified - real implementation would use OfflineAudioContext
    const duration = state.project.duration * (60 / state.project.tempo);
    const sampleRate = ctx.sampleRate;
    const buffer = ctx.createBuffer(2, Math.ceil(duration * sampleRate), sampleRate);

    return buffer;
  }, [state.project]);

  // Disconnect
  const disconnect = useCallback(() => {
    wsRef.current?.close();
    setState(prev => ({
      ...prev,
      isConnected: false,
      collaborators: [],
    }));
  }, []);

  // Create new project (for demo/local use)
  const createProject = useCallback((name: string, tempo: number = 120) => {
    const project: RemixProject = {
      id: `project-${Date.now()}`,
      name,
      ownerId: 'local-user',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      tempo,
      timeSignature: [4, 4],
      duration: 32, // 32 beats = 8 bars
      tracks: [],
      markers: [],
      version: {
        id: 'v1',
        name: 'Initial',
        createdBy: 'local-user',
        createdAt: Date.now(),
        snapshot: '',
        tags: [],
      },
      history: [],
      isPublic: false,
      allowForks: true,
    };

    const currentUser: Collaborator = {
      id: 'local-user',
      name: 'You',
      color: COLLABORATOR_COLORS[0],
      isOwner: true,
      permissions: {
        canEdit: true,
        canAddTracks: true,
        canDeleteTracks: true,
        canExport: true,
        canInvite: true,
      },
      joinedAt: Date.now(),
      lastActive: Date.now(),
    };

    setState(prev => ({
      ...prev,
      project,
      currentUser,
      collaborators: [currentUser],
    }));

    return project.id;
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    createProject,
    addTrack,
    removeTrack,
    updateTrack,
    lockTrack,
    unlockTrack,
    addClip,
    removeClip,
    moveClip,
    updateCursor,
    updateSelection,
    createVersion,
    restoreVersion,
    addComment,
    setPlaybackPosition,
    togglePlayback,
    selectTracks,
    selectClips,
    exportProject,
  };
}

export default useRemixStudio;
