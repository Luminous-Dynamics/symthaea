// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Zustand Store Index
 *
 * Global state management with:
 * - Modular store slices
 * - Persistence to IndexedDB
 * - Devtools integration
 * - Optimistic updates
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { createJSONStorage } from 'zustand/middleware';

// ==================== Types ====================

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  subscription: 'free' | 'pro' | 'artist';
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: string;
  language: string;
  audioQuality: 'low' | 'medium' | 'high' | 'lossless';
  autoPlay: boolean;
  crossfade: number;
  notifications: boolean;
}

export interface Track {
  id: string;
  title: string;
  artist: string;
  artistId: string;
  album?: string;
  albumId?: string;
  duration: number;
  coverUrl?: string;
  audioUrl: string;
  waveformData?: number[];
  bpm?: number;
  key?: string;
  genre?: string;
  plays: number;
  likes: number;
  isLiked?: boolean;
  addedAt?: Date;
}

export interface Playlist {
  id: string;
  name: string;
  description?: string;
  coverUrl?: string;
  trackIds: string[];
  ownerId: string;
  isPublic: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface PlayerState {
  currentTrack: Track | null;
  queue: Track[];
  queueIndex: number;
  isPlaying: boolean;
  isShuffle: boolean;
  repeatMode: 'off' | 'all' | 'one';
  volume: number;
  muted: boolean;
  position: number;
  duration: number;
  isLoading: boolean;
  error: string | null;
}

export interface LibraryState {
  tracks: Map<string, Track>;
  playlists: Map<string, Playlist>;
  likedTrackIds: Set<string>;
  recentlyPlayed: string[];
  downloadedTrackIds: Set<string>;
}

export interface UIState {
  sidebarOpen: boolean;
  sidebarWidth: number;
  currentView: string;
  modal: { type: string; props?: Record<string, unknown> } | null;
  toast: { message: string; type: 'success' | 'error' | 'info' } | null;
  isFullscreen: boolean;
  showLyrics: boolean;
  showQueue: boolean;
}

export interface ProjectState {
  currentProjectId: string | null;
  projects: Map<string, Project>;
  unsavedChanges: boolean;
  clipboard: ClipboardItem[];
}

export interface Project {
  id: string;
  name: string;
  tempo: number;
  timeSignature: [number, number];
  tracks: ProjectTrack[];
  clips: ProjectClip[];
  markers: ProjectMarker[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ProjectTrack {
  id: string;
  name: string;
  type: 'audio' | 'midi' | 'aux';
  color: string;
  volume: number;
  pan: number;
  mute: boolean;
  solo: boolean;
  armed: boolean;
}

export interface ProjectClip {
  id: string;
  trackId: string;
  start: number;
  duration: number;
  name: string;
  color: string;
}

export interface ProjectMarker {
  id: string;
  time: number;
  name: string;
  color: string;
}

export interface ClipboardItem {
  type: 'clip' | 'track' | 'notes';
  data: unknown;
}

// ==================== Combined Store Type ====================

export interface AppState {
  // User
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  updatePreferences: (prefs: Partial<UserPreferences>) => void;
  logout: () => void;

  // Player
  player: PlayerState;
  playTrack: (track: Track) => void;
  pauseTrack: () => void;
  resumeTrack: () => void;
  nextTrack: () => void;
  previousTrack: () => void;
  seekTo: (position: number) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;
  toggleShuffle: () => void;
  toggleRepeat: () => void;
  addToQueue: (tracks: Track[]) => void;
  clearQueue: () => void;
  playQueue: (tracks: Track[], startIndex?: number) => void;

  // Library
  library: LibraryState;
  addTrack: (track: Track) => void;
  removeTrack: (trackId: string) => void;
  likeTrack: (trackId: string) => void;
  unlikeTrack: (trackId: string) => void;
  addPlaylist: (playlist: Playlist) => void;
  updatePlaylist: (playlistId: string, updates: Partial<Playlist>) => void;
  deletePlaylist: (playlistId: string) => void;
  addTrackToPlaylist: (playlistId: string, trackId: string) => void;
  removeTrackFromPlaylist: (playlistId: string, trackId: string) => void;
  markAsDownloaded: (trackId: string) => void;
  addToRecentlyPlayed: (trackId: string) => void;

  // UI
  ui: UIState;
  setSidebarOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  setCurrentView: (view: string) => void;
  showModal: (type: string, props?: Record<string, unknown>) => void;
  hideModal: () => void;
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void;
  hideToast: () => void;
  toggleFullscreen: () => void;
  toggleLyrics: () => void;
  toggleQueue: () => void;

  // Project (DAW)
  project: ProjectState;
  createProject: (name: string) => string;
  loadProject: (projectId: string) => void;
  saveProject: () => void;
  updateProject: (updates: Partial<Project>) => void;
  addProjectTrack: (track: ProjectTrack) => void;
  updateProjectTrack: (trackId: string, updates: Partial<ProjectTrack>) => void;
  deleteProjectTrack: (trackId: string) => void;
  addClip: (clip: ProjectClip) => void;
  updateClip: (clipId: string, updates: Partial<ProjectClip>) => void;
  deleteClip: (clipId: string) => void;
  copyToClipboard: (items: ClipboardItem[]) => void;
  pasteFromClipboard: () => ClipboardItem[];
  setUnsavedChanges: (hasChanges: boolean) => void;

  // Sync
  syncStatus: 'idle' | 'syncing' | 'error';
  lastSyncTime: Date | null;
  pendingOperations: PendingOperation[];
  addPendingOperation: (op: Omit<PendingOperation, 'id' | 'timestamp'>) => void;
  clearPendingOperations: () => void;
}

export interface PendingOperation {
  id: string;
  type: 'create' | 'update' | 'delete';
  entity: string;
  entityId: string;
  data?: unknown;
  timestamp: Date;
}

// ==================== IndexedDB Storage ====================

const indexedDBStorage = {
  name: 'mycelix-storage',

  getItem: async (name: string): Promise<string | null> => {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('store', 'readonly');
        const store = tx.objectStore('store');
        const request = store.get(name);
        request.onsuccess = () => resolve(request.result?.value || null);
        request.onerror = () => reject(request.error);
      });
    } catch {
      return localStorage.getItem(name);
    }
  },

  setItem: async (name: string, value: string): Promise<void> => {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('store', 'readwrite');
        const store = tx.objectStore('store');
        const request = store.put({ key: name, value });
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch {
      localStorage.setItem(name, value);
    }
  },

  removeItem: async (name: string): Promise<void> => {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction('store', 'readwrite');
        const store = tx.objectStore('store');
        const request = store.delete(name);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch {
      localStorage.removeItem(name);
    }
  },
};

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-store', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('store')) {
        db.createObjectStore('store', { keyPath: 'key' });
      }
    };
  });
}

// ==================== Initial States ====================

const initialPlayerState: PlayerState = {
  currentTrack: null,
  queue: [],
  queueIndex: -1,
  isPlaying: false,
  isShuffle: false,
  repeatMode: 'off',
  volume: 0.8,
  muted: false,
  position: 0,
  duration: 0,
  isLoading: false,
  error: null,
};

const initialLibraryState: LibraryState = {
  tracks: new Map(),
  playlists: new Map(),
  likedTrackIds: new Set(),
  recentlyPlayed: [],
  downloadedTrackIds: new Set(),
};

const initialUIState: UIState = {
  sidebarOpen: true,
  sidebarWidth: 240,
  currentView: 'home',
  modal: null,
  toast: null,
  isFullscreen: false,
  showLyrics: false,
  showQueue: false,
};

const initialProjectState: ProjectState = {
  currentProjectId: null,
  projects: new Map(),
  unsavedChanges: false,
  clipboard: [],
};

// ==================== Store Creation ====================

export const useAppStore = create<AppState>()(
  devtools(
    subscribeWithSelector(
      persist(
        immer((set, get) => ({
          // ==================== User ====================
          user: null,
          isAuthenticated: false,

          setUser: (user) => set((state) => {
            state.user = user;
            state.isAuthenticated = !!user;
          }),

          updatePreferences: (prefs) => set((state) => {
            if (state.user) {
              state.user.preferences = { ...state.user.preferences, ...prefs };
            }
          }),

          logout: () => set((state) => {
            state.user = null;
            state.isAuthenticated = false;
            state.player = initialPlayerState;
          }),

          // ==================== Player ====================
          player: initialPlayerState,

          playTrack: (track) => set((state) => {
            state.player.currentTrack = track;
            state.player.isPlaying = true;
            state.player.position = 0;
            state.player.duration = track.duration;
            state.player.isLoading = true;

            // Add to recently played
            const recentIndex = state.library.recentlyPlayed.indexOf(track.id);
            if (recentIndex > -1) {
              state.library.recentlyPlayed.splice(recentIndex, 1);
            }
            state.library.recentlyPlayed.unshift(track.id);
            state.library.recentlyPlayed = state.library.recentlyPlayed.slice(0, 50);
          }),

          pauseTrack: () => set((state) => {
            state.player.isPlaying = false;
          }),

          resumeTrack: () => set((state) => {
            if (state.player.currentTrack) {
              state.player.isPlaying = true;
            }
          }),

          nextTrack: () => set((state) => {
            const { queue, queueIndex, isShuffle, repeatMode } = state.player;
            if (queue.length === 0) return;

            let nextIndex: number;

            if (isShuffle) {
              nextIndex = Math.floor(Math.random() * queue.length);
            } else if (queueIndex < queue.length - 1) {
              nextIndex = queueIndex + 1;
            } else if (repeatMode === 'all') {
              nextIndex = 0;
            } else {
              state.player.isPlaying = false;
              return;
            }

            state.player.queueIndex = nextIndex;
            state.player.currentTrack = queue[nextIndex];
            state.player.position = 0;
            state.player.isLoading = true;
          }),

          previousTrack: () => set((state) => {
            const { queue, queueIndex, position } = state.player;

            // If more than 3 seconds in, restart current track
            if (position > 3) {
              state.player.position = 0;
              return;
            }

            if (queue.length === 0 || queueIndex <= 0) return;

            const prevIndex = queueIndex - 1;
            state.player.queueIndex = prevIndex;
            state.player.currentTrack = queue[prevIndex];
            state.player.position = 0;
            state.player.isLoading = true;
          }),

          seekTo: (position) => set((state) => {
            state.player.position = Math.max(0, Math.min(position, state.player.duration));
          }),

          setVolume: (volume) => set((state) => {
            state.player.volume = Math.max(0, Math.min(1, volume));
            if (volume > 0) state.player.muted = false;
          }),

          toggleMute: () => set((state) => {
            state.player.muted = !state.player.muted;
          }),

          toggleShuffle: () => set((state) => {
            state.player.isShuffle = !state.player.isShuffle;
          }),

          toggleRepeat: () => set((state) => {
            const modes: PlayerState['repeatMode'][] = ['off', 'all', 'one'];
            const currentIndex = modes.indexOf(state.player.repeatMode);
            state.player.repeatMode = modes[(currentIndex + 1) % modes.length];
          }),

          addToQueue: (tracks) => set((state) => {
            state.player.queue.push(...tracks);
          }),

          clearQueue: () => set((state) => {
            state.player.queue = [];
            state.player.queueIndex = -1;
          }),

          playQueue: (tracks, startIndex = 0) => set((state) => {
            state.player.queue = tracks;
            state.player.queueIndex = startIndex;
            state.player.currentTrack = tracks[startIndex];
            state.player.isPlaying = true;
            state.player.position = 0;
            state.player.isLoading = true;
          }),

          // ==================== Library ====================
          library: initialLibraryState,

          addTrack: (track) => set((state) => {
            state.library.tracks.set(track.id, track);
          }),

          removeTrack: (trackId) => set((state) => {
            state.library.tracks.delete(trackId);
            state.library.likedTrackIds.delete(trackId);
            state.library.downloadedTrackIds.delete(trackId);
          }),

          likeTrack: (trackId) => set((state) => {
            state.library.likedTrackIds.add(trackId);
            const track = state.library.tracks.get(trackId);
            if (track) {
              track.isLiked = true;
              track.likes++;
            }

            // Add pending operation for sync
            state.pendingOperations.push({
              id: `op-${Date.now()}`,
              type: 'update',
              entity: 'track',
              entityId: trackId,
              data: { isLiked: true },
              timestamp: new Date(),
            });
          }),

          unlikeTrack: (trackId) => set((state) => {
            state.library.likedTrackIds.delete(trackId);
            const track = state.library.tracks.get(trackId);
            if (track) {
              track.isLiked = false;
              track.likes = Math.max(0, track.likes - 1);
            }

            state.pendingOperations.push({
              id: `op-${Date.now()}`,
              type: 'update',
              entity: 'track',
              entityId: trackId,
              data: { isLiked: false },
              timestamp: new Date(),
            });
          }),

          addPlaylist: (playlist) => set((state) => {
            state.library.playlists.set(playlist.id, playlist);

            state.pendingOperations.push({
              id: `op-${Date.now()}`,
              type: 'create',
              entity: 'playlist',
              entityId: playlist.id,
              data: playlist,
              timestamp: new Date(),
            });
          }),

          updatePlaylist: (playlistId, updates) => set((state) => {
            const playlist = state.library.playlists.get(playlistId);
            if (playlist) {
              Object.assign(playlist, updates, { updatedAt: new Date() });

              state.pendingOperations.push({
                id: `op-${Date.now()}`,
                type: 'update',
                entity: 'playlist',
                entityId: playlistId,
                data: updates,
                timestamp: new Date(),
              });
            }
          }),

          deletePlaylist: (playlistId) => set((state) => {
            state.library.playlists.delete(playlistId);

            state.pendingOperations.push({
              id: `op-${Date.now()}`,
              type: 'delete',
              entity: 'playlist',
              entityId: playlistId,
              timestamp: new Date(),
            });
          }),

          addTrackToPlaylist: (playlistId, trackId) => set((state) => {
            const playlist = state.library.playlists.get(playlistId);
            if (playlist && !playlist.trackIds.includes(trackId)) {
              playlist.trackIds.push(trackId);
              playlist.updatedAt = new Date();
            }
          }),

          removeTrackFromPlaylist: (playlistId, trackId) => set((state) => {
            const playlist = state.library.playlists.get(playlistId);
            if (playlist) {
              playlist.trackIds = playlist.trackIds.filter(id => id !== trackId);
              playlist.updatedAt = new Date();
            }
          }),

          markAsDownloaded: (trackId) => set((state) => {
            state.library.downloadedTrackIds.add(trackId);
          }),

          addToRecentlyPlayed: (trackId) => set((state) => {
            const index = state.library.recentlyPlayed.indexOf(trackId);
            if (index > -1) {
              state.library.recentlyPlayed.splice(index, 1);
            }
            state.library.recentlyPlayed.unshift(trackId);
            state.library.recentlyPlayed = state.library.recentlyPlayed.slice(0, 50);
          }),

          // ==================== UI ====================
          ui: initialUIState,

          setSidebarOpen: (open) => set((state) => {
            state.ui.sidebarOpen = open;
          }),

          setSidebarWidth: (width) => set((state) => {
            state.ui.sidebarWidth = Math.max(180, Math.min(400, width));
          }),

          setCurrentView: (view) => set((state) => {
            state.ui.currentView = view;
          }),

          showModal: (type, props) => set((state) => {
            state.ui.modal = { type, props };
          }),

          hideModal: () => set((state) => {
            state.ui.modal = null;
          }),

          showToast: (message, type = 'info') => set((state) => {
            state.ui.toast = { message, type };
          }),

          hideToast: () => set((state) => {
            state.ui.toast = null;
          }),

          toggleFullscreen: () => set((state) => {
            state.ui.isFullscreen = !state.ui.isFullscreen;
          }),

          toggleLyrics: () => set((state) => {
            state.ui.showLyrics = !state.ui.showLyrics;
          }),

          toggleQueue: () => set((state) => {
            state.ui.showQueue = !state.ui.showQueue;
          }),

          // ==================== Project (DAW) ====================
          project: initialProjectState,

          createProject: (name) => {
            const id = `project-${Date.now()}`;
            set((state) => {
              const project: Project = {
                id,
                name,
                tempo: 120,
                timeSignature: [4, 4],
                tracks: [],
                clips: [],
                markers: [],
                createdAt: new Date(),
                updatedAt: new Date(),
              };
              state.project.projects.set(id, project);
              state.project.currentProjectId = id;
              state.project.unsavedChanges = false;
            });
            return id;
          },

          loadProject: (projectId) => set((state) => {
            if (state.project.projects.has(projectId)) {
              state.project.currentProjectId = projectId;
              state.project.unsavedChanges = false;
            }
          }),

          saveProject: () => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                project.updatedAt = new Date();
                state.project.unsavedChanges = false;

                state.pendingOperations.push({
                  id: `op-${Date.now()}`,
                  type: 'update',
                  entity: 'project',
                  entityId: projectId,
                  data: project,
                  timestamp: new Date(),
                });
              }
            }
          }),

          updateProject: (updates) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                Object.assign(project, updates);
                state.project.unsavedChanges = true;
              }
            }
          }),

          addProjectTrack: (track) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                project.tracks.push(track);
                state.project.unsavedChanges = true;
              }
            }
          }),

          updateProjectTrack: (trackId, updates) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                const track = project.tracks.find(t => t.id === trackId);
                if (track) {
                  Object.assign(track, updates);
                  state.project.unsavedChanges = true;
                }
              }
            }
          }),

          deleteProjectTrack: (trackId) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                project.tracks = project.tracks.filter(t => t.id !== trackId);
                project.clips = project.clips.filter(c => c.trackId !== trackId);
                state.project.unsavedChanges = true;
              }
            }
          }),

          addClip: (clip) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                project.clips.push(clip);
                state.project.unsavedChanges = true;
              }
            }
          }),

          updateClip: (clipId, updates) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                const clip = project.clips.find(c => c.id === clipId);
                if (clip) {
                  Object.assign(clip, updates);
                  state.project.unsavedChanges = true;
                }
              }
            }
          }),

          deleteClip: (clipId) => set((state) => {
            const projectId = state.project.currentProjectId;
            if (projectId) {
              const project = state.project.projects.get(projectId);
              if (project) {
                project.clips = project.clips.filter(c => c.id !== clipId);
                state.project.unsavedChanges = true;
              }
            }
          }),

          copyToClipboard: (items) => set((state) => {
            state.project.clipboard = items;
          }),

          pasteFromClipboard: () => {
            return get().project.clipboard;
          },

          setUnsavedChanges: (hasChanges) => set((state) => {
            state.project.unsavedChanges = hasChanges;
          }),

          // ==================== Sync ====================
          syncStatus: 'idle',
          lastSyncTime: null,
          pendingOperations: [],

          addPendingOperation: (op) => set((state) => {
            state.pendingOperations.push({
              ...op,
              id: `op-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              timestamp: new Date(),
            });
          }),

          clearPendingOperations: () => set((state) => {
            state.pendingOperations = [];
          }),
        })),
        {
          name: 'mycelix-app-store',
          storage: createJSONStorage(() => indexedDBStorage),
          partialize: (state) => ({
            user: state.user,
            player: {
              volume: state.player.volume,
              muted: state.player.muted,
              isShuffle: state.player.isShuffle,
              repeatMode: state.player.repeatMode,
            },
            library: {
              likedTrackIds: Array.from(state.library.likedTrackIds),
              recentlyPlayed: state.library.recentlyPlayed,
              downloadedTrackIds: Array.from(state.library.downloadedTrackIds),
            },
            ui: {
              sidebarOpen: state.ui.sidebarOpen,
              sidebarWidth: state.ui.sidebarWidth,
            },
            pendingOperations: state.pendingOperations,
          }),
          onRehydrateStorage: () => (state) => {
            if (state) {
              // Convert arrays back to Sets
              if (Array.isArray((state.library as unknown as { likedTrackIds: string[] }).likedTrackIds)) {
                state.library.likedTrackIds = new Set((state.library as unknown as { likedTrackIds: string[] }).likedTrackIds);
              }
              if (Array.isArray((state.library as unknown as { downloadedTrackIds: string[] }).downloadedTrackIds)) {
                state.library.downloadedTrackIds = new Set((state.library as unknown as { downloadedTrackIds: string[] }).downloadedTrackIds);
              }
            }
          },
        }
      )
    ),
    { name: 'Mycelix Store' }
  )
);

// ==================== Selectors ====================

export const selectCurrentTrack = (state: AppState) => state.player.currentTrack;
export const selectIsPlaying = (state: AppState) => state.player.isPlaying;
export const selectVolume = (state: AppState) => state.player.volume;
export const selectQueue = (state: AppState) => state.player.queue;
export const selectUser = (state: AppState) => state.user;
export const selectIsAuthenticated = (state: AppState) => state.isAuthenticated;
export const selectCurrentView = (state: AppState) => state.ui.currentView;
export const selectCurrentProject = (state: AppState) => {
  const id = state.project.currentProjectId;
  return id ? state.project.projects.get(id) : null;
};
export const selectHasUnsavedChanges = (state: AppState) => state.project.unsavedChanges;
export const selectPendingOperations = (state: AppState) => state.pendingOperations;

// ==================== Hooks ====================

export const usePlayer = () => useAppStore((state) => ({
  ...state.player,
  playTrack: state.playTrack,
  pauseTrack: state.pauseTrack,
  resumeTrack: state.resumeTrack,
  nextTrack: state.nextTrack,
  previousTrack: state.previousTrack,
  seekTo: state.seekTo,
  setVolume: state.setVolume,
  toggleMute: state.toggleMute,
  toggleShuffle: state.toggleShuffle,
  toggleRepeat: state.toggleRepeat,
  addToQueue: state.addToQueue,
  playQueue: state.playQueue,
}));

export const useLibrary = () => useAppStore((state) => ({
  ...state.library,
  addTrack: state.addTrack,
  removeTrack: state.removeTrack,
  likeTrack: state.likeTrack,
  unlikeTrack: state.unlikeTrack,
  addPlaylist: state.addPlaylist,
  updatePlaylist: state.updatePlaylist,
  deletePlaylist: state.deletePlaylist,
}));

export const useUI = () => useAppStore((state) => ({
  ...state.ui,
  setSidebarOpen: state.setSidebarOpen,
  setCurrentView: state.setCurrentView,
  showModal: state.showModal,
  hideModal: state.hideModal,
  showToast: state.showToast,
}));

export const useProject = () => useAppStore((state) => ({
  currentProject: state.project.currentProjectId
    ? state.project.projects.get(state.project.currentProjectId)
    : null,
  unsavedChanges: state.project.unsavedChanges,
  createProject: state.createProject,
  loadProject: state.loadProject,
  saveProject: state.saveProject,
  updateProject: state.updateProject,
  addProjectTrack: state.addProjectTrack,
  updateProjectTrack: state.updateProjectTrack,
  deleteProjectTrack: state.deleteProjectTrack,
  addClip: state.addClip,
  updateClip: state.updateClip,
  deleteClip: state.deleteClip,
}));

export default useAppStore;
