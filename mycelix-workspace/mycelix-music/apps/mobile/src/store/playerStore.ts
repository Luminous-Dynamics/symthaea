// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Player State Management
 *
 * Global state for audio playback with offline support.
 * Uses Zustand for simple, performant state management.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import TrackPlayer, {
  State,
  Event,
  Track,
  RepeatMode,
  Capability,
} from 'react-native-track-player';

// ==================== Types ====================

export interface Song {
  id: string;
  title: string;
  artist: string;
  artistAddress: string;
  duration: number;
  coverArt: string;
  audioUrl: string;
  isDownloaded: boolean;
  localPath?: string;
}

export interface QueueItem extends Song {
  queueId: string;
  addedAt: number;
  source: 'library' | 'search' | 'playlist' | 'radio' | 'recommendation';
}

export interface PlayerState {
  // Playback state
  isPlaying: boolean;
  currentSong: Song | null;
  position: number;
  duration: number;
  buffered: number;

  // Queue
  queue: QueueItem[];
  queueIndex: number;
  history: Song[];

  // Settings
  volume: number;
  shuffle: boolean;
  repeat: 'off' | 'one' | 'all';
  quality: 'low' | 'medium' | 'high' | 'lossless';

  // Offline
  offlineMode: boolean;
  downloadedSongs: Map<string, string>; // songId -> localPath

  // Actions
  play: (song: Song, source?: QueueItem['source']) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  seek: (position: number) => Promise<void>;
  next: () => Promise<void>;
  previous: () => Promise<void>;

  // Queue actions
  addToQueue: (songs: Song[], position?: 'next' | 'last') => Promise<void>;
  removeFromQueue: (queueId: string) => void;
  reorderQueue: (fromIndex: number, toIndex: number) => void;
  clearQueue: () => Promise<void>;
  playFromQueue: (index: number) => Promise<void>;

  // Settings actions
  setVolume: (volume: number) => void;
  toggleShuffle: () => void;
  toggleRepeat: () => void;
  setQuality: (quality: PlayerState['quality']) => void;

  // Offline actions
  toggleOfflineMode: () => void;
  addDownloadedSong: (songId: string, localPath: string) => void;
  removeDownloadedSong: (songId: string) => void;

  // Sync
  syncPosition: (position: number, duration: number, buffered: number) => void;
  syncPlaybackState: (isPlaying: boolean) => void;
}

// ==================== Store ====================

export const usePlayerStore = create<PlayerState>()(
  persist(
    (set, get) => ({
      // Initial state
      isPlaying: false,
      currentSong: null,
      position: 0,
      duration: 0,
      buffered: 0,
      queue: [],
      queueIndex: -1,
      history: [],
      volume: 1.0,
      shuffle: false,
      repeat: 'off',
      quality: 'high',
      offlineMode: false,
      downloadedSongs: new Map(),

      // ==================== Playback Actions ====================

      play: async (song, source = 'library') => {
        const state = get();

        // Add to history
        const newHistory = [song, ...state.history.filter((s) => s.id !== song.id)].slice(0, 100);

        // Create track
        const track = songToTrack(song, state.downloadedSongs);

        // Add to queue if not already there
        const existingIndex = state.queue.findIndex((q) => q.id === song.id);
        let newQueue = state.queue;
        let newIndex = existingIndex;

        if (existingIndex === -1) {
          const queueItem: QueueItem = {
            ...song,
            queueId: generateQueueId(),
            addedAt: Date.now(),
            source,
          };

          newQueue = [...state.queue, queueItem];
          newIndex = newQueue.length - 1;
        }

        set({
          currentSong: song,
          queue: newQueue,
          queueIndex: newIndex,
          history: newHistory,
          isPlaying: true,
        });

        // Update TrackPlayer
        await TrackPlayer.reset();
        await TrackPlayer.add(track);
        await TrackPlayer.play();
      },

      pause: async () => {
        await TrackPlayer.pause();
        set({ isPlaying: false });
      },

      resume: async () => {
        await TrackPlayer.play();
        set({ isPlaying: true });
      },

      stop: async () => {
        await TrackPlayer.reset();
        set({
          isPlaying: false,
          currentSong: null,
          position: 0,
          duration: 0,
        });
      },

      seek: async (position) => {
        await TrackPlayer.seekTo(position);
        set({ position });
      },

      next: async () => {
        const state = get();

        if (state.queue.length === 0) return;

        let nextIndex: number;

        if (state.shuffle) {
          // Random next (not current)
          const availableIndices = state.queue
            .map((_, i) => i)
            .filter((i) => i !== state.queueIndex);

          if (availableIndices.length === 0) {
            nextIndex = 0;
          } else {
            nextIndex = availableIndices[Math.floor(Math.random() * availableIndices.length)];
          }
        } else {
          nextIndex = state.queueIndex + 1;
        }

        // Handle repeat
        if (nextIndex >= state.queue.length) {
          if (state.repeat === 'all') {
            nextIndex = 0;
          } else {
            // End of queue
            await TrackPlayer.pause();
            set({ isPlaying: false });
            return;
          }
        }

        const nextSong = state.queue[nextIndex];
        if (nextSong) {
          const track = songToTrack(nextSong, state.downloadedSongs);

          set({
            currentSong: nextSong,
            queueIndex: nextIndex,
            position: 0,
          });

          await TrackPlayer.reset();
          await TrackPlayer.add(track);
          await TrackPlayer.play();
        }
      },

      previous: async () => {
        const state = get();

        // If more than 3 seconds in, restart
        if (state.position > 3) {
          await TrackPlayer.seekTo(0);
          set({ position: 0 });
          return;
        }

        let prevIndex = state.queueIndex - 1;

        if (prevIndex < 0) {
          if (state.repeat === 'all') {
            prevIndex = state.queue.length - 1;
          } else {
            prevIndex = 0;
          }
        }

        const prevSong = state.queue[prevIndex];
        if (prevSong) {
          const track = songToTrack(prevSong, state.downloadedSongs);

          set({
            currentSong: prevSong,
            queueIndex: prevIndex,
            position: 0,
          });

          await TrackPlayer.reset();
          await TrackPlayer.add(track);
          await TrackPlayer.play();
        }
      },

      // ==================== Queue Actions ====================

      addToQueue: async (songs, position = 'last') => {
        const state = get();

        const queueItems: QueueItem[] = songs.map((song) => ({
          ...song,
          queueId: generateQueueId(),
          addedAt: Date.now(),
          source: 'library',
        }));

        let newQueue: QueueItem[];

        if (position === 'next') {
          newQueue = [
            ...state.queue.slice(0, state.queueIndex + 1),
            ...queueItems,
            ...state.queue.slice(state.queueIndex + 1),
          ];
        } else {
          newQueue = [...state.queue, ...queueItems];
        }

        set({ queue: newQueue });
      },

      removeFromQueue: (queueId) => {
        const state = get();
        const index = state.queue.findIndex((q) => q.queueId === queueId);

        if (index === -1) return;

        const newQueue = state.queue.filter((q) => q.queueId !== queueId);
        let newIndex = state.queueIndex;

        if (index <= state.queueIndex) {
          newIndex = Math.max(0, state.queueIndex - 1);
        }

        set({ queue: newQueue, queueIndex: newIndex });
      },

      reorderQueue: (fromIndex, toIndex) => {
        const state = get();
        const newQueue = [...state.queue];
        const [removed] = newQueue.splice(fromIndex, 1);
        newQueue.splice(toIndex, 0, removed);

        // Adjust current index
        let newIndex = state.queueIndex;
        if (fromIndex === state.queueIndex) {
          newIndex = toIndex;
        } else if (fromIndex < state.queueIndex && toIndex >= state.queueIndex) {
          newIndex--;
        } else if (fromIndex > state.queueIndex && toIndex <= state.queueIndex) {
          newIndex++;
        }

        set({ queue: newQueue, queueIndex: newIndex });
      },

      clearQueue: async () => {
        await TrackPlayer.reset();
        set({
          queue: [],
          queueIndex: -1,
          currentSong: null,
          isPlaying: false,
          position: 0,
        });
      },

      playFromQueue: async (index) => {
        const state = get();
        const song = state.queue[index];

        if (!song) return;

        const track = songToTrack(song, state.downloadedSongs);

        set({
          currentSong: song,
          queueIndex: index,
          position: 0,
          isPlaying: true,
        });

        await TrackPlayer.reset();
        await TrackPlayer.add(track);
        await TrackPlayer.play();
      },

      // ==================== Settings Actions ====================

      setVolume: (volume) => {
        TrackPlayer.setVolume(volume);
        set({ volume });
      },

      toggleShuffle: () => {
        set((state) => ({ shuffle: !state.shuffle }));
      },

      toggleRepeat: () => {
        set((state) => {
          const modes: PlayerState['repeat'][] = ['off', 'one', 'all'];
          const currentIndex = modes.indexOf(state.repeat);
          const nextMode = modes[(currentIndex + 1) % modes.length];

          // Update TrackPlayer repeat mode
          const tpMode =
            nextMode === 'one'
              ? RepeatMode.Track
              : nextMode === 'all'
              ? RepeatMode.Queue
              : RepeatMode.Off;

          TrackPlayer.setRepeatMode(tpMode);

          return { repeat: nextMode };
        });
      },

      setQuality: (quality) => {
        set({ quality });
      },

      // ==================== Offline Actions ====================

      toggleOfflineMode: () => {
        set((state) => ({ offlineMode: !state.offlineMode }));
      },

      addDownloadedSong: (songId, localPath) => {
        set((state) => {
          const newDownloads = new Map(state.downloadedSongs);
          newDownloads.set(songId, localPath);
          return { downloadedSongs: newDownloads };
        });
      },

      removeDownloadedSong: (songId) => {
        set((state) => {
          const newDownloads = new Map(state.downloadedSongs);
          newDownloads.delete(songId);
          return { downloadedSongs: newDownloads };
        });
      },

      // ==================== Sync ====================

      syncPosition: (position, duration, buffered) => {
        set({ position, duration, buffered });
      },

      syncPlaybackState: (isPlaying) => {
        set({ isPlaying });
      },
    }),
    {
      name: 'mycelix-player',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        queue: state.queue,
        history: state.history,
        volume: state.volume,
        shuffle: state.shuffle,
        repeat: state.repeat,
        quality: state.quality,
        offlineMode: state.offlineMode,
        downloadedSongs: Array.from(state.downloadedSongs.entries()),
      }),
      merge: (persisted: any, current) => ({
        ...current,
        ...persisted,
        downloadedSongs: new Map(persisted?.downloadedSongs || []),
      }),
    }
  )
);

// ==================== Helpers ====================

function generateQueueId(): string {
  return `q_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function songToTrack(song: Song, downloads: Map<string, string>): Track {
  const localPath = downloads.get(song.id);

  return {
    id: song.id,
    url: localPath || song.audioUrl,
    title: song.title,
    artist: song.artist,
    artwork: song.coverArt,
    duration: song.duration,
  };
}

// ==================== TrackPlayer Setup ====================

export async function setupPlayer(): Promise<void> {
  await TrackPlayer.setupPlayer({
    maxCacheSize: 1024 * 1024 * 500, // 500MB cache
  });

  await TrackPlayer.updateOptions({
    capabilities: [
      Capability.Play,
      Capability.Pause,
      Capability.SkipToNext,
      Capability.SkipToPrevious,
      Capability.SeekTo,
      Capability.Stop,
    ],
    compactCapabilities: [
      Capability.Play,
      Capability.Pause,
      Capability.SkipToNext,
      Capability.SkipToPrevious,
    ],
    progressUpdateEventInterval: 1,
  });
}

// ==================== Event Handlers ====================

export function registerPlayerEvents(): () => void {
  const playbackStateSubscription = TrackPlayer.addEventListener(
    Event.PlaybackState,
    async (event) => {
      const { syncPlaybackState } = usePlayerStore.getState();
      syncPlaybackState(event.state === State.Playing);
    }
  );

  const progressSubscription = TrackPlayer.addEventListener(
    Event.PlaybackProgressUpdated,
    (event) => {
      const { syncPosition } = usePlayerStore.getState();
      syncPosition(event.position, event.duration, event.buffered);
    }
  );

  const trackEndedSubscription = TrackPlayer.addEventListener(
    Event.PlaybackQueueEnded,
    async () => {
      const state = usePlayerStore.getState();

      if (state.repeat === 'one') {
        await TrackPlayer.seekTo(0);
        await TrackPlayer.play();
      } else {
        await state.next();
      }
    }
  );

  return () => {
    playbackStateSubscription.remove();
    progressSubscription.remove();
    trackEndedSubscription.remove();
  };
}

export default usePlayerStore;
