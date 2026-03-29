// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Player Store
 *
 * Global state for audio playback using Zustand.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Song } from '@/lib/api';

export interface QueueItem extends Song {
  queueId: string;
}

export interface LyricLine {
  time: number;
  text: string;
}

export interface EqualizerBand {
  frequency: number;
  gain: number;
}

interface PlayerState {
  // Playback state
  isPlaying: boolean;
  currentSong: Song | null;
  position: number;
  duration: number;
  volume: number;
  muted: boolean;

  // Queue
  queue: QueueItem[];
  queueIndex: number;
  history: Song[];

  // Lyrics
  lyrics: LyricLine[] | null;
  currentLyricIndex: number;
  lyricsVisible: boolean;

  // UI panels
  queueVisible: boolean;

  // Settings
  shuffle: boolean;
  repeat: 'off' | 'one' | 'all';
  crossfade: number; // seconds (0 = off)

  // Equalizer
  equalizerEnabled: boolean;
  equalizerPreset: string;
  equalizerBands: EqualizerBand[];

  // Audio element ref (set by PlayerProvider)
  audioRef: HTMLAudioElement | null;

  // Actions
  play: (song: Song) => void;
  pause: () => void;
  resume: () => void;
  toggle: () => void;
  seek: (position: number) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;

  next: () => void;
  previous: () => void;

  addToQueue: (songs: Song[], position?: 'next' | 'last') => void;
  removeFromQueue: (queueId: string) => void;
  clearQueue: () => void;
  playFromQueue: (index: number) => void;
  playAll: (songs: Song[], startIndex?: number) => void;

  toggleShuffle: () => void;
  toggleRepeat: () => void;

  // Lyrics actions
  setLyrics: (lyrics: LyricLine[] | null) => void;
  toggleLyrics: () => void;
  updateCurrentLyricIndex: (position: number) => void;

  // Queue panel actions
  toggleQueue: () => void;
  reorderQueue: (fromIndex: number, toIndex: number) => void;

  // Advanced settings
  setCrossfade: (seconds: number) => void;
  setEqualizerEnabled: (enabled: boolean) => void;
  setEqualizerPreset: (preset: string) => void;
  setEqualizerBand: (frequency: number, gain: number) => void;

  setAudioRef: (ref: HTMLAudioElement | null) => void;
  syncPosition: (position: number) => void;
  syncDuration: (duration: number) => void;
}

const generateQueueId = () => `q_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

// Default equalizer bands (10-band)
const defaultEqualizerBands: EqualizerBand[] = [
  { frequency: 32, gain: 0 },
  { frequency: 64, gain: 0 },
  { frequency: 125, gain: 0 },
  { frequency: 250, gain: 0 },
  { frequency: 500, gain: 0 },
  { frequency: 1000, gain: 0 },
  { frequency: 2000, gain: 0 },
  { frequency: 4000, gain: 0 },
  { frequency: 8000, gain: 0 },
  { frequency: 16000, gain: 0 },
];

// Equalizer presets
export const equalizerPresets: Record<string, EqualizerBand[]> = {
  flat: defaultEqualizerBands,
  bass: defaultEqualizerBands.map((b) => ({
    ...b,
    gain: b.frequency <= 250 ? 6 : b.frequency <= 1000 ? 2 : 0,
  })),
  treble: defaultEqualizerBands.map((b) => ({
    ...b,
    gain: b.frequency >= 4000 ? 6 : b.frequency >= 1000 ? 2 : 0,
  })),
  vocal: defaultEqualizerBands.map((b) => ({
    ...b,
    gain: b.frequency >= 250 && b.frequency <= 4000 ? 4 : -2,
  })),
  electronic: [
    { frequency: 32, gain: 4 },
    { frequency: 64, gain: 3 },
    { frequency: 125, gain: 0 },
    { frequency: 250, gain: -2 },
    { frequency: 500, gain: -1 },
    { frequency: 1000, gain: 2 },
    { frequency: 2000, gain: 3 },
    { frequency: 4000, gain: 4 },
    { frequency: 8000, gain: 5 },
    { frequency: 16000, gain: 4 },
  ],
};

export const usePlayerStore = create<PlayerState>()(
  persist(
    (set, get) => ({
      // Initial state
      isPlaying: false,
      currentSong: null,
      position: 0,
      duration: 0,
      volume: 0.8,
      muted: false,
      queue: [],
      queueIndex: -1,
      history: [],
      lyrics: null,
      currentLyricIndex: -1,
      lyricsVisible: false,
      queueVisible: false,
      shuffle: false,
      repeat: 'off',
      crossfade: 0,
      equalizerEnabled: false,
      equalizerPreset: 'flat',
      equalizerBands: defaultEqualizerBands,
      audioRef: null,

      // Play a song
      play: (song) => {
        const state = get();
        const audio = state.audioRef;

        // Add to history
        const newHistory = [song, ...state.history.filter((s) => s.id !== song.id)].slice(0, 50);

        // Check if song is in queue
        const existingIndex = state.queue.findIndex((q) => q.id === song.id);
        let newQueue = state.queue;
        let newIndex = existingIndex;

        if (existingIndex === -1) {
          const queueItem: QueueItem = { ...song, queueId: generateQueueId() };
          newQueue = [...state.queue, queueItem];
          newIndex = newQueue.length - 1;
        }

        set({
          currentSong: song,
          queue: newQueue,
          queueIndex: newIndex,
          history: newHistory,
          isPlaying: true,
          position: 0,
        });

        if (audio) {
          audio.src = song.audioUrl;
          audio.play().catch(console.error);
        }
      },

      pause: () => {
        const { audioRef } = get();
        audioRef?.pause();
        set({ isPlaying: false });
      },

      resume: () => {
        const { audioRef } = get();
        audioRef?.play().catch(console.error);
        set({ isPlaying: true });
      },

      toggle: () => {
        const { isPlaying, pause, resume } = get();
        if (isPlaying) pause();
        else resume();
      },

      seek: (position) => {
        const { audioRef } = get();
        if (audioRef) {
          audioRef.currentTime = position;
          set({ position });
        }
      },

      setVolume: (volume) => {
        const { audioRef } = get();
        if (audioRef) audioRef.volume = volume;
        set({ volume, muted: volume === 0 });
      },

      toggleMute: () => {
        const { audioRef, muted, volume } = get();
        if (audioRef) {
          audioRef.muted = !muted;
        }
        set({ muted: !muted });
      },

      next: () => {
        const state = get();
        if (state.queue.length === 0) return;

        let nextIndex: number;

        if (state.shuffle) {
          const available = state.queue.map((_, i) => i).filter((i) => i !== state.queueIndex);
          nextIndex = available.length > 0
            ? available[Math.floor(Math.random() * available.length)]
            : 0;
        } else {
          nextIndex = state.queueIndex + 1;
        }

        if (nextIndex >= state.queue.length) {
          if (state.repeat === 'all') {
            nextIndex = 0;
          } else {
            set({ isPlaying: false });
            return;
          }
        }

        const nextSong = state.queue[nextIndex];
        if (nextSong) {
          set({ queueIndex: nextIndex });
          get().play(nextSong);
        }
      },

      previous: () => {
        const state = get();

        // If more than 3 seconds in, restart
        if (state.position > 3) {
          get().seek(0);
          return;
        }

        let prevIndex = state.queueIndex - 1;
        if (prevIndex < 0) {
          prevIndex = state.repeat === 'all' ? state.queue.length - 1 : 0;
        }

        const prevSong = state.queue[prevIndex];
        if (prevSong) {
          set({ queueIndex: prevIndex });
          get().play(prevSong);
        }
      },

      addToQueue: (songs, position = 'last') => {
        const state = get();
        const queueItems: QueueItem[] = songs.map((song) => ({
          ...song,
          queueId: generateQueueId(),
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

      clearQueue: () => {
        set({ queue: [], queueIndex: -1 });
      },

      playFromQueue: (index) => {
        const { queue, play } = get();
        const song = queue[index];
        if (song) {
          set({ queueIndex: index });
          play(song);
        }
      },

      playAll: (songs, startIndex = 0) => {
        if (songs.length === 0) return;

        const queueItems: QueueItem[] = songs.map((song) => ({
          ...song,
          queueId: generateQueueId(),
        }));

        set({
          queue: queueItems,
          queueIndex: startIndex,
        });

        const songToPlay = songs[startIndex];
        if (songToPlay) {
          get().play(songToPlay);
        }
      },

      toggleShuffle: () => set((state) => ({ shuffle: !state.shuffle })),

      toggleRepeat: () => {
        set((state) => {
          const modes: PlayerState['repeat'][] = ['off', 'one', 'all'];
          const currentIndex = modes.indexOf(state.repeat);
          return { repeat: modes[(currentIndex + 1) % modes.length] };
        });
      },

      // Lyrics actions
      setLyrics: (lyrics) => set({ lyrics, currentLyricIndex: -1 }),

      toggleLyrics: () => set((state) => ({ lyricsVisible: !state.lyricsVisible })),

      updateCurrentLyricIndex: (position) => {
        const { lyrics } = get();
        if (!lyrics || lyrics.length === 0) return;

        // Find the current lyric based on position
        let index = -1;
        for (let i = 0; i < lyrics.length; i++) {
          if (lyrics[i].time <= position) {
            index = i;
          } else {
            break;
          }
        }

        set({ currentLyricIndex: index });
      },

      // Queue panel actions
      toggleQueue: () => set((state) => ({ queueVisible: !state.queueVisible })),

      reorderQueue: (fromIndex, toIndex) => {
        const { queue, queueIndex } = get();
        if (fromIndex === toIndex) return;

        const newQueue = [...queue];
        const [removed] = newQueue.splice(fromIndex, 1);
        newQueue.splice(toIndex, 0, removed);

        // Adjust queueIndex if needed
        let newQueueIndex = queueIndex;
        if (fromIndex === queueIndex) {
          newQueueIndex = toIndex;
        } else if (fromIndex < queueIndex && toIndex >= queueIndex) {
          newQueueIndex--;
        } else if (fromIndex > queueIndex && toIndex <= queueIndex) {
          newQueueIndex++;
        }

        set({ queue: newQueue, queueIndex: newQueueIndex });
      },

      // Advanced settings
      setCrossfade: (seconds) => set({ crossfade: Math.max(0, Math.min(12, seconds)) }),

      setEqualizerEnabled: (enabled) => set({ equalizerEnabled: enabled }),

      setEqualizerPreset: (preset) => {
        const bands = equalizerPresets[preset] || equalizerPresets.flat;
        set({ equalizerPreset: preset, equalizerBands: bands });
      },

      setEqualizerBand: (frequency, gain) => {
        set((state) => ({
          equalizerPreset: 'custom',
          equalizerBands: state.equalizerBands.map((band) =>
            band.frequency === frequency ? { ...band, gain } : band
          ),
        }));
      },

      setAudioRef: (ref) => set({ audioRef: ref }),
      syncPosition: (position) => set({ position }),
      syncDuration: (duration) => set({ duration }),
    }),
    {
      name: 'mycelix-player',
      partialize: (state) => ({
        volume: state.volume,
        muted: state.muted,
        shuffle: state.shuffle,
        repeat: state.repeat,
        crossfade: state.crossfade,
        equalizerEnabled: state.equalizerEnabled,
        equalizerPreset: state.equalizerPreset,
        equalizerBands: state.equalizerBands,
        history: state.history.slice(0, 20),
      }),
    }
  )
);

export default usePlayerStore;
