// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Player Context for Mycelix Mobile
 *
 * Manages audio playback state and controls.
 */

import { createContext, useContext, useState, useCallback, ReactNode, useRef, useEffect } from 'react';
import { Audio, AVPlaybackStatus } from 'expo-av';
import * as Haptics from 'expo-haptics';

export interface Track {
  id: string;
  title: string;
  artist: string;
  artistId: string;
  albumId?: string;
  artworkUri: string;
  audioUri: string;
  duration: number;
}

export interface PlayerState {
  currentTrack: Track | null;
  isPlaying: boolean;
  isLoading: boolean;
  position: number;
  duration: number;
  isExpanded: boolean;
  queue: Track[];
  queueIndex: number;
  repeatMode: 'off' | 'all' | 'one';
  shuffleEnabled: boolean;
}

interface PlayerContextValue extends PlayerState {
  playTrack: (trackId: string) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  seekTo: (position: number) => Promise<void>;
  skipNext: () => Promise<void>;
  skipPrevious: () => Promise<void>;
  setQueue: (tracks: Track[], startIndex?: number) => void;
  addToQueue: (track: Track) => void;
  toggleExpanded: () => void;
  toggleRepeat: () => void;
  toggleShuffle: () => void;
}

const PlayerContext = createContext<PlayerContextValue | undefined>(undefined);

// Mock track data (in production, this would come from API)
const mockTracks: Record<string, Track> = {
  'track-1': {
    id: 'track-1',
    title: 'Ethereal Dreams',
    artist: 'Ambient Collective',
    artistId: 'artist-1',
    artworkUri: 'https://picsum.photos/seed/track1/400',
    audioUri: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
    duration: 371,
  },
  'track-2': {
    id: 'track-2',
    title: 'Midnight Waves',
    artist: 'Ocean Sound',
    artistId: 'artist-2',
    artworkUri: 'https://picsum.photos/seed/track2/400',
    audioUri: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3',
    duration: 280,
  },
};

export function PlayerProvider({ children }: { children: ReactNode }) {
  const soundRef = useRef<Audio.Sound | null>(null);
  const [state, setState] = useState<PlayerState>({
    currentTrack: null,
    isPlaying: false,
    isLoading: false,
    position: 0,
    duration: 0,
    isExpanded: false,
    queue: [],
    queueIndex: 0,
    repeatMode: 'off',
    shuffleEnabled: false,
  });

  // Configure audio mode
  useEffect(() => {
    async function configureAudio() {
      await Audio.setAudioModeAsync({
        staysActiveInBackground: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
      });
    }
    configureAudio();

    return () => {
      if (soundRef.current) {
        soundRef.current.unloadAsync();
      }
    };
  }, []);

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    if (!status.isLoaded) return;

    setState((prev) => ({
      ...prev,
      isPlaying: status.isPlaying,
      isLoading: status.isBuffering,
      position: status.positionMillis / 1000,
      duration: (status.durationMillis ?? 0) / 1000,
    }));

    // Handle track completion
    if (status.didJustFinish) {
      // Auto-advance to next track based on repeat mode
    }
  }, []);

  const playTrack = useCallback(async (trackId: string) => {
    try {
      setState((prev) => ({ ...prev, isLoading: true }));
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

      // Unload existing sound
      if (soundRef.current) {
        await soundRef.current.unloadAsync();
      }

      // Get track data (mock for now)
      const track = mockTracks[trackId] || mockTracks['track-1'];

      // Load and play new sound
      const { sound } = await Audio.Sound.createAsync(
        { uri: track.audioUri },
        { shouldPlay: true },
        onPlaybackStatusUpdate
      );

      soundRef.current = sound;

      setState((prev) => ({
        ...prev,
        currentTrack: track,
        isLoading: false,
        isPlaying: true,
      }));
    } catch (error) {
      console.error('Failed to play track:', error);
      setState((prev) => ({ ...prev, isLoading: false }));
    }
  }, [onPlaybackStatusUpdate]);

  const pause = useCallback(async () => {
    if (soundRef.current) {
      await soundRef.current.pauseAsync();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  }, []);

  const resume = useCallback(async () => {
    if (soundRef.current) {
      await soundRef.current.playAsync();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  }, []);

  const seekTo = useCallback(async (position: number) => {
    if (soundRef.current) {
      await soundRef.current.setPositionAsync(position * 1000);
    }
  }, []);

  const skipNext = useCallback(async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Implement queue navigation
    if (state.queue.length > 0 && state.queueIndex < state.queue.length - 1) {
      const nextTrack = state.queue[state.queueIndex + 1];
      setState((prev) => ({ ...prev, queueIndex: prev.queueIndex + 1 }));
      await playTrack(nextTrack.id);
    }
  }, [state.queue, state.queueIndex, playTrack]);

  const skipPrevious = useCallback(async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // If more than 3 seconds in, restart track
    if (state.position > 3) {
      await seekTo(0);
      return;
    }
    // Otherwise go to previous track
    if (state.queueIndex > 0) {
      const prevTrack = state.queue[state.queueIndex - 1];
      setState((prev) => ({ ...prev, queueIndex: prev.queueIndex - 1 }));
      await playTrack(prevTrack.id);
    }
  }, [state.position, state.queueIndex, state.queue, seekTo, playTrack]);

  const setQueue = useCallback((tracks: Track[], startIndex = 0) => {
    setState((prev) => ({ ...prev, queue: tracks, queueIndex: startIndex }));
    if (tracks.length > 0) {
      playTrack(tracks[startIndex].id);
    }
  }, [playTrack]);

  const addToQueue = useCallback((track: Track) => {
    setState((prev) => ({ ...prev, queue: [...prev.queue, track] }));
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, []);

  const toggleExpanded = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setState((prev) => ({ ...prev, isExpanded: !prev.isExpanded }));
  }, []);

  const toggleRepeat = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setState((prev) => ({
      ...prev,
      repeatMode:
        prev.repeatMode === 'off' ? 'all' :
        prev.repeatMode === 'all' ? 'one' : 'off',
    }));
  }, []);

  const toggleShuffle = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setState((prev) => ({ ...prev, shuffleEnabled: !prev.shuffleEnabled }));
  }, []);

  const value: PlayerContextValue = {
    ...state,
    playTrack,
    pause,
    resume,
    seekTo,
    skipNext,
    skipPrevious,
    setQueue,
    addToQueue,
    toggleExpanded,
    toggleRepeat,
    toggleShuffle,
  };

  return (
    <PlayerContext.Provider value={value}>
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerContextValue {
  const context = useContext(PlayerContext);
  if (!context) {
    throw new Error('usePlayer must be used within a PlayerProvider');
  }
  return context;
}
