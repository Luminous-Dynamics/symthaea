// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useContext, createContext, useState, useRef, useEffect, ReactNode, useCallback } from 'react';
import { api } from '@/lib/api';

interface Track {
  id: string;
  title: string;
  duration: number;
  audioUrl: string;
  coverUrl?: string;
  artist: {
    id: string;
    name: string;
  };
  album?: {
    id: string;
    title: string;
  };
}

type RepeatMode = 'off' | 'all' | 'one';

interface PlayerContextType {
  currentTrack: Track | null;
  queue: Track[];
  history: Track[];
  isPlaying: boolean;
  progress: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  repeatMode: RepeatMode;
  isShuffled: boolean;

  // Controls
  play: (track?: Track) => void;
  pause: () => void;
  next: () => void;
  previous: () => void;
  seek: (time: number) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;
  toggleRepeat: () => void;
  toggleShuffle: () => void;

  // Queue management
  addToQueue: (track: Track) => void;
  removeFromQueue: (index: number) => void;
  clearQueue: () => void;
  playNext: (track: Track) => void;
  reorderQueue: (from: number, to: number) => void;

  // Play from different sources
  playTrack: (trackId: string) => Promise<void>;
  playAlbum: (albumId: string, startIndex?: number) => Promise<void>;
  playPlaylist: (playlistId: string, startIndex?: number) => Promise<void>;
  playArtist: (artistId: string) => Promise<void>;
}

const PlayerContext = createContext<PlayerContextType | undefined>(undefined);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [queue, setQueue] = useState<Track[]>([]);
  const [history, setHistory] = useState<Track[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  const [repeatMode, setRepeatMode] = useState<RepeatMode>('off');
  const [isShuffled, setIsShuffled] = useState(false);
  const [originalQueue, setOriginalQueue] = useState<Track[]>([]);

  // Initialize audio element
  useEffect(() => {
    audioRef.current = new Audio();
    audioRef.current.volume = volume;

    const audio = audioRef.current;

    audio.addEventListener('timeupdate', () => {
      setProgress(audio.currentTime);
    });

    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration);
    });

    audio.addEventListener('ended', handleTrackEnd);

    audio.addEventListener('play', () => setIsPlaying(true));
    audio.addEventListener('pause', () => setIsPlaying(false));

    return () => {
      audio.pause();
      audio.src = '';
    };
  }, []);

  const handleTrackEnd = useCallback(() => {
    if (repeatMode === 'one' && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    } else {
      next();
    }
  }, [repeatMode]);

  // Track playback reporting
  useEffect(() => {
    if (currentTrack && isPlaying && progress > 30) {
      // Report play after 30 seconds
      api.post(`/tracks/${currentTrack.id}/play`).catch(() => {});
    }
  }, [currentTrack, isPlaying, progress > 30]);

  const play = useCallback((track?: Track) => {
    if (track) {
      if (currentTrack) {
        setHistory(prev => [...prev, currentTrack]);
      }
      setCurrentTrack(track);
      if (audioRef.current) {
        audioRef.current.src = track.audioUrl;
        audioRef.current.play();
      }
    } else if (audioRef.current && currentTrack) {
      audioRef.current.play();
    }
  }, [currentTrack]);

  const pause = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  const next = useCallback(() => {
    if (queue.length > 0) {
      const nextTrack = queue[0];
      setQueue(prev => prev.slice(1));
      play(nextTrack);
    } else if (repeatMode === 'all' && history.length > 0) {
      setQueue([...history]);
      setHistory([]);
      const nextTrack = history[0];
      play(nextTrack);
    }
  }, [queue, history, repeatMode, play]);

  const previous = useCallback(() => {
    if (progress > 3 && audioRef.current) {
      // If more than 3 seconds in, restart track
      audioRef.current.currentTime = 0;
    } else if (history.length > 0) {
      const prevTrack = history[history.length - 1];
      setHistory(prev => prev.slice(0, -1));
      if (currentTrack) {
        setQueue(prev => [currentTrack, ...prev]);
      }
      play(prevTrack);
    }
  }, [progress, history, currentTrack, play]);

  const seek = useCallback((time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setProgress(time);
    }
  }, []);

  const setVolume = useCallback((newVolume: number) => {
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
    setVolumeState(newVolume);
    if (newVolume > 0) {
      setIsMuted(false);
    }
  }, []);

  const toggleMute = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
    }
    setIsMuted(prev => !prev);
  }, [isMuted]);

  const toggleRepeat = useCallback(() => {
    setRepeatMode(prev => {
      if (prev === 'off') return 'all';
      if (prev === 'all') return 'one';
      return 'off';
    });
  }, []);

  const toggleShuffle = useCallback(() => {
    if (!isShuffled) {
      setOriginalQueue([...queue]);
      const shuffled = [...queue].sort(() => Math.random() - 0.5);
      setQueue(shuffled);
    } else {
      setQueue([...originalQueue]);
    }
    setIsShuffled(prev => !prev);
  }, [isShuffled, queue, originalQueue]);

  const addToQueue = useCallback((track: Track) => {
    setQueue(prev => [...prev, track]);
  }, []);

  const removeFromQueue = useCallback((index: number) => {
    setQueue(prev => prev.filter((_, i) => i !== index));
  }, []);

  const clearQueue = useCallback(() => {
    setQueue([]);
  }, []);

  const playNext = useCallback((track: Track) => {
    setQueue(prev => [track, ...prev]);
  }, []);

  const reorderQueue = useCallback((from: number, to: number) => {
    setQueue(prev => {
      const result = [...prev];
      const [removed] = result.splice(from, 1);
      result.splice(to, 0, removed);
      return result;
    });
  }, []);

  const playTrack = useCallback(async (trackId: string) => {
    const response = await api.get(`/tracks/${trackId}`);
    play(response.data);
  }, [play]);

  const playAlbum = useCallback(async (albumId: string, startIndex = 0) => {
    const response = await api.get(`/albums/${albumId}/tracks`);
    const tracks = response.data;
    if (tracks.length > 0) {
      play(tracks[startIndex]);
      setQueue(tracks.slice(startIndex + 1));
    }
  }, [play]);

  const playPlaylist = useCallback(async (playlistId: string, startIndex = 0) => {
    const response = await api.get(`/playlists/${playlistId}/tracks`);
    const tracks = response.data;
    if (tracks.length > 0) {
      play(tracks[startIndex]);
      setQueue(tracks.slice(startIndex + 1));
    }
  }, [play]);

  const playArtist = useCallback(async (artistId: string) => {
    const response = await api.get(`/artists/${artistId}/top-tracks`);
    const tracks = response.data;
    if (tracks.length > 0) {
      play(tracks[0]);
      setQueue(tracks.slice(1));
    }
  }, [play]);

  return (
    <PlayerContext.Provider
      value={{
        currentTrack,
        queue,
        history,
        isPlaying,
        progress,
        duration,
        volume,
        isMuted,
        repeatMode,
        isShuffled,
        play,
        pause,
        next,
        previous,
        seek,
        setVolume,
        toggleMute,
        toggleRepeat,
        toggleShuffle,
        addToQueue,
        removeFromQueue,
        clearQueue,
        playNext,
        reorderQueue,
        playTrack,
        playAlbum,
        playPlaylist,
        playArtist,
      }}
    >
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer() {
  const context = useContext(PlayerContext);
  if (context === undefined) {
    throw new Error('usePlayer must be used within a PlayerProvider');
  }
  return context;
}
