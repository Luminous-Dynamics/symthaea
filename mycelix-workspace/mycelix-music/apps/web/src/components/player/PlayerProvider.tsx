// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { ReactNode, useRef, useEffect } from 'react';
import { usePlayerStore } from '@/store/playerStore';
import { api } from '@/lib/api';

interface PlayerProviderProps {
  children: ReactNode;
}

export function PlayerProvider({ children }: PlayerProviderProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const playStartRef = useRef<number>(0);

  const {
    setAudioRef,
    syncPosition,
    syncDuration,
    updateCurrentLyricIndex,
    currentSong,
    next,
    repeat,
    volume,
    muted,
  } = usePlayerStore();

  // Set audio ref on mount
  useEffect(() => {
    if (audioRef.current) {
      setAudioRef(audioRef.current);
      audioRef.current.volume = volume;
      audioRef.current.muted = muted;
    }
  }, [setAudioRef, volume, muted]);

  // Handle time update
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const currentTime = audioRef.current.currentTime;
      syncPosition(currentTime);
      updateCurrentLyricIndex(currentTime);
    }
  };

  // Handle loaded metadata
  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      syncDuration(audioRef.current.duration);
    }
  };

  // Handle play start
  const handlePlay = () => {
    playStartRef.current = Date.now();
  };

  // Handle ended
  const handleEnded = () => {
    // Record play
    if (currentSong && playStartRef.current) {
      const duration = (Date.now() - playStartRef.current) / 1000;
      api.recordPlay(currentSong.id, duration).catch(console.error);
    }

    // Handle repeat
    if (repeat === 'one' && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    } else {
      next();
    }
  };

  // Handle errors
  const handleError = (e: React.SyntheticEvent<HTMLAudioElement, Event>) => {
    console.error('Audio playback error:', e);
  };

  return (
    <>
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={handlePlay}
        onEnded={handleEnded}
        onError={handleError}
        preload="auto"
      />
      {children}
    </>
  );
}

export default PlayerProvider;
