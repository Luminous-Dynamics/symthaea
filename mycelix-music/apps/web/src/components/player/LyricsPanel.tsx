// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { usePlayerStore } from '@/store/playerStore';
import { useEffect, useRef, useCallback } from 'react';
import { X, Mic2, Music } from 'lucide-react';
import { api } from '@/lib/api';

export function LyricsPanel() {
  const {
    currentSong,
    lyrics,
    currentLyricIndex,
    lyricsVisible,
    setLyrics,
    toggleLyrics,
    seek,
  } = usePlayerStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const activeLineRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to current lyric
  useEffect(() => {
    if (activeLineRef.current && containerRef.current) {
      const container = containerRef.current;
      const activeLine = activeLineRef.current;

      const containerHeight = container.clientHeight;
      const lineTop = activeLine.offsetTop;
      const lineHeight = activeLine.clientHeight;

      // Center the active line
      container.scrollTo({
        top: lineTop - containerHeight / 2 + lineHeight / 2,
        behavior: 'smooth',
      });
    }
  }, [currentLyricIndex]);

  // Fetch lyrics when track changes
  const fetchLyrics = useCallback(async (songId: string) => {
    try {
      // Try to fetch from API first
      const response = await api.getLyrics?.(songId);
      if (response?.lyrics) {
        setLyrics(response.lyrics);
        return;
      }
    } catch {
      // Fall back to placeholder
    }

    // Use placeholder lyrics for demo
    const placeholderLyrics = [
      { time: 0, text: '...' },
      { time: 5, text: 'Welcome to the journey' },
      { time: 10, text: 'Through sound and space' },
      { time: 15, text: 'Let the music guide you' },
      { time: 20, text: 'To a better place' },
      { time: 28, text: '' },
      { time: 30, text: 'Digital waves are flowing' },
      { time: 35, text: 'Through the night' },
      { time: 40, text: 'Every beat is glowing' },
      { time: 45, text: 'Shining bright' },
      { time: 53, text: '' },
      { time: 55, text: 'Feel the rhythm in your heart' },
      { time: 60, text: 'Let it take you far' },
      { time: 65, text: 'Every ending is a start' },
      { time: 70, text: 'Like a shooting star' },
    ];

    setLyrics(placeholderLyrics);
  }, [setLyrics]);

  useEffect(() => {
    if (currentSong) {
      fetchLyrics(currentSong.id);
    } else {
      setLyrics(null);
    }
  }, [currentSong?.id, fetchLyrics, setLyrics]);

  if (!lyricsVisible) {
    return null;
  }

  if (!currentSong) {
    return null;
  }

  return (
    <div className="fixed right-0 top-0 bottom-20 w-96 bg-gray-900/95 backdrop-blur-md border-l border-white/10 z-40 overflow-hidden flex flex-col shadow-2xl">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Mic2 className="w-5 h-5 text-purple-400" />
          <div>
            <h3 className="font-semibold text-lg">Lyrics</h3>
            <p className="text-sm text-gray-400 truncate max-w-[200px]">
              {currentSong.title}
            </p>
          </div>
        </div>
        <button
          onClick={toggleLyrics}
          className="p-1 text-gray-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Lyrics Content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto p-6 relative"
      >
        {lyrics && lyrics.length > 0 ? (
          <div className="space-y-4 py-32">
            {lyrics.map((line, index) => (
              <div
                key={index}
                ref={index === currentLyricIndex ? activeLineRef : null}
                onClick={() => seek(line.time)}
                className={`cursor-pointer transition-all duration-300 origin-left ${
                  index === currentLyricIndex
                    ? 'text-white text-2xl font-bold scale-105'
                    : index < currentLyricIndex
                    ? 'text-gray-600 text-lg'
                    : 'text-gray-500 text-lg hover:text-gray-400'
                }`}
              >
                {line.text || <span className="opacity-30">...</span>}
              </div>
            ))}
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <Music className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg font-medium">No lyrics available</p>
            <p className="text-sm mt-2 text-center px-4">
              Lyrics for this track haven&apos;t been added yet
            </p>
          </div>
        )}
      </div>

      {/* Gradient overlays for better readability */}
      <div className="absolute top-16 left-0 right-0 h-16 bg-gradient-to-b from-gray-900/95 to-transparent pointer-events-none" />
      <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-gray-900/95 to-transparent pointer-events-none" />
    </div>
  );
}
