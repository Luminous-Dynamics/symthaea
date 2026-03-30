// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { usePlayerStore } from '@/store/playerStore';
import { formatDuration } from '@/lib/utils';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Shuffle,
  Repeat,
  Repeat1,
  Volume2,
  VolumeX,
  Volume1,
  ChevronDown,
  Heart,
  Share2,
  MoreHorizontal,
  ListMusic,
  Mic2,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState, useEffect, useRef } from 'react';

interface NowPlayingProps {
  isOpen: boolean;
  onClose: () => void;
}

export function NowPlaying({ isOpen, onClose }: NowPlayingProps) {
  const {
    currentSong,
    isPlaying,
    position,
    duration,
    volume,
    muted,
    shuffle,
    repeat,
    lyrics,
    currentLyricIndex,
    toggle,
    next,
    previous,
    seek,
    setVolume,
    toggleMute,
    toggleShuffle,
    toggleRepeat,
  } = usePlayerStore();

  const [showLyrics, setShowLyrics] = useState(true);
  const lyricsContainerRef = useRef<HTMLDivElement>(null);
  const activeLyricRef = useRef<HTMLDivElement>(null);

  // Auto-scroll lyrics
  useEffect(() => {
    if (activeLyricRef.current && lyricsContainerRef.current && showLyrics) {
      const container = lyricsContainerRef.current;
      const activeLine = activeLyricRef.current;

      container.scrollTo({
        top: activeLine.offsetTop - container.clientHeight / 2 + activeLine.clientHeight / 2,
        behavior: 'smooth',
      });
    }
  }, [currentLyricIndex, showLyrics]);

  if (!isOpen || !currentSong) return null;

  const progress = duration > 0 ? (position / duration) * 100 : 0;

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    seek(percent * duration);
  };

  return (
    <div className="fixed inset-0 z-[60] bg-black flex flex-col">
      {/* Background - blurred album art */}
      <div className="absolute inset-0 overflow-hidden">
        <Image
          src={currentSong.coverArt || '/placeholder-album.png'}
          alt=""
          fill
          className="object-cover blur-3xl opacity-30 scale-110"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/40 to-black" />
      </div>

      {/* Content */}
      <div className="relative flex-1 flex flex-col max-w-4xl mx-auto w-full px-8 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onClose}
            className="p-2 text-white/70 hover:text-white transition-colors"
          >
            <ChevronDown className="w-8 h-8" />
          </button>

          <div className="text-center">
            <p className="text-sm text-white/50 uppercase tracking-wider">Now Playing</p>
          </div>

          <button className="p-2 text-white/70 hover:text-white transition-colors">
            <MoreHorizontal className="w-6 h-6" />
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex gap-12 items-center">
          {/* Album Art */}
          <div className="flex-shrink-0">
            <div className="relative w-80 h-80 rounded-2xl overflow-hidden shadow-2xl">
              <Image
                src={currentSong.coverArt || '/placeholder-album.png'}
                alt={currentSong.title}
                fill
                className="object-cover"
                priority
              />
            </div>
          </div>

          {/* Lyrics / Info */}
          <div className="flex-1 min-w-0">
            {/* Toggle */}
            <div className="flex gap-4 mb-6">
              <button
                onClick={() => setShowLyrics(false)}
                className={`text-sm font-medium transition-colors ${
                  !showLyrics ? 'text-white' : 'text-white/50 hover:text-white/70'
                }`}
              >
                Up Next
              </button>
              <button
                onClick={() => setShowLyrics(true)}
                className={`text-sm font-medium transition-colors ${
                  showLyrics ? 'text-white' : 'text-white/50 hover:text-white/70'
                }`}
              >
                Lyrics
              </button>
            </div>

            {/* Content Area */}
            <div
              ref={lyricsContainerRef}
              className="h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-white/20"
            >
              {showLyrics ? (
                lyrics && lyrics.length > 0 ? (
                  <div className="space-y-3 py-20">
                    {lyrics.map((line, index) => (
                      <div
                        key={index}
                        ref={index === currentLyricIndex ? activeLyricRef : null}
                        onClick={() => seek(line.time)}
                        className={`cursor-pointer transition-all duration-300 ${
                          index === currentLyricIndex
                            ? 'text-white text-xl font-semibold'
                            : index < currentLyricIndex
                            ? 'text-white/30 text-lg'
                            : 'text-white/50 text-lg hover:text-white/70'
                        }`}
                      >
                        {line.text || <span className="opacity-30">...</span>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-white/50">
                    <div className="text-center">
                      <Mic2 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>No lyrics available</p>
                    </div>
                  </div>
                )
              ) : (
                <div className="h-full flex items-center justify-center text-white/50">
                  <div className="text-center">
                    <ListMusic className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Queue view coming soon</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Controls */}
        <div className="mt-auto pt-8">
          {/* Song Info */}
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-white truncate">
                {currentSong.title}
              </h2>
              <Link
                href={`/artist/${currentSong.artistAddress}`}
                className="text-white/70 hover:text-white transition-colors"
              >
                {currentSong.artist}
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2 text-white/70 hover:text-white transition-colors">
                <Heart className="w-6 h-6" />
              </button>
              <button className="p-2 text-white/70 hover:text-white transition-colors">
                <Share2 className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-4">
            <div
              className="h-1 bg-white/20 rounded-full cursor-pointer group"
              onClick={handleSeek}
            >
              <div
                className="h-full bg-white group-hover:bg-purple-400 rounded-full relative transition-colors"
                style={{ width: `${progress}%` }}
              >
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>
            <div className="flex justify-between text-sm text-white/50 mt-2">
              <span>{formatDuration(position)}</span>
              <span>{formatDuration(duration)}</span>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center justify-center gap-6">
            <button
              onClick={toggleShuffle}
              className={`p-2 transition-colors ${
                shuffle ? 'text-purple-400' : 'text-white/70 hover:text-white'
              }`}
            >
              <Shuffle className="w-5 h-5" />
            </button>

            <button
              onClick={previous}
              className="p-2 text-white/70 hover:text-white transition-colors"
            >
              <SkipBack className="w-8 h-8" />
            </button>

            <button
              onClick={toggle}
              className="w-16 h-16 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition-transform"
            >
              {isPlaying ? (
                <Pause className="w-7 h-7" />
              ) : (
                <Play className="w-7 h-7 ml-1" />
              )}
            </button>

            <button
              onClick={next}
              className="p-2 text-white/70 hover:text-white transition-colors"
            >
              <SkipForward className="w-8 h-8" />
            </button>

            <button
              onClick={toggleRepeat}
              className={`p-2 transition-colors ${
                repeat !== 'off' ? 'text-purple-400' : 'text-white/70 hover:text-white'
              }`}
            >
              {repeat === 'one' ? (
                <Repeat1 className="w-5 h-5" />
              ) : (
                <Repeat className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Volume */}
          <div className="flex items-center justify-center gap-2 mt-6">
            <button
              onClick={toggleMute}
              className="p-2 text-white/50 hover:text-white transition-colors"
            >
              {muted || volume === 0 ? (
                <VolumeX className="w-5 h-5" />
              ) : volume < 0.5 ? (
                <Volume1 className="w-5 h-5" />
              ) : (
                <Volume2 className="w-5 h-5" />
              )}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={muted ? 0 : volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-32 h-1 bg-white/20 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
