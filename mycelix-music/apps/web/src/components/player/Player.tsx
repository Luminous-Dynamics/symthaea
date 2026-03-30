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
  ListMusic,
  Heart,
  Maximize2,
  Mic2,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { QueuePanel } from './QueuePanel';
import { LyricsPanel } from './LyricsPanel';

export function Player() {
  const {
    currentSong,
    isPlaying,
    position,
    duration,
    volume,
    muted,
    shuffle,
    repeat,
    queueVisible,
    lyricsVisible,
    toggle,
    next,
    previous,
    seek,
    setVolume,
    toggleMute,
    toggleShuffle,
    toggleRepeat,
    toggleQueue,
    toggleLyrics,
  } = usePlayerStore();

  if (!currentSong) {
    return null;
  }

  const progress = duration > 0 ? (position / duration) * 100 : 0;

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    seek(percent * duration);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setVolume(parseFloat(e.target.value));
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-black to-black/95 border-t border-white/10 px-4 z-50">
      <div className="h-full max-w-screen-2xl mx-auto flex items-center gap-4">
        {/* Song Info */}
        <div className="flex items-center gap-3 w-72 min-w-0">
          <div className="relative w-14 h-14 rounded-md overflow-hidden flex-shrink-0 group">
            <Image
              src={currentSong.coverArt || '/placeholder-album.png'}
              alt={currentSong.title}
              fill
              className="object-cover"
            />
            <Link
              href={`/song/${currentSong.id}`}
              className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
            >
              <Maximize2 className="w-5 h-5" />
            </Link>
          </div>

          <div className="min-w-0">
            <Link
              href={`/song/${currentSong.id}`}
              className="block text-sm font-medium truncate hover:underline"
            >
              {currentSong.title}
            </Link>
            <Link
              href={`/artist/${currentSong.artistAddress}`}
              className="block text-xs text-muted-foreground truncate hover:underline"
            >
              {currentSong.artist}
            </Link>
          </div>

          <button className="p-2 text-muted-foreground hover:text-white transition-colors">
            <Heart className="w-4 h-4" />
          </button>
        </div>

        {/* Main Controls */}
        <div className="flex-1 flex flex-col items-center gap-1 max-w-2xl">
          {/* Buttons */}
          <div className="flex items-center gap-4">
            <button
              onClick={toggleShuffle}
              className={`p-2 transition-colors ${
                shuffle ? 'text-primary' : 'text-muted-foreground hover:text-white'
              }`}
            >
              <Shuffle className="w-4 h-4" />
            </button>

            <button
              onClick={previous}
              className="p-2 text-muted-foreground hover:text-white transition-colors"
            >
              <SkipBack className="w-5 h-5" />
            </button>

            <button
              onClick={toggle}
              className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition-transform"
            >
              {isPlaying ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5 ml-0.5" />
              )}
            </button>

            <button
              onClick={next}
              className="p-2 text-muted-foreground hover:text-white transition-colors"
            >
              <SkipForward className="w-5 h-5" />
            </button>

            <button
              onClick={toggleRepeat}
              className={`p-2 transition-colors ${
                repeat !== 'off' ? 'text-primary' : 'text-muted-foreground hover:text-white'
              }`}
            >
              {repeat === 'one' ? (
                <Repeat1 className="w-4 h-4" />
              ) : (
                <Repeat className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Progress Bar */}
          <div className="w-full flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-10 text-right">
              {formatDuration(position)}
            </span>

            <div
              className="flex-1 h-1 bg-white/20 rounded-full cursor-pointer group"
              onClick={handleSeek}
            >
              <div
                className="h-full bg-white group-hover:bg-primary rounded-full relative transition-colors"
                style={{ width: `${progress}%` }}
              >
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>

            <span className="text-xs text-muted-foreground w-10">
              {formatDuration(duration)}
            </span>
          </div>
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3 w-72 justify-end">
          {/* Lyrics Toggle */}
          <button
            onClick={toggleLyrics}
            className={`p-2 transition-colors ${
              lyricsVisible ? 'text-primary' : 'text-muted-foreground hover:text-white'
            }`}
            title="Lyrics"
          >
            <Mic2 className="w-5 h-5" />
          </button>

          {/* Queue Toggle */}
          <button
            onClick={toggleQueue}
            className={`p-2 transition-colors ${
              queueVisible ? 'text-primary' : 'text-muted-foreground hover:text-white'
            }`}
            title="Queue"
          >
            <ListMusic className="w-5 h-5" />
          </button>

          {/* Volume */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleMute}
              className="p-2 text-muted-foreground hover:text-white transition-colors"
              title={muted ? 'Unmute' : 'Mute'}
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
              onChange={handleVolumeChange}
              className="w-24 h-1 bg-white/20 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
            />
          </div>
        </div>
      </div>

      {/* Panels */}
      <QueuePanel />
      <LyricsPanel />
    </div>
  );
}

export default Player;
