// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Repeat,
  Repeat1,
  Shuffle,
  Heart,
  ListMusic,
  Maximize2,
  Minimize2,
  Share2,
  MoreHorizontal,
  Radio,
  Mic2,
  Airplay,
  Download,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/design-system/Button';

// ==================== Types ====================

interface Track {
  id: string;
  title: string;
  artist: string;
  album?: string;
  duration: number;
  coverUrl?: string;
  streamUrl: string;
  waveformData?: number[];
  isLiked?: boolean;
  explicit?: boolean;
}

interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  repeatMode: 'off' | 'all' | 'one';
  isShuffled: boolean;
  isBuffering: boolean;
  audioQuality: 'low' | 'medium' | 'high' | 'lossless';
}

interface EnhancedPlayerProps {
  track: Track | null;
  queue: Track[];
  onPlay: () => void;
  onPause: () => void;
  onSeek: (time: number) => void;
  onSkipNext: () => void;
  onSkipPrevious: () => void;
  onVolumeChange: (volume: number) => void;
  onRepeatChange: (mode: 'off' | 'all' | 'one') => void;
  onShuffleChange: (shuffled: boolean) => void;
  onLikeToggle: (trackId: string) => void;
  onQueueOpen: () => void;
  onShare: (trackId: string) => void;
  playerState: PlayerState;
  className?: string;
}

// ==================== Waveform Component ====================

const Waveform: React.FC<{
  data: number[];
  progress: number;
  onSeek: (progress: number) => void;
  height?: number;
}> = ({ data, progress, onSeek, height = 48 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const barWidth = 2;
    const gap = 1;
    const totalBars = Math.floor(rect.width / (barWidth + gap));
    const samplesPerBar = Math.floor(data.length / totalBars);

    ctx.clearRect(0, 0, rect.width, rect.height);

    for (let i = 0; i < totalBars; i++) {
      const sampleIndex = i * samplesPerBar;
      const amplitude = data[sampleIndex] || 0;
      const barHeight = Math.max(2, amplitude * rect.height * 0.8);
      const x = i * (barWidth + gap);
      const y = (rect.height - barHeight) / 2;

      const progressX = progress * rect.width;
      const isPlayed = x < progressX;

      ctx.fillStyle = isPlayed
        ? 'rgb(99, 102, 241)'
        : 'rgba(156, 163, 175, 0.5)';
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 1);
      ctx.fill();
    }
  }, [data, progress]);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const progress = x / rect.width;
    onSeek(Math.max(0, Math.min(1, progress)));
  };

  return (
    <div
      ref={containerRef}
      className="relative cursor-pointer group"
      style={{ height }}
      onClick={handleClick}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ height }}
      />
      <div
        className="absolute top-0 left-0 h-full bg-primary-500/20 pointer-events-none transition-all"
        style={{ width: `${progress * 100}%` }}
      />
    </div>
  );
};

// ==================== Progress Bar ====================

const ProgressBar: React.FC<{
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  buffered?: number;
}> = ({ currentTime, duration, onSeek, buffered = 0 }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [hoverPosition, setHoverPosition] = useState<number | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
  const bufferedProgress = duration > 0 ? (buffered / duration) * 100 : 0;

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = barRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const position = (x / rect.width) * duration;
    setHoverPosition(Math.max(0, Math.min(duration, position)));
  };

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = barRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const time = (x / rect.width) * duration;
    onSeek(Math.max(0, Math.min(duration, time)));
  };

  return (
    <div className="flex items-center gap-2 w-full">
      <span className="text-xs text-gray-400 w-10 text-right font-mono">
        {formatTime(currentTime)}
      </span>
      <div
        ref={barRef}
        className="relative flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer group"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverPosition(null)}
      >
        {/* Buffered */}
        <div
          className="absolute h-full bg-gray-300 dark:bg-gray-600 rounded-full"
          style={{ width: `${bufferedProgress}%` }}
        />
        {/* Progress */}
        <div
          className="absolute h-full bg-primary-500 rounded-full transition-all"
          style={{ width: `${progress}%` }}
        />
        {/* Thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-primary-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-md"
          style={{ left: `${progress}%` }}
        />
        {/* Hover tooltip */}
        {hoverPosition !== null && (
          <div
            className="absolute -top-8 -translate-x-1/2 px-2 py-1 bg-gray-900 text-white text-xs rounded shadow-lg"
            style={{ left: `${(hoverPosition / duration) * 100}%` }}
          >
            {formatTime(hoverPosition)}
          </div>
        )}
      </div>
      <span className="text-xs text-gray-400 w-10 font-mono">
        {formatTime(duration)}
      </span>
    </div>
  );
};

// ==================== Volume Control ====================

const VolumeControl: React.FC<{
  volume: number;
  isMuted: boolean;
  onVolumeChange: (volume: number) => void;
  onMuteToggle: () => void;
}> = ({ volume, isMuted, onVolumeChange, onMuteToggle }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className="flex items-center gap-2"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <button
        onClick={onMuteToggle}
        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
      >
        {isMuted || volume === 0 ? (
          <VolumeX className="w-5 h-5 text-gray-500" />
        ) : (
          <Volume2 className="w-5 h-5 text-gray-500" />
        )}
      </button>
      <div
        className={cn(
          'overflow-hidden transition-all duration-200',
          isHovered ? 'w-24 opacity-100' : 'w-0 opacity-0'
        )}
      >
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={isMuted ? 0 : volume}
          onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
          className="w-full h-1 bg-gray-200 dark:bg-gray-700 rounded-full appearance-none cursor-pointer accent-primary-500"
        />
      </div>
    </div>
  );
};

// ==================== Main Player Component ====================

export const EnhancedPlayer: React.FC<EnhancedPlayerProps> = ({
  track,
  queue,
  onPlay,
  onPause,
  onSeek,
  onSkipNext,
  onSkipPrevious,
  onVolumeChange,
  onRepeatChange,
  onShuffleChange,
  onLikeToggle,
  onQueueOpen,
  onShare,
  playerState,
  className,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showLyrics, setShowLyrics] = useState(false);

  const {
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    repeatMode,
    isShuffled,
    isBuffering,
    audioQuality,
  } = playerState;

  const handleSeek = (time: number) => {
    onSeek(time);
  };

  const handleRepeatClick = () => {
    const modes: ('off' | 'all' | 'one')[] = ['off', 'all', 'one'];
    const currentIndex = modes.indexOf(repeatMode);
    const nextMode = modes[(currentIndex + 1) % modes.length];
    onRepeatChange(nextMode);
  };

  if (!track) {
    return (
      <div className={cn(
        'fixed bottom-0 left-0 right-0 h-20 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 flex items-center justify-center text-gray-400',
        className
      )}>
        <p>No track selected</p>
      </div>
    );
  }

  // Mini player
  if (!isExpanded) {
    return (
      <div className={cn(
        'fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 shadow-lg z-50',
        className
      )}>
        {/* Progress bar at top */}
        <div className="h-1 bg-gray-200 dark:bg-gray-800">
          <div
            className="h-full bg-primary-500 transition-all"
            style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>

        <div className="h-[72px] px-4 flex items-center gap-4">
          {/* Track info */}
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-200 dark:bg-gray-800 flex-shrink-0">
              {track.coverUrl ? (
                <img
                  src={track.coverUrl}
                  alt={track.title}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-2xl">
                  🎵
                </div>
              )}
            </div>
            <div className="min-w-0">
              <p className="font-medium text-gray-900 dark:text-white truncate">
                {track.title}
              </p>
              <p className="text-sm text-gray-500 truncate">{track.artist}</p>
            </div>
            <button
              onClick={() => onLikeToggle(track.id)}
              className="ml-2 p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Heart
                className={cn(
                  'w-5 h-5',
                  track.isLiked ? 'fill-red-500 text-red-500' : 'text-gray-400'
                )}
              />
            </button>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={onSkipPrevious}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <SkipBack className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button
              onClick={isPlaying ? onPause : onPlay}
              className="w-12 h-12 bg-primary-600 hover:bg-primary-700 rounded-full flex items-center justify-center transition-colors"
            >
              {isBuffering ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : isPlaying ? (
                <Pause className="w-6 h-6 text-white" />
              ) : (
                <Play className="w-6 h-6 text-white ml-0.5" />
              )}
            </button>
            <button
              onClick={onSkipNext}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <SkipForward className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-2">
            <VolumeControl
              volume={volume}
              isMuted={isMuted}
              onVolumeChange={onVolumeChange}
              onMuteToggle={() => onVolumeChange(isMuted ? volume : 0)}
            />
            <button
              onClick={onQueueOpen}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <ListMusic className="w-5 h-5 text-gray-500" />
            </button>
            <button
              onClick={() => setIsExpanded(true)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Maximize2 className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Full screen player
  return (
    <div className="fixed inset-0 bg-gradient-to-b from-gray-900 via-gray-900 to-black z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4">
        <button
          onClick={() => setIsExpanded(false)}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
        >
          <Minimize2 className="w-6 h-6 text-white" />
        </button>
        <div className="text-center">
          <p className="text-xs text-gray-400 uppercase tracking-wider">
            Playing from
          </p>
          <p className="text-sm text-white font-medium">{track.album || 'Library'}</p>
        </div>
        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <MoreHorizontal className="w-6 h-6 text-white" />
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center px-8 py-4">
        {/* Album art */}
        <div className="w-80 h-80 rounded-2xl overflow-hidden shadow-2xl mb-8">
          {track.coverUrl ? (
            <img
              src={track.coverUrl}
              alt={track.title}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center text-8xl">
              🎵
            </div>
          )}
        </div>

        {/* Track info */}
        <div className="text-center mb-8 w-full max-w-md">
          <h2 className="text-2xl font-bold text-white mb-1 truncate">
            {track.title}
          </h2>
          <p className="text-lg text-gray-400">{track.artist}</p>
        </div>

        {/* Progress */}
        <div className="w-full max-w-md mb-8">
          {track.waveformData ? (
            <Waveform
              data={track.waveformData}
              progress={duration > 0 ? currentTime / duration : 0}
              onSeek={(p) => handleSeek(p * duration)}
              height={64}
            />
          ) : (
            <ProgressBar
              currentTime={currentTime}
              duration={duration}
              onSeek={handleSeek}
            />
          )}
          <div className="flex justify-between text-xs text-gray-400 mt-2 font-mono">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Main controls */}
        <div className="flex items-center gap-6 mb-8">
          <button
            onClick={() => onShuffleChange(!isShuffled)}
            className={cn(
              'p-3 rounded-full transition-colors',
              isShuffled ? 'text-primary-400' : 'text-gray-400 hover:text-white'
            )}
          >
            <Shuffle className="w-6 h-6" />
          </button>
          <button
            onClick={onSkipPrevious}
            className="p-3 text-white hover:scale-105 transition-transform"
          >
            <SkipBack className="w-8 h-8" />
          </button>
          <button
            onClick={isPlaying ? onPause : onPlay}
            className="w-16 h-16 bg-white rounded-full flex items-center justify-center hover:scale-105 transition-transform"
          >
            {isBuffering ? (
              <div className="w-8 h-8 border-3 border-gray-900 border-t-transparent rounded-full animate-spin" />
            ) : isPlaying ? (
              <Pause className="w-8 h-8 text-gray-900" />
            ) : (
              <Play className="w-8 h-8 text-gray-900 ml-1" />
            )}
          </button>
          <button
            onClick={onSkipNext}
            className="p-3 text-white hover:scale-105 transition-transform"
          >
            <SkipForward className="w-8 h-8" />
          </button>
          <button
            onClick={handleRepeatClick}
            className={cn(
              'p-3 rounded-full transition-colors',
              repeatMode !== 'off' ? 'text-primary-400' : 'text-gray-400 hover:text-white'
            )}
          >
            {repeatMode === 'one' ? (
              <Repeat1 className="w-6 h-6" />
            ) : (
              <Repeat className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* Secondary controls */}
        <div className="flex items-center justify-between w-full max-w-md">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onLikeToggle(track.id)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <Heart
                className={cn(
                  'w-6 h-6',
                  track.isLiked ? 'fill-red-500 text-red-500' : 'text-gray-400'
                )}
              />
            </button>
            <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
              <Download className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          <VolumeControl
            volume={volume}
            isMuted={isMuted}
            onVolumeChange={onVolumeChange}
            onMuteToggle={() => onVolumeChange(isMuted ? volume : 0)}
          />

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowLyrics(!showLyrics)}
              className={cn(
                'p-2 rounded-lg transition-colors',
                showLyrics ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-gray-400'
              )}
            >
              <Mic2 className="w-6 h-6" />
            </button>
            <button
              onClick={onQueueOpen}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <ListMusic className="w-6 h-6 text-gray-400" />
            </button>
            <button
              onClick={() => onShare(track.id)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <Share2 className="w-6 h-6 text-gray-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Audio quality indicator */}
      <div className="flex justify-center pb-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-white/10 rounded-full">
          <Radio className="w-4 h-4 text-primary-400" />
          <span className="text-xs text-gray-300 uppercase">{audioQuality}</span>
        </div>
      </div>
    </div>
  );
};

// Helper function
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export default EnhancedPlayer;
