// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import React, { useState } from 'react';

/**
 * Player Component Stories
 *
 * The Player component provides audio playback controls with
 * progress tracking, volume control, and queue management.
 */

interface Track {
  id: string;
  title: string;
  artist: string;
  coverArt: string;
  duration: number;
}

interface PlayerProps {
  currentTrack?: Track;
  isPlaying?: boolean;
  progress?: number;
  volume?: number;
  isMuted?: boolean;
  isShuffled?: boolean;
  repeatMode?: 'off' | 'all' | 'one';
  onPlay?: () => void;
  onPause?: () => void;
  onNext?: () => void;
  onPrevious?: () => void;
  onSeek?: (progress: number) => void;
  onVolumeChange?: (volume: number) => void;
  onMuteToggle?: () => void;
  onShuffleToggle?: () => void;
  onRepeatToggle?: () => void;
}

// Mock Player component
const Player = ({
  currentTrack,
  isPlaying = false,
  progress = 0,
  volume = 80,
  isMuted = false,
  isShuffled = false,
  repeatMode = 'off',
  onPlay,
  onPause,
  onNext,
  onPrevious,
  onSeek,
  onVolumeChange,
  onMuteToggle,
  onShuffleToggle,
  onRepeatToggle,
}: PlayerProps) => {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!currentTrack) {
    return (
      <div className="bg-gray-900 border-t border-gray-800 p-4 text-center text-gray-500">
        No track selected
      </div>
    );
  }

  const currentTime = (progress / 100) * currentTrack.duration;

  return (
    <div className="bg-gray-900 border-t border-gray-800 p-4">
      <div className="max-w-7xl mx-auto flex items-center gap-6">
        {/* Track Info */}
        <div className="flex items-center gap-4 w-64">
          <img
            src={currentTrack.coverArt}
            alt={currentTrack.title}
            className="w-14 h-14 rounded-md object-cover"
          />
          <div className="min-w-0">
            <div className="font-medium text-white truncate">{currentTrack.title}</div>
            <div className="text-sm text-gray-400 truncate">{currentTrack.artist}</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex-1 flex flex-col items-center gap-2">
          <div className="flex items-center gap-4">
            <button
              onClick={onShuffleToggle}
              className={`p-2 rounded-full ${isShuffled ? 'text-purple-500' : 'text-gray-400 hover:text-white'}`}
              aria-label="Shuffle"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M4 4h3l3 3-3 3H4V4zm12 0h-3l-3 3 3 3h3V4zM4 16h3l3-3-3-3H4v6zm12 0h-3l-3-3 3-3h3v6z" />
              </svg>
            </button>

            <button
              onClick={onPrevious}
              className="p-2 text-gray-400 hover:text-white"
              aria-label="Previous"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M4 5h2v10H4V5zm10 0l-8 5 8 5V5z" />
              </svg>
            </button>

            <button
              onClick={isPlaying ? onPause : onPlay}
              className="p-3 bg-white rounded-full text-black hover:scale-105 transition"
              aria-label={isPlaying ? 'Pause' : 'Play'}
              aria-pressed={isPlaying}
            >
              {isPlaying ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                </svg>
              )}
            </button>

            <button
              onClick={onNext}
              className="p-2 text-gray-400 hover:text-white"
              aria-label="Next"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M14 5h2v10h-2V5zM4 5l8 5-8 5V5z" />
              </svg>
            </button>

            <button
              onClick={onRepeatToggle}
              className={`p-2 rounded-full ${repeatMode !== 'off' ? 'text-purple-500' : 'text-gray-400 hover:text-white'}`}
              aria-label={`Repeat: ${repeatMode}`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M4 4h12v2H6v6h3l-4 4-4-4h3V4zm12 12H4v-2h10V8h-3l4-4 4 4h-3v8z" />
              </svg>
              {repeatMode === 'one' && (
                <span className="absolute text-xs">1</span>
              )}
            </button>
          </div>

          {/* Progress Bar */}
          <div className="w-full flex items-center gap-2">
            <span className="text-xs text-gray-400 w-10 text-right">
              {formatTime(currentTime)}
            </span>
            <div
              className="flex-1 h-1 bg-gray-700 rounded-full cursor-pointer group"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const percent = ((e.clientX - rect.left) / rect.width) * 100;
                onSeek?.(percent);
              }}
            >
              <div
                className="h-full bg-purple-500 rounded-full relative"
                style={{ width: `${progress}%` }}
              >
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 transition" />
              </div>
            </div>
            <span className="text-xs text-gray-400 w-10">
              {formatTime(currentTrack.duration)}
            </span>
          </div>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2 w-40">
          <button
            onClick={onMuteToggle}
            className="p-2 text-gray-400 hover:text-white"
            aria-label={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted || volume === 0 ? (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM12.293 7.293a1 1 0 011.414 0L15 8.586l1.293-1.293a1 1 0 111.414 1.414L16.414 10l1.293 1.293a1 1 0 01-1.414 1.414L15 11.414l-1.293 1.293a1 1 0 01-1.414-1.414L13.586 10l-1.293-1.293a1 1 0 010-1.414z" />
              </svg>
            ) : volume < 50 ? (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 5.757a1 1 0 00-1.414 1.414A4.987 4.987 0 0114.5 10a4.987 4.987 0 01-1.257 2.829 1 1 0 101.414 1.414A6.987 6.987 0 0016.5 10a6.987 6.987 0 00-1.843-4.243z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.983 5.983 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" />
              </svg>
            )}
          </button>
          <input
            type="range"
            min="0"
            max="100"
            value={isMuted ? 0 : volume}
            onChange={(e) => onVolumeChange?.(Number(e.target.value))}
            className="w-full h-1 bg-gray-700 rounded-full appearance-none cursor-pointer"
            aria-label="Volume"
          />
        </div>
      </div>
    </div>
  );
};

const meta: Meta<typeof Player> = {
  title: 'Music/Player',
  component: Player,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'The main audio player component with playback controls, progress tracking, and volume control.',
      },
    },
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div className="bg-black min-h-[200px] flex items-end">
        <div className="w-full">
          <Story />
        </div>
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

const sampleTrack: Track = {
  id: '1',
  title: 'Midnight Dreams',
  artist: 'Luna Eclipse',
  coverArt: 'https://picsum.photos/seed/track1/200/200',
  duration: 234,
};

export const Default: Story = {
  args: {
    currentTrack: sampleTrack,
    isPlaying: false,
    progress: 0,
    volume: 80,
  },
};

export const Playing: Story = {
  args: {
    currentTrack: sampleTrack,
    isPlaying: true,
    progress: 35,
    volume: 80,
  },
};

export const Muted: Story = {
  args: {
    currentTrack: sampleTrack,
    isPlaying: true,
    progress: 50,
    volume: 80,
    isMuted: true,
  },
};

export const ShuffleOn: Story = {
  args: {
    currentTrack: sampleTrack,
    isPlaying: true,
    progress: 25,
    isShuffled: true,
  },
};

export const RepeatOne: Story = {
  args: {
    currentTrack: sampleTrack,
    isPlaying: true,
    progress: 75,
    repeatMode: 'one',
  },
};

export const NoTrack: Story = {
  args: {
    currentTrack: undefined,
  },
};

export const Interactive: Story = {
  render: function InteractivePlayer() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [volume, setVolume] = useState(80);
    const [isMuted, setIsMuted] = useState(false);
    const [isShuffled, setIsShuffled] = useState(false);
    const [repeatMode, setRepeatMode] = useState<'off' | 'all' | 'one'>('off');

    // Simulate playback
    React.useEffect(() => {
      if (isPlaying && progress < 100) {
        const timer = setInterval(() => {
          setProgress((p) => Math.min(p + 0.5, 100));
        }, 500);
        return () => clearInterval(timer);
      }
    }, [isPlaying, progress]);

    return (
      <Player
        currentTrack={sampleTrack}
        isPlaying={isPlaying}
        progress={progress}
        volume={volume}
        isMuted={isMuted}
        isShuffled={isShuffled}
        repeatMode={repeatMode}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onNext={() => setProgress(0)}
        onPrevious={() => setProgress(0)}
        onSeek={setProgress}
        onVolumeChange={setVolume}
        onMuteToggle={() => setIsMuted(!isMuted)}
        onShuffleToggle={() => setIsShuffled(!isShuffled)}
        onRepeatToggle={() => {
          const modes: ('off' | 'all' | 'one')[] = ['off', 'all', 'one'];
          const currentIndex = modes.indexOf(repeatMode);
          setRepeatMode(modes[(currentIndex + 1) % 3]);
        }}
      />
    );
  },
};

export const LongTrackTitle: Story = {
  args: {
    currentTrack: {
      ...sampleTrack,
      title: 'This Is A Very Long Track Title That Should Be Truncated',
      artist: 'Artist With A Really Long Name That Needs Truncation',
    },
    isPlaying: true,
    progress: 45,
  },
};
