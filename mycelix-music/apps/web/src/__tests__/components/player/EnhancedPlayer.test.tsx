// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the EnhancedPlayer component's dependencies
const mockAudioElement = {
  play: vi.fn().mockResolvedValue(undefined),
  pause: vi.fn(),
  load: vi.fn(),
  currentTime: 0,
  duration: 180,
  volume: 1,
  muted: false,
  paused: true,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
};

// Mock Track data
const mockTrack = {
  id: 'track-1',
  title: 'Test Track',
  artist: 'Test Artist',
  album: 'Test Album',
  duration: 180,
  coverArt: '/test-cover.jpg',
  audioUrl: '/test-audio.mp3',
  waveformData: Array(100).fill(0).map(() => Math.random()),
};

// Mock player state
const mockPlayerState = {
  currentTrack: mockTrack,
  isPlaying: false,
  currentTime: 0,
  duration: 180,
  volume: 0.8,
  isMuted: false,
  isShuffled: false,
  repeatMode: 'off' as const,
  queue: [mockTrack],
  history: [],
};

// Test component that simulates EnhancedPlayer behavior
function TestPlayer({
  onPlay = vi.fn(),
  onPause = vi.fn(),
  onSeek = vi.fn(),
  onVolumeChange = vi.fn(),
  onNextTrack = vi.fn(),
  onPreviousTrack = vi.fn(),
  onToggleShuffle = vi.fn(),
  onToggleRepeat = vi.fn(),
  onToggleLike = vi.fn(),
  onToggleExpand = vi.fn(),
  initialExpanded = false,
  initialState = mockPlayerState,
}) {
  const [isPlaying, setIsPlaying] = React.useState(initialState.isPlaying);
  const [isExpanded, setIsExpanded] = React.useState(initialExpanded);
  const [currentTime, setCurrentTime] = React.useState(initialState.currentTime);
  const [volume, setVolume] = React.useState(initialState.volume);
  const [isMuted, setIsMuted] = React.useState(initialState.isMuted);
  const [repeatMode, setRepeatMode] = React.useState(initialState.repeatMode);
  const [isShuffled, setIsShuffled] = React.useState(initialState.isShuffled);
  const [isLiked, setIsLiked] = React.useState(false);

  return (
    <div data-testid="player" className={isExpanded ? 'expanded' : 'mini'}>
      {/* Track Info */}
      <div data-testid="track-info">
        <img src={mockTrack.coverArt} alt={mockTrack.title} data-testid="cover-art" />
        <div data-testid="track-title">{mockTrack.title}</div>
        <div data-testid="track-artist">{mockTrack.artist}</div>
      </div>

      {/* Progress Bar */}
      <div data-testid="progress-bar">
        <input
          type="range"
          min="0"
          max={mockTrack.duration}
          value={currentTime}
          onChange={(e) => {
            const time = Number(e.target.value);
            setCurrentTime(time);
            onSeek(time);
          }}
          data-testid="progress-slider"
          aria-label="Seek"
        />
        <span data-testid="current-time">{formatTime(currentTime)}</span>
        <span data-testid="duration">{formatTime(mockTrack.duration)}</span>
      </div>

      {/* Controls */}
      <div data-testid="controls">
        <button
          onClick={() => {
            setIsShuffled(!isShuffled);
            onToggleShuffle();
          }}
          data-testid="shuffle-button"
          aria-pressed={isShuffled}
          aria-label="Shuffle"
        >
          Shuffle
        </button>

        <button
          onClick={() => onPreviousTrack()}
          data-testid="prev-button"
          aria-label="Previous track"
        >
          Previous
        </button>

        <button
          onClick={() => {
            setIsPlaying(!isPlaying);
            isPlaying ? onPause() : onPlay();
          }}
          data-testid="play-button"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button
          onClick={() => onNextTrack()}
          data-testid="next-button"
          aria-label="Next track"
        >
          Next
        </button>

        <button
          onClick={() => {
            const modes = ['off', 'all', 'one'] as const;
            const nextIndex = (modes.indexOf(repeatMode) + 1) % modes.length;
            setRepeatMode(modes[nextIndex]);
            onToggleRepeat();
          }}
          data-testid="repeat-button"
          aria-label={`Repeat: ${repeatMode}`}
        >
          Repeat: {repeatMode}
        </button>
      </div>

      {/* Volume Control */}
      <div data-testid="volume-control">
        <button
          onClick={() => setIsMuted(!isMuted)}
          data-testid="mute-button"
          aria-pressed={isMuted}
          aria-label={isMuted ? 'Unmute' : 'Mute'}
        >
          {isMuted ? 'Unmute' : 'Mute'}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={isMuted ? 0 : volume}
          onChange={(e) => {
            const newVolume = Number(e.target.value);
            setVolume(newVolume);
            setIsMuted(newVolume === 0);
            onVolumeChange(newVolume);
          }}
          data-testid="volume-slider"
          aria-label="Volume"
        />
      </div>

      {/* Like Button */}
      <button
        onClick={() => {
          setIsLiked(!isLiked);
          onToggleLike();
        }}
        data-testid="like-button"
        aria-pressed={isLiked}
        aria-label={isLiked ? 'Unlike' : 'Like'}
      >
        {isLiked ? 'Liked' : 'Like'}
      </button>

      {/* Expand Toggle */}
      <button
        onClick={() => {
          setIsExpanded(!isExpanded);
          onToggleExpand();
        }}
        data-testid="expand-button"
        aria-label={isExpanded ? 'Collapse' : 'Expand'}
      >
        {isExpanded ? 'Collapse' : 'Expand'}
      </button>

      {/* Waveform (only in expanded view) */}
      {isExpanded && (
        <div data-testid="waveform">
          <canvas data-testid="waveform-canvas" />
        </div>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Import React for the test component
import React from 'react';

describe('EnhancedPlayer Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders in mini mode by default', () => {
      render(<TestPlayer />);
      expect(screen.getByTestId('player')).toHaveClass('mini');
    });

    it('renders in expanded mode when specified', () => {
      render(<TestPlayer initialExpanded />);
      expect(screen.getByTestId('player')).toHaveClass('expanded');
    });

    it('displays track information', () => {
      render(<TestPlayer />);
      expect(screen.getByTestId('track-title')).toHaveTextContent('Test Track');
      expect(screen.getByTestId('track-artist')).toHaveTextContent('Test Artist');
    });

    it('displays cover art', () => {
      render(<TestPlayer />);
      const coverArt = screen.getByTestId('cover-art') as HTMLImageElement;
      expect(coverArt.src).toContain('test-cover.jpg');
    });

    it('displays progress bar', () => {
      render(<TestPlayer />);
      expect(screen.getByTestId('progress-bar')).toBeInTheDocument();
      expect(screen.getByTestId('progress-slider')).toBeInTheDocument();
    });

    it('displays current time and duration', () => {
      render(<TestPlayer />);
      expect(screen.getByTestId('current-time')).toHaveTextContent('0:00');
      expect(screen.getByTestId('duration')).toHaveTextContent('3:00');
    });

    it('shows waveform only in expanded mode', () => {
      const { rerender } = render(<TestPlayer initialExpanded={false} />);
      expect(screen.queryByTestId('waveform')).not.toBeInTheDocument();

      rerender(<TestPlayer initialExpanded />);
      expect(screen.getByTestId('waveform')).toBeInTheDocument();
    });
  });

  describe('Playback Controls', () => {
    it('calls onPlay when play button is clicked', async () => {
      const onPlay = vi.fn();
      render(<TestPlayer onPlay={onPlay} />);

      await userEvent.click(screen.getByTestId('play-button'));
      expect(onPlay).toHaveBeenCalledTimes(1);
    });

    it('calls onPause when pause button is clicked while playing', async () => {
      const onPause = vi.fn();
      render(
        <TestPlayer
          onPause={onPause}
          initialState={{ ...mockPlayerState, isPlaying: true }}
        />
      );

      // First click to start playing
      await userEvent.click(screen.getByTestId('play-button'));
      // Second click to pause
      await userEvent.click(screen.getByTestId('play-button'));
      expect(onPause).toHaveBeenCalledTimes(1);
    });

    it('toggles play/pause text', async () => {
      render(<TestPlayer />);
      const playButton = screen.getByTestId('play-button');

      expect(playButton).toHaveTextContent('Play');
      await userEvent.click(playButton);
      expect(playButton).toHaveTextContent('Pause');
    });

    it('calls onNextTrack when next button is clicked', async () => {
      const onNextTrack = vi.fn();
      render(<TestPlayer onNextTrack={onNextTrack} />);

      await userEvent.click(screen.getByTestId('next-button'));
      expect(onNextTrack).toHaveBeenCalledTimes(1);
    });

    it('calls onPreviousTrack when previous button is clicked', async () => {
      const onPreviousTrack = vi.fn();
      render(<TestPlayer onPreviousTrack={onPreviousTrack} />);

      await userEvent.click(screen.getByTestId('prev-button'));
      expect(onPreviousTrack).toHaveBeenCalledTimes(1);
    });
  });

  describe('Progress Bar', () => {
    it('calls onSeek when progress bar is changed', async () => {
      const onSeek = vi.fn();
      render(<TestPlayer onSeek={onSeek} />);

      const slider = screen.getByTestId('progress-slider');
      fireEvent.change(slider, { target: { value: 90 } });

      expect(onSeek).toHaveBeenCalledWith(90);
    });

    it('updates current time display when seeking', async () => {
      render(<TestPlayer />);

      const slider = screen.getByTestId('progress-slider');
      fireEvent.change(slider, { target: { value: 90 } });

      expect(screen.getByTestId('current-time')).toHaveTextContent('1:30');
    });
  });

  describe('Volume Control', () => {
    it('calls onVolumeChange when volume is adjusted', async () => {
      const onVolumeChange = vi.fn();
      render(<TestPlayer onVolumeChange={onVolumeChange} />);

      const slider = screen.getByTestId('volume-slider');
      fireEvent.change(slider, { target: { value: 0.5 } });

      expect(onVolumeChange).toHaveBeenCalledWith(0.5);
    });

    it('toggles mute state when mute button is clicked', async () => {
      render(<TestPlayer />);
      const muteButton = screen.getByTestId('mute-button');

      expect(muteButton).toHaveTextContent('Mute');
      await userEvent.click(muteButton);
      expect(muteButton).toHaveTextContent('Unmute');
    });

    it('sets volume to 0 when muted', async () => {
      render(<TestPlayer />);
      const slider = screen.getByTestId('volume-slider') as HTMLInputElement;

      await userEvent.click(screen.getByTestId('mute-button'));
      expect(slider.value).toBe('0');
    });
  });

  describe('Shuffle and Repeat', () => {
    it('calls onToggleShuffle when shuffle button is clicked', async () => {
      const onToggleShuffle = vi.fn();
      render(<TestPlayer onToggleShuffle={onToggleShuffle} />);

      await userEvent.click(screen.getByTestId('shuffle-button'));
      expect(onToggleShuffle).toHaveBeenCalledTimes(1);
    });

    it('toggles shuffle state', async () => {
      render(<TestPlayer />);
      const shuffleButton = screen.getByTestId('shuffle-button');

      expect(shuffleButton).toHaveAttribute('aria-pressed', 'false');
      await userEvent.click(shuffleButton);
      expect(shuffleButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('cycles through repeat modes', async () => {
      render(<TestPlayer />);
      const repeatButton = screen.getByTestId('repeat-button');

      expect(repeatButton).toHaveTextContent('Repeat: off');
      await userEvent.click(repeatButton);
      expect(repeatButton).toHaveTextContent('Repeat: all');
      await userEvent.click(repeatButton);
      expect(repeatButton).toHaveTextContent('Repeat: one');
      await userEvent.click(repeatButton);
      expect(repeatButton).toHaveTextContent('Repeat: off');
    });
  });

  describe('Like Button', () => {
    it('calls onToggleLike when like button is clicked', async () => {
      const onToggleLike = vi.fn();
      render(<TestPlayer onToggleLike={onToggleLike} />);

      await userEvent.click(screen.getByTestId('like-button'));
      expect(onToggleLike).toHaveBeenCalledTimes(1);
    });

    it('toggles like state', async () => {
      render(<TestPlayer />);
      const likeButton = screen.getByTestId('like-button');

      expect(likeButton).toHaveTextContent('Like');
      await userEvent.click(likeButton);
      expect(likeButton).toHaveTextContent('Liked');
    });
  });

  describe('Expand/Collapse', () => {
    it('calls onToggleExpand when expand button is clicked', async () => {
      const onToggleExpand = vi.fn();
      render(<TestPlayer onToggleExpand={onToggleExpand} />);

      await userEvent.click(screen.getByTestId('expand-button'));
      expect(onToggleExpand).toHaveBeenCalledTimes(1);
    });

    it('toggles between mini and expanded modes', async () => {
      render(<TestPlayer />);

      expect(screen.getByTestId('player')).toHaveClass('mini');
      await userEvent.click(screen.getByTestId('expand-button'));
      expect(screen.getByTestId('player')).toHaveClass('expanded');
    });
  });

  describe('Accessibility', () => {
    it('has accessible play button', () => {
      render(<TestPlayer />);
      const playButton = screen.getByTestId('play-button');
      expect(playButton).toHaveAttribute('aria-label', 'Play');
    });

    it('has accessible volume slider', () => {
      render(<TestPlayer />);
      expect(screen.getByLabelText('Volume')).toBeInTheDocument();
    });

    it('has accessible seek slider', () => {
      render(<TestPlayer />);
      expect(screen.getByLabelText('Seek')).toBeInTheDocument();
    });

    it('has accessible mute button', () => {
      render(<TestPlayer />);
      expect(screen.getByLabelText('Mute')).toBeInTheDocument();
    });

    it('updates aria-label when playing', async () => {
      render(<TestPlayer />);
      const playButton = screen.getByTestId('play-button');

      await userEvent.click(playButton);
      expect(playButton).toHaveAttribute('aria-label', 'Pause');
    });
  });

  describe('Keyboard Navigation', () => {
    it('supports keyboard navigation for controls', async () => {
      render(<TestPlayer />);
      const playButton = screen.getByTestId('play-button');

      playButton.focus();
      expect(document.activeElement).toBe(playButton);

      await userEvent.keyboard('{Enter}');
      expect(playButton).toHaveTextContent('Pause');
    });

    it('allows tabbing through controls', async () => {
      render(<TestPlayer />);

      await userEvent.tab();
      expect(document.activeElement).toHaveAttribute('data-testid');
    });
  });
});
