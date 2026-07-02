// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { usePlayer } from '@/hooks/use-player';
import { useAuth } from '@/hooks/use-auth';
import { formatTime } from '@/lib/format';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Repeat,
  Repeat1,
  Shuffle,
  Volume2,
  VolumeX,
  Volume1,
  Heart,
  ListMusic,
  Maximize2,
  Mic2,
  MonitorSpeaker,
  Share2,
  MoreHorizontal,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { QueueSheet } from './queue-sheet';
import { LyricsSheet } from './lyrics-sheet';
import { DevicesSheet } from './devices-sheet';
import { FullscreenPlayer } from './fullscreen-player';

export function Player() {
  const {
    currentTrack,
    isPlaying,
    progress,
    duration,
    volume,
    isMuted,
    repeatMode,
    isShuffled,
    queue,
    play,
    pause,
    next,
    previous,
    seek,
    setVolume,
    toggleMute,
    toggleRepeat,
    toggleShuffle,
  } = usePlayer();

  const { isAuthenticated } = useAuth();
  const [showQueue, setShowQueue] = useState(false);
  const [showLyrics, setShowLyrics] = useState(false);
  const [showDevices, setShowDevices] = useState(false);
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const progressRef = useRef<HTMLDivElement>(null);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case ' ':
          e.preventDefault();
          isPlaying ? pause() : play();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          seek(Math.max(0, progress - 10));
          break;
        case 'ArrowRight':
          e.preventDefault();
          seek(Math.min(duration, progress + 10));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setVolume(Math.min(1, volume + 0.1));
          break;
        case 'ArrowDown':
          e.preventDefault();
          setVolume(Math.max(0, volume - 0.1));
          break;
        case 'm':
          toggleMute();
          break;
        case 'l':
          setShowLyrics((prev) => !prev);
          break;
        case 'q':
          setShowQueue((prev) => !prev);
          break;
        case 'f':
          setShowFullscreen((prev) => !prev);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, progress, duration, volume, play, pause, seek, setVolume, toggleMute]);

  const handleProgressClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!progressRef.current) return;
      const rect = progressRef.current.getBoundingClientRect();
      const percent = (e.clientX - rect.left) / rect.width;
      seek(percent * duration);
    },
    [duration, seek]
  );

  const VolumeIcon = isMuted || volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;

  if (!currentTrack) {
    return (
      <div className="fixed bottom-0 left-0 right-0 h-20 border-t bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Select a track to start playing
        </div>
      </div>
    );
  }

  return (
    <>
      <TooltipProvider delayDuration={300}>
        <div className="fixed bottom-0 left-0 right-0 h-20 border-t bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
          <div className="grid h-full grid-cols-3 gap-4 px-4">
            {/* Track Info */}
            <div className="flex items-center gap-3">
              <div className="relative h-14 w-14 overflow-hidden rounded">
                <Image
                  src={currentTrack.coverUrl || '/images/default-cover.png'}
                  alt={currentTrack.title}
                  fill
                  className="object-cover"
                />
                <button
                  onClick={() => setShowFullscreen(true)}
                  className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition-opacity hover:opacity-100"
                >
                  <Maximize2 className="h-5 w-5 text-white" />
                </button>
              </div>
              <div className="min-w-0">
                <Link
                  href={`/track/${currentTrack.id}`}
                  className="block truncate text-sm font-medium hover:underline"
                >
                  {currentTrack.title}
                </Link>
                <Link
                  href={`/artist/${currentTrack.artist?.id}`}
                  className="block truncate text-xs text-muted-foreground hover:underline"
                >
                  {currentTrack.artist?.name}
                </Link>
              </div>
              {isAuthenticated && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 flex-shrink-0"
                      onClick={() => setIsLiked(!isLiked)}
                    >
                      <Heart
                        className={cn('h-4 w-4', isLiked && 'fill-red-500 text-red-500')}
                      />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{isLiked ? 'Remove from Liked' : 'Add to Liked'}</TooltipContent>
                </Tooltip>
              )}
            </div>

            {/* Playback Controls */}
            <div className="flex flex-col items-center justify-center gap-1">
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={toggleShuffle}
                    >
                      <Shuffle
                        className={cn('h-4 w-4', isShuffled && 'text-primary')}
                      />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Shuffle</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={previous}
                    >
                      <SkipBack className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Previous</TooltipContent>
                </Tooltip>

                <Button
                  size="icon"
                  className="h-10 w-10 rounded-full"
                  onClick={() => (isPlaying ? pause() : play())}
                >
                  {isPlaying ? (
                    <Pause className="h-5 w-5" />
                  ) : (
                    <Play className="h-5 w-5 pl-0.5" />
                  )}
                </Button>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={next}
                    >
                      <SkipForward className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Next</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={toggleRepeat}
                    >
                      {repeatMode === 'one' ? (
                        <Repeat1 className="h-4 w-4 text-primary" />
                      ) : (
                        <Repeat
                          className={cn('h-4 w-4', repeatMode === 'all' && 'text-primary')}
                        />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {repeatMode === 'off' ? 'Enable repeat' : repeatMode === 'all' ? 'Repeat one' : 'Disable repeat'}
                  </TooltipContent>
                </Tooltip>
              </div>

              {/* Progress Bar */}
              <div className="flex w-full max-w-md items-center gap-2">
                <span className="w-10 text-right text-xs text-muted-foreground">
                  {formatTime(progress)}
                </span>
                <div
                  ref={progressRef}
                  className="relative h-1 flex-1 cursor-pointer rounded-full bg-muted"
                  onClick={handleProgressClick}
                >
                  <div
                    className="absolute h-full rounded-full bg-primary"
                    style={{ width: `${(progress / duration) * 100}%` }}
                  />
                  <div
                    className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-primary opacity-0 transition-opacity hover:opacity-100"
                    style={{ left: `calc(${(progress / duration) * 100}% - 6px)` }}
                  />
                </div>
                <span className="w-10 text-xs text-muted-foreground">
                  {formatTime(duration)}
                </span>
              </div>
            </div>

            {/* Secondary Controls */}
            <div className="flex items-center justify-end gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setShowLyrics(!showLyrics)}
                  >
                    <Mic2 className={cn('h-4 w-4', showLyrics && 'text-primary')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Lyrics</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setShowQueue(!showQueue)}
                  >
                    <ListMusic className={cn('h-4 w-4', showQueue && 'text-primary')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Queue</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setShowDevices(!showDevices)}
                  >
                    <MonitorSpeaker className={cn('h-4 w-4', showDevices && 'text-primary')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Devices</TooltipContent>
              </Tooltip>

              {/* Volume */}
              <div className="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={toggleMute}
                    >
                      <VolumeIcon className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{isMuted ? 'Unmute' : 'Mute'}</TooltipContent>
                </Tooltip>
                <Slider
                  value={[isMuted ? 0 : volume * 100]}
                  max={100}
                  step={1}
                  className="w-24"
                  onValueChange={([v]) => setVolume(v / 100)}
                />
              </div>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>
                    <Share2 className="mr-2 h-4 w-4" />
                    Share
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setShowFullscreen(true)}>
                    <Maximize2 className="mr-2 h-4 w-4" />
                    Fullscreen
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    Go to Artist
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    Go to Album
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>
      </TooltipProvider>

      {/* Sheets */}
      <QueueSheet open={showQueue} onOpenChange={setShowQueue} />
      <LyricsSheet open={showLyrics} onOpenChange={setShowLyrics} track={currentTrack} />
      <DevicesSheet open={showDevices} onOpenChange={setShowDevices} />
      <FullscreenPlayer open={showFullscreen} onOpenChange={setShowFullscreen} />
    </>
  );
}
