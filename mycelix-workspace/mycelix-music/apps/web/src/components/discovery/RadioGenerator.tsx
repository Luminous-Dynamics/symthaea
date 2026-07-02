// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';
import { usePlayerStore } from '@/store/playerStore';
import { api, Song } from '@/lib/api';
import { Radio, Play, Pause, SkipForward, Heart, X, Sparkles } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

interface RadioGeneratorProps {
  seedType: 'song' | 'artist' | 'genre';
  seedId: string;
  seedName: string;
  seedImage?: string;
  onClose?: () => void;
}

export function RadioGenerator({
  seedType,
  seedId,
  seedName,
  seedImage,
  onClose,
}: RadioGeneratorProps) {
  const { currentSong, isPlaying, playAll, toggle, next } = usePlayerStore();
  const [radioQueue, setRadioQueue] = useState<Song[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Generate radio playlist
  useEffect(() => {
    const generateRadio = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In production, call the recommendation API
        // For now, use mock data
        const mockSongs: Song[] = Array.from({ length: 25 }, (_, i) => ({
          id: `radio-${seedId}-${i}`,
          title: `Radio Track ${i + 1}`,
          artist: `Similar Artist ${(i % 5) + 1}`,
          artistAddress: `0x${(i + 1).toString().padStart(40, '0')}`,
          duration: 180 + Math.floor(Math.random() * 120),
          coverArt: `https://picsum.photos/seed/radio-${seedId}-${i}/200`,
          audioUrl: '/sample-audio.mp3',
          genre: 'Electronic',
        }));

        setRadioQueue(mockSongs);

        // Auto-start playback
        if (mockSongs.length > 0) {
          playAll(mockSongs, 0);
        }
      } catch (err) {
        setError('Failed to generate radio. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    generateRadio();
  }, [seedType, seedId, playAll]);

  const currentIndex = radioQueue.findIndex((s) => s.id === currentSong?.id);
  const upcomingSongs = radioQueue.slice(currentIndex + 1, currentIndex + 6);

  return (
    <div className="fixed inset-0 z-[55] bg-black/90 backdrop-blur-lg flex items-center justify-center">
      <div className="w-full max-w-2xl mx-auto p-8">
        {/* Close Button */}
        {onClose && (
          <button
            onClick={onClose}
            className="absolute top-6 right-6 p-2 text-white/70 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        )}

        {/* Radio Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Radio className="w-6 h-6 text-purple-400" />
            <span className="text-sm font-medium text-purple-400 uppercase tracking-wider">
              {seedType} Radio
            </span>
          </div>
          <h1 className="text-3xl font-bold mb-2">
            {seedName} Radio
          </h1>
          <p className="text-muted-foreground">
            Endless music inspired by {seedType === 'song' ? 'this track' : seedName}
          </p>
        </div>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <div className="w-16 h-16 rounded-full border-4 border-purple-500 border-t-transparent animate-spin mb-4" />
            <p className="text-muted-foreground">Generating your radio station...</p>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <p className="text-red-500 mb-4">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
            >
              Try Again
            </button>
          </div>
        ) : (
          <>
            {/* Now Playing */}
            {currentSong && (
              <div className="bg-white/5 rounded-2xl p-6 mb-6">
                <div className="flex items-center gap-6">
                  <div className="relative w-24 h-24 rounded-lg overflow-hidden">
                    <Image
                      src={currentSong.coverArt || '/placeholder-album.png'}
                      alt={currentSong.title}
                      fill
                      className="object-cover"
                    />
                    {isPlaying && (
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                        <div className="flex gap-1 items-end h-6">
                          <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite]" style={{ height: '60%' }} />
                          <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite_0.2s]" style={{ height: '100%' }} />
                          <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite_0.4s]" style={{ height: '40%' }} />
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-purple-400 mb-1">Now Playing</p>
                    <h3 className="text-xl font-bold truncate mb-1">
                      {currentSong.title}
                    </h3>
                    <p className="text-muted-foreground truncate">
                      {currentSong.artist}
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <button className="p-3 text-white/70 hover:text-white transition-colors">
                      <Heart className="w-6 h-6" />
                    </button>
                    <button
                      onClick={toggle}
                      className="w-14 h-14 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition-transform"
                    >
                      {isPlaying ? (
                        <Pause className="w-6 h-6" />
                      ) : (
                        <Play className="w-6 h-6 ml-1" />
                      )}
                    </button>
                    <button
                      onClick={next}
                      className="p-3 text-white/70 hover:text-white transition-colors"
                    >
                      <SkipForward className="w-6 h-6" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Up Next */}
            <div className="bg-white/5 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Up Next</h3>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Sparkles className="w-4 h-4" />
                  Auto-generated queue
                </div>
              </div>

              <div className="space-y-2">
                {upcomingSongs.map((song, i) => (
                  <div
                    key={song.id}
                    className="flex items-center gap-4 p-2 rounded-lg hover:bg-white/5 transition-colors"
                  >
                    <span className="w-6 text-center text-muted-foreground text-sm">
                      {i + 1}
                    </span>
                    <div className="relative w-10 h-10 rounded overflow-hidden">
                      <Image
                        src={song.coverArt || '/placeholder-album.png'}
                        alt={song.title}
                        fill
                        className="object-cover"
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{song.title}</p>
                      <p className="text-sm text-muted-foreground truncate">
                        {song.artist}
                      </p>
                    </div>
                    <button className="p-2 text-white/50 hover:text-white transition-colors opacity-0 group-hover:opacity-100">
                      <Heart className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>

              {radioQueue.length > upcomingSongs.length + currentIndex + 1 && (
                <p className="text-center text-sm text-muted-foreground mt-4">
                  + {radioQueue.length - upcomingSongs.length - currentIndex - 1} more tracks
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Button to start radio from a song
export function StartRadioButton({
  seedType,
  seedId,
  seedName,
  seedImage,
  className = '',
}: Omit<RadioGeneratorProps, 'onClose'> & { className?: string }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={`flex items-center gap-2 ${className}`}
      >
        <Radio className="w-4 h-4" />
        Start Radio
      </button>

      {isOpen && (
        <RadioGenerator
          seedType={seedType}
          seedId={seedId}
          seedName={seedName}
          seedImage={seedImage}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
