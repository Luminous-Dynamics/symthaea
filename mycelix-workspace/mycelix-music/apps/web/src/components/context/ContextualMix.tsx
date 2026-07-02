// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useContextStore, DEFAULT_PROFILES } from '@/store/contextStore';
import { useContextualRecommendations } from '@/hooks/useContextDetection';
import { usePlayerStore } from '@/store/playerStore';
import { api } from '@/lib/api';
import {
  Play,
  Pause,
  RefreshCw,
  Sparkles,
  Clock,
  Music2,
  ChevronRight,
  Volume2,
} from 'lucide-react';
import Image from 'next/image';

interface ContextualTrack {
  id: string;
  title: string;
  artist: string;
  artistId: string;
  coverArt: string;
  duration: number;
  audioUrl: string;
  matchScore: number; // How well it matches the context (0-100)
}

interface ContextualMixProps {
  compact?: boolean;
  autoPlay?: boolean;
}

export function ContextualMix({ compact = false, autoPlay = false }: ContextualMixProps) {
  const { context, attributes, getRecommendationParams, isEnabled } = useContextualRecommendations();
  const { activeProfileId, customProfiles } = useContextStore();
  const { currentSong, isPlaying, toggle, setQueue, setSong } = usePlayerStore();

  const [isRefreshing, setIsRefreshing] = useState(false);

  // Find active profile name
  const allProfiles = [...DEFAULT_PROFILES, ...customProfiles];
  const activeProfile = activeProfileId
    ? allProfiles.find(p => p.id === activeProfileId)
    : null;

  // Get contextual label
  const getContextLabel = () => {
    if (activeProfile) return activeProfile.name;

    const labels: string[] = [];
    labels.push(context.timeOfDay.replace('_', ' '));
    if (context.weather) labels.push(context.weather);
    labels.push(context.mood);

    return labels.slice(0, 2).map(l => l.charAt(0).toUpperCase() + l.slice(1)).join(' • ');
  };

  // Fetch contextual recommendations
  const { data: tracks, isLoading, refetch } = useQuery<ContextualTrack[]>({
    queryKey: ['contextual-mix', attributes, activeProfileId],
    queryFn: async () => {
      const params = getRecommendationParams();
      if (!params) return [];

      // In production, this would call the recommendation API
      // const response = await api.get('/recommendations/contextual', { params });
      // return response.data;

      // Mock data for now
      return generateMockContextualTracks(attributes);
    },
    enabled: isEnabled,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });

  // Handle refresh
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetch();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  // Play the contextual mix
  const playMix = () => {
    if (tracks && tracks.length > 0) {
      const queueSongs = tracks.map(t => ({
        id: t.id,
        title: t.title,
        artist: t.artist,
        artistId: t.artistId,
        coverArt: t.coverArt,
        duration: t.duration,
        audioUrl: t.audioUrl,
      }));

      setSong(queueSongs[0]);
      setQueue(queueSongs.slice(1));
    }
  };

  // Check if current song is from this mix
  const isPlayingMix = tracks?.some(t => t.id === currentSong?.id);

  if (!isEnabled) {
    return null;
  }

  if (compact) {
    return (
      <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs text-purple-300 uppercase tracking-wider">For You Now</p>
              <p className="font-semibold">{getContextLabel()}</p>
            </div>
          </div>

          <button
            onClick={isPlayingMix && isPlaying ? toggle : playMix}
            className="w-10 h-10 rounded-full bg-white flex items-center justify-center text-black"
          >
            {isPlayingMix && isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5 ml-0.5" />
            )}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-purple-500/10 via-pink-500/10 to-orange-500/10 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="p-6 pb-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Sparkles className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm text-purple-300 uppercase tracking-wider">
                {activeProfile ? 'Selected Vibe' : 'For Your Moment'}
              </p>
              <h2 className="text-2xl font-bold">{getContextLabel()}</h2>
              <p className="text-sm text-muted-foreground">
                {tracks?.length || 0} tracks • ~{Math.round((tracks?.reduce((sum, t) => sum + t.duration, 0) || 0) / 60)} min
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={isLoading || isRefreshing}
              className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-50"
              title="Refresh recommendations"
            >
              <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={isPlayingMix && isPlaying ? toggle : playMix}
              className="flex items-center gap-2 px-6 py-3 bg-white text-black rounded-full font-semibold hover:scale-105 transition-transform"
            >
              {isPlayingMix && isPlaying ? (
                <>
                  <Pause className="w-5 h-5" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Play Mix
                </>
              )}
            </button>
          </div>
        </div>

        {/* Music attributes display */}
        <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
          <div className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            <span>{attributes.tempo.min}-{attributes.tempo.max} BPM</span>
          </div>
          <div className="flex items-center gap-1">
            <Volume2 className="w-4 h-4" />
            <span>Energy: {Math.round(attributes.energy * 100)}%</span>
          </div>
          {attributes.genres && (
            <div className="flex items-center gap-1">
              <Music2 className="w-4 h-4" />
              <span>{attributes.genres.slice(0, 3).join(', ')}</span>
            </div>
          )}
        </div>
      </div>

      {/* Track List */}
      <div className="p-4">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="flex items-center gap-3 animate-pulse">
                <div className="w-12 h-12 rounded-lg bg-white/10" />
                <div className="flex-1">
                  <div className="h-4 w-32 bg-white/10 rounded mb-2" />
                  <div className="h-3 w-24 bg-white/10 rounded" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {tracks?.slice(0, 8).map((track, index) => (
              <TrackRow
                key={track.id}
                track={track}
                index={index}
                isPlaying={currentSong?.id === track.id && isPlaying}
                onPlay={() => {
                  setSong({
                    id: track.id,
                    title: track.title,
                    artist: track.artist,
                    artistId: track.artistId,
                    coverArt: track.coverArt,
                    duration: track.duration,
                    audioUrl: track.audioUrl,
                  });
                  const remaining = tracks.slice(index + 1).map(t => ({
                    id: t.id,
                    title: t.title,
                    artist: t.artist,
                    artistId: t.artistId,
                    coverArt: t.coverArt,
                    duration: t.duration,
                    audioUrl: t.audioUrl,
                  }));
                  setQueue(remaining);
                }}
              />
            ))}

            {tracks && tracks.length > 8 && (
              <button className="w-full py-3 text-sm text-purple-400 hover:text-purple-300 flex items-center justify-center gap-1">
                See all {tracks.length} tracks
                <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface TrackRowProps {
  track: ContextualTrack;
  index: number;
  isPlaying: boolean;
  onPlay: () => void;
}

function TrackRow({ track, index, isPlaying, onPlay }: TrackRowProps) {
  return (
    <button
      onClick={onPlay}
      className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 group transition-colors"
    >
      <div className="relative w-12 h-12 rounded-lg overflow-hidden flex-shrink-0">
        <Image
          src={track.coverArt}
          alt={track.title}
          fill
          className="object-cover"
        />
        <div className={`absolute inset-0 bg-black/50 flex items-center justify-center transition-opacity ${
          isPlaying ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
        }`}>
          {isPlaying ? (
            <div className="flex items-center gap-0.5">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="w-1 bg-white rounded-full animate-pulse"
                  style={{
                    height: `${8 + Math.random() * 8}px`,
                    animationDelay: `${i * 0.15}s`,
                  }}
                />
              ))}
            </div>
          ) : (
            <Play className="w-5 h-5" />
          )}
        </div>
      </div>

      <div className="flex-1 min-w-0 text-left">
        <p className={`font-medium truncate ${isPlaying ? 'text-purple-400' : ''}`}>
          {track.title}
        </p>
        <p className="text-sm text-muted-foreground truncate">{track.artist}</p>
      </div>

      {/* Match score indicator */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
            style={{ width: `${track.matchScore}%` }}
          />
        </div>
        <span>{track.matchScore}%</span>
      </div>

      <span className="text-sm text-muted-foreground">
        {formatDuration(track.duration)}
      </span>
    </button>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Mock data generator
function generateMockContextualTracks(attributes: any): ContextualTrack[] {
  const mockTracks: ContextualTrack[] = [
    { id: 'ctx-1', title: 'Ambient Dreams', artist: 'Ethereal Waves', artistId: '1', coverArt: 'https://picsum.photos/seed/ctx1/200', duration: 245, audioUrl: '', matchScore: 95 },
    { id: 'ctx-2', title: 'Golden Hour', artist: 'Sunset Collective', artistId: '2', coverArt: 'https://picsum.photos/seed/ctx2/200', duration: 198, audioUrl: '', matchScore: 92 },
    { id: 'ctx-3', title: 'Mindful Flow', artist: 'Zen Masters', artistId: '3', coverArt: 'https://picsum.photos/seed/ctx3/200', duration: 312, audioUrl: '', matchScore: 88 },
    { id: 'ctx-4', title: 'Celestial Drift', artist: 'Space Echoes', artistId: '4', coverArt: 'https://picsum.photos/seed/ctx4/200', duration: 267, audioUrl: '', matchScore: 85 },
    { id: 'ctx-5', title: 'Morning Mist', artist: 'Nature Sounds', artistId: '5', coverArt: 'https://picsum.photos/seed/ctx5/200', duration: 189, audioUrl: '', matchScore: 82 },
    { id: 'ctx-6', title: 'Deep Focus', artist: 'Study Beats', artistId: '6', coverArt: 'https://picsum.photos/seed/ctx6/200', duration: 224, audioUrl: '', matchScore: 80 },
    { id: 'ctx-7', title: 'Peaceful Journey', artist: 'Calm Collective', artistId: '7', coverArt: 'https://picsum.photos/seed/ctx7/200', duration: 356, audioUrl: '', matchScore: 78 },
    { id: 'ctx-8', title: 'Tranquil Waters', artist: 'Ocean Vibes', artistId: '8', coverArt: 'https://picsum.photos/seed/ctx8/200', duration: 298, audioUrl: '', matchScore: 75 },
  ];

  // Adjust match scores based on attributes
  return mockTracks.map(track => ({
    ...track,
    matchScore: Math.round(70 + Math.random() * 30),
  })).sort((a, b) => b.matchScore - a.matchScore);
}
