// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { usePlayerStore } from '@/store/playerStore';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { SongCard } from '@/components/ui/SongCard';
import { formatDuration, formatNumber, formatDate, cn } from '@/lib/utils';
import {
  Play,
  Pause,
  Heart,
  Share2,
  MoreHorizontal,
  Plus,
  Music2,
  Clock,
  Calendar,
  Disc3,
  ExternalLink,
  ListMusic,
  X,
  Check,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

export default function SongPage() {
  const params = useParams();
  const queryClient = useQueryClient();
  const songId = params.id as string;

  const { authenticated } = useAuth();
  const { play, pause, resume, isPlaying, currentSong, addToQueue } = usePlayerStore();

  const [showAddToPlaylist, setShowAddToPlaylist] = useState(false);
  const [addedToPlaylist, setAddedToPlaylist] = useState<string | null>(null);

  const { data: song, isLoading } = useQuery({
    queryKey: ['song', songId],
    queryFn: () => api.getSong(songId),
  });

  const { data: moreBySameArtist } = useQuery({
    queryKey: ['artistSongs', song?.artistAddress],
    queryFn: () => api.getArtistPublicSongs(song?.artistAddress || ''),
    enabled: !!song?.artistAddress,
  });

  const { data: userPlaylists } = useQuery({
    queryKey: ['userPlaylists'],
    queryFn: () => api.getUserPlaylists(),
    enabled: authenticated && showAddToPlaylist,
  });

  const likeMutation = useMutation({
    mutationFn: () => api.likeSong(songId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['song', songId] });
    },
  });

  const addToPlaylistMutation = useMutation({
    mutationFn: (playlistId: string) => api.addToPlaylist(playlistId, songId),
    onSuccess: (_, playlistId) => {
      setAddedToPlaylist(playlistId);
      setTimeout(() => {
        setShowAddToPlaylist(false);
        setAddedToPlaylist(null);
      }, 1500);
    },
  });

  const isCurrentSong = currentSong?.id === song?.id;
  const isCurrentlyPlaying = isCurrentSong && isPlaying;

  const handlePlay = () => {
    if (!song) return;
    if (isCurrentSong) {
      if (isPlaying) pause();
      else resume();
    } else {
      play(song);
    }
  };

  const handleAddToQueue = () => {
    if (song) {
      addToQueue(song);
    }
  };

  const relatedSongs = moreBySameArtist?.filter((s) => s.id !== songId).slice(0, 5);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!song) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Music2 className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-xl font-bold mb-2">Song not found</h2>
          <p className="text-muted-foreground">This song doesn't exist or was removed</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        {/* Song Header */}
        <div className="relative">
          {/* Background Gradient */}
          <div
            className="absolute inset-0 h-96"
            style={{
              background: `linear-gradient(to bottom, ${song.dominantColor || '#8B5CF6'}40, transparent)`,
            }}
          />

          <div className="relative px-6 pt-8 pb-6">
            <div className="flex items-end gap-8">
              {/* Cover Art */}
              <div className="relative w-64 h-64 rounded-lg shadow-2xl overflow-hidden flex-shrink-0">
                {song.coverArt ? (
                  <Image
                    src={song.coverArt}
                    alt={song.title}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-purple-600 to-fuchsia-700 flex items-center justify-center">
                    <Music2 className="w-24 h-24 text-white/60" />
                  </div>
                )}
              </div>

              {/* Song Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium mb-2">Song</p>
                <h1 className="text-5xl font-black mb-4">{song.title}</h1>

                <div className="flex items-center gap-3 text-sm">
                  {/* Artist */}
                  <Link
                    href={`/artist/${song.artistAddress}`}
                    className="flex items-center gap-2 hover:underline"
                  >
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-fuchsia-600 overflow-hidden">
                      {song.artistAvatar && (
                        <Image
                          src={song.artistAvatar}
                          alt={song.artist}
                          width={24}
                          height={24}
                          className="object-cover"
                        />
                      )}
                    </div>
                    <span className="font-medium">{song.artist}</span>
                  </Link>

                  {song.album && (
                    <>
                      <span className="text-muted-foreground">•</span>
                      <span className="text-muted-foreground">{song.album}</span>
                    </>
                  )}

                  <span className="text-muted-foreground">•</span>
                  <span className="text-muted-foreground">
                    {formatDuration(song.duration)}
                  </span>

                  <span className="text-muted-foreground">•</span>
                  <span className="text-muted-foreground">
                    {formatNumber(song.playCount || 0)} plays
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-6 flex items-center gap-4">
          <button
            onClick={handlePlay}
            className="w-14 h-14 rounded-full bg-primary flex items-center justify-center hover:scale-105 transition-transform"
          >
            {isCurrentlyPlaying ? (
              <Pause className="w-6 h-6 text-black" />
            ) : (
              <Play className="w-6 h-6 text-black ml-1" />
            )}
          </button>

          <button
            onClick={() => likeMutation.mutate()}
            disabled={likeMutation.isPending}
            className={cn(
              'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
              song.isLiked
                ? 'text-primary'
                : 'text-muted-foreground hover:text-white'
            )}
          >
            <Heart className={cn('w-6 h-6', song.isLiked && 'fill-current')} />
          </button>

          {authenticated && (
            <div className="relative">
              <button
                onClick={() => setShowAddToPlaylist(!showAddToPlaylist)}
                className="w-10 h-10 rounded-full flex items-center justify-center text-muted-foreground hover:text-white transition-colors"
              >
                <Plus className="w-6 h-6" />
              </button>

              {/* Add to Playlist Dropdown */}
              {showAddToPlaylist && (
                <div className="absolute top-full left-0 mt-2 w-64 py-2 bg-[#282828] rounded-lg shadow-xl border border-white/10 z-50">
                  <div className="px-3 py-2 border-b border-white/10">
                    <p className="text-sm font-medium">Add to playlist</p>
                  </div>

                  <div className="max-h-64 overflow-y-auto py-1">
                    {userPlaylists?.length ? (
                      userPlaylists.map((playlist) => (
                        <button
                          key={playlist.id}
                          onClick={() => addToPlaylistMutation.mutate(playlist.id)}
                          disabled={addToPlaylistMutation.isPending}
                          className="w-full px-3 py-2 text-left text-sm hover:bg-white/10 transition-colors flex items-center gap-3"
                        >
                          <div className="w-10 h-10 rounded bg-white/10 flex items-center justify-center flex-shrink-0 overflow-hidden">
                            {playlist.coverImage ? (
                              <Image
                                src={playlist.coverImage}
                                alt=""
                                width={40}
                                height={40}
                                className="object-cover"
                              />
                            ) : (
                              <ListMusic className="w-5 h-5 text-muted-foreground" />
                            )}
                          </div>
                          <span className="truncate flex-1">{playlist.name}</span>
                          {addedToPlaylist === playlist.id && (
                            <Check className="w-4 h-4 text-green-500" />
                          )}
                        </button>
                      ))
                    ) : (
                      <p className="px-3 py-4 text-sm text-muted-foreground text-center">
                        No playlists yet
                      </p>
                    )}
                  </div>

                  <div className="border-t border-white/10 pt-1">
                    <Link
                      href="/library"
                      className="w-full px-3 py-2 text-left text-sm hover:bg-white/10 transition-colors flex items-center gap-3"
                    >
                      <div className="w-10 h-10 rounded bg-white/10 flex items-center justify-center">
                        <Plus className="w-5 h-5" />
                      </div>
                      Create new playlist
                    </Link>
                  </div>
                </div>
              )}
            </div>
          )}

          <button
            onClick={handleAddToQueue}
            className="text-muted-foreground hover:text-white transition-colors text-sm"
          >
            Add to queue
          </button>

          <button className="text-muted-foreground hover:text-white transition-colors">
            <Share2 className="w-5 h-5" />
          </button>

          <button className="ml-auto text-muted-foreground hover:text-white transition-colors">
            <MoreHorizontal className="w-6 h-6" />
          </button>
        </div>

        {/* Song Details */}
        <div className="px-6 mb-8">
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Duration</span>
              </div>
              <p className="font-medium">{formatDuration(song.duration)}</p>
            </div>

            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Calendar className="w-4 h-4" />
                <span className="text-sm">Released</span>
              </div>
              <p className="font-medium">
                {song.releaseDate ? formatDate(song.releaseDate) : 'Unknown'}
              </p>
            </div>

            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Disc3 className="w-4 h-4" />
                <span className="text-sm">Genre</span>
              </div>
              <p className="font-medium">{song.genre || 'Unknown'}</p>
            </div>

            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground mb-2">
                <Heart className="w-4 h-4" />
                <span className="text-sm">Likes</span>
              </div>
              <p className="font-medium">{formatNumber(song.likeCount || 0)}</p>
            </div>
          </div>
        </div>

        {/* Lyrics (if available) */}
        {song.lyrics && (
          <div className="px-6 mb-8">
            <h2 className="text-xl font-bold mb-4">Lyrics</h2>
            <div className="p-6 bg-white/5 rounded-lg max-w-2xl">
              <pre className="whitespace-pre-wrap font-sans text-muted-foreground text-sm leading-relaxed">
                {song.lyrics}
              </pre>
            </div>
          </div>
        )}

        {/* More by Artist */}
        {relatedSongs && relatedSongs.length > 0 && (
          <div className="px-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">More by {song.artist}</h2>
              <Link
                href={`/artist/${song.artistAddress}`}
                className="text-sm text-muted-foreground hover:text-white transition-colors"
              >
                View all
              </Link>
            </div>
            <div className="grid grid-cols-5 gap-6">
              {relatedSongs.map((relatedSong) => (
                <SongCard key={relatedSong.id} song={relatedSong} showArtist={false} />
              ))}
            </div>
          </div>
        )}

        {/* NFT / Blockchain Info */}
        {song.nftTokenId && (
          <div className="px-6 mb-8">
            <h2 className="text-xl font-bold mb-4">Blockchain</h2>
            <div className="p-4 bg-white/5 rounded-lg inline-flex items-center gap-4">
              <div>
                <p className="text-sm text-muted-foreground">NFT Token ID</p>
                <code className="text-xs font-mono">{song.nftTokenId}</code>
              </div>
              <a
                href={`https://etherscan.io/token/${song.contractAddress}?a=${song.nftTokenId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-primary text-sm hover:underline"
              >
                View on Etherscan
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        )}
      </main>

      <Player />
    </div>
  );
}
