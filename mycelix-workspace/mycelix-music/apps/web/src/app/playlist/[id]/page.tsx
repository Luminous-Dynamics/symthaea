// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { usePlayerStore } from '@/store/playerStore';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { SongCard } from '@/components/ui/SongCard';
import { formatDuration, formatNumber, cn } from '@/lib/utils';
import {
  Play,
  Pause,
  Shuffle,
  MoreHorizontal,
  Clock,
  Heart,
  Share2,
  Pencil,
  Trash2,
  Plus,
  ListMusic,
  X,
  GripVertical,
} from 'lucide-react';
import Image from 'next/image';

export default function PlaylistPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const playlistId = params.id as string;

  const { user, walletAddress } = useAuth();
  const { play, playAll, isPlaying, currentSong, shuffle, toggleShuffle } = usePlayerStore();

  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const { data: playlist, isLoading } = useQuery({
    queryKey: ['playlist', playlistId],
    queryFn: () => api.getPlaylist(playlistId),
  });

  const updatePlaylist = useMutation({
    mutationFn: (data: { name?: string; description?: string }) =>
      api.updatePlaylist(playlistId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['playlist', playlistId] });
      setIsEditing(false);
    },
  });

  const deletePlaylist = useMutation({
    mutationFn: () => api.deletePlaylist(playlistId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['playlists'] });
      router.push('/library');
    },
  });

  const removeSong = useMutation({
    mutationFn: (songId: string) => api.removeFromPlaylist(playlistId, songId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['playlist', playlistId] });
    },
  });

  const isOwner = playlist?.ownerAddress === walletAddress;
  const isPlayingPlaylist = playlist?.songs.some((s) => s.id === currentSong?.id) && isPlaying;

  const totalDuration = playlist?.songs.reduce((acc, song) => acc + song.duration, 0) || 0;

  const handlePlayAll = () => {
    if (playlist?.songs.length) {
      playAll(playlist.songs, shuffle ? Math.floor(Math.random() * playlist.songs.length) : 0);
    }
  };

  const handleEdit = () => {
    setEditName(playlist?.name || '');
    setEditDescription(playlist?.description || '');
    setIsEditing(true);
  };

  const handleSaveEdit = () => {
    updatePlaylist.mutate({
      name: editName,
      description: editDescription,
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!playlist) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <ListMusic className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-xl font-bold mb-2">Playlist not found</h2>
          <p className="text-muted-foreground">This playlist doesn't exist or was deleted</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        {/* Playlist Header */}
        <div className="relative">
          {/* Background Gradient */}
          <div
            className="absolute inset-0 h-80"
            style={{
              background: `linear-gradient(to bottom, ${playlist.dominantColor || '#8B5CF6'}40, transparent)`,
            }}
          />

          <div className="relative px-6 pt-8 pb-6">
            <div className="flex items-end gap-6">
              {/* Cover Image */}
              <div className="relative w-56 h-56 rounded-lg shadow-2xl overflow-hidden flex-shrink-0">
                {playlist.coverImage ? (
                  <Image
                    src={playlist.coverImage}
                    alt={playlist.name}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-purple-600 to-fuchsia-700 flex items-center justify-center">
                    <ListMusic className="w-20 h-20 text-white/60" />
                  </div>
                )}
              </div>

              {/* Playlist Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium mb-2">Playlist</p>

                {isEditing ? (
                  <div className="space-y-3 mb-4">
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full text-5xl font-black bg-white/10 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="Playlist name"
                    />
                    <textarea
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      className="w-full text-sm bg-white/10 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                      placeholder="Add a description..."
                      rows={2}
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={handleSaveEdit}
                        disabled={updatePlaylist.isPending}
                        className="px-4 py-1.5 bg-primary text-black rounded-full text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
                      >
                        {updatePlaylist.isPending ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={() => setIsEditing(false)}
                        className="px-4 py-1.5 bg-white/10 rounded-full text-sm font-medium hover:bg-white/20"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <h1 className="text-5xl font-black mb-4 truncate">{playlist.name}</h1>
                    {playlist.description && (
                      <p className="text-sm text-muted-foreground mb-4">{playlist.description}</p>
                    )}
                  </>
                )}

                <div className="flex items-center gap-2 text-sm">
                  <span className="font-medium">{playlist.ownerName}</span>
                  <span className="text-muted-foreground">•</span>
                  <span className="text-muted-foreground">
                    {formatNumber(playlist.followers || 0)} saves
                  </span>
                  <span className="text-muted-foreground">•</span>
                  <span className="text-muted-foreground">
                    {playlist.songs.length} songs, {formatDuration(totalDuration)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-4 flex items-center gap-4">
          <button
            onClick={handlePlayAll}
            disabled={!playlist.songs.length}
            className="w-14 h-14 rounded-full bg-primary flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPlayingPlaylist ? (
              <Pause className="w-6 h-6 text-black" />
            ) : (
              <Play className="w-6 h-6 text-black ml-1" />
            )}
          </button>

          <button
            onClick={toggleShuffle}
            className={cn(
              'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
              shuffle ? 'text-primary' : 'text-muted-foreground hover:text-white'
            )}
          >
            <Shuffle className="w-5 h-5" />
          </button>

          <button className="text-muted-foreground hover:text-white transition-colors">
            <Heart className="w-6 h-6" />
          </button>

          <button className="text-muted-foreground hover:text-white transition-colors">
            <Share2 className="w-6 h-6" />
          </button>

          {isOwner && (
            <>
              <button
                onClick={handleEdit}
                className="text-muted-foreground hover:text-white transition-colors"
              >
                <Pencil className="w-5 h-5" />
              </button>

              <div className="relative">
                <button
                  onClick={() => setShowDeleteConfirm(!showDeleteConfirm)}
                  className="text-muted-foreground hover:text-red-500 transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>

                {showDeleteConfirm && (
                  <div className="absolute top-full left-0 mt-2 p-4 bg-[#282828] rounded-lg shadow-xl border border-white/10 w-64 z-50">
                    <p className="text-sm mb-3">Delete this playlist?</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => deletePlaylist.mutate()}
                        disabled={deletePlaylist.isPending}
                        className="flex-1 px-3 py-1.5 bg-red-500 text-white rounded text-sm font-medium hover:bg-red-600 disabled:opacity-50"
                      >
                        {deletePlaylist.isPending ? 'Deleting...' : 'Delete'}
                      </button>
                      <button
                        onClick={() => setShowDeleteConfirm(false)}
                        className="flex-1 px-3 py-1.5 bg-white/10 rounded text-sm font-medium hover:bg-white/20"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          <button className="ml-auto text-muted-foreground hover:text-white transition-colors">
            <MoreHorizontal className="w-6 h-6" />
          </button>
        </div>

        {/* Song List */}
        <div className="px-6">
          {playlist.songs.length > 0 ? (
            <>
              {/* Header Row */}
              <div className="grid grid-cols-[16px_4fr_3fr_1fr] gap-4 px-4 py-2 text-sm text-muted-foreground border-b border-white/10 mb-2">
                <span>#</span>
                <span>Title</span>
                <span>Album</span>
                <span className="text-right">
                  <Clock className="w-4 h-4 inline" />
                </span>
              </div>

              {/* Songs */}
              <div className="space-y-1">
                {playlist.songs.map((song, index) => (
                  <PlaylistSongRow
                    key={`${song.id}-${index}`}
                    song={song}
                    index={index}
                    isOwner={isOwner}
                    onRemove={() => removeSong.mutate(song.id)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-20">
              <ListMusic className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-xl font-bold mb-2">This playlist is empty</h3>
              <p className="text-muted-foreground mb-6">
                Start adding songs to build your perfect playlist
              </p>
              <button
                onClick={() => router.push('/search')}
                className="inline-flex items-center gap-2 px-6 py-3 bg-white text-black rounded-full font-medium hover:scale-105 transition-transform"
              >
                <Plus className="w-5 h-5" />
                Find songs
              </button>
            </div>
          )}
        </div>
      </main>

      <Player />
    </div>
  );
}

// Playlist Song Row Component
function PlaylistSongRow({
  song,
  index,
  isOwner,
  onRemove,
}: {
  song: any;
  index: number;
  isOwner: boolean;
  onRemove: () => void;
}) {
  const { currentSong, isPlaying, play, pause, resume } = usePlayerStore();
  const [isHovered, setIsHovered] = useState(false);

  const isCurrentSong = currentSong?.id === song.id;
  const isCurrentlyPlaying = isCurrentSong && isPlaying;

  const handlePlay = () => {
    if (isCurrentSong) {
      if (isPlaying) pause();
      else resume();
    } else {
      play(song);
    }
  };

  return (
    <div
      className={cn(
        'grid grid-cols-[16px_4fr_3fr_1fr] gap-4 px-4 py-2 rounded-md group',
        'hover:bg-white/10 transition-colors',
        isCurrentSong && 'bg-white/5'
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Index / Play */}
      <div className="flex items-center justify-center">
        {isHovered || isCurrentlyPlaying ? (
          <button onClick={handlePlay} className="w-4 h-4 flex items-center justify-center">
            {isCurrentlyPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4" />
            )}
          </button>
        ) : (
          <span
            className={cn(
              'text-sm',
              isCurrentSong ? 'text-primary' : 'text-muted-foreground'
            )}
          >
            {index + 1}
          </span>
        )}
      </div>

      {/* Title & Artist */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="relative w-10 h-10 rounded overflow-hidden flex-shrink-0">
          <Image
            src={song.coverArt || '/placeholder-album.png'}
            alt={song.title}
            fill
            className="object-cover"
          />
        </div>
        <div className="min-w-0">
          <p
            className={cn(
              'font-medium truncate',
              isCurrentSong && 'text-primary'
            )}
          >
            {song.title}
          </p>
          <p className="text-sm text-muted-foreground truncate">{song.artist}</p>
        </div>
      </div>

      {/* Album */}
      <div className="flex items-center min-w-0">
        <span className="text-sm text-muted-foreground truncate">
          {song.album || 'Single'}
        </span>
      </div>

      {/* Duration & Actions */}
      <div className="flex items-center justify-end gap-2">
        {isOwner && (
          <button
            onClick={onRemove}
            className="p-1 text-muted-foreground hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
            title="Remove from playlist"
          >
            <X className="w-4 h-4" />
          </button>
        )}
        <span className="text-sm text-muted-foreground">
          {formatDuration(song.duration)}
        </span>
      </div>
    </div>
  );
}
