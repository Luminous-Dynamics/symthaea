// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { SongCard } from '@/components/ui/SongCard';
import { formatNumber, cn } from '@/lib/utils';
import {
  Plus,
  ListMusic,
  Heart,
  Clock,
  Grid,
  List,
  Music2,
  User,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { redirect } from 'next/navigation';

type Tab = 'playlists' | 'artists' | 'albums' | 'songs';
type ViewMode = 'grid' | 'list';

export default function LibraryPage() {
  const { authenticated, user } = useAuth();
  const queryClient = useQueryClient();

  const [activeTab, setActiveTab] = useState<Tab>('playlists');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPlaylistName, setNewPlaylistName] = useState('');

  if (!authenticated) {
    redirect('/');
  }

  const { data: playlists, isLoading: playlistsLoading } = useQuery({
    queryKey: ['userPlaylists'],
    queryFn: () => api.getUserPlaylists(),
    enabled: authenticated,
  });

  const { data: likedSongs, isLoading: likedLoading } = useQuery({
    queryKey: ['likedSongs'],
    queryFn: () => api.getLikedSongs(),
    enabled: authenticated,
  });

  const { data: followedArtists, isLoading: artistsLoading } = useQuery({
    queryKey: ['followedArtists'],
    queryFn: () => api.getFollowedArtists(),
    enabled: authenticated,
  });

  const createPlaylist = useMutation({
    mutationFn: (name: string) => api.createPlaylist({ name, description: '' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userPlaylists'] });
      setShowCreateModal(false);
      setNewPlaylistName('');
    },
  });

  const tabs: { value: Tab; label: string; count?: number }[] = [
    { value: 'playlists', label: 'Playlists', count: playlists?.length },
    { value: 'artists', label: 'Artists', count: followedArtists?.length },
    { value: 'songs', label: 'Liked Songs', count: likedSongs?.length },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold">Your Library</h1>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode('grid')}
                className={cn(
                  'p-2 rounded-md transition-colors',
                  viewMode === 'grid' ? 'bg-white/10' : 'hover:bg-white/5'
                )}
              >
                <Grid className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={cn(
                  'p-2 rounded-md transition-colors',
                  viewMode === 'list' ? 'bg-white/10' : 'hover:bg-white/5'
                )}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-2 mb-6">
            {tabs.map((tab) => (
              <button
                key={tab.value}
                onClick={() => setActiveTab(tab.value)}
                className={cn(
                  'px-4 py-2 rounded-full text-sm font-medium transition-colors',
                  activeTab === tab.value
                    ? 'bg-white text-black'
                    : 'bg-white/10 hover:bg-white/20'
                )}
              >
                {tab.label}
                {tab.count !== undefined && (
                  <span className="ml-2 opacity-70">{tab.count}</span>
                )}
              </button>
            ))}
          </div>

          {/* Content */}
          {activeTab === 'playlists' && (
            <div className={cn(
              viewMode === 'grid'
                ? 'grid grid-cols-5 gap-6'
                : 'space-y-2'
            )}>
              {/* Create Playlist Card */}
              <button
                onClick={() => setShowCreateModal(true)}
                className={cn(
                  'flex items-center gap-4 transition-colors',
                  viewMode === 'grid'
                    ? 'flex-col p-4 rounded-lg bg-white/5 hover:bg-white/10'
                    : 'p-4 rounded-lg bg-white/5 hover:bg-white/10'
                )}
              >
                <div className={cn(
                  'flex items-center justify-center bg-gradient-to-br from-purple-500/20 to-fuchsia-600/20',
                  viewMode === 'grid'
                    ? 'w-full aspect-square rounded-md'
                    : 'w-14 h-14 rounded-md'
                )}>
                  <Plus className="w-12 h-12 text-muted-foreground" />
                </div>
                <span className={cn(
                  'font-medium',
                  viewMode === 'grid' && 'text-center'
                )}>
                  Create Playlist
                </span>
              </button>

              {/* Liked Songs Card */}
              <Link
                href="/library/liked"
                className={cn(
                  'flex items-center gap-4 transition-colors',
                  viewMode === 'grid'
                    ? 'flex-col p-4 rounded-lg bg-white/5 hover:bg-white/10'
                    : 'p-4 rounded-lg bg-white/5 hover:bg-white/10'
                )}
              >
                <div className={cn(
                  'flex items-center justify-center bg-gradient-to-br from-purple-600 to-blue-500',
                  viewMode === 'grid'
                    ? 'w-full aspect-square rounded-md'
                    : 'w-14 h-14 rounded-md'
                )}>
                  <Heart className="w-8 h-8 fill-white" />
                </div>
                <div className={viewMode === 'grid' ? 'text-center' : ''}>
                  <h3 className="font-medium">Liked Songs</h3>
                  <p className="text-sm text-muted-foreground">
                    {likedSongs?.length || 0} songs
                  </p>
                </div>
              </Link>

              {/* User Playlists */}
              {playlistsLoading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'animate-pulse',
                      viewMode === 'grid'
                        ? 'p-4 rounded-lg bg-white/5'
                        : 'p-4 rounded-lg bg-white/5'
                    )}
                  >
                    <div className={cn(
                      'bg-white/10 rounded-md',
                      viewMode === 'grid'
                        ? 'aspect-square mb-4'
                        : 'w-14 h-14'
                    )} />
                    <div className="h-4 bg-white/10 rounded w-3/4 mb-2" />
                    <div className="h-3 bg-white/10 rounded w-1/2" />
                  </div>
                ))
              ) : (
                playlists?.map((playlist) => (
                  <Link
                    key={playlist.id}
                    href={`/playlist/${playlist.id}`}
                    className={cn(
                      'group flex items-center gap-4 transition-colors',
                      viewMode === 'grid'
                        ? 'flex-col p-4 rounded-lg bg-white/5 hover:bg-white/10'
                        : 'p-4 rounded-lg bg-white/5 hover:bg-white/10'
                    )}
                  >
                    <div className={cn(
                      'relative overflow-hidden',
                      viewMode === 'grid'
                        ? 'w-full aspect-square rounded-md'
                        : 'w-14 h-14 rounded-md flex-shrink-0'
                    )}>
                      {playlist.coverImage ? (
                        <Image
                          src={playlist.coverImage}
                          alt={playlist.name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-purple-500/30 to-fuchsia-600/30 flex items-center justify-center">
                          <ListMusic className="w-8 h-8 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                    <div className={cn(
                      'min-w-0',
                      viewMode === 'grid' ? 'text-center' : 'flex-1'
                    )}>
                      <h3 className="font-medium truncate">{playlist.name}</h3>
                      <p className="text-sm text-muted-foreground">
                        Playlist • {playlist.songCount} songs
                      </p>
                    </div>
                  </Link>
                ))
              )}
            </div>
          )}

          {activeTab === 'artists' && (
            <div className={cn(
              viewMode === 'grid'
                ? 'grid grid-cols-6 gap-6'
                : 'space-y-2'
            )}>
              {artistsLoading ? (
                Array.from({ length: 6 }).map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'animate-pulse',
                      viewMode === 'grid'
                        ? 'p-4 text-center'
                        : 'flex items-center gap-4 p-4 rounded-lg bg-white/5'
                    )}
                  >
                    <div className={cn(
                      'bg-white/10 rounded-full',
                      viewMode === 'grid'
                        ? 'w-32 h-32 mx-auto mb-4'
                        : 'w-14 h-14'
                    )} />
                    <div className="h-4 bg-white/10 rounded w-3/4 mx-auto" />
                  </div>
                ))
              ) : followedArtists?.length ? (
                followedArtists.map((artist) => (
                  <Link
                    key={artist.address}
                    href={`/artist/${artist.address}`}
                    className={cn(
                      'group transition-colors',
                      viewMode === 'grid'
                        ? 'p-4 rounded-lg bg-white/5 hover:bg-white/10 text-center'
                        : 'flex items-center gap-4 p-4 rounded-lg bg-white/5 hover:bg-white/10'
                    )}
                  >
                    <div className={cn(
                      'relative rounded-full overflow-hidden',
                      viewMode === 'grid'
                        ? 'w-32 h-32 mx-auto mb-4'
                        : 'w-14 h-14 flex-shrink-0'
                    )}>
                      {artist.avatar ? (
                        <Image
                          src={artist.avatar}
                          alt={artist.name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center">
                          <User className="w-8 h-8" />
                        </div>
                      )}
                    </div>
                    <div className={viewMode === 'grid' ? '' : 'flex-1 min-w-0'}>
                      <h3 className="font-medium truncate">{artist.name}</h3>
                      <p className="text-sm text-muted-foreground">Artist</p>
                    </div>
                  </Link>
                ))
              ) : (
                <div className="col-span-full text-center py-20">
                  <User className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-xl font-bold mb-2">No followed artists</h3>
                  <p className="text-muted-foreground">
                    Follow artists to see them here
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'songs' && (
            <div>
              {likedLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-4 p-4 animate-pulse">
                      <div className="w-10 h-10 bg-white/10 rounded" />
                      <div className="flex-1">
                        <div className="h-4 bg-white/10 rounded w-48 mb-2" />
                        <div className="h-3 bg-white/10 rounded w-32" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : likedSongs?.length ? (
                <div className="bg-white/5 rounded-lg">
                  {likedSongs.map((song, index) => (
                    <SongCard key={song.id} song={song} variant="row" index={index} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-20">
                  <Heart className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-xl font-bold mb-2">No liked songs yet</h3>
                  <p className="text-muted-foreground">
                    Songs you like will appear here
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      <Player />

      {/* Create Playlist Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md p-6 bg-[#282828] rounded-xl shadow-2xl">
            <h2 className="text-xl font-bold mb-4">Create Playlist</h2>
            <input
              type="text"
              value={newPlaylistName}
              onChange={(e) => setNewPlaylistName(e.target.value)}
              placeholder="Playlist name"
              className="w-full px-4 py-3 bg-white/10 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary mb-4"
              autoFocus
            />
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewPlaylistName('');
                }}
                className="px-4 py-2 text-sm font-medium hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => newPlaylistName.trim() && createPlaylist.mutate(newPlaylistName.trim())}
                disabled={!newPlaylistName.trim() || createPlaylist.isPending}
                className="px-6 py-2 bg-primary text-black rounded-full text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {createPlaylist.isPending ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
