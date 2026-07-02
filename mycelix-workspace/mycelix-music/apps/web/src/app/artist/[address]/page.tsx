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
import { formatNumber, truncateAddress, cn } from '@/lib/utils';
import {
  Play,
  Pause,
  Shuffle,
  MoreHorizontal,
  UserPlus,
  UserCheck,
  Share2,
  ExternalLink,
  Verified,
  Music2,
  Users,
  Heart,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

type Tab = 'popular' | 'releases' | 'about';

export default function ArtistPage() {
  const params = useParams();
  const queryClient = useQueryClient();
  const artistAddress = params.address as string;

  const { walletAddress, authenticated } = useAuth();
  const { playAll, isPlaying, currentSong, shuffle, toggleShuffle } = usePlayerStore();

  const [activeTab, setActiveTab] = useState<Tab>('popular');
  const [showAllSongs, setShowAllSongs] = useState(false);

  const { data: artist, isLoading } = useQuery({
    queryKey: ['artist', artistAddress],
    queryFn: () => api.getArtist(artistAddress),
  });

  const { data: songs } = useQuery({
    queryKey: ['artistSongs', artistAddress],
    queryFn: () => api.getArtistPublicSongs(artistAddress),
  });

  const followMutation = useMutation({
    mutationFn: () => api.followArtist(artistAddress),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['artist', artistAddress] });
    },
  });

  const unfollowMutation = useMutation({
    mutationFn: () => api.unfollowArtist(artistAddress),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['artist', artistAddress] });
    },
  });

  const isOwnProfile = walletAddress === artistAddress;
  const isPlayingArtist = songs?.some((s) => s.id === currentSong?.id) && isPlaying;

  const handlePlayAll = () => {
    if (songs?.length) {
      playAll(songs, shuffle ? Math.floor(Math.random() * songs.length) : 0);
    }
  };

  const handleFollowToggle = () => {
    if (artist?.isFollowing) {
      unfollowMutation.mutate();
    } else {
      followMutation.mutate();
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!artist) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Users className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-xl font-bold mb-2">Artist not found</h2>
          <p className="text-muted-foreground">This artist doesn't exist</p>
        </div>
      </div>
    );
  }

  const displaySongs = showAllSongs ? songs : songs?.slice(0, 5);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        {/* Artist Header */}
        <div className="relative">
          {/* Banner Image */}
          <div className="relative h-72 overflow-hidden">
            {artist.bannerImage ? (
              <Image
                src={artist.bannerImage}
                alt=""
                fill
                className="object-cover"
              />
            ) : (
              <div
                className="w-full h-full"
                style={{
                  background: `linear-gradient(to bottom right, ${artist.dominantColor || '#8B5CF6'}, #1a1a1a)`,
                }}
              />
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-background via-background/60 to-transparent" />
          </div>

          {/* Artist Info */}
          <div className="absolute bottom-0 left-0 right-0 px-6 pb-6">
            <div className="flex items-end gap-6">
              {/* Avatar */}
              <div className="relative w-48 h-48 rounded-full shadow-2xl overflow-hidden border-4 border-background flex-shrink-0">
                {artist.avatar ? (
                  <Image
                    src={artist.avatar}
                    alt={artist.name}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center">
                    <Users className="w-20 h-20" />
                  </div>
                )}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0 mb-2">
                <div className="flex items-center gap-2 mb-2">
                  {artist.isVerified && (
                    <div className="flex items-center gap-1 px-2 py-0.5 bg-primary/20 rounded-full text-primary text-xs font-medium">
                      <Verified className="w-3 h-3" />
                      Verified Artist
                    </div>
                  )}
                </div>

                <h1 className="text-6xl font-black mb-3">{artist.name}</h1>

                <div className="flex items-center gap-4 text-sm">
                  <span>
                    <strong className="font-semibold">{formatNumber(artist.followerCount || 0)}</strong>{' '}
                    <span className="text-muted-foreground">followers</span>
                  </span>
                  <span className="text-muted-foreground">•</span>
                  <span>
                    <strong className="font-semibold">{formatNumber(artist.totalStreams || 0)}</strong>{' '}
                    <span className="text-muted-foreground">streams</span>
                  </span>
                  <span className="text-muted-foreground">•</span>
                  <span className="text-muted-foreground font-mono text-xs">
                    {truncateAddress(artistAddress)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-6 flex items-center gap-4">
          <button
            onClick={handlePlayAll}
            disabled={!songs?.length}
            className="w-14 h-14 rounded-full bg-primary flex items-center justify-center hover:scale-105 transition-transform disabled:opacity-50"
          >
            {isPlayingArtist ? (
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

          {authenticated && !isOwnProfile && (
            <button
              onClick={handleFollowToggle}
              disabled={followMutation.isPending || unfollowMutation.isPending}
              className={cn(
                'flex items-center gap-2 px-6 py-2 rounded-full text-sm font-medium transition-colors',
                artist.isFollowing
                  ? 'border border-white/20 hover:border-white/40'
                  : 'bg-white text-black hover:bg-white/90'
              )}
            >
              {artist.isFollowing ? (
                <>
                  <UserCheck className="w-4 h-4" />
                  Following
                </>
              ) : (
                <>
                  <UserPlus className="w-4 h-4" />
                  Follow
                </>
              )}
            </button>
          )}

          {isOwnProfile && (
            <Link
              href="/dashboard"
              className="flex items-center gap-2 px-6 py-2 bg-white/10 rounded-full text-sm font-medium hover:bg-white/20 transition-colors"
            >
              Go to Dashboard
            </Link>
          )}

          <button className="text-muted-foreground hover:text-white transition-colors">
            <Share2 className="w-6 h-6" />
          </button>

          <button className="ml-auto text-muted-foreground hover:text-white transition-colors">
            <MoreHorizontal className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="px-6 mb-6">
          <div className="flex gap-6 border-b border-white/10">
            {(['popular', 'releases', 'about'] as Tab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={cn(
                  'pb-3 text-sm font-medium transition-colors relative',
                  activeTab === tab
                    ? 'text-white'
                    : 'text-muted-foreground hover:text-white'
                )}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                {activeTab === tab && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="px-6">
          {activeTab === 'popular' && (
            <div>
              <h2 className="text-xl font-bold mb-4">Popular</h2>
              {songs?.length ? (
                <>
                  <div className="space-y-1 mb-4">
                    {displaySongs?.map((song, index) => (
                      <SongCard
                        key={song.id}
                        song={song}
                        variant="row"
                        index={index}
                        showArtist={false}
                      />
                    ))}
                  </div>
                  {songs.length > 5 && (
                    <button
                      onClick={() => setShowAllSongs(!showAllSongs)}
                      className="text-sm text-muted-foreground hover:text-white transition-colors"
                    >
                      {showAllSongs ? 'Show less' : `See all ${songs.length} songs`}
                    </button>
                  )}
                </>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Music2 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No songs yet</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'releases' && (
            <div>
              <h2 className="text-xl font-bold mb-4">Releases</h2>
              {songs?.length ? (
                <div className="grid grid-cols-5 gap-6">
                  {songs.map((song) => (
                    <SongCard key={song.id} song={song} showArtist={false} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Music2 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No releases yet</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'about' && (
            <div className="max-w-2xl">
              <h2 className="text-xl font-bold mb-4">About</h2>

              {artist.bio ? (
                <p className="text-muted-foreground mb-6 whitespace-pre-wrap">
                  {artist.bio}
                </p>
              ) : (
                <p className="text-muted-foreground mb-6 italic">
                  No bio available
                </p>
              )}

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-2xl font-bold">
                    {formatNumber(artist.followerCount || 0)}
                  </p>
                  <p className="text-sm text-muted-foreground">Followers</p>
                </div>
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-2xl font-bold">
                    {formatNumber(artist.totalStreams || 0)}
                  </p>
                  <p className="text-sm text-muted-foreground">Total Streams</p>
                </div>
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-2xl font-bold">{songs?.length || 0}</p>
                  <p className="text-sm text-muted-foreground">Releases</p>
                </div>
              </div>

              {/* Social Links */}
              {artist.socialLinks && Object.keys(artist.socialLinks).length > 0 && (
                <div>
                  <h3 className="text-sm font-medium mb-3 text-muted-foreground">
                    Social Links
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(artist.socialLinks).map(([platform, url]) => (
                      <a
                        key={platform}
                        href={url as string}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-full text-sm hover:bg-white/20 transition-colors"
                      >
                        {platform}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Wallet Address */}
              <div className="mt-6 pt-6 border-t border-white/10">
                <p className="text-sm text-muted-foreground mb-2">Wallet Address</p>
                <code className="text-xs font-mono bg-white/5 px-3 py-2 rounded block">
                  {artistAddress}
                </code>
              </div>
            </div>
          )}
        </div>
      </main>

      <Player />
    </div>
  );
}
