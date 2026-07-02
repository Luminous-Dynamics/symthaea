// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { SongCard } from '@/components/ui/SongCard';
import { ChevronRight } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  const { data: trending, isLoading: trendingLoading } = useQuery({
    queryKey: ['trending'],
    queryFn: () => api.getTrendingSongs(10),
  });

  const { data: newReleases, isLoading: newReleasesLoading } = useQuery({
    queryKey: ['newReleases'],
    queryFn: () => api.getNewReleases(10),
  });

  const { data: recommendations, isLoading: recommendationsLoading } = useQuery({
    queryKey: ['recommendations'],
    queryFn: () => api.getRecommendations(10),
  });

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-900/20 to-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Greeting */}
          <h1 className="text-3xl font-bold mb-6">{getGreeting()}</h1>

          {/* Quick Play Grid */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            {trending?.slice(0, 6).map((song) => (
              <div
                key={song.id}
                className="group flex items-center gap-4 bg-white/10 rounded-md overflow-hidden hover:bg-white/20 transition-colors cursor-pointer"
              >
                <img
                  src={song.coverArt || '/placeholder-album.png'}
                  alt={song.title}
                  className="w-20 h-20 object-cover"
                />
                <span className="font-medium truncate pr-4">{song.title}</span>
              </div>
            ))}
          </div>

          {/* Trending Section */}
          <Section title="Trending Now" href="/trending">
            {trendingLoading ? (
              <LoadingGrid count={5} />
            ) : (
              <div className="grid grid-cols-5 gap-6">
                {trending?.slice(0, 5).map((song) => (
                  <SongCard key={song.id} song={song} />
                ))}
              </div>
            )}
          </Section>

          {/* New Releases */}
          <Section title="New Releases" href="/new-releases">
            {newReleasesLoading ? (
              <LoadingGrid count={5} />
            ) : (
              <div className="grid grid-cols-5 gap-6">
                {newReleases?.slice(0, 5).map((song) => (
                  <SongCard key={song.id} song={song} />
                ))}
              </div>
            )}
          </Section>

          {/* Made For You */}
          <Section title="Made For You" href="/recommendations">
            {recommendationsLoading ? (
              <LoadingGrid count={5} />
            ) : (
              <div className="grid grid-cols-5 gap-6">
                {recommendations?.slice(0, 5).map((song) => (
                  <SongCard key={song.id} song={song} />
                ))}
              </div>
            )}
          </Section>

          {/* Genres */}
          <Section title="Browse by Genre">
            <div className="grid grid-cols-5 gap-4">
              {genres.map((genre) => (
                <Link
                  key={genre.name}
                  href={`/genre/${genre.slug}`}
                  className="relative aspect-square rounded-lg overflow-hidden group"
                  style={{ backgroundColor: genre.color }}
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-transparent to-black/50" />
                  <h3 className="absolute bottom-4 left-4 text-xl font-bold">
                    {genre.name}
                  </h3>
                </Link>
              ))}
            </div>
          </Section>
        </div>
      </main>

      <Player />
    </div>
  );
}

// Section Component
function Section({
  title,
  href,
  children,
}: {
  title: string;
  href?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">{title}</h2>
        {href && (
          <Link
            href={href}
            className="text-sm font-medium text-muted-foreground hover:text-white flex items-center gap-1"
          >
            Show all
            <ChevronRight className="w-4 h-4" />
          </Link>
        )}
      </div>
      {children}
    </section>
  );
}

// Loading Grid
function LoadingGrid({ count }: { count: number }) {
  return (
    <div className="grid grid-cols-5 gap-6">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="animate-pulse">
          <div className="aspect-square bg-white/10 rounded-md mb-4" />
          <div className="h-4 bg-white/10 rounded w-3/4 mb-2" />
          <div className="h-3 bg-white/10 rounded w-1/2" />
        </div>
      ))}
    </div>
  );
}

// Genre Data
const genres = [
  { name: 'Electronic', slug: 'electronic', color: '#8B5CF6' },
  { name: 'Hip Hop', slug: 'hip-hop', color: '#EC4899' },
  { name: 'Pop', slug: 'pop', color: '#3B82F6' },
  { name: 'Rock', slug: 'rock', color: '#EF4444' },
  { name: 'Jazz', slug: 'jazz', color: '#F59E0B' },
];
