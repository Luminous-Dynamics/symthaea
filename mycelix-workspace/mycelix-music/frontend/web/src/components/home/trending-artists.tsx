// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Play, TrendingUp, Verified } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useTrendingArtists } from '@/hooks/use-trending-artists';
import { usePlayer } from '@/hooks/use-player';
import { formatNumber } from '@/lib/format';
import { cn } from '@/lib/utils';

export function TrendingArtists() {
  const { data: artists, isLoading } = useTrendingArtists();
  const { playArtist } = usePlayer();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">Trending Artists</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="animate-pulse text-center">
              <div className="mx-auto aspect-square w-full max-w-[160px] rounded-full bg-muted" />
              <div className="mx-auto mt-3 h-4 w-3/4 rounded bg-muted" />
              <div className="mx-auto mt-2 h-3 w-1/2 rounded bg-muted" />
            </div>
          ))}
        </div>
      </section>
    );
  }

  const artistList = artists?.slice(0, 6) || [];

  if (artistList.length === 0) return null;

  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-6 w-6 text-green-500" />
          <h2 className="text-2xl font-bold">Trending Artists</h2>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/artists/trending">Show all</Link>
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-6">
        {artistList.map((artist, index) => (
          <div
            key={artist.id}
            className="group relative text-center"
          >
            {/* Rank Badge */}
            <div
              className={cn(
                'absolute -left-1 -top-1 z-10 flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold text-white',
                index === 0 ? 'bg-yellow-500' :
                index === 1 ? 'bg-gray-400' :
                index === 2 ? 'bg-amber-700' :
                'bg-muted-foreground'
              )}
            >
              {index + 1}
            </div>

            <div className="relative mx-auto aspect-square w-full max-w-[160px]">
              <Avatar className="h-full w-full">
                <AvatarImage
                  src={artist.imageUrl}
                  alt={artist.name}
                  className="object-cover"
                />
                <AvatarFallback className="text-4xl">
                  {artist.name.charAt(0)}
                </AvatarFallback>
              </Avatar>

              {/* Play Button */}
              <button
                onClick={() => playArtist(artist.id)}
                className="absolute inset-0 flex items-center justify-center rounded-full bg-black/50 opacity-0 transition-opacity group-hover:opacity-100"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary">
                  <Play className="h-6 w-6 text-primary-foreground pl-1" />
                </div>
              </button>
            </div>

            <div className="mt-3">
              <Link
                href={`/artist/${artist.id}`}
                className="inline-flex items-center gap-1 font-medium hover:underline"
              >
                {artist.name}
                {artist.verified && (
                  <Verified className="h-4 w-4 fill-primary text-primary-foreground" />
                )}
              </Link>
              <p className="text-sm text-muted-foreground">
                {formatNumber(artist.followers)} followers
              </p>
              {artist.trendChange && (
                <p className={cn(
                  'text-xs',
                  artist.trendChange > 0 ? 'text-green-500' : 'text-red-500'
                )}>
                  {artist.trendChange > 0 ? '+' : ''}{artist.trendChange}% this week
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
