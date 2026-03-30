// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Play, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useFeaturedContent } from '@/hooks/use-featured-content';
import { usePlayer } from '@/hooks/use-player';

export function FeaturedSection() {
  const { data: featured, isLoading } = useFeaturedContent();
  const { playPlaylist } = usePlayer();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">Featured Playlists</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
          {[...Array(5)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-4">
                <div className="aspect-square rounded-lg bg-muted" />
                <div className="mt-3 h-4 w-3/4 rounded bg-muted" />
                <div className="mt-2 h-3 w-1/2 rounded bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    );
  }

  const playlists = featured?.playlists || [];

  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold">Featured Playlists</h2>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/browse/playlists">Show all</Link>
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        {playlists.map((playlist) => (
          <Card
            key={playlist.id}
            className="group relative overflow-hidden transition-colors hover:bg-accent"
          >
            <CardContent className="p-4">
              <div className="relative aspect-square overflow-hidden rounded-lg shadow-lg">
                <Image
                  src={playlist.coverUrl || '/images/default-playlist.png'}
                  alt={playlist.name}
                  fill
                  className="object-cover transition-transform group-hover:scale-105"
                />

                {/* Play Button Overlay */}
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity group-hover:opacity-100">
                  <Button
                    size="icon"
                    className="h-12 w-12 rounded-full shadow-lg"
                    onClick={(e) => {
                      e.preventDefault();
                      playPlaylist(playlist.id);
                    }}
                  >
                    <Play className="h-6 w-6 pl-1" />
                  </Button>
                </div>
              </div>

              <div className="mt-3">
                <Link
                  href={`/playlist/${playlist.id}`}
                  className="block truncate font-medium hover:underline"
                >
                  {playlist.name}
                </Link>
                <p className="mt-1 truncate text-sm text-muted-foreground">
                  {playlist.description || `${playlist.trackCount} tracks`}
                </p>
              </div>

              {/* More Options */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-2 top-2 h-8 w-8 opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>Add to Library</DropdownMenuItem>
                  <DropdownMenuItem>Add to Queue</DropdownMenuItem>
                  <DropdownMenuItem>Share</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
