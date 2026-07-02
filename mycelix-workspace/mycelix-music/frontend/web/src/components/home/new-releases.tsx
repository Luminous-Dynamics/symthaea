// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Play, Calendar, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useNewReleases } from '@/hooks/use-new-releases';
import { usePlayer } from '@/hooks/use-player';
import { formatDate } from '@/lib/format';

export function NewReleases() {
  const { data: releases, isLoading } = useNewReleases();
  const { playAlbum } = usePlayer();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">New Releases</h2>
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

  const albums = releases?.slice(0, 10) || [];

  if (albums.length === 0) return null;

  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Calendar className="h-6 w-6 text-blue-500" />
          <h2 className="text-2xl font-bold">New Releases</h2>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/new-releases">Show all</Link>
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        {albums.map((album) => (
          <Card
            key={album.id}
            className="group relative overflow-hidden transition-colors hover:bg-accent"
          >
            <CardContent className="p-4">
              <div className="relative aspect-square overflow-hidden rounded-lg shadow-lg">
                <Image
                  src={album.coverUrl || '/images/default-album.png'}
                  alt={album.title}
                  fill
                  className="object-cover transition-transform group-hover:scale-105"
                />

                {/* Release Type Badge */}
                <div className="absolute left-2 top-2">
                  <span className="rounded-full bg-black/60 px-2 py-1 text-xs font-medium text-white backdrop-blur">
                    {album.type === 'single' ? 'Single' :
                     album.type === 'ep' ? 'EP' : 'Album'}
                  </span>
                </div>

                {/* Play Button Overlay */}
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity group-hover:opacity-100">
                  <Button
                    size="icon"
                    className="h-12 w-12 rounded-full shadow-lg"
                    onClick={(e) => {
                      e.preventDefault();
                      playAlbum(album.id);
                    }}
                  >
                    <Play className="h-6 w-6 pl-1" />
                  </Button>
                </div>
              </div>

              <div className="mt-3">
                <Link
                  href={`/album/${album.id}`}
                  className="block truncate font-medium hover:underline"
                >
                  {album.title}
                </Link>
                <Link
                  href={`/artist/${album.artist.id}`}
                  className="mt-1 block truncate text-sm text-muted-foreground hover:underline"
                >
                  {album.artist.name}
                </Link>
                <p className="mt-1 text-xs text-muted-foreground">
                  {formatDate(album.releaseDate)}
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
                  <DropdownMenuItem>Go to Artist</DropdownMenuItem>
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
