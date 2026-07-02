// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Play, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useRecentlyPlayed } from '@/hooks/use-recently-played';
import { usePlayer } from '@/hooks/use-player';
import { formatRelativeTime } from '@/lib/format';

interface RecentlyPlayedProps {
  userId?: string;
}

export function RecentlyPlayed({ userId }: RecentlyPlayedProps) {
  const { data: recentItems, isLoading } = useRecentlyPlayed(userId);
  const { playTrack, playAlbum, playPlaylist } = usePlayer();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">Recently Played</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="flex items-center gap-3 rounded-lg bg-muted/50 p-2">
                <div className="h-12 w-12 rounded bg-muted" />
                <div className="flex-1">
                  <div className="h-3 w-3/4 rounded bg-muted" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  const items = recentItems?.slice(0, 6) || [];

  if (items.length === 0) return null;

  const handlePlay = (item: typeof items[0]) => {
    switch (item.type) {
      case 'track':
        playTrack(item.id);
        break;
      case 'album':
        playAlbum(item.id);
        break;
      case 'playlist':
        playPlaylist(item.id);
        break;
    }
  };

  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold">Recently Played</h2>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/library/history">
            <Clock className="mr-2 h-4 w-4" />
            History
          </Link>
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        {items.map((item) => (
          <div
            key={`${item.type}-${item.id}`}
            className="group relative flex items-center gap-3 rounded-lg bg-muted/50 p-2 transition-colors hover:bg-muted"
          >
            <div className="relative h-12 w-12 flex-shrink-0 overflow-hidden rounded">
              <Image
                src={item.imageUrl || '/images/default-cover.png'}
                alt={item.title}
                fill
                className="object-cover"
              />
              <button
                onClick={() => handlePlay(item)}
                className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 transition-opacity group-hover:opacity-100"
              >
                <Play className="h-5 w-5 text-white" />
              </button>
            </div>

            <div className="min-w-0 flex-1">
              <Link
                href={`/${item.type}/${item.id}`}
                className="block truncate text-sm font-medium hover:underline"
              >
                {item.title}
              </Link>
              <p className="truncate text-xs text-muted-foreground">
                {formatRelativeTime(item.playedAt)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
