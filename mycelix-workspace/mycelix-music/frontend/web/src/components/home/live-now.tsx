// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Radio, Users, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useLiveStreams } from '@/hooks/use-live-streams';
import { formatNumber } from '@/lib/format';
import { cn } from '@/lib/utils';

export function LiveNow() {
  const { data: streams, isLoading } = useLiveStreams();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">Live Now</h2>
        <div className="flex gap-4 overflow-x-auto pb-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="w-72 flex-shrink-0 animate-pulse">
              <div className="aspect-video rounded-lg bg-muted" />
              <div className="mt-3 flex gap-3">
                <div className="h-10 w-10 rounded-full bg-muted" />
                <div className="flex-1">
                  <div className="h-4 w-3/4 rounded bg-muted" />
                  <div className="mt-2 h-3 w-1/2 rounded bg-muted" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  const liveStreams = streams?.filter(s => s.isLive) || [];

  if (liveStreams.length === 0) {
    return (
      <section>
        <div className="mb-4 flex items-center gap-2">
          <Radio className="h-6 w-6 text-red-500" />
          <h2 className="text-2xl font-bold">Live Now</h2>
        </div>
        <div className="rounded-lg border border-dashed p-8 text-center">
          <Radio className="mx-auto h-12 w-12 text-muted-foreground" />
          <p className="mt-4 text-muted-foreground">
            No live streams right now. Check back later!
          </p>
          <Button variant="outline" className="mt-4" asChild>
            <Link href="/live/schedule">View Schedule</Link>
          </Button>
        </div>
      </section>
    );
  }

  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radio className="h-6 w-6 animate-pulse text-red-500" />
          <h2 className="text-2xl font-bold">Live Now</h2>
          <Badge variant="destructive" className="animate-pulse">
            {liveStreams.length} LIVE
          </Badge>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/live">View All</Link>
        </Button>
      </div>

      <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-muted">
        {liveStreams.map((stream) => (
          <Link
            key={stream.id}
            href={`/live/${stream.id}`}
            className="group w-72 flex-shrink-0"
          >
            {/* Stream Preview */}
            <div className="relative aspect-video overflow-hidden rounded-lg">
              <Image
                src={stream.thumbnailUrl || '/images/default-stream.png'}
                alt={stream.title}
                fill
                className="object-cover transition-transform group-hover:scale-105"
              />

              {/* Live Badge */}
              <div className="absolute left-2 top-2 flex items-center gap-1">
                <Badge variant="destructive" className="animate-pulse gap-1">
                  <span className="h-2 w-2 rounded-full bg-white" />
                  LIVE
                </Badge>
              </div>

              {/* Viewer Count */}
              <div className="absolute bottom-2 right-2 flex items-center gap-1 rounded-full bg-black/60 px-2 py-1 text-xs text-white backdrop-blur">
                <Users className="h-3 w-3" />
                {formatNumber(stream.viewerCount)}
              </div>

              {/* Hover Overlay */}
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity group-hover:opacity-100">
                <Button variant="secondary" className="gap-2">
                  Watch Now
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Stream Info */}
            <div className="mt-3 flex gap-3">
              <Avatar className="h-10 w-10">
                <AvatarImage src={stream.artist.imageUrl} />
                <AvatarFallback>{stream.artist.name.charAt(0)}</AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1">
                <h3 className="truncate font-medium group-hover:text-primary">
                  {stream.title}
                </h3>
                <p className="truncate text-sm text-muted-foreground">
                  {stream.artist.name}
                </p>
                <div className="mt-1 flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {stream.category}
                  </Badge>
                  {stream.tags?.slice(0, 2).map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}
