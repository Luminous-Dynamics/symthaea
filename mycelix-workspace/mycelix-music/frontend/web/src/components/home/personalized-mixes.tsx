// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Play, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { usePersonalizedMixes } from '@/hooks/use-personalized-mixes';
import { usePlayer } from '@/hooks/use-player';
import { cn } from '@/lib/utils';

interface PersonalizedMixesProps {
  userId?: string;
}

const mixGradients = [
  'from-violet-600 to-purple-600',
  'from-blue-600 to-cyan-600',
  'from-green-600 to-emerald-600',
  'from-orange-600 to-red-600',
  'from-pink-600 to-rose-600',
  'from-yellow-600 to-amber-600',
];

export function PersonalizedMixes({ userId }: PersonalizedMixesProps) {
  const { data: mixes, isLoading } = usePersonalizedMixes(userId);
  const { playPlaylist } = usePlayer();

  if (isLoading) {
    return (
      <section>
        <h2 className="mb-4 text-2xl font-bold">Made For You</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          {[...Array(6)].map((_, i) => (
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

  const mixList = mixes || [];

  if (mixList.length === 0) return null;

  return (
    <section>
      <div className="mb-4 flex items-center gap-2">
        <Sparkles className="h-6 w-6 text-primary" />
        <h2 className="text-2xl font-bold">Made For You</h2>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        {mixList.map((mix, index) => (
          <Card
            key={mix.id}
            className="group relative overflow-hidden transition-colors hover:bg-accent"
          >
            <CardContent className="p-4">
              <div
                className={cn(
                  'relative aspect-square overflow-hidden rounded-lg bg-gradient-to-br shadow-lg',
                  mixGradients[index % mixGradients.length]
                )}
              >
                {/* Mix Cover Art Grid */}
                <div className="absolute inset-0 grid grid-cols-2 grid-rows-2 gap-0.5 p-2">
                  {mix.previewTracks?.slice(0, 4).map((track, i) => (
                    <div key={i} className="relative overflow-hidden rounded">
                      <Image
                        src={track.coverUrl || '/images/default-cover.png'}
                        alt=""
                        fill
                        className="object-cover opacity-80"
                      />
                    </div>
                  ))}
                </div>

                {/* Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />

                {/* Mix Icon */}
                <div className="absolute bottom-2 left-2">
                  <span className="rounded-full bg-black/40 px-2 py-1 text-xs font-medium text-white backdrop-blur">
                    {mix.type === 'daily' ? 'Daily Mix' : mix.type === 'genre' ? 'Genre Mix' : 'Artist Mix'}
                  </span>
                </div>

                {/* Play Button Overlay */}
                <div className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100">
                  <Button
                    size="icon"
                    className="h-12 w-12 rounded-full shadow-lg"
                    onClick={(e) => {
                      e.preventDefault();
                      playPlaylist(mix.id);
                    }}
                  >
                    <Play className="h-6 w-6 pl-1" />
                  </Button>
                </div>
              </div>

              <div className="mt-3">
                <Link
                  href={`/mix/${mix.id}`}
                  className="block truncate font-medium hover:underline"
                >
                  {mix.name}
                </Link>
                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                  {mix.description}
                </p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
