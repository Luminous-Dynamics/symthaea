// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-muted', className)}
    />
  );
}

export function SectionSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-48" />
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="space-y-3">
            <Skeleton className="aspect-square w-full rounded-lg" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function TrackSkeleton() {
  return (
    <div className="flex items-center gap-3 p-2">
      <Skeleton className="h-12 w-12 rounded" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
      </div>
      <Skeleton className="h-4 w-12" />
    </div>
  );
}

export function ArtistSkeleton() {
  return (
    <div className="text-center">
      <Skeleton className="mx-auto aspect-square w-full max-w-[160px] rounded-full" />
      <Skeleton className="mx-auto mt-3 h-4 w-3/4" />
      <Skeleton className="mx-auto mt-2 h-3 w-1/2" />
    </div>
  );
}

export function PlaylistSkeleton() {
  return (
    <div className="space-y-3 p-4">
      <Skeleton className="aspect-square w-full rounded-lg" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-3 w-1/2" />
    </div>
  );
}

export function PlayerSkeleton() {
  return (
    <div className="flex h-20 items-center justify-between px-4">
      <div className="flex items-center gap-3">
        <Skeleton className="h-14 w-14 rounded" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-24" />
        </div>
      </div>
      <div className="flex flex-col items-center gap-2">
        <div className="flex gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-10 w-10 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
        <Skeleton className="h-1 w-96 rounded-full" />
      </div>
      <div className="flex gap-2">
        <Skeleton className="h-8 w-8 rounded" />
        <Skeleton className="h-8 w-24 rounded" />
      </div>
    </div>
  );
}

export function ProfileSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <Skeleton className="h-40 w-40 rounded-full" />
        <div className="space-y-3">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
      </div>
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-lg" />
        ))}
      </div>
    </div>
  );
}

export function AlbumPageSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex gap-6">
        <Skeleton className="h-56 w-56 rounded-lg" />
        <div className="space-y-3">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-5 w-48" />
          <Skeleton className="h-4 w-32" />
          <div className="flex gap-3 pt-4">
            <Skeleton className="h-12 w-32 rounded-full" />
            <Skeleton className="h-12 w-12 rounded-full" />
          </div>
        </div>
      </div>
      <div className="space-y-2">
        {[...Array(10)].map((_, i) => (
          <TrackSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

export function SearchResultsSkeleton() {
  return (
    <div className="space-y-8">
      <div>
        <Skeleton className="mb-4 h-6 w-24" />
        <div className="flex gap-4">
          {[...Array(4)].map((_, i) => (
            <ArtistSkeleton key={i} />
          ))}
        </div>
      </div>
      <div>
        <Skeleton className="mb-4 h-6 w-24" />
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <TrackSkeleton key={i} />
          ))}
        </div>
      </div>
      <div>
        <Skeleton className="mb-4 h-6 w-24" />
        <div className="grid grid-cols-5 gap-4">
          {[...Array(5)].map((_, i) => (
            <PlaylistSkeleton key={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
