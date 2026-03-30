// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { Suspense } from 'react';
import { HeroSection } from '@/components/home/hero-section';
import { FeaturedSection } from '@/components/home/featured-section';
import { RecentlyPlayed } from '@/components/home/recently-played';
import { PersonalizedMixes } from '@/components/home/personalized-mixes';
import { TrendingArtists } from '@/components/home/trending-artists';
import { NewReleases } from '@/components/home/new-releases';
import { LiveNow } from '@/components/home/live-now';
import { FriendActivity } from '@/components/home/friend-activity';
import { SectionSkeleton } from '@/components/ui/skeletons';
import { useAuth } from '@/hooks/use-auth';

export default function HomePage() {
  const { isAuthenticated, user } = useAuth();

  return (
    <div className="space-y-8 p-6">
      {/* Hero / Welcome */}
      <HeroSection
        greeting={isAuthenticated ? `Welcome back, ${user?.displayName}` : 'Discover Your Sound'}
      />

      {/* Recently Played (authenticated users) */}
      {isAuthenticated && (
        <Suspense fallback={<SectionSkeleton />}>
          <RecentlyPlayed userId={user?.id} />
        </Suspense>
      )}

      {/* Personalized Mixes */}
      {isAuthenticated && (
        <Suspense fallback={<SectionSkeleton />}>
          <PersonalizedMixes userId={user?.id} />
        </Suspense>
      )}

      {/* Live Now */}
      <Suspense fallback={<SectionSkeleton />}>
        <LiveNow />
      </Suspense>

      {/* Featured Content */}
      <Suspense fallback={<SectionSkeleton />}>
        <FeaturedSection />
      </Suspense>

      {/* Trending Artists */}
      <Suspense fallback={<SectionSkeleton />}>
        <TrendingArtists />
      </Suspense>

      {/* New Releases */}
      <Suspense fallback={<SectionSkeleton />}>
        <NewReleases />
      </Suspense>

      {/* Friend Activity Sidebar (authenticated) */}
      {isAuthenticated && (
        <aside className="fixed right-0 top-0 hidden h-screen w-80 border-l bg-card p-4 lg:block">
          <FriendActivity />
        </aside>
      )}
    </div>
  );
}
