// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { Play, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useFeaturedContent } from '@/hooks/use-featured-content';

interface HeroSectionProps {
  greeting: string;
}

export function HeroSection({ greeting }: HeroSectionProps) {
  const { data: featured, isLoading } = useFeaturedContent();
  const [currentIndex, setCurrentIndex] = useState(0);

  const items = featured?.hero || [];
  const currentItem = items[currentIndex];

  useEffect(() => {
    if (items.length <= 1) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % items.length);
    }, 8000);

    return () => clearInterval(interval);
  }, [items.length]);

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev - 1 + items.length) % items.length);
  };

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % items.length);
  };

  return (
    <section className="relative">
      {/* Greeting */}
      <h1 className="mb-6 text-3xl font-bold">{greeting}</h1>

      {/* Hero Carousel */}
      <div className="relative overflow-hidden rounded-2xl">
        {isLoading ? (
          <div className="h-80 animate-pulse bg-muted" />
        ) : currentItem ? (
          <>
            <div className="relative h-80">
              {/* Background Image */}
              <Image
                src={currentItem.imageUrl}
                alt={currentItem.title}
                fill
                className="object-cover"
                priority
              />

              {/* Gradient Overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent" />

              {/* Content */}
              <div className="absolute bottom-0 left-0 right-0 p-8">
                <div className="max-w-2xl">
                  <span className="mb-2 inline-block rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary backdrop-blur">
                    {currentItem.type === 'album' ? 'New Album' : currentItem.type === 'artist' ? 'Featured Artist' : 'Playlist'}
                  </span>
                  <h2 className="mb-2 text-4xl font-bold text-white">{currentItem.title}</h2>
                  <p className="mb-4 text-lg text-gray-300">{currentItem.subtitle}</p>
                  <div className="flex gap-3">
                    <Button size="lg" className="gap-2" asChild>
                      <Link href={currentItem.href}>
                        <Play className="h-5 w-5" />
                        Play Now
                      </Link>
                    </Button>
                    <Button size="lg" variant="secondary" asChild>
                      <Link href={currentItem.href}>
                        Learn More
                      </Link>
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            {/* Navigation Arrows */}
            {items.length > 1 && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute left-4 top-1/2 h-10 w-10 -translate-y-1/2 rounded-full bg-black/50 text-white hover:bg-black/70"
                  onClick={goToPrevious}
                >
                  <ChevronLeft className="h-6 w-6" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-4 top-1/2 h-10 w-10 -translate-y-1/2 rounded-full bg-black/50 text-white hover:bg-black/70"
                  onClick={goToNext}
                >
                  <ChevronRight className="h-6 w-6" />
                </Button>

                {/* Dots Indicator */}
                <div className="absolute bottom-4 right-8 flex gap-2">
                  {items.map((_, index) => (
                    <button
                      key={index}
                      className={cn(
                        'h-2 w-2 rounded-full transition-all',
                        index === currentIndex
                          ? 'w-6 bg-white'
                          : 'bg-white/50 hover:bg-white/70'
                      )}
                      onClick={() => setCurrentIndex(index)}
                    />
                  ))}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="flex h-80 items-center justify-center bg-gradient-to-br from-violet-600 to-purple-800">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-white">Welcome to Mycelix</h2>
              <p className="mt-2 text-gray-200">Discover your next favorite song</p>
            </div>
          </div>
        )}
      </div>

      {/* Quick Access Cards */}
      <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <QuickAccessCard
          title="Made For You"
          href="/discover"
          gradient="from-green-500 to-emerald-600"
        />
        <QuickAccessCard
          title="New Releases"
          href="/new-releases"
          gradient="from-blue-500 to-cyan-600"
        />
        <QuickAccessCard
          title="Top Charts"
          href="/charts"
          gradient="from-orange-500 to-red-600"
        />
        <QuickAccessCard
          title="Live Now"
          href="/live"
          gradient="from-pink-500 to-rose-600"
        />
      </div>
    </section>
  );
}

interface QuickAccessCardProps {
  title: string;
  href: string;
  gradient: string;
}

function QuickAccessCard({ title, href, gradient }: QuickAccessCardProps) {
  return (
    <Link
      href={href}
      className={cn(
        'group flex items-center gap-3 rounded-lg bg-gradient-to-r p-4 transition-transform hover:scale-[1.02]',
        gradient
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-white/20 backdrop-blur">
        <Play className="h-6 w-6 text-white opacity-0 transition-opacity group-hover:opacity-100" />
      </div>
      <span className="font-semibold text-white">{title}</span>
    </Link>
  );
}
