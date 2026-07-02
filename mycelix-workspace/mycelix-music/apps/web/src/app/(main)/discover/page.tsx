// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { usePlayerStore } from '@/store/playerStore';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import {
  Radio,
  Sparkles,
  Clock,
  Compass,
  TrendingUp,
  Music2,
  Play,
  Shuffle,
  RefreshCw,
  ChevronRight,
  Zap,
  Heart,
  Headphones,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

// Mock data for personalized playlists
const mockDailyMixes = [
  {
    id: 'daily-1',
    name: 'Daily Mix 1',
    description: 'Ethereal Waves, Neon Pulse, and more',
    coverImages: [
      'https://picsum.photos/seed/daily1a/200',
      'https://picsum.photos/seed/daily1b/200',
      'https://picsum.photos/seed/daily1c/200',
      'https://picsum.photos/seed/daily1d/200',
    ],
    color: 'from-purple-600 to-blue-600',
  },
  {
    id: 'daily-2',
    name: 'Daily Mix 2',
    description: 'Forest Protocol, Ambient Dreams, and more',
    coverImages: [
      'https://picsum.photos/seed/daily2a/200',
      'https://picsum.photos/seed/daily2b/200',
      'https://picsum.photos/seed/daily2c/200',
      'https://picsum.photos/seed/daily2d/200',
    ],
    color: 'from-green-600 to-teal-600',
  },
  {
    id: 'daily-3',
    name: 'Daily Mix 3',
    description: 'Synthwave classics and new releases',
    coverImages: [
      'https://picsum.photos/seed/daily3a/200',
      'https://picsum.photos/seed/daily3b/200',
      'https://picsum.photos/seed/daily3c/200',
      'https://picsum.photos/seed/daily3d/200',
    ],
    color: 'from-pink-600 to-orange-600',
  },
  {
    id: 'daily-4',
    name: 'Daily Mix 4',
    description: 'Deep house and electronic vibes',
    coverImages: [
      'https://picsum.photos/seed/daily4a/200',
      'https://picsum.photos/seed/daily4b/200',
      'https://picsum.photos/seed/daily4c/200',
      'https://picsum.photos/seed/daily4d/200',
    ],
    color: 'from-cyan-600 to-blue-600',
  },
];

const mockDiscoverWeekly = {
  id: 'discover-weekly',
  name: 'Discover Weekly',
  description: 'Your personalized playlist, updated every Monday',
  songCount: 30,
  duration: '2h 15m',
  coverArt: 'https://picsum.photos/seed/discover-weekly/400',
  lastUpdated: 'Updated Monday',
};

const mockReleaseRadar = {
  id: 'release-radar',
  name: 'Release Radar',
  description: 'New music from artists you follow',
  songCount: 25,
  duration: '1h 45m',
  coverArt: 'https://picsum.photos/seed/release-radar/400',
  lastUpdated: 'Updated Friday',
};

const mockGenreMixes = [
  { id: 'ambient', name: 'Ambient', color: 'from-blue-900 to-indigo-900' },
  { id: 'electronic', name: 'Electronic', color: 'from-purple-900 to-pink-900' },
  { id: 'synthwave', name: 'Synthwave', color: 'from-pink-900 to-orange-900' },
  { id: 'downtempo', name: 'Downtempo', color: 'from-teal-900 to-cyan-900' },
  { id: 'house', name: 'House', color: 'from-green-900 to-emerald-900' },
  { id: 'techno', name: 'Techno', color: 'from-gray-900 to-slate-900' },
];

const mockRadioStations = [
  {
    id: 'chill-radio',
    name: 'Chill Radio',
    description: 'Relaxing beats and ambient soundscapes',
    listeners: 2340,
    coverArt: 'https://picsum.photos/seed/chill-radio/200',
  },
  {
    id: 'electronic-radio',
    name: 'Electronic Radio',
    description: 'The best in electronic music',
    listeners: 5670,
    coverArt: 'https://picsum.photos/seed/electronic-radio/200',
  },
  {
    id: 'focus-radio',
    name: 'Focus Radio',
    description: 'Music for concentration and productivity',
    listeners: 3210,
    coverArt: 'https://picsum.photos/seed/focus-radio/200',
  },
  {
    id: 'workout-radio',
    name: 'Workout Radio',
    description: 'High energy tracks to keep you moving',
    listeners: 4560,
    coverArt: 'https://picsum.photos/seed/workout-radio/200',
  },
];

const mockNewReleases = [
  {
    id: 'nr1',
    title: 'Cosmic Journey',
    artist: 'Ethereal Waves',
    coverArt: 'https://picsum.photos/seed/nr1/200',
    releaseDate: '2024-01-05',
  },
  {
    id: 'nr2',
    title: 'Neon Dreams',
    artist: 'Neon Pulse',
    coverArt: 'https://picsum.photos/seed/nr2/200',
    releaseDate: '2024-01-04',
  },
  {
    id: 'nr3',
    title: 'Digital Forest',
    artist: 'Forest Protocol',
    coverArt: 'https://picsum.photos/seed/nr3/200',
    releaseDate: '2024-01-03',
  },
  {
    id: 'nr4',
    title: 'Midnight Protocol',
    artist: 'Cyber Dreams',
    coverArt: 'https://picsum.photos/seed/nr4/200',
    releaseDate: '2024-01-02',
  },
  {
    id: 'nr5',
    title: 'Aurora',
    artist: 'Northern Lights',
    coverArt: 'https://picsum.photos/seed/nr5/200',
    releaseDate: '2024-01-01',
  },
];

export default function DiscoverPage() {
  const { playAll } = usePlayerStore();
  const [activeRadio, setActiveRadio] = useState<string | null>(null);

  const handleStartRadio = (radioId: string) => {
    setActiveRadio(radioId);
    // In production, fetch tracks and start playing
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero Section */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-600">
            <div className="absolute inset-0 bg-black/30" />
            <div className="relative p-8 md:p-12">
              <div className="flex items-center gap-3 mb-4">
                <Sparkles className="w-8 h-8" />
                <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                  Made for You
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">Discover</h1>
              <p className="text-lg opacity-80 max-w-xl mb-6">
                Personalized music recommendations powered by your listening history and preferences.
              </p>
              <button className="flex items-center gap-2 px-6 py-3 bg-white text-black rounded-full font-semibold hover:scale-105 transition-transform">
                <Shuffle className="w-5 h-5" />
                Start Discovery Session
              </button>
            </div>
          </div>

          {/* Personalized Playlists */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Made for You</h2>
              <Link
                href="/discover/personalized"
                className="text-sm text-muted-foreground hover:text-white flex items-center gap-1"
              >
                Show all <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              {/* Discover Weekly */}
              <FeaturedPlaylistCard
                playlist={mockDiscoverWeekly}
                icon={<Compass className="w-6 h-6" />}
                gradient="from-green-600 to-emerald-600"
              />

              {/* Release Radar */}
              <FeaturedPlaylistCard
                playlist={mockReleaseRadar}
                icon={<Zap className="w-6 h-6" />}
                gradient="from-blue-600 to-cyan-600"
              />
            </div>

            {/* Daily Mixes */}
            <div className="grid grid-cols-4 gap-4">
              {mockDailyMixes.map((mix) => (
                <DailyMixCard key={mix.id} mix={mix} />
              ))}
            </div>
          </section>

          {/* Radio Stations */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Radio className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold">Radio</h2>
              </div>
              <Link
                href="/discover/radio"
                className="text-sm text-muted-foreground hover:text-white flex items-center gap-1"
              >
                Show all <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="grid grid-cols-4 gap-4">
              {mockRadioStations.map((station) => (
                <RadioStationCard
                  key={station.id}
                  station={station}
                  isActive={activeRadio === station.id}
                  onStart={() => handleStartRadio(station.id)}
                />
              ))}
            </div>
          </section>

          {/* Genre Mixes */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Browse by Genre</h2>
            </div>

            <div className="grid grid-cols-6 gap-3">
              {mockGenreMixes.map((genre) => (
                <Link
                  key={genre.id}
                  href={`/genre/${genre.id}`}
                  className={`p-6 rounded-xl bg-gradient-to-br ${genre.color} hover:scale-105 transition-transform`}
                >
                  <span className="font-semibold">{genre.name}</span>
                </Link>
              ))}
            </div>
          </section>

          {/* New Releases */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-green-400" />
                <h2 className="text-2xl font-bold">New Releases</h2>
              </div>
              <Link
                href="/discover/new"
                className="text-sm text-muted-foreground hover:text-white flex items-center gap-1"
              >
                Show all <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="grid grid-cols-5 gap-4">
              {mockNewReleases.map((release) => (
                <NewReleaseCard key={release.id} release={release} />
              ))}
            </div>
          </section>

          {/* Listening Moods */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Moods & Activities</h2>
            </div>

            <div className="grid grid-cols-4 gap-4">
              <MoodCard
                name="Focus"
                description="Concentrate and be productive"
                icon={<Headphones className="w-8 h-8" />}
                gradient="from-indigo-600 to-purple-600"
              />
              <MoodCard
                name="Chill"
                description="Relax and unwind"
                icon={<Heart className="w-8 h-8" />}
                gradient="from-pink-600 to-rose-600"
              />
              <MoodCard
                name="Energy"
                description="Get pumped and motivated"
                icon={<Zap className="w-8 h-8" />}
                gradient="from-orange-600 to-red-600"
              />
              <MoodCard
                name="Sleep"
                description="Peaceful sounds for rest"
                icon={<Clock className="w-8 h-8" />}
                gradient="from-blue-600 to-indigo-600"
              />
            </div>
          </section>
        </div>
      </main>

      <Player />
    </div>
  );
}

function FeaturedPlaylistCard({
  playlist,
  icon,
  gradient,
}: {
  playlist: typeof mockDiscoverWeekly;
  icon: React.ReactNode;
  gradient: string;
}) {
  return (
    <div className={`relative rounded-xl overflow-hidden bg-gradient-to-br ${gradient} group`}>
      <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
      <div className="relative p-6 flex gap-6">
        <div className="relative w-40 h-40 rounded-lg overflow-hidden shadow-2xl flex-shrink-0">
          <Image
            src={playlist.coverArt}
            alt={playlist.name}
            fill
            className="object-cover"
          />
        </div>
        <div className="flex flex-col justify-between py-2">
          <div>
            <div className="flex items-center gap-2 mb-2">
              {icon}
              <span className="text-sm font-medium opacity-80">{playlist.lastUpdated}</span>
            </div>
            <h3 className="text-2xl font-bold mb-2">{playlist.name}</h3>
            <p className="text-sm opacity-80">{playlist.description}</p>
          </div>
          <div className="flex items-center gap-4">
            <button className="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center hover:scale-110 transition-transform shadow-lg">
              <Play className="w-5 h-5 ml-0.5" />
            </button>
            <span className="text-sm opacity-80">
              {playlist.songCount} songs &bull; {playlist.duration}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function DailyMixCard({ mix }: { mix: (typeof mockDailyMixes)[0] }) {
  return (
    <Link
      href={`/playlist/${mix.id}`}
      className="group relative rounded-xl overflow-hidden bg-white/5 hover:bg-white/10 transition-colors"
    >
      {/* 2x2 Album Grid */}
      <div className="aspect-square relative">
        <div className="absolute inset-0 grid grid-cols-2 grid-rows-2">
          {mix.coverImages.map((img, i) => (
            <div key={i} className="relative">
              <Image src={img} alt="" fill className="object-cover" />
            </div>
          ))}
        </div>
        <div className={`absolute inset-0 bg-gradient-to-t ${mix.color} opacity-60`} />

        {/* Play button */}
        <button className="absolute bottom-2 right-2 w-12 h-12 rounded-full bg-purple-500 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 transition-all shadow-lg">
          <Play className="w-5 h-5 ml-0.5" />
        </button>
      </div>

      <div className="p-4">
        <h3 className="font-semibold mb-1">{mix.name}</h3>
        <p className="text-sm text-muted-foreground line-clamp-2">{mix.description}</p>
      </div>
    </Link>
  );
}

function RadioStationCard({
  station,
  isActive,
  onStart,
}: {
  station: (typeof mockRadioStations)[0];
  isActive: boolean;
  onStart: () => void;
}) {
  return (
    <div
      className={`relative rounded-xl overflow-hidden bg-white/5 hover:bg-white/10 transition-colors group ${
        isActive ? 'ring-2 ring-purple-500' : ''
      }`}
    >
      <div className="aspect-square relative">
        <Image
          src={station.coverArt}
          alt={station.name}
          fill
          className="object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />

        {isActive && (
          <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 bg-purple-500 rounded-full">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span className="text-xs font-medium">LIVE</span>
          </div>
        )}

        <button
          onClick={onStart}
          className="absolute bottom-3 right-3 w-12 h-12 rounded-full bg-purple-500 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 transition-all shadow-lg"
        >
          {isActive ? <RefreshCw className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>

        <div className="absolute bottom-3 left-3">
          <h3 className="font-semibold">{station.name}</h3>
          <p className="text-sm text-white/70">
            {station.listeners.toLocaleString()} listening
          </p>
        </div>
      </div>
    </div>
  );
}

function NewReleaseCard({ release }: { release: (typeof mockNewReleases)[0] }) {
  return (
    <Link href={`/song/${release.id}`} className="group">
      <div className="aspect-square relative rounded-lg overflow-hidden mb-3">
        <Image
          src={release.coverArt}
          alt={release.title}
          fill
          className="object-cover group-hover:scale-105 transition-transform"
        />
        <button className="absolute bottom-2 right-2 w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 transition-all shadow-lg">
          <Play className="w-4 h-4 ml-0.5" />
        </button>
      </div>
      <h4 className="font-medium truncate">{release.title}</h4>
      <p className="text-sm text-muted-foreground truncate">{release.artist}</p>
    </Link>
  );
}

function MoodCard({
  name,
  description,
  icon,
  gradient,
}: {
  name: string;
  description: string;
  icon: React.ReactNode;
  gradient: string;
}) {
  return (
    <Link
      href={`/mood/${name.toLowerCase()}`}
      className={`p-6 rounded-xl bg-gradient-to-br ${gradient} hover:scale-105 transition-transform`}
    >
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-bold mb-1">{name}</h3>
      <p className="text-sm opacity-80">{description}</p>
    </Link>
  );
}
