// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { PresenceToken, PresenceCollection, Presence, PresenceType } from '@/components/presence/PresenceToken';
import {
  Shield,
  Sparkles,
  Calendar,
  MapPin,
  Users,
  Crown,
  Music2,
  Mic2,
  Radio,
  Heart,
  Star,
  Zap,
  Trophy,
  Filter,
  Grid,
  List,
} from 'lucide-react';
import Image from 'next/image';

// Mock presence data
const mockPresences: Presence[] = [
  {
    id: '1',
    type: 'LIVE_CONCERT',
    eventId: 'evt-1',
    eventName: 'Summer Solstice Festival',
    timestamp: new Date('2024-06-21'),
    duration: 7200,
    artistId: 'artist-1',
    artistName: 'Aurora Rising',
    location: 'Red Rocks, Colorado',
    resonanceLevel: 95,
    imageUrl: 'https://picsum.photos/seed/concert1/400',
    abilities: ['Early Access', 'Backstage Pass'],
  },
  {
    id: '2',
    type: 'ALBUM_RELEASE',
    eventId: 'evt-2',
    eventName: 'Ethereal Dreams - Midnight Release',
    timestamp: new Date('2024-05-15'),
    duration: 3600,
    artistId: 'artist-2',
    artistName: 'Cosmic Drift',
    location: 'Global Virtual Release',
    resonanceLevel: 88,
    abilities: ['First 1000 Listeners'],
  },
  {
    id: '3',
    type: 'LISTENING_CIRCLE',
    eventId: 'evt-3',
    eventName: 'Deep Focus Sunday',
    timestamp: new Date('2024-06-02'),
    duration: 5400,
    artistId: 'artist-3',
    artistName: 'Ambient Collective',
    location: 'Virtual Circle',
    resonanceLevel: 72,
  },
  {
    id: '4',
    type: 'ARTIST_MILESTONE',
    eventId: 'evt-4',
    eventName: '1 Million Streams Celebration',
    timestamp: new Date('2024-04-20'),
    duration: 1800,
    artistId: 'artist-4',
    artistName: 'Neon Pulse',
    location: 'Mycelix Platform',
    resonanceLevel: 85,
    abilities: ['Milestone Badge', 'Special Thanks'],
  },
  {
    id: '5',
    type: 'COLLABORATIVE_SESSION',
    eventId: 'evt-5',
    eventName: 'Beats & Breaks Jam',
    timestamp: new Date('2024-05-28'),
    duration: 10800,
    artistId: 'artist-5',
    artistName: 'Studio Collective',
    location: 'LA Studio',
    resonanceLevel: 92,
    abilities: ['Contributor Credit'],
  },
  {
    id: '6',
    type: 'SEASONAL_GATHERING',
    eventId: 'evt-6',
    eventName: 'Spring Equinox Ceremony',
    timestamp: new Date('2024-03-20'),
    duration: 4500,
    artistId: 'artist-6',
    artistName: 'Earth Rhythms',
    location: 'Global Synchronous',
    resonanceLevel: 78,
  },
];

// Upcoming events
const upcomingEvents = [
  {
    id: 'upcoming-1',
    name: 'Summer Solstice 2024',
    type: 'SEASONAL_GATHERING' as PresenceType,
    date: new Date('2024-06-21'),
    artist: 'Community',
    location: 'Global',
    imageUrl: 'https://picsum.photos/seed/solstice/400/200',
  },
  {
    id: 'upcoming-2',
    name: 'Album Release: Nebula',
    type: 'ALBUM_RELEASE' as PresenceType,
    date: new Date('2024-07-15'),
    artist: 'Stellar Dreams',
    location: 'Virtual',
    imageUrl: 'https://picsum.photos/seed/nebula/400/200',
  },
  {
    id: 'upcoming-3',
    name: 'Live at The Venue',
    type: 'LIVE_CONCERT' as PresenceType,
    date: new Date('2024-08-10'),
    artist: 'Electric Soul',
    location: 'Brooklyn, NY',
    imageUrl: 'https://picsum.photos/seed/venue/400/200',
  },
];

type FilterType = 'all' | PresenceType;

export default function PresencePage() {
  const [filter, setFilter] = useState<FilterType>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const filteredPresences = filter === 'all'
    ? mockPresences
    : mockPresences.filter((p) => p.type === filter);

  // Calculate stats
  const totalResonance = mockPresences.reduce((sum, p) => sum + p.resonanceLevel, 0);
  const avgResonance = Math.round(totalResonance / mockPresences.length);
  const totalDuration = mockPresences.reduce((sum, p) => sum + p.duration, 0);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600">
            <div className="absolute inset-0 bg-black/30" />

            {/* Floating tokens background */}
            <div className="absolute inset-0 overflow-hidden">
              {[...Array(8)].map((_, i) => (
                <div
                  key={i}
                  className="absolute animate-float"
                  style={{
                    left: `${10 + i * 12}%`,
                    top: `${20 + (i % 3) * 20}%`,
                    animationDelay: `${i * 0.5}s`,
                    opacity: 0.3,
                  }}
                >
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-400 to-pink-400" />
                </div>
              ))}
            </div>

            <div className="relative p-8 md:p-12">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-8 h-8" />
                <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                  Proof of Presence
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Your Musical Journey
              </h1>
              <p className="text-lg opacity-80 max-w-xl mb-6">
                Soulbound tokens that commemorate your authentic musical experiences.
                These cannot be bought or sold — only earned by being present.
              </p>

              {/* Stats */}
              <div className="flex items-center gap-8">
                <div>
                  <p className="text-3xl font-bold">{mockPresences.length}</p>
                  <p className="text-sm opacity-70">Moments Captured</p>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <p className="text-3xl font-bold">{avgResonance}%</p>
                  <p className="text-sm opacity-70">Avg Resonance</p>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <p className="text-3xl font-bold">{Math.round(totalDuration / 3600)}h</p>
                  <p className="text-sm opacity-70">Time Present</p>
                </div>
              </div>
            </div>
          </div>

          {/* Upcoming Events */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Calendar className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold">Upcoming Experiences</h2>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {upcomingEvents.map((event) => (
                <UpcomingEventCard key={event.id} event={event} />
              ))}
            </div>
          </section>

          {/* Collection */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Sparkles className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold">Your Collection</h2>
              </div>

              <div className="flex items-center gap-4">
                {/* Filter */}
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-muted-foreground" />
                  <select
                    value={filter}
                    onChange={(e) => setFilter(e.target.value as FilterType)}
                    className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="LIVE_CONCERT">Live Concerts</option>
                    <option value="VIRTUAL_CONCERT">Virtual Concerts</option>
                    <option value="ALBUM_RELEASE">Album Releases</option>
                    <option value="LISTENING_CIRCLE">Listening Circles</option>
                    <option value="ARTIST_MILESTONE">Artist Milestones</option>
                    <option value="COLLABORATIVE_SESSION">Collaborations</option>
                    <option value="SEASONAL_GATHERING">Seasonal Gatherings</option>
                  </select>
                </div>

                {/* View mode */}
                <div className="flex items-center bg-white/5 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-1.5 rounded ${viewMode === 'grid' ? 'bg-white/10' : ''}`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`p-1.5 rounded ${viewMode === 'list' ? 'bg-white/10' : ''}`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {viewMode === 'grid' ? (
              <div className="flex flex-wrap gap-6">
                {filteredPresences.map((presence) => (
                  <PresenceToken
                    key={presence.id}
                    presence={presence}
                    size="large"
                    showDetails
                  />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredPresences.map((presence) => (
                  <PresenceListItem key={presence.id} presence={presence} />
                ))}
              </div>
            )}
          </section>

          {/* Abilities Unlocked */}
          <section className="mb-10">
            <div className="flex items-center gap-3 mb-6">
              <Zap className="w-6 h-6 text-yellow-400" />
              <h2 className="text-2xl font-bold">Abilities Unlocked</h2>
            </div>

            <div className="grid grid-cols-4 gap-4">
              <AbilityCard
                icon={Crown}
                title="Early Access"
                description="Get new releases 24h before everyone else"
                source="Summer Solstice Festival"
              />
              <AbilityCard
                icon={Star}
                title="First 1000 Badge"
                description="Exclusive badge for early supporters"
                source="Ethereal Dreams Release"
              />
              <AbilityCard
                icon={Trophy}
                title="Contributor Credit"
                description="Your name in album credits"
                source="Beats & Breaks Jam"
              />
              <AbilityCard
                icon={Heart}
                title="Special Thanks"
                description="Personal shoutout from artist"
                source="1 Million Streams"
              />
            </div>
          </section>

          {/* Philosophy */}
          <section className="mb-10">
            <div className="p-8 rounded-2xl bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-white/10">
              <div className="flex items-start gap-6">
                <Shield className="w-12 h-12 text-purple-400 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold mb-3">Why Soulbound?</h3>
                  <p className="text-muted-foreground mb-4">
                    Proof of Presence tokens are <strong className="text-white">non-transferable</strong>.
                    They cannot be bought, sold, or traded. This design choice is intentional:
                  </p>
                  <ul className="space-y-2 text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-purple-400">•</span>
                      <span>They represent <em>authentic experience</em>, not speculation</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-400">•</span>
                      <span>They create a verifiable record of your musical journey</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-400">•</span>
                      <span>They unlock real abilities based on genuine participation</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-400">•</span>
                      <span>They celebrate presence as the most valuable form of support</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>

      <Player />

      <style jsx global>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(5deg); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}

interface UpcomingEventCardProps {
  event: {
    id: string;
    name: string;
    type: PresenceType;
    date: Date;
    artist: string;
    location: string;
    imageUrl: string;
  };
}

function UpcomingEventCard({ event }: UpcomingEventCardProps) {
  const daysUntil = Math.ceil((event.date.getTime() - Date.now()) / (1000 * 60 * 60 * 24));

  return (
    <div className="group relative rounded-xl overflow-hidden bg-white/5 hover:bg-white/10 transition-colors">
      <div className="aspect-video relative">
        <Image src={event.imageUrl} alt={event.name} fill className="object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />

        <div className="absolute top-3 right-3 px-2 py-1 bg-purple-500 rounded-full text-xs font-medium">
          {daysUntil > 0 ? `${daysUntil} days` : 'Today!'}
        </div>

        <div className="absolute bottom-3 left-3 right-3">
          <p className="font-semibold truncate">{event.name}</p>
          <p className="text-sm text-white/70">{event.artist}</p>
        </div>
      </div>

      <div className="p-3 flex items-center justify-between text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <Calendar className="w-4 h-4" />
          <span>{event.date.toLocaleDateString()}</span>
        </div>
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4" />
          <span>{event.location}</span>
        </div>
      </div>
    </div>
  );
}

function PresenceListItem({ presence }: { presence: Presence }) {
  return (
    <div className="flex items-center gap-4 p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors">
      <PresenceToken presence={presence} size="small" />

      <div className="flex-1 min-w-0">
        <h4 className="font-semibold truncate">{presence.eventName}</h4>
        <p className="text-sm text-muted-foreground">{presence.artistName}</p>
      </div>

      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <Calendar className="w-4 h-4" />
          <span>{presence.timestamp.toLocaleDateString()}</span>
        </div>
        <div className="flex items-center gap-1">
          <MapPin className="w-4 h-4" />
          <span>{presence.location}</span>
        </div>
        <div className="flex items-center gap-1 text-purple-400">
          <Zap className="w-4 h-4" />
          <span>{presence.resonanceLevel}%</span>
        </div>
      </div>
    </div>
  );
}

interface AbilityCardProps {
  icon: typeof Crown;
  title: string;
  description: string;
  source: string;
}

function AbilityCard({ icon: Icon, title, description, source }: AbilityCardProps) {
  return (
    <div className="p-4 bg-white/5 rounded-xl border border-white/10 hover:border-purple-500/50 transition-colors">
      <Icon className="w-8 h-8 text-purple-400 mb-3" />
      <h4 className="font-semibold mb-1">{title}</h4>
      <p className="text-sm text-muted-foreground mb-3">{description}</p>
      <p className="text-xs text-purple-400">From: {source}</p>
    </div>
  );
}
