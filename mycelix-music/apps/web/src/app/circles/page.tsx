// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@/hooks/useAuth';
import { api } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import {
  Users,
  Plus,
  Radio,
  Music2,
  Crown,
  Clock,
  TrendingUp,
  Sparkles,
  Play,
  Lock,
  Globe,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

interface Circle {
  id: string;
  name: string;
  description?: string;
  hostId: string;
  hostName: string;
  hostAvatar?: string;
  participantCount: number;
  maxParticipants: number;
  isPrivate: boolean;
  currentSong?: {
    title: string;
    artist: string;
    coverArt: string;
  };
  genre?: string;
  createdAt: Date;
}

const mockCircles: Circle[] = [
  {
    id: '1',
    name: 'Late Night Chill',
    description: 'Relaxing ambient and downtempo for night owls',
    hostId: '1',
    hostName: 'DJ Moonlight',
    hostAvatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=moonlight',
    participantCount: 23,
    maxParticipants: 50,
    isPrivate: false,
    currentSong: {
      title: 'Midnight Dreams',
      artist: 'Ethereal Waves',
      coverArt: 'https://picsum.photos/seed/midnight/200',
    },
    genre: 'Ambient',
    createdAt: new Date(),
  },
  {
    id: '2',
    name: 'Synthwave Sunday',
    description: 'Weekly synthwave listening party',
    hostId: '2',
    hostName: 'RetroWave',
    hostAvatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=retro',
    participantCount: 45,
    maxParticipants: 100,
    isPrivate: false,
    currentSong: {
      title: 'Neon Nights',
      artist: 'Neon Pulse',
      coverArt: 'https://picsum.photos/seed/neon/200',
    },
    genre: 'Synthwave',
    createdAt: new Date(),
  },
  {
    id: '3',
    name: 'Focus Flow',
    description: 'Instrumental beats for productivity',
    hostId: '3',
    hostName: 'WorkMode',
    hostAvatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=work',
    participantCount: 67,
    maxParticipants: 100,
    isPrivate: false,
    currentSong: {
      title: 'Deep Focus',
      artist: 'Study Beats',
      coverArt: 'https://picsum.photos/seed/focus/200',
    },
    genre: 'Lo-Fi',
    createdAt: new Date(),
  },
];

const mockFriendCircles: Circle[] = [
  {
    id: '4',
    name: 'Alex\'s Room',
    hostId: '4',
    hostName: 'Alex',
    hostAvatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=alex',
    participantCount: 5,
    maxParticipants: 10,
    isPrivate: true,
    currentSong: {
      title: 'Electric Dreams',
      artist: 'Future Past',
      coverArt: 'https://picsum.photos/seed/electric/200',
    },
    createdAt: new Date(),
  },
];

export default function CirclesPage() {
  const { authenticated, user } = useAuth();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filter, setFilter] = useState<'all' | 'friends' | 'genre'>('all');

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500">
            <div className="absolute inset-0 bg-black/20" />
            <div className="relative p-8 md:p-12 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Users className="w-8 h-8" />
                  <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                    Listening Circles
                  </span>
                </div>
                <h1 className="text-4xl md:text-5xl font-bold mb-4">
                  Listen Together
                </h1>
                <p className="text-lg opacity-80 max-w-xl mb-6">
                  Join synchronized listening sessions with friends or discover
                  public circles based on your taste.
                </p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="flex items-center gap-2 px-6 py-3 bg-white text-black rounded-full font-semibold hover:scale-105 transition-transform"
                >
                  <Plus className="w-5 h-5" />
                  Create Circle
                </button>
              </div>

              {/* Animated listeners */}
              <div className="hidden md:flex items-center -space-x-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="w-16 h-16 rounded-full border-4 border-white/20 overflow-hidden animate-pulse"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  >
                    <Image
                      src={`https://api.dicebear.com/7.x/avataaars/svg?seed=user${i}`}
                      alt=""
                      width={64}
                      height={64}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Friend Circles */}
          {mockFriendCircles.length > 0 && (
            <section className="mb-10">
              <div className="flex items-center gap-3 mb-6">
                <Sparkles className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold">Friends Listening</h2>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {mockFriendCircles.map((circle) => (
                  <CircleCard key={circle.id} circle={circle} isFriend />
                ))}
              </div>
            </section>
          )}

          {/* Filter Tabs */}
          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                filter === 'all'
                  ? 'bg-white text-black'
                  : 'bg-white/10 hover:bg-white/20'
              }`}
            >
              All Circles
            </button>
            <button
              onClick={() => setFilter('genre')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                filter === 'genre'
                  ? 'bg-white text-black'
                  : 'bg-white/10 hover:bg-white/20'
              }`}
            >
              By Genre
            </button>
          </div>

          {/* Public Circles */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Globe className="w-6 h-6 text-green-400" />
                <h2 className="text-2xl font-bold">Live Now</h2>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Radio className="w-4 h-4 text-green-400 animate-pulse" />
                {mockCircles.reduce((sum, c) => sum + c.participantCount, 0)} listening
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {mockCircles.map((circle) => (
                <CircleCard key={circle.id} circle={circle} />
              ))}
            </div>
          </section>

          {/* Start Your Own */}
          <section className="mb-10">
            <div className="p-8 rounded-2xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-white/10">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold mb-2">Host Your Own Circle</h3>
                  <p className="text-muted-foreground max-w-md">
                    Create a listening circle for your friends, fans, or community.
                    Share music in real-time with reactions and chat.
                  </p>
                </div>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="px-6 py-3 bg-purple-500 rounded-lg font-medium hover:bg-purple-600 transition-colors"
                >
                  Get Started
                </button>
              </div>
            </div>
          </section>
        </div>
      </main>

      <Player />

      {/* Create Modal */}
      {showCreateModal && (
        <CreateCircleModal onClose={() => setShowCreateModal(false)} />
      )}
    </div>
  );
}

function CircleCard({ circle, isFriend = false }: { circle: Circle; isFriend?: boolean }) {
  return (
    <Link
      href={`/circles/${circle.id}`}
      className="group bg-white/5 rounded-xl overflow-hidden hover:bg-white/10 transition-colors"
    >
      {/* Cover Art */}
      <div className="aspect-video relative">
        {circle.currentSong ? (
          <Image
            src={circle.currentSong.coverArt}
            alt={circle.currentSong.title}
            fill
            className="object-cover"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-purple-500 to-pink-500" />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />

        {/* Live Badge */}
        <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 bg-red-500 rounded-full">
          <Radio className="w-3 h-3 animate-pulse" />
          <span className="text-xs font-medium">LIVE</span>
        </div>

        {/* Privacy Badge */}
        {circle.isPrivate && (
          <div className="absolute top-3 right-3">
            <Lock className="w-4 h-4" />
          </div>
        )}

        {/* Current Song */}
        {circle.currentSong && (
          <div className="absolute bottom-3 left-3 right-3">
            <p className="text-sm font-medium truncate">{circle.currentSong.title}</p>
            <p className="text-xs text-white/70 truncate">{circle.currentSong.artist}</p>
          </div>
        )}

        {/* Play Overlay */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="w-14 h-14 rounded-full bg-purple-500 flex items-center justify-center">
            <Play className="w-6 h-6 ml-1" />
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h3 className="font-semibold group-hover:text-purple-400 transition-colors">
              {circle.name}
            </h3>
            {circle.description && (
              <p className="text-sm text-muted-foreground line-clamp-1">
                {circle.description}
              </p>
            )}
          </div>
          {circle.genre && (
            <span className="px-2 py-0.5 bg-white/10 rounded text-xs">
              {circle.genre}
            </span>
          )}
        </div>

        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              {circle.hostAvatar ? (
                <Image
                  src={circle.hostAvatar}
                  alt={circle.hostName}
                  width={20}
                  height={20}
                  className="rounded-full"
                />
              ) : (
                <Crown className="w-4 h-4 text-yellow-400" />
              )}
              <span>{circle.hostName}</span>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Users className="w-4 h-4" />
            <span>{circle.participantCount}/{circle.maxParticipants}</span>
          </div>
        </div>
      </div>
    </Link>
  );
}

function CreateCircleModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [maxParticipants, setMaxParticipants] = useState(50);

  const handleCreate = () => {
    // Create circle via API
    console.log({ name, description, isPrivate, maxParticipants });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-2xl max-w-md w-full p-6">
        <h2 className="text-2xl font-bold mb-6">Create Listening Circle</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Circle Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Late Night Vibes"
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What's this circle about?"
              rows={3}
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Max Participants</label>
            <select
              value={maxParticipants}
              onChange={(e) => setMaxParticipants(parseInt(e.target.value))}
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            >
              <option value={10}>10 people</option>
              <option value={25}>25 people</option>
              <option value={50}>50 people</option>
              <option value={100}>100 people</option>
            </select>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Private Circle</p>
              <p className="text-sm text-muted-foreground">Only invited friends can join</p>
            </div>
            <button
              onClick={() => setIsPrivate(!isPrivate)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                isPrivate ? 'bg-purple-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  isPrivate ? 'left-6' : 'left-0.5'
                }`}
              />
            </button>
          </div>
        </div>

        <div className="flex gap-3 mt-8">
          <button
            onClick={onClose}
            className="flex-1 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name.trim()}
            className="flex-1 py-2 bg-purple-500 rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Create Circle
          </button>
        </div>
      </div>
    </div>
  );
}
