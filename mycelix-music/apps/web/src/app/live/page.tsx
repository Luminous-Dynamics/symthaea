// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Link from 'next/link';
import { Radio, Users, Clock, TrendingUp, Play, Heart } from 'lucide-react';

// Mock live streams
const liveStreams = [
  {
    id: '1',
    title: 'Deep House Session',
    dj: 'DJ Aurora',
    avatar: null,
    listeners: 342,
    genre: 'House',
    duration: '2:15:30',
    thumbnail: null,
    isFollowing: true,
  },
  {
    id: '2',
    title: 'Techno Warehouse',
    dj: 'Pulse',
    avatar: null,
    listeners: 218,
    genre: 'Techno',
    duration: '1:42:15',
    thumbnail: null,
    isFollowing: false,
  },
  {
    id: '3',
    title: 'Sunday Chill Vibes',
    dj: 'SoulFlow',
    avatar: null,
    listeners: 156,
    genre: 'Lo-Fi',
    duration: '3:05:00',
    thumbnail: null,
    isFollowing: true,
  },
  {
    id: '4',
    title: 'Bass Nation',
    dj: 'BASSMASTER',
    avatar: null,
    listeners: 489,
    genre: 'Dubstep',
    duration: '0:45:20',
    thumbnail: null,
    isFollowing: false,
  },
  {
    id: '5',
    title: 'Trance Journey',
    dj: 'Ethereal',
    avatar: null,
    listeners: 178,
    genre: 'Trance',
    duration: '4:30:00',
    thumbnail: null,
    isFollowing: false,
  },
  {
    id: '6',
    title: 'Hip-Hop Freestyle',
    dj: 'MC Thunder',
    avatar: null,
    listeners: 267,
    genre: 'Hip-Hop',
    duration: '1:20:45',
    thumbnail: null,
    isFollowing: true,
  },
];

const genres = ['All', 'House', 'Techno', 'Lo-Fi', 'Dubstep', 'Trance', 'Hip-Hop', 'Drum & Bass'];

export default function LiveStreamsPage() {
  const totalListeners = liveStreams.reduce((acc, s) => acc + s.listeners, 0);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black">
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-red-500/20 via-transparent to-orange-500/20" />
        <div className="relative max-w-6xl mx-auto px-4 py-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
                  <Radio className="w-7 h-7 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Live Streams</h1>
                <p className="text-gray-400">Tune into live DJ sets from around the world</p>
              </div>
            </div>
            <Link
              href="/studio/broadcast"
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-red-500 to-orange-500 text-white font-medium hover:opacity-90 transition-opacity"
            >
              <Radio className="w-5 h-5" />
              Go Live
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 mt-8">
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
              <div className="flex items-center gap-2 text-gray-400 mb-1">
                <Radio className="w-4 h-4" />
                <span className="text-sm">Live Now</span>
              </div>
              <span className="text-2xl font-bold text-white">{liveStreams.length}</span>
            </div>
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
              <div className="flex items-center gap-2 text-gray-400 mb-1">
                <Users className="w-4 h-4" />
                <span className="text-sm">Total Listeners</span>
              </div>
              <span className="text-2xl font-bold text-white">{totalListeners.toLocaleString()}</span>
            </div>
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
              <div className="flex items-center gap-2 text-gray-400 mb-1">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm">Peak Today</span>
              </div>
              <span className="text-2xl font-bold text-white">2,847</span>
            </div>
          </div>
        </div>
      </div>

      {/* Genre Filter */}
      <div className="max-w-6xl mx-auto px-4 py-4">
        <div className="flex gap-2 overflow-x-auto pb-2">
          {genres.map((genre) => (
            <button
              key={genre}
              className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
                genre === 'All'
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>

      {/* Stream Grid */}
      <div className="max-w-6xl mx-auto px-4 pb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {liveStreams.map((stream) => (
            <div
              key={stream.id}
              className="group rounded-xl bg-gray-900/50 border border-gray-800 hover:border-gray-700 transition-all overflow-hidden cursor-pointer"
            >
              {/* Thumbnail */}
              <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
                    <Radio className="w-10 h-10 text-white" />
                  </div>
                </div>
                {/* Live Badge */}
                <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1 rounded-full bg-red-500 text-white text-sm font-medium">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  LIVE
                </div>
                {/* Listeners */}
                <div className="absolute top-3 right-3 flex items-center gap-1 px-2 py-1 rounded-full bg-black/50 text-white text-sm">
                  <Users className="w-3 h-3" />
                  {stream.listeners}
                </div>
                {/* Play Overlay */}
                <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/40 transition-colors">
                  <div className="w-16 h-16 rounded-full bg-white/0 group-hover:bg-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                    <Play className="w-8 h-8 text-black ml-1" />
                  </div>
                </div>
              </div>

              {/* Info */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h3 className="font-semibold text-white group-hover:text-red-400 transition-colors">
                      {stream.title}
                    </h3>
                    <p className="text-sm text-gray-400">{stream.dj}</p>
                  </div>
                  <button
                    className={`p-2 rounded-full transition-colors ${
                      stream.isFollowing
                        ? 'bg-red-500/20 text-red-400'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    <Heart className={`w-4 h-4 ${stream.isFollowing ? 'fill-current' : ''}`} />
                  </button>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-500">
                  <span className="px-2 py-0.5 rounded-full bg-gray-800 text-gray-300">
                    {stream.genre}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {stream.duration}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
