// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { BroadcastStudio } from '@/components/broadcast';
import Link from 'next/link';
import { ArrowLeft, Radio, Users, Clock, TrendingUp } from 'lucide-react';
import { useState, useEffect } from 'react';

// Mock live streams for the discovery section
const liveStreams = [
  { id: '1', title: 'Late Night House Session', dj: 'DJ Aurora', listeners: 234, genre: 'House', duration: '1:42:30' },
  { id: '2', title: 'Techno Warehouse', dj: 'Pulse', listeners: 156, genre: 'Techno', duration: '0:58:15' },
  { id: '3', title: 'Chill Sunday Vibes', dj: 'SoulFlow', listeners: 89, genre: 'Lo-Fi', duration: '2:15:00' },
];

export default function BroadcastPage() {
  const [mode, setMode] = useState<'host' | 'discover'>('host');

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black">
      {/* Header */}
      <div className="sticky top-0 z-50 bg-black/80 backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/studio"
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Studio</span>
            </Link>
            <div className="w-px h-6 bg-gray-800" />
            <h1 className="text-xl font-semibold bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
              Live Broadcast
            </h1>
          </div>
          <div className="flex items-center bg-gray-800 rounded-lg p-1">
            <button
              onClick={() => setMode('host')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === 'host'
                  ? 'bg-red-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Go Live
            </button>
            <button
              onClick={() => setMode('discover')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === 'discover'
                  ? 'bg-red-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Discover
            </button>
          </div>
        </div>
      </div>

      {mode === 'host' ? (
        /* Broadcast Studio */
        <div className="max-w-4xl mx-auto px-4 py-8">
          <BroadcastStudio />
        </div>
      ) : (
        /* Discover Live Streams */
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <Radio className="w-5 h-5 text-red-400" />
                <span className="text-sm text-gray-400">Live Now</span>
              </div>
              <span className="text-3xl font-bold text-white">{liveStreams.length}</span>
            </div>
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <Users className="w-5 h-5 text-blue-400" />
                <span className="text-sm text-gray-400">Total Listeners</span>
              </div>
              <span className="text-3xl font-bold text-white">
                {liveStreams.reduce((acc, s) => acc + s.listeners, 0)}
              </span>
            </div>
            <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
              <div className="flex items-center gap-3 mb-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <span className="text-sm text-gray-400">Trending Genre</span>
              </div>
              <span className="text-3xl font-bold text-white">House</span>
            </div>
          </div>

          {/* Live Streams List */}
          <h2 className="text-xl font-semibold text-white mb-4">Live Streams</h2>
          <div className="space-y-4">
            {liveStreams.map((stream) => (
              <div
                key={stream.id}
                className="rounded-xl bg-gray-900/50 border border-gray-800 p-6 hover:border-gray-700 transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
                        <Radio className="w-8 h-8 text-white" />
                      </div>
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full animate-pulse" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{stream.title}</h3>
                      <p className="text-gray-400">{stream.dj}</p>
                      <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Users className="w-4 h-4" />
                          {stream.listeners}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {stream.duration}
                        </span>
                        <span className="px-2 py-0.5 rounded-full bg-gray-800 text-gray-300">
                          {stream.genre}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button className="px-6 py-3 rounded-lg bg-red-500 text-white font-medium hover:bg-red-600 transition-colors">
                    Join
                  </button>
                </div>
              </div>
            ))}
          </div>

          {liveStreams.length === 0 && (
            <div className="text-center py-12">
              <Radio className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No live streams right now</p>
              <p className="text-gray-500 text-sm mt-1">Be the first to go live!</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
