// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { DiscoveryPanel } from '@/components/discovery';
import { Sparkles, Compass, Music, Mic2, Zap } from 'lucide-react';

const discoverModes = [
  { id: 'smart', label: 'Smart Discovery', icon: Sparkles, description: 'AI-powered recommendations based on your taste' },
  { id: 'explore', label: 'Explore', icon: Compass, description: 'Browse by genre, mood, and energy' },
  { id: 'similar', label: 'Similar Tracks', icon: Music, description: 'Find tracks similar to your favorites' },
  { id: 'artists', label: 'Artists', icon: Mic2, description: 'Discover new artists you might like' },
];

export default function DiscoverPage() {
  const [activeMode, setActiveMode] = useState('smart');

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 via-transparent to-fuchsia-500/20" />
        <div className="relative max-w-6xl mx-auto px-4 py-16">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-fuchsia-500 flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Discover</h1>
              <p className="text-gray-400">Find your next favorite track</p>
            </div>
          </div>

          {/* Mode Selector */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
            {discoverModes.map((mode) => {
              const Icon = mode.icon;
              const isActive = activeMode === mode.id;

              return (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`relative p-4 rounded-xl border transition-all text-left ${
                    isActive
                      ? 'bg-purple-500/20 border-purple-500/50'
                      : 'bg-gray-900/50 border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <Icon className={`w-6 h-6 mb-2 ${isActive ? 'text-purple-400' : 'text-gray-400'}`} />
                  <h3 className={`font-medium ${isActive ? 'text-white' : 'text-gray-300'}`}>
                    {mode.label}
                  </h3>
                  <p className="text-xs text-gray-500 mt-1">{mode.description}</p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Discovery Panel */}
      <div className="max-w-6xl mx-auto px-4 pb-8">
        <DiscoveryPanel />
      </div>
    </div>
  );
}
