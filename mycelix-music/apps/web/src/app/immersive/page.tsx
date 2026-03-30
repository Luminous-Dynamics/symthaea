// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';
import { SpatialVenue } from '@/components/immersive/SpatialVenue';
import { AudioVisualizer3D } from '@/components/immersive/AudioVisualizer3D';
import { Headphones, Sparkles, Box, Music } from 'lucide-react';

export default function ImmersivePage() {
  const [activeTab, setActiveTab] = useState<'spatial' | 'visualizer'>('spatial');

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2">Immersive Audio</h1>
          <p className="text-gray-400">
            Experience music in 3D space with spatial audio and reactive visuals
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex">
          <button
            onClick={() => setActiveTab('spatial')}
            className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
              activeTab === 'spatial'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <Headphones className="w-5 h-5" />
            3D Spatial Venue
          </button>
          <button
            onClick={() => setActiveTab('visualizer')}
            className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
              activeTab === 'visualizer'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <Sparkles className="w-5 h-5" />
            Audio Visualizer
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          {activeTab === 'spatial' && (
            <div className="space-y-6">
              <SpatialVenue className="w-full" />

              {/* Info Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-900 rounded-lg">
                  <Box className="w-8 h-8 text-green-400 mb-3" />
                  <h3 className="text-white font-semibold mb-2">HRTF Processing</h3>
                  <p className="text-sm text-gray-400">
                    Head-Related Transfer Function for realistic 3D audio positioning.
                  </p>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <Music className="w-8 h-8 text-blue-400 mb-3" />
                  <h3 className="text-white font-semibold mb-2">Room Acoustics</h3>
                  <p className="text-sm text-gray-400">
                    Realistic reverb and room simulation for immersive experiences.
                  </p>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <Headphones className="w-8 h-8 text-purple-400 mb-3" />
                  <h3 className="text-white font-semibold mb-2">Binaural Audio</h3>
                  <p className="text-sm text-gray-400">
                    True 3D sound that works with any stereo headphones.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'visualizer' && (
            <div className="space-y-6">
              <AudioVisualizer3D className="w-full h-[600px]" />

              {/* Info */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-900 rounded-lg">
                  <h3 className="text-white font-semibold mb-2">Particle Systems</h3>
                  <p className="text-sm text-gray-400">
                    Thousands of particles react to bass, mids, and treble frequencies.
                  </p>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <h3 className="text-white font-semibold mb-2">Beat Detection</h3>
                  <p className="text-sm text-gray-400">
                    Real-time beat detection triggers visual effects and transitions.
                  </p>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <h3 className="text-white font-semibold mb-2">Customizable</h3>
                  <p className="text-sm text-gray-400">
                    Multiple color schemes and visualization modes to choose from.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
