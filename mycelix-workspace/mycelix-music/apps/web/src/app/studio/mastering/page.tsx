// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React from 'react';
import { MasteringPanel } from '@/components/intelligence';

export default function MasteringPage() {
  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">AI Mastering</h1>
          <p className="text-gray-400">
            Professional mastering with intelligent analysis and platform-optimized output
          </p>
        </div>

        {/* Main Panel */}
        <MasteringPanel />

        {/* Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Intelligent Analysis</h3>
            <p className="text-sm text-gray-400">
              Automatic detection of audio issues including clipping, phase problems,
              and spectral imbalances with suggestions for improvement.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Platform Optimization</h3>
            <p className="text-sm text-gray-400">
              Optimize your masters for Spotify, Apple Music, YouTube, and other
              platforms with correct loudness targets and true peak limiting.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Reference Matching</h3>
            <p className="text-sm text-gray-400">
              Compare your track to reference masters and get suggestions to match
              their tonal balance and loudness characteristics.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Mastering Presets</h3>
            <p className="text-sm text-gray-400">
              Choose from genre-specific presets or customize every aspect of the
              mastering chain including EQ, compression, and saturation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
