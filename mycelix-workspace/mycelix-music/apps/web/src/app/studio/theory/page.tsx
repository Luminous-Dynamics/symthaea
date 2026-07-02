// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React from 'react';
import { TheoryAssistant } from '@/components/intelligence';

export default function TheoryPage() {
  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Music Theory Assistant</h1>
          <p className="text-gray-400">
            Analyze audio for key, chords, and progressions with educational insights
          </p>
        </div>

        {/* Main Panel */}
        <TheoryAssistant />

        {/* Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Key Detection</h3>
            <p className="text-sm text-gray-400">
              Automatic detection of the musical key using Krumhansl-Kessler algorithm
              with confidence scores and alternative key suggestions.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Chord Recognition</h3>
            <p className="text-sm text-gray-400">
              Identify chords throughout your track with Roman numeral analysis
              showing their function within the key.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Scale Explorer</h3>
            <p className="text-sm text-gray-400">
              Explore all scales that work with your key including modes,
              pentatonics, and exotic scales with related chords.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Composition Suggestions</h3>
            <p className="text-sm text-gray-400">
              Get intelligent suggestions for chord progressions, modulations,
              and harmonic variations based on music theory.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
