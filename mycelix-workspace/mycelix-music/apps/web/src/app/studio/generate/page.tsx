// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React from 'react';
import { BeatGenerator, StyleTransfer } from '@/components/generation';

export default function GeneratePage() {
  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">AI Music Generation</h1>
          <p className="text-gray-400">
            Create unique beats, melodies, and transform existing tracks with AI
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <BeatGenerator />
          <StyleTransfer />
        </div>

        {/* Features Info */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Beat Generation</h3>
            <p className="text-sm text-gray-400">
              Generate unique drum patterns and rhythms in various styles from electronic to jazz.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Melody Creation</h3>
            <p className="text-sm text-gray-400">
              Create melodies that follow music theory rules with customizable scales and complexity.
            </p>
          </div>
          <div className="p-4 bg-gray-900 rounded-lg">
            <h3 className="text-white font-semibold mb-2">Style Transfer</h3>
            <p className="text-sm text-gray-400">
              Transform your tracks into different musical styles while preserving the original character.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
