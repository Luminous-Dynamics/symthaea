// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { DJMixer } from '@/components/dj';
import { VoiceControl } from '@/components/voice';
import Link from 'next/link';
import { ArrowLeft, Mic } from 'lucide-react';
import { useState } from 'react';

export default function DJMixerPage() {
  const [showVoice, setShowVoice] = useState(false);

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
            <h1 className="text-xl font-semibold bg-gradient-to-r from-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
              DJ Mixer
            </h1>
          </div>
          <button
            onClick={() => setShowVoice(!showVoice)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              showVoice
                ? 'bg-purple-500/20 text-purple-400'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            <Mic className="w-4 h-4" />
            Voice Control
          </button>
        </div>
      </div>

      {/* Voice Control Panel */}
      {showVoice && (
        <div className="max-w-7xl mx-auto px-4 py-4">
          <VoiceControl compact={false} />
        </div>
      )}

      {/* DJ Mixer */}
      <div className="p-4">
        <DJMixer className="max-w-7xl mx-auto" />
      </div>
    </div>
  );
}
