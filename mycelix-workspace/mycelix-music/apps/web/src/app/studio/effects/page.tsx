// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { EffectsPanel } from '@/components/effects';
import Link from 'next/link';
import { ArrowLeft, Save, RotateCcw } from 'lucide-react';
import { useState } from 'react';

export default function AudioEffectsPage() {
  const [presetName, setPresetName] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);

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
            <h1 className="text-xl font-semibold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
              Audio Effects
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSaveModal(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition-colors"
            >
              <Save className="w-4 h-4" />
              Save Preset
            </button>
          </div>
        </div>
      </div>

      {/* Preset Bar */}
      <div className="max-w-4xl mx-auto px-4 py-4">
        <div className="flex items-center gap-4 overflow-x-auto pb-2">
          <span className="text-sm text-gray-500 whitespace-nowrap">Presets:</span>
          {['Flat', 'Bass Boost', 'Vocal Clarity', 'Live Room', 'Wide Stereo', 'Radio Ready'].map((preset) => (
            <button
              key={preset}
              className="px-4 py-2 rounded-full text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors whitespace-nowrap"
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Effects Panel */}
      <div className="max-w-4xl mx-auto px-4 pb-8">
        <EffectsPanel />
      </div>

      {/* Effect Chain Info */}
      <div className="max-w-4xl mx-auto px-4 pb-8">
        <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Signal Chain</h3>
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            <div className="flex items-center">
              <div className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 text-sm font-medium">
                Input
              </div>
              <div className="w-8 h-px bg-gray-700" />
            </div>
            <div className="flex items-center">
              <div className="px-4 py-2 rounded-lg bg-purple-500/20 text-purple-400 text-sm font-medium">
                EQ
              </div>
              <div className="w-8 h-px bg-gray-700" />
            </div>
            <div className="flex items-center">
              <div className="px-4 py-2 rounded-lg bg-green-500/20 text-green-400 text-sm font-medium">
                Compressor
              </div>
              <div className="w-8 h-px bg-gray-700" />
            </div>
            <div className="flex items-center">
              <div className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400 text-sm font-medium">
                Reverb
              </div>
              <div className="w-8 h-px bg-gray-700" />
            </div>
            <div className="flex items-center">
              <div className="px-4 py-2 rounded-lg bg-orange-500/20 text-orange-400 text-sm font-medium">
                Limiter
              </div>
              <div className="w-8 h-px bg-gray-700" />
            </div>
            <div className="px-4 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 text-sm font-medium">
              Output
            </div>
          </div>
          <p className="mt-4 text-sm text-gray-400">
            Effects are processed in order from left to right. Drag to reorder (coming soon).
          </p>
        </div>
      </div>

      {/* Save Preset Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Save Preset</h3>
            <input
              type="text"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="Preset name..."
              className="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowSaveModal(false)}
                className="flex-1 px-4 py-2 rounded-lg border border-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // Save preset logic
                  setShowSaveModal(false);
                  setPresetName('');
                }}
                className="flex-1 px-4 py-2 rounded-lg bg-purple-500 text-white font-medium hover:bg-purple-600 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
