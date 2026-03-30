// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import Link from 'next/link';
import { Disc3, Layers, Sliders, Wifi, Mic, Sparkles } from 'lucide-react';
import { VoiceControl } from '@/components/voice';

const studioFeatures = [
  {
    href: '/studio/dj',
    title: 'DJ Mixer',
    description: 'Professional dual-deck mixing with crossfader, EQ, loops, and beatmatching',
    icon: Disc3,
    color: 'from-purple-500 to-fuchsia-500',
  },
  {
    href: '/studio/stems',
    title: 'Stem Separation',
    description: 'AI-powered vocal, drum, bass, and instrument isolation',
    icon: Layers,
    color: 'from-blue-500 to-cyan-500',
  },
  {
    href: '/studio/effects',
    title: 'Audio Effects',
    description: 'Real-time EQ, reverb, spatial audio, and dynamics processing',
    icon: Sliders,
    color: 'from-green-500 to-emerald-500',
  },
  {
    href: '/studio/broadcast',
    title: 'Go Live',
    description: 'Stream your DJ sets live to listeners worldwide',
    icon: Wifi,
    color: 'from-red-500 to-orange-500',
  },
];

export default function StudioPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-black to-black p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
            Studio
          </h1>
          <p className="text-gray-400 text-lg">
            Professional audio production tools powered by AI and WebAssembly
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {studioFeatures.map((feature) => {
            const Icon = feature.icon;
            return (
              <Link
                key={feature.href}
                href={feature.href}
                className="group relative overflow-hidden rounded-2xl bg-gray-900/50 border border-gray-800 hover:border-gray-700 transition-all duration-300 hover:scale-[1.02]"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
                <div className="p-8">
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6`}>
                    <Icon className="w-7 h-7 text-white" />
                  </div>
                  <h2 className="text-2xl font-semibold text-white mb-2">
                    {feature.title}
                  </h2>
                  <p className="text-gray-400">
                    {feature.description}
                  </p>
                  <div className="mt-6 flex items-center text-sm font-medium text-purple-400 group-hover:text-purple-300">
                    Open Studio
                    <svg className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Voice Control Section */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
              <Mic className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">Voice Control</h2>
              <p className="text-sm text-gray-400">Control the studio hands-free</p>
            </div>
          </div>
          <VoiceControl />
        </div>

        {/* Quick Tips */}
        <div className="rounded-2xl bg-gray-900/50 border border-gray-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Sparkles className="w-5 h-5 text-yellow-400" />
            <h3 className="text-lg font-semibold text-white">Pro Tips</h3>
          </div>
          <ul className="space-y-3 text-gray-400">
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              Use voice commands like "play", "pause", "crossfade", or "loop" for hands-free control
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              The DJ Mixer supports beatmatching - press Sync to align BPMs automatically
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              Stem separation works best with high-quality audio files (WAV or FLAC)
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              Go Live to broadcast your DJ sets and interact with listeners in real-time
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
