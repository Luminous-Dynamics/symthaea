// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect } from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { ContextualAudio } from '@/components/context/ContextualAudio';
import { ContextualMix } from '@/components/context/ContextualMix';
import { useContextDetection } from '@/hooks/useContextDetection';
import {
  Sparkles,
  Clock,
  CloudSun,
  Brain,
  Music2,
  Sliders,
  History,
} from 'lucide-react';

export default function ContextPage() {
  // Initialize context detection
  useContextDetection();

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500">
            <div className="absolute inset-0 bg-black/20" />

            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="absolute rounded-full bg-white/10 animate-pulse"
                  style={{
                    width: `${20 + Math.random() * 40}px`,
                    height: `${20 + Math.random() * 40}px`,
                    left: `${Math.random() * 100}%`,
                    top: `${Math.random() * 100}%`,
                    animationDelay: `${Math.random() * 2}s`,
                    animationDuration: `${2 + Math.random() * 2}s`,
                  }}
                />
              ))}
            </div>

            <div className="relative p-8 md:p-12">
              <div className="flex items-center gap-3 mb-4">
                <Sparkles className="w-8 h-8" />
                <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                  Contextual Audio
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Music That Knows You
              </h1>
              <p className="text-lg opacity-80 max-w-xl">
                Experience adaptive soundscapes that respond to your time, weather,
                mood, and activity. The perfect soundtrack for every moment.
              </p>
            </div>
          </div>

          {/* Current Contextual Mix */}
          <section className="mb-10">
            <ContextualMix />
          </section>

          {/* Context Controls */}
          <section className="mb-10">
            <ContextualAudio />
          </section>

          {/* How It Works */}
          <section className="mb-10">
            <h2 className="text-2xl font-bold mb-6">How It Works</h2>
            <div className="grid grid-cols-4 gap-4">
              <FeatureCard
                icon={Clock}
                title="Time Awareness"
                description="Music adapts to dawn, morning, afternoon, evening, and night rhythms"
                color="from-yellow-500 to-orange-500"
              />
              <FeatureCard
                icon={CloudSun}
                title="Weather Responsive"
                description="Rainy day jazz, sunny upbeat pop, or stormy ambient atmospheres"
                color="from-blue-500 to-cyan-500"
              />
              <FeatureCard
                icon={Brain}
                title="Activity Detection"
                description="Working, exercising, relaxing - each moment gets its own sound"
                color="from-green-500 to-emerald-500"
              />
              <FeatureCard
                icon={Music2}
                title="Mood Matching"
                description="Express how you feel and discover music that resonates"
                color="from-purple-500 to-pink-500"
              />
            </div>
          </section>

          {/* Context History */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <History className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold">Recent Contexts</h2>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <ContextHistoryCard
                icon="🌅"
                label="Morning Energy"
                time="Today, 7:30 AM"
                tracks={12}
              />
              <ContextHistoryCard
                icon="🎯"
                label="Deep Focus"
                time="Yesterday, 2:15 PM"
                tracks={24}
              />
              <ContextHistoryCard
                icon="🌧️"
                label="Rainy Day"
                time="Yesterday, 6:45 PM"
                tracks={18}
              />
            </div>
          </section>

          {/* Create Custom Profile CTA */}
          <section className="mb-10">
            <div className="p-8 rounded-2xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-white/10">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <Sliders className="w-7 h-7" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-1">Create Custom Context</h3>
                    <p className="text-muted-foreground">
                      Design your own contextual profile with specific tempo, energy, and genre preferences.
                    </p>
                  </div>
                </div>
                <button className="px-6 py-3 bg-purple-500 rounded-lg font-medium hover:bg-purple-600 transition-colors">
                  Create Profile
                </button>
              </div>
            </div>
          </section>
        </div>
      </main>

      <Player />
    </div>
  );
}

interface FeatureCardProps {
  icon: typeof Clock;
  title: string;
  description: string;
  color: string;
}

function FeatureCard({ icon: Icon, title, description, color }: FeatureCardProps) {
  return (
    <div className="bg-white/5 rounded-xl p-6 hover:bg-white/10 transition-colors group">
      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
        <Icon className="w-6 h-6" />
      </div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}

interface ContextHistoryCardProps {
  icon: string;
  label: string;
  time: string;
  tracks: number;
}

function ContextHistoryCard({ icon, label, time, tracks }: ContextHistoryCardProps) {
  return (
    <button className="bg-white/5 rounded-xl p-4 text-left hover:bg-white/10 transition-colors">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-2xl">{icon}</span>
        <div>
          <p className="font-medium">{label}</p>
          <p className="text-sm text-muted-foreground">{time}</p>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">{tracks} tracks played</p>
    </button>
  );
}
