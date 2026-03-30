// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { Sparkles, TrendingUp, TrendingDown, Clock, Info, Vote, ChevronRight } from 'lucide-react';
import Link from 'next/link';

interface SporeBalanceProps {
  balance: number;
  tier: number;
  tierName: string;
  decayRate: number;
  lastActivity: Date;
  recentEarnings: {
    activity: string;
    amount: number;
    date: Date;
  }[];
}

const tierColors = [
  'from-gray-500 to-gray-600',      // Tier 0
  'from-green-500 to-emerald-500',  // Tier 1 - Sprout
  'from-teal-500 to-cyan-500',      // Tier 2 - Mycelium
  'from-purple-500 to-pink-500',    // Tier 3 - Network
  'from-amber-500 to-orange-500',   // Tier 4 - Grove
  'from-rose-500 to-red-500',       // Tier 5 - Ancient
];

const tierNames = ['Dormant', 'Sprout', 'Mycelium', 'Network', 'Grove', 'Ancient'];

export function SporeBalance({
  balance,
  tier,
  tierName,
  decayRate,
  lastActivity,
  recentEarnings,
}: SporeBalanceProps) {
  const [showDetails, setShowDetails] = useState(false);

  // Calculate days until significant decay
  const daysSinceActivity = Math.floor(
    (Date.now() - lastActivity.getTime()) / (1000 * 60 * 60 * 24)
  );
  const decayWarning = daysSinceActivity > 14;

  // Calculate progress to next tier
  const tierThresholds = [0, 100, 500, 2000, 10000, 50000];
  const nextTier = Math.min(tier + 1, 5);
  const currentThreshold = tierThresholds[tier];
  const nextThreshold = tierThresholds[nextTier];
  const progressToNext = ((balance - currentThreshold) / (nextThreshold - currentThreshold)) * 100;

  return (
    <div className="bg-white/5 rounded-2xl overflow-hidden">
      {/* Header with balance */}
      <div className={`p-6 bg-gradient-to-r ${tierColors[tier]} relative overflow-hidden`}>
        <div className="absolute inset-0 bg-black/20" />
        <div className="relative">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              <span className="font-medium">Spore Balance</span>
            </div>
            <span className="px-2 py-1 bg-white/20 rounded-full text-xs font-medium">
              {tierName}
            </span>
          </div>

          <div className="text-4xl font-bold mb-2">
            {balance.toLocaleString()}
          </div>

          {/* Progress to next tier */}
          {tier < 5 && (
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="opacity-80">Progress to {tierNames[nextTier]}</span>
                <span>{Math.min(progressToNext, 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-white/20 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all"
                  style={{ width: `${Math.min(progressToNext, 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Decay Warning */}
      {decayWarning && (
        <div className="px-6 py-3 bg-amber-500/10 border-b border-white/10 flex items-center gap-3">
          <Clock className="w-4 h-4 text-amber-400" />
          <span className="text-sm text-amber-400">
            {daysSinceActivity} days inactive - Spores will begin decaying soon
          </span>
        </div>
      )}

      {/* Stats */}
      <div className="p-6 border-b border-white/10">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold">{tier}</p>
            <p className="text-xs text-muted-foreground">Current Tier</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold">{decayRate}%</p>
            <p className="text-xs text-muted-foreground">Monthly Decay</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold">
              {Math.floor(balance * 0.1)}
            </p>
            <p className="text-xs text-muted-foreground">Voting Power</p>
          </div>
        </div>
      </div>

      {/* Recent Earnings */}
      <div className="p-6">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center justify-between w-full mb-4"
        >
          <span className="font-medium">Recent Activity</span>
          <ChevronRight className={`w-4 h-4 text-muted-foreground transition-transform ${showDetails ? 'rotate-90' : ''}`} />
        </button>

        {showDetails && (
          <div className="space-y-3">
            {recentEarnings.map((earning, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-400" />
                  <span className="text-muted-foreground">{earning.activity}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-green-400">+{earning.amount}</span>
                  <Sparkles className="w-3 h-3 text-purple-400" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="p-6 bg-white/5 border-t border-white/10 grid grid-cols-2 gap-3">
        <Link
          href="/governance"
          className="flex items-center justify-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
        >
          <Vote className="w-4 h-4" />
          <span className="text-sm font-medium">Vote</span>
        </Link>
        <Link
          href="/spores/earn"
          className="flex items-center justify-center gap-2 px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
        >
          <TrendingUp className="w-4 h-4" />
          <span className="text-sm font-medium">Earn More</span>
        </Link>
      </div>

      {/* Info Note */}
      <div className="px-6 py-4 bg-gradient-to-r from-purple-500/5 to-pink-500/5">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">
            Spores are earned through genuine participation - listening, creating playlists,
            supporting artists, and community contribution. They cannot be bought, sold, or
            transferred. Stay active to maintain your balance!
          </p>
        </div>
      </div>
    </div>
  );
}

// Simple inline display for headers/nav
export function SporeBalanceInline({ balance, tier }: { balance: number; tier: number }) {
  return (
    <Link
      href="/profile/spores"
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r ${tierColors[tier]} hover:opacity-90 transition-opacity`}
    >
      <Sparkles className="w-4 h-4" />
      <span className="font-medium text-sm">{balance.toLocaleString()}</span>
    </Link>
  );
}
