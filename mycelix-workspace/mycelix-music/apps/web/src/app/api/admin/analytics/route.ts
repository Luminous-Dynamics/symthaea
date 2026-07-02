// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const range = searchParams.get('range') || '7d';

  // Generate mock data based on time range
  const days = range === '24h' ? 24 : range === '7d' ? 7 : range === '30d' ? 30 : 90;
  const isHourly = range === '24h';

  const playsChart = Array.from({ length: days }, (_, i) => {
    const date = new Date();
    if (isHourly) {
      date.setHours(date.getHours() - (days - i));
    } else {
      date.setDate(date.getDate() - (days - i));
    }
    return {
      date: isHourly
        ? date.toLocaleTimeString('en-US', { hour: '2-digit' })
        : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      plays: Math.floor(Math.random() * 50000 + 30000),
    };
  });

  const usersChart = Array.from({ length: days }, (_, i) => {
    const date = new Date();
    if (isHourly) {
      date.setHours(date.getHours() - (days - i));
    } else {
      date.setDate(date.getDate() - (days - i));
    }
    return {
      date: isHourly
        ? date.toLocaleTimeString('en-US', { hour: '2-digit' })
        : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      users: Math.floor(Math.random() * 500 + 100),
    };
  });

  const analytics = {
    overview: {
      totalPlays: 1250000 + Math.floor(Math.random() * 100000),
      uniqueListeners: 89000 + Math.floor(Math.random() * 10000),
      newUsers: 1500 + Math.floor(Math.random() * 500),
      revenue: `${(45.5 + Math.random() * 10).toFixed(2)} ETH`,
      playsChange: Math.floor(Math.random() * 30 - 5),
      listenersChange: Math.floor(Math.random() * 25 - 3),
      usersChange: Math.floor(Math.random() * 40),
      revenueChange: Math.floor(Math.random() * 35 - 5),
    },
    playsChart,
    usersChart,
    topSongs: [
      { id: '1', title: 'Digital Dreams', artist: 'CryptoBeats', plays: 125000, change: 15 },
      { id: '2', title: 'Neon Nights', artist: 'SynthMaster', plays: 98000, change: 8 },
      { id: '3', title: 'Chain Reaction', artist: 'BlockBeat', plays: 87000, change: -3 },
      { id: '4', title: 'Token of Love', artist: 'Web3 Romance', plays: 76000, change: 22 },
      { id: '5', title: 'Decentralized', artist: 'DeFi Sounds', plays: 65000, change: 5 },
    ],
    topArtists: [
      { id: '1', name: 'CryptoBeats', plays: 1250000, followers: 45000 },
      { id: '2', name: 'SynthMaster', plays: 980000, followers: 38000 },
      { id: '3', name: 'BlockBeat', plays: 870000, followers: 32000 },
      { id: '4', name: 'Web3 Romance', plays: 760000, followers: 28000 },
      { id: '5', name: 'DeFi Sounds', plays: 650000, followers: 24000 },
    ],
    genreDistribution: [
      { genre: 'Electronic', percentage: 35 },
      { genre: 'Hip Hop', percentage: 22 },
      { genre: 'Pop', percentage: 18 },
      { genre: 'Rock', percentage: 12 },
      { genre: 'R&B', percentage: 8 },
      { genre: 'Other', percentage: 5 },
    ],
  };

  return NextResponse.json(analytics);
}
