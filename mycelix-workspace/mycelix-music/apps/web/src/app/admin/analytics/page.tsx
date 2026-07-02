// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';

interface AnalyticsData {
  overview: {
    totalPlays: number;
    uniqueListeners: number;
    newUsers: number;
    revenue: string;
    playsChange: number;
    listenersChange: number;
    usersChange: number;
    revenueChange: number;
  };
  playsChart: Array<{ date: string; plays: number }>;
  usersChart: Array<{ date: string; users: number }>;
  topSongs: Array<{
    id: string;
    title: string;
    artist: string;
    plays: number;
    change: number;
  }>;
  topArtists: Array<{
    id: string;
    name: string;
    plays: number;
    followers: number;
  }>;
  genreDistribution: Array<{ genre: string; percentage: number }>;
}

type TimeRange = '24h' | '7d' | '30d' | '90d';

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/admin/analytics?range=${timeRange}`);
      if (response.ok) {
        const result = await response.json();
        setData(result);
      }
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-purple-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Platform Analytics</h1>
          <div className="flex gap-2">
            {(['24h', '7d', '30d', '90d'] as TimeRange[]).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  timeRange === range
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Total Plays"
            value={data?.overview.totalPlays || 0}
            change={data?.overview.playsChange || 0}
            format="number"
          />
          <MetricCard
            title="Unique Listeners"
            value={data?.overview.uniqueListeners || 0}
            change={data?.overview.listenersChange || 0}
            format="number"
          />
          <MetricCard
            title="New Users"
            value={data?.overview.newUsers || 0}
            change={data?.overview.usersChange || 0}
            format="number"
          />
          <MetricCard
            title="Revenue"
            value={data?.overview.revenue || '0'}
            change={data?.overview.revenueChange || 0}
            format="currency"
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4">Plays Over Time</h2>
            <SimpleChart data={data?.playsChart || []} dataKey="plays" color="#8b5cf6" />
          </div>
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4">User Growth</h2>
            <SimpleChart data={data?.usersChart || []} dataKey="users" color="#10b981" />
          </div>
        </div>

        {/* Tables */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Top Songs */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4">Top Songs</h2>
            <div className="space-y-3">
              {data?.topSongs.map((song, index) => (
                <div
                  key={song.id}
                  className="flex items-center justify-between py-2 border-b border-gray-700 last:border-0"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-gray-500 w-6">{index + 1}</span>
                    <div>
                      <p className="font-medium">{song.title}</p>
                      <p className="text-gray-400 text-sm">{song.artist}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{song.plays.toLocaleString()}</p>
                    <p className={`text-sm ${song.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {song.change >= 0 ? '+' : ''}{song.change}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Artists */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4">Top Artists</h2>
            <div className="space-y-3">
              {data?.topArtists.map((artist, index) => (
                <div
                  key={artist.id}
                  className="flex items-center justify-between py-2 border-b border-gray-700 last:border-0"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-gray-500 w-6">{index + 1}</span>
                    <p className="font-medium">{artist.name}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{artist.plays.toLocaleString()} plays</p>
                    <p className="text-gray-400 text-sm">{artist.followers.toLocaleString()} followers</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Genre Distribution */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Genre Distribution</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {data?.genreDistribution.map((item) => (
              <div key={item.genre} className="text-center">
                <div className="relative w-24 h-24 mx-auto mb-2">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#374151"
                      strokeWidth="8"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#8b5cf6"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${item.percentage * 2.83} 283`}
                      transform="rotate(-90 50 50)"
                    />
                    <text
                      x="50"
                      y="55"
                      textAnchor="middle"
                      className="fill-white text-lg font-bold"
                    >
                      {item.percentage}%
                    </text>
                  </svg>
                </div>
                <p className="text-gray-300 text-sm">{item.genre}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  change,
  format,
}: {
  title: string;
  value: number | string;
  change: number;
  format: 'number' | 'currency';
}) {
  const formattedValue = format === 'number' && typeof value === 'number'
    ? value.toLocaleString()
    : value;

  return (
    <div className="bg-gray-800 rounded-xl p-6">
      <p className="text-gray-400 text-sm">{title}</p>
      <p className="text-3xl font-bold mt-1">{formattedValue}</p>
      <div className={`flex items-center gap-1 mt-2 ${
        change >= 0 ? 'text-green-400' : 'text-red-400'
      }`}>
        <span>{change >= 0 ? '↑' : '↓'}</span>
        <span className="text-sm">{Math.abs(change)}% vs previous period</span>
      </div>
    </div>
  );
}

function SimpleChart({
  data,
  dataKey,
  color,
}: {
  data: Array<{ date: string; [key: string]: unknown }>;
  dataKey: string;
  color: string;
}) {
  if (data.length === 0) {
    return <div className="h-48 flex items-center justify-center text-gray-500">No data</div>;
  }

  const values = data.map(d => d[dataKey] as number);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - ((d[dataKey] as number - min) / range) * 80 - 10;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="h-48">
      <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
        {data.map((d, i) => {
          const x = (i / (data.length - 1)) * 100;
          const y = 100 - ((d[dataKey] as number - min) / range) * 80 - 10;
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="1.5"
              fill={color}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>{data[0]?.date}</span>
        <span>{data[data.length - 1]?.date}</span>
      </div>
    </div>
  );
}
