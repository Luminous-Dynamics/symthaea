// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { formatNumber, formatCurrency, formatRelativeTime } from '@/lib/utils';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Users,
  DollarSign,
  Music2,
  Upload,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Activity,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import Link from 'next/link';
import { redirect } from 'next/navigation';

type Period = '24h' | '7d' | '30d' | '90d' | '1y';

export default function DashboardPage() {
  const { authenticated, user } = useAuth();
  const [period, setPeriod] = useState<Period>('30d');

  // Redirect if not authenticated or not an artist
  if (!authenticated) {
    redirect('/');
  }

  const { data: dashboard, isLoading } = useQuery({
    queryKey: ['dashboard', period],
    queryFn: () => api.getDashboard(period),
    enabled: authenticated,
  });

  const { data: songs } = useQuery({
    queryKey: ['artistSongs'],
    queryFn: () => api.getArtistSongs(),
    enabled: authenticated,
  });

  const periods: { value: Period; label: string }[] = [
    { value: '24h', label: '24 hours' },
    { value: '7d', label: '7 days' },
    { value: '30d', label: '30 days' },
    { value: '90d', label: '90 days' },
    { value: '1y', label: '1 year' },
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-muted-foreground">
                Welcome back, {user?.displayName}
              </p>
            </div>

            <div className="flex items-center gap-4">
              {/* Period Selector */}
              <div className="flex items-center gap-2 bg-white/5 rounded-lg p-1">
                {periods.map((p) => (
                  <button
                    key={p.value}
                    onClick={() => setPeriod(p.value)}
                    className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      period === p.value
                        ? 'bg-white text-black'
                        : 'text-muted-foreground hover:text-white'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Upload Button */}
              <Link
                href="/dashboard/upload"
                className="flex items-center gap-2 px-4 py-2 bg-primary text-black rounded-lg font-medium hover:bg-primary/90 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Upload Track
              </Link>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-4 gap-4 mb-8">
            <StatCard
              title="Total Streams"
              value={formatNumber(dashboard?.streams.totalStreams || 0)}
              change={12.5}
              icon={<BarChart3 className="w-5 h-5" />}
            />
            <StatCard
              title="Unique Listeners"
              value={formatNumber(dashboard?.streams.uniqueListeners || 0)}
              change={8.2}
              icon={<Users className="w-5 h-5" />}
            />
            <StatCard
              title="Total Revenue"
              value={formatCurrency(dashboard?.revenue.totalRevenue || 0)}
              change={15.3}
              icon={<DollarSign className="w-5 h-5" />}
            />
            <StatCard
              title="Followers"
              value={formatNumber(dashboard?.engagement.followers || 0)}
              change={dashboard?.engagement.followerGrowthPercent || 0}
              icon={<Users className="w-5 h-5" />}
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            {/* Streams Chart */}
            <div className="p-6 bg-white/5 rounded-xl">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-semibold">Streams Over Time</h3>
                <Activity className="w-5 h-5 text-muted-foreground" />
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={dashboard?.streamHistory || []}>
                    <defs>
                      <linearGradient id="streamGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="date"
                      tickFormatter={(val) => new Date(val).toLocaleDateString('en', { month: 'short', day: 'numeric' })}
                      stroke="#666"
                      fontSize={12}
                    />
                    <YAxis stroke="#666" fontSize={12} tickFormatter={formatNumber} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1a1a',
                        border: '1px solid #333',
                        borderRadius: '8px',
                      }}
                      labelFormatter={(val) => new Date(val).toLocaleDateString()}
                      formatter={(val: number) => [formatNumber(val), 'Streams']}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      fill="url(#streamGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Revenue Chart */}
            <div className="p-6 bg-white/5 rounded-xl">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-semibold">Revenue</h3>
                <DollarSign className="w-5 h-5 text-muted-foreground" />
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                  <div>
                    <p className="text-sm text-muted-foreground">Streaming Revenue</p>
                    <p className="text-2xl font-bold">
                      {formatCurrency(dashboard?.revenue.streamingRevenue || 0)}
                    </p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                    <Music2 className="w-6 h-6 text-green-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                  <div>
                    <p className="text-sm text-muted-foreground">Pending Payout</p>
                    <p className="text-2xl font-bold">
                      {formatCurrency(dashboard?.revenue.pendingPayout || 0)}
                    </p>
                  </div>
                  <Link
                    href="/dashboard/payouts"
                    className="px-4 py-2 bg-primary text-black rounded-lg text-sm font-medium"
                  >
                    Withdraw
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Top Songs */}
          <div className="p-6 bg-white/5 rounded-xl mb-8">
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-semibold">Top Performing Tracks</h3>
              <Link
                href="/dashboard/songs"
                className="text-sm text-primary hover:underline"
              >
                View all
              </Link>
            </div>
            <div className="space-y-2">
              {dashboard?.topSongs?.slice(0, 5).map((song, index) => (
                <div
                  key={song.songId}
                  className="flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <span className="w-6 text-center text-muted-foreground font-medium">
                    {index + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{song.title}</p>
                    <p className="text-sm text-muted-foreground">
                      {formatNumber(song.streams)} streams
                    </p>
                  </div>
                  <div className="flex items-center gap-1 text-sm">
                    {song.trend === 'up' ? (
                      <>
                        <ArrowUpRight className="w-4 h-4 text-green-500" />
                        <span className="text-green-500">+12%</span>
                      </>
                    ) : song.trend === 'down' ? (
                      <>
                        <ArrowDownRight className="w-4 h-4 text-red-500" />
                        <span className="text-red-500">-5%</span>
                      </>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="p-6 bg-white/5 rounded-xl">
            <h3 className="font-semibold mb-6">Recent Activity</h3>
            <div className="space-y-4">
              <ActivityItem
                icon={<Users className="w-4 h-4" />}
                title="New follower"
                description="Someone started following you"
                time="2 min ago"
              />
              <ActivityItem
                icon={<Music2 className="w-4 h-4" />}
                title="Track milestone"
                description="'Midnight Dreams' reached 10K streams"
                time="1 hour ago"
              />
              <ActivityItem
                icon={<DollarSign className="w-4 h-4" />}
                title="Payout processed"
                description="$125.00 has been sent to your wallet"
                time="Yesterday"
              />
            </div>
          </div>
        </div>
      </main>

      <Player />
    </div>
  );
}

// Stat Card Component
function StatCard({
  title,
  value,
  change,
  icon,
}: {
  title: string;
  value: string;
  change: number;
  icon: React.ReactNode;
}) {
  const isPositive = change >= 0;

  return (
    <div className="p-6 bg-white/5 rounded-xl">
      <div className="flex items-center justify-between mb-4">
        <span className="text-muted-foreground text-sm">{title}</span>
        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
          {icon}
        </div>
      </div>
      <p className="text-3xl font-bold mb-2">{value}</p>
      <div className="flex items-center gap-1 text-sm">
        {isPositive ? (
          <TrendingUp className="w-4 h-4 text-green-500" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-500" />
        )}
        <span className={isPositive ? 'text-green-500' : 'text-red-500'}>
          {isPositive ? '+' : ''}
          {change.toFixed(1)}%
        </span>
        <span className="text-muted-foreground">vs last period</span>
      </div>
    </div>
  );
}

// Activity Item Component
function ActivityItem({
  icon,
  title,
  description,
  time,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  time: string;
}) {
  return (
    <div className="flex items-start gap-4">
      <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <span className="text-sm text-muted-foreground flex-shrink-0">{time}</span>
    </div>
  );
}
