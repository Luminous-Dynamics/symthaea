// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { formatNumber, formatDuration } from '@/lib/utils';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import {
  BarChart3,
  Globe,
  Clock,
  Users,
  TrendingUp,
  Music2,
  Headphones,
  Share2,
  Heart,
  ListMusic,
  Smartphone,
  Monitor,
  Tablet,
  ChevronLeft,
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
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import Link from 'next/link';
import { redirect } from 'next/navigation';

type Period = '7d' | '30d' | '90d' | '1y' | 'all';
type Tab = 'overview' | 'audience' | 'songs' | 'engagement' | 'sources';

// Mock data for demonstration
const mockAudienceData = {
  countries: [
    { name: 'United States', streams: 45230, percent: 35 },
    { name: 'United Kingdom', streams: 28540, percent: 22 },
    { name: 'Germany', streams: 18920, percent: 15 },
    { name: 'France', streams: 12650, percent: 10 },
    { name: 'Canada', streams: 9870, percent: 8 },
    { name: 'Other', streams: 12890, percent: 10 },
  ],
  cities: [
    { name: 'Los Angeles', streams: 12450 },
    { name: 'London', streams: 10230 },
    { name: 'New York', streams: 9870 },
    { name: 'Berlin', streams: 7650 },
    { name: 'Paris', streams: 6540 },
  ],
  ageGroups: [
    { range: '13-17', percent: 8 },
    { range: '18-24', percent: 32 },
    { range: '25-34', percent: 35 },
    { range: '35-44', percent: 15 },
    { range: '45-54', percent: 7 },
    { range: '55+', percent: 3 },
  ],
  gender: [
    { name: 'Male', percent: 58 },
    { name: 'Female', percent: 38 },
    { name: 'Other', percent: 4 },
  ],
};

const mockListeningPatterns = {
  hourly: Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    streams: Math.floor(Math.random() * 1000) + 200 + (i >= 18 && i <= 23 ? 500 : 0),
  })),
  daily: [
    { day: 'Mon', streams: 4520 },
    { day: 'Tue', streams: 4890 },
    { day: 'Wed', streams: 5120 },
    { day: 'Thu', streams: 5340 },
    { day: 'Fri', streams: 6780 },
    { day: 'Sat', streams: 7890 },
    { day: 'Sun', streams: 6540 },
  ],
};

const mockSources = [
  { name: 'Home', streams: 35000, percent: 28 },
  { name: 'Search', streams: 28000, percent: 22 },
  { name: 'Playlists', streams: 25000, percent: 20 },
  { name: 'Artist Profile', streams: 18000, percent: 14 },
  { name: 'External Links', streams: 12000, percent: 10 },
  { name: 'Other', streams: 7000, percent: 6 },
];

const mockDevices = [
  { name: 'Mobile', streams: 65000, percent: 52, icon: Smartphone },
  { name: 'Desktop', streams: 45000, percent: 36, icon: Monitor },
  { name: 'Tablet', streams: 15000, percent: 12, icon: Tablet },
];

const mockEngagement = {
  saves: 8540,
  shares: 3250,
  playlistAdds: 12340,
  avgListenDuration: 187, // seconds
  completionRate: 78, // percent
  skipRate: 22, // percent
};

const mockSongPerformance = [
  { title: 'Midnight Dreams', streams: 45230, saves: 2340, shares: 890, completion: 82 },
  { title: 'Neon Lights', streams: 38920, saves: 1980, shares: 720, completion: 79 },
  { title: 'Digital Wave', streams: 32450, saves: 1650, shares: 580, completion: 75 },
  { title: 'Crystal Night', streams: 28760, saves: 1420, shares: 510, completion: 81 },
  { title: 'Electric Pulse', streams: 24890, saves: 1280, shares: 450, completion: 77 },
];

const COLORS = ['#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6', '#6366F1'];

export default function AnalyticsPage() {
  const { authenticated, user } = useAuth();
  const [period, setPeriod] = useState<Period>('30d');
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  if (!authenticated) {
    redirect('/');
  }

  const periods: { value: Period; label: string }[] = [
    { value: '7d', label: '7 days' },
    { value: '30d', label: '30 days' },
    { value: '90d', label: '90 days' },
    { value: '1y', label: '1 year' },
    { value: 'all', label: 'All time' },
  ];

  const tabs: { value: Tab; label: string; icon: React.ReactNode }[] = [
    { value: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
    { value: 'audience', label: 'Audience', icon: <Users className="w-4 h-4" /> },
    { value: 'songs', label: 'Songs', icon: <Music2 className="w-4 h-4" /> },
    { value: 'engagement', label: 'Engagement', icon: <Heart className="w-4 h-4" /> },
    { value: 'sources', label: 'Sources', icon: <Globe className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Back Link & Header */}
          <div className="mb-6">
            <Link
              href="/dashboard"
              className="flex items-center gap-2 text-muted-foreground hover:text-white transition-colors mb-4"
            >
              <ChevronLeft className="w-4 h-4" />
              Back to Dashboard
            </Link>
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Analytics</h1>
                <p className="text-muted-foreground">
                  Deep insights into your music performance
                </p>
              </div>

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
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-2 mb-8 border-b border-white/10 pb-4">
            {tabs.map((tab) => (
              <button
                key={tab.value}
                onClick={() => setActiveTab(tab.value)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.value
                    ? 'bg-purple-500 text-white'
                    : 'text-muted-foreground hover:text-white hover:bg-white/5'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'audience' && <AudienceTab />}
          {activeTab === 'songs' && <SongsTab />}
          {activeTab === 'engagement' && <EngagementTab />}
          {activeTab === 'sources' && <SourcesTab />}
        </div>
      </main>

      <Player />
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <QuickStat
          label="Total Streams"
          value="128,100"
          change={12.5}
          icon={<Headphones className="w-5 h-5" />}
        />
        <QuickStat
          label="Unique Listeners"
          value="45,230"
          change={8.2}
          icon={<Users className="w-5 h-5" />}
        />
        <QuickStat
          label="Avg. Stream Duration"
          value="3:07"
          change={-2.1}
          icon={<Clock className="w-5 h-5" />}
        />
        <QuickStat
          label="Engagement Rate"
          value="18.4%"
          change={5.8}
          icon={<Heart className="w-5 h-5" />}
        />
      </div>

      {/* Listening Patterns */}
      <div className="grid grid-cols-2 gap-6">
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Streams by Hour</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockListeningPatterns.hourly}>
                <XAxis
                  dataKey="hour"
                  tickFormatter={(h) => `${h}:00`}
                  stroke="#666"
                  fontSize={10}
                />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  labelFormatter={(h) => `${h}:00 - ${(h + 1) % 24}:00`}
                  formatter={(val: number) => [formatNumber(val), 'Streams']}
                />
                <Bar dataKey="streams" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Streams by Day</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockListeningPatterns.daily}>
                <XAxis dataKey="day" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  formatter={(val: number) => [formatNumber(val), 'Streams']}
                />
                <Bar dataKey="streams" fill="#EC4899" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Top Countries */}
      <div className="p-6 bg-white/5 rounded-xl">
        <h3 className="font-semibold mb-6">Top Countries</h3>
        <div className="space-y-3">
          {mockAudienceData.countries.slice(0, 5).map((country, i) => (
            <div key={country.name} className="flex items-center gap-4">
              <span className="w-6 text-muted-foreground">{i + 1}</span>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">{country.name}</span>
                  <span className="text-sm text-muted-foreground">
                    {formatNumber(country.streams)} streams
                  </span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                    style={{ width: `${country.percent}%` }}
                  />
                </div>
              </div>
              <span className="text-sm text-muted-foreground w-12 text-right">
                {country.percent}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AudienceTab() {
  return (
    <div className="space-y-6">
      {/* Demographics */}
      <div className="grid grid-cols-2 gap-6">
        {/* Age Distribution */}
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Age Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockAudienceData.ageGroups} layout="vertical">
                <XAxis type="number" stroke="#666" fontSize={12} />
                <YAxis dataKey="range" type="category" stroke="#666" fontSize={12} width={50} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  formatter={(val: number) => [`${val}%`, 'Listeners']}
                />
                <Bar dataKey="percent" fill="#8B5CF6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Gender Distribution */}
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Gender Distribution</h3>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={mockAudienceData.gender}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="percent"
                  nameKey="name"
                  label={({ name, percent }) => `${name}: ${percent}%`}
                >
                  {mockAudienceData.gender.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Top Countries & Cities */}
      <div className="grid grid-cols-2 gap-6">
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Top Countries</h3>
          <div className="space-y-3">
            {mockAudienceData.countries.map((country, i) => (
              <div key={country.name} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="w-6 text-muted-foreground">{i + 1}</span>
                  <span>{country.name}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-muted-foreground">
                    {formatNumber(country.streams)}
                  </span>
                  <div className="w-24 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{ width: `${country.percent}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Top Cities</h3>
          <div className="space-y-3">
            {mockAudienceData.cities.map((city, i) => (
              <div key={city.name} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="w-6 text-muted-foreground">{i + 1}</span>
                  <span>{city.name}</span>
                </div>
                <span className="text-muted-foreground">
                  {formatNumber(city.streams)} streams
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function SongsTab() {
  return (
    <div className="space-y-6">
      {/* Song Performance Table */}
      <div className="p-6 bg-white/5 rounded-xl">
        <h3 className="font-semibold mb-6">Song Performance</h3>
        <table className="w-full">
          <thead>
            <tr className="text-left text-muted-foreground text-sm border-b border-white/10">
              <th className="pb-4 font-medium">#</th>
              <th className="pb-4 font-medium">Song</th>
              <th className="pb-4 font-medium text-right">Streams</th>
              <th className="pb-4 font-medium text-right">Saves</th>
              <th className="pb-4 font-medium text-right">Shares</th>
              <th className="pb-4 font-medium text-right">Completion</th>
            </tr>
          </thead>
          <tbody>
            {mockSongPerformance.map((song, i) => (
              <tr key={song.title} className="border-b border-white/5 hover:bg-white/5">
                <td className="py-4 text-muted-foreground">{i + 1}</td>
                <td className="py-4 font-medium">{song.title}</td>
                <td className="py-4 text-right">{formatNumber(song.streams)}</td>
                <td className="py-4 text-right text-muted-foreground">
                  {formatNumber(song.saves)}
                </td>
                <td className="py-4 text-right text-muted-foreground">
                  {formatNumber(song.shares)}
                </td>
                <td className="py-4 text-right">
                  <span className={song.completion >= 80 ? 'text-green-500' : 'text-yellow-500'}>
                    {song.completion}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Song Comparison Radar */}
      <div className="p-6 bg-white/5 rounded-xl">
        <h3 className="font-semibold mb-6">Song Comparison</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={[
              { metric: 'Streams', 'Midnight Dreams': 100, 'Neon Lights': 86, 'Digital Wave': 72 },
              { metric: 'Saves', 'Midnight Dreams': 100, 'Neon Lights': 85, 'Digital Wave': 71 },
              { metric: 'Shares', 'Midnight Dreams': 100, 'Neon Lights': 81, 'Digital Wave': 65 },
              { metric: 'Completion', 'Midnight Dreams': 82, 'Neon Lights': 79, 'Digital Wave': 75 },
              { metric: 'Playlists', 'Midnight Dreams': 95, 'Neon Lights': 78, 'Digital Wave': 62 },
            ]}>
              <PolarGrid stroke="#333" />
              <PolarAngleAxis dataKey="metric" stroke="#666" fontSize={12} />
              <PolarRadiusAxis stroke="#666" fontSize={10} />
              <Radar
                name="Midnight Dreams"
                dataKey="Midnight Dreams"
                stroke="#8B5CF6"
                fill="#8B5CF6"
                fillOpacity={0.3}
              />
              <Radar
                name="Neon Lights"
                dataKey="Neon Lights"
                stroke="#EC4899"
                fill="#EC4899"
                fillOpacity={0.3}
              />
              <Radar
                name="Digital Wave"
                dataKey="Digital Wave"
                stroke="#10B981"
                fill="#10B981"
                fillOpacity={0.3}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333',
                  borderRadius: '8px',
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center justify-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-500" />
            <span className="text-sm">Midnight Dreams</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-pink-500" />
            <span className="text-sm">Neon Lights</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-sm">Digital Wave</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function EngagementTab() {
  return (
    <div className="space-y-6">
      {/* Engagement Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-6 bg-white/5 rounded-xl text-center">
          <Heart className="w-8 h-8 text-pink-500 mx-auto mb-3" />
          <p className="text-3xl font-bold mb-1">{formatNumber(mockEngagement.saves)}</p>
          <p className="text-muted-foreground">Total Saves</p>
        </div>
        <div className="p-6 bg-white/5 rounded-xl text-center">
          <Share2 className="w-8 h-8 text-blue-500 mx-auto mb-3" />
          <p className="text-3xl font-bold mb-1">{formatNumber(mockEngagement.shares)}</p>
          <p className="text-muted-foreground">Total Shares</p>
        </div>
        <div className="p-6 bg-white/5 rounded-xl text-center">
          <ListMusic className="w-8 h-8 text-green-500 mx-auto mb-3" />
          <p className="text-3xl font-bold mb-1">{formatNumber(mockEngagement.playlistAdds)}</p>
          <p className="text-muted-foreground">Playlist Adds</p>
        </div>
      </div>

      {/* Listening Metrics */}
      <div className="grid grid-cols-2 gap-6">
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Average Listen Duration</h3>
          <div className="flex items-center justify-center py-8">
            <div className="relative w-48 h-48">
              <svg className="w-full h-full -rotate-90">
                <circle
                  cx="96"
                  cy="96"
                  r="88"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="12"
                  className="text-white/10"
                />
                <circle
                  cx="96"
                  cy="96"
                  r="88"
                  fill="none"
                  stroke="url(#gradient)"
                  strokeWidth="12"
                  strokeLinecap="round"
                  strokeDasharray={2 * Math.PI * 88}
                  strokeDashoffset={2 * Math.PI * 88 * (1 - mockEngagement.completionRate / 100)}
                />
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#EC4899" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <p className="text-4xl font-bold">
                  {formatDuration(mockEngagement.avgListenDuration)}
                </p>
                <p className="text-muted-foreground">average</p>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Completion vs Skip Rate</h3>
          <div className="space-y-6 py-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span>Completion Rate</span>
                <span className="text-green-500 font-medium">
                  {mockEngagement.completionRate}%
                </span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
                  style={{ width: `${mockEngagement.completionRate}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span>Skip Rate</span>
                <span className="text-red-500 font-medium">{mockEngagement.skipRate}%</span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-red-500 to-orange-400 rounded-full"
                  style={{ width: `${mockEngagement.skipRate}%` }}
                />
              </div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            Your completion rate is above average, indicating strong engagement with your music.
          </p>
        </div>
      </div>
    </div>
  );
}

function SourcesTab() {
  return (
    <div className="space-y-6">
      {/* Source Breakdown */}
      <div className="grid grid-cols-2 gap-6">
        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Traffic Sources</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={mockSources}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="percent"
                  nameKey="name"
                >
                  {mockSources.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  formatter={(val: number) => [`${val}%`, 'Share']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-3 justify-center mt-4">
            {mockSources.map((source, i) => (
              <div key={source.name} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: COLORS[i % COLORS.length] }}
                />
                <span className="text-sm">{source.name}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 bg-white/5 rounded-xl">
          <h3 className="font-semibold mb-6">Source Details</h3>
          <div className="space-y-3">
            {mockSources.map((source, i) => (
              <div key={source.name} className="flex items-center gap-4">
                <div
                  className="w-2 h-8 rounded-full"
                  style={{ backgroundColor: COLORS[i % COLORS.length] }}
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">{source.name}</span>
                    <span className="text-muted-foreground">
                      {formatNumber(source.streams)}
                    </span>
                  </div>
                  <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${source.percent}%`,
                        backgroundColor: COLORS[i % COLORS.length],
                      }}
                    />
                  </div>
                </div>
                <span className="text-sm text-muted-foreground w-12 text-right">
                  {source.percent}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Device Breakdown */}
      <div className="p-6 bg-white/5 rounded-xl">
        <h3 className="font-semibold mb-6">Devices</h3>
        <div className="grid grid-cols-3 gap-6">
          {mockDevices.map((device) => (
            <div
              key={device.name}
              className="flex items-center gap-4 p-4 bg-white/5 rounded-lg"
            >
              <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center">
                <device.icon className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <p className="font-medium">{device.name}</p>
                <p className="text-sm text-muted-foreground">
                  {formatNumber(device.streams)} ({device.percent}%)
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function QuickStat({
  label,
  value,
  change,
  icon,
}: {
  label: string;
  value: string;
  change: number;
  icon: React.ReactNode;
}) {
  const isPositive = change >= 0;

  return (
    <div className="p-6 bg-white/5 rounded-xl">
      <div className="flex items-center justify-between mb-4">
        <span className="text-muted-foreground text-sm">{label}</span>
        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
          {icon}
        </div>
      </div>
      <p className="text-3xl font-bold mb-2">{value}</p>
      <div className="flex items-center gap-1 text-sm">
        {isPositive ? (
          <TrendingUp className="w-4 h-4 text-green-500" />
        ) : (
          <TrendingUp className="w-4 h-4 text-red-500 rotate-180" />
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
