// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Artist Dashboard
 *
 * Comprehensive analytics dashboard for artists showing:
 * - Real-time listener counts
 * - Revenue breakdown
 * - Audience demographics
 * - Track performance
 * - Engagement metrics
 */

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { formatDistanceToNow } from 'date-fns';

// ============================================================================
// Types
// ============================================================================

interface DashboardProps {
  artistId: string;
}

interface TimeRange {
  label: string;
  value: '24h' | '7d' | '30d' | '90d' | '1y' | 'all';
  start: Date;
  end: Date;
}

// ============================================================================
// API Hooks
// ============================================================================

function useArtistOverview(artistId: string, range: TimeRange) {
  return useQuery({
    queryKey: ['artist-overview', artistId, range.value],
    queryFn: async () => {
      const response = await fetch(
        `/api/analytics/artist/${artistId}/overview?start=${range.start.toISOString()}&end=${range.end.toISOString()}`
      );
      return response.json();
    },
    refetchInterval: 60000, // Refresh every minute
  });
}

function useRealTimeStats(artistId: string) {
  return useQuery({
    queryKey: ['artist-realtime', artistId],
    queryFn: async () => {
      const response = await fetch(`/api/analytics/artist/${artistId}/realtime`);
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });
}

function useTrackPerformance(artistId: string, range: TimeRange) {
  return useQuery({
    queryKey: ['artist-tracks', artistId, range.value],
    queryFn: async () => {
      const response = await fetch(
        `/api/analytics/artist/${artistId}/tracks?start=${range.start.toISOString()}&end=${range.end.toISOString()}`
      );
      return response.json();
    },
  });
}

function useAudienceDemographics(artistId: string, range: TimeRange) {
  return useQuery({
    queryKey: ['artist-audience', artistId, range.value],
    queryFn: async () => {
      const response = await fetch(
        `/api/analytics/artist/${artistId}/audience?start=${range.start.toISOString()}&end=${range.end.toISOString()}`
      );
      return response.json();
    },
  });
}

function useRevenueBreakdown(artistId: string, range: TimeRange) {
  return useQuery({
    queryKey: ['artist-revenue', artistId, range.value],
    queryFn: async () => {
      const response = await fetch(
        `/api/analytics/artist/${artistId}/revenue?start=${range.start.toISOString()}&end=${range.end.toISOString()}`
      );
      return response.json();
    },
  });
}

// ============================================================================
// Helper Components
// ============================================================================

const StatCard: React.FC<{
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  loading?: boolean;
}> = ({ title, value, change, icon, loading }) => (
  <div className="stat-card">
    <div className="stat-icon">{icon}</div>
    <div className="stat-content">
      <span className="stat-title">{title}</span>
      {loading ? (
        <div className="stat-skeleton" />
      ) : (
        <>
          <span className="stat-value">{value}</span>
          {change !== undefined && (
            <span className={`stat-change ${change >= 0 ? 'positive' : 'negative'}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(1)}%
            </span>
          )}
        </>
      )}
    </div>
  </div>
);

const LiveIndicator: React.FC<{ count: number }> = ({ count }) => (
  <div className="live-indicator">
    <span className="live-dot" />
    <span className="live-count">{count.toLocaleString()}</span>
    <span className="live-label">listening now</span>
  </div>
);

// ============================================================================
// Chart Colors
// ============================================================================

const COLORS = {
  primary: '#8b5cf6',
  secondary: '#06b6d4',
  accent: '#f59e0b',
  success: '#10b981',
  danger: '#ef4444',
  chart: ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#6366f1'],
};

// ============================================================================
// Main Dashboard Component
// ============================================================================

export const ArtistDashboard: React.FC<DashboardProps> = ({ artistId }) => {
  const [timeRange, setTimeRange] = useState<TimeRange>(getTimeRange('30d'));
  const [activeTab, setActiveTab] = useState<'overview' | 'tracks' | 'audience' | 'revenue'>('overview');

  const { data: overview, isLoading: overviewLoading } = useArtistOverview(artistId, timeRange);
  const { data: realtime } = useRealTimeStats(artistId);
  const { data: tracks, isLoading: tracksLoading } = useTrackPerformance(artistId, timeRange);
  const { data: audience, isLoading: audienceLoading } = useAudienceDemographics(artistId, timeRange);
  const { data: revenue, isLoading: revenueLoading } = useRevenueBreakdown(artistId, timeRange);

  return (
    <div className="artist-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>Artist Dashboard</h1>
          {realtime && <LiveIndicator count={realtime.activeListeners} />}
        </div>
        <div className="header-right">
          <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
        </div>
      </header>

      {/* Stats Overview */}
      <section className="stats-grid">
        <StatCard
          title="Total Plays"
          value={overview?.summary.totalPlays.toLocaleString() || '-'}
          change={overview?.growth.playsChange}
          icon={<PlayIcon />}
          loading={overviewLoading}
        />
        <StatCard
          title="Unique Listeners"
          value={overview?.summary.uniqueListeners.toLocaleString() || '-'}
          change={overview?.growth.listenersChange}
          icon={<UsersIcon />}
          loading={overviewLoading}
        />
        <StatCard
          title="Revenue"
          value={overview ? formatCurrency(overview.revenue.total) : '-'}
          change={overview?.growth.revenueChange}
          icon={<DollarIcon />}
          loading={overviewLoading}
        />
        <StatCard
          title="Completion Rate"
          value={overview ? `${overview.summary.completionRate.toFixed(1)}%` : '-'}
          icon={<CheckIcon />}
          loading={overviewLoading}
        />
      </section>

      {/* Navigation Tabs */}
      <nav className="dashboard-tabs">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={activeTab === 'tracks' ? 'active' : ''}
          onClick={() => setActiveTab('tracks')}
        >
          Tracks
        </button>
        <button
          className={activeTab === 'audience' ? 'active' : ''}
          onClick={() => setActiveTab('audience')}
        >
          Audience
        </button>
        <button
          className={activeTab === 'revenue' ? 'active' : ''}
          onClick={() => setActiveTab('revenue')}
        >
          Revenue
        </button>
      </nav>

      {/* Tab Content */}
      <div className="dashboard-content">
        {activeTab === 'overview' && (
          <OverviewTab
            overview={overview}
            realtime={realtime}
            loading={overviewLoading}
          />
        )}
        {activeTab === 'tracks' && (
          <TracksTab tracks={tracks} loading={tracksLoading} />
        )}
        {activeTab === 'audience' && (
          <AudienceTab audience={audience} loading={audienceLoading} />
        )}
        {activeTab === 'revenue' && (
          <RevenueTab revenue={revenue} loading={revenueLoading} />
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Tab Components
// ============================================================================

const OverviewTab: React.FC<{
  overview: any;
  realtime: any;
  loading: boolean;
}> = ({ overview, realtime, loading }) => {
  if (loading) return <LoadingState />;

  return (
    <div className="overview-tab">
      {/* Plays Over Time Chart */}
      <div className="chart-card full-width">
        <h3>Plays Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={overview?.daily || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="plays"
              stroke={COLORS.primary}
              fill={COLORS.primary}
              fillOpacity={0.3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-grid">
        {/* Source Distribution */}
        <div className="chart-card">
          <h3>Traffic Sources</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={formatPieData(overview?.sources || {})}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                dataKey="value"
                nameKey="name"
                label
              >
                {formatPieData(overview?.sources || {}).map((_, index) => (
                  <Cell key={index} fill={COLORS.chart[index % COLORS.chart.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Device Distribution */}
        <div className="chart-card">
          <h3>Devices</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={formatBarData(overview?.devices || {})} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={80} />
              <Tooltip />
              <Bar dataKey="value" fill={COLORS.secondary} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="chart-card">
        <h3>Recent Activity</h3>
        <div className="activity-feed">
          {realtime?.recentActivity?.slice(0, 10).map((activity: any, index: number) => (
            <div key={index} className="activity-item">
              <span className="activity-icon">{getActivityIcon(activity.type)}</span>
              <span className="activity-text">{formatActivity(activity)}</span>
              <span className="activity-time">
                {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const TracksTab: React.FC<{ tracks: any; loading: boolean }> = ({ tracks, loading }) => {
  if (loading) return <LoadingState />;

  return (
    <div className="tracks-tab">
      <table className="tracks-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Track</th>
            <th>Plays</th>
            <th>Listeners</th>
            <th>Saves</th>
            <th>Completion</th>
            <th>Revenue</th>
            <th>Trend</th>
          </tr>
        </thead>
        <tbody>
          {tracks?.map((track: any, index: number) => (
            <tr key={track.trackId}>
              <td>{index + 1}</td>
              <td className="track-title">{track.title}</td>
              <td>{track.plays.toLocaleString()}</td>
              <td>{track.uniqueListeners.toLocaleString()}</td>
              <td>{track.saves.toLocaleString()}</td>
              <td>{track.completionRate.toFixed(1)}%</td>
              <td>{formatCurrency(track.revenue)}</td>
              <td>
                <TrendIndicator trend={track.trend} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const AudienceTab: React.FC<{ audience: any; loading: boolean }> = ({ audience, loading }) => {
  if (loading) return <LoadingState />;

  return (
    <div className="audience-tab">
      <div className="chart-grid">
        {/* Top Countries */}
        <div className="chart-card">
          <h3>Top Countries</h3>
          <div className="country-list">
            {audience?.topCountries?.map((country: any) => (
              <div key={country.country} className="country-item">
                <span className="country-flag">{getCountryFlag(country.country)}</span>
                <span className="country-name">{country.country}</span>
                <div className="country-bar">
                  <div
                    className="country-bar-fill"
                    style={{ width: `${country.percentage}%` }}
                  />
                </div>
                <span className="country-percent">{country.percentage.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Age Distribution */}
        <div className="chart-card">
          <h3>Age Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={formatBarData(audience?.ageRanges || {})}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill={COLORS.primary} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Listening Times */}
        <div className="chart-card full-width">
          <h3>When Your Audience Listens</h3>
          <div className="heatmap-container">
            <ListeningHeatmap data={audience?.listeningTimes} />
          </div>
        </div>

        {/* Premium vs Free */}
        <div className="chart-card">
          <h3>Listener Types</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Premium', value: audience?.premiumVsFree?.premium || 0 },
                  { name: 'Free', value: audience?.premiumVsFree?.free || 0 },
                ]}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                dataKey="value"
              >
                <Cell fill={COLORS.primary} />
                <Cell fill={COLORS.secondary} />
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const RevenueTab: React.FC<{ revenue: any; loading: boolean }> = ({ revenue, loading }) => {
  if (loading) return <LoadingState />;

  return (
    <div className="revenue-tab">
      {/* Revenue Summary */}
      <div className="revenue-summary">
        <div className="revenue-total">
          <span className="label">Total Revenue</span>
          <span className="value">{formatCurrency(revenue?.total || 0)}</span>
          <span className="projected">
            Projected: {formatCurrency(revenue?.projectedMonthly || 0)}/month
          </span>
        </div>
      </div>

      <div className="chart-grid">
        {/* Revenue by Source */}
        <div className="chart-card">
          <h3>Revenue by Source</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Streaming', value: revenue?.bySource?.streaming?.amount || 0 },
                  { name: 'Patronage', value: revenue?.bySource?.patronage?.amount || 0 },
                  { name: 'NFT Sales', value: revenue?.bySource?.nftSales?.amount || 0 },
                  { name: 'Licensing', value: revenue?.bySource?.licensing?.amount || 0 },
                  { name: 'Tips', value: revenue?.bySource?.tips?.amount || 0 },
                ].filter(d => d.value > 0)}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {COLORS.chart.map((color, index) => (
                  <Cell key={index} fill={color} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => formatCurrency(value)} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Revenue by Track */}
        <div className="chart-card">
          <h3>Top Earning Tracks</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={revenue?.byTrack?.slice(0, 10) || []} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tickFormatter={(v) => formatCurrency(v)} />
              <YAxis dataKey="title" type="category" width={150} />
              <Tooltip formatter={(value: number) => formatCurrency(value)} />
              <Bar dataKey="amount" fill={COLORS.success} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Payout Schedule */}
      <div className="chart-card">
        <h3>Payout Schedule</h3>
        <table className="payout-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Amount</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {revenue?.payoutSchedule?.map((payout: any, index: number) => (
              <tr key={index}>
                <td>{new Date(payout.date).toLocaleDateString()}</td>
                <td>{formatCurrency(payout.amount)}</td>
                <td>
                  <span className={`status-badge ${payout.status}`}>
                    {payout.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ============================================================================
// Subcomponents
// ============================================================================

const TimeRangeSelector: React.FC<{
  value: TimeRange;
  onChange: (range: TimeRange) => void;
}> = ({ value, onChange }) => {
  const options: Array<{ label: string; value: TimeRange['value'] }> = [
    { label: '24 Hours', value: '24h' },
    { label: '7 Days', value: '7d' },
    { label: '30 Days', value: '30d' },
    { label: '90 Days', value: '90d' },
    { label: '1 Year', value: '1y' },
    { label: 'All Time', value: 'all' },
  ];

  return (
    <div className="time-range-selector">
      {options.map(option => (
        <button
          key={option.value}
          className={value.value === option.value ? 'active' : ''}
          onClick={() => onChange(getTimeRange(option.value))}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
};

const TrendIndicator: React.FC<{ trend: 'rising' | 'stable' | 'declining' }> = ({ trend }) => {
  const icons = {
    rising: <span className="trend rising">+</span>,
    stable: <span className="trend stable">-</span>,
    declining: <span className="trend declining">-</span>,
  };
  return icons[trend];
};

const ListeningHeatmap: React.FC<{ data?: { byHour: number[]; byDayOfWeek: number[] } }> = ({ data }) => {
  if (!data) return null;

  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const maxValue = Math.max(...data.byHour, ...data.byDayOfWeek);

  return (
    <div className="listening-heatmap">
      <div className="heatmap-hours">
        <h4>By Hour</h4>
        <div className="heatmap-row">
          {data.byHour.map((value, hour) => (
            <div
              key={hour}
              className="heatmap-cell"
              style={{ opacity: 0.2 + (value / maxValue) * 0.8 }}
              title={`${hour}:00 - ${value.toLocaleString()} listeners`}
            >
              {hour % 6 === 0 ? hour : ''}
            </div>
          ))}
        </div>
      </div>
      <div className="heatmap-days">
        <h4>By Day</h4>
        <div className="heatmap-row">
          {data.byDayOfWeek.map((value, day) => (
            <div
              key={day}
              className="heatmap-cell"
              style={{ opacity: 0.2 + (value / maxValue) * 0.8 }}
              title={`${days[day]} - ${value.toLocaleString()} listeners`}
            >
              {days[day]}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const LoadingState: React.FC = () => (
  <div className="loading-state">
    <div className="spinner" />
    <span>Loading analytics...</span>
  </div>
);

// ============================================================================
// Icons (simplified)
// ============================================================================

const PlayIcon = () => <span>|></span>;
const UsersIcon = () => <span>U</span>;
const DollarIcon = () => <span>$</span>;
const CheckIcon = () => <span>v</span>;

// ============================================================================
// Utilities
// ============================================================================

function getTimeRange(value: TimeRange['value']): TimeRange {
  const end = new Date();
  let start: Date;

  switch (value) {
    case '24h':
      start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
      break;
    case '7d':
      start = new Date(end.getTime() - 7 * 24 * 60 * 60 * 1000);
      break;
    case '30d':
      start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
      break;
    case '90d':
      start = new Date(end.getTime() - 90 * 24 * 60 * 60 * 1000);
      break;
    case '1y':
      start = new Date(end.getTime() - 365 * 24 * 60 * 60 * 1000);
      break;
    case 'all':
    default:
      start = new Date(0);
  }

  return { label: value, value, start, end };
}

function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(amount);
}

function formatPieData(obj: Record<string, number>): Array<{ name: string; value: number }> {
  return Object.entries(obj).map(([name, value]) => ({ name, value }));
}

function formatBarData(obj: Record<string, number>): Array<{ name: string; value: number }> {
  return Object.entries(obj)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
}

function getActivityIcon(type: string): string {
  const icons: Record<string, string> = {
    play: '|>',
    save: '*',
    follow: '+',
    share: '>',
    comment: '#',
  };
  return icons[type] || '-';
}

function formatActivity(activity: any): string {
  switch (activity.type) {
    case 'play':
      return `${activity.data.listenerName} played "${activity.data.trackTitle}"`;
    case 'save':
      return `${activity.data.listenerName} saved "${activity.data.trackTitle}"`;
    case 'follow':
      return `${activity.data.listenerName} started following you`;
    default:
      return JSON.stringify(activity.data);
  }
}

function getCountryFlag(countryCode: string): string {
  // Simplified - would use proper flag emoji lookup
  return countryCode.slice(0, 2).toUpperCase();
}

export default ArtistDashboard;
