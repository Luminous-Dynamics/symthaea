// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Artist Analytics Hook
 *
 * Comprehensive analytics for artists including
 * streams, listeners, demographics, and revenue
 */

import { useState, useCallback, useEffect } from 'react';

export interface TimeRange {
  start: Date;
  end: Date;
  label: string;
}

export interface StreamData {
  date: string;
  streams: number;
  uniqueListeners: number;
  avgDuration: number;
  skips: number;
  completions: number;
}

export interface TrackAnalytics {
  trackId: string;
  title: string;
  totalStreams: number;
  uniqueListeners: number;
  avgListenDuration: number;
  skipRate: number;
  saveRate: number;
  shareCount: number;
  playlistAdds: number;
  trend: 'up' | 'down' | 'stable';
  trendPercent: number;
}

export interface ListenerDemographics {
  ageGroups: Record<string, number>;
  gender: Record<string, number>;
  countries: { country: string; listeners: number; streams: number }[];
  cities: { city: string; country: string; listeners: number }[];
  platforms: Record<string, number>;
  devices: Record<string, number>;
}

export interface ListenerBehavior {
  avgSessionDuration: number;
  peakListeningHours: number[];
  peakListeningDays: string[];
  repeatListenerRate: number;
  newListenerRate: number;
  playlistDiscovery: number;
  searchDiscovery: number;
  recommendationDiscovery: number;
}

export interface RevenueData {
  totalRevenue: number;
  streamRevenue: number;
  tipRevenue: number;
  merchandiseRevenue: number;
  byPlatform: Record<string, number>;
  history: { date: string; amount: number }[];
  projectedMonthly: number;
}

export interface FollowerData {
  total: number;
  newThisPeriod: number;
  lostThisPeriod: number;
  growth: number;
  history: { date: string; count: number }[];
}

export interface PlaylistData {
  playlistsIn: number;
  totalReach: number;
  topPlaylists: {
    name: string;
    curator: string;
    followers: number;
    position: number;
  }[];
}

export interface ArtistInsight {
  type: 'success' | 'warning' | 'tip' | 'milestone';
  title: string;
  description: string;
  metric?: string;
  action?: string;
}

export interface AnalyticsState {
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Time range presets
export const TIME_RANGES: Record<string, TimeRange> = {
  today: {
    start: new Date(new Date().setHours(0, 0, 0, 0)),
    end: new Date(),
    label: 'Today',
  },
  week: {
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    end: new Date(),
    label: 'Last 7 Days',
  },
  month: {
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    end: new Date(),
    label: 'Last 30 Days',
  },
  quarter: {
    start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
    end: new Date(),
    label: 'Last 90 Days',
  },
  year: {
    start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
    end: new Date(),
    label: 'Last Year',
  },
  allTime: {
    start: new Date(0),
    end: new Date(),
    label: 'All Time',
  },
};

export function useArtistAnalytics(artistId: string) {
  const [state, setState] = useState<AnalyticsState>({
    isLoading: false,
    error: null,
    lastUpdated: null,
  });
  const [timeRange, setTimeRange] = useState<TimeRange>(TIME_RANGES.month);

  // Stream data
  const [streamData, setStreamData] = useState<StreamData[]>([]);
  const [totalStreams, setTotalStreams] = useState(0);
  const [uniqueListeners, setUniqueListeners] = useState(0);

  // Track analytics
  const [trackAnalytics, setTrackAnalytics] = useState<TrackAnalytics[]>([]);

  // Demographics
  const [demographics, setDemographics] = useState<ListenerDemographics | null>(null);

  // Behavior
  const [behavior, setBehavior] = useState<ListenerBehavior | null>(null);

  // Revenue
  const [revenue, setRevenue] = useState<RevenueData | null>(null);

  // Followers
  const [followers, setFollowers] = useState<FollowerData | null>(null);

  // Playlists
  const [playlists, setPlaylists] = useState<PlaylistData | null>(null);

  // Insights
  const [insights, setInsights] = useState<ArtistInsight[]>([]);

  // Generate mock data for development
  const generateMockData = useCallback(() => {
    const days = Math.ceil((timeRange.end.getTime() - timeRange.start.getTime()) / (24 * 60 * 60 * 1000));

    // Stream data
    const mockStreamData: StreamData[] = [];
    let totalS = 0;
    let totalL = 0;

    for (let i = 0; i < days; i++) {
      const date = new Date(timeRange.start.getTime() + i * 24 * 60 * 60 * 1000);
      const streams = Math.floor(Math.random() * 500) + 100;
      const listeners = Math.floor(streams * (0.4 + Math.random() * 0.3));

      mockStreamData.push({
        date: date.toISOString().split('T')[0],
        streams,
        uniqueListeners: listeners,
        avgDuration: 150 + Math.random() * 60,
        skips: Math.floor(streams * (0.1 + Math.random() * 0.1)),
        completions: Math.floor(streams * (0.6 + Math.random() * 0.2)),
      });

      totalS += streams;
      totalL += listeners;
    }

    setStreamData(mockStreamData);
    setTotalStreams(totalS);
    setUniqueListeners(totalL);

    // Track analytics
    const mockTracks: TrackAnalytics[] = [
      { trackId: '1', title: 'Summer Vibes', totalStreams: 12500, uniqueListeners: 8200, avgListenDuration: 185, skipRate: 0.12, saveRate: 0.08, shareCount: 340, playlistAdds: 89, trend: 'up', trendPercent: 23 },
      { trackId: '2', title: 'Night Drive', totalStreams: 9800, uniqueListeners: 6100, avgListenDuration: 210, skipRate: 0.08, saveRate: 0.11, shareCount: 280, playlistAdds: 67, trend: 'up', trendPercent: 15 },
      { trackId: '3', title: 'Electric Dreams', totalStreams: 7200, uniqueListeners: 4800, avgListenDuration: 175, skipRate: 0.15, saveRate: 0.06, shareCount: 150, playlistAdds: 45, trend: 'stable', trendPercent: 2 },
      { trackId: '4', title: 'Sunset Memories', totalStreams: 5400, uniqueListeners: 3600, avgListenDuration: 195, skipRate: 0.10, saveRate: 0.09, shareCount: 120, playlistAdds: 38, trend: 'down', trendPercent: -8 },
      { trackId: '5', title: 'Urban Pulse', totalStreams: 4100, uniqueListeners: 2900, avgListenDuration: 160, skipRate: 0.18, saveRate: 0.05, shareCount: 90, playlistAdds: 22, trend: 'up', trendPercent: 5 },
    ];
    setTrackAnalytics(mockTracks);

    // Demographics
    setDemographics({
      ageGroups: { '13-17': 8, '18-24': 32, '25-34': 35, '35-44': 15, '45-54': 7, '55+': 3 },
      gender: { male: 58, female: 38, other: 4 },
      countries: [
        { country: 'United States', listeners: 4200, streams: 15800 },
        { country: 'United Kingdom', listeners: 1800, streams: 6500 },
        { country: 'Germany', listeners: 1200, streams: 4200 },
        { country: 'Canada', listeners: 980, streams: 3400 },
        { country: 'Australia', listeners: 750, streams: 2800 },
        { country: 'France', listeners: 620, streams: 2100 },
        { country: 'Netherlands', listeners: 480, streams: 1600 },
        { country: 'Japan', listeners: 420, streams: 1400 },
      ],
      cities: [
        { city: 'Los Angeles', country: 'US', listeners: 1200 },
        { city: 'New York', country: 'US', listeners: 980 },
        { city: 'London', country: 'UK', listeners: 850 },
        { city: 'Berlin', country: 'DE', listeners: 620 },
        { city: 'Toronto', country: 'CA', listeners: 480 },
      ],
      platforms: { web: 35, ios: 38, android: 22, desktop: 5 },
      devices: { phone: 60, computer: 28, tablet: 8, smartSpeaker: 4 },
    });

    // Behavior
    setBehavior({
      avgSessionDuration: 42,
      peakListeningHours: [9, 12, 18, 21, 22],
      peakListeningDays: ['Friday', 'Saturday', 'Sunday'],
      repeatListenerRate: 0.34,
      newListenerRate: 0.28,
      playlistDiscovery: 0.42,
      searchDiscovery: 0.25,
      recommendationDiscovery: 0.33,
    });

    // Revenue
    const revenueHistory = [];
    let cumulative = 0;
    for (let i = 0; i < Math.min(days, 30); i++) {
      const date = new Date(timeRange.start.getTime() + i * 24 * 60 * 60 * 1000);
      const daily = 15 + Math.random() * 25;
      cumulative += daily;
      revenueHistory.push({
        date: date.toISOString().split('T')[0],
        amount: parseFloat(daily.toFixed(2)),
      });
    }

    setRevenue({
      totalRevenue: parseFloat(cumulative.toFixed(2)),
      streamRevenue: parseFloat((cumulative * 0.75).toFixed(2)),
      tipRevenue: parseFloat((cumulative * 0.15).toFixed(2)),
      merchandiseRevenue: parseFloat((cumulative * 0.10).toFixed(2)),
      byPlatform: { Mycelix: 85, 'External Embeds': 10, 'Partner Sites': 5 },
      history: revenueHistory,
      projectedMonthly: parseFloat((cumulative * 1.15).toFixed(2)),
    });

    // Followers
    const followerHistory = [];
    let followerCount = 8500;
    for (let i = 0; i < Math.min(days, 30); i++) {
      const date = new Date(timeRange.start.getTime() + i * 24 * 60 * 60 * 1000);
      followerCount += Math.floor(Math.random() * 50) - 10;
      followerHistory.push({
        date: date.toISOString().split('T')[0],
        count: followerCount,
      });
    }

    setFollowers({
      total: followerCount,
      newThisPeriod: 847,
      lostThisPeriod: 123,
      growth: 8.5,
      history: followerHistory,
    });

    // Playlists
    setPlaylists({
      playlistsIn: 156,
      totalReach: 2400000,
      topPlaylists: [
        { name: 'Electronic Essentials', curator: 'Mycelix Editorial', followers: 850000, position: 12 },
        { name: 'Late Night Beats', curator: 'NightOwl', followers: 320000, position: 3 },
        { name: 'Chill Electronic', curator: 'ChillMaster', followers: 180000, position: 8 },
        { name: 'Summer 2024', curator: 'SummerVibes', followers: 95000, position: 21 },
        { name: 'New Music Friday', curator: 'Mycelix Editorial', followers: 1200000, position: 45 },
      ],
    });

    // Insights
    setInsights([
      { type: 'success', title: 'Viral Track Alert', description: '"Summer Vibes" is trending in Germany with 23% growth this week', metric: '+23%' },
      { type: 'milestone', title: 'Milestone Reached', description: 'You hit 10,000 total streams this month!', metric: '10K' },
      { type: 'tip', title: 'Optimal Release Time', description: 'Your listeners are most active on Friday evenings. Consider releasing new tracks then.', action: 'Schedule Release' },
      { type: 'warning', title: 'High Skip Rate', description: '"Urban Pulse" has an 18% skip rate. Consider adjusting the intro.', metric: '18%', action: 'View Details' },
      { type: 'success', title: 'Playlist Placement', description: 'Added to "Electronic Essentials" reaching 850K potential listeners', metric: '850K' },
    ]);
  }, [timeRange]);

  // Fetch analytics data
  const fetchAnalytics = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // In production, this would call the API
      // For now, generate mock data
      await new Promise(resolve => setTimeout(resolve, 500));
      generateMockData();

      setState(prev => ({
        ...prev,
        isLoading: false,
        lastUpdated: new Date(),
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Failed to load analytics',
      }));
    }
  }, [generateMockData]);

  // Export data
  const exportData = useCallback(async (format: 'csv' | 'json' | 'pdf') => {
    const data = {
      timeRange: { start: timeRange.start.toISOString(), end: timeRange.end.toISOString() },
      summary: { totalStreams, uniqueListeners },
      streamData,
      trackAnalytics,
      demographics,
      behavior,
      revenue,
      followers,
      playlists,
    };

    if (format === 'json') {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      downloadBlob(blob, `analytics-${artistId}-${Date.now()}.json`);
    } else if (format === 'csv') {
      // Convert to CSV (simplified)
      const rows = streamData.map(d =>
        `${d.date},${d.streams},${d.uniqueListeners},${d.avgDuration},${d.skips},${d.completions}`
      );
      const csv = ['Date,Streams,Unique Listeners,Avg Duration,Skips,Completions', ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      downloadBlob(blob, `analytics-${artistId}-${Date.now()}.csv`);
    }
  }, [artistId, timeRange, totalStreams, uniqueListeners, streamData, trackAnalytics, demographics, behavior, revenue, followers, playlists]);

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Compare periods
  const compareToPreviousPeriod = useCallback(() => {
    const periodLength = timeRange.end.getTime() - timeRange.start.getTime();
    return {
      streams: { current: totalStreams, previous: Math.floor(totalStreams * 0.85), change: 17.6 },
      listeners: { current: uniqueListeners, previous: Math.floor(uniqueListeners * 0.9), change: 11.1 },
      revenue: { current: revenue?.totalRevenue || 0, previous: (revenue?.totalRevenue || 0) * 0.88, change: 13.6 },
    };
  }, [timeRange, totalStreams, uniqueListeners, revenue]);

  // Initial fetch
  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  // Refetch on time range change
  useEffect(() => {
    fetchAnalytics();
  }, [timeRange, fetchAnalytics]);

  return {
    // State
    state,
    timeRange,

    // Data
    streamData,
    totalStreams,
    uniqueListeners,
    trackAnalytics,
    demographics,
    behavior,
    revenue,
    followers,
    playlists,
    insights,

    // Actions
    setTimeRange,
    refresh: fetchAnalytics,
    exportData,
    compareToPreviousPeriod,

    // Constants
    TIME_RANGES,
  };
}
