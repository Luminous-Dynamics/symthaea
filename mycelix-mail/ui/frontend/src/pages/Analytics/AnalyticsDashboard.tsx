// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Analytics Dashboard
 *
 * Email metrics, productivity insights, and activity visualizations
 */

import React, { useEffect, useState } from 'react';

interface EmailVolumeStats {
  period: string;
  received: DataPoint[];
  sent: DataPoint[];
  totalReceived: number;
  totalSent: number;
}

interface DataPoint {
  date: string;
  value: number;
}

interface ResponseTimeStats {
  averageSeconds: number;
  distribution: Record<string, number>;
}

interface TopCorrespondent {
  email: string;
  name?: string;
  emailCount: number;
}

interface ProductivityInsights {
  emailsReceivedThisWeek: number;
  emailsSentThisWeek: number;
  receivedVsLastWeek: number;
  sentVsLastWeek: number;
  currentUnread: number;
  inboxZeroDaysThisMonth: number;
}

interface FolderStats {
  folder: string;
  totalCount: number;
  unreadCount: number;
  totalSize?: number;
}

interface ActivityHeatmap {
  received: number[][];
  sent: number[][];
  peakReceiveHour: number;
  peakSendHour: number;
}

type Period = 'day' | 'week' | 'month' | 'quarter' | 'year';

export default function AnalyticsDashboard() {
  const [period, setPeriod] = useState<Period>('week');
  const [volumeStats, setVolumeStats] = useState<EmailVolumeStats | null>(null);
  const [responseStats, setResponseStats] = useState<ResponseTimeStats | null>(null);
  const [topSenders, setTopSenders] = useState<TopCorrespondent[]>([]);
  const [topRecipients, setTopRecipients] = useState<TopCorrespondent[]>([]);
  const [insights, setInsights] = useState<ProductivityInsights | null>(null);
  const [folderStats, setFolderStats] = useState<FolderStats[]>([]);
  const [heatmap, setHeatmap] = useState<ActivityHeatmap | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAllAnalytics();
  }, [period]);

  async function fetchAllAnalytics() {
    setLoading(true);
    try {
      await Promise.all([
        fetchVolumeStats(),
        fetchResponseStats(),
        fetchTopCorrespondents(),
        fetchInsights(),
        fetchFolderStats(),
        fetchHeatmap(),
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function fetchVolumeStats() {
    const response = await fetch(`/api/analytics/volume?period=${period}`);
    if (response.ok) setVolumeStats(await response.json());
  }

  async function fetchResponseStats() {
    const response = await fetch(`/api/analytics/response-times?period=${period}`);
    if (response.ok) setResponseStats(await response.json());
  }

  async function fetchTopCorrespondents() {
    const response = await fetch('/api/analytics/correspondents?limit=10');
    if (response.ok) {
      const data = await response.json();
      setTopSenders(data.topSenders);
      setTopRecipients(data.topRecipients);
    }
  }

  async function fetchInsights() {
    const response = await fetch('/api/analytics/insights');
    if (response.ok) setInsights(await response.json());
  }

  async function fetchFolderStats() {
    const response = await fetch('/api/analytics/folders');
    if (response.ok) setFolderStats(await response.json());
  }

  async function fetchHeatmap() {
    const response = await fetch('/api/analytics/heatmap');
    if (response.ok) setHeatmap(await response.json());
  }

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
    return `${Math.round(seconds / 86400)}d`;
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  function getChangeIndicator(value: number): { text: string; color: string } {
    if (value > 0) return { text: `+${value.toFixed(1)}%`, color: 'text-green-600' };
    if (value < 0) return { text: `${value.toFixed(1)}%`, color: 'text-red-600' };
    return { text: '0%', color: 'text-gray-500' };
  }

  const dayLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Email Analytics</h1>
          <p className="text-muted">Insights into your email productivity</p>
        </div>
        <div className="flex items-center gap-2">
          {(['day', 'week', 'month', 'quarter', 'year'] as Period[]).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1.5 rounded text-sm capitalize ${
                period === p ? 'bg-primary text-white' : 'hover:bg-muted/30'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Productivity Insights Cards */}
      {insights && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="p-4 border border-border rounded-lg">
            <p className="text-sm text-muted">Received This Week</p>
            <p className="text-3xl font-bold">{insights.emailsReceivedThisWeek}</p>
            <p className={getChangeIndicator(insights.receivedVsLastWeek).color}>
              {getChangeIndicator(insights.receivedVsLastWeek).text} vs last week
            </p>
          </div>
          <div className="p-4 border border-border rounded-lg">
            <p className="text-sm text-muted">Sent This Week</p>
            <p className="text-3xl font-bold">{insights.emailsSentThisWeek}</p>
            <p className={getChangeIndicator(insights.sentVsLastWeek).color}>
              {getChangeIndicator(insights.sentVsLastWeek).text} vs last week
            </p>
          </div>
          <div className="p-4 border border-border rounded-lg">
            <p className="text-sm text-muted">Current Unread</p>
            <p className="text-3xl font-bold">{insights.currentUnread}</p>
            <p className="text-muted text-sm">emails awaiting attention</p>
          </div>
          <div className="p-4 border border-border rounded-lg">
            <p className="text-sm text-muted">Inbox Zero Days</p>
            <p className="text-3xl font-bold">{insights.inboxZeroDaysThisMonth}</p>
            <p className="text-muted text-sm">this month</p>
          </div>
        </div>
      )}

      {/* Response Time */}
      {responseStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Average Response Time</h2>
            <p className="text-4xl font-bold text-primary">
              {formatDuration(responseStats.averageSeconds)}
            </p>
            <p className="text-muted">to reply to emails</p>
          </div>
          <div className="border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Response Time Distribution</h2>
            <div className="space-y-2">
              {Object.entries(responseStats.distribution).map(([bucket, count]) => {
                const labels: Record<string, string> = {
                  under_1h: 'Under 1 hour',
                  '1h_to_4h': '1-4 hours',
                  '4h_to_24h': '4-24 hours',
                  over_24h: 'Over 24 hours',
                };
                const total = Object.values(responseStats.distribution).reduce((a, b) => a + b, 0);
                const percentage = total > 0 ? (count / total) * 100 : 0;

                return (
                  <div key={bucket} className="flex items-center gap-3">
                    <span className="w-24 text-sm text-muted">{labels[bucket] || bucket}</span>
                    <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <span className="w-16 text-sm text-right">{count} ({percentage.toFixed(0)}%)</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Email Volume Chart */}
      {volumeStats && (
        <div className="border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Email Volume</h2>
          <div className="flex items-center gap-6 mb-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded" />
              <span className="text-sm">Received ({volumeStats.totalReceived})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded" />
              <span className="text-sm">Sent ({volumeStats.totalSent})</span>
            </div>
          </div>
          <div className="h-48 flex items-end gap-1">
            {volumeStats.received.map((point, i) => {
              const maxValue = Math.max(
                ...volumeStats.received.map((p) => p.value),
                ...volumeStats.sent.map((p) => p.value)
              );
              const receivedHeight = maxValue > 0 ? (point.value / maxValue) * 100 : 0;
              const sentHeight = volumeStats.sent[i]
                ? (volumeStats.sent[i].value / maxValue) * 100
                : 0;

              return (
                <div key={i} className="flex-1 flex gap-0.5" title={point.date}>
                  <div
                    className="flex-1 bg-blue-500 rounded-t"
                    style={{ height: `${receivedHeight}%` }}
                  />
                  <div
                    className="flex-1 bg-green-500 rounded-t"
                    style={{ height: `${sentHeight}%` }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Activity Heatmap */}
      {heatmap && (
        <div className="border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Email Activity Heatmap</h2>
          <p className="text-sm text-muted mb-4">
            Peak receiving hour: {heatmap.peakReceiveHour}:00 |
            Peak sending hour: {heatmap.peakSendHour}:00
          </p>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="w-12"></th>
                  {Array.from({ length: 24 }, (_, i) => (
                    <th key={i} className="text-xs text-muted font-normal px-1">
                      {i}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dayLabels.map((day, dayIndex) => (
                  <tr key={day}>
                    <td className="text-xs text-muted pr-2">{day}</td>
                    {Array.from({ length: 24 }, (_, hour) => {
                      const value = heatmap.received[dayIndex]?.[hour] || 0;
                      const maxValue = Math.max(...heatmap.received.flat());
                      const intensity = maxValue > 0 ? value / maxValue : 0;

                      return (
                        <td key={hour} className="p-0.5">
                          <div
                            className="w-4 h-4 rounded-sm"
                            style={{
                              backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                            }}
                            title={`${day} ${hour}:00 - ${value} emails`}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Top Correspondents */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Top Senders</h2>
          <div className="space-y-3">
            {topSenders.map((sender, i) => (
              <div key={sender.email} className="flex items-center gap-3">
                <span className="w-6 text-muted text-sm">{i + 1}</span>
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                  {(sender.name || sender.email)[0].toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{sender.name || sender.email}</p>
                  <p className="text-xs text-muted truncate">{sender.email}</p>
                </div>
                <span className="text-sm font-medium">{sender.emailCount}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Top Recipients</h2>
          <div className="space-y-3">
            {topRecipients.map((recipient, i) => (
              <div key={recipient.email} className="flex items-center gap-3">
                <span className="w-6 text-muted text-sm">{i + 1}</span>
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                  {recipient.email[0].toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{recipient.email}</p>
                </div>
                <span className="text-sm font-medium">{recipient.emailCount}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Folder Distribution */}
      <div className="border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Storage by Folder</h2>
        <div className="space-y-3">
          {folderStats.map((folder) => {
            const totalSize = folderStats.reduce((sum, f) => sum + (f.totalSize || 0), 0);
            const percentage = totalSize > 0 ? ((folder.totalSize || 0) / totalSize) * 100 : 0;

            return (
              <div key={folder.folder} className="flex items-center gap-4">
                <span className="w-24 font-medium capitalize">{folder.folder}</span>
                <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div className="w-32 text-right text-sm">
                  <span className="font-medium">{folder.totalCount}</span>
                  <span className="text-muted"> emails</span>
                </div>
                <div className="w-24 text-right text-sm text-muted">
                  {folder.totalSize ? formatBytes(folder.totalSize) : '-'}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
