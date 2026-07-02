// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AnalyticsDashboard Component
 *
 * Comprehensive artist analytics dashboard with
 * charts, demographics, insights, and export
 */

import React, { useState, useMemo } from 'react';
import { useArtistAnalytics, TimeRange, ArtistInsight } from '../../hooks/useArtistAnalytics';

interface AnalyticsDashboardProps {
  artistId: string;
  className?: string;
}

export function AnalyticsDashboard({ artistId, className = '' }: AnalyticsDashboardProps) {
  const {
    state,
    timeRange,
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
    setTimeRange,
    refresh,
    exportData,
    compareToPreviousPeriod,
    TIME_RANGES,
  } = useArtistAnalytics(artistId);

  const [activeTab, setActiveTab] = useState<'overview' | 'tracks' | 'audience' | 'revenue'>('overview');

  const comparison = useMemo(() => compareToPreviousPeriod(), [compareToPreviousPeriod]);

  // Format number with abbreviation
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  // Format currency
  const formatCurrency = (num: number): string => {
    return `$${num.toFixed(2)}`;
  };

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Simple bar chart component
  const BarChart = ({ data, maxValue, color }: { data: number[]; maxValue: number; color: string }) => (
    <div style={styles.barChart}>
      {data.map((value, idx) => (
        <div
          key={idx}
          style={{
            ...styles.bar,
            height: `${(value / maxValue) * 100}%`,
            backgroundColor: color,
          }}
        />
      ))}
    </div>
  );

  // Insight card component
  const InsightCard = ({ insight }: { insight: ArtistInsight }) => {
    const colors = {
      success: '#10b981',
      warning: '#f59e0b',
      tip: '#3b82f6',
      milestone: '#8b5cf6',
    };

    const icons = {
      success: '🎉',
      warning: '⚠️',
      tip: '💡',
      milestone: '🏆',
    };

    return (
      <div style={{ ...styles.insightCard, borderLeftColor: colors[insight.type] }}>
        <div style={styles.insightHeader}>
          <span style={styles.insightIcon}>{icons[insight.type]}</span>
          <span style={styles.insightTitle}>{insight.title}</span>
          {insight.metric && (
            <span style={{ ...styles.insightMetric, color: colors[insight.type] }}>
              {insight.metric}
            </span>
          )}
        </div>
        <p style={styles.insightDesc}>{insight.description}</p>
        {insight.action && (
          <button style={styles.insightAction}>{insight.action}</button>
        )}
      </div>
    );
  };

  if (state.isLoading && !streamData.length) {
    return (
      <div className={className} style={styles.container}>
        <div style={styles.loading}>
          <div style={styles.spinner} />
          <p>Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={className} style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h2 style={styles.title}>Analytics Dashboard</h2>
          {state.lastUpdated && (
            <p style={styles.lastUpdated}>
              Last updated: {state.lastUpdated.toLocaleTimeString()}
            </p>
          )}
        </div>
        <div style={styles.headerActions}>
          <select
            value={Object.keys(TIME_RANGES).find(k => TIME_RANGES[k].label === timeRange.label)}
            onChange={(e) => setTimeRange(TIME_RANGES[e.target.value])}
            style={styles.select}
          >
            {Object.entries(TIME_RANGES).map(([key, range]) => (
              <option key={key} value={key}>{range.label}</option>
            ))}
          </select>
          <button onClick={refresh} style={styles.refreshBtn} disabled={state.isLoading}>
            {state.isLoading ? '...' : '↻'}
          </button>
          <button onClick={() => exportData('csv')} style={styles.exportBtn}>
            Export
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        {(['overview', 'tracks', 'audience', 'revenue'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              ...styles.tab,
              backgroundColor: activeTab === tab ? '#374151' : 'transparent',
              color: activeTab === tab ? '#fff' : '#9ca3af',
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          {/* Key Metrics */}
          <div style={styles.metricsGrid}>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Total Streams</span>
              <span style={styles.metricValue}>{formatNumber(totalStreams)}</span>
              <span style={{ ...styles.metricChange, color: comparison.streams.change >= 0 ? '#10b981' : '#ef4444' }}>
                {comparison.streams.change >= 0 ? '↑' : '↓'} {Math.abs(comparison.streams.change).toFixed(1)}%
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Unique Listeners</span>
              <span style={styles.metricValue}>{formatNumber(uniqueListeners)}</span>
              <span style={{ ...styles.metricChange, color: comparison.listeners.change >= 0 ? '#10b981' : '#ef4444' }}>
                {comparison.listeners.change >= 0 ? '↑' : '↓'} {Math.abs(comparison.listeners.change).toFixed(1)}%
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Followers</span>
              <span style={styles.metricValue}>{formatNumber(followers?.total || 0)}</span>
              <span style={{ ...styles.metricChange, color: (followers?.growth || 0) >= 0 ? '#10b981' : '#ef4444' }}>
                {(followers?.growth || 0) >= 0 ? '↑' : '↓'} {Math.abs(followers?.growth || 0).toFixed(1)}%
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Revenue</span>
              <span style={styles.metricValue}>{formatCurrency(revenue?.totalRevenue || 0)}</span>
              <span style={{ ...styles.metricChange, color: comparison.revenue.change >= 0 ? '#10b981' : '#ef4444' }}>
                {comparison.revenue.change >= 0 ? '↑' : '↓'} {Math.abs(comparison.revenue.change).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Stream Chart */}
          <div style={styles.chartCard}>
            <h3 style={styles.chartTitle}>Streams Over Time</h3>
            <div style={styles.chartContainer}>
              <BarChart
                data={streamData.map(d => d.streams)}
                maxValue={Math.max(...streamData.map(d => d.streams))}
                color="#818cf8"
              />
            </div>
            <div style={styles.chartLegend}>
              {streamData.length > 0 && (
                <>
                  <span>{streamData[0].date}</span>
                  <span>{streamData[streamData.length - 1].date}</span>
                </>
              )}
            </div>
          </div>

          {/* Insights */}
          <div style={styles.insightsSection}>
            <h3 style={styles.sectionTitle}>Insights & Recommendations</h3>
            <div style={styles.insightsList}>
              {insights.map((insight, idx) => (
                <InsightCard key={idx} insight={insight} />
              ))}
            </div>
          </div>

          {/* Top Tracks */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Top Performing Tracks</h3>
            <div style={styles.tracksList}>
              {trackAnalytics.slice(0, 5).map((track, idx) => (
                <div key={track.trackId} style={styles.trackItem}>
                  <span style={styles.trackRank}>{idx + 1}</span>
                  <div style={styles.trackInfo}>
                    <span style={styles.trackTitle}>{track.title}</span>
                    <span style={styles.trackMeta}>
                      {formatNumber(track.totalStreams)} streams
                    </span>
                  </div>
                  <span
                    style={{
                      ...styles.trackTrend,
                      color: track.trend === 'up' ? '#10b981' : track.trend === 'down' ? '#ef4444' : '#9ca3af',
                    }}
                  >
                    {track.trend === 'up' ? '↑' : track.trend === 'down' ? '↓' : '→'}
                    {Math.abs(track.trendPercent)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Tracks Tab */}
      {activeTab === 'tracks' && (
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>Track Analytics</h3>
          <div style={styles.tracksTable}>
            <div style={styles.tableHeader}>
              <span style={{ flex: 2 }}>Track</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Streams</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Listeners</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Avg Duration</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Skip Rate</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Saves</span>
              <span style={{ flex: 1, textAlign: 'center' }}>Trend</span>
            </div>
            {trackAnalytics.map(track => (
              <div key={track.trackId} style={styles.tableRow}>
                <span style={{ flex: 2, fontWeight: 500 }}>{track.title}</span>
                <span style={{ flex: 1, textAlign: 'center' }}>{formatNumber(track.totalStreams)}</span>
                <span style={{ flex: 1, textAlign: 'center' }}>{formatNumber(track.uniqueListeners)}</span>
                <span style={{ flex: 1, textAlign: 'center' }}>{formatDuration(track.avgListenDuration)}</span>
                <span style={{ flex: 1, textAlign: 'center', color: track.skipRate > 0.15 ? '#ef4444' : '#10b981' }}>
                  {(track.skipRate * 100).toFixed(0)}%
                </span>
                <span style={{ flex: 1, textAlign: 'center' }}>{(track.saveRate * 100).toFixed(0)}%</span>
                <span
                  style={{
                    flex: 1,
                    textAlign: 'center',
                    color: track.trend === 'up' ? '#10b981' : track.trend === 'down' ? '#ef4444' : '#9ca3af',
                  }}
                >
                  {track.trend === 'up' ? '↑' : track.trend === 'down' ? '↓' : '→'} {Math.abs(track.trendPercent)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Audience Tab */}
      {activeTab === 'audience' && demographics && behavior && (
        <>
          <div style={styles.gridTwo}>
            {/* Age Distribution */}
            <div style={styles.chartCard}>
              <h3 style={styles.chartTitle}>Age Distribution</h3>
              <div style={styles.distributionList}>
                {Object.entries(demographics.ageGroups).map(([age, percent]) => (
                  <div key={age} style={styles.distributionItem}>
                    <span style={styles.distributionLabel}>{age}</span>
                    <div style={styles.distributionBar}>
                      <div style={{ ...styles.distributionFill, width: `${percent}%` }} />
                    </div>
                    <span style={styles.distributionValue}>{percent}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Gender Distribution */}
            <div style={styles.chartCard}>
              <h3 style={styles.chartTitle}>Gender Distribution</h3>
              <div style={styles.distributionList}>
                {Object.entries(demographics.gender).map(([gender, percent]) => (
                  <div key={gender} style={styles.distributionItem}>
                    <span style={styles.distributionLabel}>{gender.charAt(0).toUpperCase() + gender.slice(1)}</span>
                    <div style={styles.distributionBar}>
                      <div
                        style={{
                          ...styles.distributionFill,
                          width: `${percent}%`,
                          backgroundColor: gender === 'male' ? '#3b82f6' : gender === 'female' ? '#ec4899' : '#9ca3af',
                        }}
                      />
                    </div>
                    <span style={styles.distributionValue}>{percent}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Countries */}
          <div style={styles.chartCard}>
            <h3 style={styles.chartTitle}>Top Countries</h3>
            <div style={styles.countriesList}>
              {demographics.countries.slice(0, 8).map((country, idx) => (
                <div key={country.country} style={styles.countryItem}>
                  <span style={styles.countryRank}>{idx + 1}</span>
                  <span style={styles.countryName}>{country.country}</span>
                  <span style={styles.countryListeners}>{formatNumber(country.listeners)} listeners</span>
                  <span style={styles.countryStreams}>{formatNumber(country.streams)} streams</span>
                </div>
              ))}
            </div>
          </div>

          {/* Listening Behavior */}
          <div style={styles.gridTwo}>
            <div style={styles.chartCard}>
              <h3 style={styles.chartTitle}>Discovery Sources</h3>
              <div style={styles.distributionList}>
                <div style={styles.distributionItem}>
                  <span style={styles.distributionLabel}>Playlists</span>
                  <div style={styles.distributionBar}>
                    <div style={{ ...styles.distributionFill, width: `${behavior.playlistDiscovery * 100}%`, backgroundColor: '#8b5cf6' }} />
                  </div>
                  <span style={styles.distributionValue}>{(behavior.playlistDiscovery * 100).toFixed(0)}%</span>
                </div>
                <div style={styles.distributionItem}>
                  <span style={styles.distributionLabel}>Recommendations</span>
                  <div style={styles.distributionBar}>
                    <div style={{ ...styles.distributionFill, width: `${behavior.recommendationDiscovery * 100}%`, backgroundColor: '#10b981' }} />
                  </div>
                  <span style={styles.distributionValue}>{(behavior.recommendationDiscovery * 100).toFixed(0)}%</span>
                </div>
                <div style={styles.distributionItem}>
                  <span style={styles.distributionLabel}>Search</span>
                  <div style={styles.distributionBar}>
                    <div style={{ ...styles.distributionFill, width: `${behavior.searchDiscovery * 100}%`, backgroundColor: '#3b82f6' }} />
                  </div>
                  <span style={styles.distributionValue}>{(behavior.searchDiscovery * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            <div style={styles.chartCard}>
              <h3 style={styles.chartTitle}>Platform Usage</h3>
              <div style={styles.distributionList}>
                {Object.entries(demographics.platforms).map(([platform, percent]) => (
                  <div key={platform} style={styles.distributionItem}>
                    <span style={styles.distributionLabel}>{platform.charAt(0).toUpperCase() + platform.slice(1)}</span>
                    <div style={styles.distributionBar}>
                      <div style={{ ...styles.distributionFill, width: `${percent}%` }} />
                    </div>
                    <span style={styles.distributionValue}>{percent}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Revenue Tab */}
      {activeTab === 'revenue' && revenue && (
        <>
          {/* Revenue Summary */}
          <div style={styles.metricsGrid}>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Total Revenue</span>
              <span style={styles.metricValue}>{formatCurrency(revenue.totalRevenue)}</span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Stream Revenue</span>
              <span style={styles.metricValue}>{formatCurrency(revenue.streamRevenue)}</span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Tips</span>
              <span style={styles.metricValue}>{formatCurrency(revenue.tipRevenue)}</span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Projected Monthly</span>
              <span style={styles.metricValue}>{formatCurrency(revenue.projectedMonthly)}</span>
            </div>
          </div>

          {/* Revenue Chart */}
          <div style={styles.chartCard}>
            <h3 style={styles.chartTitle}>Revenue Over Time</h3>
            <div style={styles.chartContainer}>
              <BarChart
                data={revenue.history.map(d => d.amount)}
                maxValue={Math.max(...revenue.history.map(d => d.amount))}
                color="#10b981"
              />
            </div>
          </div>

          {/* Playlist Placements */}
          {playlists && (
            <div style={styles.chartCard}>
              <h3 style={styles.chartTitle}>Playlist Placements</h3>
              <div style={styles.playlistStats}>
                <div style={styles.playlistStat}>
                  <span style={styles.playlistStatValue}>{playlists.playlistsIn}</span>
                  <span style={styles.playlistStatLabel}>Playlists</span>
                </div>
                <div style={styles.playlistStat}>
                  <span style={styles.playlistStatValue}>{formatNumber(playlists.totalReach)}</span>
                  <span style={styles.playlistStatLabel}>Total Reach</span>
                </div>
              </div>
              <div style={styles.topPlaylists}>
                {playlists.topPlaylists.map((pl, idx) => (
                  <div key={idx} style={styles.playlistItem}>
                    <div style={styles.playlistInfo}>
                      <span style={styles.playlistName}>{pl.name}</span>
                      <span style={styles.playlistCurator}>by {pl.curator}</span>
                    </div>
                    <span style={styles.playlistFollowers}>{formatNumber(pl.followers)}</span>
                    <span style={styles.playlistPosition}>#{pl.position}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1f2937',
    borderRadius: '12px',
    padding: '24px',
    color: '#f9fafb',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  loading: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '48px',
    gap: '16px',
  },
  spinner: {
    width: '40px',
    height: '40px',
    border: '4px solid #374151',
    borderTopColor: '#818cf8',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '24px',
  },
  title: {
    fontSize: '24px',
    fontWeight: 700,
    margin: 0,
  },
  lastUpdated: {
    fontSize: '12px',
    color: '#6b7280',
    margin: '4px 0 0 0',
  },
  headerActions: {
    display: 'flex',
    gap: '8px',
  },
  select: {
    padding: '8px 12px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '6px',
    color: '#f9fafb',
    fontSize: '13px',
  },
  refreshBtn: {
    padding: '8px 12px',
    backgroundColor: '#374151',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '16px',
    cursor: 'pointer',
  },
  exportBtn: {
    padding: '8px 16px',
    backgroundColor: '#818cf8',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  tabs: {
    display: 'flex',
    gap: '4px',
    marginBottom: '24px',
    backgroundColor: '#111827',
    padding: '4px',
    borderRadius: '8px',
  },
  tab: {
    flex: 1,
    padding: '10px 16px',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '16px',
    marginBottom: '24px',
  },
  metricCard: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  metricLabel: {
    fontSize: '12px',
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  metricValue: {
    fontSize: '28px',
    fontWeight: 700,
    color: '#fff',
  },
  metricChange: {
    fontSize: '13px',
    fontWeight: 500,
  },
  chartCard: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '16px',
  },
  chartTitle: {
    fontSize: '14px',
    fontWeight: 600,
    margin: '0 0 16px 0',
  },
  chartContainer: {
    height: '120px',
    display: 'flex',
    alignItems: 'flex-end',
  },
  chartLegend: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '8px',
    fontSize: '11px',
    color: '#6b7280',
  },
  barChart: {
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'flex-end',
    gap: '2px',
  },
  bar: {
    flex: 1,
    borderRadius: '2px 2px 0 0',
    minWidth: '4px',
    transition: 'height 0.3s',
  },
  section: {
    marginBottom: '24px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: 600,
    margin: '0 0 16px 0',
  },
  insightsSection: {
    marginBottom: '24px',
  },
  insightsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  insightCard: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '16px',
    borderLeft: '4px solid',
  },
  insightHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
  },
  insightIcon: {
    fontSize: '16px',
  },
  insightTitle: {
    flex: 1,
    fontWeight: 600,
    fontSize: '14px',
  },
  insightMetric: {
    fontWeight: 700,
    fontSize: '14px',
  },
  insightDesc: {
    margin: 0,
    fontSize: '13px',
    color: '#9ca3af',
  },
  insightAction: {
    marginTop: '12px',
    padding: '6px 12px',
    backgroundColor: '#374151',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '12px',
    cursor: 'pointer',
  },
  tracksList: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '8px',
  },
  trackItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '12px',
    borderRadius: '6px',
  },
  trackRank: {
    width: '24px',
    height: '24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#374151',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: 600,
  },
  trackInfo: {
    flex: 1,
  },
  trackTitle: {
    display: 'block',
    fontWeight: 500,
    fontSize: '14px',
  },
  trackMeta: {
    fontSize: '12px',
    color: '#6b7280',
  },
  trackTrend: {
    fontSize: '13px',
    fontWeight: 500,
  },
  tracksTable: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  tableHeader: {
    display: 'flex',
    padding: '12px 16px',
    backgroundColor: '#0f172a',
    fontSize: '11px',
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  tableRow: {
    display: 'flex',
    padding: '12px 16px',
    borderBottom: '1px solid #1f2937',
    fontSize: '13px',
  },
  gridTwo: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
    marginBottom: '16px',
  },
  distributionList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  distributionItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  distributionLabel: {
    width: '80px',
    fontSize: '12px',
    color: '#9ca3af',
  },
  distributionBar: {
    flex: 1,
    height: '8px',
    backgroundColor: '#374151',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  distributionFill: {
    height: '100%',
    backgroundColor: '#818cf8',
    borderRadius: '4px',
  },
  distributionValue: {
    width: '40px',
    textAlign: 'right',
    fontSize: '12px',
    fontWeight: 500,
  },
  countriesList: {
    display: 'flex',
    flexDirection: 'column',
  },
  countryItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '10px 0',
    borderBottom: '1px solid #1f2937',
    gap: '16px',
  },
  countryRank: {
    width: '24px',
    textAlign: 'center',
    fontSize: '12px',
    color: '#6b7280',
  },
  countryName: {
    flex: 1,
    fontWeight: 500,
    fontSize: '13px',
  },
  countryListeners: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  countryStreams: {
    fontSize: '12px',
    color: '#818cf8',
  },
  playlistStats: {
    display: 'flex',
    gap: '32px',
    marginBottom: '20px',
  },
  playlistStat: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  playlistStatValue: {
    fontSize: '24px',
    fontWeight: 700,
    color: '#818cf8',
  },
  playlistStatLabel: {
    fontSize: '12px',
    color: '#6b7280',
  },
  topPlaylists: {
    display: 'flex',
    flexDirection: 'column',
  },
  playlistItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '12px 0',
    borderBottom: '1px solid #1f2937',
    gap: '16px',
  },
  playlistInfo: {
    flex: 1,
  },
  playlistName: {
    display: 'block',
    fontWeight: 500,
    fontSize: '13px',
  },
  playlistCurator: {
    fontSize: '11px',
    color: '#6b7280',
  },
  playlistFollowers: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  playlistPosition: {
    fontSize: '12px',
    color: '#818cf8',
    fontWeight: 500,
  },
};

export default AnalyticsDashboard;
