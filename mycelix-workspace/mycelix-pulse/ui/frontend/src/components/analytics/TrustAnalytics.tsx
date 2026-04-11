// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Analytics Dashboard
 *
 * Visual analytics for trust network health:
 * - Trust score trends over time
 * - Network growth charts
 * - Attestation activity
 * - Tier distribution
 * - Relationship breakdown
 */

import { useMemo, useState, useCallback } from 'react';
import { Counter, ProgressBar } from '../ui/Animations';

// ============================================
// Types
// ============================================

interface DataPoint {
  date: string;
  value: number;
  label?: string;
}

interface TrustMetrics {
  currentScore: number;
  previousScore: number;
  trend: 'up' | 'down' | 'stable';
  totalConnections: number;
  directConnections: number;
  transitiveConnections: number;
  attestationsGiven: number;
  attestationsReceived: number;
  averagePathLength: number;
}

interface AnalyticsData {
  metrics: TrustMetrics;
  scoreHistory: DataPoint[];
  networkGrowth: DataPoint[];
  attestationActivity: DataPoint[];
  tierDistribution: { tier: number; count: number; percentage: number }[];
  relationshipBreakdown: { type: string; count: number; percentage: number }[];
}

// ============================================
// Simple Chart Components (no external deps)
// ============================================

interface LineChartProps {
  data: DataPoint[];
  height?: number;
  color?: string;
  showArea?: boolean;
  showDots?: boolean;
  showLabels?: boolean;
  className?: string;
}

function LineChart({
  data,
  height = 200,
  color = '#3b82f6',
  showArea = true,
  showDots = true,
  showLabels = false,
  className = '',
}: LineChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const { path, areaPath, points, minY, maxY } = useMemo(() => {
    if (data.length === 0) return { path: '', areaPath: '', points: [], minY: 0, maxY: 100 };

    const values = data.map((d) => d.value);
    const minY = Math.min(...values) * 0.9;
    const maxY = Math.max(...values) * 1.1;
    const rangeY = maxY - minY || 1;

    const width = 100;
    const padding = 5;

    const points = data.map((d, i) => ({
      x: padding + ((width - 2 * padding) * i) / Math.max(1, data.length - 1),
      y: height - padding - ((d.value - minY) / rangeY) * (height - 2 * padding),
      data: d,
    }));

    const path = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
      .join(' ');

    const areaPath = `${path} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`;

    return { path, areaPath, points, minY, maxY };
  }, [data, height]);

  if (data.length === 0) {
    return (
      <div className={`flex items-center justify-center ${className}`} style={{ height }}>
        <span className="text-gray-400">No data available</span>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <svg viewBox={`0 0 100 ${height}`} className="w-full" preserveAspectRatio="none">
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((y) => (
          <line
            key={y}
            x1="5"
            y1={5 + ((height - 10) * (100 - y)) / 100}
            x2="95"
            y2={5 + ((height - 10) * (100 - y)) / 100}
            stroke="currentColor"
            strokeOpacity="0.1"
            strokeWidth="0.5"
          />
        ))}

        {/* Area */}
        {showArea && (
          <path
            d={areaPath}
            fill={color}
            fillOpacity="0.1"
          />
        )}

        {/* Line */}
        <path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />

        {/* Dots */}
        {showDots &&
          points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={hoveredIndex === i ? 4 : 2}
              fill={color}
              className="transition-all duration-150 cursor-pointer"
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
          ))}
      </svg>

      {/* Tooltip */}
      {hoveredIndex !== null && points[hoveredIndex] && (
        <div
          className="absolute px-2 py-1 bg-gray-900 text-white text-xs rounded shadow-lg pointer-events-none z-10"
          style={{
            left: `${points[hoveredIndex].x}%`,
            top: points[hoveredIndex].y - 30,
            transform: 'translateX(-50%)',
          }}
        >
          <div className="font-medium">{points[hoveredIndex].data.value.toFixed(1)}</div>
          <div className="text-gray-400">{points[hoveredIndex].data.label || points[hoveredIndex].data.date}</div>
        </div>
      )}

      {/* X-axis labels */}
      {showLabels && data.length > 0 && (
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>{data[0].date}</span>
          <span>{data[data.length - 1].date}</span>
        </div>
      )}
    </div>
  );
}

interface BarChartProps {
  data: { label: string; value: number; color?: string }[];
  height?: number;
  showValues?: boolean;
  className?: string;
}

function BarChart({
  data,
  height = 200,
  showValues = true,
  className = '',
}: BarChartProps) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className={`flex items-end gap-2 ${className}`} style={{ height }}>
      {data.map((item, i) => {
        const barHeight = (item.value / maxValue) * 100;
        const color = item.color || `hsl(${(i * 360) / data.length}, 70%, 50%)`;

        return (
          <div
            key={item.label}
            className="flex-1 flex flex-col items-center"
          >
            <div
              className="w-full rounded-t transition-all duration-500 ease-out"
              style={{
                height: `${barHeight}%`,
                backgroundColor: color,
                minHeight: 4,
              }}
            />
            {showValues && (
              <span className="mt-1 text-xs font-medium text-gray-700 dark:text-gray-300">
                {item.value}
              </span>
            )}
            <span className="mt-0.5 text-xs text-gray-500 truncate max-w-full">
              {item.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

interface DonutChartProps {
  data: { label: string; value: number; color: string }[];
  size?: number;
  thickness?: number;
  className?: string;
}

function DonutChart({
  data,
  size = 200,
  thickness = 40,
  className = '',
}: DonutChartProps) {
  const total = data.reduce((sum, d) => sum + d.value, 0) || 1;
  const radius = (size - thickness) / 2;
  const circumference = 2 * Math.PI * radius;

  let currentOffset = 0;

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} className="transform -rotate-90">
        {data.map((item, i) => {
          const percentage = item.value / total;
          const strokeLength = circumference * percentage;
          const offset = currentOffset;
          currentOffset += strokeLength;

          return (
            <circle
              key={item.label}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={item.color}
              strokeWidth={thickness}
              strokeDasharray={`${strokeLength} ${circumference - strokeLength}`}
              strokeDashoffset={-offset}
              className="transition-all duration-500"
            />
          );
        })}
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {total}
          </div>
          <div className="text-xs text-gray-500">Total</div>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Stat Cards
// ============================================

interface StatCardProps {
  title: string;
  value: number | string;
  change?: number;
  icon?: string;
  color?: string;
}

function StatCard({ title, value, change, icon, color = 'blue' }: StatCardProps) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
    emerald: 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400',
    amber: 'bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400',
    purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3">
        {icon && (
          <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
            <span className="text-xl">{icon}</span>
          </div>
        )}
        <div className="flex-1">
          <div className="text-sm text-gray-500 dark:text-gray-400">{title}</div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {typeof value === 'number' ? <Counter value={value} /> : value}
          </div>
        </div>
        {change !== undefined && (
          <div
            className={`flex items-center gap-1 text-sm font-medium ${
              change > 0
                ? 'text-emerald-600 dark:text-emerald-400'
                : change < 0
                ? 'text-red-600 dark:text-red-400'
                : 'text-gray-500'
            }`}
          >
            {change > 0 ? '↑' : change < 0 ? '↓' : '→'}
            {Math.abs(change)}%
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// Main Analytics Dashboard
// ============================================

interface TrustAnalyticsDashboardProps {
  data?: AnalyticsData;
  isLoading?: boolean;
  className?: string;
}

export function TrustAnalyticsDashboard({
  data,
  isLoading,
  className = '',
}: TrustAnalyticsDashboardProps) {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');

  // Generate mock data if none provided
  const analyticsData = useMemo((): AnalyticsData => {
    if (data) return data;

    // Generate mock data for demo
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : timeRange === '90d' ? 90 : 365;
    const scoreHistory: DataPoint[] = [];
    const networkGrowth: DataPoint[] = [];
    const attestationActivity: DataPoint[] = [];

    let score = 75;
    let connections = 20;

    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

      score = Math.max(0, Math.min(100, score + (Math.random() - 0.48) * 5));
      connections += Math.random() > 0.7 ? Math.floor(Math.random() * 3) : 0;

      scoreHistory.push({ date: dateStr, value: score });
      networkGrowth.push({ date: dateStr, value: connections });
      attestationActivity.push({ date: dateStr, value: Math.floor(Math.random() * 5) });
    }

    return {
      metrics: {
        currentScore: Math.round(score),
        previousScore: Math.round(score - 3),
        trend: score > 75 ? 'up' : score < 70 ? 'down' : 'stable',
        totalConnections: connections,
        directConnections: Math.floor(connections * 0.4),
        transitiveConnections: Math.floor(connections * 0.6),
        attestationsGiven: 15,
        attestationsReceived: 23,
        averagePathLength: 1.8,
      },
      scoreHistory,
      networkGrowth,
      attestationActivity,
      tierDistribution: [
        { tier: 0, count: 5, percentage: 10 },
        { tier: 1, count: 15, percentage: 30 },
        { tier: 2, count: 12, percentage: 24 },
        { tier: 3, count: 10, percentage: 20 },
        { tier: 4, count: 8, percentage: 16 },
      ],
      relationshipBreakdown: [
        { type: 'Direct Trust', count: 20, percentage: 40 },
        { type: 'Introduction', count: 15, percentage: 30 },
        { type: 'Organization', count: 10, percentage: 20 },
        { type: 'Vouch', count: 5, percentage: 10 },
      ],
    };
  }, [data, timeRange]);

  const tierColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6'];

  if (isLoading) {
    return (
      <div className={`animate-pulse space-y-6 ${className}`}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 bg-gray-200 dark:bg-gray-700 rounded-xl" />
          ))}
        </div>
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded-xl" />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Time range selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Trust Analytics
        </h2>
        <div className="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
          {(['7d', '30d', '90d', '1y'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                timeRange === range
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Trust Score"
          value={analyticsData.metrics.currentScore}
          change={analyticsData.metrics.currentScore - analyticsData.metrics.previousScore}
          icon="🛡️"
          color="blue"
        />
        <StatCard
          title="Connections"
          value={analyticsData.metrics.totalConnections}
          icon="🔗"
          color="emerald"
        />
        <StatCard
          title="Attestations Given"
          value={analyticsData.metrics.attestationsGiven}
          icon="✍️"
          color="purple"
        />
        <StatCard
          title="Avg Path Length"
          value={analyticsData.metrics.averagePathLength.toFixed(1)}
          icon="📏"
          color="amber"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trust score trend */}
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Trust Score Trend
          </h3>
          <LineChart
            data={analyticsData.scoreHistory}
            height={200}
            color="#3b82f6"
            showArea
            showDots
            showLabels
          />
        </div>

        {/* Network growth */}
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Network Growth
          </h3>
          <LineChart
            data={analyticsData.networkGrowth}
            height={200}
            color="#10b981"
            showArea
            showDots
            showLabels
          />
        </div>
      </div>

      {/* Distribution charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Tier distribution */}
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Tier Distribution
          </h3>
          <div className="flex items-center justify-center">
            <DonutChart
              data={analyticsData.tierDistribution.map((t, i) => ({
                label: `T${t.tier}`,
                value: t.count,
                color: tierColors[i],
              }))}
              size={180}
              thickness={35}
            />
          </div>
          <div className="mt-4 grid grid-cols-5 gap-2">
            {analyticsData.tierDistribution.map((t, i) => (
              <div key={t.tier} className="text-center">
                <div
                  className="w-3 h-3 rounded-full mx-auto mb-1"
                  style={{ backgroundColor: tierColors[i] }}
                />
                <div className="text-xs text-gray-600 dark:text-gray-400">T{t.tier}</div>
                <div className="text-sm font-medium">{t.percentage}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Relationship breakdown */}
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Relationship Types
          </h3>
          <div className="space-y-3">
            {analyticsData.relationshipBreakdown.map((r, i) => (
              <div key={r.type}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">{r.type}</span>
                  <span className="font-medium">{r.count}</span>
                </div>
                <ProgressBar
                  value={r.percentage}
                  color={`bg-${['blue', 'emerald', 'amber', 'purple'][i]}-500`}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Attestation activity */}
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Recent Activity
          </h3>
          <BarChart
            data={analyticsData.attestationActivity.slice(-7).map((d) => ({
              label: d.date.split(' ')[1],
              value: d.value,
              color: '#8b5cf6',
            }))}
            height={150}
          />
        </div>
      </div>

      {/* Connection breakdown */}
      <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Connection Breakdown
        </h3>
        <div className="grid grid-cols-2 gap-8">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 dark:text-gray-400">Direct Connections</span>
              <span className="font-semibold">{analyticsData.metrics.directConnections}</span>
            </div>
            <ProgressBar
              value={(analyticsData.metrics.directConnections / analyticsData.metrics.totalConnections) * 100}
              color="bg-blue-500"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 dark:text-gray-400">Transitive Connections</span>
              <span className="font-semibold">{analyticsData.metrics.transitiveConnections}</span>
            </div>
            <ProgressBar
              value={(analyticsData.metrics.transitiveConnections / analyticsData.metrics.totalConnections) * 100}
              color="bg-emerald-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export { LineChart, BarChart, DonutChart, StatCard };
export default TrustAnalyticsDashboard;
