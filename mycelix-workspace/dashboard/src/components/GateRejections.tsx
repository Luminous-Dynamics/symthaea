// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { GateDecisionPoint } from '../types';

interface Props {
  data: GateDecisionPoint[];
}

export default function GateRejections({ data }: Props) {
  const actionTypes = useMemo(
    () => Array.from(new Set(data.map((d) => d.action_type))).sort(),
    [data],
  );

  const [selectedAction, setSelectedAction] = useState<string>('all');

  const filtered = useMemo(() => {
    const source =
      selectedAction === 'all'
        ? data
        : data.filter((d) => d.action_type === selectedAction);

    // Aggregate by timestamp across action types
    const byTime = new Map<string, { timestamp: string; approvals: number; rejections: number }>();
    for (const point of source) {
      const existing = byTime.get(point.timestamp);
      if (existing) {
        existing.approvals += point.approvals;
        existing.rejections += point.rejections;
      } else {
        byTime.set(point.timestamp, {
          timestamp: point.timestamp,
          approvals: point.approvals,
          rejections: point.rejections,
        });
      }
    }

    return Array.from(byTime.values())
      .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
      .slice(-30); // Last 30 time buckets
  }, [data, selectedAction]);

  const totalApprovals = filtered.reduce((s, d) => s + d.approvals, 0);
  const totalRejections = filtered.reduce((s, d) => s + d.rejections, 0);
  const rejectionRate =
    totalApprovals + totalRejections > 0
      ? ((totalRejections / (totalApprovals + totalRejections)) * 100).toFixed(1)
      : '0.0';

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold">Gate Decisions</h2>
        <span className="text-sm text-gray-400">
          Rejection rate: <span className="text-red-400 font-mono">{rejectionRate}%</span>
        </span>
      </div>

      <div className="mb-3">
        <select
          value={selectedAction}
          onChange={(e) => setSelectedAction(e.target.value)}
          className="bg-gray-800 text-gray-200 text-sm rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-blue-500"
        >
          <option value="all">All actions</option>
          {actionTypes.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={filtered} margin={{ left: 0 }}>
          <XAxis
            dataKey="timestamp"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} allowDecimals={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '6px',
              color: '#f3f4f6',
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="approvals" stackId="a" fill="#22c55e" name="Approved" />
          <Bar dataKey="rejections" stackId="a" fill="#ef4444" name="Rejected" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
