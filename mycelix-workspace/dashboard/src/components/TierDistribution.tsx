// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { TierDistributionEntry } from '../types';

const TIER_COLORS: Record<string, string> = {
  Observer: '#6b7280',
  Participant: '#3b82f6',
  Citizen: '#22c55e',
  Steward: '#f59e0b',
  Guardian: '#a855f7',
};

interface Props {
  data: TierDistributionEntry[];
}

export default function TierDistribution({ data }: Props) {
  const total = data.reduce((sum, d) => sum + d.count, 0);

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h2 className="text-lg font-semibold mb-1">Tier Distribution</h2>
      <p className="text-sm text-gray-400 mb-4">{total} agents total</p>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="tier"
            tick={{ fill: '#d1d5db', fontSize: 13 }}
            width={80}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '6px',
              color: '#f3f4f6',
            }}
            formatter={(value: number, _name: string, props: { payload: TierDistributionEntry }) => [
              `${value} (${props.payload.percentage}%)`,
              'Agents',
            ]}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((entry) => (
              <Cell key={entry.tier} fill={TIER_COLORS[entry.tier]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-3 grid grid-cols-5 gap-1 text-xs text-center">
        {data.map((entry) => (
          <div key={entry.tier}>
            <div
              className="w-3 h-3 rounded-full mx-auto mb-1"
              style={{ backgroundColor: TIER_COLORS[entry.tier] }}
            />
            <span className="text-gray-400">{entry.tier}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
