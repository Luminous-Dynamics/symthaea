// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useMemo } from 'react';
import type { GateAuditEntry } from '../types';

interface Props {
  entries: GateAuditEntry[];
}

export default function AuditTrail({ entries }: Props) {
  const [filterAction, setFilterAction] = useState('');
  const [filterZome, setFilterZome] = useState('');
  const [filterEligible, setFilterEligible] = useState<'all' | 'yes' | 'no'>('all');
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const actionNames = useMemo(
    () => Array.from(new Set(entries.map((e) => e.action_name))).sort(),
    [entries],
  );

  const zomeNames = useMemo(
    () => Array.from(new Set(entries.map((e) => e.zome_name))).sort(),
    [entries],
  );

  const filtered = useMemo(() => {
    return entries.filter((e) => {
      if (filterAction && e.action_name !== filterAction) return false;
      if (filterZome && e.zome_name !== filterZome) return false;
      if (filterEligible === 'yes' && !e.eligible) return false;
      if (filterEligible === 'no' && e.eligible) return false;
      return true;
    });
  }, [entries, filterAction, filterZome, filterEligible]);

  const pageCount = Math.ceil(filtered.length / pageSize);
  const pageEntries = filtered.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Audit Trail</h2>
        <span className="text-sm text-gray-400">{filtered.length} events</span>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-3 text-sm">
        <select
          value={filterAction}
          onChange={(e) => { setFilterAction(e.target.value); setPage(0); }}
          className="bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-blue-500"
        >
          <option value="">All actions</option>
          {actionNames.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        <select
          value={filterZome}
          onChange={(e) => { setFilterZome(e.target.value); setPage(0); }}
          className="bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-blue-500"
        >
          <option value="">All zomes</option>
          {zomeNames.map((z) => (
            <option key={z} value={z}>{z}</option>
          ))}
        </select>

        <select
          value={filterEligible}
          onChange={(e) => { setFilterEligible(e.target.value as 'all' | 'yes' | 'no'); setPage(0); }}
          className="bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-blue-500"
        >
          <option value="all">All outcomes</option>
          <option value="yes">Approved only</option>
          <option value="no">Rejected only</option>
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-900">
            <tr className="text-left text-gray-400 border-b border-gray-800">
              <th className="pb-2 pr-3">Time</th>
              <th className="pb-2 pr-3">Agent</th>
              <th className="pb-2 pr-3">Action</th>
              <th className="pb-2 pr-3">Zome</th>
              <th className="pb-2 pr-3">Tier</th>
              <th className="pb-2 pr-3">Required</th>
              <th className="pb-2 pr-3">Weight</th>
              <th className="pb-2">Result</th>
            </tr>
          </thead>
          <tbody>
            {pageEntries.map((entry, i) => (
              <tr
                key={`${entry.correlation_id}-${i}`}
                className="border-b border-gray-800/50 hover:bg-gray-800/30"
              >
                <td className="py-1.5 pr-3 text-gray-400 font-mono text-xs whitespace-nowrap">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </td>
                <td className="py-1.5 pr-3 font-mono text-xs truncate max-w-[120px]" title={entry.agent_did}>
                  {entry.agent_did.replace('did:mycelix:', '')}
                </td>
                <td className="py-1.5 pr-3 text-gray-300">{entry.action_name}</td>
                <td className="py-1.5 pr-3 text-gray-400">{entry.zome_name}</td>
                <td className="py-1.5 pr-3">
                  <TierBadge tier={entry.actual_tier} />
                </td>
                <td className="py-1.5 pr-3">
                  <TierBadge tier={entry.required_tier} />
                </td>
                <td className="py-1.5 pr-3 font-mono text-xs text-gray-300">
                  {entry.weight_bp > 0 ? `${(entry.weight_bp / 100).toFixed(0)}%` : '-'}
                </td>
                <td className="py-1.5">
                  {entry.eligible ? (
                    <span className="text-green-400 text-xs font-medium">PASS</span>
                  ) : (
                    <span className="text-red-400 text-xs font-medium">DENY</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="flex items-center justify-between mt-3 text-sm text-gray-400">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Prev
          </button>
          <span>
            Page {page + 1} of {pageCount}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
            disabled={page >= pageCount - 1}
            className="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    Observer: 'bg-gray-700 text-gray-300',
    Participant: 'bg-blue-900/50 text-blue-300',
    Citizen: 'bg-green-900/50 text-green-300',
    Steward: 'bg-yellow-900/50 text-yellow-300',
    Guardian: 'bg-purple-900/50 text-purple-300',
  };

  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colors[tier] ?? 'bg-gray-700 text-gray-300'}`}>
      {tier}
    </span>
  );
}
