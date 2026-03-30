// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useMemo } from 'react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';
import type {
  ConsciousnessProfile,
  ConsciousnessTier,
  GovernanceEligibility,
  ConsciousnessCredential,
} from '../types';
import { combinedScore, TIER_VOTE_WEIGHT_BP } from '../types';

interface LookupResult {
  profile: ConsciousnessProfile;
  tier: ConsciousnessTier;
  credential: ConsciousnessCredential;
  eligibility: Record<string, GovernanceEligibility>;
}

interface Props {
  lookupProfile: (did: string) => LookupResult | null;
  agents: { did: string; profile: ConsciousnessProfile }[];
}

const TIER_COLORS: Record<string, string> = {
  Observer: '#6b7280',
  Participant: '#3b82f6',
  Citizen: '#22c55e',
  Steward: '#f59e0b',
  Guardian: '#a855f7',
};

export default function ProfileInspector({ lookupProfile, agents }: Props) {
  const [didInput, setDidInput] = useState('');
  const [result, setResult] = useState<LookupResult | null>(null);
  const [notFound, setNotFound] = useState(false);

  const handleLookup = () => {
    const did = didInput.startsWith('did:') ? didInput : `did:mycelix:${didInput}`;
    const r = lookupProfile(did);
    setResult(r);
    setNotFound(!r);
  };

  const radarData = useMemo(() => {
    if (!result) return [];
    return [
      { dimension: 'Identity', value: result.profile.identity, fullMark: 1 },
      { dimension: 'Reputation', value: result.profile.reputation, fullMark: 1 },
      { dimension: 'Community', value: result.profile.community, fullMark: 1 },
      { dimension: 'Engagement', value: result.profile.engagement, fullMark: 1 },
    ];
  }, [result]);

  const score = result ? combinedScore(result.profile) : 0;

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h2 className="text-lg font-semibold mb-3">Profile Inspector</h2>

      {/* DID Input */}
      <div className="flex gap-2 mb-4">
        <div className="flex-1 relative">
          <input
            type="text"
            value={didInput}
            onChange={(e) => { setDidInput(e.target.value); setNotFound(false); }}
            onKeyDown={(e) => e.key === 'Enter' && handleLookup()}
            placeholder="Enter DID (e.g. alice or did:mycelix:alice)"
            className="w-full bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:outline-none focus:border-blue-500 placeholder-gray-500"
            list="agent-suggestions"
          />
          <datalist id="agent-suggestions">
            {agents.map((a) => (
              <option key={a.did} value={a.did} />
            ))}
          </datalist>
        </div>
        <button
          onClick={handleLookup}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded font-medium"
        >
          Lookup
        </button>
      </div>

      {notFound && (
        <p className="text-sm text-red-400 mb-3">
          No agent found. Try: {agents.slice(0, 4).map((a) => a.did.replace('did:mycelix:', '')).join(', ')}
        </p>
      )}

      {result && (
        <div className="space-y-4">
          {/* Tier badge and score */}
          <div className="flex items-center gap-3">
            <span
              className="text-sm font-bold px-3 py-1 rounded-full"
              style={{
                backgroundColor: TIER_COLORS[result.tier] + '30',
                color: TIER_COLORS[result.tier],
              }}
            >
              {result.tier}
            </span>
            <span className="text-sm text-gray-400">
              Combined score: <span className="text-gray-200 font-mono">{score.toFixed(3)}</span>
            </span>
            <span className="text-sm text-gray-400">
              Vote weight: <span className="text-gray-200 font-mono">{(TIER_VOTE_WEIGHT_BP[result.tier] / 100).toFixed(0)}%</span>
            </span>
          </div>

          {/* Radar chart */}
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis
                dataKey="dimension"
                tick={{ fill: '#d1d5db', fontSize: 12 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 1]}
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickCount={5}
              />
              <Radar
                dataKey="value"
                stroke={TIER_COLORS[result.tier]}
                fill={TIER_COLORS[result.tier]}
                fillOpacity={0.25}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>

          {/* Dimension detail */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            {(['identity', 'reputation', 'community', 'engagement'] as const).map((dim) => (
              <div key={dim} className="flex justify-between bg-gray-800/50 rounded px-3 py-1.5">
                <span className="text-gray-400 capitalize">{dim}</span>
                <span className="font-mono text-gray-200">{result.profile[dim].toFixed(2)}</span>
              </div>
            ))}
          </div>

          {/* Action eligibility */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">Action Eligibility</h3>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {Object.entries(result.eligibility).map(([action, elig]) => (
                <div
                  key={action}
                  className="flex items-center justify-between text-sm bg-gray-800/30 rounded px-3 py-1.5"
                >
                  <span className="text-gray-300 font-mono text-xs">{action}</span>
                  {elig.eligible ? (
                    <span className="text-green-400 text-xs font-medium">ELIGIBLE</span>
                  ) : (
                    <span className="text-red-400 text-xs font-medium" title={elig.reasons.join('; ')}>
                      DENIED
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Credential info */}
          <div className="text-xs text-gray-500 border-t border-gray-800 pt-2">
            Credential issued by {result.credential.issuer} | Expires{' '}
            {new Date(result.credential.expires_at / 1000).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}
