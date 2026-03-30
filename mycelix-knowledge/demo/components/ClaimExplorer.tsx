// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';

interface Claim {
  id: string;
  content: string;
  classification: { empirical: number; normative: number; mythic: number };
  credibility: number;
  author: string;
  tags: string[];
  createdAt: number;
}

const SAMPLE_CLAIMS: Claim[] = [
  {
    id: '1',
    content: 'Global temperatures have risen by approximately 1.1°C since the pre-industrial era.',
    classification: { empirical: 0.92, normative: 0.05, mythic: 0.03 },
    credibility: 0.95,
    author: 'did:example:scientist1',
    tags: ['climate', 'science', 'temperature'],
    createdAt: Date.now() - 86400000 * 30,
  },
  {
    id: '2',
    content: 'Universal healthcare should be a fundamental human right.',
    classification: { empirical: 0.15, normative: 0.8, mythic: 0.05 },
    credibility: 0.7,
    author: 'did:example:ethicist1',
    tags: ['healthcare', 'ethics', 'policy'],
    createdAt: Date.now() - 86400000 * 15,
  },
  {
    id: '3',
    content: 'The universe exhibits patterns suggesting underlying consciousness.',
    classification: { empirical: 0.2, normative: 0.1, mythic: 0.7 },
    credibility: 0.35,
    author: 'did:example:philosopher1',
    tags: ['philosophy', 'consciousness', 'metaphysics'],
    createdAt: Date.now() - 86400000 * 7,
  },
  {
    id: '4',
    content: 'Renewable energy is now cost-competitive with fossil fuels in most markets.',
    classification: { empirical: 0.85, normative: 0.1, mythic: 0.05 },
    credibility: 0.88,
    author: 'did:example:economist1',
    tags: ['energy', 'economics', 'climate'],
    createdAt: Date.now() - 86400000 * 3,
  },
];

export function ClaimExplorer() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedClaim, setSelectedClaim] = useState<Claim | null>(null);
  const [filter, setFilter] = useState<'all' | 'empirical' | 'normative' | 'mythic'>('all');

  const filteredClaims = SAMPLE_CLAIMS.filter(claim => {
    const matchesSearch = claim.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      claim.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    if (filter === 'all') return matchesSearch;

    const { empirical, normative, mythic } = claim.classification;
    const max = Math.max(empirical, normative, mythic);

    if (filter === 'empirical') return matchesSearch && max === empirical;
    if (filter === 'normative') return matchesSearch && max === normative;
    if (filter === 'mythic') return matchesSearch && max === mythic;

    return matchesSearch;
  });

  const getCredibilityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-lime-600';
    if (score >= 0.4) return 'text-amber-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Claims Explorer</h2>
          <p className="text-slate-600 mt-1">Browse and search the knowledge graph</p>
        </div>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
          <span>+</span> Add Claim
        </button>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search claims..."
              className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex gap-2">
            {(['all', 'empirical', 'normative', 'mythic'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg font-medium text-sm capitalize transition-colors ${
                  filter === f
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Claims Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredClaims.map(claim => (
          <div
            key={claim.id}
            onClick={() => setSelectedClaim(claim)}
            className={`bg-white rounded-xl border-2 p-5 cursor-pointer transition-all ${
              selectedClaim?.id === claim.id
                ? 'border-blue-500 shadow-lg'
                : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            {/* Content */}
            <p className="text-slate-900 font-medium mb-3">{claim.content}</p>

            {/* E-N-M Bar */}
            <div className="h-2 rounded-full overflow-hidden flex mb-3">
              <div
                className="bg-red-500"
                style={{ width: `${claim.classification.empirical * 100}%` }}
              />
              <div
                className="bg-green-500"
                style={{ width: `${claim.classification.normative * 100}%` }}
              />
              <div
                className="bg-blue-500"
                style={{ width: `${claim.classification.mythic * 100}%` }}
              />
            </div>

            {/* Meta */}
            <div className="flex items-center justify-between text-sm">
              <div className="flex gap-2">
                {claim.tags.slice(0, 3).map(tag => (
                  <span key={tag} className="px-2 py-1 bg-slate-100 rounded text-slate-600">
                    {tag}
                  </span>
                ))}
              </div>
              <div className={`font-bold ${getCredibilityColor(claim.credibility)}`}>
                {Math.round(claim.credibility * 100)}%
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Detail Panel */}
      {selectedClaim && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-bold text-slate-900 mb-4">Claim Details</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="text-sm text-slate-500 mb-1">Content</div>
              <p className="text-slate-900">{selectedClaim.content}</p>
            </div>

            <div>
              <div className="text-sm text-slate-500 mb-1">Epistemic Classification</div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-20 text-sm text-red-600">Empirical</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-red-500" style={{ width: `${selectedClaim.classification.empirical * 100}%` }} />
                  </div>
                  <span className="text-sm font-medium">{Math.round(selectedClaim.classification.empirical * 100)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-20 text-sm text-green-600">Normative</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500" style={{ width: `${selectedClaim.classification.normative * 100}%` }} />
                  </div>
                  <span className="text-sm font-medium">{Math.round(selectedClaim.classification.normative * 100)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-20 text-sm text-blue-600">Mythic</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500" style={{ width: `${selectedClaim.classification.mythic * 100}%` }} />
                  </div>
                  <span className="text-sm font-medium">{Math.round(selectedClaim.classification.mythic * 100)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
