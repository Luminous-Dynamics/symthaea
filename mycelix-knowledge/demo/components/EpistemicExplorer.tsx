// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';

export function EpistemicExplorer() {
  const [filter, setFilter] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Epistemic Cube</h2>
        <p className="text-slate-600 mt-1">
          Explore claims in 3D E-N-M classification space
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Cube Visualization */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="h-[500px] bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🎲</div>
              <div className="text-slate-600">3D Epistemic Cube</div>
              <div className="text-sm text-slate-500 mt-2">
                Drag to rotate • Scroll to zoom • Click points to select
              </div>
            </div>
          </div>
        </div>

        {/* Controls & Legend */}
        <div className="space-y-4">
          {/* Axis Legend */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="font-semibold text-slate-900 mb-3">Axes</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded bg-red-500"></div>
                <div>
                  <div className="font-medium text-slate-900">Empirical</div>
                  <div className="text-xs text-slate-500">Verifiable through observation</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded bg-green-500"></div>
                <div>
                  <div className="font-medium text-slate-900">Normative</div>
                  <div className="text-xs text-slate-500">Value-based, ethical claims</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded bg-blue-500"></div>
                <div>
                  <div className="font-medium text-slate-900">Mythic</div>
                  <div className="text-xs text-slate-500">Meaning-making, narrative</div>
                </div>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="font-semibold text-slate-900 mb-3">Filter by Quadrant</h3>
            <div className="space-y-2">
              {[
                { id: 'empirical', label: 'High Empirical', color: 'bg-red-100 text-red-700' },
                { id: 'normative', label: 'High Normative', color: 'bg-green-100 text-green-700' },
                { id: 'mythic', label: 'High Mythic', color: 'bg-blue-100 text-blue-700' },
                { id: 'balanced', label: 'Balanced', color: 'bg-purple-100 text-purple-700' },
              ].map(f => (
                <button
                  key={f.id}
                  onClick={() => setFilter(filter === f.id ? null : f.id)}
                  className={`w-full px-3 py-2 rounded-lg text-left text-sm font-medium transition-colors ${
                    filter === f.id ? f.color : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="font-semibold text-slate-900 mb-3">Statistics</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-600">Total Claims</span>
                <span className="font-medium">1,234</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Avg. Credibility</span>
                <span className="font-medium">72%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Visible</span>
                <span className="font-medium">{filter ? '312' : '1,234'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
