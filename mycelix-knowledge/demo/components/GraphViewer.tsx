// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';

export function GraphViewer() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Belief Graph</h2>
        <p className="text-slate-600 mt-1">Visualize relationships between claims</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        {/* Placeholder for BeliefGraph component */}
        <div className="h-[500px] bg-slate-50 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">🔗</div>
            <div className="text-slate-600">Interactive Belief Graph</div>
            <div className="text-sm text-slate-500 mt-2">
              Connect to Holochain to visualize live data
            </div>
            <button className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Load Sample Graph
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="p-4 border-t border-slate-200 bg-slate-50">
          <div className="flex flex-wrap gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>Supports</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span>Contradicts</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span>Refines</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-amber-500"></div>
              <span>Depends On</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="font-semibold text-slate-900 mb-2">Layout</h3>
          <select className="w-full px-3 py-2 border border-slate-300 rounded-lg">
            <option>Force-Directed</option>
            <option>Hierarchical</option>
            <option>Radial</option>
          </select>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="font-semibold text-slate-900 mb-2">Color By</h3>
          <select className="w-full px-3 py-2 border border-slate-300 rounded-lg">
            <option>Credibility</option>
            <option>Epistemic Type</option>
            <option>Node Type</option>
          </select>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="font-semibold text-slate-900 mb-2">Depth</h3>
          <input
            type="range"
            min="1"
            max="5"
            defaultValue="3"
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
}
