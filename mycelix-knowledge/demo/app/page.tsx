// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';
import { ClaimExplorer } from '../components/ClaimExplorer';
import { FactChecker } from '../components/FactChecker';
import { GraphViewer } from '../components/GraphViewer';
import { EpistemicExplorer } from '../components/EpistemicExplorer';
import { AIAnalyzer } from '../components/AIAnalyzer';

type Tab = 'claims' | 'factcheck' | 'graph' | 'epistemic' | 'ai';

export default function DemoPage() {
  const [activeTab, setActiveTab] = useState<Tab>('claims');

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'claims', label: 'Claims', icon: '📋' },
    { id: 'factcheck', label: 'Fact Check', icon: '✓' },
    { id: 'graph', label: 'Belief Graph', icon: '🔗' },
    { id: 'epistemic', label: 'E-N-M Cube', icon: '🎲' },
    { id: 'ai', label: 'AI Analysis', icon: '🤖' },
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
                K
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Mycelix Knowledge</h1>
                <p className="text-sm text-slate-500">Decentralized Knowledge Graph Demo</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/mycelix/knowledge"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-600 hover:text-slate-900"
              >
                GitHub
              </a>
              <a
                href="/docs"
                className="text-slate-600 hover:text-slate-900"
              >
                Docs
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'claims' && <ClaimExplorer />}
        {activeTab === 'factcheck' && <FactChecker />}
        {activeTab === 'graph' && <GraphViewer />}
        {activeTab === 'epistemic' && <EpistemicExplorer />}
        {activeTab === 'ai' && <AIAnalyzer />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold text-slate-900 mb-3">Mycelix Knowledge</h3>
              <p className="text-sm text-slate-600">
                A decentralized knowledge graph with epistemic classification,
                credibility scoring, and bidirectional Epistemic Markets integration.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 mb-3">Resources</h3>
              <ul className="space-y-2 text-sm">
                <li><a href="/docs" className="text-blue-600 hover:underline">Documentation</a></li>
                <li><a href="/api" className="text-blue-600 hover:underline">API Reference</a></li>
                <li><a href="/storybook" className="text-blue-600 hover:underline">Component Library</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 mb-3">Community</h3>
              <ul className="space-y-2 text-sm">
                <li><a href="https://github.com/mycelix/knowledge" className="text-blue-600 hover:underline">GitHub</a></li>
                <li><a href="https://discord.gg/mycelix" className="text-blue-600 hover:underline">Discord</a></li>
                <li><a href="https://twitter.com/mycelix" className="text-blue-600 hover:underline">Twitter</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-slate-200 text-center text-sm text-slate-500">
            Built with Holochain &middot; MIT License &middot; &copy; {new Date().getFullYear()} Mycelix
          </div>
        </div>
      </footer>
    </div>
  );
}
