// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';

export function AIAnalyzer() {
  const [text, setText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setAnalyzing(true);

    // Simulate AI analysis
    await new Promise(r => setTimeout(r, 2000));

    setResults({
      classification: {
        empirical: 0.75,
        normative: 0.15,
        mythic: 0.10,
        dominantType: 'empirical',
        confidence: 0.85,
      },
      evidence: {
        sources: ['Scientific American', 'Nature Journal'],
        statistics: ['45% increase', '2.3 million affected'],
        quotes: 1,
      },
      similarity: [
        { claim: 'Related claim about similar topic...', score: 0.78 },
        { claim: 'Another related claim from the graph...', score: 0.65 },
      ],
    });

    setAnalyzing(false);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">AI Analysis</h2>
        <p className="text-slate-600 mt-1">
          Auto-classify claims, extract evidence, and detect contradictions
        </p>
      </div>

      {/* Input */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Text to analyze
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste an article, claim, or any text to analyze..."
          className="w-full h-40 px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 resize-none"
        />
        <div className="mt-4 flex justify-end">
          <button
            onClick={handleAnalyze}
            disabled={analyzing || !text.trim()}
            className="px-6 py-2 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
          >
            {analyzing ? (
              <>
                <span className="animate-spin">⟳</span>
                Analyzing...
              </>
            ) : (
              <>
                <span>🤖</span>
                Analyze with AI
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Classification */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <span>🎯</span> Epistemic Classification
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-red-600">Empirical</span>
                  <span>{Math.round(results.classification.empirical * 100)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full"
                    style={{ width: `${results.classification.empirical * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-green-600">Normative</span>
                  <span>{Math.round(results.classification.normative * 100)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 rounded-full"
                    style={{ width: `${results.classification.normative * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-blue-600">Mythic</span>
                  <span>{Math.round(results.classification.mythic * 100)}%</span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${results.classification.mythic * 100}%` }}
                  />
                </div>
              </div>
              <div className="pt-4 border-t border-slate-200">
                <div className="flex justify-between">
                  <span className="text-slate-600">Dominant Type</span>
                  <span className="font-medium capitalize">{results.classification.dominantType}</span>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-slate-600">Confidence</span>
                  <span className="font-medium">{Math.round(results.classification.confidence * 100)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Evidence Extraction */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <span>📊</span> Extracted Evidence
            </h3>
            <div className="space-y-4">
              <div>
                <div className="text-sm text-slate-500 mb-2">Sources Found</div>
                <div className="flex flex-wrap gap-2">
                  {results.evidence.sources.map((s: string, i: number) => (
                    <span key={i} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-500 mb-2">Statistics</div>
                <div className="flex flex-wrap gap-2">
                  {results.evidence.statistics.map((s: string, i: number) => (
                    <span key={i} className="px-3 py-1 bg-amber-100 text-amber-700 rounded-full text-sm">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-500 mb-2">Quotes</div>
                <span className="text-2xl font-bold text-slate-900">{results.evidence.quotes}</span>
              </div>
            </div>
          </div>

          {/* Similar Claims */}
          <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <span>🔗</span> Similar Claims in Knowledge Graph
            </h3>
            <div className="space-y-3">
              {results.similarity.map((item: any, i: number) => (
                <div key={i} className="flex items-center gap-4 p-3 bg-slate-50 rounded-lg">
                  <div className="w-12 h-12 rounded-full bg-purple-100 text-purple-700 flex items-center justify-center font-bold">
                    {Math.round(item.score * 100)}%
                  </div>
                  <p className="text-slate-700 flex-1">{item.claim}</p>
                  <button className="text-blue-600 hover:underline text-sm">View</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Features */}
      <div className="bg-purple-50 rounded-xl p-6">
        <h3 className="font-semibold text-slate-900 mb-4">AI Analysis Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { icon: '🎯', title: 'Auto-Classification', desc: 'Classify into E-N-M dimensions' },
            { icon: '📊', title: 'Evidence Extraction', desc: 'Find sources, stats, quotes' },
            { icon: '⚡', title: 'Contradiction Detection', desc: 'Identify logical conflicts' },
            { icon: '🔗', title: 'Similarity Analysis', desc: 'Find related claims' },
          ].map((f, i) => (
            <div key={i} className="text-center">
              <div className="text-3xl mb-2">{f.icon}</div>
              <div className="font-medium text-slate-900">{f.title}</div>
              <div className="text-sm text-slate-600">{f.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
