// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState } from 'react';

interface FactCheckResult {
  verdict: string;
  confidence: number;
  explanation: string;
  supportingClaims: Array<{ content: string; credibility: number }>;
  contradictingClaims: Array<{ content: string; credibility: number }>;
}

const SAMPLE_RESULTS: Record<string, FactCheckResult> = {
  'climate': {
    verdict: 'True',
    confidence: 0.92,
    explanation: 'This claim is well-supported by scientific consensus and empirical data from multiple independent sources.',
    supportingClaims: [
      { content: 'NASA data shows global temperature has risen 1.1°C since 1880', credibility: 0.98 },
      { content: 'IPCC reports confirm human-caused climate change with >95% certainty', credibility: 0.96 },
      { content: 'Ocean temperatures have increased measurably over the past century', credibility: 0.94 },
    ],
    contradictingClaims: [],
  },
  'default': {
    verdict: 'InsufficientEvidence',
    confidence: 0.45,
    explanation: 'There is not enough verified information in the knowledge graph to make a confident determination.',
    supportingClaims: [],
    contradictingClaims: [],
  },
};

export function FactChecker() {
  const [statement, setStatement] = useState('');
  const [isChecking, setIsChecking] = useState(false);
  const [result, setResult] = useState<FactCheckResult | null>(null);

  const handleCheck = async () => {
    if (!statement.trim()) return;

    setIsChecking(true);
    setResult(null);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Demo logic
    if (statement.toLowerCase().includes('climate') || statement.toLowerCase().includes('temperature')) {
      setResult(SAMPLE_RESULTS.climate);
    } else {
      setResult(SAMPLE_RESULTS.default);
    }

    setIsChecking(false);
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'True': return 'bg-green-100 text-green-800 border-green-300';
      case 'MostlyTrue': return 'bg-lime-100 text-lime-800 border-lime-300';
      case 'Mixed': return 'bg-amber-100 text-amber-800 border-amber-300';
      case 'MostlyFalse': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'False': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case 'True': return '✓';
      case 'MostlyTrue': return '◐';
      case 'Mixed': return '◑';
      case 'MostlyFalse': return '◔';
      case 'False': return '✗';
      default: return '?';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Fact Checker</h2>
        <p className="text-slate-600 mt-1">
          Enter a statement to check it against the decentralized knowledge graph
        </p>
      </div>

      {/* Input */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Statement to verify
        </label>
        <textarea
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          placeholder="Enter a claim or statement to fact-check..."
          className="w-full h-32 px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
        />

        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-slate-500">
            Try: "Global temperatures have risen significantly due to human activity"
          </div>
          <button
            onClick={handleCheck}
            disabled={isChecking || !statement.trim()}
            className="px-6 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isChecking ? (
              <>
                <span className="animate-spin">⟳</span>
                Checking...
              </>
            ) : (
              <>
                <span>✓</span>
                Check Statement
              </>
            )}
          </button>
        </div>
      </div>

      {/* Result */}
      {result && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          {/* Verdict Header */}
          <div className={`p-6 ${getVerdictColor(result.verdict)} border-b`}>
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center text-3xl shadow-sm">
                {getVerdictIcon(result.verdict)}
              </div>
              <div>
                <div className="text-2xl font-bold">{result.verdict.replace(/([A-Z])/g, ' $1').trim()}</div>
                <div className="text-sm opacity-75">
                  {Math.round(result.confidence * 100)}% confidence
                </div>
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="p-6 border-b border-slate-200">
            <h3 className="font-semibold text-slate-900 mb-2">Explanation</h3>
            <p className="text-slate-600">{result.explanation}</p>
          </div>

          {/* Supporting Claims */}
          {result.supportingClaims.length > 0 && (
            <div className="p-6 border-b border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                <span className="text-green-600">+</span>
                Supporting Evidence ({result.supportingClaims.length})
              </h3>
              <ul className="space-y-3">
                {result.supportingClaims.map((claim, i) => (
                  <li key={i} className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                    <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center text-green-700 font-bold text-sm shrink-0">
                      {Math.round(claim.credibility * 100)}
                    </div>
                    <p className="text-slate-700">{claim.content}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Contradicting Claims */}
          {result.contradictingClaims.length > 0 && (
            <div className="p-6">
              <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                <span className="text-red-600">-</span>
                Contradicting Evidence ({result.contradictingClaims.length})
              </h3>
              <ul className="space-y-3">
                {result.contradictingClaims.map((claim, i) => (
                  <li key={i} className="flex items-start gap-3 p-3 bg-red-50 rounded-lg">
                    <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center text-red-700 font-bold text-sm shrink-0">
                      {Math.round(claim.credibility * 100)}
                    </div>
                    <p className="text-slate-700">{claim.content}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* How it works */}
      <div className="bg-slate-100 rounded-xl p-6">
        <h3 className="font-semibold text-slate-900 mb-4">How Fact Checking Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold shrink-0">1</div>
            <div>
              <div className="font-medium text-slate-900">Parse & Analyze</div>
              <div className="text-sm text-slate-600">The statement is analyzed for key claims and semantic meaning</div>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold shrink-0">2</div>
            <div>
              <div className="font-medium text-slate-900">Cross-Reference</div>
              <div className="text-sm text-slate-600">Claims are matched against the decentralized knowledge graph</div>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold shrink-0">3</div>
            <div>
              <div className="font-medium text-slate-900">Verdict</div>
              <div className="text-sm text-slate-600">A verdict is generated based on credibility-weighted evidence</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
