// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect } from 'react'

interface AIScoreBadgeProps {
  score?: number
  viability?: 'excellent' | 'good' | 'fair' | 'poor' | 'not_viable'
  showLabel?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function AIScoreBadge({
  score,
  viability,
  showLabel = true,
  size = 'md',
  className = '',
}: AIScoreBadgeProps) {
  // Determine color based on score
  const getScoreColor = (s: number) => {
    if (s >= 80) return { bg: 'from-emerald-500 to-green-500', text: 'text-emerald-300', glow: 'shadow-emerald-500/40' }
    if (s >= 65) return { bg: 'from-cyan-500 to-blue-500', text: 'text-cyan-300', glow: 'shadow-cyan-500/40' }
    if (s >= 50) return { bg: 'from-yellow-500 to-amber-500', text: 'text-yellow-300', glow: 'shadow-yellow-500/40' }
    if (s >= 30) return { bg: 'from-orange-500 to-red-500', text: 'text-orange-300', glow: 'shadow-orange-500/40' }
    return { bg: 'from-red-500 to-rose-600', text: 'text-red-300', glow: 'shadow-red-500/40' }
  }

  const getViabilityLabel = (v: string) => {
    const labels: Record<string, string> = {
      excellent: 'Excellent',
      good: 'Good',
      fair: 'Fair',
      poor: 'Poor',
      not_viable: 'Not Viable',
    }
    return labels[v] || v
  }

  const sizeClasses = {
    sm: 'w-8 h-8 text-xs',
    md: 'w-12 h-12 text-sm',
    lg: 'w-16 h-16 text-base',
  }

  if (score === undefined) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className={`${sizeClasses[size]} rounded-full bg-white/5 border border-white/10 flex items-center justify-center animate-pulse`}>
          <span className="text-white/30">?</span>
        </div>
        {showLabel && <span className="text-xs text-white/40">Analyzing...</span>}
      </div>
    )
  }

  const colors = getScoreColor(score)

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div
        className={`${sizeClasses[size]} rounded-full bg-gradient-to-br ${colors.bg} flex items-center justify-center shadow-lg ${colors.glow} border border-white/20`}
      >
        <span className="font-bold text-white drop-shadow-md">{score}</span>
      </div>
      {showLabel && (
        <div className="flex flex-col">
          <span className={`text-xs font-semibold ${colors.text}`}>
            AI Score
          </span>
          {viability && (
            <span className="text-[10px] text-white/50">
              {getViabilityLabel(viability)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// Compact inline version for lists
export function AIScoreInline({ score }: { score?: number }) {
  if (score === undefined) return null

  const getColor = (s: number) => {
    if (s >= 80) return 'text-emerald-400'
    if (s >= 65) return 'text-cyan-400'
    if (s >= 50) return 'text-yellow-400'
    return 'text-orange-400'
  }

  return (
    <span className={`text-xs font-bold ${getColor(score)}`}>
      {score}
    </span>
  )
}

// Full analysis card
interface AnalysisCardProps {
  analysis: {
    overallScore: number
    viabilityRating: string
    analyses: Record<string, { score: number; recommendations: string[] }>
    investmentReadiness: {
      score: number
      stage: string
      estimatedTimeToRevenue: string
    }
  }
}

export function AIAnalysisCard({ analysis }: AnalysisCardProps) {
  const { overallScore, viabilityRating, analyses, investmentReadiness } = analysis

  const getAnalyzerName = (key: string) => {
    const names: Record<string, string> = {
      helios: '☀️ Solar',
      aeolus: '💨 Wind',
      hydra: '💧 Hydro',
      vulcan: '🌋 Geothermal',
      poseidon: '🌊 Ocean',
      apollo: '⚛️ Nuclear',
      atlas: '🔌 Grid',
    }
    return names[key] || key
  }

  return (
    <div className="p-4 bg-gradient-to-b from-emerald-950/30 to-transparent border border-emerald-500/30 rounded-lg">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-emerald-400">AI Analysis</h4>
          <p className="text-[10px] text-white/50">Sacred Seven Score</p>
        </div>
        <AIScoreBadge
          score={overallScore}
          viability={viabilityRating as any}
          showLabel={false}
          size="md"
        />
      </div>

      {/* Individual analyzer scores */}
      <div className="space-y-2 mb-4">
        {Object.entries(analyses).map(([key, result]) => (
          <div key={key} className="flex items-center justify-between text-xs">
            <span className="text-white/60">{getAnalyzerName(key)}</span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full"
                  style={{ width: `${result.score}%` }}
                />
              </div>
              <span className="text-white/80 w-6 text-right">{result.score}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Investment readiness */}
      <div className="pt-3 border-t border-white/10">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-white/60">Investment Ready</span>
          <span className="text-cyan-400">{investmentReadiness.score}%</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-white/60">Time to Revenue</span>
          <span className="text-white/80">{investmentReadiness.estimatedTimeToRevenue}</span>
        </div>
      </div>

      {/* Top recommendation */}
      {Object.values(analyses).some(a => a.recommendations?.length > 0) && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <p className="text-[10px] text-white/50 mb-1">Top Recommendation</p>
          <p className="text-xs text-emerald-300/80">
            {Object.values(analyses).find(a => a.recommendations?.length > 0)?.recommendations[0]}
          </p>
        </div>
      )}
    </div>
  )
}
