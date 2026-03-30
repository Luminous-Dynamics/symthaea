// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

interface ConsciousnessScoreProps {
  phiScore?: number | null
  harmonyAlignment?: number | null
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  className?: string
}

/**
 * Displays Symthaea consciousness scores (Phi + Harmony Alignment) for a project.
 * Used on project cards and detail pages in the Conscious Allocation Network.
 */
export function ConsciousnessScore({
  phiScore,
  harmonyAlignment,
  size = 'md',
  showLabel = true,
  className = '',
}: ConsciousnessScoreProps) {
  const phi = phiScore ?? 0
  const harmony = harmonyAlignment ?? 0
  const hasData = phiScore !== null && phiScore !== undefined

  const getPhiColor = (score: number) => {
    if (score >= 0.8) return { bg: 'from-violet-500 to-purple-600', text: 'text-violet-200', glow: 'shadow-violet-500/30' }
    if (score >= 0.6) return { bg: 'from-blue-500 to-indigo-600', text: 'text-blue-200', glow: 'shadow-blue-500/30' }
    if (score >= 0.4) return { bg: 'from-cyan-500 to-blue-500', text: 'text-cyan-200', glow: 'shadow-cyan-500/30' }
    if (score >= 0.2) return { bg: 'from-amber-500 to-orange-500', text: 'text-amber-200', glow: 'shadow-amber-500/30' }
    return { bg: 'from-gray-500 to-gray-600', text: 'text-gray-300', glow: '' }
  }

  const getHarmonyColor = (score: number) => {
    if (score >= 0.8) return 'text-emerald-400'
    if (score >= 0.6) return 'text-green-400'
    if (score >= 0.4) return 'text-yellow-400'
    if (score >= 0.2) return 'text-orange-400'
    return 'text-gray-400'
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-3 py-1 gap-2',
    lg: 'text-base px-4 py-1.5 gap-2',
  }

  if (!hasData) {
    return (
      <span className={`inline-flex items-center rounded-full bg-gray-800/60 border border-gray-700/50 ${sizeClasses[size]} ${className}`}>
        <span className="text-gray-500">No consciousness data</span>
      </span>
    )
  }

  const phiColors = getPhiColor(phi)

  return (
    <span className={`inline-flex items-center rounded-full bg-gradient-to-r ${phiColors.bg} shadow-lg ${phiColors.glow} ${sizeClasses[size]} ${className}`}>
      <span className={`font-mono font-bold ${phiColors.text}`} title="Integrated Information (Phi)">
        {'\u03A6'} {(phi * 100).toFixed(0)}
      </span>
      <span className="text-white/30">|</span>
      <span className={`font-mono ${getHarmonyColor(harmony)}`} title="Eight Harmonies Alignment">
        {'\u2261'} {(harmony * 100).toFixed(0)}
      </span>
      {showLabel && (
        <span className="text-white/60 text-xs ml-1">
          {phi >= 0.7 ? 'Aligned' : phi >= 0.4 ? 'Emerging' : 'Nascent'}
        </span>
      )}
    </span>
  )
}
