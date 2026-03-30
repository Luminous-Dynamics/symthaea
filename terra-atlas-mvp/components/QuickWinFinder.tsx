// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import IntelligenceScoreDisplay from './IntelligenceScoreDisplay'
import logger from '@/lib/logger'

interface QuickWin {
  id: string
  name: string
  type: string
  state: string
  score: number
  capacity_mw: number
  investment_million: number
  key_insight: string
  irr?: number
  payback_years?: number
  status?: string
  quick_win_score: number
  quick_win_reason: string
}

export default function QuickWinFinder() {
  const [searching, setSearching] = useState(false)
  const [quickWins, setQuickWins] = useState<QuickWin[]>([])
  const [selectedView, setSelectedView] = useState<'grid' | 'list'>('grid')
  const [filterCategory, setFilterCategory] = useState<'all' | 'immediate' | 'near-term' | 'strategic'>('all')

  const findQuickWins = useCallback(async () => {
    setSearching(true)
    
    try {
      // Fetch projects and analyze for quick wins
      const response = await fetch('/api/intelligence/top-projects?limit=100')
      const data = await response.json()
      
      if (data.projects && data.projects.length > 0) {
        // Calculate quick win scores
        const quickWinProjects = analyzeQuickWins(data.projects)
        setQuickWins(quickWinProjects)
      }
    } catch (error) {
      logger.error('Error finding quick wins', error)
    } finally {
      setSearching(false)
    }
  }, [])

  const analyzeQuickWins = (projects: any[]): QuickWin[] => {
    // Score projects based on quick win criteria
    const scoredProjects = projects.map(project => {
      let quickWinScore = 0
      const reasons: string[] = []
      
      // 1. Low investment requirement (30% weight)
      if (project.investment_million < 10) {
        quickWinScore += 30
        reasons.push('Low capital requirement')
      } else if (project.investment_million < 50) {
        quickWinScore += 20
        reasons.push('Moderate investment')
      } else if (project.investment_million < 100) {
        quickWinScore += 10
      }
      
      // 2. Quick payback (25% weight)
      const payback = project.data?.payback_years || 10
      if (payback < 3) {
        quickWinScore += 25
        reasons.push(`${payback.toFixed(1)} year payback`)
      } else if (payback < 5) {
        quickWinScore += 18
        reasons.push(`${payback.toFixed(1)} year payback`)
      } else if (payback < 7) {
        quickWinScore += 10
      }
      
      // 3. Project readiness (20% weight)
      const readyStatuses = ['Permitted', 'Construction', 'Operating']
      if (readyStatuses.includes(project.status)) {
        quickWinScore += 20
        reasons.push('Ready to deploy')
      } else if (project.status === 'In Development') {
        quickWinScore += 10
      }
      
      // 4. High IRR (15% weight)
      const irr = project.data?.irr || project.data?.estimated_irr || 10
      if (irr > 20) {
        quickWinScore += 15
        reasons.push(`${irr.toFixed(1)}% IRR`)
      } else if (irr > 15) {
        quickWinScore += 10
        reasons.push(`${irr.toFixed(1)}% IRR`)
      } else if (irr > 12) {
        quickWinScore += 5
      }
      
      // 5. Grid connection available (10% weight)
      if (project.data?.transmission_available || project.data?.grid_distance_km < 5) {
        quickWinScore += 10
        reasons.push('Grid ready')
      } else if (project.data?.grid_distance_km < 20) {
        quickWinScore += 5
      }
      
      // Determine category
      let category = 'strategic'
      if (quickWinScore >= 75) {
        category = 'immediate'
      } else if (quickWinScore >= 50) {
        category = 'near-term'
      }
      
      return {
        ...project,
        quick_win_score: quickWinScore,
        quick_win_reason: reasons.slice(0, 2).join(' • ') || 'Strategic opportunity',
        category
      }
    })
    
    // Filter and sort by quick win score
    return scoredProjects
      .filter(p => p.quick_win_score > 30)
      .sort((a, b) => b.quick_win_score - a.quick_win_score)
      .slice(0, 20)
  }

  const getScoreColor = (score: number) => {
    if (score >= 75) return 'from-emerald-500 to-green-500'
    if (score >= 50) return 'from-cyan-500 to-blue-500'
    return 'from-purple-500 to-pink-500'
  }

  const getCategoryBadge = (score: number) => {
    if (score >= 75) return { text: 'Immediate', color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' }
    if (score >= 50) return { text: 'Near-term', color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' }
    return { text: 'Strategic', color: 'bg-purple-500/20 text-purple-400 border-purple-500/30' }
  }

  const filteredWins = filterCategory === 'all' 
    ? quickWins
    : quickWins.filter(w => {
        if (filterCategory === 'immediate') return w.quick_win_score >= 75
        if (filterCategory === 'near-term') return w.quick_win_score >= 50 && w.quick_win_score < 75
        if (filterCategory === 'strategic') return w.quick_win_score < 50
        return true
      })

  useEffect(() => {
    findQuickWins()
  }, [findQuickWins])

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-white mb-3">
          <span className="text-4xl mr-3">⚡</span>
          Quick Win Finder
        </h2>
        <p className="text-white/60 max-w-2xl mx-auto">
          AI identifies projects with immediate returns, low barriers to entry, and rapid deployment potential
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
        {/* Category Filter */}
        <div className="flex gap-2">
          {(['all', 'immediate', 'near-term', 'strategic'] as const).map(cat => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={`px-4 py-2 rounded-lg border transition-all capitalize ${
                filterCategory === cat
                  ? 'bg-white/10 border-white/30 text-white'
                  : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10'
              }`}
            >
              {cat === 'all' ? 'All Opportunities' : cat}
            </button>
          ))}
        </div>

        {/* View Toggle */}
        <div className="flex gap-2 bg-white/5 rounded-lg p-1">
          <button
            onClick={() => setSelectedView('grid')}
            className={`px-3 py-1.5 rounded transition-all ${
              selectedView === 'grid'
                ? 'bg-white/20 text-white'
                : 'text-white/60 hover:text-white'
            }`}
          >
            <span className="text-sm">Grid</span>
          </button>
          <button
            onClick={() => setSelectedView('list')}
            className={`px-3 py-1.5 rounded transition-all ${
              selectedView === 'list'
                ? 'bg-white/20 text-white'
                : 'text-white/60 hover:text-white'
            }`}
          >
            <span className="text-sm">List</span>
          </button>
        </div>

        {/* Refresh Button */}
        <button
          onClick={findQuickWins}
          disabled={searching}
          className="px-6 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50"
        >
          {searching ? (
            <span className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analyzing...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              🔍 Find Quick Wins
            </span>
          )}
        </button>
      </div>

      {/* Results Summary */}
      {quickWins.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4">
            <div className="text-3xl font-light text-emerald-400">
              {quickWins.filter(w => w.quick_win_score >= 75).length}
            </div>
            <div className="text-sm text-white/60 mt-1">Immediate Opportunities</div>
            <div className="text-xs text-white/40 mt-2">Deploy within 6 months</div>
          </div>
          <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-xl p-4">
            <div className="text-3xl font-light text-cyan-400">
              {quickWins.filter(w => w.quick_win_score >= 50 && w.quick_win_score < 75).length}
            </div>
            <div className="text-sm text-white/60 mt-1">Near-term Wins</div>
            <div className="text-xs text-white/40 mt-2">6-12 month horizon</div>
          </div>
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
            <div className="text-3xl font-light text-purple-400">
              ${(quickWins.slice(0, 5).reduce((sum, w) => sum + w.investment_million, 0) / 1000).toFixed(1)}B
            </div>
            <div className="text-sm text-white/60 mt-1">Top 5 Investment</div>
            <div className="text-xs text-white/40 mt-2">Total capital required</div>
          </div>
        </div>
      )}

      {/* Quick Wins Display */}
      <AnimatePresence mode="wait">
        {searching ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center py-20"
          >
            <div className="text-center">
              <div className="w-16 h-16 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mx-auto mb-4" />
              <p className="text-white/60">Analyzing opportunities...</p>
            </div>
          </motion.div>
        ) : filteredWins.length > 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {selectedView === 'grid' ? (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredWins.map((win, index) => {
                  const badge = getCategoryBadge(win.quick_win_score)
                  return (
                    <motion.div
                      key={win.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="group relative"
                    >
                      <div className="absolute -inset-0.5 bg-gradient-to-r opacity-0 group-hover:opacity-100 blur-sm transition-opacity from-emerald-500/20 to-cyan-500/20 rounded-xl" />
                      
                      <div className="relative bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:border-white/20 transition-all">
                        {/* Quick Win Score with Intelligence Display */}
                        <div className="absolute top-4 right-4">
                          <IntelligenceScoreDisplay
                            score={win.quick_win_score}
                            size="small"
                            showDetails={false}
                            showAnimation={true}
                          />
                        </div>

                        {/* Category Badge */}
                        <div className={`inline-flex px-3 py-1 rounded-full text-xs border ${badge.color} mb-3`}>
                          {badge.text}
                        </div>

                        {/* Project Info */}
                        <h3 className="text-lg font-light text-white mb-2 pr-20">
                          {win.name}
                        </h3>
                        
                        <div className="flex items-center gap-3 text-sm text-white/60 mb-3">
                          <span className={`px-2 py-0.5 rounded ${
                            win.type === 'solar' ? 'bg-amber-500/20 text-amber-300' :
                            win.type === 'wind' ? 'bg-blue-500/20 text-blue-300' :
                            win.type === 'hydro' ? 'bg-emerald-500/20 text-emerald-300' :
                            'bg-purple-500/20 text-purple-300'
                          }`}>
                            {win.type}
                          </span>
                          <span>{win.capacity_mw} MW</span>
                          <span>{win.state}</span>
                        </div>

                        {/* Quick Win Reason */}
                        <div className="text-sm text-emerald-400 mb-4">
                          ✓ {win.quick_win_reason}
                        </div>

                        {/* Key Metrics */}
                        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-white/10">
                          <div>
                            <div className="text-xs text-white/40">Investment</div>
                            <div className="text-white">${win.investment_million}M</div>
                          </div>
                          <div>
                            <div className="text-xs text-white/40">AI Score</div>
                            <div className="text-white">{win.score}/100</div>
                          </div>
                        </div>

                        {/* CTA */}
                        <Link 
                          href={`/project/${win.id}`}
                          className="mt-4 w-full py-2 bg-white/10 hover:bg-white/20 rounded-lg text-center text-white/80 hover:text-white transition-all flex items-center justify-center gap-2"
                        >
                          View Details
                          <span className="opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                        </Link>
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredWins.map((win, index) => {
                  const badge = getCategoryBadge(win.quick_win_score)
                  return (
                    <motion.div
                      key={win.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.03 }}
                      className="bg-white/5 backdrop-blur-md border border-white/10 rounded-lg p-5 hover:bg-white/10 hover:border-white/20 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-4 mb-2">
                            <div className={`px-3 py-1 rounded-full text-xs border ${badge.color}`}>
                              {badge.text}
                            </div>
                            <h3 className="text-white font-light">{win.name}</h3>
                            <span className={`px-2 py-0.5 rounded text-xs ${
                              win.type === 'solar' ? 'bg-amber-500/20 text-amber-300' :
                              win.type === 'wind' ? 'bg-blue-500/20 text-blue-300' :
                              win.type === 'hydro' ? 'bg-emerald-500/20 text-emerald-300' :
                              'bg-purple-500/20 text-purple-300'
                            }`}>
                              {win.type}
                            </span>
                          </div>
                          <div className="flex items-center gap-6 text-sm text-white/60">
                            <span>{win.capacity_mw} MW</span>
                            <span>{win.state}</span>
                            <span>${win.investment_million}M</span>
                            <span className="text-emerald-400">✓ {win.quick_win_reason}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <IntelligenceScoreDisplay
                            score={win.quick_win_score}
                            size="small"
                            showDetails={false}
                            showAnimation={false}
                          />
                          <div className="text-center">
                            <div className="text-2xl font-light text-white">
                              {win.score}
                            </div>
                            <div className="text-xs text-white/40">AI Score</div>
                          </div>
                          <Link 
                            href={`/project/${win.id}`}
                            className="px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 rounded-lg text-emerald-400 hover:bg-emerald-500/30 transition-all"
                          >
                            View →
                          </Link>
                        </div>
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <div className="text-6xl mb-4">🔍</div>
            <p className="text-white/60">No quick wins found matching criteria</p>
            <button
              onClick={findQuickWins}
              className="mt-4 px-6 py-2 bg-white/10 border border-white/20 rounded-lg text-white/80 hover:bg-white/20 transition-all"
            >
              Try Again
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tips Section */}
      {quickWins.length > 0 && (
        <div className="bg-gradient-to-r from-emerald-500/5 to-cyan-500/5 rounded-2xl border border-white/10 p-6">
          <h3 className="text-lg font-light text-white mb-4">
            💡 Quick Win Strategy Tips
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-white/60">
            <div className="flex items-start gap-2">
              <span className="text-emerald-400">•</span>
              <p>Focus on projects with &lt;3 year payback for fastest capital recovery</p>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400">•</span>
              <p>Permitted projects can start generating returns 6-12 months faster</p>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400">•</span>
              <p>Grid-ready sites save $5-20M and 12-24 months in interconnection</p>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400">•</span>
              <p>Operating assets offer immediate cash flow with proven performance</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}