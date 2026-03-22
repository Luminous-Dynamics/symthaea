'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import IntelligenceScoreDisplay from './IntelligenceScoreDisplay'

interface OptimizationCriteria {
  maxInvestment: number
  minIRR: number
  maxPayback: number
  preferredTypes: string[]
  riskTolerance: 'conservative' | 'moderate' | 'aggressive'
  diversification: boolean
  esgFocus: boolean
}

interface OptimizedPortfolio {
  projects: any[]
  totalInvestment: number
  projectedIRR: number
  avgPayback: number
  riskScore: number
  diversificationScore: number
  esgScore: number
}

export default function PortfolioOptimizer() {
  const [criteria, setCriteria] = useState<OptimizationCriteria>({
    maxInvestment: 10000000, // $10M default
    minIRR: 12,
    maxPayback: 7,
    preferredTypes: ['solar', 'wind'],
    riskTolerance: 'moderate',
    diversification: true,
    esgFocus: false
  })

  const [optimizing, setOptimizing] = useState(false)
  const [portfolio, setPortfolio] = useState<OptimizedPortfolio | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const runOptimization = async () => {
    setOptimizing(true)
    
    try {
      // Fetch projects that meet basic criteria
      const response = await fetch('/api/intelligence/top-projects?limit=100')
      const data = await response.json()
      
      if (data.projects && data.projects.length > 0) {
        // Run optimization algorithm
        const optimized = optimizePortfolio(data.projects, criteria)
        setPortfolio(optimized)
      }
    } catch (error) {
      console.error('Optimization error:', error)
    } finally {
      setOptimizing(false)
    }
  }

  const optimizePortfolio = (projects: any[], criteria: OptimizationCriteria): OptimizedPortfolio => {
    // Filter by basic criteria
    let eligible = projects.filter((p: any) => {
      const irr = p.data?.irr || p.data?.estimated_irr || 10
      const payback = p.data?.payback_years || 10
      
      return irr >= criteria.minIRR && 
             payback <= criteria.maxPayback &&
             (criteria.preferredTypes.length === 0 || criteria.preferredTypes.includes(p.type))
    })

    // Apply risk tolerance filter
    if (criteria.riskTolerance === 'conservative') {
      eligible = eligible.filter((p: any) => p.score >= 85)
    } else if (criteria.riskTolerance === 'moderate') {
      eligible = eligible.filter((p: any) => p.score >= 70)
    }

    // Sort by optimization score
    eligible.sort((a: any, b: any) => {
      let scoreA = a.score
      let scoreB = b.score
      
      // Boost ESG-friendly projects if focus is enabled
      if (criteria.esgFocus) {
        const esgBoostA = (a.data?.environmental_score || 0) * 0.2
        const esgBoostB = (b.data?.environmental_score || 0) * 0.2
        scoreA += esgBoostA
        scoreB += esgBoostB
      }
      
      return scoreB - scoreA
    })

    // Build portfolio within investment limit
    const selected: any[] = []
    let totalInvestment = 0
    const typeCount: Record<string, number> = {}

    for (const project of eligible) {
      const investment = project.investment_million * 1000000
      
      // Check if we can afford this project
      if (totalInvestment + investment <= criteria.maxInvestment) {
        // Check diversification requirements
        if (criteria.diversification) {
          const projectType = project.type
          if (typeCount[projectType] >= 2) {
            continue // Skip to maintain diversity
          }
        }
        
        selected.push(project)
        totalInvestment += investment
        typeCount[project.type] = (typeCount[project.type] || 0) + 1
        
        // Stop if we've selected enough projects
        if (selected.length >= 10) break
      }
    }

    // Calculate portfolio metrics
    const avgIRR = selected.reduce((sum, p) => 
      sum + (p.data?.irr || p.data?.estimated_irr || 10), 0) / selected.length
    
    const avgPayback = selected.reduce((sum, p) => 
      sum + (p.data?.payback_years || 7), 0) / selected.length
    
    const avgScore = selected.reduce((sum, p) => sum + p.score, 0) / selected.length
    
    const diversificationScore = Object.keys(typeCount).length / 5 * 100 // 5 main types
    
    const esgScore = selected.reduce((sum, p) => 
      sum + (p.data?.environmental_score || 70), 0) / selected.length

    return {
      projects: selected,
      totalInvestment: totalInvestment / 1000000, // Convert to millions
      projectedIRR: avgIRR,
      avgPayback: avgPayback,
      riskScore: avgScore,
      diversificationScore,
      esgScore
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value)
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-light text-white mb-3">
          <span className="text-4xl mr-3">🎯</span>
          AI Portfolio Optimizer
        </h2>
        <p className="text-white/60 max-w-2xl mx-auto">
          Our AI analyzes thousands of combinations to build your optimal energy investment portfolio
        </p>
      </div>

      {/* Optimization Criteria */}
      <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6">
        <h3 className="text-xl font-light text-white mb-6">Investment Criteria</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Investment Amount */}
          <div>
            <label className="block text-sm text-white/60 mb-2">
              Maximum Investment
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40">$</span>
              <input
                type="number"
                value={criteria.maxInvestment}
                onChange={(e) => setCriteria({ ...criteria, maxInvestment: Number(e.target.value) })}
                className="w-full pl-8 pr-3 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
                step="1000000"
              />
            </div>
            <p className="text-xs text-white/40 mt-1">
              {formatCurrency(criteria.maxInvestment)}
            </p>
          </div>

          {/* Minimum IRR */}
          <div>
            <label className="block text-sm text-white/60 mb-2">
              Minimum IRR Target
            </label>
            <div className="relative">
              <input
                type="number"
                value={criteria.minIRR}
                onChange={(e) => setCriteria({ ...criteria, minIRR: Number(e.target.value) })}
                className="w-full pr-8 px-3 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
                step="1"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40">%</span>
            </div>
          </div>

          {/* Risk Tolerance */}
          <div>
            <label className="block text-sm text-white/60 mb-2">
              Risk Tolerance
            </label>
            <div className="flex gap-2">
              {(['conservative', 'moderate', 'aggressive'] as const).map((risk) => (
                <button
                  key={risk}
                  onClick={() => setCriteria({ ...criteria, riskTolerance: risk })}
                  className={`flex-1 py-3 rounded-lg border transition-all capitalize ${
                    criteria.riskTolerance === risk
                      ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                      : 'bg-white/5 border-white/20 text-white/60 hover:bg-white/10'
                  }`}
                >
                  {risk}
                </button>
              ))}
            </div>
          </div>

          {/* Payback Period */}
          <div>
            <label className="block text-sm text-white/60 mb-2">
              Maximum Payback Period
            </label>
            <div className="relative">
              <input
                type="number"
                value={criteria.maxPayback}
                onChange={(e) => setCriteria({ ...criteria, maxPayback: Number(e.target.value) })}
                className="w-full px-3 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
                step="1"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40">years</span>
            </div>
          </div>
        </div>

        {/* Energy Type Preferences */}
        <div className="mt-6">
          <label className="block text-sm text-white/60 mb-2">
            Preferred Energy Types
          </label>
          <div className="flex flex-wrap gap-2">
            {['solar', 'wind', 'hydro', 'nuclear', 'geothermal', 'battery'].map((type) => (
              <button
                key={type}
                onClick={() => {
                  const types = criteria.preferredTypes.includes(type)
                    ? criteria.preferredTypes.filter(t => t !== type)
                    : [...criteria.preferredTypes, type]
                  setCriteria({ ...criteria, preferredTypes: types })
                }}
                className={`px-4 py-2 rounded-full border transition-all capitalize ${
                  criteria.preferredTypes.includes(type)
                    ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400'
                    : 'bg-white/5 border-white/20 text-white/60 hover:bg-white/10'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="mt-6 text-sm text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-2"
        >
          {showAdvanced ? '▼' : '▶'} Advanced Options
        </button>

        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="flex gap-6 mt-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={criteria.diversification}
                    onChange={(e) => setCriteria({ ...criteria, diversification: e.target.checked })}
                    className="w-5 h-5 rounded bg-white/10 border border-white/20 checked:bg-emerald-500"
                  />
                  <span className="text-white/60">Maximize Diversification</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={criteria.esgFocus}
                    onChange={(e) => setCriteria({ ...criteria, esgFocus: e.target.checked })}
                    className="w-5 h-5 rounded bg-white/10 border border-white/20 checked:bg-emerald-500"
                  />
                  <span className="text-white/60">ESG Focus</span>
                </label>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Optimize Button */}
        <button
          onClick={runOptimization}
          disabled={optimizing}
          className="mt-8 w-full py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl text-white font-medium hover:shadow-lg hover:shadow-emerald-500/25 transition-all transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {optimizing ? (
            <span className="flex items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Optimizing Portfolio...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <span className="text-xl">🚀</span>
              Optimize Portfolio
            </span>
          )}
        </button>
      </div>

      {/* Optimized Portfolio Results */}
      <AnimatePresence>
        {portfolio && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 backdrop-blur-md rounded-2xl border border-emerald-500/20 p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-light text-white">
                <span className="text-2xl mr-2">✨</span>
                Optimized Portfolio
              </h3>
              <div className="text-right">
                <div className="text-2xl font-light text-emerald-400">
                  {portfolio.projectedIRR.toFixed(1)}%
                </div>
                <div className="text-xs text-white/60 uppercase">Projected IRR</div>
              </div>
            </div>

            {/* Portfolio Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-sm text-white/60 mb-1">Total Investment</div>
                <div className="text-lg text-white">${portfolio.totalInvestment.toFixed(1)}M</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-sm text-white/60 mb-1">Avg Payback</div>
                <div className="text-lg text-white">{portfolio.avgPayback.toFixed(1)} years</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-sm text-white/60 mb-1">Risk Score</div>
                <div className="flex justify-center mt-1">
                  <IntelligenceScoreDisplay
                    score={portfolio.riskScore}
                    size="small"
                    showDetails={false}
                    showAnimation={true}
                  />
                </div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-sm text-white/60 mb-1">Diversification</div>
                <div className="text-lg text-white">{portfolio.diversificationScore.toFixed(0)}%</div>
              </div>
            </div>

            {/* Selected Projects */}
            <div className="space-y-3">
              <h4 className="text-sm text-white/60 uppercase tracking-wider">
                Selected Projects ({portfolio.projects.length})
              </h4>
              {portfolio.projects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white/5 rounded-lg p-4 flex items-center justify-between hover:bg-white/10 transition-colors cursor-pointer"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-white/40 text-sm">#{index + 1}</span>
                      <h5 className="text-white">{project.name}</h5>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        project.type === 'solar' ? 'bg-amber-500/20 text-amber-300' :
                        project.type === 'wind' ? 'bg-blue-500/20 text-blue-300' :
                        project.type === 'hydro' ? 'bg-emerald-500/20 text-emerald-300' :
                        'bg-purple-500/20 text-purple-300'
                      }`}>
                        {project.type}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 mt-1 text-sm text-white/60">
                      <span>{project.capacity_mw} MW</span>
                      <span>${project.investment_million}M</span>
                      <span className="text-emerald-400">
                        {(project.data?.irr || project.data?.estimated_irr || 12).toFixed(1)}% IRR
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-end">
                    <IntelligenceScoreDisplay
                      score={project.score}
                      size="small"
                      showDetails={false}
                      showAnimation={false}
                    />
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 mt-6">
              <button className="flex-1 py-3 bg-emerald-500/20 border border-emerald-500/40 rounded-lg text-emerald-400 hover:bg-emerald-500/30 transition-all">
                Export Portfolio
              </button>
              <button className="flex-1 py-3 bg-cyan-500/20 border border-cyan-500/40 rounded-lg text-cyan-400 hover:bg-cyan-500/30 transition-all">
                Save & Track
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}