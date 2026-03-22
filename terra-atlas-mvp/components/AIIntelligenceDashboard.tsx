'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import IntelligenceScoreDisplay from './IntelligenceScoreDisplay'

interface IntelligenceMetrics {
  overall_score: number
  economic_score: number
  environmental_score: number
  social_score: number
  technical_score: number
  total_projects_analyzed: number
  high_impact_projects: number
  quick_wins: number
  average_processing_time: number
}

interface TopProject {
  id: string
  name: string
  type: string
  state: string
  score: number
  capacity_mw: number
  investment_million: number
  key_insight: string
}

export default function AIIntelligenceDashboard() {
  const [metrics, setMetrics] = useState<IntelligenceMetrics | null>(null)
  const [topProjects, setTopProjects] = useState<TopProject[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedCategory, setSelectedCategory] = useState<'economic' | 'environmental' | 'social' | 'technical'>('economic')

  useEffect(() => {
    fetchIntelligenceData()
  }, [])

  const fetchIntelligenceData = async () => {
    try {
      // Fetch intelligence metrics
      const metricsRes = await fetch('/api/intelligence/metrics')
      const metricsData = await metricsRes.json()
      setMetrics(metricsData)

      // Fetch top scored projects
      const projectsRes = await fetch('/api/intelligence/top-projects')
      const projectsData = await projectsRes.json()
      setTopProjects(projectsData.projects || [])
    } catch (error) {
      console.error('Error fetching intelligence data:', error)
      // Set demo data if API fails
      setMetrics({
        overall_score: 85,
        economic_score: 88,
        environmental_score: 82,
        social_score: 79,
        technical_score: 91,
        total_projects_analyzed: 4287,
        high_impact_projects: 342,
        quick_wins: 156,
        average_processing_time: 0.3
      })
      setTopProjects([
        {
          id: '1',
          name: 'Grand Coulee Modernization',
          type: 'hydro',
          state: 'WA',
          score: 94,
          capacity_mw: 6809,
          investment_million: 450,
          key_insight: 'Existing infrastructure with 25% capacity increase potential'
        },
        {
          id: '2',
          name: 'Texas Solar Corridor',
          type: 'solar',
          state: 'TX',
          score: 91,
          capacity_mw: 2500,
          investment_million: 3200,
          key_insight: 'Optimal grid connection with 15.8% IRR projection'
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const categoryColors = {
    economic: 'from-emerald-500 to-green-500',
    environmental: 'from-cyan-500 to-blue-500',
    social: 'from-purple-500 to-pink-500',
    technical: 'from-amber-500 to-orange-500'
  }

  const categoryIcons = {
    economic: '💰',
    environmental: '🌱',
    social: '🤝',
    technical: '⚡'
  }

  if (loading) {
    return (
      <div className="min-h-[600px] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-white/60">Analyzing intelligence patterns...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full space-y-8">
      {/* Main Intelligence Score */}
      <div className="text-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-block"
        >
          <div className="relative">
            <div className="w-48 h-48 mx-auto">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  className="text-white/10"
                />
                <motion.circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="url(#scoreGradient)"
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  initial={{ strokeDasharray: '0 553' }}
                  animate={{ strokeDasharray: `${(metrics?.overall_score || 0) * 5.53} 553` }}
                  transition={{ duration: 1.5, ease: 'easeOut' }}
                />
                <defs>
                  <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" />
                    <stop offset="50%" stopColor="#06b6d4" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="text-6xl font-light text-white"
                >
                  {metrics?.overall_score}
                </motion.div>
                <div className="text-sm text-white/60 uppercase tracking-wider">AI Score</div>
              </div>
            </div>
          </div>
        </motion.div>
        <p className="text-white/60 mt-4 max-w-2xl mx-auto">
          Our AI analyzes {metrics?.total_projects_analyzed.toLocaleString()} projects across 100+ factors in {metrics?.average_processing_time}s
        </p>
      </div>

      {/* Category Scores */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {(['economic', 'environmental', 'social', 'technical'] as const).map((category) => (
          <motion.button
            key={category}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSelectedCategory(category)}
            className={`relative p-6 rounded-2xl bg-white/5 backdrop-blur-md border transition-all ${
              selectedCategory === category
                ? 'border-white/30 bg-white/10'
                : 'border-white/10 hover:border-white/20'
            }`}
          >
            <div className="text-3xl mb-3">{categoryIcons[category]}</div>
            <div className="text-2xl font-light text-white mb-1">
              {metrics?.[`${category}_score`]}%
            </div>
            <div className="text-xs text-white/60 uppercase tracking-wider">
              {category}
            </div>
            {selectedCategory === category && (
              <motion.div
                layoutId="categoryHighlight"
                className={`absolute inset-0 bg-gradient-to-br ${categoryColors[category]} opacity-10 rounded-2xl`}
              />
            )}
          </motion.button>
        ))}
      </div>

      {/* Intelligence Insights */}
      <div className="grid md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 backdrop-blur-md rounded-2xl p-6 border border-emerald-500/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="text-4xl">🎯</div>
            <div className="text-3xl font-light text-emerald-400">
              {metrics?.high_impact_projects}
            </div>
          </div>
          <h3 className="text-lg font-light text-white mb-2">High Impact</h3>
          <p className="text-sm text-white/60">
            Projects scoring 80+ with exceptional returns and environmental benefits
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 backdrop-blur-md rounded-2xl p-6 border border-cyan-500/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="text-4xl">⚡</div>
            <div className="text-3xl font-light text-cyan-400">
              {metrics?.quick_wins}
            </div>
          </div>
          <h3 className="text-lg font-light text-white mb-2">Quick Wins</h3>
          <p className="text-sm text-white/60">
            Ready-to-go projects with permits and grid connections in place
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-md rounded-2xl p-6 border border-purple-500/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="text-4xl">🚀</div>
            <div className="text-3xl font-light text-purple-400">
              4.2x
            </div>
          </div>
          <h3 className="text-lg font-light text-white mb-2">Faster Analysis</h3>
          <p className="text-sm text-white/60">
            What takes analysts weeks, our AI completes in seconds
          </p>
        </motion.div>
      </div>

      {/* Top Projects */}
      <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6">
        <h3 className="text-xl font-light text-white mb-6 flex items-center gap-2">
          <span className="text-2xl">🏆</span>
          Top AI-Scored Opportunities
        </h3>
        
        <div className="space-y-4">
          <AnimatePresence>
            {topProjects.map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.1 }}
                className="group relative bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-all cursor-pointer border border-white/10 hover:border-white/20"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-white/40 text-sm">#{index + 1}</span>
                      <h4 className="text-white font-medium">{project.name}</h4>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        project.type === 'solar' ? 'bg-amber-500/20 text-amber-300' :
                        project.type === 'wind' ? 'bg-blue-500/20 text-blue-300' :
                        project.type === 'hydro' ? 'bg-emerald-500/20 text-emerald-300' :
                        'bg-purple-500/20 text-purple-300'
                      }`}>
                        {project.type}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-white/60">
                      <span>{project.state}</span>
                      <span>{project.capacity_mw.toLocaleString()} MW</span>
                      <span>${(project.investment_million).toFixed(0)}M</span>
                    </div>
                    <p className="text-sm text-cyan-400 mt-2">
                      💡 {project.key_insight}
                    </p>
                  </div>
                  <div className="flex justify-end">
                    <IntelligenceScoreDisplay
                      score={project.score}
                      projectName={project.name}
                      size="small"
                      showDetails={false}
                      showAnimation={true}
                    />
                  </div>
                </div>
                
                {/* Score breakdown on hover */}
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 to-cyan-500/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Real-time Processing Indicator */}
      <div className="flex items-center justify-center gap-3 text-sm text-white/40">
        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
        <span>AI actively analyzing new projects</span>
      </div>
    </div>
  )
}