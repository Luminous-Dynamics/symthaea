'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface IntelligenceScoreDisplayProps {
  score: number
  projectName?: string
  size?: 'small' | 'medium' | 'large' | 'xlarge'
  showDetails?: boolean
  showAnimation?: boolean
  showBreakdown?: boolean
  economicScore?: number
  environmentalScore?: number
  socialScore?: number
  technicalScore?: number
}

export default function IntelligenceScoreDisplay({
  score,
  projectName,
  size = 'medium',
  showDetails = true,
  showAnimation = true,
  showBreakdown = false,
  economicScore,
  environmentalScore,
  socialScore,
  technicalScore
}: IntelligenceScoreDisplayProps) {
  const [animatedScore, setAnimatedScore] = useState(showAnimation ? 0 : score)
  const [showTooltip, setShowTooltip] = useState(false)

  useEffect(() => {
    if (showAnimation) {
      const timer = setTimeout(() => {
        setAnimatedScore(score)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [score, showAnimation])

  // Determine score color and status
  const getScoreColor = () => {
    if (score >= 90) return {
      gradient: 'from-emerald-400 to-green-500',
      glow: 'shadow-emerald-500/50',
      text: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      status: 'Elite',
      icon: '⭐'
    }
    if (score >= 80) return {
      gradient: 'from-cyan-400 to-blue-500',
      glow: 'shadow-cyan-500/50',
      text: 'text-cyan-400',
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/30',
      status: 'Premium',
      icon: '✨'
    }
    if (score >= 70) return {
      gradient: 'from-purple-400 to-pink-500',
      glow: 'shadow-purple-500/50',
      text: 'text-purple-400',
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      status: 'Strong',
      icon: '📈'
    }
    return {
      gradient: 'from-amber-400 to-orange-500',
      glow: 'shadow-amber-500/50',
      text: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      status: 'Strategic',
      icon: '🔍'
    }
  }

  const scoreStyle = getScoreColor()

  // Size configurations
  const sizeConfig = {
    small: {
      container: 'w-16 h-16',
      text: 'text-lg',
      label: 'text-xs',
      ring: 'w-14 h-14',
      strokeWidth: 4
    },
    medium: {
      container: 'w-24 h-24',
      text: 'text-2xl',
      label: 'text-sm',
      ring: 'w-20 h-20',
      strokeWidth: 6
    },
    large: {
      container: 'w-32 h-32',
      text: 'text-3xl',
      label: 'text-base',
      ring: 'w-28 h-28',
      strokeWidth: 8
    },
    xlarge: {
      container: 'w-40 h-40',
      text: 'text-4xl',
      label: 'text-lg',
      ring: 'w-36 h-36',
      strokeWidth: 10
    }
  }

  const config = sizeConfig[size]

  // Calculate ring progress
  const circumference = 2 * Math.PI * 40
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference

  return (
    <div className="relative inline-block">
      {/* Main Score Display */}
      <div 
        className="relative"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {/* Glowing Background */}
        {showAnimation && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className={`absolute inset-0 ${config.container} bg-gradient-to-br ${scoreStyle.gradient} opacity-20 blur-xl`}
          />
        )}

        {/* Circular Progress Ring */}
        <div className={`relative ${config.container}`}>
          <svg className="absolute inset-0 -rotate-90" viewBox="0 0 100 100">
            {/* Background Circle */}
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="currentColor"
              strokeWidth={config.strokeWidth}
              className="text-white/10"
            />
            {/* Progress Circle */}
            <motion.circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              strokeWidth={config.strokeWidth}
              strokeLinecap="round"
              strokeDasharray={strokeDasharray}
              initial={{ strokeDashoffset: strokeDasharray }}
              animate={{ strokeDashoffset }}
              transition={{ 
                duration: showAnimation ? 1.5 : 0,
                ease: "easeInOut"
              }}
              className={`${scoreStyle.text}`}
            />
          </svg>

          {/* Score Number */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className={`${config.text} font-light ${scoreStyle.text}`}
            >
              {Math.round(animatedScore)}
            </motion.div>
            {size !== 'small' && (
              <div className={`${config.label} text-white/40`}>Score</div>
            )}
          </div>
        </div>

        {/* Status Badge */}
        {showDetails && size !== 'small' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className={`absolute -bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-1 px-3 py-1 rounded-full ${scoreStyle.bg} border ${scoreStyle.border}`}
          >
            <span>{scoreStyle.icon}</span>
            <span className={`text-xs ${scoreStyle.text}`}>{scoreStyle.status}</span>
          </motion.div>
        )}
      </div>

      {/* Tooltip with Breakdown */}
      <AnimatePresence>
        {showTooltip && showBreakdown && (economicScore || environmentalScore || socialScore || technicalScore) && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 z-50"
          >
            <div className="bg-black/95 backdrop-blur-xl border border-white/20 rounded-xl p-4 min-w-[250px] shadow-2xl">
              {projectName && (
                <div className="text-white text-sm font-medium mb-3 pb-3 border-b border-white/10">
                  {projectName}
                </div>
              )}
              
              <div className="space-y-2">
                {/* Overall Score */}
                <div className="flex justify-between items-center mb-3">
                  <span className="text-white/60 text-sm">Intelligence Score</span>
                  <span className={`text-lg font-medium ${scoreStyle.text}`}>{score}</span>
                </div>

                {/* Breakdown Bars */}
                {economicScore !== undefined && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Economic (30%)</span>
                      <span className="text-emerald-400">{Math.round(economicScore)}</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${economicScore}%` }}
                        className="h-full bg-gradient-to-r from-emerald-500 to-green-500"
                      />
                    </div>
                  </div>
                )}

                {environmentalScore !== undefined && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Environmental (25%)</span>
                      <span className="text-cyan-400">{Math.round(environmentalScore)}</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${environmentalScore}%` }}
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      />
                    </div>
                  </div>
                )}

                {socialScore !== undefined && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Social (20%)</span>
                      <span className="text-purple-400">{Math.round(socialScore)}</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${socialScore}%` }}
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                      />
                    </div>
                  </div>
                )}

                {technicalScore !== undefined && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Technical (25%)</span>
                      <span className="text-amber-400">{Math.round(technicalScore)}</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${technicalScore}%` }}
                        className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Insight */}
              <div className="mt-3 pt-3 border-t border-white/10">
                <p className="text-xs text-white/60">
                  {score >= 90 ? '🎯 Top 5% opportunity - immediate action recommended' :
                   score >= 80 ? '✨ High-potential project with strong fundamentals' :
                   score >= 70 ? '📊 Solid opportunity with balanced risk-return' :
                   '🔍 Strategic play requiring deeper analysis'}
                </p>
              </div>
            </div>

            {/* Arrow */}
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-white/20" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}