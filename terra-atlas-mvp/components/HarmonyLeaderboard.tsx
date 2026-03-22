'use client'

import { useEffect, useState } from 'react'
import { ConsciousnessScore } from './ConsciousnessScore'

interface LeaderboardProject {
  id: string
  name: string
  type: string
  capacity_mw: number | null
  consciousness?: {
    phi_score: number
    harmony_alignment: number
  } | null
  discovery_score: number
  carbon_avoided_tons_per_year?: number
}

interface HarmonyLeaderboardProps {
  limit?: number
  className?: string
}

/**
 * Ranks projects by Net Humanity Benefit / discovery score.
 * Uses sort=harmony on the projects API for consciousness-weighted ordering.
 * Projects with higher consciousness scores surface first.
 */
export function HarmonyLeaderboard({ limit = 10, className = '' }: HarmonyLeaderboardProps) {
  const [projects, setProjects] = useState<LeaderboardProject[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        const res = await fetch(`/api/projects?sort=harmony&limit=${limit}`)
        if (res.ok) {
          const data = await res.json()
          setProjects(data.projects || [])
        }
      } catch {
        // API not available
      } finally {
        setLoading(false)
      }
    }
    fetchLeaderboard()
  }, [limit])

  if (loading) {
    return (
      <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 ${className}`}>
        <div className="h-5 bg-gray-700 rounded w-1/3 mb-4" />
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-12 bg-gray-800 rounded mb-2 animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-200">Harmony Leaderboard</h3>
        <span className="text-xs text-gray-500">Ranked by consciousness + impact</span>
      </div>

      {projects.length === 0 ? (
        <p className="text-gray-500 text-sm">No projects with consciousness data yet.</p>
      ) : (
        <div className="space-y-2">
          {projects.map((project, index) => (
            <div
              key={project.id}
              className="flex items-center gap-3 rounded-lg bg-gray-800/40 hover:bg-gray-800/70 transition-colors p-2.5"
            >
              <span className={`
                font-mono font-bold text-sm w-6 text-center
                ${index === 0 ? 'text-amber-400' : index === 1 ? 'text-gray-300' : index === 2 ? 'text-orange-400' : 'text-gray-500'}
              `}>
                {index + 1}
              </span>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-200 truncate">{project.name}</span>
                  <span className="text-xs text-gray-500 shrink-0">{project.type}</span>
                </div>
                {project.capacity_mw && (
                  <span className="text-xs text-gray-500">{project.capacity_mw} MW</span>
                )}
              </div>

              <ConsciousnessScore
                phiScore={project.consciousness?.phi_score}
                harmonyAlignment={project.consciousness?.harmony_alignment}
                size="sm"
                showLabel={false}
              />

              <span className="font-mono text-xs text-gray-400 w-12 text-right" title="Discovery Score">
                {project.discovery_score?.toFixed(2) || '---'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
