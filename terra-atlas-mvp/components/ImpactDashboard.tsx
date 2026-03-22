'use client'

import { useEffect, useState } from 'react'

interface ImpactData {
  total_co2_avoided: number
  total_jobs_created: number
  total_trust_delta: number
  peak_households: number
  report_count: number
  net_humanity_benefit: number
}

interface ImpactDashboardProps {
  projectId: string
  className?: string
}

/**
 * Displays QOL impact metrics for a project in the Conscious Allocation Network.
 * Fetches from /api/projects/:id/impact and shows CO2, jobs, trust, households,
 * and Net Humanity Benefit composite.
 */
export function ImpactDashboard({ projectId, className = '' }: ImpactDashboardProps) {
  const [impact, setImpact] = useState<ImpactData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchImpact() {
      try {
        const res = await fetch(`/api/projects/${projectId}/impact`)
        if (res.ok) {
          const data = await res.json()
          setImpact(data.summary)
        }
      } catch {
        // Impact data not available yet
      } finally {
        setLoading(false)
      }
    }
    fetchImpact()
  }, [projectId])

  if (loading) {
    return (
      <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 animate-pulse ${className}`}>
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-3" />
        <div className="grid grid-cols-2 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-16 bg-gray-800 rounded" />
          ))}
        </div>
      </div>
    )
  }

  if (!impact || impact.report_count === 0) {
    return (
      <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 ${className}`}>
        <h3 className="text-sm font-semibold text-gray-400 mb-2">Impact Metrics</h3>
        <p className="text-gray-500 text-sm">No impact data reported yet. Impact will appear here once field reports are submitted.</p>
      </div>
    )
  }

  const nhbColor = impact.net_humanity_benefit > 0.5
    ? 'text-emerald-400'
    : impact.net_humanity_benefit > 0
      ? 'text-green-400'
      : 'text-amber-400'

  return (
    <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">Impact Metrics</h3>
        <span className="text-xs text-gray-500">{impact.report_count} report{impact.report_count !== 1 ? 's' : ''}</span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <MetricCard
          label="CO2 Avoided"
          value={`${impact.total_co2_avoided.toLocaleString()} t`}
          icon="leaf"
          color="text-green-400"
        />
        <MetricCard
          label="Jobs Created"
          value={impact.total_jobs_created.toLocaleString()}
          icon="users"
          color="text-blue-400"
        />
        <MetricCard
          label="Trust Delta"
          value={`${impact.total_trust_delta >= 0 ? '+' : ''}${(impact.total_trust_delta * 100).toFixed(1)}%`}
          icon="heart"
          color={impact.total_trust_delta >= 0 ? 'text-rose-400' : 'text-orange-400'}
        />
        <MetricCard
          label="Households"
          value={impact.peak_households.toLocaleString()}
          icon="home"
          color="text-cyan-400"
        />
      </div>

      <div className="border-t border-gray-800 pt-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Net Humanity Benefit</span>
          <span className={`font-mono font-bold text-lg ${nhbColor}`}>
            {impact.net_humanity_benefit.toFixed(3)}
          </span>
        </div>
        <div className="mt-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-green-400 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(Math.max(impact.net_humanity_benefit * 100, 0), 100)}%` }}
          />
        </div>
      </div>
    </div>
  )
}

function MetricCard({ label, value, icon, color }: {
  label: string
  value: string
  icon: string
  color: string
}) {
  const icons: Record<string, string> = {
    leaf: '\uD83C\uDF3F',
    users: '\uD83D\uDC65',
    heart: '\uD83D\uDC9A',
    home: '\uD83C\uDFE0',
  }

  return (
    <div className="rounded-lg bg-gray-800/60 p-2.5">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-sm">{icons[icon] || ''}</span>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <span className={`font-mono font-semibold text-sm ${color}`}>{value}</span>
    </div>
  )
}
