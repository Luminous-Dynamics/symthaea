// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

interface NationalData {
  summary: {
    total_projects: number
    total_capacity_gw: number
    total_capacity_tw: number
    total_investment_billion: number
    total_co2_avoided_million_tons: number
    total_construction_jobs: number
    total_permanent_jobs: number
    states_covered: number
  }
  scorecard: {
    target_clean_gw: number
    pipeline_gw: number
    progress_pct: number
    gap_gw: number
    note: string
  }
  by_state: { state: string; projects: number; capacity_mw: number; co2_avoided: number }[]
  by_technology: { type: string; projects: number; capacity_mw: number; avg_capacity_mw: number }[]
  by_status: { status: string; projects: number; capacity_mw: number }[]
  top_trust_scored: { name: string; type: string; capacity_mw: number; state: string; trust_score: number; harmony_alignment: number }[]
}

export default function NationalDashboard() {
  const [data, setData] = useState<NationalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [stateFilter, setStateFilter] = useState('')

  useEffect(() => {
    async function fetchData() {
      try {
        const url = stateFilter
          ? `/api/coordinator/national?state=${stateFilter}`
          : '/api/coordinator/national'
        const res = await fetch(url)
        if (res.ok) setData(await res.json())
      } catch (e) {
        console.error('Failed to load national data:', e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [stateFilter])

  if (loading) {
    return (
      <main className="min-h-screen bg-gray-950 text-white p-8">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-800 rounded w-1/3" />
          <div className="grid grid-cols-4 gap-4">
            {[1,2,3,4].map(i => <div key={i} className="h-24 bg-gray-800 rounded" />)}
          </div>
        </div>
      </main>
    )
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-gray-950 text-white p-8">
        <p className="text-gray-500">Failed to load national data. Ensure PostgreSQL is running with the terra_atlas database.</p>
      </main>
    )
  }

  const { summary, scorecard, by_state, by_technology, by_status, top_trust_scored } = data

  return (
    <main className="min-h-screen bg-gray-950 text-white p-8">
      <header className="mb-8">
        <Link href="/dashboard" className="text-gray-400 hover:text-white text-sm mb-2 inline-block">&larr; Dashboard</Link>
        <h1 className="text-3xl font-bold tracking-tight">National Energy Transition</h1>
        <p className="text-gray-400 mt-1">
          {summary.total_projects.toLocaleString()} real projects across {summary.states_covered} states
        </p>
      </header>

      {/* Scorecard */}
      <div className="bg-gradient-to-r from-violet-900/20 to-emerald-900/20 rounded-xl p-6 border border-violet-500/20 mb-8">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">2035 Clean Electricity Scorecard</h2>
          <span className="text-xs text-gray-400">{scorecard.note}</span>
        </div>
        <div className="flex items-end gap-4 mb-3">
          <span className="text-5xl font-bold text-emerald-400">{scorecard.pipeline_gw.toLocaleString()}</span>
          <span className="text-gray-400 text-lg mb-1">/ {scorecard.target_clean_gw.toLocaleString()} GW target</span>
        </div>
        <div className="h-4 bg-gray-800 rounded-full overflow-hidden mb-2">
          <div
            className="h-full bg-gradient-to-r from-violet-500 via-emerald-500 to-green-400 rounded-full transition-all duration-1000"
            style={{ width: `${Math.min(scorecard.progress_pct, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-400">
          <span>{scorecard.progress_pct.toFixed(1)}% of target in pipeline</span>
          <span>Gap: {scorecard.gap_gw > 0 ? scorecard.gap_gw.toLocaleString() + ' GW' : 'Target exceeded'}</span>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Capacity" value={`${summary.total_capacity_tw} TW`} sub={`${summary.total_capacity_gw.toLocaleString()} GW`} color="text-emerald-400" />
        <StatCard label="Total Investment" value={`$${summary.total_investment_billion}B`} sub="estimated" color="text-blue-400" />
        <StatCard label="CO2 Avoided" value={`${summary.total_co2_avoided_million_tons}M`} sub="tonnes/year" color="text-green-400" />
        <StatCard label="Jobs" value={(summary.total_construction_jobs + summary.total_permanent_jobs).toLocaleString()} sub={`${summary.total_construction_jobs.toLocaleString()} construction`} color="text-cyan-400" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* By State */}
        <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">By State</h3>
            {stateFilter && (
              <button onClick={() => setStateFilter('')} className="text-xs text-violet-400 hover:text-violet-300">
                Clear filter
              </button>
            )}
          </div>
          <div className="space-y-1.5 max-h-96 overflow-y-auto">
            {by_state.map((s, i) => {
              const maxCap = by_state[0]?.capacity_mw || 1
              const pct = (s.capacity_mw / maxCap) * 100
              return (
                <button
                  key={s.state}
                  onClick={() => setStateFilter(s.state)}
                  className={`w-full text-left flex items-center gap-3 p-2 rounded-lg transition-colors ${
                    stateFilter === s.state ? 'bg-violet-900/30 border border-violet-500/30' : 'hover:bg-gray-800/50'
                  }`}
                >
                  <span className="font-mono text-xs text-gray-500 w-5">{i + 1}</span>
                  <span className="font-mono font-bold text-sm w-8">{s.state}</span>
                  <div className="flex-1">
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-emerald-600 to-green-400 rounded-full" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                  <span className="font-mono text-xs text-gray-400 w-20 text-right">{(s.capacity_mw / 1000).toFixed(0)} GW</span>
                  <span className="text-xs text-gray-500 w-16 text-right">{s.projects.toLocaleString()}</span>
                </button>
              )
            })}
          </div>
        </div>

        {/* By Technology */}
        <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">By Technology</h3>
          <div className="space-y-3">
            {by_technology.map(t => {
              const colors: Record<string, string> = {
                'Hydro': 'text-blue-400', 'Solar': 'text-amber-400', 'Solar+Storage': 'text-amber-300',
                'Wind': 'text-cyan-400', 'Wind+Storage': 'text-cyan-300',
                'Battery Storage': 'text-purple-400', 'Offshore Wind': 'text-teal-400', 'Nuclear': 'text-red-400',
              }
              return (
                <div key={t.type} className="flex items-center justify-between">
                  <span className={`text-sm ${colors[t.type] || 'text-gray-300'}`}>{t.type}</span>
                  <div className="text-right">
                    <span className="font-mono text-xs text-gray-400">{(t.capacity_mw / 1000).toFixed(0)} GW</span>
                    <span className="text-xs text-gray-600 ml-2">({t.projects.toLocaleString()})</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Top Trust-Scored Projects */}
      {top_trust_scored.length > 0 && (
        <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 mb-8">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Highest Trust-Scored Projects</h3>
          <div className="space-y-2">
            {top_trust_scored.slice(0, 10).map((p, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800/50">
                <span className={`font-mono font-bold text-sm w-6 text-center ${i === 0 ? 'text-amber-400' : i < 3 ? 'text-gray-300' : 'text-gray-500'}`}>
                  {i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-gray-200 truncate block">{p.name}</span>
                  <span className="text-xs text-gray-500">{p.type} &middot; {p.capacity_mw} MW &middot; {p.state}</span>
                </div>
                <span className="font-mono text-sm text-violet-400" title="Trust Score">
                  &#934; {(p.trust_score * 100).toFixed(0)}
                </span>
                <span className="font-mono text-xs text-blue-400" title="Harmony Alignment">
                  &#8801; {(p.harmony_alignment * 100).toFixed(0)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* By Status */}
      <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Pipeline Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {by_status.slice(0, 8).map(s => (
            <div key={s.status} className="bg-gray-800/50 rounded-lg p-3">
              <span className="text-xs text-gray-400 block mb-1">{s.status}</span>
              <span className="font-mono text-sm text-gray-200">{s.projects.toLocaleString()}</span>
              <span className="text-xs text-gray-500 ml-1">({(s.capacity_mw / 1000).toFixed(0)} GW)</span>
            </div>
          ))}
        </div>
      </div>

      <footer className="mt-12 text-center text-xs text-gray-600">
        <p>Data sources: USACE National Inventory of Dams, FERC Interconnection Queue, NRC SMR Pipeline</p>
        <p className="mt-1">National Energy Transition Coordinator v0.1.0</p>
      </footer>
    </main>
  )
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
      <span className="text-xs text-gray-400 block mb-1">{label}</span>
      <span className={`font-mono text-2xl font-bold ${color}`}>{value}</span>
      <span className="text-xs text-gray-500 block mt-0.5">{sub}</span>
    </div>
  )
}
