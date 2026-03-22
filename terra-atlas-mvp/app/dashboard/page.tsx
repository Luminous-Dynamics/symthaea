// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import {
  Activity, TrendingUp, Users, Zap, Wind, Sun, Battery, Droplets,
  Globe, DollarSign, Briefcase, Building, MapPin, Award, Target,
  Calendar, Clock, AlertCircle, CheckCircle, BarChart3, PieChart
} from 'lucide-react'
import { HarmonyLeaderboard } from '@/components/HarmonyLeaderboard'

// Dynamically import RealtimeCharts for better performance
const RealtimeCharts = dynamic(() => import('@/components/RealtimeCharts'), {
  loading: () => <div className="animate-pulse bg-gray-900/50 rounded-xl h-96" />,
  ssr: false
})

interface Stats {
  total_projects: number
  total_capacity_mw: number
  countries: number
  total_investment: number
  operational: number
  construction: number
  development: number
  planned: number
  solar_projects: number
  wind_projects: number
  battery_projects: number
  hydro_projects: number
}

interface TrendData {
  month: string
  projects: number
  capacity: number
  investment: number
}

interface TopProject {
  id: number
  name: string
  type: string
  capacity_mw: number
  investment: number
  state: string
  country: string
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [trends, setTrends] = useState<TrendData[]>([])
  const [topProjects, setTopProjects] = useState<TopProject[]>([])
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState('30d')
  const [selectedType, setSelectedType] = useState('all')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [selectedCountry, setSelectedCountry] = useState('all')
  const [capacityRange, setCapacityRange] = useState({ min: 0, max: 10000 })
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)

    const fetchDashboardData = async () => {
      try {
        const baseParams = new URLSearchParams()
        if (selectedType !== 'all') baseParams.append('type', selectedType)
        if (selectedStatus !== 'all') baseParams.append('status', selectedStatus)
        if (selectedCountry !== 'all') baseParams.append('country', selectedCountry)
        if (capacityRange.min > 0) baseParams.append('minCapacity', capacityRange.min.toString())
        if (capacityRange.max < 10000) baseParams.append('maxCapacity', capacityRange.max.toString())

        const statsPromise = fetch(`/api/stats?${baseParams}`, { signal: controller.signal }).then(res =>
          res.json()
        )

        const projectParams = new URLSearchParams(baseParams)
        projectParams.append('limit', '10')
        projectParams.append('sort', 'capacity')

        const projectsPromise = fetch(`/api/projects?${projectParams}`, {
          signal: controller.signal,
        }).then(res => res.json())

        const [statsData, projectsData] = await Promise.all([statsPromise, projectsPromise])

        if (controller.signal.aborted) return

        setStats(statsData)
        setTopProjects(projectsData.projects || [])

        // Generate trend data (mocked)
        const mockTrends: TrendData[] = [
          { month: 'Jan', projects: 3200, capacity: 45000, investment: 120 },
          { month: 'Feb', projects: 3850, capacity: 52000, investment: 145 },
          { month: 'Mar', projects: 4200, capacity: 58000, investment: 165 },
          { month: 'Apr', projects: 4900, capacity: 65000, investment: 190 },
          { month: 'May', projects: 5500, capacity: 72000, investment: 220 },
          { month: 'Jun', projects: 6200, capacity: 81000, investment: 250 },
        ]
        setTrends(mockTrends)
      } catch (error) {
        if (controller.signal.aborted) return
        console.error('Error fetching dashboard data:', error)
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    fetchDashboardData()

    return () => controller.abort()
  }, [timeRange, selectedType, selectedStatus, selectedCountry, capacityRange])

  const formatNumber = (num: number) => {
    if (num >= 1000000000) return `${(num / 1000000000).toFixed(1)}B`
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const getTypeIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'solar': return <Sun className="h-4 w-4" />
      case 'wind': return <Wind className="h-4 w-4" />
      case 'battery': return <Battery className="h-4 w-4" />
      case 'hydro': return <Droplets className="h-4 w-4" />
      default: return <Zap className="h-4 w-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'solar': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'wind': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
      case 'battery': return 'text-purple-400 bg-purple-500/20 border-purple-500/30'
      case 'hydro': return 'text-cyan-400 bg-cyan-500/20 border-cyan-500/30'
      default: return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400"></div>
          <p className="mt-4 text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas
              </Link>
              <span className="ml-2 text-xs text-gray-400">/ Dashboard</span>
            </div>
            <div className="flex space-x-8">
              <Link href="/" className="text-white/70 hover:text-white transition">Home</Link>
              <Link href="/explore" className="text-white/70 hover:text-white transition">Explore</Link>
              <Link href="/dashboard" className="text-white transition">Dashboard</Link>
              <Link href="/invest" className="text-white/70 hover:text-white transition">Invest</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            Platform Dashboard
          </h1>
          <p className="text-gray-400 mt-2">Real-time insights into global energy transformation</p>
        </div>

        {/* Time Range Selector & Filters */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex gap-2">
          {['24h', '7d', '30d', '90d', '1y', 'All'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg transition ${
                timeRange === range 
                  ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
                  : 'bg-gray-900/50 border border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              {range}
            </button>
          ))}
          </div>
          
          {/* Filter Toggle Button */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`px-4 py-2 rounded-lg transition flex items-center gap-2 ${
              showFilters 
                ? 'bg-purple-500/20 border border-purple-500/50 text-purple-400'
                : 'bg-gray-900/50 border border-gray-700 text-gray-400 hover:border-gray-600'
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            Advanced Filters
            {(selectedType !== 'all' || selectedStatus !== 'all' || selectedCountry !== 'all') && (
              <span className="bg-purple-500/30 px-2 py-0.5 rounded-full text-xs">
                Active
              </span>
            )}
          </button>
        </div>

        {/* Advanced Filters Panel */}
        {showFilters && (
          <div className="mb-8 p-6 bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {/* Project Type Filter */}
              <div>
                <label className="text-sm font-semibold text-gray-400 mb-2 block">Project Type</label>
                <select 
                  value={selectedType} 
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-purple-500 focus:outline-none"
                >
                  <option value="all">All Types</option>
                  <option value="Solar">Solar</option>
                  <option value="Wind">Wind</option>
                  <option value="Battery">Battery Storage</option>
                  <option value="Hydro">Hydro</option>
                  <option value="Nuclear">Nuclear</option>
                  <option value="Geothermal">Geothermal</option>
                </select>
              </div>
              
              {/* Status Filter */}
              <div>
                <label className="text-sm font-semibold text-gray-400 mb-2 block">Project Status</label>
                <select 
                  value={selectedStatus} 
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-purple-500 focus:outline-none"
                >
                  <option value="all">All Statuses</option>
                  <option value="Operational">Operational</option>
                  <option value="Construction">Under Construction</option>
                  <option value="Development">In Development</option>
                  <option value="Planned">Planned</option>
                </select>
              </div>
              
              {/* Country Filter */}
              <div>
                <label className="text-sm font-semibold text-gray-400 mb-2 block">Country</label>
                <select 
                  value={selectedCountry} 
                  onChange={(e) => setSelectedCountry(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-purple-500 focus:outline-none"
                >
                  <option value="all">All Countries</option>
                  <option value="USA">United States</option>
                  <option value="China">China</option>
                  <option value="Germany">Germany</option>
                  <option value="India">India</option>
                  <option value="Brazil">Brazil</option>
                  <option value="Australia">Australia</option>
                  <option value="Canada">Canada</option>
                  <option value="UK">United Kingdom</option>
                  <option value="Japan">Japan</option>
                </select>
              </div>
              
              {/* Capacity Range */}
              <div>
                <label className="text-sm font-semibold text-gray-400 mb-2 block">Capacity (MW)</label>
                <div className="flex gap-2">
                  <input 
                    type="number" 
                    value={capacityRange.min}
                    onChange={(e) => setCapacityRange(prev => ({ ...prev, min: Number(e.target.value) }))}
                    placeholder="Min"
                    className="w-1/2 px-2 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white text-sm focus:border-purple-500 focus:outline-none"
                  />
                  <input 
                    type="number" 
                    value={capacityRange.max}
                    onChange={(e) => setCapacityRange(prev => ({ ...prev, max: Number(e.target.value) }))}
                    placeholder="Max"
                    className="w-1/2 px-2 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
            
            {/* Filter Actions */}
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => {
                  setSelectedType('all')
                  setSelectedStatus('all')
                  setSelectedCountry('all')
                  setCapacityRange({ min: 0, max: 10000 })
                }}
                className="px-4 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-gray-400 hover:border-gray-600 transition"
              >
                Clear Filters
              </button>
              <button
                onClick={() => fetchDashboardData()}
                className="px-4 py-2 text-sm bg-purple-500/20 border border-purple-500/50 rounded-lg text-purple-400 hover:bg-purple-500/30 transition"
              >
                Apply Filters
              </button>
            </div>
          </div>
        )}

        {/* Primary Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gradient-to-br from-emerald-950/20 to-transparent border border-emerald-500/20 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Globe className="h-8 w-8 text-emerald-400" />
              <span className="text-xs text-emerald-400 bg-emerald-500/20 px-2 py-1 rounded-full">
                +12.3%
              </span>
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {formatNumber(stats?.total_projects || 0)}
            </div>
            <div className="text-sm text-gray-400">Total Projects</div>
            <div className="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 w-4/5"></div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-cyan-950/20 to-transparent border border-cyan-500/20 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Zap className="h-8 w-8 text-cyan-400" />
              <span className="text-xs text-cyan-400 bg-cyan-500/20 px-2 py-1 rounded-full">
                +18.7%
              </span>
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {((stats?.total_capacity_mw || 0) / 1000).toFixed(1)} GW
            </div>
            <div className="text-sm text-gray-400">Total Capacity</div>
            <div className="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-400 to-cyan-600 w-3/4"></div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-950/20 to-transparent border border-purple-500/20 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <DollarSign className="h-8 w-8 text-purple-400" />
              <span className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded-full">
                +25.4%
              </span>
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              ${formatNumber(stats?.total_investment || 0)}
            </div>
            <div className="text-sm text-gray-400">Investment Opportunity</div>
            <div className="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-purple-400 to-purple-600 w-5/6"></div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-yellow-950/20 to-transparent border border-yellow-500/20 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <MapPin className="h-8 w-8 text-yellow-400" />
              <span className="text-xs text-yellow-400 bg-yellow-500/20 px-2 py-1 rounded-full">
                +8 new
              </span>
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {stats?.countries || 0}
            </div>
            <div className="text-sm text-gray-400">Countries</div>
            <div className="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-yellow-400 to-yellow-600 w-2/3"></div>
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Project Type Distribution */}
          <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Project Distribution</h3>
              <PieChart className="h-5 w-5 text-gray-400" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sun className="h-4 w-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">Solar</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.solar_projects || 0)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wind className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-gray-300">Wind</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.wind_projects || 0)}</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Battery className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Storage</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.battery_projects || 0)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Droplets className="h-4 w-4 text-cyan-400" />
                    <span className="text-sm text-gray-300">Hydro</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.hydro_projects || 0)}</span>
                </div>
              </div>
            </div>
            
            {/* Visual Chart */}
            <div className="mt-6 h-32 flex items-end gap-2">
              {[
                { type: 'Solar', value: stats?.solar_projects || 0, color: 'bg-yellow-500' },
                { type: 'Wind', value: stats?.wind_projects || 0, color: 'bg-blue-500' },
                { type: 'Storage', value: stats?.battery_projects || 0, color: 'bg-purple-500' },
                { type: 'Hydro', value: stats?.hydro_projects || 0, color: 'bg-cyan-500' },
              ].map((item, idx) => (
                <div key={idx} className="flex-1 flex flex-col items-center">
                  <div 
                    className={`w-full ${item.color} rounded-t-lg transition-all hover:opacity-80`}
                    style={{ height: `${(item.value / 20000) * 100}%` }}
                  ></div>
                  <span className="text-xs text-gray-500 mt-2">{item.type}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Status Overview */}
          <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Project Status</h3>
              <Activity className="h-5 w-5 text-gray-400" />
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-sm text-gray-300">Operational</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.operational || 0)}</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-green-500 w-3/5"></div>
                </div>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">Under Construction</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.construction || 0)}</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-yellow-500 w-1/4"></div>
                </div>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-gray-300">In Development</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.development || 0)}</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 w-1/6"></div>
                </div>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Planned</span>
                  </div>
                  <span className="text-sm font-semibold text-white">{formatNumber(stats?.planned || 0)}</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500 w-1/3"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Real-time Analytics Section */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
            <Activity className="h-7 w-7 text-emerald-400" />
            Real-time Analytics
          </h2>
          <RealtimeCharts 
            data={{ 
              projects: topProjects, 
              stats: stats 
            }} 
            filters={{ 
              type: selectedType, 
              status: selectedStatus, 
              country: selectedCountry, 
              capacityRange 
            }}
            liveMode={true}
            refreshInterval={5000}
          />
        </div>

        {/* Growth Trends */}
        <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Growth Trends</h3>
            <BarChart3 className="h-5 w-5 text-gray-400" />
          </div>
          <div className="h-64 flex items-end gap-4">
            {trends.map((data, idx) => (
              <div key={idx} className="flex-1 flex flex-col items-center">
                <div className="relative w-full flex flex-col items-center gap-1">
                  <div 
                    className="w-12 bg-gradient-to-t from-emerald-500 to-emerald-400 rounded-t-lg hover:opacity-80 transition cursor-pointer"
                    style={{ height: `${(data.projects / 6200) * 200}px` }}
                  >
                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs text-gray-400 whitespace-nowrap">
                      {formatNumber(data.projects)}
                    </div>
                  </div>
                  <span className="text-xs text-gray-500">{data.month}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Projects Table */}
        <div className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Top Projects by Capacity</h3>
            <Award className="h-5 w-5 text-gray-400" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left text-xs font-semibold text-gray-400 pb-3">Project</th>
                  <th className="text-left text-xs font-semibold text-gray-400 pb-3">Type</th>
                  <th className="text-left text-xs font-semibold text-gray-400 pb-3">Location</th>
                  <th className="text-right text-xs font-semibold text-gray-400 pb-3">Capacity</th>
                  <th className="text-right text-xs font-semibold text-gray-400 pb-3">Investment</th>
                  <th className="text-right text-xs font-semibold text-gray-400 pb-3">Action</th>
                </tr>
              </thead>
              <tbody>
                {topProjects.map((project, idx) => (
                  <tr key={project.id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">#{idx + 1}</span>
                        <span className="text-sm font-medium text-white">{project.name}</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg border ${getTypeColor(project.type)}`}>
                        {getTypeIcon(project.type)}
                        <span className="text-xs">{project.type}</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1 text-sm text-gray-400">
                        <MapPin className="h-3 w-3" />
                        {project.state}, {project.country}
                      </div>
                    </td>
                    <td className="py-3 text-right">
                      <span className="text-sm font-semibold text-emerald-400">
                        {formatNumber(project.capacity_mw)} MW
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <span className="text-sm font-semibold text-cyan-400">
                        ${formatNumber(project.investment)}
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <Link
                        href={`/project/${project.id}`}
                        className="text-xs px-3 py-1 bg-purple-500/20 border border-purple-500/50 rounded-lg text-purple-400 hover:bg-purple-500/30 transition"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link href="/explore" className="p-4 bg-gradient-to-r from-emerald-950/20 to-transparent border border-emerald-500/20 rounded-xl hover:border-emerald-500/40 transition group">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white mb-1">Explore Projects</h4>
                <p className="text-sm text-gray-400">Interactive 3D globe view</p>
              </div>
              <Globe className="h-8 w-8 text-emerald-400 group-hover:rotate-12 transition-transform" />
            </div>
          </Link>
          
          <Link href="/invest" className="p-4 bg-gradient-to-r from-purple-950/20 to-transparent border border-purple-500/20 rounded-xl hover:border-purple-500/40 transition group">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-white mb-1">Start Investing</h4>
                <p className="text-sm text-gray-400">Join the energy revolution</p>
              </div>
              <DollarSign className="h-8 w-8 text-purple-400 group-hover:scale-110 transition-transform" />
            </div>
          </Link>
          
          {/* Consciousness-Weighted Harmony Leaderboard */}
          <HarmonyLeaderboard limit={5} className="col-span-full" />

          <div className="p-4 bg-gradient-to-r from-cyan-950/20 to-transparent border border-cyan-500/20 rounded-xl hover:border-cyan-500/40 transition group">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="font-semibold text-white mb-1">Export Data</h4>
                <p className="text-sm text-gray-400">Download reports</p>
              </div>
              <BarChart3 className="h-8 w-8 text-cyan-400 group-hover:translate-y-[-2px] transition-transform" />
            </div>
            <div className="space-y-2">
              <button 
                onClick={() => {
                  const params = new URLSearchParams()
                  params.append('dataset', 'projects')
                  params.append('format', 'csv')
                  if (selectedType !== 'all') params.append('type', selectedType)
                  if (selectedStatus !== 'all') params.append('status', selectedStatus)
                  if (capacityRange.min > 0) params.append('minCapacity', capacityRange.min.toString())
                  if (capacityRange.max < 10000) params.append('maxCapacity', capacityRange.max.toString())
                  window.open(`/api/export?${params}`, '_blank')
                }}
                className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 hover:bg-cyan-500/20 transition text-sm"
              >
                {selectedType !== 'all' || selectedStatus !== 'all' ? 'Filtered' : 'All'} Projects (CSV)
              </button>
              <button 
                onClick={() => window.open('/api/export?dataset=statistics&format=csv', '_blank')}
                className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 hover:bg-cyan-500/20 transition text-sm"
              >
                Statistics (CSV)
              </button>
              <button 
                onClick={() => window.open('/api/export?dataset=summary&format=csv', '_blank')}
                className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 hover:bg-cyan-500/20 transition text-sm"
              >
                State Summary (CSV)
              </button>
              <button 
                onClick={() => {
                  const params = new URLSearchParams()
                  params.append('dataset', 'projects')
                  params.append('format', 'json')
                  if (selectedType !== 'all') params.append('type', selectedType)
                  if (selectedStatus !== 'all') params.append('status', selectedStatus)
                  window.open(`/api/export?${params}`, '_blank')
                }}
                className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 hover:bg-cyan-500/20 transition text-sm"
              >
                Export as JSON
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
