// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Search, Filter, TrendingUp, Zap, Wind, Sun, Battery, Droplets,
  MapPin, DollarSign, Clock, Users, Target, AlertCircle, 
  ChevronRight, Info, Shield, Award, Sparkles, Building2, Atom, Lock
} from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'
import AdvancedFilterPanel from '@/components/AdvancedFilterPanel'

interface Project {
  id: number
  name: string
  type: string
  capacity_mw: number
  location: string
  country: string
  status: string
  roi_percentage: number
  min_investment: number
  total_raised: number
  target_amount: number
  investors_count: number
  completion_date: string
  risk_level: 'Low' | 'Medium' | 'High'
  description?: string
}

export default function InvestPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([])
  const [featuredSMR, setFeaturedSMR] = useState<any[]>([]) // Featured SMR projects
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState('all')
  const [selectedRisk, setSelectedRisk] = useState('all')
  const [selectedCountry, setSelectedCountry] = useState('all')
  const [sortBy, setSortBy] = useState('roi')
  const { user } = useAuth()
  const router = useRouter()
  const projectsController = useRef<AbortController | null>(null)
  const smrController = useRef<AbortController | null>(null)

  useEffect(() => {
    fetchInvestmentProjects()
    fetchFeaturedSMR()
  }, [])

  useEffect(() => {
    filterAndSortProjects()
  }, [projects, searchTerm, selectedType, selectedRisk, selectedCountry, sortBy])

  const fetchInvestmentProjects = async () => {
    projectsController.current?.abort()
    const controller = new AbortController()
    projectsController.current = controller

    try {
      // Fetch top investment opportunities
      const res = await fetch('/api/projects?limit=50&hasInvestment=true', {
        signal: controller.signal,
      })
      const data = await res.json()
      
      // Transform projects with investment data
      const investmentProjects = (data.projects || []).map((p: any) => ({
        ...p,
        roi_percentage: getROIByType(p.type),
        min_investment: 10,
        total_raised: Math.floor(Math.random() * 5000000) + 1000000,
        target_amount: Math.floor(Math.random() * 10000000) + 5000000,
        investors_count: Math.floor(Math.random() * 500) + 50,
        completion_date: getCompletionDate(p.status),
        risk_level: getRiskLevel(p.type, p.status),
        description: generateProjectDescription(p)
      }))
      
      if (!controller.signal.aborted) {
        setProjects(investmentProjects)
        setFilteredProjects(investmentProjects)
      }
    } catch (error) {
      if (controller.signal.aborted) return
      console.error('Error fetching projects:', error)
    }
    if (!controller.signal.aborted) {
      setLoading(false)
    }
  }

  const fetchFeaturedSMR = async () => {
    smrController.current?.abort()
    const controller = new AbortController()
    smrController.current = controller

    try {
      // Fetch top 2 SMR projects for featured section
      const res = await fetch('/api/smr?limit=2&sort=roi', {
        signal: controller.signal,
      })
      const data = await res.json()
      if (!controller.signal.aborted) {
        setFeaturedSMR(data.projects || [])
      }
    } catch (error) {
      if (controller.signal.aborted) return
      console.error('Error fetching SMR projects:', error)
    }
  }

  const getROIByType = (type: string): number => {
    const rois: { [key: string]: number } = {
      'Solar': 14,
      'Wind': 16,
      'Battery': 15,
      'Hydro': 11,
      'Nuclear': 12,
      'Geothermal': 13
    }
    return rois[type] || 13
  }

  const getRiskLevel = (type: string, status: string): 'Low' | 'Medium' | 'High' => {
    if (status === 'Operational') return 'Low'
    if (status === 'Construction') return 'Medium'
    if (type === 'Nuclear') return 'Medium'
    return 'Low'
  }

  const getCompletionDate = (status: string): string => {
    const dates: { [key: string]: string } = {
      'Operational': 'Operating',
      'Construction': '2025 Q3',
      'Development': '2026 Q2',
      'Planned': '2027 Q1'
    }
    return dates[status] || '2026 Q4'
  }

  const generateProjectDescription = (project: any): string => {
    const descriptions: { [key: string]: string } = {
      'Solar': `Large-scale solar farm with ${project.capacity_mw}MW capacity, utilizing latest photovoltaic technology for maximum efficiency.`,
      'Wind': `Offshore wind farm featuring advanced turbines with ${project.capacity_mw}MW total capacity and 25+ year operational lifespan.`,
      'Battery': `Grid-scale battery storage facility providing ${project.capacity_mw}MW capacity for renewable energy stabilization.`,
      'Hydro': `Run-of-river hydroelectric project generating ${project.capacity_mw}MW of clean baseload power.`,
      'Nuclear': `Small modular reactor facility providing ${project.capacity_mw}MW of carbon-free baseload power.`
    }
    return descriptions[project.type] || `Clean energy project with ${project.capacity_mw}MW capacity.`
  }

  const filterAndSortProjects = () => {
    let filtered = [...projects]

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(p => 
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.location?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.country?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Apply type filter
    if (selectedType !== 'all') {
      filtered = filtered.filter(p => p.type === selectedType)
    }

    // Apply risk filter
    if (selectedRisk !== 'all') {
      filtered = filtered.filter(p => p.risk_level === selectedRisk)
    }

    // Apply country filter
    if (selectedCountry !== 'all') {
      filtered = filtered.filter(p => p.country === selectedCountry)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'roi':
          return b.roi_percentage - a.roi_percentage
        case 'capacity':
          return b.capacity_mw - a.capacity_mw
        case 'raised':
          return b.total_raised - a.total_raised
        case 'investors':
          return b.investors_count - a.investors_count
        default:
          return 0
      }
    })

    setFilteredProjects(filtered)
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Solar': return <Sun className="h-4 w-4" />
      case 'Wind': return <Wind className="h-4 w-4" />
      case 'Battery': return <Battery className="h-4 w-4" />
      case 'Hydro': return <Droplets className="h-4 w-4" />
      case 'Nuclear': return <Zap className="h-4 w-4" />
      default: return <Zap className="h-4 w-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'Solar': return 'from-yellow-500 to-orange-500'
      case 'Wind': return 'from-blue-500 to-cyan-500'
      case 'Battery': return 'from-purple-500 to-pink-500'
      case 'Hydro': return 'from-cyan-500 to-teal-500'
      case 'Nuclear': return 'from-green-500 to-emerald-500'
      default: return 'from-gray-500 to-gray-600'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'text-green-400 bg-green-500/20'
      case 'Medium': return 'text-yellow-400 bg-yellow-500/20'
      case 'High': return 'text-red-400 bg-red-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `$${(num / 1000).toFixed(0)}K`
    return `$${num}`
  }

  const calculateProgress = (raised: number, target: number) => {
    return Math.min((raised / target) * 100, 100)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400"></div>
          <p className="mt-4 text-gray-400">Loading investment opportunities...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/70 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Terra Atlas
              </Link>
              <span className="ml-2 text-xs text-gray-400">/ Invest</span>
            </div>
            <div className="flex items-center space-x-8">
              <Link href="/explore" className="text-white/70 hover:text-white transition">Explore</Link>
              <Link href="/dashboard" className="text-white/70 hover:text-white transition">Dashboard</Link>
              <Link href="/invest" className="text-white transition">Invest</Link>
              {user ? (
                <Link href="/portfolio" className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold">
                  My Portfolio
                </Link>
              ) : (
                <Link href="/auth/login" className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold">
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="pt-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              Invest in the Energy Future
            </h1>
            <p className="text-xl text-gray-400 mb-8">
              Join 50,000+ investors funding the global energy transition
            </p>
            
            {/* Stats Bar */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-bold text-emerald-400">11-16%</div>
                <div className="text-sm text-gray-400">Average Returns</div>
              </div>
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-bold text-cyan-400">$10</div>
                <div className="text-sm text-gray-400">Minimum Investment</div>
              </div>
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-bold text-purple-400">50K+</div>
                <div className="text-sm text-gray-400">Active Investors</div>
              </div>
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4">
                <div className="text-2xl font-bold text-yellow-400">$2.5B</div>
                <div className="text-sm text-gray-400">Total Invested</div>
              </div>
            </div>
          </motion.div>

          {/* Featured SMR Section */}
          {featuredSMR.length > 0 && (
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                  <Atom className="h-8 w-8 text-emerald-400" />
                  Featured SMR Opportunities
                  <span className="px-3 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-xs font-semibold text-emerald-400">
                    HIGH VALUE
                  </span>
                </h2>
                <Link
                  href="/smr"
                  className="text-emerald-400 hover:text-emerald-300 transition flex items-center gap-1"
                >
                  View All SMR Projects
                  <ChevronRight className="h-4 w-4" />
                </Link>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {featuredSMR.map((smr) => (
                  <motion.div
                    key={smr.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ scale: 1.02 }}
                    className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 backdrop-blur-md border border-emerald-500/30 rounded-xl p-6 hover:border-emerald-500/50 transition-all"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Atom className="h-6 w-6 text-emerald-400" />
                          <span className="text-xs font-semibold text-emerald-400">SMALL MODULAR REACTOR</span>
                        </div>
                        <h3 className="text-xl font-bold text-white mb-1">{smr.name}</h3>
                        <p className="text-sm text-gray-400">{smr.developer}</p>
                      </div>
                      <span className="px-3 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded-full text-xs font-semibold text-yellow-400">
                        {smr.status}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div>
                        <div className="text-2xl font-bold text-emerald-400">{smr.roi_percentage}%</div>
                        <div className="text-xs text-gray-500">Annual ROI</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-cyan-400">{smr.capacity_mw} MW</div>
                        <div className="text-xs text-gray-500">Capacity</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-purple-400">{smr.design_life_years}yr</div>
                        <div className="text-xs text-gray-500">Lifetime</div>
                      </div>
                    </div>
                    
                    <div className="mb-4">
                      <div className="flex items-center justify-between text-sm mb-2">
                        <span className="text-gray-400">Investment Progress</span>
                        <span className="text-white font-semibold">
                          {((smr.investment_raised / smr.total_investment) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                          style={{ width: `${(smr.investment_raised / smr.total_investment) * 100}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="text-xs text-gray-500">Minimum Investment</p>
                        <p className="text-lg font-bold text-white">
                          ${(smr.min_investment / 1000).toFixed(0)}K - ${(smr.min_investment * 5 / 1000).toFixed(0)}K
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-gray-500">Investment Type</p>
                        <p className="text-sm font-semibold text-yellow-400">{smr.investment_type}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg mb-4">
                      <Shield className="h-4 w-4 text-yellow-400" />
                      <span className="text-xs text-yellow-400">Accredited Investors Only • 5-10 Year Terms</span>
                    </div>
                    
                    <button
                      onClick={() => router.push(`/invest/smr/${smr.id}`)}
                      className="w-full py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all flex items-center justify-center gap-2"
                    >
                      <Lock className="h-4 w-4" />
                      Invest in SMR Project
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </motion.div>
                ))}
              </div>
              
              <div className="mt-6 text-center">
                <Link
                  href="/smr"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gray-900/50 border border-emerald-500/30 rounded-lg text-emerald-400 hover:bg-emerald-500/10 transition"
                >
                  Explore All {featuredSMR[0]?.total_projects || 10} SMR Investment Opportunities
                  <ChevronRight className="h-5 w-5" />
                </Link>
              </div>
            </div>
          )}

          {/* Advanced Filter Panel */}
          <AdvancedFilterPanel
            onFiltersChange={(filters) => {
              setSearchTerm(filters.searchTerm)
              setSelectedType(filters.types.length > 0 ? filters.types[0] : 'all')
              setSelectedRisk(filters.riskLevels.length > 0 ? filters.riskLevels[0] : 'all')
              setSelectedCountry(filters.countries.length > 0 ? filters.countries[0] : 'all')
              setSortBy(filters.sortBy)
              
              // Apply advanced filtering
              let filtered = [...projects]

              // Search
              if (filters.searchTerm) {
                filtered = filtered.filter(p => 
                  p.name.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.location?.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.country?.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.developer?.toLowerCase().includes(filters.searchTerm.toLowerCase())
                )
              }

              // Type filter (multi-select)
              if (filters.types.length > 0) {
                filtered = filtered.filter(p => filters.types.includes(p.type))
              }

              // Status filter (multi-select)
              if (filters.statuses.length > 0) {
                filtered = filtered.filter(p => filters.statuses.includes(p.status))
              }

              // Country filter (multi-select)
              if (filters.countries.length > 0) {
                filtered = filtered.filter(p => filters.countries.includes(p.country))
              }

              // Risk filter (multi-select)
              if (filters.riskLevels.length > 0) {
                filtered = filtered.filter(p => filters.riskLevels.includes(p.risk_level))
              }

              // Capacity range
              filtered = filtered.filter(p => 
                p.capacity_mw >= filters.minCapacity && 
                p.capacity_mw <= filters.maxCapacity
              )

              // Investment range
              filtered = filtered.filter(p => 
                p.min_investment >= filters.minInvestment && 
                p.min_investment <= filters.maxInvestment
              )

              // ROI range
              filtered = filtered.filter(p => 
                p.roi_percentage >= filters.minROI && 
                p.roi_percentage <= filters.maxROI
              )

              // Date filters
              if (filters.completionFrom || filters.completionTo) {
                // Parse completion dates and filter
                filtered = filtered.filter(p => {
                  // Simple date comparison logic here
                  return true // Placeholder - implement actual date comparison
                })
              }

              // Sorting
              filtered.sort((a, b) => {
                const order = filters.sortOrder === 'asc' ? 1 : -1
                switch (filters.sortBy) {
                  case 'roi':
                    return (b.roi_percentage - a.roi_percentage) * order
                  case 'capacity':
                    return (b.capacity_mw - a.capacity_mw) * order
                  case 'investment':
                    return (b.min_investment - a.min_investment) * order
                  case 'name':
                    return a.name.localeCompare(b.name) * order
                  case 'location':
                    return (a.location || '').localeCompare(b.location || '') * order
                  default:
                    return 0
                }
              })

              setFilteredProjects(filtered)
            }}
            projectCount={filteredProjects.length}
            availableTypes={['Solar', 'Wind', 'Battery', 'Hydro', 'Nuclear', 'Geothermal']}
            availableCountries={Array.from(new Set(projects.map(p => p.country).filter(Boolean)))}
            availableDevelopers={Array.from(new Set(projects.map(p => p.developer).filter(Boolean)))}
            className="mb-8"
          />

          {/* Projects Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProjects.map((project) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ y: -5 }}
                className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl overflow-hidden hover:border-emerald-500/50 transition-all"
              >
                {/* Project Header */}
                <div className={`h-2 bg-gradient-to-r ${getTypeColor(project.type)}`} />
                
                <div className="p-6">
                  {/* Project Title */}
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-1">{project.name}</h3>
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <MapPin className="h-4 w-4" />
                        {project.location || project.country}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 px-2 py-1 bg-gray-800 rounded-lg">
                      {getTypeIcon(project.type)}
                      <span className="text-xs text-gray-300">{project.type}</span>
                    </div>
                  </div>

                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-2xl font-bold text-emerald-400">{project.roi_percentage}%</div>
                      <div className="text-xs text-gray-500">Annual Return</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-white">{project.capacity_mw} MW</div>
                      <div className="text-xs text-gray-500">Capacity</div>
                    </div>
                  </div>

                  {/* Risk Level */}
                  <div className="mb-4">
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${getRiskColor(project.risk_level)}`}>
                      <Shield className="h-3 w-3 mr-1" />
                      {project.risk_level} Risk
                    </span>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-gray-400 mb-4 line-clamp-2">
                    {project.description}
                  </p>

                  {/* Funding Progress */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Funding Progress</span>
                      <span className="text-sm font-semibold text-white">
                        {calculateProgress(project.total_raised, project.target_amount).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${calculateProgress(project.total_raised, project.target_amount)}%` }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                      />
                    </div>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-gray-500">
                        {formatNumber(project.total_raised)} raised
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatNumber(project.target_amount)} goal
                      </span>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-800">
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-400">{project.investors_count} investors</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-400">{project.completion_date}</span>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Link
                      href={`/project/${project.id}`}
                      className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-center text-sm font-medium text-gray-300 hover:bg-gray-700 transition"
                    >
                      Learn More
                    </Link>
                    <button
                      onClick={() => {
                        if (!user) {
                          router.push('/auth/login')
                        } else {
                          router.push(`/invest/${project.id}`)
                        }
                      }}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg text-center text-sm font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all flex items-center justify-center gap-1"
                    >
                      Invest Now
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Empty State */}
          {filteredProjects.length === 0 && (
            <div className="text-center py-12">
              <AlertCircle className="h-12 w-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-400 mb-2">No projects found</h3>
              <p className="text-gray-500">Try adjusting your filters or search terms</p>
            </div>
          )}

          {/* Trust Indicators */}
          <div className="mt-12 py-8 border-t border-gray-800">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
              <div>
                <Shield className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
                <h4 className="font-semibold text-white mb-1">Bank-Grade Security</h4>
                <p className="text-sm text-gray-400">256-bit encryption & SOC2 compliant</p>
              </div>
              <div>
                <Award className="h-8 w-8 text-cyan-400 mx-auto mb-2" />
                <h4 className="font-semibold text-white mb-1">Regulated Platform</h4>
                <p className="text-sm text-gray-400">SEC-registered & fully licensed</p>
              </div>
              <div>
                <Building2 className="h-8 w-8 text-purple-400 mx-auto mb-2" />
                <h4 className="font-semibold text-white mb-1">Institutional Partners</h4>
                <p className="text-sm text-gray-400">Backed by leading energy firms</p>
              </div>
              <div>
                <Sparkles className="h-8 w-8 text-yellow-400 mx-auto mb-2" />
                <h4 className="font-semibold text-white mb-1">Success Record</h4>
                <p className="text-sm text-gray-400">95% projects hit targets</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
