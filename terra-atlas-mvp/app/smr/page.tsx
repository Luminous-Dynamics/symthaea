'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Atom, Building, MapPin, DollarSign, TrendingUp, Clock, Shield, 
  Award, Users, Target, ChevronRight, AlertCircle, CheckCircle,
  Zap, Activity, Calendar, Globe, Filter, Search, Info, Lock,
  Sparkles, Battery, BarChart3, Building2, Briefcase, ArrowUp
} from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'
import AdvancedFilterPanel from '@/components/AdvancedFilterPanel'

interface SMRProject {
  id: number
  name: string
  developer: string
  type: string
  subtype: string
  capacity_mw: number
  modules: number
  module_capacity_mw: number
  location: string
  state: string
  country: string
  lat: number
  lon: number
  status: string
  construction_start: string
  commercial_operation: string
  total_investment: number
  investment_raised: number
  investment_needed: number
  levelized_cost_mwh: number
  power_purchase_agreement: string
  capacity_factor: number
  design_life_years: number
  regulatory_status: string
  description: string
  key_features: string[]
  investors: string[]
  risk_factors: string[]
  roi_percentage: number
  min_investment: number
  investment_type: string
  website: string
}

interface SMRStats {
  total_projects: number
  total_capacity_mw: number
  total_investment_needed: number
  average_roi: number
  countries: number
  developers: number
  status_breakdown: {
    construction: number
    development: number
    planning: number
  }
}

export default function SMRPage() {
  const [projects, setProjects] = useState<SMRProject[]>([])
  const [stats, setStats] = useState<SMRStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedProject, setSelectedProject] = useState<SMRProject | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCountry, setSelectedCountry] = useState('all')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [sortBy, setSortBy] = useState('roi')
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    fetchSMRProjects()
  }, [sortBy])

  const fetchSMRProjects = async () => {
    try {
      const params = new URLSearchParams()
      params.append('sort', sortBy)
      
      const res = await fetch(`/api/smr?${params}`)
      const data = await res.json()
      
      setProjects(data.projects || [])
      setStats(data.stats || null)
    } catch (error) {
      console.error('Error fetching SMR projects:', error)
    }
    setLoading(false)
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000000) return `$${(num / 1000000000).toFixed(2)}B`
    if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `$${(num / 1000).toFixed(0)}K`
    return `$${num.toFixed(0)}`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Construction': return 'text-green-400 bg-green-500/20'
      case 'Development': return 'text-blue-400 bg-blue-500/20'
      case 'Planning': return 'text-yellow-400 bg-yellow-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const getRiskColor = (factors: string[]) => {
    if (factors.some(f => f.toLowerCase().includes('first'))) {
      return 'text-yellow-400'
    }
    return 'text-green-400'
  }

  const filteredProjects = projects.filter(project => {
    const matchesSearch = searchTerm === '' || 
      project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      project.developer.toLowerCase().includes(searchTerm.toLowerCase()) ||
      project.location.toLowerCase().includes(searchTerm.toLowerCase())
    
    const matchesCountry = selectedCountry === 'all' || project.country === selectedCountry
    const matchesStatus = selectedStatus === 'all' || project.status === selectedStatus
    
    return matchesSearch && matchesCountry && matchesStatus
  })

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-950 text-white flex items-center justify-center">
        <div className="text-center">
          <Atom className="h-12 w-12 text-emerald-400 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-400">Loading SMR investment opportunities...</p>
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
              <span className="ml-2 text-xs text-gray-400">/ SMR Investments</span>
            </div>
            <div className="flex items-center space-x-8">
              <Link href="/explore" className="text-white/70 hover:text-white transition">Explore</Link>
              <Link href="/invest" className="text-white/70 hover:text-white transition">All Projects</Link>
              <Link href="/smr" className="text-white transition">SMR Projects</Link>
              {user ? (
                <Link href="/portfolio" className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold">
                  My Portfolio
                </Link>
              ) : (
                <Link href="/auth/login" className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold">
                  Login to Invest
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="pt-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <Atom className="h-12 w-12 text-emerald-400" />
              <h1 className="text-5xl font-bold bg-gradient-to-r from-emerald-400 via-green-400 to-cyan-400 bg-clip-text text-transparent">
                Small Modular Reactor Investments
              </h1>
            </div>
            <p className="text-xl text-gray-400 mb-4">
              Invest in the future of carbon-free baseload power
            </p>
            <p className="text-lg text-gray-500 max-w-3xl mx-auto">
              Small Modular Reactors represent the cutting edge of nuclear technology - 
              safer, more flexible, and economically competitive with fossil fuels.
            </p>
          </motion.div>

          {/* Stats Cards */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
                className="bg-gradient-to-br from-emerald-950/30 to-transparent border border-emerald-500/30 rounded-xl p-4"
              >
                <Atom className="h-6 w-6 text-emerald-400 mb-2" />
                <div className="text-2xl font-bold text-white">{stats.total_projects}</div>
                <div className="text-xs text-gray-400">Active Projects</div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="bg-gradient-to-br from-cyan-950/30 to-transparent border border-cyan-500/30 rounded-xl p-4"
              >
                <Zap className="h-6 w-6 text-cyan-400 mb-2" />
                <div className="text-2xl font-bold text-white">{(stats.total_capacity_mw / 1000).toFixed(1)} GW</div>
                <div className="text-xs text-gray-400">Total Capacity</div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="bg-gradient-to-br from-purple-950/30 to-transparent border border-purple-500/30 rounded-xl p-4"
              >
                <DollarSign className="h-6 w-6 text-purple-400 mb-2" />
                <div className="text-2xl font-bold text-white">{formatNumber(stats.total_investment_needed)}</div>
                <div className="text-xs text-gray-400">Capital Needed</div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 }}
                className="bg-gradient-to-br from-yellow-950/30 to-transparent border border-yellow-500/30 rounded-xl p-4"
              >
                <TrendingUp className="h-6 w-6 text-yellow-400 mb-2" />
                <div className="text-2xl font-bold text-white">{stats.average_roi.toFixed(1)}%</div>
                <div className="text-xs text-gray-400">Avg. ROI</div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
                className="bg-gradient-to-br from-green-950/30 to-transparent border border-green-500/30 rounded-xl p-4"
              >
                <Globe className="h-6 w-6 text-green-400 mb-2" />
                <div className="text-2xl font-bold text-white">{stats.countries}</div>
                <div className="text-xs text-gray-400">Countries</div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 }}
                className="bg-gradient-to-br from-red-950/30 to-transparent border border-red-500/30 rounded-xl p-4"
              >
                <Building2 className="h-6 w-6 text-red-400 mb-2" />
                <div className="text-2xl font-bold text-white">{stats.developers}</div>
                <div className="text-xs text-gray-400">Developers</div>
              </motion.div>
            </div>
          )}

          {/* Advanced Filter Panel */}
          <AdvancedFilterPanel
            onFiltersChange={(filters) => {
              // Apply advanced filtering
              let filtered = [...projects]

              // Search
              if (filters.searchTerm) {
                filtered = filtered.filter(p => 
                  p.name.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.developer.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.location.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                  p.subtype.toLowerCase().includes(filters.searchTerm.toLowerCase())
                )
              }

              // Type filter (SMR subtypes)
              if (filters.types.length > 0) {
                filtered = filtered.filter(p => {
                  // Map generic types to SMR subtypes
                  if (filters.types.includes('Nuclear') || filters.types.includes('SMR')) {
                    return true // Show all SMRs if Nuclear/SMR is selected
                  }
                  return filters.types.some(type => p.subtype.includes(type))
                })
              }

              // Status filter
              if (filters.statuses.length > 0) {
                filtered = filtered.filter(p => filters.statuses.includes(p.status))
              }

              // Country filter
              if (filters.countries.length > 0) {
                filtered = filtered.filter(p => filters.countries.includes(p.country))
              }

              // Developer filter
              if (filters.developers.length > 0) {
                filtered = filtered.filter(p => filters.developers.includes(p.developer))
              }

              // Risk level filter
              if (filters.riskLevels.length > 0) {
                filtered = filtered.filter(p => {
                  const riskLevel = p.risk_factors.some(f => f.toLowerCase().includes('first')) ? 'Medium' : 'Low'
                  return filters.riskLevels.includes(riskLevel)
                })
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
              if (filters.startDateFrom || filters.startDateTo) {
                filtered = filtered.filter(p => {
                  const startDate = new Date(p.construction_start)
                  if (filters.startDateFrom && startDate < new Date(filters.startDateFrom)) return false
                  if (filters.startDateTo && startDate > new Date(filters.startDateTo)) return false
                  return true
                })
              }

              if (filters.completionFrom || filters.completionTo) {
                filtered = filtered.filter(p => {
                  const completionDate = new Date(p.commercial_operation)
                  if (filters.completionFrom && completionDate < new Date(filters.completionFrom)) return false
                  if (filters.completionTo && completionDate > new Date(filters.completionTo)) return false
                  return true
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
                  case 'completion':
                    return new Date(a.commercial_operation).getTime() - new Date(b.commercial_operation).getTime() * order
                  case 'name':
                    return a.name.localeCompare(b.name) * order
                  case 'location':
                    return a.location.localeCompare(b.location) * order
                  default:
                    return 0
                }
              })

              setFilteredProjects(filtered)
            }}
            projectCount={filteredProjects.length}
            availableTypes={['Nuclear', 'SMR', 'PWR', 'BWR', 'MSR', 'HTGR', 'FNR']}
            availableCountries={Array.from(new Set(projects.map(p => p.country).filter(Boolean)))}
            availableDevelopers={Array.from(new Set(projects.map(p => p.developer).filter(Boolean)))}
            className="mb-8"
          />

          {/* Project Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
            {filteredProjects.map((project) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.02 }}
                className="bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl overflow-hidden hover:border-emerald-500/50 transition-all"
              >
                {/* Project Header */}
                <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 p-6 border-b border-gray-800">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2">{project.name}</h3>
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <div className="flex items-center gap-1">
                          <Building className="h-4 w-4" />
                          {project.developer}
                        </div>
                        <div className="flex items-center gap-1">
                          <MapPin className="h-4 w-4" />
                          {project.location}, {project.country}
                        </div>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(project.status)}`}>
                      {project.status}
                    </span>
                  </div>
                  
                  {/* Key Metrics */}
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <div className="text-2xl font-bold text-emerald-400">{project.roi_percentage}%</div>
                      <div className="text-xs text-gray-500">ROI</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-white">{project.capacity_mw} MW</div>
                      <div className="text-xs text-gray-500">Capacity</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-cyan-400">{project.modules}</div>
                      <div className="text-xs text-gray-500">Modules</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-purple-400">{project.design_life_years}yr</div>
                      <div className="text-xs text-gray-500">Lifetime</div>
                    </div>
                  </div>
                </div>

                {/* Project Details */}
                <div className="p-6">
                  {/* Technology */}
                  <div className="mb-4">
                    <div className="text-sm text-gray-400 mb-1">Technology</div>
                    <div className="text-white font-medium">{project.subtype}</div>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-gray-300 mb-4 line-clamp-2">
                    {project.description}
                  </p>

                  {/* Investment Details */}
                  <div className="bg-black/30 rounded-lg p-4 mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Investment Progress</span>
                      <span className="text-sm font-semibold text-white">
                        {((project.investment_raised / project.total_investment) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden mb-2">
                      <motion.div
                        className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(project.investment_raised / project.total_investment) * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{formatNumber(project.investment_raised)} raised</span>
                      <span>{formatNumber(project.investment_needed)} needed</span>
                    </div>
                  </div>

                  {/* Key Features */}
                  <div className="mb-4">
                    <div className="text-sm text-gray-400 mb-2">Key Features</div>
                    <div className="flex flex-wrap gap-2">
                      {project.key_features.slice(0, 3).map((feature, idx) => (
                        <span key={idx} className="px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-xs text-emerald-400">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Investment Info */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Minimum Investment</div>
                      <div className="text-lg font-semibold text-white">{formatNumber(project.min_investment)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Investment Type</div>
                      <div className="text-lg font-semibold text-white">{project.investment_type}</div>
                    </div>
                  </div>

                  {/* Timeline */}
                  <div className="flex items-center gap-4 mb-4 text-sm">
                    <div className="flex items-center gap-1 text-gray-400">
                      <Clock className="h-4 w-4" />
                      Start: {project.construction_start}
                    </div>
                    <div className="flex items-center gap-1 text-emerald-400">
                      <CheckCircle className="h-4 w-4" />
                      Complete: {project.commercial_operation}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <button
                      onClick={() => setSelectedProject(project)}
                      className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-center text-sm font-medium text-gray-300 hover:bg-gray-700 transition"
                    >
                      View Details
                    </button>
                    <button
                      onClick={() => {
                        if (!user) {
                          router.push('/auth/login')
                        } else {
                          // Create a special investment page for SMR projects
                          router.push(`/invest/smr/${project.id}`)
                        }
                      }}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg text-center text-sm font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all flex items-center justify-center gap-2"
                    >
                      Invest Now
                      <ArrowUp className="h-4 w-4" />
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
              <h3 className="text-lg font-semibold text-gray-400 mb-2">No SMR projects found</h3>
              <p className="text-gray-500">Try adjusting your filters</p>
            </div>
          )}

          {/* Why SMR Section */}
          <div className="bg-gradient-to-br from-emerald-950/20 to-cyan-950/20 border border-emerald-500/20 rounded-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Sparkles className="h-8 w-8 text-yellow-400" />
              Why Invest in Small Modular Reactors?
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <Shield className="h-8 w-8 text-emerald-400 mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Enhanced Safety</h3>
                <p className="text-sm text-gray-400">
                  Passive safety systems, walk-away safe designs, and underground containment make SMRs 
                  inherently safer than traditional nuclear plants.
                </p>
              </div>
              <div>
                <Battery className="h-8 w-8 text-cyan-400 mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Grid Flexibility</h3>
                <p className="text-sm text-gray-400">
                  Load-following capability, black-start capability, and integration with renewables 
                  provide essential grid stability services.
                </p>
              </div>
              <div>
                <Briefcase className="h-8 w-8 text-purple-400 mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Strong Economics</h3>
                <p className="text-sm text-gray-400">
                  Factory manufacturing, shorter construction times, and 60+ year lifespans deliver 
                  competitive levelized costs and attractive returns.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Project Details Modal */}
      {selectedProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 border border-gray-700 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
          >
            <div className="sticky top-0 bg-gray-900 border-b border-gray-800 p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">{selectedProject.name}</h2>
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <span>{selectedProject.developer}</span>
                    <span>•</span>
                    <span>{selectedProject.location}, {selectedProject.country}</span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedProject(null)}
                  className="text-gray-400 hover:text-white transition"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="p-6">
              {/* Full Description */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Project Overview</h3>
                <p className="text-gray-300 leading-relaxed">{selectedProject.description}</p>
              </div>

              {/* Technical Specifications */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Technical Specifications</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Reactor Type</div>
                    <div className="text-white font-medium">{selectedProject.subtype}</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Total Capacity</div>
                    <div className="text-white font-medium">{selectedProject.capacity_mw} MW</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Modules</div>
                    <div className="text-white font-medium">{selectedProject.modules} × {selectedProject.module_capacity_mw} MW</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Capacity Factor</div>
                    <div className="text-white font-medium">{selectedProject.capacity_factor}%</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Design Life</div>
                    <div className="text-white font-medium">{selectedProject.design_life_years} years</div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">LCOE</div>
                    <div className="text-white font-medium">${selectedProject.levelized_cost_mwh}/MWh</div>
                  </div>
                </div>
              </div>

              {/* All Key Features */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Key Features</h3>
                <div className="space-y-2">
                  {selectedProject.key_features.map((feature, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <CheckCircle className="h-5 w-5 text-emerald-400 mt-0.5" />
                      <span className="text-gray-300">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Investors */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Current Investors</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedProject.investors.map((investor, idx) => (
                    <span key={idx} className="px-3 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg text-sm text-purple-400">
                      {investor}
                    </span>
                  ))}
                </div>
              </div>

              {/* Risk Factors */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Risk Factors</h3>
                <div className="space-y-2">
                  {selectedProject.risk_factors.map((risk, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5" />
                      <span className="text-gray-300">{risk}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Investment Details */}
              <div className="bg-gradient-to-br from-emerald-950/30 to-cyan-950/30 border border-emerald-500/30 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Investment Opportunity</h3>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Total Investment Needed</div>
                    <div className="text-2xl font-bold text-emerald-400">{formatNumber(selectedProject.investment_needed)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Expected ROI</div>
                    <div className="text-2xl font-bold text-cyan-400">{selectedProject.roi_percentage}% Annual</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Minimum Investment</div>
                    <div className="text-lg font-semibold text-white">{formatNumber(selectedProject.min_investment)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Investment Type</div>
                    <div className="text-lg font-semibold text-white">{selectedProject.investment_type}</div>
                  </div>
                </div>
                
                <div className="mt-6 flex gap-3">
                  <a
                    href={selectedProject.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-center font-medium text-gray-300 hover:bg-gray-700 transition"
                  >
                    Learn More
                  </a>
                  <button
                    onClick={() => {
                      setSelectedProject(null)
                      if (!user) {
                        router.push('/auth/login')
                      } else {
                        router.push(`/invest/smr/${selectedProject.id}`)
                      }
                    }}
                    className="flex-1 px-4 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all"
                  >
                    Invest in This Project
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}