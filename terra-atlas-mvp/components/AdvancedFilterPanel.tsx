'use client'

import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, Filter, X, ChevronDown, ChevronUp, Calendar, DollarSign,
  TrendingUp, MapPin, Zap, Building2, Save, Clock, Check, Star,
  BarChart3, Globe, Battery, Sun, Wind, Droplets, Atom, AlertCircle
} from 'lucide-react'

interface FilterOptions {
  types: string[]
  statuses: string[]
  countries: string[]
  developers: string[]
  minCapacity: number
  maxCapacity: number
  minInvestment: number
  maxInvestment: number
  minROI: number
  maxROI: number
  startDateFrom: string
  startDateTo: string
  completionFrom: string
  completionTo: string
  riskLevels: string[]
  searchTerm: string
  sortBy: string
  sortOrder: 'asc' | 'desc'
}

interface FilterPreset {
  id: string
  name: string
  filters: Partial<FilterOptions>
  icon?: string
  description?: string
}

interface AdvancedFilterPanelProps {
  onFiltersChange: (filters: FilterOptions) => void
  onClose?: () => void
  projectCount?: number
  availableTypes?: string[]
  availableCountries?: string[]
  availableDevelopers?: string[]
  className?: string
}

const defaultFilters: FilterOptions = {
  types: [],
  statuses: [],
  countries: [],
  developers: [],
  minCapacity: 0,
  maxCapacity: 10000,
  minInvestment: 10,
  maxInvestment: 10000000,
  minROI: 0,
  maxROI: 30,
  startDateFrom: '',
  startDateTo: '',
  completionFrom: '',
  completionTo: '',
  riskLevels: [],
  searchTerm: '',
  sortBy: 'roi',
  sortOrder: 'desc'
}

const filterPresets: FilterPreset[] = [
  {
    id: 'high-roi',
    name: 'High ROI',
    icon: 'trending',
    description: 'Projects with 15%+ annual returns',
    filters: { minROI: 15 }
  },
  {
    id: 'solar-wind',
    name: 'Solar & Wind',
    icon: 'renewable',
    description: 'Renewable energy projects',
    filters: { types: ['Solar', 'Wind'] }
  },
  {
    id: 'operational',
    name: 'Operating Now',
    icon: 'check',
    description: 'Currently operational projects',
    filters: { statuses: ['Operational'] }
  },
  {
    id: 'small-investor',
    name: 'Small Investor',
    icon: 'dollar',
    description: 'Projects under $1000 minimum',
    filters: { maxInvestment: 1000 }
  },
  {
    id: 'nuclear',
    name: 'Nuclear/SMR',
    icon: 'atom',
    description: 'Nuclear and SMR projects',
    filters: { types: ['Nuclear', 'SMR'] }
  },
  {
    id: 'near-completion',
    name: 'Near Completion',
    icon: 'clock',
    description: 'Completing within 1 year',
    filters: {
      completionFrom: new Date().toISOString().split('T')[0],
      completionTo: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    }
  }
]

export default function AdvancedFilterPanel({
  onFiltersChange,
  onClose,
  projectCount = 0,
  availableTypes = ['Solar', 'Wind', 'Battery', 'Hydro', 'Nuclear', 'SMR', 'Geothermal'],
  availableCountries = ['USA', 'China', 'Germany', 'India', 'Brazil', 'Australia', 'Canada', 'UK', 'Japan'],
  availableDevelopers = [],
  className = ''
}: AdvancedFilterPanelProps) {
  const [filters, setFilters] = useState<FilterOptions>(defaultFilters)
  const [expandedSections, setExpandedSections] = useState<string[]>(['type', 'search'])
  const [savedPresets, setSavedPresets] = useState<FilterPreset[]>([])
  const [activePreset, setActivePreset] = useState<string | null>(null)
  const [searchSuggestions, setSearchSuggestions] = useState<string[]>([])
  const [isSearchFocused, setIsSearchFocused] = useState(false)

  // Load saved presets from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('terraAtlasFilterPresets')
    if (saved) {
      setSavedPresets(JSON.parse(saved))
    }
  }, [])

  // Update parent component when filters change
  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      onFiltersChange(filters)
    }, 300)

    return () => clearTimeout(debounceTimer)
  }, [filters, onFiltersChange])

  const toggleSection = (section: string) => {
    setExpandedSections(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }

  const handleFilterChange = (key: keyof FilterOptions, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }))
    setActivePreset(null) // Clear active preset when manually changing filters
  }

  const toggleArrayFilter = (key: keyof FilterOptions, value: string) => {
    const currentArray = filters[key] as string[]
    const newArray = currentArray.includes(value)
      ? currentArray.filter(v => v !== value)
      : [...currentArray, value]
    handleFilterChange(key, newArray)
  }

  const applyPreset = (preset: FilterPreset) => {
    setFilters(prev => ({ ...defaultFilters, ...preset.filters }))
    setActivePreset(preset.id)
  }

  const saveCurrentAsPreset = () => {
    const name = prompt('Enter a name for this filter preset:')
    if (!name) return

    const newPreset: FilterPreset = {
      id: `custom-${Date.now()}`,
      name,
      filters: { ...filters },
      description: 'Custom saved filter'
    }

    const updated = [...savedPresets, newPreset]
    setSavedPresets(updated)
    localStorage.setItem('terraAtlasFilterPresets', JSON.stringify(updated))
  }

  const deletePreset = (id: string) => {
    const updated = savedPresets.filter(p => p.id !== id)
    setSavedPresets(updated)
    localStorage.setItem('terraAtlasFilterPresets', JSON.stringify(updated))
    if (activePreset === id) setActivePreset(null)
  }

  const clearAllFilters = () => {
    setFilters(defaultFilters)
    setActivePreset(null)
  }

  const activeFilterCount = useMemo(() => {
    let count = 0
    if (filters.types.length > 0) count++
    if (filters.statuses.length > 0) count++
    if (filters.countries.length > 0) count++
    if (filters.developers.length > 0) count++
    if (filters.riskLevels.length > 0) count++
    if (filters.minCapacity > 0 || filters.maxCapacity < 10000) count++
    if (filters.minInvestment > 10 || filters.maxInvestment < 10000000) count++
    if (filters.minROI > 0 || filters.maxROI < 30) count++
    if (filters.startDateFrom || filters.startDateTo) count++
    if (filters.completionFrom || filters.completionTo) count++
    if (filters.searchTerm) count++
    return count
  }, [filters])

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'solar': return <Sun className="h-4 w-4" />
      case 'wind': return <Wind className="h-4 w-4" />
      case 'battery': return <Battery className="h-4 w-4" />
      case 'hydro': return <Droplets className="h-4 w-4" />
      case 'nuclear': case 'smr': return <Atom className="h-4 w-4" />
      default: return <Zap className="h-4 w-4" />
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`bg-gray-900/95 backdrop-blur-xl border border-gray-700 rounded-2xl overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border-b border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <Filter className="h-6 w-6 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Advanced Filters</h2>
              <p className="text-sm text-gray-400">
                {activeFilterCount > 0 ? `${activeFilterCount} filter${activeFilterCount > 1 ? 's' : ''} active • ` : ''}
                {projectCount} projects found
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={saveCurrentAsPreset}
              className="p-2 bg-purple-500/20 border border-purple-500/30 rounded-lg text-purple-400 hover:bg-purple-500/30 transition"
              title="Save current filters as preset"
            >
              <Save className="h-4 w-4" />
            </button>
            <button
              onClick={clearAllFilters}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-400 hover:text-white transition"
            >
              Clear All
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-white transition"
              >
                <X className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>

        {/* Filter Presets */}
        <div className="flex flex-wrap gap-2">
          {[...filterPresets, ...savedPresets].map(preset => (
            <div key={preset.id} className="relative group">
              <button
                onClick={() => applyPreset(preset)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition flex items-center gap-1.5 ${
                  activePreset === preset.id
                    ? 'bg-emerald-500/30 border border-emerald-500 text-emerald-400'
                    : 'bg-gray-800/50 border border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300'
                }`}
                title={preset.description}
              >
                {preset.id === 'high-roi' && <TrendingUp className="h-3.5 w-3.5" />}
                {preset.id === 'solar-wind' && <Sun className="h-3.5 w-3.5" />}
                {preset.id === 'operational' && <Check className="h-3.5 w-3.5" />}
                {preset.id === 'small-investor' && <DollarSign className="h-3.5 w-3.5" />}
                {preset.id === 'nuclear' && <Atom className="h-3.5 w-3.5" />}
                {preset.id === 'near-completion' && <Clock className="h-3.5 w-3.5" />}
                {preset.id.startsWith('custom-') && <Star className="h-3.5 w-3.5" />}
                {preset.name}
              </button>
              {preset.id.startsWith('custom-') && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deletePreset(preset.id)
                  }}
                  className="absolute -top-1 -right-1 p-0.5 bg-red-500 rounded-full text-white opacity-0 group-hover:opacity-100 transition"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Filter Sections */}
      <div className="max-h-[600px] overflow-y-auto">
        {/* Search */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('search')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <Search className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Search</span>
              {filters.searchTerm && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  Active
                </span>
              )}
            </div>
            {expandedSections.includes('search') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('search') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                    <input
                      type="text"
                      value={filters.searchTerm}
                      onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
                      onFocus={() => setIsSearchFocused(true)}
                      onBlur={() => setTimeout(() => setIsSearchFocused(false), 200)}
                      placeholder="Search projects, locations, developers..."
                      className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 pl-10 pr-4 text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition"
                    />
                  </div>
                  {isSearchFocused && searchSuggestions.length > 0 && (
                    <div className="absolute mt-1 w-full bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-10">
                      {searchSuggestions.map((suggestion, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            handleFilterChange('searchTerm', suggestion)
                            setIsSearchFocused(false)
                          }}
                          className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 transition"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Project Type */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('type')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <Zap className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Project Type</span>
              {filters.types.length > 0 && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  {filters.types.length} selected
                </span>
              )}
            </div>
            {expandedSections.includes('type') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('type') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 grid grid-cols-2 md:grid-cols-3 gap-2">
                  {availableTypes.map(type => (
                    <button
                      key={type}
                      onClick={() => toggleArrayFilter('types', type)}
                      className={`px-3 py-2 rounded-lg border text-sm font-medium transition flex items-center gap-2 ${
                        filters.types.includes(type)
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400'
                          : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {getTypeIcon(type)}
                      {type}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Status */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('status')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Project Status</span>
              {filters.statuses.length > 0 && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  {filters.statuses.length} selected
                </span>
              )}
            </div>
            {expandedSections.includes('status') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('status') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 grid grid-cols-2 gap-2">
                  {['Operational', 'Construction', 'Development', 'Planned'].map(status => (
                    <button
                      key={status}
                      onClick={() => toggleArrayFilter('statuses', status)}
                      className={`px-3 py-2 rounded-lg border text-sm font-medium transition ${
                        filters.statuses.includes(status)
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400'
                          : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {status}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Location */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('location')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <Globe className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Location</span>
              {filters.countries.length > 0 && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  {filters.countries.length} selected
                </span>
              )}
            </div>
            {expandedSections.includes('location') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('location') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 grid grid-cols-2 md:grid-cols-3 gap-2">
                  {availableCountries.map(country => (
                    <button
                      key={country}
                      onClick={() => toggleArrayFilter('countries', country)}
                      className={`px-3 py-2 rounded-lg border text-sm font-medium transition ${
                        filters.countries.includes(country)
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400'
                          : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {country}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Investment Range */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('investment')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <DollarSign className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Investment Range</span>
              {(filters.minInvestment > 10 || filters.maxInvestment < 10000000) && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  Custom
                </span>
              )}
            </div>
            {expandedSections.includes('investment') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('investment') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs text-gray-400 mb-1 block">Minimum ($)</label>
                      <input
                        type="number"
                        value={filters.minInvestment}
                        onChange={(e) => handleFilterChange('minInvestment', Number(e.target.value))}
                        className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-400 mb-1 block">Maximum ($)</label>
                      <input
                        type="number"
                        value={filters.maxInvestment}
                        onChange={(e) => handleFilterChange('maxInvestment', Number(e.target.value))}
                        className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        handleFilterChange('minInvestment', 10)
                        handleFilterChange('maxInvestment', 1000)
                      }}
                      className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-400 hover:text-white transition"
                    >
                      $10-$1K
                    </button>
                    <button
                      onClick={() => {
                        handleFilterChange('minInvestment', 1000)
                        handleFilterChange('maxInvestment', 10000)
                      }}
                      className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-400 hover:text-white transition"
                    >
                      $1K-$10K
                    </button>
                    <button
                      onClick={() => {
                        handleFilterChange('minInvestment', 10000)
                        handleFilterChange('maxInvestment', 100000)
                      }}
                      className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-400 hover:text-white transition"
                    >
                      $10K-$100K
                    </button>
                    <button
                      onClick={() => {
                        handleFilterChange('minInvestment', 100000)
                        handleFilterChange('maxInvestment', 10000000)
                      }}
                      className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-400 hover:text-white transition"
                    >
                      $100K+
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ROI Range */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('roi')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">ROI Range</span>
              {(filters.minROI > 0 || filters.maxROI < 30) && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  {filters.minROI}%-{filters.maxROI}%
                </span>
              )}
            </div>
            {expandedSections.includes('roi') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('roi') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs text-gray-400 mb-1 block">Min ROI (%)</label>
                      <input
                        type="number"
                        value={filters.minROI}
                        onChange={(e) => handleFilterChange('minROI', Number(e.target.value))}
                        min="0"
                        max="30"
                        className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-400 mb-1 block">Max ROI (%)</label>
                      <input
                        type="number"
                        value={filters.maxROI}
                        onChange={(e) => handleFilterChange('maxROI', Number(e.target.value))}
                        min="0"
                        max="30"
                        className="w-full bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>0%</span>
                      <span>15%</span>
                      <span>30%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="30"
                      value={filters.minROI}
                      onChange={(e) => handleFilterChange('minROI', Number(e.target.value))}
                      className="w-full"
                    />
                    <input
                      type="range"
                      min="0"
                      max="30"
                      value={filters.maxROI}
                      onChange={(e) => handleFilterChange('maxROI', Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Dates */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('dates')}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-800/30 transition"
          >
            <div className="flex items-center gap-3">
              <Calendar className="h-5 w-5 text-gray-400" />
              <span className="font-medium text-white">Timeline</span>
              {(filters.startDateFrom || filters.startDateTo || filters.completionFrom || filters.completionTo) && (
                <span className="px-2 py-0.5 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                  Custom
                </span>
              )}
            </div>
            {expandedSections.includes('dates') ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.includes('dates') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-4 space-y-4">
                  <div>
                    <label className="text-xs text-gray-400 mb-2 block">Construction Start Date</label>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="date"
                        value={filters.startDateFrom}
                        onChange={(e) => handleFilterChange('startDateFrom', e.target.value)}
                        className="bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                      <input
                        type="date"
                        value={filters.startDateTo}
                        onChange={(e) => handleFilterChange('startDateTo', e.target.value)}
                        className="bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 mb-2 block">Completion Date</label>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="date"
                        value={filters.completionFrom}
                        onChange={(e) => handleFilterChange('completionFrom', e.target.value)}
                        className="bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                      <input
                        type="date"
                        value={filters.completionTo}
                        onChange={(e) => handleFilterChange('completionTo', e.target.value)}
                        className="bg-black/30 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Sort Options */}
        <div className="px-6 py-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-white">Sort By</span>
            <div className="flex gap-2">
              <button
                onClick={() => handleFilterChange('sortOrder', filters.sortOrder === 'asc' ? 'desc' : 'asc')}
                className="p-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white transition"
              >
                <BarChart3 className={`h-4 w-4 transition-transform ${filters.sortOrder === 'desc' ? 'rotate-180' : ''}`} />
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[
              { value: 'roi', label: 'ROI' },
              { value: 'capacity', label: 'Capacity' },
              { value: 'investment', label: 'Investment' },
              { value: 'completion', label: 'Completion' },
              { value: 'name', label: 'Name' },
              { value: 'location', label: 'Location' }
            ].map(option => (
              <button
                key={option.value}
                onClick={() => handleFilterChange('sortBy', option.value)}
                className={`px-3 py-2 rounded-lg border text-sm font-medium transition ${
                  filters.sortBy === option.value
                    ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400'
                    : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="bg-gray-800/30 border-t border-gray-700 p-4 flex items-center justify-between">
        <div className="text-sm text-gray-400">
          {activeFilterCount === 0 ? 'No filters applied' : `${activeFilterCount} filter${activeFilterCount > 1 ? 's' : ''} active`}
        </div>
        <div className="flex gap-2">
          <button
            onClick={clearAllFilters}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm font-medium text-gray-300 hover:bg-gray-700 transition"
          >
            Reset
          </button>
          <button
            onClick={() => onFiltersChange(filters)}
            className="px-6 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg text-sm font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all"
          >
            Apply Filters
          </button>
        </div>
      </div>
    </motion.div>
  )
}