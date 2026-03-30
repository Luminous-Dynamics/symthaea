// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Filter, X, ChevronDown, ChevronUp, Search, 
  Zap, Wind, Droplet, Mountain, Battery, 
  DollarSign, MapPin, Calendar, TrendingUp,
  RotateCcw, Download, Save
} from 'lucide-react'

export interface FilterState {
  energyTypes: string[]
  capacityRange: [number, number]
  investmentRange: [number, number]
  locations: string[]
  status: string[]
  yearRange: [number, number]
  riskLevel: string[]
  sortBy: string
  sortOrder: 'asc' | 'desc'
}

interface AdvancedFiltersProps {
  onFiltersChange: (filters: FilterState) => void
  onReset: () => void
  projectCount?: number
}

const defaultFilters: FilterState = {
  energyTypes: [],
  capacityRange: [0, 10000],
  investmentRange: [0, 1000000000],
  locations: [],
  status: [],
  yearRange: [2020, 2030],
  riskLevel: [],
  sortBy: 'capacity',
  sortOrder: 'desc'
}

export default function AdvancedFilters({ onFiltersChange, onReset, projectCount = 0 }: AdvancedFiltersProps) {
  const [filters, setFilters] = useState<FilterState>(defaultFilters)
  const [expandedSections, setExpandedSections] = useState<string[]>(['energy', 'capacity'])
  const [searchQuery, setSearchQuery] = useState('')
  const [savedFilters, setSavedFilters] = useState<{ name: string; filters: FilterState }[]>([])

  // Energy type options with icons
  const energyTypes = [
    { value: 'solar', label: 'Solar', icon: '☀️', color: 'from-yellow-500 to-orange-500' },
    { value: 'wind', label: 'Wind', icon: '💨', color: 'from-blue-500 to-cyan-500' },
    { value: 'hydro', label: 'Hydro', icon: '💧', color: 'from-blue-600 to-blue-400' },
    { value: 'geothermal', label: 'Geothermal', icon: '🌋', color: 'from-red-500 to-orange-600' },
    { value: 'battery', label: 'Battery Storage', icon: '🔋', color: 'from-green-500 to-emerald-500' },
    { value: 'hybrid', label: 'Hybrid', icon: '⚡', color: 'from-purple-500 to-pink-500' }
  ]

  // Status options
  const statusOptions = [
    { value: 'operational', label: 'Operational', color: 'bg-green-500' },
    { value: 'construction', label: 'Under Construction', color: 'bg-yellow-500' },
    { value: 'planning', label: 'Planning', color: 'bg-blue-500' },
    { value: 'proposed', label: 'Proposed', color: 'bg-purple-500' }
  ]

  // Risk levels
  const riskLevels = [
    { value: 'low', label: 'Low Risk', color: 'bg-green-500' },
    { value: 'medium', label: 'Medium Risk', color: 'bg-yellow-500' },
    { value: 'high', label: 'High Risk', color: 'bg-red-500' }
  ]

  // Popular locations
  const popularLocations = [
    'United States', 'China', 'India', 'Germany', 'Japan',
    'Brazil', 'Australia', 'United Kingdom', 'France', 'Canada'
  ]

  useEffect(() => {
    // Debounce filter changes
    const timer = setTimeout(() => {
      onFiltersChange(filters)
    }, 300)
    return () => clearTimeout(timer)
  }, [filters])

  useEffect(() => {
    // Load saved filters from localStorage
    const saved = localStorage.getItem('savedFilters')
    if (saved) {
      setSavedFilters(JSON.parse(saved))
    }
  }, [])

  const toggleSection = (section: string) => {
    setExpandedSections(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }

  const handleEnergyTypeToggle = (type: string) => {
    setFilters(prev => ({
      ...prev,
      energyTypes: prev.energyTypes.includes(type)
        ? prev.energyTypes.filter(t => t !== type)
        : [...prev.energyTypes, type]
    }))
  }

  const handleStatusToggle = (status: string) => {
    setFilters(prev => ({
      ...prev,
      status: prev.status.includes(status)
        ? prev.status.filter(s => s !== status)
        : [...prev.status, status]
    }))
  }

  const handleRiskToggle = (risk: string) => {
    setFilters(prev => ({
      ...prev,
      riskLevel: prev.riskLevel.includes(risk)
        ? prev.riskLevel.filter(r => r !== risk)
        : [...prev.riskLevel, risk]
    }))
  }

  const handleLocationToggle = (location: string) => {
    setFilters(prev => ({
      ...prev,
      locations: prev.locations.includes(location)
        ? prev.locations.filter(l => l !== location)
        : [...prev.locations, location]
    }))
  }

  const handleReset = () => {
    setFilters(defaultFilters)
    onReset()
  }

  const saveCurrentFilters = () => {
    const name = prompt('Enter a name for this filter preset:')
    if (name) {
      const newSaved = [...savedFilters, { name, filters }]
      setSavedFilters(newSaved)
      localStorage.setItem('savedFilters', JSON.stringify(newSaved))
    }
  }

  const loadSavedFilter = (saved: FilterState) => {
    setFilters(saved)
  }

  const exportFilters = () => {
    const dataStr = JSON.stringify(filters, null, 2)
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`
    const exportFileDefaultName = 'terra-atlas-filters.json'
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  return (
    <div className="w-full max-w-sm bg-gradient-to-b from-gray-900 to-black rounded-xl border border-gray-800">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-green-400" />
            <h2 className="text-lg font-bold text-white">Filters</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={saveCurrentFilters}
              className="p-1.5 text-gray-400 hover:text-white transition-colors"
              title="Save filters"
            >
              <Save className="w-4 h-4" />
            </button>
            <button
              onClick={exportFilters}
              className="p-1.5 text-gray-400 hover:text-white transition-colors"
              title="Export filters"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              className="p-1.5 text-gray-400 hover:text-white transition-colors"
              title="Reset filters"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Quick search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Quick search projects..."
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-green-500 focus:outline-none text-sm"
          />
        </div>

        {/* Results count */}
        <div className="mt-3 text-sm text-gray-400">
          <span className="font-semibold text-green-400">{projectCount}</span> projects match
        </div>
      </div>

      {/* Filter sections */}
      <div className="max-h-[600px] overflow-y-auto">
        {/* Energy Types */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('energy')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Energy Type</span>
            {expandedSections.includes('energy') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          <AnimatePresence>
            {expandedSections.includes('energy') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="p-4 grid grid-cols-2 gap-2">
                  {energyTypes.map(type => (
                    <button
                      key={type.value}
                      onClick={() => handleEnergyTypeToggle(type.value)}
                      className={`p-3 rounded-lg border transition-all ${
                        filters.energyTypes.includes(type.value)
                          ? `bg-gradient-to-r ${type.color} border-transparent text-white`
                          : 'bg-gray-800/50 border-gray-700 text-gray-300 hover:border-gray-600'
                      }`}
                    >
                      <div className="text-xl mb-1">{type.icon}</div>
                      <div className="text-xs font-medium">{type.label}</div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Capacity Range */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('capacity')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Capacity (MW)</span>
            {expandedSections.includes('capacity') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          <AnimatePresence>
            {expandedSections.includes('capacity') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <input
                      type="number"
                      value={filters.capacityRange[0]}
                      onChange={(e) => setFilters(prev => ({
                        ...prev,
                        capacityRange: [parseInt(e.target.value) || 0, prev.capacityRange[1]]
                      }))}
                      className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                      placeholder="Min"
                    />
                    <span className="text-gray-400">to</span>
                    <input
                      type="number"
                      value={filters.capacityRange[1]}
                      onChange={(e) => setFilters(prev => ({
                        ...prev,
                        capacityRange: [prev.capacityRange[0], parseInt(e.target.value) || 10000]
                      }))}
                      className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                      placeholder="Max"
                    />
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="10000"
                    value={filters.capacityRange[1]}
                    onChange={(e) => setFilters(prev => ({
                      ...prev,
                      capacityRange: [prev.capacityRange[0], parseInt(e.target.value)]
                    }))}
                    className="w-full"
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Investment Range */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('investment')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Investment Size</span>
            {expandedSections.includes('investment') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
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
                <div className="p-4">
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: '< $10M', min: 0, max: 10000000 },
                      { label: '$10M - $50M', min: 10000000, max: 50000000 },
                      { label: '$50M - $100M', min: 50000000, max: 100000000 },
                      { label: '$100M - $500M', min: 100000000, max: 500000000 },
                      { label: '$500M - $1B', min: 500000000, max: 1000000000 },
                      { label: '> $1B', min: 1000000000, max: 10000000000 }
                    ].map(range => (
                      <button
                        key={range.label}
                        onClick={() => setFilters(prev => ({
                          ...prev,
                          investmentRange: [range.min, range.max]
                        }))}
                        className={`px-3 py-2 rounded-lg border text-xs font-medium transition-all ${
                          filters.investmentRange[0] === range.min && filters.investmentRange[1] === range.max
                            ? 'bg-green-500/20 border-green-500 text-green-400'
                            : 'bg-gray-800/50 border-gray-700 text-gray-300 hover:border-gray-600'
                        }`}
                      >
                        {range.label}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Location */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('location')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Location</span>
            {expandedSections.includes('location') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
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
                <div className="p-4">
                  <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                    {popularLocations.map(location => (
                      <button
                        key={location}
                        onClick={() => handleLocationToggle(location)}
                        className={`px-3 py-2 rounded-lg border text-xs font-medium transition-all ${
                          filters.locations.includes(location)
                            ? 'bg-blue-500/20 border-blue-500 text-blue-400'
                            : 'bg-gray-800/50 border-gray-700 text-gray-300 hover:border-gray-600'
                        }`}
                      >
                        {location}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Project Status */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('status')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Project Status</span>
            {expandedSections.includes('status') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
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
                <div className="p-4 space-y-2">
                  {statusOptions.map(status => (
                    <button
                      key={status.value}
                      onClick={() => handleStatusToggle(status.value)}
                      className={`w-full px-3 py-2 rounded-lg border flex items-center gap-2 transition-all ${
                        filters.status.includes(status.value)
                          ? 'bg-gray-700 border-gray-600 text-white'
                          : 'bg-gray-800/50 border-gray-700 text-gray-300 hover:border-gray-600'
                      }`}
                    >
                      <div className={`w-2 h-2 rounded-full ${status.color}`} />
                      <span className="text-sm">{status.label}</span>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Risk Level */}
        <div className="border-b border-gray-800">
          <button
            onClick={() => toggleSection('risk')}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <span className="text-white font-medium">Risk Level</span>
            {expandedSections.includes('risk') ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          <AnimatePresence>
            {expandedSections.includes('risk') && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden"
              >
                <div className="p-4 space-y-2">
                  {riskLevels.map(risk => (
                    <button
                      key={risk.value}
                      onClick={() => handleRiskToggle(risk.value)}
                      className={`w-full px-3 py-2 rounded-lg border flex items-center gap-2 transition-all ${
                        filters.riskLevel.includes(risk.value)
                          ? 'bg-gray-700 border-gray-600 text-white'
                          : 'bg-gray-800/50 border-gray-700 text-gray-300 hover:border-gray-600'
                      }`}
                    >
                      <div className={`w-2 h-2 rounded-full ${risk.color}`} />
                      <span className="text-sm">{risk.label}</span>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Saved Filters */}
        {savedFilters.length > 0 && (
          <div className="p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Saved Filters</h3>
            <div className="space-y-2">
              {savedFilters.map((saved, index) => (
                <button
                  key={index}
                  onClick={() => loadSavedFilter(saved.filters)}
                  className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-left hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">{saved.name}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSavedFilters(prev => prev.filter((_, i) => i !== index))
                        localStorage.setItem('savedFilters', JSON.stringify(savedFilters.filter((_, i) => i !== index)))
                      }}
                      className="text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <button
          onClick={handleReset}
          className="w-full py-2 bg-gray-800 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors text-sm"
        >
          Clear All Filters
        </button>
      </div>
    </div>
  )
}