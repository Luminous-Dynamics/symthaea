'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Filter, X, ChevronDown, Zap, MapPin, Building, DollarSign } from 'lucide-react'
import Link from 'next/link'

interface SearchResult {
  id: number
  name: string
  type: string
  state: string
  country: string
  developer: string
  capacity_mw: number
  status: string
}

interface FilterOptions {
  types: string[]
  statuses: string[]
  countries: string[]
  capacityMin: number
  capacityMax: number
  investmentMin: number
  investmentMax: number
}

export default function ProjectSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<FilterOptions>({
    types: [],
    statuses: [],
    countries: [],
    capacityMin: 0,
    capacityMax: 10000,
    investmentMin: 0,
    investmentMax: 1000000000
  })
  
  const searchRef = useRef<HTMLDivElement>(null)
  const debounceTimer = useRef<NodeJS.Timeout>()

  // Click outside handler
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false)
        setShowFilters(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Debounced search
  const performSearch = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim() && filters.types.length === 0 && filters.statuses.length === 0) {
      setResults([])
      return
    }

    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (searchQuery) params.append('q', searchQuery)
      if (filters.types.length > 0) params.append('types', filters.types.join(','))
      if (filters.statuses.length > 0) params.append('statuses', filters.statuses.join(','))
      if (filters.countries.length > 0) params.append('countries', filters.countries.join(','))
      params.append('limit', '20')

      const response = await fetch(`/api/projects/search?${params}`)
      if (response.ok) {
        const data = await response.json()
        setResults(data.results || [])
        setShowResults(true)
      }
    } catch (error) {
      console.error('Search error:', error)
    }
    setLoading(false)
  }, [filters])

  // Handle search input
  const handleSearch = (value: string) => {
    setQuery(value)
    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      performSearch(value)
    }, 300)
  }

  // Handle filter changes
  const toggleFilter = (category: keyof FilterOptions, value: string) => {
    setFilters(prev => {
      const updated = { ...prev }
      const arr = updated[category] as string[]
      const index = arr.indexOf(value)
      if (index > -1) {
        arr.splice(index, 1)
      } else {
        arr.push(value)
      }
      return updated
    })
  }

  // Clear all filters
  const clearFilters = () => {
    setFilters({
      types: [],
      statuses: [],
      countries: [],
      capacityMin: 0,
      capacityMax: 10000,
      investmentMin: 0,
      investmentMax: 1000000000
    })
    setQuery('')
    setResults([])
    setShowResults(false)
  }

  const activeFilterCount = 
    filters.types.length + 
    filters.statuses.length + 
    filters.countries.length

  return (
    <div ref={searchRef} className="relative w-full max-w-3xl mx-auto">
      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onFocus={() => query && setShowResults(true)}
          placeholder="Search projects by name, location, or developer..."
          className="w-full pl-12 pr-24 py-4 bg-gray-900/50 backdrop-blur-md border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none transition-colors"
        />
        
        {/* Filter Button */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`absolute right-2 top-1/2 -translate-y-1/2 px-3 py-2 rounded-lg border transition-colors flex items-center gap-2 ${
            showFilters || activeFilterCount > 0
              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
              : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Filter className="h-4 w-4" />
          {activeFilterCount > 0 && (
            <span className="text-xs font-bold">{activeFilterCount}</span>
          )}
        </button>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <div className="absolute top-full mt-2 w-full bg-gray-900/95 backdrop-blur-md border border-gray-700 rounded-xl shadow-2xl z-50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white">Advanced Filters</h3>
            <button
              onClick={clearFilters}
              className="text-sm text-gray-400 hover:text-white transition"
            >
              Clear All
            </button>
          </div>

          {/* Project Types */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-400 mb-3">Project Type</h4>
            <div className="flex flex-wrap gap-2">
              {['Solar', 'Wind', 'Hydro', 'Battery', 'Nuclear', 'Geothermal'].map(type => (
                <button
                  key={type}
                  onClick={() => toggleFilter('types', type)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    filters.types.includes(type)
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                      : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Status */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-400 mb-3">Status</h4>
            <div className="flex flex-wrap gap-2">
              {['Operational', 'Construction', 'Development', 'Planned'].map(status => (
                <button
                  key={status}
                  onClick={() => toggleFilter('statuses', status)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    filters.statuses.includes(status)
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                      : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
                  }`}
                >
                  {status}
                </button>
              ))}
            </div>
          </div>

          {/* Capacity Range */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-400 mb-3">Capacity (MW)</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-gray-500">Min</label>
                <input
                  type="number"
                  value={filters.capacityMin}
                  onChange={(e) => setFilters(prev => ({ ...prev, capacityMin: Number(e.target.value) }))}
                  className="w-full mt-1 px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white text-sm focus:border-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Max</label>
                <input
                  type="number"
                  value={filters.capacityMax}
                  onChange={(e) => setFilters(prev => ({ ...prev, capacityMax: Number(e.target.value) }))}
                  className="w-full mt-1 px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white text-sm focus:border-emerald-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Apply Filters Button */}
          <button
            onClick={() => performSearch(query)}
            className="w-full py-2 bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 font-semibold rounded-lg hover:bg-emerald-500/30 transition"
          >
            Apply Filters
          </button>
        </div>
      )}

      {/* Search Results */}
      {showResults && (query || activeFilterCount > 0) && (
        <div className="absolute top-full mt-2 w-full bg-gray-900/95 backdrop-blur-md border border-gray-700 rounded-xl shadow-2xl z-40 max-h-96 overflow-y-auto">
          {loading ? (
            <div className="p-8 text-center text-gray-400">
              <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-400"></div>
              <p className="mt-2">Searching...</p>
            </div>
          ) : results.length > 0 ? (
            <div className="divide-y divide-gray-800">
              {results.map((project) => (
                <Link
                  key={project.id}
                  href={`/project/${project.id}`}
                  className="block p-4 hover:bg-gray-800/50 transition-colors"
                  onClick={() => setShowResults(false)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-white mb-1">{project.name}</h4>
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <span className="flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {project.state}, {project.country}
                        </span>
                        <span className="flex items-center gap-1">
                          <Zap className="h-3 w-3" />
                          {project.capacity_mw} MW
                        </span>
                        <span className="flex items-center gap-1">
                          <Building className="h-3 w-3" />
                          {project.developer}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                        project.type === 'Solar' ? 'bg-yellow-500/20 text-yellow-400' :
                        project.type === 'Wind' ? 'bg-blue-500/20 text-blue-400' :
                        project.type === 'Hydro' ? 'bg-cyan-500/20 text-cyan-400' :
                        'bg-purple-500/20 text-purple-400'
                      }`}>
                        {project.type}
                      </span>
                      <p className="text-xs text-gray-500 mt-1">{project.status}</p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-gray-400">
              <p>No projects found</p>
              <p className="text-sm mt-2">Try adjusting your search or filters</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}