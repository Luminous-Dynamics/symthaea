'use client'

import { useState, useEffect, useRef, useMemo } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { AIScoreBadge, AIAnalysisCard } from '@/components/AIScoreBadge'
import { clusterSites, sampleSites } from '@/lib/clustering'

// Dynamically import Three.js globe for SSR safety
const TerraGlobe = dynamic(() => import('@/components/TerraGlobeWithSites'), {
  loading: () => (
    <div className="w-full h-full bg-gradient-to-b from-gray-950 to-black flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-4 mx-auto"></div>
        <p className="text-white/60">Loading 3D Globe...</p>
      </div>
    </div>
  ),
  ssr: false
})

interface Project {
  id: string
  name: string
  type: string
  location: { lat: number; lon: number }
  capacity_mw: number
  status: string
  investment: number
  state: string
  aiScore?: number
}

interface AIAnalysis {
  overallScore: number
  viabilityRating: string
  analyses: Record<string, { score: number; recommendations: string[] }>
  investmentReadiness: {
    score: number
    stage: string
    estimatedTimeToRevenue: string
  }
}

export default function ExplorePage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [selectedAnalysis, setSelectedAnalysis] = useState<AIAnalysis | null>(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [minCapacity, setMinCapacity] = useState(0)
  const projectsCache = useRef<Record<string, Project[]>>({})
  const inFlight = useRef<AbortController | null>(null)

  // Handle site selection from globe
  const handleSiteSelect = (site: any) => {
    // Find matching project in our list
    const project = projects.find(p => p.id === site.id)
    if (project) {
      setSelectedProject(project)
    }
  }

  // Fetch AI analysis when project is selected
  useEffect(() => {
    if (!selectedProject) {
      setSelectedAnalysis(null)
      return
    }

    const fetchAnalysis = async () => {
      setAnalysisLoading(true)
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project: {
              id: selectedProject.id,
              name: selectedProject.name,
              type: selectedProject.type === 'battery' ? 'storage' :
                    selectedProject.type === 'smr' ? 'nuclear' :
                    selectedProject.type === 'dams' ? 'hydro' :
                    selectedProject.type,
              location: {
                latitude: selectedProject.location.lat,
                longitude: selectedProject.location.lon,
                state: selectedProject.state,
              },
              capacityMw: selectedProject.capacity_mw,
              status: selectedProject.status,
            },
          }),
        })

        if (response.ok) {
          const data = await response.json()
          setSelectedAnalysis(data.analysis)
        }
      } catch (error) {
        console.error('Failed to fetch analysis:', error)
      } finally {
        setAnalysisLoading(false)
      }
    }

    fetchAnalysis()
  }, [selectedProject])

  useEffect(() => {
    const controller = new AbortController()
    inFlight.current?.abort()
    inFlight.current = controller

    setLoading(true)
    const cached = projectsCache.current[filter]
    if (cached) {
      setProjects(cached)
      setLoading(false)
      return () => controller.abort()
    }

    const fetchProjects = async () => {
      try {
        // Use new sites API with regional clustering for explore page
        const typeQuery = filter === 'all' ? '' : `&type=${filter}`
        const response = await fetch(`/api/sites?level=regional${typeQuery}`, {
          signal: controller.signal,
        })
        if (!response.ok) {
          console.warn('Sites API failed, falling back to legacy projects API')
          // Fallback to old API
          const fallbackResponse = await fetch(`/api/projects?${filter === 'all' ? '' : `type=${filter}&`}limit=500`, {
            signal: controller.signal,
          })
          if (!fallbackResponse.ok) throw new Error('Failed to load projects')
          const fallbackData = await fallbackResponse.json()

          const projectsWithLocation =
            fallbackData.projects?.map((p: any) => ({
              ...p,
              location: {
                lat: p.latitude || 39.8283 + (Math.random() - 0.5) * 30,
                lon: p.longitude || -98.5795 + (Math.random() - 0.5) * 50,
              },
            })) || []

          if (!controller.signal.aborted) {
            projectsCache.current[filter] = projectsWithLocation
            setProjects(projectsWithLocation)
          }
          return
        }

        const data = await response.json()

        // Transform sites API format to project format
        const projectsWithLocation =
          data.sites?.map((s: any) => ({
            id: s.id,
            name: s.name,
            type: s.type,
            location: {
              lat: s.latitude,
              lon: s.longitude,
            },
            capacity_mw: s.estimated_capacity_mw || 0,
            status: s.status || 'Available',
            investment: s.estimated_capacity_mw * 2500000, // $2.5M per MW estimate
            state: s.state,
            aiScore: s.avg_score,
            isCluster: s.isCluster,
            site_count: s.site_count,
          })) || []

        if (!controller.signal.aborted) {
          projectsCache.current[filter] = projectsWithLocation
          setProjects(projectsWithLocation)
        }
      } catch (error) {
        if (controller.signal.aborted) return
        console.error('Error fetching projects:', error)
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    fetchProjects()

    return () => controller.abort()
  }, [filter])

  // Calculate stats based on filtered projects
  const filteredProjects = useMemo(() => {
    let filtered = projects

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(query) ||
        p.state?.toLowerCase().includes(query) ||
        p.type.toLowerCase().includes(query)
      )
    }

    if (minCapacity > 0) {
      filtered = filtered.filter(p => p.capacity_mw >= minCapacity)
    }

    return filtered
  }, [projects, searchQuery, minCapacity])

  // Calculate total actual sites (including clusters)
  const totalSites = projects.reduce((sum, p) => sum + ((p as any).site_count || 1), 0)
  const filteredTotalSites = filteredProjects.reduce((sum, p) => sum + ((p as any).site_count || 1), 0)

  const stats = {
    total: filteredTotalSites,
    clusters: filteredProjects.length,
    capacity: filteredProjects.reduce((sum, p) => sum + (p.capacity_mw || 0), 0),
    investment: filteredProjects.reduce((sum, p) => sum + (p.investment || 0), 0),
    isFiltered: filteredProjects.length !== projects.length || filteredTotalSites !== totalSites
  }

  // Filter and cluster projects for display
  const displayProjects = useMemo(() => {
    if (projects.length === 0) return []

    // Apply search and capacity filters
    let filtered = projects

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(query) ||
        p.state?.toLowerCase().includes(query) ||
        p.type.toLowerCase().includes(query)
      )
    }

    if (minCapacity > 0) {
      filtered = filtered.filter(p => p.capacity_mw >= minCapacity)
    }

    if (filtered.length === 0) return []

    // For large datasets, sample first for performance
    if (filtered.length > 5000) {
      const sampled = sampleSites(
        filtered.map(p => ({
          ...p,
          latitude: p.location.lat,
          longitude: p.location.lon,
        })),
        500
      )

      return clusterSites(sampled, {
        gridSize: 2,
        maxSitesPerCluster: 100,
        minClusterSize: 3,
      })
    }

    // For moderate datasets, just cluster
    if (filtered.length > 200) {
      return clusterSites(
        filtered.map(p => ({
          ...p,
          latitude: p.location.lat,
          longitude: p.location.lon,
        })),
        {
          gridSize: 1.5,
          maxSitesPerCluster: 100,
          minClusterSize: 3,
        }
      )
    }

    // Small datasets show as-is
    return filtered.map(p => ({
      ...p,
      latitude: p.location.lat,
      longitude: p.location.lon,
    }))
  }, [projects, searchQuery, minCapacity])

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
              <span className="ml-2 text-xs text-gray-400">/ Explore</span>
            </div>
            <div className="flex space-x-8">
              <Link href="/" className="text-white/70 hover:text-white transition">Home</Link>
              <Link href="/explore" className="text-white transition">Explore</Link>
              <Link href="/horizon" className="text-white/70 hover:text-white transition">Horizon</Link>
              <Link href="/invest" className="text-white/70 hover:text-white transition">Invest</Link>
              <Link href="/api" className="text-white/70 hover:text-white transition">API</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-16 flex h-screen">
        {/* Left Panel - Filters & Stats */}
        <div className="w-80 bg-gray-950 border-r border-white/10 p-6 overflow-y-auto">
          <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            Interactive Explorer
          </h2>

          {/* Search */}
          <div className="mb-6">
            <div className="relative">
              <input
                type="text"
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-white/40 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/30"
              />
              <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>

          {/* Stats */}
          <div className="mb-8 p-4 bg-gradient-to-b from-emerald-950/20 to-transparent border border-emerald-500/20 rounded-lg">
            {stats.isFiltered && (
              <div className="text-xs text-cyan-400 mb-2 flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                Filtered Results
              </div>
            )}
            <div className="grid grid-cols-1 gap-3">
              <div>
                <div className="text-2xl font-bold text-emerald-400">{stats.total.toLocaleString()}</div>
                <div className="text-xs text-white/60">
                  Sites {stats.isFiltered && <span className="text-white/40">of {totalSites.toLocaleString()}</span>}
                </div>
                <div className="text-xs text-white/40 mt-0.5">
                  {stats.clusters.toLocaleString()} clusters
                </div>
              </div>
              <div>
                <div className="text-2xl font-bold text-cyan-400">{(stats.capacity / 1000).toFixed(1)} GW</div>
                <div className="text-xs text-white/60">Total Capacity</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-400">${(stats.investment / 1000000).toFixed(1)}M</div>
                <div className="text-xs text-white/60">Investment</div>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-white/80 mb-3">Filter by Type</h3>
            <div className="space-y-2">
              {[
                { value: 'all', label: 'All Projects', color: 'bg-white', shadow: 'shadow-[0_0_6px_rgba(255,255,255,0.4)]' },
                { value: 'solar', label: 'Solar', color: 'bg-amber-500', shadow: 'shadow-[0_0_6px_rgba(251,191,36,0.6)]' },
                { value: 'wind', label: 'Wind', color: 'bg-cyan-500', shadow: 'shadow-[0_0_6px_rgba(6,182,212,0.6)]' },
                { value: 'battery', label: 'Storage', color: 'bg-purple-500', shadow: 'shadow-[0_0_6px_rgba(168,85,247,0.6)]' },
                { value: 'hydro', label: 'Hydro', color: 'bg-blue-500', shadow: 'shadow-[0_0_6px_rgba(59,130,246,0.6)]' },
                { value: 'smr', label: 'Nuclear SMR', color: 'bg-purple-500', shadow: 'shadow-[0_0_6px_rgba(168,85,247,0.6)]' },
                { value: 'dams', label: 'Dam Retrofits', color: 'bg-blue-500', shadow: 'shadow-[0_0_6px_rgba(59,130,246,0.6)]' },
                { value: 'geothermal', label: 'Geothermal', color: 'bg-red-500', shadow: 'shadow-[0_0_6px_rgba(239,68,68,0.6)]' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => setFilter(option.value)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition ${
                    filter === option.value
                      ? 'bg-white/10 text-white'
                      : 'text-white/60 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <span className={`inline-block w-2.5 h-2.5 rounded-full ${option.color} ${option.shadow} mr-2`}></span>
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Capacity Filter */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-white/80 mb-3">Minimum Capacity</h3>
            <div className="space-y-3">
              <input
                type="range"
                min="0"
                max="500"
                step="10"
                value={minCapacity}
                onChange={(e) => setMinCapacity(Number(e.target.value))}
                className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-xs text-white/50">
                <span>0 MW</span>
                <span className="text-emerald-400 font-semibold">{minCapacity} MW+</span>
                <span>500 MW</span>
              </div>
            </div>
          </div>

          {/* Selected Project */}
          {selectedProject && (
            <div className="space-y-4">
              <div className="p-4 bg-gradient-to-b from-purple-950/20 to-transparent border border-purple-500/20 rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-bold text-purple-400">{selectedProject.name}</h3>
                  {selectedAnalysis && (
                    <AIScoreBadge
                      score={selectedAnalysis.overallScore}
                      viability={selectedAnalysis.viabilityRating as any}
                      showLabel={false}
                      size="sm"
                    />
                  )}
                </div>
                {(selectedProject as any).isCluster && (selectedProject as any).site_count && (
                  <div className="mb-2 px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-xs text-blue-400">
                    🔷 Cluster of {(selectedProject as any).site_count.toLocaleString()} sites
                  </div>
                )}
                <div className="space-y-1 text-sm">
                  <p className="text-white/60">Type: <span className="text-white capitalize">{selectedProject.type}</span></p>
                  <p className="text-white/60">State: <span className="text-white">{selectedProject.state}</span></p>
                  <p className="text-white/60">
                    {(selectedProject as any).isCluster ? 'Total Capacity:' : 'Capacity:'}
                    <span className="text-emerald-400"> {selectedProject.capacity_mw.toLocaleString()} MW</span>
                  </p>
                  {selectedProject.investment > 0 && (
                    <p className="text-white/60">Investment: <span className="text-cyan-400">${(selectedProject.investment / 1000000).toFixed(1)}M</span></p>
                  )}
                  <p className="text-white/60">Status: <span className="text-yellow-400">{selectedProject.status}</span></p>
                </div>
                <Link
                  href={(selectedProject as any).isCluster ? `/explore?region=${selectedProject.state}` : `/project/${selectedProject.id}`}
                  className="mt-4 w-full px-3 py-2 bg-purple-500/20 border border-purple-500/50 rounded-lg text-purple-400 hover:bg-purple-500/30 transition inline-block text-center"
                >
                  {(selectedProject as any).isCluster ? `Explore ${(selectedProject as any).site_count} Sites` : 'View Details'}
                </Link>
              </div>

              {/* AI Analysis Card */}
              {analysisLoading ? (
                <div className="p-4 bg-gradient-to-b from-emerald-950/20 to-transparent border border-emerald-500/20 rounded-lg">
                  <div className="animate-pulse">
                    <div className="h-4 w-24 bg-white/10 rounded mb-3"></div>
                    <div className="space-y-2">
                      <div className="h-3 w-full bg-white/5 rounded"></div>
                      <div className="h-3 w-3/4 bg-white/5 rounded"></div>
                      <div className="h-3 w-1/2 bg-white/5 rounded"></div>
                    </div>
                  </div>
                </div>
              ) : selectedAnalysis ? (
                <AIAnalysisCard analysis={selectedAnalysis} />
              ) : null}
            </div>
          )}
        </div>

        {/* Globe Container */}
        <div className="flex-1 relative bg-gradient-to-b from-gray-950 to-black">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-white/60">Loading projects...</div>
            </div>
          ) : (
            <>
              <TerraGlobe
                sites={displayProjects}
                onSiteSelect={handleSiteSelect}
                showLegend={false}
                showStats={false}
              />

              {/* Overlay Info */}
              <div className="absolute top-6 left-6 pointer-events-none">
                <h3 className="text-lg font-semibold text-white/80 mb-2">Interactive 3D Globe</h3>
                <p className="text-sm text-white/50">Click and drag to rotate • Scroll to zoom</p>
                <p className="text-sm text-white/50">Click markers to view details • Press ESC to deselect</p>
                {projects.length > displayProjects.length && (
                  <p className="text-xs text-emerald-400/70 mt-2">
                    Showing {displayProjects.length} clusters of {projects.length.toLocaleString()} sites
                  </p>
                )}
                {(searchQuery || minCapacity > 0) && (
                  <p className="text-xs text-cyan-400/70 mt-1">
                    Filtered: {displayProjects.length} of {projects.length} projects
                  </p>
                )}
              </div>

              {/* Clear Filters Button */}
              {(searchQuery || minCapacity > 0 || filter !== 'all') && (
                <div className="absolute top-6 right-6">
                  <button
                    onClick={() => {
                      setSearchQuery('')
                      setMinCapacity(0)
                      setFilter('all')
                    }}
                    className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 text-white/80 hover:bg-white/20 transition text-sm"
                  >
                    Clear Filters
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
