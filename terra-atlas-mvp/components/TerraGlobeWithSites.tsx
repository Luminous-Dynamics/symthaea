// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import logger from '@/lib/logger'
import { performanceMonitor } from '../utils/performance'
import {
  trackScorecardView,
  trackComparisonMode,
  trackTimelineMode,
  trackTimelineYear,
  trackTimelineAnimation
} from '@/lib/analytics'

interface Site {
  id: string
  name: string
  latitude: number
  longitude: number
  type?: string  // Energy type (geothermal, solar, wind, hydro, nuclear)
  estimated_capacity_mw?: number
  capacity_mw?: number  // Alternative field name
  estimated_irr?: number
  country?: string
  state?: string
  isCluster?: boolean
  site_count?: number  // For clusters
  avg_score?: number   // For clusters

  // ✨ Investment Scorecard Data
  scorecard?: {
    // Financial metrics
    totalInvestment?: number        // Total $ needed
    paybackPeriod?: number          // Years to break even
    capacityFactor?: number         // 0-1 (e.g., 0.35 = 35%)

    // Technical metrics
    gridConnection?: 'easy' | 'moderate' | 'difficult'
    permittingStatus?: 'approved' | 'pending' | 'not_started'

    // Environmental impact
    co2AvoidedPerYear?: number      // Tons CO2 avoided
    waterImpact?: 'positive' | 'neutral' | 'negative'

    // Risk assessment
    riskScore?: number              // 0-100 (lower is better)
    regulatoryRisk?: 'low' | 'medium' | 'high'
  }
}

interface TerraGlobeProps {
  sites?: Site[]  // Optional external sites
  onSiteSelect?: (site: Site) => void
  showLegend?: boolean  // Show color legend overlay
  showStats?: boolean   // Show site count badge
  minimal?: boolean     // Minimal mode for homepage hero
}

export default function TerraGlobeWithSites({
  sites: externalSites,
  onSiteSelect,
  showLegend = true,
  showStats = true,
  minimal = false
}: TerraGlobeProps = {}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [sites, setSites] = useState<Site[]>([])
  const [selectedSite, setSelectedSite] = useState<Site | null>(null)
  const [hoveredSite, setHoveredSite] = useState<Site | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const [loading, setLoading] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [currentZoomLevel, setCurrentZoomLevel] = useState<'world' | 'regional' | 'local'>('world')
  const [isTransitioning, setIsTransitioning] = useState(false)  // ✨ Fade transition state
  // ✨ Real-time statistics state
  const [viewportStats, setViewportStats] = useState({
    visibleSites: 0,
    totalCapacity: 0,
    states: new Set<string>(),
    typeBreakdown: {} as Record<string, number>,
    avgIRR: 0,
    investmentPotential: 0,
  })

  // ✨ Search & Filter state
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [selectedState, setSelectedState] = useState<string>('all')
  const [minCapacity, setMinCapacity] = useState<string>('')
  const [sortBy, setSortBy] = useState<'name' | 'capacity' | 'irr'>('capacity')
  const [showFilters, setShowFilters] = useState(false)

  // ✨ Regional Comparison state
  const [comparisonMode, setComparisonMode] = useState(false)
  const [selectedRegions, setSelectedRegions] = useState<string[]>([])
  const [regionalStats, setRegionalStats] = useState<Record<string, {
    totalSites: number
    totalCapacity: number
    avgIRR: number
    avgRiskScore: number
    totalInvestment: number
    co2Avoided: number
    typeBreakdown: Record<string, number>
  }>>({})

  // ✨ Time-Series Projections state
  const [timelineMode, setTimelineMode] = useState(false)
  const [timelineYear, setTimelineYear] = useState(2025)
  const [isAnimating, setIsAnimating] = useState(false)
  const [projections, setProjections] = useState<Record<number, {
    cumulativeCapacity: number
    cumulativeInvestment: number
    cumulativeJobs: number
    cumulativeCO2Avoided: number
    deployedSites: number
    activeProjects: number
  }>>({})

  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const animationIdRef = useRef<number | null>(null)
  const rotationVelocityRef = useRef({ x: 0, y: 0 })
  const markersRef = useRef<THREE.Mesh[]>([])
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const earthRef = useRef<THREE.Mesh | null>(null)
  // ✨ Camera flyover animation state
  const cameraAnimationRef = useRef<{
    isAnimating: boolean
    startPosition: THREE.Vector3
    targetPosition: THREE.Vector3
    startRotation: { x: number, y: number }
    targetRotation: { x: number, y: number }
    progress: number
    duration: number
  } | null>(null)

  // ✨ Generate investment scorecard data for a site
  const generateScorecard = (site: Site) => {
    // If scorecard already exists, return it
    if (site.scorecard) return site.scorecard

    // Generate estimated scorecard based on available data
    const capacity = site.estimated_capacity_mw || site.capacity_mw || 100
    const irr = site.estimated_irr || 10
    const type = site.type || 'unknown'

    // Calculate total investment ($2.5M per MW industry standard)
    const totalInvestment = capacity * 2500000

    // Estimate payback period from IRR (simplified: 100 / IRR)
    const paybackPeriod = Math.round(100 / irr)

    // Capacity factors by energy type (industry averages)
    const capacityFactors: Record<string, number> = {
      solar: 0.25,
      wind: 0.35,
      hydro: 0.52,
      nuclear: 0.93,
      geothermal: 0.85,
    }
    const capacityFactor = capacityFactors[type] || 0.40

    // Grid connection difficulty (hydro=easy, nuclear=difficult, others=moderate)
    const gridConnection: 'easy' | 'moderate' | 'difficult' =
      type === 'hydro' ? 'easy' :
      type === 'nuclear' ? 'difficult' :
      'moderate'

    // Permitting status (for demo: 70% approved, 20% pending, 10% not started)
    const rand = Math.random()
    const permittingStatus: 'approved' | 'pending' | 'not_started' =
      rand > 0.7 ? 'approved' :
      rand > 0.5 ? 'pending' :
      'not_started'

    // CO2 avoided calculation (tons per year)
    // Formula: capacity * capacity_factor * 8760 hours * 0.5 tons/MWh (coal offset)
    const co2AvoidedPerYear = Math.round(capacity * capacityFactor * 8760 * 0.5)

    // Water impact (hydro=positive, solar/wind=positive, nuclear=neutral)
    const waterImpact: 'positive' | 'neutral' | 'negative' =
      type === 'hydro' || type === 'solar' || type === 'wind' ? 'positive' :
      type === 'nuclear' ? 'neutral' :
      'positive'

    // Risk score (0-100, lower is better)
    // Based on: IRR (higher = lower risk), capacity factor, permitting
    const baseRisk = 100 - irr * 3 // IRR of 10% = 70 risk, 14% = 58 risk
    const permitRisk = permittingStatus === 'approved' ? -10 : permittingStatus === 'pending' ? 5 : 15
    const typeRisk = type === 'nuclear' ? 15 : type === 'solar' ? -10 : 0
    const riskScore = Math.max(0, Math.min(100, Math.round(baseRisk + permitRisk + typeRisk)))

    // Regulatory risk (nuclear=high, others=low/medium)
    const regulatoryRisk: 'low' | 'medium' | 'high' =
      type === 'nuclear' ? 'high' :
      type === 'hydro' ? 'medium' :
      'low'

    return {
      totalInvestment,
      paybackPeriod,
      capacityFactor,
      gridConnection,
      permittingStatus,
      co2AvoidedPerYear,
      waterImpact,
      riskScore,
      regulatoryRisk,
    }
  }

  useEffect(() => {
    const route = typeof window !== 'undefined' ? window.location.pathname : 'unknown'
    performanceMonitor.setContext({
      route,
      buildId: process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA || 'local',
    })
  }, [])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedSite(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Load energy sites from local data or use external sites
  useEffect(() => {
    // If external sites provided, use those
    if (externalSites && externalSites.length > 0) {
      // Normalize the data format
      const normalizedSites = externalSites.map(site => ({
        ...site,
        estimated_capacity_mw: site.estimated_capacity_mw || site.capacity_mw || 10,
        estimated_irr: site.estimated_irr || 10,
      }))
      logger.info(`✅ Using ${normalizedSites.length} external sites`)
      setSites(normalizedSites)
      return
    }

    // Otherwise load from API with dynamic zoom levels
    const loadSites = async (zoomLevel: string = 'world', filters?: {
      type?: string
      state?: string
      minCapacity?: string
    }) => {
      try {
        // Build query parameters
        const params = new URLSearchParams({ level: zoomLevel })
        if (filters?.type && filters.type !== 'all') params.append('type', filters.type)
        if (filters?.state && filters.state !== 'all') params.append('state', filters.state)
        if (filters?.minCapacity) params.append('minCapacity', filters.minCapacity)

        // Use API for dynamic multi-level clustering
        const response = await fetch(`/api/sites?${params.toString()}`)
        const data = await response.json()

        if (data.success && data.sites) {
          logger.info(`✅ Loaded ${data.sites.length} ${zoomLevel} clusters (${data.metadata.totalSites.toLocaleString()} total sites)`)
          setSites(data.sites)
          setCurrentZoomLevel(zoomLevel as any)
        } else {
          throw new Error('Invalid API response')
        }
      } catch (err) {
        logger.error('❌ Failed to load from API, using fallback:', err)
        // Fallback to static clustered data
        try {
          const response = await fetch('/data/sites-clustered.json')
          const data = await response.json()
          const sites = data.clusters.map((cluster: any) => ({
            id: cluster.id,
            name: cluster.name,
            latitude: cluster.latitude,
            longitude: cluster.longitude,
            type: cluster.type,
            estimated_capacity_mw: cluster.total_capacity_mw,
            estimated_irr: cluster.avg_irr,
            country: 'USA',
            state: cluster.states?.[0] || '',
            isCluster: cluster.site_count > 1,
            site_count: cluster.site_count,
            avg_score: cluster.avg_score
          }))
          logger.info(`✅ Fallback: Loaded ${sites.length} clusters`)
          setSites(sites)
        } catch {
          logger.error('❌ No site data available')
        }
      }
    }

    loadSites('world')
  }, [externalSites])

  // ✨ Calculate real-time viewport statistics
  useEffect(() => {
    if (sites.length === 0) return

    // Calculate stats from all currently visible sites
    const states = new Set<string>()
    const typeBreakdown: Record<string, number> = {}
    let totalCapacity = 0
    let totalIRR = 0
    let visibleSiteCount = 0

    sites.forEach(site => {
      // Count individual sites (accounting for clusters)
      const siteCount = site.site_count || 1
      visibleSiteCount += siteCount

      // Accumulate capacity
      const capacity = site.estimated_capacity_mw || 0
      totalCapacity += capacity

      // Accumulate IRR for averaging
      const irr = site.estimated_irr || 0
      totalIRR += irr * siteCount // Weight by site count

      // Track states
      if (site.state) states.add(site.state)

      // Track energy types
      const type = site.type || 'unknown'
      typeBreakdown[type] = (typeBreakdown[type] || 0) + capacity
    })

    const avgIRR = sites.length > 0 ? totalIRR / visibleSiteCount : 0
    // Rough investment potential estimate: $2.5M per MW
    const investmentPotential = totalCapacity * 2500000

    setViewportStats({
      visibleSites: visibleSiteCount,
      totalCapacity,
      states,
      typeBreakdown,
      avgIRR,
      investmentPotential,
    })

    logger.info(`📊 Stats updated: ${visibleSiteCount.toLocaleString()} sites, ${(totalCapacity/1000).toFixed(1)} GW capacity`)
  }, [sites])

  // ✨ Calculate regional statistics for comparison
  useEffect(() => {
    if (sites.length === 0) return

    const statsByState: Record<string, {
      totalSites: number
      totalCapacity: number
      avgIRR: number
      avgRiskScore: number
      totalInvestment: number
      co2Avoided: number
      typeBreakdown: Record<string, number>
    }> = {}

    sites.forEach(site => {
      const state = site.state || 'Unknown'
      if (!statsByState[state]) {
        statsByState[state] = {
          totalSites: 0,
          totalCapacity: 0,
          avgIRR: 0,
          avgRiskScore: 0,
          totalInvestment: 0,
          co2Avoided: 0,
          typeBreakdown: {}
        }
      }

      const siteCount = site.site_count || 1
      const capacity = site.estimated_capacity_mw || 0
      const irr = site.estimated_irr || 0
      const type = site.type || 'unknown'

      // Aggregate metrics
      statsByState[state].totalSites += siteCount
      statsByState[state].totalCapacity += capacity
      statsByState[state].avgIRR += irr * siteCount // Will divide later

      // Generate scorecard to get risk score and CO2
      const scorecard = generateScorecard(site)
      statsByState[state].avgRiskScore += scorecard.riskScore * siteCount
      statsByState[state].totalInvestment += scorecard.totalInvestment
      statsByState[state].co2Avoided += scorecard.co2AvoidedPerYear

      // Type breakdown
      statsByState[state].typeBreakdown[type] = (statsByState[state].typeBreakdown[type] || 0) + capacity
    })

    // Calculate averages
    Object.keys(statsByState).forEach(state => {
      const stats = statsByState[state]
      stats.avgIRR = stats.avgIRR / stats.totalSites
      stats.avgRiskScore = stats.avgRiskScore / stats.totalSites
    })

    setRegionalStats(statsByState)
    logger.info(`📍 Regional stats calculated for ${Object.keys(statsByState).length} states`)
  }, [sites])

  // ✨ Calculate time-series projections for deployment timeline
  useEffect(() => {
    if (sites.length === 0) return

    // Define deployment schedule: 2025-2035 (10 years)
    // Assume gradual rollout with ~10% deployment per year
    const startYear = 2025
    const endYear = 2035
    const yearsToProject = endYear - startYear + 1

    const projectionsByYear: Record<number, {
      cumulativeCapacity: number
      cumulativeInvestment: number
      cumulativeJobs: number
      cumulativeCO2Avoided: number
      deployedSites: number
      activeProjects: number
    }> = {}

    // Calculate total potential from all sites
    let totalPotentialCapacity = 0
    let totalPotentialInvestment = 0
    let totalPotentialJobs = 0
    let totalPotentialCO2 = 0
    let totalSites = 0

    sites.forEach(site => {
      const siteCount = site.site_count || 1
      const capacity = site.estimated_capacity_mw || 0
      const scorecard = generateScorecard(site)

      totalSites += siteCount
      totalPotentialCapacity += capacity
      totalPotentialInvestment += scorecard.totalInvestment
      totalPotentialJobs += capacity * 10 // ~10 jobs per MW (industry standard)
      totalPotentialCO2 += scorecard.co2AvoidedPerYear
    })

    // Generate projections for each year (S-curve deployment)
    for (let year = startYear; year <= endYear; year++) {
      const yearIndex = year - startYear
      const progress = yearIndex / (yearsToProject - 1) // 0 to 1

      // S-curve deployment (logistic function for realistic ramp-up/plateau)
      // Fast initial growth, then plateau: f(x) = 1 / (1 + e^(-k(x-x0)))
      const k = 0.8 // Steepness factor
      const x0 = 0.5 // Midpoint (50% deployment at year 5.5)
      const deploymentFactor = 1 / (1 + Math.exp(-k * (progress - x0)))

      // Calculate cumulative metrics based on deployment factor
      projectionsByYear[year] = {
        cumulativeCapacity: Math.round(totalPotentialCapacity * deploymentFactor),
        cumulativeInvestment: Math.round(totalPotentialInvestment * deploymentFactor),
        cumulativeJobs: Math.round(totalPotentialJobs * deploymentFactor),
        cumulativeCO2Avoided: Math.round(totalPotentialCO2 * deploymentFactor * (year - startYear)), // Cumulative CO2
        deployedSites: Math.round(totalSites * deploymentFactor),
        activeProjects: Math.round(totalSites * 0.15) // Assume ~15% of sites in active development
      }
    }

    setProjections(projectionsByYear)
    logger.info(`📅 Time-series projections calculated for ${yearsToProject} years (${startYear}-${endYear})`)
  }, [sites])

  // ✨ Apply filters when they change
  useEffect(() => {
    if (externalSites && externalSites.length > 0) return // Don't reload if using external sites

    const filters = {
      type: selectedType,
      state: selectedState,
      minCapacity: minCapacity
    }

    // Reload with filters
    const loadFilteredSites = async () => {
      setIsTransitioning(true)
      try {
        // Build query parameters
        const params = new URLSearchParams({ level: currentZoomLevel })
        if (selectedType !== 'all') params.append('type', selectedType)
        if (selectedState !== 'all') params.append('state', selectedState)
        if (minCapacity) params.append('minCapacity', minCapacity)

        const response = await fetch(`/api/sites?${params.toString()}`)
        const data = await response.json()

        if (data.success && data.sites) {
          // Apply client-side search filter
          let filteredSites = data.sites
          if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase()
            filteredSites = data.sites.filter((site: Site) =>
              site.name.toLowerCase().includes(query) ||
              site.state?.toLowerCase().includes(query) ||
              site.type?.toLowerCase().includes(query)
            )
          }

          // Apply sorting
          filteredSites.sort((a: Site, b: Site) => {
            if (sortBy === 'name') return a.name.localeCompare(b.name)
            if (sortBy === 'capacity') return (b.estimated_capacity_mw || 0) - (a.estimated_capacity_mw || 0)
            if (sortBy === 'irr') return (b.estimated_irr || 0) - (a.estimated_irr || 0)
            return 0
          })

          logger.info(`🔍 Filtered to ${filteredSites.length} sites (from ${data.sites.length})`)
          setSites(filteredSites)
        }
      } catch (err) {
        logger.error('❌ Failed to apply filters:', err)
      } finally {
        setTimeout(() => setIsTransitioning(false), 600)
      }
    }

    loadFilteredSites()
  }, [selectedType, selectedState, minCapacity, searchQuery, sortBy, currentZoomLevel, externalSites])

  useEffect(() => {
    if (!containerRef.current) return

    // 📊 START PERFORMANCE TRACKING
    performanceMonitor.markGlobeMountStart()

    const initTimer = setTimeout(() => {
      try {
        const width = containerRef.current?.clientWidth || 800
        const height = containerRef.current?.clientHeight || 600

        logger.info('🌍 Creating Terra Atlas globe with real site data')
        performanceMonitor.markThreeJsInitStart()

        // Scene with very dark space background
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x000408)

        // Camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
        camera.position.z = 4.2
        camera.position.y = 0.0
        cameraRef.current = camera

        // Renderer
        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: false
        })
        renderer.setSize(width, height)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

        if (containerRef.current) {
          containerRef.current.appendChild(renderer.domElement)
          rendererRef.current = renderer
        }

        // 📊 Three.js initialization complete
        performanceMonitor.markThreeJsInitEnd()

        // Star field (abbreviated for brevity - same as before)
        const starLayers = []
        const stars1Geometry = new THREE.BufferGeometry()
        const stars1Vertices = []
        const stars1Colors = []
        const stars1Sizes = []

        for (let i = 0; i < 800; i++) {
          const x = (Math.random() - 0.5) * 2000
          const y = (Math.random() - 0.5) * 2000
          const z = (Math.random() - 0.5) * 2000
          stars1Vertices.push(x, y, z)

          const colorVariation = Math.random()
          stars1Colors.push(0.8 + colorVariation * 0.2, 0.85 + colorVariation * 0.15, 1.0)
          stars1Sizes.push(0.8 + Math.random() * 0.5)
        }

        stars1Geometry.setAttribute('position', new THREE.Float32BufferAttribute(stars1Vertices, 3))
        stars1Geometry.setAttribute('color', new THREE.Float32BufferAttribute(stars1Colors, 3))
        stars1Geometry.setAttribute('size', new THREE.Float32BufferAttribute(stars1Sizes, 1))

        const stars1Material = new THREE.PointsMaterial({
          size: 1.2,
          sizeAttenuation: true,
          vertexColors: true,
          transparent: true,
          opacity: 0.7
        })

        const stars1 = new THREE.Points(stars1Geometry, stars1Material)
        scene.add(stars1)
        starLayers.push({ mesh: stars1, speed: 0.00002 })

        // Earth sphere
        const geometry = new THREE.SphereGeometry(1, 128, 128)

        // Progressive texture loading
        const textureLoader = new THREE.TextureLoader()
        let lowResLoaded = 0
        let highResLoaded = 0
        const totalTextures = 2

        const updateProgress = (stage: 'low' | 'high') => {
          if (stage === 'low') {
            lowResLoaded++
            const progress = (lowResLoaded / totalTextures) * 50
            setLoadingProgress(progress)
            if (lowResLoaded === totalTextures) {
              setTimeout(() => setLoading(false), 200)
            }
          } else {
            highResLoaded++
            const progress = 50 + (highResLoaded / totalTextures) * 50
            setLoadingProgress(progress)
          }
        }

        const earthTextureLow = textureLoader.load(
          '/textures/earth-blue-marble-low.webp',
          () => updateProgress('low'),
          undefined,
          () => updateProgress('low')
        )

        const boundariesTextureLow = textureLoader.load(
          '/textures/earth-topology-low.webp',
          () => updateProgress('low'),
          undefined,
          () => updateProgress('low')
        )

        let earthTexture = earthTextureLow
        let boundariesTexture = boundariesTextureLow

        // Earth shader material - DRAMATICALLY IMPROVED CONTRAST
        const earthMaterial = new THREE.ShaderMaterial({
          vertexShader: `
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vViewPosition;

            void main() {
              vNormal = normalize(normalMatrix * normal);
              vUv = uv;
              vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
              vViewPosition = -mvPosition.xyz;
              gl_Position = projectionMatrix * mvPosition;
            }
          `,
          fragmentShader: `
            uniform float time;
            uniform sampler2D earthTexture;
            uniform sampler2D boundariesTexture;
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vViewPosition;

            void main() {
              vec4 texColor = texture2D(earthTexture, vUv);

              float brightness = (texColor.r + texColor.g + texColor.b) / 3.0;
              float blueDominance = texColor.b - ((texColor.r + texColor.g) * 0.5);

              float isWater = smoothstep(0.05, 0.15, blueDominance);
              float isDark = smoothstep(0.20, 0.10, brightness);
              float isOcean = max(isWater, isDark);
              float isLand = 1.0 - isOcean;

              // HIGH CONTRAST COLORS - Much more visible!
              vec3 landColor = vec3(0.08, 0.18, 0.12);   // Bright emerald green
              vec3 oceanColor = vec3(0.02, 0.06, 0.15);  // Deep navy blue

              // Add subtle variation based on texture
              landColor += texColor.rgb * 0.08;

              vec3 baseColor = mix(oceanColor, landColor, isLand);

              // SUBTLE grid lines - not dominant
              float gridSpacing = 30.0;
              float latLines = fract(vUv.y * 180.0 / gridSpacing);
              float lonLines = fract(vUv.x * 360.0 / gridSpacing);

              float latGrid = smoothstep(0.990, 1.0, latLines) + smoothstep(0.010, 0.0, latLines);
              float lonGrid = smoothstep(0.990, 1.0, lonLines) + smoothstep(0.010, 0.0, lonLines);
              float techGrid = max(latGrid, lonGrid);

              // Very subtle cyan grid - decorative only
              vec3 gridColor = vec3(0.10, 0.25, 0.30);
              baseColor = mix(baseColor, gridColor, techGrid * 0.15);

              // Add coastline glow for land edges
              float coastlineWidth = 0.08;
              float coastGlow = smoothstep(coastlineWidth, 0.0, abs(isLand - 0.5) * 2.0);
              vec3 coastColor = vec3(0.15, 0.35, 0.40);
              baseColor = mix(baseColor, coastColor, coastGlow * 0.4);

              // Enhanced rim lighting with glow
              vec3 viewDir = normalize(vViewPosition);
              float rimPower = 2.5;
              float rim = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
              float rimIntensity = pow(rim, rimPower);
              vec3 rimColor = vec3(0.08, 0.22, 0.30) * rimIntensity * 0.6;

              // Surface illumination
              float surfaceGlow = pow(1.0 - rim, 3.0) * 0.1;
              vec3 glowColor = vec3(0.04, 0.12, 0.16) * surfaceGlow;

              vec3 finalColor = baseColor + rimColor + glowColor;

              gl_FragColor = vec4(finalColor, 1.0);
            }
          `,
          uniforms: {
            time: { value: 0 },
            earthTexture: { value: earthTexture },
            boundariesTexture: { value: boundariesTexture }
          }
        })

        const earth = new THREE.Mesh(geometry, earthMaterial)
        earth.rotation.y = -0.5
        scene.add(earth)
        earthRef.current = earth

        // Load high-res textures
        textureLoader.load('/textures/earth-blue-marble.webp', (texture) => {
          updateProgress('high')
          earthMaterial.uniforms.earthTexture.value = texture
          earthMaterial.needsUpdate = true
        })

        textureLoader.load('/textures/earth-topology.webp', (texture) => {
          updateProgress('high')
          earthMaterial.uniforms.boundariesTexture.value = texture
          earthMaterial.needsUpdate = true
        })

        // REFINED atmosphere - subtle and elegant
        const atmosphereGeometry = new THREE.SphereGeometry(1.08, 64, 64)
        const atmosphereMaterial = new THREE.ShaderMaterial({
          vertexShader: `
            varying vec3 vNormal;
            void main() {
              vNormal = normalize(normalMatrix * normal);
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            uniform float time;
            varying vec3 vNormal;

            void main() {
              float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.5);

              // Subtle blue atmosphere glow
              vec3 atmosphereColor = vec3(0.06, 0.18, 0.32);
              float alpha = intensity * 0.45;

              gl_FragColor = vec4(atmosphereColor, alpha);
            }
          `,
          uniforms: { time: { value: 0 } },
          blending: THREE.AdditiveBlending,
          side: THREE.BackSide,
          transparent: true
        })

        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial)
        scene.add(atmosphere)

        // Outer atmosphere halo - very subtle
        const atmosphere2Geometry = new THREE.SphereGeometry(1.12, 64, 64)
        const atmosphere2Material = new THREE.ShaderMaterial({
          vertexShader: `
            varying vec3 vNormal;
            void main() {
              vNormal = normalize(normalMatrix * normal);
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            uniform float time;
            varying vec3 vNormal;

            void main() {
              float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.5);

              vec3 outerGlow = vec3(0.04, 0.12, 0.22);
              float alpha = intensity * 0.25;

              gl_FragColor = vec4(outerGlow, alpha);
            }
          `,
          uniforms: { time: { value: 0 } },
          blending: THREE.AdditiveBlending,
          side: THREE.BackSide,
          transparent: true
        })

        const atmosphere2 = new THREE.Mesh(atmosphere2Geometry, atmosphere2Material)
        scene.add(atmosphere2)

        // Enhanced dramatic lighting
        const ambientLight = new THREE.AmbientLight(0x778899, 0.7)
        const directionalLight = new THREE.DirectionalLight(0xaaccee, 0.8)
        directionalLight.position.set(5, 3, 5)

        // Add rim light from opposite side
        const rimLight = new THREE.DirectionalLight(0x66ddff, 0.3)
        rimLight.position.set(-5, 2, -3)

        scene.add(ambientLight, directionalLight, rimLight)

        // Energy-type color mapping for visual storytelling
        const getEnergyColor = (type: string | undefined): number => {
          const normalizedType = type?.toLowerCase() || ''

          if (normalizedType.includes('solar')) {
            return 0xfbbf24 // Amber/Gold
          } else if (normalizedType.includes('wind')) {
            return 0x06b6d4 // Cyan/Blue
          } else if (normalizedType.includes('hydro') || normalizedType.includes('dam')) {
            return 0x3b82f6 // Deep Blue
          } else if (normalizedType.includes('nuclear') || normalizedType.includes('smr')) {
            return 0xa855f7 // Purple
          } else if (normalizedType.includes('geothermal')) {
            return 0xef4444 // Red/Orange
          } else if (normalizedType.includes('battery') || normalizedType.includes('storage')) {
            return 0xa855f7 // Purple
          } else {
            return 0x9ca3af // Gray default
          }
        }

        // 🎯 ADD SITE MARKERS - CLEAN FLAT DOT DESIGN (No additive blending!)
        const addSiteMarkers = (sitesData: Site[]) => {
          performanceMonitor.markMarkersRenderStart()
          logger.info(`📍 Adding ${sitesData.length} site markers to globe`)

          sitesData.forEach(site => {
            // Convert lat/lon to 3D coordinates
            const phi = (90 - site.latitude) * (Math.PI / 180)
            const theta = (site.longitude + 180) * (Math.PI / 180)

            const radius = 1.012 // Slightly above Earth surface
            const x = -(radius * Math.sin(phi) * Math.cos(theta))
            const z = radius * Math.sin(phi) * Math.sin(theta)
            const y = radius * Math.cos(phi)

            // Color code by energy type for visual storytelling
            const colorHex = getEnergyColor(site.type)
            const color = new THREE.Color(colorHex)

            // Capacity-based sizing - visible but refined
            const capacity = site.estimated_capacity_mw || site.capacity_mw || 10
            const baseSize = Math.log(capacity + 1) * 0.0018
            const size = Math.min(Math.max(baseSize, 0.006), 0.022)

            // PREMIUM MARKER - Bright core with glowing ring outline
            const markerGeometry = new THREE.CircleGeometry(size, 24)
            const markerMaterial = new THREE.ShaderMaterial({
              uniforms: {
                time: { value: 0 },
                color: { value: color }
              },
              vertexShader: `
                varying vec2 vUv;
                void main() {
                  vUv = uv;
                  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
              `,
              fragmentShader: `
                uniform float time;
                uniform vec3 color;
                varying vec2 vUv;

                void main() {
                  vec2 center = vUv - vec2(0.5);
                  float dist = length(center) * 2.0;

                  // Subtle breathing pulse
                  float pulse = sin(time * 1.2) * 0.08 + 0.92;

                  // Bright filled core (inner 60%)
                  float core = 1.0 - smoothstep(0.55, 0.65, dist);

                  // Glowing ring outline (65% to 90%)
                  float ringInner = smoothstep(0.60, 0.70, dist);
                  float ringOuter = 1.0 - smoothstep(0.85, 1.0, dist);
                  float ring = ringInner * ringOuter;

                  // Outer glow halo (very subtle, no overlap issues)
                  float halo = (1.0 - smoothstep(0.85, 1.0, dist)) * 0.3;

                  // Combine: bright core + luminous ring
                  vec3 coreColor = color * 1.3 * pulse;
                  vec3 ringColor = color * 1.8 * pulse; // Ring brighter than core
                  vec3 haloColor = color * 0.5;

                  vec3 finalColor = coreColor * core + ringColor * ring + haloColor * halo;

                  // Alpha: solid core, visible ring, subtle halo
                  float alpha = max(core * 0.95, ring * 0.9);
                  alpha = max(alpha, halo * 0.4);

                  if (alpha < 0.01) discard;
                  gl_FragColor = vec4(finalColor, alpha);
                }
              `,
              transparent: true,
              blending: THREE.NormalBlending,
              depthWrite: true,
              side: THREE.DoubleSide
            })

            const marker = new THREE.Mesh(markerGeometry, markerMaterial)
            marker.position.set(x, y, z)

            // Orient the circle to face outward from globe center
            marker.lookAt(0, 0, 0)
            marker.rotateY(Math.PI) // Flip to face camera

            // Staggered pop-in animation
            const distance = Math.sqrt(x * x + y * y + z * z)
            const animDelay = (sitesData.indexOf(site) % 50) * 0.02 // Stagger by index
            marker.scale.setScalar(0.01) // Start tiny
            marker.userData.animationProgress = 0
            marker.userData.animationDelay = animDelay

            // Store site data and animation uniforms
            marker.userData = {
              ...marker.userData,
              site,
              material: markerMaterial,
              pulseOffset: Math.random() * Math.PI * 2,
              targetScale: 1.0,
              fadeOpacity: 0.0,  // ✨ Start invisible, will fade in
              isFadingOut: false,
              isFadingIn: true,  // ✨ Trigger fade-in on creation
            }

            earth.add(marker)
            markersRef.current.push(marker)
          })

          logger.info('✅ Site markers added to globe')

          // 🌐 ADD ENERGY NETWORK CONNECTIONS
          // Connect nearby sites to show interconnected energy grid
          const addEnergyConnections = (sitesData: Site[]) => {
            logger.info('🔗 Creating energy network connections...')

            const connectionDistance = 0.4 // Max distance for connections (sphere radius = 1)
            const maxConnectionsPerSite = 3 // Limit connections to avoid clutter
            const connections: Array<{start: THREE.Vector3, end: THREE.Vector3, color: THREE.Color}> = []

            sitesData.forEach((site, i) => {
              const sitePhi = (90 - site.latitude) * (Math.PI / 180)
              const siteTheta = (site.longitude + 180) * (Math.PI / 180)
              const siteRadius = 1.02

              const sitePos = new THREE.Vector3(
                -(siteRadius * Math.sin(sitePhi) * Math.cos(siteTheta)),
                siteRadius * Math.cos(sitePhi),
                siteRadius * Math.sin(sitePhi) * Math.sin(siteTheta)
              )

              // Find nearby sites (only look ahead to avoid duplicates)
              const nearbyConnections: Array<{distance: number, targetPos: THREE.Vector3, color: THREE.Color}> = []

              for (let j = i + 1; j < sitesData.length; j++) {
                const targetSite = sitesData[j]
                const targetPhi = (90 - targetSite.latitude) * (Math.PI / 180)
                const targetTheta = (targetSite.longitude + 180) * (Math.PI / 180)

                const targetPos = new THREE.Vector3(
                  -(siteRadius * Math.sin(targetPhi) * Math.cos(targetTheta)),
                  siteRadius * Math.cos(targetPhi),
                  siteRadius * Math.sin(targetPhi) * Math.sin(targetTheta)
                )

                const distance = sitePos.distanceTo(targetPos)

                if (distance < connectionDistance) {
                  // Blend colors of connected sites for visual harmony
                  const color1 = new THREE.Color(getEnergyColor(site.type))
                  const color2 = new THREE.Color(getEnergyColor(targetSite.type))
                  const blendedColor = new THREE.Color().copy(color1).lerp(color2, 0.5)

                  nearbyConnections.push({ distance, targetPos, color: blendedColor })
                }
              }

              // Sort by distance and take only closest connections
              nearbyConnections
                .sort((a, b) => a.distance - b.distance)
                .slice(0, maxConnectionsPerSite)
                .forEach(conn => {
                  connections.push({
                    start: sitePos.clone(),
                    end: conn.targetPos.clone(),
                    color: conn.color
                  })
                })
            })

            logger.info(`  → Creating ${connections.length} energy connections`)

            // Create flowing connection lines with shader material
            connections.forEach((conn, idx) => {
              // Create curved line using quadratic bezier (arc over globe surface)
              const midPoint = new THREE.Vector3()
                .addVectors(conn.start, conn.end)
                .multiplyScalar(0.5)
                .normalize()
                .multiplyScalar(1.08) // Lift above surface

              const curve = new THREE.QuadraticBezierCurve3(
                conn.start,
                midPoint,
                conn.end
              )

              const points = curve.getPoints(20) // Smooth curve
              const geometry = new THREE.BufferGeometry().setFromPoints(points)

              // Create flowing energy line shader
              const material = new THREE.ShaderMaterial({
                uniforms: {
                  time: { value: 0 },
                  color: { value: conn.color },
                  flowOffset: { value: Math.random() * Math.PI * 2 } // Random flow phase
                },
                vertexShader: `
                  varying float vProgress;
                  void main() {
                    vProgress = position.z; // Use as flow position
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                  }
                `,
                fragmentShader: `
                  uniform float time;
                  uniform vec3 color;
                  uniform float flowOffset;
                  varying float vProgress;

                  void main() {
                    // Flowing energy pulse along the line - faster and brighter
                    float flow = sin(vProgress * 4.0 - time * 3.0 + flowOffset) * 0.5 + 0.5;

                    // Fade at ends
                    float fadeStart = smoothstep(0.0, 0.15, vProgress);
                    float fadeEnd = smoothstep(1.0, 0.85, vProgress);
                    float fade = fadeStart * fadeEnd;

                    // Brighter, more visible connections
                    vec3 finalColor = color * (0.6 + flow * 0.8);
                    float alpha = 0.35 * fade * (0.5 + flow * 0.5);

                    gl_FragColor = vec4(finalColor, alpha);
                  }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false,
                linewidth: 1
              })

              const line = new THREE.Line(geometry, material)
              line.userData = {
                material: material,
                flowOffset: material.uniforms.flowOffset.value,
                isEnergyConnection: true
              }
              earth.add(line)
            })

            logger.info('✅ Energy network connections created')
          }

          // Add connections after markers
          if (sitesData.length > 10) { // Only add if we have enough sites
            addEnergyConnections(sitesData)
          }

          // ✨ ADD ENERGY PARTICLES FOR HIGH-CAPACITY SITES
          // Subtle particles rising to show energy generation
          const addEnergyParticles = (sitesData: Site[]) => {
            logger.info('✨ Creating energy generation particles...')

            const particleSystems: Array<{
              particles: THREE.Points
              material: THREE.ShaderMaterial
              sitePos: THREE.Vector3
              color: THREE.Color
            }> = []

            // Only add particles for high-capacity sites (>200 MW)
            const highCapacitySites = sitesData.filter(site => site.estimated_capacity_mw > 200)
            logger.info(`  → Found ${highCapacitySites.length} high-capacity sites for particles`)

            highCapacitySites.forEach(site => {
              const phi = (90 - site.latitude) * (Math.PI / 180)
              const theta = (site.longitude + 180) * (Math.PI / 180)
              const radius = 1.02

              const sitePos = new THREE.Vector3(
                -(radius * Math.sin(phi) * Math.cos(theta)),
                radius * Math.cos(phi),
                radius * Math.sin(phi) * Math.sin(theta)
              )

              const color = new THREE.Color(getEnergyColor(site.type))

              // Create particle geometry
              const particleCount = 8 // Subtle, not overwhelming
              const positions = new Float32Array(particleCount * 3)
              const lifetimes = new Float32Array(particleCount) // Random start times

              for (let i = 0; i < particleCount; i++) {
                // Start at site position
                positions[i * 3] = sitePos.x
                positions[i * 3 + 1] = sitePos.y
                positions[i * 3 + 2] = sitePos.z
                // Random lifetime offset for continuous flow
                lifetimes[i] = Math.random() * Math.PI * 2
              }

              const geometry = new THREE.BufferGeometry()
              geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
              geometry.setAttribute('lifetime', new THREE.BufferAttribute(lifetimes, 1))

              // Custom particle shader
              const material = new THREE.ShaderMaterial({
                uniforms: {
                  time: { value: 0 },
                  color: { value: color },
                  sitePos: { value: sitePos },
                  siteNormal: { value: sitePos.clone().normalize() }
                },
                vertexShader: `
                  uniform float time;
                  uniform vec3 sitePos;
                  uniform vec3 siteNormal;

                  attribute float lifetime;

                  varying float vAlpha;
                  varying float vProgress;

                  void main() {
                    // Particle rises over time with lifetime offset
                    float t = mod(time * 0.3 + lifetime, 6.28); // ~2π = full cycle
                    float progress = t / 6.28;

                    // Rise along surface normal (upward from site)
                    float riseHeight = progress * 0.15; // Rise 0.15 units
                    vec3 particlePos = sitePos + siteNormal * riseHeight;

                    // Slight random drift outward
                    float drift = sin(lifetime * 3.0) * 0.02 * progress;
                    particlePos += siteNormal * drift;

                    // Fade in at start, fade out at end
                    float fadeIn = smoothstep(0.0, 0.1, progress);
                    float fadeOut = smoothstep(1.0, 0.7, progress);
                    vAlpha = fadeIn * fadeOut;
                    vProgress = progress;

                    vec4 mvPosition = modelViewMatrix * vec4(particlePos, 1.0);
                    gl_Position = projectionMatrix * mvPosition;

                    // Particle size based on distance
                    gl_PointSize = 2.5 * (300.0 / -mvPosition.z);
                  }
                `,
                fragmentShader: `
                  uniform vec3 color;
                  varying float vAlpha;
                  varying float vProgress;

                  void main() {
                    // Circular particle shape
                    vec2 center = gl_PointCoord - vec2(0.5);
                    float dist = length(center);
                    if (dist > 0.5) discard;

                    // Soft edge
                    float softEdge = 1.0 - smoothstep(0.3, 0.5, dist);

                    // Pulse brightness as it rises
                    float pulse = 0.6 + sin(vProgress * 6.28 * 2.0) * 0.4;

                    vec3 finalColor = color * pulse;
                    float alpha = vAlpha * softEdge * 0.4; // Subtle opacity

                    gl_FragColor = vec4(finalColor, alpha);
                  }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false
              })

              const particles = new THREE.Points(geometry, material)
              earth.add(particles)

              particleSystems.push({
                particles,
                material,
                sitePos,
                color
              })
            })

            logger.info('✅ Energy particles created')

            // Store particle systems for animation
            earth.userData.particleSystems = particleSystems
          }

          // Add particles after connections
          if (sitesData.length > 10) { // Only if we have enough sites
            addEnergyParticles(sitesData)
          }

          performanceMonitor.markMarkersRenderEnd()

          // 📊 Generate performance report after markers are rendered
          performanceMonitor.markGlobeMountEnd()
          setTimeout(() => {
            performanceMonitor.logReport()
          }, 100)
        }

        // Add markers when sites are loaded
        if (sites.length > 0) {
          addSiteMarkers(sites)
        }

        // Mouse interaction with raycasting for site selection
        let isDragging = false
        let previousMousePosition = { x: 0, y: 0 }
        let mouseDownPosition = { x: 0, y: 0 }
        const raycaster = new THREE.Raycaster()
        const mouse = new THREE.Vector2()
        let hoveredMarker: THREE.Mesh | null = null

        const handleMouseDown = (e: MouseEvent) => {
          isDragging = true
          previousMousePosition = { x: e.clientX, y: e.clientY }
          mouseDownPosition = { x: e.clientX, y: e.clientY }
          if (containerRef.current) {
            containerRef.current.style.cursor = 'grabbing'
          }
        }

        const handleMouseMove = (e: MouseEvent) => {
          if (!containerRef.current) return

          const rect = containerRef.current.getBoundingClientRect()
          mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
          mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

          // Check for hover on markers
          raycaster.setFromCamera(mouse, camera)
          const intersects = raycaster.intersectObjects(markersRef.current)

          if (intersects.length > 0) {
            const marker = intersects[0].object as THREE.Mesh
            if (hoveredMarker !== marker) {
              // Reset previous hover
              if (hoveredMarker && hoveredMarker.userData.material) {
                hoveredMarker.scale.setScalar(1)
              }
              // Apply hover effect
              hoveredMarker = marker
              marker.scale.setScalar(1.3)
              if (containerRef.current) {
                containerRef.current.style.cursor = 'pointer'
              }
              // Show tooltip with site name
              const site = marker.userData.site
              if (site) {
                setHoveredSite(site)
              }
            }
            // Update tooltip position
            setTooltipPos({ x: e.clientX, y: e.clientY })
          } else {
            // Reset hover
            if (hoveredMarker) {
              hoveredMarker.scale.setScalar(1)
              hoveredMarker = null
              setHoveredSite(null)
            }
            if (containerRef.current && !isDragging) {
              containerRef.current.style.cursor = 'grab'
            }
          }

          if (isDragging) {
            const deltaX = e.clientX - previousMousePosition.x
            const deltaY = e.clientY - previousMousePosition.y

            rotationVelocityRef.current.x += deltaY * 0.002
            rotationVelocityRef.current.y += deltaX * 0.002

            previousMousePosition = { x: e.clientX, y: e.clientY }
          }
        }

        const handleMouseUp = (e: MouseEvent) => {
          // Check if it was a click (not a drag)
          const dx = e.clientX - mouseDownPosition.x
          const dy = e.clientY - mouseDownPosition.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < 5 && containerRef.current) {
            // It's a click - check for marker intersection
            const rect = containerRef.current.getBoundingClientRect()
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

            raycaster.setFromCamera(mouse, camera)
            const intersects = raycaster.intersectObjects(markersRef.current)

            if (intersects.length > 0) {
              const marker = intersects[0].object as THREE.Mesh
              const site = marker.userData.site
              if (site) {
                setSelectedSite(site)
                // ✨ Track Investment Scorecard view
                trackScorecardView(site.id, site.name, site.type)
                // ✨ Fly camera to selected site for cinematic focus
                flyToLocation(site.latitude, site.longitude, 2.5, 1500)
                // Callback if provided
                if (onSiteSelect) {
                  onSiteSelect(site)
                }
                logger.info(`📍 Selected site: ${site.name}`)
              }
            } else {
              // Clicked empty space - deselect (zoom out slightly)
              setSelectedSite(null)
              if (camera.position.length() < 3.5) {
                flyToLocation(0, 0, 4.2, 1000) // Return to overview
              }
            }
          }

          isDragging = false
          if (containerRef.current && !hoveredMarker) {
            containerRef.current.style.cursor = 'grab'
          }
        }

        // Mouse wheel zoom
        const handleWheel = (e: WheelEvent) => {
          e.preventDefault()
          const zoomSpeed = 0.001
          const delta = e.deltaY * zoomSpeed

          // Zoom in/out by moving camera
          const direction = camera.position.clone().normalize()
          camera.position.addScaledVector(direction, delta)

          // Clamp zoom distance
          const distance = camera.position.length()
          if (distance < 1.8) {
            camera.position.setLength(1.8) // Min distance
          } else if (distance > 8) {
            camera.position.setLength(8) // Max distance
          }
        }

        if (containerRef.current) {
          containerRef.current.style.cursor = 'grab'
          containerRef.current.addEventListener('mousedown', handleMouseDown)
          containerRef.current.addEventListener('mousemove', handleMouseMove)
          containerRef.current.addEventListener('mouseup', handleMouseUp)
          containerRef.current.addEventListener('mouseleave', handleMouseUp)
          containerRef.current.addEventListener('wheel', handleWheel, { passive: false })
        }

        // ✨ Camera flyover function - smoothly fly to any location on globe
        const flyToLocation = (latitude: number, longitude: number, distance: number = 3.0, durationMs: number = 2000) => {
          // Convert lat/lng to 3D position on sphere
          const phi = (90 - latitude) * (Math.PI / 180)
          const theta = (longitude + 180) * (Math.PI / 180)

          const targetPos = new THREE.Vector3(
            -(distance * Math.sin(phi) * Math.cos(theta)),
            distance * Math.cos(phi),
            distance * Math.sin(phi) * Math.sin(theta)
          )

          // Calculate target rotation to face the location
          const targetRotX = -latitude * (Math.PI / 180)
          const targetRotY = -longitude * (Math.PI / 180)

          // Store animation state
          cameraAnimationRef.current = {
            isAnimating: true,
            startPosition: camera.position.clone(),
            targetPosition: targetPos,
            startRotation: { x: earth.rotation.x, y: earth.rotation.y },
            targetRotation: { x: targetRotX, y: targetRotY },
            progress: 0,
            duration: durationMs
          }

          logger.info(`✈️  Flying to location: ${latitude}°, ${longitude}° (${durationMs}ms)`)
        }

        // Animation loop
        let time = 0
        let lastZoomCheck = 0
        let isLoadingSites = false

        const animate = () => {
          if (!rendererRef.current) return

          animationIdRef.current = requestAnimationFrame(animate)
          time += 0.008

          // 📊 Track FPS
          performanceMonitor.trackFrame()

          // ✨ Camera flyover animation
          if (cameraAnimationRef.current?.isAnimating) {
            const anim = cameraAnimationRef.current
            anim.progress += 16 / anim.duration // Assuming ~60fps (16ms per frame)

            if (anim.progress >= 1) {
              // Animation complete
              anim.progress = 1
              anim.isAnimating = false
            }

            // Smooth easing (ease-in-out cubic)
            const easeProgress = anim.progress < 0.5
              ? 4 * anim.progress * anim.progress * anim.progress
              : 1 - Math.pow(-2 * anim.progress + 2, 3) / 2

            // Interpolate camera position
            camera.position.lerpVectors(anim.startPosition, anim.targetPosition, easeProgress)

            // Interpolate earth rotation
            earth.rotation.x = anim.startRotation.x + (anim.targetRotation.x - anim.startRotation.x) * easeProgress
            earth.rotation.y = anim.startRotation.y + (anim.targetRotation.y - anim.startRotation.y) * easeProgress

            // Stop manual rotation during flyover
            rotationVelocityRef.current.x = 0
            rotationVelocityRef.current.y = 0
          }

          // 🔄 Dynamic zoom-based clustering (check every 2 seconds)
          if (!minimal && !isLoadingSites && Date.now() - lastZoomCheck > 2000) {
            const distance = camera.position.length()
            const newLevel = distance > 5 ? 'world' : distance > 2.5 ? 'regional' : 'local'

            if (newLevel !== currentZoomLevel) {
              logger.info(`🔄 Zoom changed: ${currentZoomLevel} → ${newLevel} (distance: ${distance.toFixed(2)})`)
              isLoadingSites = true
              lastZoomCheck = Date.now()
              setIsTransitioning(true)  // ✨ Show loading indicator

              // ✨ Trigger fade-out for all existing markers
              markersRef.current.forEach(marker => {
                marker.userData.isFadingOut = true
                marker.userData.isFadingIn = false
              })

              // Wait for fade-out (300ms) then load new data
              setTimeout(() => {
                // Load new cluster level
                fetch(`/api/sites?level=${newLevel}`)
                  .then(res => res.json())
                  .then(data => {
                    if (data.success && data.sites) {
                      setSites(data.sites)
                      setCurrentZoomLevel(newLevel as any)
                      logger.info(`✅ Loaded ${data.sites.length} ${newLevel} clusters`)
                    }
                  })
                  .catch(err => logger.error('Failed to reload clusters:', err))
                  .finally(() => {
                    isLoadingSites = false
                    // Keep transition state for fade-in to complete (300ms more)
                    setTimeout(() => setIsTransitioning(false), 300)
                  })
              }, 300)
            }
            lastZoomCheck = Date.now()
          }

          earthMaterial.uniforms.time.value = time
          atmosphereMaterial.uniforms.time.value = time
          atmosphere2Material.uniforms.time.value = time

          // Interactive rotation
          if (Math.abs(rotationVelocityRef.current.x) > 0.0001) {
            earth.rotation.x += rotationVelocityRef.current.x
            rotationVelocityRef.current.x *= 0.92
          }
          if (Math.abs(rotationVelocityRef.current.y) > 0.0001) {
            earth.rotation.y += rotationVelocityRef.current.y
            rotationVelocityRef.current.y *= 0.92
          }

          if (!isDragging && Math.abs(rotationVelocityRef.current.y) < 0.001) {
            earth.rotation.y += 0.0002
          }

          atmosphere.rotation.y += 0.0003

          starLayers.forEach((layer) => {
            layer.mesh.rotation.y += layer.speed
            layer.mesh.rotation.x += layer.speed * 0.5
          })

          // Animate premium pulsing markers with pop-in
          markersRef.current.forEach((marker) => {
            // ✨ Fade-out animation when transitioning
            if (marker.userData.isFadingOut) {
              marker.userData.fadeOpacity = Math.max(0, marker.userData.fadeOpacity - 0.05)
              if (marker.userData.material) {
                marker.userData.material.opacity = marker.userData.fadeOpacity
              }
            }

            // ✨ Fade-in animation for new markers
            if (marker.userData.isFadingIn === undefined || marker.userData.isFadingIn) {
              marker.userData.fadeOpacity = Math.min(1, (marker.userData.fadeOpacity || 0) + 0.05)
              if (marker.userData.material) {
                marker.userData.material.opacity = marker.userData.fadeOpacity
              }
              if (marker.userData.fadeOpacity >= 1) {
                marker.userData.isFadingIn = false
              }
            }

            // Pop-in animation
            if (marker.userData.animationProgress !== undefined && marker.userData.animationProgress < 1) {
              const delay = marker.userData.animationDelay || 0
              if (time > delay) {
                marker.userData.animationProgress = Math.min(1, marker.userData.animationProgress + 0.04)
                // Elastic ease-out for bouncy effect
                const progress = marker.userData.animationProgress
                const elastic = progress === 1 ? 1 : Math.pow(2, -10 * progress) * Math.sin((progress - 0.1) * 5 * Math.PI) + 1
                marker.scale.setScalar(elastic * (marker.userData.targetScale || 1.0))
              }
            }

            if (marker.userData.material) {
              // Update time uniform with random phase offset for organic feel
              const pulseOffset = marker.userData.pulseOffset || 0
              marker.userData.material.uniforms.time.value = time + pulseOffset
            }
          })

          // Animate glow layers (stored in earth's children)
          earth.children.forEach((child) => {
            if (child.userData.material && child.userData.material.uniforms?.time) {
              const pulseOffset = child.userData.pulseOffset || 0
              child.userData.material.uniforms.time.value = time + pulseOffset
            }

            // Animate energy connection flows
            if (child.userData.isEnergyConnection && child.userData.material) {
              child.userData.material.uniforms.time.value = time
            }
          })

          // Animate energy particles rising from high-capacity sites
          if (earth.userData.particleSystems) {
            earth.userData.particleSystems.forEach((system: any) => {
              if (system.material && system.material.uniforms) {
                system.material.uniforms.time.value = time
              }
            })
          }

          camera.position.x = Math.sin(time * 0.08) * 0.01
          camera.position.y = 0.0 + Math.cos(time * 0.1) * 0.01

          renderer.render(scene, camera)
        }
        animate()

        // Handle resize
        const handleResize = () => {
          if (!containerRef.current || !renderer) return

          const width = containerRef.current.clientWidth
          const height = containerRef.current.clientHeight

          camera.aspect = width / height
          camera.updateProjectionMatrix()
          renderer.setSize(width, height)
        }
        window.addEventListener('resize', handleResize)

        return () => {
          window.removeEventListener('resize', handleResize)
          if (containerRef.current) {
            containerRef.current.removeEventListener('mousedown', handleMouseDown)
            containerRef.current.removeEventListener('mousemove', handleMouseMove)
            containerRef.current.removeEventListener('mouseup', handleMouseUp)
            containerRef.current.removeEventListener('mouseleave', handleMouseUp)
            containerRef.current.removeEventListener('wheel', handleWheel)
          }
        }
      } catch (err) {
        logger.error('❌ Globe initialization error:', err)
        setError('Failed to initialize 3D globe')
      }
    }, 100)

    return () => {
      clearTimeout(initTimer)

      if (animationIdRef.current !== null) {
        cancelAnimationFrame(animationIdRef.current)
      }

      if (rendererRef.current) {
        rendererRef.current.dispose()
        if (containerRef.current && rendererRef.current.domElement) {
          try {
            containerRef.current.removeChild(rendererRef.current.domElement)
          } catch (e) {
            logger.error('Cleanup error:', e)
          }
        }
      }
    }
  }, [sites]) // Re-run when sites data changes

  if (error) {
    return (
      <div className="absolute inset-0 bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="text-blue-400 text-6xl mb-4">🌍</div>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="absolute inset-0 min-h-[600px]" />

      {/* Hover Tooltip */}
      {hoveredSite && !minimal && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltipPos.x + 15,
            top: tooltipPos.y - 10,
            transform: 'translateY(-100%)'
          }}
        >
          <div className="bg-black/90 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/20 shadow-xl max-w-xs">
            <div className="text-sm font-semibold text-white truncate">{hoveredSite.name}</div>
            <div className="flex items-center gap-3 text-xs text-white/60 mt-1">
              {hoveredSite.isCluster && hoveredSite.site_count && (
                <span className="text-blue-400">{hoveredSite.site_count} sites</span>
              )}
              {hoveredSite.type && <span className="capitalize">{hoveredSite.type}</span>}
              {(hoveredSite.estimated_capacity_mw || hoveredSite.capacity_mw) && (
                <span className="text-emerald-400">
                  {(hoveredSite.estimated_capacity_mw || hoveredSite.capacity_mw || 0).toLocaleString()} MW
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ✨ Real-Time Statistics Dashboard */}
      {showStats && sites.length > 0 && !minimal && (
        <div className="absolute top-6 left-6 bg-black/70 backdrop-blur-md rounded-xl p-4 border border-emerald-500/20 min-w-[280px]">
          <h4 className="text-xs font-semibold text-emerald-400 mb-3 uppercase tracking-wider flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
            Investment Potential
          </h4>

          {/* Main Investment Stats */}
          <div className="space-y-3">
            {/* Total Investment Potential */}
            <div>
              <div className="text-2xl font-bold text-white">
                ${(viewportStats.investmentPotential / 1_000_000_000).toFixed(1)}B
              </div>
              <div className="text-xs text-white/50">Estimated capital needed</div>
            </div>

            {/* Capacity */}
            <div className="flex items-end justify-between gap-4">
              <div>
                <div className="text-lg font-semibold text-emerald-400">
                  {(viewportStats.totalCapacity / 1000).toFixed(1)} GW
                </div>
                <div className="text-xs text-white/50">Total capacity</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-cyan-400">
                  {viewportStats.avgIRR.toFixed(1)}%
                </div>
                <div className="text-xs text-white/50">Avg IRR</div>
              </div>
            </div>

            {/* Regional Coverage */}
            <div className="pt-2 border-t border-white/10">
              <div className="flex items-center justify-between text-xs">
                <span className="text-white/50">{viewportStats.states.size} states</span>
                <span className="text-white/50">{viewportStats.visibleSites.toLocaleString()} sites</span>
              </div>
            </div>

            {/* Energy Mix Breakdown */}
            <div className="pt-2 border-t border-white/10">
              <div className="text-xs text-white/50 mb-2">Energy Mix</div>
              <div className="space-y-1.5">
                {Object.entries(viewportStats.typeBreakdown)
                  .sort(([, a], [, b]) => (b as number) - (a as number))
                  .slice(0, 4) // Top 4 types
                  .map(([type, capacity]) => {
                    const percentage = (capacity / viewportStats.totalCapacity) * 100
                    const colors = {
                      solar: 'bg-amber-400',
                      wind: 'bg-cyan-400',
                      hydro: 'bg-blue-500',
                      nuclear: 'bg-purple-500',
                      geothermal: 'bg-red-500',
                    }
                    const bgColor = colors[type as keyof typeof colors] || 'bg-gray-400'

                    return (
                      <div key={type} className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${bgColor}`}></div>
                        <div className="flex-1 flex items-center justify-between text-xs">
                          <span className="text-white/60 capitalize">{type}</span>
                          <span className="text-white/80 font-medium">{percentage.toFixed(0)}%</span>
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Color Legend - Energy Types */}
      {showLegend && !minimal && !comparisonMode && (
        <div className="absolute bottom-6 left-6 bg-black/60 backdrop-blur-md rounded-xl p-4 border border-white/10">
          <h4 className="text-xs font-semibold text-white/80 mb-3 uppercase tracking-wider">Energy Types</h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-amber-400 shadow-lg shadow-amber-400/50"></div>
              <span className="text-xs text-white/70">Solar</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"></div>
              <span className="text-xs text-white/70">Wind</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500 shadow-lg shadow-blue-500/50"></div>
              <span className="text-xs text-white/70">Hydro</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500 shadow-lg shadow-purple-500/50"></div>
              <span className="text-xs text-white/70">Nuclear</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500 shadow-lg shadow-red-500/50"></div>
              <span className="text-xs text-white/70">Geothermal</span>
            </div>
          </div>

          {/* ✨ Comparison Mode Toggle */}
          <button
            onClick={() => {
              const newMode = !comparisonMode
              setComparisonMode(newMode)
              trackComparisonMode(newMode) // ✨ Track panel open/close
              if (!newMode) {
                setSelectedRegions([])
              }
            }}
            className="mt-3 w-full px-3 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-400/30 rounded-lg text-xs text-white hover:from-purple-500/30 hover:to-pink-500/30 transition-all"
          >
            📊 Compare Regions
          </button>

          {/* ✨ Timeline Mode Toggle */}
          <button
            onClick={() => {
              const newMode = !timelineMode
              setTimelineMode(newMode)
              trackTimelineMode(newMode) // ✨ Track panel open/close
              if (!newMode) {
                setTimelineYear(2025)
                setIsAnimating(false)
              }
            }}
            className="mt-2 w-full px-3 py-2 bg-gradient-to-r from-blue-500/20 to-indigo-500/20 border border-blue-400/30 rounded-lg text-xs text-white hover:from-blue-500/30 hover:to-indigo-500/30 transition-all"
          >
            📅 Deployment Timeline
          </button>
        </div>
      )}

      {/* ✨ Regional Comparison Panel */}
      {comparisonMode && !minimal && (
        <div className="absolute bottom-2 sm:bottom-6 left-2 sm:left-6 right-2 sm:right-auto bg-black/90 backdrop-blur-md rounded-xl border border-purple-500/30 sm:max-w-2xl shadow-2xl shadow-purple-500/10 motion-safe:animate-[fadeInUp_0.3s_ease-out] overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-3 sm:p-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <h4 className="text-xs sm:text-sm font-bold text-white">Regional Comparison</h4>
              <button
                onClick={() => setComparisonMode(false)}
                className="text-white/40 hover:text-white/80 transition-colors text-xl leading-none"
              >
                ×
              </button>
            </div>
          </div>

          <div className="p-3 sm:p-4 space-y-3 sm:space-y-4 max-h-[500px] sm:max-h-[600px] overflow-y-auto">
            {/* State Selection */}
            <div>
              <label className="text-[10px] sm:text-xs text-white/50 mb-2 block">Select States to Compare (max 3)</label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {Object.keys(regionalStats)
                  .sort((a, b) => regionalStats[b].totalCapacity - regionalStats[a].totalCapacity)
                  .slice(0, 12) // Top 12 states
                  .map(state => (
                    <button
                      key={state}
                      onClick={() => {
                        if (selectedRegions.includes(state)) {
                          setSelectedRegions(selectedRegions.filter(s => s !== state))
                        } else if (selectedRegions.length < 3) {
                          setSelectedRegions([...selectedRegions, state])
                        }
                      }}
                      className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        selectedRegions.includes(state)
                          ? 'bg-purple-500/30 text-purple-300 border border-purple-400/50'
                          : 'bg-white/5 text-white/70 border border-white/10 hover:bg-white/10'
                      }`}
                      disabled={!selectedRegions.includes(state) && selectedRegions.length >= 3}
                    >
                      {state}
                    </button>
                  ))}
              </div>
            </div>

            {/* Comparison Table */}
            {selectedRegions.length > 0 && (
              <div className="border border-white/10 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-white/5">
                      <tr>
                        <th className="px-3 py-2 text-left text-white/50 font-semibold">Metric</th>
                        {selectedRegions.map(state => (
                          <th key={state} className="px-3 py-2 text-center text-white font-semibold">
                            {state}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {/* Total Sites */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">Total Sites</td>
                        {selectedRegions.map(state => (
                          <td key={state} className="px-3 py-2 text-center text-white">
                            {regionalStats[state].totalSites.toLocaleString()}
                          </td>
                        ))}
                      </tr>

                      {/* Total Capacity */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">Total Capacity</td>
                        {selectedRegions.map(state => (
                          <td key={state} className="px-3 py-2 text-center text-emerald-400 font-semibold">
                            {(regionalStats[state].totalCapacity / 1000).toFixed(1)} GW
                          </td>
                        ))}
                      </tr>

                      {/* Average IRR */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">Avg IRR</td>
                        {selectedRegions.map(state => (
                          <td key={state} className="px-3 py-2 text-center text-cyan-400 font-semibold">
                            {regionalStats[state].avgIRR.toFixed(1)}%
                          </td>
                        ))}
                      </tr>

                      {/* Total Investment */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">Total Investment</td>
                        {selectedRegions.map(state => (
                          <td key={state} className="px-3 py-2 text-center text-white">
                            ${(regionalStats[state].totalInvestment / 1_000_000_000).toFixed(1)}B
                          </td>
                        ))}
                      </tr>

                      {/* CO2 Avoided */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">CO₂ Avoided/Year</td>
                        {selectedRegions.map(state => (
                          <td key={state} className="px-3 py-2 text-center text-green-400">
                            {(regionalStats[state].co2Avoided / 1_000_000).toFixed(1)}M tons
                          </td>
                        ))}
                      </tr>

                      {/* Risk Score */}
                      <tr className="hover:bg-white/5">
                        <td className="px-3 py-2 text-white/60">Avg Risk Score</td>
                        {selectedRegions.map(state => {
                          const risk = regionalStats[state].avgRiskScore
                          return (
                            <td key={state} className="px-3 py-2 text-center">
                              <span className={`font-semibold ${
                                risk < 40 ? 'text-green-400' :
                                risk < 70 ? 'text-yellow-400' :
                                'text-red-400'
                              }`}>
                                {risk.toFixed(0)}/100
                              </span>
                            </td>
                          )
                        })}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Helper Text */}
            {selectedRegions.length === 0 && (
              <div className="text-center py-6 text-white/40 text-sm">
                Select up to 3 states above to compare investment opportunities
              </div>
            )}
          </div>
        </div>
      )}

      {/* ✨ Time-Series Projections Timeline */}
      {timelineMode && !minimal && Object.keys(projections).length > 0 && (
        <div className="absolute bottom-2 sm:bottom-6 left-2 sm:left-1/2 sm:-translate-x-1/2 right-2 sm:right-auto bg-black/90 backdrop-blur-md rounded-xl border border-blue-500/30 sm:max-w-4xl w-auto sm:w-full sm:mx-4 shadow-2xl shadow-blue-500/10 motion-safe:animate-[fadeInUp_0.3s_ease-out] overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-500/20 to-indigo-500/20 p-3 sm:p-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 sm:gap-4">
                <h4 className="text-xs sm:text-sm font-bold text-white">Deployment Timeline</h4>
                <div className="flex items-center gap-1 sm:gap-2">
                  <span className="text-2xl sm:text-3xl font-bold text-blue-400">{timelineYear}</span>
                  <span className="text-[10px] sm:text-xs text-white/50">Projections</span>
                </div>
              </div>
              <button
                onClick={() => setTimelineMode(false)}
                className="text-white/40 hover:text-white/80 transition-colors text-xl leading-none"
              >
                ×
              </button>
            </div>
          </div>

          <div className="p-3 sm:p-6 space-y-4 sm:space-y-6">
            {/* Timeline Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-white/50">
                <span>2025</span>
                <span>2030</span>
                <span>2035</span>
              </div>

              <div className="relative">
                {/* Slider Track */}
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  {/* Progress Fill */}
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-300"
                    style={{ width: `${((timelineYear - 2025) / 10) * 100}%` }}
                  />
                </div>

                {/* Milestone Markers */}
                <div className="absolute top-0 left-0 w-full h-2 pointer-events-none">
                  {[2027, 2030, 2033].map(year => (
                    <div
                      key={year}
                      className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-white/30"
                      style={{ left: `${((year - 2025) / 10) * 100}%` }}
                    />
                  ))}
                </div>

                {/* Slider Input */}
                <input
                  type="range"
                  min={2025}
                  max={2035}
                  value={timelineYear}
                  onChange={(e) => {
                    const newYear = Number(e.target.value)
                    setTimelineYear(newYear)
                    trackTimelineYear(newYear) // ✨ Track year selection
                  }}
                  className="absolute top-0 left-0 w-full h-2 opacity-0 cursor-pointer"
                />
              </div>

              <div className="flex items-center justify-center gap-2">
                <button
                  onClick={() => {
                    const newState = !isAnimating
                    setIsAnimating(newState)
                    trackTimelineAnimation(newState ? 'play' : 'pause') // ✨ Track animation control
                    if (newState) {
                      // Start animation
                      const interval = setInterval(() => {
                        setTimelineYear(prev => {
                          if (prev >= 2035) {
                            setIsAnimating(false)
                            clearInterval(interval)
                            return 2025 // Reset to start
                          }
                          return prev + 1
                        })
                      }, 800)
                    }
                  }}
                  className="px-4 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg text-xs font-medium transition-colors border border-blue-400/30"
                >
                  {isAnimating ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  onClick={() => {
                    setTimelineYear(2025)
                    trackTimelineAnimation('reset') // ✨ Track reset action
                  }}
                  className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-white/70 rounded-lg text-xs transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Cumulative Metrics Grid */}
            {projections[timelineYear] && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
                {/* Cumulative Capacity */}
                <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 rounded-lg p-4 border border-emerald-500/20">
                  <div className="text-xs text-emerald-400/70 mb-1">Cumulative Capacity</div>
                  <div className="text-2xl font-bold text-emerald-400">
                    {(projections[timelineYear].cumulativeCapacity / 1000).toFixed(1)}
                    <span className="text-sm ml-1">GW</span>
                  </div>
                  <div className="text-xs text-white/40 mt-1">
                    {projections[timelineYear].deployedSites.toLocaleString()} sites
                  </div>
                </div>

                {/* Cumulative Investment */}
                <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-600/5 rounded-lg p-4 border border-cyan-500/20">
                  <div className="text-xs text-cyan-400/70 mb-1">Cumulative Investment</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    ${(projections[timelineYear].cumulativeInvestment / 1_000_000_000).toFixed(1)}
                    <span className="text-sm ml-1">B</span>
                  </div>
                  <div className="text-xs text-white/40 mt-1">
                    {projections[timelineYear].activeProjects.toLocaleString()} active projects
                  </div>
                </div>

                {/* Cumulative Jobs */}
                <div className="bg-gradient-to-br from-yellow-500/10 to-yellow-600/5 rounded-lg p-4 border border-yellow-500/20">
                  <div className="text-xs text-yellow-400/70 mb-1">Jobs Created</div>
                  <div className="text-2xl font-bold text-yellow-400">
                    {(projections[timelineYear].cumulativeJobs / 1000).toFixed(0)}
                    <span className="text-sm ml-1">K</span>
                  </div>
                  <div className="text-xs text-white/40 mt-1">
                    ~{(projections[timelineYear].cumulativeJobs / projections[timelineYear].deployedSites).toFixed(0)} jobs/site
                  </div>
                </div>

                {/* Cumulative CO₂ Avoided */}
                <div className="bg-gradient-to-br from-green-500/10 to-green-600/5 rounded-lg p-4 border border-green-500/20">
                  <div className="text-xs text-green-400/70 mb-1">CO₂ Avoided</div>
                  <div className="text-2xl font-bold text-green-400">
                    {(projections[timelineYear].cumulativeCO2Avoided / 1_000_000).toFixed(1)}
                    <span className="text-sm ml-1">M tons</span>
                  </div>
                  <div className="text-xs text-white/40 mt-1">
                    Since 2025
                  </div>
                </div>
              </div>
            )}

            {/* Deployment Milestones */}
            <div className="border-t border-white/10 pt-4">
              <div className="text-xs text-white/50 mb-2">Key Milestones</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className={`p-2 rounded ${timelineYear >= 2027 ? 'bg-blue-500/20 text-blue-300' : 'bg-white/5 text-white/40'}`}>
                  <div className="font-semibold">2027</div>
                  <div className="text-[10px]">First Wave Complete</div>
                </div>
                <div className={`p-2 rounded ${timelineYear >= 2030 ? 'bg-blue-500/20 text-blue-300' : 'bg-white/5 text-white/40'}`}>
                  <div className="font-semibold">2030</div>
                  <div className="text-[10px]">50% Deployment</div>
                </div>
                <div className={`p-2 rounded ${timelineYear >= 2033 ? 'bg-blue-500/20 text-blue-300' : 'bg-white/5 text-white/40'}`}>
                  <div className="font-semibold">2033</div>
                  <div className="text-[10px]">Final Phase Launch</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Site Count Badge with Zoom Level */}
      {showStats && sites.length > 0 && !minimal && (
        <div className="absolute top-6 right-6 bg-black/60 backdrop-blur-md rounded-xl px-4 py-3 border border-white/10">
          <div className="text-2xl font-bold text-white">
            {sites.reduce((sum, s) => sum + (s.site_count || 1), 0).toLocaleString()}
          </div>
          <div className="text-xs text-white/60 uppercase tracking-wider">Sites Mapped</div>
          <div className="text-xs text-white/40 mt-1">{sites.length.toLocaleString()} clusters</div>
          <div className="mt-2 pt-2 border-t border-white/10">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                currentZoomLevel === 'world' ? 'bg-blue-400' :
                currentZoomLevel === 'regional' ? 'bg-emerald-400' :
                'bg-amber-400'
              }`}></div>
              <span className="text-xs text-white/60 capitalize">{currentZoomLevel} View</span>
            </div>
            {/* ✨ Loading indicator during transitions */}
            {isTransitioning ? (
              <div className="text-xs text-emerald-400 mt-1 flex items-center gap-1">
                <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse"></div>
                <span>Updating...</span>
              </div>
            ) : (
              <div className="text-xs text-white/40 mt-1">Scroll to zoom</div>
            )}
          </div>
        </div>
      )}

      {/* ✨ Search & Filter Panel */}
      {!minimal && (
        <div className="absolute top-[180px] right-6 bg-black/60 backdrop-blur-md rounded-xl border border-white/10 overflow-hidden" style={{ maxWidth: '280px' }}>
          {/* Header with toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/5 transition-colors"
          >
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
              </svg>
              <span className="text-sm font-semibold text-white">Search & Filter</span>
            </div>
            <svg
              className={`w-4 h-4 text-white/60 transition-transform ${showFilters ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Collapsible Filter Controls */}
          {showFilters && (
            <div className="px-4 pb-4 space-y-3 border-t border-white/10">
              {/* Search Input */}
              <div className="pt-3">
                <label className="text-xs text-white/50 mb-1.5 block">Search by name or location</label>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Type to search..."
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-sm text-white placeholder-white/40 focus:outline-none focus:border-cyan-400/50 focus:bg-white/15 transition-colors"
                />
              </div>

              {/* Energy Type Filter */}
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">Energy Type</label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-400/50 focus:bg-white/15 transition-colors cursor-pointer"
                >
                  <option value="all">All Types</option>
                  <option value="solar">Solar</option>
                  <option value="wind">Wind</option>
                  <option value="hydro">Hydro</option>
                  <option value="nuclear">Nuclear</option>
                  <option value="geothermal">Geothermal</option>
                </select>
              </div>

              {/* State Filter */}
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">State</label>
                <select
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-400/50 focus:bg-white/15 transition-colors cursor-pointer"
                >
                  <option value="all">All States</option>
                  <option value="CA">California</option>
                  <option value="TX">Texas</option>
                  <option value="FL">Florida</option>
                  <option value="NY">New York</option>
                  <option value="WA">Washington</option>
                  {/* Add more states as needed */}
                </select>
              </div>

              {/* Min Capacity Filter */}
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">Min Capacity (MW)</label>
                <input
                  type="number"
                  value={minCapacity}
                  onChange={(e) => setMinCapacity(e.target.value)}
                  placeholder="e.g., 100"
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-sm text-white placeholder-white/40 focus:outline-none focus:border-cyan-400/50 focus:bg-white/15 transition-colors"
                />
              </div>

              {/* Sort By */}
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">Sort By</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSortBy('capacity')}
                    className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                      sortBy === 'capacity'
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/50'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
                    }`}
                  >
                    Capacity
                  </button>
                  <button
                    onClick={() => setSortBy('irr')}
                    className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                      sortBy === 'irr'
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/50'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
                    }`}
                  >
                    IRR
                  </button>
                  <button
                    onClick={() => setSortBy('name')}
                    className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                      sortBy === 'name'
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/50'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
                    }`}
                  >
                    Name
                  </button>
                </div>
              </div>

              {/* Quick Filters (Presets) */}
              <div className="pt-2 border-t border-white/10">
                <label className="text-xs text-white/50 mb-2 block">Quick Filters</label>
                <div className="space-y-2">
                  <button
                    onClick={() => {
                      setSelectedType('all')
                      setMinCapacity('100')
                      setSortBy('capacity')
                    }}
                    className="w-full px-3 py-2 bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 border border-emerald-400/30 rounded-lg text-xs text-white hover:from-emerald-500/30 hover:to-cyan-500/30 transition-all"
                  >
                    🏆 Large Projects (100+ MW)
                  </button>
                  <button
                    onClick={() => {
                      setSortBy('irr')
                      setMinCapacity('')
                      setSelectedType('all')
                    }}
                    className="w-full px-3 py-2 bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-400/30 rounded-lg text-xs text-white hover:from-amber-500/30 hover:to-orange-500/30 transition-all"
                  >
                    💰 Best Returns (by IRR)
                  </button>
                  <button
                    onClick={() => {
                      setSelectedType('solar')
                      setSortBy('capacity')
                    }}
                    className="w-full px-3 py-2 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-400/30 rounded-lg text-xs text-white hover:from-amber-500/30 hover:to-yellow-500/30 transition-all"
                  >
                    ☀️ Solar Only
                  </button>
                </div>
              </div>

              {/* Clear Filters Button */}
              <button
                onClick={() => {
                  setSearchQuery('')
                  setSelectedType('all')
                  setSelectedState('all')
                  setMinCapacity('')
                  setSortBy('capacity')
                }}
                className="w-full px-3 py-2 bg-red-500/10 border border-red-400/30 rounded-lg text-xs text-red-400 hover:bg-red-500/20 transition-colors"
              >
                Clear All Filters
              </button>
            </div>
          )}
        </div>
      )}

      {/* ✨ Investment Scorecard Panel */}
      {selectedSite && !minimal && (() => {
        const scorecard = generateScorecard(selectedSite)
        const capacity = selectedSite.estimated_capacity_mw || selectedSite.capacity_mw || 0
        const irr = selectedSite.estimated_irr || 0

        return (
          <div className="absolute bottom-2 sm:bottom-6 right-2 sm:right-6 left-2 sm:left-auto bg-black/90 backdrop-blur-md rounded-xl border border-emerald-500/30 sm:max-w-md shadow-2xl shadow-emerald-500/10 motion-safe:animate-[fadeInUp_0.3s_ease-out] overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 p-3 sm:p-4 border-b border-white/10">
              <div className="flex items-start justify-between">
                <div className="min-w-0 flex-1">
                  <h4 className="text-sm sm:text-base font-bold text-white leading-tight mb-1 truncate">{selectedSite.name}</h4>
                  <div className="flex items-center gap-2 sm:gap-3 text-[10px] sm:text-xs text-white/60 flex-wrap">
                    {selectedSite.type && (
                      <span className="capitalize">{selectedSite.type}</span>
                    )}
                    {selectedSite.state && (
                      <span>{selectedSite.state}, USA</span>
                    )}
                    {selectedSite.isCluster && selectedSite.site_count && (
                      <span className="text-blue-400">{selectedSite.site_count} sites</span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setSelectedSite(null)}
                  className="text-white/40 hover:text-white/80 transition-colors text-xl leading-none -mt-1"
                >
                  ×
                </button>
              </div>
            </div>

            <div className="p-3 sm:p-4 space-y-3 sm:space-y-4 max-h-[400px] sm:max-h-[500px] overflow-y-auto">
              {/* Financial Metrics */}
              <div>
                <h5 className="text-xs font-semibold text-emerald-400 mb-2 uppercase tracking-wider">Financial</h5>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="text-xs text-white/50 mb-1">Total Investment</div>
                    <div className="text-lg font-bold text-white">
                      ${(scorecard.totalInvestment / 1_000_000).toFixed(1)}M
                    </div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="text-xs text-white/50 mb-1">IRR</div>
                    <div className="text-lg font-bold text-emerald-400">
                      {irr.toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="text-xs text-white/50 mb-1">Payback Period</div>
                    <div className="text-lg font-bold text-cyan-400">
                      {scorecard.paybackPeriod} yrs
                    </div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="text-xs text-white/50 mb-1">Capacity Factor</div>
                    <div className="text-lg font-bold text-white">
                      {(scorecard.capacityFactor * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical Metrics */}
              <div>
                <h5 className="text-xs font-semibold text-cyan-400 mb-2 uppercase tracking-wider">Technical</h5>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-white/60">Capacity</span>
                    <span className="text-sm font-semibold text-white">{capacity.toLocaleString()} MW</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-white/60">Grid Connection</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      scorecard.gridConnection === 'easy' ? 'bg-green-500/20 text-green-400' :
                      scorecard.gridConnection === 'moderate' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {scorecard.gridConnection}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-white/60">Permitting Status</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      scorecard.permittingStatus === 'approved' ? 'bg-green-500/20 text-green-400' :
                      scorecard.permittingStatus === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {scorecard.permittingStatus === 'approved' ? 'Approved' :
                       scorecard.permittingStatus === 'pending' ? 'Pending' :
                       'Not Started'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Environmental Impact */}
              <div>
                <h5 className="text-xs font-semibold text-green-400 mb-2 uppercase tracking-wider">Environmental</h5>
                <div className="space-y-2">
                  <div className="bg-green-500/10 rounded-lg p-3 border border-green-500/20">
                    <div className="text-xs text-green-400/80 mb-1">CO₂ Avoided Per Year</div>
                    <div className="text-lg font-bold text-green-400">
                      {scorecard.co2AvoidedPerYear.toLocaleString()} tons
                    </div>
                    <div className="text-xs text-green-400/60 mt-1">
                      vs. coal baseline
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-white/60">Water Impact</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      scorecard.waterImpact === 'positive' ? 'bg-blue-500/20 text-blue-400' :
                      scorecard.waterImpact === 'neutral' ? 'bg-gray-500/20 text-gray-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {scorecard.waterImpact}
                    </span>
                  </div>
                </div>
              </div>

              {/* Risk Assessment */}
              <div>
                <h5 className="text-xs font-semibold text-orange-400 mb-2 uppercase tracking-wider">Risk Assessment</h5>
                <div className="space-y-3">
                  {/* Risk Score Bar */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs text-white/60">Overall Risk Score</span>
                      <span className={`text-sm font-bold ${
                        scorecard.riskScore < 40 ? 'text-green-400' :
                        scorecard.riskScore < 70 ? 'text-yellow-400' :
                        'text-red-400'
                      }`}>
                        {scorecard.riskScore}/100
                      </span>
                    </div>
                    <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          scorecard.riskScore < 40 ? 'bg-green-500' :
                          scorecard.riskScore < 70 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${scorecard.riskScore}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-white/40 mt-1">Lower is better</div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs text-white/60">Regulatory Risk</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      scorecard.regulatoryRisk === 'low' ? 'bg-green-500/20 text-green-400' :
                      scorecard.regulatoryRisk === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {scorecard.regulatoryRisk}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Button */}
            <div className="p-4 border-t border-white/10">
              <a
                href={selectedSite.isCluster ? `/explore?region=${selectedSite.state}` : `/project/${selectedSite.id}`}
                className="block w-full bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black font-bold py-3 px-4 rounded-lg text-center text-sm transition-all shadow-lg shadow-emerald-500/20"
              >
                {selectedSite.isCluster ? `Explore ${selectedSite.site_count} Sites →` : 'View Full Investment Details →'}
              </a>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
