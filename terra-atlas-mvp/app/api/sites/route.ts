import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

interface ClusterLevel {
  gridSize: number
  clusterCount: number
  avgClusterSize: string
  clusters: any[]
}

interface MultiLevelData {
  metadata: {
    generatedAt: string
    totalSites: number
    levels: Array<{ name: string; gridSize: number }>
    totalCapacityMw: number
  }
  levels: {
    world: ClusterLevel
    regional: ClusterLevel
    local: ClusterLevel
  }
}

// ✨ PERFORMANCE OPTIMIZATION: Multi-level caching
// 1. Raw data cache (40MB JSON file)
let cachedData: MultiLevelData | null = null
let cacheLoadTime: number = 0

// 2. Response cache (stores already-filtered results)
interface CachedResponse {
  data: any
  timestamp: number
}

const responseCache = new Map<string, CachedResponse>()
const CACHE_TTL = 60 * 60 * 1000 // 1 hour
const MAX_CACHE_ENTRIES = 100

function loadData(): MultiLevelData {
  if (cachedData) return cachedData

  console.time('⚡ Load sites data')
  const dataPath = path.join(process.cwd(), 'public/data/sites-multi-level.json')
  const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'))
  cachedData = data
  cacheLoadTime = Date.now()
  console.timeEnd('⚡ Load sites data')
  console.log(`✅ Cached ${data.metadata.totalSites.toLocaleString()} sites`)

  return data
}

// Generate cache key from request parameters
function getCacheKey(searchParams: URLSearchParams): string {
  const level = searchParams.get('level') || 'world'
  const type = searchParams.get('type') || 'all'
  const state = searchParams.get('state') || 'all'
  const minCapacity = searchParams.get('minCapacity') || '0'
  const bounds = searchParams.get('bounds') || 'none'
  return `${level}-${type}-${state}-${minCapacity}-${bounds}`
}

// Clean old cache entries if needed
function cleanCache() {
  if (responseCache.size > MAX_CACHE_ENTRIES) {
    const now = Date.now()
    // Remove expired entries
    for (const [key, value] of responseCache.entries()) {
      if (now - value.timestamp > CACHE_TTL) {
        responseCache.delete(key)
      }
    }
    // If still too many, remove oldest
    if (responseCache.size > MAX_CACHE_ENTRIES) {
      const entries = Array.from(responseCache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp)
      const toRemove = entries.slice(0, entries.length - MAX_CACHE_ENTRIES)
      toRemove.forEach(([key]) => responseCache.delete(key))
    }
  }
}

/**
 * GET /api/sites
 *
 * Query Parameters:
 * - level: 'world' | 'regional' | 'local' (default: 'world')
 * - zoom: number (camera distance, auto-selects level)
 * - bounds: 'minLat,maxLat,minLng,maxLng' (filter by viewport)
 * - type: energy type filter
 * - minCapacity: minimum capacity in MW
 * - state: state filter
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now()

  try {
    const searchParams = request.nextUrl.searchParams

    // ✨ Check response cache first
    const cacheKey = getCacheKey(searchParams)
    const cached = responseCache.get(cacheKey)

    if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
      const responseTime = Date.now() - startTime
      console.log(`⚡ Cache HIT for ${cacheKey} (${responseTime}ms)`)

      // Return cached response with cache headers
      return new NextResponse(JSON.stringify(cached.data), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
          'X-Cache': 'HIT',
          'X-Response-Time': `${responseTime}ms`,
        },
      })
    }

    console.log(`💾 Cache MISS for ${cacheKey}, generating...`)

    // Load data (will use file cache)
    const data = loadData()

    // Determine zoom level
    let levelName = searchParams.get('level') || 'world'
    const zoom = searchParams.get('zoom')

    if (zoom) {
      const zoomNum = parseFloat(zoom)
      // Auto-select level based on camera distance
      if (zoomNum > 5) levelName = 'world'
      else if (zoomNum > 2.5) levelName = 'regional'
      else levelName = 'local'
    }

    // Get clusters for the level
    let clusters = data.levels[levelName as keyof typeof data.levels]?.clusters || []

    // Apply filters
    const bounds = searchParams.get('bounds')
    if (bounds) {
      const [minLat, maxLat, minLng, maxLng] = bounds.split(',').map(Number)
      clusters = clusters.filter(c =>
        c.latitude >= minLat && c.latitude <= maxLat &&
        c.longitude >= minLng && c.longitude <= maxLng
      )
    }

    const typeFilter = searchParams.get('type')
    if (typeFilter) {
      clusters = clusters.filter(c => c.type === typeFilter)
    }

    const minCapacity = searchParams.get('minCapacity')
    if (minCapacity) {
      const minCap = parseFloat(minCapacity)
      clusters = clusters.filter(c => c.total_capacity_mw >= minCap)
    }

    const stateFilter = searchParams.get('state')
    if (stateFilter) {
      clusters = clusters.filter(c => c.states?.includes(stateFilter))
    }

    // Transform to Site format
    const sites = clusters.map(cluster => ({
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
      avg_score: cluster.avg_score,
    }))

    const responseData = {
      success: true,
      level: levelName,
      metadata: {
        total: sites.length,
        totalSites: data.metadata.totalSites,
        totalCapacity: sites.reduce((sum, s) => sum + s.estimated_capacity_mw, 0),
      },
      sites,
    }

    // ✨ Store in response cache
    responseCache.set(cacheKey, {
      data: responseData,
      timestamp: Date.now(),
    })
    cleanCache() // Clean old entries if needed

    const responseTime = Date.now() - startTime
    console.log(`✅ Generated response for ${cacheKey} (${responseTime}ms)`)

    return new NextResponse(JSON.stringify(responseData), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
        'X-Cache': 'MISS',
        'X-Response-Time': `${responseTime}ms`,
      },
    })
  } catch (error) {
    console.error('Sites API error:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to load sites' },
      { status: 500 }
    )
  }
}
