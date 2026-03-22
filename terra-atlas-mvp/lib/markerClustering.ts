/**
 * Marker Clustering for 3D Globe
 *
 * Efficiently clusters thousands of markers for optimal rendering performance.
 * Uses a grid-based approach with dynamic cell size based on zoom level.
 */

interface Marker {
  id: number
  name: string
  lat: number
  lng: number
  type: string
  power: number
  color: string
  size: number
  status: string
}

interface Cluster {
  lat: number
  lng: number
  count: number
  totalPower: number
  markers: Marker[]
  avgColor: string
  dominantType: string
}

interface ClusterOptions {
  /** Minimum distance (in degrees) for clustering. Smaller = more clusters */
  clusterRadius?: number
  /** Maximum number of markers in a single cluster */
  maxClusterSize?: number
  /** Minimum zoom level to show individual markers */
  minZoomForIndividual?: number
}

/**
 * Calculate distance between two lat/lng points in degrees
 * (Fast approximation, good enough for clustering)
 */
function fastDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = lat2 - lat1
  const dLng = lng2 - lng1
  return Math.sqrt(dLat * dLat + dLng * dLng)
}

/**
 * Grid-based clustering algorithm
 * O(n) complexity - fast for large datasets
 */
export function clusterMarkers(
  markers: Marker[],
  options: ClusterOptions = {}
): (Marker | Cluster)[] {
  const {
    clusterRadius = 5, // 5 degrees ~= 550km at equator
    maxClusterSize = 100,
    minZoomForIndividual = 1.5 // Camera distance
  } = options

  if (markers.length === 0) return []

  // Create a grid for fast spatial lookup
  const cellSize = clusterRadius
  const grid = new Map<string, Marker[]>()

  // Place markers in grid cells
  markers.forEach(marker => {
    const cellX = Math.floor(marker.lng / cellSize)
    const cellY = Math.floor(marker.lat / cellSize)
    const key = `${cellX},${cellY}`

    if (!grid.has(key)) {
      grid.set(key, [])
    }
    grid.get(key)!.push(marker)
  })

  const result: (Marker | Cluster)[] = []

  // Process each grid cell
  grid.forEach(cellMarkers => {
    if (cellMarkers.length === 1) {
      // Single marker - keep as is
      result.push(cellMarkers[0])
    } else if (cellMarkers.length <= maxClusterSize) {
      // Create a cluster
      const cluster = createCluster(cellMarkers)
      result.push(cluster)
    } else {
      // Too many in one cell - subdivide
      const subclusters = subdivideCell(cellMarkers, clusterRadius / 2)
      result.push(...subclusters)
    }
  })

  return result
}

/**
 * Create a cluster from a group of markers
 */
function createCluster(markers: Marker[]): Cluster {
  // Calculate centroid
  let totalLat = 0
  let totalLng = 0
  let totalPower = 0
  const typeCounts = new Map<string, number>()

  markers.forEach(marker => {
    totalLat += marker.lat
    totalLng += marker.lng
    totalPower += marker.power
    typeCounts.set(marker.type, (typeCounts.get(marker.type) || 0) + 1)
  })

  const count = markers.length
  const avgLat = totalLat / count
  const avgLng = totalLng / count

  // Find dominant type
  let dominantType = markers[0].type
  let maxCount = 0
  typeCounts.forEach((count, type) => {
    if (count > maxCount) {
      maxCount = count
      dominantType = type
    }
  })

  // Get color for dominant type
  const dominantMarker = markers.find(m => m.type === dominantType) || markers[0]

  return {
    lat: avgLat,
    lng: avgLng,
    count,
    totalPower,
    markers,
    avgColor: dominantMarker.color,
    dominantType
  }
}

/**
 * Subdivide a cell that has too many markers
 */
function subdivideCell(markers: Marker[], radius: number): (Marker | Cluster)[] {
  const subclusters: (Marker | Cluster)[] = []
  const used = new Set<number>()

  markers.forEach((marker, i) => {
    if (used.has(i)) return

    const nearby: Marker[] = [marker]
    used.add(i)

    // Find nearby markers
    markers.forEach((other, j) => {
      if (i === j || used.has(j)) return

      const dist = fastDistance(marker.lat, marker.lng, other.lat, other.lng)
      if (dist < radius) {
        nearby.push(other)
        used.add(j)
      }
    })

    if (nearby.length === 1) {
      subclusters.push(nearby[0])
    } else {
      subclusters.push(createCluster(nearby))
    }
  })

  return subclusters
}

/**
 * Dynamic clustering based on camera distance
 * Closer camera = less clustering (more detail)
 */
export function getDynamicClusterRadius(cameraDistance: number): number {
  // Camera distance typically ranges from 1.5 (close) to 5+ (far)

  if (cameraDistance < 2) {
    return 2 // ~220km - show individual markers
  } else if (cameraDistance < 3) {
    return 5 // ~550km - moderate clustering
  } else if (cameraDistance < 4) {
    return 10 // ~1100km - aggressive clustering
  } else {
    return 15 // ~1650km - very aggressive for distant view
  }
}

/**
 * Check if clustering should be applied based on marker count and camera distance
 */
export function shouldCluster(markerCount: number, cameraDistance: number): boolean {
  // Always cluster if we have a lot of markers
  if (markerCount > 1000) return true

  // For medium counts, cluster only when zoomed out
  if (markerCount > 100 && cameraDistance > 2.5) return true

  // For small counts, only cluster when very zoomed out
  if (markerCount > 20 && cameraDistance > 4) return true

  return false
}

/**
 * Get a summary of cluster types for display
 */
export function getClusterSummary(cluster: Cluster): string {
  const typeCounts = new Map<string, number>()

  cluster.markers.forEach(marker => {
    typeCounts.set(marker.type, (typeCounts.get(marker.type) || 0) + 1)
  })

  const types = Array.from(typeCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([type, count]) => `${count} ${type}`)
    .join(', ')

  return types
}

/**
 * Color blending for mixed clusters
 * Returns a hex color that represents the mix of marker types
 */
export function blendColors(colors: string[], weights: number[]): string {
  let r = 0, g = 0, b = 0
  let totalWeight = 0

  colors.forEach((color, i) => {
    const weight = weights[i] || 1
    totalWeight += weight

    // Parse hex color
    const hex = color.replace('#', '')
    r += parseInt(hex.substring(0, 2), 16) * weight
    g += parseInt(hex.substring(2, 4), 16) * weight
    b += parseInt(hex.substring(4, 6), 16) * weight
  })

  r = Math.round(r / totalWeight)
  g = Math.round(g / totalWeight)
  b = Math.round(b / totalWeight)

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

export type { Marker, Cluster, ClusterOptions }
