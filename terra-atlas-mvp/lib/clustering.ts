/**
 * Site Clustering Utility
 * Groups nearby energy sites into clusters for efficient rendering
 */

interface Site {
  id: string
  name: string
  latitude: number
  longitude: number
  type: string
  capacity_mw?: number
  [key: string]: any
}

interface Cluster {
  id: string
  latitude: number
  longitude: number
  sites: Site[]
  count: number
  totalCapacity: number
  dominantType: string
  isCluster: true
}

interface ClusterOptions {
  gridSize?: number // Degrees per grid cell (smaller = more detail)
  maxSitesPerCluster?: number // When to show individual sites
  minClusterSize?: number // Minimum sites to form a cluster
}

/**
 * Cluster sites using a simple grid-based approach
 * Fast O(n) performance suitable for 100K+ sites
 */
export function clusterSites(
  sites: Site[],
  options: ClusterOptions = {}
): (Site | Cluster)[] {
  const {
    gridSize = 2, // 2 degrees per cell (~220km at equator)
    maxSitesPerCluster = 50, // Show individuals below this
    minClusterSize = 3, // Minimum to form cluster
  } = options

  if (sites.length <= maxSitesPerCluster) {
    return sites
  }

  // Grid-based clustering
  const grid = new Map<string, Site[]>()

  for (const site of sites) {
    if (site.latitude == null || site.longitude == null) continue

    // Calculate grid cell key
    const gridLat = Math.floor(site.latitude / gridSize)
    const gridLon = Math.floor(site.longitude / gridSize)
    const key = `${gridLat},${gridLon}`

    if (!grid.has(key)) {
      grid.set(key, [])
    }
    grid.get(key)!.push(site)
  }

  // Convert grid cells to clusters or individual sites
  const results: (Site | Cluster)[] = []

  for (const [key, cellSites] of grid) {
    if (cellSites.length >= minClusterSize) {
      // Create cluster
      const cluster = createCluster(key, cellSites)
      results.push(cluster)
    } else {
      // Keep as individual sites
      results.push(...cellSites)
    }
  }

  return results
}

/**
 * Create a cluster from multiple sites
 */
function createCluster(key: string, sites: Site[]): Cluster {
  // Calculate centroid
  let sumLat = 0
  let sumLon = 0
  let totalCapacity = 0
  const typeCounts = new Map<string, number>()

  for (const site of sites) {
    sumLat += site.latitude
    sumLon += site.longitude
    totalCapacity += site.capacity_mw || 0

    const type = site.type || 'unknown'
    typeCounts.set(type, (typeCounts.get(type) || 0) + 1)
  }

  // Find dominant type
  let dominantType = 'mixed'
  let maxCount = 0
  for (const [type, count] of typeCounts) {
    if (count > maxCount) {
      maxCount = count
      dominantType = type
    }
  }

  return {
    id: `cluster-${key}`,
    latitude: sumLat / sites.length,
    longitude: sumLon / sites.length,
    sites,
    count: sites.length,
    totalCapacity,
    dominantType,
    isCluster: true,
  }
}

/**
 * Adaptive clustering based on zoom level
 * Returns appropriate grid size for visualization
 */
export function getClusterGridSize(zoomLevel: number): number {
  // Higher zoom = smaller grid = more detail
  if (zoomLevel >= 8) return 0.1   // ~11km - show individuals
  if (zoomLevel >= 6) return 0.5   // ~55km
  if (zoomLevel >= 4) return 1     // ~110km
  if (zoomLevel >= 2) return 2     // ~220km
  return 5                          // ~550km - country level
}

/**
 * Format cluster for display
 */
export function formatClusterLabel(cluster: Cluster): string {
  const capacity = cluster.totalCapacity >= 1000
    ? `${(cluster.totalCapacity / 1000).toFixed(1)} GW`
    : `${cluster.totalCapacity.toFixed(0)} MW`

  return `${cluster.count} sites • ${capacity}`
}

/**
 * Check if item is a cluster
 */
export function isCluster(item: Site | Cluster): item is Cluster {
  return 'isCluster' in item && item.isCluster === true
}

/**
 * Get cluster/site display properties
 */
export function getDisplayProperties(item: Site | Cluster) {
  if (isCluster(item)) {
    return {
      id: item.id,
      latitude: item.latitude,
      longitude: item.longitude,
      type: item.dominantType,
      capacity_mw: item.totalCapacity,
      name: `${item.count} Energy Sites`,
      isCluster: true,
      count: item.count,
    }
  }

  return {
    id: item.id,
    latitude: item.latitude,
    longitude: item.longitude,
    type: item.type,
    capacity_mw: item.capacity_mw || 0,
    name: item.name,
    isCluster: false,
    count: 1,
  }
}

/**
 * Sample sites for initial load (fast preview)
 * Takes representative samples across the grid
 */
export function sampleSites(sites: Site[], maxSamples: number = 500): Site[] {
  if (sites.length <= maxSamples) return sites

  // Grid-based sampling for geographic distribution
  const gridSize = 3 // degrees
  const grid = new Map<string, Site[]>()

  for (const site of sites) {
    if (site.latitude == null || site.longitude == null) continue

    const gridLat = Math.floor(site.latitude / gridSize)
    const gridLon = Math.floor(site.longitude / gridSize)
    const key = `${gridLat},${gridLon}`

    if (!grid.has(key)) {
      grid.set(key, [])
    }
    grid.get(key)!.push(site)
  }

  // Take top sites from each cell
  const samplesPerCell = Math.ceil(maxSamples / grid.size)
  const samples: Site[] = []

  for (const cellSites of grid.values()) {
    // Sort by capacity and take top
    const sorted = cellSites
      .sort((a, b) => (b.capacity_mw || 0) - (a.capacity_mw || 0))
      .slice(0, samplesPerCell)

    samples.push(...sorted)
  }

  // Trim to exact count
  return samples.slice(0, maxSamples)
}
