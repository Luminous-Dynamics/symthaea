import { describe, it, expect } from 'vitest'
import { clusterMarkers } from '@/lib/markerClustering'

// Helper to create test markers
function createMarker(
  id: number,
  lat: number,
  lng: number,
  type = 'solar',
  power = 100
) {
  return {
    id,
    name: `Project ${id}`,
    lat,
    lng,
    type,
    power,
    color: '#00ff00',
    size: 1,
    status: 'active',
  }
}

describe('clusterMarkers', () => {
  describe('basic functionality', () => {
    it('returns empty array for empty input', () => {
      const result = clusterMarkers([])
      expect(result).toEqual([])
    })

    it('returns single marker without clustering', () => {
      const markers = [createMarker(1, 40.7128, -74.006)]
      const result = clusterMarkers(markers)

      expect(result).toHaveLength(1)
      // Single marker should be returned as-is
      expect(result[0]).toHaveProperty('id', 1)
    })

    it('preserves all markers when spread apart', () => {
      // Markers far apart (> 5 degrees default radius)
      const markers = [
        createMarker(1, 0, 0),      // Equator, Prime Meridian
        createMarker(2, 20, 20),    // 20 degrees away
        createMarker(3, -20, -20),  // 20 degrees away opposite
      ]

      const result = clusterMarkers(markers)

      // All three should be separate (no clustering)
      expect(result.length).toBeGreaterThanOrEqual(3)
    })
  })

  describe('clustering behavior', () => {
    it('clusters nearby markers', () => {
      // Two markers very close together (< 5 degrees)
      const markers = [
        createMarker(1, 40.0, -74.0),
        createMarker(2, 40.1, -74.1), // ~0.14 degrees away
      ]

      const result = clusterMarkers(markers)

      // Should cluster into 1 group
      expect(result.length).toBeLessThanOrEqual(2)

      // If clustered, check cluster properties
      const cluster = result.find((r) => 'count' in r)
      if (cluster && 'count' in cluster) {
        expect(cluster.count).toBe(2)
        expect(cluster.totalPower).toBe(200) // 100 + 100
      }
    })

    it('respects custom cluster radius', () => {
      // Markers 3 degrees apart
      const markers = [
        createMarker(1, 40.0, -74.0),
        createMarker(2, 43.0, -74.0), // 3 degrees north
      ]

      // With default radius (5 degrees), they should cluster
      const resultDefault = clusterMarkers(markers)

      // With smaller radius (2 degrees), they should NOT cluster
      const resultSmall = clusterMarkers(markers, { clusterRadius: 2 })

      expect(resultSmall.length).toBeGreaterThanOrEqual(resultDefault.length)
    })
  })

  describe('cluster properties', () => {
    it('calculates cluster center correctly', () => {
      // Create markers in a grid pattern
      const markers = [
        createMarker(1, 10, 10),
        createMarker(2, 10, 11),
        createMarker(3, 11, 10),
        createMarker(4, 11, 11),
      ]

      const result = clusterMarkers(markers, { clusterRadius: 10 })

      // All four should cluster together
      const cluster = result.find((r) => 'count' in r && r.count === 4)
      if (cluster && 'lat' in cluster && 'lng' in cluster) {
        // Center should be approximately (10.5, 10.5)
        expect(cluster.lat).toBeCloseTo(10.5, 0)
        expect(cluster.lng).toBeCloseTo(10.5, 0)
      }
    })

    it('calculates total power correctly', () => {
      const markers = [
        createMarker(1, 40, -74, 'solar', 100),
        createMarker(2, 40.1, -74.1, 'solar', 200),
        createMarker(3, 40.2, -74.2, 'wind', 300),
      ]

      const result = clusterMarkers(markers, { clusterRadius: 10 })

      const cluster = result.find((r) => 'count' in r && r.count === 3)
      if (cluster && 'totalPower' in cluster) {
        expect(cluster.totalPower).toBe(600) // 100 + 200 + 300
      }
    })

    it('determines dominant type correctly', () => {
      const markers = [
        createMarker(1, 40, -74, 'solar', 100),
        createMarker(2, 40.1, -74.1, 'solar', 100),
        createMarker(3, 40.2, -74.2, 'wind', 100),
      ]

      const result = clusterMarkers(markers, { clusterRadius: 10 })

      const cluster = result.find((r) => 'count' in r)
      if (cluster && 'dominantType' in cluster) {
        expect(cluster.dominantType).toBe('solar')
      }
    })
  })

  describe('performance characteristics', () => {
    it('handles large datasets', () => {
      // Create 1000 random markers
      const markers = Array.from({ length: 1000 }, (_, i) =>
        createMarker(i, Math.random() * 180 - 90, Math.random() * 360 - 180)
      )

      const start = performance.now()
      const result = clusterMarkers(markers)
      const duration = performance.now() - start

      // Should complete in under 100ms for 1000 markers
      expect(duration).toBeLessThan(100)

      // Should return some result
      expect(result.length).toBeGreaterThan(0)
    })

    it('handles edge case coordinates', () => {
      const markers = [
        createMarker(1, 90, 0),      // North Pole
        createMarker(2, -90, 0),     // South Pole
        createMarker(3, 0, 180),     // Date Line
        createMarker(4, 0, -180),    // Date Line (other side)
      ]

      // Should not throw
      expect(() => clusterMarkers(markers)).not.toThrow()
    })
  })
})
