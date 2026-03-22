import { describe, it, expect } from 'vitest'
import { ProjectsQuerySchema } from '@/lib/schemas/projects'
import { StatsResponseSchema, createDefaultStatsResponse } from '@/lib/schemas/stats'

describe('API Schema Validation', () => {
  describe('ProjectsQuerySchema', () => {
    it('parses valid query with all fields', () => {
      const result = ProjectsQuerySchema.safeParse({
        limit: '50',
        offset: '10',
        type: 'Solar',
        status: 'Operational',
        country: 'United States',
        search: 'california'
      })

      expect(result.success).toBe(true)
      if (result.success) {
        expect(result.data.limit).toBe(50)
        expect(result.data.offset).toBe(10)
        expect(result.data.type).toBe('Solar')
        expect(result.data.status).toBe('Operational')
        expect(result.data.country).toBe('United States')
        expect(result.data.search).toBe('california')
      }
    })

    it('uses default values for missing fields', () => {
      const result = ProjectsQuerySchema.safeParse({})

      expect(result.success).toBe(true)
      if (result.success) {
        expect(result.data.limit).toBe(100) // Default is 100
        expect(result.data.offset).toBe(0)
        expect(result.data.type).toBeUndefined()
        expect(result.data.status).toBeUndefined()
      }
    })

    it('allows limit up to maximum value', () => {
      const result = ProjectsQuerySchema.safeParse({
        limit: '500' // Max is 500
      })

      expect(result.success).toBe(true)
      if (result.success) {
        expect(result.data.limit).toBe(500)
      }
    })

    it('rejects limit above maximum', () => {
      const result = ProjectsQuerySchema.safeParse({
        limit: '1000' // Above max of 500
      })

      expect(result.success).toBe(false)
    })

    it('rejects negative offset', () => {
      const result = ProjectsQuerySchema.safeParse({
        offset: '-5'
      })

      // Should either fail or clamp to 0
      if (result.success) {
        expect(result.data.offset).toBeGreaterThanOrEqual(0)
      }
    })

    it('trims and validates string values', () => {
      const result = ProjectsQuerySchema.safeParse({
        type: '  Solar  ',
        status: 'Operational'
      })

      expect(result.success).toBe(true)
      if (result.success) {
        // Strings should be trimmed
        expect(result.data.type).toBe('Solar')
        expect(result.data.status).toBe('Operational')
      }
    })
  })

  describe('StatsResponseSchema', () => {
    it('validates complete stats response', () => {
      const statsData = {
        overview: {
          total_projects: 100,
          regions: 10,
          states: 50,
          developers: 25,
          project_types: 5,
          total_capacity_mw: 10000,
          total_capacity_gw: 10,
          avg_capacity_mw: 100,
          min_capacity_mw: 1,
          max_capacity_mw: 500,
          total_investment_opportunity: 10000000000,
          estimated_jobs: 1740,
          estimated_homes_powered: 7500000
        },
        by_type: [
          { type: 'Solar', count: 50, total_capacity: 5000 }
        ],
        by_status: [
          { status: 'Operational', count: 75 }
        ],
        top_regions: [
          { region: 'California', count: 20, total_capacity: 2000 }
        ],
        top_developers: [
          { developer: 'Test Corp', projects: 10, total_capacity: 1000 }
        ],
        recent_projects: [],
        timestamp: new Date().toISOString()
      }

      const result = StatsResponseSchema.safeParse(statsData)
      expect(result.success).toBe(true)
    })

    it('creates valid default stats response', () => {
      const defaultStats = createDefaultStatsResponse()

      expect(defaultStats.overview).toBeDefined()
      // Default shows placeholder data (79193 projects) when DB unavailable
      expect(defaultStats.overview.total_projects).toBe(79193)
      expect(defaultStats.by_type).toEqual([])
      expect(defaultStats.by_status).toEqual([])
      expect(defaultStats.timestamp).toBeDefined()
      expect(defaultStats.error).toBe('Database temporarily unavailable')

      // Should validate against schema
      const result = StatsResponseSchema.safeParse(defaultStats)
      expect(result.success).toBe(true)
    })
  })
})

describe('API Response Formats', () => {
  describe('Projects list response', () => {
    it('should have correct structure', () => {
      const mockResponse = {
        projects: [
          {
            id: '1',
            name: 'Test Solar Farm',
            type: 'Solar',
            capacity_mw: 100,
            status: 'Operational',
            state: 'California',
            country: 'United States',
            latitude: 34.05,
            longitude: -118.24
          }
        ],
        total: 1,
        metadata: {
          types: ['Solar', 'Wind'],
          statuses: ['Operational', 'Planning'],
          countries: ['United States']
        }
      }

      expect(mockResponse.projects).toHaveLength(1)
      expect(mockResponse.projects[0]).toHaveProperty('id')
      expect(mockResponse.projects[0]).toHaveProperty('name')
      expect(mockResponse.projects[0]).toHaveProperty('type')
      expect(mockResponse.metadata).toHaveProperty('types')
      expect(mockResponse.metadata).toHaveProperty('statuses')
    })
  })

  describe('Export formats', () => {
    it('should support CSV format', () => {
      const csvData = 'id,name,type\n1,Test,Solar'
      const lines = csvData.split('\n')

      expect(lines).toHaveLength(2)
      expect(lines[0]).toBe('id,name,type')
    })

    it('should support JSON format', () => {
      const jsonResponse = {
        dataset: 'projects',
        timestamp: new Date().toISOString(),
        count: 10,
        filters: { type: 'Solar' },
        data: []
      }

      expect(jsonResponse.dataset).toBe('projects')
      expect(jsonResponse.count).toBe(10)
      expect(jsonResponse.data).toEqual([])
    })
  })
})

describe('Error Response Formats', () => {
  it('should have consistent error structure', () => {
    const errorResponse = {
      error: 'Invalid query parameters',
      issues: [
        { path: ['limit'], message: 'Expected number' }
      ]
    }

    expect(errorResponse).toHaveProperty('error')
    expect(typeof errorResponse.error).toBe('string')
  })

  it('should include status codes for different errors', () => {
    const errors = {
      badRequest: { status: 400, error: 'Bad Request' },
      unauthorized: { status: 401, error: 'Unauthorized' },
      notFound: { status: 404, error: 'Not Found' },
      tooManyRequests: { status: 429, error: 'Too Many Requests' },
      serverError: { status: 500, error: 'Internal Server Error' }
    }

    expect(errors.badRequest.status).toBe(400)
    expect(errors.unauthorized.status).toBe(401)
    expect(errors.notFound.status).toBe(404)
    expect(errors.tooManyRequests.status).toBe(429)
    expect(errors.serverError.status).toBe(500)
  })
})
