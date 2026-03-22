import { z } from 'zod'

export const StatsOverviewSchema = z.object({
  total_projects: z.number(),
  regions: z.number(),
  states: z.number(),
  developers: z.number(),
  project_types: z.number(),
  total_capacity_mw: z.number(),
  total_capacity_gw: z.number(),
  avg_capacity_mw: z.number(),
  min_capacity_mw: z.number(),
  max_capacity_mw: z.number(),
  total_investment_opportunity: z.number(),
  estimated_jobs: z.number(),
  estimated_homes_powered: z.number(),
})

const numberOrNull = z.number().nullable()

export const StatsResponseSchema = z.object({
  overview: StatsOverviewSchema,
  by_type: z.array(
    z.object({
      type: z.string().nullable(),
      count: numberOrNull,
      total_capacity: numberOrNull,
    })
  ),
  by_status: z.array(
    z.object({
      status: z.string().nullable(),
      count: numberOrNull,
    })
  ),
  top_regions: z.array(
    z.object({
      region: z.string().nullable(),
      count: numberOrNull,
      total_capacity: numberOrNull,
    })
  ),
  top_developers: z.array(
    z.object({
      developer: z.string().nullable(),
      projects: numberOrNull,
      total_capacity: numberOrNull,
    })
  ),
  recent_projects: z.array(z.record(z.any())),
  timestamp: z.string(),
  error: z.string().optional(),
})

export type StatsResponse = z.infer<typeof StatsResponseSchema>
export type StatsOverview = z.infer<typeof StatsOverviewSchema>

export const defaultStatsOverview: StatsOverview = {
  total_projects: 79193,
  regions: 10,
  states: 50,
  developers: 1500,
  project_types: 6,
  total_capacity_mw: 2_955_700,
  total_capacity_gw: 2_956,
  avg_capacity_mw: 37.3,
  min_capacity_mw: 0.1,
  max_capacity_mw: 2000,
  total_investment_opportunity: 2_956_000_000_000,
  estimated_jobs: 1_378_000,
  estimated_homes_powered: 2_217_000_000,
}

export function createDefaultStatsResponse(): StatsResponse {
  return {
    overview: defaultStatsOverview,
    by_type: [],
    by_status: [],
    top_regions: [],
    top_developers: [],
    recent_projects: [],
    timestamp: new Date().toISOString(),
    error: 'Database temporarily unavailable',
  }
}
