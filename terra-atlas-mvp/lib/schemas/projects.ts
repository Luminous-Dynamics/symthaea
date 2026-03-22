import { z } from 'zod'

const optionalText = (max = 120) =>
  z
    .preprocess((value) => {
      if (typeof value !== 'string') return undefined
      const trimmed = value.trim()
      return trimmed.length ? trimmed : undefined
    }, z.string().max(max))
    .optional()

const numberWithDefault = (fallback: number, { min, max }: { min?: number; max?: number }) =>
  z.preprocess((value) => {
    if (typeof value === 'number' && !Number.isNaN(value)) {
      return value
    }

    if (typeof value === 'string') {
      const trimmed = value.trim()
      if (!trimmed) return fallback
      const parsed = Number(trimmed)
      return Number.isNaN(parsed) ? value : parsed
    }

    if (value === undefined || value === null) {
      return fallback
    }

    return value
  }, z.number().int().min(min ?? Number.MIN_SAFE_INTEGER).max(max ?? Number.MAX_SAFE_INTEGER))

export const ProjectsQuerySchema = z.object({
  limit: numberWithDefault(100, { min: 1, max: 500 }),
  offset: numberWithDefault(0, { min: 0, max: 5000 }),
  type: optionalText(),
  status: optionalText(),
  country: optionalText(),
  search: optionalText(200),
})

export type ProjectsQuery = z.infer<typeof ProjectsQuerySchema>
