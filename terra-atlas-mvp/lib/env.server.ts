import { env as publicEnv } from './env'

const serverOnlyEnv = {
  SUPABASE_SERVICE_ROLE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY,
  STRIPE_SECRET_KEY: process.env.STRIPE_SECRET_KEY,
  STRIPE_WEBHOOK_SECRET: process.env.STRIPE_WEBHOOK_SECRET,
  DATABASE_URL: process.env.DATABASE_URL,
  JWT_SECRET: process.env.JWT_SECRET,
} as const

type ServerOnlyEnvKey = keyof typeof serverOnlyEnv

function normalize(value: string | undefined | null) {
  return typeof value === 'string' ? value.trim() : value ?? undefined
}

function readServerEnv(key: ServerOnlyEnvKey, options?: { optional?: boolean }): string | undefined {
  const value = normalize(serverOnlyEnv[key])

  if (!value && !options?.optional) {
    throw new Error(`[env] Missing required server environment variable: ${key}`)
  }

  return value
}

export function requireServerEnv(key: ServerOnlyEnvKey): string {
  const value = readServerEnv(key)
  if (!value) {
    throw new Error(`[env] Missing required server environment variable: ${key}`)
  }
  return value
}

export const serverEnv = {
  ...publicEnv,
  SUPABASE_SERVICE_ROLE_KEY: readServerEnv('SUPABASE_SERVICE_ROLE_KEY', { optional: true }),
  DATABASE_URL: readServerEnv('DATABASE_URL', { optional: true }),
  JWT_SECRET: readServerEnv('JWT_SECRET', { optional: true }),
  STRIPE_SECRET_KEY: readServerEnv('STRIPE_SECRET_KEY', { optional: true }),
  STRIPE_WEBHOOK_SECRET: readServerEnv('STRIPE_WEBHOOK_SECRET', { optional: true }),
}

export type ServerEnv = typeof serverEnv
