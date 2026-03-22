const publicEnv = {
  NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
  NEXT_PUBLIC_SUPABASE_ANON_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
} as const

type PublicEnvKey = keyof typeof publicEnv

function requirePublicEnv(key: PublicEnvKey): string {
  const value = publicEnv[key]
  if (!value) {
    throw new Error(`[env] Missing required environment variable: ${key}`)
  }
  return value
}

export const env = {
  NEXT_PUBLIC_SUPABASE_URL: requirePublicEnv('NEXT_PUBLIC_SUPABASE_URL'),
  NEXT_PUBLIC_SUPABASE_ANON_KEY: requirePublicEnv('NEXT_PUBLIC_SUPABASE_ANON_KEY'),
}

export type PublicEnv = typeof env
