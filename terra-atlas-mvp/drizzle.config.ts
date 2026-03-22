import { defineConfig } from 'drizzle-kit'

export default defineConfig({
  schema: './lib/drizzle/schema.ts',
  out: './lib/drizzle/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.DATABASE_URL || 'postgresql://tstoltz@localhost:5434/terra_atlas?host=/srv/luminous-dynamics/terra-atlas-mvp/postgres-data',
  },
  verbose: true,
  strict: true,
})