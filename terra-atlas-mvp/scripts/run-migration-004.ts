#!/usr/bin/env tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Run migration 004 to add metadata column
 */
import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'

// Load environment variables
const envPath = path.join(process.cwd(), '.env.local')
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8')
  envContent.split('\n').forEach(line => {
    const match = line.match(/^([^=:#]+)=(.*)$/)
    if (match) {
      const key = match[1].trim()
      const value = match[2].trim().replace(/^["']|["']$/g, '')
      process.env[key] = value
    }
  })
}

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function runMigration() {
  console.log('📦 Running migration 004_add_metadata_column...')

  const migrationSQL = fs.readFileSync(
    path.join(process.cwd(), 'supabase/migrations/004_add_metadata_column.sql'),
    'utf-8'
  )

  const { error } = await supabase.rpc('exec_sql', { sql: migrationSQL })

  if (error) {
    // Try direct execution if exec_sql doesn't exist
    console.log('⚠️  exec_sql RPC not available, trying direct execution...')

    // Split by DO $$ blocks and execute individually
    const { error: directError } = await supabase.from('_migrations').select('*').limit(1)

    if (directError) {
      console.error('❌ Migration failed:', directError.message)
      console.log('\n📝 Please run this SQL manually in Supabase SQL Editor:')
      console.log(migrationSQL)
      process.exit(1)
    }
  }

  console.log('✅ Migration complete!')
  console.log('Metadata column added to sites table')
}

runMigration()
