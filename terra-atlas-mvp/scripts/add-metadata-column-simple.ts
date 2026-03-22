#!/usr/bin/env tsx
import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'

// Load .env.local
const envPath = path.join(process.cwd(), '.env.local')
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8')
  envContent.split('\n').forEach(line => {
    const match = line.match(/^([^=:#]+)=(.*)$/)
    if (match) {
      process.env[match[1].trim()] = match[2].trim().replace(/^["']|["']$/g, '')
    }
  })
}

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function addMetadataColumn() {
  console.log('📦 Adding metadata column to sites table...')

  // Check if column already exists
  const { data: columns } = await supabase
    .from('sites')
    .select('*')
    .limit(0)

  if (columns !== null) {
    console.log('✅ Sites table accessed successfully')
    console.log('\n📝 Please run this SQL in Supabase SQL Editor (https://fyyszjyixenujgbjaqkd.supabase.co/project/fyyszjyixenujgbjaqkd/editor):')
    console.log('\nALTER TABLE sites ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT \'{}\'::jsonb;')
    console.log('CREATE INDEX IF NOT EXISTS idx_sites_metadata ON sites USING GIN (metadata);')
    console.log('\nThen rerun the import script.')
  }
}

addMetadataColumn()
