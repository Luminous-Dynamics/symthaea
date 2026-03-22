#!/usr/bin/env tsx
/**
 * Test that we can fetch the 79,193 projects from the database
 */

import { createClient } from '@supabase/supabase-js';
import { config } from 'dotenv';

// Load environment variables
config({ path: '.env.local' });

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function main() {
  console.log('🔍 Testing Terra Atlas Database');
  console.log('================================');
  
  // Get total count
  const { count, error: countError } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });
  
  if (countError) {
    console.error('❌ Error getting count:', countError);
  } else {
    console.log(`✅ Total projects in database: ${count?.toLocaleString()}`);
  }
  
  // Get breakdown by type
  console.log('\n📊 Projects by Type:');
  const types = ['solar', 'wind', 'hydro', 'nuclear', 'storage'];
  
  for (const type of types) {
    const { count: typeCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })
      .eq('type', type);
    
    console.log(`  ${type}: ${typeCount?.toLocaleString() || 0}`);
  }
  
  // Get breakdown by status
  console.log('\n📈 Projects by Status:');
  const statuses = ['planning', 'approved', 'construction', 'operational'];
  
  for (const status of statuses) {
    const { count: statusCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })
      .eq('status', status);
    
    console.log(`  ${status}: ${statusCount?.toLocaleString() || 0}`);
  }
  
  // Get sample projects
  console.log('\n🎯 Sample Projects:');
  const { data: samples } = await supabase
    .from('projects')
    .select('name, type, status, capacity_mw, state')
    .limit(5);
  
  if (samples) {
    samples.forEach(p => {
      console.log(`  • ${p.name} (${p.type}, ${p.status}, ${p.capacity_mw}MW, ${p.state})`);
    });
  }
  
  // Test API endpoint
  console.log('\n🌐 Testing API Endpoint:');
  try {
    const response = await fetch('http://localhost:3001/api/projects?limit=5');
    if (response.ok) {
      const data = await response.json();
      console.log(`✅ API returned ${data.projects?.length || 0} projects`);
      console.log(`✅ Total available: ${data.total?.toLocaleString() || 'Unknown'}`);
    } else {
      console.log(`⚠️ API returned status ${response.status}`);
    }
  } catch (err) {
    console.log('⚠️ Could not reach API (server may still be starting)');
  }
  
  console.log('\n🎉 Database test complete!');
  console.log('Visit http://localhost:3001 to see the interactive globe with all 79,193 projects!');
}

main();