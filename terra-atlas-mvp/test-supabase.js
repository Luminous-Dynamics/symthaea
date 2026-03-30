// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Quick Supabase connection test
require('dotenv').config({ path: '.env.local' });
const { createClient } = require('@supabase/supabase-js')

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

async function testConnection() {
  console.log('🔍 Testing Supabase connection...\n')

  const supabase = createClient(supabaseUrl, supabaseAnonKey)

  // Test 1: Check connection
  console.log('Test 1: Basic connection')
  console.log('URL:', supabaseUrl)
  console.log('Key:', supabaseAnonKey.substring(0, 20) + '...\n')

  // Test 2: List tables
  console.log('Test 2: Query sites table')
  try {
    const { data, error, count } = await supabase
      .from('sites')
      .select('*', { count: 'exact' })
      .limit(5)

    if (error) {
      console.error('❌ Error:', error)
      console.error('Error code:', error.code)
      console.error('Error message:', error.message)
      console.error('Error details:', error.details)
      console.error('Error hint:', error.hint)
    } else {
      console.log(`✅ Success! Found ${count} total sites`)
      console.log(`Fetched ${data.length} sample sites:`)
      data.forEach((site, i) => {
        console.log(`  ${i + 1}. ${site.name || 'Unnamed'} - ${site.latitude}, ${site.longitude} - IRR: ${site.estimated_irr}%`)
      })
    }
  } catch (err) {
    console.error('❌ Exception:', err)
  }

  // Test 3: Filtered query (what TerraGlobeWithSites uses)
  console.log('\nTest 3: Filtered query (IRR >= 11%)')
  try {
    const { data, error, count } = await supabase
      .from('sites')
      .select('*', { count: 'exact' })
      .gte('estimated_irr', 11)
      .limit(5)

    if (error) {
      console.error('❌ Error:', error)
    } else {
      console.log(`✅ Success! Found ${count} viable sites (IRR >= 11%)`)
      console.log(`Fetched ${data.length} sample sites`)
    }
  } catch (err) {
    console.error('❌ Exception:', err)
  }
}

testConnection().then(() => {
  console.log('\n✅ Test complete')
  process.exit(0)
}).catch(err => {
  console.error('\n❌ Test failed:', err)
  process.exit(1)
})
