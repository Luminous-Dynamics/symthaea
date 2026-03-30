#!/usr/bin/env tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Check what columns actually exist in the Supabase projects table
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
  console.log('🔍 Checking Supabase Projects Table Schema');
  console.log('==========================================');
  
  try {
    // Try to select one row to see what columns exist
    const { data, error } = await supabase
      .from('projects')
      .select('*')
      .limit(1);
    
    if (error) {
      console.error('Error fetching sample:', error);
    } else if (data && data.length > 0) {
      console.log('\n✅ Available columns:');
      console.log(Object.keys(data[0]).join(', '));
      
      console.log('\n📋 Sample row structure:');
      console.log(JSON.stringify(data[0], null, 2));
    } else {
      console.log('\nℹ️ Table exists but is empty');
      
      // Try to insert a minimal row to see what's required
      const testRow = {
        name: 'Test Project',
        type: 'solar',
        status: 'planning',
        latitude: 40.7128,
        longitude: -74.0060,
        capacity_mw: 100,
        metadata: { test: true }
      };
      
      const { data: insertData, error: insertError } = await supabase
        .from('projects')
        .insert(testRow)
        .select();
      
      if (insertError) {
        console.log('\n❌ Insert test failed:', insertError.message);
        console.log('This tells us which columns are missing or have issues');
      } else {
        console.log('\n✅ Minimal insert succeeded with these columns:');
        console.log(Object.keys(testRow).join(', '));
        
        // Clean up test row
        if (insertData && insertData[0]?.id) {
          await supabase.from('projects').delete().eq('id', insertData[0].id);
        }
      }
    }
    
  } catch (err) {
    console.error('Fatal error:', err);
  }
}

main();