#!/usr/bin/env tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Get the exact schema from Supabase using their API
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
  console.log('🔍 Getting Exact Schema from Supabase');
  console.log('=====================================');
  
  try {
    // Try inserting with absolutely minimal fields
    const testProject = {
      name: 'Test',
      type: 'solar',
      status: 'planning',
      latitude: 40.0,
      longitude: -74.0
    };
    
    console.log('\nTrying minimal insert with only required fields:');
    console.log(JSON.stringify(testProject, null, 2));
    
    const { data, error } = await supabase
      .from('projects')
      .insert(testProject)
      .select();
    
    if (error) {
      console.log('\n❌ Error:', error.message);
      
      // Try even more minimal
      const minimalProject = {
        name: 'Test'
      };
      
      console.log('\nTrying ultra-minimal with just name:');
      const { error: minError } = await supabase
        .from('projects')
        .insert(minimalProject)
        .select();
      
      if (minError) {
        console.log('❌ Still failed:', minError.message);
      }
    } else {
      console.log('\n✅ Success! These columns work:');
      console.log(Object.keys(testProject).join(', '));
      
      if (data && data[0]) {
        console.log('\n📋 Actual returned data:');
        console.log(JSON.stringify(data[0], null, 2));
        console.log('\n✅ Available columns in response:');
        console.log(Object.keys(data[0]).join(', '));
        
        // Clean up
        await supabase.from('projects').delete().eq('id', data[0].id);
      }
    }
    
    // Also try to query to see what comes back
    console.log('\n📊 Attempting to select all columns:');
    const { data: queryData, error: queryError } = await supabase
      .from('projects')
      .select('*')
      .limit(1);
    
    if (!queryError && queryData) {
      if (queryData.length > 0) {
        console.log('Found existing data with columns:', Object.keys(queryData[0]).join(', '));
      } else {
        console.log('Table is empty but query succeeded');
      }
    }
    
  } catch (err) {
    console.error('Fatal error:', err);
  }
}

main();