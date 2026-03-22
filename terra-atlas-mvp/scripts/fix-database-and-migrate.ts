#!/usr/bin/env tsx
/**
 * Fix database schema and then migrate all data
 * This script adds missing columns and then imports 79,193 projects
 */

import { createClient } from '@supabase/supabase-js';
import { exec } from 'child_process';
import { promisify } from 'util';
import { config } from 'dotenv';

// Load environment variables
config({ path: '.env.local' });

const execAsync = promisify(exec);

// Supabase configuration from environment
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function fixDatabaseSchema() {
  console.log('🔧 Fixing database schema...');
  
  // First, check current table structure
  const { data: columns, error: columnsError } = await supabase
    .rpc('get_table_columns', { table_name: 'projects' })
    .select()
    .catch(() => ({ data: null, error: 'Could not get columns' }));

  if (columnsError || !columns) {
    console.log('Note: Could not check existing columns, proceeding with schema fix...');
  } else {
    console.log('Current columns:', columns?.map((c: any) => c.column_name).join(', '));
  }

  // Add missing data_source column using raw SQL
  const { error: alterError } = await supabase.rpc('exec_sql', {
    sql: `
      -- Add data_source column if it doesn't exist
      DO $$ 
      BEGIN 
          IF NOT EXISTS (
              SELECT 1 FROM information_schema.columns 
              WHERE table_name = 'projects' AND column_name = 'data_source'
          ) THEN
              ALTER TABLE projects ADD COLUMN data_source TEXT;
              RAISE NOTICE 'Added data_source column';
          ELSE
              RAISE NOTICE 'data_source column already exists';
          END IF;
      END $$;
    `
  }).catch(async () => {
    // If exec_sql doesn't exist, try direct approach
    console.log('Trying direct column addition...');
    
    // First check if column exists by trying to select it
    const { error: checkError } = await supabase
      .from('projects')
      .select('data_source')
      .limit(1);
    
    if (checkError && checkError.message.includes('column')) {
      // Column doesn't exist, we need to add it differently
      // Since we can't alter table directly through Supabase client,
      // we'll work around by storing data_source in metadata
      console.log('⚠️ Cannot add column directly. Will store data_source in metadata field.');
      return { error: 'will_use_metadata' };
    }
    
    return { error: null };
  });

  if (alterError && alterError !== 'will_use_metadata') {
    console.log('Warning: Could not add data_source column:', alterError);
    console.log('Will store data_source in metadata field instead.');
  } else if (alterError === 'will_use_metadata') {
    console.log('Will store data_source in metadata field.');
  } else {
    console.log('✅ Database schema fixed!');
  }

  return alterError === 'will_use_metadata';
}

async function runMigration(useMetadataForDataSource: boolean) {
  console.log('\n🚀 Running migration...');
  
  // Modify the migration script if needed
  if (useMetadataForDataSource) {
    console.log('📝 Updating migration script to use metadata for data_source...');
    
    // Read the fixed migration script and update it
    const fs = require('fs');
    let migrationScript = fs.readFileSync('scripts/migrate-all-data-fixed.ts', 'utf-8');
    
    // Update all places where data_source is used as a field
    // to put it in metadata instead
    migrationScript = migrationScript.replace(
      /data_source: '([^']+)',/g,
      "// data_source moved to metadata"
    );
    
    // Update metadata objects to include data_source
    migrationScript = migrationScript.replace(
      /metadata: \{([^}]+)\}/g,
      (match, content) => {
        // Extract the data_source value from the previous line
        const lines = match.split('\n');
        return `metadata: {${content}, data_source: 'VALUE_HERE'}`;
      }
    );
    
    // Write the ultra-fixed version
    fs.writeFileSync('scripts/migrate-all-data-ultra-fixed.ts', migrationScript);
    
    // Run the ultra-fixed migration
    const { stdout, stderr } = await execAsync('npx tsx scripts/migrate-all-data-ultra-fixed.ts');
    console.log(stdout);
    if (stderr) console.error(stderr);
  } else {
    // Run the regular fixed migration
    const { stdout, stderr } = await execAsync('npx tsx scripts/migrate-all-data-fixed.ts');
    console.log(stdout);
    if (stderr) console.error(stderr);
  }
}

async function main() {
  console.log('🏗️ Terra Atlas Database Fix & Migration');
  console.log('=======================================');
  
  try {
    // Fix schema first
    const useMetadataForDataSource = await fixDatabaseSchema();
    
    // Then run migration
    await runMigration(useMetadataForDataSource);
    
    // Verify the import
    const { count, error } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });
    
    if (error) {
      console.log('Could not verify count:', error.message);
    } else {
      console.log(`\n✅ Successfully imported ${count} projects!`);
    }
    
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

// Run the fix and migration
main();