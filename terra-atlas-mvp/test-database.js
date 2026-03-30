// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Simple database connectivity test for Terra Atlas
const { Client } = require('pg');

async function testDatabase() {
  const client = new Client({
    host: 'localhost',
    port: 5434,
    database: 'terra_atlas',
    user: process.env.USER || 'tstoltz',
    connectionString: process.env.DATABASE_URL
  });

  try {
    console.log('🔌 Connecting to PostgreSQL...');
    await client.connect();
    console.log('✅ Connected successfully!');

    // Test basic query
    const result = await client.query('SELECT NOW() as current_time');
    console.log('📅 Current database time:', result.rows[0].current_time);

    // Count tables
    const tablesResult = await client.query(`
      SELECT COUNT(*) as table_count 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
    `);
    console.log('📊 Number of tables:', tablesResult.rows[0].table_count);

    // List tables
    const tableListResult = await client.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      ORDER BY table_name
    `);
    console.log('\n📋 Tables in database:');
    tableListResult.rows.forEach(row => {
      console.log('  -', row.table_name);
    });

    // Check data sources
    const sourcesResult = await client.query('SELECT name FROM data_sources');
    console.log('\n🌐 Data sources configured:');
    sourcesResult.rows.forEach(row => {
      console.log('  -', row.name);
    });

    console.log('\n✅ Database is properly configured and ready!');
    console.log('🌍 Terra Atlas database setup complete.');

  } catch (error) {
    console.error('❌ Database connection error:', error.message);
    console.error('💡 Tip: Make sure PostgreSQL is running on port 5434');
  } finally {
    await client.end();
  }
}

// Run the test
testDatabase();