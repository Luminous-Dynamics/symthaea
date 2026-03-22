const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
require('dotenv').config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function importAllRenewableSites() {
  console.log('🌍 Importing ALL renewable energy sites from unified-all-sites.json\n');
  
  // Load the unified data file
  const unifiedData = JSON.parse(fs.readFileSync('./data/unified-all-sites.json', 'utf8'));
  
  console.log(`📊 Found ${unifiedData.sites.length} total sites to import\n`);
  
  // Group by type for reporting
  const typeCount = {};
  unifiedData.sites.forEach(site => {
    typeCount[site.type] = (typeCount[site.type] || 0) + 1;
  });
  
  console.log('Site breakdown by type:');
  Object.entries(typeCount).forEach(([type, count]) => {
    console.log(`  ${type}: ${count}`);
  });
  console.log('');
  
  // Prepare sites for batch insert
  const sitesToInsert = unifiedData.sites.map(site => ({
    id: site.id,
    name: site.name || 'Unnamed Site',
    type: site.type || 'unknown',
    subtype: site.subtype || null,
    country: site.country || 'Unknown',
    state: site.state || site.region || null,
    latitude: parseFloat(site.latitude) || parseFloat(site.lat) || 0,
    longitude: parseFloat(site.longitude) || parseFloat(site.lng) || parseFloat(site.lon) || 0,
    power_mw: parseFloat(site.power_mw) || parseFloat(site.capacity_mw) || parseFloat(site.capacity) || 0,
    annual_generation_gwh: parseFloat(site.annual_generation_gwh) || parseFloat(site.annual_gwh) || null,
    capacity_factor: parseFloat(site.capacity_factor) || null,
    status: site.status || 'operational',
    owner: site.owner || site.operator || null,
    developer: site.developer || null,
    investment_usd: parseInt(site.investment_usd) || parseInt(site.investment_required) || parseInt(site.cost_usd) || null,
    lcoe_usd: parseFloat(site.lcoe_usd) || parseFloat(site.lcoe) || null,
    construction_start: parseInt(site.construction_start) || parseInt(site.start_year) || null,
    operational_year: parseInt(site.operational_year) || parseInt(site.commissioned) || parseInt(site.year) || null,
    renewable_integration_mw: parseFloat(site.renewable_integration_mw) || null,
    storage_capacity_mwh: parseFloat(site.storage_capacity_mwh) || parseFloat(site.storage_mwh) || null,
    co2_avoided_tons_year: parseInt(site.co2_avoided_tons_year) || parseInt(site.co2_reduction) || null,
    jobs_construction: parseInt(site.jobs_construction) || null,
    jobs_permanent: parseInt(site.jobs_permanent) || parseInt(site.jobs_operational) || null
  }));
  
  console.log('📤 Importing sites to Supabase...\n');
  
  // Import in batches of 50 to avoid timeout
  const batchSize = 50;
  let successCount = 0;
  let errorCount = 0;
  
  for (let i = 0; i < sitesToInsert.length; i += batchSize) {
    const batch = sitesToInsert.slice(i, i + batchSize);
    
    try {
      const { data, error } = await supabase
        .from('sites')
        .upsert(batch, { onConflict: 'id' });
      
      if (error) {
        console.error(`❌ Error in batch ${i / batchSize + 1}:`, error.message);
        errorCount += batch.length;
      } else {
        successCount += batch.length;
        console.log(`✅ Batch ${i / batchSize + 1}: Imported ${batch.length} sites`);
      }
    } catch (err) {
      console.error(`❌ Error in batch ${i / batchSize + 1}:`, err.message);
      errorCount += batch.length;
    }
  }
  
  console.log('\n📊 Import Summary:');
  console.log(`✅ Successfully imported: ${successCount} sites`);
  console.log(`❌ Failed to import: ${errorCount} sites`);
  console.log(`📈 Total processed: ${sitesToInsert.length} sites`);
  
  // Verify the import
  const { count } = await supabase
    .from('sites')
    .select('*', { count: 'exact', head: true });
  
  console.log(`\n🎯 Total sites now in database: ${count}`);
  
  // Get breakdown by type from database
  const { data: typeStats } = await supabase
    .from('sites')
    .select('type')
    .order('type');
  
  if (typeStats) {
    const dbTypes = {};
    typeStats.forEach(row => {
      dbTypes[row.type] = (dbTypes[row.type] || 0) + 1;
    });
    
    console.log('\n📊 Database breakdown by type:');
    Object.entries(dbTypes).forEach(([type, count]) => {
      console.log(`  ${type}: ${count}`);
    });
  }
  
  console.log('\n✨ Import complete! Your database now has comprehensive renewable energy data.');
  console.log('🌐 View at: https://atlas.luminousdynamics.io');
}

importAllRenewableSites().catch(console.error);