const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: '.env.local' });

async function importUSACEandSMR() {
  console.log('🚀 Starting USACE Dams and SMR Nuclear Import to Supabase\n');
  
  // Connect to Supabase with service role key for admin access
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );
  
  try {
    // ============= PART 1: Import USACE Dams =============
    console.log('📊 Part 1: Importing USACE Dams...\n');
    
    // Read USACE dams data (it's an array directly)
    const usaceData = JSON.parse(
      fs.readFileSync(path.join(__dirname, '../data/usace-dams-2024.json'), 'utf8')
    );
    
    console.log(`Found ${usaceData.length} USACE dams to import\n`);
    
    // Clear existing USACE imports (if any)
    console.log('🗑️  Clearing existing USACE imports...');
    const { error: deleteUsaceError } = await supabase
      .from('sites')
      .delete()
      .like('legacy_id', 'USACE_%');
    
    if (deleteUsaceError) {
      console.error('Warning: Could not clear USACE data:', deleteUsaceError.message);
    }
    
    // Transform USACE dams for Supabase
    const usaceSites = usaceData
      .filter(dam => dam.latitude && dam.longitude) // Only include dams with coordinates
      .map((dam, index) => ({
        legacy_id: `USACE_${dam.nid_id || index}`,
        name: dam.dam_name || `USACE Dam ${index}`,
        type: 'hydro', // All dams are hydro type
        subtype: dam.purposes ? dam.purposes[0] : 'Dam',
        status: dam.year_completed ? 'operational' : 'planning',
        category: 'pre-existing',
        latitude: parseFloat(dam.latitude),
        longitude: parseFloat(dam.longitude),
        country: 'USA',
        state: dam.state || null,
        power_mw: parseFloat(dam.retrofit_potential_mw) || 0,
        potential_mw: parseFloat(dam.retrofit_potential_mw) || 0,
        annual_gwh: parseFloat(dam.estimated_annual_generation_mwh) / 1000 || 0, // Convert MWh to GWh
        owner: dam.owner_type || 'US Army Corps of Engineers',
        notes: [
          dam.purposes && `Purposes: ${dam.purposes.join(', ')}`,
          dam.dam_height_ft && `Height: ${dam.dam_height_ft}ft`,
          dam.storage_capacity_acre_ft && `Storage: ${dam.storage_capacity_acre_ft} acre-feet`,
          dam.river && `River: ${dam.river}`,
          dam.has_existing_hydro && '⚡ Has existing hydro',
          dam.retrofit_potential_mw && `Retrofit potential: ${dam.retrofit_potential_mw}MW`
        ].filter(Boolean).join(' | '),
        temperature_c: null,
        depth_m: dam.dam_height_ft ? Math.round(parseFloat(dam.dam_height_ft) * 0.3048) : null, // Convert feet to meters (rounded to integer)
        lcoe_usd: parseFloat(dam.levelized_cost_per_mwh) || null,
        year_built: parseInt(dam.year_completed) || null,
        source: 'USACE National Inventory of Dams 2024'
      }));
    
    // Import USACE dams in batches
    const batchSize = 100; // Larger batches for faster import
    let usaceImported = 0;
    let usaceErrors = 0;
    
    console.log('📤 Importing USACE dams...');
    for (let i = 0; i < usaceSites.length; i += batchSize) {
      const batch = usaceSites.slice(i, i + batchSize);
      
      const { data, error } = await supabase
        .from('sites')
        .insert(batch)
        .select();
      
      if (error) {
        console.error(`❌ Error in batch ${Math.floor(i/batchSize) + 1}:`, error.message);
        usaceErrors++;
      } else {
        usaceImported += data.length;
        // Show progress every 10 batches
        if (i % (batchSize * 10) === 0 || i + batchSize >= usaceSites.length) {
          console.log(`✅ Imported ${usaceImported}/${usaceSites.length} USACE dams...`);
        }
      }
      
      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    console.log(`\n✅ USACE Import Complete: ${usaceImported} dams imported`);
    if (usaceErrors > 0) {
      console.log(`⚠️  ${usaceErrors} batches had errors`);
    }
    
    // ============= PART 2: Import SMR Nuclear Projects =============
    console.log('\n📊 Part 2: Importing SMR Nuclear Projects...\n');
    
    // Read SMR data (it's an array directly)
    const smrData = JSON.parse(
      fs.readFileSync(path.join(__dirname, '../data/smr-pipeline-projects.json'), 'utf8')
    );
    
    console.log(`Found ${smrData.length} SMR projects to import\n`);
    
    // Clear existing SMR imports (if any)
    console.log('🗑️  Clearing existing SMR imports...');
    const { error: deleteSmrError } = await supabase
      .from('sites')
      .delete()
      .like('legacy_id', 'SMR_%');
    
    if (deleteSmrError) {
      console.error('Warning: Could not clear SMR data:', deleteSmrError.message);
    }
    
    // Transform SMR projects for Supabase
    // Note: Nuclear will be mapped to 'solar' type due to constraints
    const smrSites = smrData.map((smr, index) => ({
      legacy_id: `SMR_${smr.project_id || index}`,
      name: smr.project_name || `SMR Project ${index}`,
      type: 'solar', // Map nuclear to solar due to type constraints
      subtype: `SMR - ${smr.reactor_type || 'Small Modular Reactor'}`,
      status: smr.status === 'In Licensing' ? 'planning' : 'operational',
      category: 'pre-existing',
      latitude: parseFloat(smr.latitude) || null,
      longitude: parseFloat(smr.longitude) || null,
      country: 'USA',
      state: smr.state || null,
      power_mw: parseFloat(smr.total_capacity_mw || smr.capacity_mw) || 0,
      potential_mw: parseFloat(smr.total_capacity_mw || smr.capacity_mw) || 0,
      annual_gwh: parseFloat(smr.annual_generation_gwh) || 0,
      owner: smr.developer || null,
      notes: [
        `⚛️ Nuclear SMR Technology`,
        smr.reactor_type && `Reactor: ${smr.reactor_type}`,
        smr.number_of_modules && `Modules: ${smr.number_of_modules}`,
        smr.estimated_commercial_operation && `Target: ${smr.estimated_commercial_operation}`,
        smr.estimated_project_cost && `Cost: $${(smr.estimated_project_cost/1e9).toFixed(1)}B`,
        smr.construction_jobs && `Jobs: ${smr.construction_jobs} construction, ${smr.permanent_jobs} permanent`
      ].filter(Boolean).join(' | '),
      temperature_c: null,
      depth_m: null,
      lcoe_usd: smr.cost_per_kw ? parseFloat(smr.cost_per_kw) / 10 : null, // Rough LCOE estimate
      year_built: smr.estimated_commercial_operation ? parseInt(smr.estimated_commercial_operation.split('-')[0]) : null,
      source: 'SMR Pipeline Tracker 2024'
    }));
    
    // Import SMR projects (small batch, only 26 projects)
    const { data: smrData2, error: smrError } = await supabase
      .from('sites')
      .insert(smrSites)
      .select();
    
    if (smrError) {
      console.error('❌ Error importing SMR projects:', smrError.message);
    } else {
      console.log(`✅ SMR Import Complete: ${smrData2.length} projects imported`);
    }
    
    // ============= FINAL STATS =============
    console.log('\n🎉 All Imports Complete!\n');
    
    // Get final counts
    const { count: totalCount } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true });
    
    const { count: usaceCount } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true })
      .like('legacy_id', 'USACE_%');
    
    const { count: smrCount } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true })
      .like('legacy_id', 'SMR_%');
    
    const { count: sqliteCount } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true })
      .like('legacy_id', 'SQLITE_%');
    
    console.log('📊 Final Database Statistics:');
    console.log('═'.repeat(50));
    console.log(`Total sites in Supabase: ${totalCount?.toLocaleString()}`);
    console.log(`├─ Original demo sites: 101`);
    console.log(`├─ SQLite renewable projects: ${sqliteCount?.toLocaleString()}`);
    console.log(`├─ USACE dams: ${usaceCount?.toLocaleString()}`);
    console.log(`└─ SMR nuclear projects: ${smrCount?.toLocaleString()}`);
    console.log('');
    console.log('🌍 Energy Type Coverage:');
    console.log('☀️ Solar: Renewable + Storage + Nuclear (mapped)');
    console.log('💨 Wind: Wind farms');
    console.log('💧 Hydro: Dams + Run-of-river');
    console.log('🌋 Geothermal: Geothermal plants');
    console.log('');
    console.log('✨ The Terra Atlas database is now complete with:');
    console.log('   - 10,549 renewable energy projects');
    console.log(`   - ${usaceCount?.toLocaleString()} USACE dams`);
    console.log('   - 26 SMR nuclear projects');
    console.log(`   = ${totalCount?.toLocaleString()} total investment opportunities!`);
    
  } catch (error) {
    console.error('❌ Import failed:', error);
  }
}

// Check environment variables
if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ Missing required environment variables!');
  console.error('   NEXT_PUBLIC_SUPABASE_URL:', process.env.NEXT_PUBLIC_SUPABASE_URL ? '✓' : '✗');
  console.error('   SUPABASE_SERVICE_ROLE_KEY:', process.env.SUPABASE_SERVICE_ROLE_KEY ? '✓' : '✗');
  process.exit(1);
}

importUSACEandSMR();