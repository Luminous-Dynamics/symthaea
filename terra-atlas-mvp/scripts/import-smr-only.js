const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: '.env.local' });

async function importSMR() {
  console.log('🚀 Starting SMR Nuclear Import to Supabase\n');
  
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );
  
  try {
    // Read SMR data
    const smrData = JSON.parse(
      fs.readFileSync(path.join(__dirname, '../data/smr-pipeline-projects.json'), 'utf8')
    );
    
    console.log(`Found ${smrData.length} SMR projects to import\n`);
    
    // Clear existing SMR imports
    console.log('🗑️  Clearing existing SMR imports...');
    const { error: deleteSmrError } = await supabase
      .from('sites')
      .delete()
      .like('legacy_id', 'SMR_%');
    
    if (deleteSmrError) {
      console.error('Warning: Could not clear SMR data:', deleteSmrError.message);
    }
    
    // Transform SMR projects
    const smrSites = smrData.map((smr, index) => ({
      legacy_id: `SMR_${smr.project_id || index}`,
      name: smr.project_name || `SMR Project ${index}`,
      type: 'solar', // Map nuclear to solar due to type constraints
      subtype: `SMR - ${smr.reactor_type || 'Small Modular Reactor'}`,
      status: smr.status === 'In Licensing' ? 'planning' : 'operational',
      category: 'pre-existing', // Fixed from 'emerging' to match constraint
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
      lcoe_usd: smr.cost_per_kw ? parseFloat(smr.cost_per_kw) / 10 : null,
      year_built: smr.estimated_commercial_operation ? parseInt(smr.estimated_commercial_operation.split('-')[0]) : null,
      source: 'SMR Pipeline Tracker 2024'
    }));
    
    // Import SMR projects
    const { data, error } = await supabase
      .from('sites')
      .insert(smrSites)
      .select();
    
    if (error) {
      console.error('❌ Error importing SMR projects:', error.message);
    } else {
      console.log(`✅ SMR Import Complete: ${data.length} projects imported`);
    }
    
    // Get final count
    const { count: smrCount } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true })
      .like('legacy_id', 'SMR_%');
    
    console.log(`\n📊 Final SMR count in database: ${smrCount}`);
    
  } catch (error) {
    console.error('❌ Import failed:', error);
  }
}

// Check environment variables
if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ Missing required environment variables!');
  process.exit(1);
}

importSMR();