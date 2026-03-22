const { createClient } = require('@supabase/supabase-js');
const fs = require('fs').promises;
require('dotenv').config({ path: '.env.local' });

async function importData() {
  console.log('🚀 Importing Terra Atlas Data to Supabase\n');
  
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  );
  
  try {
    // Load the enhanced data
    const data = JSON.parse(
      await fs.readFile('./data/enhanced-global-sites-v2.json', 'utf8')
    );
    
    console.log(`📊 Found ${data.sites.length} sites to import\n`);
    
    // Transform data for Supabase
    const sites = data.sites.map(site => ({
      name: site.name,
      slug: site.name.toLowerCase().replace(/\s+/g, '-'),
      site_type: site.type,
      status: site.status || 'operational',
      country: site.country,
      region: site.region,
      latitude: site.lat || site.coordinates?.lat || site.coordinates?.[1],
      longitude: site.lng || site.coordinates?.lng || site.coordinates?.[0],
      capacity_mw: site.capacity_mw,
      annual_generation_gwh: site.annual_generation_gwh,
      capacity_factor: site.capacity_factor,
      investment_required_usd: site.investment_required || site.investment_usd,
      commissioning_date: site.commissioning_date,
      tags: [site.type, site.country, site.status].filter(Boolean),
      data_completeness: 75,
      public_visible: true,
      featured: site.capacity_mw > 500
    }));
    
    // Insert in batches
    const batchSize = 20;
    let imported = 0;
    
    for (let i = 0; i < sites.length; i += batchSize) {
      const batch = sites.slice(i, i + batchSize);
      
      const { data, error } = await supabase
        .from('sites')
        .insert(batch)
        .select();
      
      if (error) {
        console.error(`❌ Error inserting batch ${i/batchSize + 1}:`, error.message);
      } else {
        imported += data.length;
        console.log(`✅ Imported ${imported}/${sites.length} sites...`);
      }
    }
    
    console.log('\n🎉 Import Complete!');
    console.log(`✅ Successfully imported ${imported} sites to Supabase`);
    
    // Verify
    const { count } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true });
    
    console.log(`\n📊 Total sites in database: ${count}`);
    
  } catch (error) {
    console.error('❌ Import failed:', error.message);
    process.exit(1);
  }
}

importData();