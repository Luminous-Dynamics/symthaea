require('dotenv').config({ path: '.env.local' });
const https = require('https');
const fs = require('fs');

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !serviceKey) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

// Fetch sites with ROI >= 11% (using expected_roi_percentage)
const url = `${supabaseUrl}/rest/v1/sites?select=id,name,latitude,longitude,power_mw,potential_mw,annual_gwh,type,subtype,status,country,lcoe_usd,expected_roi_percentage&expected_roi_percentage=gte.11&limit=200`;

const options = {
  headers: {
    'apikey': serviceKey,
    'Authorization': `Bearer ${serviceKey}`
  }
};

https.get(url, options, (res) => {
  let data = '';
  res.on('data', (chunk) => data += chunk);
  res.on('end', () => {
    if (res.statusCode === 200) {
      const sites = JSON.parse(data);
      
      // Map to the format TerraGlobeWithSites expects
      const formattedSites = sites.map(site => ({
        id: site.id,
        name: site.name,
        latitude: site.latitude,
        longitude: site.longitude,
        estimated_capacity_mw: site.power_mw || site.potential_mw || 1,
        estimated_irr: site.expected_roi_percentage || 11,
        type: site.type,
        subtype: site.subtype,
        status: site.status,
        country: site.country,
        lcoe_usd: site.lcoe_usd,
        annual_gwh: site.annual_gwh
      }));
      
      console.error(`✅ Fetched ${formattedSites.length} viable energy sites (ROI >= 11%)`);
      
      // Calculate stats
      const totalCapacity = formattedSites.reduce((sum, s) => sum + (s.estimated_capacity_mw || 0), 0);
      const avgIRR = formattedSites.reduce((sum, s) => sum + (s.estimated_irr || 0), 0) / formattedSites.length;
      
      console.error(`Total capacity: ${totalCapacity.toFixed(0)} MW`);
      console.error(`Average IRR: ${avgIRR.toFixed(1)}%`);
      
      // Output formatted JSON
      console.log(JSON.stringify(formattedSites, null, 2));
    } else {
      console.error('❌ Failed:', data);
      process.exit(1);
    }
  });
}).on('error', (err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
