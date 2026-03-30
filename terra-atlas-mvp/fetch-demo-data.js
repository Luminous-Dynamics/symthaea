// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
require('dotenv').config({ path: '.env.local' });
const https = require('https');

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !serviceKey) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

const url = `${supabaseUrl}/rest/v1/sites?select=*&estimated_irr=gte.11&limit=200`;

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
      console.log(`✅ Fetched ${sites.length} viable energy sites (IRR >= 11%)`);
      console.log(JSON.stringify(sites, null, 2));
    } else {
      console.error('❌ Failed:', data);
    }
  });
}).on('error', (err) => {
  console.error('Error:', err.message);
});
