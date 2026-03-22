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

const url = `${supabaseUrl}/rest/v1/sites?select=*&limit=1`;

const options = {
  headers: {
    'apikey': serviceKey,
    'Authorization': `Bearer ${serviceKey}`,
    'Prefer': 'return=representation'
  }
};

https.get(url, options, (res) => {
  let data = '';
  res.on('data', (chunk) => data += chunk);
  res.on('end', () => {
    if (res.statusCode === 200) {
      const sites = JSON.parse(data);
      console.log('✅ Site schema (first row):');
      console.log(JSON.stringify(sites[0], null, 2));
      console.log('\nColumn names:');
      console.log(Object.keys(sites[0]).join(', '));
    } else {
      console.error('❌ Failed:', data);
    }
  });
}).on('error', (err) => {
  console.error('Error:', err.message);
});
