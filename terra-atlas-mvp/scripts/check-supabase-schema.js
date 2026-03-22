const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '.env.local' });

async function checkSchema() {
  console.log('🔍 Checking Supabase Schema\n');
  
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );
  
  try {
    // Get one row to see the actual columns
    const { data, error } = await supabase
      .from('sites')
      .select('*')
      .limit(1);
    
    if (error) {
      console.error('❌ Error:', error.message);
      
      // Try to create the table if it doesn't exist
      console.log('\n📊 Table might not exist. Let me check what tables are available...');
      
      // Get all tables (this might not work depending on permissions)
      const { data: tables, error: tablesError } = await supabase
        .from('information_schema.tables')
        .select('table_name')
        .eq('table_schema', 'public');
      
      if (tables) {
        console.log('Available tables:', tables);
      }
      
      return;
    }
    
    if (data && data.length > 0) {
      console.log('✅ Sites table exists with columns:');
      console.log(Object.keys(data[0]));
      console.log('\n📝 Sample row:');
      console.log(data[0]);
    } else {
      console.log('✅ Sites table exists but is empty');
      
      // Try to insert a test row to see what columns are expected
      const testSite = {
        name: 'Test Site',
        site_type: 'solar',
        status: 'operational',
        country: 'USA',
        region: 'TX',
        latitude: 30.0,
        longitude: -95.0,
        capacity_mw: 100
      };
      
      const { data: insertData, error: insertError } = await supabase
        .from('sites')
        .insert([testSite])
        .select();
      
      if (insertError) {
        console.log('\n❌ Insert error reveals missing columns:');
        console.log(insertError.message);
      } else if (insertData) {
        console.log('\n✅ Successfully inserted test row with columns:');
        console.log(Object.keys(insertData[0]));
        
        // Clean up test row
        await supabase
          .from('sites')
          .delete()
          .eq('id', insertData[0].id);
      }
    }
    
    // Get row count
    const { count } = await supabase
      .from('sites')
      .select('*', { count: 'exact', head: true });
    
    console.log(`\n📊 Total rows in sites table: ${count}`);
    
  } catch (error) {
    console.error('❌ Unexpected error:', error);
  }
}

checkSchema();